"""
R26-IT-042 — WorkPlus  |  Employee Activity Monitoring System
main.py

Application entry point.

Startup sequence
────────────────
1. Load config (config/settings.py)
2. Connect to MongoDB Atlas (common/database.py)
3. Show 3-Step Login (app/login.py): Password → MFA → Face + Liveness
4. On successful login →
   a. Open employee dashboard (dashboard/employee_panel.py)
   b. Start C3 activity monitoring in background (initialize_monitoring.py)
   c. Start C1 user interaction profiling
   d. Start C4 productivity prediction logger
5. On logout / shutdown → graceful teardown of all threads + flush logs

Admin panel:
   Run: python dashboard/admin_panel.py  (separate process)
   Or:  python main.py --admin

Cross-platform notes
────────────────────
• sys.platform == "win32"  → Windows (pynput keyboard hook OK)
• sys.platform == "darwin" → macOS (prompt user to grant Accessibility +
                             Camera permissions in System Preferences)
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Optional

import customtkinter as ctk

# Keep TensorFlow/MediaPipe runtime logs quieter in console output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# ── Path bootstrap ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _ensure_project_venv_runtime() -> None:
    """Re-exec into the project virtualenv interpreter when available."""
    if os.environ.get("WORKPLUS_SKIP_VENV_REEXEC") == "1":
        return

    venv_python = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return

    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except Exception:
        return

    if current == target:
        return

    env = os.environ.copy()
    env["WORKPLUS_SKIP_VENV_REEXEC"] = "1"
    cmd = [str(target), str(_PROJECT_ROOT / "main.py"), *sys.argv[1:]]
    print(f"[WorkPlus] Switching runtime to project venv: {target}")
    subprocess.Popen(cmd, cwd=str(_PROJECT_ROOT), env=env)
    raise SystemExit(0)


_ensure_project_venv_runtime()

# ── Internal imports ──────────────────────────────────────────────────────
from config.settings import settings
from common.database import MongoDBClient
from common.logger import SecureLogger
from common.alerts import AlertSender

# ── Logging bootstrap ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

# ── Global appearance ─────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Application Controller
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Application:
    """
    Top-level application controller.
    Owns all resources and orchestrates the startup / shutdown lifecycle.
    """

    def __init__(self) -> None:
        self._db_client: Optional[MongoDBClient] = None
        self._secure_logger: Optional[SecureLogger] = None
        self._alert_sender: Optional[AlertSender] = None
        self._user_id: Optional[str] = None
        self._session_id: Optional[str] = None
        self._employee: Optional[dict] = None
        self._monitor_threads: list[threading.Thread] = []
        self._employee_panel = None
        self._shutdown_event = threading.Event()
        self._root: Optional[ctk.CTk] = None
        self._liveness_score = 1.0
        self._cam_streaming = False
        self._screen_streaming = False
        self._location_mode = "unknown"
        self._wifi_ssid_match = False
        self._current_city = "Unknown"
        self._location_context: dict = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def run(self, admin_mode: bool = False) -> None:
        """Main entry point — runs until logout or window close."""
        self._check_platform()
        self._load_config()
        self._connect_db()
        self._init_services()

        if admin_mode:
            self._launch_admin_panel()
        else:
            self._launch_login()

    def _check_platform(self) -> None:
        log.info("Platform: %s", sys.platform)
        log.info("Python executable: %s", sys.executable)
        try:
            mp_version = version("mediapipe")
            log.info("MediaPipe package version=%s", mp_version)
        except PackageNotFoundError:
            log.warning("MediaPipe package is not installed in this interpreter.")
        except Exception as exc:
            log.warning("MediaPipe package check failed: %s", exc)
        if sys.platform == "darwin":
            from tkinter import messagebox
            messagebox.showinfo(
                "macOS Permissions Required",
                "This application requires:\n\n"
                "• Camera access (facial liveness)\n"
                "• Accessibility access (keyboard / mouse tracking)\n\n"
                "Grant these in System Preferences → Privacy & Security.",
            )

    def _load_config(self) -> None:
        missing = settings.validate()
        if missing:
            from tkinter import messagebox
            messagebox.showwarning(
                "Configuration Warning",
                f"Missing environment variables:\n\n"
                + "\n".join(f"  • {k}" for k in missing)
                + "\n\nEdit .env and restart.",
            )
        log.info("Settings loaded: %s", settings)

    def _connect_db(self) -> None:
        self._db_client = MongoDBClient(
            uri=settings.MONGO_URI,
            db_name=settings.MONGO_DB_NAME,
        )
        ok = self._db_client.connect()
        if ok:
            # Ensure all required collections are indexed
            self._ensure_extra_indexes()
        else:
            log.warning("MongoDB offline — running in offline mode.")

    def _ensure_extra_indexes(self) -> None:
        """Create indexes for collections added by the full system."""
        if not self._db_client or not self._db_client.is_connected:
            return
        try:
            import pymongo
            db = self._db_client._db  # internal reference
            if db is None:
                return
            db["activity_logs"].create_index(
                [("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
                name="user_activity_time", background=True,
            )
            db["employees"].create_index(
                [("employee_id", pymongo.ASCENDING)],
                name="employee_id_unique", unique=True, background=True,
            )
            db["tasks"].create_index(
                [("employee_id", pymongo.ASCENDING), ("assigned_at", pymongo.DESCENDING)],
                name="employee_tasks", background=True,
            )
            db["attendance_logs"].create_index(
                [("employee_id", pymongo.ASCENDING), ("date", pymongo.DESCENDING)],
                name="employee_attendance", background=True,
            )
            db["policy_violations"].create_index(
                [("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
                name="user_violations", background=True,
            )
            db["antispoofing_checks"].create_index(
                [("user_id", pymongo.ASCENDING), ("timestamp", pymongo.DESCENDING)],
                name="user_antispoofing_checks", background=True,
            )
            log.debug("Extra MongoDB indexes ensured.")
        except Exception as exc:
            log.warning("Could not ensure extra indexes: %s", exc)

    def _init_services(self) -> None:
        self._secure_logger = SecureLogger(user_id="SYSTEM")
        self._alert_sender = AlertSender(
            ws_url=settings.WEBSOCKET_URL,
            fallback_logger=self._secure_logger,
        )

    # ------------------------------------------------------------------
    # Login (employee mode)
    # ------------------------------------------------------------------

    def _launch_login(self) -> None:
        from app.login import LoginWindow

        win = LoginWindow(
            db=self._db_client,
            on_success=self._on_login_success,
            alert_sender=self._alert_sender,
        )
        win.protocol("WM_DELETE_WINDOW", self._shutdown)
        self._root = win
        win.mainloop()

    def _on_login_success(self, employee: dict, session_id: str) -> None:
        """Called by LoginWindow after all 3 steps pass."""
        self._employee = employee
        self._user_id = employee.get("employee_id", "?")
        self._session_id = session_id
        self._liveness_score = employee.get("face_liveness_score", 1.0)
        self._location_mode = str(employee.get("location_mode") or "unknown").lower()
        self._wifi_ssid_match = bool(employee.get("wifi_ssid_match", False))
        self._current_city = employee.get("geo_city") or "Unknown"
        self._location_context = {
            "city": employee.get("geo_city") or "Unknown",
            "region": employee.get("geo_region") or "Unknown",
            "country": employee.get("geo_country") or "Unknown",
            "timezone": employee.get("geo_timezone") or "Unknown",
            "isp": employee.get("geo_isp") or "Unknown",
            "org": employee.get("geo_org") or "Unknown",
            "asn": employee.get("geo_asn") or "Unknown",
            "geo_source": employee.get("geo_source") or "unknown",
            "lat": employee.get("geo_lat"),
            "lon": employee.get("geo_lon"),
            "confidence": float(employee.get("geo_confidence") or 0.0),
            "location_hint": employee.get("geo_hint") or "Unknown",
            "vpn_proxy_detected": bool(employee.get("vpn_proxy_detected", False)),
            "hosting_detected": bool(employee.get("hosting_detected", False)),
            "geolocation_deviation": employee.get("geolocation_deviation"),
            "inside_office_geofence": employee.get("inside_office_geofence"),
            "geolocation_resolved": bool(employee.get("geolocation_resolved", False)),
            "office_radius_km": float(employee.get("office_radius_km") or 0.0),
            "location_trust_score": float(employee.get("location_trust_score") or 0.0),
        }
        log.info("Login success: %s session=%s liveness=%.2f", self._user_id, session_id, self._liveness_score)

        if self._secure_logger:
            self._secure_logger._user_id = self._user_id

        # Close login window
        if self._root:
            try:
                self._root.withdraw()
            except Exception:
                pass

        # Start monitoring threads
        self._init_monitoring()

        # Open employee panel
        self._open_employee_panel()

    def _open_employee_panel(self) -> None:
        from dashboard.employee_panel import EmployeePanel

        # Need a CTk root for Toplevel
        root = ctk.CTk()
        root.withdraw()
        root.protocol("WM_DELETE_WINDOW", self._shutdown)
        root._app = self
        self._root = root

        panel = EmployeePanel(
            parent=root,
            employee=self._employee,
            db=self._db_client,
            session_id=self._session_id,
            break_manager=getattr(self, "_break_manager", None),
            alert_sender=self._alert_sender,
        )
        self._employee_panel = panel
        self._bind_break_manager_to_panel_async(panel)
        # Add a helper to the root so panel can call it
        root.show_login = self._restart_at_login
        panel.protocol("WM_DELETE_WINDOW", self._shutdown)

        # Log attendance sign-in
        self._log_attendance_signin()

        root.mainloop()

    def _bind_break_manager_to_panel_async(self, panel) -> None:
        """Bind break manager to the panel even when C3 starts slightly later."""
        def _worker() -> None:
            # Retry for a short period because C3 startup is asynchronous.
            deadline = time.time() + 60.0
            while time.time() < deadline and not self._shutdown_event.is_set():
                bm = getattr(self, "_break_manager", None)
                if bm is not None:
                    try:
                        if panel is not None and panel.winfo_exists():
                            panel.after(0, lambda b=bm: panel.set_break_manager(b))
                    except Exception:
                        pass
                    return
                time.sleep(0.5)

        threading.Thread(target=_worker, daemon=True, name="BreakManager-Binder").start()

    def _log_attendance_signin(self) -> None:
        try:
            from datetime import datetime, timezone
            col = self._db_client.get_collection("attendance_logs")
            if col is not None:
                today = datetime.now().strftime("%Y-%m-%d")
                existing = col.find_one({"employee_id": self._user_id, "date": today})
                
                if not existing:
                    col.insert_one({
                        "employee_id": self._user_id,
                        "full_name": self._employee.get("full_name", ""),
                        "date": today,
                        "signin": datetime.now().strftime("%H:%M:%S"),
                        "signout": None,
                        "duration": None,
                        "location": getattr(self, "_current_city", "Unknown"),
                        "status": "On Time",
                    })
                else:
                    # Relog on the same day — clear signout/duration so they show as online
                    col.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {"signout": None, "duration": None, "status": "On Time"}}
                    )
        except Exception as exc:
            log.warning("Attendance signin log error: %s", exc)

    def _log_attendance_signout(self) -> None:
        try:
            from datetime import datetime
            col = self._db_client.get_collection("attendance_logs")
            if col is not None:
                today = datetime.now().strftime("%Y-%m-%d")
                now_str = datetime.now().strftime("%H:%M:%S")
                
                # Update today's record
                doc = col.find_one({"employee_id": self._user_id, "date": today})
                if doc and doc.get("signin"):
                    # Calculate duration
                    s_t = datetime.strptime(doc["signin"], "%H:%M:%S")
                    e_t = datetime.strptime(now_str, "%H:%M:%S")
                    diff = e_t - s_t
                    duration = str(diff).split(".")[0] # HH:MM:SS

                    # Derive final attendance status with overtime support.
                    sign_in_cutoff = datetime.strptime("09:15:00", "%H:%M:%S").time()
                    early_departure_cutoff = datetime.strptime("17:00:00", "%H:%M:%S").time()
                    overtime_cutoff = datetime.strptime("18:00:00", "%H:%M:%S").time()
                    sign_in_time = s_t.time()
                    sign_out_time = e_t.time()

                    if sign_out_time >= overtime_cutoff:
                        final_status = "Overtime"
                    elif sign_out_time < early_departure_cutoff:
                        final_status = "Early Departure"
                    elif sign_in_time > sign_in_cutoff:
                        final_status = "Late"
                    else:
                        final_status = "On Time"
                    
                    col.update_one(
                        {"employee_id": self._user_id, "date": today},
                        {"$set": {
                            "signout": now_str,
                            "duration": duration,
                            "status": final_status
                        }}
                    )
        except Exception as exc:
            log.warning("Attendance signout log error: %s", exc)

    # ------------------------------------------------------------------
    # Monitoring (C1, C3, C4)
    # ------------------------------------------------------------------

    def _init_monitoring(self) -> None:
        # C3: Activity monitoring
        self._start_thread("C3-ActivityMonitoring", self._start_c3)
        # C1: User interaction
        self._start_thread("C1-UserInteraction", self._start_c1)
        # C4: Productivity prediction
        self._start_thread("C4-ProductivityPrediction", self._start_c4)
        # REMOTE COMMANDS: Admin-to-Employee instructions
        self._start_thread("COMMANDS", self._start_command_poller)

    def _restart_at_login(self) -> None:
        """Called on logout to return to the login screen."""
        log.info("Restarting at login...")
        self._log_attendance_signout()
        self._shutdown_monitoring()
        if self._root:
            self._root.destroy()
        self._launch_login()

    def _shutdown_monitoring(self) -> None:
        self._shutdown_event.set()
        for t in self._monitor_threads:
            t.join(timeout=1.0)
        self._monitor_threads = []
        self._shutdown_event.clear()

    def _start_thread(self, name: str, target) -> None:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()
        self._monitor_threads.append(t)
        log.info("Thread started: %s", name)

    def _start_c3(self) -> None:
        try:
            from C3_activity_monitoring.src.initialize_monitoring import start_monitoring
            self._break_manager = start_monitoring(
                user_id=self._user_id,
                db_client=self._db_client,
                alert_sender=self._alert_sender,
                shutdown_event=self._shutdown_event,
                session_id=self._session_id,
                location_mode=self._location_mode,
                location_context=self._location_context,
                wifi_ssid_match=self._wifi_ssid_match,
                face_liveness_score=self._liveness_score,
            )
            try:
                panel = getattr(self, "_employee_panel", None)
                if panel is not None and hasattr(panel, "set_break_manager"):
                    panel.set_break_manager(self._break_manager)
            except Exception:
                pass
        except ImportError:
            log.warning("C3 not available — skipping activity monitoring.")
        except Exception as exc:
            log.error("C3 crashed: %s", exc)

    def _start_c1(self) -> None:
        try:
            from C1_user_interaction.src import start_interaction_profiling
            start_interaction_profiling(
                user_id=self._user_id,
                shutdown_event=self._shutdown_event,
            )
        except ImportError:
            log.warning("C1 not available — skipping.")
        except Exception as exc:
            log.error("C1 crashed: %s", exc)

    def _start_c4(self) -> None:
        try:
            from C4_productivity_prediction.src import start_productivity_logger
            start_productivity_logger(
                user_id=self._user_id,
                db_client=self._db_client,
                shutdown_event=self._shutdown_event,
            )
        except ImportError:
            log.warning("C4 not available — skipping.")
        except Exception as exc:
            log.error("C4 crashed: %s", exc)

    def _start_command_poller(self) -> None:
        """
        Background thread that listens for instructions from the admin panel
        (e.g., force screenshot, lock workstation, display message).
        """
        try:
            from common.commands import CommandPoller
            from C3_activity_monitoring.src.screenshot_trigger import ScreenshotTrigger
            from common.encryption import AESEncryptor
            import cv2
            import base64
            from datetime import datetime

            poller = CommandPoller(
                user_id=self._user_id,
                db_client=self._db_client,
                shutdown_event=self._shutdown_event,
                interval_sec=10
            )

            # --- Handler: Remote Screenshot ---
            def handle_screenshot(cmd: dict):
                log.info("Executing remote command: force_screenshot")
                try:
                    enc = AESEncryptor()
                    st = ScreenshotTrigger(db_client=self._db_client, encryptor=enc)
                    st.capture(
                        user_id=self._user_id,
                        session_id=self._session_id or "admin_remote",
                        risk_score=cmd.get("risk_score", 0.0),
                        trigger_reason="admin_remote_force"
                    )
                except Exception as e:
                    log.error("Remote screenshot execution failed: %s", e)
                    raise

            # --- Handler: Live Cam & Screen ---
            self._cam_streaming = False
            self._screen_streaming = False
            
            def handle_start_cam(cmd: dict):
                if self._cam_streaming: return
                log.info("Starting live camera stream for admin")
                self._cam_streaming = True
                
                def stream_loop():
                    col = self._db_client.get_collection("camera_streams")
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        log.error("LIVE CAM: Could not open camera.")
                        if col is not None:
                            col.update_one({"user_id": self._user_id}, {"$set": {"status": "off", "error": "Camera unavailable"}}, upsert=True)
                        self._cam_streaming = False
                        return

                    log.info("LIVE CAM: Camera stream started.")
                    try:
                        while self._cam_streaming and not self._shutdown_event.is_set():
                            ret, frame = cap.read()
                            if ret:
                                # Resize and encode
                                frame = cv2.resize(frame, (320, 240))
                                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                                b64 = base64.b64encode(buffer).decode('utf-8')
                                
                                if col is not None:
                                    col.update_one(
                                        {"user_id": self._user_id},
                                        {"$set": {
                                            "image_base64": b64,
                                            "timestamp": datetime.utcnow().isoformat(),
                                            "status": "streaming"
                                        }},
                                        upsert=True
                                    )
                            else:
                                log.warning("LIVE CAM: Failed to capture frame.")
                            time.sleep(1.0)
                    except Exception as e:
                        log.error("LIVE CAM: Loop error: %s", e)
                    finally:
                        cap.release()
                        if col is not None:
                            col.update_one({"user_id": self._user_id}, {"$set": {"status": "off"}})
                        self._cam_streaming = False
                        log.info("LIVE CAM: Camera stream stopped.")

                threading.Thread(target=stream_loop, daemon=True).start()

            def handle_stop_cam(cmd: dict):
                log.info("Stopping live camera stream")
                self._cam_streaming = False

            def handle_start_screen(cmd: dict):
                if self._screen_streaming: return
                log.info("Starting live screen stream for admin")
                self._screen_streaming = True
                
                def screen_loop():
                    import pyautogui, io
                    col = self._db_client.get_collection("screen_streams")
                    try:
                        while self._screen_streaming and not self._shutdown_event.is_set():
                            # Capture
                            img = pyautogui.screenshot()
                            # Resize to reasonable size
                            img = img.resize((640, 360))
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=40)
                            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                            
                            if col is not None:
                                col.update_one(
                                    {"user_id": self._user_id},
                                    {"$set": {
                                        "image_base64": b64,
                                        "timestamp": datetime.utcnow().isoformat(),
                                        "status": "streaming"
                                    }},
                                    upsert=True
                                )
                            time.sleep(2.0) # Slightly slower for screen
                    except Exception as e:
                        log.error("LIVE SCREEN: Loop error: %s", e)
                    finally:
                        if col is not None:
                            col.update_one({"user_id": self._user_id}, {"$set": {"status": "off"}})
                        self._screen_streaming = False
                        log.info("LIVE SCREEN: Screen stream stopped.")

                threading.Thread(target=screen_loop, daemon=True).start()

            def handle_stop_screen(cmd: dict):
                log.info("Stopping live screen stream")
                self._screen_streaming = False

            def handle_antispoofing_check(cmd: dict):
                """
                Remote anti-spoofing check initiated by admin.
                Runs ResNet50 model on live camera frames, determines spoof status,
                and compares the live face against the logged-in employee's stored face.
                """
                log.info("Executing remote command: antispoofing_check")
                try:
                    from C2_Anti_Spoofing_Detection.src.antispoofing_detector import AntiSpoofingDetector
                    from C3_activity_monitoring.src.face_verifier import FaceVerifier
                    from common.antispoofing_utils import store_antispoofing_result
                    import time as time_module
                    
                    detector = AntiSpoofingDetector()
                    if not detector.load_model():
                        log.error("Anti-spoofing model failed to load.")
                        return

                    stored_embedding = None
                    identity_available = False
                    verifier = None
                    legacy_histogram = False
                    emp_doc = None
                    try:
                        emp_col = self._db_client.get_collection("employees")
                        if emp_col is not None:
                            emp_doc = emp_col.find_one(
                                {"employee_id": self._user_id},
                                {"face_embedding": 1, "face_images": 1}
                            )
                            if emp_doc is not None:
                                stored_embedding = emp_doc.get("face_embedding")
                                if stored_embedding:
                                    identity_available = True
                    except Exception as exc:
                        log.warning("Unable to load stored user face embedding: %s", exc)

                    FACE_RECOGNITION_THRESHOLD = 0.70

                    def is_facenet_embedding(embedding) -> bool:
                        try:
                            import numpy as np
                            vec = np.array(embedding, dtype=np.float32).flatten()
                            if vec.size != 128:
                                return False
                            max_abs = float(np.max(np.abs(vec)))
                            mean_abs = float(np.mean(np.abs(vec)))
                            return max_abs <= 5.0 and mean_abs <= 1.0
                        except Exception:
                            return False

                    def bootstrap_facenet_embedding(images_b64):
                        try:
                            import base64
                            import cv2
                            import numpy as np
                            embeddings = []
                            for img_b64 in (images_b64 or [])[:5]:
                                try:
                                    raw = base64.b64decode(img_b64)
                                    arr = np.frombuffer(raw, dtype=np.uint8)
                                    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                                    if gray is None:
                                        continue
                                    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                                    emb = verifier.get_embedding(bgr)
                                    if emb is not None:
                                        embeddings.append(emb)
                                except Exception:
                                    continue
                            if not embeddings:
                                return []
                            avg_emb = np.mean(np.array(embeddings), axis=0).tolist()
                            return avg_emb
                        except Exception:
                            return []

                    def histogram_similarity(face_roi, stored_hist):
                        try:
                            import cv2
                            import numpy as np
                            if face_roi is None or face_roi.size == 0:
                                return 0.0
                            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            gray = cv2.equalizeHist(gray)
                            current_hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
                            stored_vec = np.array(stored_hist, dtype=np.float32).flatten()
                            if current_hist.size != stored_vec.size:
                                min_len = min(current_hist.size, stored_vec.size)
                                current_hist = current_hist[:min_len]
                                stored_vec = stored_vec[:min_len]
                            norm_current = np.linalg.norm(current_hist)
                            norm_stored = np.linalg.norm(stored_vec)
                            if norm_current < 1e-6 or norm_stored < 1e-6:
                                return 0.0
                            return float(np.dot(current_hist, stored_vec) / (norm_current * norm_stored))
                        except Exception:
                            return 0.0

                    if identity_available:
                        try:
                            verifier = FaceVerifier(model_path="models/face_recognition_sface.onnx")
                        except Exception as exc:
                            log.warning("Face identity verifier unavailable: %s", exc)
                            verifier = None

                    if identity_available and verifier is not None and stored_embedding is not None:
                        if not is_facenet_embedding(stored_embedding):
                            face_images = emp_doc.get("face_images", []) if emp_doc is not None else []
                            migrated_embedding = []
                            if face_images:
                                migrated_embedding = bootstrap_facenet_embedding(face_images)
                            if migrated_embedding:
                                stored_embedding = migrated_embedding
                                try:
                                    emp_col.update_one(
                                        {"employee_id": self._user_id},
                                        {"$set": {"face_embedding": migrated_embedding}}
                                    )
                                    log.info("Migrated legacy face_embedding to FaceNet template for user %s", self._user_id)
                                except Exception:
                                    log.warning("Unable to persist migrated FaceNet embedding for user %s", self._user_id)
                            else:
                                legacy_histogram = True
                                log.info("Using legacy histogram fallback for identity verification for user %s", self._user_id)

                    import cv2
                    import numpy as np
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        log.warning("Camera not available for anti-spoofing check.")
                        detector.close()
                        if verifier:
                            verifier.close()
                        return

                    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    face_cascade = cv2.CascadeClassifier(cascade_path)
                    if face_cascade.empty():
                        log.warning("Failed to load face cascade for identity verification.")
                        face_cascade = None
                    
                    # Run check: collect predictions over ~10 seconds
                    predictions = []
                    scores = []
                    identity_scores = []
                    identity_match_count = 0
                    best_identity_score = 0.0
                    start_time = time_module.time()
                    timeout_sec = 10.0
                    
                    while (time_module.time() - start_time) < timeout_sec:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        is_real, confidence, _ = detector.predict(frame)
                        predictions.append(is_real)
                        scores.append(1.0 - confidence if is_real else confidence)

                        detection_box = None
                        if face_cascade is not None:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                            if len(faces) > 0:
                                detection_box = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]

                        if identity_available and stored_embedding is not None:
                            identity_score = 0.0
                            matched = False
                            try:
                                if detection_box is not None:
                                    if verifier is not None and not legacy_histogram:
                                        matched, identity_score = verifier.verify(
                                            frame,
                                            stored_embedding,
                                            threshold=FACE_RECOGNITION_THRESHOLD,
                                            detection_box=np.array(detection_box, dtype=np.uint32),
                                        )
                                    else:
                                        x, y, w, h = detection_box
                                        face_roi = frame[y:y + h, x:x + w]
                                        identity_score = histogram_similarity(face_roi, stored_embedding)
                                        matched = identity_score >= FACE_RECOGNITION_THRESHOLD
                                else:
                                    log.debug("No face region detected by cascade; falling back to full-frame identity verification.")
                                    if verifier is not None and not legacy_histogram:
                                        matched, identity_score = verifier.verify(
                                            frame,
                                            stored_embedding,
                                            threshold=FACE_RECOGNITION_THRESHOLD,
                                        )
                                    else:
                                        identity_score = histogram_similarity(frame, stored_embedding)
                                        matched = identity_score >= FACE_RECOGNITION_THRESHOLD
                            except Exception as exc:
                                log.debug("Identity verification failed for frame: %s", exc)
                                matched = False
                                identity_score = 0.0

                            identity_scores.append(identity_score)
                            best_identity_score = max(best_identity_score, identity_score)
                            if matched:
                                identity_match_count += 1

                    cap.release()
                    detector.close()
                    if verifier:
                        verifier.close()
                    
                    if not predictions:
                        log.warning("No frames captured for anti-spoofing check.")
                        return
                    
                    # Average results
                    duration = time_module.time() - start_time
                    avg_is_real = sum(predictions) / len(predictions) >= 0.5
                    avg_confidence = sum(confidence for confidence in scores) / len(scores) if scores else 0.0
                    avg_score = sum(scores) / len(scores) if scores else 0.5
                    best_identity_score = best_identity_score if identity_scores else 0.0
                    identity_match = False
                    identity_status = "UNKNOWN"

                    if identity_available and verifier is not None and stored_embedding is not None:
                        if identity_match_count >= 1 or best_identity_score >= FACE_RECOGNITION_THRESHOLD:
                            identity_match = True
                            identity_status = "SAME_PERSON"
                        else:
                            identity_match = False
                            identity_status = "DIFFERENT_PERSON"
                    elif identity_available and verifier is None:
                        identity_status = "VERIFIER_UNAVAILABLE"
                    elif not identity_available:
                        identity_status = "NO_TEMPLATE"

                    if not avg_is_real:
                        verdict = "FAKE"
                    elif identity_status == "SAME_PERSON":
                        verdict = "REAL_SAME_PERSON"
                    elif identity_status == "DIFFERENT_PERSON":
                        verdict = "REAL_DIFFERENT_PERSON"
                    else:
                        verdict = "REAL_UNKNOWN"
                    
                    success = store_antispoofing_result(
                        db_client=self._db_client,
                        user_id=self._user_id,
                        is_real=avg_is_real,
                        confidence=avg_confidence,
                        frame_count=len(predictions),
                        avg_score=avg_score,
                        duration_sec=duration,
                        identity_match=identity_match if identity_status in {"SAME_PERSON", "DIFFERENT_PERSON"} else None,
                        identity_score=best_identity_score,
                        identity_status=identity_status,
                        verdict=verdict,
                    )
                    
                    if success:
                        log.info(
                            "Anti-spoofing check completed: user=%s verdict=%s confidence=%.2f frames=%d identity_status=%s identity_score=%.2f",
                            self._user_id,
                            verdict,
                            avg_confidence,
                            len(predictions),
                            identity_status,
                            best_identity_score,

                            identity_status,
                            avg_identity_score,
                        )
                    
                except Exception as e:
                    log.error("Anti-spoofing check execution failed: %s", e)
            
            poller.register_handler("force_screenshot", handle_screenshot)
            poller.register_handler("start_live_cam", handle_start_cam)
            poller.register_handler("stop_live_cam", handle_stop_cam)
            poller.register_handler("start_live_screen", handle_start_screen)
            poller.register_handler("stop_live_screen", handle_stop_screen)
            poller.register_handler("start_antispoofing_check", handle_antispoofing_check)
            
            # Start polling
            poller.start()

        except Exception as exc:
            log.error("Command poller initialization failed: %s", exc)

    # ------------------------------------------------------------------
    # Admin panel mode
    # ------------------------------------------------------------------

    def _launch_admin_panel(self) -> None:
        from dashboard.admin_panel import AdminPanel
        panel = AdminPanel(db=self._db_client)
        panel.protocol("WM_DELETE_WINDOW", self._shutdown)
        panel.mainloop()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        log.info("Shutting down — signalling all monitor threads.")
        self._shutdown_event.set()

        # Log attendance sign-out
        try:
            if self._user_id and self._db_client and self._db_client.is_connected:
                from datetime import datetime
                col = self._db_client.get_collection("attendance_logs")
                today = datetime.now().strftime("%Y-%m-%d")
                if col:
                    col.update_one(
                        {"employee_id": self._user_id, "date": today},
                        {"$set": {"signout": datetime.now().strftime("%H:%M:%S")}},
                    )
        except Exception:
            pass

        # Update session status
        try:
            if self._session_id and self._db_client and self._db_client.is_connected:
                from datetime import datetime
                col = self._db_client.get_collection("sessions")
                if col is not None:
                    col.update_one(
                        {"session_id": self._session_id},
                        {"$set": {"status": "ended", "logout_at": datetime.utcnow().isoformat()}},
                    )
        except Exception:
            pass

        # Wait up to 5 s for threads
        for t in self._monitor_threads:
            t.join(timeout=5.0)

        # Final offline flush
        try:
            if self._secure_logger and self._db_client and self._db_client.is_connected:
                col = self._db_client.get_collection("sessions")
                if col is not None:
                    self._secure_logger.flush_queue(col)
        except Exception:
            pass

        if self._db_client:
            self._db_client.close()

        log.info("Shutdown complete.")
        sys.exit(0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _select_venv_python() -> Optional[Path]:
    """Return local venv python path if it exists."""
    venv_py = _PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return venv_py if venv_py.exists() else None


def _ensure_venv_interpreter() -> None:
    """Relaunch with local venv interpreter when started from system Python."""
    if os.environ.get("WORKPLUS_SKIP_VENV_REEXEC") == "1":
        return

    venv_py = _select_venv_python()
    if venv_py is None:
        return

    current = str(Path(sys.executable).resolve()).lower()
    target = str(venv_py.resolve()).lower()
    if current == target:
        return

    os.environ["WORKPLUS_SKIP_VENV_REEXEC"] = "1"
    print(f"[WorkPlus] Switching interpreter to venv: {venv_py}")
    env = os.environ.copy()
    env["WORKPLUS_SKIP_VENV_REEXEC"] = "1"
    subprocess.Popen([str(venv_py), *sys.argv], cwd=str(_PROJECT_ROOT), env=env)
    raise SystemExit(0)

def main() -> None:
    _ensure_venv_interpreter()

    parser = argparse.ArgumentParser(description="WorkPlus Employee Monitoring System")
    parser.add_argument("--admin", action="store_true", help="Open the admin panel directly")
    args = parser.parse_args()

    try:
        app = Application()
        app.run(admin_mode=args.admin)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        log.critical("Unhandled exception: %s", exc, exc_info=True)
        try:
            from tkinter import messagebox
            messagebox.showerror(
                "Fatal Error",
                f"An unexpected error occurred:\n\n{exc}\n\nThe application will close.",
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
