"""
R26-IT-042 — Employee Activity Monitoring System
app/login.py

3-Step Login Flow:
  Step 1 — Employee ID + Password (bcrypt verify, lockout after 3 fails)
  Step 2 — TOTP MFA (pyotp, 6-digit Google Authenticator code)
  Step 3 — Face Verification + Liveness Check (OpenCV DNN + C2 liveness)

On full success:
  - Creates session document in MongoDB "sessions" collection
  - Logs auth event to "auth_events"
  - Starts C3 activity trackers
  - Opens employee_panel.py

Lockout:
  - Password: 5 minutes after 3 failed attempts
  - Face:     15 minutes after 3 failed attempts → CRITICAL alert
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import numpy as np

from common.database import MongoDBClient
from common.alerts import AlertSender
from config.settings import settings

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Color palette ─────────────────────────────────────────────────────────
C_BG    = "#0b0e17"
C_CARD  = "#151b2d"
C_BORDER= "#1e2a40"
C_TEAL  = "#14b8a6"
C_TEAL_D= "#0d9488"
C_RED   = "#ef4444"
C_GREEN = "#22c55e"
C_BLUE  = "#3b82f6"
C_TEXT  = "#e2e8f0"
C_MUTED = "#64748b"

# Lockout constants
_PW_LOCKOUT_MINUTES = 5
_FACE_LOCKOUT_MINUTES = 15
_MAX_ATTEMPTS = 3

# Face similarity threshold (cosine similarity of histogram embeddings)
_FACE_THRESHOLD = 0.45
_FACENET_THRESHOLD = 0.78
_FACENET_SOFT_THRESHOLD = 0.72
_FACE_VERIFY_HITS_REQUIRED = 4

LOGO_PATH = _ROOT / "assets" / "logo.png"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step Indicator Widget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StepIndicator(ctk.CTkFrame):
    """Three-step progress indicator at the top of the login window."""

    STEP_LABELS = ["Password", "MFA Code", "Face Check"]

    def __init__(self, parent, **kw) -> None:
        super().__init__(parent, fg_color="transparent", **kw)
        self._circles: list[ctk.CTkLabel] = []
        self._step_labels: list[ctk.CTkLabel] = []

        for i, label in enumerate(self.STEP_LABELS):
            col = ctk.CTkFrame(self, fg_color="transparent")
            col.pack(side="left", expand=True)

            circle = ctk.CTkLabel(
                col, text=str(i + 1),
                width=36, height=36,
                corner_radius=18,
                fg_color=C_BORDER,
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=C_MUTED,
            )
            circle.pack()

            lbl = ctk.CTkLabel(
                col, text=label,
                font=ctk.CTkFont(size=10),
                text_color=C_MUTED,
            )
            lbl.pack(pady=(4, 0))

            self._circles.append(circle)
            self._step_labels.append(lbl)

            # Connector line between steps
            if i < len(self.STEP_LABELS) - 1:
                line = ctk.CTkFrame(self, height=2, fg_color=C_BORDER, width=60)
                line.pack(side="left", pady=(0, 20))

        self.set_step(0)

    def set_step(self, active: int) -> None:
        """Highlight the active step (0-indexed)."""
        for i, (circle, lbl) in enumerate(zip(self._circles, self._step_labels)):
            if i < active:
                circle.configure(fg_color=C_GREEN, text_color="#fff", text="✓")
                lbl.configure(text_color=C_GREEN)
            elif i == active:
                circle.configure(fg_color=C_TEAL, text_color="#fff", text=str(i + 1))
                lbl.configure(text_color=C_TEAL)
            else:
                circle.configure(fg_color=C_BORDER, text_color=C_MUTED, text=str(i + 1))
                lbl.configure(text_color=C_MUTED)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Login Window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LoginWindow(ctk.CTk):
    """
    3-Step login: Password → MFA → Face + Liveness.
    """

    def __init__(
        self,
        db: MongoDBClient,
        on_success: Callable[[dict, str], None],
        alert_sender: Optional[AlertSender] = None,
    ) -> None:
        super().__init__()
        self._db = db
        self._on_success = on_success
        self._alert_sender = alert_sender

        self._current_employee: Optional[dict] = None
        self._session_id = str(uuid.uuid4())
        self._step = 0
        self._pw_attempts = 0
        self._mfa_attempts = 0
        self._face_attempts = 0

        self.title(f"{settings.APP_NAME} — Login")
        self.geometry("480x680")
        self.resizable(False, False)
        self.configure(fg_color=C_BG)

        self.update_idletasks()
        x = (self.winfo_screenwidth() - 480) // 2
        y = (self.winfo_screenheight() - 680) // 2
        self.geometry(f"480x680+{x}+{y}")

        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        # Logo
        try:
            from PIL import Image
            raw = Image.open(LOGO_PATH).convert("RGBA").resize((90, 90), Image.LANCZOS)
            self._logo = ctk.CTkImage(light_image=raw, dark_image=raw, size=(90, 90))
            ctk.CTkLabel(self, image=self._logo, text="").pack(pady=(30, 4))
        except Exception:
            ctk.CTkLabel(self, text="WorkPlus", font=ctk.CTkFont(size=32, weight="bold"),
                         text_color=C_TEAL).pack(pady=(30, 4))

        ctk.CTkLabel(self, text=settings.APP_NAME,
                     font=ctk.CTkFont(size=18, weight="bold"), text_color=C_TEXT).pack()
        ctk.CTkLabel(self, text="Employee Activity Monitoring System",
                     font=ctk.CTkFont(size=10), text_color=C_MUTED).pack(pady=(0, 16))

        # Step indicator
        self._indicator = StepIndicator(self)
        self._indicator.pack(padx=40, fill="x")

        ctk.CTkFrame(self, height=1, fg_color=C_BORDER).pack(fill="x", padx=32, pady=12)

        # Step content container
        self._step_container = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=16)
        self._step_container.pack(fill="x", padx=32)

        # Status label (shared)
        self._status_var = ctk.StringVar()
        self._status_lbl = ctk.CTkLabel(
            self, textvariable=self._status_var,
            font=ctk.CTkFont(size=11), text_color=C_RED, wraplength=380,
        )
        self._status_lbl.pack(pady=8)

        self._show_step_1()

    def _clear_step(self) -> None:
        for w in self._step_container.winfo_children():
            w.destroy()
        self._status_var.set("")

    # ------------------------------------------------------------------
    # Step 1 — Password
    # ------------------------------------------------------------------

    def _show_step_1(self) -> None:
        self._step = 0
        self._clear_step()
        self._indicator.set_step(0)

        ctk.CTkLabel(self._step_container, text="Sign In",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color=C_TEXT).pack(padx=24, pady=(20, 8), anchor="w")

        ctk.CTkLabel(self._step_container, text="Employee ID",
                     font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(padx=24, fill="x")
        self._emp_id_var = ctk.StringVar()
        self._emp_id_entry = ctk.CTkEntry(
            self._step_container, textvariable=self._emp_id_var,
            placeholder_text="e.g. EMP-001", height=42,
            fg_color="#0f1117", border_color=C_BORDER,
        )
        self._emp_id_entry.pack(padx=24, fill="x", pady=(2, 10))

        ctk.CTkLabel(self._step_container, text="Password",
                     font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(padx=24, fill="x")

        pw_row = ctk.CTkFrame(self._step_container, fg_color="transparent")
        pw_row.pack(padx=24, fill="x", pady=(2, 10))
        self._pw_var = ctk.StringVar()
        self._pw_entry = ctk.CTkEntry(pw_row, textvariable=self._pw_var,
                                       show="•", height=42,
                                       fg_color="#0f1117", border_color=C_BORDER)
        self._pw_entry.pack(side="left", fill="x", expand=True)
        self._show_pw = False
        toggle_btn = ctk.CTkButton(pw_row, text="Show", width=56, height=42,
                                    fg_color=C_BORDER, hover_color="#374151",
                                    command=self._toggle_pw)
        toggle_btn.pack(side="left", padx=(4, 0))
        self._toggle_btn = toggle_btn

        self._login_btn = ctk.CTkButton(
            self._step_container, text="Continue",
            height=44, fg_color=C_TEAL, hover_color=C_TEAL_D,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_step1_submit,
        )
        self._login_btn.pack(padx=24, fill="x", pady=(4, 20))
        self.bind("<Return>", lambda _: self._on_step1_submit())
        self._emp_id_entry.focus()

    def _toggle_pw(self) -> None:
        self._show_pw = not self._show_pw
        self._pw_entry.configure(show="" if self._show_pw else "•")
        self._toggle_btn.configure(text="Hide" if self._show_pw else "Show")

    def _on_step1_submit(self) -> None:
        emp_id = self._emp_id_var.get().strip()
        password = self._pw_var.get()
        if not emp_id or not password:
            self._set_status("Please enter both Employee ID and password.")
            return
        self._login_btn.configure(state="disabled", text="Verifying…")
        threading.Thread(target=self._verify_password, args=(emp_id, password), daemon=True).start()

    def _verify_password(self, emp_id: str, password: str) -> None:
        try:
            col = self._db.get_collection("employees")
            if col is None:
                self.after(0, lambda: self._set_status("Database unavailable. Try again."))
                self.after(0, lambda: self._login_btn.configure(state="normal", text="Continue"))
                return

            emp = col.find_one({"employee_id": emp_id})
            if emp is None:
                self._pw_attempts += 1
                self.after(0, lambda: self._set_status(f"Employee ID not found. ({self._pw_attempts}/{_MAX_ATTEMPTS})"))
                self.after(0, lambda: self._login_btn.configure(state="normal", text="Continue"))
                return

            # Check lockout
            locked_until = emp.get("locked_until")
            if locked_until:
                try:
                    lu = datetime.fromisoformat(locked_until)
                    if datetime.utcnow() < lu.replace(tzinfo=None):
                        remaining = (lu.replace(tzinfo=None) - datetime.utcnow()).seconds // 60 + 1
                        self.after(0, lambda r=remaining: self._set_status(f"Account locked. Try again in {r} minute(s)."))
                        self.after(0, lambda: self._login_btn.configure(state="normal", text="Continue"))
                        return
                except Exception:
                    pass

            # Verify password
            import bcrypt
            pw_hash = emp.get("password_hash", "").encode()
            if not bcrypt.checkpw(password.encode(), pw_hash):
                self._pw_attempts += 1
                if self._pw_attempts >= _MAX_ATTEMPTS:
                    lock_until = (datetime.utcnow() + timedelta(minutes=_PW_LOCKOUT_MINUTES)).isoformat()
                    col.update_one({"employee_id": emp_id}, {"$set": {"locked_until": lock_until}})
                    self.after(0, lambda: self._set_status(f"Too many attempts. Account locked for {_PW_LOCKOUT_MINUTES} minutes."))
                else:
                    self.after(0, lambda a=self._pw_attempts: self._set_status(f"Incorrect password. ({a}/{_MAX_ATTEMPTS})"))
                self.after(0, lambda: self._login_btn.configure(state="normal", text="Continue"))
                return

            # Success — move to step 2
            self._current_employee = emp
            self._pw_attempts = 0
            self.after(0, self._show_step_2)

        except Exception as exc:
            logger.error("Password verify error: %s", exc)
            self.after(0, lambda: self._set_status("Verification error. Please try again."))
            self.after(0, lambda: self._login_btn.configure(state="normal", text="Continue"))

    # ------------------------------------------------------------------
    # Step 2 — MFA TOTP
    # ------------------------------------------------------------------

    def _show_step_2(self) -> None:
        self._step = 1
        self._clear_step()
        self._indicator.set_step(1)

        ctk.CTkLabel(self._step_container, text="Two-Factor Authentication",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color=C_TEXT).pack(padx=24, pady=(20, 4), anchor="w")
        ctk.CTkLabel(self._step_container, text="Enter the 6-digit code from your authenticator app",
                     font=ctk.CTkFont(size=11), text_color=C_MUTED, wraplength=380).pack(padx=24, anchor="w")

        self._otp_var = ctk.StringVar()
        otp_entry = ctk.CTkEntry(
            self._step_container, textvariable=self._otp_var,
            placeholder_text="● ● ● ● ● ●", height=52,
            font=ctk.CTkFont(size=22), justify="center",
            fg_color="#0f1117", border_color=C_BORDER,
        )
        otp_entry.pack(padx=24, fill="x", pady=(14, 4))
        otp_entry.focus()

        ctk.CTkLabel(self._step_container, text="Use Google Authenticator or compatible app",
                     font=ctk.CTkFont(size=10), text_color=C_MUTED).pack(padx=24, anchor="w")

        self._mfa_btn = ctk.CTkButton(
            self._step_container, text="Verify Code",
            height=44, fg_color=C_TEAL, hover_color=C_TEAL_D,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_step2_submit,
        )
        self._mfa_btn.pack(padx=24, fill="x", pady=(14, 20))

        # Auto-submit when 6 digits entered
        self._otp_var.trace_add("write", lambda *_: self._otp_auto_submit())
        self.bind("<Return>", lambda _: self._on_step2_submit())

    def _otp_auto_submit(self) -> None:
        if len(self._otp_var.get()) == 6 and self._otp_var.get().isdigit():
            self.after(200, self._on_step2_submit)

    def _on_step2_submit(self) -> None:
        code = self._otp_var.get().strip()
        if len(code) != 6 or not code.isdigit():
            self._set_status("Enter a valid 6-digit code.")
            return
        self._mfa_btn.configure(state="disabled", text="Verifying…")
        threading.Thread(target=self._verify_mfa, args=(code,), daemon=True).start()

    def _verify_mfa(self, code: str) -> None:
        try:
            import pyotp
            secret = self._current_employee.get("mfa_secret", "")
            if not secret:
                # Dev bypass: accept any valid 6-digit code
                logger.warning("No MFA secret for employee — dev bypass active.")
                self.after(0, self._show_step_3)
                return

            totp = pyotp.TOTP(secret)
            if totp.verify(code, valid_window=1):
                self._mfa_attempts = 0
                self.after(0, self._show_step_3)
            else:
                self._mfa_attempts += 1
                if self._mfa_attempts >= _MAX_ATTEMPTS:
                    self.after(0, lambda: self._set_status("Too many invalid codes. Please restart login."))
                    self.after(2000, self._show_step_1)
                else:
                    self.after(0, lambda a=self._mfa_attempts: self._set_status(f"Invalid code. ({a}/{_MAX_ATTEMPTS})"))
                    self.after(0, lambda: self._mfa_btn.configure(state="normal", text="Verify Code"))
        except Exception as exc:
            logger.error("MFA verify error: %s", exc)
            self.after(0, lambda: self._set_status("MFA verification error."))
            self.after(0, lambda: self._mfa_btn.configure(state="normal", text="Verify Code"))

    # ------------------------------------------------------------------
    # Step 3 — Face Verification + Liveness
    # ------------------------------------------------------------------

    def _show_step_3(self) -> None:
        self._step = 2
        self._clear_step()
        self._indicator.set_step(2)

        ctk.CTkLabel(self._step_container, text="Face Verification",
                     font=ctk.CTkFont(size=15, weight="bold"), text_color=C_TEXT).pack(padx=24, pady=(20, 4), anchor="w")
        ctk.CTkLabel(self._step_container, text="Look directly at the camera and blink naturally",
                     font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(padx=24, anchor="w", pady=(0, 8))

        # macOS permission warning
        if sys.platform == "darwin":
            ctk.CTkLabel(self._step_container,
                         text="macOS: Grant camera access in System Preferences → Privacy.",
                         font=ctk.CTkFont(size=10), text_color=C_AMBER).pack(padx=24, anchor="w")

        # Camera canvas with rounded corners
        canvas_frame = ctk.CTkFrame(self._step_container, fg_color=C_BORDER, corner_radius=16)
        canvas_frame.pack(padx=24, pady=8)
        import tkinter as tk
        self._cam_canvas = tk.Canvas(canvas_frame, width=360, height=270,
                                      bg="#0b0e17", highlightthickness=0)
        self._cam_canvas.pack(padx=2, pady=2)

        self._face_status_var = ctk.StringVar(value="Look at the camera…")
        ctk.CTkLabel(self._step_container, textvariable=self._face_status_var,
                     font=ctk.CTkFont(size=12), text_color=C_MUTED).pack(pady=4)

        self._face_btn = ctk.CTkButton(
            self._step_container, text="Cancel",
            height=40, fg_color=C_BORDER, hover_color="#374151",
            command=self._cancel_face,
        )
        self._face_btn.pack(padx=24, fill="x", pady=(4, 16))

        # Start face check in background
        self._face_running = True
        threading.Thread(target=self._run_face_check, daemon=True).start()

    def _run_face_check(self) -> None:
        cap = None
        detector = None
        verifier = None
        try:
            import cv2
            from C3_activity_monitoring.src.liveness_detector import LivenessDetector
            from PIL import Image, ImageTk

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.after(0, lambda: self._face_status_var.set("Camera not available — check permissions."))
                self.after(0, self._on_face_fail)
                return

            detector = LivenessDetector()
            detector.initialize()

            # Initialize FaceNet verifier
            try:
                from C3_activity_monitoring.src.face_verifier import FaceVerifier
                verifier = FaceVerifier(model_path="models/face_recognition_sface.onnx")
            except Exception as e:
                logger.warning(f"FaceNet verifier failed to initialize: {e}. Using histogram fallback.")
                verifier = None

            # Get stored FaceNet embedding (pre-computed at registration)
            stored_embedding = self._current_employee.get("face_embedding", [])
            if verifier is not None:
                # Some legacy records store 128-bin histogram vectors under face_embedding.
                # Those are incompatible with FaceNet and produce very low scores.
                needs_migration = not self._is_facenet_embedding(stored_embedding)
                if not needs_migration and not self._embedding_consistent_with_images(verifier, stored_embedding):
                    needs_migration = True

                if needs_migration:
                    stored_embedding = self._bootstrap_facenet_embedding(verifier)
                    if stored_embedding:
                        logger.info("FaceNet embedding migrated from stored face images.")
                    else:
                        logger.warning("No FaceNet embedding found. Using histogram embeddings.")
                        verifier = None

            # Face detection (use Haar Cascade for speed)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)

            # 2. Verification Loop
            verified_count = 0
            strong_match_count = 0
            soft_match_count = 0
            best_match_score = 0.0
            liveness_passed_once = False
            start_ts = time.time()
            max_duration = 30.0  # 30 seconds to verify

            while self._face_running and (time.time() - start_ts) < max_duration:
                # Check if window/canvas still exists before continuing
                try:
                    if not self.winfo_exists() or not self._cam_canvas.winfo_exists():
                        break
                except Exception:
                    break

                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detector.process_frame(frame_rgb)

                # Detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5)
                
                display = frame_rgb.copy()
                for (x, y, w_, h_) in faces:
                    cv2.rectangle(display, (x, y), (x + w_, y + h_), (20, 184, 166), 2)

                # Render
                try:
                    img_p = Image.fromarray(display).resize((360, 270))
                    photo = ImageTk.PhotoImage(img_p)
                    self.after(0, lambda p=photo: self._draw_cam(p))
                except Exception:
                    pass

                # Face Verification
                is_match = False
                match_score = 0.0
                
                if len(faces) > 0:
                    (fx, fy, fw, fh) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                    
                    # FaceNet verification (recommended)
                    if verifier is not None and stored_embedding:
                        try:
                            detection_box = np.array([fx, fy, fw, fh], dtype=np.uint32)
                            is_match, match_score = verifier.verify(
                                frame, 
                                stored_embedding,
                                threshold=_FACENET_THRESHOLD,
                                detection_box=detection_box
                            )
                        except Exception as e:
                            logger.debug(f"FaceNet verification failed: {e}")
                            is_match = False
                    else:
                        # Fallback: Histogram similarity (for old registrations)
                        face_roi = gray[fy:fy+fh, fx:fx+fw]
                        face_roi = cv2.equalizeHist(face_roi)
                        stored_emb = self._current_employee.get("face_embedding", [])
                        if stored_emb:
                            current_emb = self._compute_embedding(face_roi)
                            match_score = self._cosine_similarity(current_emb, stored_emb)
                            if match_score >= _FACE_THRESHOLD:
                                is_match = True

                    if is_match:
                        verified_count += 1
                        if match_score >= 0.90:
                            strong_match_count += 1
                        if match_score >= _FACENET_SOFT_THRESHOLD:
                            soft_match_count += 1
                    else:
                        verified_count = max(verified_count - 1, 0)
                        strong_match_count = max(strong_match_count - 1, 0)
                        if match_score >= _FACENET_SOFT_THRESHOLD:
                            soft_match_count += 1
                        else:
                            soft_match_count = max(soft_match_count - 1, 0)

                    best_match_score = max(best_match_score, match_score)

                liveness = detector.get_result()
                if liveness.passed:
                    liveness_passed_once = True
                if (
                    verified_count >= _FACE_VERIFY_HITS_REQUIRED
                    or strong_match_count >= 2
                    or soft_match_count >= 10
                ) and liveness.passed:
                    self.after(0, lambda s=liveness.liveness_score: self._on_face_success(s))
                    return

                # Status strings
                if len(faces) == 0:
                    msg = "Position your face in the box"
                elif not liveness.passed:
                    msg = f"Liveness check... (Blinks: {liveness.blink_count}, Move: {'Y' if liveness.head_moved else 'N'})"
                elif verified_count < _FACE_VERIFY_HITS_REQUIRED:
                    method = "FaceNet" if (verifier is not None) else "Histogram"
                    msg = f"Verifying identity [{method}: {match_score:.2f}] ({verified_count}/{_FACE_VERIFY_HITS_REQUIRED})..."
                else:
                    msg = "Keep still..."
                
                try:
                    if self.winfo_exists():
                        self.after(0, lambda m=msg: self._face_status_var.set(m))
                except Exception:
                    pass

                time.sleep(1 / 20)

            reason = "face"
            if best_match_score >= _FACENET_SOFT_THRESHOLD and not liveness_passed_once:
                reason = "liveness"
            self.after(0, lambda r=reason: self._on_face_fail(r))

        except Exception as exc:
            logger.error("Face check internal error: %s", exc)
            self.after(0, lambda: self._on_face_fail("internal"))
        finally:
            self._face_running = False
            if cap and cap.isOpened():
                cap.release()
            if detector:
                detector.close()
            if verifier:
                verifier.close()

    def _draw_cam(self, photo) -> None:
        try:
            self._cam_canvas._photo = photo
            self._cam_canvas.delete("all")
            self._cam_canvas.create_image(0, 0, anchor="nw", image=photo)
        except Exception:
            pass

    def _compute_embedding(self, face_roi) -> list:
        """Histogram embedding (fallback for old registrations)."""
        try:
            import cv2
            # face_roi is already grayscale from the caller
            hist = cv2.calcHist([face_roi], [0], None, [128], [0, 256])
            return [float(v[0]) for v in hist]
        except Exception:
            return []

    def _bootstrap_facenet_embedding(self, verifier) -> list:
        """Build and persist FaceNet embedding for legacy users from stored face images."""
        try:
            import cv2
            import base64

            images_b64 = self._current_employee.get("face_images", []) or []
            if not images_b64:
                return []

            embeddings = []
            for img_b64 in images_b64[:5]:
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
            self._current_employee["face_embedding"] = avg_emb

            emp_id = self._current_employee.get("employee_id", "")
            col = self._db.get_collection("employees")
            if col is not None and emp_id:
                col.update_one({"employee_id": emp_id}, {"$set": {"face_embedding": avg_emb}})

            return avg_emb
        except Exception as exc:
            logger.debug("Failed to backfill FaceNet embedding: %s", exc)
            return []

    def _is_facenet_embedding(self, embedding: list) -> bool:
        """Heuristic check: distinguish FaceNet vectors from legacy histogram vectors."""
        try:
            if not embedding:
                return False
            vec = np.array(embedding, dtype=np.float32).flatten()
            if vec.size != 128:
                return False

            # Legacy histogram vectors have large bin counts (often >> 10).
            # FaceNet vectors are compact float features near zero-centered range.
            max_abs = float(np.max(np.abs(vec)))
            mean_abs = float(np.mean(np.abs(vec)))
            return max_abs <= 5.0 and mean_abs <= 1.0
        except Exception:
            return False

    def _embedding_consistent_with_images(self, verifier, stored_embedding: list) -> bool:
        """Quickly verify stored embedding still matches employee's enrolled face images."""
        try:
            import cv2
            import base64

            images_b64 = self._current_employee.get("face_images", []) or []
            if not images_b64:
                return True

            sample = images_b64[0]
            arr = np.frombuffer(base64.b64decode(sample), dtype=np.uint8)
            gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return True

            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            emb = verifier.get_embedding(bgr)
            if emb is None:
                return True

            score = verifier.cosine_similarity(emb, np.array(stored_embedding, dtype=np.float32))
            return score >= 0.45
        except Exception:
            return True

    def _cosine_similarity(self, a: list, b: list) -> float:
        """Cosine similarity between two embeddings."""
        try:
            va = np.array(a, dtype=np.float32)
            vb = np.array(b, dtype=np.float32)
            if len(va) != len(vb):
                # Pad/truncate to match
                min_len = min(len(va), len(vb))
                va, vb = va[:min_len], vb[:min_len]
            norm_a = np.linalg.norm(va)
            norm_b = np.linalg.norm(vb)
            if norm_a < 1e-6 or norm_b < 1e-6:
                return 0.0
            return float(np.dot(va, vb) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _on_face_success(self, liveness_score: float) -> None:
        self._face_running = False
        self._face_status_var.set("Identity verified!")
        self._status_lbl.configure(text_color=C_GREEN)
        self._set_status("Identity verified! Starting session…", color=C_GREEN)
        self.after(800, lambda: self._finalize_login(liveness_score))

    def _on_face_fail(self, reason: str = "face") -> None:
        self._face_running = False
        self._face_attempts += 1

        if self._face_attempts >= _MAX_ATTEMPTS:
            # Lock account for 15 minutes
            emp_id = self._current_employee.get("employee_id", "")
            lock_until = (datetime.utcnow() + timedelta(minutes=_FACE_LOCKOUT_MINUTES)).isoformat()
            try:
                col = self._db.get_collection("employees")
                if col is not None:
                    col.update_one({"employee_id": emp_id}, {"$set": {"locked_until": lock_until}})
            except Exception:
                pass

            # Log CRITICAL security event
            self._log_auth_event("face_fail_lockout", emp_id, success=False)

            if self._alert_sender:
                try:
                    self._alert_sender.send_alert(
                        user_id=emp_id, risk_score=95.0,
                        factors=["face_verification_failed", f"attempts_{self._face_attempts}"],
                        level="CRITICAL",
                    )
                except Exception:
                    pass

            self._set_status(f"Face verification failed {_MAX_ATTEMPTS} times. Account locked for {_FACE_LOCKOUT_MINUTES} minutes.")
            self.after(3000, self._show_step_1)
        else:
            if reason == "liveness":
                self._face_status_var.set("Liveness check not passed")
                self._set_status(
                    f"Liveness not confirmed. Blink once or move your head slightly. ({self._face_attempts}/{_MAX_ATTEMPTS})"
                )
            elif reason == "internal":
                self._face_status_var.set("Verification interrupted")
                self._set_status(
                    f"Face verification interrupted. Please retry. ({self._face_attempts}/{_MAX_ATTEMPTS})"
                )
            else:
                self._face_status_var.set("Face not recognized")
                self._set_status(f"Face not recognized. ({self._face_attempts}/{_MAX_ATTEMPTS}). Try again.")
            self._face_btn.configure(text="Retry", command=self._show_step_3)

    def _cancel_face(self) -> None:
        self._face_running = False
        self._show_step_1()

    # ------------------------------------------------------------------
    # Finalize login
    # ------------------------------------------------------------------

    def _finalize_login(self, liveness_score: float) -> None:
        emp_id = self._current_employee.get("employee_id", "")
        try:
            # Geo context
            location_mode = "unknown"
            geo_city = "Unknown"
            geo_country = "Unknown"
            geo_region = "Unknown"
            geo_ip = "unknown"
            geo_lat = None
            geo_lon = None
            geo_timezone = "Unknown"
            geo_isp = "Unknown"
            geo_org = "Unknown"
            geo_asn = "Unknown"
            geo_confidence = 0.0
            geo_hint = "Unknown"
            vpn_proxy_detected = False
            hosting_detected = False
            geolocation_deviation = None
            inside_office_geofence = None
            geolocation_resolved = False
            geo_source = "unknown"
            office_radius_km = 25.0
            location_trust_score = 0.0
            wifi_ssid_hash = ""
            wifi_ssid_match = False
            device_fingerprint = ""
            try:
                from C3_activity_monitoring.src.geo_context import get_geo_context, haversine_km
                geo = get_geo_context()
                geo_city = geo.get("city") or "Unknown"
                geo_country = geo.get("country") or "Unknown"
                geo_region = geo.get("region") or "Unknown"
                geo_ip = geo.get("ip") or "unknown"
                geo_lat = geo.get("lat")
                geo_lon = geo.get("lon")
                geo_timezone = geo.get("timezone") or "Unknown"
                geo_isp = geo.get("isp") or "Unknown"
                geo_org = geo.get("org") or "Unknown"
                geo_asn = geo.get("asn") or "Unknown"
                geo_confidence = float(geo.get("confidence") or 0.0)
                geo_hint = geo.get("location_hint") or "Unknown"
                geo_source = str(geo.get("source") or "unknown")
                vpn_proxy_detected = bool(geo.get("is_proxy", False))
                hosting_detected = bool(geo.get("is_hosting", False))

                work_location = str(self._current_employee.get("work_location") or "").strip().lower()

                # Compare employee estimated location with configured office geofence.
                policy_doc = None
                try:
                    settings_col = self._db.get_collection("system_settings")
                    if settings_col is not None:
                        policy_doc = settings_col.find_one({"_id": "geo_policy"}) or {}
                except Exception:
                    policy_doc = {}

                office = (policy_doc or {}).get("office") or {}
                risk_cfg = (policy_doc or {}).get("risk") or {}

                wifi_ssid_hash = self._get_wifi_ssid_hash()
                office_wifi_hash = str(
                    office.get("wifi_ssid_hash")
                    or office.get("ssid_hash")
                    or ""
                ).strip()
                if wifi_ssid_hash and office_wifi_hash:
                    wifi_ssid_match = wifi_ssid_hash == office_wifi_hash

                try:
                    office_radius_km = float(office.get("radius_km", 25.0) or 25.0)
                except Exception:
                    office_radius_km = 25.0

                distance_km = haversine_km(
                    geo_lat,
                    geo_lon,
                    office.get("lat"),
                    office.get("lon"),
                )
                if distance_km is not None:
                    geolocation_deviation = float(distance_km)
                    inside_office_geofence = geolocation_deviation <= office_radius_km
                    geolocation_resolved = True

                # Set location mode based on resolved geofence facts first.
                if inside_office_geofence is True:
                    location_mode = "office"
                    # Backward-compatible fallback until office Wi-Fi hash is configured.
                    if not office_wifi_hash and wifi_ssid_hash:
                        wifi_ssid_match = True
                elif inside_office_geofence is False:
                    if work_location in {"home", "hybrid"}:
                        location_mode = work_location
                    else:
                        location_mode = "outside"
                else:
                    if work_location in {"office", "home", "hybrid"}:
                        location_mode = work_location
                    else:
                        location_mode = "unknown"

                strict_vpn_proxy = bool(risk_cfg.get("strict_vpn_proxy", True))
                outside_penalty = float(risk_cfg.get("outside_penalty", 20.0) or 20.0)
                vpn_penalty = float(risk_cfg.get("vpn_proxy_penalty", 25.0) or 25.0)
                hosting_penalty = float(risk_cfg.get("hosting_penalty", 20.0) or 20.0)

                trust = float(geo_confidence) * 100.0
                if inside_office_geofence is False:
                    trust -= outside_penalty
                if vpn_proxy_detected:
                    trust -= vpn_penalty if strict_vpn_proxy else (vpn_penalty * 0.5)
                if hosting_detected:
                    trust -= hosting_penalty
                if geo_city == "Unknown":
                    trust -= 15.0
                location_trust_score = round(max(0.0, min(trust, 100.0)), 2)
            except Exception:
                pass

            device_fingerprint = self._get_device_fp()

            # Create session document
            session_doc = {
                "session_id": self._session_id,
                "employee_id": emp_id,
                "login_at": datetime.now(timezone.utc).isoformat(),
                "location_mode": location_mode,
                "city": geo_city,
                "country": geo_country,
                "region": geo_region,
                "ip": geo_ip,
                "lat": geo_lat,
                "lon": geo_lon,
                "timezone": geo_timezone,
                "isp": geo_isp,
                "org": geo_org,
                "asn": geo_asn,
                "location_confidence": geo_confidence,
                "location_hint": geo_hint,
                "vpn_proxy_detected": vpn_proxy_detected,
                "hosting_detected": hosting_detected,
                "geo_source": geo_source,
                "geolocation_deviation": geolocation_deviation,
                "inside_office_geofence": inside_office_geofence,
                "geolocation_resolved": geolocation_resolved,
                "office_radius_km": office_radius_km,
                "location_trust_score": location_trust_score,
                "wifi_ssid_hash": wifi_ssid_hash,
                "wifi_ssid_match": wifi_ssid_match,
                "device_fingerprint": device_fingerprint,
                "face_liveness_score": liveness_score,
                "mfa_verified": True,
                "status": "active",
            }

            # Keep resolved geo data on employee payload for main.py callback.
            self._current_employee["location_mode"] = location_mode
            self._current_employee["geo_city"] = geo_city
            self._current_employee["geo_country"] = geo_country
            self._current_employee["geo_region"] = geo_region
            self._current_employee["geo_ip"] = geo_ip
            self._current_employee["geo_lat"] = geo_lat
            self._current_employee["geo_lon"] = geo_lon
            self._current_employee["geo_timezone"] = geo_timezone
            self._current_employee["geo_isp"] = geo_isp
            self._current_employee["geo_org"] = geo_org
            self._current_employee["geo_asn"] = geo_asn
            self._current_employee["geo_confidence"] = geo_confidence
            self._current_employee["geo_hint"] = geo_hint
            self._current_employee["geo_source"] = geo_source
            self._current_employee["vpn_proxy_detected"] = vpn_proxy_detected
            self._current_employee["hosting_detected"] = hosting_detected
            self._current_employee["geolocation_deviation"] = geolocation_deviation
            self._current_employee["inside_office_geofence"] = inside_office_geofence
            self._current_employee["geolocation_resolved"] = geolocation_resolved
            self._current_employee["office_radius_km"] = office_radius_km
            self._current_employee["location_trust_score"] = location_trust_score
            self._current_employee["wifi_ssid_hash"] = wifi_ssid_hash
            self._current_employee["wifi_ssid_match"] = wifi_ssid_match

            col = self._db.get_collection("sessions")
            if col is not None:
                # Ensure only one active session per employee.
                col.update_many(
                    {
                        "employee_id": emp_id,
                        "status": "active",
                        "session_id": {"$ne": self._session_id},
                    },
                    {
                        "$set": {
                            "status": "ended",
                            "logout_at": datetime.now(timezone.utc).isoformat(),
                            "ended_reason": "superseded_by_new_login",
                        }
                    },
                )
                col.insert_one(session_doc)

            # Log auth event
            self._log_auth_event("login_success", emp_id, success=True, extra={"liveness_score": liveness_score})

            # Clear face_attempts and locked_until
            emp_col = self._db.get_collection("employees")
            if emp_col is not None:
                emp_col.update_one(
                    {"employee_id": emp_id},
                    {"$unset": {"locked_until": ""}, "$set": {"last_login": datetime.utcnow().isoformat()}},
                )

            # Call success callback
            self.after(0, lambda: self._on_success(self._current_employee, self._session_id))

        except Exception as exc:
            logger.error("Session creation error: %s", exc)
            self.after(0, lambda: self._on_success(self._current_employee, self._session_id))

    def _log_auth_event(self, event_type: str, emp_id: str, success: bool, extra: dict = None) -> None:
        try:
            col = self._db.get_collection("auth_events")
            if col is not None:
                col.insert_one({
                    "event_type":  event_type,
                    "employee_id": emp_id,
                    "session_id":  self._session_id,
                    "timestamp":   datetime.utcnow().isoformat(),
                    "success":     success,
                    **(extra or {}),
                })
        except Exception as exc:
            logger.debug("Auth event log error: %s", exc)

    def _get_device_fp(self) -> str:
        import hashlib, platform, uuid as _uuid
        parts = [platform.node(), platform.machine(), str(_uuid.getnode())]
        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def _get_wifi_ssid_hash(self) -> str:
        """Return a short hash of the currently connected Wi-Fi SSID (best effort)."""
        import hashlib
        import platform
        import subprocess

        ssid = ""
        try:
            os_name = platform.system()
            if os_name == "Windows":
                result = subprocess.run(
                    ["netsh", "wlan", "show", "interfaces"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for line in result.stdout.splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    k = key.strip().lower()
                    if k == "ssid" and "bssid" not in k:
                        candidate = value.strip()
                        if candidate and candidate.lower() != "n/a":
                            ssid = candidate
                            break
            elif os_name == "Darwin":
                result = subprocess.run(
                    [
                        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                        "-I",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for line in result.stdout.splitlines():
                    if "SSID:" in line and "BSSID" not in line:
                        ssid = line.split(":", 1)[1].strip()
                        break
            else:
                result = subprocess.run(
                    ["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                for line in result.stdout.splitlines():
                    if line.startswith("yes:"):
                        ssid = line.split(":", 1)[1].strip()
                        break
        except Exception:
            ssid = ""

        if not ssid:
            return ""
        return hashlib.sha256(ssid.encode("utf-8", errors="ignore")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, color: str = C_RED) -> None:
        self._status_var.set(msg)
        self._status_lbl.configure(text_color=color)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def launch_login(
    db: MongoDBClient,
    on_success: Callable[[dict, str], None],
    alert_sender: Optional[AlertSender] = None,
) -> LoginWindow:
    win = LoginWindow(db=db, on_success=on_success, alert_sender=alert_sender)
    return win


if __name__ == "__main__":
    db = MongoDBClient(uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME)
    db.connect()

    def _on_success(emp, session_id):
        print(f"Login SUCCESS: {emp['employee_id']} session={session_id}")

    win = launch_login(db, _on_success)
    win.mainloop()
