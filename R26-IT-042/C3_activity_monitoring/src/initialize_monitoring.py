"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/initialize_monitoring.py

Orchestrates all C3 sub-trackers: keyboard, mouse, app usage,
idle detection, anomaly engine, activity logger, break manager,
and screenshot trigger.

Called by main.py in a background thread. Runs until shutdown_event is set.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def start_monitoring(
    user_id: str,
    db_client=None,
    alert_sender=None,
    shutdown_event: Optional[threading.Event] = None,
    session_id: Optional[str] = None,
    location_mode: str = "unknown",
    location_context: Optional[dict] = None,
    wifi_ssid_match: bool = False,
    face_liveness_score: float = 1.0,
) -> Optional[BreakManager]:
    """
    Start all C3 activity monitoring sub-components.

    Parameters
    ----------
    user_id:
        Employee identifier.
    db_client:
        MongoDBClient instance from common/database.py.
    alert_sender:
        AlertSender instance from common/alerts.py.
    shutdown_event:
        threading.Event — monitored to exit all loops cleanly.
    session_id:
        Current session UUID. Generated if not provided.
    location_mode:
        "office" | "home" | "unknown"
    location_context:
        Optional approximate location context (city/country/isp/confidence).
    wifi_ssid_match:
        Whether the current WiFi SSID matches the known office network.
    face_liveness_score:
        Liveness score from C2 (passed at login).
    """
    logger.info("C3 activity monitoring starting for user: %s", user_id)

    if shutdown_event is None:
        shutdown_event = threading.Event()

    if session_id is None:
        session_id = str(uuid.uuid4())

    session_start = time.perf_counter()

    # ── Import sub-components ─────────────────────────────────────────
    try:
        from C3_activity_monitoring.src.keyboard_tracker import KeyboardTracker
        from C3_activity_monitoring.src.mouse_tracker import MouseTracker
        from C3_activity_monitoring.src.app_usage_monitor import AppUsageMonitor
        from C3_activity_monitoring.src.idle_detector import IdleDetector
        from C3_activity_monitoring.src.feature_extractor import FeatureExtractor
        from C3_activity_monitoring.src.anomaly_engine import AnomalyEngine
        from C3_activity_monitoring.src.offline_queue import OfflineQueue
        from C3_activity_monitoring.src.activity_logger import ActivityLogger
        from C3_activity_monitoring.src.screenshot_trigger import ScreenshotTrigger
        from C3_activity_monitoring.src.break_manager import BreakManager
    except ImportError as exc:
        logger.error("C3 sub-component import failed: %s", exc)
        shutdown_event.wait()
        return

    # ── Instantiate components ────────────────────────────────────────
    keyboard = KeyboardTracker(window_sec=30.0)
    mouse = MouseTracker(window_sec=30.0)
    app = AppUsageMonitor(window_sec=30.0)

    def _persist_alert(level: str, risk_score: float, factors: list[str], extra: Optional[dict] = None) -> None:
        """Persist alert documents so Admin Panel alert feed can display them."""
        if db_client is None or not getattr(db_client, "is_connected", False):
            return
        try:
            alerts_col = db_client.get_collection("alerts")
            if alerts_col is None:
                return
            alerts_col.insert_one(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "risk_score": float(risk_score),
                    "level": level,
                    "factors": factors,
                    **(extra or {}),
                }
            )
        except Exception as exc:
            logger.warning("Could not persist alert document: %s", exc)

    def _on_idle_detected(idle_duration_sec: float) -> None:
        logger.warning("Employee idle for %.0fs", idle_duration_sec)
        factors = ["idle_timeout", f"idle_{int(idle_duration_sec)}s"]

        # Send via websocket channel (if configured)
        if alert_sender is not None:
            try:
                alert_sender.send_alert(
                    user_id=user_id,
                    risk_score=55.0,
                    factors=factors,
                    level="MEDIUM",
                    session_id=session_id,
                    extra={"reason": "idle_inactivity", "idle_duration_sec": round(idle_duration_sec, 1)},
                )
            except Exception as exc:
                logger.warning("Idle alert websocket send failed: %s", exc)

        # Persist for Admin Panel Alerts tab
        _persist_alert(
            level="MEDIUM",
            risk_score=55.0,
            factors=factors,
            extra={"reason": "idle_inactivity", "idle_duration_sec": round(idle_duration_sec, 1)},
        )

    def _on_idle_resume() -> None:
        logger.info("Employee resumed activity after idle period.")

    idle = IdleDetector(
        threshold_sec=120,
        check_interval=5.0,
        window_sec=60.0,
        on_idle=_on_idle_detected,
        on_resume=_on_idle_resume,
    )

    # Try to load break manager early so employee break controls remain usable
    # even if anomaly model startup is blocked.
    break_mgr = None
    try:
        break_mgr = BreakManager(
            trackers=(keyboard, mouse, app),
            db_client=db_client,
            alert_sender=alert_sender,
            user_id=user_id,
        )
        break_mgr.load_breaks()
    except Exception as exc:
        logger.warning("BreakManager init error (non-fatal): %s", exc)

    offline_queue = OfflineQueue()
    anomaly_engine = AnomalyEngine()
    model_loaded = anomaly_engine.load_model()
    if not model_loaded:
        logger.error(
            "Anomaly model failed to load for user=%s session=%s. "
            "Fail-closed mode enabled; C3 monitoring startup aborted.",
            user_id,
            session_id,
        )
        _persist_alert(
            level="HIGH",
            risk_score=100.0,
            factors=["model_unavailable_startup", "monitoring_blocked"],
            extra={
                "reason": "anomaly_model_unavailable",
                "session_id": session_id,
            },
        )
        if break_mgr is not None:
            try:
                _bm_thread = threading.Thread(
                    target=break_mgr._run_loop,
                    args=(shutdown_event,),
                    daemon=True,
                    name="BreakManager-Loop",
                )
                _bm_thread.start()
            except Exception as exc:
                logger.warning("BreakManager loop start error: %s", exc)
        # Fail closed for C3 only; do not stop the whole app runtime.
        return break_mgr

    # Wire keyboard and mouse activity → idle detector
    _orig_kb_on_press = keyboard._listener  # patched after start

    extractor = FeatureExtractor(
        keyboard=keyboard,
        mouse=mouse,
        app=app,
        idle=idle,
        user_id=user_id,
        session_id=session_id,
        session_start=session_start,
        location_mode=location_mode,
        location_context=location_context,
        wifi_ssid_match=wifi_ssid_match,
        face_liveness_score=face_liveness_score,
    )

    # Screenshot trigger
    screenshot_trigger = None
    try:
        from common.encryption import AESEncryptor
        encryptor = AESEncryptor()
        screenshot_trigger = ScreenshotTrigger(db_client=db_client, encryptor=encryptor)
    except Exception as exc:
        logger.warning("ScreenshotTrigger init error (non-fatal): %s", exc)

    activity_logger = ActivityLogger(
        feature_extractor=extractor,
        anomaly_engine=anomaly_engine,
        db_client=db_client,
        offline_queue=offline_queue,
        user_id=user_id,
        session_id=session_id,
        alert_sender=alert_sender,
        break_manager=break_mgr,
        screenshot_trigger=screenshot_trigger,
    )

    # ── Start all components ──────────────────────────────────────────
    # Wire activity hooks before starting
    keyboard._on_activity = idle.record_activity
    mouse._on_activity    = idle.record_activity

    keyboard.start()
    mouse.start()
    app.start(shutdown_event=shutdown_event)
    idle.start(shutdown_event=shutdown_event)

    # Keyboard and mouse both record to idle via polling their data
    activity_logger.start(shutdown_event=shutdown_event)

    # Run break manager if available
    if break_mgr is not None:
        try:
            _bm_thread = threading.Thread(
                target=break_mgr._run_loop,
                args=(shutdown_event,),
                daemon=True,
                name="BreakManager-Loop",
            )
            _bm_thread.start()
        except Exception as exc:
            logger.warning("BreakManager loop start error: %s", exc)

    # ── Flush offline queue on reconnect ──────────────────────────────
    def _flush_loop():
        while not shutdown_event.is_set():
            if offline_queue.size > 0 and offline_queue.is_online() and db_client and db_client.is_connected:
                col = db_client.get_collection("activity_logs")
                if col is not None:
                    offline_queue.flush(col)
            shutdown_event.wait(timeout=60.0)

    threading.Thread(target=_flush_loop, daemon=True, name="OfflineQueueFlusher").start()

    def _teardown_when_stopped() -> None:
        shutdown_event.wait()

        keyboard.stop()
        mouse.stop()
        app.stop()
        idle.stop()
        if break_mgr is not None:
            try:
                break_mgr.pause_monitoring()
            except Exception:
                pass

        if db_client and db_client.is_connected and offline_queue.size > 0:
            col = db_client.get_collection("activity_logs")
            if col is not None:
                offline_queue.flush(col)

        logger.info("C3 activity monitoring stopped for user: %s", user_id)

    threading.Thread(
        target=_teardown_when_stopped,
        daemon=True,
        name="C3-ShutdownWatcher",
    ).start()

    logger.info("C3: all sub-components running for user %s (session=%s)", user_id, session_id)
    return break_mgr
