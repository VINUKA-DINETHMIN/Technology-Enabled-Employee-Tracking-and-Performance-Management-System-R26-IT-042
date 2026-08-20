"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/activity_logger.py

ActivityLogger — 60-second loop that:
  1. Calls FeatureExtractor.extract() to get the full 27-field vector
  2. Runs AnomalyEngine to get composite_risk_score
  3. Builds a complete MongoDB activity_logs document
  4. Encrypts the feature_vector field with AES-256
  5. Signs the document with HMAC
  6. Saves to MongoDB (or OfflineQueue if offline)
  7. Triggers a screenshot if risk >= 75 for 2 consecutive windows

MongoDB document schema
───────────────────────
{
  "timestamp": ISO str,
  "user_id": str,
  "session_id": str,
  "feature_vector": <encrypted bytes>,
  "composite_risk_score": float (0-100),
  "productivity_score": float (0-100),
  "alert_triggered": bool,
  "contributing_factors": list[str],
  "label": "normal" | "low_risk_anomaly" | "high_risk_anomaly",
  "location_mode": str,
  "in_break": bool,
  "break_type": str | null,
  "encrypted": true,
  "hmac_signature": str
}
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from C3_activity_monitoring.src.feature_extractor import FeatureExtractor
    from C3_activity_monitoring.src.anomaly_engine import AnomalyEngine
    from C3_activity_monitoring.src.offline_queue import OfflineQueue

# Risk score thresholds
_SOFT_WARN = 50.0
_HARD_WARN = 75.0

# Consecutive high-risk windows before screenshot is triggered
_SCREENSHOT_CONSECUTIVE_THRESHOLD = 2

# Log interval in seconds
_LOG_INTERVAL = 60.0

# Unproductive apps/sites list
_UNPRODUCTIVE_APPS = ["youtube", "netflix", "facebook", "instagram", "tiktok", "gaming", "steam"]

# Low-activity mouse thresholds used for explicit anomaly factors.
_LOW_MOUSE_VELOCITY = 80.0
_LOW_MOUSE_CLICK_FREQUENCY = 5.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _risk_to_label(score: float) -> str:
    if score < _SOFT_WARN:
        return "normal"
    if score < _HARD_WARN:
        return "low_risk_anomaly"
    return "high_risk_anomaly"


def _risk_to_productivity(
    score: float,
    idle_ratio: float,
    typing_speed: float,
) -> float:
    """
    Heuristic productivity score (0-100).
    Higher typing speed + lower idle + lower risk → higher productivity.
    """
    base = 100.0 - score
    idle_penalty = idle_ratio * 30.0
    typing_bonus = min(typing_speed / 60.0 * 10.0, 10.0)  # cap at 10
    return round(max(0.0, min(100.0, base - idle_penalty + typing_bonus)), 2)


def _get_contributing_factors(fv: dict, risk_score: float) -> list[str]:
    """Return a list of human-readable factor labels from the feature vector."""
    factors = []
    if fv.get("idle_ratio", 0.0) > 0.5:
        factors.append("high_idle_ratio")
    if fv.get("typing_speed_wpm", 0.0) < 5.0:
        factors.append("very_low_typing_speed")
    if fv.get("error_rate", 0.0) > 0.3:
        factors.append("high_error_rate")
    if fv.get("app_switch_frequency", 0.0) > 20.0:
        factors.append("rapid_app_switching")
    if fv.get("active_app_entropy", 0.0) < 0.3:
        factors.append("low_app_entropy")
    if not fv.get("wifi_ssid_match", True):
        factors.append("unknown_wifi_network")
    if not fv.get("device_fingerprint_match", True):
        factors.append("unknown_device")
    if fv.get("face_liveness_score", 1.0) < 0.5:
        factors.append("low_liveness_score")
    if fv.get("geolocation_deviation", 0.0) > 50.0:
        factors.append("unusual_location")
    if fv.get("inside_office_geofence") is False:
        factors.append("outside_office_geofence")
    if fv.get("vpn_proxy_detected", False):
        factors.append("vpn_or_proxy_detected")
    if fv.get("hosting_detected", False):
        factors.append("hosting_network_detected")
    if float(fv.get("location_trust_score", 100.0) or 100.0) < 50.0:
        factors.append("low_location_trust")
    if fv.get("top_app", "").lower() in _UNPRODUCTIVE_APPS:
        factors.append("unproductive_app_usage")
    if fv.get("active_task_id") is None and fv.get("typing_speed_wpm", 0.0) > 20:
        factors.append("off_task_activity")
    
    # ── Mouse Specific Factors ─────────────────────────────────────────
    if fv.get("click_frequency", 0.0) > 100.0:
        factors.append("abnormal_click_frequency")
    if fv.get("mean_curvature", 0.0) > 0.8:
        factors.append("erratic_mouse_movement")
    if fv.get("mean_velocity", 0.0) > 1500.0:
        factors.append("high_velocity_movement")
    if (
        _safe_float(fv.get("mean_velocity", 0.0), 0.0) < _LOW_MOUSE_VELOCITY
        and _safe_float(fv.get("click_frequency", 0.0), 0.0) < _LOW_MOUSE_CLICK_FREQUENCY
    ):
        factors.append("low_mouse_movement")

    if risk_score >= _HARD_WARN:
        factors.append("high_composite_risk")
    return factors


def _fallback_risk_score(fv: dict) -> float:
    """Heuristic risk score used only when the ML model is unavailable."""
    score = 0.0

    if fv.get("idle_ratio", 0.0) > 0.5:
        score += 18.0
    if fv.get("typing_speed_wpm", 0.0) < 5.0:
        score += 18.0
    if fv.get("error_rate", 0.0) > 0.3:
        score += 8.0
    if fv.get("app_switch_frequency", 0.0) > 20.0:
        score += 10.0
    if fv.get("active_app_entropy", 0.0) < 0.3:
        score += 8.0
    if not fv.get("wifi_ssid_match", True):
        score += 8.0
    if not fv.get("device_fingerprint_match", True):
        score += 8.0
    if fv.get("face_liveness_score", 1.0) < 0.5:
        score += 10.0
    if fv.get("geolocation_deviation", 0.0) > 50.0:
        score += 8.0
    if fv.get("inside_office_geofence") is False:
        score += 10.0
    if fv.get("vpn_proxy_detected", False):
        score += 10.0
    if fv.get("hosting_detected", False):
        score += 8.0
    if float(fv.get("location_trust_score", 100.0) or 100.0) < 50.0:
        score += 8.0
    if fv.get("top_app", "").lower() in _UNPRODUCTIVE_APPS:
        score += 8.0
    if fv.get("active_task_id") is None and fv.get("typing_speed_wpm", 0.0) > 20:
        score += 8.0
    if fv.get("click_frequency", 0.0) > 100.0:
        score += 10.0
    if fv.get("mean_curvature", 0.0) > 0.8:
        score += 8.0
    if fv.get("mean_velocity", 0.0) > 1500.0:
        score += 8.0
    if (
        _safe_float(fv.get("mean_velocity", 0.0), 0.0) < _LOW_MOUSE_VELOCITY
        and _safe_float(fv.get("click_frequency", 0.0), 0.0) < _LOW_MOUSE_CLICK_FREQUENCY
    ):
        score += 8.0

    return round(max(0.0, min(100.0, score)), 2)


def _geo_risk_adjustment(fv: dict) -> float:
    """Extra risk penalties for location mismatch and suspicious network context."""
    penalty = 0.0

    try:
        deviation = float(fv.get("geolocation_deviation", 0.0) or 0.0)
    except Exception:
        deviation = 0.0

    if fv.get("inside_office_geofence") is False:
        if deviation > 150.0:
            penalty += 20.0
        elif deviation > 50.0:
            penalty += 12.0
        else:
            penalty += 8.0

    if fv.get("vpn_proxy_detected", False):
        penalty += 20.0
    if fv.get("hosting_detected", False):
        penalty += 12.0

    try:
        trust = float(fv.get("location_trust_score", 100.0) or 100.0)
    except Exception:
        trust = 100.0

    if trust < 30.0:
        penalty += 10.0
    elif trust < 50.0:
        penalty += 6.0

    return penalty


class ActivityLogger:
    """
    Runs a 60-second periodic loop that logs activity feature vectors
    to MongoDB with encryption, HMAC signing, and screenshot triggering.

    Usage
    ─────
    >>> logger_obj = ActivityLogger(
    ...     feature_extractor=extractor,
    ...     anomaly_engine=engine,
    ...     db_client=db,
    ...     offline_queue=queue,
    ...     user_id="EMP001",
    ...     session_id="sess-xyz",
    ...     alert_sender=sender,
    ... )
    >>> logger_obj.start(shutdown_event)
    """

    def __init__(
        self,
        feature_extractor: "FeatureExtractor",
        anomaly_engine: "AnomalyEngine",
        db_client,
        offline_queue: "OfflineQueue",
        user_id: str,
        session_id: str,
        alert_sender=None,
        break_manager=None,
        screenshot_trigger=None,
        log_interval: float = _LOG_INTERVAL,
    ) -> None:
        self._extractor = feature_extractor
        self._engine = anomaly_engine
        self._db = db_client
        self._queue = offline_queue
        self._user_id = user_id
        self._session_id = session_id
        self._alert_sender = alert_sender
        self._break_manager = break_manager
        self._screenshot_trigger = screenshot_trigger
        self._log_interval = log_interval
        self._high_risk_consecutive = 0
        self._model_guard_alert_sent = False

        # Lazy-load encryptor
        self._encryptor = None
        self._thread: Optional[threading.Thread] = None
        
        # ThreadPool for async DB writes (non-blocking)
        self._db_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ActivityLogDB")

    def start(self, shutdown_event: Optional[threading.Event] = None) -> None:
        """Start the 60-second logging loop in a daemon thread."""
        self._thread = threading.Thread(
            target=self._log_loop,
            args=(shutdown_event or threading.Event(),),
            daemon=True,
            name="ActivityLogger",
        )
        self._thread.start()
        logger.info("ActivityLogger started (interval=%ds).", int(self._log_interval))

    def stop(self) -> None:
        logger.info("ActivityLogger stop requested.")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _log_loop(self, shutdown_event: threading.Event) -> None:
        while not shutdown_event.is_set():
            try:
                self._do_log()
            except Exception as exc:
                logger.error("ActivityLogger loop error: %s", exc)
            shutdown_event.wait(timeout=self._log_interval)

    def _do_log(self) -> None:
        self._maybe_reload_model()

        # ── Determine break context ───────────────────────────────────
        in_break = False
        break_type = None
        if self._break_manager is not None:
            try:
                in_break = self._break_manager.is_in_break()
                break_type = self._break_manager.get_active_break()
            except Exception:
                pass

        # ── Skip logging during break (no data collected) ─────────────
        if in_break:
            logger.debug("In break — skipping activity log.")
            return

        # ── Extract feature vector ────────────────────────────────────
        fv = self._extractor.extract(in_break=in_break, break_type=break_type)

        # ── Find Active Task ──────────────────────────────────────────
        active_task_id = None
        active_task_title = None
        try:
            if self._db and self._db.is_connected:
                col = self._db.get_collection("tasks")
                task = col.find_one({"employee_id": self._user_id, "status": "in_progress"})
                if task:
                    active_task_id = task.get("task_id")
                    active_task_title = task.get("title")
        except Exception:
            pass
        
        fv["active_task_id"] = active_task_id
        fv["active_task_title"] = active_task_title

        # ── Score anomaly ─────────────────────────────────────────────
        model_loaded = bool(self._engine.is_loaded)
        if not model_loaded:
            self._warn_model_unavailable_once()
            logger.error(
                "Fail-closed: skipping activity log for user=%s session=%s because anomaly model is unavailable.",
                self._user_id,
                self._session_id,
            )
            return

        model_score = None
        risk_score = 0.0
        import numpy as np
        numeric_fields = [
            "mean_dwell_time", "std_dwell_time", "mean_flight_time",
            "typing_speed_wpm", "error_rate", "mean_velocity", "std_velocity",
            "mean_acceleration", "mean_curvature", "click_frequency",
            "idle_ratio", "app_switch_frequency", "active_app_entropy",
            "total_focus_duration", "session_duration_min", "geolocation_deviation",
            "wifi_ssid_match", "device_fingerprint_match", "face_liveness_score",
        ]
        arr = np.array([_safe_float(fv.get(f, 0.0), 0.0) for f in numeric_fields], dtype=np.float32)
        model_score = float(self._engine.score(arr))
        risk_score = model_score

        # Add deterministic policy penalties only when model score is available.
        # This prevents silent geo-only fallback patterns (for example fixed 8.0)
        # if the model failed to load in a running process.
        risk_score = max(0.0, min(100.0, risk_score + _geo_risk_adjustment(fv)))

        model_error = None

        productivity_score = _risk_to_productivity(
            risk_score,
            fv.get("idle_ratio", 0.0),
            fv.get("typing_speed_wpm", 0.0),
        )
        # Extra penalty for unproductive apps
        if fv.get("top_app", "").lower() in _UNPRODUCTIVE_APPS:
            productivity_score = max(0.0, productivity_score - 40.0)

        # ── Determine contributing factors and label ───────────────────
        factors = _get_contributing_factors(fv, risk_score)
        label = _risk_to_label(risk_score)
        alert_triggered = risk_score >= _HARD_WARN

        # ── Encrypt feature vector ────────────────────────────────────
        enc = self._get_encryptor()
        fv_json = json.dumps(fv, ensure_ascii=False, default=str)
        if enc is not None:
            encrypted_fv = enc.encrypt(fv_json).decode("utf-8")
        else:
            encrypted_fv = fv_json  # unencrypted fallback

        # ── Build document ────────────────────────────────────────────
        geo_resolved = bool(fv.get("geolocation_resolved", False))
        geo_dev_doc = None
        if geo_resolved:
            geo_dev_doc = round(_safe_float(fv.get("geolocation_deviation", 0.0), 0.0), 3)

        doc = {
            "timestamp": fv["timestamp"],
            "user_id": self._user_id,
            "session_id": self._session_id,
            "feature_vector": encrypted_fv,
            "composite_risk_score": round(risk_score, 2),
            "anomaly_model_loaded": model_loaded,
            "anomaly_model_score": None if model_score is None else round(_safe_float(model_score, 0.0), 2),
            "anomaly_model_error": model_error,
            "productivity_score": productivity_score,
            "idle_ratio": round(_safe_float(fv.get("idle_ratio", 0.0), 0.0), 4),
            "app_switch_frequency": round(_safe_float(fv.get("app_switch_frequency", 0.0), 0.0), 3),
            "active_app_entropy": round(_safe_float(fv.get("active_app_entropy", 0.0), 0.0), 4),
            "total_focus_duration": round(_safe_float(fv.get("total_focus_duration", 0.0), 0.0), 2),
            "alert_triggered": alert_triggered,
            "contributing_factors": factors,
            "label": label,
            "location_mode": fv.get("location_mode", "unknown"),
            "city": fv.get("geo_city", "Unknown"),
            "region": fv.get("geo_region", "Unknown"),
            "country": fv.get("geo_country", "Unknown"),
            "timezone": fv.get("geo_timezone", "Unknown"),
            "isp": fv.get("geo_isp", "Unknown"),
            "org": fv.get("geo_org", "Unknown"),
            "asn": fv.get("geo_asn", "Unknown"),
            "geo_source": fv.get("geo_source", "unknown"),
            "lat": fv.get("geo_lat"),
            "lon": fv.get("geo_lon"),
            "location_confidence": round(_safe_float(fv.get("geo_confidence", 0.0), 0.0), 2),
            "location_hint": fv.get("location_hint", "Unknown"),
            "geolocation_deviation": geo_dev_doc,
            "inside_office_geofence": fv.get("inside_office_geofence"),
            "geolocation_resolved": geo_resolved,
            "vpn_proxy_detected": bool(fv.get("vpn_proxy_detected", False)),
            "hosting_detected": bool(fv.get("hosting_detected", False)),
            "location_trust_score": round(_safe_float(fv.get("location_trust_score", 0.0), 0.0), 2),
            "office_radius_km": round(_safe_float(fv.get("office_radius_km", 0.0), 0.0), 2),
            "in_break": in_break,
            "break_type": break_type,
            "active_task_id": active_task_id,
            "active_task_title": active_task_title,
            "top_app": fv.get("top_app"),
            "encrypted": enc is not None,
        }

        # ── HMAC sign ─────────────────────────────────────────────────
        if enc is not None:
            doc["hmac_signature"] = enc.hmac_sign(
                json.dumps(doc, sort_keys=True, default=str)
            )
        else:
            doc["hmac_signature"] = ""

        # ── Save to MongoDB or offline queue ──────────────────────────
        self._save_document(doc)

        # ── Alert ─────────────────────────────────────────────────────
        if alert_triggered and self._alert_sender is not None:
            try:
                self._alert_sender.send_alert(
                    user_id=self._user_id,
                    risk_score=risk_score,
                    factors=factors,
                    session_id=self._session_id,
                )
            except Exception as exc:
                logger.warning("Alert send error: %s", exc)

        # ── Screenshot trigger (2 consecutive high-risk windows) ──────
        if risk_score >= _HARD_WARN:
            self._high_risk_consecutive += 1
        else:
            self._high_risk_consecutive = 0

        if (
            self._high_risk_consecutive >= _SCREENSHOT_CONSECUTIVE_THRESHOLD
            and self._screenshot_trigger is not None
        ):
            try:
                self._screenshot_trigger.capture(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    risk_score=risk_score,
                )
                self._high_risk_consecutive = 0
            except Exception as exc:
                logger.warning("Screenshot trigger error: %s", exc)

        logger.info(
            "ActivityLog saved — user=%s risk=%.1f label=%s",
            self._user_id, risk_score, label,
        )

    def _maybe_reload_model(self) -> None:
        """Periodically retry model loading when engine is currently unavailable."""
        try:
            if self._engine.is_loaded:
                return
            if self._engine.load_model():
                logger.info("Anomaly model reload succeeded for user=%s", self._user_id)
                # Reset one-time guard so future genuine outages are still reported.
                self._model_guard_alert_sent = False
                return

            # If the existing engine instance is in a bad state, try a fresh engine object.
            try:
                from C3_activity_monitoring.src.anomaly_engine import AnomalyEngine

                fresh_engine = AnomalyEngine()
                if fresh_engine.load_model():
                    self._engine = fresh_engine
                    logger.info(
                        "Anomaly model hard-reload with fresh engine succeeded for user=%s",
                        self._user_id,
                    )
                    self._model_guard_alert_sent = False
            except Exception as inner_exc:
                logger.debug("Anomaly model hard-reload failed: %s", inner_exc)
        except Exception as exc:
            logger.debug("Anomaly model reload retry failed: %s", exc)

    def _warn_model_unavailable_once(self) -> None:
        """Emit a one-time warning/alert when anomaly model is unavailable."""
        if self._model_guard_alert_sent:
            return
        self._model_guard_alert_sent = True
        logger.error(
            "Anomaly model unavailable for user=%s session=%s; risk fallback blocked.",
            self._user_id,
            self._session_id,
        )
        if self._alert_sender is not None:
            try:
                self._alert_sender.send_alert(
                    user_id=self._user_id,
                    risk_score=0.0,
                    factors=["model_unavailable"],
                    session_id=self._session_id,
                )
            except Exception as exc:
                logger.warning("Model-unavailable alert send error: %s", exc)

    def _save_document(self, doc: dict) -> None:
        """Save document to MongoDB or enqueue offline (async background)."""
        # Submit to thread pool so DB write doesn't block the 60-sec logging loop
        self._db_executor.submit(self._save_document_blocking, doc)
    
    def _save_document_blocking(self, doc: dict) -> None:
        """Blocking DB save (runs in background thread pool)."""
        if self._queue.is_online() and self._db is not None and self._db.is_connected:
            col = self._db.get_collection("activity_logs")
            if col is not None:
                try:
                    col.insert_one(doc)
                    return
                except Exception as exc:
                    logger.warning("MongoDB insert error — queuing offline: %s", exc)
        self._queue.enqueue(doc)

    def _get_encryptor(self):
        if self._encryptor is not None:
            return self._encryptor
        try:
            from common.encryption import AESEncryptor
            self._encryptor = AESEncryptor()
        except Exception as exc:
            logger.warning("AESEncryptor unavailable: %s", exc)
        return self._encryptor
