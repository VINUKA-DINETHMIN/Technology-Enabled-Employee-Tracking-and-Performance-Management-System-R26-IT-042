"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/feature_extractor.py

FeatureExtractor — Aggregates outputs from KeyboardTracker, MouseTracker,
AppUsageMonitor, and IdleDetector every 60 seconds into one complete
27-field feature vector dict for anomaly scoring and MongoDB storage.

Feature schema (27 fields)
──────────────────────────
  timestamp, user_id, session_id, location_mode, in_break, break_type,
  mean_dwell_time, std_dwell_time, mean_flight_time, typing_speed_wpm,
  error_rate, mean_velocity, std_velocity, mean_acceleration,
  mean_curvature, click_frequency, idle_ratio, app_switch_frequency,
  active_app_entropy, total_focus_duration, session_duration_min,
  hour_of_day, day_of_week, geolocation_deviation, wifi_ssid_match,
  device_fingerprint_match, face_liveness_score
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from C3_activity_monitoring.src.keyboard_tracker import KeyboardTracker
    from C3_activity_monitoring.src.mouse_tracker import MouseTracker
    from C3_activity_monitoring.src.app_usage_monitor import AppUsageMonitor
    from C3_activity_monitoring.src.idle_detector import IdleDetector


def _device_fingerprint() -> str:
    """Return a stable, non-PII device fingerprint hash."""
    parts = [
        platform.node(),
        platform.machine(),
        platform.processor(),
        str(uuid.getnode()),  # MAC address integer
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FeatureExtractor:
    """
    Aggregates outputs from all 4 trackers into one feature vector per call.

    Usage
    ─────
    >>> extractor = FeatureExtractor(
    ...     keyboard=kb_tracker,
    ...     mouse=mouse_tracker,
    ...     app=app_monitor,
    ...     idle=idle_detector,
    ...     user_id="EMP001",
    ...     session_id="sess-xyz",
    ... )
    >>> feature_vector = extractor.extract()
    """

    def __init__(
        self,
        keyboard: "KeyboardTracker",
        mouse: "MouseTracker",
        app: "AppUsageMonitor",
        idle: "IdleDetector",
        user_id: str,
        session_id: str,
        session_start: Optional[float] = None,
        location_mode: str = "unknown",
        location_context: Optional[dict] = None,
        wifi_ssid_match: bool = False,
        face_liveness_score: float = 0.0,
    ) -> None:
        self._keyboard = keyboard
        self._mouse = mouse
        self._app = app
        self._idle = idle
        self._user_id = user_id
        self._session_id = session_id
        self._session_start = session_start or time.perf_counter()
        self._location_mode = location_mode
        self._location_context = dict(location_context or {})
        self._wifi_ssid_match = wifi_ssid_match
        self._face_liveness_score = face_liveness_score
        self._device_fp = _device_fingerprint()
        self._known_fp: Optional[str] = None  # loaded from MongoDB on first call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        in_break: bool = False,
        break_type: Optional[str] = None,
        geolocation_deviation: Optional[float] = None,
    ) -> dict:
        """
        Produce one complete 27-field feature vector.

        Parameters
        ----------
        in_break:
            Whether the employee is currently in a break period.
        break_type:
            "lunch" | "short" | None
        geolocation_deviation:
            Distance from known location in km (None if not supplied).

        Returns
        -------
        dict
            All 27 feature fields plus timestamp metadata.
        """
        now_utc = datetime.now(timezone.utc)
        now_perf = time.perf_counter()

        kb_feat = self._keyboard.get_features()
        ms_feat = self._mouse.get_features()
        app_feat = self._app.get_features()

        session_elapsed_min = (now_perf - self._session_start) / 60.0
        idle_ratio = self._idle.get_idle_ratio()

        # Device fingerprint match (True if same as registered device)
        device_fp_match = (
            self._known_fp is None or self._device_fp == self._known_fp
        )

        geo_dev: Optional[float] = None
        try:
            if geolocation_deviation not in (None, ""):
                geo_dev = float(geolocation_deviation)
        except Exception:
            geo_dev = None

        if geo_dev is None:
            try:
                ctx_geo_dev = self._location_context.get("geolocation_deviation")
                if ctx_geo_dev not in (None, ""):
                    geo_dev = float(ctx_geo_dev)
            except Exception:
                geo_dev = None

        if geo_dev is None:
            geo_dev = 0.0

        feature_vector = {
            # ── Metadata ───────────────────────────────────────────
            "timestamp": now_utc.isoformat(),
            "user_id": self._user_id,
            "session_id": self._session_id,
            "location_mode": self._location_mode,
            "geo_city": self._location_context.get("city", "Unknown"),
            "geo_region": self._location_context.get("region", "Unknown"),
            "geo_country": self._location_context.get("country", "Unknown"),
            "geo_timezone": self._location_context.get("timezone", "Unknown"),
            "geo_isp": self._location_context.get("isp", "Unknown"),
            "geo_org": self._location_context.get("org", "Unknown"),
            "geo_asn": self._location_context.get("asn", "Unknown"),
            "geo_source": self._location_context.get("geo_source", "unknown"),
            "geo_lat": self._location_context.get("lat"),
            "geo_lon": self._location_context.get("lon"),
            "geo_confidence": float(self._location_context.get("confidence", 0.0) or 0.0),
            "location_hint": self._location_context.get("location_hint", "Unknown"),
            "inside_office_geofence": self._location_context.get("inside_office_geofence"),
            "geolocation_resolved": bool(self._location_context.get("geolocation_resolved", False)),
            "vpn_proxy_detected": bool(self._location_context.get("vpn_proxy_detected", False)),
            "hosting_detected": bool(self._location_context.get("hosting_detected", False)),
            "location_trust_score": float(self._location_context.get("location_trust_score", 0.0) or 0.0),
            "office_radius_km": float(self._location_context.get("office_radius_km", 0.0) or 0.0),
            "in_break": in_break,
            "break_type": break_type,
            # ── Keyboard features ──────────────────────────────────
            "mean_dwell_time": kb_feat["mean_dwell_time"],
            "std_dwell_time": kb_feat["std_dwell_time"],
            "mean_flight_time": kb_feat["mean_flight_time"],
            "typing_speed_wpm": kb_feat["typing_speed_wpm"],
            "error_rate": kb_feat["error_rate"],
            # ── Mouse features ─────────────────────────────────────
            "mean_velocity": ms_feat["mean_velocity"],
            "std_velocity": ms_feat["std_velocity"],
            "mean_acceleration": ms_feat["mean_acceleration"],
            "mean_curvature": ms_feat["mean_curvature"],
            "click_frequency": ms_feat["click_frequency"],
            "idle_ratio": idle_ratio,
            # ── App features ───────────────────────────────────────
            "app_switch_frequency": app_feat["app_switch_frequency"],
            "active_app_entropy": app_feat["active_app_entropy"],
            "total_focus_duration": app_feat["total_focus_duration"],
            "top_app": app_feat.get("top_app"),  # ← FIX: Add missing app tracking
            # ── Session / temporal context ─────────────────────────
            "session_duration_min": round(session_elapsed_min, 2),
            "hour_of_day": now_utc.hour,
            "day_of_week": now_utc.weekday(),  # 0=Monday … 6=Sunday
            # ── Environmental context ──────────────────────────────
            "geolocation_deviation": geo_dev,
            "wifi_ssid_match": self._wifi_ssid_match,
            "device_fingerprint_match": device_fp_match,
            "face_liveness_score": self._face_liveness_score,
        }

        return feature_vector

    def update_liveness_score(self, score: float) -> None:
        """Update the face liveness score (called after periodic re-check)."""
        self._face_liveness_score = score

    def update_location(self, mode: str, wifi_match: bool) -> None:
        """Update location context (office/home/unknown)."""
        self._location_mode = mode
        self._wifi_ssid_match = wifi_match

    def update_location_context(self, context: Optional[dict]) -> None:
        """Update approximate geo context used in activity logs."""
        self._location_context = dict(context or {})

    def set_known_device_fp(self, fp: str) -> None:
        """Set the registered device fingerprint for matching."""
        self._known_fp = fp
