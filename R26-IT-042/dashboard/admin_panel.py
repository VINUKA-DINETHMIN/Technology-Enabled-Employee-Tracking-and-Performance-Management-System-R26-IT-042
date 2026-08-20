"""
R26-IT-042 — Employee Activity Monitoring System
dashboard/admin_panel.py

Full CustomTkinter Admin Panel with sidebar navigation:
  - Dashboard   : Live employee overview, color-coded risk scores
  - Alerts      : Real-time WebSocket feed, alert management
  - Tasks       : Task assignment with tkcalendar date picker
  - Attendance  : Date-filterable attendance log
  - Settings    : Application configuration

Run standalone:
    python dashboard/admin_panel.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import re
import asyncio
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Path bootstrap ────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk

try:
    from tkcalendar import DateEntry
    _HAS_CALENDAR = True
except ImportError:
    _HAS_CALENDAR = False

from common.database import MongoDBClient
from common.alerts import AlertSender
from config.settings import settings

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False

# ── Appearance ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Color palette
C_BG        = "#0b0e17"
C_SIDEBAR   = "#0f1420"
C_CARD      = "#151b2d"
C_BORDER    = "#1e2a40"
C_TEAL      = "#14b8a6"
C_TEAL_D    = "#0d9488"
C_RED       = "#ef4444"
C_AMBER     = "#f59e0b"
C_GREEN     = "#22c55e"
C_TEXT      = "#e2e8f0"
C_MUTED     = "#64748b"
C_BLUE      = "#3b82f6"

POLL_INTERVAL_MS = 10_000   # 10 seconds
ONLINE_SESSION_MAX_AGE_HOURS = 16

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _risk_color(score: float) -> str:
    if score < 50:
        return C_GREEN
    if score < 75:
        return C_AMBER
    return C_RED


def _level_color(level: str) -> str:
    return {
        "LOW": "#6366f1",
        "MEDIUM": C_AMBER,
        "HIGH": C_RED,
        "CRITICAL": "#dc2626",
    }.get(level.upper(), C_MUTED)


def _play_alert_sound() -> None:
    """Play system beep for CRITICAL alerts (cross-platform)."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif sys.platform == "darwin":
            os.system("afplay /System/Library/Sounds/Funk.aiff &")
    except Exception:
        pass


def _parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse an ISO timestamp string and return a timezone-aware local datetime."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone()
    except Exception:
        return None


def _clip_text(text: str, max_chars: int) -> str:
    """Clip long values to keep table columns visually aligned."""
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 3)] + "..."


def _fmt_time(ts_str: str) -> str:
    """Format ISO timestamp to HH:MM string."""
    dt = _parse_iso_timestamp(ts_str)
    if dt is not None:
        return dt.strftime("%H:%M")
    return ts_str[:5] if ts_str else "--"


def _fmt_last_seen(ts_str: str) -> str:
    """Format last-seen text similar to chat apps (today/yesterday/date + time)."""
    dt = _parse_iso_timestamp(ts_str)
    if dt is None:
        return "—"

    now_local = datetime.now(dt.tzinfo)
    day_diff = (now_local.date() - dt.date()).days
    time_text = dt.strftime("%I:%M %p").lstrip("0")

    if day_diff == 0:
        return f"Today at {time_text}"
    if day_diff == 1:
        return f"Yesterday at {time_text}"
    return f"{dt.strftime('%d %b %Y')} at {time_text}"


def _fmt_minutes(total_minutes: int) -> str:
    """Format minutes into compact human-readable text."""
    minutes = max(0, int(total_minutes or 0))
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins}m"


def _fmt_seconds(total_seconds: int) -> str:
    """Format seconds into compact human-readable text."""
    seconds = max(0, int(total_seconds or 0))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {mins:02d}m"
    return f"{mins:02d}m {secs:02d}s"


def _is_hhmm(value: str) -> bool:
    """Validate HH:MM (24-hour) text."""
    return bool(re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", str(value or "").strip()))


def _combine_due_datetime(due_date: str, due_time: str) -> str:
    """Combine date and time into ISO-like datetime string (local naive)."""
    try:
        dt = datetime.strptime(f"{due_date} {due_time}", "%Y-%m-%d %H:%M")
        return dt.isoformat()
    except Exception:
        return ""


def _parse_due_datetime_local(due_date: str, due_time: str) -> Optional[datetime]:
    """Parse due date/time in local time; return None when invalid."""
    try:
        return datetime.strptime(f"{due_date} {due_time}", "%Y-%m-%d %H:%M")
    except Exception:
        return None


def _fmt_due_display(due_date: str, due_time: str) -> str:
    """Format due date/time as Today/Tomorrow/or full date with time."""
    date_text = str(due_date or "").strip()
    time_text = str(due_time or "").strip()

    if not date_text and not time_text:
        return "—"

    try:
        due_d = datetime.strptime(date_text, "%Y-%m-%d").date()
        today = datetime.now().date()
        day_delta = (due_d - today).days

        if day_delta == 0:
            prefix = "Today"
        elif day_delta == 1:
            prefix = "Tomorrow"
        elif day_delta == -1:
            prefix = "Yesterday"
        else:
            prefix = due_d.strftime("%d %b %Y")

        if _is_hhmm(time_text):
            return f"{prefix} at {time_text}"
        return prefix
    except Exception:
        if date_text and time_text:
            return f"{date_text} {time_text}"
        return date_text or time_text


def _due_color(due_date: str, status: str) -> str:
    """Return color for due date urgency in task table."""
    if status == "completed":
        return C_MUTED
    date_text = str(due_date or "").strip()
    try:
        due_d = datetime.strptime(date_text, "%Y-%m-%d").date()
        today = datetime.now().date()
        if due_d < today:
            return C_RED
        if due_d == today:
            return C_AMBER
        return C_MUTED
    except Exception:
        return C_MUTED


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Summary Card Widget
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SummaryCard(ctk.CTkFrame):
    def __init__(self, parent, title: str, value: str, accent: str = C_TEAL, **kw):
        super().__init__(parent, fg_color=C_CARD, corner_radius=14, **kw)
        ctk.CTkLabel(
            self, text=title,
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
        ).pack(anchor="w", padx=16, pady=(14, 2))
        self._val = ctk.CTkLabel(
            self, text=value,
            font=ctk.CTkFont(size=28, weight="bold"), text_color=accent,
        )
        self._val.pack(anchor="w", padx=16, pady=(0, 14))

    def set_value(self, val: str) -> None:
        self._val.configure(text=val)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Employee Detail Window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class EmployeeDetailWindow(ctk.CTkToplevel):
    def __init__(self, parent, employee: dict, db: MongoDBClient):
        super().__init__(parent)
        self._db = db
        self._emp = employee
        self._emp_id = employee.get("employee_id", "?")
        self._detail_closed = False
        self._ss_poll_after_id = None
        self._screenshot_widgets = {}  # {timestamp: widget_frame} - Track existing screenshot widgets
        emp_id = employee.get("employee_id", "?")
        name = employee.get("full_name", emp_id)

        self.title(f"Employee Detail — {name}")
        self.geometry("780x700")
        self.configure(fg_color=C_BG)
        self.attributes("-topmost", True)

        # Header
        hdr = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=0, height=64)
        hdr.pack(fill="x")
        ctk.CTkLabel(
            hdr, text=f"{name}  •  {emp_id}",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=C_TEXT,
        ).pack(side="left", padx=20, pady=16)

        body = ctk.CTkScrollableFrame(self, fg_color=C_BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)

        # Risk score
        risk_doc = self._latest_activity(emp_id)
        risk = risk_doc.get("composite_risk_score", 0.0) if risk_doc else 0.0
        risk_color = _risk_color(risk)
        if not risk_doc:
            score_source = "Unknown"
            score_source_color = C_MUTED
        elif bool(risk_doc.get("anomaly_model_loaded")) and risk_doc.get("anomaly_model_score") is not None:
            score_source = "ML"
            score_source_color = C_GREEN
        else:
            score_source = "Fallback"
            score_source_color = C_AMBER

        rframe = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        rframe.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(rframe, text="Composite Risk Score", font=ctk.CTkFont(size=12), text_color=C_MUTED).pack(anchor="w", padx=16, pady=(12, 0))
        ctk.CTkLabel(rframe, text=f"{risk:.1f} / 100", font=ctk.CTkFont(size=32, weight="bold"), text_color=risk_color).pack(anchor="w", padx=16)
        progress = ctk.CTkProgressBar(rframe, height=10, progress_color=risk_color, fg_color=C_BORDER)
        progress.set(risk / 100.0)
        progress.pack(fill="x", padx=16, pady=(4, 12))
        ctk.CTkLabel(
            rframe,
            text=f"Scoring Source: {score_source}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=score_source_color,
        ).pack(anchor="w", padx=16, pady=(0, 12))

        # Currently Active Task & Top App
        if risk_doc:
            status_frame = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
            status_frame.pack(fill="x", pady=(0, 12))
            
            top_app = risk_doc.get("top_app") or "None"
            is_unproductive = top_app.lower() in ["youtube", "netflix", "facebook", "instagram", "tiktok", "gaming", "steam"]
            app_color = C_RED if is_unproductive else C_GREEN
            
            ctk.CTkLabel(status_frame, text="Current App Focus:", font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(anchor="w", padx=16, pady=(12, 0))
            ctk.CTkLabel(status_frame, text=top_app.upper(), font=ctk.CTkFont(size=14, weight="bold"), text_color=app_color).pack(anchor="w", padx=16)

            # --- New Detailed App Metrics ---
            metrics_row = ctk.CTkFrame(status_frame, fg_color="transparent")
            metrics_row.pack(fill="x", padx=16, pady=(8, 0))
            
            def m_box(parent, label, val, color=C_TEXT):
                f = ctk.CTkFrame(parent, fg_color=C_BORDER, corner_radius=6)
                f.pack(side="left", expand=True, fill="both", padx=2)
                ctk.CTkLabel(f, text=label, font=ctk.CTkFont(size=9), text_color=C_MUTED).pack(pady=(4, 0))
                ctk.CTkLabel(f, text=val, font=ctk.CTkFont(size=11, weight="bold"), text_color=color).pack(pady=(0, 4))

            sw_freq = risk_doc.get("app_switch_frequency", 0.0)
            entropy = risk_doc.get("active_app_entropy", 0.0)
            duration = risk_doc.get("total_focus_duration", 0.0)

            m_box(metrics_row, "SWITCH RATE", f"{sw_freq:.1f}/min", C_AMBER if sw_freq > 15 else C_TEXT)
            m_box(metrics_row, "FOCUS ENTROPY", f"{entropy:.2f}", C_RED if entropy > 2.0 else C_TEXT)
            m_box(metrics_row, "FOCUS TIME", f"{duration:.0f}s")

            active_task = risk_doc.get("active_task_title", "No Active Task")
            ctk.CTkLabel(status_frame, text="Active Working Task:", font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(anchor="w", padx=16, pady=(8, 0))
            ctk.CTkLabel(status_frame, text=active_task, font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=16, pady=(0, 12))
            
            prod = risk_doc.get("productivity_score", 0.0)
            ctk.CTkLabel(status_frame, text="Current Productivity:", font=ctk.CTkFont(size=11), text_color=C_MUTED).pack(anchor="w", padx=16)
            ctk.CTkLabel(status_frame, text=f"{prod:.0f}%", font=ctk.CTkFont(size=18, weight="bold"), text_color=_risk_color(100-prod)).pack(anchor="w", padx=16, pady=(0, 12))

        # Contributing factors
        if risk_doc:
            factors = risk_doc.get("contributing_factors", [])
            if factors:
                ff = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
                ff.pack(fill="x", pady=(0, 12))
                ctk.CTkLabel(ff, text="Anomaly Factors", font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=16, pady=(12, 4))
                for f in factors:
                    f_name = f.replace('_', ' ').title()
                    f_color = C_RED if "unproductive" in f or "off_task" in f else C_AMBER
                    ctk.CTkLabel(ff, text=f"  • {f_name}", font=ctk.CTkFont(size=12, weight="bold" if f_color==C_RED else "normal"), text_color=f_color).pack(anchor="w", padx=16)
                ctk.CTkFrame(ff, fg_color="transparent", height=8).pack()

        # App Usage Analytics
        try:
            from dashboard.app_usage_tracker import AppUsageTrackerUI
            tracker = AppUsageTrackerUI(body, db_client=self._db, emp_id=emp_id)
            tracker.show()
        except Exception as exc:
            logger.debug("App usage tracker load error: %s", exc)

        # ─────────────────────────────────────────────────
        # Anti-Spoofing Verification
        # ─────────────────────────────────────────────────
        self._render_antispoofing_check(body, emp_id)

        # Alert history
        alerts = self._get_alerts(emp_id)
        af = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        af.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(af, text=f"Recent Alerts ({len(alerts)})", font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=16, pady=(12, 4))
        for a in alerts[:5]:
            row = ctk.CTkFrame(af, fg_color=C_BORDER, corner_radius=8)
            row.pack(fill="x", padx=16, pady=3)
            lvl = a.get("level", "LOW")
            ctk.CTkLabel(row, text=f"[{lvl}]", text_color=_level_color(lvl), font=ctk.CTkFont(size=11, weight="bold"), width=60).pack(side="left", padx=8, pady=6)
            ctk.CTkLabel(row, text=", ".join(a.get("factors", [])) or "—", text_color=C_TEXT, font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(row, text=_fmt_time(a.get("timestamp", "")), text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(side="right", padx=8)
        ctk.CTkFrame(af, fg_color="transparent", height=8).pack()

        # Activity logs (recent windows)
        activity_logs = self._get_activity_logs(emp_id)
        total_activity_logs = self._get_activity_log_count(emp_id)
        lgf = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        lgf.pack(fill="x", pady=(0, 12))
        logs_header = ctk.CTkFrame(lgf, fg_color="transparent")
        logs_header.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            logs_header,
            text=f"Recent Activity Logs ({len(activity_logs)} shown of {total_activity_logs})",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=C_TEXT,
        ).pack(side="left")

        ctk.CTkButton(
            logs_header,
            text="Delete Excess",
            width=120,
            height=28,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._delete_excess_activity_logs(emp_id, keep_latest=200),
        ).pack(side="right")

        if not activity_logs:
            ctk.CTkLabel(
                lgf,
                text="No activity logs found for this employee.",
                text_color=C_MUTED,
                font=ctk.CTkFont(size=11),
            ).pack(anchor="w", padx=16, pady=(0, 10))
        else:
            for item in activity_logs[:12]:
                row = ctk.CTkFrame(lgf, fg_color=C_BORDER, corner_radius=8)
                row.pack(fill="x", padx=16, pady=3)

                risk = float(item.get("composite_risk_score", 0.0) or 0.0)
                prod = float(item.get("productivity_score", 0.0) or 0.0)
                lbl = str(item.get("label", "normal"))
                top_app = str(item.get("top_app") or "-")
                idle_ratio = float(item.get("idle_ratio", 0.0) or 0.0)

                ctk.CTkLabel(
                    row,
                    text=_fmt_time(item.get("timestamp", "")),
                    text_color=C_MUTED,
                    font=ctk.CTkFont(size=11),
                    width=65,
                ).pack(side="left", padx=(8, 4), pady=6)
                ctk.CTkLabel(
                    row,
                    text=f"Risk {risk:.1f}",
                    text_color=_risk_color(risk),
                    font=ctk.CTkFont(size=11, weight="bold"),
                    width=85,
                ).pack(side="left", padx=4)
                ctk.CTkLabel(
                    row,
                    text=f"Prod {prod:.0f}%",
                    text_color=C_TEXT,
                    font=ctk.CTkFont(size=11),
                    width=80,
                ).pack(side="left", padx=4)
                ctk.CTkLabel(
                    row,
                    text=f"Idle {idle_ratio:.2f}",
                    text_color=C_TEXT,
                    font=ctk.CTkFont(size=11),
                    width=75,
                ).pack(side="left", padx=4)
                ctk.CTkLabel(
                    row,
                    text=f"App {top_app}",
                    text_color=C_TEXT,
                    font=ctk.CTkFont(size=11),
                ).pack(side="left", padx=6)
                ctk.CTkLabel(
                    row,
                    text=lbl.replace("_", " ").title(),
                    text_color=C_AMBER if "risk" in lbl else C_GREEN,
                    font=ctk.CTkFont(size=11),
                ).pack(side="right", padx=(4, 8))

                ctk.CTkButton(
                    row,
                    text="View",
                    width=52,
                    height=24,
                    fg_color=C_SIDEBAR,
                    hover_color=C_BLUE,
                    font=ctk.CTkFont(size=10),
                    command=lambda d=item: self._show_activity_log_detail(d),
                ).pack(side="right", padx=4)

                ctk.CTkButton(
                    row,
                    text="Delete",
                    width=58,
                    height=24,
                    fg_color="#7f1d1d",
                    hover_color="#991b1b",
                    font=ctk.CTkFont(size=10, weight="bold"),
                    command=lambda d=item: self._delete_activity_log(emp_id, d),
                ).pack(side="right", padx=4)

        ctk.CTkFrame(lgf, fg_color="transparent", height=8).pack()

        # Tasks assigned
        tasks = self._get_tasks(emp_id)
        tf = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        tf.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(tf, text=f"Assigned Tasks ({len(tasks)})", font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=16, pady=(12, 4))
        for t in tasks[:5]:
            row = ctk.CTkFrame(tf, fg_color=C_BORDER, corner_radius=8)
            row.pack(fill="x", padx=16, pady=3)
            status_color = {"pending": C_MUTED, "in_progress": C_AMBER, "completed": C_GREEN, "paused": C_RED}.get(t.get("status", ""), C_MUTED)
            ctk.CTkLabel(row, text=t.get("title", "?"), text_color=C_TEXT, font=ctk.CTkFont(size=11)).pack(side="left", padx=8, pady=6)
            ctk.CTkLabel(row, text=t.get("status", "").replace("_", " ").title(), text_color=status_color, font=ctk.CTkFont(size=11)).pack(side="right", padx=8)
        ctk.CTkFrame(tf, fg_color="transparent", height=8).pack()

        # Screenshots list
        self._ssf = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=12)
        self._ssf.pack(fill="x", pady=(0, 12))
        self._render_screenshots(self._ssf, emp_id)

        # Action Buttons
        ctrl_frame = ctk.CTkFrame(body, fg_color="transparent")
        ctrl_frame.pack(fill="x", pady=20)
        
        ctk.CTkButton(ctrl_frame, text="Resend MFA Email", fg_color=C_TEAL, hover_color=C_TEAL_D, height=38, 
                      command=self._resend_mfa).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(ctrl_frame, text="Force Screenshot", fg_color="#7c3aed", hover_color="#6d28d9", height=38, 
                      command=lambda: self._force_screenshot(emp_id)).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(ctrl_frame, text="Live Camera", fg_color=C_RED, hover_color="#b91c1c", height=38, 
                      command=lambda: LiveCamViewer(self, emp_id, self._db)).pack(side="left", expand=True, padx=4)
        ctk.CTkButton(ctrl_frame, text="Live Screen", fg_color="#3b82f6", hover_color="#2563eb", height=38, 
                      command=lambda: LiveScreenViewer(self, emp_id, self._db)).pack(side="left", expand=True, padx=4)

    def _schedule_screenshot_refresh(self, initial_delay_ms: int = 10000) -> None:  # Increased from 1500 to 10000
        if self._detail_closed:
            return
        try:
            if self._ss_poll_after_id is not None:
                self.after_cancel(self._ss_poll_after_id)
        except Exception:
            pass
        self._schedule_screenshot_refresh(initial_delay_ms=10000)  # Poll every 10 seconds instead of 3

    def _on_close_detail(self) -> None:
        self._detail_closed = True
        try:
            if self._ss_poll_after_id is not None:
                self.after_cancel(self._ss_poll_after_id)
        except Exception:
            pass
        self.destroy()

    def _schedule_screenshot_refresh(self, initial_delay_ms: int = 10000) -> None:  # Increased from 3000 to 10000
        if self._detail_closed:
            return
        try:
            if self._ss_poll_after_id is not None:
                self.after_cancel(self._ss_poll_after_id)
        except Exception:
            pass
        self._ss_poll_after_id = self.after(initial_delay_ms, self._poll_screenshots)

    def _poll_screenshots(self) -> None:
        if self._detail_closed or not self.winfo_exists():
            return
        try:
            self._render_screenshots(self._ssf, self._emp_id)
        except Exception:
            pass
        self._ss_poll_after_id = self.after(10000, self._poll_screenshots)  # Poll every 10 seconds instead of 3

    def _force_screenshot(self, emp_id: str) -> None:
        if not self._db or not self._db.is_connected: return
        import uuid
        try:
            col = self._db.get_collection("commands")
            if col is not None:
                now = datetime.utcnow()
                expires = (now + timedelta(minutes=5)).isoformat()
                col.insert_one({
                    "command_id": str(uuid.uuid4()),
                    "target_user_id": emp_id,
                    "command_type": "force_screenshot",
                    "status": "pending",
                    "timestamp": now.isoformat(),
                    "expires_at": expires
                })

            # Trigger quick UI refresh attempts so new screenshot appears as soon as written.
            self._schedule_screenshot_refresh(initial_delay_ms=800)
        except Exception: pass

    def _resend_mfa(self) -> None:
        from common.email_utils import send_mfa_setup_email
        email = self._emp.get("email")
        name = self._emp.get("full_name")
        mfa_secret = self._emp.get("mfa_secret")
        if not email or not mfa_secret:
            messagebox.showerror("Error", "Missing email or MFA secret.")
            return
        if send_mfa_setup_email(email, name, mfa_secret):
            messagebox.showinfo("Success", f"MFA email sent to {email}.")
        else:
            messagebox.showerror("Failed", "Check SMTP settings in .env.")

    def _render_screenshots(self, parent_frame: ctk.CTkFrame, emp_id: str) -> None:
        """Fetch and render screenshot list rows with delta updates."""
        # Get current screenshots
        screens = self._get_screenshots(emp_id)
        current_timestamps = {s.get("timestamp", "") for s in screens[:5]}
        existing_timestamps = set(self._screenshot_widgets.keys())

        # Remove widgets for screenshots that no longer exist
        for ts in existing_timestamps - current_timestamps:
            if ts in self._screenshot_widgets:
                try:
                    self._screenshot_widgets[ts].destroy()
                except Exception:
                    pass
                del self._screenshot_widgets[ts]

        # Ensure header exists (only create once)
        if not hasattr(self, '_ss_header_label'):
            self._ss_header_label = ctk.CTkLabel(parent_frame, text="Recent Screenshots", font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT)
            self._ss_header_label.pack(anchor="w", padx=16, pady=(12, 4))

        # Handle empty state
        if not screens:
            if not hasattr(self, '_ss_empty_label'):
                self._ss_empty_label = ctk.CTkLabel(parent_frame, text="No screenshots captured yet.", font=ctk.CTkFont(size=11), text_color=C_MUTED)
                self._ss_empty_label.pack(anchor="w", padx=16, pady=(0, 12))
        else:
            # Remove empty label if it exists
            if hasattr(self, '_ss_empty_label'):
                try:
                    self._ss_empty_label.destroy()
                except Exception:
                    pass
                delattr(self, '_ss_empty_label')

            # Add/update screenshot widgets
            for s in screens[:5]:
                ts = s.get("timestamp", "")
                if ts not in self._screenshot_widgets:
                    # Create new widget
                    row = ctk.CTkFrame(parent_frame, fg_color=C_BORDER, corner_radius=8)
                    row.pack(fill="x", padx=16, pady=3)
                    self._create_screenshot_row_content(row, s, emp_id)
                    self._screenshot_widgets[ts] = row
                # Existing widgets are not updated (timestamps don't change, so no need to refresh)

        # Ensure footer spacing
        if not hasattr(self, '_ss_footer_frame'):
            self._ss_footer_frame = ctk.CTkFrame(parent_frame, fg_color="transparent", height=8)
            self._ss_footer_frame.pack()

    def _create_screenshot_row_content(self, row: ctk.CTkFrame, s: dict, emp_id: str) -> None:
        """Create the content for a screenshot row widget."""
        reason = s.get("trigger_reason", "manual").title()
        risk = s.get("risk_score_at_capture", 0.0)
        ctk.CTkLabel(row, text=f"📸 {reason}", text_color=C_TEXT, font=ctk.CTkFont(size=11)).pack(side="left", padx=8, pady=6)
        ctk.CTkLabel(row, text=f"Risk: {risk:.0f}", text_color=_risk_color(risk), font=ctk.CTkFont(size=11)).pack(side="left", padx=12)

        # Buttons
        path = s.get("file_path") or s.get("image_path") or ""
        b64 = s.get("image_base64") or s.get("thumbnail_base64")

        # Delete (🗑️)
        ctk.CTkButton(row, text="🗑️", width=32, height=24, fg_color="#450a0a", hover_color=C_RED,
                      font=ctk.CTkFont(size=12), command=lambda sc=s: self._delete_screenshot(sc)).pack(side="right", padx=(8, 4))

        # Folder Opener
        if path:
            ctk.CTkButton(row, text="📁", width=28, height=24, fg_color=C_SIDEBAR, hover_color=C_BORDER,
                          font=ctk.CTkFont(size=10), command=lambda p=path: self._open_file(p)).pack(side="right", padx=2)

        # View Local
        if path and os.path.exists(path):
            ctk.CTkButton(row, text="View Local", width=70, height=24, fg_color=C_BLUE, hover_color="#2563eb",
                          font=ctk.CTkFont(size=10), command=lambda p=path: ScreenshotViewer(self, p, is_path=True, user_id=emp_id)).pack(side="right", padx=4)

        # View Cloud
        if b64:
            ctk.CTkButton(row, text="View Cloud", width=70, height=24, fg_color=C_TEAL, hover_color=C_TEAL_D,
                          font=ctk.CTkFont(size=10), command=lambda b=b64: ScreenshotViewer(self, b)).pack(side="right", padx=4)

        ctk.CTkLabel(row, text=_fmt_time(s.get("timestamp", "")), text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(side="right", padx=4)

    def _delete_screenshot(self, screenshot: dict) -> None:
        """Confirm and delete a screenshot from DB and Disk."""
        msg = "Are you sure you want to delete this screenshot?\n\nThis will remove it from MongoDB and your local storage."
        if not messagebox.askyesno("Delete Screenshot", msg):
            return

        try:
            emp_id = screenshot.get("user_id")
            # 1. Delete from DB
            col = self._db.get_collection("screenshots")
            if col is not None:
                # We need the real _id. In _get_screenshots we excluded it... 
                # Let's check _get_screenshots implementation.
                # Actually, if we don't have _id, we use timestamp + user_id.
                res = col.delete_one({"user_id": emp_id, "timestamp": screenshot.get("timestamp")})
                if res.deleted_count == 0:
                    # Try by filename if path exists
                    path = screenshot.get("file_path") or screenshot.get("image_path")
                    if path:
                        col.delete_one({"file_path": path})
                        col.delete_one({"image_path": path})

            # 2. Delete from Filesystem
            path = screenshot.get("file_path") or screenshot.get("image_path")
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning("Failed to delete local file: %s", e)

            # 3. Refresh UI
            self._render_screenshots(self._ssf, emp_id)
            messagebox.showinfo("Deleted", "Screenshot successfully removed.")

        except Exception as exc:
            messagebox.showerror("Error", f"Deletion failed: {exc}")

    def _latest_activity(self, emp_id: str) -> Optional[dict]:
        try:
            col = self._db.get_collection("activity_logs")
            if col is not None:
                # Optimized: Only fetch essential fields for dashboard display
                projection = {
                    "_id": 0,
                    "composite_risk_score": 1,
                    "anomaly_model_loaded": 1,
                    "anomaly_model_score": 1,
                    "productivity_score": 1,
                    "top_app": 1,
                    "idle_ratio": 1,
                    "contributing_factors": 1,
                    "in_break": 1,
                    "active_task_title": 1,
                    "app_switch_frequency": 1,
                    "active_app_entropy": 1,
                    "total_focus_duration": 1,
                    "timestamp": 1
                }
                return col.find_one({"user_id": emp_id}, projection, sort=[("timestamp", -1)])
        except Exception: pass
        return None

    def _get_alerts(self, emp_id: str) -> list:
        try:
            col = self._db.get_collection("alerts")
            if col is not None:
                # Optimized: Only fetch essential fields for alert display
                projection = {
                    "_id": 0,
                    "level": 1,
                    "factors": 1,
                    "timestamp": 1,
                    "risk_score": 1
                }
                return list(col.find({"user_id": emp_id}, projection).sort("timestamp", -1).limit(10))
        except Exception: pass
        return []

    def _get_tasks(self, emp_id: str) -> list:
        try:
            col = self._db.get_collection("tasks")
            if col is not None:
                return list(col.find({"employee_id": emp_id}, {"_id": 0}).sort("assigned_at", -1).limit(10))
        except Exception: pass
        return []

    def _get_screenshots(self, emp_id: str) -> list:
        try:
            col = self._db.get_collection("screenshots")
            if col is not None:
                # Optimized: Only fetch essential fields, but keep both current and legacy field names.
                projection = {
                    "_id": 0,
                    "user_id": 1,
                    "timestamp": 1,
                    "file_path": 1,
                    "image_path": 1,
                    "image_base64": 1,
                    "thumbnail_base64": 1,
                    "metadata": 1
                }
                return list(col.find({"user_id": emp_id}, projection).sort("timestamp", -1).limit(5))  # Reduced from 10 to 5
        except Exception: pass
        return []

    def _render_antispoofing_check(self, parent, emp_id: str) -> None:
        """Render the anti-spoofing verification section."""
        try:
            asf = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=12)
            asf.pack(fill="x", pady=(0, 12))

            # Header with button
            header = ctk.CTkFrame(asf, fg_color="transparent")
            header.pack(fill="x", padx=16, pady=(12, 4))
            ctk.CTkLabel(
                header, text="Anti-Spoofing Verification",
                font=ctk.CTkFont(size=12, weight="bold"), text_color=C_TEXT,
            ).pack(side="left")

            # Trigger button
            trigger_btn = ctk.CTkButton(
                header,
                text="🔍 Verify Face",
                width=140, height=28,
                fg_color=C_TEAL, hover_color=C_TEAL_D,
                font=ctk.CTkFont(size=11, weight="bold"),
                command=lambda: self._trigger_antispoofing_check(emp_id),
            )
            trigger_btn.pack(side="right")
            self._antispoofing_btn = trigger_btn

            # Result container
            self._antispoofing_result_frame = ctk.CTkFrame(asf, fg_color="transparent")
            self._antispoofing_result_frame.pack(fill="x", padx=16, pady=(0, 12))

            # Load and display latest result
            self._update_antispoofing_display(asf, emp_id)

            ctk.CTkFrame(asf, fg_color="transparent", height=8).pack()

        except Exception as exc:
            logger.debug("Anti-spoofing UI render error: %s", exc)

    def _trigger_antispoofing_check(self, emp_id: str) -> None:
        """Send antispoofing check command to employee device."""
        try:
            col = self._db.get_collection("commands")
            if col is None:
                messagebox.showerror("Error", "Commands collection unavailable.")
                return

            cmd = {
                "command_id": str(uuid.uuid4()),
                "target_user_id": emp_id,
                "command_type": "start_antispoofing_check",
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
            }
            col.insert_one(cmd)

            # Show status message
            messagebox.showinfo(
                "Verification Initiated",
                f"Anti-spoofing check started for {emp_id}.\n\n"
                "The employee will see a camera prompt.\n"
                "Results will appear below in 10-15 seconds."
            )

            # Schedule refresh
            self.after(3000, lambda: self._update_antispoofing_display(
                self._antispoofing_result_frame, emp_id
            ))

        except Exception as exc:
            messagebox.showerror("Error", f"Failed to trigger check: {exc}")

    def _update_antispoofing_display(self, parent, emp_id: str) -> None:
        """Update antispoofing result display."""
        try:
            # Clear previous results
            for w in self._antispoofing_result_frame.winfo_children():
                w.destroy()

            col = self._db.get_collection("antispoofing_checks")
            if col is None:
                return

            result = col.find_one({"user_id": emp_id}, sort=[("timestamp", -1)])

            if result is None:
                ctk.CTkLabel(
                    self._antispoofing_result_frame,
                    text="No checks yet. Click 'Verify Face' to start.",
                    text_color=C_MUTED, font=ctk.CTkFont(size=11),
                ).pack(anchor="w", pady=4)
                return

            # Display result
            verdict = result.get("verdict", "UNKNOWN")
            is_real = result.get("is_real", False)
            confidence = float(result.get("confidence", 0.0) or 0.0)
            frames = int(result.get("frame_count", 0) or 0)
            avg_score = float(result.get("avg_score", 0.5) or 0.5)
            duration = float(result.get("check_duration_sec", 0.0) or 0.0)

            # Color based on verdict
            verdict_color = C_GREEN if is_real else C_RED
            verdict_icon = "✓ REAL" if is_real else "✗ FAKE"

            # Result box
            result_box = ctk.CTkFrame(self._antispoofing_result_frame, fg_color=C_BORDER, corner_radius=8)
            result_box.pack(fill="x", pady=4)

            # Verdict
            ctk.CTkLabel(
                result_box, text=verdict_icon,
                font=ctk.CTkFont(size=16, weight="bold"), text_color=verdict_color,
            ).pack(anchor="w", padx=12, pady=(8, 0))

            # Details
            details_text = (
                f"Confidence: {confidence:.1%} | "
                f"Frames: {frames} | "
                f"Duration: {duration:.1f}s"
            )
            ctk.CTkLabel(
                result_box, text=details_text,
                font=ctk.CTkFont(size=10), text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=(2, 4))

            identity_status = result.get("identity_status", "UNKNOWN")
            identity_score = float(result.get("identity_score", 0.0) or 0.0)
            # Keep display aligned with current policy even for older stored records.
            if identity_score >= 0.70 and identity_status in {"DIFFERENT_PERSON", "UNKNOWN"}:
                identity_status = "SAME_PERSON"
            identity_text = ""
            if identity_status == "SAME_PERSON":
                identity_text = f"Identity: Same person ({identity_score:.2f})"
            elif identity_status == "DIFFERENT_PERSON":
                identity_text = f"Identity: Different person ({identity_score:.2f})"
            elif identity_status == "NO_FACE_DETECTED":
                identity_text = "Identity: No person detected in front of camera"
            elif identity_status == "NO_TEMPLATE":
                identity_text = "Identity: No stored face template available"
            elif identity_status == "VERIFIER_UNAVAILABLE":
                identity_text = "Identity: Verifier unavailable"
            else:
                identity_text = "Identity: Unknown"

            ctk.CTkLabel(
                result_box, text=identity_text,
                font=ctk.CTkFont(size=10), text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=(0, 8))

            source_text = result.get("check_source") or result.get("source") or result.get("context")
            if source_text:
                ctk.CTkLabel(
                    result_box,
                    text=f"Source: {source_text}",
                    font=ctk.CTkFont(size=9),
                    text_color=C_MUTED,
                ).pack(anchor="w", padx=12, pady=(0, 4))

            # Timestamp
            ts_text = _fmt_time(result.get("timestamp", ""))
            ctk.CTkLabel(
                result_box, text=f"Last checked: {ts_text}",
                font=ctk.CTkFont(size=9), text_color=C_MUTED,
            ).pack(anchor="w", padx=12, pady=(0, 8))

        except Exception as exc:
            logger.debug("Update antispoofing display error: %s", exc)

    def _get_activity_logs(self, emp_id: str) -> list:
        try:
            col = self._db.get_collection("activity_logs")
            if col is not None:
                # Optimized: Time-filtered query (last 7 days) + essential fields only
                seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
                query = {
                    "user_id": emp_id,
                    "timestamp": {"$gte": seven_days_ago}
                }

                # Reduced projection: Only fields needed for UI display
                projection = {
                    "_id": 1,  # Keep for delete operations
                    "user_id": 1,
                    "session_id": 1,
                    "timestamp": 1,
                    "composite_risk_score": 1,
                    "anomaly_model_loaded": 1,
                    "anomaly_model_score": 1,
                    "productivity_score": 1,
                    "idle_ratio": 1,
                    "top_app": 1,
                    "label": 1,
                    "contributing_factors": 1,
                    "location_mode": 1,
                    "location_hint": 1,
                    "geo_source": 1,
                    "lat": 1,
                    "lon": 1,
                    "location_confidence": 1,
                    "geolocation_deviation": 1,
                    "inside_office_geofence": 1,
                    "geolocation_resolved": 1,
                    "location_trust_score": 1,
                    "vpn_proxy_detected": 1,
                    "hosting_detected": 1,
                    "isp": 1,
                    "city": 1,
                    "country": 1,
                    "in_break": 1,
                    "break_type": 1,
                    "alert_triggered": 1
                }
                return list(col.find(query, projection).sort("timestamp", -1).limit(20))  # Reduced from 50 to 20
        except Exception:
            pass
        return []

    def _get_activity_log_count(self, emp_id: str) -> int:
        try:
            col = self._db.get_collection("activity_logs")
            if col is not None:
                return int(col.count_documents({"user_id": emp_id}))
        except Exception:
            pass
        return 0

    def _delete_activity_log(self, emp_id: str, log_doc: dict) -> None:
        try:
            col = self._db.get_collection("activity_logs")
            if col is None:
                messagebox.showerror("Delete Activity Log", "Activity log collection is unavailable.")
                return

            ts = str(log_doc.get("timestamp") or "")
            confirm = messagebox.askyesno(
                "Delete Activity Log",
                f"Delete this activity log for {emp_id}?\n\nTime: {_fmt_time(ts)}\nThis action cannot be undone.",
            )
            if not confirm:
                return

            query = {"user_id": emp_id}
            if log_doc.get("_id") is not None:
                query["_id"] = log_doc.get("_id")
            elif ts:
                query["timestamp"] = ts

            result = col.delete_one(query)
            if result.deleted_count == 0:
                messagebox.showwarning("Delete Activity Log", "No matching log found. It may already be deleted.")
                return

            self.destroy()
            EmployeeDetailWindow(self.master, self._emp, self._db)

        except Exception as exc:
            messagebox.showerror("Delete Activity Log", f"Failed to delete log: {exc}")

    def _delete_excess_activity_logs(self, emp_id: str, keep_latest: int = 200) -> None:
        try:
            col = self._db.get_collection("activity_logs")
            if col is None:
                messagebox.showerror("Delete Excess Logs", "Activity log collection is unavailable.")
                return

            total_logs = int(col.count_documents({"user_id": emp_id}))
            if total_logs <= keep_latest:
                messagebox.showinfo(
                    "Delete Excess Logs",
                    f"No excess logs to delete for {emp_id}.\nCurrent logs: {total_logs}\nRetention limit: {keep_latest}",
                )
                return

            delete_count = total_logs - keep_latest
            confirm = messagebox.askyesno(
                "Delete Excess Logs",
                (
                    f"Delete {delete_count} old activity logs for {emp_id}?\n\n"
                    f"This will keep the newest {keep_latest} logs and permanently remove older records."
                ),
            )
            if not confirm:
                return

            cutoff_doc = col.find({"user_id": emp_id}, {"timestamp": 1, "_id": 0}).sort("timestamp", -1).skip(keep_latest).limit(1)
            cutoff_list = list(cutoff_doc)
            if not cutoff_list:
                messagebox.showinfo("Delete Excess Logs", "No cutoff point found. Nothing was deleted.")
                return

            cutoff_ts = cutoff_list[0].get("timestamp")
            if not cutoff_ts:
                messagebox.showerror("Delete Excess Logs", "Could not determine timestamp cutoff.")
                return

            result = col.delete_many({"user_id": emp_id, "timestamp": {"$lt": cutoff_ts}})
            messagebox.showinfo(
                "Delete Excess Logs",
                f"Deleted {result.deleted_count} old activity logs for {emp_id}.",
            )

            self.destroy()
            EmployeeDetailWindow(self.master, self._emp, self._db)

        except Exception as exc:
            messagebox.showerror("Delete Excess Logs", f"Failed to delete logs: {exc}")

    def _show_activity_log_detail(self, log_doc: dict) -> None:
        loc_hint = log_doc.get("location_hint") or "Unknown"
        loc_conf = float(log_doc.get("location_confidence", 0.0) or 0.0)
        geo_dev_raw = log_doc.get("geolocation_deviation")
        geo_resolved = bool(log_doc.get("geolocation_resolved", False))
        in_fence = log_doc.get("inside_office_geofence")
        trust = float(log_doc.get("location_trust_score", 0.0) or 0.0)
        geo_source = str(log_doc.get("geo_source") or "unknown")
        isp_val = str(log_doc.get("isp") or "Unknown")
        location_mode = str(log_doc.get("location_mode") or "unknown")
        lat_val = log_doc.get("lat")
        lon_val = log_doc.get("lon")

        # Backward compatibility: older activity logs may miss geo fields.
        # Try filling from the matching session record.
        try:
            if self._db and self._db.is_connected:
                sess_col = self._db.get_collection("sessions")
                if sess_col is not None:
                    sess = None
                    sess_id = log_doc.get("session_id")
                    if sess_id:
                        sess = sess_col.find_one({"session_id": sess_id}, {"_id": 0})
                    if sess is None:
                        uid = log_doc.get("user_id")
                        if uid:
                            sess = sess_col.find_one({"employee_id": uid}, {"_id": 0}, sort=[("login_at", -1)])

                    if sess:
                        if geo_source in {"", "unknown"}:
                            if sess.get("geo_source"):
                                geo_source = str(sess.get("geo_source"))
                            elif sess.get("ip") or sess.get("city"):
                                geo_source = "session_fallback"
                        if lat_val is None:
                            lat_val = sess.get("lat")
                        if lon_val is None:
                            lon_val = sess.get("lon")
                        if (loc_hint == "Unknown") and sess.get("location_hint"):
                            loc_hint = str(sess.get("location_hint"))
                        if (isp_val == "Unknown") and sess.get("isp"):
                            isp_val = str(sess.get("isp"))
                        if location_mode in {"", "unknown"} and sess.get("location_mode"):
                            location_mode = str(sess.get("location_mode"))

                        if (loc_conf <= 0.0) and (sess.get("location_confidence") is not None):
                            try:
                                loc_conf = float(sess.get("location_confidence") or 0.0)
                            except Exception:
                                pass

                        if (trust <= 0.0) and (sess.get("location_trust_score") is not None):
                            try:
                                trust = float(sess.get("location_trust_score") or 0.0)
                            except Exception:
                                pass

                        if in_fence is None and ("inside_office_geofence" in sess):
                            in_fence = sess.get("inside_office_geofence")

                        if (not geo_resolved) and (sess.get("geolocation_resolved") is not None):
                            geo_resolved = bool(sess.get("geolocation_resolved"))

                        if geo_dev_raw is None and (sess.get("geolocation_deviation") is not None):
                            geo_dev_raw = sess.get("geolocation_deviation")
        except Exception:
            pass

        coords = "Unknown"
        try:
            if lat_val is not None and lon_val is not None:
                coords = f"{float(lat_val):.5f}, {float(lon_val):.5f}"
        except Exception:
            coords = "Unknown"

        geo_dev_text = "Unknown"
        if geo_resolved and geo_dev_raw is not None:
            try:
                geo_dev_text = f"{float(geo_dev_raw):.3f}"
            except Exception:
                geo_dev_text = "Unknown"

        in_fence_text = "Unknown"
        if in_fence is True:
            in_fence_text = "True"
        elif in_fence is False:
            in_fence_text = "False"

        lines = [
            f"Time: {_fmt_time(log_doc.get('timestamp', ''))}",
            f"Risk: {float(log_doc.get('composite_risk_score', 0.0) or 0.0):.2f}",
            f"Productivity: {float(log_doc.get('productivity_score', 0.0) or 0.0):.2f}%",
            f"Label: {str(log_doc.get('label', 'normal')).replace('_', ' ').title()}",
            f"Top App: {log_doc.get('top_app', '-')}",
            f"Idle Ratio: {float(log_doc.get('idle_ratio', 0.0) or 0.0):.3f}",
            f"Location: {location_mode}",
            f"Estimated Place: {loc_hint}",
            f"Geo Source: {geo_source}",
            f"Coordinates: {coords}",
            f"Location Confidence: {loc_conf:.2f}",
            f"ISP: {isp_val}",
            f"Geo Deviation (KM): {geo_dev_text}",
            f"Inside Office Geofence: {in_fence_text}",
            f"Geo Check Resolved: {geo_resolved}",
            f"VPN/Proxy Detected: {bool(log_doc.get('vpn_proxy_detected', False))}",
            f"Hosting Network: {bool(log_doc.get('hosting_detected', False))}",
            f"Location Trust: {trust:.2f}",
            f"In Break: {bool(log_doc.get('in_break', False))}",
            f"Break Type: {log_doc.get('break_type') or '-'}",
            f"Alert Triggered: {bool(log_doc.get('alert_triggered', False))}",
        ]

        factors = log_doc.get("contributing_factors", []) or []
        if factors:
            lines.append("Contributing Factors: " + ", ".join(str(f) for f in factors))
        else:
            lines.append("Contributing Factors: -")

        messagebox.showinfo("Activity Log Detail", "\n".join(lines))

    def _open_file(self, path: str) -> None:
        try:
            if not path: return
            dir_path = os.path.dirname(path)
            if sys.platform == "win32": os.startfile(dir_path)
            else:
                import subprocess
                subprocess.call(["open", dir_path])
        except Exception as exc: messagebox.showerror("Error", str(exc))

    def _force_screenshot(self, emp_id: str) -> None:
        """
        Send a remote command to the employee's app to trigger a screenshot.
        """
        try:
            col = self._db.get_collection("commands")
            if col is not None:
                cmd = {
                    "command_id": str(uuid.uuid4()),
                    "target_user_id": emp_id,
                    "command_type": "force_screenshot",
                    "status": "pending",
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
                }
                col.insert_one(cmd)
                messagebox.showinfo("Command Sent", f"Force Screenshot command queued for {emp_id}.\nIt will be captured on their next heartbeat.")
            else:
                messagebox.showerror("Error", "Command collection unavailable.")
        except Exception as exc: 
            messagebox.showerror("Error", f"Failed to send command: {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Screenshot Viewer Window
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ScreenshotViewer(ctk.CTkToplevel):
    def __init__(self, parent, data: str, title: str = "Screenshot Viewer", is_path: bool = False, user_id: str = ""):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x650")
        self.attributes("-topmost", True)
        self.configure(fg_color=C_BG)

        import base64
        import io
        from PIL import Image
        from common.encryption import AESEncryptor

        try:
            if is_path:
                p = Path(data)
                if not p.exists():
                    raise FileNotFoundError(f"File not found: {p}")
                
                raw_bytes = p.read_bytes()
                
                # PNG Signature check for legacy/unencrypted files
                if raw_bytes.startswith(b"\x89PNG") or raw_bytes.startswith(b"\xff\xd8"):
                    img_bytes = raw_bytes
                    source_text = "Viewing Local File (Unencrypted)"
                elif p.suffix == ".enc":
                    # Decrypt high-quality local image
                    enc = AESEncryptor()
                    # Use user_id as associated data if available (matches ScreenshotTrigger logic)
                    assoc = user_id.encode() if user_id else None
                    img_bytes = enc.decrypt_bytes(raw_bytes, associated_data=assoc)
                    source_text = "Viewing Local File (Secure Decrypted)"
                else:
                    img_bytes = raw_bytes
                    source_text = "Viewing Local File"
            else:
                img_bytes = base64.b64decode(data)
                source_text = "Viewing Cloud Preview (Optimized)"

            img = Image.open(io.BytesIO(img_bytes))
            
            # Resize
            display_w, display_h = 860, 540
            img.thumbnail((display_w, display_h))
            
            self._photo = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            self._lbl = ctk.CTkLabel(self, image=self._photo, text="")
            self._lbl.pack(expand=True, padx=20, pady=20)
            
            ctk.CTkLabel(self, text=source_text, font=ctk.CTkFont(size=10), text_color=C_MUTED).pack(pady=(0, 10))
            
        except Exception as exc:
            ctk.CTkLabel(self, text=f"Failed to load image: {exc}", text_color=C_RED, wraplength=400).pack(expand=True)

class LiveCamViewer(ctk.CTkToplevel):
    """
    Real-time camera feed viewer that polls MongoDB for latest snapshots.
    """
    def __init__(self, parent, user_id: str, db: Optional[MongoDBClient]):
        super().__init__(parent)
        self.user_id = user_id
        self._db = db
        self.title(f"Live Cam — {user_id}")
        self.geometry("680x560")
        self.attributes("-topmost", True)
        self.configure(fg_color=C_BG)

        self._lbl = ctk.CTkLabel(self, text="Initializing Stream...", font=ctk.CTkFont(size=14), text_color=C_MUTED)
        self._lbl.pack(expand=True, fill="both", padx=20, pady=20)

        self._status_lbl = ctk.CTkLabel(self, text="Connecting to remote device...", font=ctk.CTkFont(size=11), text_color=C_AMBER)
        self._status_lbl.pack(pady=(0, 10))

        # Send command to start streaming
        self._send_command("start_live_cam")
        
        self._closed = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_loop()

    def _send_command(self, cmd_type: str):
        if not self._db or not self._db.is_connected: return
        import uuid
        try:
            col = self._db.get_collection("commands")
            if col is not None:
                now = datetime.utcnow()
                expires = (now + timedelta(minutes=5)).isoformat()
                col.insert_one({
                    "command_id": str(uuid.uuid4()),
                    "target_user_id": self.user_id,
                    "command_type": cmd_type,
                    "status": "pending",
                    "timestamp": now.isoformat(),
                    "expires_at": expires
                })
        except Exception: pass

    def _update_loop(self):
        if self._closed or not self.winfo_exists(): return
        if not self._db or not self._db.is_connected: 
            self.after(1000, self._update_loop)
            return
        
        try:
            import base64, io
            from PIL import Image
            
            col = self._db.get_collection("camera_streams")
            if col is not None:
                doc = col.find_one({"user_id": self.user_id})
                if doc:
                    status = doc.get("status")
                    if status == "streaming":
                        b64 = doc.get("image_base64")
                        if b64:
                            img_bytes = base64.b64decode(b64)
                            img = Image.open(io.BytesIO(img_bytes))
                            # Display
                            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                            self._lbl.configure(image=photo, text="")
                            self._lbl._image = photo # Keep reference
                            self._status_lbl.configure(text=f"Live • Last Update: {doc.get('timestamp','?')[-8:]}", text_color=C_GREEN)
                    elif status == "off":
                        err = doc.get("error", "Stream stopped by employee system.")
                        self._status_lbl.configure(text=err, text_color=C_RED)
                else:
                    self._status_lbl.configure(text="Waiting for remote device to respond...", text_color=C_AMBER)
        except Exception as e:
            if self.winfo_exists():
                self._status_lbl.configure(text=f"Update failed: {e}", text_color=C_RED)
        
        if self.winfo_exists():
            self.after(1000, self._update_loop)

    def _on_close(self):
        self._closed = True
        self._send_command("stop_live_cam")
        self.destroy()

class LiveScreenViewer(ctk.CTkToplevel):
    """
    Real-time screen feed viewer that polls MongoDB for latest snapshots.
    """
    def __init__(self, parent, user_id: str, db: Optional[MongoDBClient]):
        super().__init__(parent)
        self.user_id = user_id
        self._db = db
        self.title(f"Live Screen — {user_id}")
        self.geometry("820x600")
        self.attributes("-topmost", True)
        self.configure(fg_color=C_BG)

        self._lbl = ctk.CTkLabel(self, text="Initializing Stream...", font=ctk.CTkFont(size=14), text_color=C_MUTED)
        self._lbl.pack(expand=True, fill="both", padx=20, pady=20)

        self._status_lbl = ctk.CTkLabel(self, text="Connecting to remote device...", font=ctk.CTkFont(size=11), text_color=C_AMBER)
        self._status_lbl.pack(pady=(0, 10))

        # Send command to start streaming
        self._send_command("start_live_screen")
        
        self._closed = False
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_loop()

    def _send_command(self, cmd_type: str):
        if not self._db or not self._db.is_connected: return
        import uuid
        try:
            col = self._db.get_collection("commands")
            if col is not None:
                now = datetime.utcnow()
                expires = (now + timedelta(minutes=5)).isoformat()
                col.insert_one({
                    "command_id": str(uuid.uuid4()),
                    "target_user_id": self.user_id,
                    "command_type": cmd_type,
                    "status": "pending",
                    "timestamp": now.isoformat(),
                    "expires_at": expires
                })
        except Exception: pass

    def _update_loop(self):
        if self._closed or not self.winfo_exists(): return
        if not self._db or not self._db.is_connected: 
            self.after(1000, self._update_loop)
            return
        
        try:
            import base64, io
            from PIL import Image
            
            col = self._db.get_collection("screen_streams")
            if col is not None:
                doc = col.find_one({"user_id": self.user_id})
                if doc:
                    status = doc.get("status")
                    if status == "streaming":
                        b64 = doc.get("image_base64")
                        if b64:
                            img_bytes = base64.b64decode(b64)
                            img = Image.open(io.BytesIO(img_bytes))
                            # Display
                            photo = ctk.CTkImage(light_image=img, dark_image=img, size=(780, 440))
                            self._lbl.configure(image=photo, text="")
                            self._lbl._image = photo # Keep reference
                            self._status_lbl.configure(text=f"Live Screen • Last Update: {doc.get('timestamp','?')[-8:]}", text_color=C_GREEN)
                    elif status == "off":
                        err = doc.get("error", "Stream stopped by employee system.")
                        self._status_lbl.configure(text=err, text_color=C_RED)
                else:
                    self._status_lbl.configure(text="Waiting for remote screen capture...", text_color=C_AMBER)
        except Exception as e:
            if self.winfo_exists():
                self._status_lbl.configure(text=f"Update failed: {e}", text_color=C_RED)
        
        if self.winfo_exists():
            self.after(2000, self._update_loop)

    def _on_close(self):
        self._closed = True
        self._send_command("stop_live_screen")
        self.destroy()

class AdminPanel(ctk.CTk):
    """
    Full-featured CustomTkinter Admin Panel for the monitoring system.
    """

    def __init__(self, db: Optional[MongoDBClient] = None) -> None:
        super().__init__()

        self._db = db or self._init_db()
        self._alert_sender = AlertSender(ws_url=settings.WEBSOCKET_URL)
        self._active_tab = "dashboard"
        self._employee_rows: dict = {}
        self._employee_name_cache: dict[str, str] = {}
        self._critical_sound_seen: set[str] = set()
        self._efficiency_period_var = ctk.StringVar(value="Last Month")
        self._efficiency_service = None
        self._eff_pie_canvas = None
        self._eff_bar_canvas = None
        self._alerts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AlertsRefresh")
        self._alerts_refresh_inflight = False
        self._alerts_last_signature = ""
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws_server = None
        self._closed = False
        self._poll_after_id = None
        
        # Efficiency optimization: async rendering + caching
        self._efficiency_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EfficiencyRenderer")
        self._efficiency_cache = None
        self._efficiency_cache_ts = 0
        self._efficiency_cache_ttl_sec = 60.0  # Cache for 60 seconds

        self.title(f"{settings.APP_NAME} — Admin Panel")
        w, h = 1200, 780
        self.geometry(f"{w}x{h}")
        self.minsize(900, 600)
        self.configure(fg_color=C_BG)

        # Centre on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        self._build_layout()
        self._switch_tab("dashboard")
        self._start_alert_ws_server()
        self._start_polling()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        # Sidebar
        self._sidebar = ctk.CTkFrame(self, width=210, fg_color=C_SIDEBAR, corner_radius=0)
        self._sidebar.pack(side="left", fill="y")
        self._sidebar.pack_propagate(False)

        # App logo/title area
        logo_frame = ctk.CTkFrame(self._sidebar, fg_color="transparent")
        logo_frame.pack(fill="x", padx=16, pady=(24, 20))
        ctk.CTkLabel(
            logo_frame, text=settings.APP_NAME,
            font=ctk.CTkFont(size=20, weight="bold"), text_color=C_TEAL,
        ).pack(anchor="w")
        ctk.CTkLabel(
            logo_frame, text="Admin Console",
            font=ctk.CTkFont(size=11), text_color=C_MUTED,
        ).pack(anchor="w")

        # Separator
        ctk.CTkFrame(self._sidebar, height=1, fg_color=C_BORDER).pack(fill="x", padx=16, pady=(0, 16))

        nav_items = [
            ("dashboard",  "  Dashboard"),
            ("employees",  "  Employees"),
            ("live_grid",  "  Live Monitor"),
            ("efficiency", "  Efficiency"),
            ("alerts",     "  Alerts"),
            ("tasks",      "  Tasks"),
            ("attendance", "  Attendance"),
        ]
        self._nav_btns: dict = {}
        for tab_id, label in nav_items:
            btn = ctk.CTkButton(
                self._sidebar,
                text=label,
                height=44,
                font=ctk.CTkFont(size=13),
                anchor="w",
                fg_color="transparent",
                text_color=C_MUTED,
                hover_color="#1a2133",
                corner_radius=8,
                command=lambda t=tab_id: self._switch_tab(t),
            )
            btn.pack(fill="x", padx=12, pady=2)
            self._nav_btns[tab_id] = btn

        ctk.CTkButton(
            self._sidebar,
            text="  Behavioral Baseline Viewer",
            height=44,
            font=ctk.CTkFont(size=13),
            anchor="w",
            fg_color="transparent",
            text_color=C_MUTED,
            hover_color="#1a2133",
            corner_radius=8,
            command=self._open_baseline_window,
        ).pack(fill="x", padx=12, pady=(8, 2))

        btn = ctk.CTkButton(
            self._sidebar,
            text="  Settings",
            height=44,
            font=ctk.CTkFont(size=13),
            anchor="w",
            fg_color="transparent",
            text_color=C_MUTED,
            hover_color="#1a2133",
            corner_radius=8,
            command=lambda: self._switch_tab("settings"),
        )
        btn.pack(fill="x", padx=12, pady=2)
        self._nav_btns["settings"] = btn

        ctk.CTkFrame(self._sidebar, height=1, fg_color=C_BORDER).pack(fill="x", padx=16, pady=(20, 10))

        # Main content area
        self._content = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True)

        # Header bar
        self._header = ctk.CTkFrame(self._content, height=58, fg_color=C_CARD, corner_radius=0)
        self._header.pack(fill="x")
        self._page_title = ctk.CTkLabel(
            self._header, text="Dashboard",
            font=ctk.CTkFont(size=17, weight="bold"), text_color=C_TEXT,
        )
        self._page_title.pack(side="left", padx=24, pady=14)

        self._conn_lbl = ctk.CTkLabel(
            self._header,
            text="⬤ Connected" if (self._db and self._db.is_connected) else "⬤ Offline",
            font=ctk.CTkFont(size=11),
            text_color=C_GREEN if (self._db and self._db.is_connected) else C_RED,
        )
        self._conn_lbl.pack(side="right", padx=24)

        # Tab frame container
        self._tab_frame = ctk.CTkFrame(self._content, fg_color=C_BG, corner_radius=0)
        self._tab_frame.pack(fill="both", expand=True)

        # Initialise all tab panels
        self._tabs: dict = {
            "dashboard":  self._build_dashboard_tab(),
            "employees":  self._build_employees_tab(),
            "live_grid":  self._tab_live_grid(),
            "efficiency": self._build_efficiency_tab(),
            "alerts":     self._build_alerts_tab(),
            "tasks":      self._build_tasks_tab(),
            "attendance": self._build_attendance_tab(),
            "settings":   self._build_settings_tab(),
        }

    def _tab_live_grid(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color="transparent")
        
        header = ctk.CTkFrame(frame, fg_color=C_CARD, height=60, corner_radius=12)
        header.pack(fill="x", pady=(0, 20))
        header.pack_propagate(False)
        
        ctk.CTkLabel(header, text="Live Screen Monitor Grid", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left", padx=20)
        
        self._grid_scroll = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        self._grid_scroll.pack(fill="both", expand=True)
        
        self._grid_items = {} # {user_id: {frame, label, status_label}}
        self._image_cache = {} # {user_id: {image, timestamp, ctk_image}} - Cache for 30 seconds
        
        return frame

    def _build_efficiency_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        header = ctk.CTkFrame(frame, fg_color=C_CARD, corner_radius=12)
        header.pack(fill="x", padx=20, pady=(18, 12))

        ctk.CTkLabel(
            header,
            text="Employee Efficiency Predictions",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=18, pady=(16, 4))

        ctk.CTkLabel(
            header,
            text="Open the separate read-only C4 window to review predicted efficiency for each employee.",
            font=ctk.CTkFont(size=12),
            text_color=C_MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 10))

        body = ctk.CTkFrame(frame, fg_color=C_CARD, corner_radius=12)
        body.pack(fill="x", padx=20, pady=(0, 12))

        ctk.CTkLabel(
            body,
            text="This view does not change tasks, attendance, or monitoring data.",
            font=ctk.CTkFont(size=12),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=18, pady=(16, 6))

        ctk.CTkLabel(
            body,
            text="Use the button below to launch the separate efficiency analysis window.",
            font=ctk.CTkFont(size=12),
            text_color=C_MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 14))

        ctk.CTkButton(
            body,
            text="Open Efficiency Window",
            fg_color=C_TEAL,
            hover_color=C_TEAL_D,
            height=40,
            command=self._open_efficiency_window,
        ).pack(anchor="w", padx=18, pady=(0, 18))

        charts_card = ctk.CTkFrame(frame, fg_color=C_CARD, corner_radius=12)
        charts_card.pack(fill="both", expand=True, padx=20, pady=(0, 12))

        control_row = ctk.CTkFrame(charts_card, fg_color="transparent")
        control_row.pack(fill="x", padx=16, pady=(14, 6))

        ctk.CTkLabel(
            control_row,
            text="Overall Efficiency Overview",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C_TEXT,
        ).pack(side="left")

        ctk.CTkOptionMenu(
            control_row,
            values=["Last Month", "Last 3 Months", "Last 6 Months", "All Time"],
            variable=self._efficiency_period_var,
            command=lambda _v: self._clear_efficiency_cache_and_refresh(),
            fg_color=C_BORDER,
            button_color=C_BLUE,
            button_hover_color="#2563eb",
            width=170,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            control_row,
            text="Refresh Charts",
            fg_color=C_TEAL,
            hover_color=C_TEAL_D,
            width=120,
            command=self._refresh_efficiency_overview,
        ).pack(side="right")

        charts_row = ctk.CTkFrame(charts_card, fg_color="transparent")
        charts_row.pack(fill="both", expand=True, padx=16, pady=(4, 12))

        self._eff_pie_frame = ctk.CTkFrame(charts_row, fg_color="#10172b", corner_radius=10)
        self._eff_pie_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._eff_bar_frame = ctk.CTkFrame(charts_row, fg_color="#10172b", corner_radius=10)
        self._eff_bar_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        ctk.CTkLabel(
            self._eff_pie_frame,
            text="Prediction Distribution",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(
            self._eff_bar_frame,
            text="Overall Efficiency Indicators",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 2))

        return frame

    def _efficiency_period_range(self):
        choice = self._efficiency_period_var.get().strip().lower()
        now = datetime.now(timezone.utc)

        if choice == "all time":
            return None, None
        if choice == "last month":
            month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            prev_month_end = month_start - timedelta(seconds=1)
            prev_month_start = datetime(prev_month_end.year, prev_month_end.month, 1, tzinfo=timezone.utc)
            return prev_month_start, prev_month_end
        if choice == "last 3 months":
            return now - timedelta(days=90), now
        if choice == "last 6 months":
            return now - timedelta(days=180), now
        return now - timedelta(days=30), now

    def _refresh_efficiency_overview(self) -> None:
        if not self._db or not self._db.is_connected:
            return

        # Check cache first (60 sec TTL)
        now = time.time()
        if self._efficiency_cache is not None and (now - self._efficiency_cache_ts) < self._efficiency_cache_ttl_sec:
            self._render_efficiency_charts(self._efficiency_cache)
            return

        # Show loading indicator
        for host in [self._eff_pie_frame, self._eff_bar_frame]:
            for child in host.winfo_children()[1:]:
                child.destroy()
        ctk.CTkLabel(self._eff_pie_frame, text="Loading...", text_color=C_MUTED).pack(anchor="w", padx=10, pady=(8, 12))
        ctk.CTkLabel(self._eff_bar_frame, text="Loading...", text_color=C_MUTED).pack(anchor="w", padx=10, pady=(8, 12))

        # Submit to background thread (non-blocking)
        self._efficiency_executor.submit(self._fetch_and_render_efficiency)

    def _fetch_and_render_efficiency(self) -> None:
        """Fetch efficiency data in background thread."""
        try:
            from C4_productivity_prediction.src.efficiency_service import EfficiencyPredictionService
        except Exception as exc:
            logger.warning("Efficiency service import failed: %s", exc)
            return

        if self._efficiency_service is None:
            self._efficiency_service = EfficiencyPredictionService()

        try:
            start, end = self._efficiency_period_range()
            rows = self._efficiency_service.predict_all(self._db, period_start=start, period_end=end)
            
            # Cache results
            self._efficiency_cache = rows
            self._efficiency_cache_ts = time.time()
            
            # Render on main thread
            self.after(0, lambda: self._render_efficiency_charts(rows))
        except Exception as exc:
            logger.warning("Efficiency overview refresh failed: %s", exc)
            self.after(0, lambda: self._show_efficiency_error("Failed to load data"))
    
    def _show_efficiency_error(self, msg: str) -> None:
        """Show error in efficiency charts."""
        for host in [self._eff_pie_frame, self._eff_bar_frame]:
            for child in host.winfo_children()[1:]:
                child.destroy()
        ctk.CTkLabel(self._eff_pie_frame, text=msg, text_color=C_RED).pack(anchor="w", padx=10, pady=(8, 12))
        ctk.CTkLabel(self._eff_bar_frame, text=msg, text_color=C_RED).pack(anchor="w", padx=10, pady=(8, 12))
    
    def _clear_efficiency_cache_and_refresh(self) -> None:
        """Clear cache and refresh when user changes period."""
        self._efficiency_cache = None
        self._refresh_efficiency_overview()

    def _render_efficiency_charts(self, rows) -> None:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as exc:
            logger.warning("Matplotlib unavailable for efficiency charts: %s", exc)
            return

        for host in [self._eff_pie_frame, self._eff_bar_frame]:
            for child in host.winfo_children()[1:]:
                child.destroy()

        if self._eff_pie_canvas is not None:
            try:
                self._eff_pie_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self._eff_pie_canvas = None
        if self._eff_bar_canvas is not None:
            try:
                self._eff_bar_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self._eff_bar_canvas = None

        if not rows:
            ctk.CTkLabel(self._eff_pie_frame, text="No data for selected period.", text_color=C_MUTED).pack(anchor="w", padx=10, pady=(8, 12))
            ctk.CTkLabel(self._eff_bar_frame, text="No data for selected period.", text_color=C_MUTED).pack(anchor="w", padx=10, pady=(8, 12))
            return

        high = sum(1 for r in rows if str(r.predicted_label).lower() == "high")
        medium = sum(1 for r in rows if str(r.predicted_label).lower() == "medium")
        low = sum(1 for r in rows if str(r.predicted_label).lower() == "low")

        total_assigned = sum(int(r.total_tasks_assigned) for r in rows)
        total_pending = sum(int(r.total_tasks_pending) for r in rows)
        total_on_time = sum(int(r.total_tasks_completed_on_time) for r in rows)
        total_late = sum(int(r.total_tasks_completed_late) for r in rows)
        total_completed = total_on_time + total_late

        avg_prod = sum(float(r.productivity_score_input) for r in rows) / len(rows)
        avg_work = sum(float(r.workload_score) for r in rows) / len(rows)
        on_time_pct = (100.0 * total_on_time / total_completed) if total_completed else 0.0
        pending_pct = (100.0 * total_pending / total_assigned) if total_assigned else 0.0
        late_pct = (100.0 * total_late / total_completed) if total_completed else 0.0

        fig_pie = Figure(figsize=(4.8, 2.8), dpi=100, facecolor="#10172b")
        ax_pie = fig_pie.add_subplot(111)
        ax_pie.set_facecolor("#10172b")

        parts = [("High", high, C_GREEN), ("Medium", medium, C_AMBER), ("Low", low, C_RED)]
        parts = [p for p in parts if p[1] > 0]
        if not parts:
            parts = [("No Data", 1, C_MUTED)]

        labels = [p[0] for p in parts]
        sizes = [p[1] for p in parts]
        colors = [p[2] for p in parts]
        _, _, autotexts = ax_pie.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"color": C_TEXT, "fontsize": 9},
        )
        for t in autotexts:
            t.set_color(C_TEXT)
            t.set_fontsize(8)
        ax_pie.axis("equal")
        fig_pie.tight_layout()

        self._eff_pie_canvas = FigureCanvasTkAgg(fig_pie, master=self._eff_pie_frame)
        self._eff_pie_canvas.draw()
        self._eff_pie_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 10))

        fig_bar = Figure(figsize=(4.8, 2.8), dpi=100, facecolor="#10172b")
        ax_bar = fig_bar.add_subplot(111)
        ax_bar.set_facecolor("#10172b")
        names = ["Avg Prod", "Avg Work", "On-time %", "Pending %", "Late %"]
        values = [avg_prod, avg_work, on_time_pct, pending_pct, late_pct]
        colors = [C_TEAL, C_BLUE, C_GREEN, C_AMBER, C_RED]
        ax_bar.bar(names, values, color=colors)
        ax_bar.set_ylim(0, 100)
        ax_bar.grid(axis="y", color="#23324d", linestyle="-", linewidth=0.6)
        ax_bar.tick_params(axis="x", colors=C_TEXT, labelsize=8, rotation=10)
        ax_bar.tick_params(axis="y", colors=C_TEXT, labelsize=8)
        for sp in ax_bar.spines.values():
            sp.set_color("#23324d")
        fig_bar.tight_layout()

        self._eff_bar_canvas = FigureCanvasTkAgg(fig_bar, master=self._eff_bar_frame)
        self._eff_bar_canvas.draw()
        self._eff_bar_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 10))

    def _open_efficiency_window(self) -> None:
        script_path = _PROJECT_ROOT / "C4_productivity_prediction" / "src" / "launch_efficiency_window.py"
        if not script_path.exists():
            messagebox.showerror("Efficiency Window", f"Launcher not found: {script_path}")
            return

        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=str(_PROJECT_ROOT))
        except Exception as exc:
            logger.exception("Failed to launch efficiency window")
            messagebox.showerror("Efficiency Window", f"Failed to launch efficiency window: {exc}")

    def _open_baseline_window(self) -> None:
        script_path = _PROJECT_ROOT / "C1_user_Behavioural_Baseline" / "dashboard.py"
        if not script_path.exists():
            messagebox.showerror("Baseline Viewer", f"Launcher not found: {script_path}")
            return

        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=str(_PROJECT_ROOT))
        except Exception as exc:
            logger.exception("Failed to launch baseline viewer")
            messagebox.showerror("Baseline Viewer", f"Failed to launch baseline viewer: {exc}")

    def _refresh_live_grid(self) -> None:
        threading.Thread(target=self._fetch_live_grid, daemon=True).start()

    def _fetch_live_grid(self) -> None:
        if not self._db or not self._db.is_connected: return
        try:
            import base64, io
            from PIL import Image
            col = self._db.get_collection("screen_streams")
            if col is None: return
            streams = list(col.find({"status": "streaming"}))
            
            # Process images in background thread with caching
            processed = []
            current_time = time.time()
            
            for s in streams:
                uid = s["user_id"]
                img = None
                ctk_img = None
                b64 = s.get("image_base64")
                timestamp = s.get("timestamp", "?")
                
                # Check cache (valid for 30 seconds)
                cache_key = f"{uid}_{hash(b64) if b64 else 'no_image'}"  # Use hash of b64 as key
                cached = self._image_cache.get(cache_key)
                if cached and (current_time - cached["timestamp"]) < 30:
                    img = cached["pil_image"]
                    ctk_img = cached["ctk_image"]
                elif b64:
                    try:
                        img_bytes = base64.b64decode(b64)
                        img = Image.open(io.BytesIO(img_bytes))
                        # Cache the processed image
                        self._image_cache[cache_key] = {
                            "pil_image": img,
                            "ctk_image": None,  # Will be created in UI thread
                            "timestamp": current_time
                        }
                    except Exception: pass
                
                processed.append({
                    "user_id": uid, 
                    "image": img, 
                    "ctk_image": ctk_img,
                    "timestamp": timestamp,
                    "cache_key": cache_key  # Pass cache key to UI thread
                })
            
            # Clean old cache entries (keep cache size reasonable)
            if len(self._image_cache) > 20:  # Max 20 cached images
                oldest_keys = sorted(self._image_cache.keys(), 
                                   key=lambda k: self._image_cache[k]["timestamp"])[:10]
                for key in oldest_keys:
                    del self._image_cache[key]
            
            self.after(0, lambda: self._update_live_grid_ui(processed))
        except Exception: pass

    def _update_live_grid_ui(self, processed: list) -> None:
        current_ids = set(self._grid_items.keys())
        active_ids = {p["user_id"] for p in processed}
        
        # Remove dead
        for uid in current_ids - active_ids:
            if uid in self._grid_items:
                self._grid_items[uid]["frame"].destroy()
                del self._grid_items[uid]
                
        # Update/Add
        for i, p in enumerate(processed):
            uid = p["user_id"]
            if uid not in self._grid_items:
                tile = ctk.CTkFrame(self._grid_scroll, fg_color=C_CARD, corner_radius=12, width=380, height=240)
                tile.grid(row=len(self._grid_items)//3, column=len(self._grid_items)%3, padx=10, pady=10)
                tile.grid_propagate(False)
                n_lbl = ctk.CTkLabel(tile, text=f"Employee: {uid}", font=ctk.CTkFont(size=12, weight="bold"))
                n_lbl.pack(pady=(10, 5))
                s_lbl = ctk.CTkLabel(tile, text="Loading...", font=ctk.CTkFont(size=10), text_color=C_MUTED)
                s_lbl.pack(expand=True, fill="both", padx=10)
                st_lbl = ctk.CTkLabel(tile, text="Live • Capturing", font=ctk.CTkFont(size=9), text_color=C_GREEN)
                st_lbl.pack(pady=(0, 10))
                self._grid_items[uid] = {"frame": tile, "label": s_lbl, "status": st_lbl}

            if p["image"]:
                img = p["image"]
                ctk_img = p.get("ctk_image")
                
                if ctk_img is None:
                    # Create new CTkImage if not cached
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(340, 180))
                    # Update cache with CTkImage
                    cache_key = f"{uid}_{p.get('cache_key', '')}"
                    if cache_key in self._image_cache:
                        self._image_cache[cache_key]["ctk_image"] = ctk_img
                
                self._grid_items[uid]["label"].configure(image=ctk_img, text="")
                self._grid_items[uid]["label"]._image = ctk_img
                self._grid_items[uid]["status"].configure(text=f"Live • Last sync: {p['timestamp'][-8:]}")

    def _switch_tab(self, tab_id: str) -> None:
        self._active_tab = tab_id
        for tid, widget in self._tabs.items():
            widget.pack_forget()
        self._tabs[tab_id].pack(fill="both", expand=True)

        if tab_id == "settings":
            try:
                self._refresh_geo_policy_ui(status_msg="")
            except Exception:
                pass
        elif tab_id == "efficiency":
            self._refresh_efficiency_overview()

        self._page_title.configure(text=tab_id.title())
        for tid, btn in self._nav_btns.items():
            if tid == tab_id:
                btn.configure(fg_color="#1e3a5f", text_color=C_TEAL)
            else:
                btn.configure(fg_color="transparent", text_color=C_MUTED)

    # ------------------------------------------------------------------
    # Dashboard Tab
    # ------------------------------------------------------------------

    def _build_dashboard_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        # Summary cards row
        cards_row = ctk.CTkFrame(frame, fg_color="transparent")
        cards_row.pack(fill="x", padx=20, pady=(16, 12))

        self._card_online    = SummaryCard(cards_row, "Employees Online", "—", accent=C_TEAL)
        self._card_alerts    = SummaryCard(cards_row, "Alerts Today",     "—", accent=C_AMBER)
        self._card_highrisk  = SummaryCard(cards_row, "High Risk",        "—", accent=C_RED)
        self._card_avg_prod  = SummaryCard(cards_row, "Avg Productivity",  "—", accent=C_GREEN)

        for card in (self._card_online, self._card_alerts, self._card_highrisk, self._card_avg_prod):
            card.pack(side="left", expand=True, fill="both", padx=6)

        # Employee list
        ctk.CTkLabel(
            frame, text="Live Employee Status",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT,
        ).pack(anchor="w", padx=24, pady=(8, 4))

        # Column headers
        hdr = ctk.CTkFrame(frame, fg_color=C_SIDEBAR, corner_radius=8, height=34)
        hdr.pack(fill="x", padx=20)
        for col, w in [("Employee", 190), ("ID", 90), ("Risk", 70), ("Location", 230), ("Status", 120), ("Last Seen", 220), ("", 90)]:
            ctk.CTkLabel(hdr, text=col, font=ctk.CTkFont(size=11), text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=4)

        self._emp_list_frame = ctk.CTkScrollableFrame(frame, fg_color=C_BG, corner_radius=0)
        self._emp_list_frame.pack(fill="both", expand=True, padx=20, pady=(4, 16))

        return frame

    def _refresh_dashboard(self) -> None:
        """Refresh summary cards and employee list from MongoDB."""
        if not self._db or not self._db.is_connected:
            return
        try:
            sessions_col = self._db.get_collection("sessions")
            alerts_col   = self._db.get_collection("alerts")
            activity_col = self._db.get_collection("activity_logs")
            emps_col     = self._db.get_collection("employees")

            # Employee rows (exclude heavy fields but keep enough for details)
            employees = list(emps_col.find({}, {"_id": 0, "password_hash": 0, "face_images": 0, "face_embedding": 0}).limit(50)) if emps_col is not None else []
            employee_ids = {e.get("employee_id") for e in employees if e.get("employee_id")}

            # Pre-fetch active sessions by employee (deduped + stale filtered)
            active_sessions = self._get_active_sessions_by_employee(employee_ids)

            # Count currently online employees (not raw session rows)
            online_cnt = len(active_sessions)
            self.after(0, lambda v=online_cnt: self._card_online.set_value(str(v)))

            # Alerts today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            alerts_today = alerts_col.count_documents({"timestamp": {"$gte": today_start.isoformat()}}) if alerts_col is not None else 0
            self.after(0, lambda v=alerts_today: self._card_alerts.set_value(str(v)))

            # High risk
            high_risk = activity_col.count_documents({"composite_risk_score": {"$gte": 75}}) if activity_col is not None else 0
            self.after(0, lambda v=high_risk: self._card_highrisk.set_value(str(v)))

            # Avg productivity (last 24 hours only - optimized)
            yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            pipeline = [
                {"$match": {"timestamp": {"$gte": yesterday.isoformat()}}},
                {"$group": {"_id": None, "avg": {"$avg": "$productivity_score"}}}
            ]
            avg_result = list(activity_col.aggregate(pipeline)) if activity_col is not None else []
            avg_prod = avg_result[0]["avg"] if avg_result else 0.0
            self._card_avg_prod.set_value(f"{avg_prod:.0f}%")

            self._update_employee_list(employees, activity_col, active_sessions)

        except Exception as exc:
            import logging
            logging.getLogger(__name__).error("Dashboard refresh error: %s", exc)

    def _update_employee_list(self, employees: list, activity_col, active_sessions: dict) -> None:
        if not hasattr(self, "_emp_rows"):
            self._emp_rows = {} # {emp_id: {"frame": frame, "labels": {name: label}}}

        # Sync IDs
        new_ids = {e.get("employee_id") for e in employees if e.get("employee_id")}
        old_ids = set(self._emp_rows.keys())

        # 1. Remove rows
        for eid in (old_ids - new_ids):
            try:
                self._emp_rows[eid]["frame"].pack_forget()
                self._emp_rows[eid]["frame"].destroy()
            except Exception: pass
            del self._emp_rows[eid]

        # 2. Update or Create
        for emp in employees:
            eid = emp.get("employee_id")
            if not eid: continue
            
            act = None
            try:
                if activity_col is not None:
                    act = activity_col.find_one({"user_id": eid}, sort=[("timestamp", -1)])
            except Exception: pass

            risk = act.get("composite_risk_score", 0.0) if act else 0.0
            risk_color = _risk_color(risk)
            idle_ratio = float(act.get("idle_ratio", 0.0)) if act else 0.0
            factors = act.get("contributing_factors", []) if act else []
            is_idle = idle_ratio >= 0.5 or ("high_idle_ratio" in factors)
            status = "Break" if (act and act.get("in_break")) else ("Idle" if is_idle else "Active")
            
            # Use activity log location or fallback to session metadata (city)
            sess = active_sessions.get(eid)
            is_online = (sess is not None)
            
            loc = "—"
            if act:
                act_city = act.get("city")
                act_country = act.get("country")
                if act_city and act_city != "Unknown":
                    loc = act_city
                    if act_country and act_country != "Unknown":
                        loc = f"{act_city}, {act_country}"
                elif act.get("location_hint") and act.get("location_hint") != "Unknown":
                    loc = str(act.get("location_hint"))
                elif act.get("location_mode"):
                    loc = act.get("location_mode").title()
            elif sess:
                # If no activity log yet, pull from session
                city = sess.get("city")
                loc = city if (city and city != "Unknown") else (sess.get("location_mode") or "—").title()
            
            act_ts = str(act.get("timestamp") or "").strip() if act else ""
            sess_ts = str(sess.get("login_at") or "").strip() if sess else ""

            # Prefer the most recent known timestamp across activity and session login.
            chosen_ts = act_ts or sess_ts
            if act_ts and sess_ts:
                act_dt = _parse_iso_timestamp(act_ts)
                sess_dt = _parse_iso_timestamp(sess_ts)
                if sess_dt is not None and (act_dt is None or sess_dt > act_dt):
                    chosen_ts = sess_ts

            last_seen = _fmt_last_seen(chosen_ts) if chosen_ts else "—"
            name = emp.get("full_name", eid)
            display_name = _clip_text(name, 26)
            display_loc = _clip_text(loc, 30)

            status_text = status
            status_color = C_TEXT
            if is_online:
                status_text = f"⬤ {status}"
                if status == "Idle":
                    status_color = C_AMBER
                elif status == "Break":
                    status_color = C_AMBER
                else:
                    status_color = C_GREEN
            elif status == "Break":
                status_text = f"○ {status}"
                status_color = C_AMBER
            else:
                status_text = f"✖ Offline"
                status_color = C_RED

            if eid in self._emp_rows:
                # Update
                labels = self._emp_rows[eid]["labels"]
                labels["name"].configure(text=display_name)
                labels["risk"].configure(text=f"{risk:.0f}", text_color=risk_color)
                labels["loc"].configure(text=display_loc)
                labels["status"].configure(text=status_text, text_color=status_color)
                labels["seen"].configure(text=last_seen)
            else:
                # Create
                row = ctk.CTkFrame(self._emp_list_frame, fg_color=C_CARD, corner_radius=10, height=44)
                row.pack(fill="x", pady=3)
                row.pack_propagate(False)

                l_name = ctk.CTkLabel(row, text=display_name, text_color=C_TEXT, font=ctk.CTkFont(size=12, weight="bold"), width=190, anchor="w")
                l_name.pack(side="left", padx=8)
                l_id = ctk.CTkLabel(row, text=eid, text_color=C_MUTED, font=ctk.CTkFont(size=11), width=90, anchor="w")
                l_id.pack(side="left")
                l_risk = ctk.CTkLabel(row, text=f"{risk:.0f}", text_color=risk_color, font=ctk.CTkFont(size=12, weight="bold"), width=70, anchor="w")
                l_risk.pack(side="left")
                l_loc = ctk.CTkLabel(row, text=display_loc, text_color=C_MUTED, font=ctk.CTkFont(size=11), width=230, anchor="w")
                l_loc.pack(side="left")
                l_status = ctk.CTkLabel(row, text=status_text, text_color=status_color, font=ctk.CTkFont(size=11), width=120, anchor="w")
                l_status.pack(side="left")
                l_seen = ctk.CTkLabel(row, text=last_seen, text_color=C_MUTED, font=ctk.CTkFont(size=11), width=220, anchor="w")
                l_seen.pack(side="left")

                ctk.CTkButton(
                    row, text="Details", width=72, height=28, fg_color=C_BORDER, hover_color=C_BLUE,
                    font=ctk.CTkFont(size=11), command=lambda e=emp: EmployeeDetailWindow(self, e, self._db)
                ).pack(side="right", padx=8)

                self._emp_rows[eid] = {
                    "frame": row,
                    "labels": {"name": l_name, "id": l_id, "risk": l_risk, "loc": l_loc, "status": l_status, "seen": l_seen}
                }

    # ------------------------------------------------------------------
    # Employees Tab
    # ------------------------------------------------------------------

    def _build_employees_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        topbar = ctk.CTkFrame(frame, fg_color="transparent")
        topbar.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkLabel(topbar, text="Employee Directory", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(side="left")
        ctk.CTkButton(
            topbar, text="Register New Employee", width=180,
            fg_color=C_TEAL, hover_color=C_TEAL_D,
            command=self._open_registration,
        ).pack(side="right")

        # Column headers
        hdr = ctk.CTkFrame(frame, fg_color=C_SIDEBAR, corner_radius=8, height=34)
        hdr.pack(fill="x", padx=20, pady=(0, 4))
        for col, w in [("Employee ID", 120), ("Name", 200), ("Email", 250), ("Actions", 150)]:
            ctk.CTkLabel(hdr, text=col, font=ctk.CTkFont(size=11), text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=8)

        self._all_emp_frame = ctk.CTkScrollableFrame(frame, fg_color=C_BG)
        self._all_emp_frame.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        return frame

    def _refresh_employees(self) -> None:
        if not self._db or not self._db.is_connected: return
        try:
            col = self._db.get_collection("employees")
            if col is None: return
            emps = list(col.find({}, {"_id": 0}))
            self.after(0, lambda: self._update_employees_ui(emps))
        except Exception:
            pass

    def _update_employees_ui(self, employees: list) -> None:
        for w in self._all_emp_frame.winfo_children():
            w.destroy()
        if not employees:
            ctk.CTkLabel(self._all_emp_frame, text="No employees found.", text_color=C_MUTED).pack(pady=20)
            return
            
        for emp in employees:
            eid = emp.get("employee_id", "Unknown")
            name = emp.get("full_name", "Unknown")
            email = emp.get("email", "—")
            
            row = ctk.CTkFrame(self._all_emp_frame, fg_color=C_CARD, corner_radius=8, height=44)
            row.pack(fill="x", pady=3)
            row.pack_propagate(False)
            
            l_id = ctk.CTkLabel(row, text=eid, text_color=C_MUTED, font=ctk.CTkFont(size=11), width=120, anchor="w")
            l_id.pack(side="left", padx=(8, 0))
            l_name = ctk.CTkLabel(row, text=name, text_color=C_TEXT, font=ctk.CTkFont(size=12, weight="bold"), width=200, anchor="w")
            l_name.pack(side="left", padx=8)
            l_email = ctk.CTkLabel(row, text=email, text_color=C_TEXT, font=ctk.CTkFont(size=11), width=250, anchor="w")
            l_email.pack(side="left", padx=8)
            
            # Actions
            btn_frame = ctk.CTkFrame(row, fg_color="transparent")
            btn_frame.pack(side="right", padx=8)
            
            ctk.CTkButton(
                btn_frame, text="✏️ Edit", width=60, height=28, fg_color=C_BLUE, hover_color="#2563eb",
                font=ctk.CTkFont(size=11), command=lambda e=emp: self._edit_employee(e)
            ).pack(side="left", padx=4)
            
            ctk.CTkButton(
                btn_frame, text="❌ Delete", width=70, height=28, fg_color="#450a0a", hover_color=C_RED,
                font=ctk.CTkFont(size=11), command=lambda e=emp: self._delete_employee(e)
            ).pack(side="left", padx=4)

    def _edit_employee(self, emp: dict) -> None:
        eid = emp.get("employee_id")
        win = ctk.CTkToplevel(self)
        win.title(f"Edit Employee - {eid}")
        win.geometry("440x550")
        win.attributes("-topmost", True)
        win.configure(fg_color=C_BG)
        
        ctk.CTkLabel(win, text="Edit Information", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(pady=(20, 10))
        
        form = ctk.CTkScrollableFrame(win, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=20, pady=10)

        def create_entry(label_text, default_val):
            f = ctk.CTkFrame(form, fg_color="transparent")
            f.pack(fill="x", pady=5)
            ctk.CTkLabel(f, text=label_text, width=120, anchor="w").pack(side="left")
            e = ctk.CTkEntry(f, width=220)
            e.pack(side="right")
            e.insert(0, default_val)
            return e

        def create_dropdown(label_text, default_val, options):
            f = ctk.CTkFrame(form, fg_color="transparent")
            f.pack(fill="x", pady=5)
            ctk.CTkLabel(f, text=label_text, width=120, anchor="w").pack(side="left")
            var = ctk.StringVar(value=default_val if default_val in options else options[0])
            om = ctk.CTkOptionMenu(f, variable=var, values=options, width=220, fg_color=C_BORDER, button_color=C_BORDER)
            om.pack(side="right")
            return var

        e_name = create_entry("Full Name", emp.get("full_name", ""))
        e_email = create_entry("Email", emp.get("email", ""))
        
        DEPARTMENTS = ["IT", "HR", "Finance", "Operations", "Management"]
        ROLES       = ["Employee", "Senior Employee", "Team Lead"]
        LOCATIONS   = ["Office", "Home", "Hybrid"]
        
        v_dept = create_dropdown("Department", emp.get("department", "IT"), DEPARTMENTS)
        v_role = create_dropdown("Role", emp.get("role", "Employee"), ROLES)
        e_shift_start = create_entry("Shift Start", emp.get("shift_start", "09:00"))
        e_shift_end = create_entry("Shift End", emp.get("shift_end", "18:00"))
        v_loc = create_dropdown("Work Location", emp.get("work_location", "Office"), LOCATIONS)

        def save():
            if not self._db or not self._db.is_connected: return
            new_name = e_name.get().strip()
            new_email = e_email.get().strip()
            if not new_name: return messagebox.showerror("Error", "Name required", parent=win)
            
            update_data = {
                "full_name": new_name,
                "email": new_email,
                "department": v_dept.get(),
                "role": v_role.get(),
                "shift_start": e_shift_start.get().strip(),
                "shift_end": e_shift_end.get().strip(),
                "work_location": v_loc.get()
            }
            try:
                col = self._db.get_collection("employees")
                col.update_one({"employee_id": eid}, {"$set": update_data})
                messagebox.showinfo("Success", "Employee updated.", parent=win)
                win.destroy()
                self._refresh_employees()
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=win)
                
        ctk.CTkButton(win, text="Save Changes", fg_color=C_TEAL, hover_color=C_TEAL_D, command=save).pack(pady=(10, 20))

    def _delete_employee(self, emp: dict) -> None:
        eid = emp.get("employee_id")
        msg = f"Are you sure you want to permanently delete {eid} ({emp.get('full_name')})?\n\nThis action cannot be undone."
        if not messagebox.askyesno("Delete Employee", msg):
            return
            
        try:
            if not self._db or not self._db.is_connected: return
            
            # 1. Delete Employee
            e_col = self._db.get_collection("employees")
            e_col.delete_one({"employee_id": eid})
            
            # 2. Invalidate sessions
            s_col = self._db.get_collection("sessions")
            s_col.update_many({"employee_id": eid}, {"$set": {"status": "terminated"}})
            
            messagebox.showinfo("Deleted", f"Employee {eid} has been removed.")
            self._refresh_employees()
            
        except Exception as exc:
            messagebox.showerror("Error", f"Could not delete: {exc}")


    def _open_registration(self) -> None:
        try:
            from dashboard.employee_registration import EmployeeRegistration
            EmployeeRegistration(self, db=self._db)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not open registration: {exc}")

    # ------------------------------------------------------------------
    # Alerts Tab
    # ------------------------------------------------------------------

    def _employee_name(self, emp_id: str) -> str:
        if not emp_id:
            return "Unknown"
        cached = self._employee_name_cache.get(emp_id)
        if cached:
            return cached
        try:
            col = self._db.get_collection("employees") if self._db else None
            doc = col.find_one({"employee_id": emp_id}, {"_id": 0, "full_name": 1}) if col is not None else None
            name = (doc or {}).get("full_name") or emp_id
            self._employee_name_cache[emp_id] = name
            return name
        except Exception:
            return emp_id

    def _alert_dedupe_key(self, alert: dict) -> str:
        factors = ",".join(sorted(alert.get("factors", []) or []))
        return "|".join(
            [
                str(alert.get("user_id", "")),
                str(alert.get("timestamp", "")),
                str(alert.get("level", "")).upper(),
                str(round(float(alert.get("risk_score", 0.0)), 2)),
                factors,
            ]
        )

    def _persist_realtime_alert(self, alert: dict) -> dict:
        if not self._db or not self._db.is_connected:
            return alert
        # Idle timeout alerts are already persisted by initialize_monitoring.
        if alert.get("reason") == "idle_inactivity":
            return alert
        try:
            col = self._db.get_collection("alerts")
            if col is None:
                return alert
            dedupe_key = self._alert_dedupe_key(alert)
            existing = col.find_one({"dedupe_key": dedupe_key}, {"_id": 1})
            if existing:
                merged = dict(alert)
                merged["_id"] = existing.get("_id")
                return merged
            doc = {
                "type": "alert",
                "timestamp": alert.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "user_id": alert.get("user_id", "?"),
                "session_id": alert.get("session_id"),
                "risk_score": float(alert.get("risk_score", 0.0)),
                "level": str(alert.get("level", "LOW")).upper(),
                "factors": list(alert.get("factors", [])),
                "resolved": False,
                "dedupe_key": dedupe_key,
            }
            for k, v in alert.items():
                if k not in doc:
                    doc[k] = v
            result = col.insert_one(doc)
            doc["_id"] = result.inserted_id
            return doc
        except Exception:
            return alert
        return alert

    async def _ws_message_handler(self, websocket) -> None:
        async for raw in websocket:
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if payload.get("type") != "alert":
                continue
            persisted = self._persist_realtime_alert(payload)
            self.after(0, lambda p=persisted: self._add_realtime_alert_card(p))

    async def _ws_server_task(self, host: str, port: int) -> None:
        if not _HAS_WEBSOCKETS:
            return
        self._ws_server = await websockets.serve(self._ws_message_handler, host, port)
        await self._ws_server.wait_closed()

    def _start_alert_ws_server(self) -> None:
        if not _HAS_WEBSOCKETS:
            return
        try:
            parsed = urlparse(settings.WEBSOCKET_URL)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8765
        except Exception:
            host, port = "localhost", 8765

        def _run_server() -> None:
            try:
                loop = asyncio.new_event_loop()
                self._ws_loop = loop
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._ws_server_task(host, port))
            except Exception:
                pass
            finally:
                if self._ws_loop is not None:
                    try:
                        self._ws_loop.close()
                    except Exception:
                        pass

        threading.Thread(target=_run_server, daemon=True, name="AdminAlertWebSocketServer").start()

    def _open_break_config_popup(self) -> None:
        """Allow admin to configure the shared break schedule."""
        try:
            from C3_activity_monitoring.src.break_manager import BreakManager
            bm = BreakManager()
            bm.load_breaks()
            bm.show_configure_popup(parent=self)
        except Exception as exc:
            messagebox.showerror("Break Configuration", f"Could not open break config: {exc}")

    def _build_alerts_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        topbar = ctk.CTkFrame(frame, fg_color="transparent")
        topbar.pack(fill="x", padx=20, pady=(16, 8))
        ctk.CTkLabel(topbar, text="Alert Feed", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(side="left")
        ctk.CTkButton(topbar, text="Refresh", width=90, fg_color=C_BORDER, hover_color=C_BLUE, command=lambda: self._refresh_alerts(force=True)).pack(side="right")
        ctk.CTkButton(
            topbar,
            text="Delete Old",
            width=100,
            fg_color="#7f1d1d",
            hover_color="#991b1b",
            command=self._delete_old_alerts,
        ).pack(side="right", padx=(0, 8))
        ctk.CTkButton(topbar, text="Configure Breaks", width=130, fg_color=C_TEAL_D, hover_color=C_TEAL,
                      command=self._open_break_config_popup).pack(side="right", padx=(0, 8))

        self._alerts_frame = ctk.CTkScrollableFrame(frame, fg_color=C_BG)
        self._alerts_frame.pack(fill="both", expand=True, padx=20, pady=(4, 16))
        return frame

    def _refresh_alerts(self, force: bool = False) -> None:
        if not self._db or not self._db.is_connected:
            return

        # Avoid overlapping refreshes when polling is active.
        if self._alerts_refresh_inflight:
            return
        self._alerts_refresh_inflight = True

        if force:
            for w in self._alerts_frame.winfo_children():
                w.destroy()
            ctk.CTkLabel(self._alerts_frame, text="Loading alerts...", text_color=C_MUTED).pack(anchor="w", padx=8, pady=8)

        self._alerts_executor.submit(self._fetch_alerts_worker)

    def _fetch_alerts_worker(self) -> None:
        try:
            col = self._db.get_collection("alerts")
            if col is None:
                self.after(0, lambda: self._apply_alerts_refresh([], "", None))
                return

            projection = {
                "_id": 1,
                "timestamp": 1,
                "user_id": 1,
                "employee_name": 1,
                "risk_score": 1,
                "level": 1,
                "factors": 1,
                "resolved": 1,
            }
            alerts = list(
                col.find({"resolved": {"$ne": True}}, projection)
                .sort("timestamp", -1)
                .limit(50)
            )

            signature_parts = []
            for a in alerts:
                signature_parts.append(
                    f"{a.get('_id')}|{a.get('timestamp')}|{a.get('level')}|{a.get('risk_score')}"
                )
            signature = "\n".join(signature_parts)
            self.after(0, lambda rows=alerts, sig=signature: self._apply_alerts_refresh(rows, sig, None))
        except Exception as exc:
            self.after(0, lambda err=str(exc): self._apply_alerts_refresh([], "", err))

    def _apply_alerts_refresh(self, alerts: list, signature: str, error: Optional[str]) -> None:
        self._alerts_refresh_inflight = False

        if error:
            for w in self._alerts_frame.winfo_children():
                w.destroy()
            ctk.CTkLabel(self._alerts_frame, text=f"Error: {error}", text_color=C_RED).pack(anchor="w", padx=8, pady=8)
            return

        # Skip expensive UI rebuild when data did not change.
        if signature == self._alerts_last_signature:
            return
        self._alerts_last_signature = signature

        for w in self._alerts_frame.winfo_children():
            w.destroy()

        if not alerts:
            ctk.CTkLabel(self._alerts_frame, text="No active alerts.", text_color=C_MUTED).pack(anchor="w", padx=8, pady=8)
            return

        for alert in alerts:
            self._add_alert_card(alert)

    def _delete_old_alerts(self) -> None:
        if not self._db or not self._db.is_connected:
            return

        msg = (
            "Delete old alerts now?\n\n"
            "This removes:\n"
            "- Resolved alerts\n"
            "- Alerts older than 30 days\n\n"
            "Unresolved recent alerts are kept."
        )
        if not messagebox.askyesno("Delete Old Alerts", msg):
            return

        try:
            col = self._db.get_collection("alerts")
            if col is None:
                return

            cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
            result = col.delete_many(
                {
                    "$or": [
                        {"resolved": True},
                        {"timestamp": {"$lt": cutoff}},
                    ]
                }
            )
            self._alerts_last_signature = ""
            self._refresh_alerts(force=True)
            messagebox.showinfo("Delete Old Alerts", f"Deleted {result.deleted_count} alert(s).")
        except Exception as exc:
            messagebox.showerror("Delete Old Alerts", str(exc))

    def _add_realtime_alert_card(self, alert: dict) -> None:
        """Safely add realtime alert card; avoids UI race issues during refresh."""
        if self._closed or not hasattr(self, "_alerts_frame"):
            return
        try:
            if not self._alerts_frame.winfo_exists():
                return
        except Exception:
            return

        # Force next poll refresh to reconcile with DB order/state.
        self._alerts_last_signature = ""
        try:
            self._add_alert_card(alert, prepend=True)
        except Exception as exc:
            logger.debug("Realtime alert insert skipped: %s", exc)

    def _add_alert_card(self, alert: dict, prepend: bool = False) -> None:
        if not hasattr(self, "_alerts_frame"):
            return
        try:
            if not self._alerts_frame.winfo_exists():
                return
        except Exception:
            return

        level = alert.get("level", "LOW").upper()
        color = _level_color(level)
        emp_id = str(alert.get("user_id", "?"))
        emp_name = alert.get("employee_name") or self._employee_name(emp_id)

        card = ctk.CTkFrame(self._alerts_frame, fg_color=C_CARD, corner_radius=12)
        if prepend:
            before_widget = None
            for child in self._alerts_frame.winfo_children():
                try:
                    if child.winfo_exists() and child.winfo_manager() == "pack":
                        before_widget = child
                        break
                except Exception:
                    continue

            if before_widget is not None:
                try:
                    card.pack(fill="x", pady=4, before=before_widget)
                except Exception:
                    card.pack(fill="x", pady=4)
            else:
                card.pack(fill="x", pady=4)
        else:
            card.pack(fill="x", pady=4)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(12, 4))

        ctk.CTkLabel(top, text=f"  {level}  ", fg_color=color, corner_radius=6,
                     font=ctk.CTkFont(size=11, weight="bold"), text_color="#fff", width=70).pack(side="left")
        ctk.CTkLabel(top, text=f"  {emp_name} ({emp_id})", text_color=C_TEXT, font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=8)
        ctk.CTkLabel(top, text=f"Risk: {alert.get('risk_score', 0):.1f}", text_color=_risk_color(alert.get("risk_score", 0)), font=ctk.CTkFont(size=12)).pack(side="left", padx=12)
        ctk.CTkLabel(top, text=_fmt_time(alert.get("timestamp", "")), text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(side="right")

        factors_str = ", ".join(alert.get("factors", [])) or "No factors listed"
        ctk.CTkLabel(card, text=factors_str, text_color=C_MUTED, font=ctk.CTkFont(size=11), anchor="w").pack(fill="x", padx=16, pady=(0, 6))

        btn_row = ctk.CTkFrame(card, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))

        alert_id = str(alert.get("_id", ""))
        ctk.CTkButton(btn_row, text="Mark Resolved", width=120, height=28, fg_color="#166534",
                      hover_color="#15803d", font=ctk.CTkFont(size=11),
                      command=lambda aid=alert_id: self._mark_resolved(aid, card)).pack(side="left", padx=(0, 8))
        ctk.CTkButton(btn_row, text="View Details", width=120, height=28, fg_color=C_BORDER,
                      hover_color=C_BLUE, font=ctk.CTkFont(size=11),
                      command=lambda a=alert: self._view_emp_from_alert(a)).pack(side="left")

        if level == "CRITICAL":
            dedupe_key = self._alert_dedupe_key(alert)
            if dedupe_key not in self._critical_sound_seen:
                self._critical_sound_seen.add(dedupe_key)
                _play_alert_sound()

    def _mark_resolved(self, alert_id: str, card_widget) -> None:
        try:
            from bson import ObjectId
            col = self._db.get_collection("alerts")
            if col is not None and alert_id:
                query = None
                try:
                    query = {"_id": ObjectId(alert_id)}
                except Exception:
                    query = {"_id": alert_id}
                col.update_one(query, {"$set": {"resolved": True, "resolved_at": datetime.now(timezone.utc).isoformat()}})
            self._alerts_last_signature = ""
            card_widget.destroy()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _view_emp_from_alert(self, alert: dict) -> None:
        emp_id = alert.get("user_id")
        try:
            col = self._db.get_collection("employees")
            emp = col.find_one({"employee_id": emp_id}, {"_id": 0, "password_hash": 0, "face_images": 0, "face_embedding": 0}) if col is not None else None
            if emp:
                EmployeeDetailWindow(self, emp, self._db)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tasks Tab
    # ------------------------------------------------------------------

    def _build_tasks_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        top = ctk.CTkFrame(frame, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(16, 0))

        # Assignment form
        form = ctk.CTkFrame(top, fg_color=C_CARD, corner_radius=14)
        form.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(form, text="Assign New Task", font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=16, pady=(14, 8))

        row1 = ctk.CTkFrame(form, fg_color="transparent")
        row1.pack(fill="x", padx=16)

        # Employee dropdown
        ctk.CTkLabel(row1, text="Employee:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=90, anchor="w").pack(side="left")
        self._task_emp_var = ctk.StringVar(value="Select employee")
        self._task_emp_dd = ctk.CTkOptionMenu(row1, variable=self._task_emp_var, values=["Loading..."], width=180, fg_color=C_BORDER, button_color=C_BORDER)
        self._task_emp_dd.pack(side="left", padx=(0, 16))
        self._refresh_employee_dropdown()

        # Priority
        ctk.CTkLabel(row1, text="Priority:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=60, anchor="w").pack(side="left")
        self._task_priority = ctk.StringVar(value="medium")
        ctk.CTkOptionMenu(row1, variable=self._task_priority, values=["low", "medium", "high"], width=100, fg_color=C_BORDER, button_color=C_BORDER).pack(side="left", padx=(0, 16))

        # Due date
        ctk.CTkLabel(row1, text="Due:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=40, anchor="w").pack(side="left")
        if _HAS_CALENDAR:
            self._due_date = DateEntry(row1, width=12, background="#1a1d27", foreground="white", borderwidth=0, date_pattern="yyyy-mm-dd")
            # Disallow selecting past dates from calendar UI.
            try:
                self._due_date.config(mindate=datetime.now().date())
            except Exception:
                pass
            self._due_date.pack(side="left")
        else:
            self._due_var = ctk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
            ctk.CTkEntry(row1, textvariable=self._due_var, width=110).pack(side="left")

        ctk.CTkLabel(row1, text="Time:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=46, anchor="w").pack(side="left", padx=(12, 0))
        self._due_time_var = ctk.StringVar(value="17:00")
        ctk.CTkEntry(row1, textvariable=self._due_time_var, width=80, placeholder_text="HH:MM").pack(side="left")

        ctk.CTkLabel(row1, text="Allocated:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=72, anchor="w").pack(side="left", padx=(12, 0))
        self._task_alloc_minutes_var = ctk.StringVar(value="60")
        ctk.CTkEntry(row1, textvariable=self._task_alloc_minutes_var, width=70, placeholder_text="min").pack(side="left")

        row2 = ctk.CTkFrame(form, fg_color="transparent")
        row2.pack(fill="x", padx=16, pady=(8, 0))
        ctk.CTkLabel(row2, text="Title:", text_color=C_MUTED, font=ctk.CTkFont(size=12), width=90, anchor="w").pack(side="left")
        self._task_title_var = ctk.StringVar()
        ctk.CTkEntry(row2, textvariable=self._task_title_var, placeholder_text="Task title...", width=400).pack(side="left", fill="x", expand=True)

        row3 = ctk.CTkFrame(form, fg_color="transparent")
        row3.pack(fill="x", padx=16, pady=(8, 0))
        ctk.CTkLabel(row3, text="Description:", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(anchor="w")
        self._task_desc_box = ctk.CTkTextbox(form, height=80, fg_color=C_BG, border_color=C_BORDER)
        self._task_desc_box.pack(fill="x", padx=16, pady=(4, 0))

        ctk.CTkButton(form, text="Assign Task", fg_color=C_TEAL, hover_color=C_TEAL_D, height=38,
                      command=self._assign_task).pack(padx=16, pady=12, anchor="e")

        # Task list
        ctk.CTkLabel(frame, text="All Tasks", font=ctk.CTkFont(size=13, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=24, pady=(4, 6))

        hdr = ctk.CTkFrame(frame, fg_color=C_SIDEBAR, corner_radius=8, height=34)
        hdr.pack(fill="x", padx=20, pady=(0, 4))
        for col, w in [
            ("Title", 260), ("Employee", 95), ("Priority", 80), ("Due", 170),
            ("Allocated", 95), ("Actual", 100), ("Status", 100), ("", 160),
        ]:
            ctk.CTkLabel(hdr, text=col, font=ctk.CTkFont(size=11), text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=4)

        self._task_list_frame = ctk.CTkScrollableFrame(frame, fg_color=C_BG)
        self._task_list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        return frame

    def _refresh_employee_dropdown(self) -> None:
        if not self._db or not self._db.is_connected:
            return
        try:
            col = self._db.get_collection("employees")
            if col is not None:
                emps = list(col.find({}, {"employee_id": 1, "full_name": 1, "_id": 0}))
                values = [f"{e['employee_id']} — {e.get('full_name','')}" for e in emps]
                values = values or ["No employees"]

                current = self._task_emp_var.get().strip()
                self._task_emp_dd.configure(values=values)

                if current and current in values:
                    self._task_emp_var.set(current)
                elif current in {"", "Loading...", "No employees", "Select employee"}:
                    self._task_emp_var.set(values[0])
        except Exception:
            pass

    def _assign_task(self) -> None:
        title = self._task_title_var.get().strip()
        desc = self._task_desc_box.get("1.0", "end").strip()
        emp_raw = self._task_emp_var.get()
        priority = self._task_priority.get()

        if not title or not emp_raw or "Select" in emp_raw:
            messagebox.showwarning("Validation", "Employee and title are required.")
            return

        emp_id = emp_raw.split("—")[0].strip()
        due = ""
        if _HAS_CALENDAR:
            due = self._due_date.get_date().isoformat()
        else:
            due = self._due_var.get()

        due_time = self._due_time_var.get().strip()
        if not _is_hhmm(due_time):
            messagebox.showwarning("Validation", "Due time must be in HH:MM format (24-hour).")
            return

        due_dt_local = _parse_due_datetime_local(due, due_time)
        if due_dt_local is None:
            messagebox.showwarning("Validation", "Due date/time is invalid.")
            return

        if due_dt_local < datetime.now():
            messagebox.showwarning("Validation", "Due date/time cannot be in the past.")
            return

        try:
            allocated_minutes = max(1, int(self._task_alloc_minutes_var.get().strip() or "0"))
        except Exception:
            messagebox.showwarning("Validation", "Allocated time must be a valid number of minutes.")
            return

        due_at = _combine_due_datetime(due, due_time)

        task_doc = {
            "task_id": str(uuid.uuid4()),
            "employee_id": emp_id,
            "title": title,
            "description": desc,
            "due_date": due,
            "due_time": due_time,
            "due_at": due_at,
            "priority": priority,
            "status": "pending",
            "assigned_by": "ADMIN",
            "assigned_at": datetime.utcnow().isoformat(),
            "allocated_minutes": allocated_minutes,
            "actual_seconds": 0,
            "last_started_at": None,
            "started_at": None,
            "completed_at": None,
        }

        try:
            col = self._db.get_collection("tasks")
            if col is not None:
                col.insert_one(task_doc)
                messagebox.showinfo("Success", f"Task '{title}' assigned to {emp_id}.")
                self._task_title_var.set("")
                self._task_desc_box.delete("1.0", "end")
                self._task_alloc_minutes_var.set("60")
                self._refresh_task_list()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def _fetch_tasks(self) -> None:
        if self._closed or not self.winfo_exists():
            return
        if not self._db or not self._db.is_connected:
            return
        try:
            col = self._db.get_collection("tasks")
            if col is None: return
            tasks = list(col.find({}, {"_id": 0}).sort("assigned_at", -1).limit(40))
            if self._closed or not self.winfo_exists():
                return
            self.after(0, lambda: self._update_task_ui(tasks))
        except Exception as exc:
            if self._closed or not self.winfo_exists():
                return
            self.after(0, lambda: self._update_task_ui([], error=str(exc)))

    def _update_task_ui(self, tasks: list, error: str = "") -> None:
        if self._closed or not self.winfo_exists() or not hasattr(self, "_task_list_frame"):
            return
        if not self._task_list_frame.winfo_exists():
            return
        for w in self._task_list_frame.winfo_children():
            try:
                if w.winfo_exists():
                    w.destroy()
            except Exception:
                pass
        if error:
            ctk.CTkLabel(self._task_list_frame, text=error, text_color=C_RED).pack()
            return
        for t in tasks:
            status = t.get("status", "pending")
            s_color = {"pending": C_MUTED, "in_progress": C_AMBER, "completed": C_GREEN}.get(status, C_MUTED)
            p_color = {"low": C_BLUE, "medium": C_AMBER, "high": C_RED}.get(t.get("priority", ""), C_MUTED)

            allocated_minutes = int(t.get("allocated_minutes", 0) or 0)
            actual_seconds = int(t.get("actual_seconds", 0) or 0)
            if status == "in_progress":
                run_start = _parse_iso_timestamp(t.get("last_started_at") or t.get("started_at") or "")
                if run_start is not None:
                    actual_seconds += max(0, int((datetime.now(run_start.tzinfo) - run_start).total_seconds()))

            due_date = str(t.get("due_date", "") or "")
            due_time = str(t.get("due_time", "") or "")
            due_text = _fmt_due_display(due_date, due_time)
            due_text_color = _due_color(due_date, status)

            row = ctk.CTkFrame(self._task_list_frame, fg_color=C_CARD, corner_radius=10, height=46)
            row.pack(fill="x", pady=3)
            row.pack_propagate(False)

            ctk.CTkLabel(row, text=_clip_text(t.get("title", "?"), 34), text_color=C_TEXT, font=ctk.CTkFont(size=12), width=260, anchor="w").pack(side="left", padx=(8, 4))
            ctk.CTkLabel(row, text=t.get("employee_id", "?"), text_color=C_MUTED, font=ctk.CTkFont(size=11), width=95, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=t.get("priority", "").title(), text_color=p_color, font=ctk.CTkFont(size=11), width=80, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=due_text, text_color=due_text_color, font=ctk.CTkFont(size=11), width=170, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=_fmt_minutes(allocated_minutes), text_color=C_MUTED, font=ctk.CTkFont(size=11), width=95, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=_fmt_seconds(actual_seconds), text_color=C_MUTED, font=ctk.CTkFont(size=11), width=100, anchor="w").pack(side="left", padx=4)
            ctk.CTkLabel(row, text=status.replace("_", " ").title(), text_color=s_color, font=ctk.CTkFont(size=11), width=100, anchor="w").pack(side="left", padx=4)

            # --- Action Buttons ---
            btn_frame = ctk.CTkFrame(row, fg_color="transparent", width=160)
            btn_frame.pack(side="right", padx=8)
            btn_frame.pack_propagate(False)

            ctk.CTkButton(btn_frame, text="✏️ Edit", width=60, height=28, fg_color=C_BLUE, hover_color="#2563eb",
                          font=ctk.CTkFont(size=11), command=lambda task_obj=t: self._edit_task(task_obj)).pack(side="left", padx=4)
            ctk.CTkButton(btn_frame, text="❌ Delete", width=70, height=28, fg_color="#450a0a", hover_color=C_RED,
                          font=ctk.CTkFont(size=11), command=lambda task_obj=t: self._delete_task(task_obj)).pack(side="left", padx=4)

    def _refresh_task_list(self) -> None:
        threading.Thread(target=self._fetch_tasks, daemon=True).start()

    def _delete_task(self, task: dict) -> None:
        """Confirm and delete a task from MongoDB."""
        tid = task.get("task_id")
        title = task.get("title", "Untitled")
        if not messagebox.askyesno("Delete Task", f"Are you sure you want to delete the task '{title}'?"):
            return

        try:
            col = self._db.get_collection("tasks")
            if col is not None:
                col.delete_one({"task_id": tid})
                messagebox.showinfo("Success", "Task deleted successfully.")
                self._refresh_task_list()
        except Exception as exc:
            messagebox.showerror("Error", f"Could not delete task: {exc}")

    def _edit_task(self, task: dict) -> None:
        """Open a popup to edit task title, description, and status."""
        tid = task.get("task_id")
        win = ctk.CTkToplevel(self)
        win.title(f"Edit Task - {tid[:8]}")
        win.geometry("450x560")
        win.attributes("-topmost", True)
        win.configure(fg_color=C_BG)

        ctk.CTkLabel(win, text="Edit Task Information", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(pady=(20, 10))

        form = ctk.CTkScrollableFrame(win, fg_color="transparent")
        form.pack(fill="both", expand=True, padx=20, pady=10)

        # Title
        ctk.CTkLabel(form, text="Title", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        e_title = ctk.CTkEntry(form, height=36, fg_color="#0f1117", border_color=C_BORDER)
        e_title.pack(fill="x")
        e_title.insert(0, task.get("title", ""))

        # Description
        ctk.CTkLabel(form, text="Description", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        t_desc = ctk.CTkTextbox(form, height=80, fg_color="#0f1117", border_color=C_BORDER)
        t_desc.pack(fill="x")
        t_desc.insert("1.0", task.get("description", ""))

        # Status
        ctk.CTkLabel(form, text="Status", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        v_status = ctk.StringVar(value=task.get("status", "pending"))
        ctk.CTkOptionMenu(form, variable=v_status, values=["pending", "in_progress", "completed"], 
                          fg_color=C_BORDER, button_color=C_BORDER).pack(fill="x")

        # Priority
        ctk.CTkLabel(form, text="Priority", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        v_priority = ctk.StringVar(value=task.get("priority", "medium"))
        ctk.CTkOptionMenu(form, variable=v_priority, values=["low", "medium", "high"], 
                          fg_color=C_BORDER, button_color=C_BORDER).pack(fill="x")

        # Due date
        ctk.CTkLabel(form, text="Due Date (YYYY-MM-DD)", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        v_due_date = ctk.StringVar(value=str(task.get("due_date", "")))
        ctk.CTkEntry(form, textvariable=v_due_date, height=36, fg_color="#0f1117", border_color=C_BORDER).pack(fill="x")

        # Due time
        ctk.CTkLabel(form, text="Due Time (HH:MM)", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        v_due_time = ctk.StringVar(value=str(task.get("due_time", "17:00")))
        ctk.CTkEntry(form, textvariable=v_due_time, height=36, fg_color="#0f1117", border_color=C_BORDER).pack(fill="x")

        # Allocated time
        ctk.CTkLabel(form, text="Allocated Minutes", text_color=C_MUTED, font=ctk.CTkFont(size=12), anchor="w").pack(fill="x", pady=(10, 2))
        v_alloc = ctk.StringVar(value=str(task.get("allocated_minutes", 60)))
        ctk.CTkEntry(form, textvariable=v_alloc, height=36, fg_color="#0f1117", border_color=C_BORDER).pack(fill="x")

        def save_changes():
            if not self._db or not self._db.is_connected: return
            new_title = e_title.get().strip()
            new_desc = t_desc.get("1.0", "end").strip()
            if not new_title:
                messagebox.showerror("Error", "Title cannot be empty.", parent=win)
                return

            new_due_date = v_due_date.get().strip()
            new_due_time = v_due_time.get().strip()
            if new_due_time and not _is_hhmm(new_due_time):
                messagebox.showerror("Error", "Due time must be in HH:MM format (24-hour).", parent=win)
                return

            try:
                allocated_minutes = max(1, int(v_alloc.get().strip() or "0"))
            except Exception:
                messagebox.showerror("Error", "Allocated minutes must be a number.", parent=win)
                return

            update_data = {
                "title": new_title,
                "description": new_desc,
                "status": v_status.get(),
                "priority": v_priority.get(),
                "due_date": new_due_date,
                "due_time": new_due_time,
                "due_at": _combine_due_datetime(new_due_date, new_due_time) if (new_due_date and new_due_time) else "",
                "allocated_minutes": allocated_minutes,
            }
            # Special case for completion timestamp
            if v_status.get() == "completed" and task.get("status") != "completed":
                update_data["completed_at"] = datetime.utcnow().isoformat()
            
            try:
                col = self._db.get_collection("tasks")
                if col is not None:
                    col.update_one({"task_id": tid}, {"$set": update_data})
                    messagebox.showinfo("Success", "Task updated.", parent=win)
                    win.destroy()
                    self._refresh_task_list()
            except Exception as exc:
                messagebox.showerror("Error", str(exc), parent=win)

        ctk.CTkButton(win, text="Save Changes", fg_color=C_TEAL, hover_color=C_TEAL_D, height=40,
                      command=save_changes).pack(pady=(10, 20), padx=20, fill="x")

    # ------------------------------------------------------------------
    # Attendance Tab
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_from_hms(value: Optional[str]) -> int:
        """Convert HH:MM:SS to seconds; return 0 when value is missing/invalid."""
        if not value:
            return 0
        try:
            parts = [int(p) for p in str(value).split(":")]
            if len(parts) != 3:
                return 0
            h, m, s = parts
            return max(0, (h * 3600) + (m * 60) + s)
        except Exception:
            return 0

    @staticmethod
    def _fmt_hms(seconds: int) -> str:
        seconds = max(0, int(seconds))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def _parse_date_hms(date_str: Optional[str], time_str: Optional[str]) -> Optional[datetime]:
        """Parse attendance date + HH:MM:SS into datetime."""
        d = str(date_str or "").strip()
        t = str(time_str or "").strip()
        if not d or not t or t == "—":
            return None
        try:
            return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    def _derive_attendance_status(self, signin: Optional[str], signout: Optional[str], existing_status: Optional[str]) -> str:
        """Return one of: On Time / Late / Early Departure / Overtime."""
        base_status = str(existing_status or "").strip()
        if base_status in {"On Time", "Late", "Early Departure", "Overtime"}:
            return base_status

        sign_in_cutoff = datetime.strptime("09:15:00", "%H:%M:%S").time()
        early_departure_cutoff = datetime.strptime("17:00:00", "%H:%M:%S").time()
        overtime_cutoff = datetime.strptime("18:00:00", "%H:%M:%S").time()

        sign_in_time = None
        sign_out_time = None
        try:
            if signin:
                sign_in_time = datetime.strptime(signin, "%H:%M:%S").time()
        except Exception:
            sign_in_time = None
        try:
            if signout:
                sign_out_time = datetime.strptime(signout, "%H:%M:%S").time()
        except Exception:
            sign_out_time = None

        if sign_out_time is not None:
            if sign_out_time >= overtime_cutoff:
                return "Overtime"
            if sign_out_time < early_departure_cutoff:
                return "Early Departure"

        if sign_in_time is not None and sign_in_time > sign_in_cutoff:
            return "Late"
        return "On Time"

    def _build_attendance_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)

        filter_bar = ctk.CTkFrame(frame, fg_color=C_CARD, corner_radius=10)
        filter_bar.pack(fill="x", padx=20, pady=(16, 8))

        ctk.CTkLabel(filter_bar, text="Filter:", text_color=C_MUTED, font=ctk.CTkFont(size=12)).pack(side="left", padx=12, pady=10)

        if _HAS_CALENDAR:
            self._att_date = DateEntry(filter_bar, width=12, background="#1a1d27", foreground="white", date_pattern="yyyy-mm-dd")
            self._att_date.pack(side="left", padx=(0, 12), pady=10)
        else:
            self._att_date_var = ctk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
            ctk.CTkEntry(filter_bar, textvariable=self._att_date_var, width=110).pack(side="left", padx=4, pady=10)

        self._att_emp_var = ctk.StringVar(value="All Employees")
        ctk.CTkEntry(filter_bar, textvariable=self._att_emp_var, placeholder_text="Employee ID...", width=140).pack(side="left", padx=4)
        ctk.CTkButton(filter_bar, text="Search", fg_color=C_TEAL, hover_color=C_TEAL_D, width=80, command=self._refresh_attendance).pack(side="left", padx=8)

        # Table header
        hdr = ctk.CTkFrame(frame, fg_color=C_SIDEBAR, corner_radius=8, height=34)
        hdr.pack(fill="x", padx=20, pady=(0, 2))
        for col, w in [
            ("Name", 210), ("ID", 90), ("Date", 110), ("Sign-in", 95), ("Sign-out", 95),
            ("Active Duration", 130), ("Idle Duration", 120), ("Status", 100),
        ]:
            ctk.CTkLabel(hdr, text=col, font=ctk.CTkFont(size=11), text_color=C_MUTED, width=w, anchor="w").pack(side="left", padx=4, pady=8)

        self._att_list_frame = ctk.CTkScrollableFrame(frame, fg_color=C_BG)
        self._att_list_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        return frame

    def _refresh_attendance(self) -> None:
        if _HAS_CALENDAR:
            date_str = self._att_date.get_date().isoformat()
        else:
            date_str = self._att_date_var.get()
        emp_filter = self._att_emp_var.get().strip()
        threading.Thread(target=self._fetch_attendance, args=(date_str, emp_filter), daemon=True).start()

    def _fetch_attendance(self, date_str: str, emp_filter: str) -> None:
        if not self._db or not self._db.is_connected:
            return
        try:
            col = self._db.get_collection("attendance_logs")
            if col is None: return
            query = {}
            if date_str:
                query["date"] = date_str
            if emp_filter and emp_filter != "All Employees":
                emp_text = emp_filter.split("—")[0].strip()
                esc = re.escape(emp_text)
                query["$or"] = [
                    {"employee_id": {"$regex": f"^{esc}$", "$options": "i"}},
                    {"full_name": {"$regex": esc, "$options": "i"}},
                ]

            docs = list(col.find(query, {"_id": 0}).sort("signin", -1).limit(200))

            active_sessions = set(self._get_active_sessions_by_employee().keys())

            idle_seconds_by_user: dict[str, int] = {}
            try:
                activity_col = self._db.get_collection("activity_logs")
                if activity_col is not None and date_str:
                    act_query: dict = {
                        "timestamp": {"$regex": f"^{re.escape(date_str)}"},
                    }
                    if emp_filter and emp_filter != "All Employees":
                        emp_text = emp_filter.split("—")[0].strip()
                        act_query["user_id"] = {"$regex": f"^{re.escape(emp_text)}$", "$options": "i"}

                    for act in activity_col.find(act_query, {"_id": 0, "user_id": 1, "idle_ratio": 1}):
                        uid = str(act.get("user_id", "")).strip()
                        if not uid:
                            continue
                        ratio = float(act.get("idle_ratio", 0.0) or 0.0)
                        ratio = max(0.0, min(1.0, ratio))
                        idle_seconds_by_user[uid] = idle_seconds_by_user.get(uid, 0) + int(round(ratio * 60.0))
            except Exception:
                pass

            self.after(0, lambda: self._update_attendance_ui(docs, active_sessions, idle_seconds_by_user))
        except Exception as exc:
            self.after(0, lambda: self._update_attendance_ui([], set(), {}, error=str(exc)))

    def _parse_iso_dt(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _get_active_sessions_by_employee(self, allowed_employee_ids: Optional[set[str]] = None) -> dict:
        """
        Return active sessions keyed by employee_id, deduped by latest login time,
        and filtered to recent sessions to avoid stale online state.
        """
        active_by_emp: dict[str, dict] = {}
        if not self._db or not self._db.is_connected:
            return active_by_emp

        cutoff = datetime.now(timezone.utc) - timedelta(hours=ONLINE_SESSION_MAX_AGE_HOURS)
        try:
            scol = self._db.get_collection("sessions")
            if scol is None:
                return active_by_emp

            rows = list(scol.find({"status": "active"}, {"_id": 0}).sort("login_at", -1).limit(5000))
            for sess in rows:
                eid = str(sess.get("employee_id") or "").strip()
                if not eid:
                    continue
                if allowed_employee_ids is not None and eid not in allowed_employee_ids:
                    continue
                if eid in active_by_emp:
                    continue

                login_dt = self._parse_iso_dt(sess.get("login_at"))
                if login_dt is None or login_dt < cutoff:
                    continue

                active_by_emp[eid] = sess
        except Exception:
            return {}

        return active_by_emp

    def _update_attendance_ui(self, docs: list, active_sessions: set, idle_seconds_by_user: dict[str, int], error: str = "") -> None:
        for w in self._att_list_frame.winfo_children():
            w.destroy()
        if error:
            ctk.CTkLabel(self._att_list_frame, text=error, text_color=C_RED).pack()
            return
        today_str = datetime.now().strftime("%Y-%m-%d")
        for d in docs:
            eid = str(d.get("employee_id", "")).strip()
            sign_in = d.get("signin") or "—"
            sign_out = d.get("signout") or "—"
            row_date = str(d.get("date", "")).strip()

            total_seconds = self._seconds_from_hms(d.get("duration"))
            if total_seconds <= 0 and sign_in not in (None, "—"):
                try:
                    s_dt = self._parse_date_hms(row_date, sign_in)
                    if sign_out not in (None, "—"):
                        e_dt = self._parse_date_hms(row_date, sign_out)
                        if s_dt is not None and e_dt is not None and e_dt < s_dt:
                            # Handle midnight crossover.
                            e_dt = e_dt + timedelta(days=1)
                    elif eid in active_sessions and d.get("date") == today_str:
                        e_dt = datetime.now().replace(microsecond=0)
                    else:
                        e_dt = None
                    if s_dt is not None and e_dt is not None:
                        total_seconds = max(0, int((e_dt - s_dt).total_seconds()))
                except Exception:
                    total_seconds = 0

            idle_seconds = min(total_seconds, idle_seconds_by_user.get(eid, 0))
            active_seconds = max(0, total_seconds - idle_seconds)

            status = self._derive_attendance_status(
                signin=None if sign_in == "—" else sign_in,
                signout=None if sign_out == "—" else sign_out,
                existing_status=d.get("status"),
            )

            s_color = {
                "On Time": C_GREEN,
                "Late": C_AMBER,
                "Early Departure": C_RED,
                "Overtime": C_BLUE,
            }.get(status, C_MUTED)

            row = ctk.CTkFrame(self._att_list_frame, fg_color=C_CARD, corner_radius=8, height=40)
            row.pack(fill="x", pady=2)
            row.pack_propagate(False)
            row_vals = [
                (_clip_text(d.get("full_name", "?"), 28), 210, C_TEXT),
                (eid, 90, C_TEXT),
                (row_date, 110, C_TEXT),
                (sign_in, 95, C_TEXT),
                (sign_out, 95, C_TEXT),
                (self._fmt_hms(active_seconds), 130, C_TEXT),
                (self._fmt_hms(idle_seconds), 120, C_TEXT),
                (status, 100, s_color),
            ]
            for txt, width, color in row_vals:
                ctk.CTkLabel(row, text=str(txt), text_color=color, font=ctk.CTkFont(size=11), width=width, anchor="w").pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # Settings Tab
    # ------------------------------------------------------------------

    def _default_geo_policy(self) -> dict:
        return {
            "office": {
                "city": "",
                "region": "",
                "country": "",
                "isp": "",
                "ip": "",
                "wifi_ssid_hash": "",
                "lat": None,
                "lon": None,
                "radius_km": 25.0,
                "location_hint": "",
            },
            "risk": {
                "strict_vpn_proxy": True,
                "outside_penalty": 20.0,
                "vpn_proxy_penalty": 25.0,
                "hosting_penalty": 20.0,
            },
        }

    def _load_geo_policy(self) -> dict:
        policy = self._default_geo_policy()
        if not self._db or not self._db.is_connected:
            return policy
        try:
            col = self._db.get_collection("system_settings")
            if col is None:
                return policy
            doc = col.find_one({"_id": "geo_policy"}) or {}
            office = doc.get("office") or {}
            risk = doc.get("risk") or {}
            policy["office"].update(office)
            policy["risk"].update(risk)
        except Exception:
            pass
        return policy

    def _save_geo_policy(self, policy: dict) -> None:
        if not self._db or not self._db.is_connected:
            raise RuntimeError("Database is offline")
        col = self._db.get_collection("system_settings")
        if col is None:
            raise RuntimeError("system_settings collection unavailable")
        payload = {
            **policy,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": "admin_panel",
        }
        col.update_one({"_id": "geo_policy"}, {"$set": payload}, upsert=True)

    def _refresh_geo_policy_ui(self, status_msg: str = "") -> None:
        policy = self._load_geo_policy()
        office = policy.get("office", {})
        risk = policy.get("risk", {})

        self._geo_city_var.set(str(office.get("city") or ""))
        self._geo_region_var.set(str(office.get("region") or ""))
        self._geo_country_var.set(str(office.get("country") or ""))
        self._geo_isp_var.set(str(office.get("isp") or ""))
        self._geo_ip_var.set(str(office.get("ip") or ""))
        self._geo_wifi_hash_var.set(str(office.get("wifi_ssid_hash") or office.get("ssid_hash") or ""))
        self._geo_hint_var.set(str(office.get("location_hint") or ""))
        self._geo_lat_var.set("" if office.get("lat") is None else str(office.get("lat")))
        self._geo_lon_var.set("" if office.get("lon") is None else str(office.get("lon")))
        self._geo_radius_var.set(str(office.get("radius_km", 25.0) or 25.0))

        self._geo_strict_var.set(bool(risk.get("strict_vpn_proxy", True)))
        self._geo_outside_penalty_var.set(str(risk.get("outside_penalty", 20.0) or 20.0))
        self._geo_vpn_penalty_var.set(str(risk.get("vpn_proxy_penalty", 25.0) or 25.0))
        self._geo_hosting_penalty_var.set(str(risk.get("hosting_penalty", 20.0) or 20.0))

        if status_msg and hasattr(self, "_geo_status_lbl"):
            self._geo_status_lbl.configure(text=status_msg, text_color=C_GREEN)

    def _get_current_wifi_ssid_hash(self) -> str:
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

    def _capture_office_geofence(self) -> None:
        if hasattr(self, "_geo_status_lbl"):
            self._geo_status_lbl.configure(text="Detecting current network location...", text_color=C_AMBER)

        def worker():
            try:
                from C3_activity_monitoring.src.geo_context import get_geo_context
                geo = get_geo_context()
                lat = geo.get("lat")
                lon = geo.get("lon")
                if lat is None or lon is None:
                    self.after(0, lambda: self._geo_status_lbl.configure(text="Could not detect lat/lon from current network.", text_color=C_RED))
                    return

                def apply_values():
                    self._geo_city_var.set(str(geo.get("city") or ""))
                    self._geo_region_var.set(str(geo.get("region") or ""))
                    self._geo_country_var.set(str(geo.get("country") or ""))
                    self._geo_isp_var.set(str(geo.get("isp") or ""))
                    self._geo_ip_var.set(str(geo.get("ip") or ""))
                    self._geo_wifi_hash_var.set(self._get_current_wifi_ssid_hash())
                    self._geo_hint_var.set(str(geo.get("location_hint") or ""))
                    self._geo_lat_var.set(str(lat))
                    self._geo_lon_var.set(str(lon))
                    self._geo_status_lbl.configure(text="Office geofence center captured from current network. Click Save.", text_color=C_GREEN)

                self.after(0, apply_values)
            except Exception as exc:
                self.after(0, lambda: self._geo_status_lbl.configure(text=f"Geo capture failed: {exc}", text_color=C_RED))

        threading.Thread(target=worker, daemon=True).start()

    def _save_geo_policy_from_form(self) -> None:
        try:
            lat = float(self._geo_lat_var.get().strip())
            lon = float(self._geo_lon_var.get().strip())
            radius_km = float(self._geo_radius_var.get().strip())
            outside_penalty = float(self._geo_outside_penalty_var.get().strip())
            vpn_penalty = float(self._geo_vpn_penalty_var.get().strip())
            hosting_penalty = float(self._geo_hosting_penalty_var.get().strip())
        except Exception:
            messagebox.showerror("Invalid Geo Policy", "Please enter valid numeric values for lat/lon/radius and penalties.")
            return

        policy = {
            "office": {
                "city": self._geo_city_var.get().strip(),
                "region": self._geo_region_var.get().strip(),
                "country": self._geo_country_var.get().strip(),
                "isp": self._geo_isp_var.get().strip(),
                "ip": self._geo_ip_var.get().strip(),
                "wifi_ssid_hash": self._geo_wifi_hash_var.get().strip(),
                "location_hint": self._geo_hint_var.get().strip(),
                "lat": lat,
                "lon": lon,
                "radius_km": max(0.5, radius_km),
            },
            "risk": {
                "strict_vpn_proxy": bool(self._geo_strict_var.get()),
                "outside_penalty": max(0.0, outside_penalty),
                "vpn_proxy_penalty": max(0.0, vpn_penalty),
                "hosting_penalty": max(0.0, hosting_penalty),
            },
        }

        try:
            self._save_geo_policy(policy)
            self._refresh_geo_policy_ui(status_msg="Geo-fence policy saved.")
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Could not save geo policy: {exc}")

    def _build_settings_tab(self) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(self._tab_frame, fg_color=C_BG, corner_radius=0)
        wrap = ctk.CTkScrollableFrame(frame, fg_color="transparent")
        wrap.pack(fill="both", expand=True, padx=24, pady=20)

        app_card = ctk.CTkFrame(wrap, fg_color=C_CARD, corner_radius=14)
        app_card.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkLabel(app_card, text="Application Settings", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=20, pady=(16, 8))
        ctk.CTkLabel(app_card, text=f"App Name:   {settings.APP_NAME}", text_color=C_MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=20)
        ctk.CTkLabel(app_card, text=f"Version:    {settings.VERSION}", text_color=C_MUTED, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=20)
        ctk.CTkLabel(app_card, text=f"DB Status:  {'Connected' if self._db and self._db.is_connected else 'Offline'}", text_color=C_GREEN if (self._db and self._db.is_connected) else C_RED, font=ctk.CTkFont(size=12)).pack(anchor="w", padx=20, pady=(0, 16))

        geo_card = ctk.CTkFrame(wrap, fg_color=C_CARD, corner_radius=14)
        geo_card.pack(fill="x", padx=16, pady=(0, 16))
        ctk.CTkLabel(geo_card, text="Office Geo-Fence Policy", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(anchor="w", padx=20, pady=(16, 6))
        ctk.CTkLabel(geo_card, text="Set office geo center from current admin network, then save radius and risk penalties.", text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(0, 10))

        self._geo_city_var = tk.StringVar()
        self._geo_region_var = tk.StringVar()
        self._geo_country_var = tk.StringVar()
        self._geo_isp_var = tk.StringVar()
        self._geo_ip_var = tk.StringVar()
        self._geo_wifi_hash_var = tk.StringVar()
        self._geo_hint_var = tk.StringVar()
        self._geo_lat_var = tk.StringVar()
        self._geo_lon_var = tk.StringVar()
        self._geo_radius_var = tk.StringVar(value="25")
        self._geo_strict_var = tk.BooleanVar(value=True)
        self._geo_outside_penalty_var = tk.StringVar(value="20")
        self._geo_vpn_penalty_var = tk.StringVar(value="25")
        self._geo_hosting_penalty_var = tk.StringVar(value="20")

        grid = ctk.CTkFrame(geo_card, fg_color="transparent")
        grid.pack(fill="x", padx=20, pady=(0, 8))
        grid.grid_columnconfigure((1, 3), weight=1)

        def _row(r: int, side: str, label: str, var: tk.Variable):
            is_left = side == "left"
            label_col = 0 if is_left else 2
            value_col = 1 if is_left else 3
            ctk.CTkLabel(grid, text=label, text_color=C_MUTED, font=ctk.CTkFont(size=11)).grid(row=r, column=label_col, sticky="w", padx=(0, 8), pady=6)
            ctk.CTkEntry(grid, textvariable=var, fg_color=C_SIDEBAR, border_color=C_BORDER).grid(row=r, column=value_col, sticky="ew", padx=(0, 14), pady=6)

        _row(0, "left", "City", self._geo_city_var)
        _row(0, "right", "Region", self._geo_region_var)
        _row(1, "left", "Country", self._geo_country_var)
        _row(1, "right", "ISP", self._geo_isp_var)
        _row(2, "left", "Public IP", self._geo_ip_var)
        _row(2, "right", "Office WiFi Hash", self._geo_wifi_hash_var)
        _row(3, "left", "Location Hint", self._geo_hint_var)
        _row(3, "right", "Latitude", self._geo_lat_var)
        _row(4, "left", "Longitude", self._geo_lon_var)
        _row(4, "right", "Radius (KM)", self._geo_radius_var)
        _row(5, "left", "Outside Penalty", self._geo_outside_penalty_var)
        _row(5, "right", "VPN/Proxy Penalty", self._geo_vpn_penalty_var)
        _row(6, "left", "Hosting Penalty", self._geo_hosting_penalty_var)

        strict_row = ctk.CTkFrame(geo_card, fg_color="transparent")
        strict_row.pack(fill="x", padx=20, pady=(2, 8))
        ctk.CTkSwitch(strict_row, text="Strict VPN/Proxy Check", variable=self._geo_strict_var, onvalue=True, offvalue=False).pack(side="left")

        btn_row = ctk.CTkFrame(geo_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(4, 6))
        ctk.CTkButton(btn_row, text="Capture Office From Current Network", fg_color=C_BLUE, hover_color="#2563eb", command=self._capture_office_geofence).pack(side="left")
        ctk.CTkButton(btn_row, text="Reload", width=90, fg_color=C_BORDER, hover_color=C_BLUE, command=lambda: self._refresh_geo_policy_ui(status_msg="Reloaded."),).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="Save Geo Policy", fg_color=C_TEAL, hover_color=C_TEAL_D, command=self._save_geo_policy_from_form).pack(side="right")

        self._geo_status_lbl = ctk.CTkLabel(geo_card, text="", text_color=C_MUTED, font=ctk.CTkFont(size=11))
        self._geo_status_lbl.pack(anchor="w", padx=20, pady=(0, 14))

        self._refresh_geo_policy_ui(status_msg="")
        return frame

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _start_polling(self) -> None:
        self._do_poll()

    def _do_poll(self) -> None:
        if self._closed or not self.winfo_exists():
            return
        if self._active_tab == "dashboard":
            self._refresh_dashboard()
        elif self._active_tab == "alerts":
            self._refresh_alerts()
        elif self._active_tab == "tasks":
            self._refresh_employee_dropdown()
            self._refresh_task_list()
        elif self._active_tab == "attendance":
            self._refresh_attendance()
        elif self._active_tab == "live_grid":
            self._refresh_live_grid()
        elif self._active_tab == "employees":
            self._refresh_employees()
        elif self._active_tab == "efficiency":
            self._refresh_efficiency_overview()
        self._poll_after_id = self.after(POLL_INTERVAL_MS, self._do_poll)

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

    def _init_db(self) -> MongoDBClient:
        db = MongoDBClient(uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME)
        db.connect()
        return db

    def _on_close(self) -> None:
        self._closed = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None
        try:
            if self._ws_server is not None:
                self._ws_server.close()
        except Exception:
            pass
        try:
            self._alerts_executor.shutdown(wait=False)
        except Exception:
            pass
        self.destroy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def launch_admin_panel(db: Optional[MongoDBClient] = None) -> None:
    """Launch the admin panel as a standalone window."""
    panel = AdminPanel(db=db)
    panel.mainloop()


if __name__ == "__main__":
    launch_admin_panel()
