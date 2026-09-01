from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import customtkinter as ctk
from tkinter import messagebox

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.database import MongoDBClient
from config.settings import settings
from C4_productivity_prediction.src.efficiency_service import EfficiencyPredictionService

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG = "#0b0e17"
C_CARD = "#151b2d"
C_BORDER = "#1e2a40"
C_TEXT = "#e2e8f0"
C_MUTED = "#64748b"
C_TEAL = "#14b8a6"
C_GREEN = "#22c55e"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_BLUE = "#3b82f6"


class EfficiencyWindow(ctk.CTk):
    """Standalone read-only window for per-employee efficiency prediction."""
    

    def __init__(self, db: MongoDBClient, refresh_ms: int = 60_000) -> None:
        super().__init__()
         # Database client instance
        self._db = db
        self._refresh_ms = refresh_ms
        self._service = EfficiencyPredictionService()
        self._period_var = ctk.StringVar(value="Current Month")
        self._refresh_after_id = None
        
        # Async rendering executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="EfficiencyRefresh")

        self.title(f"{settings.APP_NAME} - Employee Efficiency Predictions")
        self.geometry("1280x760")
        self.minsize(1080, 680)
        self.configure(fg_color=C_BG)

        self._last_updated_var = ctk.StringVar(value="Last updated: -")
        self._status_var = ctk.StringVar(value="Loading model and reading data...")

        self._build()
        # Schedule initial refresh after 200 ms (startup delay)
        self.after(200, self._refresh)
        
    
    #Build and layout the Employee Efficiency dashboard UI
    def _build(self) -> None:
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=18, pady=(16, 8))

        ctk.CTkLabel(
            header,
            text="Individual Employee Efficiency",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Read-only C4 prediction view from existing employee/task/activity data",
            font=ctk.CTkFont(size=12),
            text_color=C_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        topbar = ctk.CTkFrame(self, fg_color="transparent")
        topbar.pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(topbar, textvariable=self._last_updated_var, text_color=C_MUTED).pack(side="left")

        period_picker = ctk.CTkOptionMenu(
            topbar,
            values=["Current Month", "Last 3 Months", "Last 6 Months", "All Time"],
            variable=self._period_var,
            fg_color=C_BORDER,
            button_color=C_BLUE,
            button_hover_color="#2563eb",
            width=160,
        )
        period_picker.pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            topbar,
            text="Refresh Now",
            fg_color=C_TEAL,
            hover_color="#0d9488",
            command=self._refresh,
            width=130,
            height=34,
        ).pack(side="right")

        self._summary_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._summary_frame.pack(fill="x", padx=18, pady=(0, 10))

        self._cards = {
            "employees": self._make_card(self._summary_frame, "Employees", "0", C_BLUE),
            "high": self._make_card(self._summary_frame, "Predicted High", "0", C_GREEN),
            "medium": self._make_card(self._summary_frame, "Predicted Medium", "0", C_AMBER),
            "low": self._make_card(self._summary_frame, "Predicted Low", "0", C_RED),
            "avg_conf": self._make_card(self._summary_frame, "Avg Confidence", "0%", C_TEAL),
        }

        for i, key in enumerate(["employees", "high", "medium", "low", "avg_conf"]):
            self._cards[key].grid(row=0, column=i, sticky="nsew", padx=6)
            self._summary_frame.grid_columnconfigure(i, weight=1)

        body = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=12)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 16))

        ctk.CTkLabel(
            body,
            text="Predictions by Employee",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=14, pady=(12, 6))

        header_row = ctk.CTkFrame(body, fg_color="#10172b", corner_radius=8)
        header_row.pack(fill="x", padx=12, pady=(0, 8))
        for title, width in [
            ("Employee", 230),
            ("Prediction", 130),
            ("Efficiency Score", 130),
            ("Input Productivity", 150),
            ("Workload", 110),
            ("Assigned", 90),
            ("Pending", 80),
            ("On Time", 90),
            ("Late", 80),
            ("Details", 90),
        ]:
            ctk.CTkLabel(
                header_row,
                text=title,
                width=width,
                text_color=C_MUTED,
                anchor="w",
                font=ctk.CTkFont(size=11, weight="bold"),
            ).pack(side="left", padx=4, pady=8)

        self._table = ctk.CTkScrollableFrame(body, fg_color="transparent")
        self._table.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkLabel(footer, textvariable=self._status_var, text_color=C_MUTED).pack(anchor="w")

    def _make_card(self, parent, title: str, value: str, accent: str):
        card = ctk.CTkFrame(parent, fg_color=C_CARD, corner_radius=12, border_width=1, border_color=C_BORDER)
        ctk.CTkLabel(card, text=title, text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(10, 2))
        val = ctk.CTkLabel(card, text=value, text_color=accent, font=ctk.CTkFont(size=20, weight="bold"))
        val.pack(anchor="w", padx=12, pady=(0, 10))
        card._value_label = val
        return card

    def _set_card(self, key: str, value: str) -> None:
        card = self._cards.get(key)
        if card is not None:
            card._value_label.configure(text=value)

    def _refresh(self) -> None:
        # Submit fetch work to background thread
        self._executor.submit(self._fetch_and_render)
        
        # Schedule next refresh
        if self._refresh_after_id is not None:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self._refresh_after_id = self.after(self._refresh_ms, self._refresh)
    
    def _fetch_and_render(self) -> None:
        """Fetch predictions in background thread, then render on main thread."""
        try:
            period_start, period_end = self._period_range()
            rows = self._service.predict_all(self._db, period_start=period_start, period_end=period_end)
            
            # Render on main thread
            self.after(0, lambda: self._render_on_main_thread(rows))
        except Exception as exc:
            logger.exception("Efficiency window refresh failed")
            self.after(0, lambda: self._status_var.set(f"Refresh failed: {exc}"))
    
    def _render_on_main_thread(self, rows) -> None:
        """Render results (must be called on main thread)."""
        self._render_rows(rows)
        self._render_summary(rows)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_updated_var.set(f"Last updated: {now}")
        self._status_var.set(f"Read-only prediction completed for {len(rows)} employees.")

    #rendering summary in static form
    def _render_summary(self, rows) -> None:
        total = len(rows)
        high = sum(1 for r in rows if r.predicted_label.lower() == "high")
        medium = sum(1 for r in rows if r.predicted_label.lower() == "medium")
        low = sum(1 for r in rows if r.predicted_label.lower() == "low")
        avg_conf = (sum(r.confidence for r in rows) / total) if total else 0.0

        self._set_card("employees", str(total))
        self._set_card("high", str(high))
        self._set_card("medium", str(medium))
        self._set_card("low", str(low))
        self._set_card("avg_conf", f"{avg_conf * 100:.1f}%")

    def _period_range(self):
        choice = self._period_var.get().strip().lower()
        now = datetime.now(timezone.utc)

        if choice == "all time":
            return None, None

        if choice == "last 3 months":
            start = now - timedelta(days=90)
            return start, now

        if choice == "last 6 months":
            start = now - timedelta(days=180)
            return start, now

        month_start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        return month_start, now

    def _render_rows(self, rows) -> None:
        for w in self._table.winfo_children():
            w.destroy()

        if not rows:
            ctk.CTkLabel(
                self._table,
                text="No employees available for prediction.",
                text_color=C_MUTED,
            ).pack(pady=20)
            return

        for r in rows:
            row = ctk.CTkFrame(self._table, fg_color="#10172b", corner_radius=8)
            row.pack(fill="x", pady=4)

            pred_color = {
                "high": C_GREEN,
                "medium": C_AMBER,
                "low": C_RED,
            }.get(r.predicted_label.lower(), C_TEXT)

            values = [
                (f"{r.full_name} ({r.employee_id})", 230, C_TEXT),
                (r.predicted_label, 130, pred_color),
                (f"{r.efficiency_score:.1f}", 130, C_TEXT),
                (f"{r.productivity_score_input:.1f}", 150, C_TEXT),
                (f"{r.workload_score:.1f}", 110, C_TEXT),
                (str(r.total_tasks_assigned), 90, C_TEXT),
                (str(r.total_tasks_pending), 80, C_TEXT),
                (str(r.total_tasks_completed_on_time), 90, C_TEXT),
                (str(r.total_tasks_completed_late), 80, C_TEXT),
            ]

            for text, width, color in values:
                ctk.CTkLabel(
                    row,
                    text=text,
                    width=width,
                    anchor="w",
                    text_color=color,
                    font=ctk.CTkFont(size=12),
                ).pack(side="left", padx=4, pady=8)

            ctk.CTkButton(
                row,
                text="Details",
                width=86,
                height=28,
                fg_color=C_BLUE,
                hover_color="#2563eb",
                command=lambda emp_id=r.employee_id: self._open_employee_details(emp_id),
            ).pack(side="left", padx=4, pady=6)

    def _open_employee_details(self, employee_id: str) -> None:
        self._status_var.set(f"Loading productivity report for {employee_id}...")
        period_start, period_end = self._period_range()
        self._executor.submit(self._fetch_employee_report, employee_id, period_start, period_end)

    # Fetch detailed report for one employee and show in new window (background)
    def _fetch_employee_report(self, employee_id: str, period_start, period_end) -> None:
        try:
            report = self._service.get_employee_productivity_report(
                self._db,
                employee_id=employee_id,
                period_start=period_start,
                period_end=period_end,
            )
            self.after(0, lambda: self._show_employee_details(report, employee_id, period_start, period_end))
        except Exception as exc:
            logger.exception("Failed to build employee productivity report")
            self.after(0, lambda: messagebox.showerror("Productivity Report", f"Failed to load report: {exc}"))

    def _show_employee_details(self, report, employee_id: str, period_start=None, period_end=None) -> None:
        if report is None:
            self._status_var.set("No report available for selected employee.")
            messagebox.showinfo("Productivity Report", f"No report data available for {employee_id} in this period.")
            return
        
        # color based on predicted label
        pred_color = {
            "high": C_GREEN,
            "medium": C_AMBER,
            "low": C_RED,
        }.get(str(report.predicted_label).lower(), C_TEXT)

        self._status_var.set(f"Report ready for {report.employee_id}.")

        win = ctk.CTkToplevel(self)
        win.title(f"Productivity Detail Report - {report.full_name}")
        win.geometry("860x620")
        win.minsize(760, 520)
        win.configure(fg_color=C_BG)
        win.transient(self)
        win.grab_set()

        header = ctk.CTkFrame(win, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(14, 8))

        ctk.CTkLabel(
            header,
            text=f"{report.full_name} ({report.employee_id})",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text=report.summary,
            font=ctk.CTkFont(size=12),
            text_color=C_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        cards = ctk.CTkFrame(win, fg_color="transparent")
        cards.pack(fill="x", padx=16, pady=(0, 8))

        metric_cards = [
            ("Prediction", str(report.predicted_label), pred_color),
            ("Confidence", f"{report.confidence * 100:.1f}%", C_TEAL),
            ("Efficiency Score", f"{report.efficiency_score:.1f}", C_BLUE),
            ("Workload", f"{report.workload_score:.1f}", C_AMBER),
        ]

        for col, (title, value, color) in enumerate(metric_cards):
            card = ctk.CTkFrame(cards, fg_color=C_CARD, corner_radius=10, border_width=1, border_color=C_BORDER)
            card.grid(row=0, column=col, sticky="nsew", padx=4)
            cards.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(card, text=title, text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10, pady=(8, 1))
            ctk.CTkLabel(card, text=value, text_color=color, font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=10, pady=(0, 9))

        content = ctk.CTkScrollableFrame(win, fg_color=C_CARD, corner_radius=12)
        content.pack(fill="both", expand=True, padx=16, pady=(0, 14))

        ctk.CTkLabel(
            content,
            text="Productivity Detail Report",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(12, 6))

        stats = [
            f"Assigned Tasks: {report.total_tasks_assigned}",
            f"Pending Tasks: {report.total_tasks_pending}",
            f"Completed On Time: {report.total_tasks_completed_on_time}",
            f"Completed Late: {report.total_tasks_completed_late}",
            f"Completion Ratio: {report.completion_ratio * 100:.1f}%",
            f"On-Time Ratio: {report.on_time_ratio * 100:.1f}%",
            f"Backlog Ratio: {report.backlog_ratio * 100:.1f}%",
        ]

        for line in stats:
            ctk.CTkLabel(
                content,
                text=line,
                text_color=C_TEXT,
                font=ctk.CTkFont(size=12),
            ).pack(anchor="w", padx=12, pady=2)

        ctk.CTkLabel(
            content,
            text="Model Insights",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(12, 6))

        for insight in report.insights:
            ctk.CTkLabel(
                content,
                text=f"- {insight}",
                text_color=C_MUTED,
                font=ctk.CTkFont(size=12),
                justify="left",
                wraplength=780,
            ).pack(anchor="w", padx=12, pady=2)

        lime_lines = self._service.get_employee_lime_explanation(
            self._db,
            employee_id=employee_id,
            period_start=period_start,
            period_end=period_end,
        )
        weekly_forecast = self._service.get_employee_weekly_forecast(
            self._db,
            employee_id=employee_id,
            period_end=period_end,
        )

        efficiency_state = "strong" if report.efficiency_score >= 70 else "needs attention" if report.efficiency_score < 50 else "mixed"
        workload_state = "healthy" if report.workload_score >= 70 else "strained" if report.workload_score < 50 else "moderate"
        insight_summary = " ".join(report.insights[:3]) if report.insights else "The model did not find any strong warning or support signals beyond the score itself."

        ctk.CTkLabel(
            content,
            text="Business interpretation",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(12, 6))

        ctk.CTkLabel(
            content,
            text=(
                f"This employee's efficiency score is {report.efficiency_score:.1f}/100, so the current picture is {efficiency_state}. "
                f"The workload score is {report.workload_score:.1f}/100, which is {workload_state}. {insight_summary} "
                "Taken together, this helps a manager or HR reviewer decide whether the employee is performing well, holding steady, or slipping because of unfinished work, late completion, or weaker activity patterns."
            ),
            text_color=C_MUTED,
            font=ctk.CTkFont(size=12),
            wraplength=760,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 4))

        # Get weekly forecast with real-time ML predictions
        realtime_score = self._service.get_employee_realtime_forecast(self._db, employee_id=employee_id)
        if realtime_score is not None:
            weekly_forecast._realtime_score = realtime_score
        weekly_forecast.current_score = float(report.efficiency_score)

        self._render_weekly_forecast_section(content, weekly_forecast)

        ctk.CTkButton(
            win,
            text="Close",
            fg_color=C_BORDER,
            hover_color="#27364f",
            width=110,
            command=win.destroy,
        ).pack(anchor="e", padx=16, pady=(0, 14))

    def _render_weekly_forecast_section(self, parent, forecast) -> None:
        section = ctk.CTkFrame(parent, fg_color="#10172b", corner_radius=10, border_width=1, border_color=C_BORDER)
        section.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            section,
            text="Weekly Efficiency Forecast",
            text_color=C_TEXT,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w", padx=12, pady=(10, 2))

        # Check for real-time ML forecast availability
        has_realtime = hasattr(forecast, '_realtime_score') and forecast._realtime_score is not None
        has_history = getattr(forecast, "available", False) and forecast.history_scores

        # Build message - different for history vs realtime only
        message_text = forecast.message
        if not has_history and has_realtime:
            message_text = "Real-time ML forecast (insufficient historical data for trend analysis)"

        ctk.CTkLabel(
            section,
            text=message_text,
            text_color=C_MUTED,
            font=ctk.CTkFont(size=11),
            wraplength=780,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 8))

        body = ctk.CTkFrame(section, fg_color=C_CARD, corner_radius=8)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Show message if neither history nor realtime available
        if not has_history and not has_realtime:
            ctk.CTkLabel(
                body,
                text=forecast.message,
                text_color=C_MUTED,
                font=ctk.CTkFont(size=12),
                wraplength=740,
                justify="left",
            ).pack(anchor="w", padx=12, pady=18)
            return

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            from matplotlib.ticker import FuncFormatter
        except Exception as exc:
            ctk.CTkLabel(
                body,
                text=f"Forecast chart unavailable: {exc}",
                text_color=C_MUTED,
                font=ctk.CTkFont(size=12),
                wraplength=740,
                justify="left",
            ).pack(anchor="w", padx=12, pady=18)
            return

        fig = Figure(figsize=(6.2, 2.9), dpi=100, facecolor="#10172b")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#10172b")

        today = datetime.now(timezone.utc).date()
        upcoming_dates = [today + timedelta(days=i) for i in range(7)]

        # Handle two cases: (1) historical data available, or (2) only realtime forecast
        if has_history:
            # Plot with full history + both forecasts
            history_x = list(range(len(forecast.history_scores)))
            forecast_x = len(history_x)
            labels = list(forecast.history_weeks)
            labels.append(f"Next\n{forecast.forecast_week}")

            ax.plot(history_x, forecast.history_scores, color=C_TEAL, linewidth=2.4, marker="o", markersize=5)
            
            # Plot linear regression forecast (existing)
            ax.plot(
                [history_x[-1], forecast_x],
                [forecast.history_scores[-1], float(forecast.forecast_score or forecast.history_scores[-1])],
                color=C_AMBER,
                linewidth=2.2,
                linestyle="--",
                marker="o",
                markersize=5,
            )
            ax.scatter([forecast_x], [float(forecast.forecast_score or forecast.history_scores[-1])], color=C_AMBER, s=55, zorder=5)
            ax.annotate(
                f"{float(forecast.forecast_score or forecast.history_scores[-1]):.1f}",
                (forecast_x, float(forecast.forecast_score or forecast.history_scores[-1])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                color=C_AMBER,
                fontsize=9,
                fontweight="bold",
            )
            
            # Add real-time ML forecast if available
            if has_realtime:
                rt_score = float(forecast._realtime_score)
                ax.plot(
                    [history_x[-1], forecast_x],
                    [forecast.history_scores[-1], rt_score],
                    color="#10b981",
                    linewidth=2.2,
                    linestyle=":",
                    marker="s",
                    markersize=5,
                    alpha=0.8,
                )
                ax.scatter([forecast_x], [rt_score], color="#10b981", s=55, zorder=5, alpha=0.8)
                ax.annotate(
                    f"{rt_score:.1f}",
                    (forecast_x, rt_score),
                    textcoords="offset points",
                    xytext=(0, -12),
                    ha="center",
                    color="#10b981",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_ylim(0, 100)
            ax.set_xlim(-0.2, forecast_x + 0.4)
            ax.set_xticks(list(range(len(labels))))
            ax.set_xticklabels(labels, color=C_TEXT, fontsize=8)
            ax.axvline(history_x[-1], color="#23324d", linestyle=":", linewidth=1)
        else:
            # Plot realtime-only forecast across the next 7 calendar days.
            # Use the employee's actual current score as the starting point so the chart
            # reflects the current state instead of a misleading neutral 50 baseline.
            rt_score = float(forecast._realtime_score)
            current_score = getattr(forecast, "current_score", None)
            start_score = float(current_score) if current_score is not None else rt_score
            x_pos = list(range(7))
            y_values = [start_score] + [rt_score] * 6
            day_labels = [d.strftime("%b %d") for d in upcoming_dates]
            
            ax.plot(x_pos, y_values, color="#10b981", linewidth=2.8, linestyle=":", marker="o", markersize=7, alpha=0.9)
            ax.scatter([x_pos[-1]], [rt_score], color="#10b981", s=100, zorder=5, alpha=0.9)
            ax.annotate(
                f"{rt_score:.1f}",
                (x_pos[-1], rt_score),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="#10b981",
                fontsize=11,
                fontweight="bold",
            )
            
            ax.set_ylim(0, 100)
            ax.set_xlim(-0.3, 6.3)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(day_labels, color=C_TEXT, fontsize=8)

        ax.tick_params(axis="y", colors=C_TEXT, labelsize=8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}%"))
        ax.set_ylabel("Efficiency score", color=C_TEXT, fontsize=9)
        ax.set_xlabel("Date", color=C_TEXT, fontsize=9)
        ax.grid(axis="y", color="#23324d", linestyle="-", linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#23324d")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=body)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body._forecast_canvas = canvas


def launch_efficiency_window(db: MongoDBClient | None = None) -> None:
    own_db = db is None
    db_client = db
    if db_client is None:
        db_client = MongoDBClient(uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME)
        db_client.connect()

    app = EfficiencyWindow(db=db_client)
    app.mainloop()

    if own_db and db_client is not None:
        try:
            db_client.close()
        except Exception:
            pass
