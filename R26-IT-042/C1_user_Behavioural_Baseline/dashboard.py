import csv
from collections import Counter, defaultdict
from datetime import datetime
import sys
from pathlib import Path

import customtkinter as ctk
from tkinter import messagebox

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "employee_behavior_10k.csv"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

C_BG = "#0b0e17"
C_CARD = "#151b2d"
C_BORDER = "#1e2a40"
C_TEAL = "#14b8a6"
C_AMBER = "#f59e0b"
C_GREEN = "#22c55e"
C_RED = "#ef4444"
C_TEXT = "#e2e8f0"
C_MUTED = "#94a3b8"
C_BLUE = "#3b82f6"

LIST_COLUMNS = [
    ("Employee", 160, 5),
    ("ID", 72, 2),
    ("Anomaly", 82, 2),
    ("Top Location", 96, 3),
    ("Status", 72, 2),
    ("", 86, 2),
]


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    

# functions for formatting durations and login times
def format_duration(minutes):
    minutes = int(round(minutes))
    hours = minutes // 60
    mins = minutes % 60
    if hours and mins:
        return f"{hours}h {mins}m"
    if hours:
        return f"{hours}h"
    return f"{mins}m"


def format_login_time(value):
    value = safe_float(value)
    hours = int(value)
    minutes = int(round((value - hours) * 60))
    return f"{hours:02d}:{minutes:02d}"


def parse_iso_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None


def load_records_from_mongo():
    project_root = BASE_DIR.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from common.database import MongoDBClient
        from config.settings import settings
    except ImportError:
        return []

    client = MongoDBClient(uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME)
    if not client.connect():
        return []

    try:
        employees = {}
        employee_collection = client.get_collection("employees")
        if employee_collection is not None:
            for doc in employee_collection.find({}, {"employee_id": 1, "full_name": 1, "shift_start": 1, "shift_end": 1}):
                employee_id = doc.get("employee_id")
                if not employee_id:
                    continue
                employee_id = str(employee_id).strip().upper()
                employees[employee_id] = {
                    "name": doc.get("full_name") or employee_id,
                    "shift_start": doc.get("shift_start"),
                    "shift_end": doc.get("shift_end"),
                }

        task_counts = Counter()
        task_collection = client.get_collection("tasks")
        if task_collection is not None:
            for doc in task_collection.find({}, {"employee_id": 1}):
                employee_id = doc.get("employee_id")
                if not employee_id:
                    continue
                task_counts[str(employee_id).strip().upper()] += 1

        attendance_status = {}
        attendance_collection = client.get_collection("attendance_logs")
        if attendance_collection is not None:
            for doc in attendance_collection.find({}, {"employee_id": 1, "date": 1, "signin": 1, "signout": 1, "status": 1}).sort([
                ("employee_id", 1),
                ("date", -1),
            ]):
                employee_id = doc.get("employee_id")
                if not employee_id:
                    continue
                employee_id = str(employee_id).strip().upper()
                if employee_id in attendance_status:
                    continue
                attendance_status[employee_id] = doc.get("status") or "Unknown"

        auth_failures = Counter()
        auth_collection = client.get_collection("auth_events")
        if auth_collection is not None:
            for doc in auth_collection.find({"success": False}, {"employee_id": 1}):
                auth_failures[doc.get("employee_id")] += 1

        records = []
        activity_collection = client.get_collection("activity_logs")
        if activity_collection is None:
            return []

        for doc in activity_collection.find({}, {
            "user_id": 1,
            "timestamp": 1,
            "productivity_score": 1,
            "composite_risk_score": 1,
            "alert_triggered": 1,
            "in_break": 1,
            "location_mode": 1,
            "session_id": 1,
            "label": 1,
            "break_type": 1,
        }).sort("timestamp", 1):
            timestamp = parse_iso_timestamp(doc.get("timestamp"))
            employee_id = doc.get("user_id") or doc.get("employee_id")
            if not employee_id:
                continue
            employee_id = str(employee_id).strip().upper()
            if employees and employee_id not in employees:
                continue

            employee_name = employees.get(employee_id, {}).get("name", employee_id)
            label = doc.get("label") or "normal"
            location_mode = doc.get("location_mode") or "unknown"
            alert_triggered = bool(doc.get("alert_triggered"))
            in_break = bool(doc.get("in_break"))
            anomaly_type = doc.get("break_type") or (label if label != "normal" else "normal")
            login_hour = 0.0
            if timestamp is not None:
                login_hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0

            records.append({
                "date": timestamp.date().isoformat() if timestamp else "",
                "employee_id": employee_id,
                "employee_name": employee_name,
                "login_time": timestamp.strftime("%H:%M:%S") if timestamp else "",
                "logout_time": "",
                "login_hour_numeric": login_hour,
                "session_duration_min": 0,
                "idle_time_min": 0,
                "auth_failures": auth_failures.get(employee_id, 0),
                "module_accessed": location_mode,
                "activity_frequency": 1,
                "is_anomaly": 1 if alert_triggered or label != "normal" else 0,
                "anomaly_type": anomaly_type,
                "productivity_score": safe_float(doc.get("productivity_score", 0)),
                "composite_risk_score": safe_float(doc.get("composite_risk_score", 0)),
                "alert_count": int(alert_triggered),
                "break_count": int(in_break),
                "location_mode": location_mode,
                "task_count": task_counts.get(employee_id, 0),
                "attendance_status": attendance_status.get(employee_id, "Unknown"),
                "last_seen": timestamp.isoformat() if timestamp else "",
            })
        return records
    finally:
        client.close()


# Load records from MongoDB if available, otherwise fall back to CSV file
def load_records():
    records = load_records_from_mongo()
    if records:
        return records

    if not DATA_FILE.exists():
        return []

    with DATA_FILE.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            records.append({
                "date": row.get("date", ""),
                "employee_id": row.get("employee_id", "").strip(),
                "employee_name": row.get("employee_name", "").strip(),
                "login_time": row.get("login_time", ""),
                "logout_time": row.get("logout_time", ""),
                "login_hour_numeric": safe_float(row.get("login_hour_numeric", 0.0)),
                "session_duration_min": safe_int(row.get("session_duration_min", 0)),
                "idle_time_min": safe_int(row.get("idle_time_min", 0)),
                "auth_failures": safe_int(row.get("auth_failures", 0)),
                "module_accessed": row.get("module_accessed", "").strip(),
                "activity_frequency": safe_int(row.get("activity_frequency", 0)),
                "is_anomaly": safe_int(row.get("is_anomaly", 0)),
                "anomaly_type": row.get("anomaly_type", "").strip(),
            })
    return records


#Build behavioral baselines for each employee based on their activity records
def build_employee_baselines(records):
    by_employee = defaultdict(list)
    for record in records:
        if not record["employee_id"]:
            continue
        by_employee[record["employee_id"]].append(record)

    baselines = {}
    for employee_id, rows in by_employee.items():
        employee_name = rows[0]["employee_name"] or employee_id
        total = len(rows)
        anomaly_count = sum(r.get("is_anomaly", 0) for r in rows)
        average_login_hour = sum(r.get("login_hour_numeric", 0.0) for r in rows) / total
        average_productivity = sum(r.get("productivity_score", 0.0) for r in rows) / total
        average_risk = sum(r.get("composite_risk_score", 0.0) for r in rows) / total
        alert_count = sum(r.get("alert_count", 0) for r in rows)
        break_count = sum(r.get("break_count", 0) for r in rows)
        task_count = rows[0].get("task_count", 0)
        attendance_status = rows[0].get("attendance_status", "Unknown")
        common_location = Counter(r.get("location_mode", "unknown") for r in rows if r.get("location_mode"))
        top_location = common_location.most_common(1)[0][0] if common_location else "Unknown"
        anomaly_rate = round(anomaly_count * 100.0 / total, 1)
        recent_records = sorted(rows, key=lambda r: r.get("last_seen", ""), reverse=True)[:5]
        status_label = "Stable" if anomaly_rate < 10 else "Watch" if anomaly_rate < 25 else "Risk"

        baselines[employee_id] = {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "record_count": total,
            "alert_count": alert_count,
            "break_count": break_count,
            "task_count": task_count,
            "attendance_status": attendance_status,
            "anomaly_count": anomaly_count,
            "anomaly_rate": anomaly_rate,
            "average_login_time": format_login_time(average_login_hour),
            "average_productivity": round(average_productivity, 1),
            "average_risk": round(average_risk, 1),
            "top_module": top_location,
            "status_label": status_label,
            "recent_records": recent_records,
            "explanation": (
                f"{employee_name} ({employee_id}) has {total} recorded activity events, "
                f"with average productivity {round(average_productivity, 1)}% and average risk {round(average_risk, 1)}%. "
                f"Alerts have occurred {alert_count} times, and the most common location mode is {top_location}."
            ),
        }
    return baselines


class BehavioralBaselineApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("C1 Behavioral Baselines")
        self.geometry("1160x760")
        self.minsize(1040, 680)
        self.configure(fg_color=C_BG)

        self.records = load_records()
        self.baselines = build_employee_baselines(self.records)
        self.selected_employee_id = None
        self._row_frames = {}

        self._build_layout()
        self._render_summary()
        self._render_employee_list()
        if self.baselines:
            first_id = next(iter(sorted(self.baselines.keys(), key=lambda eid: self.baselines[eid]["employee_name"])))
            self.show_employee_detail(first_id)

    def _build_layout(self):
        header = ctk.CTkFrame(self, fg_color=C_CARD, corner_radius=0, height=68)
        header.pack(fill="x", padx=16, pady=(16, 0))
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="Behavioral Baseline Monitor",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=C_TEXT,
        ).pack(side="left", padx=16)

        ctk.CTkButton(
            header,
            text="Refresh Data",
            width=120,
            height=34,
            fg_color=C_TEAL,
            hover_color="#0ea5e9",
            command=self.refresh_data,
        ).pack(side="right", padx=16)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=16, pady=(12, 16))

        left_panel = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=20)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=0)

        right_panel = ctk.CTkFrame(body, fg_color=C_CARD, corner_radius=20)
        right_panel.pack(side="right", fill="both", expand=True, padx=(8, 0), pady=0)

        self._summary_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        self._summary_frame.pack(fill="x", padx=16, pady=(16, 12))

        separator = ctk.CTkFrame(left_panel, height=1, fg_color=C_BORDER)
        separator.pack(fill="x", padx=16, pady=(0, 8))

        ctk.CTkLabel(
            left_panel,
            text="Employee Baseline Summary",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=C_TEXT,
        ).pack(anchor="w", padx=16, pady=(0, 8))

        self._list_scroll = ctk.CTkScrollableFrame(left_panel, fg_color="transparent", corner_radius=0)
        self._list_scroll.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        self._detail_frame = ctk.CTkScrollableFrame(right_panel, fg_color="transparent", corner_radius=0)
        self._detail_frame.pack(fill="both", expand=True, padx=16, pady=16)

    def _render_summary(self):
        for child in self._summary_frame.winfo_children():
            child.destroy()

        total = len(self.baselines)
        avg_rate = 0.0
        if total:
            avg_rate = sum(b["anomaly_rate"] for b in self.baselines.values()) / total
        risks = sorted(self.baselines.values(), key=lambda b: b["anomaly_rate"], reverse=True)[:3]

        stats = [
            ("Employees", str(total), C_TEAL),
            ("Avg Anomaly", f"{avg_rate:.1f}%", C_AMBER),
            ("Top Risk", risks[0]["employee_name"] if risks else "—", C_RED),
        ]

        row = ctk.CTkFrame(self._summary_frame, fg_color=C_BG, corner_radius=14)
        row.pack(fill="x")
        for title, value, accent in stats:
            card = ctk.CTkFrame(row, fg_color=C_CARD, corner_radius=14)
            card.pack(side="left", fill="both", expand=True, padx=4, pady=4)
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=10), text_color=C_MUTED).pack(anchor="w", padx=12, pady=(12, 4))
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=20, weight="bold"), text_color=accent).pack(anchor="w", padx=12, pady=(0, 12))

    def _render_employee_list(self):
        for child in self._list_scroll.winfo_children():
            child.destroy()
        self._row_frames.clear()

        if not self.baselines:
            ctk.CTkLabel(self._list_scroll, text="No employee records found.", text_color=C_MUTED).pack(pady=20)
            return

        header = ctk.CTkFrame(self._list_scroll, fg_color=C_BORDER, corner_radius=12)
        header.pack(fill="x", pady=(0, 4), padx=(0, 8))
        for col_idx, (text, minsize, weight) in enumerate(LIST_COLUMNS):
            header.grid_columnconfigure(col_idx, weight=weight, minsize=minsize)
            ctk.CTkLabel(
                header,
                text=text,
                anchor="w",
                text_color=C_MUTED,
                font=ctk.CTkFont(size=11, weight="bold"),
            ).grid(row=0, column=col_idx, sticky="w", padx=(8, 4), pady=10)

        for employee_id, baseline in sorted(self.baselines.items(), key=lambda x: x[1]["employee_name"]):
            row = ctk.CTkFrame(self._list_scroll, fg_color=C_CARD, corner_radius=12)
            row.pack(fill="x", pady=4, ipady=4, padx=(0, 8))
            row.bind("<Button-1>", lambda e, eid=employee_id: self.show_employee_detail(eid))

            for col_idx, (_, minsize, weight) in enumerate(LIST_COLUMNS):
                row.grid_columnconfigure(col_idx, weight=weight, minsize=minsize)

            ctk.CTkLabel(row, text=baseline["employee_name"], anchor="w", text_color=C_TEXT, font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=0, sticky="w", padx=(8, 4), pady=8)
            ctk.CTkLabel(row, text=employee_id, anchor="w", text_color=C_MUTED, font=ctk.CTkFont(size=11)).grid(row=0, column=1, sticky="w", padx=(8, 4), pady=8)
            ctk.CTkLabel(row, text=f"{baseline['anomaly_rate']:.1f}%", anchor="w", text_color=C_AMBER if baseline["anomaly_rate"] >= 15 else C_GREEN, font=ctk.CTkFont(size=11)).grid(row=0, column=2, sticky="w", padx=(8, 4), pady=8)
            ctk.CTkLabel(row, text=baseline["top_module"], anchor="w", text_color=C_TEXT, font=ctk.CTkFont(size=11)).grid(row=0, column=3, sticky="w", padx=(8, 4), pady=8)
            status_color = C_GREEN if baseline["status_label"] == "Stable" else C_AMBER if baseline["status_label"] == "Watch" else C_RED
            ctk.CTkLabel(row, text=baseline["status_label"], anchor="w", text_color=status_color, font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=4, sticky="w", padx=(8, 4), pady=8)
            ctk.CTkButton(
                row,
                text="Details",
                width=80,
                height=28,
                fg_color=C_BLUE,
                hover_color="#2563eb",
                command=lambda eid=employee_id: self.show_employee_detail(eid),
            ).grid(row=0, column=5, sticky="e", padx=(8, 6), pady=6)

            self._row_frames[employee_id] = row

        self._update_row_selection()


    # When an employee is selected, show detailed baseline information in the right panel
    def show_employee_detail(self, employee_id: str):
        if employee_id not in self.baselines:
            return
        self.selected_employee_id = employee_id
        self._update_row_selection()
        self._render_detail_panel(self.baselines[employee_id])

    def _update_row_selection(self):
        for eid, frame in self._row_frames.items():
            bg = "#1f2937" if eid == self.selected_employee_id else C_CARD
            try:
                frame.configure(fg_color=bg)
            except Exception:
                pass

    def _render_detail_panel(self, baseline):
        for child in self._detail_frame.winfo_children():
            child.destroy()

        ctk.CTkLabel(self._detail_frame, text=f"{baseline['employee_name']} — {baseline['employee_id']}", font=ctk.CTkFont(size=18, weight="bold"), text_color=C_TEXT).pack(anchor="w", pady=(0, 8))

        status_color = C_GREEN if baseline["status_label"] == "Stable" else C_AMBER if baseline["status_label"] == "Watch" else C_RED
        ctk.CTkLabel(self._detail_frame, text=f"Status: {baseline['status_label']}", font=ctk.CTkFont(size=13, weight="bold"), text_color=status_color, fg_color=C_BORDER, corner_radius=12, width=180, height=34).pack(anchor="w", pady=(0, 10))

        ctk.CTkLabel(self._detail_frame, text=baseline["explanation"], text_color=C_MUTED, wraplength=460, justify="left", font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(0, 14))

        metrics = [
            ("Total events", str(baseline["record_count"])),
            ("Alerts", str(baseline["alert_count"])),
            ("Risk score", f"{baseline['average_risk']:.1f}%"),
            ("Avg productivity", f"{baseline['average_productivity']:.1f}%"),
            ("Avg login", baseline["average_login_time"]),
            ("Breaks", str(baseline["break_count"])),
            ("Tasks", str(baseline["task_count"])),
            ("Attendance", baseline["attendance_status"]),
            ("Top location", baseline["top_module"]),
        ]

        grid = ctk.CTkFrame(self._detail_frame, fg_color=C_BORDER, corner_radius=16)
        grid.pack(fill="x", pady=(0, 16))
        for i, (label, value) in enumerate(metrics):
            cell = ctk.CTkFrame(grid, fg_color=C_CARD, corner_radius=14)
            cell.grid(row=i // 2, column=i % 2, padx=12, pady=12, sticky="nsew")
            grid.grid_columnconfigure(i % 2, weight=1)
            ctk.CTkLabel(cell, text=label, text_color=C_MUTED, font=ctk.CTkFont(size=10)).pack(anchor="w", padx=12, pady=(10, 4))
            ctk.CTkLabel(cell, text=value, text_color=C_TEXT, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=12, pady=(0, 10))

        ctk.CTkLabel(self._detail_frame, text="Recent Activity", font=ctk.CTkFont(size=14, weight="bold"), text_color=C_TEXT).pack(anchor="w", pady=(0, 8))

        for record in baseline["recent_records"]:
            row = ctk.CTkFrame(self._detail_frame, fg_color=C_BG, corner_radius=12)
            row.pack(fill="x", pady=4, padx=0)
            ctk.CTkLabel(row, text=record.get("date", "-"), width=100, anchor="w", text_color=C_TEXT, font=ctk.CTkFont(size=11)).pack(side="left", padx=8, pady=8)
            ctk.CTkLabel(row, text=record.get("login_time", "-"), width=90, anchor="w", text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(row, text=record.get("location_mode", "-"), width=130, anchor="w", text_color=C_TEXT, font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(row, text=f"Prod {record.get('productivity_score', 0):.1f}%", width=110, anchor="w", text_color=C_AMBER if record.get("productivity_score", 0) < 50 else C_GREEN, font=ctk.CTkFont(size=11)).pack(side="left")
            ctk.CTkLabel(row, text=record.get("anomaly_type", "Normal"), width=140, anchor="w", text_color=C_MUTED, font=ctk.CTkFont(size=11)).pack(side="left")

    def refresh_data(self):
        self.records = load_records()
        self.baselines = build_employee_baselines(self.records)
        self._render_summary()
        self._render_employee_list()
        if self.selected_employee_id not in self.baselines:
            if self.baselines:
                first_id = next(iter(sorted(self.baselines.keys(), key=lambda eid: self.baselines[eid]["employee_name"])))
                self.selected_employee_id = first_id
        if self.selected_employee_id:
            self.show_employee_detail(self.selected_employee_id)


def launch_baseline_viewer():
    app = BehavioralBaselineApp()
    app.mainloop()


if __name__ == "__main__":
    launch_baseline_viewer()
