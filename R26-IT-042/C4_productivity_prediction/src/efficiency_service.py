from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import threading
import time

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EmployeeEfficiencyResult:
    employee_id: str
    full_name: str
    predicted_label: str
    confidence: float
    productivity_score_input: float
    workload_score: float
    total_tasks_assigned: int
    total_tasks_pending: int
    total_tasks_completed_on_time: int
    total_tasks_completed_late: int


@dataclass
class EmployeeProductivityReport:
    employee_id: str
    full_name: str
    predicted_label: str
    confidence: float
    productivity_score: float
    workload_score: float
    total_tasks_assigned: int
    total_tasks_pending: int
    total_tasks_completed_on_time: int
    total_tasks_completed_late: int
    completion_ratio: float
    on_time_ratio: float
    backlog_ratio: float
    summary: str
    insights: list[str]


class EfficiencyPredictionService:
    """Read-only C4 prediction service built on top of existing MongoDB data."""

    _cache_lock = threading.Lock()
    _result_cache: dict[tuple[str, str, str], tuple[float, list[EmployeeEfficiencyResult]]] = {}
    _cache_ttl_seconds = 12.0

    def __init__(self, model_path: Optional[Path] = None, label_encoder_path: Optional[Path] = None) -> None:
        base = Path(__file__).resolve().parent
        self._model_path = model_path or (base / "productivity_classifier.joblib")
        self._label_encoder_path = label_encoder_path or (base / "label_encoder.joblib")
        self._model = None
        self._label_encoder = None
        self._feature_names: list[str] = []

    def load(self) -> None:
        self._model = joblib.load(self._model_path)
        self._label_encoder = joblib.load(self._label_encoder_path)
        names = getattr(self._model, "feature_names_in_", None)
        if names is None:
            raise ValueError("Model does not expose feature_names_in_.")
        self._feature_names = [str(n) for n in names]

    def predict_all(self, db_client, period_start: Optional[datetime] = None, period_end: Optional[datetime] = None) -> list[EmployeeEfficiencyResult]:
        if self._model is None or self._label_encoder is None:
            self.load()

        if db_client is None or not getattr(db_client, "is_connected", False):
            return []

        cache_key = (
            str(getattr(db_client, "db_name", "")),
            self._period_cache_key(period_start),
            self._period_cache_key(period_end),
        )

        cached_rows = self._get_cached_results(cache_key)
        if cached_rows is not None:
            return cached_rows

        emp_col = db_client.get_collection("employees")
        task_col = db_client.get_collection("tasks")
        activity_col = db_client.get_collection("activity_logs")

        if emp_col is None:
            return []

        employees = list(emp_col.find({}, {"_id": 0}))
        if not employees:
            return []

        employee_lookup: dict[str, dict] = {}
        for emp in employees:
            employee_id = str(emp.get("employee_id") or "").strip()
            if not employee_id:
                continue
            employee_lookup.setdefault(self._normalize_employee_id(employee_id), emp)

        tasks_by_employee: dict[str, list[dict]] = {key: [] for key in employee_lookup}
        orphan_task_ids: list[str] = []
        if task_col is not None:
            for task in task_col.find({}, {"_id": 0}):
                raw_employee_id = str(task.get("employee_id") or "").strip()
                if not raw_employee_id:
                    continue
                normalized_employee_id = self._normalize_employee_id(raw_employee_id)
                if normalized_employee_id not in employee_lookup:
                    orphan_task_ids.append(raw_employee_id)
                    continue
                tasks_by_employee.setdefault(normalized_employee_id, []).append(task)

        activity_by_employee: dict[str, list[dict]] = {key: [] for key in employee_lookup}
        if activity_col is not None:
            for doc in activity_col.find({}, {"_id": 0, "productivity_score": 1, "timestamp": 1, "user_id": 1}):
                raw_user_id = str(doc.get("user_id") or "").strip()
                if not raw_user_id:
                    continue
                normalized_user_id = self._normalize_employee_id(raw_user_id)
                if normalized_user_id not in employee_lookup:
                    continue
                activity_by_employee.setdefault(normalized_user_id, []).append(doc)

        if orphan_task_ids:
            orphan_count = len(orphan_task_ids)
            preview = sorted(set(orphan_task_ids))[:5]
            logger.warning("Efficiency service ignored %s task records with no matching employee: %s", orphan_count, preview)

        rows: list[dict] = []
        meta: list[dict] = []

        for employee_key, emp in employee_lookup.items():
            employee_id = str(emp.get("employee_id") or "").strip()
            if not employee_id:
                continue

            tasks = self._filter_tasks_for_period(tasks_by_employee.get(employee_key, []), period_start, period_end)
            activity_logs = self._filter_activity_for_period(activity_by_employee.get(employee_key, []), period_start, period_end)

            row, stats = self._build_feature_row(emp, tasks, activity_logs)
            rows.append(row)
            meta.append(stats)

        if not rows:
            return []

        frame = pd.DataFrame(rows, columns=self._feature_names)
        pred_encoded = self._model.predict(frame)
        pred_proba = self._model.predict_proba(frame)
        labels = self._label_encoder.inverse_transform(pred_encoded)

        results: list[EmployeeEfficiencyResult] = []
        for idx, label in enumerate(labels):
            confidence = float(pred_proba[idx].max())
            stats = meta[idx]
            results.append(
                EmployeeEfficiencyResult(
                    employee_id=stats["employee_id"],
                    full_name=stats["full_name"],
                    predicted_label=str(label),
                    confidence=confidence,
                    productivity_score_input=float(stats["productivity_score_input"]),
                    workload_score=float(stats["workload_score"]),
                    total_tasks_assigned=int(stats["total_tasks_assigned"]),
                    total_tasks_pending=int(stats["total_tasks_pending"]),
                    total_tasks_completed_on_time=int(stats["total_tasks_completed_on_time"]),
                    total_tasks_completed_late=int(stats["total_tasks_completed_late"]),
                )
            )

        final_results = sorted(results, key=lambda r: (r.predicted_label, -r.confidence, r.employee_id))
        self._store_cached_results(cache_key, final_results)
        return final_results

    def get_employee_productivity_report(
        self,
        db_client,
        employee_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> Optional[EmployeeProductivityReport]:
        target_id = self._normalize_employee_id(employee_id)
        if not target_id:
            return None

        rows = self.predict_all(db_client, period_start=period_start, period_end=period_end)
        target = None
        for row in rows:
            if self._normalize_employee_id(row.employee_id) == target_id:
                target = row
                break

        if target is None:
            return None

        assigned = int(target.total_tasks_assigned)
        pending = int(target.total_tasks_pending)
        on_time = int(target.total_tasks_completed_on_time)
        late = int(target.total_tasks_completed_late)
        completed = on_time + late

        completion_ratio = (completed / assigned) if assigned else 0.0
        on_time_ratio = (on_time / completed) if completed else 0.0
        backlog_ratio = (pending / assigned) if assigned else 0.0

        confidence_pct = target.confidence * 100.0
        summary = (
            f"Predicted productivity is {target.predicted_label} "
            f"with {confidence_pct:.1f}% confidence."
        )

        insights: list[str] = []
        if assigned == 0:
            insights.append("No tasks were assigned in this period, so task-based metrics are limited.")
        if backlog_ratio >= 0.5:
            insights.append("Backlog is high compared to assigned tasks, which lowers workload performance.")
        elif backlog_ratio <= 0.2 and assigned > 0:
            insights.append("Pending task ratio is low, indicating good task flow.")

        if completed > 0:
            if on_time_ratio >= 0.8:
                insights.append("Completed tasks are mostly delivered on time.")
            elif on_time_ratio < 0.5:
                insights.append("On-time completion rate is low and likely affecting productivity output.")

        if target.productivity_score_input >= 70:
            insights.append("Observed productivity score is strong for this period.")
        elif target.productivity_score_input < 40:
            insights.append("Observed productivity score is low and needs attention.")

        if target.workload_score < 40:
            insights.append("Workload score is low due to limited completion or high pending tasks.")
        elif target.workload_score >= 70:
            insights.append("Workload score is healthy and indicates stable execution.")

        if confidence_pct < 60:
            insights.append("Prediction confidence is moderate; monitor additional periods for stability.")

        if not insights:
            insights.append("Productivity signals are stable for this period.")

        return EmployeeProductivityReport(
            employee_id=target.employee_id,
            full_name=target.full_name,
            predicted_label=target.predicted_label,
            confidence=target.confidence,
            productivity_score=target.productivity_score_input,
            workload_score=target.workload_score,
            total_tasks_assigned=assigned,
            total_tasks_pending=pending,
            total_tasks_completed_on_time=on_time,
            total_tasks_completed_late=late,
            completion_ratio=completion_ratio,
            on_time_ratio=on_time_ratio,
            backlog_ratio=backlog_ratio,
            summary=summary,
            insights=insights,
        )

    def _build_feature_row(self, emp: dict, tasks: list[dict], activity_logs: list[dict]) -> tuple[dict, dict]:
        now = datetime.now(timezone.utc)
        employee_id = str(emp.get("employee_id") or "UNKNOWN")
        full_name = str(emp.get("full_name") or employee_id)

        completed = [t for t in tasks if str(t.get("status") or "") == "completed"]
        pending_like = [t for t in tasks if str(t.get("status") or "") in {"pending", "in_progress", "paused"}]

        on_time = 0
        late = 0
        time_deviations_hours: list[float] = []
        allocated_hours_values: list[float] = []
        actual_hours_values: list[float] = []

        high_on_time = 0
        medium_on_time = 0
        low_on_time = 0

        categories: list[str] = []
        priorities: list[str] = []

        latest_assigned = None
        latest_deadline = None
        active_status = "pending"

        for t in tasks:
            assigned_at = self._parse_dt(t.get("assigned_at"))
            due = self._task_due_dt(t)
            completed_at = self._parse_dt(t.get("completed_at"))

            if assigned_at and (latest_assigned is None or assigned_at > latest_assigned):
                latest_assigned = assigned_at
            if due and (latest_deadline is None or due > latest_deadline):
                latest_deadline = due

            p = str(t.get("priority") or "medium").lower()
            priorities.append(p)

            category = str(t.get("task_category") or "general").strip().lower()
            if not category:
                category = "general"
            categories.append(category)

            allocated_minutes = self._to_float(t.get("allocated_minutes"), default=0.0)
            if allocated_minutes > 0:
                allocated_hours_values.append(allocated_minutes / 60.0)

            actual_seconds = self._to_float(t.get("actual_seconds"), default=0.0)
            if actual_seconds > 0:
                actual_hours_values.append(actual_seconds / 3600.0)

            status = str(t.get("status") or "pending")
            if status == "in_progress":
                active_status = "in_progress"
            elif status == "paused" and active_status != "in_progress":
                active_status = "paused"

            if completed_at and due:
                dev_h = (completed_at - due).total_seconds() / 3600.0
                time_deviations_hours.append(dev_h)
                is_on_time = completed_at <= due
                if is_on_time:
                    on_time += 1
                    if p == "high":
                        high_on_time += 1
                    elif p == "medium":
                        medium_on_time += 1
                    else:
                        low_on_time += 1
                else:
                    late += 1

        total_assigned = len(tasks)
        total_pending = len(pending_like)
        avg_dev = sum(time_deviations_hours) / len(time_deviations_hours) if time_deviations_hours else 0.0

        allocated_hours = sum(allocated_hours_values) if allocated_hours_values else 1.0
        actual_hours = sum(actual_hours_values) if actual_hours_values else 0.0

        if allocated_hours <= 0.0:
            allocated_hours = 1.0

        completion_ratio = (on_time + late) / total_assigned if total_assigned else 0.0
        backlog_ratio = total_pending / total_assigned if total_assigned else 0.0
        workload_score = max(0.0, min(100.0, 100.0 * (0.55 * completion_ratio + 0.45 * (1.0 - backlog_ratio))))

        productivity_input = self._activity_productivity(activity_logs)
        if productivity_input is None:
            productivity_input = max(0.0, min(100.0, workload_score))

        dominant_priority = self._mode_or_default(priorities, "medium")
        dominant_category = self._mode_or_default(categories, "general")
        similar_tasks_completed_count = sum(1 for t in completed if str(t.get("task_category") or "general").strip().lower() == dominant_category)

        join_date = str(emp.get("created_at") or now.date().isoformat())

        assigned_date = self._date_to_days((latest_assigned or now).date().isoformat())
        deadline_date = self._date_to_days((latest_deadline or now).date().isoformat())
        join_date_value = self._date_to_days(join_date)

        feature_row = {
            "employee_id": employee_id,
            "department": str(emp.get("department") or "IT"),
            "role": str(emp.get("role") or "Employee"),
            "join_date": join_date_value,
            "task_category": dominant_category,
            "task_priority": dominant_priority,
            "allocated_hours": float(round(allocated_hours, 3)),
            "actual_hours": float(round(actual_hours, 3)),
            "task_status": active_status,
            "assigned_date": assigned_date,
            "deadline_date": deadline_date,
            "month": int(now.month),
            "year": int(now.year),
            "total_tasks_assigned": int(total_assigned),
            "total_tasks_completed_on_time": int(on_time),
            "total_tasks_completed_late": int(late),
            "total_tasks_pending": int(total_pending),
            "high_priority_completed_on_time": int(high_on_time),
            "medium_priority_completed_on_time": int(medium_on_time),
            "low_priority_completed_on_time": int(low_on_time),
            "average_time_deviation": float(round(avg_dev, 3)),
            "similar_tasks_completed_count": int(similar_tasks_completed_count),
            "workload_score": float(round(workload_score, 3)),
            "productivity_score": float(round(productivity_input, 3)),
        }

        stats = {
            "employee_id": employee_id,
            "full_name": full_name,
            "productivity_score_input": feature_row["productivity_score"],
            "workload_score": feature_row["workload_score"],
            "total_tasks_assigned": total_assigned,
            "total_tasks_pending": total_pending,
            "total_tasks_completed_on_time": on_time,
            "total_tasks_completed_late": late,
        }

        # Ensure exact model ordering and fill any missing with reasonable defaults.
        ordered = {}
        for col in self._feature_names:
            if col in feature_row:
                ordered[col] = feature_row[col]
            else:
                ordered[col] = 0

        return ordered, stats

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _mode_or_default(values: list[str], default: str) -> str:
        if not values:
            return default
        counts: dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

    @staticmethod
    def _normalize_employee_id(value: str) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _period_cache_key(value: Optional[datetime]) -> str:
        if value is None:
            return "all"
        return value.astimezone(timezone.utc).isoformat()

    @classmethod
    def _get_cached_results(cls, cache_key: tuple[str, str, str]) -> Optional[list[EmployeeEfficiencyResult]]:
        now = time.time()
        with cls._cache_lock:
            cached = cls._result_cache.get(cache_key)
            if cached is None:
                return None
            expires_at, rows = cached
            if expires_at <= now:
                cls._result_cache.pop(cache_key, None)
                return None
            return list(rows)

    @classmethod
    def _store_cached_results(cls, cache_key: tuple[str, str, str], rows: list[EmployeeEfficiencyResult]) -> None:
        expires_at = time.time() + cls._cache_ttl_seconds
        with cls._cache_lock:
            cls._result_cache[cache_key] = (expires_at, list(rows))

    @staticmethod
    def _parse_dt(value) -> Optional[datetime]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _date_to_days(value: str) -> int:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return dt.date().toordinal()
        except Exception:
            return 0

    def _task_due_dt(self, task: dict) -> Optional[datetime]:
        due_at = task.get("due_at")
        parsed = self._parse_dt(due_at)
        if parsed is not None:
            return parsed

        due_date = str(task.get("due_date") or "").strip()
        due_time = str(task.get("due_time") or "").strip() or "23:59"
        if not due_date:
            return None
        try:
            dt = datetime.fromisoformat(f"{due_date}T{due_time}:00")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def _activity_productivity(self, activity_logs: list[dict]) -> Optional[float]:
        values: list[float] = []
        for doc in activity_logs:
            try:
                score = float(doc.get("productivity_score"))
                values.append(score)
            except Exception:
                continue
        if not values:
            return None
        return sum(values) / len(values)

    def _filter_tasks_for_period(
        self,
        tasks: list[dict],
        period_start: Optional[datetime],
        period_end: Optional[datetime],
    ) -> list[dict]:
        if period_start is None or period_end is None:
            return tasks

        filtered: list[dict] = []
        for task in tasks:
            ref_dt = self._task_reference_dt(task)
            if ref_dt is None:
                continue
            if period_start <= ref_dt <= period_end:
                filtered.append(task)
        return filtered

    def _filter_activity_for_period(
        self,
        activity_logs: list[dict],
        period_start: Optional[datetime],
        period_end: Optional[datetime],
    ) -> list[dict]:
        if period_start is None or period_end is None:
            return activity_logs

        filtered: list[dict] = []
        for doc in activity_logs:
            ts = self._parse_dt(doc.get("timestamp"))
            if ts is None:
                continue
            if period_start <= ts <= period_end:
                filtered.append(doc)
        return filtered

    def _task_reference_dt(self, task: dict) -> Optional[datetime]:
        completed = self._parse_dt(task.get("completed_at"))
        if completed is not None:
            return completed

        assigned = self._parse_dt(task.get("assigned_at"))
        if assigned is not None:
            return assigned

        due = self._task_due_dt(task)
        if due is not None:
            return due

        return None
