from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import threading
import time

import joblib
import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

logger = logging.getLogger(__name__)


@dataclass
class EmployeeEfficiencyResult:
    employee_id: str
    full_name: str
    predicted_label: str
    confidence: float
    efficiency_score: float
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
    efficiency_score: float
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


@dataclass
class EmployeeWeeklyForecast:
    employee_id: str
    full_name: str
    available: bool
    message: str
    history_weeks: list[str] = field(default_factory=list)
    history_scores: list[float] = field(default_factory=list)
    forecast_week: str = ""
    forecast_score: Optional[float] = None
    source_weeks: int = 0


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
        # Load the trained model and label encoder, then capture the model's expected feature order.
        self._model = joblib.load(self._model_path)
        self._label_encoder = joblib.load(self._label_encoder_path)
        names = getattr(self._model, "feature_names_in_", None)
        if names is None:
            raise ValueError("Model does not expose feature_names_in_.")
        self._feature_names = [str(n) for n in names]

    def predict_all(self, db_client, period_start: Optional[datetime] = None, period_end: Optional[datetime] = None) -> list[EmployeeEfficiencyResult]:
        # Build one feature row per employee, run the classifier, and return ranked prediction results.
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
                    efficiency_score=self._calculate_efficiency_score(
                        float(stats["workload_score"]),
                        str(label),
                        confidence,
                    ),
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
        # Convert the raw prediction into a readable employee-level productivity summary.
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
            efficiency_score=target.efficiency_score,
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

    def get_employee_lime_explanation(
        self,
        db_client,
        employee_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        max_features: int = 5,
    ) -> list[str]:
        # Build a local LIME explanation from the current employee slice without changing the model or stored data.
        target_id = self._normalize_employee_id(employee_id)
        if not target_id:
            return ["No employee was selected for the explanation."]

        if self._model is None or self._label_encoder is None:
            self.load()

        if db_client is None or not getattr(db_client, "is_connected", False):
            return ["No database connection is available for the explanation."]

        emp_col = db_client.get_collection("employees")
        task_col = db_client.get_collection("tasks")
        activity_col = db_client.get_collection("activity_logs")

        if emp_col is None:
            return ["No employee data is available for explanation."]

        employees = list(emp_col.find({}, {"_id": 0}))
        if not employees:
            return ["No employee records were found for explanation."]

        employee_lookup: dict[str, dict] = {}
        for emp in employees:
            employee_id_value = str(emp.get("employee_id") or "").strip()
            if not employee_id_value:
                continue
            employee_lookup.setdefault(self._normalize_employee_id(employee_id_value), emp)

        tasks_by_employee: dict[str, list[dict]] = {key: [] for key in employee_lookup}
        if task_col is not None:
            for task in task_col.find({}, {"_id": 0}):
                raw_employee_id = str(task.get("employee_id") or "").strip()
                if not raw_employee_id:
                    continue
                normalized_employee_id = self._normalize_employee_id(raw_employee_id)
                if normalized_employee_id not in employee_lookup:
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

        rows: list[dict] = []
        meta: list[dict] = []
        target_index = None

        for idx, (employee_key, emp) in enumerate(employee_lookup.items()):
            employee_id_value = str(emp.get("employee_id") or "").strip()
            if not employee_id_value:
                continue

            tasks = self._filter_tasks_for_period(tasks_by_employee.get(employee_key, []), period_start, period_end)
            activity_logs = self._filter_activity_for_period(activity_by_employee.get(employee_key, []), period_start, period_end)

            row, stats = self._build_feature_row(emp, tasks, activity_logs)
            rows.append(row)
            meta.append(stats)

            if self._normalize_employee_id(employee_id_value) == target_id:
                target_index = len(rows) - 1

        if target_index is None or not rows:
            return ["No explanation could be generated for the selected employee."]

        frame = pd.DataFrame(rows, columns=self._feature_names)
        if len(frame) < 2:
            return ["Not enough employee data is available yet to generate a reliable LIME explanation."]

        if LimeTabularExplainer is None:
            return self._fallback_explanation(meta[target_index], target_id)

        preprocessor = getattr(self._model, "named_steps", {}).get("preprocessor")
        model = getattr(self._model, "named_steps", {}).get("model")
        if preprocessor is None or model is None:
            return ["The model structure does not support LIME explanation in the current run."]

        transformed = preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = transformed.astype(float, copy=False)

        feature_names = [str(name) for name in preprocessor.get_feature_names_out()]
        class_names = [str(name) for name in getattr(self._label_encoder, "classes_", [])]

        explainer = LimeTabularExplainer(
            training_data=transformed,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
        )

        target_label = str(meta[target_index]["employee_id"])
        target_row = transformed[target_index]
        prediction = str(self._label_encoder.inverse_transform(self._model.predict(frame))[target_index])
        label_index = int(self._label_encoder.transform([prediction])[0])

        friendly_lines: list[str] = []

        explanation = explainer.explain_instance(
            data_row=target_row,
            predict_fn=model.predict_proba,
            num_features=max(1, min(max_features, len(feature_names))),
            top_labels=1,
        )

        raw_factors = explanation.as_list(label=label_index) or explanation.as_list()
        for factor, weight in raw_factors[:max_features]:
            factor_line = self._format_lime_factor(factor, float(weight), prediction)
            if factor_line:
                friendly_lines.append(factor_line)

        if not friendly_lines:
            friendly_lines.append(f"The model did not produce a detailed factor breakdown for {target_label}.")

        return friendly_lines

    def get_employee_weekly_forecast(
        self,
        db_client,
        employee_id: str,
        period_end: Optional[datetime] = None,
        lookback_weeks: int = 8,
        min_weeks: int = 4,
    ) -> EmployeeWeeklyForecast:
        target_id = self._normalize_employee_id(employee_id)
        if not target_id:
            return EmployeeWeeklyForecast(
                employee_id="",
                full_name="",
                available=False,
                message="No employee was selected for the forecast.",
            )

        if self._model is None or self._label_encoder is None:
            self.load()

        if db_client is None or not getattr(db_client, "is_connected", False):
            return EmployeeWeeklyForecast(
                employee_id=employee_id,
                full_name=employee_id,
                available=False,
                message="No database connection is available for the forecast.",
            )

        emp_col = db_client.get_collection("employees")
        task_col = db_client.get_collection("tasks")
        activity_col = db_client.get_collection("activity_logs")

        if emp_col is None or task_col is None or activity_col is None:
            return EmployeeWeeklyForecast(
                employee_id=employee_id,
                full_name=employee_id,
                available=False,
                message="The forecast cannot be generated because one or more source collections are unavailable.",
            )

        employees = list(emp_col.find({}, {"_id": 0}))
        if not employees:
            return EmployeeWeeklyForecast(
                employee_id=employee_id,
                full_name=employee_id,
                available=False,
                message="No employee records are available for forecasting.",
            )

        employee_lookup: dict[str, dict] = {}
        for emp in employees:
            employee_id_value = str(emp.get("employee_id") or "").strip()
            if not employee_id_value:
                continue
            employee_lookup.setdefault(self._normalize_employee_id(employee_id_value), emp)

        emp = employee_lookup.get(target_id)
        if emp is None:
            return EmployeeWeeklyForecast(
                employee_id=employee_id,
                full_name=employee_id,
                available=False,
                message="No employee record was found for the selected employee.",
            )

        employee_name = str(emp.get("full_name") or emp.get("employee_id") or employee_id)

        tasks = [
            task for task in task_col.find({}, {"_id": 0})
            if self._normalize_employee_id(str(task.get("employee_id") or "")) == target_id
        ]
        activity_logs = [
            doc for doc in activity_col.find({}, {"_id": 0, "productivity_score": 1, "timestamp": 1, "user_id": 1})
            if self._normalize_employee_id(str(doc.get("user_id") or "")) == target_id
        ]

        anchor_end = period_end or datetime.now(timezone.utc)
        if anchor_end.tzinfo is None:
            anchor_end = anchor_end.replace(tzinfo=timezone.utc)
        else:
            anchor_end = anchor_end.astimezone(timezone.utc)

        history_start = anchor_end - timedelta(days=7 * max(lookback_weeks, min_weeks))
        weekly_buckets: dict[date, dict[str, list[dict]]] = {}

        for task in tasks:
            ref_dt = self._task_reference_dt(task)
            if ref_dt is None or ref_dt < history_start or ref_dt > anchor_end:
                continue
            week_start = (ref_dt - timedelta(days=ref_dt.weekday())).date()
            bucket = weekly_buckets.setdefault(week_start, {"tasks": [], "activity": []})
            bucket["tasks"].append(task)

        for doc in activity_logs:
            ts = self._parse_dt(doc.get("timestamp"))
            if ts is None or ts < history_start or ts > anchor_end:
                continue
            week_start = (ts - timedelta(days=ts.weekday())).date()
            bucket = weekly_buckets.setdefault(week_start, {"tasks": [], "activity": []})
            bucket["activity"].append(doc)

        if len(weekly_buckets) < min_weeks:
            return EmployeeWeeklyForecast(
                employee_id=str(emp.get("employee_id") or employee_id),
                full_name=employee_name,
                available=False,
                message=(
                    "Current data is not sufficient for a reliable next-week forecast. "
                    f"At least {min_weeks} weekly history points are needed."
                ),
                source_weeks=len(weekly_buckets),
            )

        weekly_points: list[tuple[date, float]] = []
        for week_start in sorted(weekly_buckets):
            bucket = weekly_buckets[week_start]
            row, stats = self._build_feature_row(emp, bucket["tasks"], bucket["activity"])
            frame = pd.DataFrame([row], columns=self._feature_names)

            pred_label = str(self._label_encoder.inverse_transform(self._model.predict(frame))[0])
            confidence = float(self._model.predict_proba(frame)[0].max())
            efficiency_score = self._calculate_efficiency_score(
                float(stats["workload_score"]),
                pred_label,
                confidence,
            )
            weekly_points.append((week_start, efficiency_score))

        if len(weekly_points) < min_weeks:
            return EmployeeWeeklyForecast(
                employee_id=str(emp.get("employee_id") or employee_id),
                full_name=employee_name,
                available=False,
                message=(
                    "Current data is not sufficient for a reliable next-week forecast. "
                    f"At least {min_weeks} weekly history points are needed."
                ),
                source_weeks=len(weekly_points),
            )

        weekly_points = weekly_points[-lookback_weeks:]
        week_labels = [week_start.strftime("%b %d") for week_start, _ in weekly_points]
        week_scores = [float(score) for _, score in weekly_points]

        x = np.arange(len(week_scores), dtype=float)
        y = np.asarray(week_scores, dtype=float)
        if len(week_scores) >= 2:
            slope, intercept = np.polyfit(x, y, deg=1)
            forecast_raw = float(slope * len(week_scores) + intercept)
            fitted = slope * x + intercept
            rmse = float(np.sqrt(np.mean((y - fitted) ** 2))) if len(week_scores) else 0.0
            stability = max(0.0, min(1.0, 1.0 - (rmse / 30.0)))
            trend = "improving" if slope > 0.5 else "softening" if slope < -0.5 else "stable"
        else:
            forecast_raw = float(week_scores[-1])
            stability = 0.5
            trend = "stable"

        forecast_score = float(max(0.0, min(100.0, round(forecast_raw, 1))))
        next_week_start = weekly_points[-1][0] + timedelta(days=7)
        forecast_week = next_week_start.strftime("%b %d")
        message = (
            f"Based on the last {len(week_scores)} weekly data points, the employee's forecast for next week is "
            f"{forecast_score:.1f}/100 and the recent trend appears {trend}."
        )
        if stability < 0.45:
            message += " The pattern is somewhat volatile, so treat the forecast as directional rather than exact."

        return EmployeeWeeklyForecast(
            employee_id=str(emp.get("employee_id") or employee_id),
            full_name=employee_name,
            available=True,
            message=message,
            history_weeks=week_labels,
            history_scores=week_scores,
            forecast_week=forecast_week,
            forecast_score=forecast_score,
            source_weeks=len(week_scores),
        )

    def _fallback_explanation(self, stats: dict, employee_id: str) -> list[str]:
        lines = ["A detailed LIME breakdown could not be generated in this run, so the system is using a plain-language fallback explanation."]

        assigned = int(stats.get("total_tasks_assigned", 0) or 0)
        pending = int(stats.get("total_tasks_pending", 0) or 0)
        on_time = int(stats.get("total_tasks_completed_on_time", 0) or 0)
        late = int(stats.get("total_tasks_completed_late", 0) or 0)
        completed = on_time + late
        workload = float(stats.get("workload_score", 0.0) or 0.0)
        prod_input = float(stats.get("productivity_score_input", 0.0) or 0.0)

        if assigned == 0:
            lines.append("No tasks were assigned in this period, so the score is based mostly on activity signals.")
        else:
            backlog_ratio = pending / assigned
            completion_ratio = completed / assigned
            if backlog_ratio >= 0.5:
                lines.append("A large share of tasks is still pending, which lowers the efficiency score.")
            elif backlog_ratio <= 0.2:
                lines.append("Most assigned tasks are not pending, which supports a stronger efficiency score.")

            if completion_ratio >= 0.8:
                lines.append("Task completion is strong for this employee during the selected period.")
            elif completion_ratio < 0.5:
                lines.append("Only a small portion of assigned tasks is completed, which reduces the score.")

            if completed > 0 and late > on_time:
                lines.append("More tasks were completed late than on time, which pulls the score down.")

        if prod_input >= 70:
            lines.append("Activity productivity is strong, so the model sees stable work behavior.")
        elif prod_input < 40:
            lines.append("Activity productivity is low, which usually indicates weaker work signals.")

        if workload >= 70:
            lines.append("Workload performance is healthy and helps the employee score.")
        elif workload < 40:
            lines.append("Workload performance is weak because of limited completion or high pending work.")

        if len(lines) == 1:
            lines.append(f"No additional explanation signals were strong for {employee_id} in this period.")

        return lines

    def _build_feature_row(self, emp: dict, tasks: list[dict], activity_logs: list[dict]) -> tuple[dict, dict]:
        # Assemble model-ready inputs and the reporting metadata from employee, task, and activity data.
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

    # Helper methods for data parsing, normalization, caching, and feature extraction.
    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value in (None, ""):
                return default
            return float(value)
        except Exception:
            return default

    # For categorical features, use the most common value or a default if no data is available.
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

    # Cache keys are based on the database name and formatted period start,end datetimes 
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

    @staticmethod
    def _format_lime_factor(feature_term: str, weight: float, predicted_label: str) -> str:
        term = str(feature_term)
        base_term = term.split(" <= ")[0].split(" > ")[0].split(" = ")[0].strip()
        plain_name = EfficiencyPredictionService._plain_feature_name(base_term)
        if plain_name.lower().startswith("employee id") or plain_name.lower().startswith("employee name"):
            return ""

        direction = "helped improve" if weight >= 0 else "pulled down"
        return f"{plain_name} {direction} the employee's score."

    @staticmethod
    def _plain_feature_name(feature_name: str) -> str:
        cleaned = str(feature_name).replace("cat__", "").replace("num__", "")
        cleaned = cleaned.replace("_", " ").strip()

        replacements = [
            ("workload score", "Workload"),
            ("productivity score", "Activity productivity"),
            ("total tasks pending", "Pending tasks"),
            ("total tasks completed on time", "On-time completions"),
            ("total tasks completed late", "Late completions"),
            ("total tasks assigned", "Assigned tasks"),
            ("completion ratio", "Completion rate"),
            ("on time ratio", "On-time completion rate"),
            ("backlog ratio", "Backlog"),
            ("average time deviation", "Schedule timing"),
            ("similar tasks completed count", "Similar completed tasks"),
            ("allocated hours", "Allocated time"),
            ("actual hours", "Actual time"),
            ("task priority", "Task priority"),
            ("task category", "Task category"),
            ("task status", "Task status"),
            ("department", "Department"),
            ("role", "Role"),
        ]

        lowered = cleaned.lower()
        for needle, label in replacements:
            if needle in lowered:
                return label

        return cleaned.title() if cleaned else "A model feature"

    @staticmethod
    def _label_score(label: str) -> float:
        return {
            "high": 100.0,
            "medium": 65.0,
            "low": 30.0,
        }.get(str(label).strip().lower(), 0.0)

    @classmethod
    def _calculate_efficiency_score(cls, workload_score: float, predicted_label: str, confidence: float) -> float:
        label_score = cls._label_score(predicted_label)
        score = 0.70 * float(workload_score) + 0.30 * (label_score * float(confidence))
        return float(round(max(0.0, min(100.0, score)), 3))

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
