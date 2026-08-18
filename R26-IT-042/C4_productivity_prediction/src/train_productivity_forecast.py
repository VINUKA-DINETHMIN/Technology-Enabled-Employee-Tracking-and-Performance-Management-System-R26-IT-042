from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    mae: float
    rmse: float
    r2: float
    mape: float
    accuracy_pct: float


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])  # type: ignore[assignment]

    expected_cols = {
        "employee_id",
        "full_name",
        "department",
        "role",
        "date",
        "day_of_week",
        "is_weekend",
        "productivity_score",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    return df.sort_values(["employee_id", "date"]).reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week_num"] = out["date"].dt.dayofweek
    out["day_sin"] = np.sin(2 * np.pi * out["day_of_week_num"] / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * out["day_of_week_num"] / 7.0)
    out["month_num"] = out["date"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month_num"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month_num"] / 12.0)
    return out


def build_supervised_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    work = add_time_features(df)

    lag_cols = [
        "productivity_score",
        "workload_score",
        "tasks_assigned_today",
        "tasks_completed_today",
        "total_tasks_pending",
    ]
    lag_steps = [1, 2, 3, 7, 14]

    grouped = work.groupby("employee_id", group_keys=False)

    for col in lag_cols:
        for lag in lag_steps:
            work[f"{col}_lag_{lag}"] = grouped[col].shift(lag)

    work["productivity_roll_mean_7"] = grouped["productivity_score"].shift(1).rolling(7).mean().reset_index(level=0, drop=True)
    work["productivity_roll_std_7"] = grouped["productivity_score"].shift(1).rolling(7).std().reset_index(level=0, drop=True)
    work["productivity_roll_mean_14"] = grouped["productivity_score"].shift(1).rolling(14).mean().reset_index(level=0, drop=True)

    work["target_productivity_next_day"] = grouped["productivity_score"].shift(-1)
    work["target_date"] = grouped["date"].shift(-1)

    rows = work.dropna(subset=["target_productivity_next_day", "target_date"]).copy()

    target_date = pd.to_datetime(rows["target_date"])
    rows["target_day_of_week"] = target_date.dt.day_name()
    rows["target_is_weekend"] = (target_date.dt.dayofweek >= 5).astype(int)
    rows["target_day_of_week_num"] = target_date.dt.dayofweek
    rows["target_day_sin"] = np.sin(2 * np.pi * rows["target_day_of_week_num"] / 7.0)
    rows["target_day_cos"] = np.cos(2 * np.pi * rows["target_day_of_week_num"] / 7.0)
    rows["target_month_num"] = target_date.dt.month
    rows["target_month_sin"] = np.sin(2 * np.pi * rows["target_month_num"] / 12.0)
    rows["target_month_cos"] = np.cos(2 * np.pi * rows["target_month_num"] / 12.0)

    # Keep lag rows with sufficient history so temporal signals are reliable.
    required_lag_cols = [f"productivity_score_lag_{lag}" for lag in [1, 2, 3, 7, 14]]
    rows = rows.dropna(subset=required_lag_cols).copy()

    feature_cols = [
        "employee_id",
        "department",
        "role",
        "target_day_of_week",
        "target_is_weekend",
        "target_day_of_week_num",
        "target_day_sin",
        "target_day_cos",
        "target_month_num",
        "target_month_sin",
        "target_month_cos",
        "completion_rate_on_time",
        "late_rate",
    ]

    lag_feature_cols = [
        c
        for c in rows.columns
        if c.endswith("_lag_1")
        or c.endswith("_lag_2")
        or c.endswith("_lag_3")
        or c.endswith("_lag_7")
        or c.endswith("_lag_14")
    ]

    feature_cols.extend(sorted(lag_feature_cols))
    feature_cols.extend(["productivity_roll_mean_7", "productivity_roll_std_7", "productivity_roll_mean_14"])

    x = rows[feature_cols].copy()
    y = rows["target_productivity_next_day"].astype(float)

    meta = rows[["employee_id", "full_name", "date", "target_date"]].copy()
    return x, y, meta


def temporal_train_test_split(
    x: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, test_days: int = 21
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    max_target_date = pd.to_datetime(meta["target_date"]).max()
    cutoff_date = max_target_date - pd.Timedelta(days=test_days)

    train_mask = pd.to_datetime(meta["target_date"]) <= cutoff_date
    test_mask = ~train_mask

    x_train = x.loc[train_mask].copy()
    x_test = x.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()

    meta_train = meta.loc[train_mask].copy()
    meta_test = meta.loc[test_mask].copy()

    if x_train.empty or x_test.empty:
        raise ValueError("Temporal split failed: one split is empty.")

    return x_train, x_test, y_train, y_test, meta_train, meta_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    eps = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
    accuracy_pct = float(max(0.0, 100.0 - mape))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "accuracy_pct": accuracy_pct,
    }


def make_pipeline(base_model: Any, categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_cols,
            ),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", base_model)])
    return pipeline


def train_and_select_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[ModelResult, list[ModelResult], list[str], list[str]]:
    categorical_cols = [
        "employee_id",
        "department",
        "role",
        "target_day_of_week",
    ]
    numeric_cols = [c for c in x_train.columns if c not in categorical_cols]

    candidate_models: list[tuple[str, Any]] = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(
                n_estimators=700,
                max_depth=16,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results: list[ModelResult] = []

    for model_name, estimator in candidate_models:
        pipeline = make_pipeline(estimator, categorical_cols, numeric_cols)
        pipeline.fit(x_train, y_train)

        pred = pipeline.predict(x_test)
        pred = np.clip(pred, 0.0, 100.0)

        m = compute_metrics(y_test.to_numpy(dtype=float), pred.astype(float))
        results.append(
            ModelResult(
                name=model_name,
                pipeline=pipeline,
                mae=m["mae"],
                rmse=m["rmse"],
                r2=m["r2"],
                mape=m["mape"],
                accuracy_pct=m["accuracy_pct"],
            )
        )

    best = sorted(results, key=lambda r: (r.mae, r.rmse))[0]
    return best, results, categorical_cols, numeric_cols


def build_next_feature_row(
    employee_id: str,
    department: str,
    role: str,
    forecast_date: pd.Timestamp,
    prod_hist: list[float],
    workload_hist: list[float],
    tasks_assigned_hist: list[float],
    tasks_completed_hist: list[float],
    pending_hist: list[float],
    completion_rate_last: float,
    late_rate_last: float,
) -> dict[str, Any]:
    dow_num = int(forecast_date.dayofweek)
    month_num = int(forecast_date.month)

    row: dict[str, Any] = {
        "employee_id": employee_id,
        "department": department,
        "role": role,
        "target_day_of_week": forecast_date.day_name(),
        "target_is_weekend": int(dow_num >= 5),
        "target_day_of_week_num": dow_num,
        "target_day_sin": float(np.sin(2 * np.pi * dow_num / 7.0)),
        "target_day_cos": float(np.cos(2 * np.pi * dow_num / 7.0)),
        "target_month_num": month_num,
        "target_month_sin": float(np.sin(2 * np.pi * month_num / 12.0)),
        "target_month_cos": float(np.cos(2 * np.pi * month_num / 12.0)),
        "completion_rate_on_time": float(completion_rate_last),
        "late_rate": float(late_rate_last),
    }

    for lag in [1, 2, 3, 7, 14]:
        row[f"productivity_score_lag_{lag}"] = float(prod_hist[-lag])
        row[f"workload_score_lag_{lag}"] = float(workload_hist[-lag])
        row[f"tasks_assigned_today_lag_{lag}"] = float(tasks_assigned_hist[-lag])
        row[f"tasks_completed_today_lag_{lag}"] = float(tasks_completed_hist[-lag])
        row[f"total_tasks_pending_lag_{lag}"] = float(pending_hist[-lag])

    row["productivity_roll_mean_7"] = float(np.mean(prod_hist[-7:]))
    row["productivity_roll_std_7"] = float(np.std(prod_hist[-7:], ddof=0))
    row["productivity_roll_mean_14"] = float(np.mean(prod_hist[-14:]))

    return row


def forecast_next_week(df: pd.DataFrame, best_model: Pipeline, horizon_days: int = 7) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    base = df.sort_values(["employee_id", "date"]).copy()

    for employee_id, group in base.groupby("employee_id"):
        g = group.reset_index(drop=True)

        if len(g) < 20:
            continue

        full_name = str(g.iloc[-1]["full_name"])
        department = str(g.iloc[-1]["department"])
        role = str(g.iloc[-1]["role"])

        prod_hist = g["productivity_score"].astype(float).tolist()
        workload_hist = g["workload_score"].astype(float).tolist()
        tasks_assigned_hist = g["tasks_assigned_today"].astype(float).tolist()
        tasks_completed_hist = g["tasks_completed_today"].astype(float).tolist()
        pending_hist = g["total_tasks_pending"].astype(float).tolist()

        completion_rate_last = float(g.iloc[-1]["completion_rate_on_time"])
        late_rate_last = float(g.iloc[-1]["late_rate"])

        last_date = pd.to_datetime(g.iloc[-1]["date"])

        for step in range(1, horizon_days + 1):
            forecast_date = last_date + pd.Timedelta(days=step)
            feature_row = build_next_feature_row(
                employee_id=employee_id,
                department=department,
                role=role,
                forecast_date=forecast_date,
                prod_hist=prod_hist,
                workload_hist=workload_hist,
                tasks_assigned_hist=tasks_assigned_hist,
                tasks_completed_hist=tasks_completed_hist,
                pending_hist=pending_hist,
                completion_rate_last=completion_rate_last,
                late_rate_last=late_rate_last,
            )

            x_future = pd.DataFrame([feature_row])
            pred = float(best_model.predict(x_future)[0])
            pred = float(np.clip(pred, 0.0, 100.0))

            rows.append(
                {
                    "employee_id": employee_id,
                    "full_name": full_name,
                    "department": department,
                    "role": role,
                    "forecast_date": forecast_date.date().isoformat(),
                    "day_of_week": forecast_date.day_name(),
                    "predicted_productivity_score": round(pred, 2),
                }
            )

            # Recursive forecasting: feed prediction back into lag history.
            prod_hist.append(pred)
            workload_hist.append(workload_hist[-1])
            tasks_assigned_hist.append(tasks_assigned_hist[-1])
            tasks_completed_hist.append(tasks_completed_hist[-1])
            pending_hist.append(pending_hist[-1])

    return pd.DataFrame(rows)


def write_model_report(
    report_path: Path,
    best: ModelResult,
    all_results: list[ModelResult],
    dataset_rows: int,
    employees: int,
    train_rows: int,
    test_rows: int,
) -> None:
    lines = [
        "# Employee Productivity Forecast Model Report",
        "",
        "## Why this model was selected",
        "The script compares RandomForestRegressor and ExtraTreesRegressor using a temporal holdout split and selects the model with the lowest MAE (tie-breaker: RMSE).",
        "Tree-ensemble regressors were used because they model non-linear productivity patterns, work well on mixed feature types with one-hot encoding, and are robust on medium-sized tabular datasets.",
        "",
        "## Dataset summary",
        f"- Rows: {dataset_rows}",
        f"- Employees: {employees}",
        f"- Training samples: {train_rows}",
        f"- Test samples: {test_rows}",
        "",
        "## Candidate model performance (test set)",
        "| Model | MAE | RMSE | R2 | MAPE (%) | Accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in all_results:
        lines.append(
            f"| {r.name} | {r.mae:.4f} | {r.rmse:.4f} | {r.r2:.4f} | {r.mape:.2f} | {r.accuracy_pct:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Selected model",
            f"- Best model: {best.name}",
            f"- Test MAE: {best.mae:.4f}",
            f"- Test RMSE: {best.rmse:.4f}",
            f"- Test R2: {best.r2:.4f}",
            f"- Test MAPE: {best.mape:.2f}%",
            f"- Test accuracy (100 - MAPE): {best.accuracy_pct:.2f}%",
            "",
            "## Forecast objective",
            "Predict each employee's next-day productivity score, then recursively forecast the upcoming 7 days per employee.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset_path = base_dir / "activity_forecast_dataset.csv"
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    x, y, meta = build_supervised_table(df)

    x_train, x_test, y_train, y_test, meta_train, meta_test = temporal_train_test_split(x, y, meta, test_days=21)

    best, all_results, _, _ = train_and_select_model(x_train, y_train, x_test, y_test)

    y_pred = np.clip(best.pipeline.predict(x_test), 0.0, 100.0)
    metrics = compute_metrics(y_test.to_numpy(dtype=float), y_pred.astype(float))

    results_table = pd.DataFrame(
        [
            {
                "model_name": r.name,
                "mae": r.mae,
                "rmse": r.rmse,
                "r2": r.r2,
                "mape": r.mape,
                "accuracy_pct": r.accuracy_pct,
            }
            for r in all_results
        ]
    ).sort_values(["mae", "rmse"]).reset_index(drop=True)

    bundle = {
        "model": best.pipeline,
        "model_name": best.name,
        "feature_columns": list(x.columns),
        "target": "target_productivity_next_day",
        "metrics": metrics,
        "train_end_date": str(pd.to_datetime(meta_train["target_date"]).max().date()),
        "test_start_date": str(pd.to_datetime(meta_test["target_date"]).min().date()),
        "test_end_date": str(pd.to_datetime(meta_test["target_date"]).max().date()),
        "horizon_days": 7,
    }

    model_path = artifacts_dir / "employee_productivity_forecast_model.joblib"
    joblib.dump(bundle, model_path)

    metrics_payload = {
        "model_name": best.name,
        "target": "target_productivity_next_day",
        "metrics": metrics,
        "feature_columns": list(x.columns),
        "dataset_rows": int(len(df)),
        "employee_count": int(df["employee_id"].nunique()),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_end_date": str(pd.to_datetime(meta_train["target_date"]).max().date()),
        "test_start_date": str(pd.to_datetime(meta_test["target_date"]).min().date()),
        "test_end_date": str(pd.to_datetime(meta_test["target_date"]).max().date()),
        "horizon_days": 7,
    }

    metrics_path = artifacts_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    leaderboard_path = artifacts_dir / "model_comparison.csv"
    results_table.to_csv(leaderboard_path, index=False)

    forecast_df = forecast_next_week(df, best.pipeline, horizon_days=7)
    forecast_path = artifacts_dir / "next_week_employee_productivity_forecast.csv"
    forecast_df.to_csv(forecast_path, index=False)

    report_path = artifacts_dir / "MODEL_REPORT.md"
    write_model_report(
        report_path=report_path,
        best=best,
        all_results=all_results,
        dataset_rows=len(df),
        employees=df["employee_id"].nunique(),
        train_rows=len(x_train),
        test_rows=len(x_test),
    )

    print("Forecast training complete.")
    print(f"Model file: {model_path}")
    print(f"Metrics file: {metrics_path}")
    print(f"Model comparison: {leaderboard_path}")
    print(f"Forecast output: {forecast_path}")
    print(f"Report: {report_path}")
    print(f"Selected model: {best.name}")
    print(f"Accuracy (100 - MAPE): {metrics['accuracy_pct']:.2f}%")


if __name__ == "__main__":
    main()
