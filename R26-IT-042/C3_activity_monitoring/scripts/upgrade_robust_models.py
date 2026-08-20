"""
Upgrade pipeline for robust anomaly detection.

What it does:
1) Strict temporal split: train/val/test (60/20/20)
2) Load existing IF+AE artifacts and generate model scores
3) Train supervised stacker (RandomForest) on raw features + IF/AE scores
4) Calibrate stacker probabilities on validation only (IsotonicRegression)
5) Pick operating thresholds on validation only by target FPR
6) Evaluate once on test and persist upgraded artifacts/config

Run:
  python3 scripts/upgrade_robust_models.py
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "employee_monitoring_dataset.csv"

FEATURE_COLS = [
    "mean_dwell_time",
    "std_dwell_time",
    "mean_flight_time",
    "typing_speed_wpm",
    "error_rate",
    "mean_velocity",
    "std_velocity",
    "mean_acceleration",
    "mean_curvature",
    "click_frequency",
    "idle_ratio",
    "app_switch_frequency",
    "active_app_entropy",
    "total_focus_duration",
    "session_duration_min",
    "geolocation_deviation",
    "wifi_ssid_match",
    "device_fingerprint_match",
    "face_liveness_score",
]


def load_base_models():
    with open(MODELS_DIR / "if_model.pkl", "rb") as f:
        if_model = pickle.load(f)
    with open(MODELS_DIR / "if_scaler.pkl", "rb") as f:
        if_scaler = pickle.load(f)
    with open(MODELS_DIR / "ae_model.pkl", "rb") as f:
        ae_model = pickle.load(f)
    with open(MODELS_DIR / "ae_scaler.pkl", "rb") as f:
        ae_scaler = pickle.load(f)
    with open(MODELS_DIR / "ae_threshold.pkl", "rb") as f:
        ae_threshold = pickle.load(f)

    composite_iso = None
    if (MODELS_DIR / "composite_iso.pkl").exists():
        with open(MODELS_DIR / "composite_iso.pkl", "rb") as f:
            composite_iso = pickle.load(f)

    return if_model, if_scaler, ae_model, ae_scaler, ae_threshold, composite_iso


def compute_if_ae_scores(df, base_models, weight_if=0.6):
    if_model, if_scaler, ae_model, ae_scaler, ae_threshold, composite_iso = base_models
    x = df[FEATURE_COLS].values

    x_if = if_scaler.transform(x)
    if_raw = if_model.decision_function(x_if)
    if_score = np.clip((0.5 - if_raw) * 100, 0, 100) / 100.0

    x_ae = ae_scaler.transform(x)
    ae_rec = ae_model.predict(x_ae)
    ae_err = np.mean((x_ae - ae_rec) ** 2, axis=1)
    ae_score = np.clip((ae_err / ae_threshold) * 50, 0, 100) / 100.0

    composite_raw = (if_score * weight_if) + (ae_score * (1 - weight_if))
    if composite_iso is not None:
        composite_cal = composite_iso.transform(composite_raw)
    else:
        composite_cal = composite_raw

    return if_score, ae_score, composite_raw, composite_cal


def threshold_for_target_fpr(y_true, scores, target_fpr):
    fpr, _tpr, thresholds = roc_curve(y_true, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(np.max(scores) + 1e-9)
    return float(thresholds[valid[-1]])


def metrics_at_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    auc = roc_auc_score(y_true, scores)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(threshold),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def split_temporal(df):
    n = len(df)
    i_train = int(n * 0.60)
    i_val = int(n * 0.80)
    train = df.iloc[:i_train].copy()
    val = df.iloc[i_train:i_val].copy()
    test = df.iloc[i_val:].copy()
    return train, val, test


def build_stacking_features(df, if_score, ae_score, composite_raw):
    x_raw = df[FEATURE_COLS].values
    return np.column_stack([x_raw, if_score, ae_score, composite_raw])


def main():
    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    y_all = np.where(df["label"] == "normal", 0, 1)

    train, val, test = split_temporal(df)
    y_train = np.where(train["label"] == "normal", 0, 1)
    y_val = np.where(val["label"] == "normal", 0, 1)
    y_test = np.where(test["label"] == "normal", 0, 1)

    print(f"Split sizes -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Anomaly rates -> train: {y_train.mean():.3f}, val: {y_val.mean():.3f}, test: {y_test.mean():.3f}")

    base_models = load_base_models()

    if_tr, ae_tr, comp_raw_tr, comp_cal_tr = compute_if_ae_scores(train, base_models)
    if_va, ae_va, comp_raw_va, comp_cal_va = compute_if_ae_scores(val, base_models)
    if_te, ae_te, comp_raw_te, comp_cal_te = compute_if_ae_scores(test, base_models)

    # Baseline threshold chosen on validation only
    baseline_thr_005 = threshold_for_target_fpr(y_val, comp_cal_va, target_fpr=0.05)
    baseline_test = metrics_at_threshold(y_test, comp_cal_te, baseline_thr_005)

    # Train supervised stacker
    x_train = build_stacking_features(train, if_tr, ae_tr, comp_raw_tr)
    x_val = build_stacking_features(val, if_va, ae_va, comp_raw_va)
    x_test = build_stacking_features(test, if_te, ae_te, comp_raw_te)

    stacker = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    stacker.fit(x_train, y_train)

    stacker_val_prob_raw = stacker.predict_proba(x_val)[:, 1]
    stacker_test_prob_raw = stacker.predict_proba(x_test)[:, 1]

    # Calibration fit on validation only
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(stacker_val_prob_raw, y_val)
    stacker_val_prob = iso.transform(stacker_val_prob_raw)
    stacker_test_prob = iso.transform(stacker_test_prob_raw)

    # Thresholds selected on validation only
    thr_fpr_001 = threshold_for_target_fpr(y_val, stacker_val_prob, target_fpr=0.01)
    thr_fpr_002 = threshold_for_target_fpr(y_val, stacker_val_prob, target_fpr=0.02)
    thr_fpr_005 = threshold_for_target_fpr(y_val, stacker_val_prob, target_fpr=0.05)

    # Tier thresholds (high risk, medium risk)
    # high: very low FPR boundary; medium: operational boundary
    tier_high = thr_fpr_001
    tier_medium = thr_fpr_005

    upgraded_test_005 = metrics_at_threshold(y_test, stacker_test_prob, thr_fpr_005)
    upgraded_test_002 = metrics_at_threshold(y_test, stacker_test_prob, thr_fpr_002)
    upgraded_test_001 = metrics_at_threshold(y_test, stacker_test_prob, thr_fpr_001)

    # Tier distribution on test
    high_mask = stacker_test_prob >= tier_high
    med_mask = (stacker_test_prob >= tier_medium) & (stacker_test_prob < tier_high)
    low_mask = stacker_test_prob < tier_medium

    tier_counts = {
        "high": int(high_mask.sum()),
        "medium": int(med_mask.sum()),
        "low": int(low_mask.sum()),
    }

    # Save artifacts
    with open(MODELS_DIR / "supervised_stacker.pkl", "wb") as f:
        pickle.dump(stacker, f)
    with open(MODELS_DIR / "supervised_stacker_iso.pkl", "wb") as f:
        pickle.dump(iso, f)

    config = {
        "model": "random_forest_stacker",
        "features": FEATURE_COLS + ["if_score", "ae_score", "composite_raw"],
        "validation_thresholds": {
            "fpr_0.01": thr_fpr_001,
            "fpr_0.02": thr_fpr_002,
            "fpr_0.05": thr_fpr_005,
        },
        "tier_thresholds": {
            "high": tier_high,
            "medium": tier_medium,
        },
        "recommended_operating_threshold": thr_fpr_005,
        "notes": "All thresholds selected on validation split only.",
    }
    with open(MODELS_DIR / "supervised_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics summary
    metrics = {
        "split": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_time_range": [str(train["timestamp"].min()), str(train["timestamp"].max())],
            "val_time_range": [str(val["timestamp"].min()), str(val["timestamp"].max())],
            "test_time_range": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        },
        "baseline_calibrated_composite_test_at_val_fpr_0.05": baseline_test,
        "upgraded_stacker_test": {
            "at_val_fpr_0.05": upgraded_test_005,
            "at_val_fpr_0.02": upgraded_test_002,
            "at_val_fpr_0.01": upgraded_test_001,
        },
        "tier_counts_test": tier_counts,
    }
    with open(MODELS_DIR / "supervised_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Baseline (calibrated composite) on test @ val-selected FPR<=0.05 threshold ===")
    print(baseline_test)

    print("\n=== Upgraded supervised stacker on test ===")
    print("@FPR<=0.05 threshold:", upgraded_test_005)
    print("@FPR<=0.02 threshold:", upgraded_test_002)
    print("@FPR<=0.01 threshold:", upgraded_test_001)

    print("\n=== Tiering (test counts) ===")
    print(tier_counts)

    print("\nSaved artifacts:")
    print("-", MODELS_DIR / "supervised_stacker.pkl")
    print("-", MODELS_DIR / "supervised_stacker_iso.pkl")
    print("-", MODELS_DIR / "supervised_config.json")
    print("-", MODELS_DIR / "supervised_metrics.json")


if __name__ == "__main__":
    main()
