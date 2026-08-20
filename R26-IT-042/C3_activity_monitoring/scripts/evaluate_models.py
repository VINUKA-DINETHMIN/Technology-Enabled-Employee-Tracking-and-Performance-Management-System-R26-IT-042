"""
Evaluate trained models on temporal test split and print Precision, Recall and AUC.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
DATA_PATH = ROOT / 'data' / 'employee_monitoring_dataset.csv'
FEATURE_COLS = [
    'mean_dwell_time', 'std_dwell_time', 'mean_flight_time',
    'typing_speed_wpm', 'error_rate',
    'mean_velocity', 'std_velocity', 'mean_acceleration',
    'mean_curvature', 'click_frequency', 'idle_ratio',
    'app_switch_frequency', 'active_app_entropy', 'total_focus_duration',
    'session_duration_min', 'geolocation_deviation',
    'wifi_ssid_match', 'device_fingerprint_match', 'face_liveness_score'
]

def load_models():
    with open(MODELS_DIR / 'if_model.pkl','rb') as f:
        if_model = pickle.load(f)
    with open(MODELS_DIR / 'if_scaler.pkl','rb') as f:
        if_scaler = pickle.load(f)
    with open(MODELS_DIR / 'ae_model.pkl','rb') as f:
        ae_model = pickle.load(f)
    with open(MODELS_DIR / 'ae_scaler.pkl','rb') as f:
        ae_scaler = pickle.load(f)
    ae_threshold = None
    try:
        with open(MODELS_DIR / 'ae_threshold.pkl','rb') as f:
            ae_threshold = pickle.load(f)
    except Exception:
        ae_threshold = None
    iso = None
    try:
        with open(MODELS_DIR / 'composite_iso.pkl','rb') as f:
            iso = pickle.load(f)
    except Exception:
        iso = None
    meta = None
    try:
        with open(MODELS_DIR / 'meta_lr.pkl','rb') as f:
            meta = pickle.load(f)
    except Exception:
        meta = None
    return if_model, if_scaler, ae_model, ae_scaler, ae_threshold, iso, meta


def score_models(df, models, weight_if=0.6):
    if_model, if_scaler, ae_model, ae_scaler, ae_threshold, iso, meta = models
    X = df[FEATURE_COLS].values
    X_if = if_scaler.transform(X)
    raw_if = if_model.decision_function(X_if)
    if_risk = np.clip((0.5 - raw_if) * 100, 0, 100) / 100.0

    X_ae = ae_scaler.transform(X)
    rec = ae_model.predict(X_ae)
    ae_err = np.mean((X_ae - rec)**2, axis=1)
    ae_risk = np.clip((ae_err / ae_threshold) * 50, 0, 100) / 100.0 if ae_threshold is not None else (ae_err - ae_err.min())/(ae_err.max()-ae_err.min()+1e-9)

    composite = (if_risk * weight_if) + (ae_risk * (1-weight_if))
    calibrated = iso.transform(composite) if iso is not None else composite
    meta_prob = meta.predict_proba(np.vstack([if_risk, ae_risk]).T)[:,1] if meta is not None else None

    return if_risk, ae_risk, composite, calibrated, meta_prob


def precision_recall_at_best_f1(y_true, scores):
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * (p * r) / (p + r + 1e-12)
    idx = np.nanargmax(f1)
    return float(p[idx]), float(r[idx]), float(thr[idx])


def precision_recall_at_threshold(y_true, scores, threshold):
    preds = (scores >= threshold).astype(int)
    return float(precision_score(y_true, preds)), float(recall_score(y_true, preds))


def auc_score(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return None


def main():
    print('Loading data:', DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    split_date = df['timestamp'].median()
    train = df[df['timestamp'] <= split_date]
    test = df[df['timestamp'] > split_date]
    print('Train rows:', len(train), 'Test rows:', len(test))

    models = load_models()
    y_test = np.where(test['label']=='normal', 0, 1)

    if_risk, ae_risk, composite_raw, composite_cal, meta_prob = score_models(test, models, weight_if=0.6)

    # AUCs
    auc_if = auc_score(y_test, if_risk)
    auc_ae = auc_score(y_test, ae_risk)
    auc_comp = auc_score(y_test, composite_cal)
    auc_meta = auc_score(y_test, meta_prob) if meta_prob is not None else None

    # Best-F1 thresholds
    p_if, r_if, t_if = precision_recall_at_best_f1(y_test, if_risk)
    p_ae, r_ae, t_ae = precision_recall_at_best_f1(y_test, ae_risk)
    p_comp, r_comp, t_comp = precision_recall_at_best_f1(y_test, composite_cal)
    if meta_prob is not None:
        p_meta, r_meta, t_meta = precision_recall_at_best_f1(y_test, meta_prob)
    else:
        p_meta = r_meta = t_meta = None

    # Precision/recall at recommended thresholds
    rec_comp_threshold = 0.22
    p_comp_rec, r_comp_rec = precision_recall_at_threshold(y_test, composite_cal, rec_comp_threshold)

    print('\n=== AUC scores ===')
    print(f'IF AUC: {auc_if:.4f}')
    print(f'AE AUC: {auc_ae:.4f}')
    print(f'Calibrated Composite AUC: {auc_comp:.4f}')
    if auc_meta is not None:
        print(f'Meta AUC: {auc_meta:.4f}')

    print('\n=== Best-F1 (test) results ===')
    print(f'IF  best-F1 prec={p_if:.4f} recall={r_if:.4f} thr={t_if:.6f}')
    print(f'AE  best-F1 prec={p_ae:.4f} recall={r_ae:.4f} thr={t_ae:.6f}')
    print(f'Calibrated Composite best-F1 prec={p_comp:.4f} recall={r_comp:.4f} thr={t_comp:.6f}')
    if p_meta is not None:
        print(f'Meta best-F1 prec={p_meta:.4f} recall={r_meta:.4f} thr={t_meta:.6f}')

    print('\n=== Recommended operating point ===')
    print(f'Calibrated Composite @ threshold={rec_comp_threshold}: precision={p_comp_rec:.4f}, recall={r_comp_rec:.4f}')

if __name__ == '__main__':
    main()
