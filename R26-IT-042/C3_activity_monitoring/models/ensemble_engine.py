import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

try:
    import pickle
    from sklearn.preprocessing import StandardScaler
except Exception:
    # allow import-time failures to be raised later when functions are used
    pickle = None

FEATURE_COLS = [
    'mean_dwell_time', 'std_dwell_time', 'mean_flight_time',
    'typing_speed_wpm', 'error_rate',
    'mean_velocity', 'std_velocity', 'mean_acceleration',
    'mean_curvature', 'click_frequency', 'idle_ratio',
    'app_switch_frequency', 'active_app_entropy', 'total_focus_duration',
    'session_duration_min', 'geolocation_deviation',
    'wifi_ssid_match', 'device_fingerprint_match', 'face_liveness_score'
]


def _p(p: str) -> Path:
    return Path(p)


def _first_existing(base: Path, *names: str) -> Path:
    for name in names:
        candidate = base.joinpath(name)
        if candidate.exists():
            return candidate
    return base.joinpath(names[0])


def load_engine(models_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load models and config from models_dir (defaults to this file's parent).

    Returns a dict with keys: if_model, ae_model, scaler (optional), config
    """
    if pickle is None:
        raise RuntimeError("pickle/sklearn not available in this environment")

    base = Path(models_dir) if models_dir else Path(__file__).resolve().parent

    def _lp(name):
        return base.joinpath(name)

    # required files, with backward-compatible fallbacks for the live app names
    if_path = _first_existing(base, 'if_model.pkl', 'user_behavioral_model.pkl')
    ae_path = _first_existing(base, 'ae_model.pkl')
    cfg_path = _lp('ensemble_config.json')

    if not if_path.exists() or not ae_path.exists():
        raise FileNotFoundError(f"Missing model files in {base}. Expected one of: if_model.pkl/user_behavioral_model.pkl and ae_model.pkl")

    with open(if_path, 'rb') as f:
        if_model = pickle.load(f)
    with open(ae_path, 'rb') as f:
        ae_model = pickle.load(f)

    scaler = None
    scaler_path = _first_existing(base, 'if_scaler.pkl', 'feature_scaler.pkl')
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None

    config = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, 'r') as f:
                config = json.load(f)
        except Exception:
            config = {}

    # attempt to load a persisted meta-classifier if present
    meta_clf = None
    meta_path = base.joinpath('meta_lr.pkl')
    if meta_path.exists():
        try:
            with open(meta_path, 'rb') as f:
                meta_clf = pickle.load(f)
        except Exception:
            meta_clf = None

    return {
        'if_model': if_model,
        'ae_model': ae_model,
        'scaler': scaler,
        'config': config,
        'models_dir': str(base),
        'meta_clf': meta_clf,
    }


def load_named_config(models_dir: Optional[str], name: str) -> Dict[str, Any]:
    """Load a named config: 'default' uses ensemble_config.json, 'constrained' uses ensemble_constrained.json"""
    base = Path(models_dir) if models_dir else Path(__file__).resolve().parent
    cfg = {}
    if name == 'constrained':
        p = base.joinpath('ensemble_constrained.json')
        if p.exists():
            with open(p, 'r') as f:
                cfg = json.load(f)
            # constrained file structure differs; map into expected fields
            # move val_best -> best_weight/best_threshold
            vb = cfg.get('val_best', {})
            mapped = {
                'best_weight': vb.get('weight'),
                'best_threshold': vb.get('threshold'),
                'constrained': True,
                'constrained_meta': cfg
            }
            return mapped
        else:
            raise FileNotFoundError(str(p))
    else:
        # default/tuned
        p = base.joinpath('ensemble_config.json')
        if p.exists():
            with open(p, 'r') as f:
                cfg = json.load(f)
        return cfg


def _if_score(if_model, Xs: np.ndarray) -> np.ndarray:
    raw = if_model.decision_function(Xs)
    score = np.clip((0.5 - raw) * 100, 0, 100) / 100.0
    return score


def _ae_score(ae_model, Xs: np.ndarray, mse_train_min: Optional[float] = None, mse_train_max: Optional[float] = None) -> np.ndarray:
    recon = ae_model.predict(Xs)
    mse = np.mean((Xs - recon) ** 2, axis=1)
    if mse_train_min is None or mse_train_max is None or mse_train_max - mse_train_min <= 0:
        # fallback normalization: scale by 99th percentile of the batch
        p99 = np.percentile(mse, 99) if len(mse) > 0 else 1.0
        denom = p99 if p99 > 0 else 1.0
        ae_score = np.clip(mse / denom, 0, 1)
    else:
        ae_score = (mse - mse_train_min) / (mse_train_max - mse_train_min)
        ae_score = np.clip(ae_score, 0, 1)
    return ae_score


def score_batch(X: np.ndarray, engine: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Score a batch of raw feature vectors (shape: n_samples x n_features).

    X should be in the order of FEATURE_COLS. If a scaler is available it will be applied.
    Returns a dict: if_score, ae_score, ensemble_score, ensemble_pred, meta_prob (if available)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != len(FEATURE_COLS):
        raise ValueError(f"Expected {len(FEATURE_COLS)} features in order {FEATURE_COLS}")

    if_model = engine['if_model']
    ae_model = engine['ae_model']
    scaler = engine.get('scaler')
    cfg = engine.get('config', {}) or {}

    if scaler is None:
        raise RuntimeError('Scaler not available in models directory. Please provide a fitted scaler as if_scaler.pkl or pass pre-scaled features.')

    Xs = scaler.transform(X)

    if_score = _if_score(if_model, Xs)

    # if config contains saved train mse bounds, use them
    mse_min = cfg.get('ae_mse_min', None)
    mse_max = cfg.get('ae_mse_max', None)
    ae_score = _ae_score(ae_model, Xs, mse_min, mse_max)

    # ensemble
    w = float(cfg.get('best_weight', 0.5))
    t = float(cfg.get('best_threshold', 0.5))
    ensemble_score = w * if_score + (1.0 - w) * ae_score
    ensemble_pred = (ensemble_score >= t).astype(int)

    meta_prob = None
    meta_clf = engine.get('meta_clf')
    if meta_clf is not None:
        # use persisted sklearn classifier
        meta_X = np.vstack([if_score, ae_score]).T
        try:
            probs = meta_clf.predict_proba(meta_X)[:, 1]
            meta_prob = probs
            meta_pred = (probs >= 0.5).astype(int)
        except Exception:
            meta_prob = None
            meta_pred = None
    elif 'meta_lr' in cfg and cfg['meta_lr'].get('coef'):
        # fallback: simple logistic model application from stored coefficients
        coef = np.array(cfg['meta_lr']['coef']).reshape(-1)
        intercept = float(cfg['meta_lr'].get('intercept', [0.0])[0])
        # features order: [if_score, ae_score]
        meta_X = np.vstack([if_score, ae_score]).T
        logits = meta_X.dot(coef) + intercept
        probs = 1 / (1 + np.exp(-logits))
        meta_prob = probs
        meta_pred = (probs >= 0.5).astype(int)
    else:
        meta_pred = None

    return {
        'if_score': if_score,
        'ae_score': ae_score,
        'ensemble_score': ensemble_score,
        'ensemble_pred': ensemble_pred,
        'meta_prob': meta_prob,
        'meta_pred': meta_pred
    }


def predict_from_dict(record: Dict[str, Any], engine: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience: accept a dict or pandas Series of feature values and return scores/pred"""
    import numpy as _np
    arr = _np.array([record.get(c, None) for c in FEATURE_COLS], dtype=float)
    out = score_batch(arr, engine)
    # return scalars if single sample
    for k, v in list(out.items()):
        if isinstance(v, _np.ndarray) and v.size == 1:
            out[k] = v.item()
    return out


if __name__ == '__main__':
    # small CLI smoke-test: try to load models and score a few rows from dataset if available
    import argparse
    import pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument('--models-dir', default=str(Path(__file__).resolve().parent))
    p.add_argument('--config', choices=['default', 'constrained', 'tuned'], default='default')
    args = p.parse_args()

    print('Loading ensemble engine from models dir:', args.models_dir)
    # load models
    engine = load_engine(models_dir=args.models_dir)
    # override config with named
    try:
        named_cfg = load_named_config(args.models_dir, args.config)
        engine['config'] = named_cfg
        print(f"Using named config: {args.config}")
    except FileNotFoundError:
        print(f"Named config {args.config} not found; falling back to embedded config in ensemble_config.json")
    print('Loaded models; trying a small smoke test if dataset is available nearby...')
    # try to find a dataset in known relative locations
    candidates = [
        Path(__file__).resolve().parent.parent.joinpath('data/employee_monitoring_dataset.csv'),
        Path(__file__).resolve().parent.joinpath('..:/data/employee_monitoring_dataset.csv'),
        Path(__file__).resolve().parent.joinpath('../data/employee_monitoring_dataset.csv')
    ]
    ds = None
    for c in candidates:
        try:
            if c.exists():
                ds = str(c)
                break
        except Exception:
            continue
    if ds is None:
        print('No dataset found for smoke test. Use engine programmatically with pre-scaled data or provide if_scaler.pkl in models dir.')
    else:
        print('Using dataset for smoke test:', ds)
        df = pd.read_csv(ds)
        sample = df[FEATURE_COLS].head(3)
        engine_local = engine
        res = score_batch(sample.values, engine_local)
        print('Smoke test results (first rows):')
        for k, v in res.items():
            print(f'  {k}:', getattr(v, 'shape', v) if isinstance(v, np.ndarray) else v)
