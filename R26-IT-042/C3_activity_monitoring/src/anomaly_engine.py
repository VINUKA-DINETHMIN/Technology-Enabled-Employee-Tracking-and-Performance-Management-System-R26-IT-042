"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/anomaly_engine.py

Loads the trained IsolationForest model and scaler from models/ and
scores incoming feature vectors.  Returns a risk_score (0–100).

Model file locations
────────────────────
  C3_activity_monitoring/models/user_behavioral_model.pkl
  C3_activity_monitoring/models/feature_scaler.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODELS_DIR / "user_behavioral_model.pkl"
_SCALER_PATH = _MODELS_DIR / "feature_scaler.pkl"
_AE_MODEL_PATH = _MODELS_DIR / "ae_model.pkl"
_AE_SCALER_PATH = _MODELS_DIR / "ae_scaler.pkl"
_AE_THRESHOLD_PATH = _MODELS_DIR / "ae_threshold.pkl"
_COMPOSITE_ISO_PATH = _MODELS_DIR / "composite_iso.pkl"
_META_LR_PATH = _MODELS_DIR / "meta_lr.pkl"
_ENSEMBLE_CONFIG_PATH = _MODELS_DIR / "ensemble_config.json"

_IF_WEIGHT = 0.6
_AE_WEIGHT = 0.4

# Canonical model feature order used during training and runtime inference.
FEATURE_COLUMNS = [
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


class AnomalyEngine:
    """
    Wraps a trained sklearn IsolationForest to produce risk scores.

    Usage
    ─────
    >>> engine = AnomalyEngine()
    >>> engine.load_model()
    >>> score = engine.score(feature_vector.to_array())
    """

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._ae_model = None
        self._ae_scaler = None
        self._ae_threshold = None
        self._composite_iso = None
        self._meta_lr = None
        self._if_weight = _IF_WEIGHT
        self._ae_weight = _AE_WEIGHT
        self._ensemble_threshold = None
        self._model_loaded = False
        self._last_load_error: Optional[str] = None

    def _align_feature_length(self, x: np.ndarray, expected: int) -> np.ndarray:
        """Adapt feature vector length to what the loaded scaler/model expects."""
        current = int(x.shape[1])
        if current == expected:
            return x

        # Backward compatibility path: previous runtime sent 14 features.
        # Map to 19 by appending session/context defaults.
        if current == 14 and expected == 19:
            pad = np.array([[0.0, 0.0, 1.0, 1.0, 0.0]], dtype=np.float32)
            logger.warning(
                "AnomalyEngine received legacy 14-feature input; auto-expanding to 19 features."
            )
            return np.hstack([x.astype(np.float32), pad])

        # Generic fallback: truncate or zero-pad to avoid hard failure.
        logger.warning(
            "AnomalyEngine feature length mismatch (got=%d, expected=%d). Applying fallback alignment.",
            current,
            expected,
        )
        if current > expected:
            return x[:, :expected]
        pad = np.zeros((x.shape[0], expected - current), dtype=np.float32)
        return np.hstack([x.astype(np.float32), pad])

    def load_model(self) -> bool:
        """
        Load model and scaler from disk.

        Returns
        -------
        bool
            True if both files loaded successfully.
        """
        # Reset optional artifacts for a clean reload attempt.
        self._model = None
        self._scaler = None
        self._ae_model = None
        self._ae_scaler = None
        self._ae_threshold = None
        self._composite_iso = None
        self._meta_lr = None
        self._model_loaded = False
        self._last_load_error = None

        if not _MODEL_PATH.exists():
            self._last_load_error = f"model_file_not_found:{_MODEL_PATH}"
            logger.warning("Model file not found: %s — anomaly scoring disabled.", _MODEL_PATH)
            return False

        # Primary model is mandatory.
        try:
            with open(_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            logger.info("IsolationForest model loaded from %s", _MODEL_PATH)
        except Exception as exc:
            self._last_load_error = f"primary_model_load_failed:{type(exc).__name__}:{exc}"
            logger.error("Failed to load primary anomaly model from %s: %s", _MODEL_PATH, exc)
            return False

        # Scaler is helpful but not strictly required for runtime continuity.
        if _SCALER_PATH.exists():
            try:
                with open(_SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info("Feature scaler loaded from %s", _SCALER_PATH)
            except Exception as exc:
                logger.warning("Could not load feature scaler (%s): %s", _SCALER_PATH, exc)

        # Optional ensemble artifacts should never disable the primary IF model.
        if _AE_MODEL_PATH.exists():
            try:
                with open(_AE_MODEL_PATH, "rb") as f:
                    self._ae_model = pickle.load(f)
                logger.info("Autoencoder model loaded from %s", _AE_MODEL_PATH)
            except Exception as exc:
                logger.warning("Could not load autoencoder model (%s): %s", _AE_MODEL_PATH, exc)
                self._ae_model = None

        ae_scaler_path = _AE_SCALER_PATH if _AE_SCALER_PATH.exists() else _SCALER_PATH
        if ae_scaler_path.exists():
            try:
                with open(ae_scaler_path, "rb") as f:
                    self._ae_scaler = pickle.load(f)
                logger.info("Autoencoder scaler loaded from %s", ae_scaler_path)
            except Exception as exc:
                logger.warning("Could not load autoencoder scaler (%s): %s", ae_scaler_path, exc)
                self._ae_scaler = None

        if _AE_THRESHOLD_PATH.exists():
            try:
                with open(_AE_THRESHOLD_PATH, "rb") as f:
                    self._ae_threshold = pickle.load(f)
                logger.info("Autoencoder threshold loaded from %s", _AE_THRESHOLD_PATH)
            except Exception as exc:
                logger.warning("Could not load autoencoder threshold (%s): %s", _AE_THRESHOLD_PATH, exc)
                self._ae_threshold = None

        if _ENSEMBLE_CONFIG_PATH.exists():
            try:
                import json
                with open(_ENSEMBLE_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self._if_weight = float(cfg.get("weight", self._if_weight))
                self._ae_weight = max(0.0, 1.0 - self._if_weight)
                self._ensemble_threshold = float(cfg.get("threshold", 50.0))
                logger.info("Ensemble config loaded from %s", _ENSEMBLE_CONFIG_PATH)
            except Exception as exc:
                logger.warning("Could not load ensemble config: %s", exc)

        if _COMPOSITE_ISO_PATH.exists():
            try:
                with open(_COMPOSITE_ISO_PATH, "rb") as f:
                    self._composite_iso = pickle.load(f)
                logger.info("Composite isotonic calibrator loaded from %s", _COMPOSITE_ISO_PATH)
            except Exception as exc:
                logger.warning("Could not load composite calibrator (%s): %s", _COMPOSITE_ISO_PATH, exc)
                self._composite_iso = None

        if _META_LR_PATH.exists():
            try:
                with open(_META_LR_PATH, "rb") as f:
                    self._meta_lr = pickle.load(f)
                logger.info("Meta logistic calibrator loaded from %s", _META_LR_PATH)
            except Exception as exc:
                logger.warning("Could not load meta calibrator (%s): %s", _META_LR_PATH, exc)
                self._meta_lr = None

        self._model_loaded = True
        self._last_load_error = None
        return True

    def score(self, features: np.ndarray) -> float:
        """
        Compute a risk score (0–100) from a feature vector.

        The IsolationForest returns -1 for anomalies, +1 for normal.
        We convert its raw decision function to a 0–100 scale where
        higher = more anomalous.

        Parameters
        ----------
        features:
            1-D numpy array from FeatureVector.to_array().

        Returns
        -------
        float
            Risk score in range [0, 100].
        """
        if not self._model_loaded or self._model is None:
            # Without a model, return 0 (no risk assumed)
            return 0.0

        try:
            x = features.reshape(1, -1)

            expected_features = None
            if self._scaler is not None:
                expected_features = getattr(self._scaler, "n_features_in_", None)
            if expected_features is None:
                expected_features = getattr(self._model, "n_features_in_", None)
            if expected_features is not None:
                x = self._align_feature_length(x, int(expected_features))

            if self._scaler is not None:
                x = self._scaler.transform(x)

            # decision_function: more negative = more anomalous
            raw_score = self._model.decision_function(x)[0]
            # Map to [0, 100]: raw typically in [-0.5, 0.5]
            if_risk = max(0.0, min(100.0, (0.5 - raw_score) * 100.0))

            ae_risk = None
            if self._ae_model is not None:
                x_ae = features.reshape(1, -1)
                if expected_features is not None:
                    x_ae = self._align_feature_length(x_ae, int(expected_features))
                if self._ae_scaler is not None:
                    x_ae = self._ae_scaler.transform(x_ae)
                recon = self._ae_model.predict(x_ae)
                mae = float(np.mean(np.abs(x_ae - recon)))
                threshold = float(self._ae_threshold or 0.0)
                if threshold > 0:
                    ae_risk = max(0.0, min(100.0, (mae / threshold) * 50.0))
                else:
                    ae_risk = max(0.0, min(100.0, mae * 100.0))

            if ae_risk is None:
                base_risk = if_risk
            else:
                base_risk = (self._if_weight * if_risk) + (self._ae_weight * ae_risk)

            base_risk = max(0.0, min(100.0, base_risk))

            calibrated_risk = None
            if self._meta_lr is not None:
                try:
                    meta_input = np.array([
                        [
                            max(0.0, min(1.0, if_risk / 100.0)),
                            max(0.0, min(1.0, (ae_risk if ae_risk is not None else if_risk) / 100.0)),
                        ]
                    ], dtype=np.float64)
                    meta_prob = float(self._meta_lr.predict_proba(meta_input)[0, 1])
                    if np.isfinite(meta_prob):
                        calibrated_risk = max(0.0, min(100.0, meta_prob * 100.0))
                except Exception as exc:
                    logger.debug("Meta calibration skipped: %s", exc)

            if calibrated_risk is None and self._composite_iso is not None:
                try:
                    iso_input = np.array([base_risk / 100.0], dtype=np.float64)
                    iso_out = self._composite_iso.predict(iso_input)
                    iso_val = float(np.asarray(iso_out).reshape(-1)[0])
                    if np.isfinite(iso_val):
                        calibrated_risk = max(0.0, min(100.0, iso_val * 100.0 if iso_val <= 1.5 else iso_val))
                except Exception as exc:
                    logger.debug("Composite calibration skipped: %s", exc)

            if calibrated_risk is not None:
                return round(calibrated_risk, 2)

            return round(base_risk, 2)

        except Exception as exc:
            logger.error("Anomaly scoring error: %s", exc)
            return 0.0

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def last_load_error(self) -> Optional[str]:
        return self._last_load_error
