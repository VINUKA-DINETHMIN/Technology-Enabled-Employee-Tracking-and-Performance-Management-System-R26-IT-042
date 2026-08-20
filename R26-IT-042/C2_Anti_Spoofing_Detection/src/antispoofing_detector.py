"""
R26-IT-042 — C2: Anti-Spoofing Detection
C2_Anti_Spoofing_Detection/src/antispoofing_detector.py

ResNet50-based anti-spoofing detector to verify if a face is real (not a photo/video replay).
Trained on: real selfies, live videos vs. printouts, cut-outs, replay videos.

This module provides anti-spoofing detection to complement the existing 
MediaPipe liveness detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_PATH = _MODELS_DIR / "best_anti_spoofing_model.keras"

# Anti-spoofing thresholds
_REAL_THRESHOLD = 0.5  # Score >= 0.5 = Real, < 0.5 = Fake
_CONFIDENCE_THRESHOLD = 0.85  # Require >= 85% confidence for decision


class AntiSpoofingDetector:
    """
    ResNet50-based anti-spoofing detector.
    Detects whether a face is real (liveness) or fake (photo/video replay).

    Attributes:
        - Binary classification: 0 = Real, 1 = Fake
        - Input: 96x96 RGB image
        - Output: Score 0.0–1.0 (higher = more likely fake)
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Initialize the anti-spoofing detector.

        Parameters
        ----------
        model_path:
            Path to the .keras model file. Defaults to models/best_anti_spoofing_model.keras
        """
        self._model = None
        self._model_loaded = False
        self._model_path = model_path or _MODEL_PATH

    def load_model(self) -> bool:
        """
        Load the anti-spoofing model from disk.

        Returns
        -------
        bool
            True if model loaded successfully, False otherwise.
        """
        if self._model_loaded:
            return True

        if not self._model_path.exists():
            logger.warning(
                "Anti-spoofing model not found at %s. "
                "Please place 'best_anti_spoofing_model.keras' in the models folder.",
                self._model_path,
            )
            return False

        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(str(self._model_path))
            self._model_loaded = True
            logger.info("Anti-spoofing model loaded from %s", self._model_path)
            return True

        except ImportError:
            logger.warning("TensorFlow not installed — anti-spoofing detection disabled.")
            return False
        except Exception as exc:
            logger.error("Failed to load anti-spoofing model: %s", exc)
            return False

    def preprocess_frame(self, frame: np.ndarray, target_size: int = 96) -> Optional[np.ndarray]:
        """
        Preprocess a frame for the anti-spoofing model.

        Parameters
        ----------
        frame:
            OpenCV frame (BGR format, H×W×3).
        target_size:
            Target image size (default 96x96).

        Returns
        -------
        np.ndarray | None
            Preprocessed frame ready for model inference, or None if frame is invalid.
        """
        try:
            if frame is None or frame.size == 0:
                return None

            # Resize to target size
            resized = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0

            # Add batch dimension
            batch = np.expand_dims(normalized, axis=0)

            return batch
        except Exception as exc:
            logger.debug("Frame preprocessing error: %s", exc)
            return None

    def predict(self, frame: np.ndarray) -> Tuple[bool, float, str]:
        """
        Predict if a face in the frame is real or fake.

        Parameters
        ----------
        frame:
            OpenCV frame (BGR format, H×W×3).

        Returns
        -------
        tuple[bool, float, str]
            (is_real, confidence, decision_reason)
            - is_real: True if face is determined to be real
            - confidence: Model confidence (0.0–1.0)
            - decision_reason: Human-readable explanation
        """
        if not self._model_loaded:
            logger.warning("Anti-spoofing model not loaded.")
            return True, 0.0, "Model not available; defaulting to real"

        try:
            # Preprocess
            batch = self.preprocess_frame(frame)
            if batch is None:
                return True, 0.0, "Invalid frame"

            # Predict
            score = float(self._model.predict(batch, verbose=0)[0][0])

            # Interpret
            # score close to 0.0 = Real, score close to 1.0 = Fake
            confidence = max(score, 1.0 - score)
            is_real = score < _REAL_THRESHOLD

            if confidence < _CONFIDENCE_THRESHOLD:
                reason = f"Low confidence ({confidence:.2f}); inconclusive"
            elif is_real:
                reason = f"Real face detected (score: {score:.3f})"
            else:
                reason = f"Fake face detected (score: {score:.3f})"

            return is_real, confidence, reason

        except Exception as exc:
            logger.error("Prediction error: %s", exc)
            return True, 0.0, f"Prediction error: {exc}"

    def predict_from_camera(
        self, timeout_sec: float = 10.0, windows: int = 5
    ) -> Tuple[bool, float, str]:
        """
        Perform anti-spoofing check using camera frames.

        Parameters
        ----------
        timeout_sec:
            Maximum seconds to wait.
        windows:
            Number of frames to average predictions over.

        Returns
        -------
        tuple[bool, float, str]
            (is_real, avg_confidence, reason)
        """
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.warning("Camera not available for anti-spoofing check.")
                return True, 0.0, "Camera unavailable"

            predictions = []
            start = __import__("time").time()

            while (__import__("time").time() - start) < timeout_sec and len(predictions) < windows:
                ret, frame = cap.read()
                if not ret:
                    break

                is_real, confidence, _ = self.predict(frame)
                predictions.append((is_real, confidence))

            cap.release()

            if not predictions:
                return True, 0.0, "No frames captured"

            # Average predictions
            avg_is_real = sum(p[0] for p in predictions) / len(predictions) >= 0.5
            avg_confidence = sum(p[1] for p in predictions) / len(predictions)

            reason = f"Anti-spoofing: {'Real' if avg_is_real else 'Fake'} (avg conf: {avg_confidence:.2f})"
            return avg_is_real, avg_confidence, reason

        except Exception as exc:
            logger.error("Camera anti-spoofing check error: %s", exc)
            return True, 0.0, f"Error: {exc}"

    def close(self) -> None:
        """Clean up resources."""
        self._model = None
        self._model_loaded = False
