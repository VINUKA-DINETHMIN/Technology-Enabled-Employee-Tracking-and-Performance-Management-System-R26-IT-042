"""
R26-IT-042 — C3: Tests
C3_activity_monitoring/tests/test_macro_detector.py

Unit tests for MacroDetectorEngine.
Verifies human typing vs. synthetic macro typing detection.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from C3_activity_monitoring.src.macro_detector import MacroDetectorEngine


class TestMacroDetectorEngine(unittest.TestCase):

    def setUp(self):
        self.mock_keyboard = MagicMock()
        self.mock_mouse = MagicMock()
        self.mock_keyboard._lock = MagicMock()
        self.mock_mouse._lock = MagicMock()
        self.detector = MacroDetectorEngine(
            keyboard=self.mock_keyboard,
            mouse=self.mock_mouse,
            window_sec=30.0,
        )

    def test_empty_trackers_returns_default_features(self):
        """When no keystrokes or mouse movements are present, default neutral features are returned."""
        self.mock_keyboard._keystrokes = []
        self.mock_mouse._movements = []

        features = self.detector.get_features()

        # When no data is present, neutral defaults (0.5 for entropy/sync/jitter, 0.0 for periodicity etc) sum to ~0.275
        self.assertLess(features["macro_risk_score"], 0.35)
        self.assertEqual(features["keystroke_periodicity_score"], 0.0)

    def test_human_typing_low_macro_risk(self):
        """Simulate realistic variable human typing intervals."""
        now = time.perf_counter()
        # Human typing: variable dwell times (50-120ms) and variable flight intervals (80-350ms)
        import random
        random.seed(42)

        keystrokes = []
        curr = now - 25.0
        for _ in range(40):
            dwell = random.uniform(0.05, 0.12)
            flight = random.uniform(0.08, 0.35)
            keystrokes.append({
                "press_ts": curr,
                "release_ts": curr + dwell,
                "dwell_ms": dwell * 1000.0,
                "is_backspace": False,
            })
            curr += dwell + flight

        self.mock_keyboard._keystrokes = keystrokes

        # Add mouse movements to show sync
        movements = []
        m_curr = now - 25.0
        for _ in range(30):
            movements.append({
                "timestamp": m_curr,
                "velocity": random.uniform(100, 500),
            })
            m_curr += random.uniform(0.2, 0.8)
        self.mock_mouse._movements = movements

        features = self.detector.get_features()

        self.assertLess(features["macro_risk_score"], 0.45)
        self.assertLess(features["keystroke_timing_jitter"], 0.6)

    def test_synthetic_macro_high_macro_risk(self):
        """Simulate machine-perfect synthetic macro typing (e.g. constant 100ms interval, 50ms dwell)."""
        now = time.perf_counter()

        keystrokes = []
        curr = now - 20.0
        # Machine-perfect timing: exactly 0.05s dwell and 0.05s flight (100ms total interval)
        for _ in range(50):
            keystrokes.append({
                "press_ts": curr,
                "release_ts": curr + 0.05,
                "dwell_ms": 50.0,
                "is_backspace": False,
            })
            curr += 0.10  # constant step

        self.mock_keyboard._keystrokes = keystrokes
        self.mock_mouse._movements = []  # zero mouse movement -> desync

        features = self.detector.get_features()

        # Synthetic macro should score high on consistency, jitter, and macro_risk_score
        self.assertGreater(features["macro_risk_score"], 0.65)
        self.assertGreater(features["dwell_time_consistency"], 0.8)
        self.assertEqual(features["keystroke_timing_jitter"], 1.0)


if __name__ == "__main__":
    unittest.main()
