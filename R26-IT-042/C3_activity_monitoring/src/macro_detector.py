"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/macro_detector.py

MacroDetectorEngine — Detects synthetic input (macros, automation scripts,
keystroke injection) by analyzing keystroke timing patterns and behavioral
anomalies without modifying existing trackers.

Features extracted (12 macro-detection dimensions)
───────────────────────────────────────────────────
  keystroke_timing_entropy        — Randomness of inter-keystroke intervals
  coefficient_of_variation_iki    — Variability in inter-keystroke timing
  keystroke_periodicity_score     — Detects repeating/periodic patterns (FFT)
  dwell_time_consistency          — Uniformity of key-hold durations
  typing_pace_variance_ratio      — Detects abrupt speed changes
  sequence_repetition_score       — Identical keystroke sequence repeats
  burst_typing_ratio              — Density of typing bursts
  keystroke_mouse_sync_score      — Correlation between KB and mouse activity
  app_context_anomaly             — Unrealistic typing speed for app context
  idle_interrupt_ratio            — Unnatural resumption patterns
  keystroke_timing_jitter         — Precision regularity of keystroke intervals
  macro_risk_score                — Aggregated macro injection probability (0-1)

All features are 0-1 normalized or directly interpretable.
Reads from existing trackers without modification.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from C3_activity_monitoring.src.keyboard_tracker import KeyboardTracker
    from C3_activity_monitoring.src.mouse_tracker import MouseTracker


class MacroDetectorEngine:
    """
    Analyzes keystroke timing patterns for macro/automation signatures.
    
    Operates independently from existing trackers (read-only access).
    Designed to detect:
    - Perfect keystroke timing regularity (macro loops)
    - Inhuman typing speeds or patterns
    - Keystroke/mouse activity desynchronization
    - Periodic keystroke injection patterns
    
    Usage
    ─────
    >>> detector = MacroDetectorEngine(keyboard=kb, mouse=mouse)
    >>> macro_features = detector.get_features()
    >>> if macro_features['macro_risk_score'] > 0.7:
    ...     print("Potential macro/injection detected")
    """

    def __init__(
        self,
        keyboard: "KeyboardTracker",
        mouse: "MouseTracker",
        window_sec: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        keyboard:
            KeyboardTracker instance (read-only).
        mouse:
            MouseTracker instance (read-only).
        window_sec:
            Analysis window in seconds (typically matches keyboard window).
        """
        self._keyboard = keyboard
        self._mouse = mouse
        self._window_sec = window_sec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_features(self) -> dict:
        """
        Compute macro-detection features.
        
        Returns
        -------
        dict
            12 macro-detection features (all 0-1 or 0-100 scale).
        """
        try:
            # Extract raw keystroke data (read-only)
            kb_features_raw = self._extract_keystroke_raw_data()
            mouse_features_raw = self._extract_mouse_raw_data()

            # Compute individual detection metrics
            entropy = self._keystroke_timing_entropy(kb_features_raw)
            cv_iki = self._coefficient_of_variation_iki(kb_features_raw)
            periodicity = self._keystroke_periodicity_score(kb_features_raw)
            dwell_consistency = self._dwell_time_consistency(kb_features_raw)
            pace_variance = self._typing_pace_variance_ratio(kb_features_raw)
            repetition = self._sequence_repetition_score(kb_features_raw)
            burst_ratio = self._burst_typing_ratio(kb_features_raw)
            sync_score = self._keystroke_mouse_sync_score(kb_features_raw, mouse_features_raw)
            app_anomaly = self._app_context_anomaly(kb_features_raw)
            idle_interrupt = self._idle_interrupt_ratio(kb_features_raw)
            jitter = self._keystroke_timing_jitter(kb_features_raw)

            # Aggregate macro risk score
            macro_risk = self._aggregate_macro_risk(
                entropy, cv_iki, periodicity, dwell_consistency,
                pace_variance, repetition, burst_ratio, sync_score,
                app_anomaly, idle_interrupt, jitter
            )

            return {
                "keystroke_timing_entropy": round(entropy, 4),
                "coefficient_of_variation_iki": round(cv_iki, 4),
                "keystroke_periodicity_score": round(periodicity, 4),
                "dwell_time_consistency": round(dwell_consistency, 4),
                "typing_pace_variance_ratio": round(pace_variance, 4),
                "sequence_repetition_score": round(repetition, 4),
                "burst_typing_ratio": round(burst_ratio, 4),
                "keystroke_mouse_sync_score": round(sync_score, 4),
                "app_context_anomaly": round(app_anomaly, 4),
                "idle_interrupt_ratio": round(idle_interrupt, 4),
                "keystroke_timing_jitter": round(jitter, 4),
                "macro_risk_score": round(macro_risk, 4),
            }

        except Exception as exc:
            logger.warning("MacroDetectorEngine.get_features() error: %s", exc)
            return self._empty_macro_features()

    # ------------------------------------------------------------------
    # Internal: Raw data extraction (read-only from trackers)
    # ------------------------------------------------------------------

    def _extract_keystroke_raw_data(self) -> dict:
        """
        Extract keystroke timing data from keyboard tracker.
        Returns: {inter_keystroke_intervals, dwell_times, keystroke_sequence}
        """
        now = time.perf_counter()
        cutoff = now - self._window_sec

        try:
            with self._keyboard._lock:
                recent_keystrokes = [
                    k for k in self._keyboard._keystrokes
                    if k.get("press_ts", 0) >= cutoff
                ]
        except Exception:
            recent_keystrokes = []

        if not recent_keystrokes:
            return {
                "inter_keystroke_intervals": [],
                "dwell_times": [],
                "keystroke_count": 0,
                "keystrokes": [],
            }

        # Sort by press timestamp
        recent_keystrokes.sort(key=lambda k: k.get("press_ts", 0))

        # Extract inter-keystroke intervals (flight times between releases and next presses)
        inter_keystroke_intervals = []
        for i in range(1, len(recent_keystrokes)):
            flight_ms = (
                (recent_keystrokes[i]["press_ts"] - recent_keystrokes[i - 1]["release_ts"])
                * 1000.0
            )
            if 0 < flight_ms < 5000:  # filter unreasonable pauses
                inter_keystroke_intervals.append(flight_ms)

        # Extract dwell times
        dwell_times = [k.get("dwell_ms", 0) for k in recent_keystrokes if k.get("dwell_ms", 0) > 0]

        return {
            "inter_keystroke_intervals": inter_keystroke_intervals,
            "dwell_times": dwell_times,
            "keystroke_count": len(recent_keystrokes),
            "keystrokes": recent_keystrokes,
        }

    def _extract_mouse_raw_data(self) -> dict:
        """
        Extract mouse movement data from mouse tracker.
        Returns: {movement_count, movement_gaps, velocity_samples}
        """
        now = time.perf_counter()
        cutoff = now - self._window_sec

        try:
            with self._mouse._lock:
                recent_moves = [
                    m for m in self._mouse._movements
                    if m.get("timestamp", 0) >= cutoff
                ]
        except Exception:
            recent_moves = []

        if not recent_moves:
            return {
                "movement_count": 0,
                "movement_gaps": [],
                "velocity_samples": [],
            }

        recent_moves.sort(key=lambda m: m.get("timestamp", 0))

        # Extract gaps between movements
        movement_gaps = []
        for i in range(1, len(recent_moves)):
            gap_sec = recent_moves[i]["timestamp"] - recent_moves[i - 1]["timestamp"]
            if 0 < gap_sec < 60:
                movement_gaps.append(gap_sec * 1000.0)  # convert to ms

        # Extract velocity samples
        velocity_samples = [m.get("velocity", 0) for m in recent_moves if m.get("velocity", 0) > 0]

        return {
            "movement_count": len(recent_moves),
            "movement_gaps": movement_gaps,
            "velocity_samples": velocity_samples,
        }

    # ------------------------------------------------------------------
    # Layer 1: Statistical Keystroke Authenticity (5 features)
    # ------------------------------------------------------------------

    def _keystroke_timing_entropy(self, kb_data: dict) -> float:
        """
        Measure randomness of inter-keystroke intervals.
        Real humans: high entropy (irregular) → 0.6–1.0
        Macros: low entropy (regular) → 0.0–0.3
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 5:
            return 0.5  # neutral

        # Compute standard deviation of intervals
        mean_interval = sum(intervals) / len(intervals)
        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance) if variance > 0 else 0

        # Entropy: high std_dev = high entropy
        # Normalize to 0-1 scale (empirical: humans typically 20-150ms std_dev)
        entropy = min(std_dev / 150.0, 1.0)
        return entropy

    def _coefficient_of_variation_iki(self, kb_data: dict) -> float:
        """
        Measure variability in inter-keystroke intervals (Coefficient of Variation).
        Real humans: CV > 0.4 (high variation)
        Macros: CV < 0.15 (too uniform)
        
        Returns 0-1: lower CV (macro-like) → higher score (0 = natural, 1 = macro)
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 3:
            return 0.5

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.5

        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)

        # CV = std_dev / mean (dimensionless)
        cv = std_dev / mean_interval if mean_interval > 0 else 0

        # Invert: low CV (macro-like) should produce high score
        # Humans typically CV 0.35–0.65, macros CV 0.05–0.15
        macro_indicator = max(0, 1.0 - (cv / 0.5))  # scales 0.5 CV to 0
        return min(macro_indicator, 1.0)

    def _keystroke_periodicity_score(self, kb_data: dict) -> float:
        """
        Detect periodic/repeating keystroke patterns using autocorrelation.
        Macros often have loop delays that create periodicity.
        
        Returns 0-1: detects periodic patterns (high = macro-like).
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 10:
            return 0.0

        # Simple periodicity check: look for repeated interval patterns
        # Count how many consecutive pairs of intervals are similar
        similar_pairs = 0
        for i in range(len(intervals) - 1):
            # If two consecutive intervals are within 20% of each other, they're "similar"
            mean_of_pair = (intervals[i] + intervals[i + 1]) / 2
            if mean_of_pair > 0:
                diff_pct = abs(intervals[i] - intervals[i + 1]) / mean_of_pair
                if diff_pct < 0.2:  # within 20%
                    similar_pairs += 1

        periodicity_ratio = similar_pairs / max(len(intervals) - 1, 1)
        return min(periodicity_ratio, 1.0)

    def _dwell_time_consistency(self, kb_data: dict) -> float:
        """
        Measure uniformity of key-hold durations.
        Real humans: variable dwell times (high std_dev)
        Macros: suspiciously consistent dwell times (low std_dev)
        
        Returns 0-1: higher value indicates macro-like consistency.
        """
        dwell_times = kb_data["dwell_times"]
        if len(dwell_times) < 5:
            return 0.5

        mean_dwell = sum(dwell_times) / len(dwell_times)
        if mean_dwell == 0:
            return 0.0

        variance = sum((x - mean_dwell) ** 2 for x in dwell_times) / len(dwell_times)
        std_dev = math.sqrt(variance)

        # CV for dwell time
        cv_dwell = std_dev / mean_dwell if mean_dwell > 0 else 0

        # Low CV = high consistency = macro-like
        # Humans typically CV 0.3–0.6, macros CV 0.05–0.15
        consistency_score = max(0, 1.0 - (cv_dwell / 0.4))
        return min(consistency_score, 1.0)

    def _typing_pace_variance_ratio(self, kb_data: dict) -> float:
        """
        Detect abrupt speed changes (macros often: burst typing → long pause → burst again).
        Real humans: gradual pace changes.
        
        Returns 0-1: higher = more abrupt changes = macro-like.
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 6:
            return 0.0

        # Split into two halves and compare mean speeds
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]

        if not first_half or not second_half:
            return 0.0

        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)

        if mean_first == 0:
            return 0.0

        # Ratio of speed change
        pace_ratio = abs(mean_second - mean_first) / mean_first

        # Humans typically < 0.3 ratio change, macros often > 0.5
        abruptness = min(pace_ratio / 0.5, 1.0)
        return abruptness

    # ------------------------------------------------------------------
    # Layer 2: Sequence Pattern Detection (3 features)
    # ------------------------------------------------------------------

    def _sequence_repetition_score(self, kb_data: dict) -> float:
        """
        Detect identical keystroke sequences (substring repeats).
        Macros often repeat: type "test", backspace, retype "test" → high repetition.
        
        Returns 0-1: higher = more repetition = macro-like.
        """
        keystrokes = kb_data["keystrokes"]
        if len(keystrokes) < 8:
            return 0.0

        # Look for repeating dwell+flight patterns (sequence signatures)
        # Build a sequence of (dwell_ms, flight_ms) tuples
        patterns = []
        for i, ks in enumerate(keystrokes):
            dwell = round(ks.get("dwell_ms", 0))
            # Get flight to next keystroke
            if i + 1 < len(keystrokes):
                flight = round(
                    (keystrokes[i + 1]["press_ts"] - ks["release_ts"]) * 1000.0
                )
                if 0 < flight < 5000:
                    patterns.append((dwell, flight))

        if len(patterns) < 4:
            return 0.0

        # Count substring repetitions (3-keystroke windows)
        window_size = 3
        repetition_count = 0
        for i in range(len(patterns) - window_size):
            window = tuple(patterns[i:i + window_size])
            # Count how many times this exact window appears later
            for j in range(i + window_size, len(patterns) - window_size + 1):
                if tuple(patterns[j:j + window_size]) == window:
                    repetition_count += 1
                    break  # count each window once

        total_windows = max(len(patterns) - window_size, 1)
        repetition_ratio = repetition_count / total_windows
        return min(repetition_ratio, 1.0)

    def _burst_typing_ratio(self, kb_data: dict) -> float:
        """
        Measure sudden dense keystroke clusters (typical of macros).
        Real humans: mixed typing/pausing.
        Macros: often 2–3 sec burst, 0.5 sec pause, repeat.
        
        Returns 0-1: higher = more burst-like = macro-like.
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 10:
            return 0.0

        # Define "burst": consecutive short intervals (< 100ms)
        # Define "pause": longer interval (> 300ms)
        burst_threshold = 100  # ms
        pause_threshold = 300  # ms

        burst_count = sum(1 for i in intervals if i < burst_threshold)
        pause_count = sum(1 for i in intervals if i > pause_threshold)

        # Macro ratio: many bursts followed by pauses
        if burst_count + pause_count == 0:
            return 0.0

        macro_pattern_ratio = burst_count / (burst_count + pause_count)

        # High burst ratio might indicate macro (but not always)
        # Threshold: 0.7+ = likely macro
        burst_ratio = max(0, (macro_pattern_ratio - 0.5) / 0.3)
        return min(burst_ratio, 1.0)

    def _app_context_anomaly(self, kb_data: dict) -> float:
        """
        Detect unrealistic typing speeds for application context.
        Example: 1000+ WPM in Notepad (impossible for human).
        
        Returns 0-1: higher = more unrealistic = macro-like.
        """
        kb_count = kb_data["keystroke_count"]
        if kb_count < 10:
            return 0.0

        # Estimate WPM from keystroke count and window
        wpm = (kb_count / 5.0) / (self._window_sec / 60.0)  # 5 chars per word

        # Humans typically type: 30–80 WPM normal, 80–150 WPM fast
        # Superhuman: 300+ WPM
        # Macros can reach: 500+ WPM

        if wpm > 300:  # Definitely unrealistic
            anomaly = min((wpm - 300) / 200.0, 1.0)
            return anomaly
        elif wpm > 150:  # Borderline
            anomaly = (wpm - 150) / 300.0
            return anomaly
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Layer 3: Contextual Correlation (3 features)
    # ------------------------------------------------------------------

    def _keystroke_mouse_sync_score(self, kb_data: dict, mouse_data: dict) -> float:
        """
        Detect desynchronization between keyboard and mouse activity.
        Real humans: keyboard activity followed by mouse movement.
        Macros: often isolated keyboard bursts (no mouse correlation).
        
        Returns 0-1: higher = desynchronized = macro-like.
        """
        kb_count = kb_data["keystroke_count"]
        mouse_count = mouse_data["movement_count"]

        # If very little activity, return neutral
        if kb_count < 5 or mouse_count < 5:
            return 0.5

        # Ratio of keyboard to mouse activity
        # Humans: roughly balanced, ratio 0.3–3.0
        # Macros: often keyboard-heavy, ratio > 5.0
        ratio = kb_count / max(mouse_count, 1)

        if ratio > 5.0:
            desync = min((ratio - 5.0) / 5.0, 1.0)
            return desync
        elif ratio > 3.0:
            desync = (ratio - 3.0) / 4.0
            return desync
        else:
            return 0.0

    def _idle_interrupt_ratio(self, kb_data: dict) -> float:
        """
        Detect unnatural resumption patterns after idle.
        Real humans: gradual activity ramp-up.
        Macros: sudden dense activity bursts.
        
        Approximated by: how consistent is keystroke timing (abrupt changes = suspicious).
        Returns 0-1: higher = more suspicious = macro-like.
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 8:
            return 0.0

        # Check first half vs second half for sudden activity changes
        mid = len(intervals) // 2
        first_half = intervals[:mid]
        second_half = intervals[mid:]

        if not first_half or not second_half:
            return 0.0

        # If first half is slower (idle resumption), second half should gradually speed up
        # Macros: sudden jump to fast typing
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)

        # If suddenly goes from slow to very fast, could be macro
        if mean_first > 200 and mean_second < 80:  # Idle → sudden burst
            interrupt_score = 0.7
            return interrupt_score
        elif mean_first > 150 and mean_second < 100:
            interrupt_score = 0.4
            return interrupt_score
        else:
            return 0.0

    # ------------------------------------------------------------------
    # Layer 4: Hardware/OS Signals (1 feature)
    # ------------------------------------------------------------------

    def _keystroke_timing_jitter(self, kb_data: dict) -> float:
        """
        Measure timing precision of keystroke intervals.
        Real keyboards: small timing jitter (±10–50ms variation between similar intervals).
        Input injection tools: machine-perfect timing (jitter < 5ms).
        
        Returns 0-1: higher = too perfect = macro-like.
        """
        intervals = kb_data["inter_keystroke_intervals"]
        if len(intervals) < 6:
            return 0.5

        # Compute jitter as coefficient of variation of consecutive interval pairs
        jitter_values = []
        for i in range(len(intervals) - 1):
            diff = abs(intervals[i] - intervals[i + 1])
            mean_pair = (intervals[i] + intervals[i + 1]) / 2
            if mean_pair > 0:
                jitter_values.append(diff / mean_pair)

        if not jitter_values:
            return 0.5

        avg_jitter = sum(jitter_values) / len(jitter_values)

        # Humans: avg_jitter typically 0.15–0.40
        # Injection tools: avg_jitter < 0.05 (too perfect)
        if avg_jitter < 0.05:
            perfection_score = 1.0
        elif avg_jitter < 0.10:
            perfection_score = 0.8
        elif avg_jitter < 0.15:
            perfection_score = 0.5
        else:
            perfection_score = max(0, 1.0 - (avg_jitter / 0.5))

        return min(perfection_score, 1.0)

    # ------------------------------------------------------------------
    # Layer 5: Aggregate Score
    # ------------------------------------------------------------------

    def _aggregate_macro_risk(
        self,
        entropy: float,
        cv_iki: float,
        periodicity: float,
        dwell_consistency: float,
        pace_variance: float,
        repetition: float,
        burst_ratio: float,
        sync_score: float,
        app_anomaly: float,
        idle_interrupt: float,
        jitter: float,
    ) -> float:
        """
        Combine 11 indicators into single macro_risk_score (0–1).
        Uses weighted average + non-linear boosting for strong signals.
        """
        # Assign weights (importance)
        weights = {
            "cv_iki": 0.15,  # CV is strong indicator
            "periodicity": 0.12,
            "dwell_consistency": 0.12,
            "entropy": 0.10,  # Inverse of entropy
            "burst_ratio": 0.12,
            "repetition": 0.10,
            "jitter": 0.10,
            "sync_score": 0.08,
            "app_anomaly": 0.08,
            "pace_variance": 0.05,
            "idle_interrupt": 0.03,
        }

        # Invert entropy (high entropy = low risk)
        entropy_risk = 1.0 - entropy

        # Weighted sum
        score = (
            weights["entropy"] * entropy_risk
            + weights["cv_iki"] * cv_iki
            + weights["periodicity"] * periodicity
            + weights["dwell_consistency"] * dwell_consistency
            + weights["pace_variance"] * pace_variance
            + weights["repetition"] * repetition
            + weights["burst_ratio"] * burst_ratio
            + weights["sync_score"] * sync_score
            + weights["app_anomaly"] * app_anomaly
            + weights["idle_interrupt"] * idle_interrupt
            + weights["jitter"] * jitter
        )

        # Non-linear boosting: if 3+ indicators are high, boost the score
        high_indicators = sum(1 for x in [
            cv_iki, periodicity, dwell_consistency, burst_ratio,
            jitter, app_anomaly
        ] if x > 0.6)

        if high_indicators >= 3:
            # Multiple strong signals: boost score
            boost_factor = 1.0 + (0.2 * min(high_indicators - 2, 3))
            score = min(score * boost_factor, 1.0)

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_macro_features() -> dict:
        """Return zeroed macro features when data is unavailable."""
        return {
            "keystroke_timing_entropy": 0.5,
            "coefficient_of_variation_iki": 0.5,
            "keystroke_periodicity_score": 0.0,
            "dwell_time_consistency": 0.5,
            "typing_pace_variance_ratio": 0.0,
            "sequence_repetition_score": 0.0,
            "burst_typing_ratio": 0.0,
            "keystroke_mouse_sync_score": 0.5,
            "app_context_anomaly": 0.0,
            "idle_interrupt_ratio": 0.0,
            "keystroke_timing_jitter": 0.5,
            "macro_risk_score": 0.0,
        }
