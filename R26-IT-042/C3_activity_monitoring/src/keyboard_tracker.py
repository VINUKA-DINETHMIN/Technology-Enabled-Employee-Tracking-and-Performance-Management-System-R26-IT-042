"""
R26-IT-042 — C3: Activity Monitoring
C3_activity_monitoring/src/keyboard_tracker.py

KeyboardTracker — Records keystroke timings and computes behavioural
biometric features per configurable window (default 30 seconds).

Features extracted
──────────────────
  mean_dwell_time    — avg key-hold duration (ms)
  std_dwell_time     — std-dev of key-hold durations
  mean_flight_time   — avg gap between key-release → next key-press (ms)
  typing_speed_wpm   — estimated words per minute (5 chars/word)
  error_rate         — backspace / total keystrokes ratio

RAW KEYS ARE NEVER STORED.  Only timing metadata is kept in memory.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Average characters per word used for WPM estimation
_CHARS_PER_WORD = 5.0


class KeyboardTracker:
    """
    Cross-platform keystroke biometric recorder.

    Listens to keyboard events via pynput and computes timing-based
    features over a configurable sliding window.  Raw keys are discarded
    immediately; only press/release timestamps and whether the key was a
    backspace are stored.

    Usage
    ─────
    >>> tracker = KeyboardTracker(window_sec=30)
    >>> tracker.start()
    >>> features = tracker.get_features()
    >>> tracker.stop()
    """

    def __init__(self, window_sec: float = 30.0) -> None:
        """
        Parameters
        ----------
        window_sec:
            Feature extraction window in seconds.  Data older than this
            is discarded automatically.
        """
        self._window_sec = window_sec
        self._on_activity = None
        self._running = False
        self._listener = None
        self._lock = threading.Lock()

        # Per-key press/release timestamps: {event_id: {"press": float, "release": float, "is_backspace": bool}}
        # We use a list of completed keystroke records instead
        self._keystrokes: list[dict] = []   # {press_ts, release_ts, is_backspace}
        self._pending_press: dict[str, float] = {}  # key_name -> press timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the keyboard listener in a background daemon thread."""
        try:
            from pynput import keyboard as _kb

            self._running = True

            def _key_id(key) -> str:
                """Build a stable key identifier for press/release pairing."""
                vk = getattr(key, "vk", None)
                if vk is not None:
                    return f"vk:{vk}"
                name = getattr(key, "name", None)
                if name is not None:
                    return f"name:{name}"
                return str(key)

            def on_press(key):
                ts = time.perf_counter()
                key_id = _key_id(key)
                is_backspace = key == _kb.Key.backspace

                with self._lock:
                    self._pending_press[key_id] = (ts, is_backspace)
                if self._on_activity:
                    self._on_activity()

            def on_release(key):
                release_ts = time.perf_counter()
                key_id = _key_id(key)
                is_backspace = key == _kb.Key.backspace

                with self._lock:
                    if key_id in self._pending_press:
                        press_ts, is_bs = self._pending_press.pop(key_id)
                        dwell = (release_ts - press_ts) * 1000.0  # ms
                        if 0 <= dwell < 2000:  # sanity filter
                            self._keystrokes.append({
                                "press_ts": press_ts,
                                "release_ts": release_ts,
                                "dwell_ms": dwell,
                                "is_backspace": is_backspace or is_bs,
                            })
                    # Prune old entries outside the window
                    cutoff = time.perf_counter() - self._window_sec * 3
                    self._keystrokes = [k for k in self._keystrokes if k["press_ts"] >= cutoff]

            self._listener = _kb.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
            logger.info("KeyboardTracker started (window=%.0fs).", self._window_sec)

        except ImportError:
            logger.warning("pynput not available — KeyboardTracker disabled.")
        except Exception as exc:
            logger.error("KeyboardTracker start error: %s", exc)

    def stop(self) -> None:
        """Stop the keyboard listener."""
        self._running = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        logger.info("KeyboardTracker stopped.")

    def record_activity(self) -> None:
        """Notify external idle detector of activity (called by IdleDetector hook)."""
        # Activity is implicitly recorded through keystroke data; this is a no-op
        # kept for interface compatibility with IdleDetector.
        pass

    def get_features(self) -> dict:
        """
        Compute and return keyboard biometric features for the last window.

        Returns
        -------
        dict
            Keys: mean_dwell_time, std_dwell_time, mean_flight_time,
                  typing_speed_wpm, error_rate
        """
        now = time.perf_counter()
        cutoff = now - self._window_sec

        with self._lock:
            recent = [k for k in self._keystrokes if k["press_ts"] >= cutoff]

        if not recent:
            return _empty_keyboard_features()

        # ── Dwell time ───────────────────────────────────────────────
        dwells = [k["dwell_ms"] for k in recent]
        mean_dwell = _mean(dwells)
        std_dwell = _std(dwells)

        # ── Flight time (release[i] → press[i+1]) ────────────────────
        flights = []
        for i in range(1, len(recent)):
            flight = (recent[i]["press_ts"] - recent[i - 1]["release_ts"]) * 1000.0
            if 0 < flight < 5000:  # filter unreasonably long pauses
                flights.append(flight)

        mean_flight = _mean(flights) if flights else 0.0

        # ── Typing speed ──────────────────────────────────────────────
        elapsed_min = max(self._window_sec / 60.0, 1e-6)
        typing_speed_wpm = len(recent) / _CHARS_PER_WORD / elapsed_min

        # ── Error rate ────────────────────────────────────────────────
        backspaces = sum(1 for k in recent if k["is_backspace"])
        error_rate = backspaces / max(len(recent), 1)

        return {
            "mean_dwell_time": round(mean_dwell, 3),
            "std_dwell_time": round(std_dwell, 3),
            "mean_flight_time": round(mean_flight, 3),
            "typing_speed_wpm": round(typing_speed_wpm, 2),
            "error_rate": round(error_rate, 4),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_keyboard_features() -> dict:
    """Return zeroed keyboard features when no data is available."""
    return {
        "mean_dwell_time": 0.0,
        "std_dwell_time": 0.0,
        "mean_flight_time": 0.0,
        "typing_speed_wpm": 0.0,
        "error_rate": 0.0,
    }


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)
