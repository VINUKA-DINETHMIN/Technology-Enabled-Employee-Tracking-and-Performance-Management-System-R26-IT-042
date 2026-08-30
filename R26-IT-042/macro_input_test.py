"""
Test-only synthetic keyboard input generator.

Run this while the employee monitoring application is already running.
It sends only F15 key events, so it does not type text or click the mouse.
The purpose is to create a deliberately regular timing pattern for testing
MacroDetectorEngine. This does not modify application code or database data.
"""

from __future__ import annotations

import time

from pynput import keyboard


KEY_COUNT = 30
INTERVAL_SECONDS = 0.100


def main() -> None:
    controller = keyboard.Controller()

    print("[TEST] Ensure the employee monitor is running before continuing.")
    input("[TEST] Press Enter to send 30 regular F15 events, or Ctrl+C to cancel: ")

    print("[TEST] Sending continuous regular synthetic keyboard timing (press Ctrl+C to stop)...")
    count = 0
    while True:
        count += 1
        controller.press(keyboard.Key.f15)
        time.sleep(0.04)  # 40ms key-hold duration
        controller.release(keyboard.Key.f15)
        if count % 10 == 0:
            print(f"[TEST] Sent {count} synthetic macro events... (Monitoring active window)")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[TEST] Cancelled.")
    except Exception as exc:
        print(f"[TEST] Could not generate keyboard events: {exc}")
        print("[TEST] Install the dependency with: pip install pynput")
