from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestFormalModeInterlock(unittest.TestCase):
    def test_negative_signed_power_is_shore_regardless_of_ais(self) -> None:
        from v2.envs.formal_episode import (
            FormalOperatingMode,
            classify_formal_mode,
            interlocked_fc_power_kw,
        )

        for speed in (0.0, 0.1, 1.9, 12.0):
            with self.subTest(speed=speed):
                mode = classify_formal_mode(-70.0, speed)
                self.assertIs(mode, FormalOperatingMode.SHORE_CONNECTED)
                self.assertEqual(interlocked_fc_power_kw(mode, 500.0), 0.0)
                self.assertEqual(mode.shore_connected, 1.0)

    def test_zero_speed_does_not_turn_positive_hotel_load_into_shore(self) -> None:
        from v2.envs.formal_episode import (
            FormalOperatingMode,
            classify_formal_mode,
            interlocked_fc_power_kw,
        )

        mode = classify_formal_mode(120.0, 0.0)
        self.assertIs(mode, FormalOperatingMode.SAILING_OR_ISLANDED)
        self.assertEqual(interlocked_fc_power_kw(mode, 75.0), 75.0)
        self.assertEqual(mode.shore_connected, 0.0)

    def test_deadband_is_idle_and_action_invariant(self) -> None:
        from v2.envs.formal_episode import (
            FormalOperatingMode,
            classify_formal_mode,
            interlocked_fc_power_kw,
        )

        for load in (-1.0, 0.0, 1.0):
            mode = classify_formal_mode(load, 15.0)
            self.assertIs(mode, FormalOperatingMode.IDLE)
            self.assertEqual(interlocked_fc_power_kw(mode, 600.0), 0.0)


if __name__ == "__main__":
    unittest.main()
