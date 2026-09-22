from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class SupervisoryRuleTests(unittest.TestCase):
    def test_freshness_cap_is_inclusive_causal_and_fixed_at_ten_seconds(self) -> None:
        from v2.data.supervisory_rules import (
            FRESHNESS_CAP_SECONDS,
            is_fresh_causal_age,
        )

        self.assertEqual(FRESHNESS_CAP_SECONDS, 10.0)
        self.assertTrue(is_fresh_causal_age(0.0))
        self.assertTrue(is_fresh_causal_age(10.0))
        self.assertFalse(is_fresh_causal_age(-0.001))
        self.assertFalse(is_fresh_causal_age(10.001))
        with self.assertRaises(TypeError):
            is_fresh_causal_age(True)

    def test_mode_classification_requires_two_shore_samples_and_fails_closed(self) -> None:
        from v2.data.supervisory_rules import (
            FC_ZERO_TOLERANCE_KW,
            SPEED_ZERO_TOLERANCE_KN,
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        self.assertEqual(SPEED_ZERO_TOLERANCE_KN, 0.1)
        self.assertEqual(FC_ZERO_TOLERANCE_KW, 8.0)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = (
            ModeSample(start, 0.1, 8.0, -20.0),
            ModeSample(start + timedelta(seconds=30), 0.0, 0.0, -30.0),
            ModeSample(start + timedelta(seconds=60), 0.2, 100.0, 10.0),
            ModeSample(start + timedelta(seconds=90), 0.1, 0.0, 10.0),
            ModeSample(start + timedelta(seconds=120), 0.0, 8.1, -10.0),
            ModeSample(start + timedelta(seconds=150), 2.0, 100.0, 10.0, channels_complete=False),
            ModeSample(start + timedelta(seconds=180), 2.0, 100.0, 10.0, conflicting_duplicate=True),
            ModeSample(start + timedelta(seconds=240), 2.0, 100.0, 10.0, long_gap_contaminated=True),
            ModeSample(start + timedelta(seconds=270), 2.0, 10.0, -20.0),
        )

        self.assertEqual(
            classify_operating_modes(values),
            (
                OperatingMode.SHORE_CONNECTED,
                OperatingMode.SHORE_CONNECTED,
                OperatingMode.SAILING_ISLAND,
                OperatingMode.UNKNOWN,
                OperatingMode.UNKNOWN,
                OperatingMode.UNKNOWN,
                OperatingMode.UNKNOWN,
                OperatingMode.UNKNOWN,
                OperatingMode.UNKNOWN,
            ),
        )

    def test_load_reconstruction_is_available_only_for_sailing_island(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            reconstruct_sailing_load,
        )

        sample = ModeSample(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            5.0,
            300.0,
            -40.0,
        )
        self.assertEqual(
            reconstruct_sailing_load(sample, OperatingMode.SAILING_ISLAND),
            260.0,
        )
        for mode in (OperatingMode.SHORE_CONNECTED, OperatingMode.UNKNOWN):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                reconstruct_sailing_load(sample, mode)


if __name__ == "__main__":
    unittest.main()
