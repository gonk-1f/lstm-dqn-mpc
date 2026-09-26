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
    def test_onboard_load_normalization_clamps_only_the_frozen_deadband(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            normalize_onboard_load_kw,
            reconstruct_sailing_load,
        )

        self.assertEqual(normalize_onboard_load_kw(0.0), 0.0)
        self.assertEqual(normalize_onboard_load_kw(12.5), 12.5)
        self.assertEqual(normalize_onboard_load_kw(-1.0), 0.0)
        self.assertEqual(normalize_onboard_load_kw(-0.001), 0.0)
        with self.assertRaisesRegex(ValueError, "deadband"):
            normalize_onboard_load_kw(-1.001)
        sample = ModeSample(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            1.0,
            0.0,
            -0.5,
        )
        self.assertEqual(reconstruct_sailing_load(sample, OperatingMode.ONBOARD), 0.0)

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

    def test_mode_classification_requires_three_causal_shore_samples(self) -> None:
        from v2.data.supervisory_rules import (
            BATTERY_CHARGE_THRESHOLD_KW,
            SHORE_MIN_CONSECUTIVE_SAMPLES,
            SPEED_ZERO_TOLERANCE_KN,
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        self.assertEqual(SPEED_ZERO_TOLERANCE_KN, 0.1)
        self.assertEqual(BATTERY_CHARGE_THRESHOLD_KW, 1.0)
        self.assertEqual(SHORE_MIN_CONSECUTIVE_SAMPLES, 3)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        values = (
            ModeSample(start, 0.1, 8.0, -20.0),
            ModeSample(start + timedelta(seconds=30), 0.0, 0.0, -30.0),
            ModeSample(start + timedelta(seconds=60), 0.0, 1.0, -25.0),
            ModeSample(start + timedelta(seconds=90), 0.2, 100.0, -20.0),
            ModeSample(start + timedelta(seconds=120), 0.0, 100.0, 10.0),
            ModeSample(start + timedelta(seconds=150), 0.0, 100.0, 10.0, channels_complete=False),
        )

        self.assertEqual(
            classify_operating_modes(values),
            (
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_CHARGING,
                OperatingMode.ONBOARD,
                OperatingMode.ONBOARD,
                OperatingMode.UNRESOLVED,
            ),
        )

    def test_short_candidate_run_and_negative_unexplained_power_fail_closed(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        samples = (
            ModeSample(start, 0.0, 0.0, -20.0),
            ModeSample(start + timedelta(seconds=30), 0.0, 0.0, -20.0),
            ModeSample(start + timedelta(seconds=60), 2.0, 10.0, -30.0),
        )

        self.assertEqual(
            classify_operating_modes(samples),
            (
                OperatingMode.UNRESOLVED,
                OperatingMode.UNRESOLVED,
                OperatingMode.UNRESOLVED,
            ),
        )

    def test_nonzero_recorded_fc_does_not_block_confirmed_shore_charging(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        samples = tuple(
            ModeSample(
                start + timedelta(seconds=30 * index),
                0.0,
                fc_kw,
                -150.0,
                p_load_kw=-50.0,
            )
            for index, fc_kw in enumerate((100.0, 106.0, 107.0))
        )

        self.assertEqual(
            classify_operating_modes(samples),
            (
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_PENDING,
                OperatingMode.SHORE_CHARGING,
            ),
        )

    def test_zero_speed_hotel_load_and_moving_fc_charge_remain_onboard(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        samples = (
            ModeSample(start, 0.0, 120.0, 0.0),
            ModeSample(start + timedelta(seconds=30), 5.0, 100.0, -40.0),
        )
        self.assertEqual(
            classify_operating_modes(samples),
            (OperatingMode.ONBOARD, OperatingMode.ONBOARD),
        )

    def test_frozen_load_deadband_prevents_component_interpolation_residual_misclassification(self) -> None:
        from v2.data.supervisory_rules import (
            ModeSample,
            OperatingMode,
            classify_operating_modes,
        )

        sample = ModeSample(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            0.0,
            0.0,
            -0.01,
            p_load_kw=0.0,
        )
        self.assertEqual(classify_operating_modes((sample,)), (OperatingMode.ONBOARD,))

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
        self.assertEqual(reconstruct_sailing_load(sample, OperatingMode.ONBOARD), 260.0)
        for mode in (
            OperatingMode.SHORE_PENDING,
            OperatingMode.SHORE_CHARGING,
            OperatingMode.UNRESOLVED,
        ):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                reconstruct_sailing_load(sample, mode)


if __name__ == "__main__":
    unittest.main()
