from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestFormalS8State(unittest.TestCase):
    def test_exact_schema_order_status_and_dimension_are_frozen(self) -> None:
        from v2.dqn.state import (
            FORMAL_STATE_DIMENSION,
            FORMAL_STATE_FEATURE_NAMES,
            FORMAL_STATE_SCHEMA_DIGEST,
            FORMAL_STATE_SCHEMA_VERSION,
            FORMAL_STATE_STATUS,
        )

        self.assertEqual(FORMAL_STATE_STATUS, "FROZEN_PROJECT_BASELINE")
        self.assertEqual(FORMAL_STATE_SCHEMA_VERSION, "v2_s8_onboard_ais_v1")
        self.assertEqual(FORMAL_STATE_DIMENSION, 8)
        self.assertEqual(
            FORMAL_STATE_FEATURE_NAMES,
            (
                "soc",
                "causal_base_load_fraction",
                "load_residual_fraction",
                "recent_load_population_std_fraction",
                "recent_load_window_trend_fraction",
                "fuel_cell_power_fraction",
                "fuel_cell_delta_fraction",
                "speed_fraction",
            ),
        )
        self.assertRegex(FORMAL_STATE_SCHEMA_DIGEST, r"^[0-9a-f]{64}$")

    def test_builds_exact_s8_and_does_not_clip_speed(self) -> None:
        from v2.dqn.state import OperatingHistorySample, build_formal_operating_state

        history = (
            OperatingHistorySample(0.0, 0.60, 10.0, 0.0, 100.0, 80.0),
            OperatingHistorySample(30.0, 0.59, 40.0, 0.0, 160.0, 120.0),
        )
        state = build_formal_operating_state(
            history,
            current_time_seconds=30.0,
            speed_kn=25.0,
        )
        expected = (
            0.59,
            120.0 / 600.0,
            40.0 / 600.0,
            np.std([100.0, 160.0], ddof=0) / 600.0,
            2.0 * 150.0 / 600.0,
            40.0 / 600.0,
            30.0 / 600.0,
            1.25,
        )
        np.testing.assert_allclose(state, expected, rtol=0.0, atol=1e-12)

    def test_cold_start_and_reentry_are_causal(self) -> None:
        from v2.dqn.state import OperatingHistorySample, build_formal_operating_state

        history = (OperatingHistorySample(0.0, 0.60, 0.0, 0.0, -70.0, -70.0),)
        state = build_formal_operating_state(
            history,
            current_time_seconds=0.0,
            speed_kn=0.0,
        )
        self.assertEqual(state[1], -70.0 / 600.0)
        self.assertEqual(state[2], 0.0)
        self.assertEqual(state[3], 0.0)
        self.assertEqual(state[4], 0.0)
        self.assertEqual(state[6], 0.0)

        with self.assertRaisesRegex(ValueError, "nonnegative"):
            build_formal_operating_state(
                history,
                current_time_seconds=0.0,
                speed_kn=-0.1,
            )

    def test_future_history_tail_cannot_change_current_state(self) -> None:
        from v2.dqn.state import OperatingHistorySample, build_formal_operating_state

        prefix = (
            OperatingHistorySample(0.0, 0.60, 10.0, 0.0, 100.0, 100.0),
            OperatingHistorySample(30.0, 0.59, 20.0, 0.0, 120.0, 110.0),
        )
        future = OperatingHistorySample(60.0, 0.01, 600.0, -500.0, 999.0, 1.0)
        current = build_formal_operating_state(
            prefix, current_time_seconds=30.0, speed_kn=4.0
        )
        with_future = build_formal_operating_state(
            prefix + (future,), current_time_seconds=30.0, speed_kn=4.0
        )
        self.assertEqual(current, with_future)


if __name__ == "__main__":
    unittest.main()
