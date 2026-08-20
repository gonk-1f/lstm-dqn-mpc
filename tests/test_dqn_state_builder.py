from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from dqn.utils.state_builder import (  # noqa: E402
    DQN_MPC_STATE_DIM,
    build_dqn_mpc_state,
)


class TestDqnMpcStateBuilder(unittest.TestCase):
    def test_state_dimension_is_exactly_seven(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=280.0,
            previous_batt_kw=0.0,
            load_history_kw=[280.0, 290.0, 300.0],
        )
        self.assertEqual(DQN_MPC_STATE_DIM, 7)
        self.assertEqual(state.shape, (7,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_values_follow_required_normalization(self) -> None:
        history = [100.0, 200.0, 300.0, 400.0, 500.0, 512.0, 560.0]
        state = build_dqn_mpc_state(
            current_soc=0.60,
            previous_fc_kw=280.0,
            previous_batt_kw=312.0,
            load_history_kw=history,
        )
        mean_history = float(np.mean(history)) / 600.0
        expected = np.asarray(
            [
                1.0,
                280.0 / 600.0,
                0.5,
                560.0 / 600.0,
                1.0,
                mean_history,
                mean_history,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(state, expected, rtol=0.0, atol=1.0e-6)

    def test_history_uses_available_points_at_voyage_start(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=200.0,
            previous_batt_kw=0.0,
            load_history_kw=[300.0],
        )
        self.assertAlmostEqual(float(state[4]), 0.0, places=7)
        self.assertAlmostEqual(float(state[5]), 0.5, places=7)
        self.assertAlmostEqual(float(state[6]), 0.5, places=7)

    def test_history_windows_use_last_ten_and_sixty_seconds(self) -> None:
        history = np.arange(1.0, 71.0)
        state = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=0.0,
            previous_batt_kw=0.0,
            load_history_kw=history,
        )
        self.assertAlmostEqual(float(state[5]), np.mean(history[-10:]) / 600.0)
        self.assertAlmostEqual(float(state[6]), np.mean(history[-60:]) / 600.0)

    def test_state_is_not_clipped(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.65,
            previous_fc_kw=560.0,
            previous_batt_kw=1248.0,
            load_history_kw=[600.0, 700.0],
        )
        self.assertAlmostEqual(float(state[2]), 2.0, places=7)
        self.assertGreater(float(state[3]), 1.0)
        self.assertGreater(float(state[4]), 1.0)

    def test_non_finite_or_empty_history_is_rejected(self) -> None:
        common = {
            "current_soc": 0.55,
            "previous_fc_kw": 200.0,
            "previous_batt_kw": 0.0,
        }
        with self.assertRaises(ValueError):
            build_dqn_mpc_state(**common, load_history_kw=[])
        with self.assertRaises(ValueError):
            build_dqn_mpc_state(**common, load_history_kw=[300.0, np.inf])
        with self.assertRaises(ValueError):
            build_dqn_mpc_state(
                **{**common, "current_soc": np.nan},
                load_history_kw=[300.0],
            )


if __name__ == "__main__":
    unittest.main()
