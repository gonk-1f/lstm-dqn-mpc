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
    DQN_MPC_PREVIEW_STEPS,
    DQN_MPC_STATE_DIM,
    build_dqn_mpc_state,
)


class TestDqnMpcStateBuilder(unittest.TestCase):
    def test_state_dimension_is_exactly_eleven(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=280.0,
            previous_batt_kw=0.0,
            current_load_kw=300.0,
            previous_load_kw=290.0,
            future_load_kw=[
                310.0,
                320.0,
                330.0,
                340.0,
                350.0,
                360.0,
            ],
        )

        self.assertEqual(DQN_MPC_STATE_DIM, 11)
        self.assertEqual(DQN_MPC_PREVIEW_STEPS, 6)
        self.assertEqual(state.shape, (11,))
        self.assertEqual(state.dtype, np.float32)

    def test_state_values_follow_required_normalization(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.60,
            previous_fc_kw=280.0,
            previous_batt_kw=312.0,
            current_load_kw=560.0,
            previous_load_kw=512.0,
            future_load_kw=[
                56.0,
                112.0,
                168.0,
                224.0,
                280.0,
                336.0,
            ],
        )

        expected = np.asarray(
            [
                1.0,       # (0.60 - 0.55) / 0.05
                0.5,       # 280 / 560
                0.5,       # 312 / 624
                1.0,       # 560 / 560
                1.0,       # (560 - 512) / 48
                0.1,       # 56 / 560
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(
            state,
            expected,
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_zero_load_change_is_zero(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=0.0,
            previous_batt_kw=0.0,
            current_load_kw=250.0,
            previous_load_kw=250.0,
            future_load_kw=[250.0] * 6,
        )

        self.assertAlmostEqual(
            float(state[4]),
            0.0,
            places=7,
        )

    def test_state_is_not_clipped(self) -> None:
        state = build_dqn_mpc_state(
            current_soc=0.65,
            previous_fc_kw=560.0,
            previous_batt_kw=1248.0,
            current_load_kw=700.0,
            previous_load_kw=600.0,
            future_load_kw=[700.0] * 6,
        )

        self.assertAlmostEqual(
            float(state[2]),
            2.0,
            places=7,
        )

        self.assertGreater(
            float(state[3]),
            1.0,
        )

        self.assertGreater(
            float(state[4]),
            1.0,
        )

    def test_preview_must_have_exactly_six_values(self) -> None:
        for preview in (
            [100.0] * 5,
            [100.0] * 7,
        ):
            with self.subTest(length=len(preview)):
                with self.assertRaises(ValueError):
                    build_dqn_mpc_state(
                        current_soc=0.55,
                        previous_fc_kw=200.0,
                        previous_batt_kw=0.0,
                        current_load_kw=300.0,
                        previous_load_kw=290.0,
                        future_load_kw=preview,
                    )

    def test_non_finite_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_dqn_mpc_state(
                current_soc=np.nan,
                previous_fc_kw=200.0,
                previous_batt_kw=0.0,
                current_load_kw=300.0,
                previous_load_kw=290.0,
                future_load_kw=[300.0] * 6,
            )

        with self.assertRaises(ValueError):
            build_dqn_mpc_state(
                current_soc=0.55,
                previous_fc_kw=200.0,
                previous_batt_kw=0.0,
                current_load_kw=300.0,
                previous_load_kw=290.0,
                future_load_kw=[
                    300.0,
                    300.0,
                    np.inf,
                    300.0,
                    300.0,
                    300.0,
                ],
            )


if __name__ == "__main__":
    unittest.main()