from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from dqn.utils.reward import (  # noqa: E402
    BATTERY_POWER_REF_KW,
    FC_VARIATION_REF_KW,
    REWARD_Q_BATT,
    REWARD_Q_FC_VAR,
    REWARD_Q_H2,
    REWARD_Q_SOC,
    SOC_BAND,
    SOC_REFERENCE,
    calculate_mpc_weight_reward,
)

from mpc.solvers.fc_dp0_curve import (  # noqa: E402
    dp0_quadratic_coefficients,
)


class TestDqnMpcReward(unittest.TestCase):
    def test_fixed_reward_weights_are_candidate_c(self) -> None:
        self.assertEqual(REWARD_Q_H2, 0.25)
        self.assertEqual(REWARD_Q_BATT, 0.40)
        self.assertEqual(REWARD_Q_SOC, 12.0)
        self.assertEqual(REWARD_Q_FC_VAR, 20.0)

    def test_zero_objective_terms_give_zero_reward(self) -> None:
        reward, info = calculate_mpc_weight_reward(
            p_fc_kw=0.0,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=0.0,
        )

        self.assertAlmostEqual(reward, 0.0, places=12)
        self.assertAlmostEqual(info["h2_norm"], 0.0, places=12)
        self.assertAlmostEqual(
            info["battery_power_sq_norm"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            info["soc_tracking_sq_norm"],
            0.0,
            places=12,
        )
        self.assertAlmostEqual(
            info["fc_variation_sq_norm"],
            0.0,
            places=12,
        )

    def test_each_normalized_reference_has_unit_cost(self) -> None:
        _, batt_info = calculate_mpc_weight_reward(
            p_fc_kw=0.0,
            p_batt_kw=BATTERY_POWER_REF_KW,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=0.0,
        )

        self.assertAlmostEqual(
            batt_info["battery_power_sq_norm"],
            1.0,
            places=12,
        )

        _, soc_info = calculate_mpc_weight_reward(
            p_fc_kw=0.0,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE + SOC_BAND,
            previous_fc_kw=0.0,
        )

        self.assertAlmostEqual(
            soc_info["soc_tracking_sq_norm"],
            1.0,
            places=12,
        )

        _, fc_info = calculate_mpc_weight_reward(
            p_fc_kw=FC_VARIATION_REF_KW,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=0.0,
        )

        self.assertAlmostEqual(
            fc_info["fc_variation_sq_norm"],
            1.0,
            places=12,
        )

    def test_h2_at_600_kw_is_normalized_to_one(self) -> None:
        _, info = calculate_mpc_weight_reward(
            p_fc_kw=600.0,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=600.0,
        )

        self.assertAlmostEqual(
            info["h2_norm"],
            1.0,
            places=12,
        )

    def test_h2_term_matches_mpc_quadratic_fit(self) -> None:
        p_fc_kw = 280.0

        _, info = calculate_mpc_weight_reward(
            p_fc_kw=p_fc_kw,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=p_fc_kw,
        )

        a1, a2 = dp0_quadratic_coefficients()
        relative_power = p_fc_kw / 600.0

        expected = (
            a1 * relative_power
            + a2 * relative_power**2
        ) / (a1 + a2)

        self.assertAlmostEqual(
            info["h2_norm"],
            expected,
            places=12,
        )

    def test_total_reward_is_negative_weighted_sum(self) -> None:
        reward, info = calculate_mpc_weight_reward(
            p_fc_kw=280.0,
            p_batt_kw=312.0,
            next_soc=0.60,
            previous_fc_kw=232.0,
        )

        expected_cost = (
            REWARD_Q_H2 * info["h2_norm"]
            + REWARD_Q_BATT
            * info["battery_power_sq_norm"]
            + REWARD_Q_SOC
            * info["soc_tracking_sq_norm"]
            + REWARD_Q_FC_VAR
            * info["fc_variation_sq_norm"]
        )

        self.assertAlmostEqual(
            info["total_cost"],
            expected_cost,
            places=12,
        )

        self.assertAlmostEqual(
            reward,
            -expected_cost,
            places=12,
        )

    def test_non_finite_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_mpc_weight_reward(
                p_fc_kw=np.nan,
                p_batt_kw=0.0,
                next_soc=SOC_REFERENCE,
                previous_fc_kw=0.0,
            )


if __name__ == "__main__":
    unittest.main()