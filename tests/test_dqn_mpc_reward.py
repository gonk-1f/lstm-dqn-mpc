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
    REWARD_Q_FC_VAR_MIN,
    REWARD_Q_H2,
    REWARD_Q_SOC,
    SOC_BAND,
    SOC_MIN,
    SOC_REFERENCE,
    LOAD_DELTA_RISE_REFERENCE_KW,
    calculate_mpc_weight_reward,
)

from mpc.solvers.fc_dp0_curve import (  # noqa: E402
    dp0_quadratic_coefficients,
)


class TestDqnMpcReward(unittest.TestCase):
    def test_fixed_common_reward_weights_are_unchanged(self) -> None:
        self.assertEqual(REWARD_Q_H2, 0.25)
        self.assertEqual(REWARD_Q_BATT, 0.40)
        self.assertEqual(REWARD_Q_SOC, 12.0)
        self.assertEqual(REWARD_Q_FC_VAR, 20.0)
        self.assertEqual(REWARD_Q_FC_VAR_MIN, 8.0)

    def test_fc_variation_weight_is_20_without_gates(self) -> None:
        _, info = calculate_mpc_weight_reward(
            p_fc_kw=FC_VARIATION_REF_KW,
            p_batt_kw=0.0,
            next_soc=SOC_REFERENCE,
            previous_fc_kw=0.0,
            soc_before=SOC_REFERENCE,
            load_delta_kw=0.0,
        )

        self.assertAlmostEqual(
            info["fc_var_weight"],
            REWARD_Q_FC_VAR,
            places=12,
        )

    def test_fc_variation_weight_reaches_floor_for_rise_or_low_soc(self) -> None:
        common_inputs = {
            "p_fc_kw": FC_VARIATION_REF_KW,
            "p_batt_kw": 0.0,
            "next_soc": SOC_REFERENCE,
            "previous_fc_kw": 0.0,
        }
        _, rise_info = calculate_mpc_weight_reward(
            **common_inputs,
            soc_before=SOC_REFERENCE,
            load_delta_kw=LOAD_DELTA_RISE_REFERENCE_KW,
        )
        _, low_soc_info = calculate_mpc_weight_reward(
            **common_inputs,
            soc_before=SOC_MIN,
            load_delta_kw=0.0,
        )

        self.assertAlmostEqual(
            rise_info["g_rise"], 1.0, places=12
        )
        self.assertAlmostEqual(
            rise_info["fc_var_weight"],
            REWARD_Q_FC_VAR_MIN,
            places=12,
        )
        self.assertAlmostEqual(
            low_soc_info["g_low"], 1.0, places=12
        )
        self.assertAlmostEqual(
            low_soc_info["fc_var_weight"],
            REWARD_Q_FC_VAR_MIN,
            places=12,
        )

    def test_fc_variation_weight_always_stays_within_configured_range(self) -> None:
        for soc_before in (0.0, SOC_MIN, 0.35, SOC_REFERENCE, 0.9):
            for load_delta_kw in (-10.0, 0.0, 1.0, LOAD_DELTA_RISE_REFERENCE_KW, 100.0):
                _, info = calculate_mpc_weight_reward(
                    p_fc_kw=FC_VARIATION_REF_KW,
                    p_batt_kw=0.0,
                    next_soc=SOC_REFERENCE,
                    previous_fc_kw=0.0,
                    soc_before=soc_before,
                    load_delta_kw=load_delta_kw,
                )
                self.assertGreaterEqual(
                    info["fc_var_weight"],
                    REWARD_Q_FC_VAR_MIN,
                )
                self.assertLessEqual(
                    info["fc_var_weight"],
                    REWARD_Q_FC_VAR,
                )
                self.assertEqual(
                    info["reward_weights"]["q_fc_var_min"],
                    REWARD_Q_FC_VAR_MIN,
                )

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
