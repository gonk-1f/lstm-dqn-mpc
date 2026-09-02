from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from dqn.agents.dqn_agent import DQNTrainConfig  # noqa: E402
from dqn.utils import reward as reward_module  # noqa: E402
from mpc.solvers.fc_dp0_curve import (  # noqa: E402
    dp0_quadratic_coefficients,
)


class TestDqnMpcReward(unittest.TestCase):
    def test_reward_interface_contains_only_active_inputs(self) -> None:
        parameters = inspect.signature(
            reward_module.calculate_mpc_weight_reward
        ).parameters

        self.assertEqual(
            list(parameters),
            [
                "p_fc_kw",
                "p_batt_kw",
                "next_soc",
                "previous_fc_kw",
            ],
        )
        self.assertTrue(
            all(
                parameter.kind
                is inspect.Parameter.KEYWORD_ONLY
                for parameter in parameters.values()
            )
        )

    def test_fixed_common_reward_weights_match_final_formula(self) -> None:
        self.assertEqual(reward_module.REWARD_Q_H2, 0.25)
        self.assertEqual(reward_module.REWARD_Q_BATT, 0.40)
        self.assertEqual(reward_module.REWARD_Q_FC_VAR, 20.0)
        self.assertFalse(hasattr(reward_module, "REWARD_Q_SOC"))

    def test_soc_soft_penalty_matches_required_scale(self) -> None:
        penalty = getattr(
            reward_module,
            "soc_soft_working_range_penalty",
            None,
        )
        self.assertTrue(callable(penalty))

        expected = {
            0.45: 1.0,
            0.48: 0.16,
            0.49: 0.04,
            0.50: 0.0,
            0.55: 0.0,
            0.60: 0.0,
            0.61: 0.04,
            0.62: 0.16,
            0.65: 1.0,
        }
        for soc, expected_penalty in expected.items():
            with self.subTest(soc=soc):
                self.assertAlmostEqual(
                    penalty(soc),
                    expected_penalty,
                    places=12,
                )

    def test_soc_soft_penalty_is_symmetric(self) -> None:
        penalty = (
            reward_module.soc_soft_working_range_penalty
        )
        self.assertAlmostEqual(
            penalty(0.49), penalty(0.61), places=12
        )
        self.assertAlmostEqual(
            penalty(0.45), penalty(0.65), places=12
        )

    def test_soc_soft_penalty_is_zero_through_closed_range(
        self,
    ) -> None:
        penalty = (
            reward_module.soc_soft_working_range_penalty
        )
        for soc in np.linspace(0.50, 0.60, 21):
            with self.subTest(soc=float(soc)):
                self.assertEqual(penalty(float(soc)), 0.0)

    def test_zero_objective_terms_give_zero_reward(self) -> None:
        reward, info = (
            reward_module.calculate_mpc_weight_reward(
                p_fc_kw=0.0,
                p_batt_kw=0.0,
                next_soc=0.55,
                previous_fc_kw=0.0,
            )
        )

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["h2_norm"], 0.0)
        self.assertEqual(info["battery_power_sq_norm"], 0.0)
        self.assertEqual(info["phi_soc"], 0.0)
        self.assertEqual(info["fc_variation_sq_norm"], 0.0)

    def test_each_normalized_reference_has_unit_cost(self) -> None:
        _, batt_info = (
            reward_module.calculate_mpc_weight_reward(
                p_fc_kw=0.0,
                p_batt_kw=reward_module.BATTERY_POWER_REF_KW,
                next_soc=0.55,
                previous_fc_kw=0.0,
            )
        )
        self.assertAlmostEqual(
            batt_info["battery_power_sq_norm"], 1.0, places=12
        )

        _, soc_info = (
            reward_module.calculate_mpc_weight_reward(
                p_fc_kw=0.0,
                p_batt_kw=0.0,
                next_soc=0.45,
                previous_fc_kw=0.0,
            )
        )
        self.assertAlmostEqual(
            soc_info["phi_soc"], 1.0, places=12
        )

        _, fc_info = (
            reward_module.calculate_mpc_weight_reward(
                p_fc_kw=reward_module.FC_VARIATION_REF_KW,
                p_batt_kw=0.0,
                next_soc=0.55,
                previous_fc_kw=0.0,
            )
        )
        self.assertAlmostEqual(
            fc_info["fc_variation_sq_norm"], 1.0, places=12
        )

    def test_h2_at_600_kw_is_normalized_to_one(self) -> None:
        _, info = reward_module.calculate_mpc_weight_reward(
            p_fc_kw=600.0,
            p_batt_kw=0.0,
            next_soc=0.55,
            previous_fc_kw=600.0,
        )

        self.assertAlmostEqual(info["h2_norm"], 1.0, places=12)

    def test_h2_term_matches_mpc_quadratic_fit(self) -> None:
        p_fc_kw = 280.0
        _, info = reward_module.calculate_mpc_weight_reward(
            p_fc_kw=p_fc_kw,
            p_batt_kw=0.0,
            next_soc=0.55,
            previous_fc_kw=p_fc_kw,
        )

        a1, a2 = dp0_quadratic_coefficients()
        relative_power = p_fc_kw / 600.0
        expected = (
            a1 * relative_power + a2 * relative_power**2
        ) / (a1 + a2)

        self.assertAlmostEqual(
            info["h2_norm"], expected, places=12
        )

    def test_total_reward_is_exact_negative_four_term_sum(
        self,
    ) -> None:
        reward, info = (
            reward_module.calculate_mpc_weight_reward(
                p_fc_kw=280.0,
                p_batt_kw=312.0,
                next_soc=0.48,
                previous_fc_kw=232.0,
            )
        )

        expected_cost = (
            0.25 * info["h2_norm"]
            + 0.40 * info["battery_power_sq_norm"]
            + info["phi_soc"]
            + 20.0 * info["fc_variation_sq_norm"]
        )
        self.assertAlmostEqual(
            info["total_cost"], expected_cost, places=12
        )
        self.assertEqual(info["weighted_soc"], info["phi_soc"])
        self.assertAlmostEqual(reward, -expected_cost, places=12)

    def test_diagnostics_contain_only_final_reward_terms(
        self,
    ) -> None:
        _, info = reward_module.calculate_mpc_weight_reward(
            p_fc_kw=280.0,
            p_batt_kw=312.0,
            next_soc=0.48,
            previous_fc_kw=232.0,
        )

        self.assertEqual(
            set(info),
            {
                "h2_norm",
                "battery_power_sq_norm",
                "phi_soc",
                "fc_variation_sq_norm",
                "weighted_h2",
                "weighted_batt",
                "weighted_soc",
                "weighted_fc_var",
                "total_cost",
                "total_reward",
            },
        )

    def test_non_finite_active_input_is_rejected(self) -> None:
        valid = {
            "p_fc_kw": 100.0,
            "p_batt_kw": 20.0,
            "next_soc": 0.55,
            "previous_fc_kw": 95.0,
        }
        for name in valid:
            values = dict(valid)
            values[name] = np.nan
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    reward_module.calculate_mpc_weight_reward(
                        **values
                    )

    def test_solver_failure_reward_remains_minus_620(self) -> None:
        self.assertEqual(
            DQNTrainConfig().solver_failure_reward,
            -620.0,
        )


if __name__ == "__main__":
    unittest.main()
