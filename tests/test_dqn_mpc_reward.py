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
from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402


class TestDqnMpcReward(unittest.TestCase):
    def test_reward_interface_uses_only_selected_mpc_objective(self) -> None:
        parameters = inspect.signature(
            reward_module.calculate_mpc_weight_reward
        ).parameters
        self.assertEqual(
            list(parameters),
            ["raw_mpc_objective", "action_weights"],
        )
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in parameters.values()
            )
        )

    def test_final_action_weight_sums(self) -> None:
        expected = (56.70, 16.65, 70.75, 128.95)
        actual = tuple(
            reward_module.mpc_action_weight_sum(action.as_tuple())
            for action in DQN_MPC_WEIGHT_ACTIONS
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_reward_is_inverse_one_plus_normalized_objective(self) -> None:
        reward, info = reward_module.calculate_mpc_weight_reward(
            raw_mpc_objective=141.5,
            action_weights=DQN_MPC_WEIGHT_ACTIONS[2].as_tuple(),
        )
        expected_normalized = 141.5 / 70.75
        self.assertAlmostEqual(info["weight_sum"], 70.75, places=12)
        self.assertAlmostEqual(info["raw_mpc_objective"], 141.5, places=12)
        self.assertAlmostEqual(
            info["normalized_objective"], expected_normalized, places=12
        )
        self.assertAlmostEqual(reward, 1.0 / (1.0 + expected_normalized), places=12)
        self.assertAlmostEqual(info["total_reward"], reward, places=12)

    def test_lower_normalized_objective_has_higher_reward(self) -> None:
        weights = DQN_MPC_WEIGHT_ACTIONS[0].as_tuple()
        reward_low, _ = reward_module.calculate_mpc_weight_reward(
            raw_mpc_objective=10.0,
            action_weights=weights,
        )
        reward_high, _ = reward_module.calculate_mpc_weight_reward(
            raw_mpc_objective=20.0,
            action_weights=weights,
        )
        self.assertGreater(reward_low, reward_high)

    def test_zero_normalized_objective_returns_one_and_is_finite(self) -> None:
        reward, info = reward_module.calculate_mpc_weight_reward(
            raw_mpc_objective=0.0,
            action_weights=DQN_MPC_WEIGHT_ACTIONS[3].as_tuple(),
        )
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["normalized_objective"], 0.0)
        self.assertTrue(np.isfinite(reward))

    def test_success_reward_is_strictly_positive_and_at_most_one(self) -> None:
        weights = DQN_MPC_WEIGHT_ACTIONS[1].as_tuple()
        for objective in (0.0, 1.0e-12, 1.0, 1.0e6):
            with self.subTest(objective=objective):
                reward, _ = reward_module.calculate_mpc_weight_reward(
                    raw_mpc_objective=objective,
                    action_weights=weights,
                )
                self.assertGreater(reward, 0.0)
                self.assertLessEqual(reward, 1.0)
                self.assertTrue(np.isfinite(reward))

    def test_invalid_objective_or_weights_are_rejected(self) -> None:
        valid_weights = DQN_MPC_WEIGHT_ACTIONS[0].as_tuple()
        for objective in (np.nan, np.inf, -1.0):
            with self.subTest(objective=objective):
                with self.assertRaises(ValueError):
                    reward_module.calculate_mpc_weight_reward(
                        raw_mpc_objective=objective,
                        action_weights=valid_weights,
                    )
        for weights in (
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 3.0, np.nan),
            (1.0, 2.0, 3.0, -1.0),
            (0.0, 0.0, 0.0, 0.0),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    reward_module.calculate_mpc_weight_reward(
                        raw_mpc_objective=1.0,
                        action_weights=weights,
                    )

    def test_old_common_reward_weights_are_not_exposed(self) -> None:
        for name in (
            "REWARD_Q_H2",
            "REWARD_Q_BATT",
            "REWARD_Q_SOC",
            "REWARD_Q_FC_VAR",
        ):
            self.assertFalse(hasattr(reward_module, name), name)

    def test_solver_failure_reward_remains_minus_620(self) -> None:
        self.assertEqual(DQNTrainConfig().solver_failure_reward, -620.0)


if __name__ == "__main__":
    unittest.main()
