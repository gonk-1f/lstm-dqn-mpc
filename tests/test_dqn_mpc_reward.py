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
from dqn.utils import legacy_reward as reward_module  # noqa: E402
from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402


class TestDqnMpcReward(unittest.TestCase):
    def test_reward_interface_uses_only_selected_mpc_objective(self) -> None:
        parameters = inspect.signature(
            reward_module.legacy_yuan_like_self_cost
        ).parameters
        self.assertEqual(
            list(parameters),
            ["raw_mpc_objective"],
        )
        self.assertTrue(
            all(
                parameter.kind is inspect.Parameter.KEYWORD_ONLY
                for parameter in parameters.values()
            )
        )
        source = inspect.getsource(reward_module.legacy_yuan_like_self_cost)
        self.assertNotIn("action_weights", source)
        self.assertNotIn("mpc_action_weight_sum", source)

    def test_final_action_weight_sums_are_one(self) -> None:
        expected = np.ones(84)
        actual = tuple(
            reward_module.mpc_action_weight_sum(action.as_tuple())
            for action in DQN_MPC_WEIGHT_ACTIONS
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_reward_is_inverse_one_plus_mpc_objective(self) -> None:
        reward, info = reward_module.legacy_yuan_like_self_cost(
            raw_mpc_objective=2.0,
        )
        self.assertNotIn("weight_sum", info)
        self.assertNotIn("normalized_objective", info)
        self.assertAlmostEqual(info["raw_mpc_objective"], 2.0, places=12)
        self.assertAlmostEqual(reward, 1.0 / 3.0, places=12)
        self.assertAlmostEqual(info["total_reward"], reward, places=12)

    def test_lower_objective_has_higher_reward(self) -> None:
        reward_low, _ = reward_module.legacy_yuan_like_self_cost(
            raw_mpc_objective=10.0,
        )
        reward_high, _ = reward_module.legacy_yuan_like_self_cost(
            raw_mpc_objective=20.0,
        )
        self.assertGreater(reward_low, reward_high)

    def test_zero_objective_returns_one_and_is_finite(self) -> None:
        reward, info = reward_module.legacy_yuan_like_self_cost(
            raw_mpc_objective=0.0,
        )
        self.assertEqual(reward, 1.0)
        self.assertEqual(info["raw_mpc_objective"], 0.0)
        self.assertTrue(np.isfinite(reward))

    def test_success_reward_is_strictly_positive_and_at_most_one(self) -> None:
        for objective in (0.0, 1.0e-12, 1.0, 1.0e6):
            with self.subTest(objective=objective):
                reward, _ = reward_module.legacy_yuan_like_self_cost(
                    raw_mpc_objective=objective,
                )
                self.assertGreater(reward, 0.0)
                self.assertLessEqual(reward, 1.0)
                self.assertTrue(np.isfinite(reward))

    def test_invalid_objective_is_rejected(self) -> None:
        for objective in (np.nan, np.inf, -1.0):
            with self.subTest(objective=objective):
                with self.assertRaises(ValueError):
                    reward_module.legacy_yuan_like_self_cost(
                        raw_mpc_objective=objective,
                    )

    def test_old_common_reward_weights_are_not_exposed(self) -> None:
        for name in (
            "REWARD_Q_H2",
            "REWARD_Q_BATT",
            "REWARD_Q_SOC",
            "REWARD_Q_FC_VAR",
        ):
            self.assertFalse(hasattr(reward_module, name), name)

    def test_terminal_failure_penalty_defaults_uncalibrated(self) -> None:
        self.assertIsNone(DQNTrainConfig().terminal_failure_penalty)


if __name__ == "__main__":
    unittest.main()
