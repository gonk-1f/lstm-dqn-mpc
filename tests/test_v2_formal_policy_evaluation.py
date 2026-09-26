from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


S8 = (0.60, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0, 0.15)


def _episode(sample_id: str, *, steps: int = 1):
    from v2.data.formal_training_dataset import FormalEpisode
    from v2.data.supervisory_rules import OperatingMode

    return FormalEpisode(
        sample_id,
        sample_id,
        "validation",
        tuple(pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=30 * index) for index in range(steps)),
        np.arange(steps, dtype=float) * 30.0,
        np.full(steps, 40.0),
        np.full(steps, 3.0),
        ("synthetic",) * steps,
        np.full(steps, 40.0),
        np.zeros(steps),
        (OperatingMode.ONBOARD.value,) * steps,
        ("synthetic",) * steps,
    )


class _Backend:
    def __init__(self, executed_soc: tuple[float, ...]) -> None:
        self.executed_soc = list(executed_soc)


class _Environment:
    def __init__(self, rows, *, executed_soc=(0.60, 0.59)) -> None:
        self._rows = tuple(rows)
        self._position = 0
        self.backend = _Backend(executed_soc)

    def reset(self):
        self._position = 0
        return S8

    def step(self, action_id: str):
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND, FORMAL_FAILURE_POLICY

        values, failed, executed_mpc_steps = self._rows[self._position]
        self._position += 1
        action = next(
            item for item in FINAL_DQN_ACTION_CATALOG if item.action_id == action_id
        )
        ledger = RawCnyIntervalLedger(*values)
        penalty = FORMAL_FAILURE_POLICY.penalty_score if failed else 0.0
        return MacroTransition(
            state=S8,
            action=action,
            learning_reward=ledger.reward_cny - penalty,
            next_state=S8,
            done=failed or self._position == len(self._rows),
            executed_mpc_steps=executed_mpc_steps,
            ledger=ledger,
            failure_penalty_score=penalty,
            failure_kind=FORMAL_FAILURE_KIND if failed else None,
        )


class TestFormalPolicyEvaluation(unittest.TestCase):
    def assert_nested_equal(self, left, right) -> None:
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            self.assertIsInstance(right, np.ndarray)
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for left_item, right_item in zip(left, right):
                self.assert_nested_equal(left_item, right_item)
        else:
            self.assertEqual(left, right)

    def test_fixed_policy_preserves_order_and_reports_separate_score_parts(self) -> None:
        from v2.evaluation.formal_policy import (
            FixedActionPolicy,
            evaluate_formal_policy,
        )

        episodes = (_episode("a"), _episode("b"))
        completed = _Environment(
            (((12.0, 0.0, 0.0, 0.0), False, 1),),
            executed_soc=(0.60, 0.58),
        )
        # This terminal failure preserves two physical intervals committed
        # before the next solve proved infeasible.
        failed = _Environment(
            (((15.0, 0.0, 0.0, 0.0), True, 2),),
            executed_soc=(0.60, 0.57, 0.56),
        )
        environments = (
            (completed.backend, completed),
            (failed.backend, failed),
        )

        with mock.patch(
            "v2.evaluation.formal_policy.build_formal_environment",
            side_effect=environments,
        ):
            result = evaluate_formal_policy(
                episodes=episodes,
                policy=FixedActionPolicy("w_8_1_1"),
            )

        self.assertEqual(result.policy_id, "w_8_1_1")
        self.assertEqual(result.episode_ids, ("a", "b"))
        self.assertEqual(result.completed_episodes, 1)
        self.assertEqual(result.failed_episodes, 1)
        self.assertEqual(result.raw_economic_cost_cny, 27.0)
        self.assertEqual(result.failure_penalty_score, 50_000.0)
        self.assertEqual(result.learning_reward, -50_027.0)
        self.assertEqual(result.action_counts, (("w_8_1_1", 2),))
        self.assertEqual(result.transition_count, 2)
        self.assertEqual(result.executed_mpc_steps, 3)
        self.assertEqual(result.episodes[0].soc_min, 0.58)
        self.assertEqual(result.episodes[0].soc_max, 0.60)
        with self.assertRaises(FrozenInstanceError):
            result.episodes[0].completed = False

    def test_policy_boundary_rejects_unknown_actions_and_nonfinite_s8(self) -> None:
        from v2.evaluation.formal_policy import FixedActionPolicy, GreedyDqnPolicy

        with self.assertRaisesRegex(ValueError, "unknown action"):
            FixedActionPolicy("w_0_0_0")
        fixed = FixedActionPolicy("w_8_1_1")
        with self.assertRaisesRegex(ValueError, "finite S8"):
            fixed.action_index(S8[:-1] + (float("nan"),))

        class Agent:
            @staticmethod
            def greedy_action(state):
                return 0

        greedy = GreedyDqnPolicy(Agent())
        with self.assertRaisesRegex(ValueError, "finite S8"):
            greedy.action_index((0.0,) * 7)

    def test_public_constructors_reject_invalid_counts_order_and_greedy_index(self) -> None:
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.evaluation.formal_policy import EpisodeEvaluation, GreedyDqnPolicy

        valid = {
            "sample_id": "episode",
            "completed": True,
            "failure_kind": None,
            "transition_count": 2,
            "executed_mpc_steps": 2,
            "h2_cost_cny": 1.0,
            "fc_degradation_cost_cny": 2.0,
            "battery_degradation_cost_cny": 3.0,
            "shore_cost_cny": 4.0,
            "raw_economic_cost_cny": 10.0,
            "failure_penalty_score": 0.0,
            "learning_reward": -10.0,
            "soc_min": 0.50,
            "soc_max": 0.60,
            "action_counts": (("w_1_1_8", 2),),
        }
        invalid_counts = (
            (
                "zero transition count",
                {"transition_count": 0, "action_counts": ()},
                "positive integer",
            ),
            (
                "mismatched action total",
                {"action_counts": (("w_1_1_8", 1),)},
                "equal transition_count",
            ),
            (
                "noncanonical action order",
                {"action_counts": (("w_8_1_1", 1), ("w_1_1_8", 1))},
                "canonical action-catalog order",
            ),
        )
        for name, overrides, message in invalid_counts:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                EpisodeEvaluation(**(valid | overrides))

        for invalid_index in (-1, len(FINAL_DQN_ACTION_CATALOG), True, 1.5):
            class Agent:
                @staticmethod
                def greedy_action(state, value=invalid_index):
                    return value

            with self.subTest(invalid_index=invalid_index), self.assertRaisesRegex(
                ValueError, "canonical action index"
            ):
                GreedyDqnPolicy(Agent()).action_index(S8)

    def test_greedy_evaluation_calls_only_greedy_action_and_preserves_agent_state(self) -> None:
        from v2.evaluation.formal_policy import GreedyDqnPolicy, evaluate_formal_policy
        from v2.training.dqn import DqnAgent, DqnTrainingConfig

        agent = DqnAgent(DqnTrainingConfig.formal_baseline(), seed=19, device="cpu")
        original_greedy = agent.greedy_action
        agent.greedy_action = mock.Mock(wraps=original_greedy)
        agent.select_action = mock.Mock(side_effect=AssertionError("select_action called"))
        agent.optimize = mock.Mock(side_effect=AssertionError("optimize called"))
        agent.replay.append = mock.Mock(side_effect=AssertionError("replay append called"))
        before = copy.deepcopy(agent.state_dict())
        environment = _Environment(
            (
                ((1.0, 2.0, 3.0, 4.0), False, 1),
                ((2.0, 3.0, 4.0, 5.0), False, 1),
            ),
            executed_soc=(0.60, 0.59, 0.58),
        )

        with mock.patch(
            "v2.evaluation.formal_policy.build_formal_environment",
            return_value=(environment.backend, environment),
        ):
            result = evaluate_formal_policy(
                episodes=(_episode("greedy", steps=2),),
                policy=GreedyDqnPolicy(agent),
            )

        after = copy.deepcopy(agent.state_dict())
        self.assertEqual(agent.greedy_action.call_count, 2)
        agent.select_action.assert_not_called()
        agent.optimize.assert_not_called()
        agent.replay.append.assert_not_called()
        self.assert_nested_equal(before, after)
        self.assertEqual(result.transition_count, 2)
        self.assertEqual(sum(count for _, count in result.action_counts), 2)

    def test_greedy_action_counts_follow_exact_canonical_order(self) -> None:
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.evaluation.formal_policy import GreedyDqnPolicy, evaluate_formal_policy

        action_indices = {
            action.action_id: index
            for index, action in enumerate(FINAL_DQN_ACTION_CATALOG)
        }

        class AlternatingAgent:
            def __init__(self) -> None:
                self.indices = iter(
                    (
                        action_indices["w_8_1_1"],
                        action_indices["w_1_1_8"],
                        action_indices["w_8_1_1"],
                    )
                )
                self.calls = 0

            def greedy_action(self, state) -> int:
                self.calls += 1
                return next(self.indices)

        agent = AlternatingAgent()
        environment = _Environment(
            (
                ((1.0, 0.0, 0.0, 0.0), False, 1),
                ((2.0, 0.0, 0.0, 0.0), False, 1),
                ((3.0, 0.0, 0.0, 0.0), False, 1),
            ),
            executed_soc=(0.60, 0.59, 0.58, 0.57),
        )

        with mock.patch(
            "v2.evaluation.formal_policy.build_formal_environment",
            return_value=(environment.backend, environment),
        ):
            result = evaluate_formal_policy(
                episodes=(_episode("alternating", steps=3),),
                policy=GreedyDqnPolicy(agent),
            )

        self.assertEqual(agent.calls, 3)
        self.assertEqual(
            result.action_counts,
            (("w_1_1_8", 1), ("w_8_1_1", 2)),
        )

    def test_backend_soc_snapshot_contains_initial_and_each_committed_interval(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment
        from v2.evaluation.formal_policy import FixedActionPolicy, evaluate_formal_policy

        class Solver:
            @staticmethod
            def solve(**kwargs):
                load = float(kwargs["observed_load_kw"])
                soc = float(kwargs["current_soc"])

                class Command:
                    p_fc_kw = load
                    p_batt_bus_kw = 0.0
                    predicted_next_soc = soc

                class Plan:
                    @staticmethod
                    def first_command():
                        return Command()

                return Plan()

        episode = _episode("soc", steps=2)
        backend = FormalEpisodeBackend(
            load_kw=episode.load_kw,
            speed_kn=episode.speed_kn,
            fc_power_kw=episode.fc_power_kw,
            battery_bus_kw=episode.battery_bus_kw,
            operating_mode=episode.operating_mode,
            mpc=Solver(),
        )
        environment = MultiRateWeightEnvironment(
            timescale=TimeScaleConfig.formal_baseline(),
            action_catalog=FINAL_DQN_ACTION_CATALOG,
            backend=backend,
            state_provider=backend.state,
            formal_training_mode=True,
        )

        with mock.patch(
            "v2.evaluation.formal_policy.build_formal_environment",
            return_value=(backend, environment),
        ):
            result = evaluate_formal_policy(
                episodes=(episode,), policy=FixedActionPolicy("w_8_1_1")
            )

        self.assertEqual(backend.executed_soc, [0.60, 0.60, 0.60])
        self.assertTrue(np.isfinite(np.asarray(backend.executed_soc)).all())
        self.assertGreaterEqual(result.episodes[0].soc_min, 0.20)
        self.assertLessEqual(result.episodes[0].soc_max, 0.80)
        frozen_bounds = (result.episodes[0].soc_min, result.episodes[0].soc_max)
        backend.executed_soc[1] = 0.20
        backend.executed_soc.append(0.80)
        self.assertEqual(
            (result.episodes[0].soc_min, result.episodes[0].soc_max),
            frozen_bounds,
        )


if __name__ == "__main__":
    unittest.main()
