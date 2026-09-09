from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"

for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import train_dqn_mpc_mlp as training  # noqa: E402
from dqn.agents.dqn_agent import DQNTrainConfig  # noqa: E402


SEED = 321
REQUIRED_API = (
    "DEFAULT_OPERATING_DATASET_ROOT",
    "DEFAULT_SPLIT_MANIFEST",
    "VoyageSplit",
    "TrainingRuntime",
    "load_voyage_split",
    "load_operating_segment_loads",
    "build_formal_mpc_config",
    "create_training_runtime",
    "run_training_episode",
    "train_to_budget",
    "train_complete_voyage_rounds",
    "validate_voyages",
)


def parameter_snapshot(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
    }


def snapshots_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return (
        left.keys() == right.keys()
        and all(
            torch.equal(left[name], right[name])
            for name in left
        )
    )


def short_loads(
    *,
    start_kw: float,
    samples: int,
) -> np.ndarray:
    return np.linspace(
        start_kw,
        start_kw + 5.0 * (samples - 1),
        samples,
        dtype=np.float64,
    )


class TestDqnMpcMlpTraining(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    def require_api(self) -> None:
        missing = [
            name
            for name in REQUIRED_API
            if not hasattr(training, name)
        ]
        self.assertEqual(missing, [])

    def make_config(
        self,
        **changes: object,
    ) -> DQNTrainConfig:
        config = DQNTrainConfig(device="cpu")
        return replace(config, **changes)

    def test_formal_split_is_segment_based_and_disjoint(
        self,
    ) -> None:
        self.require_api()
        split = training.load_voyage_split(training.DEFAULT_SPLIT_MANIFEST)

        self.assertEqual(len(split.train_segments), 110)
        self.assertEqual(len(split.validation_segments), 27)
        self.assertEqual(len(split.test_segments), 8)
        self.assertEqual(len(split.train_parents), 45)
        self.assertEqual(len(split.validation_parents), 13)
        self.assertEqual(len(split.test_parents), 7)

        train = set(split.train_segments)
        validation = set(split.validation_segments)
        test = set(split.test_segments)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), 145)
        self.assertTrue(
            all(identifier.startswith(("train_parent_", "validation_parent_", "test_parent_")) for identifier in train | validation | test)
        )

    def test_effective_training_and_validation_exclude_only_physical_stress_cases(
        self,
    ) -> None:
        split = training.load_voyage_split(training.DEFAULT_SPLIT_MANIFEST)

        self.assertEqual(split.effective_train_segments, split.train_segments)
        self.assertEqual(split.effective_validation_segments, split.validation_segments)
        self.assertEqual(training.PHYSICAL_INFEASIBLE_STRESS_CASES, {})

        statistics = training.effective_split_statistics(split)
        self.assertEqual(statistics["train"], {"segment_count": 110, "point_count": 521376})
        self.assertEqual(statistics["validation"], {"segment_count": 27, "point_count": 167251})

    def test_formal_configuration_uses_long_horizon_gamma(self) -> None:
        runtime = training.create_training_runtime(self.make_config())

        self.assertEqual(runtime.config.gamma, 0.9995)
        self.assertEqual(runtime.agent.discount, 0.9995)

    def test_loader_reads_train_and_rejects_test_before_io(
        self,
    ) -> None:
        self.require_api()
        split = training.load_voyage_split(training.DEFAULT_SPLIT_MANIFEST)

        with self.assertRaisesRegex(ValueError, "test"):
            training.load_operating_segment_loads(
                "test",
                split.test_segments[0],
                split=split,
            )

        loads = training.load_operating_segment_loads(
            "train",
            split.train_segments[0],
            split=split,
        )
        self.assertGreaterEqual(float(loads.min()), 0.0)
        self.assertGreaterEqual(len(loads), 2)

    def test_runtime_locks_formal_mlp_design_and_sync_interval(
        self,
    ) -> None:
        self.require_api()
        runtime = training.create_training_runtime(
            self.make_config()
        )

        self.assertEqual(training.STATE_DIM, 7)
        self.assertEqual(
            training.ACTION_DIM,
            len(training.DQN_MPC_WEIGHT_ACTIONS),
        )
        self.assertEqual(
            tuple(runtime.config.mlp_hidden_dims),
            (128, 64),
        )
        self.assertEqual(
            [
                (
                    layer.in_features,
                    layer.out_features,
                )
                for layer in runtime.agent.q_net.layers
            ],
            [
                (7, 128),
                (128, 64),
                (64, training.ACTION_DIM),
            ],
        )
        self.assertFalse(
            runtime.config.state_normalization_enabled
        )
        self.assertFalse(runtime.config.double_dqn)
        self.assertFalse(runtime.config.dueling_dqn)

        with self.assertRaisesRegex(
            ValueError,
            "target_sync_interval",
        ):
            training.create_training_runtime(
                self.make_config(
                    target_sync_interval=0
                )
            )

    def test_two_complete_episodes_have_no_cross_voyage_transition(
        self,
    ) -> None:
        self.require_api()
        config = self.make_config(
            batch_size=2,
            warmup_steps=100,
            buffer_size=100,
            target_sync_interval=1000,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()
        initial_online = parameter_snapshot(
            runtime.agent.q_net
        )

        first = training.run_training_episode(
            voyage_id="fixture_train_1",
            loads_kw=short_loads(
                start_kw=220.0,
                samples=5,
            ),
            base_config=base_config,
            runtime=runtime,
        )
        second = training.run_training_episode(
            voyage_id="fixture_train_2",
            loads_kw=short_loads(
                start_kw=260.0,
                samples=4,
            ),
            base_config=base_config,
            runtime=runtime,
        )

        self.assertEqual(first["episode_steps"], 4)
        self.assertEqual(second["episode_steps"], 3)
        self.assertEqual(
            sum(first[f"action_count_A{action_id}"] for action_id in range(4)),
            first["episode_steps"],
        )
        self.assertEqual(runtime.global_step, 7)
        self.assertEqual(len(runtime.replay_buffer), 7)
        self.assertEqual(
            runtime.replay_buffer.dones,
            [
                False,
                False,
                False,
                True,
                False,
                False,
                True,
            ],
        )
        self.assertEqual(runtime.update_steps, [])
        self.assertTrue(
            snapshots_equal(
                initial_online,
                parameter_snapshot(runtime.agent.q_net),
            )
        )

    def test_warmup_then_first_post_warmup_step_updates_online(
        self,
    ) -> None:
        self.require_api()
        config = self.make_config(
            batch_size=2,
            warmup_steps=2,
            buffer_size=100,
            target_sync_interval=100,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()
        initial_online = parameter_snapshot(
            runtime.agent.q_net
        )
        initial_target = parameter_snapshot(
            runtime.agent.target_q_net
        )

        training.run_training_episode(
            voyage_id="fixture_warmup",
            loads_kw=short_loads(
                start_kw=230.0,
                samples=3,
            ),
            base_config=base_config,
            runtime=runtime,
        )

        self.assertEqual(runtime.global_step, 2)
        self.assertEqual(len(runtime.replay_buffer), 2)
        self.assertEqual(runtime.update_steps, [])
        self.assertTrue(
            snapshots_equal(
                initial_online,
                parameter_snapshot(runtime.agent.q_net),
            )
        )

        training.run_training_episode(
            voyage_id="fixture_update",
            loads_kw=np.asarray([245.0, 250.0]),
            base_config=base_config,
            runtime=runtime,
        )

        self.assertEqual(runtime.global_step, 3)
        self.assertEqual(runtime.update_steps, [3])
        self.assertEqual(len(runtime.losses), 1)
        self.assertTrue(np.isfinite(runtime.losses[0]))
        self.assertFalse(
            snapshots_equal(
                initial_online,
                parameter_snapshot(runtime.agent.q_net),
            )
        )
        self.assertTrue(
            snapshots_equal(
                initial_target,
                parameter_snapshot(
                    runtime.agent.target_q_net
                ),
            )
        )

    def test_target_sync_uses_completed_global_step_multiples(
        self,
    ) -> None:
        self.require_api()
        config = self.make_config(
            batch_size=1,
            warmup_steps=0,
            buffer_size=100,
            target_sync_interval=2,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()

        with patch.object(
            runtime.agent,
            "sync_target_network",
            wraps=runtime.agent.sync_target_network,
        ) as sync_target:
            training.run_training_episode(
                voyage_id="fixture_sync",
                loads_kw=short_loads(
                    start_kw=235.0,
                    samples=6,
                ),
                base_config=base_config,
                runtime=runtime,
            )

        self.assertEqual(
            runtime.update_steps,
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            runtime.target_sync_steps,
            [2, 4],
        )
        self.assertEqual(sync_target.call_count, 2)
        self.assertFalse(
            snapshots_equal(
                parameter_snapshot(runtime.agent.q_net),
                parameter_snapshot(
                    runtime.agent.target_q_net
                ),
            )
        )

    def test_budget_checks_only_between_ordered_episodes(
        self,
    ) -> None:
        self.require_api()
        config = self.make_config(
            batch_size=2,
            warmup_steps=100,
            buffer_size=100,
            target_sync_interval=1000,
            max_steps=7,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()
        loads_by_voyage = {
            "fixture_a": short_loads(
                start_kw=220.0,
                samples=4,
            ),
            "fixture_b": short_loads(
                start_kw=260.0,
                samples=4,
            ),
        }

        episodes = training.train_to_budget(
            voyage_ids=("fixture_a", "fixture_b"),
            load_voyage=loads_by_voyage.__getitem__,
            base_config=base_config,
            runtime=runtime,
        )

        self.assertEqual(
            [
                episode["voyage_id"]
                for episode in episodes
            ],
            ["fixture_a", "fixture_b", "fixture_a"],
        )
        self.assertEqual(runtime.global_step, 9)
        self.assertGreater(
            runtime.global_step,
            config.max_steps,
        )
        self.assertEqual(
            runtime.replay_buffer.dones,
            [
                False,
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                True,
            ],
        )

    def test_complete_voyage_rounds_repeat_without_runtime_reset(self) -> None:
        self.require_api()
        runtime = training.create_training_runtime(self.make_config())
        runtime_id = id(runtime)
        calls: list[str] = []

        def fake_episode(*, voyage_id, loads_kw, base_config, runtime):
            self.assertEqual(id(runtime), runtime_id)
            calls.append(voyage_id)
            runtime.global_step += len(loads_kw) - 1
            return {"voyage_id": voyage_id, "episode_steps": len(loads_kw) - 1}

        loads = {"fixture_a": np.array([1.0, 2.0]), "fixture_b": np.array([3.0, 4.0])}
        with patch.object(training, "run_training_episode", side_effect=fake_episode):
            rounds = training.train_complete_voyage_rounds(
                num_training_rounds=2,
                voyage_ids=("fixture_a", "fixture_b"),
                load_voyage=loads.__getitem__,
                base_config=training.build_formal_mpc_config(),
                runtime=runtime,
            )

        self.assertEqual(calls, ["fixture_a", "fixture_b", "fixture_a", "fixture_b"])
        self.assertEqual([round_['completed_training_episodes'] for round_ in rounds], [2, 4])

    def test_round_summary_aggregates_training_stability_metrics(self) -> None:
        runtime = training.create_training_runtime(self.make_config())
        runtime.agent.latest_update_diagnostics = {
            "q_value_mean": 1.0,
            "q_value_std": 0.2,
            "target_q_mean": 0.8,
            "target_q_std": 0.1,
        }

        def fake_episode(*, voyage_id, loads_kw, base_config, runtime):
            runtime.global_step += 2
            runtime.record_loss(torch.tensor(0.5))
            runtime.record_loss(torch.tensor(0.25))
            runtime.flush_loss_metrics()
            runtime.update_steps.extend([runtime.global_step - 1, runtime.global_step])
            return {
                "voyage_id": voyage_id,
                "episode_steps": 2,
                "solver_failure_count": 0,
                "min_soc": 0.49,
                "final_soc": 0.51,
                "action_count_A0": 1,
                "action_count_A1": 1,
                "action_count_A2": 0,
                "action_count_A3": 0,
            }

        with patch.object(training, "run_training_episode", side_effect=fake_episode):
            summary = training.train_complete_voyage_rounds(
                num_training_rounds=1,
                voyage_ids=("fixture",),
                load_voyage=lambda _: np.asarray([1.0, 2.0, 3.0]),
                base_config=training.build_formal_mpc_config(),
                runtime=runtime,
            )[0]

        self.assertEqual(summary["training_action_counts"], {"A0": 1, "A1": 1, "A2": 0, "A3": 0})
        self.assertEqual(summary["training_solver_failure_count"], 0)
        self.assertEqual(summary["loss_statistics"]["median"], 0.375)
        self.assertEqual(summary["q_value_diagnostics"]["q_value_mean"], 1.0)

    def test_validation_is_greedy_and_has_no_learning_side_effects(
        self,
    ) -> None:
        self.require_api()
        config = self.make_config(
            batch_size=2,
            warmup_steps=10,
            buffer_size=100,
            target_sync_interval=100,
        )
        runtime = training.create_training_runtime(config)
        base_config = training.build_formal_mpc_config()

        dummy_state = np.zeros(7, dtype=np.float32)
        runtime.replay_buffer.push(
            dummy_state,
            0,
            -1.0,
            True,
            dummy_state.copy(),
        )
        runtime.policy.step()

        with torch.no_grad():
            for parameter in runtime.agent.q_net.parameters():
                parameter.zero_()
            runtime.agent.q_net.layers[-1].bias[3] = 1.0

        online_before = parameter_snapshot(
            runtime.agent.q_net
        )
        target_before = parameter_snapshot(
            runtime.agent.target_q_net
        )
        replay_size_before = len(runtime.replay_buffer)
        epsilon_before = runtime.policy.epsilon
        global_step_before = runtime.global_step

        loads_by_voyage = {
            "fixture_val_1": short_loads(
                start_kw=225.0,
                samples=4,
            ),
            "fixture_val_2": short_loads(
                start_kw=250.0,
                samples=5,
            ),
        }
        validation = training.validate_voyages(
            voyage_ids=tuple(loads_by_voyage),
            load_voyage=loads_by_voyage.__getitem__,
            base_config=base_config,
            agent=runtime.agent,
        )

        self.assertEqual(
            len(validation["voyages"]),
            2,
        )
        total_steps = 0
        for voyage in validation["voyages"]:
            count_sum = sum(
                voyage[f"action_count_A{action_id}"]
                for action_id in range(training.ACTION_DIM)
            )
            self.assertEqual(
                count_sum,
                voyage["episode_steps"],
            )
            self.assertEqual(
                voyage["action_count_A3"],
                voyage["episode_steps"],
            )
            total_steps += voyage["episode_steps"]

        self.assertEqual(
            validation["action_fraction_A3"],
            1.0,
        )
        self.assertEqual(
            sum(
                validation[
                    f"action_fraction_A{action_id}"
                ]
                for action_id in range(training.ACTION_DIM)
            ),
            1.0,
        )
        self.assertEqual(total_steps, 7)
        self.assertTrue(
            np.isfinite(
                validation["mean_episode_reward"]
            )
        )
        self.assertTrue(
            np.isfinite(
                validation["mean_reward_per_step"]
            )
        )

        self.assertEqual(
            len(runtime.replay_buffer),
            replay_size_before,
        )
        self.assertEqual(
            runtime.policy.epsilon,
            epsilon_before,
        )
        self.assertEqual(
            runtime.global_step,
            global_step_before,
        )
        self.assertTrue(
            snapshots_equal(
                online_before,
                parameter_snapshot(runtime.agent.q_net),
            )
        )
        self.assertTrue(
            snapshots_equal(
                target_before,
                parameter_snapshot(
                    runtime.agent.target_q_net
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
