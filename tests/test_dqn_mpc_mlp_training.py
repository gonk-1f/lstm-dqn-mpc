from __future__ import annotations

import sys
import tempfile
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
    "DEFAULT_SPLIT_JSON",
    "DEFAULT_VOYAGE_DATA_DIR",
    "VoyageSplit",
    "TrainingRuntime",
    "load_voyage_split",
    "load_voyage_loads",
    "build_formal_mpc_config",
    "create_training_runtime",
    "run_training_episode",
    "train_to_budget",
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

    def test_formal_split_is_46_13_7_and_disjoint(
        self,
    ) -> None:
        self.require_api()
        split = training.load_voyage_split(
            training.DEFAULT_SPLIT_JSON
        )

        self.assertEqual(len(split.train_voyages), 46)
        self.assertEqual(len(split.validation_voyages), 13)
        self.assertEqual(len(split.test_voyages), 7)
        self.assertEqual(split.excluded_voyages, ())
        self.assertEqual(
            split.formal_1s_directory,
            (
                "outputs/spline_1s_diagnostics/data/"
                "natural_clipped_by_voyage"
            ),
        )
        self.assertEqual(split.target_load, "load_total_kw")
        self.assertEqual(
            split.sample_interval_seconds,
            1.0,
        )

        train = set(split.train_voyages)
        validation = set(split.validation_voyages)
        test = set(split.test_voyages)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), 66)

        self.assertEqual(
            split.train_voyages,
            tuple(
                f"voyage_{index:03d}"
                for index in range(1, 47)
            ),
        )
        self.assertEqual(
            split.validation_voyages,
            tuple(
                f"voyage_{index:03d}"
                for index in range(47, 60)
            ),
        )
        self.assertEqual(
            split.test_voyages,
            tuple(
                f"voyage_{index:03d}"
                for index in range(60, 67)
            ),
        )

    def test_loader_reads_train_and_rejects_test_before_io(
        self,
    ) -> None:
        self.require_api()
        split = training.load_voyage_split(
            training.DEFAULT_SPLIT_JSON
        )

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)

            with self.assertRaisesRegex(
                ValueError,
                "test",
            ):
                training.load_voyage_loads(
                    "test",
                    "voyage_060",
                    split=split,
                    data_dir=data_dir,
                )

            voyage_path = (
                data_dir / "voyage_001__fixture.csv"
            )
            pd.DataFrame(
                {
                    "voyage_id": ["voyage_001"] * 3,
                    "split": ["train"] * 3,
                    "time_s": [0.0, 1.0, 2.0],
                    "load_total_kw": [
                        210.0,
                        215.0,
                        220.0,
                    ],
                }
            ).to_csv(voyage_path, index=False)
            pd.DataFrame(
                {
                    "voyage_id": ["voyage_001"],
                    "split": ["train"],
                    "output_csv": [str(voyage_path)],
                }
            ).to_csv(
                data_dir / "manifest.csv",
                index=False,
            )

            loads = training.load_voyage_loads(
                "train",
                "voyage_001",
                split=split,
                data_dir=data_dir,
            )

        np.testing.assert_array_equal(
            loads,
            np.asarray([210.0, 215.0, 220.0]),
        )

    def test_runtime_locks_formal_mlp_design_and_sync_interval(
        self,
    ) -> None:
        self.require_api()
        runtime = training.create_training_runtime(
            self.make_config()
        )

        self.assertEqual(training.STATE_DIM, 11)
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
                (11, 128),
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

        dummy_state = np.zeros(11, dtype=np.float32)
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
