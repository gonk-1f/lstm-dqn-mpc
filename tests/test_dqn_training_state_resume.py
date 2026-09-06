from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import train_dqn_mpc_mlp as training  # noqa: E402
from dqn.agents.dqn_agent import DQNTrainConfig  # noqa: E402


class TrainingStateResumeTests(unittest.TestCase):
    def _runtime_with_updates(self):
        config = DQNTrainConfig(
            device="cpu",
            buffer_size=5,
            batch_size=2,
            warmup_steps=0,
        )
        runtime = training.create_training_runtime(config)
        for index in range(7):
            state = np.full(7, float(index), dtype=np.float32)
            runtime.replay_buffer.push(
                state,
                index % 4,
                float(index) / 10.0,
                bool(index % 2),
                state + 1.0,
            )
        batch = runtime.replay_buffer.sample(2)
        runtime.agent.update(batch)
        runtime.global_step = 17
        runtime.policy.epsilon = 0.37
        runtime.gradient_update_count = 1
        runtime.target_sync_count = 1
        return runtime

    def test_round_state_restores_network_optimizer_policy_replay_and_rng(self) -> None:
        random.seed(811)
        np.random.seed(811)
        torch.manual_seed(811)
        runtime = self._runtime_with_updates()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training_state_round1.pt"
            training.save_training_state(
                runtime=runtime,
                path=path,
                completed_round=1,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            expected_rng = (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item()),
            )
            restored, completed_round, metadata = training.load_training_state(path)

        self.assertEqual(completed_round, 1)
        self.assertEqual(training.next_round_id(completed_round), 2)
        self.assertEqual(metadata["gamma"], 0.9995)
        self.assertEqual(payload["training_config"]["gamma"], 0.9995)
        self.assertTrue(payload["replay_buffer_saved"])
        self.assertTrue(
            {
                "online_q_network_state_dict",
                "target_q_network_state_dict",
                "optimizer_state_dict",
                "epsilon",
                "global_step",
                "gradient_update_count",
                "random_seed",
                "python_rng_state",
                "numpy_rng_state",
                "torch_cpu_rng_state",
            }.issubset(payload)
        )
        self.assertFalse(path.with_suffix(".pt.tmp").exists())
        for left, right in zip(
            runtime.agent.q_net.state_dict().values(),
            restored.agent.q_net.state_dict().values(),
        ):
            self.assertTrue(torch.equal(left, right))
        for left, right in zip(
            runtime.agent.target_q_net.state_dict().values(),
            restored.agent.target_q_net.state_dict().values(),
        ):
            self.assertTrue(torch.equal(left, right))
        left_optimizer = runtime.agent.optimizer.state_dict()
        right_optimizer = restored.agent.optimizer.state_dict()
        self.assertEqual(left_optimizer["param_groups"], right_optimizer["param_groups"])
        self.assertEqual(left_optimizer["state"].keys(), right_optimizer["state"].keys())
        for parameter_id, left_state in left_optimizer["state"].items():
            right_state = right_optimizer["state"][parameter_id]
            self.assertEqual(left_state.keys(), right_state.keys())
            for key, left_value in left_state.items():
                right_value = right_state[key]
                if isinstance(left_value, torch.Tensor):
                    self.assertTrue(torch.equal(left_value, right_value))
                else:
                    self.assertEqual(left_value, right_value)
        self.assertEqual(restored.global_step, 17)
        self.assertEqual(restored.policy.epsilon, 0.37)
        self.assertEqual(restored.gradient_update_count, 1)
        self.assertEqual(len(restored.replay_buffer), 5)
        self.assertEqual(restored.replay_buffer.size, 7)
        self.assertEqual(restored.replay_buffer.write_position, 2)
        np.testing.assert_array_equal(restored.replay_buffer.states[0], np.full(7, 5.0, dtype=np.float32))
        np.testing.assert_array_equal(restored.replay_buffer.states[1], np.full(7, 6.0, dtype=np.float32))
        self.assertEqual(
            (random.random(), float(np.random.random()), float(torch.rand(1).item())),
            expected_rng,
        )

    def test_failed_state_save_preserves_existing_complete_checkpoint(self) -> None:
        runtime = self._runtime_with_updates()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "training_state_round1.pt"
            path.write_bytes(b"existing-complete-state")
            with patch.object(torch, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    training.save_training_state(
                        runtime=runtime,
                        path=path,
                        completed_round=1,
                    )
            self.assertEqual(path.read_bytes(), b"existing-complete-state")
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_resume_after_round_one_starts_with_round_two(self) -> None:
        runtime = training.create_training_runtime(DQNTrainConfig(device="cpu"))
        calls: list[str] = []

        def fake_episode(*, voyage_id, loads_kw, base_config, runtime):
            calls.append(voyage_id)
            runtime.global_step += 1
            return {"voyage_id": voyage_id, "episode_steps": 1}

        with patch.object(training, "run_training_episode", side_effect=fake_episode):
            summaries = training.train_complete_voyage_rounds(
                num_training_rounds=2,
                first_round_id=training.next_round_id(1),
                voyage_ids=("fixture_train",),
                load_voyage=lambda _: np.asarray([1.0, 2.0]),
                base_config=training.build_formal_mpc_config(),
                runtime=runtime,
            )

        self.assertEqual(calls, ["fixture_train"])
        self.assertEqual([summary["round_id"] for summary in summaries], [2])


if __name__ == "__main__":
    unittest.main()
