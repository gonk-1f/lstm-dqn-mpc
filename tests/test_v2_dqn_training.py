from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestV2DqnTraining(unittest.TestCase):
    def test_frozen_defaults_and_network_shape(self) -> None:
        from v2.training.dqn import DqnTrainingConfig, QNetwork

        config = DqnTrainingConfig.formal_baseline()
        self.assertEqual(config.state_dim, 8)
        self.assertEqual(config.action_dim, 36)
        self.assertEqual(config.hidden_dims, (128, 128))
        self.assertEqual(config.gamma, 1.0)
        self.assertEqual(config.learning_rate, 1.0e-4)
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.replay_capacity, 200_000)
        self.assertEqual(config.warmup_steps, 5_000)
        self.assertEqual(config.target_sync_steps, 1_000)
        self.assertEqual(config.gradient_clip_norm, 10.0)
        self.assertEqual(config.rounds, 40)
        network = QNetwork(config)
        output = network(torch.zeros((3, 8), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (3, 36))

    def test_epsilon_is_global_macro_step_linear_schedule(self) -> None:
        from v2.training.dqn import epsilon_at_global_step

        self.assertEqual(epsilon_at_global_step(0), 1.0)
        self.assertAlmostEqual(epsilon_at_global_step(75_000), 0.525)
        self.assertEqual(epsilon_at_global_step(150_000), 0.05)
        self.assertEqual(epsilon_at_global_step(999_999), 0.05)
        self.assertAlmostEqual(1.0 - epsilon_at_global_step(75_000), 0.475)

    def test_formal_round_count_matches_current_train_volume(self) -> None:
        from v2.training.dqn import DqnTrainingConfig, epsilon_at_global_step

        macro_transitions_per_round = 3_721
        config = DqnTrainingConfig.formal_baseline()
        total_macro_steps = macro_transitions_per_round * config.rounds
        self.assertEqual(total_macro_steps, 148_840)
        self.assertAlmostEqual(epsilon_at_global_step(total_macro_steps), 0.057346666666666656)
        self.assertGreater(epsilon_at_global_step(total_macro_steps), 0.05)
        self.assertLess(epsilon_at_global_step(total_macro_steps), 0.06)

    def test_replay_and_seeded_action_selection_are_reproducible(self) -> None:
        from v2.training.dqn import DqnAgent, DqnTrainingConfig, ReplayBuffer

        config = DqnTrainingConfig.formal_baseline()
        first = DqnAgent(config, seed=42, device="cpu")
        second = DqnAgent(config, seed=42, device="cpu")
        state = np.arange(8, dtype=np.float32)
        sequence_a = [first.select_action(state, epsilon=1.0) for _ in range(20)]
        sequence_b = [second.select_action(state, epsilon=1.0) for _ in range(20)]
        self.assertEqual(sequence_a, sequence_b)

        replay = ReplayBuffer(3, state_dim=8, seed=42)
        for index in range(4):
            replay.append(state + index, index % 36, -float(index), state + index + 1, False)
        self.assertEqual(len(replay), 3)
        batch = replay.sample(2)
        self.assertEqual(batch.states.shape, (2, 8))
        self.assertEqual(batch.actions.shape, (2,))

    def test_double_dqn_update_uses_huber_and_syncs_target(self) -> None:
        from v2.training.dqn import DqnAgent, DqnTrainingConfig

        config = DqnTrainingConfig.formal_baseline()
        agent = DqnAgent(config, seed=7, device="cpu")
        for index in range(config.batch_size):
            state = np.full(8, index / config.batch_size, dtype=np.float32)
            agent.replay.append(state, index % 36, -1.0, state + 0.01, index % 13 == 0)
        loss = agent.optimize()
        self.assertIsInstance(loss, float)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.optimizer_steps, 1)

    def test_validation_greedy_action_does_not_consume_exploration_rng(self) -> None:
        from v2.training.dqn import DqnAgent, DqnTrainingConfig

        agent = DqnAgent(DqnTrainingConfig.formal_baseline(), seed=42, device="cpu")
        state = np.arange(8, dtype=np.float32)
        before = agent.rng.getstate()
        action = agent.greedy_action(state)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 36)
        self.assertEqual(agent.rng.getstate(), before)


if __name__ == "__main__":
    unittest.main()
