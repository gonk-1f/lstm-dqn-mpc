from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from dataclasses import replace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestV2CheckpointResume(unittest.TestCase):
    def test_round_trip_restores_agent_replay_schedule_and_counters(self) -> None:
        from v2.training.checkpoint import load_checkpoint, save_checkpoint
        from v2.training.dqn import DqnAgent, DqnTrainingConfig
        from v2.training.schedule import EpisodeShuffleSchedule

        config = DqnTrainingConfig.formal_baseline()
        agent = DqnAgent(config, seed=42, device="cpu")
        schedule = EpisodeShuffleSchedule(("a", "b", "c"), seed=42)
        schedule.next_round()
        state = np.arange(8, dtype=np.float32)
        for index in range(10):
            agent.replay.append(state + index, index, -float(index), state + index + 1, False)
        expected_actions = [agent.select_action(state, epsilon=1.0) for _ in range(5)]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(
                path,
                agent=agent,
                schedule=schedule,
                global_macro_step=123,
                round_index=4,
                episode_position=2,
                current_permutation=("c", "a", "b"),
            )
            restored_agent = DqnAgent(config, seed=999, device="cpu")
            restored_schedule = EpisodeShuffleSchedule(("a", "b", "c"), seed=999)
            metadata = load_checkpoint(
                path, agent=restored_agent, schedule=restored_schedule
            )
            self.assertEqual(metadata.global_macro_step, 123)
            self.assertEqual(metadata.round_index, 4)
            self.assertEqual(metadata.episode_position, 2)
            self.assertEqual(metadata.current_permutation, ("c", "a", "b"))
            self.assertEqual(len(restored_agent.replay), 10)
            actual_actions = [restored_agent.select_action(state, epsilon=1.0) for _ in range(5)]
            continued_actions = [agent.select_action(state, epsilon=1.0) for _ in range(5)]
            self.assertEqual(actual_actions, continued_actions)
            self.assertNotEqual(expected_actions, actual_actions)

    def test_resume_allows_only_a_higher_round_budget(self) -> None:
        from v2.training.checkpoint import load_checkpoint, save_checkpoint
        from v2.training.dqn import DqnAgent, DqnTrainingConfig
        from v2.training.schedule import EpisodeShuffleSchedule

        base = DqnTrainingConfig.formal_baseline()
        schedule = EpisodeShuffleSchedule(("a", "b"), seed=42)
        permutation = schedule.next_round()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(
                path,
                agent=DqnAgent(base, seed=42, device="cpu"),
                schedule=schedule,
                global_macro_step=1,
                round_index=0,
                episode_position=1,
                current_permutation=permutation,
            )
            extended = replace(base, rounds=40)
            metadata = load_checkpoint(
                path,
                agent=DqnAgent(extended, seed=999, device="cpu"),
                schedule=EpisodeShuffleSchedule(("a", "b"), seed=999),
            )
            self.assertEqual(metadata.current_permutation, permutation)

    def test_rejects_legacy_state_or_action_identity(self) -> None:
        from v2.training.checkpoint import IncompatibleCheckpointError, load_checkpoint
        from v2.training.dqn import DqnAgent, DqnTrainingConfig
        from v2.training.schedule import EpisodeShuffleSchedule

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"state_schema_version": "legacy_s7", "action_dim": 84}, path)
            with self.assertRaises(IncompatibleCheckpointError):
                load_checkpoint(
                    path,
                    agent=DqnAgent(DqnTrainingConfig.formal_baseline(), seed=1, device="cpu"),
                    schedule=EpisodeShuffleSchedule(("a",), seed=1),
                )


if __name__ == "__main__":
    unittest.main()
