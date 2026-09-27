from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestHistoryDqnExperiments(unittest.TestCase):
    @staticmethod
    def _calibration(costs=(10.0, 20.0, 30.0), provenance_id="fixture"):
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import calibrate_reward_scale

        return calibrate_reward_scale(
            tuple(costs),
            provenance=DatasetProvenance(
                "operating_dataset_zero_boundary_v2",
                f"sha256:{provenance_id}",
                DataSplit.TRAIN,
            ),
            audit_id="experiment-fixture",
            reason="bind scaled experiment identities",
        )

    @staticmethod
    def _failure_transition():
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND, FORMAL_FAILURE_POLICY

        state = (0.0,) * 90
        ledger = RawCnyIntervalLedger(30.0, 0.0, 0.0, 0.0)
        return MacroTransition(
            state=state,
            action=FINAL_DQN_ACTION_CATALOG[0],
            learning_reward=-50_030.0,
            next_state=state,
            done=True,
            executed_mpc_steps=1,
            ledger=ledger,
            failure_penalty_score=FORMAL_FAILURE_POLICY.penalty_score,
            failure_kind=FORMAL_FAILURE_KIND,
        )

    def test_matrix_changes_only_reward_mode_and_learning_rate(self) -> None:
        from v2.training.experiments import history_study_profiles

        profiles = history_study_profiles(self._calibration())
        self.assertEqual(tuple(profiles), ("H1", "H2", "H3", "H4"))
        self.assertEqual(
            [profiles[key].learning_rate for key in profiles],
            [1.0e-4, 1.0e-4, 3.0e-4, 1.0e-3],
        )
        self.assertEqual(
            [profiles[key].reward_mode for key in profiles],
            ["raw", "scaled", "scaled", "scaled"],
        )
        configs = [profiles[key].dqn_config(rounds=10) for key in profiles]
        for config in configs:
            self.assertEqual(config.state_dim, 90)
            self.assertEqual(config.action_dim, 36)
            self.assertEqual(config.hidden_dims, (128, 128))
            self.assertEqual(config.gamma, 1.0)
            self.assertEqual(config.batch_size, 256)
            self.assertEqual(config.replay_capacity, 200_000)
            self.assertEqual(config.warmup_steps, 5_000)
            self.assertEqual(config.target_sync_steps, 1_000)
            self.assertEqual(config.rounds, 10)

    def test_raw_rejects_calibration_and_scaled_requires_exact_calibration(self) -> None:
        from v2.training.experiments import history_study_profiles

        calibration = self._calibration()
        profiles = history_study_profiles(calibration)
        transition = self._failure_transition()
        self.assertEqual(profiles["H1"].replay_reward(transition, None), -50_030.0)
        with self.assertRaises(ValueError):
            profiles["H1"].replay_reward(transition, calibration)
        with self.assertRaises(ValueError):
            profiles["H2"].replay_reward(transition, None)
        self.assertAlmostEqual(
            profiles["H2"].replay_reward(transition, calibration),
            -2501.5,
        )
        with self.assertRaises(ValueError):
            profiles["H2"].replay_reward(
                transition,
                self._calibration((1.0, 2.0, 3.0), "different"),
            )

    def test_checkpoint_rejects_profile_reward_and_scale_identity_changes(self) -> None:
        from v2.training.checkpoint import (
            IncompatibleCheckpointError,
            load_checkpoint,
            save_checkpoint,
        )
        from v2.training.dqn import DqnAgent
        from v2.training.experiments import history_study_profiles
        from v2.training.schedule import EpisodeShuffleSchedule

        profile = history_study_profiles(self._calibration())["H2"]
        source_config = profile.dqn_config(rounds=10)
        schedule = EpisodeShuffleSchedule(("a",), seed=42)
        permutation = schedule.next_round()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(
                path,
                agent=DqnAgent(source_config, seed=42, device="cpu"),
                schedule=schedule,
                global_macro_step=1,
                round_index=0,
                episode_position=1,
                current_permutation=permutation,
            )
            variants = (
                replace(source_config, experiment_id="H3"),
                replace(source_config, learning_rate=3.0e-4),
                replace(source_config, reward_mode="raw"),
                replace(source_config, reward_scaling_identity="different-digest"),
            )
            for config in variants:
                with self.subTest(config=config):
                    with self.assertRaises(IncompatibleCheckpointError):
                        load_checkpoint(
                            path,
                            agent=DqnAgent(config, seed=7, device="cpu"),
                            schedule=EpisodeShuffleSchedule(("a",), seed=7),
                        )


if __name__ == "__main__":
    unittest.main()
