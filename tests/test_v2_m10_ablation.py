from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class M10AblationTests(unittest.TestCase):
    @staticmethod
    def _calibration():
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import calibrate_reward_scale

        return calibrate_reward_scale(
            (1.0, 2.0, 3.0),
            provenance=DatasetProvenance(
                "operating_dataset_zero_boundary_v2",
                "sha256:m10-fixture",
                DataSplit.TRAIN,
            ),
            audit_id="m10-fixture",
            reason="M10 Train-only fixture",
        )

    @staticmethod
    def _document(timescale):
        from v2.training.reward_scaling import build_reward_scale_document

        return build_reward_scale_document(
            calibration=M10AblationTests._calibration(),
            reference_action_id="w_8_1_1",
            manifest_hashes={
                "power": "a" * 64,
                "ais": "b" * 64,
                "modes": "c" * 64,
            },
            train_segment_ids=("train-a",),
            test_payloads_opened=0,
            timescale=timescale,
        )

    def test_m10_profile_changes_only_requested_macro_step_parameters(self) -> None:
        from v2.training.experiments import m10_ablation_profile

        profile = m10_ablation_profile(self._calibration())
        config = profile.dqn_config(rounds=40)

        self.assertEqual(profile.timescale.ts_mpc_seconds, 30.0)
        self.assertEqual(profile.timescale.n_mpc, 5)
        self.assertEqual(profile.timescale.dqn_switch_steps, 10)
        self.assertEqual(profile.timescale.switch_seconds, 300.0)
        self.assertEqual(config.warmup_steps, 2_500)
        self.assertEqual(config.epsilon_decay_steps, 75_000)
        self.assertEqual(config.replay_capacity, 100_000)
        self.assertEqual(config.gamma, 1.0)
        self.assertEqual(config.learning_rate, 1.0e-3)
        self.assertEqual(config.hidden_dims, (128, 128))
        self.assertEqual(config.batch_size, 256)
        self.assertEqual(config.target_sync_steps, 1_000)
        self.assertEqual(config.action_dim, 36)

    def test_macro_transition_estimate_uses_requested_action_hold(self) -> None:
        from v2.data.formal_training_dataset import _onboard_macro_transition_count

        modes = ("onboard",) * 11 + ("shore_charging",) + ("onboard",) * 9
        self.assertEqual(_onboard_macro_transition_count(modes, 5), 5)
        self.assertEqual(_onboard_macro_transition_count(modes, 10), 3)

    def test_m10_reward_scale_is_bound_to_m10_and_rejects_m5_loader(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.contracts import control_semantics
        from v2.training.reward_scaling import load_reward_scale_document

        m10 = TimeScaleConfig(30.0, 5, 10)
        document = self._document(m10)
        self.assertEqual(document["control_semantics"], control_semantics(m10))
        loaded = load_reward_scale_document(document, timescale=m10)
        self.assertEqual(loaded.scale_cny, 2.0)
        with self.assertRaises(ValueError):
            load_reward_scale_document(document)

    def test_m10_checkpoint_cannot_be_loaded_as_m5(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.training.checkpoint import (
            IncompatibleCheckpointError,
            load_checkpoint,
            save_checkpoint,
        )
        from v2.training.dqn import DqnAgent
        from v2.training.experiments import m10_ablation_profile
        from v2.training.schedule import EpisodeShuffleSchedule

        m10 = TimeScaleConfig(30.0, 5, 10)
        config = m10_ablation_profile(self._calibration()).dqn_config(rounds=1)
        schedule = EpisodeShuffleSchedule(("a",), seed=42)
        permutation = schedule.next_round()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m10.pt"
            save_checkpoint(
                path,
                agent=DqnAgent(config, seed=42, device="cpu"),
                schedule=schedule,
                global_macro_step=0,
                round_index=0,
                episode_position=0,
                current_permutation=permutation,
                timescale=m10,
            )
            load_checkpoint(
                path,
                agent=DqnAgent(config, seed=7, device="cpu"),
                schedule=EpisodeShuffleSchedule(("a",), seed=7),
                timescale=m10,
            )
            with self.assertRaises(IncompatibleCheckpointError):
                load_checkpoint(
                    path,
                    agent=DqnAgent(config, seed=7, device="cpu"),
                    schedule=EpisodeShuffleSchedule(("a",), seed=7),
                )

    def test_formal_environment_uses_one_explicit_timescale_end_to_end(self) -> None:
        from v2.config import TimeScaleConfig
        from v2.data.formal_training_dataset import FormalEpisode
        from v2.evaluation.formal_policy import build_formal_environment

        m10 = TimeScaleConfig(30.0, 5, 10)
        count = 12
        episode = FormalEpisode(
            "parent",
            "train-fixture",
            "Train",
            tuple(f"t-{index}" for index in range(count)),
            np.arange(count, dtype=float) * 30.0,
            np.full(count, 100.0),
            np.full(count, 5.0),
            ("measured",) * count,
            np.zeros(count),
            np.zeros(count),
            ("onboard",) * count,
            ("fixture",) * count,
        )
        backend, environment = build_formal_environment(episode, timescale=m10)
        self.assertEqual(backend.timescale, m10)
        self.assertEqual(environment.timescale, m10)
        self.assertEqual(backend.mpc.config.timescale, m10)

    def test_independent_cli_defaults_to_m10_artifact_roots(self) -> None:
        from v2.main.train_m10_dqn_study import (
            DEFAULT_M10_REWARD_SCALE,
            DEFAULT_M10_STUDY_ROOT,
            _parser,
        )

        args = _parser().parse_args([])
        self.assertEqual(args.rounds, 40)
        self.assertEqual(args.reward_scale, DEFAULT_M10_REWARD_SCALE)
        self.assertEqual(args.output_dir, DEFAULT_M10_STUDY_ROOT)
        self.assertNotIn("test", str(DEFAULT_M10_STUDY_ROOT).lower())
        self.assertNotEqual(
            DEFAULT_M10_REWARD_SCALE,
            ROOT / "outputs" / "v2_history_dqn_study" / "reward_scale_calibration.json",
        )

    def test_validation_comparison_reports_distribution_entropy_and_max_share(self) -> None:
        from v2.main.compare_m5_m10_validation import _action_diagnostics

        values = _action_diagnostics((("w_1_1_8", 3), ("w_1_2_7", 1)))
        self.assertEqual(values["unique_action_count"], 2)
        self.assertEqual(values["transition_count"], 4)
        self.assertAlmostEqual(values["max_action_share"], 0.75)
        self.assertAlmostEqual(
            values["shannon_entropy"],
            -(0.75 * np.log(0.75) + 0.25 * np.log(0.25)),
        )
        self.assertEqual(
            values["action_distribution"],
            {"w_1_1_8": 3, "w_1_2_7": 1},
        )


if __name__ == "__main__":
    unittest.main()
