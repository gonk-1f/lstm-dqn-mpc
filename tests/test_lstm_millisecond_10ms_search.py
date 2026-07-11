from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import optuna
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src" / "main", ROOT / "src" / "forecasting"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lstm_millisecond_10ms_search import (  # noqa: E402
    CandidateResult,
    EpochResult,
    build_parser,
    close_study_storage,
    load_sequences,
    run_study,
    run_training_loop,
    sample_trial_config,
    select_configuration,
)


class TestSearchContract(unittest.TestCase):
    def test_default_limits_match_approved_design(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.n_trials, 24)
        self.assertEqual(args.study_timeout_s, 3600)
        self.assertEqual(args.trial_timeout_s, 180)
        self.assertEqual(args.max_epochs, 25)
        self.assertEqual(args.patience, 4)
        self.assertEqual(args.history_steps, 30)
        self.assertEqual(args.prediction_steps, 6)

    def test_seed_is_not_an_optuna_parameter(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "hidden_size": 32,
                "num_layers": 1,
                "mlp_head": "none",
                "loss": "Huber",
                "learning_rate": 1e-3,
                "batch_size": 64,
                "gradient_clip": 1.0,
                "weight_decay": 0.0,
            }
        )
        config = sample_trial_config(trial, fixed_seed=42)
        self.assertEqual(config.seed, 42)
        self.assertNotIn("seed", trial.params)
        self.assertEqual(config.dropout, 0.0)


class TestTrainingLimits(unittest.TestCase):
    def test_training_stops_when_trial_deadline_is_reached(self) -> None:
        clock = iter([0.0, 1.0, 181.0]).__next__
        result = run_training_loop(
            state=object(),
            max_epochs=25,
            patience=4,
            min_delta=1e-6,
            trial_timeout_s=180,
            clock=clock,
            run_epoch=lambda state, epoch: EpochResult(score=2.0, mae=1.0),
        )
        self.assertTrue(result.stopped_by_timeout)
        self.assertLess(result.epochs_completed, 25)

    def test_training_uses_early_stopping(self) -> None:
        scores = iter([2.0, 2.0, 2.0, 2.0])
        result = run_training_loop(
            state=object(),
            max_epochs=20,
            patience=2,
            min_delta=1e-6,
            trial_timeout_s=1000,
            clock=lambda: 0.0,
            run_epoch=lambda state, epoch: EpochResult(score=next(scores), mae=1.0),
        )
        self.assertEqual(result.epochs_completed, 3)
        self.assertTrue(result.stopped_by_early_stopping)


class TestStudyPersistence(unittest.TestCase):
    def test_study_honors_trial_count_and_persists_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls: list[int] = []
            study = run_study(
                objective=lambda trial: calls.append(trial.number) or float(trial.number),
                storage_path=root / "study.sqlite3",
                study_name="test",
                n_trials=3,
                timeout_s=60,
                sampler_seed=20260710,
                trial_csv=root / "trials.csv",
            )
            self.assertLessEqual(len(study.trials), 3)
            self.assertTrue((root / "trials.csv").exists())
            self.assertEqual(len(calls), len(study.trials))
            close_study_storage(study)

    def test_resume_refuses_mismatched_study_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = dict(
                objective=lambda trial: 1.0,
                storage_path=root / "study.sqlite3",
                study_name="test",
                n_trials=0,
                timeout_s=60,
                sampler_seed=20260710,
                trial_csv=root / "trials.csv",
            )
            study = run_study(**common, study_user_attrs={"dataset_sha256": "a"})
            close_study_storage(study)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                study = run_study(**common, study_user_attrs={"dataset_sha256": "b"})
            close_study_storage(study)


class TestSequenceLoading(unittest.TestCase):
    def test_load_sequences_preserves_atomic_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            pd.DataFrame(
                {
                    "split": ["train", "train", "train", "validation"],
                    "sequence_id": ["a", "a", "b", "v"],
                    "time_ms": [0, 10, 100, 0],
                    "load_kw": [1.0, 2.0, 9.0, 3.0],
                }
            ).to_csv(path, index=False)
            sequences = load_sequences(path, "train")
            self.assertEqual(list(sequences), ["a", "b"])
            self.assertEqual(sequences["a"].tolist(), [1.0, 2.0])
            self.assertEqual(sequences["b"].tolist(), [9.0])


class TestValidationOnlySelection(unittest.TestCase):
    def test_configuration_selection_uses_validation_only(self) -> None:
        candidates = [
            CandidateResult("a", (2.0, 2.2, 1.8), (2.0, 2.1, 1.9)),
            CandidateResult("b", (3.0, 3.1, 2.9), (1.0, 1.1, 0.9)),
        ]
        selected = select_configuration(candidates)
        self.assertEqual(selected.config_id, "a")


if __name__ == "__main__":
    unittest.main()
