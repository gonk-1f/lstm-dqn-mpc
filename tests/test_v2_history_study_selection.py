from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestHistoryStudySelection(unittest.TestCase):
    def test_cli_defaults_and_data_loader_never_open_test(self) -> None:
        from v2.main import select_history_dqn_study as module

        args = module._parser().parse_args([])
        self.assertEqual(args.rounds, 10)
        self.assertEqual(args.top_k, 2)

        class Dataset:
            opened_test_payloads = 0

            def split_episode_ids(self, split):
                self.train_split = split
                return ("train-a",)

            def load_validation(self):
                return ("validation-a",)

            def load_test(self):
                raise AssertionError("Test must not be opened")

        dataset = Dataset()
        fixture_args = type(
            "Args",
            (),
            {"power_root": Path("p"), "ais_root": Path("a"), "mode_root": Path("m")},
        )()
        with mock.patch.object(
            module.FormalTrainingDataset,
            "open",
            return_value=dataset,
        ):
            opened, train_ids, validation = module._open_selection_data(fixture_args)
        self.assertIs(opened, dataset)
        self.assertEqual(train_ids, ("train-a",))
        self.assertEqual(validation, ("validation-a",))
        self.assertEqual(dataset.train_split, "train")

    @staticmethod
    def _run(
        experiment_id: str,
        *,
        completed: int,
        raw: float,
        best_round: int,
    ):
        from v2.dqn.action_space import ACTION_CATALOG_DIGEST
        from v2.dqn.state import FORMAL_STATE_SCHEMA_DIGEST
        from v2.evaluation.checkpoint_selection import ValidationCandidate
        from v2.evaluation.study_selection import StudyProfileRun

        profile_number = int(experiment_id[1])
        candidates = []
        for round_index in range(1, 11):
            round_completed = completed if round_index == best_round else max(0, completed - 1)
            failed = 8 - round_completed
            penalty = float(failed * 50_000.0)
            round_raw = raw if round_index == best_round else raw + round_index + 100.0
            candidates.append(
                ValidationCandidate(
                    round_index=round_index,
                    checkpoint_sha256=f"{profile_number * 100 + round_index:064x}",
                    completed_episodes=round_completed,
                    failed_episodes=failed,
                    failure_penalty_score=penalty,
                    raw_economic_cost_cny=float(round_raw),
                    learning_reward=-float(round_raw) - penalty,
                )
            )
        reward_mode = "raw" if experiment_id == "H1" else "scaled"
        reward_identity = "raw_learning_reward_v1" if reward_mode == "raw" else "a" * 64
        return StudyProfileRun(
            experiment_id=experiment_id,
            reward_mode=reward_mode,
            reward_scaling_identity=reward_identity,
            state_schema_digest=FORMAL_STATE_SCHEMA_DIGEST,
            action_catalog_digest=ACTION_CATALOG_DIGEST,
            dataset_identity="d" * 64,
            completed_rounds=10,
            resume_path=Path(experiment_id) / "latest.pt",
            validation_candidates=tuple(candidates),
        )

    def test_selects_best_round_per_profile_then_top_two_profiles(self) -> None:
        from v2.evaluation.study_selection import select_study_candidates

        runs = (
            self._run("H1", completed=7, raw=50.0, best_round=3),
            self._run("H2", completed=8, raw=120.0, best_round=4),
            self._run("H3", completed=8, raw=100.0, best_round=7),
            self._run("H4", completed=7, raw=20.0, best_round=2),
        )
        selected = select_study_candidates(
            runs,
            required_profiles=("H1", "H2", "H3", "H4"),
            top_k=2,
        )
        self.assertEqual(tuple(item.experiment_id for item in selected), ("H3", "H2"))
        self.assertTrue(all(item.completed_rounds == 10 for item in selected))
        self.assertEqual(tuple(item.best_round_index for item in selected), (7, 4))
        self.assertTrue(all(item.resume_path.name == "latest.pt" for item in selected))

    def test_rejects_missing_rounds_profiles_duplicates_and_mixed_identity(self) -> None:
        from v2.evaluation.study_selection import select_study_candidates

        runs = (
            self._run("H1", completed=7, raw=50.0, best_round=3),
            self._run("H2", completed=8, raw=120.0, best_round=4),
            self._run("H3", completed=8, raw=100.0, best_round=7),
            self._run("H4", completed=7, raw=20.0, best_round=2),
        )
        invalid = (
            runs[:-1],
            (runs[0], runs[1], runs[2], runs[2]),
            (replace(runs[0], validation_candidates=runs[0].validation_candidates[:-1]),) + runs[1:],
            (replace(runs[0], state_schema_digest="f" * 64),) + runs[1:],
            (replace(runs[0], action_catalog_digest="f" * 64),) + runs[1:],
            (replace(runs[0], dataset_identity="f" * 64),) + runs[1:],
            (replace(runs[2], reward_scaling_identity="b" * 64), runs[0], runs[1], runs[3]),
        )
        for values in invalid:
            with self.subTest(ids=[value.experiment_id for value in values]):
                with self.assertRaises(ValueError):
                    select_study_candidates(
                        values,
                        required_profiles=("H1", "H2", "H3", "H4"),
                        top_k=2,
                    )

    def test_writes_atomic_manifest_metrics_and_resume_plan(self) -> None:
        from v2.evaluation.study_selection import (
            select_study_candidates,
            write_study_selection_outputs,
        )

        runs = (
            self._run("H1", completed=7, raw=50.0, best_round=3),
            self._run("H2", completed=8, raw=120.0, best_round=4),
            self._run("H3", completed=8, raw=100.0, best_round=7),
            self._run("H4", completed=7, raw=20.0, best_round=2),
        )
        selected = select_study_candidates(
            runs,
            required_profiles=("H1", "H2", "H3", "H4"),
            top_k=2,
        )
        hashes = {"power": "1" * 64, "ais": "2" * 64, "modes": "3" * 64}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "selection"
            write_study_selection_outputs(
                destination,
                runs=runs,
                selected=selected,
                input_manifest_hashes=hashes,
            )
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {
                    "profile_validation_metrics.csv",
                    "study_selection_manifest.json",
                    "top_profiles.json",
                },
            )
            top = json.loads((destination / "top_profiles.json").read_text(encoding="utf-8"))
            self.assertEqual([item["experiment_id"] for item in top["profiles"]], ["H3", "H2"])
            self.assertTrue(all(item["target_round"] == 40 for item in top["profiles"]))
            with self.assertRaises(FileExistsError):
                write_study_selection_outputs(
                    destination,
                    runs=runs,
                    selected=selected,
                    input_manifest_hashes=hashes,
                )


if __name__ == "__main__":
    unittest.main()
