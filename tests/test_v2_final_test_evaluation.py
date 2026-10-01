from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import pandas as pd

from v2.evaluation.checkpoint_selection import (
    ValidationCandidate,
    make_selection_manifest,
    write_selection_outputs,
)
from v2.evaluation.formal_policy import (
    EpisodeEvaluation,
    EpisodePowerTrace,
    PolicyEvaluation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_current_manifests(root: Path) -> tuple[dict[str, str], dict[str, Path]]:
    paths = {}
    for key in ("power", "ais", "modes"):
        path = root / key / "metadata" / "sample_manifest.csv"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"synthetic-{key}-manifest".encode("ascii"))
        paths[key] = path
    return ({key: _sha256(path) for key, path in paths.items()}, paths)


def _write_selection(root: Path, manifest_hashes: dict[str, str]) -> Path:
    checkpoint = root / "round_001.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    selected_hash = _sha256(checkpoint)
    values = tuple(
        ValidationCandidate(
            round_index=round_index,
            checkpoint_sha256=(selected_hash if round_index == 1 else f"{round_index:064x}"),
            completed_episodes=8,
            failed_episodes=0,
            failure_penalty_score=0.0,
            raw_economic_cost_cny=float(round_index),
            learning_reward=-float(round_index),
        )
        for round_index in range(1, 41)
    )
    manifest = make_selection_manifest(manifest_hashes, values)
    destination = root / "selection"
    write_selection_outputs(destination, manifest, checkpoint)
    return destination


def _episode_result(sample_id: str, *, failed: bool, raw: float, action_id: str) -> EpisodeEvaluation:
    penalty = 50_000.0 if failed else 0.0
    return EpisodeEvaluation(
        sample_id=sample_id,
        completed=not failed,
        failure_kind="physical_infeasibility" if failed else None,
        transition_count=1,
        executed_mpc_steps=1,
        h2_cost_cny=raw,
        fc_degradation_cost_cny=0.0,
        battery_degradation_cost_cny=0.0,
        shore_cost_cny=0.0,
        raw_economic_cost_cny=raw,
        failure_penalty_score=penalty,
        learning_reward=-raw - penalty,
        soc_min=0.50,
        soc_max=0.60,
        action_counts=((action_id, 1),),
    )


def _power_trace(sample_id: str) -> EpisodePowerTrace:
    return EpisodePowerTrace(
        sample_id=sample_id,
        time_s=(0.0,),
        load_power_kw=(10.0,),
        fuel_cell_power_kw=(8.0,),
        battery_bus_power_kw=(2.0,),
        operating_mode=("onboard",),
        soc_time_s=(0.0, 30.0),
        soc=(0.60, 0.599),
        completed=True,
        failure_kind=None,
    )


class TestFinalTestAuthorization(unittest.TestCase):
    def test_authenticates_selection_and_loads_exact_test_once(self) -> None:
        from v2.data.formal_training_dataset import FormalTrainingDataset
        from v2.evaluation.final_test import authenticate_final_test_selection

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, _ = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            authorization = authenticate_final_test_selection(selection)
            sample_ids = [f"test_{index}" for index in range(5, 0, -1)]
            frame = pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "split": ["test"] * 5,
                    "relative_path": [f"{value}.csv" for value in sample_ids],
                }
            )
            dataset = FormalTrainingDataset(
                root / "power", root / "ais", root / "modes",
                frame, frame.copy(), frame.copy(),
            )
            opened: list[str] = []

            def load_episode(power_row, ais_row, mode_row):
                opened.append(str(power_row.sample_id))
                return SimpleNamespace(sample_id=str(power_row.sample_id), split="test")

            with mock.patch.object(dataset, "_load_episode", side_effect=load_episode):
                episodes = dataset.load_final_test(authorization)
                with self.assertRaises(PermissionError):
                    dataset.load_final_test(authorization)
            self.assertEqual(tuple(item.sample_id for item in episodes), tuple(sorted(sample_ids)))
            self.assertEqual(opened, sorted(sample_ids))
            self.assertEqual(dataset.opened_test_payloads, 5)
            self.assertEqual(dataset._cache, {})
            with self.assertRaises(PermissionError):
                dataset.load_split("test")

    def test_rejects_forged_authority_tamper_and_current_manifest_change(self) -> None:
        from v2.data.formal_training_dataset import FormalTrainingDataset
        from v2.evaluation.final_test import (
            FinalTestAuthorization,
            authenticate_final_test_selection,
            require_final_test_authorization,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, paths = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            authorization = authenticate_final_test_selection(selection)
            frame = pd.DataFrame(
                {
                    "sample_id": [f"test_{index}" for index in range(5)],
                    "split": ["test"] * 5,
                    "relative_path": [f"test_{index}.csv" for index in range(5)],
                }
            )
            dataset = FormalTrainingDataset(
                root / "power", root / "ais", root / "modes",
                frame, frame.copy(), frame.copy(),
            )

            class Forged(FinalTestAuthorization):
                pass

            forged = object.__new__(Forged)
            for key, value in vars(authorization).items():
                object.__setattr__(forged, key, value)
            with (
                mock.patch.object(dataset, "_load_episode") as loader,
                self.assertRaises((TypeError, PermissionError, ValueError)),
            ):
                dataset.load_final_test(forged)
            loader.assert_not_called()

            tampered = object.__new__(FinalTestAuthorization)
            for key, value in vars(authorization).items():
                object.__setattr__(tampered, key, value)
            object.__setattr__(tampered, "selected_round", True)
            with self.assertRaises((TypeError, PermissionError, ValueError)):
                require_final_test_authorization(tampered)

            paths["ais"].write_bytes(b"changed")
            with (
                mock.patch.object(dataset, "_load_episode") as loader,
                self.assertRaises((PermissionError, ValueError)),
            ):
                dataset.load_final_test(authorization)
            loader.assert_not_called()
            self.assertEqual(dataset.opened_test_payloads, 0)

            manifest_path = selection / "selection_manifest.json"
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            with self.assertRaises(ValueError):
                authenticate_final_test_selection(selection)

    def test_modified_best_checkpoint_is_rejected(self) -> None:
        from v2.evaluation.final_test import authenticate_final_test_selection

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, _ = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            (selection / "best_validation.pt").write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                authenticate_final_test_selection(selection)


class TestFinalTestCli(unittest.TestCase):
    def test_rejects_wrong_confirmation_and_existing_output_before_test_access(self) -> None:
        from v2.main import evaluate_formal_dqn_test as module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(module, "authenticate_final_test_selection") as authenticate,
                mock.patch.object(module.FormalTrainingDataset, "open") as open_dataset,
                self.assertRaises(SystemExit),
            ):
                module.main(["--confirm-final-test", "wrong"])
            authenticate.assert_not_called()
            open_dataset.assert_not_called()

            output = root / "test_output"
            output.mkdir()
            with (
                mock.patch.object(module, "authenticate_final_test_selection") as authenticate,
                mock.patch.object(module.FormalTrainingDataset, "open") as open_dataset,
                self.assertRaises(FileExistsError),
            ):
                module.main([
                    "--output-dir", str(output),
                    "--confirm-final-test", "FINAL_TEST_ONCE",
                ])
            authenticate.assert_not_called()
            open_dataset.assert_not_called()

    def test_same_test_tuple_is_used_for_dqn_and_fixed_outputs(self) -> None:
        from v2.main import evaluate_formal_dqn_test as module

        class Dataset:
            def __init__(self) -> None:
                self.opened_test_payloads = 0
                self.episodes = (
                    SimpleNamespace(sample_id="test_a"),
                    SimpleNamespace(sample_id="test_b"),
                )

            def split_episode_ids(self, split):
                self.train_split = split
                return ("train_a", "train_b")

            def load_final_test(self, authorization):
                self.opened_test_payloads = 2
                return self.episodes

        class Agent:
            def __init__(self, config, *, seed, device) -> None:
                pass

            def greedy_action(self, state):
                return 0

        class Schedule:
            def __init__(self, episode_ids, *, seed) -> None:
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, _ = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            output = root / "test_output"
            dataset = Dataset()
            episode_objects: list[object] = []

            def evaluate(*, episodes, policy):
                episode_objects.append(episodes)
                if policy.policy_id == "greedy_dqn":
                    return PolicyEvaluation(
                        "greedy_dqn",
                        (
                            _episode_result("test_a", failed=False, raw=10.0, action_id="w_8_1_1"),
                            _episode_result("test_b", failed=True, raw=5.0, action_id="w_8_1_1"),
                        ),
                    )
                return PolicyEvaluation(
                    "w_8_1_1",
                    (
                        _episode_result("test_a", failed=False, raw=12.0, action_id="w_8_1_1"),
                        _episode_result("test_b", failed=False, raw=8.0, action_id="w_8_1_1"),
                    ),
                )

            with (
                mock.patch.object(module.FormalTrainingDataset, "open", return_value=dataset),
                mock.patch.object(module, "DqnAgent", Agent),
                mock.patch.object(module, "EpisodeShuffleSchedule", Schedule),
                mock.patch.object(
                    module,
                    "load_checkpoint",
                    return_value=SimpleNamespace(round_index=1, episode_position=0),
                ),
                mock.patch.object(module, "evaluate_formal_policy", side_effect=evaluate),
                redirect_stdout(io.StringIO()),
            ):
                code = module.main([
                    "--selection-dir", str(selection),
                    "--output-dir", str(output),
                    "--confirm-final-test", "FINAL_TEST_ONCE",
                ])

            self.assertEqual(code, 0)
            self.assertIs(episode_objects[0], dataset.episodes)
            self.assertIs(episode_objects[1], dataset.episodes)
            self.assertEqual(dataset.train_split, "train")
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "TEST_ACCESS_STARTED.json",
                    "run_manifest.json",
                    "test_action_distribution.csv",
                    "test_episode_metrics.csv",
                    "test_summary.json",
                ],
            )
            marker = json.loads((output / "TEST_ACCESS_STARTED.json").read_text("ascii"))
            self.assertEqual(marker["status"], "COMPLETE")
            summary = json.loads((output / "test_summary.json").read_text("ascii"))
            for policy in summary["policies"]:
                self.assertEqual(
                    policy["learning_reward"],
                    -policy["raw_economic_cost_cny"] - policy["failure_penalty_score"],
                )

    def test_evaluation_failure_leaves_started_lock(self) -> None:
        from v2.main import evaluate_formal_dqn_test as module

        dataset = SimpleNamespace(
            opened_test_payloads=0,
            split_episode_ids=lambda split: ("train",),
            load_final_test=lambda authorization: (SimpleNamespace(sample_id="test"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, _ = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            output = root / "test_output"
            with (
                mock.patch.object(module.FormalTrainingDataset, "open", return_value=dataset),
                mock.patch.object(
                    module,
                    "DqnAgent",
                    return_value=SimpleNamespace(greedy_action=lambda state: 0),
                ),
                mock.patch.object(module, "EpisodeShuffleSchedule", return_value=SimpleNamespace()),
                mock.patch.object(
                    module,
                    "load_checkpoint",
                    return_value=SimpleNamespace(round_index=1, episode_position=0),
                ),
                mock.patch.object(
                    module,
                    "evaluate_formal_policy",
                    side_effect=RuntimeError("synthetic failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "synthetic failure"),
            ):
                module.main([
                    "--selection-dir", str(selection),
                    "--output-dir", str(output),
                    "--confirm-final-test", "FINAL_TEST_ONCE",
                ])
            marker = json.loads((output / "TEST_ACCESS_STARTED.json").read_text("ascii"))
            self.assertEqual(marker["status"], "STARTED")
            self.assertTrue(output.is_dir())

    def test_optional_plot_directory_uses_same_greedy_test_execution_traces(self) -> None:
        from v2.main import evaluate_formal_dqn_test as module

        episodes = (SimpleNamespace(sample_id="test_a"),)
        dataset = SimpleNamespace(
            opened_test_payloads=0,
            split_episode_ids=lambda split: ("train",),
            load_final_test=lambda authorization: episodes,
        )
        greedy = PolicyEvaluation(
            "greedy_dqn",
            (_episode_result("test_a", failed=False, raw=10.0, action_id="w_8_1_1"),),
        )
        fixed = PolicyEvaluation(
            "w_8_1_1",
            (_episode_result("test_a", failed=False, raw=12.0, action_id="w_8_1_1"),),
        )
        traces = (_power_trace("test_a"),)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hashes, _ = _write_current_manifests(root)
            selection = _write_selection(root, hashes)
            output = root / "test_output"
            plot_output = root / "plots"
            with (
                mock.patch.object(module.FormalTrainingDataset, "open", return_value=dataset),
                mock.patch.object(module, "DqnAgent", return_value=SimpleNamespace(greedy_action=lambda state: 0)),
                mock.patch.object(module, "EpisodeShuffleSchedule", return_value=SimpleNamespace()),
                mock.patch.object(module, "load_checkpoint", return_value=SimpleNamespace(round_index=1, episode_position=0)),
                mock.patch.object(module, "evaluate_formal_policy_with_power_traces", return_value=(greedy, traces)) as traced,
                mock.patch.object(module, "evaluate_formal_policy", return_value=fixed) as aggregate,
                mock.patch.object(module, "write_power_trace_plots") as writer,
                redirect_stdout(io.StringIO()),
            ):
                code = module.main(
                    [
                        "--selection-dir", str(selection),
                        "--output-dir", str(output),
                        "--plot-dir", str(plot_output),
                        "--confirm-final-test", "FINAL_TEST_ONCE",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIs(traced.call_args.kwargs["episodes"], episodes)
        self.assertIs(aggregate.call_args.kwargs["episodes"], episodes)
        writer.assert_called_once_with(plot_output, traces, policy_id="H1_round_001")


if __name__ == "__main__":
    unittest.main()
