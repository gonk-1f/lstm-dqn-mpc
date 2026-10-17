from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _freeze(root: Path, rows: list[dict]) -> None:
    directory = root / "metadata" / "splits_v1_seed42"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "sample_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    payloads = []
    for row in rows:
        for kind, path_key, hash_key in (
            ("canonical_1s", "relative_path", "sha256"),
            ("existing_30s_preview", "preview_30s_relative_path", "preview_30s_sha256"),
        ):
            payloads.append({"candidate_id": row["candidate_id"], "split": row["split"],
                             "kind": kind, "relative_path": row[path_key], "sha256": row[hash_key]})
    pd.DataFrame(payloads).to_csv(directory / "payload_hashes.csv", index=False)
    counts = {split: sum(row["split"] == split for row in rows)
              for split in ("train", "validation", "test")}
    policy = {
        "schema": "independent_v2_split_v1", "dataset_version": "operating_dataset_independent_v2",
        "split_version": "splits_v1_seed42", "split_counts": counts,
        "sample_manifest_sha256": _sha(manifest),
        "payload_hash_manifest_sha256": _sha(directory / "payload_hashes.csv"),
        "historical_checkpoints_valid_for_new_task": False,
    }
    (directory / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    active = {
        "split_version": "splits_v1_seed42", "split_counts": counts,
        "manifest_relative_path": "metadata/splits_v1_seed42/sample_manifest.csv",
        "manifest_sha256": _sha(manifest),
        "policy_relative_path": "metadata/splits_v1_seed42/policy.json",
    }
    (root / "metadata" / "active_split.json").write_text(json.dumps(active), encoding="utf-8")


def _fixture(root: Path) -> list[dict]:
    (root / "candidates").mkdir()
    rows = []
    for number, (split, duration, preview_times) in enumerate((
        ("train", 78, [0, 30]), ("validation", 60, [0, 30, 60]), ("test", 47, [0]),
    )):
        candidate = f"parent_{number:03d}_seg01"
        parent = f"source_{number:03d}"
        canonical = pd.DataFrame({
            "candidate_id": candidate, "parent_sample_id": parent,
            "timestamp": pd.date_range("2024-04-09 08:00:00+08:00", periods=duration + 1, freq="s"),
            "time_s": np.arange(duration + 1), "load_total_kw": np.arange(duration + 1) / 10 - .4,
            "source_time_s": np.arange(duration + 1) + 300,
            "source_1s_row": np.arange(duration + 1) + 300,
        })
        relative = f"candidates/{candidate}.csv.gz"
        preview = f"candidates/{candidate}.csv"
        canonical.to_csv(root / relative, index=False, float_format="%.17g")
        canonical.iloc[preview_times].to_csv(root / preview, index=False, float_format="%.17g")
        rows.append({
            "candidate_id": candidate, "parent_sample_id": parent, "group_id": parent,
            "split": split, "new_split": split, "legacy_split": "test",
            "relative_path": relative, "sha256": _sha(root / relative),
            "preview_30s_relative_path": preview, "preview_30s_sha256": _sha(root / preview),
            "duration_s": duration, "point_count": duration + 1,
            "preview_30s_point_count": len(preview_times), "unknown_points": 0, "shore_points": 0,
        })
    _freeze(root, rows)
    return rows


class TestIndependentDataset(unittest.TestCase):
    def test_exclusion_freezes_a_new_version_without_touching_old_rows_or_test(self) -> None:
        script = ROOT / "scripts/exclude_independent_voyage_sample.py"
        self.assertTrue(script.is_file(), "versioned metadata exclusion command is missing")
        spec = importlib.util.spec_from_file_location("independent_exclusion_fixture", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _fixture(root)
            extra = rows[0] | {"candidate_id": "kept_train", "parent_sample_id": "kept_parent",
                               "group_id": "kept_parent"}
            rows.append(extra)
            _freeze(root, rows)
            old_directory = root / "metadata/splits_v1_seed42"
            old_bytes = {path.name: path.read_bytes() for path in old_directory.iterdir()}
            original_manifest = old_bytes["sample_manifest.csv"]
            test_paths = {(root / rows[2][key]).resolve()
                          for key in ("relative_path", "preview_30s_relative_path")}
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.resolve() in test_paths:
                    raise AssertionError("Test payload opened while excluding Train metadata")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                result = module.freeze_exclusion(root, rows[0]["candidate_id"], "splits_v2_fixture",
                                                 "user excluded the long idle tail")
                new_directory = root / "metadata/splits_v2_fixture"
                expected_lines = [line for line in original_manifest.splitlines(keepends=True)
                                  if not line.startswith((rows[0]["candidate_id"] + ",").encode())]
                self.assertEqual((new_directory / "sample_manifest.csv").read_bytes(), b"".join(expected_lines))
                self.assertEqual(result["split_counts"], {"train": 1, "validation": 1, "test": 1})
                for name, original in old_bytes.items():
                    self.assertEqual((old_directory / name).read_bytes(), original)
                dataset = IndependentTrainingDataset.open(root)
                self.assertEqual(dataset.split_episode_ids("train"), ("kept_train",))
                self.assertIn("metadata/splits_v2_fixture/exclusions.csv", dataset.identity_hashes())
                self.assertEqual(dataset.opened_test_payloads, 0)
                with self.assertRaises(FileExistsError):
                    module.freeze_exclusion(root, "kept_train", "splits_v2_fixture", "duplicate version")
                with self.assertRaisesRegex(ValueError, "Train|train"):
                    module.freeze_exclusion(root, rows[2]["candidate_id"], "splits_v3_forbidden", "forbidden")

    def test_complete_tail_preserves_signed_load_and_nonzero_boundaries(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            dataset = IndependentTrainingDataset.open(root)
            episode, = dataset.load_train()
            np.testing.assert_array_equal(episode.time_s, [0, 30, 60])
            np.testing.assert_array_equal(episode.dt_s, [30, 30, 18])
            np.testing.assert_allclose(episode.load_kw, [-.4, 2.6, 5.6], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(episode.source_time_s, [300, 330, 360])
            np.testing.assert_array_equal(episode.source_1s_row, [300, 330, 360])
            self.assertEqual(episode.operating_mode, ("onboard",) * 3)
            self.assertEqual(episode.terminal_boundary_time_s, 78)
            self.assertEqual(episode.terminal_boundary_load_kw, 78 / 10 - .4)
            self.assertEqual(episode.duration_s, sum(episode.dt_s))
            self.assertEqual(episode.task_mode, "independent_voyage")
            self.assertEqual(episode.step_count, 3)
            self.assertFalse(hasattr(episode, "battery_bus_kw"))
            self.assertFalse(hasattr(episode, "speed_kn"))
            self.assertEqual(dataset.root, root.resolve())
            self.assertEqual(dataset.split_sizes, (1, 1))
            self.assertEqual(dataset.task_contract["initial_soc"], .6)
            self.assertFalse(dataset.task_contract["terminal_charging"])
            self.assertIs(dataset.load_train(), dataset.load_train())

    def test_exact_terminal_point_has_no_extra_execution_interval(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            episode, = IndependentTrainingDataset.open(root).load_validation()
            np.testing.assert_array_equal(episode.time_s, [0, 30])
            np.testing.assert_array_equal(episode.dt_s, [30, 30])
            self.assertEqual(episode.terminal_boundary_load_kw, 5.6)

    def test_metadata_identity_and_train_access_never_open_test_payload(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _fixture(root)
            test_paths = {(root / rows[2][key]).resolve()
                          for key in ("relative_path", "preview_30s_relative_path")}
            original_open = Path.open

            def guarded_open(path, *args, **kwargs):
                if path.resolve() in test_paths:
                    raise AssertionError("Test payload opened")
                return original_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", guarded_open):
                dataset = IndependentTrainingDataset.open(root)
                self.assertEqual(dataset.split_episode_ids("train"), (rows[0]["candidate_id"],))
                dataset.load_train()
                dataset.load_validation()
                hashes = dataset.identity_hashes()
                self.assertEqual(set(hashes), {
                    "metadata/active_split.json", "metadata/splits_v1_seed42/sample_manifest.csv",
                    "metadata/splits_v1_seed42/policy.json", "metadata/splits_v1_seed42/payload_hashes.csv",
                })
                for relative, digest in hashes.items():
                    self.assertEqual(digest, _sha(root / relative))
                with self.assertRaises(PermissionError):
                    dataset.load_split("test")
                with self.assertRaises(PermissionError):
                    dataset.split_episode_ids("test")
                self.assertEqual(dataset.opened_test_payloads, 0)

    def test_manifest_tamper_and_parent_group_leak_are_rejected(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _fixture(root)
            manifest = root / "metadata/splits_v1_seed42/sample_manifest.csv"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                IndependentTrainingDataset.open(root)
            rows[1]["parent_sample_id"] = rows[1]["group_id"] = rows[0]["parent_sample_id"]
            _freeze(root, rows)
            with self.assertRaisesRegex(ValueError, "group|parent"):
                IndependentTrainingDataset.open(root)

    def test_changed_train_payload_fails_before_parsing(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _fixture(root)
            dataset = IndependentTrainingDataset.open(root)
            (root / rows[0]["relative_path"]).write_bytes(b"broken payload")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                dataset.load_train()

    def test_authenticated_nonuniform_canonical_time_is_rejected(self) -> None:
        from v4.independent_dataset import IndependentTrainingDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = _fixture(root)
            path = root / rows[0]["relative_path"]
            frame = pd.read_csv(path)
            frame.loc[10, "time_s"] = 9
            frame.to_csv(path, index=False)
            rows[0]["sha256"] = _sha(path)
            _freeze(root, rows)
            with self.assertRaisesRegex(ValueError, "1.s|one.second|time"):
                IndependentTrainingDataset.open(root).load_train()


if __name__ == "__main__":
    unittest.main()
