from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from v2.contracts import control_semantics
from v2.dqn.action_space import ACTION_CATALOG_DIGEST, FINAL_DQN_ACTION_CATALOG
from v2.dqn.state import FORMAL_STATE_SCHEMA_DIGEST, FORMAL_STATE_SCHEMA_VERSION
from v2.failure_policy import FORMAL_FAILURE_POLICY
from v2.training.checkpoint import CHECKPOINT_VERSION
from v2.evaluation.checkpoint_selection import (
    RANKING_RULE,
    SELECTION_SCHEMA_VERSION,
    SelectionManifest,
    ValidationCandidate,
    authenticate_checkpoint_candidates,
    canonical_result_digest,
    load_selection_outputs,
    make_selection_manifest,
    select_best_candidate,
    selection_key,
    write_selection_outputs,
)


HASHES = {
    "power": "1" * 64,
    "ais": "2" * 64,
    "modes": "3" * 64,
}


def candidate(
    round_index: int,
    *,
    completed: int = 4,
    failed: int = 0,
    penalty: float | None = None,
    raw: float = 100.0,
    reward: float | None = None,
    checkpoint_hash: str | None = None,
) -> ValidationCandidate:
    actual_penalty = (
        float(failed * FORMAL_FAILURE_POLICY.penalty_score)
        if penalty is None
        else penalty
    )
    actual_reward = -raw - actual_penalty if reward is None else reward
    return ValidationCandidate(
        round_index=round_index,
        checkpoint_sha256=checkpoint_hash or f"{round_index:064x}",
        completed_episodes=completed,
        failed_episodes=failed,
        failure_penalty_score=actual_penalty,
        raw_economic_cost_cny=raw,
        learning_reward=actual_reward,
    )


def candidates(count: int = 4) -> tuple[ValidationCandidate, ...]:
    return tuple(candidate(index, raw=100.0 + index) for index in range(1, count + 1))


def manifest(values: tuple[ValidationCandidate, ...]) -> SelectionManifest:
    return make_selection_manifest(HASHES, values, required_rounds=range(1, len(values) + 1))


class TestValidationCandidate(unittest.TestCase):
    def test_ranking_precedence_is_exact(self) -> None:
        scenarios = (
            (
                candidate(8, completed=8, failed=0, penalty=0.0, raw=100.0),
                candidate(9, completed=7, failed=1, penalty=0.0, raw=1.0),
                8,
            ),
            (
                candidate(8, completed=8, failed=1, penalty=50_000.0, raw=10.0),
                candidate(9, completed=8, failed=1, penalty=0.0, raw=100.0),
                9,
            ),
            (
                candidate(8, completed=8, failed=0, penalty=0.0, raw=100.0),
                candidate(9, completed=8, failed=0, penalty=0.0, raw=90.0),
                9,
            ),
            (
                candidate(8, completed=8, failed=0, penalty=0.0, raw=90.0),
                candidate(9, completed=8, failed=0, penalty=0.0, raw=90.0),
                8,
            ),
        )
        for first, second, expected_round in scenarios:
            with self.subTest(expected_round=expected_round):
                winner = select_best_candidate(
                    (first, second), required_rounds=range(8, 10)
                )
                self.assertEqual(winner.round_index, expected_round)
        self.assertEqual(selection_key(scenarios[2][1]), (-8, 0.0, 90.0, 9))

    def test_duplicate_missing_and_extra_rounds_are_rejected(self) -> None:
        for values in (
            (candidate(1), candidate(1), candidate(2)),
            (candidate(1), candidate(3)),
            (candidate(1), candidate(2), candidate(3)),
        ):
            with self.subTest(rounds=[value.round_index for value in values]):
                with self.assertRaises(ValueError):
                    select_best_candidate(values, required_rounds=range(1, 3))
        with self.assertRaises(ValueError):
            select_best_candidate(
                (candidate(1), candidate(3)), required_rounds=(1, 3)
            )

    def test_default_formal_selection_requires_exactly_rounds_one_to_forty(self) -> None:
        with self.assertRaises(ValueError):
            select_best_candidate(candidates())
        values = candidates(40)
        self.assertEqual(select_best_candidate(values).round_index, 1)

    def test_nonfinite_metrics_bool_aliases_and_inconsistent_totals_are_rejected(self) -> None:
        valid = vars(candidate(1)).copy()
        invalid_changes = (
            {"round_index": True},
            {"completed_episodes": True},
            {"failed_episodes": True},
            {"failure_penalty_score": float("nan")},
            {"failure_penalty_score": 0},
            {"raw_economic_cost_cny": float("inf")},
            {"raw_economic_cost_cny": 100},
            {"learning_reward": float("-inf")},
            {"failure_penalty_score": 1.0},
            {"learning_reward": -99.0},
        )
        for change in invalid_changes:
            with self.subTest(change=change):
                fields = valid | change
                with self.assertRaises((TypeError, ValueError)):
                    ValidationCandidate(**fields)

    def test_hash_must_be_exact_lowercase_sha256(self) -> None:
        for value in ("a" * 63, "A" * 64, "g" * 64, bytes(32)):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    candidate(1, checkpoint_hash=value)  # type: ignore[arg-type]


class TestCheckpointAuthentication(unittest.TestCase):
    @staticmethod
    def _checkpoint(round_index: int, *, semantics: object | None = None) -> dict[str, object]:
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "state_schema_version": FORMAL_STATE_SCHEMA_VERSION,
            "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
            "action_catalog_digest": ACTION_CATALOG_DIGEST,
            "action_dim": len(FINAL_DQN_ACTION_CATALOG),
            "semantics": control_semantics() if semantics is None else semantics,
            "failure_policy": asdict(FORMAL_FAILURE_POLICY),
            "round_index": round_index,
        }

    def _write_bank(self, root: Path, count: int = 4) -> None:
        for round_index in range(1, count + 1):
            torch.save(self._checkpoint(round_index), root / f"round_{round_index:03d}.pt")
        (root / "latest.pt").write_bytes(b"ignored")

    def test_authenticates_numeric_order_hashes_and_ignores_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_bank(root)
            authenticated = authenticate_checkpoint_candidates(
                root, required_rounds=range(1, 5)
            )
            self.assertEqual(tuple(item.round_index for item in authenticated), (1, 2, 3, 4))
            self.assertEqual(
                authenticated[0].checkpoint_sha256,
                hashlib.sha256((root / "round_001.pt").read_bytes()).hexdigest(),
            )

    def test_missing_extra_bad_filename_and_metadata_round_are_rejected(self) -> None:
        mutations = ("missing", "extra", "bad_filename", "metadata")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self._write_bank(root)
                if mutation == "missing":
                    (root / "round_004.pt").unlink()
                elif mutation == "extra":
                    torch.save(self._checkpoint(5), root / "round_005.pt")
                elif mutation == "bad_filename":
                    torch.save(self._checkpoint(4), root / "round_04.pt")
                else:
                    torch.save(self._checkpoint(3), root / "round_004.pt")
                with self.assertRaises(ValueError):
                    authenticate_checkpoint_candidates(root, required_rounds=range(1, 5))

    def test_altered_v3_semantics_are_rejected_before_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_bank(root)
            altered = control_semantics()
            altered["reward_version"] = "altered"
            torch.save(self._checkpoint(2, semantics=altered), root / "round_002.pt")
            with self.assertRaises(ValueError):
                authenticate_checkpoint_candidates(root, required_rounds=range(1, 5))

    def test_study_authentication_binds_profile_and_reward_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_bank(root)
            for round_index in range(1, 5):
                path = root / f"round_{round_index:03d}.pt"
                payload = torch.load(path, map_location="cpu", weights_only=False)
                payload["training_config_identity"] = {
                    "experiment_id": "H2",
                    "reward_mode": "scaled",
                    "reward_scaling_identity": "a" * 64,
                }
                torch.save(payload, path)
            expected = {
                "experiment_id": "H2",
                "reward_mode": "scaled",
                "reward_scaling_identity": "a" * 64,
            }
            authenticated = authenticate_checkpoint_candidates(
                root,
                required_rounds=range(1, 5),
                expected_training_identity=expected,
            )
            self.assertEqual(len(authenticated), 4)
            expected["experiment_id"] = "H3"
            with self.assertRaises(ValueError):
                authenticate_checkpoint_candidates(
                    root,
                    required_rounds=range(1, 5),
                    expected_training_identity=expected,
                )


class TestSelectionManifest(unittest.TestCase):
    def test_manifest_binds_frozen_identity_and_digest_roundtrips(self) -> None:
        value = manifest(candidates())
        document = value.to_document()
        self.assertEqual(document["schema_version"], SELECTION_SCHEMA_VERSION)
        self.assertEqual(document["ranking_rule"], list(RANKING_RULE))
        self.assertEqual(document["input_manifest_sha256"], HASHES)
        self.assertEqual(document["state_schema_digest"], FORMAL_STATE_SCHEMA_DIGEST)
        self.assertEqual(document["action_catalog_digest"], ACTION_CATALOG_DIGEST)
        self.assertEqual(document["control_semantics"], control_semantics())
        self.assertEqual(document["failure_policy"], asdict(FORMAL_FAILURE_POLICY))
        digest_body = document.copy()
        digest_body.pop("result_digest")
        self.assertEqual(value.result_digest, canonical_result_digest(digest_body))
        self.assertEqual(SelectionManifest.from_json_bytes(value.to_json_bytes()), value)

    def test_unknown_fields_noncanonical_json_and_bad_digest_are_rejected(self) -> None:
        value = manifest(candidates())
        document = value.to_document()
        variants = []
        variants.append(document | {"unknown": 1})
        bad_candidate = json.loads(json.dumps(document))
        bad_candidate["candidates"][0]["unknown"] = 1
        variants.append(bad_candidate)
        bad_digest = json.loads(json.dumps(document))
        bad_digest["result_digest"] = "f" * 64
        variants.append(bad_digest)
        for payload in variants:
            with self.subTest(keys=payload.keys()):
                data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
                with self.assertRaises(ValueError):
                    SelectionManifest.from_json_bytes(data)
        noncanonical = json.dumps(document, indent=2, sort_keys=False).encode("utf-8")
        with self.assertRaises(ValueError):
            SelectionManifest.from_json_bytes(noncanonical)

    def test_manifest_rejects_bool_numeric_aliases(self) -> None:
        document = manifest(candidates()).to_document()
        document["selected_round"] = True
        body = document.copy()
        body.pop("result_digest")
        document["result_digest"] = canonical_result_digest(body)
        data = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")
        with self.assertRaises((TypeError, ValueError)):
            SelectionManifest.from_json_bytes(data)


class TestSelectionOutputs(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[SelectionManifest, Path]:
        source = root / "round_001.pt"
        source.write_bytes(b"selected checkpoint bytes")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        values = (
            candidate(1, raw=10.0, checkpoint_hash=source_hash),
            candidate(2, raw=20.0, checkpoint_hash="2" * 64),
        )
        return manifest(values), source

    def test_writes_and_loads_atomic_deterministic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value, source = self._inputs(root)
            destination = root / "sealed"
            write_selection_outputs(destination, value, source)
            self.assertEqual(
                sorted(path.name for path in destination.iterdir()),
                ["best_validation.pt", "selection_manifest.json", "validation_checkpoint_metrics.csv"],
            )
            self.assertEqual((destination / "best_validation.pt").read_bytes(), source.read_bytes())
            self.assertEqual(load_selection_outputs(destination), value)
            with (destination / "validation_checkpoint_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["round_index"] for row in rows], ["1", "2"])

    def test_preexisting_destination_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value, source = self._inputs(root)
            destination = root / "sealed"
            destination.mkdir()
            marker = destination / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_selection_outputs(destination, value, source)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_source_mismatch_cleans_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value, source = self._inputs(root)
            source.write_bytes(b"tampered")
            destination = root / "sealed"
            with self.assertRaises(ValueError):
                write_selection_outputs(destination, value, source)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".sealed.tmp-*")), [])

    def test_write_failure_cleans_created_temporary_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value, source = self._inputs(root)
            destination = root / "sealed"
            with (
                mock.patch(
                    "v2.evaluation.checkpoint_selection.shutil.copyfile",
                    side_effect=OSError("synthetic copy failure"),
                ),
                self.assertRaises(OSError),
            ):
                write_selection_outputs(destination, value, source)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.glob(".sealed.tmp-*")), [])

    def test_corrupted_json_csv_and_checkpoint_copy_are_rejected(self) -> None:
        mutations = ("json", "csv", "checkpoint")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                value, source = self._inputs(root)
                destination = root / "sealed"
                write_selection_outputs(destination, value, source)
                if mutation == "json":
                    path = destination / "selection_manifest.json"
                    path.write_bytes(path.read_bytes() + b"\n")
                elif mutation == "csv":
                    path = destination / "validation_checkpoint_metrics.csv"
                    path.write_text(path.read_text(encoding="utf-8") + "junk\n", encoding="utf-8")
                else:
                    (destination / "best_validation.pt").write_bytes(b"tampered")
                with self.assertRaises(ValueError):
                    load_selection_outputs(destination)


if __name__ == "__main__":
    unittest.main()
