"""Authenticated independent voyages with complete, duration-aware 30 s intervals.

Only the frozen split metadata are read by ``open``. Test payloads have no
loading route here; final evaluation requires a separately authorized entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("dataset paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("manifest path escapes dataset root")
    return path


def _check_sha256(path: Path, expected: object) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"{path.name}: SHA-256 mismatch")


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


@dataclass(frozen=True)
class IndependentEpisode:
    sample_id: str
    split: str
    parent_sample_id: str
    timestamp: tuple[pd.Timestamp, ...]
    time_s: np.ndarray
    dt_s: np.ndarray
    load_kw: np.ndarray
    source_time_s: np.ndarray
    source_1s_row: np.ndarray
    duration_s: float
    terminal_boundary_time_s: float
    terminal_boundary_load_kw: float
    terminal_boundary_timestamp: pd.Timestamp
    terminal_boundary_source_time_s: float
    terminal_boundary_source_1s_row: int
    task_mode: str = "independent_voyage"

    @property
    def operating_mode(self) -> tuple[str, ...]:
        return ("onboard",) * self.step_count

    @property
    def step_count(self) -> int:
        return len(self.time_s)


class IndependentTrainingDataset:
    task_mode = "independent_voyage"

    def __init__(
        self, root: Path, manifest: pd.DataFrame, policy: dict, identity_paths: tuple[str, ...],
    ) -> None:
        self.root = root
        self._manifest = manifest
        self._policy = policy
        self._identity_paths = identity_paths
        self._initial_hashes = self.identity_hashes()
        self._cache: dict[str, tuple[IndependentEpisode, ...]] = {}

    @classmethod
    def open(cls, root: Path) -> "IndependentTrainingDataset":
        root = Path(root).resolve()
        active_relative = "metadata/active_split.json"
        active = json.loads(_contained(root, active_relative).read_text(encoding="utf-8"))
        manifest_relative = active["manifest_relative_path"]
        policy_relative = active["policy_relative_path"]
        manifest_path = _contained(root, manifest_relative)
        policy_path = _contained(root, policy_relative)
        _check_sha256(manifest_path, active["manifest_sha256"])
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if (policy.get("schema") != "independent_v2_split_v1"
                or policy.get("dataset_version") != "operating_dataset_independent_v2"
                or policy.get("split_version") != active["split_version"]
                or manifest_path.parent != policy_path.parent
                or manifest_path.parent.name != active["split_version"]
                or policy.get("historical_checkpoints_valid_for_new_task") is not False):
            raise ValueError("independent dataset split policy identity differs")
        _check_sha256(manifest_path, policy["sample_manifest_sha256"])
        payload_path = policy_path.parent / "payload_hashes.csv"
        _check_sha256(payload_path, policy["payload_hash_manifest_sha256"])
        manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", float_precision="round_trip")
        required = {
            "candidate_id", "parent_sample_id", "group_id", "split", "relative_path", "sha256",
            "preview_30s_relative_path", "preview_30s_sha256", "duration_s", "point_count",
            "unknown_points", "shore_points",
        }
        if required.difference(manifest.columns) or manifest[list(required)].isna().any().any():
            raise ValueError("independent sample manifest schema mismatch")
        if manifest["candidate_id"].duplicated().any():
            raise ValueError("independent manifest requires unique candidate identities")
        counts = {split: int(manifest["split"].eq(split).sum())
                  for split in ("train", "validation", "test")}
        if (set(manifest["split"]) != set(counts)
                or any(value <= 0 for value in counts.values())
                or counts != active["split_counts"] or counts != policy["split_counts"]):
            raise ValueError("independent split counts differ from frozen policy")
        if (not manifest["group_id"].equals(manifest["parent_sample_id"])
                or manifest.groupby("parent_sample_id")["split"].nunique().max() != 1):
            raise ValueError("independent parent group crosses split identities")
        if "new_split" in manifest and not manifest["new_split"].equals(manifest["split"]):
            raise ValueError("independent new_split and split differ")
        numeric = manifest[["duration_s", "point_count", "unknown_points", "shore_points"]].apply(
            pd.to_numeric, errors="coerce")
        if (not np.isfinite(numeric.to_numpy()).all()
                or not numeric.eq(np.floor(numeric)).all().all()
                or not numeric["duration_s"].gt(0).all()
                or not numeric["point_count"].eq(numeric["duration_s"] + 1).all()
                or not numeric[["unknown_points", "shore_points"]].eq(0).all().all()):
            raise ValueError("independent duration, 1 s count or excluded-mode metadata differs")
        payloads = pd.read_csv(payload_path, encoding="utf-8-sig")
        payload_columns = ["candidate_id", "split", "kind", "relative_path", "sha256"]
        if set(payload_columns).difference(payloads.columns):
            raise ValueError("independent payload hash manifest schema mismatch")
        expected_payloads = []
        for row in manifest.itertuples(index=False):
            for kind, relative, digest in (
                ("canonical_1s", row.relative_path, row.sha256),
                ("existing_30s_preview", row.preview_30s_relative_path, row.preview_30s_sha256),
            ):
                _contained(root, relative)  # Resolve paths without opening any payload.
                expected_payloads.append((row.candidate_id, row.split, kind, relative, digest))
        expected = pd.DataFrame(expected_payloads, columns=payload_columns)
        order = ["candidate_id", "kind"]
        if not (expected.sort_values(order).reset_index(drop=True).equals(
                payloads[payload_columns].sort_values(order).reset_index(drop=True))):
            raise ValueError("independent payload hashes differ from sample manifest")
        identity_paths = (active_relative, manifest_relative, policy_relative,
                          payload_path.relative_to(root).as_posix())
        if "exclusions_relative_path" in policy:
            exclusions_relative = policy["exclusions_relative_path"]
            _check_sha256(_contained(root, exclusions_relative), policy["exclusions_sha256"])
            identity_paths += (exclusions_relative,)
        return cls(root, manifest, policy, identity_paths)

    @property
    def split_sizes(self) -> tuple[int, int]:
        return tuple(int(self._manifest["split"].eq(split).sum())
                     for split in ("train", "validation"))

    @property
    def split_counts(self) -> dict[str, int]:
        return {split: int(self._manifest["split"].eq(split).sum())
                for split in ("train", "validation", "test")}

    @property
    def opened_test_payloads(self) -> int:
        return 0

    @property
    def task_contract(self) -> dict:
        return {
            "task_mode": self.task_mode, "dataset_version": self._policy["dataset_version"],
            "split_version": self._policy["split_version"], "initial_soc": .6,
            "shore_control": False, "terminal_charging": False, "model_step_s": 30,
            "execution_grid": "t=30*k<duration_s", "tail_policy": "exact_partial_interval",
            "load_sampling": "canonical_left_endpoint_no_interpolation",
            "loader_load_policy": "preserve_signed_source_values",
            "control_load_policy": "existing_minus1_to_zero_numerical_tolerance",
            "control_negative_load_tolerance_kw": 1.0,
            "historical_checkpoints_valid_for_new_task": False,
        }

    def identity_hashes(self) -> dict[str, str]:
        return {relative: _sha256(_contained(self.root, relative)) for relative in self._identity_paths}

    @staticmethod
    def _training_split(split: str) -> None:
        if type(split) is not str:
            raise TypeError("split must be an exact str")
        if split == "test":
            raise PermissionError("Test payload access is forbidden during training/model selection")
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")

    def split_episode_ids(self, split: str) -> tuple[str, ...]:
        self._training_split(split)
        return tuple(sorted(self._manifest.loc[self._manifest["split"].eq(split), "candidate_id"]))

    def _load_episode(self, row: object) -> IndependentEpisode:
        path = _contained(self.root, row.relative_path)
        _check_sha256(path, row.sha256)
        frame = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
        required = {"candidate_id", "parent_sample_id", "timestamp", "time_s", "load_total_kw",
                    "source_time_s", "source_1s_row"}
        if required.difference(frame.columns) or len(frame) != int(row.point_count):
            raise ValueError(f"{row.candidate_id}: canonical 1 s payload schema/count differs")
        if (not frame["candidate_id"].eq(row.candidate_id).all()
                or not frame["parent_sample_id"].eq(row.parent_sample_id).all()):
            raise ValueError(f"{row.candidate_id}: canonical payload identities differ")
        values = frame[["time_s", "load_total_kw", "source_time_s", "source_1s_row"]].apply(
            pd.to_numeric, errors="coerce")
        duration = int(row.duration_s)
        time = values["time_s"].to_numpy(dtype=float)
        source_time = values["source_time_s"].to_numpy(dtype=float)
        source_rows = values["source_1s_row"].to_numpy(dtype=float)
        if (not np.isfinite(values.to_numpy()).all()
                or not np.array_equal(time, np.arange(duration + 1, dtype=float))
                or not np.array_equal(np.diff(source_time), np.ones(duration))
                or not np.array_equal(np.diff(source_rows), np.ones(duration))
                or not np.equal(source_rows, np.floor(source_rows)).all()):
            raise ValueError(f"{row.candidate_id}: canonical 1 s time/source grid differs")
        stamps = tuple(pd.Timestamp(value) for value in frame["timestamp"])
        if (any(pd.isna(value) or value.tzinfo is None for value in stamps)
                or any((right - left).total_seconds() != 1 for left, right in zip(stamps, stamps[1:]))):
            raise ValueError(f"{row.candidate_id}: canonical timestamp grid differs")
        selected = np.arange(0, duration, 30, dtype=int)
        loads = values["load_total_kw"].to_numpy(dtype=float)
        return IndependentEpisode(
            sample_id=str(row.candidate_id), split=str(row.split),
            parent_sample_id=str(row.parent_sample_id), timestamp=tuple(stamps[index] for index in selected),
            time_s=_readonly(time[selected]), dt_s=_readonly(np.minimum(30, duration - selected).astype(float)),
            load_kw=_readonly(loads[selected]), source_time_s=_readonly(source_time[selected]),
            source_1s_row=_readonly(source_rows[selected].astype(np.int64)), duration_s=float(duration),
            terminal_boundary_time_s=float(time[-1]), terminal_boundary_load_kw=float(loads[-1]),
            terminal_boundary_timestamp=stamps[-1], terminal_boundary_source_time_s=float(source_time[-1]),
            terminal_boundary_source_1s_row=int(source_rows[-1]),
        )

    def load_split(self, split: str) -> tuple[IndependentEpisode, ...]:
        self._training_split(split)
        if self.identity_hashes() != self._initial_hashes:
            raise ValueError("frozen independent dataset metadata SHA-256 changed after opening")
        if split not in self._cache:
            rows = self._manifest[self._manifest["split"].eq(split)].sort_values("candidate_id")
            self._cache[split] = tuple(self._load_episode(row) for row in rows.itertuples(index=False))
        return self._cache[split]

    def load_train(self) -> tuple[IndependentEpisode, ...]:
        return self.load_split("train")

    def load_validation(self) -> tuple[IndependentEpisode, ...]:
        return self.load_split("validation")
