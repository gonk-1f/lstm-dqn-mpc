"""Validation-only ranking and sealed outputs for H1-H4 pilot runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import uuid

from ..dqn.action_space import ACTION_CATALOG_DIGEST
from ..dqn.state import FORMAL_STATE_SCHEMA_DIGEST
from .checkpoint_selection import (
    ValidationCandidate,
    canonical_result_digest,
    select_best_candidate,
)


STUDY_SELECTION_SCHEMA_VERSION = "v2_history_dqn_study_selection_v1"
STUDY_PILOT_ROUNDS = 10
STUDY_TARGET_ROUND = 40
STUDY_RANKING_RULE = (
    "-completed_episodes",
    "failure_penalty_score",
    "raw_economic_cost_cny",
    "experiment_id",
    "round_index",
)
_PROFILE_IDS = ("H1", "H2", "H3", "H4")
_HASH_KEYS = ("power", "ais", "modes")


def _hash(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class StudyProfileRun:
    experiment_id: str
    reward_mode: str
    reward_scaling_identity: str
    state_schema_digest: str
    action_catalog_digest: str
    dataset_identity: str
    completed_rounds: int
    resume_path: Path
    validation_candidates: tuple[ValidationCandidate, ...]

    def __post_init__(self) -> None:
        if self.experiment_id not in _PROFILE_IDS:
            raise ValueError("experiment_id must be H1-H4")
        expected_mode = "raw" if self.experiment_id == "H1" else "scaled"
        if self.reward_mode != expected_mode:
            raise ValueError("reward_mode differs from the study matrix")
        if type(self.reward_scaling_identity) is not str or not self.reward_scaling_identity:
            raise ValueError("reward_scaling_identity must be nonempty")
        if self.reward_mode == "raw":
            if self.reward_scaling_identity != "raw_learning_reward_v1":
                raise ValueError("H1 reward identity differs")
        else:
            _hash(self.reward_scaling_identity, "reward_scaling_identity")
        _hash(self.state_schema_digest, "state_schema_digest")
        _hash(self.action_catalog_digest, "action_catalog_digest")
        _hash(self.dataset_identity, "dataset_identity")
        if type(self.completed_rounds) is not int or self.completed_rounds <= 0:
            raise ValueError("completed_rounds must be positive")
        if not isinstance(self.resume_path, Path) or self.resume_path.name != "latest.pt":
            raise ValueError("resume_path must identify latest.pt")
        if type(self.validation_candidates) is not tuple or any(
            type(item) is not ValidationCandidate for item in self.validation_candidates
        ):
            raise TypeError("validation_candidates must contain exact candidates")


@dataclass(frozen=True)
class SelectedStudyCandidate:
    experiment_id: str
    best_round_index: int
    completed_rounds: int
    resume_path: Path
    validation: ValidationCandidate


def select_study_candidates(
    candidates: tuple[StudyProfileRun, ...] | list[StudyProfileRun],
    *,
    required_profiles: tuple[str, ...] = _PROFILE_IDS,
    top_k: int = 2,
) -> tuple[SelectedStudyCandidate, ...]:
    if type(candidates) not in (tuple, list):
        raise TypeError("candidates must be an exact tuple or list")
    runs = tuple(candidates)
    if type(required_profiles) is not tuple or not required_profiles:
        raise ValueError("required_profiles must be a nonempty exact tuple")
    if len(set(required_profiles)) != len(required_profiles):
        raise ValueError("required_profiles must be unique")
    if any(type(run) is not StudyProfileRun for run in runs):
        raise TypeError("study candidates must be exact StudyProfileRun values")
    actual_ids = tuple(run.experiment_id for run in runs)
    if len(set(actual_ids)) != len(actual_ids) or set(actual_ids) != set(required_profiles):
        raise ValueError("study runs do not exactly match required profiles")
    if type(top_k) is not int or not 0 < top_k <= len(required_profiles):
        raise ValueError("top_k is outside the required profile count")
    if any(run.completed_rounds != STUDY_PILOT_ROUNDS for run in runs):
        raise ValueError("every pilot must complete exactly ten rounds")
    if {run.state_schema_digest for run in runs} != {FORMAL_STATE_SCHEMA_DIGEST}:
        raise ValueError("study state identities differ")
    if {run.action_catalog_digest for run in runs} != {ACTION_CATALOG_DIGEST}:
        raise ValueError("study action identities differ")
    if len({run.dataset_identity for run in runs}) != 1:
        raise ValueError("study dataset identities differ")
    scaled_identities = {
        run.reward_scaling_identity for run in runs if run.reward_mode == "scaled"
    }
    if len(scaled_identities) != 1:
        raise ValueError("scaled profiles do not share one reward calibration")

    selected: list[SelectedStudyCandidate] = []
    required_rounds = range(1, STUDY_PILOT_ROUNDS + 1)
    for run in runs:
        best = select_best_candidate(
            run.validation_candidates,
            required_rounds=required_rounds,
        )
        selected.append(
            SelectedStudyCandidate(
                experiment_id=run.experiment_id,
                best_round_index=best.round_index,
                completed_rounds=run.completed_rounds,
                resume_path=run.resume_path,
                validation=best,
            )
        )
    selected.sort(
        key=lambda item: (
            -item.validation.completed_episodes,
            item.validation.failure_penalty_score,
            item.validation.raw_economic_cost_cny,
            item.experiment_id,
            item.best_round_index,
        )
    )
    return tuple(selected[:top_k])


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _validation_document(value: ValidationCandidate) -> dict[str, object]:
    return value.to_document()


def write_study_selection_outputs(
    destination: Path,
    *,
    runs: tuple[StudyProfileRun, ...],
    selected: tuple[SelectedStudyCandidate, ...],
    input_manifest_hashes: dict[str, str],
) -> None:
    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if type(input_manifest_hashes) is not dict or set(input_manifest_hashes) != set(_HASH_KEYS):
        raise ValueError("input manifest hashes must use exact keys")
    hashes = {key: _hash(input_manifest_hashes[key], key) for key in _HASH_KEYS}
    canonical_selected = select_study_candidates(
        runs,
        required_profiles=_PROFILE_IDS,
        top_k=len(selected),
    )
    if selected != canonical_selected:
        raise ValueError("selected profiles do not follow the study ranking rule")

    profile_rows = []
    for run in sorted(runs, key=lambda value: value.experiment_id):
        for candidate in sorted(run.validation_candidates, key=lambda value: value.round_index):
            profile_rows.append(
                {"experiment_id": run.experiment_id, **candidate.to_document()}
            )
    selected_rows = [
        {
            "experiment_id": item.experiment_id,
            "best_round_index": item.best_round_index,
            "completed_rounds": item.completed_rounds,
            "resume_path": str(item.resume_path),
            "target_round": STUDY_TARGET_ROUND,
        }
        for item in selected
    ]
    body: dict[str, object] = {
        "schema_version": STUDY_SELECTION_SCHEMA_VERSION,
        "input_manifest_sha256": hashes,
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "dataset_identity": runs[0].dataset_identity,
        "ranking_rule": list(STUDY_RANKING_RULE),
        "profile_runs": [
            {
                "experiment_id": run.experiment_id,
                "reward_mode": run.reward_mode,
                "reward_scaling_identity": run.reward_scaling_identity,
                "completed_rounds": run.completed_rounds,
                "resume_path": str(run.resume_path),
                "validation_candidates": [
                    _validation_document(item) for item in run.validation_candidates
                ],
            }
            for run in sorted(runs, key=lambda value: value.experiment_id)
        ],
        "selected_profiles": selected_rows,
        "test_payloads_opened": 0,
    }
    body["result_digest"] = canonical_result_digest(body)
    top_document = {
        "schema_version": STUDY_SELECTION_SCHEMA_VERSION,
        "profiles": selected_rows,
        "selection_result_digest": body["result_digest"],
    }
    stream = io.StringIO(newline="")
    fields = (
        "experiment_id",
        "round_index",
        "checkpoint_sha256",
        "completed_episodes",
        "failed_episodes",
        "failure_penalty_score",
        "raw_economic_cost_cny",
        "learning_reward",
    )
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(profile_rows)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        (temporary / "profile_validation_metrics.csv").write_text(
            stream.getvalue(), encoding="utf-8", newline=""
        )
        (temporary / "study_selection_manifest.json").write_bytes(_canonical_bytes(body))
        (temporary / "top_profiles.json").write_bytes(_canonical_bytes(top_document))
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


__all__ = [
    "STUDY_PILOT_ROUNDS",
    "STUDY_RANKING_RULE",
    "STUDY_SELECTION_SCHEMA_VERSION",
    "STUDY_TARGET_ROUND",
    "SelectedStudyCandidate",
    "StudyProfileRun",
    "select_study_candidates",
    "write_study_selection_outputs",
]
