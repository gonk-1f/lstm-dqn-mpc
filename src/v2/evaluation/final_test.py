"""Sealed authorization and deterministic artifacts for one final Test run."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable

from ..contracts import control_semantics
from ..dqn.action_space import ACTION_CATALOG_DIGEST
from ..dqn.state import FORMAL_STATE_SCHEMA_DIGEST
from ..failure_policy import FORMAL_FAILURE_POLICY
from .checkpoint_selection import (
    FORMAL_SELECTION_ROUNDS,
    SELECTION_SCHEMA_VERSION,
    SelectionManifest,
    canonical_result_digest,
    load_selection_outputs,
)
from .formal_policy import EpisodeEvaluation, PolicyEvaluation


FINAL_TEST_CONFIRMATION = "FINAL_TEST_ONCE"
FINAL_TEST_RUN_SCHEMA_VERSION = "v2_final_test_evaluation_v1"
FINAL_TEST_ACCESS_SCHEMA_VERSION = "v2_final_test_access_v1"
_AUTHORIZATION_SEAL = object()
_MANIFEST_KEYS = ("power", "ais", "modes")
_OUTPUT_FILES = {
    "TEST_ACCESS_STARTED.json",
    "test_summary.json",
    "test_episode_metrics.csv",
    "test_action_distribution.csv",
    "run_manifest.json",
}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, init=False)
class FinalTestAuthorization:
    selection_schema_version: str
    selection_result_digest: str
    selected_round: int
    selected_checkpoint_sha256: str
    input_manifest_hashes: tuple[tuple[str, str], ...]
    state_schema_digest: str
    action_catalog_digest: str
    control_semantics_json: bytes
    failure_policy_json: bytes
    authorization_digest: str
    _seal: object


def _authorization_body(values: dict[str, object]) -> dict[str, object]:
    return {
        "selection_schema_version": values["selection_schema_version"],
        "selection_result_digest": values["selection_result_digest"],
        "selected_round": values["selected_round"],
        "selected_checkpoint_sha256": values["selected_checkpoint_sha256"],
        "input_manifest_sha256": dict(values["input_manifest_hashes"]),
        "state_schema_digest": values["state_schema_digest"],
        "action_catalog_digest": values["action_catalog_digest"],
        "control_semantics_json": values["control_semantics_json"].decode("ascii"),
        "failure_policy_json": values["failure_policy_json"].decode("ascii"),
    }


def _issue_authorization(manifest: SelectionManifest) -> FinalTestAuthorization:
    value = object.__new__(FinalTestAuthorization)
    stored = {
        "selection_schema_version": SELECTION_SCHEMA_VERSION,
        "selection_result_digest": manifest.result_digest,
        "selected_round": manifest.selected_round,
        "selected_checkpoint_sha256": manifest.copied_best_sha256,
        "input_manifest_hashes": tuple(
            (str(key), str(item)) for key, item in manifest.input_manifest_hashes
        ),
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "control_semantics_json": _canonical_json(control_semantics()),
        "failure_policy_json": _canonical_json(asdict(FORMAL_FAILURE_POLICY)),
    }
    stored["authorization_digest"] = canonical_result_digest(
        _authorization_body(stored)
    )
    stored["_seal"] = _AUTHORIZATION_SEAL
    for key, item in stored.items():
        object.__setattr__(value, key, item)
    return value


def require_final_test_authorization(value: object) -> FinalTestAuthorization:
    """Reject anything except an intact authority issued by this module."""

    if type(value) is not FinalTestAuthorization:
        raise TypeError("final Test access requires an exact authorization")
    expected_fields = {item.name for item in fields(FinalTestAuthorization)}
    if set(vars(value)) != expected_fields or value._seal is not _AUTHORIZATION_SEAL:
        raise PermissionError("final Test authorization is forged or incomplete")
    if value.selection_schema_version != SELECTION_SCHEMA_VERSION:
        raise PermissionError("selection schema authorization differs")
    if type(value.selected_round) is not int or value.selected_round not in FORMAL_SELECTION_ROUNDS:
        raise PermissionError("selected round is outside the formal checkpoint bank")
    for digest in (
        value.selection_result_digest,
        value.selected_checkpoint_sha256,
        value.state_schema_digest,
        value.action_catalog_digest,
        value.authorization_digest,
    ):
        if type(digest) is not str or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise PermissionError("authorization contains a malformed digest")
    if value.state_schema_digest != FORMAL_STATE_SCHEMA_DIGEST:
        raise PermissionError("authorized state schema differs")
    if value.action_catalog_digest != ACTION_CATALOG_DIGEST:
        raise PermissionError("authorized action catalog differs")
    if type(value.control_semantics_json) is not bytes:
        raise PermissionError("authorized control semantics representation differs")
    if type(value.failure_policy_json) is not bytes:
        raise PermissionError("authorized failure policy representation differs")
    if value.control_semantics_json != _canonical_json(control_semantics()):
        raise PermissionError("authorized control semantics differ")
    if value.failure_policy_json != _canonical_json(asdict(FORMAL_FAILURE_POLICY)):
        raise PermissionError("authorized failure policy differs")
    if (
        type(value.input_manifest_hashes) is not tuple
        or tuple(key for key, _ in value.input_manifest_hashes) != _MANIFEST_KEYS
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or len(pair[1]) != 64
            or any(character not in "0123456789abcdef" for character in pair[1])
            for pair in value.input_manifest_hashes
        )
    ):
        raise PermissionError("authorized input manifest identity differs")
    if value.authorization_digest != canonical_result_digest(
        _authorization_body(vars(value))
    ):
        raise PermissionError("final Test authorization digest differs")
    return value


def authorization_manifest_hashes(value: object) -> dict[str, str]:
    checked = require_final_test_authorization(value)
    return dict(checked.input_manifest_hashes)


def authenticate_final_test_selection(selection_directory: Path) -> FinalTestAuthorization:
    """Authenticate a complete 40-round selection and issue Test authority."""

    manifest = load_selection_outputs(Path(selection_directory))
    rounds = tuple(candidate.round_index for candidate in manifest.candidates)
    if rounds != FORMAL_SELECTION_ROUNDS:
        raise ValueError("final Test requires the complete formal 40-round selection")
    return require_final_test_authorization(_issue_authorization(manifest))


def _access_document(
    authorization: FinalTestAuthorization,
    *,
    status: str,
    run_result_digest: str | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": FINAL_TEST_ACCESS_SCHEMA_VERSION,
        "status": status,
        "selection_result_digest": authorization.selection_result_digest,
        "selected_round": authorization.selected_round,
        "best_checkpoint_sha256": authorization.selected_checkpoint_sha256,
    }
    if run_result_digest is not None:
        document["run_result_digest"] = run_result_digest
    return document


def create_final_test_lock(
    output_directory: Path,
    authorization: FinalTestAuthorization,
) -> Path:
    checked = require_final_test_authorization(authorization)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "TEST_ACCESS_STARTED.json"
    marker.write_bytes(_canonical_json(_access_document(checked, status="STARTED")))
    return marker


def _summary(value: PolicyEvaluation) -> dict[str, object]:
    return {
        "policy_id": value.policy_id,
        "episode_count": len(value.episodes),
        "completed_episodes": value.completed_episodes,
        "failed_episodes": value.failed_episodes,
        "completion_rate": value.completion_rate,
        "transition_count": value.transition_count,
        "executed_mpc_steps": value.executed_mpc_steps,
        "h2_cost_cny": value.h2_cost_cny,
        "fc_degradation_cost_cny": value.fc_degradation_cost_cny,
        "battery_degradation_cost_cny": value.battery_degradation_cost_cny,
        "shore_cost_cny": value.shore_cost_cny,
        "raw_economic_cost_cny": value.raw_economic_cost_cny,
        "failure_penalty_score": value.failure_penalty_score,
        "learning_reward": value.learning_reward,
        "soc_min": min(episode.soc_min for episode in value.episodes),
        "soc_max": max(episode.soc_max for episode in value.episodes),
    }


def _episode_csv(evaluations: tuple[PolicyEvaluation, ...]) -> bytes:
    field_names = ("policy_id",) + tuple(
        item.name for item in fields(EpisodeEvaluation) if item.name != "action_counts"
    ) + ("action_counts_json",)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, lineterminator="\n")
    writer.writeheader()
    for evaluation in evaluations:
        for episode in evaluation.episodes:
            row = {
                item.name: getattr(episode, item.name)
                for item in fields(EpisodeEvaluation)
                if item.name != "action_counts"
            }
            row["policy_id"] = evaluation.policy_id
            row["action_counts_json"] = _canonical_json(
                [[action_id, count] for action_id, count in episode.action_counts]
            ).decode("ascii")
            writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _action_csv(evaluations: tuple[PolicyEvaluation, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("policy_id", "action_id", "count"),
        lineterminator="\n",
    )
    writer.writeheader()
    for evaluation in evaluations:
        for action_id, count in evaluation.action_counts:
            writer.writerow(
                {"policy_id": evaluation.policy_id, "action_id": action_id, "count": count}
            )
    return stream.getvalue().encode("utf-8")


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_final_test_results(
    output_directory: Path,
    authorization: FinalTestAuthorization,
    evaluations: Iterable[PolicyEvaluation],
) -> str:
    """Write complete deterministic results and then change STARTED to COMPLETE."""

    checked = require_final_test_authorization(authorization)
    output = Path(output_directory)
    marker = output / "TEST_ACCESS_STARTED.json"
    expected_started = _canonical_json(_access_document(checked, status="STARTED"))
    if not output.is_dir() or not marker.is_file() or marker.read_bytes() != expected_started:
        raise PermissionError("final Test output does not hold the exact STARTED lock")
    values = tuple(evaluations)
    if (
        len(values) != 2
        or any(type(value) is not PolicyEvaluation for value in values)
        or tuple(value.policy_id for value in values) != ("greedy_dqn", "w_8_1_1")
    ):
        raise ValueError("final Test requires greedy DQN and w_8_1_1 evaluations")
    if values[0].episode_ids != values[1].episode_ids:
        raise ValueError("final Test policies must use the same ordered episodes")

    summary_document = {
        "schema_version": FINAL_TEST_RUN_SCHEMA_VERSION,
        "selection_result_digest": checked.selection_result_digest,
        "selected_round": checked.selected_round,
        "test_episode_ids": list(values[0].episode_ids),
        "policies": [_summary(value) for value in values],
    }
    payloads = {
        "test_summary.json": _canonical_json(summary_document),
        "test_episode_metrics.csv": _episode_csv(values),
        "test_action_distribution.csv": _action_csv(values),
    }
    for name, data in payloads.items():
        _write_atomic(output / name, data)
    artifact_hashes = {name: _sha256_bytes(data) for name, data in payloads.items()}
    run_body = {
        "schema_version": FINAL_TEST_RUN_SCHEMA_VERSION,
        "status": "COMPLETE",
        "selection_result_digest": checked.selection_result_digest,
        "selected_round": checked.selected_round,
        "best_checkpoint_sha256": checked.selected_checkpoint_sha256,
        "input_manifest_sha256": dict(checked.input_manifest_hashes),
        "test_episode_ids": list(values[0].episode_ids),
        "policy_ids": [value.policy_id for value in values],
        "artifact_sha256": artifact_hashes,
    }
    run_document = dict(run_body)
    run_document["result_digest"] = canonical_result_digest(run_body)
    _write_atomic(output / "run_manifest.json", _canonical_json(run_document))
    if {path.name for path in output.iterdir()} != _OUTPUT_FILES:
        raise ValueError("final Test output file set differs")
    _write_atomic(
        marker,
        _canonical_json(
            _access_document(
                checked,
                status="COMPLETE",
                run_result_digest=run_document["result_digest"],
            )
        ),
    )
    return str(run_document["result_digest"])


__all__ = [
    "FINAL_TEST_ACCESS_SCHEMA_VERSION",
    "FINAL_TEST_CONFIRMATION",
    "FINAL_TEST_RUN_SCHEMA_VERSION",
    "FinalTestAuthorization",
    "authenticate_final_test_selection",
    "authorization_manifest_hashes",
    "create_final_test_lock",
    "require_final_test_authorization",
    "write_final_test_results",
]
