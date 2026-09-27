"""Pure Validation ranking and sealed checkpoint-selection artifacts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

import torch

from ..contracts import control_semantics, require_v2_semantics
from ..dqn.action_space import ACTION_CATALOG_DIGEST, FINAL_DQN_ACTION_CATALOG
from ..dqn.state import FORMAL_STATE_SCHEMA_DIGEST, FORMAL_STATE_SCHEMA_VERSION
from ..failure_policy import FORMAL_FAILURE_POLICY
from ..training.checkpoint import CHECKPOINT_VERSION


SELECTION_SCHEMA_VERSION = "v2_validation_checkpoint_selection_v1"
FORMAL_SELECTION_ROUNDS = tuple(range(1, 41))
RANKING_RULE = (
    "-completed_episodes",
    "failure_penalty_score",
    "raw_economic_cost_cny",
    "round_index",
)
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_RE = re.compile(r"round_([0-9]{3})\.pt")
_INPUT_HASH_KEYS = ("power", "ais", "modes")
_OUTPUT_FILENAMES = {
    "validation_checkpoint_metrics.csv",
    "selection_manifest.json",
    "best_validation.pt",
}


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256 digest")
    return value


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_result_digest(payload_without_digest: dict[str, object]) -> str:
    """Hash one manifest body using the repository canonical JSON convention."""

    if type(payload_without_digest) is not dict:
        raise TypeError("digest body must be an exact dict")
    if "result_digest" in payload_without_digest:
        raise ValueError("digest body must exclude result_digest")
    return hashlib.sha256(_canonical_json(payload_without_digest)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationCandidate:
    """The exact aggregate Validation metrics for one completed training round."""

    round_index: int
    checkpoint_sha256: str
    completed_episodes: int
    failed_episodes: int
    failure_penalty_score: float
    raw_economic_cost_cny: float
    learning_reward: float

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or self.round_index <= 0:
            raise ValueError("round_index must be an exact positive integer")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        for name, value in (
            ("completed_episodes", self.completed_episodes),
            ("failed_episodes", self.failed_episodes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an exact nonnegative integer")
        if self.completed_episodes + self.failed_episodes <= 0:
            raise ValueError("Validation must contain at least one episode")
        penalty = _finite_float(
            self.failure_penalty_score,
            "failure_penalty_score",
            nonnegative=True,
        )
        raw_cost = _finite_float(
            self.raw_economic_cost_cny,
            "raw_economic_cost_cny",
            nonnegative=True,
        )
        reward = _finite_float(self.learning_reward, "learning_reward")
        if self.failed_episodes == 0 and penalty != 0.0:
            raise ValueError("fully completed Validation cannot carry a failure penalty")
        if reward != -raw_cost - penalty:
            raise ValueError("learning reward must equal negative raw cost and penalty")

    def to_document(self) -> dict[str, object]:
        return {
            item.name: getattr(self, item.name)
            for item in fields(ValidationCandidate)
        }


def _validated_candidate(value: object) -> ValidationCandidate:
    if type(value) is not ValidationCandidate:
        raise TypeError("candidate set must contain exact ValidationCandidate values")
    expected_fields = {item.name for item in fields(ValidationCandidate)}
    if set(vars(value)) != expected_fields:
        raise ValueError("candidate contains altered stored fields")
    return ValidationCandidate(**vars(value))


def selection_key(candidate: ValidationCandidate) -> tuple[int, float, float, int]:
    """Return the approved deterministic Validation ranking key."""

    value = _validated_candidate(candidate)
    return (
        -value.completed_episodes,
        value.failure_penalty_score,
        value.raw_economic_cost_cny,
        value.round_index,
    )


def _required_round_tuple(required_rounds: Iterable[int]) -> tuple[int, ...]:
    rounds = tuple(required_rounds)
    if (
        not rounds
        or any(type(value) is not int or value <= 0 for value in rounds)
        or rounds != tuple(range(rounds[0], rounds[-1] + 1))
    ):
        raise ValueError("required rounds must be one ascending contiguous sequence")
    return rounds


def _validated_candidates(
    candidates: Iterable[ValidationCandidate],
    required_rounds: Iterable[int],
) -> tuple[ValidationCandidate, ...]:
    if type(candidates) not in (tuple, list):
        raise TypeError("candidates must be an exact tuple or list")
    values = tuple(_validated_candidate(value) for value in candidates)
    required = _required_round_tuple(required_rounds)
    actual = tuple(value.round_index for value in values)
    if len(set(actual)) != len(actual):
        raise ValueError("candidate rounds must be unique")
    if set(actual) != set(required):
        raise ValueError("candidate rounds do not exactly match the required rounds")
    episode_totals = {
        value.completed_episodes + value.failed_episodes for value in values
    }
    if len(episode_totals) != 1:
        raise ValueError("all candidates must cover the same Validation episodes")
    return values


def select_best_candidate(
    candidates: tuple[ValidationCandidate, ...] | list[ValidationCandidate],
    *,
    required_rounds: Iterable[int] = FORMAL_SELECTION_ROUNDS,
) -> ValidationCandidate:
    """Select one checkpoint using Validation only and the frozen ranking rule."""

    values = _validated_candidates(candidates, required_rounds)
    return min(values, key=selection_key)


@dataclass(frozen=True)
class AuthenticatedCheckpoint:
    round_index: int
    path: Path
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if type(self.round_index) is not int or self.round_index <= 0:
            raise ValueError("round_index must be an exact positive integer")
        if not isinstance(self.path, Path):
            raise TypeError("checkpoint path must be a Path")
        _require_sha256(self.checkpoint_sha256, "checkpoint_sha256")


def _load_trusted_checkpoint_metadata(path: Path) -> object:
    """Read a trusted local training checkpoint using the repository convention."""

    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_checkpoint_metadata(
    value: object,
    expected_round: int,
    expected_training_identity: dict[str, object] | None = None,
) -> None:
    if type(value) is not dict:
        raise ValueError("checkpoint payload must be an exact dict")
    expected_scalars = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "state_schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "action_dim": len(FINAL_DQN_ACTION_CATALOG),
        "round_index": expected_round,
    }
    for key, expected in expected_scalars.items():
        actual = value.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {key} identity differs")
    try:
        require_v2_semantics(value.get("semantics"))
    except Exception as exc:
        raise ValueError("checkpoint control semantics differ") from exc
    expected_policy = asdict(FORMAL_FAILURE_POLICY)
    policy = value.get("failure_policy")
    if type(policy) is not dict or policy.keys() != expected_policy.keys():
        raise ValueError("checkpoint failure policy differs")
    for key, expected in expected_policy.items():
        actual = policy[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError("checkpoint failure policy differs")
    if expected_training_identity is not None:
        if type(expected_training_identity) is not dict or set(
            expected_training_identity
        ) != {
            "experiment_id",
            "reward_mode",
            "reward_scaling_identity",
        }:
            raise ValueError("expected training identity keys are not exact")
        stored = value.get("training_config_identity")
        if type(stored) is not dict:
            raise ValueError("checkpoint training config identity is missing")
        for key, expected in expected_training_identity.items():
            if type(stored.get(key)) is not type(expected) or stored.get(key) != expected:
                raise ValueError(f"checkpoint {key} identity differs")


def authenticate_checkpoint_candidates(
    checkpoint_directory: Path,
    *,
    required_rounds: Iterable[int] = FORMAL_SELECTION_ROUNDS,
    metadata_loader: Callable[[Path], object] | None = None,
    expected_training_identity: dict[str, object] | None = None,
) -> tuple[AuthenticatedCheckpoint, ...]:
    """Authenticate an exact round checkpoint bank; ``latest.pt`` is ignored."""

    directory = Path(checkpoint_directory)
    if not directory.is_dir():
        raise ValueError("checkpoint directory does not exist")
    required = _required_round_tuple(required_rounds)
    discovered: dict[int, Path] = {}
    round_files = sorted(directory.glob("round_*.pt"), key=lambda path: path.name)
    for path in round_files:
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"checkpoint filename is not canonical: {path.name}")
        round_index = int(match.group(1))
        if round_index in discovered:
            raise ValueError("checkpoint round filenames are not unique")
        discovered[round_index] = path
    if set(discovered) != set(required):
        raise ValueError("checkpoint files do not exactly match required rounds")
    loader = metadata_loader or _load_trusted_checkpoint_metadata
    authenticated: list[AuthenticatedCheckpoint] = []
    for round_index in required:
        path = discovered[round_index]
        before_hash = _sha256(path)
        try:
            metadata = loader(path)
        except Exception as exc:
            raise ValueError(f"checkpoint cannot be read: {path.name}") from exc
        _validate_checkpoint_metadata(
            metadata,
            round_index,
            expected_training_identity,
        )
        after_hash = _sha256(path)
        if after_hash != before_hash:
            raise ValueError("checkpoint changed while it was authenticated")
        authenticated.append(AuthenticatedCheckpoint(round_index, path, after_hash))
    return tuple(authenticated)


def _pairs_from_mapping(
    value: object,
    *,
    keys: tuple[str, ...],
    name: str,
    hashes: bool = False,
) -> tuple[tuple[str, object], ...]:
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{name} must use exact keys")
    pairs: list[tuple[str, object]] = []
    for key in keys:
        item = value[key]
        if hashes:
            item = _require_sha256(item, f"{name}.{key}")
        pairs.append((key, item))
    return tuple(pairs)


def _manifest_body(
    *,
    input_manifest_hashes: tuple[tuple[str, object], ...],
    candidates: tuple[ValidationCandidate, ...],
    selected_round: int,
    selected_source_sha256: str,
    copied_best_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "input_manifest_sha256": dict(input_manifest_hashes),
        "state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "action_catalog_digest": ACTION_CATALOG_DIGEST,
        "control_semantics": control_semantics(),
        "failure_policy": asdict(FORMAL_FAILURE_POLICY),
        "ranking_rule": list(RANKING_RULE),
        "candidates": [candidate.to_document() for candidate in candidates],
        "selected_round": selected_round,
        "selected_source_sha256": selected_source_sha256,
        "copied_best_sha256": copied_best_sha256,
    }


@dataclass(frozen=True)
class SelectionManifest:
    """Canonical sealed identity of a completed Validation-only selection."""

    input_manifest_hashes: tuple[tuple[str, object], ...]
    candidates: tuple[ValidationCandidate, ...]
    selected_round: int
    selected_source_sha256: str
    copied_best_sha256: str
    result_digest: str

    def __post_init__(self) -> None:
        if type(self.input_manifest_hashes) is not tuple:
            raise TypeError("input manifest hashes must be an exact tuple")
        if any(
            type(item) is not tuple or len(item) != 2
            for item in self.input_manifest_hashes
        ):
            raise TypeError("input manifest hash entries must be exact pairs")
        hashes = dict(self.input_manifest_hashes)
        if len(hashes) != len(self.input_manifest_hashes):
            raise ValueError("input manifest hash keys must be unique")
        if tuple(key for key, _ in self.input_manifest_hashes) != _INPUT_HASH_KEYS:
            raise ValueError("input manifest hashes are not canonically ordered")
        _pairs_from_mapping(
            hashes,
            keys=_INPUT_HASH_KEYS,
            name="input_manifest_sha256",
            hashes=True,
        )
        if type(self.candidates) is not tuple:
            raise TypeError("manifest candidates must be an exact tuple")
        if any(type(value) is not ValidationCandidate for value in self.candidates):
            raise TypeError("manifest candidates must be exact ValidationCandidate values")
        manifest_rounds = tuple(value.round_index for value in self.candidates)
        values = _validated_candidates(self.candidates, manifest_rounds)
        if values != tuple(sorted(values, key=lambda value: value.round_index)):
            raise ValueError("manifest candidates must be ordered by round")
        if type(self.selected_round) is not int:
            raise TypeError("selected_round must be an exact integer")
        selected = select_best_candidate(values, required_rounds=manifest_rounds)
        if self.selected_round != selected.round_index:
            raise ValueError("selected round does not follow the ranking rule")
        _require_sha256(self.selected_source_sha256, "selected_source_sha256")
        _require_sha256(self.copied_best_sha256, "copied_best_sha256")
        if self.selected_source_sha256 != selected.checkpoint_sha256:
            raise ValueError("selected source hash differs from selected candidate")
        if self.copied_best_sha256 != self.selected_source_sha256:
            raise ValueError("copied best hash differs from selected source")
        _require_sha256(self.result_digest, "result_digest")
        body = _manifest_body(
            input_manifest_hashes=self.input_manifest_hashes,
            candidates=values,
            selected_round=self.selected_round,
            selected_source_sha256=self.selected_source_sha256,
            copied_best_sha256=self.copied_best_sha256,
        )
        if self.result_digest != canonical_result_digest(body):
            raise ValueError("selection manifest result digest differs")

    def to_document(self) -> dict[str, object]:
        body = _manifest_body(
            input_manifest_hashes=self.input_manifest_hashes,
            candidates=self.candidates,
            selected_round=self.selected_round,
            selected_source_sha256=self.selected_source_sha256,
            copied_best_sha256=self.copied_best_sha256,
        )
        body["result_digest"] = self.result_digest
        return body

    def to_json_bytes(self) -> bytes:
        return _canonical_json(self.to_document())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "SelectionManifest":
        if type(data) is not bytes:
            raise TypeError("manifest input must be exact bytes")

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            value: dict[str, object] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("selection manifest JSON contains duplicate keys")
                value[key] = item
            return value

        try:
            document = json.loads(data.decode("ascii"), object_pairs_hook=exact_object)
        except Exception as exc:
            raise ValueError("selection manifest is not valid canonical JSON") from exc
        if type(document) is not dict:
            raise ValueError("selection manifest must be a JSON object")
        expected_keys = {
            "schema_version",
            "input_manifest_sha256",
            "state_schema_digest",
            "action_catalog_digest",
            "control_semantics",
            "failure_policy",
            "ranking_rule",
            "candidates",
            "selected_round",
            "selected_source_sha256",
            "copied_best_sha256",
            "result_digest",
        }
        if set(document) != expected_keys:
            raise ValueError("selection manifest keys are not exact")
        if _canonical_json(document) != data:
            raise ValueError("selection manifest bytes are not canonical")
        if document["schema_version"] != SELECTION_SCHEMA_VERSION:
            raise ValueError("selection manifest schema is incompatible")
        if document["state_schema_digest"] != FORMAL_STATE_SCHEMA_DIGEST:
            raise ValueError("state schema digest differs")
        if document["action_catalog_digest"] != ACTION_CATALOG_DIGEST:
            raise ValueError("action catalog digest differs")
        try:
            require_v2_semantics(document["control_semantics"])
        except Exception as exc:
            raise ValueError("control semantics differ") from exc
        if document["failure_policy"] != asdict(FORMAL_FAILURE_POLICY):
            raise ValueError("failure policy differs")
        if document["ranking_rule"] != list(RANKING_RULE):
            raise ValueError("ranking rule differs")
        hashes = _pairs_from_mapping(
            document["input_manifest_sha256"],
            keys=_INPUT_HASH_KEYS,
            name="input_manifest_sha256",
            hashes=True,
        )
        raw_candidates = document["candidates"]
        if type(raw_candidates) is not list or not raw_candidates:
            raise ValueError("candidates must be a nonempty JSON array")
        candidate_keys = {item.name for item in fields(ValidationCandidate)}
        candidates: list[ValidationCandidate] = []
        for item in raw_candidates:
            if type(item) is not dict or set(item) != candidate_keys:
                raise ValueError("candidate fields are not exact")
            candidates.append(ValidationCandidate(**item))
        result = cls(
            input_manifest_hashes=hashes,
            candidates=tuple(candidates),
            selected_round=document["selected_round"],
            selected_source_sha256=document["selected_source_sha256"],
            copied_best_sha256=document["copied_best_sha256"],
            result_digest=document["result_digest"],
        )
        if result.to_json_bytes() != data:
            raise ValueError("selection manifest representation is noncanonical")
        return result


def make_selection_manifest(
    input_manifest_sha256: dict[str, str],
    candidates: tuple[ValidationCandidate, ...] | list[ValidationCandidate],
    *,
    required_rounds: Iterable[int] = FORMAL_SELECTION_ROUNDS,
) -> SelectionManifest:
    """Build and seal a manifest from caller-supplied current manifest hashes."""

    hashes = _pairs_from_mapping(
        input_manifest_sha256,
        keys=_INPUT_HASH_KEYS,
        name="input_manifest_sha256",
        hashes=True,
    )
    values = _validated_candidates(candidates, required_rounds)
    ordered = tuple(sorted(values, key=lambda value: value.round_index))
    selected = select_best_candidate(ordered, required_rounds=required_rounds)
    body = _manifest_body(
        input_manifest_hashes=hashes,
        candidates=ordered,
        selected_round=selected.round_index,
        selected_source_sha256=selected.checkpoint_sha256,
        copied_best_sha256=selected.checkpoint_sha256,
    )
    return SelectionManifest(
        input_manifest_hashes=hashes,
        candidates=ordered,
        selected_round=selected.round_index,
        selected_source_sha256=selected.checkpoint_sha256,
        copied_best_sha256=selected.checkpoint_sha256,
        result_digest=canonical_result_digest(body),
    )


_CSV_FIELDS = tuple(item.name for item in fields(ValidationCandidate))


def _metrics_csv(candidates: tuple[ValidationCandidate, ...]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(candidate.to_document())
    return stream.getvalue().encode("utf-8")


def write_selection_outputs(
    destination: Path,
    manifest: SelectionManifest,
    selected_source: Path,
) -> None:
    """Atomically write the three-file sealed Validation selection bundle."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if type(manifest) is not SelectionManifest:
        raise TypeError("manifest must be an exact SelectionManifest")
    SelectionManifest.from_json_bytes(manifest.to_json_bytes())
    source = Path(selected_source)
    expected_name = f"round_{manifest.selected_round:03d}.pt"
    if not source.is_file() or source.name != expected_name:
        raise ValueError("selected source path does not match the selected round")
    if _sha256(source) != manifest.selected_source_sha256:
        raise ValueError("selected source checkpoint hash differs from manifest")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        (temporary / "validation_checkpoint_metrics.csv").write_bytes(
            _metrics_csv(manifest.candidates)
        )
        (temporary / "selection_manifest.json").write_bytes(manifest.to_json_bytes())
        copied = temporary / "best_validation.pt"
        shutil.copyfile(source, copied)
        if _sha256(copied) != manifest.copied_best_sha256:
            raise ValueError("copied best checkpoint hash differs from manifest")
        load_selection_outputs(temporary)
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        temporary.rename(target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_selection_outputs(directory: Path) -> SelectionManifest:
    """Strictly authenticate a previously written three-file selection bundle."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError("selection output directory does not exist")
    if {path.name for path in root.iterdir()} != _OUTPUT_FILENAMES:
        raise ValueError("selection output files are not exact")
    manifest = SelectionManifest.from_json_bytes(
        (root / "selection_manifest.json").read_bytes()
    )
    expected_csv = _metrics_csv(manifest.candidates)
    if (root / "validation_checkpoint_metrics.csv").read_bytes() != expected_csv:
        raise ValueError("Validation metrics CSV differs from the manifest")
    if _sha256(root / "best_validation.pt") != manifest.copied_best_sha256:
        raise ValueError("best Validation checkpoint differs from the manifest")
    return manifest


__all__ = [
    "AuthenticatedCheckpoint",
    "FORMAL_SELECTION_ROUNDS",
    "RANKING_RULE",
    "SELECTION_SCHEMA_VERSION",
    "SelectionManifest",
    "ValidationCandidate",
    "authenticate_checkpoint_candidates",
    "canonical_result_digest",
    "load_selection_outputs",
    "make_selection_manifest",
    "select_best_candidate",
    "selection_key",
    "write_selection_outputs",
]
