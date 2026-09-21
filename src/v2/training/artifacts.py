"""Canonical, fail-closed checkpoint and replay envelopes for v2.

Payloads are opaque bytes.  This module deliberately does not use pickle or
``torch.load``; metadata is fully validated before payload bytes are decoded or
passed to a consumer.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, TypeVar

from ..config import TimeScaleConfig
from ..contracts import (
    IncompatibleArtifactError,
    control_semantics,
    require_v2_semantics,
)
from ..dqn.action_space import ActionCandidate
from ..economics import RawCnyIntervalLedger
from ..envs.multirate_weight_env import MacroTransition


ARTIFACT_SCHEMA_VERSION = "v2_training_artifact_v1"
_ARTIFACT_KINDS = ("checkpoint", "replay")
_REWARD_UNITS = "raw_CNY"
_HEX_256 = re.compile(r"[0-9a-f]{64}")
_ACTION_ID = re.compile(r"w_([1-9])_([1-9])_([1-9])")
_T = TypeVar("_T")


def _exact_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, OverflowError) as exc:
        raise IncompatibleArtifactError("artifact is not canonical JSON data") from exc


def _catalog_digest(identity: tuple[str, ...]) -> str:
    return hashlib.sha256(
        _canonical_json({"action_ids": list(identity)})
    ).hexdigest()


def _action_identity_key(value: str) -> tuple[int, int, int]:
    match = _ACTION_ID.fullmatch(value)
    if match is None:
        raise ValueError("action catalog contains a non-canonical v2 action ID")
    numerators = tuple(int(item) for item in match.groups())
    if sum(numerators) != 10:
        raise ValueError("action catalog identity does not sum to ten")
    return numerators


def _validate_action(value: object) -> ActionCandidate:
    if type(value) is not ActionCandidate:
        raise TypeError("action catalog must contain exact ActionCandidate values")
    numerators = (value.n_base, value.n_smooth, value.n_soc)
    if any(type(item) is not int for item in numerators):
        raise TypeError("stored action numerators must remain exact integers")
    if any(item < 1 for item in numerators) or sum(numerators) != 10:
        raise ValueError("stored action numerators are invalid")
    canonical = ActionCandidate(*numerators)
    if vars(value) != vars(canonical):
        raise ValueError("stored action contains injected or altered attributes")
    if value.action_id != canonical.action_id:
        raise ValueError("stored action identity is not canonical")
    return canonical


def _rng_state_to_json_data(value: object) -> object:
    """Normalize a restored NumPy bit-generator state for exact comparison."""
    if type(value) is dict:
        return {key: _rng_state_to_json_data(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_rng_state_to_json_data(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _rng_state_to_json_data(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _rng_state_to_json_data(item())
    return value


def _validate_rng_state(algorithm: str, state_value: dict[str, object]) -> None:
    import numpy as np

    if state_value.get("bit_generator") != algorithm:
        raise ValueError("rng_state_json does not identify rng_algorithm")
    bit_generator_type = getattr(np.random, algorithm, None)
    if (
        type(bit_generator_type) is not type
        or not issubclass(bit_generator_type, np.random.BitGenerator)
    ):
        raise ValueError("rng_algorithm is not a supported NumPy bit generator")
    try:
        restored = bit_generator_type()
        restored.state = state_value
        restored_state = _rng_state_to_json_data(restored.state)
    except Exception as exc:
        raise ValueError("rng_state_json is not recoverable by rng_algorithm") from exc
    if _canonical_json(restored_state) != _canonical_json(state_value):
        raise ValueError("rng_state_json changes during RNG state recovery")


@dataclass(frozen=True)
class ArtifactMetadata:
    """Immutable metadata sufficient to reject incompatible resume state."""

    schema_version: str
    kind: str
    control_semantics_items: tuple[tuple[str, object], ...]
    action_catalog_identity: tuple[str, ...]
    action_catalog_digest: str
    state_schema_version: str
    state_dimension: int
    reward_units: str
    reward_scaling_identity: str
    episode_index: int
    macro_index: int
    seed: int
    rng_algorithm: str
    rng_state_json: str
    rng_state_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION or type(self.schema_version) is not str:
            raise ValueError("artifact schema version is incompatible")
        if type(self.kind) is not str or self.kind not in _ARTIFACT_KINDS:
            raise ValueError("artifact kind must be checkpoint or replay")
        if type(self.control_semantics_items) is not tuple or any(
            type(item) is not tuple or len(item) != 2 or type(item[0]) is not str
            for item in self.control_semantics_items
        ):
            raise TypeError("control semantics must be immutable key/value tuples")
        semantics = dict(self.control_semantics_items)
        if len(semantics) != len(self.control_semantics_items):
            raise ValueError("control semantics contain duplicate keys")
        require_v2_semantics(semantics, self.timescale)
        if tuple(sorted(semantics.items())) != self.control_semantics_items:
            raise ValueError("control semantics key order is not canonical")
        if type(self.action_catalog_identity) is not tuple or not self.action_catalog_identity:
            raise TypeError("action catalog identity must be a non-empty exact tuple")
        if any(type(item) is not str or not item for item in self.action_catalog_identity):
            raise TypeError("action catalog IDs must be non-empty exact strings")
        if len(set(self.action_catalog_identity)) != len(self.action_catalog_identity):
            raise ValueError("action catalog identity contains duplicates")
        identity_keys = tuple(
            _action_identity_key(item) for item in self.action_catalog_identity
        )
        if identity_keys != tuple(sorted(identity_keys)):
            raise ValueError("action catalog identity order is not canonical")
        if type(self.action_catalog_digest) is not str or self.action_catalog_digest != _catalog_digest(
            self.action_catalog_identity
        ):
            raise ValueError("action catalog digest is invalid")
        _exact_text(self.state_schema_version, "state_schema_version")
        if type(self.state_dimension) is not int or self.state_dimension <= 0:
            raise TypeError("state_dimension must be an exact positive integer")
        if type(self.reward_units) is not str or self.reward_units != _REWARD_UNITS:
            raise ValueError("reward units must be raw_CNY")
        _exact_text(self.reward_scaling_identity, "reward_scaling_identity")
        for name in ("episode_index", "macro_index", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise TypeError(f"{name} must be an exact non-negative integer")
        _exact_text(self.rng_algorithm, "rng_algorithm")
        state_json = _exact_text(self.rng_state_json, "rng_state_json")
        try:
            state_value = json.loads(
                state_json, object_pairs_hook=_reject_duplicate_keys
            )
            state_bytes = state_json.encode("ascii")
        except (UnicodeEncodeError, json.JSONDecodeError, IncompatibleArtifactError) as exc:
            raise ValueError("rng_state_json must be canonical ASCII JSON") from exc
        if type(state_value) is not dict or not state_value:
            raise ValueError("rng_state_json must encode a non-empty JSON object")
        if _canonical_json(state_value) != state_bytes:
            raise ValueError("rng_state_json must use canonical JSON serialization")
        _validate_rng_state(self.rng_algorithm, state_value)
        expected_rng_digest = hashlib.sha256(state_bytes).hexdigest()
        if (
            type(self.rng_state_digest) is not str
            or _HEX_256.fullmatch(self.rng_state_digest) is None
            or self.rng_state_digest != expected_rng_digest
        ):
            raise ValueError("rng_state_digest must be a lowercase SHA-256 digest")

    @property
    def timescale(self) -> TimeScaleConfig:
        values = dict(self.control_semantics_items)
        return TimeScaleConfig(
            values["ts_mpc_seconds"],
            values["n_mpc"],
            values["dqn_switch_steps"],
        )

    @property
    def n_mpc(self) -> int:
        return self.timescale.n_mpc

    @property
    def dqn_switch_steps(self) -> int:
        return self.timescale.dqn_switch_steps


@dataclass(frozen=True)
class ArtifactEnvelope:
    metadata: ArtifactMetadata
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.metadata) is not ArtifactMetadata:
            raise TypeError("metadata must be an exact ArtifactMetadata")
        if type(self.payload) is not bytes:
            raise TypeError("payload must be exact bytes")


def make_artifact_metadata(
    *,
    kind: str,
    timescale: TimeScaleConfig,
    action_catalog: tuple[ActionCandidate, ...],
    state_schema_version: str,
    state_dimension: int,
    reward_scaling_identity: str,
    episode_index: int,
    macro_index: int,
    seed: int,
    rng_algorithm: str,
    rng_state_json: str,
) -> ArtifactMetadata:
    if type(timescale) is not TimeScaleConfig:
        raise TypeError("timescale must be an exact TimeScaleConfig")
    if type(action_catalog) is not tuple or not action_catalog:
        raise TypeError("action_catalog must be a non-empty exact tuple")
    actions = tuple(_validate_action(action) for action in action_catalog)
    identity = tuple(action.action_id for action in actions)
    if type(rng_state_json) is not str:
        raise TypeError("rng_state_json must be an exact str")
    try:
        rng_state_digest = hashlib.sha256(rng_state_json.encode("ascii")).hexdigest()
    except UnicodeEncodeError as exc:
        raise ValueError("rng_state_json must be canonical ASCII JSON") from exc
    return ArtifactMetadata(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        kind=kind,
        control_semantics_items=tuple(sorted(control_semantics(timescale).items())),
        action_catalog_identity=identity,
        action_catalog_digest=_catalog_digest(identity),
        state_schema_version=state_schema_version,
        state_dimension=state_dimension,
        reward_units=_REWARD_UNITS,
        reward_scaling_identity=reward_scaling_identity,
        episode_index=episode_index,
        macro_index=macro_index,
        seed=seed,
        rng_algorithm=rng_algorithm,
        rng_state_json=rng_state_json,
        rng_state_digest=rng_state_digest,
    )


_METADATA_KEYS = {
    "schema_version",
    "kind",
    "control_semantics",
    "action_catalog_identity",
    "action_catalog_digest",
    "state_schema_version",
    "state_dimension",
    "reward_units",
    "reward_scaling_identity",
    "episode_index",
    "macro_index",
    "seed",
    "rng_algorithm",
    "rng_state_json",
    "rng_state_digest",
}


def _metadata_document(metadata: ArtifactMetadata) -> dict[str, object]:
    if type(metadata) is not ArtifactMetadata:
        raise TypeError("metadata must be an exact ArtifactMetadata")
    # Reconstructing also catches post-construction object.__setattr__ tampering.
    ArtifactMetadata(**metadata.__dict__)
    return {
        "schema_version": metadata.schema_version,
        "kind": metadata.kind,
        "control_semantics": dict(metadata.control_semantics_items),
        "action_catalog_identity": list(metadata.action_catalog_identity),
        "action_catalog_digest": metadata.action_catalog_digest,
        "state_schema_version": metadata.state_schema_version,
        "state_dimension": metadata.state_dimension,
        "reward_units": metadata.reward_units,
        "reward_scaling_identity": metadata.reward_scaling_identity,
        "episode_index": metadata.episode_index,
        "macro_index": metadata.macro_index,
        "seed": metadata.seed,
        "rng_algorithm": metadata.rng_algorithm,
        "rng_state_json": metadata.rng_state_json,
        "rng_state_digest": metadata.rng_state_digest,
    }


def _metadata_from_document(value: object) -> ArtifactMetadata:
    if type(value) is not dict or set(value) != _METADATA_KEYS:
        raise IncompatibleArtifactError("artifact metadata keys are not exact v2 keys")
    semantics = value["control_semantics"]
    identity = value["action_catalog_identity"]
    if type(semantics) is not dict:
        raise IncompatibleArtifactError("control semantics must be a JSON object")
    if type(identity) is not list:
        raise IncompatibleArtifactError("action catalog identity must be a JSON array")
    try:
        return ArtifactMetadata(
            schema_version=value["schema_version"],
            kind=value["kind"],
            control_semantics_items=tuple(semantics.items()),
            action_catalog_identity=tuple(identity),
            action_catalog_digest=value["action_catalog_digest"],
            state_schema_version=value["state_schema_version"],
            state_dimension=value["state_dimension"],
            reward_units=value["reward_units"],
            reward_scaling_identity=value["reward_scaling_identity"],
            episode_index=value["episode_index"],
            macro_index=value["macro_index"],
            seed=value["seed"],
            rng_algorithm=value["rng_algorithm"],
            rng_state_json=value["rng_state_json"],
            rng_state_digest=value["rng_state_digest"],
        )
    except (KeyError, TypeError, ValueError, IncompatibleArtifactError) as exc:
        raise IncompatibleArtifactError("artifact metadata is incompatible with v2") from exc


def _envelope_digest(metadata_document: dict[str, object], payload_base64: str) -> str:
    body = {
        "metadata": metadata_document,
        "payload_base64": payload_base64,
        "payload_encoding": "base64",
    }
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def encode_artifact(metadata: ArtifactMetadata, payload: bytes) -> bytes:
    if type(payload) is not bytes:
        raise TypeError("artifact payload must be exact bytes")
    metadata_document = _metadata_document(metadata)
    payload_base64 = base64.b64encode(payload).decode("ascii")
    document = {
        "metadata": metadata_document,
        "payload_base64": payload_base64,
        "payload_encoding": "base64",
        "sha256": _envelope_digest(metadata_document, payload_base64),
    }
    return _canonical_json(document)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IncompatibleArtifactError("artifact JSON contains duplicate keys")
        result[key] = value
    return result


def decode_artifact(
    data: bytes,
    *,
    expected_metadata: ArtifactMetadata,
    payload_consumer: Callable[[bytes], _T] | None = None,
) -> ArtifactEnvelope | _T:
    if type(data) is not bytes:
        raise IncompatibleArtifactError("artifact input must be exact bytes")
    if type(expected_metadata) is not ArtifactMetadata:
        raise TypeError("expected_metadata must be an exact ArtifactMetadata")
    _metadata_document(expected_metadata)
    if payload_consumer is not None and not callable(payload_consumer):
        raise TypeError("payload_consumer must be callable or None")
    try:
        document = json.loads(data.decode("ascii"), object_pairs_hook=_reject_duplicate_keys)
    except IncompatibleArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompatibleArtifactError("artifact is not a v2 canonical JSON envelope") from exc
    top_keys = {"metadata", "payload_base64", "payload_encoding", "sha256"}
    if type(document) is not dict or set(document) != top_keys:
        raise IncompatibleArtifactError("artifact envelope keys are not exact v2 keys")

    # Validate and compare every metadata field before decoding payload bytes.
    metadata = _metadata_from_document(document["metadata"])
    if metadata != expected_metadata:
        raise IncompatibleArtifactError("artifact metadata does not exactly match expected v2 metadata")
    if type(document["payload_encoding"]) is not str or document["payload_encoding"] != "base64":
        raise IncompatibleArtifactError("artifact payload encoding is incompatible")
    payload_base64 = document["payload_base64"]
    digest = document["sha256"]
    if type(payload_base64) is not str or type(digest) is not str:
        raise IncompatibleArtifactError("artifact payload or digest has an invalid type")
    metadata_document = _metadata_document(metadata)
    if digest != _envelope_digest(metadata_document, payload_base64):
        raise IncompatibleArtifactError("artifact digest does not match its contents")
    if _canonical_json(document) != data:
        raise IncompatibleArtifactError("artifact bytes are not in canonical form")
    try:
        payload = base64.b64decode(payload_base64, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise IncompatibleArtifactError("artifact payload is not canonical base64") from exc
    envelope = ArtifactEnvelope(metadata, payload)
    if encode_artifact(metadata, payload) != data:
        raise IncompatibleArtifactError("artifact bytes are not in canonical form")
    if payload_consumer is not None:
        return payload_consumer(payload)
    return envelope


def encode_checkpoint(metadata: ArtifactMetadata, payload: bytes) -> bytes:
    if type(metadata) is not ArtifactMetadata or metadata.kind != "checkpoint":
        raise TypeError("checkpoint encoding requires exact checkpoint metadata")
    return encode_artifact(metadata, payload)


def decode_checkpoint(
    data: bytes,
    *,
    expected_metadata: ArtifactMetadata,
    payload_consumer: Callable[[bytes], _T] | None = None,
) -> ArtifactEnvelope | _T:
    if type(expected_metadata) is not ArtifactMetadata or expected_metadata.kind != "checkpoint":
        raise TypeError("checkpoint decoding requires exact checkpoint metadata")
    return decode_artifact(
        data,
        expected_metadata=expected_metadata,
        payload_consumer=payload_consumer,
    )


def _transition_document(value: MacroTransition) -> dict[str, object]:
    if type(value) is not MacroTransition:
        raise TypeError("replay must contain exact MacroTransition values")
    # Reconstruct to detect post-construction mutation before serialization.
    MacroTransition(**value.__dict__)
    return {
        "state": list(value.state),
        "action_numerators": list(value.action.numerators),
        "reward_cny": value.reward_cny,
        "next_state": list(value.next_state),
        "done": value.done,
        "executed_mpc_steps": value.executed_mpc_steps,
        "ledger_components_cny": list(value.ledger.components_cny),
    }


_TRANSITION_KEYS = {
    "state",
    "action_numerators",
    "reward_cny",
    "next_state",
    "done",
    "executed_mpc_steps",
    "ledger_components_cny",
}


def _transition_from_document(value: object) -> MacroTransition:
    if type(value) is not dict or set(value) != _TRANSITION_KEYS:
        raise IncompatibleArtifactError("replay transition keys are not exact v2 keys")
    for key, expected_length in (
        ("action_numerators", 3),
        ("ledger_components_cny", 4),
    ):
        if type(value[key]) is not list or len(value[key]) != expected_length:
            raise IncompatibleArtifactError(f"{key} is malformed")
    if type(value["state"]) is not list or type(value["next_state"]) is not list:
        raise IncompatibleArtifactError("replay states must be JSON arrays")
    try:
        action = ActionCandidate(*value["action_numerators"])
        ledger = RawCnyIntervalLedger(*value["ledger_components_cny"])
        return MacroTransition(
            state=tuple(value["state"]),
            action=action,
            reward_cny=value["reward_cny"],
            next_state=tuple(value["next_state"]),
            done=value["done"],
            executed_mpc_steps=value["executed_mpc_steps"],
            ledger=ledger,
        )
    except (TypeError, ValueError) as exc:
        raise IncompatibleArtifactError("replay transition is incompatible with v2") from exc


def encode_replay(
    metadata: ArtifactMetadata,
    transitions: tuple[MacroTransition, ...],
) -> bytes:
    _validate_replay_transitions(metadata, transitions)
    payload = _canonical_json([_transition_document(item) for item in transitions])
    return encode_artifact(metadata, payload)


def _validate_replay_transitions(
    metadata: ArtifactMetadata,
    transitions: tuple[MacroTransition, ...],
) -> tuple[MacroTransition, ...]:
    if type(metadata) is not ArtifactMetadata or metadata.kind != "replay":
        raise TypeError("replay validation requires exact replay metadata")
    _metadata_document(metadata)
    if type(transitions) is not tuple:
        raise TypeError("transitions must be an immutable exact tuple")

    for transition in transitions:
        if type(transition) is not MacroTransition:
            raise TypeError("replay must contain exact MacroTransition values")
        # Reconstruct first so forged bool/subclass/nested fields fail before
        # metadata binding checks.
        MacroTransition(**transition.__dict__)
        if (
            len(transition.state) != metadata.state_dimension
            or len(transition.next_state) != metadata.state_dimension
        ):
            raise ValueError("replay state dimensions do not match metadata")
        if transition.action.action_id not in metadata.action_catalog_identity:
            raise ValueError("replay action is outside the metadata action catalog")
        if not 1 <= transition.executed_mpc_steps <= metadata.dqn_switch_steps:
            raise ValueError("replay execution count is outside the metadata timescale")
    return transitions


def decode_replay(
    data: bytes,
    *,
    expected_metadata: ArtifactMetadata,
) -> tuple[MacroTransition, ...]:
    if type(expected_metadata) is not ArtifactMetadata or expected_metadata.kind != "replay":
        raise TypeError("replay decoding requires exact replay metadata")
    envelope = decode_artifact(data, expected_metadata=expected_metadata)
    if type(envelope) is not ArtifactEnvelope:  # pragma: no cover - static narrowing
        raise AssertionError("unreachable")
    try:
        values = json.loads(
            envelope.payload.decode("ascii"), object_pairs_hook=_reject_duplicate_keys
        )
    except IncompatibleArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IncompatibleArtifactError("replay payload is not canonical JSON") from exc
    if type(values) is not list:
        raise IncompatibleArtifactError("replay payload must be a JSON array")
    transitions = tuple(_transition_from_document(item) for item in values)
    if _canonical_json([_transition_document(item) for item in transitions]) != envelope.payload:
        raise IncompatibleArtifactError("replay payload bytes are not canonical")
    try:
        _validate_replay_transitions(envelope.metadata, transitions)
    except (TypeError, ValueError, IncompatibleArtifactError) as exc:
        raise IncompatibleArtifactError(
            "replay transitions do not match artifact metadata"
        ) from exc
    return transitions


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactEnvelope",
    "ArtifactMetadata",
    "decode_artifact",
    "decode_checkpoint",
    "decode_replay",
    "encode_artifact",
    "encode_checkpoint",
    "encode_replay",
    "make_artifact_metadata",
]
