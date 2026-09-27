"""Authenticated Train-only reward-scale documents."""

from __future__ import annotations

from ..analysis.action_screening import (
    DataSplit,
    DatasetProvenance,
    HeldOutSelectionError,
)
from ..economics import (
    REWARD_SCALE_DERIVATION_RULE,
    RewardScaleCalibration,
    _validate_reward_scale,
    calibrate_reward_scale,
)
from ..evaluation.checkpoint_selection import canonical_result_digest


REWARD_SCALE_DOCUMENT_VERSION = "v2_train_interval_reward_scale_v1"
REFERENCE_ACTION_ID = "w_8_1_1"
_MANIFEST_KEYS = ("power", "ais", "modes")


def _manifest_hashes(value: object) -> dict[str, str]:
    if type(value) is not dict:
        raise TypeError("manifest_hashes must be an exact dict")
    if set(value) != set(_MANIFEST_KEYS):
        raise ValueError("manifest_hashes must contain power, ais, and modes")
    checked: dict[str, str] = {}
    for key in _MANIFEST_KEYS:
        item = value[key]
        if (
            type(item) is not str
            or len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"{key} manifest hash must be lowercase SHA-256")
        checked[key] = item
    return checked


def _segment_ids(value: object) -> tuple[str, ...]:
    if type(value) not in (tuple, list) or not value:
        raise ValueError("train_segment_ids must be a nonempty tuple/list")
    checked = tuple(value)
    if any(type(item) is not str or not item for item in checked):
        raise ValueError("train_segment_ids must contain nonempty exact strings")
    if len(set(checked)) != len(checked):
        raise ValueError("train_segment_ids must be unique")
    return checked


def build_reward_scale_document(
    *,
    calibration: RewardScaleCalibration,
    reference_action_id: str,
    manifest_hashes: dict[str, str],
    train_segment_ids: tuple[str, ...],
    test_payloads_opened: int,
) -> dict[str, object]:
    checked = _validate_reward_scale(calibration)
    if reference_action_id != REFERENCE_ACTION_ID:
        raise ValueError(f"reference_action_id must be {REFERENCE_ACTION_ID}")
    hashes = _manifest_hashes(manifest_hashes)
    segment_ids = _segment_ids(train_segment_ids)
    if type(test_payloads_opened) is not int or test_payloads_opened != 0:
        raise HeldOutSelectionError("reward calibration must open zero Test payloads")
    body: dict[str, object] = {
        "schema_version": REWARD_SCALE_DOCUMENT_VERSION,
        "dataset_version": checked.provenance.dataset_version,
        "provenance_id": checked.provenance.provenance_id,
        "split": "train",
        "reference_action_id": reference_action_id,
        "input_manifest_sha256": hashes,
        "train_segment_ids": list(segment_ids),
        "train_raw_interval_costs_cny": list(checked.train_raw_costs_cny),
        "sample_count": checked.sample_count,
        "scale_cny": checked.scale_cny,
        "derivation_rule": checked.derivation_rule,
        "audit_id": checked.audit_id,
        "reason": checked.reason,
        "calibration_digest": checked.digest,
        "test_payloads_opened": 0,
    }
    body["result_digest"] = canonical_result_digest(body)
    return body


def load_reward_scale_document(
    document: object,
    *,
    expected_manifest_hashes: dict[str, str] | None = None,
) -> RewardScaleCalibration:
    if type(document) is not dict:
        raise TypeError("reward-scale document must be an exact dict")
    if document.get("split") != "train":
        raise HeldOutSelectionError("reward-scale document must remain Train-only")
    if document.get("test_payloads_opened") != 0:
        raise HeldOutSelectionError("reward-scale document opened Test payloads")
    try:
        stored_digest = document["result_digest"]
        body = {key: value for key, value in document.items() if key != "result_digest"}
        if type(stored_digest) is not str or stored_digest != canonical_result_digest(body):
            raise ValueError("reward-scale document digest differs")
        if document["schema_version"] != REWARD_SCALE_DOCUMENT_VERSION:
            raise ValueError("reward-scale document schema differs")
        if document["reference_action_id"] != REFERENCE_ACTION_ID:
            raise ValueError("reward-scale reference action differs")
        hashes = _manifest_hashes(document["input_manifest_sha256"])
        if expected_manifest_hashes is not None:
            expected = _manifest_hashes(expected_manifest_hashes)
            if hashes != expected:
                raise ValueError("reward-scale input manifest hashes differ")
        _segment_ids(document["train_segment_ids"])
        costs_value = document["train_raw_interval_costs_cny"]
        if type(costs_value) is not list or not costs_value:
            raise ValueError("reward-scale Train costs must be a nonempty list")
        if any(type(value) is not float for value in costs_value):
            raise TypeError("reward-scale Train costs must contain exact floats")
        provenance = DatasetProvenance(
            document["dataset_version"],
            document["provenance_id"],
            DataSplit.TRAIN,
        )
        calibration = calibrate_reward_scale(
            tuple(costs_value),
            provenance=provenance,
            audit_id=document["audit_id"],
            reason=document["reason"],
        )
        expected_fields = {
            "sample_count": calibration.sample_count,
            "scale_cny": calibration.scale_cny,
            "derivation_rule": REWARD_SCALE_DERIVATION_RULE,
            "calibration_digest": calibration.digest,
        }
        if any(document[key] != value for key, value in expected_fields.items()):
            raise ValueError("reward-scale derived fields differ")
    except KeyError as exc:
        raise ValueError(f"reward-scale document is missing {exc.args[0]}") from exc
    return calibration


__all__ = [
    "REFERENCE_ACTION_ID",
    "REWARD_SCALE_DOCUMENT_VERSION",
    "build_reward_scale_document",
    "load_reward_scale_document",
]
