from types import SimpleNamespace

import pytest

from v3.closed_loop_screen import catalog_from_document, select_train_voyage


def synthetic_episode(split="train"):
    return SimpleNamespace(
        sample_id="sample_a", split=split,
        operating_mode=("shore_pending", "shore_charging", "onboard", "onboard", "shore_pending", "shore_charging", "onboard"),
        load_kw=(0.0, 0.0, 100.0, 120.0, 0.0, 0.0, 80.0),
        battery_bus_kw=(-10.0, -20.0, 0.0, 0.0, -30.0, -40.0, 0.0),
    )


def test_select_train_voyage_keeps_complete_onboard_and_following_shore_block():
    selected = select_train_voyage((synthetic_episode(),), "sample_a", 2)
    assert selected.source_sample_id == "sample_a"
    assert selected.start_index == 2
    assert selected.end_index_exclusive == 6
    assert selected.episode.operating_mode == (
        "onboard", "onboard", "shore_pending", "shore_charging",
    )
    assert selected.episode.battery_bus_kw == (0.0, 0.0, -30.0, -40.0)
    with pytest.raises(ValueError, match="start"):
        select_train_voyage((synthetic_episode(),), "sample_a", 3)
    with pytest.raises(PermissionError, match="Train"):
        select_train_voyage((synthetic_episode("validation"),), "sample_a", 2)


def test_catalog_requires_train_provenance_matching_hashes_and_unique_sixteen_actions():
    hashes = {"power": "p", "ais": "a", "modes": "m"}
    actions = [
        {"action_id": index, "lambda_ref": float(index // 4), "lambda_soc": float(index % 4)}
        for index in range(16)
    ]
    payload = {
        "schema_version": "v3_persistence_action_calibration_v1",
        "split_used": "train", "test_payloads_opened": 0,
        "manifest_sha256": hashes, "nominal_cost_cny": 8.9, "actions": actions,
    }
    catalog = catalog_from_document(payload, expected_manifest_sha256=hashes)
    assert catalog.nominal_cost_cny == 8.9
    assert len(catalog.actions) == 16
    assert catalog.actions[0].lambda_ref == 0.0
    with pytest.raises(PermissionError, match="manifest"):
        catalog_from_document(payload, expected_manifest_sha256={**hashes, "power": "changed"})
    with pytest.raises(PermissionError, match="Train"):
        catalog_from_document({**payload, "split_used": "test"}, expected_manifest_sha256=hashes)
    with pytest.raises(ValueError, match="unique"):
        catalog_from_document({**payload, "actions": actions[:-1] + [actions[0]]}, expected_manifest_sha256=hashes)
