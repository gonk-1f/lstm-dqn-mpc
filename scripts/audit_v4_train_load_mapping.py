"""Reconcile frozen Train source loads with archived round-40 V4 transitions.

This audit opens only Train payloads. It does not execute a policy or read Test.
Run from the repository root with PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import pandas as pd

from v2.data.formal_training_dataset import FormalTrainingDataset
from v2.data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from v3.control import AccountState, EconomicMPC
from v4.control import feasible_fc_actions
from v4.review import _manifest_hashes
from v4.train import _default_data_root


ARCHIVE = Path(
    "docs/results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/"
    "raw/round_040_train_trajectories.json.gz"
)


def _source_location(power_root: Path, manifest: pd.DataFrame, sample_id: str, time_s: float) -> dict:
    match = manifest.loc[manifest.sample_id.eq(sample_id) & manifest.split.eq("train")]
    assert len(match) == 1
    relative = str(match.iloc[0].relative_path)
    path = power_root / relative
    power = pd.read_csv(path, encoding="utf-8-sig")
    rows = power.index[power.time_s.eq(time_s)].tolist()
    assert len(rows) == 1
    return {"power_csv": str(path), "power_data_row_0_based": rows[0],
            "power_file_line_1_based": rows[0] + 2}


def audit(archive_path: Path) -> dict:
    roots = tuple(_default_data_root(name) for name in (
        "operating_dataset_zero_boundary_v2", "operating_dataset_zero_boundary_v2_ais",
        "operating_dataset_zero_boundary_v2_modes"))
    hashes_before = _manifest_hashes(roots)
    archived_run = json.loads((archive_path.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert archived_run["manifest_sha256"] == hashes_before
    dataset = FormalTrainingDataset.open(*roots)
    episodes = {episode.sample_id: episode for episode in dataset.load_train()}
    assert len(episodes) == 30 and dataset.opened_test_payloads == 0
    with gzip.open(archive_path, "rt", encoding="utf-8") as handle:
        archive = json.load(handle)
    profiles = {row["sample_id"]: row for group in ("completed", "failed") for row in archive[group]}
    assert profiles.keys() == episodes.keys()
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    source = Counter()
    executed = Counter()
    examples = []
    manifest = pd.read_csv(roots[0] / "metadata/sample_manifest.csv", encoding="utf-8-sig")
    for sample_id, episode in episodes.items():
        onboard_rows = [i for i, mode in enumerate(episode.operating_mode)
                        if OperatingMode(mode) is OperatingMode.ONBOARD]
        for i in onboard_rows:
            load = float(episode.load_kw[i])
            source["onboard"] += 1
            source["raw_exact_zero"] += load == 0
            source["raw_positive_below_10"] += 0 < load < 10
            source["raw_negative"] += load < 0
            source["effective_zero"] += normalize_onboard_load_kw(load) == 0
        transitions = profiles[sample_id]["transitions"]
        assert len(transitions) <= len(onboard_rows)
        voyage = 0
        for j, transition in enumerate(transitions):
            source_row = onboard_rows[j]
            raw = float(episode.load_kw[source_row])
            effective = normalize_onboard_load_kw(raw)
            assert abs(effective - float(transition["load_kw"])) <= 1e-9, (sample_id, j)
            assert transition["voyage_index"] == voyage
            voyage += bool(transition["done"])
            executed["onboard"] += 1
            executed["raw_exact_zero"] += raw == 0
            executed["raw_positive_below_10"] += 0 < raw < 10
            executed["raw_negative"] += raw < 0
            executed["effective_zero"] += effective == 0
            if transition["fc_kw"] > 0 and (
                j == 0 or transitions[j-1]["voyage_index"] != transition["voyage_index"]
                or transitions[j-1]["fc_kw"] == 0
            ):
                executed["starts_total"] += 1
                if j == 0 or transitions[j-1]["voyage_index"] != transition["voyage_index"]:
                    executed["starts_first_in_voyage"] += 1
                    executed["starts_at_departure_first_action"] += 1
                elif not any(t["fc_kw"] > 0 for t in transitions[:j]
                             if t["voyage_index"] == transition["voyage_index"]):
                    executed["starts_first_in_voyage"] += 1
                    executed["starts_delayed_in_voyage"] += 1
                else:
                    executed["starts_internal_restart"] += 1
            if transition["soc_before"] >= 0.79 and transition["fc_kw"] == 0:
                previous = float(transition["soc_before"])
                mask = feasible_fc_actions(AccountState(soc=previous), raw, accountant)
                assert 0 in mask
                key = "high_soc_fc_zero_positive_feasible" if any(action > 0 for action in mask) else "high_soc_fc_zero_only_zero"
                executed[key] += 1
                if key.endswith("only_zero"):
                    executed["only_zero_raw_negative"] += raw < 0
                    executed["only_zero_raw_exact_zero"] += raw == 0
                    executed["only_zero_raw_positive_below_10"] += 0 < raw < 10
            if sample_id == "zero_boundary_029" and j == 355:
                examples.append({"sample_id": sample_id, "archive_transition_0_based": j,
                                 "supervisory_row_0_based": source_row,
                                 "time_s": float(episode.time_s[source_row]),
                                 "source_load_kw": raw, "effective_load_kw": effective,
                                 **_source_location(roots[0], manifest, sample_id,
                                                    float(episode.time_s[source_row]))})
    assert dataset.opened_test_payloads == 0
    assert hashes_before == _manifest_hashes(roots)
    result = {"manifest_sha256": hashes_before, "source": dict(source),
              "executed_round40": dict(executed), "examples": examples,
              "test_payloads_opened": dataset.opened_test_payloads}
    assert source["onboard"] == 18448
    assert source["raw_exact_zero"] == 30
    assert source["raw_positive_below_10"] == 457
    assert executed["high_soc_fc_zero_only_zero"] == 2372
    assert executed["high_soc_fc_zero_positive_feasible"] == 235
    assert executed["starts_total"] == 326
    assert executed["starts_first_in_voyage"] == 62
    assert executed["starts_at_departure_first_action"] == 54
    assert executed["starts_delayed_in_voyage"] == 8
    assert executed["starts_internal_restart"] == 264
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    args = parser.parse_args()
    print(json.dumps(audit(args.archive), ensure_ascii=False, indent=2))
