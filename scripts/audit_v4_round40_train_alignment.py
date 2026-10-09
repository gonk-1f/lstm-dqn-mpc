"""Read-only Train vs round-40 greedy trace alignment audit.

Run from repository root:
    python scripts/audit_v4_round40_train_alignment.py --output /tmp/v4_round40_train_alignment.json

Only load_train() is called; Test payloads are never opened.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "docs/results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/round_040_train_trajectories.json.gz"
META = ARCHIVE.parent / "run_metadata.json"
ROOTS = (
    REPO / "data/processed/operating_dataset_zero_boundary_v2",
    REPO / "data/processed/operating_dataset_zero_boundary_v2_ais",
    REPO / "data/processed/operating_dataset_zero_boundary_v2_modes",
)
MANIFESTS = (
    "operating_dataset_zero_boundary_v2/sample_manifest.csv",
    "operating_dataset_zero_boundary_v2_ais/sample_manifest.csv",
    "operating_dataset_zero_boundary_v2_modes/sample_manifest.csv",
)


def close(a: float, b: float, *, atol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=atol)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON diagnostics outside the frozen archives")
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "src"))
    from v2.data.formal_training_dataset import FormalTrainingDataset
    from v2.data.supervisory_rules import normalize_onboard_load_kw
    from v3.control import AccountState, EconomicMPC
    from v4.control import feasible_fc_actions

    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Missing archive: {ARCHIVE}")
    metadata = json.loads(META.read_text(encoding="utf-8"))
    manifest_checks = {}
    for name in MANIFESTS:
        current = hashlib.sha256((REPO / "data/processed" / name).read_bytes()).hexdigest()
        original = metadata["manifest_sha256"][name]
        manifest_checks[name] = {"current": current, "run_recorded": original, "match": current == original}
        if current != original:
            raise RuntimeError(f"Manifest SHA256 mismatch: {name}")

    dataset = FormalTrainingDataset.open(*ROOTS)
    episodes = dataset.load_train()
    assert dataset.opened_test_payloads == 0
    if len(episodes) != 30:
        raise RuntimeError(f"Expected 30 Train samples, found {len(episodes)}")
    by_id = {ep.sample_id: ep for ep in episodes}
    with gzip.open(ARCHIVE, "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    profiles = tuple(archive["completed"]) + tuple(archive["failed"])
    if len(profiles) != 30 or set(by_id) != {p["sample_id"] for p in profiles}:
        raise RuntimeError("Greedy archive has missing/duplicate Train identities")
    if len({p["sample_id"] for p in profiles}) != 30:
        raise RuntimeError("Duplicate greedy sample profiles")
    by_profile = {p["sample_id"]: p for p in profiles}
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    source = Counter()
    events = Counter()
    discrepancies = []
    examples = []
    samples = []

    for ep in episodes:
        onboard_indices = [j for j, mode in enumerate(ep.operating_mode) if mode == "onboard"]
        source["onboard"] += len(onboard_indices)
        for j in onboard_indices:
            load = float(ep.load_kw[j])
            source["raw_zero"] += int(load == 0.0)
            source["raw_open_0_to_10"] += int(0.0 < load < 10.0)
            source["raw_negative"] += int(load < 0.0)
        profile = by_profile[ep.sample_id]
        transitions = profile["transitions"]
        if len(transitions) > len(onboard_indices):
            raise RuntimeError(f"{ep.sample_id}: more transitions than ONBOARD data rows")
        prior_power = {}
        started = set()
        local = Counter()
        for i, t in enumerate(transitions):
            j = onboard_indices[i]
            load = float(ep.load_kw[j])
            normalized = normalize_onboard_load_kw(load)
            archived = float(t["load_kw"])
            fc = float(t["fc_kw"])
            battery = float(t["battery_bus_kw"])
            if not close(archived, fc + battery):
                raise RuntimeError(f"{ep.sample_id}, transition {i}: load != fc + battery")
            if not close(normalized, archived):
                discrepancies.append({
                    "sample_id": ep.sample_id, "transition_index": i,
                    "mode_csv_row_zero_based": j, "mode_csv_line_one_based": j + 2,
                    "time_s": float(ep.time_s[j]), "source_load_kw": load,
                    "archived_load_kw": archived, "fc_kw": fc,
                    "battery_bus_kw": battery, "soc_before": float(t["soc_before"]),
                })
            voyage = int(t["voyage_index"])
            previous = prior_power.get(voyage, 0.0)
            if fc > 0.0 and previous == 0.0:
                events["total_starts"] += 1
                if voyage not in started:
                    events["voyage_first_starts"] += 1
                    started.add(voyage)
                else:
                    events["onboard_internal_restarts"] += 1
            prior_power[voyage] = fc
            events["executed_onboard"] += 1
            if float(t["soc_before"]) >= 0.79 and fc == 0.0:
                mask = feasible_fc_actions(AccountState(soc=float(t["soc_before"])),
                                           normalized, accountant)
                if not mask:
                    raise RuntimeError(f"{ep.sample_id}, transition {i}: action 0 executed with empty mask")
                events["high_soc_fc_zero"] += 1
                if mask == (0,):
                    events["only_zero_feasible"] += 1
                    local["only_zero_feasible"] += 1
                else:
                    events["positive_feasible_but_chose_zero"] += 1
                    local["positive_feasible_but_chose_zero"] += 1
                if load == 0.0:
                    events["high_soc_fc_zero_raw_load_zero"] += 1
                if close(archived, 0.0, atol=1e-9):
                    events["high_soc_fc_zero_archived_load_zero"] += 1
                if len(examples) < 20:
                    examples.append({
                        "sample_id": ep.sample_id, "voyage_index": voyage,
                        "transition_index": i, "mode_csv_line_one_based": j + 2,
                        "time_s": float(ep.time_s[j]), "source_load_kw": load,
                        "archived_load_kw": archived, "soc_before": float(t["soc_before"]),
                        "physical_actions_kw": list(mask), "only_zero": mask == (0,),
                    })
        samples.append({
            "sample_id": ep.sample_id, "source_onboard": len(onboard_indices),
            "executed_onboard": len(transitions), "completed": bool(profile["completed"]),
            **dict(local),
        })

    assert dataset.opened_test_payloads == 0
    claims = {
        "only_zero_feasible": 2372, "positive_feasible_but_chose_zero": 235,
        "high_soc_fc_zero_archived_load_zero": 2057, "total_starts": 326,
        "voyage_first_starts": 62, "onboard_internal_restarts": 264,
    }
    comparison = {name: {
        "observed": events.get(name, 0), "reported": reported,
        "match": events.get(name, 0) == reported,
    } for name, reported in claims.items()}
    result = {
        "status": "PASS" if not discrepancies and all(x["match"] for x in comparison.values())
                  else "BLOCKED_MISMATCH",
        "source_counts": dict(source), "greedy_events": dict(events),
        "p1_p0_claim_comparison": comparison,
        "source_archive_load_mismatch_count": len(discrepancies),
        "source_archive_load_mismatch_examples": discrepancies[:20],
        "high_soc_fc_zero_examples": examples,
        "manifest_checks": manifest_checks, "by_sample": samples,
        "test_payloads_opened": dataset.opened_test_payloads,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
