"""Train-only offline failure-scale audit; never replay, train, or change F.

Archived per-voyage summaries may substantiate cumulative ledgers even when
the separately retained per-step failure observations are sparse. The two
evidence levels are deliberately kept separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def train_only_path(path):
    path = Path(path)
    if any(set(re.split(r"[_\-.]", part.lower())) & {"test", "validation", "val"}
           for part in path.parts):
        raise ValueError(f"Train-only audit rejects non-Train payload: {path}")
    return path


def distribution(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(count=0, min=None, median=None, p95=None, max=None)
    return dict(count=len(values), min=values[0], median=statistics.median(values),
                p95=values[math.ceil(.95*len(values))-1], max=values[-1])


def number(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"nonfinite archived field {key}")
    return value


def summarize_train_row(row):
    completed = str(row["completed"]).lower() == "true"
    parts = ("fc_h2_cost_cny", "fc_degradation_cost_cny",
             "battery_degradation_cost_cny", "battery_h2eq_cost_cny",
             "shore_electricity_cost_cny")
    economic = math.fsum(number(row, key) for key in parts)
    logged = number(row, "executed_prefix_economic_cost_cny")
    penalty = number(row, "failure_penalty_reward_units")
    full = number(row, "complete_economic_cost_cny") if completed else None
    if abs(economic-logged) > 1e-5 or (full is not None and abs(full-economic) > 1e-5):
        raise ValueError(f"archived economic summary inconsistent: {row['sample_id']}")
    return dict(sample_id=row["sample_id"], source=row.get("source", "fixture"),
        completed=completed, status=row["status"], executed_steps=int(row["executed_steps"]),
        voyage_duration_s=number(row, "voyage_duration_s"),
        duration_coverage=number(row, "duration_coverage"), soc_final=number(row, "soc_terminal"),
        **{key:number(row, key) for key in parts}, economic_cost_cny=economic,
        complete_economic_cost_cny=full, failure_penalty_cny=penalty,
        economic_only_return=-math.fsum((economic, penalty)),
        economic_ledger_identity_residual_cny=economic-logged,
        evidence="archived_cumulative_summary_not_replayed")


def summarize_sparse_group(rows):
    times = [number(r, "time_s") for r in rows]
    if any(r.get("source_split") != "train" for r in rows):
        raise ValueError("Train-only sparse source contains a non-Train row")
    contiguous = (times[0] == 0 and all(b-a == 1 for a,b in zip(times, times[1:])))
    return dict(sample_id=rows[0]["sample_id"], origin_round=rows[0]["origin_round"],
        recorded_steps=len(rows), first_time_s=min(times), last_time_s=max(times),
        contiguous_from_reset=contiguous, has_complete_step_ledger=False,
        economic_prefix_cost_cny=None, complete_economic_cost_cny=None,
        step_economic_cost_distribution_cny=None,
        recorded_h2_step_cost_distribution_cny=distribution(number(r,"fc_h2") for r in rows),
        recorded_signed_soc_step_cost_distribution_cny=distribution(number(r,"battery_h2eq") for r in rows),
        note="Sparse observed rows omit intermediate actions and both degradation ledgers; do not sum as a voyage.")


def compare_same_sample(rows, *, penalty):
    comparisons = []
    for failed in (r for r in rows if not r["completed"]):
        for completed in (r for r in rows if r["completed"] and r["sample_id"] == failed["sample_id"]):
            gap = completed["economic_cost_cny"] - failed["economic_cost_cny"]
            comparisons.append(dict(sample_id=failed["sample_id"],
                failed_source=failed["source"], completed_source=completed["source"],
                failed_economic_prefix_cny=failed["economic_cost_cny"],
                completed_economic_cost_cny=completed["economic_cost_cny"],
                failed_coverage=failed["duration_coverage"],
                failure_penalty_cny=penalty, failed_pure_economic_return=-(failed["economic_cost_cny"]+penalty),
                completed_pure_economic_return=-completed["economic_cost_cny"],
                break_even_failure_penalty_cny=gap,
                failed_return_exceeds_completed_return=gap>penalty))
    return comparisons


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT/"docs/results/sb3_dqn_1s_20261028/raw")
    parser.add_argument("--dataset", type=Path, default=ROOT/"data/processed/operating_dataset_independent_v2")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/audits/v6_episode500_20261029")
    args = parser.parse_args(argv)
    train_only_path(args.source_root)
    sources = []

    def provenance(path, kind):
        path = Path(path)
        sources.append(dict(path=path.resolve().relative_to(ROOT).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), kind=kind))

    # Manifest identities are allowed for split verification, never calibration.
    split_path = args.dataset/"metadata/active_split.json"
    metadata = json.loads(split_path.read_text(encoding="utf-8"))
    manifest_path = args.dataset/metadata["manifest_relative_path"]
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    provenance(split_path, "split_metadata_only")
    provenance(manifest_path, "split_identity_and_Train_duration_metadata_only")
    train_rows = [r for r in manifest if r["split"] == "train"]
    allowed_ids = {r["candidate_id"] for r in train_rows}
    if len(allowed_ids) != 35:
        raise ValueError("expected frozen 35-Train split")
    reference_path = ROOT/"src/v4/failure_penalty_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    provenance(reference_path, "inherited_reference_provenance_not_recalibration")
    ref_costs = sorted(reference["training_costs_cny"].values())
    penalty = ref_costs[math.ceil(.95*len(ref_costs))-1]
    ref_ids = set(reference["training_costs_cny"])
    identity_overlap = {split:sorted({r["parent_sample_id"] for r in manifest if r["split"]==split}&ref_ids)
                        for split in ("train", "validation", "test")}
    # Hash the historical mixed-split file for provenance, without decoding any
    # trajectory or consulting its Validation section or economic values.
    archived_reference = ROOT/reference["source_file"]
    source_hash = hashlib.sha256(archived_reference.read_bytes()).hexdigest() if archived_reference.exists() else None

    summaries = []
    for path in sorted(args.source_root.glob("round_*_train_samples.csv")):
        train_only_path(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 35 or {r["sample_id"] for r in rows} != allowed_ids:
            raise ValueError(f"summary must cover only the fixed 35 Train IDs: {path}")
        summaries.extend(summarize_train_row(dict(r, source=path.name)) for r in rows)
        provenance(path, "Train_cumulative_summary")
    if not summaries:
        raise ValueError("no archived Train summaries; do not synthesize missing costs")
    sparse = []
    for path in sorted(args.source_root.glob("round_*_train_failure_prefixes.csv.gz")):
        train_only_path(path)
        groups = defaultdict(list)
        with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as handle:
            for row in csv.DictReader(handle):
                if row["sample_id"] not in allowed_ids:
                    raise ValueError("sparse observations contain a non-Train ID")
                groups[row["sample_id"]].append(row)
        sparse.extend(dict(source=path.name, **summarize_sparse_group(rows)) for rows in groups.values())
        provenance(path, "Train_sparse_observation_no_full_step_ledger")
    completed = [r for r in summaries if r["completed"]]
    failed = [r for r in summaries if not r["completed"]]
    comparisons = compare_same_sample(summaries, penalty=penalty)
    complete_costs = distribution(r["economic_cost_cny"] for r in completed)
    failed_costs = distribution(r["economic_cost_cny"] for r in failed)
    gap_costs = distribution(max(0., r["break_even_failure_penalty_cny"]) for r in comparisons)
    best_per_id = [min(r["economic_cost_cny"] for r in completed if r["sample_id"]==sid)
                   for sid in sorted({r["sample_id"] for r in completed})]
    sensitivity = []
    for label, amount in (("inherited",penalty), ("Train_complete_p95",complete_costs["p95"]),
                          ("Train_complete_max",complete_costs["max"]),
                          ("observed_same_id_gap_max",gap_costs["max"])):
        sensitivity.append(dict(candidate=label, proposed_only=True, penalty_cny=amount,
            same_id_pairs_where_failed_return_is_higher=sum(r["break_even_failure_penalty_cny"]>amount for r in comparisons),
            same_id_pairs=len(comparisons)))
    durations = [float(r["duration_s"]) for r in train_rows]
    result = dict(audit_scope="Train-only; no new controller execution or training",
        inherited_penalty_cny=penalty, penalty_changed=False, formal_training_requires_confirmation=True,
        reference=dict(count=len(ref_costs), cost_distribution_cny=distribution(ref_costs),
            source_hash_matches=source_hash==reference["source_sha256"], source_sha256=source_hash,
            old_train_reference_ids_by_current_split=identity_overlap,
            current_Train_only_calibration=False,
            caveat="Historical zero_boundary/old split comparable ledger; no signed SOC term in its stored comparable sum. Not a 35-Train 1-second calibration."),
        archive=dict(rows=len(summaries), complete=len(completed), failed=len(failed),
            complete_Train_ids=len(best_per_id), failure_reasons=dict(Counter(r["status"] for r in failed)),
            Train_ids_without_any_archived_completion=sorted(allowed_ids-{r["sample_id"] for r in completed}),
            complete_economic_cost_distribution_cny=complete_costs,
            per_id_best_complete_economic_cost_distribution_cny=distribution(best_per_id),
            failed_prefix_economic_cost_distribution_cny=failed_costs,
            complete_costs_exceeding_inherited_penalty=sum(r["economic_cost_cny"]>penalty for r in completed),
            max_abs_summary_ledger_residual_cny=max(abs(r["economic_ledger_identity_residual_cny"]) for r in summaries),
            sparse_groups=len(sparse), sparse_observation_rows=sum(r["recorded_steps"] for r in sparse),
            full_Train_step_economic_cost_distribution_cny=None),
        same_id_rescoring=dict(pairs=len(comparisons),
            failed_return_exceeds_completed_count=sum(r["failed_return_exceeds_completed_return"] for r in comparisons),
            positive_break_even_penalty_distribution_cny=gap_costs,
            interpretation="Same voyage and reset contract, different archived behaviors/rounds; economic re-scoring only. Not same-prefix continuation, causal intent, or new-policy performance."),
        candidate_sensitivity_not_applied=sensitivity,
        schedule_metadata=dict(train_voyages=len(durations), full_episode_steps=distribution(durations),
            full_completion_500_min_steps=500*min(durations), full_completion_500_max_steps=500*max(durations),
            full_completion_500_expected_steps=500*math.fsum(durations)/len(durations),
            actual_with_failure_lower_bound_steps=0,
            note="Bounds assume no failure and 1-second grid. Failures can shorten episodes; zero-action failures contribute no transition."),
        structural_failure_metadata=dict(Train_peak_max_kw=max(float(r["load_peak_kw"]) for r in train_rows),
            note="No structural failure in these archived Train summaries; metadata peaks alone do not rule out SOC/increment infeasibility."),
        new_payloads_opened=dict(train=0, validation=0, test=0),
        limitations=["Cumulative Train summaries were not independently replayed; internal component sums were verified.",
            "Sparse failure observations cannot reconstruct full step costs, capped degradation, or a complete trajectory.",
            "Old reference hash checks bytes only; mixed-split archived profiles were not decoded.",
            "No Validation/Test trajectory or cost was used to derive new candidates.",
            "Observed maximum/p95 are policy-dependent and not a bound on unseen policies."])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/"train_economic_summary_evidence.csv", summaries)
    write_csv(args.output/"train_same_id_failure_comparisons.csv", comparisons)
    (args.output/"train_sparse_failure_evidence.json").write_text(json.dumps(sparse,indent=2,allow_nan=False),encoding="utf-8")
    (args.output/"failure_penalty_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (args.output/"failure_penalty_source_manifest.json").write_text(json.dumps(sources,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), complete=len(completed), failed=len(failed),
        comparisons=len(comparisons), higher_failure_returns=result["same_id_rescoring"]["failed_return_exceeds_completed_count"],
        penalty_cny=penalty, formal_confirmation_required=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
