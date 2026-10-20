"""Read saved JSON/CSV diagnostics only; never import training code or torch.

Run from any directory. Output is beside this script. No model execution,
sampling, training, fitting, gradients, Test data, or checkpoint deserialization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


AUDIT = Path(__file__).resolve().parent
RUN = AUDIT.parent
ROOT = RUN.parent.parent
CATEGORIES = (
    "failure_terminal", "normal_terminal", "ordinary_nonterminal",
    "failure_prefix", "unresolved_prefix",
)
ERROR_FIELDS = (
    "mean_td_error", "mean_absolute_td_error", "root_mean_square_td_error",
    "max_absolute_td_error", "mean_q_scaled", "mean_target_scaled",
    "absolute_td_above_one_count", "absolute_td_above_one_fraction",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_source_hash():
    # Same definition as monitored_training.py:283-290. Reads src only.
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json"):
            digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def categories(td):
    return {
        name: td[name] if name in ("failure_terminal", "normal_terminal")
        else td["nonterminal_categories"][name]
        for name in CATEGORIES
    }


def flatten(values, prefix=""):
    for key, value in values.items():
        name = prefix + key
        if isinstance(value, dict):
            yield from flatten(value, name + "_")
        elif isinstance(value, (list, tuple)):
            yield name, json.dumps(value, ensure_ascii=False)
        else:
            yield name, value


def verify_csv(rounds):
    mismatch = []
    checked = 0
    with (RUN / "round_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for saved, row in zip(rows, rounds):
        expected = {
            "round": row["round"],
            "economic_optimizer_updates_this_round": row["economic_optimizer_updates"],
            "economic_optimizer_updates": row["economic_optimizer_updates_cumulative"],
            "economic_replay_insertions_this_round": row["economic_replay_insertions"],
            "economic_replay_insertions": row["economic_replay_insertions_cumulative"],
            "target_sync_calls_cumulative": row["target_sync_calls_cumulative"],
            "target_soft_update_calls_cumulative": row["target_soft_update_calls_cumulative"],
            **dict(flatten(row["td_statistics"], "td_")),
        }
        for key, value in expected.items():
            observed = saved.get(key)
            if value is None:
                okay = observed == ""
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    okay = math.isclose(float(observed), value, rel_tol=1e-12, abs_tol=1e-12)
                except (ValueError, TypeError):
                    okay = False
            else:
                okay = observed == str(value)
            checked += 1
            if not okay:
                mismatch.append({"round": row["round"], "column": key,
                                 "json": value, "csv": observed})
    return {"json_rounds": len(rounds), "csv_rows": len(rows),
            "checked_values": checked, "mismatches": mismatch,
            "all_passed": len(rows) == len(rounds) and not mismatch}


def concise_category(value, sample_total):
    n = value["sample_count"]
    return {"sample_count": n, "draw_fraction": n / sample_total if sample_total else None,
            **{key: value[key] for key in ERROR_FIELDS if key in value}}


def round_summary(row):
    td = row["td_statistics"]
    coverage = td["replay_coverage"]
    return {
        "round": row["round"], "epsilon": row["epsilon"],
        "updates": row["economic_optimizer_updates"],
        "updates_cumulative": row["economic_optimizer_updates_cumulative"],
        "insertions": row["economic_replay_insertions"],
        "draw_count": td["sample_count"],
        "mean_td_error": td["mean_td_error"],
        "mae": td["mean_absolute_td_error"],
        "rmse": td["root_mean_square_td_error"],
        "weighted_huber_loss": td["mean_smooth_l1_loss"],
        "unweighted_huber_loss": td["replay_sampling"]["mean_unweighted_smooth_l1_loss"],
        "mean_importance_weight": td["replay_sampling"]["mean_importance_weight"],
        "within_batch_repeated_draws": td["replay_sampling"]["within_batch_repeat_draws"],
        "categories": {key: concise_category(value, td["sample_count"])
                       for key, value in categories(td).items()},
        "action_delta_draw_counts": coverage["actual_draws_since_reset"].get("action_delta_kw", {}),
        "unique_id_coverage": coverage["unique_replay_ids_since_reset"],
        "pool_categories_at_round_end": coverage["pool_categories"],
        "greedy_completions": {
            key: {"completed": row[key]["evaluable_completed"],
                  "total": row[key]["evaluable_episodes"]}
            for key in ("greedy_train", "greedy_validation")},
    }


def aggregate(rows):
    totals = {"optimizer_updates": sum(row["economic_optimizer_updates"] for row in rows),
              "draw_count": sum(row["td_statistics"]["sample_count"] for row in rows)}
    cats = {name: [] for name in CATEGORIES}
    action_counts = Counter()
    for row in rows:
        td = row["td_statistics"]
        for name, value in categories(td).items():
            cats[name].append(value)
        action_counts.update(td["replay_coverage"]["actual_draws_since_reset"].get("action_delta_kw", {}))
    output = {}
    for name, items in cats.items():
        n = sum(value["sample_count"] for value in items)
        result = {"sample_count": n, "draw_fraction": n / totals["draw_count"]}
        for key in ("mean_td_error", "mean_absolute_td_error", "mean_q_scaled", "mean_target_scaled"):
            if all(key in value for value in items):
                result[key] = (sum(value[key] * value["sample_count"] for value in items
                                   if value["sample_count"]) / n if n else None)
        result["root_mean_square_td_error"] = (
            math.sqrt(sum(value["root_mean_square_td_error"] ** 2 * value["sample_count"]
                          for value in items if value["sample_count"]) / n) if n else None)
        result["max_absolute_td_error"] = max(
            (value["max_absolute_td_error"] for value in items if value["sample_count"]), default=None)
        if all("absolute_td_above_one_count" in value for value in items):
            result["absolute_td_above_one_count"] = sum(value["absolute_td_above_one_count"] for value in items)
            result["absolute_td_above_one_fraction"] = result["absolute_td_above_one_count"] / n if n else None
        output[name] = result
    return {**totals, "categories": output,
            "action_delta_draw_counts": dict(sorted(action_counts.items(), key=lambda kv: float(kv[0]))),
            "weighted_huber_loss": sum(row["td_statistics"]["mean_smooth_l1_loss"] *
                                       row["economic_optimizer_updates"] for row in rows) / totals["optimizer_updates"],
            "mean_preclip_gradient_norm": sum(row["td_statistics"]["gradient_statistics"]["mean_preclip_norm"] *
                                              row["economic_optimizer_updates"] for row in rows) / totals["optimizer_updates"],
            "clipped_updates": sum(row["td_statistics"]["gradient_statistics"]["clipped_updates"] for row in rows)}


def snapshot(row):
    td = row["td_statistics"]
    coverage = td["replay_coverage"]
    result = round_summary(row)
    result["replay_sampling"] = td["replay_sampling"]
    result["gradient_statistics"] = td["gradient_statistics"]
    result["target_sync_calls_cumulative"] = row["target_sync_calls_cumulative"]
    result["target_soft_update_calls_cumulative"] = row["target_soft_update_calls_cumulative"]
    result["actual_draw_marginals"] = coverage["actual_draws_since_reset"]
    result["pool_marginals_at_round_end"] = coverage["pool"]
    result["pool_vs_draw"] = {
        key: {"pool_fraction_at_end": value / sum(coverage["pool_categories"].values()),
              "draw_fraction_during_round": result["categories"][key]["draw_fraction"]}
        for key, value in coverage["pool_categories"].items()
    }
    result["pool_vs_draw_scope"] = "End-pool composition versus full-round draw counts; pool evolves, so this is descriptive and is not exact time-matched PER enrichment."
    return result


def main():
    report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
    rounds = report["rounds"]
    hp = report["hyperparameters"]
    counts = report["execution_counts"]
    report_hash, history_hash = sha256(RUN / "report.json"), sha256(RUN / "round_history.json")
    actual_source_hash = runtime_source_hash()
    csv_check = verify_csv(rounds)
    all_td = [report["bootstrap_td_statistics"], *[row["td_statistics"] for row in rounds]]
    integrity_issues = []
    for label, td in zip(["bootstrap", *range(1, len(rounds) + 1)], all_td):
        if td["sample_count"] != td["optimizer_updates"] * hp["batch_size"]:
            integrity_issues.append([label, "draws != updates*batch"])
        if sum(value["sample_count"] for value in categories(td).values()) != td["sample_count"]:
            integrity_issues.append([label, "category sum != draw count"])
        if td["failure_terminal"]["sample_count"] != td["optimizer_updates"] * hp["failure_terminal_quota"]:
            integrity_issues.append([label, "quota does not equal 2 per successful batch"])
        for dimension, marginal in td["replay_coverage"]["actual_draws_since_reset"].items():
            if sum(marginal.values()) != td["sample_count"]:
                integrity_issues.append([label, dimension + " draw marginal sum mismatch"])
    missing_actions = [{"round": row["round"], "missing": [a for a in hp["action_kw"]
                        if str(a) not in row["td_statistics"]["replay_coverage"]["actual_draws_since_reset"].get("action_delta_kw", {})]}
                       for row in rounds]
    watched_n = [len(td.get("watched_replay_targets", [])) for td in all_td]
    refs = {
        "actual_loss_and_td": "src/v4/dqn.py:453-549",
        "conditional_sampling_and_quota": "src/v4/dqn.py:660-688",
        "category_assignment": "src/v4/dqn.py:560-570",
        "actual_draw_marginal_accumulation": "src/v4/dqn.py:519-535,572-586",
        "round_reset_and_unique_coverage": "src/v4/dqn.py:690-714,731-734,797-813; src/v4/monitored_training.py:877-897",
        "priority_mass_initialization_and_replacement": "src/v4/prioritized_replay.py:63-81,94-116",
        "conditional_IS_normalizer": "src/v4/prioritized_replay.py:118-133; src/v4/dqn.py:676-686",
        "priority_refresh_uses_unweighted_preupdate_td": "src/v4/dqn.py:467-498",
        "target_initial_copy_and_every_optimizer_soft_update": "src/v4/dqn.py:131-143,486-488,820-828",
        "cadence": "src/v4/experiment_schedule.py:20-44; src/v4/monitored_training.py:763-768,908-972",
        "csv_semantics_and_report_identity": "src/v4/monitored_training.py:189-238",
        "checkpoint_current_priorities_not_history": "src/v4/prioritized_replay.py:135-140; src/v4/dqn.py:830-858",
        "bounded_watched_counts_not_universal_counts": "src/v4/dqn.py:29-30,278-304,534-535,606-639",
        "source_hash_definition": "src/v4/monitored_training.py:283-290",
    }
    output = {
        "audit_kind": "Saved category TD history and source semantics; descriptive, no model or training-state loading",
        "run_path": str(RUN),
        "inputs": {
            "report_sha256": report_hash, "round_history_sha256": history_hash,
            "report_history_byte_identical": report_hash == history_hash,
            "csv_sha256": sha256(RUN / "round_metrics.csv"), "csv_consistency": csv_check,
            "recorded_runtime_source_sha256": report["source_tree_sha256"],
            "current_runtime_source_sha256": actual_source_hash,
            "current_source_matches_recorded_run": actual_source_hash == report["source_tree_sha256"],
            "training_state_deserialized": False,
        },
        "units_and_estimand": {
            "all_td_numbers": "Scaled training reward units (reward_scale=0.001); original units are x1000; includes shaping/failure, not expenditure.",
            "residual_direction": "delta=y-q from the actual pre-update sampled Q/target. Positive calls for raising Q, negative for lowering Q only in scalar output space.",
            "history_scope": "Each round aggregates actual successful optimizer batches since its diagnostic reset, before each batch update, with changing models and a changing pool. These are not a fixed cohort evaluated at one snapshot.",
            "weights": "Category TD means/MAE/RMSE are unweighted draw statistics. The optimized loss applies per-item IS to Huber before the batch mean.",
            "window_aggregation": "Means weighted by sample_count, RMSE from pooled second moments, maxima across rounds. No regression/fitting.",
        },
        "source_references": refs,
        "actual_training_semantics": {
            "batch_size": hp["batch_size"], "cadence": hp["cadence"],
            "credit_rule": "One successful optimizer step consumes 32 economic replay insertions; residual credit crosses samples and rounds. consume() runs after a completed or failed sample is added; bootstrap grants the same insertion credit.",
            "not_episode16_despite_metadata_field": "updates_per_completed_episode=16 is a legacy field, inactive under replay32.",
            "quota": "2 failure-terminal draws without replacement per batch; remaining 62 drawn with replacement from ordinary PER, which includes normal terminals, ordinary nonterminals, failure prefixes, and any unresolved prefixes. Pools can fall back when short; actual saved history always met quota2.",
            "priority": "raw=abs(pre-update unweighted TD) on refresh; mass=(raw+1e-6)^0.6. New eligible entries get historical maximum raw priority. Duplicate IDs remain real loss terms but one max absolute TD refresh per batch ID. Failure terminals are excluded from the priority index with quota2.",
            "IS": "beta=1 fixed. Ordinary conditional weight=(Nordinary*p)^(-1)/Z. Failure-terminal weight=1/Z. One pool-level Z=max(1,(Nordinary*pmin)^(-1)) shared by both classes. Conditional IS recovers the old fixed-quota objective up to a common per-batch scale; it does not recover uniform replay over both pools.",
            "huber": "Smooth L1 beta=1 in scaled units: L=mean(w_i*Huber(q_i-y_i)). Scalar dL/dq_i=-w_i*clip(delta_i,-1,1)/64. Loss target is detached. abs(delta)>1 saturates its scalar derivative; raw TD magnitude alone is not a gradient-contribution magnitude.",
            "target": "Initial hard copy once. learn() always performs target.lerp_(online,0.001) immediately after every successful optimizer step; soft calls therefore equal optimizer steps in this run.",
        },
        "counts_and_integrity": {
            "execution_counts": counts,
            "bootstrap_draws": report["bootstrap_td_statistics"]["sample_count"],
            "formal_draws": sum(row["td_statistics"]["sample_count"] for row in rounds),
            "all_draws": sum(td["sample_count"] for td in all_td),
            "all_failure_terminal_draws": sum(td["failure_terminal"]["sample_count"] for td in all_td),
            "replay_insertions_minus_32_updates": counts["economic_replay_insertions"] - 32 * counts["economic_optimizer_updates"],
            "all_batch_category_quota_marginal_checks_passed": not integrity_issues,
            "integrity_issues": integrity_issues,
            "rounds_with_missing_action_delta_labels": [value for value in missing_actions if value["missing"]],
            "watched_replay_targets_count_min_max": [min(watched_n), max(watched_n)],
        },
        "selected_rounds": {str(n): snapshot(rounds[n-1]) for n in (38, 62, 70, 80, 90, 100)},
        "late_windows": {f"R{start}-R{end}": aggregate(rounds[start-1:end])
                         for start, end in ((61,70), (71,80), (81,90), (91,100))},
        "formal_all_rounds": aggregate(rounds),
        "round_series": [round_summary(row) for row in rounds],
        "interpretation_boundaries": {
            "supported": [
                "R62 to R100 sampled RMSE increases for ordinary_nonterminal, failure_prefix and failure_terminal; normal_terminal RMSE decreases. A claim that all categories worsen is unsupported.",
                "Late pooled windows show ordinary_nonterminal RMSE rising throughout; failure-terminal RMSE is low in R71-80 and rises strongly afterward. Failure-prefix RMSE rises into R81-90 and is slightly lower in R91-100. Normal-terminal RMSE falls across all four late windows.",
                "At R100 mean residual is positive for ordinary_nonterminal and negative for failure_prefix, failure_terminal and normal_terminal. This is a scalar TD correction tendency, not a measured parameter-gradient relation.",
                "All seven increment labels are represented in the actual round draw marginals. This establishes action-label coverage only, not coverage of each action at every state, category or source.",
            ],
            "not_identified": [
                "No category-separated parameter gradients, gradient cosine/dot products, per-item Jacobians, controlled class-specific updates, or fixed-cohort before/after effects are saved here. Opposite mean residuals and simultaneous errors cannot establish causal gradient interference.",
                "The pool, policy, PER priorities, sampled states and online/target functions change during/between rounds. Cross-round residual changes are not pure forgetting estimates for identical experiences.",
                "Marginal draw counts cannot reconstruct joint state/action/category/source sampling frequencies or per-item exposure distributions.",
                "Saved loss/weight scales do not by themselves quantify Adam parameter-step magnitudes or class-specific optimization influence.",
            ],
        },
        "saved_evidence_limits": {
            "general_per_experience_actual_draw_count_saved": False,
            "actual_draw_sequence_or_full_sampled_ID_list_saved": False,
            "round_unique_ID_count_saved": True,
            "round_unique_ID_set_saved_in_report": False,
            "bounded_watch_exception": "Only WATCHED_REPLAY_SOURCES retain limited live ID counters, evicted/old watched IDs are dropped. This run's saved probe lists are empty (see counts). This is not a complete lifetime exposure ledger.",
            "per_ID_priority_time_history_saved": False,
            "checkpoint_priority_content_from_source_only": "Current live (ID,raw_priority) pairs plus a historical maximum scalar; not timestamps, refresh counts or each priority's trajectory. This audit did not deserialize checkpoints.",
            "category_failure_terminal_missing_fields": "Saved failure_terminal summary has counts and TD moments, but not class mean Q/target or abs(TD)>1 fraction even though internal category accumulation exists.",
            "no_new_training_or_gradient_computation": True,
            "no_test_payload_read": True,
        },
    }
    target = AUDIT / "category_history_review.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "bytes": target.stat().st_size,
                      "report_history_identical": output["inputs"]["report_history_byte_identical"],
                      "source_matches": output["inputs"]["current_source_matches_recorded_run"],
                      "csv_checks": csv_check["checked_values"], "csv_pass": csv_check["all_passed"],
                      "history_checks_pass": not integrity_issues,
                      "missing_action_rounds": len(output["counts_and_integrity"]["rounds_with_missing_action_delta_labels"]),
                      "watched_counts_min_max": [min(watched_n), max(watched_n)]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
