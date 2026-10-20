"""Targeted two-voyage witness reconstruction and frozen Greedy-state comparison.

No training, fitting, optimizer, Test payload, or full-dataset search. Outputs
are opened exclusively: reruns cannot overwrite an earlier audit artifact.
"""
from dataclasses import asdict
import csv
import hashlib
import json
from math import fsum
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import audit_support as a

sys.path.insert(0, str(a.ROOT / "scripts"))
from check_v4_increment_reachability import search
from counterfactual_audit import compare_archive
from v2.data.supervisory_rules import normalize_onboard_load_kw
from v3.control import SOC_MIN, SOC_MAX
from v4.control import ReplayExecutionError, replay_episode
from v4.independent_economics import ZhouEvaluationConfig


def git(*args):
    return subprocess.check_output(["git", *args], cwd=a.ROOT, text=True).strip()


def sequence_hash(sequence):
    return hashlib.sha256(sequence.astype("<i2").tobytes()).hexdigest()


def formal_witness(episode, sequence, accountant, config, hp):
    cursor = 0
    previous = 0

    def policy(state, candidates):
        nonlocal cursor, previous
        command = int(sequence[cursor])
        assert abs(state[2] * 600 - previous) < 1e-9
        assert command - previous in a.FC_INCREMENT_KW
        assert command in candidates
        previous = command
        cursor += 1
        return command

    result = replay_episode(
        episode, policy, accountant=accountant, initial_state=a.AccountState(),
        state_mode=hp["state_mode"], action_mode=hp["action_mode"],
        reward_cost_mode=hp["reward_cost_mode"], beta_soc=hp["beta_soc"],
        economic_evaluation_config=config,
    )
    ts = result.transitions
    assert cursor == episode.step_count == len(ts)
    assert ts[-1].done and ts[-1].is_successful_terminal
    assert not result.shore_blocks and result.modeled_terminal_settlement is None
    assert not any(t.modeled_terminal_ledger or t.shore_ledger for t in ts)
    assert all(t.action_kw == int(sequence[i]) for i, t in enumerate(ts))
    assert all(t.action_delta_kw in a.FC_INCREMENT_KW for t in ts)
    assert all(t.action_kw in t.physical_feasible_actions
               and t.action_kw in t.policy_candidate_actions for t in ts)
    assert all(SOC_MIN <= t.actual_soc <= SOC_MAX for t in ts)
    assert all(accountant.plant.battery_charge_min_kw <= t.actual_battery_kw
               <= accountant.plant.battery_discharge_max_kw for t in ts)
    raw_reward = fsum(t.reward_cny for t in ts)
    soc_penalty = fsum(t.soc_soft_penalty_cny for t in ts)
    ramp_penalty = fsum(t.fc_ramp_penalty_cny for t in ts)
    economic_sum = fsum(t.executed_ledger.h2_cost_cny
                       + t.soc_equivalent_hydrogen_cost_cny
                       + t.executed_ledger.fuel_cell_degradation_cost_cny
                       + t.executed_ledger.battery_degradation_cost_cny for t in ts)
    errors = {
        "step_economics_vs_zhou_cost": abs(economic_sum - result.comparable_cost_cny),
        "raw_reward_decomposition": abs(-raw_reward - result.comparable_cost_cny
                                         - soc_penalty - ramp_penalty),
    }
    assert max(errors.values()) < 1e-8
    return result, {
        "sample_id": episode.sample_id, "split": episode.split,
        "status": "formal_increment7_zhou_joint_full_replay_verified",
        "steps": len(ts), "duration_s": fsum(t.duration_s for t in ts),
        "first_fc_kw": ts[0].action_kw,
        "maximum_absolute_delta_kw": max(abs(t.action_delta_kw) for t in ts),
        "minimum_soc": min(.6, *(t.actual_soc for t in ts)),
        "maximum_soc": max(.6, *(t.actual_soc for t in ts)),
        "terminal_soc": result.final_state.soc,
        "zhou_total_cost_cny": result.comparable_cost_cny,
        "sequence_sha256": sequence_hash(sequence),
        "soc_and_power_bounds_and_increment_masks_all_pass": True,
        "initial_state": asdict(a.AccountState()),
        "final_state": asdict(result.final_state),
        "terminal_settlement": None, "shore_blocks": 0,
        "raw_reward_sum": raw_reward, "soc_penalty_sum": soc_penalty,
        "quadratic_ramp_penalty_sum": ramp_penalty,
        "reconciliation_absolute_errors": errors,
        "economic_evaluation": asdict(result.economic_evaluation),
        "sequence_fc_kw": sequence.tolist(),
        "replayed_soc_after": [t.actual_soc for t in ts],
        "replayed_battery_bus_kw": [t.actual_battery_kw for t in ts],
    }


def main():
    paths = [a.OUT / "witness_vs_greedy_parent015.csv", a.OUT / "witness_review.json"]
    assert not any(p.exists() for p in paths), "refuse to overwrite audit outputs"
    baseline = json.loads((a.OUT / "readonly_baseline.json").read_text(encoding="utf-8"))
    protected = {
        str(p.relative_to(a.ROOT)): a.sha256(p)
        for p in [a.RUN / row["relative_path"] for row in baseline["protected_files"]]
        + [a.ROOT / "docs/results/v4_increment7_reachability_20261017.csv",
           a.ROOT / "scripts/check_v4_increment_reachability.py"]
    }
    assert all(a.sha256(a.RUN / row["relative_path"]) == row["sha256"]
               for row in baseline["protected_files"])
    frozen, cp, metadata = a.load_policy()
    hp = cp["hyperparameters"]
    accountant, config = a.accountant_config(cp)
    assert config.fc_average_efficiency == ZhouEvaluationConfig.from_formal_curve().fc_average_efficiency
    assert hp["initial_soc"] == .6 and hp["initial_fc_kw"] == 0
    assert tuple(hp["action_kw"]) == (-60, -30, -10, 0, 10, 30, 60)
    assert not frozen.model.training and not any(p.requires_grad for p in frozen.model.parameters())
    original_path = a.ROOT / "docs/results/v4_increment7_reachability_20261017.csv"
    with original_path.open(encoding="utf-8-sig", newline="") as handle:
        original = {r["sample_id"]: r for r in csv.DictReader(handle)}
    dataset = a.IndependentTrainingDataset.open(a.ROOT / "data/processed/operating_dataset_independent_v2")
    assert dataset.identity_hashes() == cp["manifest_sha256"] == metadata["manifest_sha256"]
    targets = (("parent_015_seg01", "train"), ("parent_044_seg01", "validation"))
    witnesses, episodes, replayed, payload_checks = {}, {}, {}, []
    for sid, split in targets:
        dataset._training_split(split)
        selected = dataset._manifest[(dataset._manifest.candidate_id == sid)
                                      & (dataset._manifest.split == split)]
        assert len(selected) == 1
        record = next(selected.itertuples(index=False))
        # Only the requested canonical Train/Validation payload is loaded.
        episode = dataset._load_episode(record)
        episodes[sid] = episode
        payload_checks.append({"sample_id": sid, "split": split,
                               "canonical_relative_path": record.relative_path,
                               "manifest_sha256": record.sha256,
                               "current_sha256": a.sha256(dataset.root / record.relative_path)})
        assert payload_checks[-1]["manifest_sha256"] == payload_checks[-1]["current_sha256"]
        status, sequence, evidence = search(episode, accountant)
        assert status == "candidate_witness", (sid, status, evidence)
        result, summary = formal_witness(episode, sequence, accountant, config, hp)
        old = original[sid]
        assert summary["sequence_sha256"] == old["sequence_sha256"]
        differences = {key: abs(summary[key] - float(old[key])) for key in
                       ("steps", "duration_s", "first_fc_kw", "maximum_absolute_delta_kw",
                        "minimum_soc", "maximum_soc", "terminal_soc", "zhou_total_cost_cny")}
        assert max(differences.values()) < 1e-8, differences
        summary.update({"original_csv_row": old, "original_csv_sequence_hash_matches": True,
                        "original_csv_metric_absolute_differences": differences})
        witnesses[sid], replayed[sid] = summary, result
        print(sid, "formal full replay", len(result.transitions), "matching sequence SHA", flush=True)

    episode = episodes["parent_015_seg01"]
    profile = a.load_profile(episode.sample_id, "train")
    try:
        greedy_result = replay_episode(
            episode, frozen, accountant=accountant, initial_state=a.AccountState(),
            state_mode=hp["state_mode"], action_mode=hp["action_mode"],
            reward_cost_mode=hp["reward_cost_mode"], beta_soc=hp["beta_soc"],
            economic_evaluation_config=config,
        )
        greedy_ts, error = greedy_result.transitions, None
    except ReplayExecutionError as exc:
        greedy_ts, error = exc.executed_transitions, exc
    archive_check = compare_archive(greedy_ts, profile, error)
    assert error is not None and error.row_index == 155
    with (a.OUT / "q_action_audit_parent015.csv").open(encoding="utf-8-sig", newline="") as handle:
        archived_q = list(csv.DictReader(handle))
    with (a.OUT / "counterfactual_parent_015_seg01_greedy_r100.csv").open(encoding="utf-8-sig", newline="") as handle:
        previous_greedy_csv = list(csv.DictReader(handle))
    assert len(archived_q) == len(previous_greedy_csv) == len(greedy_ts)
    comparison, q_maxdiff = [], 0.
    witness_ts = replayed[episode.sample_id].transitions
    for index, w in enumerate(witness_ts):
        row = {
            "step_1based": index + 1, "relative_time_s": w.relative_time_s,
            "duration_s": w.duration_s, "raw_load_kw": w.raw_load_kw,
            "witness_previous_fc_kw": round(w.state[2] * 600),
            "witness_fc_kw": w.action_kw, "witness_delta_kw": w.action_delta_kw,
            "witness_soc_before": w.state[0], "witness_soc_after": w.actual_soc,
            "witness_battery_bus_kw": w.actual_battery_kw,
            "witness_legal_absolute_fc_kw": json.dumps(w.physical_feasible_actions),
            "witness_increment_candidate_fc_kw": json.dumps(w.policy_candidate_actions),
            "witness_raw_reward": w.reward_cny,
            "witness_ledger_h2_cost_cny": w.executed_ledger.h2_cost_cny,
            "witness_soc_equivalent_h2_cost_cny": w.soc_equivalent_hydrogen_cost_cny,
            "witness_ledger_fc_degradation_cost_cny": w.executed_ledger.fuel_cell_degradation_cost_cny,
            "witness_ledger_battery_degradation_cost_cny": w.executed_ledger.battery_degradation_cost_cny,
            "greedy_status_at_step": "executed" if index < len(greedy_ts) else
                                     "failed_before_action" if index == error.row_index else "no_greedy_state_after_failure",
        }
        if index < len(greedy_ts):
            g = greedy_ts[index]
            state = g.state
            assert g.action_kw == int(previous_greedy_csv[index]["fc_kw"])
            assert g.state[0] == float(previous_greedy_csv[index]["soc_before"])
            row.update({"greedy_fc_kw": g.action_kw, "greedy_delta_kw": g.action_delta_kw,
                        "greedy_soc_after": g.actual_soc,
                        "greedy_battery_bus_kw": g.actual_battery_kw,
                        "same_command_as_greedy": w.action_kw == g.action_kw,
                        "greedy_raw_reward_before_failure_attachment": g.reward_cny})
        elif index == error.row_index:
            state = error.failed_state
            assert list(state) == profile["failed_observed_state"]
        else:
            comparison.append(row)
            continue
        previous = round(state[2] * 600)
        physical = a.AccountState(soc=state[0], previous_fc_kw=previous)
        full, legal_fc = a.physical_mask(physical, w.raw_load_kw, accountant, w.duration_s)
        qs = frozen.q(state)
        legal = [i for i, delta in enumerate(a.FC_INCREMENT_KW) if previous + delta in legal_fc]
        ranked = sorted(legal, key=lambda i: (-qs[i], i))
        raw_ranked = sorted(range(7), key=lambda i: (-qs[i], i))
        wd_index = a.FC_INCREMENT_KW.index(w.action_delta_kw)
        required_delta = w.action_kw - previous
        wa_index = a.FC_INCREMENT_KW.index(required_delta) if required_delta in a.FC_INCREMENT_KW else None
        best = ranked[0] if ranked else None
        if index < len(greedy_ts):
            assert g.action_index == best
            for j, delta in enumerate(a.FC_INCREMENT_KW):
                q_maxdiff = max(q_maxdiff, abs(qs[j] - float(archived_q[index][f"q_delta_{delta:+d}_kw"])))
        row.update({
            "greedy_soc_before": state[0], "greedy_previous_fc_kw": previous,
            "q_state_origin": "formal_R100_greedy_actual_state",
            "q_state_soc": state[0], "q_state_load_over600": state[1],
            "q_state_previous_fc_over600": state[2],
            "greedy_physical_absolute_fc_kw": json.dumps(full),
            "greedy_increment_candidate_fc_kw": json.dumps(legal_fc),
            "greedy_legal_action_indices": json.dumps(legal),
            "greedy_best_delta_kw": a.FC_INCREMENT_KW[best] if best is not None else None,
            "greedy_best_q": qs[best] if best is not None else None,
            "witness_delta_applied_to_greedy_fc_kw": previous + w.action_delta_kw,
            "witness_delta_legal_at_greedy_state": wd_index in legal,
            "witness_delta_q_at_greedy_state": qs[wd_index],
            "witness_delta_rank_among_all7_at_greedy_state": raw_ranked.index(wd_index) + 1,
            "witness_delta_rank_among_legal_at_greedy_state": ranked.index(wd_index) + 1 if wd_index in legal else None,
            "witness_absolute_fc_physically_legal_at_greedy_state": w.action_kw in full,
            "witness_absolute_fc_legal_increment_at_greedy_state": wa_index in legal,
            "required_delta_to_witness_absolute_fc_kw": required_delta,
            "required_delta_action_index": wa_index,
            "witness_absolute_fc_q_at_greedy_state": qs[wa_index] if wa_index is not None else None,
            "witness_absolute_fc_rank_among_all7_at_greedy_state": raw_ranked.index(wa_index) + 1 if wa_index is not None else None,
            "witness_absolute_fc_rank_among_legal_at_greedy_state": ranked.index(wa_index) + 1 if wa_index in legal else None,
        })
        for delta, value in zip(a.FC_INCREMENT_KW, qs):
            row[f"greedy_state_q_delta_{delta:+d}_kw"] = value
        comparison.append(row)
    assert q_maxdiff <= 1e-8
    first = next(r for r in comparison if r.get("same_command_as_greedy") is False)
    high_entry = comparison[104]
    assert high_entry["step_1based"] == 105 and high_entry["raw_load_kw"] > 600
    assert comparison[103]["raw_load_kw"] < 600
    # Current source and historical witness/run source have the same formal physics.
    witness_commit = git("log", "-1", "--format=%H", "--", "scripts/check_v4_increment_reachability.py")
    scope = ("src/v2", "src/v3", "src/v4/control.py", "src/v4/actions.py",
             "src/v4/independent_dataset.py", "src/v4/independent_economics.py",
             "src/v4/reward_feedback.py", "scripts/check_v4_increment_reachability.py")
    source_checks = {}
    for label, before, after in (("witness_commit_to_training_source", witness_commit, cp["source_commit"]),
                                 ("training_source_to_current_head", cp["source_commit"], "HEAD")):
        changed = git("diff", "--name-only", before, after, "--", *scope)
        assert not changed, changed
        source_checks[label] = {"before": before, "after": after, "changed_files_in_formal_physics_scope": []}
    assert not git("diff", "--name-only", "--", *scope)
    assert not git("diff", "--cached", "--name-only", "--", *scope)
    assert dataset.identity_hashes() == metadata["manifest_sha256"]
    assert dataset.opened_test_payloads == 0
    assert all(a.sha256(a.ROOT / relative) == digest for relative, digest in protected.items())
    for row in baseline["source_files"]:
        assert a.sha256(a.ROOT / row["file"]) == row["sha256"]
    columns = list(dict.fromkeys(key for row in comparison for key in row))
    with paths[0].open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(comparison)
    report = {
        "scope": "only Train parent_015_seg01 and Validation parent_044_seg01; targeted existing-search replay",
        "head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
        "checkpoint": str(a.RUN / "diagnostic_checkpoints/round_100.pt"),
        "checkpoint_sha256": a.sha256(a.RUN / "diagnostic_checkpoints/round_100.pt"),
        "checkpoint_round": cp["round"], "checkpoint_metadata_identity_matches": True,
        "experiment_id": cp["experiment_id"], "training_source_commit": cp["source_commit"],
        "current_dataset_metadata_hashes": dataset.identity_hashes(),
        "dataset_metadata_hashes_equal_checkpoint_and_run_metadata": True,
        "target_canonical_payload_sha256_checks": payload_checks,
        "task_contract": dataset.task_contract,
        "physical_plant": asdict(accountant.plant),
        "battery_efficiency": asdict(accountant.efficiency),
        "soc_bounds": [SOC_MIN, SOC_MAX], "initial_state": asdict(a.AccountState()),
        "action_deltas_kw": list(a.FC_INCREMENT_KW),
        "fc_physical_grid_kw": list(range(0, 601, 10)),
        "formal_source_identity_checks": source_checks,
        "formal_source_scope": list(scope),
        "original_csv_sha256": a.sha256(original_path),
        "original_csv_is_complete_action_sequence": False,
        "original_csv_limitation": "43 per-voyage summary rows; sequence SHA is not an encoded/recoverable action list; no per-witness data SHA is stored there. Matching current metadata to run and reproducing each original sequence SHA plus metrics supply the present audit evidence.",
        "search_method": "exact existing search(): SOC outer hull plus two actually reached endpoints per FC with predecessor backtracking; full observed load offline",
        "witnesses": witnesses,
        "greedy_parent015": {
            "completed": False, "executed_steps": len(greedy_ts),
            "failed_row_index_0based": error.row_index, "failed_step_1based": error.row_index + 1,
            "failure_kind": error.failure_kind, "failure_cause": error.failure_cause,
            "failed_observed_state": error.failed_state,
            "formal_replay_matches_archived_profile": archive_check,
            "recomputed_q_vs_prior_audit_max_absolute_difference": q_maxdiff,
            "first_divergence": first, "original_high_load_block_entry": high_entry,
        },
        "q_interpretation": "All compared Q vectors use formal R100 Greedy actual states only. Witness delta applied from Greedy previous FC and the absolute witness FC target are distinct counterfactual commands after paths diverge. Unavailable increments have no corresponding action Q; illegal actions have no legal-mask rank. All-7 rank is an unmasked output rank, not an executable-policy rank.",
        "limitations": [
            "Existence witnesses exploit full known episode load offline; no causal controller, economic optimum, Q ground truth, or generalization result is established.",
            "Replaying these two records cannot independently reverify all 43 historical witness rows.",
            "Witnesses approach SOC constraints closely; nominal replay is not evidence of robust feasibility with model/load uncertainty.",
            "A failed max-next-SOC heuristic is not a proof of physical infeasibility. Both audited complete witnesses establish nominal feasibility from the stipulated initial condition.",
            "A later witness state differs from the actual Greedy state; witness-path Q is not substituted into this comparison.",
            "Checkpoint Q rankings are learned estimates; ranking a witness increment does not measure the realized return of switching to that increment from Greedy state.",
        ],
        "output_files": {paths[0].name: {"rows": len(comparison), "sha256": a.sha256(paths[0])}},
        "protected_files_before_and_after_sha256": protected,
        "protected_files_and_existing_source_unchanged": True,
        "opened_test_payloads": 0, "training_calls": 0, "fitting_calls": 0,
        "optimizer_calls": 0, "gradient_calls": 0,
        "torch_mode": "CPU, eval, requires_grad False, inference_mode",
    }
    with paths[1].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"first_divergence": first, "high_load_entry": high_entry,
                      "q_maxdiff": q_maxdiff, "output_rows": len(comparison)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
