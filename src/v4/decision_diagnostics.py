"""Bounded, read-only Q diagnostics at actual greedy Train decisions."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .control import ACTION_KW, DirectTransition


DIAGNOSTIC_SAMPLE_IDS = frozenset((
    "zero_boundary_015", "zero_boundary_017", "zero_boundary_044",
))
MAX_DECISIONS_PER_SAMPLE = 16


def actual_greedy_q_records(
    sample_id: str, transitions: tuple[DirectTransition, ...], agent, *,
    failed: bool = False,
) -> list[dict]:
    """Select a few executed decisions, then read all 61 online Q values.

    Tags describe the observed trajectory only. No action is selected here and
    no counterfactual state, replay entry, RNG draw, or optimizer step is made.
    """
    values = tuple(transitions)
    if not values:
        return []
    tagged: dict[int, set[str]] = {0: {"first_decision"}}
    tagged.setdefault(len(values) - 1, set()).add("last_decision")
    low_fc_run = 0
    for index, item in enumerate(values):
        if index and values[index - 1].done:
            low_fc_run = 0
        load = item.actual_battery_kw + item.action_kw
        tags = tagged.setdefault(index, set())
        if item.state[0] < .4 and load >= 600.:
            tags.add("low_soc_high_load")
        if load >= 600. and item.action_kw <= 100:
            low_fc_run += 1
            if low_fc_run >= 3:
                tags.add("consecutive_low_fc")
        else:
            low_fc_run = 0
        if item.state[0] >= .79:
            tags.add("near_soc_ceiling")
        if item.action_kw == 0 and item.state[4] > 0 and item.physical_feasible_actions == (0,):
            tags.add("forced_stop")
    if failed:
        for index in range(max(0, len(values) - 3), len(values)):
            tagged[index].add("near_failure_end")

    # Two early and two late examples per priority, then chronological output.
    chosen: set[int] = set()
    for label in ("forced_stop", "low_soc_high_load", "consecutive_low_fc",
                  "near_soc_ceiling", "near_failure_end", "first_decision", "last_decision"):
        matches = [index for index, tags in tagged.items() if label in tags]
        for index in (*matches[:2], *matches[-2:]):
            if len(chosen) < MAX_DECISIONS_PER_SAMPLE:
                chosen.add(index)
    indices = sorted(chosen)
    with torch.inference_mode():
        # Match select_power's single-state inference shape exactly. Batched
        # GEMM can change final-bit rounding and a near-tie action ranking.
        q_rows = [agent.online(torch.tensor([values[index].state], dtype=torch.float32))[0].tolist()
                  for index in indices]
    scale = agent.reward_scale
    records = []
    for index, q in zip(indices, q_rows):
        item = values[index]
        physical = item.physical_feasible_actions
        candidate = item.policy_candidate_actions
        if not candidate or item.action_kw not in candidate or not set(candidate).issubset(physical):
            raise ValueError("diagnostic decision violates the recorded Scheme A candidate mask")
        ordered = sorted(candidate, key=lambda action: (-q[action // 10], action))
        records.append({
            "sample_id": sample_id, "archive_transition_index": index,
            "priority_tags": sorted(tagged[index]), "state": list(item.state),
            "physical_actions_kw": list(physical), "candidate_actions_kw": list(candidate),
            "action_grid_kw": list(ACTION_KW), "q_values": q,
            "selected_action_kw": item.action_kw, "legal_action_order_kw": ordered,
            "selected_action_legal_rank": ordered.index(item.action_kw) + 1,
            "q_unit": "scaled reward-equivalent CNY", "reward_scale": scale,
        })
    return records


def save_diagnostic_snapshot(
    output_dir: Path, *, round_index: int, agent, decisions: list[dict],
    run_metadata: dict, training_environment_transitions: int,
    test_payloads_opened: int,
) -> dict[str, str]:
    """Write a separate network snapshot and bounded decision table atomically."""
    if test_payloads_opened != 0:
        raise RuntimeError("Test must remain closed before writing diagnostic artifacts")
    directory = output_dir / "diagnostic_checkpoints"
    directory.mkdir(exist_ok=True)
    checkpoint = directory / f"round_{round_index:03d}.pt"
    decision_file = directory / f"round_{round_index:03d}_decisions.json"
    if checkpoint.exists() or decision_file.exists():
        raise FileExistsError("diagnostic snapshot for this round already exists")
    payload = {
        "purpose": "diagnostic_only", "eligible_for_best_model_selection": False,
        "architecture": "v4_mlp_double_dqn_direct_power", "round": round_index,
        "online_state": {key: value.detach().cpu().clone()
                         for key, value in agent.online.state_dict().items()},
        "target_state": {key: value.detach().cpu().clone()
                         for key, value in agent.target.state_dict().items()},
        "source_commit": run_metadata["source_commit"],
        "source_worktree_dirty": run_metadata["source_worktree_dirty"],
        "hyperparameters": run_metadata["hyperparameters"],
        "manifest_sha256": run_metadata["manifest_sha256"],
        "dataset_roots": run_metadata["dataset_roots"],
        "economic_optimizer_updates": agent.economic_optimizer_updates,
        "target_sync_calls": agent.target_sync_calls,
        "economic_replay_insertions": agent.economic_replay_insertions,
        "outcome_optimizer_updates": agent.outcome_optimizer_updates,
        "training_environment_transitions": training_environment_transitions,
        "test_payloads_opened": test_payloads_opened,
    }
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    decision_payload = {
        "purpose": "diagnostic_only", "round": round_index,
        "source_commit": run_metadata["source_commit"],
        "checkpoint": checkpoint.relative_to(output_dir).as_posix(),
        "sample_ids": sorted(DIAGNOSTIC_SAMPLE_IDS),
        "max_decisions_per_sample": MAX_DECISIONS_PER_SAMPLE,
        "decisions": decisions,
    }
    temporary_json = decision_file.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_json.replace(decision_file)
    return {"checkpoint": checkpoint.relative_to(output_dir).as_posix(),
            "decisions": decision_file.relative_to(output_dir).as_posix()}
