"""Per-round greedy monitoring with qualified economic checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time

import torch

from v3.control import EconomicMPC

from .control import ACTION_KW, ReplayExecutionError, replay_episode
from .dqn import DirectPowerDDQN
from .experiment_schedule import BestCheckpoint, EconomicUpdateSchedule, fully_completed
from .train import _run_episodes, _summarize


def greedy_evaluate(episodes, agent, accountant, beta_soc: float) -> dict:
    """Preserve all training RNG streams; never add replay or run optimizers."""
    random_state = agent.random.getstate()
    outcome_random_state = agent.outcome_random.getstate()
    torch_state = torch.random.get_rng_state()
    failed_transitions = []
    try:
        results, failures = _run_episodes(
            episodes, lambda state, feasible: agent.select_power(state, feasible, epsilon=0.0),
            accountant, beta_soc=beta_soc,
            on_failure=lambda exc: failed_transitions.extend(exc.executed_transitions),
        )
        return _summarize(results, len(episodes), failures, failed_transitions)
    finally:
        agent.random.setstate(random_state)
        agent.outcome_random.setstate(outcome_random_state)
        torch.random.set_rng_state(torch_state)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_monitored_training(
    dataset, *, output_dir: Path, rounds: int = 40, beta_soc: float,
    cadence: str = "episode16", target_mode: str = "round", target_interval: int = 1000,
    seed: int = 42, batch_size: int = 64, updates_per_episode: int = 16,
    epsilon_start: float = 1.0, epsilon_end: float = 0.05, progress_every_steps: int = 0,
):
    if not 1 <= rounds <= 40 or updates_per_episode != 16 or batch_size < 1:
        raise ValueError("study permits at most 40 rounds and fixes episode cadence at 16")
    if not 0 <= epsilon_end <= epsilon_start <= 1 or beta_soc < 0:
        raise ValueError("invalid beta or epsilon schedule")
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test must be closed before the study")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in ("report.json", "round_history.json", "best_agent.pt")):
        raise FileExistsError("training requires a fresh result directory")
    started = time.monotonic()
    train = dataset.load_train()
    validation = dataset.load_validation()
    if not train or not validation:
        raise ValueError("Train and Validation must be nonempty")
    if any(item.split != "train" for item in train) or any(item.split != "validation" for item in validation):
        raise ValueError("dataset split identity differs")
    agent = DirectPowerDDQN(seed=seed)
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    schedule = EconomicUpdateSchedule(cadence, target_mode=target_mode, target_interval=target_interval)
    selector = BestCheckpoint()
    bootstrap_failed = []
    bootstrap_prefix = 0

    def bootstrap_failure(exc):
        nonlocal bootstrap_prefix
        bootstrap_failed.extend(exc.executed_transitions)
        if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
            bootstrap_prefix += agent.remember_completed_prefix(exc.executed_transitions)
            agent.remember_outcome_trajectory(exc.executed_transitions, failed=True)

    print(f"bootstrap beta={beta_soc:g} cadence={cadence} target={target_mode} start", flush=True)
    bootstrap, bootstrap_errors = _run_episodes(
        train, lambda state, feasible: min(feasible, key=lambda power: abs(power-state[1]*600)),
        accountant, beta_soc=beta_soc, on_failure=bootstrap_failure,
    )
    for result in bootstrap:
        for transition in result.transitions:
            agent.remember_transition(transition)
        if result.transitions:
            agent.remember_outcome_trajectory(result.transitions, failed=False)
    schedule.grant(insertions=agent.economic_replay_insertions, completed_episodes=len(bootstrap))
    bootstrap_losses = schedule.consume(agent, batch_size=batch_size)
    for _ in range(16 * (len(bootstrap) + len(bootstrap_errors))):
        agent.learn_outcome(batch_size=batch_size)
    if target_mode == "round":
        agent.sync_target()
    bootstrap_summary = _summarize(bootstrap, len(train), bootstrap_errors, bootstrap_failed)
    print(f"bootstrap completed={len(bootstrap)}/{len(train)} economic_updates={agent.economic_optimizer_updates}", flush=True)
    history = []
    selected_steps = 0
    evaluation_steps = 0
    best_record = None
    first_qualification = None
    hyperparameters = {
        "beta_soc": beta_soc, "rounds": rounds, "seed": seed, "batch_size": batch_size,
        "gamma": agent.gamma, "learning_rate": 1e-4, "hidden_dims": [128,64],
        "action_kw": list(ACTION_KW), "replay_capacity": 100000,
        "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
        "cadence": cadence, "updates_per_completed_episode": 16,
        "target_mode": target_mode, "target_interval_optimizer_updates": target_interval if target_mode == "optimizer" else None,
        "bootstrap_uses_same_cadence": True, "torch_threads": torch.get_num_threads(),
    }
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()

    def snapshot_report(completed_training=False):
        return {
            "completed_training": completed_training, "source_commit": source_commit,
            "hyperparameters": hyperparameters, "bootstrap": bootstrap_summary,
            "bootstrap_optimizer_updates": len(bootstrap_losses),
            "bootstrap_economic_replay_insertions": sum(len(x.transitions) for x in bootstrap)+bootstrap_prefix,
            "rounds": history, "best_checkpoint": best_record,
            "first_qualification": first_qualification, "eligible_rounds": selector.eligible_rounds,
            "test_payloads_opened": dataset.opened_test_payloads,
            "execution_counts": {
                "training_environment_transitions": selected_steps,
                "bootstrap_environment_transitions": bootstrap_summary["executed_onboard_transitions"],
                "greedy_evaluation_environment_transitions": evaluation_steps,
                "economic_replay_insertions": agent.economic_replay_insertions,
                "economic_optimizer_updates": agent.economic_optimizer_updates,
                "outcome_optimizer_updates": agent.outcome_optimizer_updates,
                "target_sync_calls_including_initial_copy": agent.target_sync_calls,
                "remaining_transition_credit": schedule.remaining_transition_credit,
            },
            "elapsed_seconds": time.monotonic()-started,
            "cost_rule": "Actual plus modeled terminal settlement; SOC shaping excluded; incomplete split cost is null",
            "monitoring_rng_isolation": True,
        }

    for round_index in range(1, rounds+1):
        epsilon = epsilon_start if rounds == 1 else epsilon_end if round_index == rounds else (
            epsilon_start+(epsilon_end-epsilon_start)*(round_index-1)/(rounds-1)
        )
        print(f"round {round_index}/{rounds} start epsilon={epsilon:.4f}",flush=True)
        results, failures, failed_transitions, losses = [], [], [], []
        before_insertions = agent.economic_replay_insertions
        before_updates = agent.economic_optimizer_updates
        for episode_index, episode in enumerate(train, start=1):
            def training_policy(state, feasible):
                nonlocal selected_steps
                action = agent.select_power(state, feasible, epsilon=epsilon)
                selected_steps += 1
                if progress_every_steps and selected_steps % progress_every_steps == 0:
                    print(f"progress step={selected_steps} round={round_index}/{rounds} episode={episode_index}/{len(train)} sample={episode.sample_id} soc={state[0]:.4f} fc={action}",flush=True)
                return action

            before_episode_insertions = agent.economic_replay_insertions
            try:
                result = replay_episode(episode,training_policy,accountant=accountant,beta_soc=beta_soc)
            except ReplayExecutionError as exc:
                failures.append(f"{episode.sample_id}: {exc}")
                failed_transitions.extend(exc.executed_transitions)
                if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    agent.remember_completed_prefix(exc.executed_transitions)
                    agent.remember_outcome_trajectory(exc.executed_transitions,failed=True)
                schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,completed_episodes=0)
                losses.extend(schedule.consume(agent,batch_size=batch_size))
                if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    for _ in range(16):
                        agent.learn_outcome(batch_size=batch_size)
                continue
            results.append(result)
            for transition in result.transitions:
                agent.remember_transition(transition)
            if result.transitions:
                agent.remember_outcome_trajectory(result.transitions,failed=False)
            schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,completed_episodes=1)
            losses.extend(schedule.consume(agent,batch_size=batch_size))
            for _ in range(16):
                agent.learn_outcome(batch_size=batch_size)
        if target_mode == "round":
            agent.sync_target()
        explore = _summarize(results,len(train),failures,failed_transitions)
        print(f"round {round_index}/{rounds} exploratory={len(results)}/{len(train)} greedy_train start",flush=True)
        greedy_train = greedy_evaluate(train,agent,accountant,beta_soc)
        evaluation_steps += greedy_train["executed_onboard_transitions"]
        greedy_validation = None
        if fully_completed(greedy_train):
            print(f"round {round_index}/{rounds} greedy_train={len(train)}/{len(train)} validation start",flush=True)
            greedy_validation = greedy_evaluate(validation,agent,accountant,beta_soc)
            evaluation_steps += greedy_validation["executed_onboard_transitions"]
        row = {
            "round":round_index,"epsilon":epsilon,"exploratory_train":explore,
            "greedy_train":greedy_train,"greedy_validation":greedy_validation,
            "validation_skip_reason":None if greedy_validation is not None else "greedy_train_incomplete",
            "mean_loss":sum(losses)/len(losses) if losses else None,
            "economic_optimizer_updates":agent.economic_optimizer_updates-before_updates,
            "economic_optimizer_updates_cumulative":agent.economic_optimizer_updates,
            "economic_replay_insertions":agent.economic_replay_insertions-before_insertions,
            "economic_replay_insertions_cumulative":agent.economic_replay_insertions,
            "training_environment_transitions_cumulative":selected_steps,
            "target_sync_calls_cumulative":agent.target_sync_calls,
            "remaining_transition_credit":schedule.remaining_transition_credit,
        }
        improved = selector.consider(round_index,greedy_train,greedy_validation)
        row["qualified_checkpoint"] = fully_completed(greedy_train) and fully_completed(greedy_validation)
        row["best_checkpoint_improved"] = improved
        if row["qualified_checkpoint"] and first_qualification is None:
            first_qualification = {
                "round":round_index,"economic_optimizer_updates":agent.economic_optimizer_updates,
                "environment_transitions_including_bootstrap":selected_steps+bootstrap_summary["executed_onboard_transitions"],
                "economic_replay_insertions":agent.economic_replay_insertions,
            }
        if improved:
            best_record = {
                "round":round_index,"validation_comparable_cost_cny":selector.cost_cny,
                "train":greedy_train,"validation":greedy_validation,"path":"best_agent.pt",
                "economic_optimizer_updates":agent.economic_optimizer_updates,
                "economic_replay_insertions":agent.economic_replay_insertions,
            }
            temporary = output_dir/"best_agent.pt.tmp"
            torch.save({
                "architecture":"v4_mlp_double_dqn_direct_power",
                "model_state":{key:value.detach().cpu().clone() for key,value in agent.online.state_dict().items()},
                "hyperparameters":hyperparameters,"selection":best_record,
                "source_commit":source_commit,"test_payloads_opened":0,
            },temporary)
            temporary.replace(output_dir/"best_agent.pt")
        history.append(row)
        if dataset.opened_test_payloads != 0:
            raise RuntimeError("Test opened during monitored training")
        _write_json(output_dir/"round_history.json",snapshot_report())
        val = "SKIPPED" if greedy_validation is None else f"{greedy_validation['completed']}/{len(validation)}"
        cost = None if greedy_validation is None else greedy_validation["cost_cny"]
        print(f"MONITOR round={round_index} exploratory={len(results)}/{len(train)} greedy_train={greedy_train['completed']}/{len(train)} validation={val} cost={cost} updates={agent.economic_optimizer_updates} best_round={selector.round}",flush=True)
    report = snapshot_report(completed_training=True)
    _write_json(output_dir/"report.json",report)
    return agent, report
