"""Per-round greedy monitoring with qualified economic checkpoint selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
import subprocess
import time

import torch

from v3.control import EconomicMPC

from .control import ACTION_KW, ReplayExecutionError, replay_episode
from .dqn import DirectPowerDDQN
from .experiment_schedule import BestCheckpoint, EconomicUpdateSchedule, fully_completed
from .train import _run_episodes, _summarize
from .diagnostics import fixed_q_diagnostics, trajectory_record
from .reward_feedback import BatteryEnergyValue
from .experiment_paths import unarchived_output_path
from .failure_replay import FailurePenalty, prepare_failure_replay


def greedy_evaluate(episodes, agent, accountant, beta_soc: float, *,
                    redistribute_battery_energy: bool = False, trajectory_path: Path | None = None,
                    learn_no_feasible_failures: bool = False, failure_penalty: FailurePenalty | None = None) -> dict:
    """Preserve all training RNG streams; never add replay or run optimizers."""
    random_state = agent.random.getstate()
    outcome_random_state = agent.outcome_random.getstate()
    torch_state = torch.random.get_rng_state()
    failed_transitions = []
    failure_profiles = []
    reason_counts={'soc_limited':0,'structural_power':0,'execution_error':0}
    event_only_count=0
    def on_failure(exc):
        nonlocal event_only_count
        cause=exc.failure_cause if exc.failure_kind=='no_feasible_action' else 'execution_error'
        reason_counts[cause]+=1
        transitions=exc.executed_transitions
        if learn_no_feasible_failures and exc.failure_kind=='no_feasible_action':
            preview=prepare_failure_replay(exc,penalty=failure_penalty or FailurePenalty.from_reference(),
                energy_value=BatteryEnergyValue.from_accountant(accountant),
                redistribute_battery_energy=redistribute_battery_energy)
            transitions=preview.transitions
            event_only_count+=int(preview.event_only)
        failed_transitions.extend(transitions)
        profile=trajectory_record(
            exc.sample_id,
            transitions, completed=False, failure=str(exc))
        profile.update(failure_cause=cause,failed_observed_state=exc.failed_state,
                       controllable_by_power_policy=False if cause=='structural_power' else None)
        failure_profiles.append(profile)
    try:
        results, failures = _run_episodes(
            episodes, lambda state, feasible: agent.select_power(state, feasible, epsilon=0.0),
            accountant, beta_soc=beta_soc,
            on_failure=on_failure, redistribute_battery_energy=redistribute_battery_energy,
        )
        summary = _summarize(results, len(episodes), failures, failed_transitions)
        summary.update(failure_reason_counts=reason_counts,event_only_failures=event_only_count)
        if trajectory_path is not None:
            _write_json(trajectory_path, {
                'completed': [trajectory_record(r.sample_id,r.transitions,completed=True) for r in results],
                'failed': failure_profiles,
            })
            summary['trajectory_file'] = trajectory_path.name
        return summary
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
    redistribute_battery_energy: bool = False, n_step: int = 1,
    episode_credit_scope: str = 'sample', required_split_sizes: tuple[int,int] | None = None,
    capture_trajectories: bool = False,
    learn_no_feasible_failures: bool = True, failure_penalty_scale: float = 1.0,
):
    if not 1 <= rounds <= 40 or updates_per_episode != 16 or batch_size < 1:
        raise ValueError("study permits at most 40 rounds and fixes episode cadence at 16")
    if (not 0 <= epsilon_end <= epsilon_start <= 1 or not math.isfinite(beta_soc) or beta_soc < 0
            or episode_credit_scope not in ('sample','voyage')):
        raise ValueError("invalid beta or epsilon schedule")
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test must be closed before the study")
    output_dir = unarchived_output_path(output_dir)
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
    if required_split_sizes is not None and (len(train),len(validation)) != required_split_sizes:
        raise ValueError('dataset sizes differ from the required checkpoint qualification set')
    agent = DirectPowerDDQN(seed=seed,n_step=n_step)
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    schedule = EconomicUpdateSchedule(cadence, target_mode=target_mode, target_interval=target_interval)
    failure_penalty=FailurePenalty.from_reference(scale=failure_penalty_scale)
    selector = BestCheckpoint()
    bootstrap_failed = []
    bootstrap_prefix = 0
    bootstrap_completed_voyages = 0
    failure_suffix_count = 0
    no_suffix_failure_events = 0
    initial_q_diagnostics = fixed_q_diagnostics(agent,accountant)
    bootstrap_reason_counts={'soc_limited':0,'structural_power':0,'execution_error':0}

    def replay_counts():
        n=agent.economic_replay_insertions
        return {
            'success':agent.economic_success_replay_insertions,'failure':agent.economic_failure_replay_insertions,
            'failure_terminals':agent.economic_failure_terminal_insertions,
            'success_fraction':agent.economic_success_replay_insertions/n if n else None,
            'failure_fraction':agent.economic_failure_replay_insertions/n if n else None,
        }

    def failure_view(exc):
        if learn_no_feasible_failures and exc.failure_kind=='no_feasible_action':
            return prepare_failure_replay(exc,penalty=failure_penalty,
                energy_value=BatteryEnergyValue.from_accountant(accountant),
                redistribute_battery_energy=redistribute_battery_energy)
        return None

    def bootstrap_failure(exc):
        nonlocal bootstrap_prefix, bootstrap_completed_voyages, failure_suffix_count, no_suffix_failure_events
        cause=exc.failure_cause if exc.failure_kind=='no_feasible_action' else 'execution_error'
        bootstrap_reason_counts[cause]+=1
        view=failure_view(exc)
        bootstrap_failed.extend(view.transitions if view else exc.executed_transitions)
        if exc.failure_kind == 'no_feasible_action' and (
                not exc.executed_transitions or exc.executed_transitions[-1].done):
            no_suffix_failure_events += 1
        if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
            if view is not None:
                retained=agent.remember_trajectory(view.transitions)
                prefix=exc.executed_transitions[:view.successful_prefix_transitions]
            else:
                retained=agent.remember_completed_prefix(exc.executed_transitions)
                prefix=exc.executed_transitions[:retained]
            bootstrap_prefix += retained
            bootstrap_completed_voyages += sum(t.is_successful_terminal for t in prefix)
            failure_suffix_count += len(exc.executed_transitions)-retained
            agent.remember_outcome_trajectory(exc.executed_transitions,
                                              failed=not exc.executed_transitions[-1].done)

    print(f"bootstrap beta={beta_soc:g} cadence={cadence} target={target_mode} start", flush=True)
    bootstrap, bootstrap_errors = _run_episodes(
        train, lambda state, feasible: min(feasible, key=lambda power: abs(power-state[1]*600)),
        accountant, beta_soc=beta_soc, on_failure=bootstrap_failure,
        redistribute_battery_energy=redistribute_battery_energy,
    )
    for result in bootstrap:
        agent.remember_trajectory(result.transitions)
        bootstrap_completed_voyages += sum(t.done for t in result.transitions)
        if result.transitions:
            agent.remember_outcome_trajectory(result.transitions, failed=False)
    schedule.grant(insertions=agent.economic_replay_insertions,
                   completed_episodes=(len(bootstrap) if episode_credit_scope == 'sample' else bootstrap_completed_voyages))
    bootstrap_losses = schedule.consume(agent, batch_size=batch_size)
    bootstrap_td_statistics = agent.td_statistics()
    for _ in range(16 * (len(bootstrap) + len(bootstrap_errors))):
        agent.learn_outcome(batch_size=batch_size)
    if target_mode == "round":
        agent.sync_target()
    bootstrap_summary = _summarize(bootstrap, len(train), bootstrap_errors, bootstrap_failed)
    bootstrap_summary['failure_reason_counts']=bootstrap_reason_counts
    print(f"bootstrap completed={len(bootstrap)}/{len(train)} economic_updates={agent.economic_optimizer_updates}", flush=True)
    history = []
    selected_steps = 0
    executed_training_steps = 0
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
        'redistribute_battery_energy': redistribute_battery_energy, 'n_step': n_step,
        'episode_credit_scope': episode_credit_scope,
        'required_split_sizes': list(required_split_sizes) if required_split_sizes is not None else None,
        'learn_no_feasible_failures':learn_no_feasible_failures,
        'failure_penalty':asdict(failure_penalty)|{'amount_equivalent_cny':failure_penalty.amount},
        'battery_energy_value': {
            'coefficient_cny': BatteryEnergyValue.from_accountant(accountant).coefficient_cny,
            'reference_soc': BatteryEnergyValue.from_accountant(accountant).reference_soc,
        },
    }
    source_commit = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    source_worktree_dirty = bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())

    def snapshot_report(completed_training=False):
        return {
            "completed_training": completed_training, "source_commit": source_commit,
            'source_worktree_dirty': source_worktree_dirty,
            "hyperparameters": hyperparameters, "bootstrap": bootstrap_summary,
            "bootstrap_optimizer_updates": len(bootstrap_losses),
            'formal_training_optimizer_updates': agent.economic_optimizer_updates-len(bootstrap_losses),
            'bootstrap_td_statistics': bootstrap_td_statistics,
            'initial_fixed_state_q_diagnostics': initial_q_diagnostics,
            'economic_replay_outcome_counts':replay_counts(),
            "bootstrap_economic_replay_insertions": sum(len(x.transitions) for x in bootstrap)+bootstrap_prefix,
            "rounds": history, "best_checkpoint": best_record,
            "first_qualification": first_qualification, "eligible_rounds": selector.eligible_rounds,
            "test_payloads_opened": dataset.opened_test_payloads,
            "execution_counts": {
                "training_environment_transitions": executed_training_steps,
                'training_policy_calls': selected_steps,
                "bootstrap_environment_transitions": bootstrap_summary["executed_onboard_transitions"],
                "greedy_evaluation_environment_transitions": evaluation_steps,
                "economic_replay_insertions": agent.economic_replay_insertions,
                "economic_optimizer_updates": agent.economic_optimizer_updates,
                "outcome_optimizer_updates": agent.outcome_optimizer_updates,
                "target_sync_calls_including_initial_copy": agent.target_sync_calls,
                "remaining_transition_credit": schedule.remaining_transition_credit,
                'bootstrap_economic_optimizer_updates': len(bootstrap_losses),
                'formal_training_economic_optimizer_updates': agent.economic_optimizer_updates-len(bootstrap_losses),
                'failed_suffix_transitions_excluded_from_economic_replay': failure_suffix_count,
                'no_feasible_action_events_without_executed_suffix': no_suffix_failure_events,
                'economic_success_replay_insertions':agent.economic_success_replay_insertions,
                'economic_failure_replay_insertions':agent.economic_failure_replay_insertions,
                'economic_failure_terminal_insertions':agent.economic_failure_terminal_insertions,
            },
            "elapsed_seconds": time.monotonic()-started,
            "cost_rule": "Actual plus modeled terminal settlement; SOC shaping excluded; incomplete split cost is null",
            "monitoring_rng_isolation": True,
            'failure_training_semantics': ('Genuine no-feasible-action suffixes enter economic Q with a tagged terminal, once-only training penalty and potential correction; ledgers unchanged'
                if learn_no_feasible_failures else 'Historical outcome-only unfinished suffix protocol'),
            'dataset_class': type(dataset).__name__,
        }

    for round_index in range(1, rounds+1):
        epsilon = epsilon_start if rounds == 1 else epsilon_end if round_index == rounds else (
            epsilon_start+(epsilon_end-epsilon_start)*(round_index-1)/(rounds-1)
        )
        print(f"round {round_index}/{rounds} start epsilon={epsilon:.4f}",flush=True)
        results, failures, failed_transitions, losses = [], [], [], []
        failure_profiles = []
        reason_counts={'soc_limited':0,'structural_power':0,'execution_error':0}
        completed_voyages = 0
        agent.reset_td_statistics()
        before_insertions = agent.economic_replay_insertions
        before_updates = agent.economic_optimizer_updates
        before_success=agent.economic_success_replay_insertions
        before_failure=agent.economic_failure_replay_insertions
        before_failure_terminals=agent.economic_failure_terminal_insertions
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
                result = replay_episode(episode,training_policy,accountant=accountant,beta_soc=beta_soc,
                                         redistribute_battery_energy=redistribute_battery_energy)
            except ReplayExecutionError as exc:
                failures.append(f"{episode.sample_id}: {exc}")
                cause=exc.failure_cause if exc.failure_kind=='no_feasible_action' else 'execution_error'
                reason_counts[cause]+=1
                view=failure_view(exc)
                transitions=view.transitions if view else exc.executed_transitions
                failed_transitions.extend(transitions)
                profile=trajectory_record(episode.sample_id,transitions,completed=False,failure=str(exc))
                profile.update(failure_cause=cause,failed_observed_state=exc.failed_state,
                               controllable_by_power_policy=False if cause=='structural_power' else None)
                failure_profiles.append(profile)
                prefix_voyages = 0
                if exc.failure_kind == 'no_feasible_action' and (
                        not exc.executed_transitions or exc.executed_transitions[-1].done):
                    no_suffix_failure_events += 1
                if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    if view is not None:
                        retained=agent.remember_trajectory(view.transitions)
                        prefix=exc.executed_transitions[:view.successful_prefix_transitions]
                    else:
                        retained=agent.remember_completed_prefix(exc.executed_transitions)
                        prefix=exc.executed_transitions[:retained]
                    prefix_voyages = sum(t.is_successful_terminal for t in prefix)
                    completed_voyages += prefix_voyages
                    failure_suffix_count += len(exc.executed_transitions)-retained
                    agent.remember_outcome_trajectory(exc.executed_transitions,
                                                      failed=not exc.executed_transitions[-1].done)
                schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,
                               completed_episodes=(prefix_voyages if episode_credit_scope == 'voyage' else 0))
                losses.extend(schedule.consume(agent,batch_size=batch_size))
                if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    for _ in range(16):
                        agent.learn_outcome(batch_size=batch_size)
                continue
            results.append(result)
            agent.remember_trajectory(result.transitions)
            voyages = sum(t.done for t in result.transitions)
            completed_voyages += voyages
            if result.transitions:
                agent.remember_outcome_trajectory(result.transitions,failed=False)
            schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,
                           completed_episodes=(voyages if episode_credit_scope == 'voyage' else 1))
            losses.extend(schedule.consume(agent,batch_size=batch_size))
            for _ in range(16):
                agent.learn_outcome(batch_size=batch_size)
        if target_mode == "round":
            agent.sync_target()
        explore = _summarize(results,len(train),failures,failed_transitions)
        explore['failure_reason_counts']=reason_counts
        executed_training_steps += explore['executed_onboard_transitions']
        if capture_trajectories:
            path = output_dir/f'round_{round_index:03d}_exploratory_trajectories.json'
            _write_json(path,{'completed':[trajectory_record(r.sample_id,r.transitions,completed=True) for r in results],
                              'failed':failure_profiles})
            explore['trajectory_file'] = path.name
        print(f"round {round_index}/{rounds} exploratory={len(results)}/{len(train)} greedy_train start",flush=True)
        greedy_train = greedy_evaluate(train,agent,accountant,beta_soc,
            redistribute_battery_energy=redistribute_battery_energy,
            trajectory_path=output_dir/f'round_{round_index:03d}_train_trajectories.json' if capture_trajectories else None,
            learn_no_feasible_failures=learn_no_feasible_failures,failure_penalty=failure_penalty)
        evaluation_steps += greedy_train["executed_onboard_transitions"]
        greedy_validation = None
        if fully_completed(greedy_train):
            print(f"round {round_index}/{rounds} greedy_train={len(train)}/{len(train)} validation start",flush=True)
            greedy_validation = greedy_evaluate(validation,agent,accountant,beta_soc,
                redistribute_battery_energy=redistribute_battery_energy,
                trajectory_path=output_dir/f'round_{round_index:03d}_validation_trajectories.json' if capture_trajectories else None,
                learn_no_feasible_failures=learn_no_feasible_failures,failure_penalty=failure_penalty)
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
            "training_environment_transitions_cumulative":executed_training_steps,
            'training_policy_calls_cumulative': selected_steps,
            "target_sync_calls_cumulative":agent.target_sync_calls,
            "remaining_transition_credit":schedule.remaining_transition_credit,
            'td_statistics': agent.td_statistics(),
            'fixed_state_q_diagnostics': fixed_q_diagnostics(agent,accountant),
            'completed_voyages_entering_economic_replay': completed_voyages,
            'failed_sample_count':len(failures),'failure_reason_counts':reason_counts,
            'economic_success_replay_insertions':agent.economic_success_replay_insertions-before_success,
            'economic_failure_replay_insertions':agent.economic_failure_replay_insertions-before_failure,
            'economic_failure_terminal_insertions':agent.economic_failure_terminal_insertions-before_failure_terminals,
            'economic_replay_outcome_counts_cumulative':replay_counts(),
        }
        improved = selector.consider(round_index,greedy_train,greedy_validation)
        row["qualified_checkpoint"] = fully_completed(greedy_train) and fully_completed(greedy_validation)
        row["best_checkpoint_improved"] = improved
        if row["qualified_checkpoint"] and first_qualification is None:
            first_qualification = {
                "round":round_index,"economic_optimizer_updates":agent.economic_optimizer_updates,
                "environment_transitions_including_bootstrap":executed_training_steps+bootstrap_summary["executed_onboard_transitions"],
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
