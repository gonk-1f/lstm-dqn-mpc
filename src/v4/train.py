"""Legacy experimental runner and shared v4 evaluation helpers.

This CLI is NOT the current formal 100-round configuration. Use
``python -m v4.feedback_study`` for formal runs. Historical defaults remain
only for legacy callers; _default_data_root, _run_episodes and _summarize
are also dependencies of the formal monitored training path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .control import (
    ACTION_KW, DEFAULT_STATE_MODE, SOC_REF, DirectReplay, DirectTransition,
    ReplayExecutionError, replay_episode, state_feature_value,
)
from .dqn import (
    DirectPowerDDQN, ECONOMIC_HIDDEN_DIMS, ECONOMIC_LEARNING_RATE, STATE_DIM, STATE_FEATURES,
)
from .telemetry import duration_weighted_mean, soc_time_occupancy
from .diagnostics import COMPONENT_NAMES, reward_totals
from .experiment_schedule import annotate_evaluation, fully_completed
from .independent_economics import EVALUATION_VERSION, ZhouEconomicEvaluation


def _summarize(
    results: list[DirectReplay], requested: int, failures: list[str],
    failed_transitions: Sequence[DirectTransition] = (),
    *, task_mode: str = 'legacy',
) -> dict[str, object]:
    if task_mode not in ('legacy', 'independent_voyage'):
        raise ValueError('summary task_mode must be legacy or independent_voyage')
    if any(getattr(item, 'task_mode', 'legacy') != task_mode for item in results):
        raise ValueError('completed replay task modes differ from the summary task')
    independent = task_mode == 'independent_voyage'
    if independent and any(
            not isinstance(item.economic_evaluation, ZhouEconomicEvaluation)
            or item.economic_evaluation.evaluation_version != EVALUATION_VERSION for item in results):
        raise ValueError('independent completed replays require matching Zhou economic evaluations')
    starts = sum(
        sum(left == 0 and right > 0 for left, right in zip((0.0, *item.fc_power_kw_by_row[:-1]), item.fc_power_kw_by_row))
        for item in results
    )
    soc = [value for item in results for value in item.soc_by_row]
    onboard_soc = [transition.actual_soc for item in results for transition in item.transitions]
    all_onboard_soc = [*onboard_soc, *(transition.actual_soc for transition in failed_transitions)]
    all_fc = [transition.action_kw for item in results for transition in item.transitions]
    all_fc.extend(transition.action_kw for transition in failed_transitions)
    terminal_onboard_soc = [item.transitions[-1].actual_soc for item in results if item.transitions]
    observed_cost = sum(item.total_cost_cny for item in results)
    modeled_shore_ledgers = [block.ledger for result in results for block in result.shore_blocks
                             if block.settlement_basis == 'modeled_fixed_target_soc_0.6']
    modeled_shore_cost = math.fsum(ledger.total_cost_cny for ledger in modeled_shore_ledgers)
    modeled_cost = sum(
        0.0 if item.modeled_terminal_settlement is None
        else item.modeled_terminal_settlement.ledger.total_cost_cny
        for item in results
    )
    comparable_cost = (math.fsum(item.comparable_cost_cny for item in results)
                       if independent else observed_cost + modeled_cost)
    completed_transitions = [t for item in results for t in item.transitions]
    executed_transitions = [*completed_transitions, *failed_transitions]
    executed_durations = [getattr(item, 'duration_s', 30.0) for item in executed_transitions]
    voyage_terminal_soc = [t.actual_soc for t in executed_transitions
                           if t.shore_ledger is not None or t.is_successful_terminal]
    completed_onboard_segments = (sum(t.shore_ledger is not None for t in executed_transitions)
                                  + sum(t.is_successful_terminal and t.shore_ledger is None
                                        for t in executed_transitions))
    executed_components = {
        name: math.fsum(getattr(item.total_ledger,name) for item in results)
              + math.fsum(getattr(t.original_economic_ledger,name) for t in failed_transitions)
        for name in COMPONENT_NAMES
    }
    executed_modeled_shore_ledgers = [*modeled_shore_ledgers,
                                     *(t.shore_ledger for t in failed_transitions
                                       if t.shore_ledger is not None)]
    executed_observed_components = {
        name: value - math.fsum(getattr(ledger, name) for ledger in executed_modeled_shore_ledgers)
        for name, value in executed_components.items()
    }
    summary = {
        "episodes": requested,
        "completed": len(results),
        "failed": failures,
        "cost_cny": comparable_cost if not failures and (not independent or len(results) == requested) else None,
        "completed_cost_cny": comparable_cost,
        "completed_observed_cost_cny": observed_cost - modeled_shore_cost,
        "completed_accounted_cost_including_modeled_shore_cny": observed_cost,
        "completed_modeled_fixed_target_shore_cost_cny": modeled_shore_cost,
        "completed_modeled_fixed_target_shore_components_cny": {
            name: math.fsum(getattr(ledger, name) for ledger in modeled_shore_ledgers)
            for name in COMPONENT_NAMES
        },
        'executed_observed_components_cny': executed_observed_components,
        'executed_observed_cost_cny': math.fsum(executed_observed_components.values()),
        'executed_accounted_components_cny': executed_components,
        'executed_accounted_cost_cny': math.fsum(executed_components.values()),
        'executed_modeled_fixed_target_shore_cost_cny': math.fsum(
            ledger.total_cost_cny for ledger in executed_modeled_shore_ledgers),
        'executed_cost_scope': 'Completed sample ledgers plus failed executed transitions; modeled fixed-target SHORE shown separately; unattached leading SHORE in failed samples is unavailable',
        "completed_observed_components_cny": {
            name: math.fsum(getattr(item.total_ledger, name) for item in results)
                  - math.fsum(getattr(ledger, name) for ledger in modeled_shore_ledgers)
            for name in COMPONENT_NAMES
        },
        "completed_modeled_components_cny": {
            name: math.fsum(getattr(item.modeled_terminal_settlement.ledger, name)
                           for item in results if item.modeled_terminal_settlement is not None)
            for name in COMPONENT_NAMES
        },
        "completed_reward_feedback": reward_totals(completed_transitions),
        "executed_reward_feedback_including_failed_prefix": reward_totals(executed_transitions),
        "completed_modeled_terminal_cost_cny": modeled_cost,
        "completed_soc_soft_penalty_cny": sum(item.soc_soft_penalty_cny for item in results),
        "transitions": sum(len(item.transitions) for item in results),
        "executed_onboard_transitions": len(all_onboard_soc),
        "executed_soc_soft_penalty_cny": (
            sum(item.soc_soft_penalty_cny for item in results)
            + sum(item.soc_soft_penalty_cny for item in failed_transitions)
        ),
        "soc_time_occupancy": soc_time_occupancy(
            all_onboard_soc, durations_s=executed_durations if independent else None),
        "onboard_soc_mean": duration_weighted_mean(all_onboard_soc, executed_durations),
        "fc_zero_fraction": duration_weighted_mean((float(value == 0) for value in all_fc), executed_durations),
        "fc_starts": starts,
        "executed_fc_starts_including_failed_prefix": sum(
            state_feature_value(t.state, 'previous_fc_kw_over_600') == 0 and t.action_kw > 0
            for t in executed_transitions),
        "completed_voyages": completed_onboard_segments,
        "completed_onboard_segments": completed_onboard_segments,
        'failed_samples':len(failures),
        'failure_terminal_count':sum(t.done and not t.is_successful_terminal for t in failed_transitions),
        'voyage_terminal_soc_mean': math.fsum(voyage_terminal_soc)/len(voyage_terminal_soc) if voyage_terminal_soc else None,
        'voyage_terminal_soc_min': min(voyage_terminal_soc) if voyage_terminal_soc else None,
        'voyage_terminal_soc_max': max(voyage_terminal_soc) if voyage_terminal_soc else None,
        "soc_min": min(soc) if soc else None,
        "soc_max": max(soc) if soc else None,
        "onboard_soc_min": min(onboard_soc) if onboard_soc else None,
        "onboard_soc_max": max(onboard_soc) if onboard_soc else None,
        "onboard_soc_min_including_failed_prefix": min(all_onboard_soc) if all_onboard_soc else None,
        "terminal_onboard_soc_mean": (
            sum(terminal_onboard_soc) / len(terminal_onboard_soc) if terminal_onboard_soc else None
        ),
        "terminal_onboard_soc_min": min(terminal_onboard_soc) if terminal_onboard_soc else None,
        "terminal_onboard_soc_max": max(terminal_onboard_soc) if terminal_onboard_soc else None,
    }
    if independent:
        evaluations = [item.economic_evaluation for item in results]
        summary.update(
            task_mode=task_mode,
            economic_evaluation_basis=EVALUATION_VERSION,
            completed_three_cost_components_cny={
                name: math.fsum(getattr(evaluation, name) for evaluation in evaluations)
                for name in ('equivalent_hydrogen_cost_cny', 'fuel_cell_degradation_cost_cny',
                             'battery_degradation_cost_cny')},
            completed_hydrogen_quantities_kg={
                'actual': math.fsum(item.actual_hydrogen_mass_kg for item in evaluations),
                'soc_signed': math.fsum(item.soc_equivalent_hydrogen_mass_kg for item in evaluations),
                'equivalent': math.fsum(item.equivalent_hydrogen_mass_kg for item in evaluations),
            },
            executed_onboard_duration_s=math.fsum(executed_durations),
        )
    return summary


def _run_episodes(episodes, policy, accountant, *, on_failure=None, beta_soc: float = 0.0,
                  redistribute_battery_energy: bool = False,
                  reward_cost_mode: str = 'full_economic', state_mode: str = DEFAULT_STATE_MODE,
                  action_mode: str = 'absolute61',
                  economic_evaluation_config=None):
    results: list[DirectReplay] = []
    failures: list[str] = []
    for episode in episodes:
        try:
            results.append(replay_episode(episode, policy, accountant=accountant, beta_soc=beta_soc,
                                          redistribute_battery_energy=redistribute_battery_energy,
                                          reward_cost_mode=reward_cost_mode, state_mode=state_mode,
                                          action_mode=action_mode,
                                          economic_evaluation_config=economic_evaluation_config))
        except ReplayExecutionError as exc:
            exc.sample_id = str(episode.sample_id)
            failures.append(f"{episode.sample_id}: {exc}")
            if on_failure is not None:
                on_failure(exc)
    return results, failures


def run_train_validation(
    dataset: FormalTrainingDataset, *, rounds: int = 1, seed: int = 42,
    batch_size: int = 64, updates_per_episode: int = 16,
    max_train_episodes: int | None = None,
    max_validation_episodes: int | None = None,
    progress_every_steps: int = 0,
    epsilon_start: float = 0.15, epsilon_end: float = 0.02,
    beta_soc: float = 0.0,
) -> tuple[DirectPowerDDQN, dict[str, object]]:
    """Fit off-policy from causal load-following rollouts, then DQN rollouts.

    Failed physical episodes are reported and excluded from cost comparisons.
    A checkpoint is eligible only if every Validation episode completes.
    """
    if min(rounds, batch_size, updates_per_episode) < 1:
        raise ValueError("rounds, batch size, and updates per episode must be positive")
    if max_train_episodes is not None and max_train_episodes < 1:
        raise ValueError("max_train_episodes must be positive")
    if max_validation_episodes is not None and max_validation_episodes < 1:
        raise ValueError("max_validation_episodes must be positive")
    if type(progress_every_steps) is not int or progress_every_steps < 0:
        raise ValueError("progress_every_steps must be a nonnegative integer")
    if not (math.isfinite(epsilon_start) and math.isfinite(epsilon_end)
            and 0.0 <= epsilon_end <= epsilon_start <= 1.0):
        raise ValueError("epsilon schedule must satisfy 0 <= end <= start <= 1")
    if not math.isfinite(beta_soc) or beta_soc < 0.0:
        raise ValueError("beta_soc must be finite and nonnegative")
    before_test = dataset.opened_test_payloads
    train = dataset.load_train()
    validation = dataset.load_validation()
    if not train or not validation:
        raise ValueError("Train and Validation must each contain episodes")
    if any(getattr(item, "split", None) != "train" for item in train):
        raise ValueError("Train loader returned a non-Train episode")
    if any(getattr(item, "split", None) != "validation" for item in validation):
        raise ValueError("Validation loader returned a non-Validation episode")
    train = train[:max_train_episodes]
    validation = validation[:max_validation_episodes]
    from .monitored_training import greedy_evaluate
    accountant = EconomicMPC(nominal_cost_cny=1.0)  # only interval accounting; MPC solve is never called
    agent = DirectPowerDDQN(seed=seed)
    bootstrap_failed_prefixes = 0
    bootstrap_completed_prefixes = 0
    bootstrap_failed_transitions: list[DirectTransition] = []
    bootstrap_events: list[ReplayExecutionError] = []
    bootstrap_unknown_discarded = 0

    def remember_bootstrap_failure(exc: ReplayExecutionError) -> None:
        nonlocal bootstrap_failed_prefixes, bootstrap_completed_prefixes, bootstrap_unknown_discarded
        bootstrap_events.append(exc)
        bootstrap_failed_transitions.extend(exc.executed_transitions)
        if exc.failure_kind == "data_truncation":
            retained = agent.remember_completed_prefix(exc.executed_transitions)
            bootstrap_unknown_discarded += len(exc.executed_transitions) - retained
        elif exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
            bootstrap_completed_prefixes += agent.remember_completed_prefix(exc.executed_transitions)
            bootstrap_failed_prefixes += len(exc.executed_transitions)

    if progress_every_steps:
        print(f"bootstrap start episodes={len(train)}", flush=True)
    bootstrap, bootstrap_failures = _run_episodes(
        train,
        lambda state, feasible: min(feasible, key=lambda power: abs(
            power - state_feature_value(state, 'current_load_kw_over_600') * 600.0)),
        accountant,
        on_failure=remember_bootstrap_failure,
        beta_soc=beta_soc,
    )
    if progress_every_steps:
        print(f"bootstrap completed={len(bootstrap)}/{len(train)} failed={len(bootstrap_failures)}", flush=True)
    for result in bootstrap:
        for transition in result.transitions:
            agent.remember_transition(transition)
    bootstrap_losses = [
        loss for _ in range(updates_per_episode * len(bootstrap))
        if (loss := agent.learn(batch_size=batch_size)) is not None
    ]
    agent.sync_target()
    training_rounds: list[dict[str, object]] = []
    selected_steps = 0
    for round_index in range(rounds):
        epsilon = (
            epsilon_start if rounds == 1
            else epsilon_end if round_index == rounds - 1
            else epsilon_start + (epsilon_end - epsilon_start) * round_index / (rounds - 1)
        )
        if progress_every_steps:
            print(f"round {round_index + 1}/{rounds} start epsilon={epsilon:.4f}", flush=True)
        completed: list[DirectReplay] = []
        failures: list[str] = []
        losses: list[float] = []
        failed_prefix_transitions = 0
        completed_prefix_transitions = 0
        round_failed_transitions: list[DirectTransition] = []
        round_events: list[ReplayExecutionError] = []
        round_unknown_discarded = 0
        for episode_index, episode in enumerate(train, start=1):
            episode_steps = 0

            def training_policy(state, feasible):
                nonlocal selected_steps, episode_steps
                action = agent.select_power(state, feasible, epsilon=epsilon)
                selected_steps += 1
                episode_steps += 1
                if progress_every_steps:
                    if selected_steps % progress_every_steps == 0:
                        load_kw = (state_feature_value(state, 'current_load_kw_over_600')
                                   * accountant.plant.fuel_cell_rated_total_kw)
                        print(
                            f"progress step={selected_steps} phase=train round={round_index + 1}/{rounds} "
                            f"episode={episode_index}/{len(train)} sample={episode.sample_id} "
                            f"onboard_step={episode_steps} load_kw={load_kw:.1f} "
                            f"soc_before={state[0]:.4f} selected_fc_kw={action}",
                            flush=True,
                        )
                return action

            try:
                result = replay_episode(
                    episode,
                    training_policy,
                    accountant=accountant,
                    beta_soc=beta_soc,
                )
            except ReplayExecutionError as exc:
                exc.sample_id = str(episode.sample_id)
                round_events.append(exc)
                failures.append(f"{episode.sample_id}: {exc}")
                round_failed_transitions.extend(exc.executed_transitions)
                if exc.failure_kind == "data_truncation":
                    retained = agent.remember_completed_prefix(exc.executed_transitions)
                    round_unknown_discarded += len(exc.executed_transitions) - retained
                elif exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    completed_prefix_transitions += agent.remember_completed_prefix(exc.executed_transitions)
                    failed_prefix_transitions += len(exc.executed_transitions)
                if progress_every_steps:
                    print(f"failure phase=train round={round_index + 1}/{rounds} sample={episode.sample_id} {exc}", flush=True)
                continue
            completed.append(result)
            for transition in result.transitions:
                agent.remember_transition(transition)
            for _ in range(updates_per_episode):
                loss = agent.learn(batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)
        agent.sync_target()
        summary = annotate_evaluation(
            _summarize(completed, len(train), failures, round_failed_transitions),
            train, completed, round_events)
        summary.update({
            "round": round_index + 1, "epsilon": epsilon,
            "mean_loss": sum(losses) / len(losses) if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "economic_q_optimizer_updates": len(losses),
            "economic_q_optimizer_updates_cumulative": agent.economic_optimizer_updates,
            "target_sync_calls_cumulative": agent.target_sync_calls,
            "failed_prefix_transitions": failed_prefix_transitions,
            "completed_prefix_transitions": completed_prefix_transitions,
            "unknown_discarded_economic_q_transitions": round_unknown_discarded,
        })
        summary["greedy_train"] = greedy_evaluate(train, agent, accountant, beta_soc)
        summary["greedy_validation"] = greedy_evaluate(validation, agent, accountant, beta_soc)
        training_rounds.append(summary)
        if progress_every_steps:
            print(
                f"round {round_index + 1}/{rounds} completed={len(completed)}/{len(train)} "
                f"failed={len(failures)} selected_steps={selected_steps} "
                f"mean_loss={summary['mean_loss']}",
                flush=True,
            )
    if dataset.opened_test_payloads != before_test or before_test != 0:
        raise RuntimeError("Test payload was opened during Train/Validation selection")
    last_executed_onboard_train = sum(item.operating_mode[-1] == "onboard" for item in train)
    last_executed_onboard_validation = sum(item.operating_mode[-1] == "onboard" for item in validation)
    train_greedy_summary = training_rounds[-1]["greedy_train"]
    validation_summary = training_rounds[-1]["greedy_validation"]
    ineligibility_reasons: list[str] = []
    if not agent.replay:
        ineligibility_reasons.append("no_training_experience")
    if not fully_completed(train_greedy_summary):
        ineligibility_reasons.append("train_greedy_incomplete")
    if not fully_completed(validation_summary):
        ineligibility_reasons.append("validation_incomplete")
    report: dict[str, object] = {
        "controller": "MLP_Double_DQN_direct_FC_power",
        "sample_seconds": 30,
        "action_kw": list(ACTION_KW),
        "state_dim": STATE_DIM,
        "state_features": list(STATE_FEATURES),
        "reward_definition": "negative_four_component_cost_with_existing_settlement_minus_beta_soc_times_post_action_soc_reference_deviation_squared",
        "soc_penalty_units": "auxiliary reward-equivalent units; not actual CNY expenditure",
        "bootstrap": annotate_evaluation(
            _summarize(bootstrap, len(train), bootstrap_failures, bootstrap_failed_transitions),
            train, bootstrap, bootstrap_events),
        "bootstrap_unknown_discarded_economic_q_transitions": bootstrap_unknown_discarded,
        "bootstrap_economic_q_optimizer_updates": len(bootstrap_losses),
        "bootstrap_mean_loss": sum(bootstrap_losses) / len(bootstrap_losses) if bootstrap_losses else None,
        "bootstrap_failed_prefix_transitions": bootstrap_failed_prefixes,
        "bootstrap_completed_prefix_transitions": bootstrap_completed_prefixes,
        "training_rounds": training_rounds,
        "train_greedy_evaluation": train_greedy_summary,
        "validation": validation_summary,
        "execution_counts": {
            "training_onboard_selected_actions": selected_steps,
            "training_onboard_executed_transitions": sum(
                row["executed_onboard_transitions"] for row in training_rounds
            ),
            "bootstrap_onboard_executed_transitions": (
                sum(len(item.transitions) for item in bootstrap) + len(bootstrap_failed_transitions)
            ),
            "greedy_evaluation_onboard_executed_transitions": (
                train_greedy_summary["executed_onboard_transitions"]
                + validation_summary["executed_onboard_transitions"]
            ),
            "economic_q_optimizer_updates": agent.economic_optimizer_updates,
            "training_economic_q_optimizer_updates": agent.economic_optimizer_updates - len(bootstrap_losses),
            "target_sync_calls_including_initial_copy": agent.target_sync_calls,
            "target_sync_calls_after_initialization": agent.target_sync_calls - 1,
            "scope": "ONBOARD decision transitions; excludes SHORE accounting and virtual modeled charging",
        },
        "last_executed_onboard_episodes": {
            "train": last_executed_onboard_train, "validation": last_executed_onboard_validation,
        },
        "selection_eligible": not ineligibility_reasons,
        "selection_ineligibility_reasons": ineligibility_reasons,
        "test_payloads_opened": dataset.opened_test_payloads,
        "hyperparameters": {
            "seed": seed, "rounds": rounds, "batch_size": batch_size,
            "updates_per_episode": updates_per_episode, "gamma": agent.gamma,
            "learning_rate": ECONOMIC_LEARNING_RATE, "replay_capacity": 100_000,
            "hidden_dims": list(ECONOMIC_HIDDEN_DIMS),
            "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
            "beta_soc": beta_soc,
            "soc_ref": SOC_REF,
            "target_sync_schedule": "initial hard copy; once after bootstrap; once after each round",
            "economic_update_schedule": "16 by default per completed sample, unchanged",
        },
    }
    return agent, report


def _default_data_root(name: str) -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "processed" / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"cannot locate data/processed/{name}; pass an explicit dataset root")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-root", type=Path)
    parser.add_argument("--ais-root", type=Path)
    parser.add_argument("--mode-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v4_direct_power"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-episode", type=int, default=16)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-validation-episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every-steps", type=int, default=50)
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--beta-soc", type=float, default=0.0)
    args = parser.parse_args(argv)
    power = args.power_root or _default_data_root("operating_dataset_zero_boundary_v2")
    ais = args.ais_root or _default_data_root("operating_dataset_zero_boundary_v2_ais")
    mode = args.mode_root or _default_data_root("operating_dataset_zero_boundary_v2_modes_v3")
    dataset = FormalTrainingDataset.open(power, ais, mode)
    agent, report = run_train_validation(
        dataset, rounds=args.rounds, batch_size=args.batch_size,
        updates_per_episode=args.updates_per_episode,
        max_train_episodes=args.max_train_episodes,
        max_validation_episodes=args.max_validation_episodes, seed=args.seed,
        progress_every_steps=args.progress_every_steps,
        epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
        beta_soc=args.beta_soc,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if report["selection_eligible"]:
        torch.save({
            "architecture": "v4_mlp_double_dqn_direct_power",
            "model_state": agent.online.state_dict(),
            "state_dim": STATE_DIM, "action_kw": ACTION_KW,
            "hyperparameters": report["hyperparameters"],
        }, args.output_dir / "selected_agent.pt")
    else:
        (args.output_dir / "selected_agent.pt").unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["selection_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
