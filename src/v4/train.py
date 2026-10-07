"""Train and validate a direct-power MLP Double-DQN without opening Test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import torch

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .control import ACTION_KW, DirectReplay, DirectTransition, ReplayExecutionError, replay_episode
from .dqn import DirectPowerDDQN
from .telemetry import soc_time_occupancy


def _summarize(
    results: list[DirectReplay], requested: int, failures: list[str],
    failed_transitions: Sequence[DirectTransition] = (),
) -> dict[str, object]:
    starts = sum(
        sum(left == 0 and right > 0 for left, right in zip((0.0, *item.fc_power_kw_by_row[:-1]), item.fc_power_kw_by_row))
        for item in results
    )
    soc = [value for item in results for value in item.soc_by_row]
    onboard_soc = [transition.actual_soc for item in results for transition in item.transitions]
    all_onboard_soc = [*onboard_soc, *(transition.actual_soc for transition in failed_transitions)]
    terminal_onboard_soc = [item.transitions[-1].actual_soc for item in results if item.transitions]
    observed_cost = sum(item.total_cost_cny for item in results)
    modeled_cost = sum(
        0.0 if item.modeled_terminal_settlement is None
        else item.modeled_terminal_settlement.ledger.total_cost_cny
        for item in results
    )
    comparable_cost = observed_cost + modeled_cost
    return {
        "episodes": requested,
        "completed": len(results),
        "failed": failures,
        "cost_cny": comparable_cost if not failures else None,
        "completed_cost_cny": comparable_cost,
        "completed_observed_cost_cny": observed_cost,
        "completed_modeled_terminal_cost_cny": modeled_cost,
        "completed_soc_soft_penalty_cny": sum(item.soc_soft_penalty_cny for item in results),
        "transitions": sum(len(item.transitions) for item in results),
        "executed_onboard_transitions": len(all_onboard_soc),
        "executed_soc_soft_penalty_cny": (
            sum(item.soc_soft_penalty_cny for item in results)
            + sum(item.soc_soft_penalty_cny for item in failed_transitions)
        ),
        "soc_time_occupancy": soc_time_occupancy(all_onboard_soc),
        "fc_starts": starts,
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


def _run_episodes(episodes, policy, accountant, *, on_failure=None, beta_soc: float = 0.0):
    results: list[DirectReplay] = []
    failures: list[str] = []
    for episode in episodes:
        try:
            results.append(replay_episode(episode, policy, accountant=accountant, beta_soc=beta_soc))
        except ReplayExecutionError as exc:
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
    accountant = EconomicMPC(nominal_cost_cny=1.0)  # only interval accounting; MPC solve is never called
    agent = DirectPowerDDQN(seed=seed)
    bootstrap_failed_prefixes = 0
    bootstrap_completed_prefixes = 0
    bootstrap_failed_transitions: list[DirectTransition] = []

    def remember_bootstrap_failure(exc: ReplayExecutionError) -> None:
        nonlocal bootstrap_failed_prefixes, bootstrap_completed_prefixes
        bootstrap_failed_transitions.extend(exc.executed_transitions)
        if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
            bootstrap_completed_prefixes += agent.remember_completed_prefix(exc.executed_transitions)
            agent.remember_outcome_trajectory(exc.executed_transitions, failed=True)
            bootstrap_failed_prefixes += len(exc.executed_transitions)

    if progress_every_steps:
        print(f"bootstrap start episodes={len(train)}", flush=True)
    bootstrap, bootstrap_failures = _run_episodes(
        train,
        lambda state, feasible: min(feasible, key=lambda power: abs(power - state[1] * 600.0)),
        accountant,
        on_failure=remember_bootstrap_failure,
        beta_soc=beta_soc,
    )
    if progress_every_steps:
        print(f"bootstrap completed={len(bootstrap)}/{len(train)} failed={len(bootstrap_failures)}", flush=True)
    for result in bootstrap:
        for transition in result.transitions:
            agent.remember_transition(transition)
        if result.transitions:
            agent.remember_outcome_trajectory(result.transitions, failed=False)
    bootstrap_losses = [
        loss for _ in range(updates_per_episode * len(bootstrap))
        if (loss := agent.learn(batch_size=batch_size)) is not None
    ]
    bootstrap_outcome_losses = [
        loss for _ in range(updates_per_episode * (len(bootstrap) + len(bootstrap_failures)))
        if (loss := agent.learn_outcome(batch_size=batch_size)) is not None
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
        outcome_losses: list[float] = []
        failed_prefix_transitions = 0
        completed_prefix_transitions = 0
        round_failed_transitions: list[DirectTransition] = []
        for episode_index, episode in enumerate(train, start=1):
            episode_steps = 0

            def training_policy(state, feasible):
                nonlocal selected_steps, episode_steps
                action = agent.select_power(state, feasible, epsilon=epsilon)
                selected_steps += 1
                episode_steps += 1
                if progress_every_steps:
                    if selected_steps % progress_every_steps == 0:
                        load_kw = state[1] * accountant.plant.fuel_cell_rated_total_kw
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
                failures.append(f"{episode.sample_id}: {exc}")
                round_failed_transitions.extend(exc.executed_transitions)
                if exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    completed_prefix_transitions += agent.remember_completed_prefix(exc.executed_transitions)
                    agent.remember_outcome_trajectory(exc.executed_transitions, failed=True)
                    failed_prefix_transitions += len(exc.executed_transitions)
                    for _ in range(updates_per_episode):
                        loss = agent.learn_outcome(batch_size=batch_size)
                        if loss is not None:
                            outcome_losses.append(loss)
                if progress_every_steps:
                    print(f"failure phase=train round={round_index + 1}/{rounds} sample={episode.sample_id} {exc}", flush=True)
                continue
            completed.append(result)
            for transition in result.transitions:
                agent.remember_transition(transition)
            if result.transitions:
                agent.remember_outcome_trajectory(result.transitions, failed=False)
            for _ in range(updates_per_episode):
                loss = agent.learn(batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)
                outcome_loss = agent.learn_outcome(batch_size=batch_size)
                if outcome_loss is not None:
                    outcome_losses.append(outcome_loss)
        agent.sync_target()
        summary = _summarize(completed, len(train), failures, round_failed_transitions)
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
            "outcome_replay_size": len(agent.outcome_replay),
            "outcome_training_updates": len(outcome_losses),
            "outcome_mean_loss": sum(outcome_losses) / len(outcome_losses) if outcome_losses else None,
        })
        training_rounds.append(summary)
        if progress_every_steps:
            print(
                f"round {round_index + 1}/{rounds} completed={len(completed)}/{len(train)} "
                f"failed={len(failures)} selected_steps={selected_steps} "
                f"mean_loss={summary['mean_loss']}",
                flush=True,
            )
    if progress_every_steps:
        print(f"train greedy evaluation start episodes={len(train)} epsilon=0", flush=True)
    train_greedy_failed_transitions: list[DirectTransition] = []
    train_greedy_results, train_greedy_failures = _run_episodes(
        train,
        lambda state, feasible: agent.select_power(state, feasible),
        accountant,
        beta_soc=beta_soc,
        on_failure=lambda exc: train_greedy_failed_transitions.extend(exc.executed_transitions),
    )
    if progress_every_steps:
        print(
            f"train greedy evaluation completed={len(train_greedy_results)}/{len(train)} "
            f"failed={len(train_greedy_failures)}", flush=True,
        )
        print(f"validation start episodes={len(validation)} epsilon=0", flush=True)
    validation_failed_transitions: list[DirectTransition] = []
    validation_results, validation_failures = _run_episodes(
        validation,
        lambda state, feasible: agent.select_power(state, feasible),
        accountant,
        beta_soc=beta_soc,
        on_failure=lambda exc: validation_failed_transitions.extend(exc.executed_transitions),
    )
    if progress_every_steps:
        print(f"validation completed={len(validation_results)}/{len(validation)} failed={len(validation_failures)}", flush=True)
    if dataset.opened_test_payloads != before_test or before_test != 0:
        raise RuntimeError("Test payload was opened during Train/Validation selection")
    last_executed_onboard_train = sum(item.operating_mode[-1] == "onboard" for item in train)
    last_executed_onboard_validation = sum(item.operating_mode[-1] == "onboard" for item in validation)
    ineligibility_reasons: list[str] = []
    if not agent.replay:
        ineligibility_reasons.append("no_training_experience")
    if train_greedy_failures or len(train_greedy_results) != len(train):
        ineligibility_reasons.append("train_greedy_incomplete")
    if validation_failures or not validation_results:
        ineligibility_reasons.append("validation_incomplete")
    report: dict[str, object] = {
        "controller": "MLP_Double_DQN_direct_FC_power",
        "sample_seconds": 30,
        "action_kw": list(ACTION_KW),
        "state_dim": 8,
        "reward_definition": "negative_observed_plus_modeled_terminal_four_component_CNY_minus_beta_soc_phi",
        "bootstrap": _summarize(bootstrap, len(train), bootstrap_failures, bootstrap_failed_transitions),
        "bootstrap_economic_q_optimizer_updates": len(bootstrap_losses),
        "bootstrap_mean_loss": sum(bootstrap_losses) / len(bootstrap_losses) if bootstrap_losses else None,
        "bootstrap_failed_prefix_transitions": bootstrap_failed_prefixes,
        "bootstrap_completed_prefix_transitions": bootstrap_completed_prefixes,
        "bootstrap_outcome_training_updates": len(bootstrap_outcome_losses),
        "training_rounds": training_rounds,
        "train_greedy_evaluation": _summarize(
            train_greedy_results, len(train), train_greedy_failures, train_greedy_failed_transitions,
        ),
        "validation": _summarize(
            validation_results, len(validation), validation_failures, validation_failed_transitions,
        ),
        "execution_counts": {
            "training_onboard_selected_actions": selected_steps,
            "training_onboard_executed_transitions": sum(
                row["executed_onboard_transitions"] for row in training_rounds
            ),
            "bootstrap_onboard_executed_transitions": (
                sum(len(item.transitions) for item in bootstrap) + len(bootstrap_failed_transitions)
            ),
            "greedy_evaluation_onboard_executed_transitions": (
                sum(len(item.transitions) for item in (*train_greedy_results, *validation_results))
                + len(train_greedy_failed_transitions) + len(validation_failed_transitions)
            ),
            "economic_q_optimizer_updates": agent.economic_optimizer_updates,
            "training_economic_q_optimizer_updates": agent.economic_optimizer_updates - len(bootstrap_losses),
            "outcome_optimizer_updates": agent.outcome_optimizer_updates,
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
        "outcome_model_role": "diagnostic_observed_rollout_failure_only_no_action_filter",
        "hyperparameters": {
            "seed": seed, "rounds": rounds, "batch_size": batch_size,
            "updates_per_episode": updates_per_episode, "gamma": agent.gamma,
            "learning_rate": 1e-4, "replay_capacity": 100_000,
            "hidden_dims": [128, 64], "epsilon_start": epsilon_start, "epsilon_end": epsilon_end,
            "beta_soc": beta_soc,
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
    mode = args.mode_root or _default_data_root("operating_dataset_zero_boundary_v2_modes")
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
            "state_dim": 8, "action_kw": ACTION_KW,
            "hyperparameters": report["hyperparameters"],
        }, args.output_dir / "selected_agent.pt")
    else:
        (args.output_dir / "selected_agent.pt").unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["selection_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
