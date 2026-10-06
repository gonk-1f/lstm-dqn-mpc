"""Train and validate a direct-power MLP Double-DQN without opening Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .control import ACTION_KW, DirectReplay, ReplayExecutionError, replay_episode
from .dqn import DirectPowerDDQN


def _summarize(results: list[DirectReplay], requested: int, failures: list[str]) -> dict[str, object]:
    starts = sum(
        sum(left == 0 and right > 0 for left, right in zip((0.0, *item.fc_power_kw_by_row[:-1]), item.fc_power_kw_by_row))
        for item in results
    )
    soc = [value for item in results for value in item.soc_by_row]
    completed_cost = sum(item.total_cost_cny for item in results)
    return {
        "episodes": requested,
        "completed": len(results),
        "failed": failures,
        "cost_cny": completed_cost if not failures else None,
        "completed_cost_cny": completed_cost,
        "transitions": sum(len(item.transitions) for item in results),
        "fc_starts": starts,
        "soc_min": min(soc) if soc else None,
        "soc_max": max(soc) if soc else None,
    }


def _run_episodes(episodes, policy, accountant):
    results: list[DirectReplay] = []
    failures: list[str] = []
    for episode in episodes:
        try:
            results.append(replay_episode(episode, policy, accountant=accountant))
        except ReplayExecutionError as exc:
            failures.append(f"{episode.sample_id}: {exc}")
    return results, failures


def run_train_validation(
    dataset: FormalTrainingDataset, *, rounds: int = 1, seed: int = 42,
    batch_size: int = 64, updates_per_episode: int = 16,
    max_train_episodes: int | None = None,
    max_validation_episodes: int | None = None,
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
    bootstrap, bootstrap_failures = _run_episodes(
        train,
        lambda state, feasible: min(feasible, key=lambda power: abs(power - state[1] * 600.0)),
        accountant,
    )
    for result in bootstrap:
        for transition in result.transitions:
            agent.remember_transition(transition)
    bootstrap_losses = [
        loss for _ in range(updates_per_episode * len(bootstrap))
        if (loss := agent.learn(batch_size=batch_size)) is not None
    ]
    agent.sync_target()
    training_rounds: list[dict[str, object]] = []
    for round_index in range(rounds):
        epsilon = 0.15 if rounds == 1 else 0.15 - 0.13 * round_index / (rounds - 1)
        completed: list[DirectReplay] = []
        failures: list[str] = []
        losses: list[float] = []
        for episode in train:
            try:
                result = replay_episode(
                    episode,
                    lambda state, feasible: agent.select_power(state, feasible, epsilon=epsilon),
                    accountant=accountant,
                )
            except ReplayExecutionError as exc:
                failures.append(f"{episode.sample_id}: {exc}")
                continue
            completed.append(result)
            for transition in result.transitions:
                agent.remember_transition(transition)
            for _ in range(updates_per_episode):
                loss = agent.learn(batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)
        agent.sync_target()
        summary = _summarize(completed, len(train), failures)
        summary.update({"round": round_index + 1, "epsilon": epsilon, "mean_loss": sum(losses) / len(losses) if losses else None})
        training_rounds.append(summary)
    validation_results, validation_failures = _run_episodes(
        validation,
        lambda state, feasible: agent.select_power(state, feasible),
        accountant,
    )
    if dataset.opened_test_payloads != before_test or before_test != 0:
        raise RuntimeError("Test payload was opened during Train/Validation selection")
    terminal_onboard_train = sum(item.operating_mode[-1] == "onboard" for item in train)
    terminal_onboard_validation = sum(item.operating_mode[-1] == "onboard" for item in validation)
    ineligibility_reasons: list[str] = []
    if not agent.replay:
        ineligibility_reasons.append("no_training_experience")
    if validation_failures or not validation_results:
        ineligibility_reasons.append("validation_incomplete")
    if terminal_onboard_train or terminal_onboard_validation:
        ineligibility_reasons.append("unsettled_terminal_energy")
    report: dict[str, object] = {
        "controller": "MLP_Double_DQN_direct_FC_power",
        "sample_seconds": 30,
        "action_kw": list(ACTION_KW),
        "state_dim": 8,
        "reward_definition": "negative_actual_four_component_CNY",
        "bootstrap": _summarize(bootstrap, len(train), bootstrap_failures),
        "bootstrap_mean_loss": sum(bootstrap_losses) / len(bootstrap_losses) if bootstrap_losses else None,
        "training_rounds": training_rounds,
        "validation": _summarize(validation_results, len(validation), validation_failures),
        "unsettled_terminal_onboard_episodes": {
            "train": terminal_onboard_train, "validation": terminal_onboard_validation,
        },
        "selection_eligible": not ineligibility_reasons,
        "selection_ineligibility_reasons": ineligibility_reasons,
        "test_payloads_opened": dataset.opened_test_payloads,
        "hyperparameters": {
            "seed": seed, "rounds": rounds, "batch_size": batch_size,
            "updates_per_episode": updates_per_episode, "gamma": agent.gamma,
            "learning_rate": 1e-4, "replay_capacity": 100_000,
            "hidden_dims": [128, 64], "epsilon_start": 0.15, "epsilon_end": 0.02,
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
