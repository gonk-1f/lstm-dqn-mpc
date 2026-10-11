"""Matched Train/Validation screening of the v4 onboard SOC soft penalty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .control import ReplayExecutionError, replay_episode
from .dqn import DirectPowerDDQN
from .train import _default_data_root, run_train_validation


DEFAULT_BETAS = (0.0, 250.0, 1000.0, 4000.0)


def _beta_label(beta: float) -> str:
    return str(int(beta)) if beta.is_integer() else str(beta).replace(".", "p")


def _profile(
    episode: object, agent: DirectPowerDDQN, accountant: EconomicMPC, beta_soc: float,
) -> dict[str, object]:
    failure: str | None = None
    result = None
    try:
        result = replay_episode(
            episode, lambda state, feasible: agent.select_power(state, feasible),
            accountant=accountant, beta_soc=beta_soc,
        )
        transitions = result.transitions
    except ReplayExecutionError as exc:
        if exc.failure_kind != "no_feasible_action":
            raise
        transitions = exc.executed_transitions
        failure = str(exc)
    fc = [item.action_kw for item in transitions]
    battery = [item.actual_battery_kw for item in transitions]
    soc = [item.actual_soc for item in transitions]
    component_names = (
        "h2_cost_cny", "fuel_cell_degradation_cost_cny",
        "battery_degradation_cost_cny", "shore_cost_cny",
    )
    return {
        "sample_id": str(episode.sample_id),
        "completed": result is not None,
        "failure": failure,
        "observed_cost_cny": result.total_cost_cny if result is not None else None,
        "observed_components_cny": {
            name: getattr(result.total_ledger, name) for name in component_names
        } if result is not None else None,
        "modeled_components_cny": {
            name: getattr(result.modeled_terminal_settlement.ledger, name)
            if result.modeled_terminal_settlement is not None else 0.0
            for name in component_names
        } if result is not None else None,
        "modeled_terminal_cost_cny": (
            result.modeled_terminal_settlement.ledger.total_cost_cny
            if result is not None and result.modeled_terminal_settlement is not None else 0.0
        ) if result is not None else None,
        "comparable_cost_cny": result.comparable_cost_cny if result is not None else None,
        "soc_soft_penalty_cny": sum(item.soc_soft_penalty_cny for item in transitions),
        "terminal_onboard_soc": soc[-1] if result is not None and soc else None,
        "last_executed_soc": soc[-1] if soc else None,
        "soc_min_onboard": min(soc) if soc else None,
        "fc_total_variation_kw": sum(
            abs(right - left) for left, right in zip((0, *fc[:-1]), fc)
        ),
        "battery_peak_abs_kw": max((abs(value) for value in battery), default=None),
        "onboard_step": list(range(len(transitions))),
        "load_kw": [item.state[4] * 600.0 for item in transitions],
        "fc_kw": fc,
        "battery_bus_kw": battery,
        "soc_after": soc,
    }


def _common_cost(
    profiles_by_beta: dict[str, dict[str, dict[str, dict[str, object]]]],
    split: str,
) -> dict[str, object]:
    completed_sets = [
        {sample_id for sample_id, row in profiles[split].items() if row["completed"]}
        for profiles in profiles_by_beta.values()
    ]
    common = sorted(set.intersection(*completed_sets)) if completed_sets else []
    return {
        "sample_ids": common,
        "count": len(common),
        "cost_by_beta_cny": {
            beta: sum(float(profiles[split][sample_id]["comparable_cost_cny"]) for sample_id in common)
            if common else None
            for beta, profiles in profiles_by_beta.items()
        },
    }


def _plot_validation(
    profiles_by_beta: dict[str, dict[str, dict[str, dict[str, object]]]],
    destination: Path,
) -> None:
    sample_ids = sorted(set.union(*(
        set(profiles["validation"]) for profiles in profiles_by_beta.values()
    )))
    for sample_id in sample_ids:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for beta, profiles in profiles_by_beta.items():
            row = profiles["validation"].get(sample_id)
            if row is None or not row["onboard_step"]:
                continue
            minutes = [step * 0.5 for step in row["onboard_step"]]
            suffix = "" if row["completed"] else " (failed)"
            label = f"beta={beta}{suffix}"
            axes[0].plot(minutes, row["fc_kw"], label=label, linewidth=1.1)
            axes[1].plot(minutes, row["battery_bus_kw"], label=label, linewidth=1.1)
            axes[2].plot(minutes, row["soc_after"], label=label, linewidth=1.1)
        axes[0].set_ylabel("FC (kW)")
        axes[1].set_ylabel("Battery bus (kW)")
        axes[2].set_ylabel("SOC")
        axes[2].set_xlabel("Executed ONBOARD time (min)")
        axes[2].axhspan(0.4, 0.6, color="green", alpha=0.08)
        for axis in axes:
            axis.grid(alpha=0.2)
        axes[0].legend(ncol=2, fontsize=8)
        fig.suptitle(sample_id)
        fig.tight_layout()
        fig.savefig(destination / f"{sample_id}_power_soc.png", dpi=140)
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v4_soc_beta_study_5r_seed42"))
    parser.add_argument("--betas", nargs="+", type=float, default=DEFAULT_BETAS)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--updates-per-episode", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--progress-every-steps", type=int, default=50)
    parser.add_argument("--max-train-episodes", type=int)
    parser.add_argument("--max-validation-episodes", type=int)
    args = parser.parse_args(argv)
    betas = tuple(float(beta) for beta in args.betas)
    if len(set(betas)) != len(betas) or not betas:
        raise ValueError("beta candidates must be unique and nonempty")
    dataset = FormalTrainingDataset.open(*(
        _default_data_root(name) for name in (
            "operating_dataset_zero_boundary_v2",
            "operating_dataset_zero_boundary_v2_ais",
            "operating_dataset_zero_boundary_v2_modes",
        )
    ))
    train = dataset.load_train()[:args.max_train_episodes]
    validation = dataset.load_validation()[:args.max_validation_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profiles_by_beta: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    reports: dict[str, dict[str, object]] = {}
    for beta in betas:
        label = _beta_label(beta)
        if label in reports:
            raise ValueError("beta labels must be unique")
        print(f"beta={beta:g} start", flush=True)
        agent, report = run_train_validation(
            dataset, rounds=args.rounds, seed=args.seed, batch_size=args.batch_size,
            updates_per_episode=args.updates_per_episode,
            max_train_episodes=args.max_train_episodes,
            max_validation_episodes=args.max_validation_episodes,
            progress_every_steps=args.progress_every_steps,
            epsilon_start=args.epsilon_start, epsilon_end=args.epsilon_end,
            beta_soc=beta,
        )
        accountant = EconomicMPC(nominal_cost_cny=1.0)
        profiles = {
            split: {
                str(episode.sample_id): _profile(episode, agent, accountant, beta)
                for episode in episodes
            }
            for split, episodes in (("train", train), ("validation", validation))
        }
        if sum(row["completed"] for row in profiles["train"].values()) != report["train_greedy_evaluation"]["completed"]:
            raise RuntimeError("greedy Train profile count differs from training report")
        if sum(row["completed"] for row in profiles["validation"].values()) != report["validation"]["completed"]:
            raise RuntimeError("Validation profile count differs from training report")
        beta_dir = args.output_dir / f"beta_{label}"
        beta_dir.mkdir(parents=True, exist_ok=True)
        (beta_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (beta_dir / "profiles.json").write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")
        reports[label] = report
        profiles_by_beta[label] = profiles
        print(
            f"beta={beta:g} greedy_train={report['train_greedy_evaluation']['completed']}/{len(train)} "
            f"validation={report['validation']['completed']}/{len(validation)}",
            flush=True,
        )
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test payload was opened during SOC beta screening")
    common = {
        split: _common_cost(profiles_by_beta, split)
        for split in ("train", "validation")
    }
    summary = {
        "betas": betas,
        "rounds": args.rounds,
        "train_episodes": len(train),
        "validation_episodes": len(validation),
        "test_payloads_opened": dataset.opened_test_payloads,
        "reports": {
            beta: {
                "train_greedy_evaluation": report["train_greedy_evaluation"],
                "validation": report["validation"],
                "training_rounds": report["training_rounds"],
            }
            for beta, report in reports.items()
        },
        "common_completed_cost": common,
        "cost_rule": "Full-split cost is null if any episode fails; matched cost uses only sample IDs completed by every beta.",
    }
    (args.output_dir / "study_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    _plot_validation(profiles_by_beta, args.output_dir)
    print(json.dumps({
        "common_completed_cost": common,
        "test_payloads_opened": dataset.opened_test_payloads,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
