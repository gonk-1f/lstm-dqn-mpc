"""Fixed beta=250, 40-round Train/Validation review; Test remains closed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC

from .soc_beta_study import _profile
from .telemetry import soc_time_occupancy
from .train import _default_data_root, run_train_validation


FIXED_CONFIGURATION = {
    "rounds": 40, "seed": 42, "epsilon_start": 1.0, "epsilon_end": 0.05,
    "batch_size": 64, "updates_per_episode": 16, "beta_soc": 250.0,
    "progress_every_steps": 50,
}


def _manifest_hashes(roots: Sequence[Path]) -> dict[str, str]:
    return {
        f"{root.name}/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots for path in sorted((root / "metadata").glob("*manifest.csv"))
    }


def _trajectory_plot(row: dict[str, object], destination: Path, beta_soc: float = 250.0) -> None:
    minutes = [step * 0.5 for step in row["onboard_step"]]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(minutes, row["load_kw"], color="black", alpha=0.5, label="Load")
    axes[0].plot(minutes, row["fc_kw"], label="FC", linewidth=1)
    axes[1].plot(minutes, row["battery_bus_kw"], label="Battery", linewidth=1)
    axes[2].plot(minutes, row["soc_after"], label="Post-action SOC", linewidth=1)
    for axis, ylabel in zip(axes, ("Power (kW)", "Battery (kW)", "SOC")):
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(loc="best", fontsize=8)
    axes[2].axhspan(0.4, 0.6, color="green", alpha=0.1)
    axes[2].axhline(0.79, linestyle="--", color="orange", linewidth=0.8)
    axes[2].set_ylim(0.18, 0.82)
    axes[2].set_xlabel("Executed ONBOARD time (min); SHORE excluded")
    fig.suptitle(f"{row['sample_id']} | beta={beta_soc:g} | {'completed' if row['completed'] else 'failed'}")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _training_plot(rounds: list[dict[str, object]], destination: Path) -> None:
    indices = [row["round"] for row in rounds]
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(indices, [row["completed"] for row in rounds], marker="o", markersize=3)
    axes[0].set_ylabel("Train completed / 30")
    axes[0].set_ylim(0, 31)
    for name in rounds[0]["soc_time_occupancy"]["fractions"]:
        axes[1].plot(indices, [row["soc_time_occupancy"]["fractions"][name] for row in rounds], label=name)
    axes[1].set_ylabel("SOC time fraction")
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].plot(indices, [row["mean_loss"] for row in rounds])
    axes[2].set_ylabel("Mean Smooth L1 loss")
    axes[3].plot(indices, [row["economic_q_optimizer_updates"] for row in rounds])
    axes[3].set_ylabel("Economic Q updates")
    axes[3].set_xlabel("Round; epsilon decreases linearly 1.0 to 0.05")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/v4_beta250_40r_seed42"))
    args = parser.parse_args(argv)
    if (args.output_dir / "report.json").exists():
        raise FileExistsError("choose a fresh output directory; fixed review does not overwrite a prior report")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roots = tuple(_default_data_root(name) for name in (
        "operating_dataset_zero_boundary_v2",
        "operating_dataset_zero_boundary_v2_ais",
        "operating_dataset_zero_boundary_v2_modes_v3",
    ))
    before_hashes = _manifest_hashes(roots)
    dataset = FormalTrainingDataset.open(*roots)
    print(f"fixed_configuration={json.dumps(FIXED_CONFIGURATION)}", flush=True)
    started = time.monotonic()
    agent, report = run_train_validation(dataset, **FIXED_CONFIGURATION)
    counts = report["execution_counts"]
    counts["learning_rollout_onboard_executed_transitions"] = (
        counts["bootstrap_onboard_executed_transitions"] + counts["training_onboard_executed_transitions"]
    )
    counts["all_train_validation_onboard_executed_transitions"] = (
        counts["learning_rollout_onboard_executed_transitions"]
        + counts["greedy_evaluation_onboard_executed_transitions"]
    )
    report["training_and_greedy_evaluation_elapsed_seconds"] = time.monotonic() - started
    report["source_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()
    report["manifest_sha256"] = before_hashes
    checkpoint = {
        "architecture": "v4_mlp_double_dqn_direct_power",
        "model_state": agent.online.state_dict(),
        "state_dim": 8, "action_kw": report["action_kw"],
        "hyperparameters": report["hyperparameters"],
        "selection_eligible": report["selection_eligible"],
        "purpose": "Train/Validation diagnostics; not authorization to open Test",
    }
    torch.save(checkpoint, args.output_dir / "diagnostic_final_agent.pt")
    if report["selection_eligible"]:
        torch.save(checkpoint, args.output_dir / "selected_agent.pt")
    # Save the training evidence before additional read-only trajectory replay.
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    profiles = {}
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    for split, episodes in (("train", dataset.load_train()), ("validation", dataset.load_validation())):
        profiles[split] = {
            str(episode.sample_id): _profile(episode, agent, accountant, 250.0)
            for episode in episodes
        }
        rows = list(profiles[split].values())
        for row in rows:
            row["soc_time_occupancy"] = soc_time_occupancy(row["soc_after"])
        key = "train_greedy_evaluation" if split == "train" else "validation"
        summary = report[key]
        if sum(row["completed"] for row in rows) != summary["completed"]:
            raise RuntimeError("profile completion differs from final greedy evaluation")
        occupancy = soc_time_occupancy(soc for row in rows for soc in row["soc_after"])
        if occupancy["counts"] != summary["soc_time_occupancy"]["counts"]:
            raise RuntimeError("profile SOC occupancy differs from final greedy evaluation")
        summary["completed_observed_components_cny"] = {
            component: sum(row["observed_components_cny"][component] for row in rows if row["completed"])
            for component in (
                "h2_cost_cny", "fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny", "shore_cost_cny",
            )
        }
        summary["completed_modeled_components_cny"] = {
            component: sum(row["modeled_components_cny"][component] for row in rows if row["completed"])
            for component in summary["completed_observed_components_cny"]
        }
    if dataset.opened_test_payloads != 0 or before_hashes != _manifest_hashes(roots):
        raise RuntimeError("Test opened or dataset manifests changed during review")
    report["test_payloads_opened"] = dataset.opened_test_payloads
    report["dataset_manifests_unchanged"] = True
    counts["additional_read_only_profile_replay_transitions"] = sum(
        len(row["soc_after"]) for rows in profiles.values() for row in rows.values()
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
    fields = (
        "sample_id", "completed", "observed_cost_cny", "modeled_terminal_cost_cny",
        "comparable_cost_cny", "soc_soft_penalty_cny", "soc_min_onboard", "terminal_onboard_soc",
    )
    with (args.output_dir / "episode_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("split", *fields))
        writer.writeheader()
        for split, rows in profiles.items():
            writer.writerows({"split": split, **{key: row[key] for key in fields}} for row in rows.values())
    for row in profiles["validation"].values():
        _trajectory_plot(row, args.output_dir / f"{row['sample_id']}_power_soc.png")
    representative_train = sorted(
        profiles["train"].values(), key=lambda row: max(row["load_kw"], default=0.0), reverse=True,
    )[:2]
    for row in representative_train:
        _trajectory_plot(row, args.output_dir / f"train_{row['sample_id']}_power_soc.png")
    _training_plot(report["training_rounds"], args.output_dir / "training_curves.png")
    print(json.dumps({
        "train_greedy": report["train_greedy_evaluation"],
        "validation": report["validation"], "execution_counts": report["execution_counts"],
        "selection_eligible": report["selection_eligible"], "test_payloads_opened": 0,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
