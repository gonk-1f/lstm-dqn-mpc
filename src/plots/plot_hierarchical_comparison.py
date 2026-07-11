from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_hierarchical_comparison(
    dqn_csv: str | Path,
    baseline_csv: str | Path,
    output_png: str | Path,
    max_points: int = 2000,
) -> Path:
    dqn_df = pd.read_csv(dqn_csv)
    baseline_df = pd.read_csv(baseline_csv)

    if max_points > 0 and len(dqn_df) > max_points:
        step = max(1, len(dqn_df) // max_points)
        dqn_df = dqn_df.iloc[::step].reset_index(drop=True)
        baseline_df = baseline_df.iloc[::step].reset_index(drop=True)

    output = Path(output_png)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(dqn_df["tracking_error_kw"].abs(), label="DQN abs tracking error", linewidth=1.0)
    axes[0].plot(baseline_df["tracking_error_kw"].abs(), label="Baseline abs tracking error", linewidth=1.0)
    axes[0].set_ylabel("kW")
    axes[0].set_title("Reference Tracking Comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dqn_df["left_soc"], label="DQN left SOC", linewidth=1.0)
    axes[1].plot(dqn_df["right_soc"], label="DQN right SOC", linewidth=1.0)
    axes[1].plot(baseline_df["left_soc"], label="Baseline left SOC", linewidth=1.0, linestyle="--")
    axes[1].plot(baseline_df["right_soc"], label="Baseline right SOC", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("SOC")
    axes[1].set_title("SOC Evolution")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(dqn_df["left_tracking_error_kw"].abs(), label="DQN left error", linewidth=1.0)
    axes[2].plot(dqn_df["right_tracking_error_kw"].abs(), label="DQN right error", linewidth=1.0)
    axes[2].plot(baseline_df["left_tracking_error_kw"].abs(), label="Baseline left error", linewidth=1.0, linestyle="--")
    axes[2].plot(baseline_df["right_tracking_error_kw"].abs(), label="Baseline right error", linewidth=1.0, linestyle="--")
    axes[2].set_ylabel("kW")
    axes[2].set_xlabel("Step")
    axes[2].set_title("Left/Right Tracking Error")
    axes[2].legend(ncol=2)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
