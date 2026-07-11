from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _time_axis_hours(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    if "dqn_time_seconds" in df.columns:
        return df["dqn_time_seconds"].to_numpy(dtype=float) / 3600.0, "Time (h)"
    if "dqn_control_seconds" in df.columns and not df["dqn_control_seconds"].dropna().empty:
        dt = float(df["dqn_control_seconds"].dropna().iloc[0])
        return np.arange(len(df), dtype=float) * dt / 3600.0, "Time (h)"
    return np.arange(len(df), dtype=float), "Step"


def plot_soc_curve(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(df["soc"], label="SOC")
    plt.xlabel("Step")
    plt.ylabel("SOC")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_dual_side_soc_curve(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x, xlabel = _time_axis_hours(df)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=False)
    ax = axes[0]
    if "left_soc" in df.columns:
        ax.plot(x, df["left_soc"], label="Left SOC", color="tab:green", linewidth=1.2)
    if "right_soc" in df.columns:
        ax.plot(x, df["right_soc"], label="Right SOC", color="tab:red", linewidth=1.2)
    if "soc_ref" in df.columns:
        ax.plot(x, df["soc_ref"], label="SOC Ref", color="tab:blue", linestyle="--", linewidth=1.0)
    ax.axhspan(0.45, 0.75, color="tab:green", alpha=0.08, label="Suggested target band")
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1.0, label="SOC Min")
    ax.axhline(0.95, color="gray", linestyle="-.", linewidth=1.0, label="SOC Max")
    ax.set_ylabel("SOC")
    ax.set_ylim(0, 1)
    ax.set_title("SOC Trajectory - Full Evaluation")
    ax.legend(ncol=3, fontsize=8)

    zoom_rows = min(len(df), 2400)
    zoom = df.iloc[:zoom_rows]
    zoom_x = x[:zoom_rows]
    if "left_soc" in zoom.columns:
        axes[1].plot(zoom_x, zoom["left_soc"], label="Left SOC", color="tab:green", linewidth=1.2)
    if "right_soc" in zoom.columns:
        axes[1].plot(zoom_x, zoom["right_soc"], label="Right SOC", color="tab:red", linewidth=1.2)
    if "soc_ref" in zoom.columns:
        axes[1].plot(zoom_x, zoom["soc_ref"], label="SOC Ref", color="tab:blue", linestyle="--", linewidth=1.0)
    axes[1].axhline(0.2, color="gray", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.95, color="gray", linestyle="-.", linewidth=1.0)
    axes[1].set_title("SOC Trajectory - First 2400 DQN Steps (5 s scale)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("SOC")
    axes[1].set_ylim(0, 1)
    axes[1].legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path
