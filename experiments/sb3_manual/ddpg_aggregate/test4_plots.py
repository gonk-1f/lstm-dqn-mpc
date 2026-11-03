"""Plots of actual deterministic Test4 traces; never fill unexecuted intervals."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_test_plots(rows, traces, output_dir) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.2,
    })
    count = len(rows)
    episodes = np.arange(1, count + 1)
    rewards = [row["episode_reward"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(episodes, rewards, marker="o", color="#147d72")
    for episode, reward in zip(episodes, rewards):
        ax.annotate(f"{reward:,.1f}", (episode, reward), xytext=(0, 9),
                    textcoords="offset points", ha="center")
    ax.set_xticks(episodes, [f"{i}\n{r['sample_id']}" for i, r in zip(episodes, rows)])
    ax.set_xlabel("Test episode (fixed voyage order)")
    ax.set_ylabel("Episode cumulative reward")
    ax.set_title("Deterministic Test4 episode returns")
    ax.margins(y=0.18)
    fig.savefig(output_dir / "test_reward_curve.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(count, 3, figsize=(18, max(3.2, 2.4 * count)),
                             squeeze=False, constrained_layout=True)
    colors = {"fc_kw": "#147d72", "battery_kw": "#3475b8", "load_kw": "#222b36"}
    for index, (trace, row) in enumerate(zip(traces, rows)):
        plot_axes = axes[index]
        for ax in plot_axes:
            ax.set_xlabel("Time (min)")
            ax.axhline(0, color="#7e8791", linewidth=0.5)
        plot_axes[0].set_ylabel(f"{trace['sample_id']}\nPower (kW)")
        if not trace["interval_start_s"]:
            for ax in plot_axes:
                ax.text(0.5, 0.5, "No physical interval executed",
                        transform=ax.transAxes, ha="center", va="center")
            continue
        times = np.array(trace["interval_start_s"] + [trace["time_s"][-1]]) / 60.0

        def line(ax, field, label):
            values = trace[field]
            ax.step(times, values + [values[-1]], where="post",
                    color=colors[field], linewidth=0.8, label=label)
        line(plot_axes[0], "fc_kw", "FC")
        line(plot_axes[1], "battery_kw", "LB")
        line(plot_axes[2], "load_kw", "Load")
        line(plot_axes[2], "fc_kw", "FC")
        line(plot_axes[2], "battery_kw", "LB")
        plot_axes[2].legend(loc="upper right", fontsize=7, ncol=3)
        plot_axes[2].text(0.01, 0.02, row["terminal_reason"],
                          transform=plot_axes[2].transAxes, fontsize=7)
    for ax, title in zip(axes[0], ("Fuel cell", "Battery bus (+ discharge / - charge)",
                                  "Load and power allocation")):
        ax.set_title(title)
    fig.savefig(output_dir / "test_power_curves.png", dpi=150)
    plt.close(fig)

    columns = 1 if count == 1 else 2
    fig, axes = plt.subplots(math.ceil(count / columns), columns,
                             figsize=(13, max(3.2, 2.6 * math.ceil(count / columns))),
                             squeeze=False, constrained_layout=True)
    for ax, trace, row in zip(axes.flat, traces, rows):
        ax.plot(np.array(trace["time_s"]) / 60.0, trace["soc"], color="#147d72", linewidth=1)
        ax.set_title(f"{trace['sample_id']} ({row['terminal_reason']})", fontsize=9)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("SOC")
        ax.set_ylim(0.18, 0.82)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    fig.savefig(output_dir / "test_soc_curves.png", dpi=150)
    plt.close(fig)
