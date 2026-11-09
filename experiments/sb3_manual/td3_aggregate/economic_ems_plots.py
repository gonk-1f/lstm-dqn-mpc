"""Plot economic TD3 training curves from saved episode records only."""
import argparse
import csv
import math
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plot-fixed-evaluation", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.run_dir / "training_episodes.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["mean_reward_per_step"]) for row in rows]
    warmup = [i for i, row in enumerate(rows) if int(row["warmup_steps"]) > 0]
    postwarmup = [i for i, row in enumerate(rows) if int(row["warmup_steps"]) == 0]
    failures = {i for i, row in enumerate(rows) if row["completed"] != "True"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    (overview, zoom), (cost, soc) = axes
    overview.plot(episodes, rewards, color="tab:blue", label="All attempts")
    if warmup:
        overview.axvspan(episodes[warmup[0]] - 0.5, episodes[warmup[-1]] + 0.5,
                         color="tab:orange", alpha=0.16, label="Contains warm-up")
    overview.set_title("Mean raw reward per step")

    if postwarmup:
        post_episodes = [episodes[i] for i in postwarmup]
        post_rewards = [rewards[i] for i in postwarmup]
        zoom.plot(post_episodes, post_rewards, color="tab:blue", alpha=0.55, label="Post-warm-up attempts")
        if len(post_rewards) >= 20:
            moving_mean = [
                math.nan if end < 19 else fmean(post_rewards[end - 19:end + 1])
                for end in range(len(post_rewards))
            ]
            zoom.plot(post_episodes, moving_mean, color="tab:orange", label="20-episode mean, failures included")
    else:
        zoom.text(0.5, 0.5, "No post-warm-up episodes yet", ha="center", transform=zoom.transAxes)
    zoom.set_title("Mean raw reward per step after warm-up")
    for axis, scope in ((overview, range(len(rows))), (zoom, postwarmup)):
        failed = [i for i in scope if i in failures]
        if failed:
            axis.scatter([episodes[i] for i in failed], [rewards[i] for i in failed],
                         color="tab:red", zorder=3, label="Physical failure")
        axis.set_ylabel("Raw episode reward / environment steps")

    cost.plot(episodes, [float(row["physical_economic_cost_cny"]) for row in rows], label="Physical economic cost")
    cost.plot(episodes, [float(row["reward_economic_cost_cny"]) for row in rows], alpha=0.6, label="Learning economic cost")
    cost.set_title("Voyage costs")
    cost.set_ylabel("CNY per voyage")
    soc.plot(episodes, [float(row["soc_final"]) for row in rows], label="Terminal SOC")
    soc.axhline(0.5, linestyle="--", label="SOC midpoint 0.5")
    soc.axhline(0.2, linestyle=":", label="Hard lower bound")
    soc.axhline(0.8, linestyle=":", label="Hard upper bound")
    soc.set_title("Terminal SOC")
    for axis in axes.flat:
        axis.set_xlabel("Training episode")
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    fig.savefig(args.run_dir / "economic_ems_reward_curves.png", dpi=160)
    plt.close(fig)

    if args.plot_fixed_evaluation:
        path = args.run_dir / "fixed_train_evaluation" / "fixed_train_evaluation.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            evaluations = list(csv.DictReader(handle))
        fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
        axis.plot(
            [int(row["training_episode"]) for row in evaluations],
            [float(row["mean_reward_per_step"]) for row in evaluations],
            marker="o",
        )
        axis.set(
            title="Fixed-voyage deterministic evaluation",
            xlabel="Completed training voyages",
            ylabel="Raw evaluation reward / environment steps",
        )
        axis.grid(alpha=0.2)
        fig.savefig(args.run_dir / "economic_ems_fixed_evaluation_reward_curve.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
