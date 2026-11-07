"""Plot real episode logs from the independent simplified-reward TD3 run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


COMPONENT_FIELDS = (
    "reward_h2_cost_cny", "reward_fc_variation_penalty", "reward_soc_penalty",
    "failure_reward_total",
)
NUMERIC_FIELDS = (
    "episode", "raw_reward_sum", "algorithm_reward_sum", "episode_length",
    "coverage", "soc_min", "soc_final",
) + COMPONENT_FIELDS


def _rolling_mean(values: list[float], window: int) -> list[float]:
    """Keep partial windows absent instead of implying a full-window average."""
    return [
        math.nan if index + 1 < window else math.fsum(values[index + 1 - window:index + 1]) / window
        for index in range(len(values))
    ]


def plot_training(run_dir: str | Path, window: int = 20) -> dict:
    """Read saved episodes only; never construct an environment or load a model."""
    if window < 1:
        raise ValueError("The moving-average window must be positive")
    directory = Path(run_dir).resolve()
    metadata_path = directory / "td3_training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    identity = metadata.get("reward_identity", {})
    include_hydrogen = identity.get("include_hydrogen", True)
    soc_weight = identity.get("soc_deviation_weight", 10.0)
    source = directory / "training_episodes.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(NUMERIC_FIELDS + ("completed",)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Simplified-reward episode log is missing fields: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No completed episode records in {source}")
    columns = {name: [float(row[name]) for row in rows] for name in NUMERIC_FIELDS}
    if any(not math.isfinite(value) for values in columns.values() for value in values):
        raise ValueError("Episode log contains a non-finite plotted value")
    if any(length <= 0 for length in columns["episode_length"]):
        raise ValueError("Recorded episode lengths must be positive")
    successes = [row["completed"].strip().lower() == "true" for row in rows]
    episodes = columns["episode"]
    raw = columns["raw_reward_sum"]
    mean_step = [reward / length for reward, length in zip(raw, columns["episode_length"])]
    failed_indices = [index for index, success in enumerate(successes) if not success]

    # Import plotting only inside the CPU plotting subprocess. This module has
    # no dependency on SB3, PyTorch, Replay Buffer or the vessel environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 10,
    })
    figure, axes = plt.subplots(2, 2, figsize=(15, 9.2), layout="constrained")
    failures = len(rows) - sum(successes)
    figure.suptitle(
        f"TD3简化奖励实验：已记录{len(rows)}个Episode（成功{sum(successes)}，失败{failures}）\n"
        + ("正常原始奖励 = −[" + ("实际氢耗成本 + " if include_hydrogen else "")
         + f"0.0025×ΔFC² + {soc_weight:g}×(SOC_next−0.5)²]；失败−20000；训练奖励 = 原始值/10000"),
        fontsize=14,
    )

    reward_axis = axes[0, 0]
    reward_axis.plot(episodes, raw, color="#377eb8", marker=".", linewidth=0.8,
                     alpha=0.65, label="单Episode累计奖励")
    if len(rows) >= window:
        reward_axis.plot(episodes, _rolling_mean(raw, window), color="#163b63", linewidth=2,
                         label=f"{window}个Episode完整窗口均值")
    if failed_indices:
        reward_axis.scatter([episodes[index] for index in failed_indices],
                            [raw[index] for index in failed_indices],
                            marker="x", color="#d62728", s=30, label="物理失败", zorder=4)
    reward_axis.set(title="每个Episode累计奖励（越高越好）", ylabel="原始累计奖励")
    reward_axis.legend(fontsize=9)

    average_axis = axes[0, 1]
    length_axis = average_axis.twinx()
    length_axis.bar(episodes, columns["episode_length"], color="#777777", alpha=0.12,
                    width=0.8, label="Episode长度")
    length_axis.set_ylabel("Episode长度（步）", color="#666666")
    length_axis.tick_params(axis="y", colors="#666666")
    average_axis.plot(episodes, mean_step, color="#238b45", marker=".", linewidth=0.8,
                      label="累计奖励 / Episode长度")
    if len(rows) >= window:
        average_axis.plot(episodes, _rolling_mean(mean_step, window), color="#005a32", linewidth=2,
                          label=f"{window}个Episode完整窗口均值")
    average_axis.set(title="每步平均奖励；失败覆盖奖励仍计入", ylabel="原始奖励 / 步")
    handles, labels = average_axis.get_legend_handles_labels()
    length_handles, length_labels = length_axis.get_legend_handles_labels()
    average_axis.legend(handles + length_handles, labels + length_labels, fontsize=9)

    component_axis = axes[1, 0]
    for field, color, label in (
        ("reward_h2_cost_cny", "#377eb8", "实际氢耗成本"),
        ("reward_fc_variation_penalty", "#ff7f00", "FC平方波动惩罚"),
        ("reward_soc_penalty", "#984ea3", "SOC偏离惩罚"),
    ):
        if field == "reward_h2_cost_cny" and not include_hydrogen:
            continue
        component_axis.plot(episodes, columns[field], color=color, marker=".", linewidth=1, label=label)
    component_axis.set(title="每个Episode的奖励分项（SOC/波动惩罚不是设备账本）",
                       ylabel="累计成本 / 惩罚，原始尺度")
    failure_values = columns["failure_reward_total"]
    failure_indices = [index for index, value in enumerate(failure_values) if value != 0.0]
    if failure_indices:
        failure_axis = component_axis.twinx()
        failure_axis.scatter([episodes[index] for index in failure_indices],
                             [failure_values[index] for index in failure_indices],
                             color="#d62728", marker="x", s=35, label="失败覆盖奖励")
        failure_axis.set_ylabel("失败覆盖奖励（原始尺度）", color="#d62728")
        failure_axis.tick_params(axis="y", colors="#d62728")
        handles, labels = component_axis.get_legend_handles_labels()
        failure_handles, failure_labels = failure_axis.get_legend_handles_labels()
        component_axis.legend(handles + failure_handles, labels + failure_labels, fontsize=9)
    else:
        component_axis.legend(fontsize=9)

    soc_axis = axes[1, 1]
    soc_axis.plot(episodes, columns["soc_final"], color="#238b45", marker=".", label="最终SOC")
    soc_axis.plot(episodes, columns["soc_min"], color="#74c476", linewidth=1, label="航段最低SOC")
    soc_axis.axhline(0.5, color="#c17d23", linestyle="--", linewidth=1, label="目标0.5")
    soc_axis.axhline(0.2, color="#777777", linestyle=":", linewidth=1)
    soc_axis.axhline(0.8, color="#777777", linestyle=":", linewidth=1, label="物理边界0.2/0.8")
    if failed_indices:
        soc_axis.scatter([episodes[index] for index in failed_indices],
                         [columns["soc_final"][index] for index in failed_indices],
                         marker="x", color="#d62728", s=30, label="失败Episode", zorder=4)
    coverage_axis = soc_axis.twinx()
    coverage_axis.plot(episodes, columns["coverage"], color="#555555", linestyle="--",
                       alpha=0.55, linewidth=1, label="航段时长覆盖率")
    coverage_axis.set(ylim=(-0.03, 1.05), ylabel="航段时长覆盖率")
    soc_axis.set(title="SOC与实际航段完成情况", ylabel="SOC", ylim=(0.15, 0.85))
    handles, labels = soc_axis.get_legend_handles_labels()
    coverage_handles, coverage_labels = coverage_axis.get_legend_handles_labels()
    soc_axis.legend(handles + coverage_handles, labels + coverage_labels, fontsize=8, loc="best")
    for axis in axes.flat:
        axis.set_xlabel("训练Episode")
        axis.grid(alpha=0.2)
        if len(episodes) == 1:
            axis.set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)

    figure_path = directory / "reward_curves.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    errors = [
        abs(raw[index] - (
            -columns["reward_h2_cost_cny"][index]
            -columns["reward_fc_variation_penalty"][index]
            -columns["reward_soc_penalty"][index]
            +failure_values[index]
        )) for index in range(len(rows))
    ]
    latest = {name: columns[name][-1] for name in NUMERIC_FIELDS}
    latest.update(completed=successes[-1], mean_reward_per_step=mean_step[-1])
    summary = {
        "source_csv": str(source), "figure": str(figure_path),
        "include_hydrogen_in_reward": include_hydrogen,
        "recorded_episodes": len(rows), "latest_episode": int(episodes[-1]),
        "successful_episodes": sum(successes), "failed_episodes": failures,
        "completion_rate": sum(successes) / len(rows),
        "moving_average_window_episodes": window,
        "moving_average_uses_complete_windows_only": True,
        "reward_accounting_max_abs_error": max(errors),
        "latest": latest,
        "scope": "Observed training episodes only; no environment rollout, inference or training is performed",
    }
    (directory / "reward_curve_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    )
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Zoo run directory containing training_episodes.csv")
    parser.add_argument("--window", type=int, default=20,
                        help="Moving-average window in complete episodes (default: 20)")
    args = parser.parse_args(argv)
    result = plot_training(args.run_dir, args.window)
    print(f"Reward curves: {result['figure']}", flush=True)


if __name__ == "__main__":
    main()
