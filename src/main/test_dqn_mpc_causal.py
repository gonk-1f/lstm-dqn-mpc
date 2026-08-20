from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dqn.agents.dqn_agent import DQNTrainConfig
from envs.dqn_mpc_weight_env import (
    DqnMpcWeightEnv,
    MpcSolveFailure,
)
import train_dqn_mpc_mlp as training


MODEL_DIR = (
    REPO_ROOT
    / "outputs"
    / "dqn_mpc_mlp_causal_1epoch_20260820"
)

MODEL_PATH = MODEL_DIR / "model_final.pt"

TEST_OUTPUT_DIR = MODEL_DIR / "formal_test"
TRACE_DIR = TEST_OUTPUT_DIR / "traces"
PLOT_DIR = TEST_OUTPUT_DIR / "plots"

TEST_CSV = TEST_OUTPUT_DIR / "test_by_voyage.csv"
TEST_SUMMARY = TEST_OUTPUT_DIR / "test_summary.json"


def load_test_voyage(
    voyage_id: str,
    *,
    split,
) -> np.ndarray:
    voyage_id = str(voyage_id)

    if voyage_id not in split.test_voyages:
        raise ValueError(
            f"{voyage_id} is not a test voyage"
        )

    root = training.DEFAULT_VOYAGE_DATA_DIR.resolve()

    manifest = pd.read_csv(
        root / "manifest.csv"
    )

    rows = manifest.loc[
        manifest["voyage_id"].astype(str)
        == voyage_id
    ]

    if len(rows) != 1:
        raise ValueError(
            f"Expected one manifest row for {voyage_id}"
        )

    row = rows.iloc[0]

    if str(row["split"]).strip().lower() != "test":
        raise ValueError(
            f"{voyage_id} is not labelled test"
        )

    raw_path = Path(str(row["output_csv"]))

    voyage_path = (
        raw_path
        if raw_path.is_absolute()
        else REPO_ROOT / raw_path
    ).resolve()

    frame = pd.read_csv(
        voyage_path,
        usecols=[
            "voyage_id",
            "split",
            "time_s",
            split.target_load,
        ],
    )

    frame = frame.sort_values(
        "time_s",
        kind="stable",
    ).reset_index(drop=True)

    loads_kw = pd.to_numeric(
        frame[split.target_load],
        errors="raise",
    ).to_numpy(dtype=np.float64)

    if len(loads_kw) < 2:
        raise ValueError(
            f"{voyage_id} contains too few samples"
        )

    return loads_kw


def run_test_episode(
    *,
    voyage_id: str,
    loads_kw: np.ndarray,
    base_config,
    agent,
) -> dict[str, object]:
    env = DqnMpcWeightEnv(
        loads_kw=loads_kw,
        base_config=base_config,
        initial_soc=0.55,
    )

    state = env.reset()

    completed = True
    solver_failure_count = 0
    failure_index = None

    episode_reward = 0.0
    episode_steps = 0

    min_soc = float(env.current_soc)

    action_counts = np.zeros(
        training.ACTION_DIM,
        dtype=np.int64,
    )
    trace_rows: list[dict[str, object]] = []
    done = False

    while not done:
        action = agent.greedy_action(state)

        try:
            next_state, reward, done, info = env.step(
                action
            )

        except MpcSolveFailure:
            completed = False
            solver_failure_count = 1
            failure_index = int(env.decision_index)

            reward = float(
                agent.config.solver_failure_reward
            )

            action_counts[action] += 1
            episode_reward += reward
            episode_steps += 1

            break

        action_counts[action] += 1

        episode_reward += float(reward)
        episode_steps += 1

        min_soc = min(
            min_soc,
            float(info["soc_after"]),
        )
        trace_rows.append(
            {
                "decision_index": int(info["decision_index"]),
                "execution_index": int(info["execution_index"]),
                "load_kw": float(info["load_actual_kw"]),
                "p_fc_kw": float(info["p_fc_kw"]),
                "p_batt_kw": float(info["p_batt_kw"]),
                "soc_before": float(info["soc_before"]),
                "soc_after": float(info["soc_after"]),
                "action_id": int(action),
                "reward": float(reward),
                "solver_status": str(info["solver_status"]),
            }
        )
        state = next_state

    result = {
        "voyage_id": voyage_id,
        "completed": completed,
        "solver_failure_count": (
            solver_failure_count
        ),
        "failure_index": failure_index,
        "episode_steps": episode_steps,
        "episode_reward": episode_reward,
        "mean_reward_per_step": (
            episode_reward / episode_steps
        ),
        "min_soc": min_soc,
        "final_soc": float(env.current_soc),
    }

    for action_id in range(training.ACTION_DIM):
        result[f"action_count_A{action_id}"] = int(
            action_counts[action_id]
        )

    return result, pd.DataFrame(trace_rows)

def plot_power_allocation(
    voyage_id: str,
    trace: pd.DataFrame,
) -> None:
    x = trace["execution_index"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        x,
        trace["load_kw"],
        label="Load",
    )
    ax.plot(
        x,
        trace["p_fc_kw"],
        label="Fuel cell",
    )
    ax.plot(
        x,
        trace["p_batt_kw"],
        label="Battery",
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (kW)")
    ax.set_title(
        f"{voyage_id} - Power Allocation"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / f"{voyage_id}_power.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_soc_trajectory(
    voyage_id: str,
    trace: pd.DataFrame,
) -> None:
    x = trace["execution_index"].to_numpy(dtype=float)
    soc = trace["soc_after"].to_numpy(dtype=float)

    soc_min = float(np.min(soc))
    soc_max = float(np.max(soc))

    padding = max(
        0.01,
        0.15 * (soc_max - soc_min),
    )

    y_min = max(
        0.18,
        min(soc_min, 0.55) - padding,
    )
    y_max = min(
        0.82,
        max(soc_max, 0.55) + padding,
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        x,
        soc,
        label="SOC",
    )

    ax.axhline(
        0.55,
        linewidth=1.0,
        label="SOC reference = 0.55",
    )

    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("SOC")
    ax.set_title(
        f"{voyage_id} - SOC Trajectory "
        f"(min={soc_min:.4f}, hard limit=0.20)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / f"{voyage_id}_soc.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)
def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if TEST_OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Formal test output already exists: "
            f"{TEST_OUTPUT_DIR}"
        )

    TEST_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )
    TRACE_DIR.mkdir()
    PLOT_DIR.mkdir()
    # 防止无意重复覆盖第一次正式测试结果。
    if TEST_CSV.exists() or TEST_SUMMARY.exists():
        raise FileExistsError(
            "Formal test output already exists."
        )

    split = training.load_voyage_split()

    print("Test voyages:", split.test_voyages)

    runtime = training.create_training_runtime(
        DQNTrainConfig()
    )

    runtime.agent.load(MODEL_PATH)
    runtime.agent.q_net.eval()

    base_config = training.build_formal_mpc_config()

    results = []

    for index, voyage_id in enumerate(
        split.test_voyages,
        start=1,
    ):
        loads_kw = load_test_voyage(
            voyage_id,
            split=split,
        )

        result, trace = run_test_episode(
            voyage_id=voyage_id,
            loads_kw=loads_kw,
            base_config=base_config,
            agent=runtime.agent,
        )

        results.append(result)

        trace_path = (
                TRACE_DIR
                / f"{voyage_id}_trace.csv"
        )

        trace.to_csv(
            trace_path,
            index=False,
        )

        plot_power_allocation(
            voyage_id,
            trace,
        )

        plot_soc_trajectory(
            voyage_id,
            trace,
        )

        print(
            f"[test] {index}/{len(split.test_voyages)} "
            f"voyage={voyage_id} "
            f"completed={result['completed']} "
            f"steps={result['episode_steps']} "
            f"min_soc={result['min_soc']:.6f} "
            f"reward_mean="
            f"{result['mean_reward_per_step']:.6f}"
        )

    frame = pd.DataFrame(results)

    frame.to_csv(
        TEST_CSV,
        index=False,
    )

    completed = int(frame["completed"].sum())

    summary = {
        "model_path": str(MODEL_PATH),
        "test_voyages": list(split.test_voyages),
        "completed_voyages": completed,
        "total_voyages": len(frame),
        "success_rate": (
            completed / len(frame)
        ),
        "failed_voyages": frame.loc[
            ~frame["completed"],
            "voyage_id",
        ].tolist(),
    }

    TEST_SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nFormal test finished.")
    print(
        f"Completed: {completed}/{len(frame)}"
    )
    print(
        "Failed:",
        summary["failed_voyages"],
    )
    print("CSV:", TEST_CSV)
    print("Summary:", TEST_SUMMARY)


if __name__ == "__main__":
    main()