from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
from envs.ship_env_dual_side import DualSideEnvConfig, ShipEnvDualSide, load_dual_side_env_config


def rollout_dual_side_reference(eval_csv: str | Path, output_path: str | Path | None = None) -> dict:
    df = pd.read_csv(eval_csv)
    env = ShipEnvDualSide(
        df,
        config=load_dual_side_env_config(Path.cwd()),
    )
    env.config.random_reset = False
    env.config.episode_steps = max(1, len(df) - 1)
    state, _ = env.reset(start_index=0)
    center_tuple = (0.5, 0.5, 0.0, 0.0)
    center_action = env.actions.index(center_tuple) if center_tuple in env.actions else len(env.actions) // 2

    rewards = []
    rows = []
    done = False
    while not done:
        state, reward, done, truncated, info = env.step(center_action)
        rewards.append(reward)
        info["truncated"] = truncated
        rows.append(info)

    summary = {
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "tracking_error_mae_kw": float(np.mean([abs(row["tracking_error_kw"]) for row in rows])),
        "fc_tracking_error_mae_kw": float(np.mean([abs(row["fc_tracking_error_kw"]) for row in rows])),
        "fc_support_adjusted_tracking_error_mae_kw": float(
            np.mean([abs(row["fc_support_adjusted_tracking_error_kw"]) for row in rows])
        ),
        "battery_reference_tracking_error_mae_kw": float(
            np.mean([abs(row["battery_reference_tracking_error_kw"]) for row in rows])
        ),
        "battery_command_tracking_error_mae_kw": float(
            np.mean([abs(row["battery_command_tracking_error_kw"]) for row in rows])
        ),
        "battery_balance_correction_mae_kw": float(
            np.mean([abs(row["battery_balance_correction_kw"]) for row in rows])
        ),
        "left_tracking_error_mae_kw": float(np.mean([abs(row["left_tracking_error_kw"]) for row in rows])),
        "right_tracking_error_mae_kw": float(np.mean([abs(row["right_tracking_error_kw"]) for row in rows])),
        "total_balance_error_mae_kw": float(np.mean([abs(row["total_balance_error_kw"]) for row in rows])),
        "left_soc_final": float(rows[-1]["left_soc"]) if rows else None,
        "right_soc_final": float(rows[-1]["right_soc"]) if rows else None,
        "steps": len(rows),
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    return summary


def evaluate_dual_side_dqn(eval_csv: str | Path, model_path: str | Path, output_path: str | Path | None = None) -> dict:
    df = pd.read_csv(eval_csv)
    env = ShipEnvDualSide(
        df,
        config=load_dual_side_env_config(Path.cwd()),
    )
    env.config.random_reset = False
    env.config.episode_steps = max(1, len(df) - 1)
    agent = DQNAgent(env.state_dim, env.action_dim, DQNTrainConfig())
    agent.load(model_path)

    state, _ = env.reset(start_index=0)
    done = False
    rewards = []
    rows = []
    while not done:
        action = agent.greedy_action(state)
        state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        info["truncated"] = truncated
        rows.append(info)

    summary = {
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "tracking_error_mae_kw": float(np.mean([abs(row["tracking_error_kw"]) for row in rows])),
        "fc_tracking_error_mae_kw": float(np.mean([abs(row["fc_tracking_error_kw"]) for row in rows])),
        "fc_support_adjusted_tracking_error_mae_kw": float(
            np.mean([abs(row["fc_support_adjusted_tracking_error_kw"]) for row in rows])
        ),
        "battery_reference_tracking_error_mae_kw": float(
            np.mean([abs(row["battery_reference_tracking_error_kw"]) for row in rows])
        ),
        "battery_command_tracking_error_mae_kw": float(
            np.mean([abs(row["battery_command_tracking_error_kw"]) for row in rows])
        ),
        "battery_balance_correction_mae_kw": float(
            np.mean([abs(row["battery_balance_correction_kw"]) for row in rows])
        ),
        "left_tracking_error_mae_kw": float(np.mean([abs(row["left_tracking_error_kw"]) for row in rows])),
        "right_tracking_error_mae_kw": float(np.mean([abs(row["right_tracking_error_kw"]) for row in rows])),
        "total_balance_error_mae_kw": float(np.mean([abs(row["total_balance_error_kw"]) for row in rows])),
        "left_soc_final": float(rows[-1]["left_soc"]) if rows else None,
        "right_soc_final": float(rows[-1]["right_soc"]) if rows else None,
        "steps": len(rows),
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    return summary
