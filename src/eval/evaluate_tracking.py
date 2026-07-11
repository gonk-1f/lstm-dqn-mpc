from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
from envs.ship_env_simple import ShipEnvSimple, SimpleEnvConfig


def evaluate_simple_tracking(eval_csv: str | Path, model_path: str | Path, output_path: str | Path | None = None) -> dict:
    df = pd.read_csv(eval_csv)
    env = ShipEnvSimple(df, config=SimpleEnvConfig(random_reset=False, episode_steps=max(1, len(df) - 1)))
    config = DQNTrainConfig()
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    agent.load(model_path)

    state, _ = env.reset()
    done = False
    rewards = []
    tracking_errors = []
    balance_errors = []
    rows = []

    while not done:
        action = agent.greedy_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        tracking_errors.append(abs(info["tracking_error_kw"]))
        balance_errors.append(abs(info["balance_error_kw"]))
        rows.append(info)
        state = next_state

    summary = {
        "reward_sum": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "tracking_error_mae_kw": float(np.mean(tracking_errors)),
        "balance_error_mae_kw": float(np.mean(balance_errors)),
        "steps": len(rewards),
        "truncated": bool(truncated) if len(rewards) > 0 else False,
    }

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    return summary
