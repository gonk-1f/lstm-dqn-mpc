from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from mpc.controllers.upper_mpc import UpperMPCController


def benchmark_dqn_inference(model_path: str | Path, data_csv: str | Path, n_runs: int = 200) -> dict:
    from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
    from envs.ship_env_simple import ShipEnvSimple, SimpleEnvConfig

    df = pd.read_csv(data_csv)
    env = ShipEnvSimple(df, SimpleEnvConfig(random_reset=False, episode_steps=max(2, min(len(df) - 1, n_runs + 1))))
    config = DQNTrainConfig()
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    agent.load(model_path)

    state, _ = env.reset(start_index=0)
    times_ms = []
    for _ in range(n_runs):
        start = time.perf_counter()
        action = agent.greedy_action(state)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        state, _, done, _, _ = env.step(action)
        if done:
            state, _ = env.reset(start_index=0)
    return {
        "runs": len(times_ms),
        "mean_ms": float(np.mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
    }


def benchmark_upper_reference(data_csv: str | Path, n_runs: int = 200, horizon_steps: int = 8) -> dict:
    df = pd.read_csv(data_csv)
    controller = UpperMPCController()
    generator = controller.reference_generator
    times_ms = []
    prev_fc = float(df["fuel_cell_power_total_kw"].iloc[0]) if "fuel_cell_power_total_kw" in df else 0.0
    prev_fc_left = float(df.get("fuel_cell_power_left_kw", pd.Series([prev_fc * 0.5])).iloc[0])
    prev_fc_right = float(df.get("fuel_cell_power_right_kw", pd.Series([prev_fc * 0.5])).iloc[0])
    config = getattr(generator, "config", None)
    horizon = int(getattr(config, "horizon_steps", getattr(config, "prediction_horizon", horizon_steps)))
    dual_supported = hasattr(generator, "generate_dual") and "load_left_kw" in df.columns and "load_right_kw" in df.columns

    for idx in range(min(n_runs, len(df))):
        start = time.perf_counter()
        if dual_supported:
            load_left = df["load_left_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
            load_right = df["load_right_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
            if len(load_left) < horizon:
                load_left = np.pad(load_left, (0, horizon - len(load_left)), mode="edge")
            if len(load_right) < horizon:
                load_right = np.pad(load_right, (0, horizon - len(load_right)), mode="edge")
            if hasattr(generator, "generate_dual_tuple"):
                fc_left, fc_right, _, _ = generator.generate_dual_tuple(
                    load_left_forecast_kw=load_left,
                    load_right_forecast_kw=load_right,
                    soc_left=float(df.get("soc_left", df["soc_mean"]).iloc[idx]),
                    soc_right=float(df.get("soc_right", df["soc_mean"]).iloc[idx]),
                    prev_fc_left_kw=prev_fc_left,
                    prev_fc_right_kw=prev_fc_right,
                )
            else:
                result = generator.generate_dual(
                    load_left_forecast_kw=load_left,
                    load_right_forecast_kw=load_right,
                    soc_left=float(df.get("soc_left", df["soc_mean"]).iloc[idx]),
                    soc_right=float(df.get("soc_right", df["soc_mean"]).iloc[idx]),
                    prev_fc_left_kw=prev_fc_left,
                    prev_fc_right_kw=prev_fc_right,
                )
                fc_left = result.fuel_cell_ref_left_kw
                fc_right = result.fuel_cell_ref_right_kw
            prev_fc_left = fc_left
            prev_fc_right = fc_right
            prev_fc = fc_left + fc_right
        else:
            forecast = df["load_total_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
            if len(forecast) < horizon:
                forecast = np.pad(forecast, (0, horizon - len(forecast)), mode="edge")
            current_soc = float(df["soc_mean"].iloc[idx])
            fc_ref, _ = generator.generate(forecast, current_soc=current_soc, prev_fc_kw=prev_fc)
            prev_fc = fc_ref
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)

    return {
        "mode": controller.mode,
        "runs": len(times_ms),
        "mean_ms": float(np.mean(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
    }
