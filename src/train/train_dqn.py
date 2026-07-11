from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from dqn.agents.dqn_agent import DQNAgent, DQNTrainConfig
from dqn.memory.replay_buffer import ReplayBuffer
from dqn.policies.epsilon_greedy import EpsilonGreedyPolicy
from envs.ship_env_dual_side import ShipEnvDualSide, load_dual_side_env_config
from envs.ship_env_simple import ShipEnvSimple, SimpleEnvConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ModuleNotFoundError:
        pass


def _resolve_run_settings(
    df: pd.DataFrame,
    config: DQNTrainConfig,
    episode_steps: int | None,
    random_reset: bool | None,
    full_data_pass: bool,
) -> tuple[DQNTrainConfig, int | None, bool | None]:
    """Resolve long-run data coverage without changing the default smoke-test path."""
    if full_data_pass:
        steps_for_one_pass = max(1, len(df) - 1)
        config = replace(config, max_steps=steps_for_one_pass)
        episode_steps = steps_for_one_pass
        random_reset = False
    return config, episode_steps, random_reset


def make_simple_env(
    train_csv: str | Path,
    episode_steps: int | None = None,
    random_reset: bool | None = None,
) -> ShipEnvSimple:
    df = pd.read_csv(train_csv)
    env_config = SimpleEnvConfig()
    if episode_steps is not None:
        env_config.episode_steps = int(episode_steps)
    if random_reset is not None:
        env_config.random_reset = bool(random_reset)
    return ShipEnvSimple(df, config=env_config)


def make_dual_side_env(
    train_csv: str | Path,
    episode_steps: int | None = None,
    random_reset: bool | None = None,
) -> ShipEnvDualSide:
    df = pd.read_csv(train_csv)
    env_config = load_dual_side_env_config(Path.cwd())
    if episode_steps is not None:
        env_config.episode_steps = int(episode_steps)
    if random_reset is not None:
        env_config.random_reset = bool(random_reset)
    return ShipEnvDualSide(df, config=env_config)


def _train_env(env, output_root: Path, config: DQNTrainConfig) -> dict:
    agent = DQNAgent(env.state_dim, env.action_dim, config)
    replay_buffer = ReplayBuffer(config.buffer_size)
    policy = EpsilonGreedyPolicy(config.epsilon_start, config.epsilon_min, config.epsilon_decay)

    state, _ = env.reset()
    episode_reward = 0.0
    episode_length = 0
    best_episode_reward: float | None = None
    reward_component_keys = [
        "fuel_cost_norm", "fc_smooth_norm", "soc_protect_norm", "soc_dev_norm",
        "soc_direction_norm", "soc_battery_direction_norm", "fc_soc_direction_norm",
        "soc_correction_credit_norm", "fc_trim_norm", "fc_trim_penalty_norm",
        "battery_deg_norm", "battery_use_raw_norm", "battery_use_soc_factor",
        "smooth_norm", "battery_smooth_norm", "soc_boundary_norm",
        "action_smooth_norm", "soc_balance_norm", "battery_balance_norm",
        "constraint_penalty_norm", "total_reward",
    ]
    episode_metric_keys = [
        "tracking_error_kw", "balance_error_kw", "total_balance_error_kw",
        "left_tracking_error_kw", "right_tracking_error_kw", "soc_abs_error",
    ]
    episode_reward_components = {key: 0.0 for key in reward_component_keys}
    episode_metric_sums = {key: 0.0 for key in episode_metric_keys}
    logs: dict[str, object] = {
        "loss": [], "q_value_mean": [], "q_value_std": [],
        "target_q_mean": [], "target_q_std": [],
        "episode_reward": [], "episode_reward_mean": [], "episode_length": [],
        "episode_epsilon": [],
        "episode_reward_components": {key: [] for key in reward_component_keys},
        "episode_metrics": {key: [] for key in episode_metric_keys},
        "step_window": {
            "end_step": [], "reward_mean": [], "reward_sum": [],
            "window_length": [], "epsilon": [],
            "metrics": {key: [] for key in episode_metric_keys},
        },
        "kan_grid_update_steps": [],
        "target_sync_steps": [],
    }
    window_reward_sum = 0.0
    window_metric_sums = {key: 0.0 for key in episode_metric_keys}
    window_count = 0

    def flush_step_window(end_step: int) -> None:
        nonlocal window_reward_sum, window_metric_sums, window_count
        if window_count <= 0:
            return
        sw = logs["step_window"]
        sw["end_step"].append(float(end_step))
        sw["reward_mean"].append(float(window_reward_sum / window_count))
        sw["reward_sum"].append(float(window_reward_sum))
        sw["window_length"].append(float(window_count))
        sw["epsilon"].append(float(policy.epsilon))
        for mk in episode_metric_keys:
            sw["metrics"][mk].append(float(window_metric_sums[mk] / window_count))
        window_reward_sum = 0.0
        window_metric_sums = {key: 0.0 for key in episode_metric_keys}
        window_count = 0

    for step in range(config.max_steps):
        agent.observe_states_for_normalization(state)
        greedy_action = agent.greedy_action(state)
        action = policy.select_action(greedy_action, env.action_dim, warmup=step < config.warmup_steps)
        next_state, reward, done, truncated, info = env.step(action)
        replay_buffer.push(state, action, reward, done, next_state)
        agent.observe_states_for_normalization(next_state)
        if step >= config.warmup_steps:
            policy.step()

        state = next_state
        episode_reward += reward
        window_reward_sum += reward
        window_count += 1
        episode_length += 1
        for key in reward_component_keys:
            episode_reward_components[key] += float(info.get(key, 0.0))
        for key in episode_metric_keys:
            if key == "soc_abs_error":
                soc_v = info.get("soc"); soc_r = info.get("soc_ref")
                if soc_v is not None and soc_r is not None:
                    mv = abs(float(soc_v) - float(soc_r))
                    episode_metric_sums[key] += mv
                    window_metric_sums[key] += mv
            else:
                mv = float(info.get(key, 0.0))
                episode_metric_sums[key] += mv
                window_metric_sums[key] += mv

        if len(replay_buffer) >= config.batch_size and step >= config.warmup_steps:
            batch = replay_buffer.sample(config.batch_size)
            loss = agent.update(batch)
            if (config.kan_grid_update_enabled and config.kan_grid_update_interval_steps > 0
                    and (step + 1) % int(config.kan_grid_update_interval_steps) == 0
                    and (config.kan_grid_update_until_step <= 0 or (step + 1) <= int(config.kan_grid_update_until_step))):
                grid_states = replay_buffer.sample_states(int(config.kan_grid_update_samples))
                if agent.update_kan_grid_from_samples(grid_states, update_target=True):
                    logs["kan_grid_update_steps"].append(float(step + 1))
            # Hard-sync target network every target_sync_interval update steps
            loss_count = len(logs["loss"]) + 1
            if config.target_sync_interval > 0 and loss_count % config.target_sync_interval == 0:
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())
                logs["target_sync_steps"].append(float(step + 1))
            logs["loss"].append(loss)
            diag = getattr(agent, "latest_update_diagnostics", {})
            logs["q_value_mean"].append(float(diag.get("q_value_mean", np.nan)))
            logs["q_value_std"].append(float(diag.get("q_value_std", np.nan)))
            logs["target_q_mean"].append(float(diag.get("target_q_mean", np.nan)))
            logs["target_q_std"].append(float(diag.get("target_q_std", np.nan)))

        if config.log_window_steps > 0 and (step + 1) % config.log_window_steps == 0:
            flush_step_window(step + 1)

        if done:
            logs["episode_reward"].append(float(episode_reward))
            logs["episode_reward_mean"].append(float(episode_reward / max(episode_length, 1)))
            logs["episode_length"].append(float(episode_length))
            logs["episode_epsilon"].append(float(policy.epsilon))
            for key in reward_component_keys:
                logs["episode_reward_components"][key].append(
                    float(episode_reward_components[key] / max(episode_length, 1)))
            for key in episode_metric_keys:
                logs["episode_metrics"][key].append(float(episode_metric_sums[key] / max(episode_length, 1)))

            if best_episode_reward is None or episode_reward > best_episode_reward:
                best_episode_reward = episode_reward
                agent.save(output_root / "best_ship_dqn.pt")

            state, _ = env.reset()
            episode_reward = 0.0; episode_length = 0
            episode_reward_components = {key: 0.0 for key in reward_component_keys}
            episode_metric_sums = {key: 0.0 for key in episode_metric_keys}

    flush_step_window(config.max_steps)
    agent.save(output_root / "last_ship_dqn.pt")
    summary = {
        "episodes": len(logs["episode_reward"]),
        "loss_updates": len(logs["loss"]),
        "best_episode_reward": best_episode_reward,
        "mean_episode_reward": float(np.mean(logs["episode_reward_mean"])) if logs["episode_reward_mean"] else None,
        "mean_tracking_error_kw": float(np.mean(logs["episode_metrics"]["tracking_error_kw"]))
        if logs["episode_metrics"]["tracking_error_kw"] else None,
        "mean_balance_error_kw": float(np.mean(logs["episode_metrics"]["balance_error_kw"]))
        if logs["episode_metrics"]["balance_error_kw"] else None,
        "mean_total_balance_error_kw": float(np.mean(logs["episode_metrics"]["total_balance_error_kw"]))
        if logs["episode_metrics"]["total_balance_error_kw"] else None,
        "mean_soc_abs_error": float(np.mean(logs["episode_metrics"]["soc_abs_error"]))
        if logs["episode_metrics"]["soc_abs_error"] else None,
        "final_epsilon": policy.epsilon,
        "device": agent.device_name,
        "hyperparameters": {
            "discount": config.discount, "lr": config.lr,
            "batch_size": config.batch_size, "warmup_steps": config.warmup_steps,
            "buffer_size": config.buffer_size,
            "epsilon_start": config.epsilon_start, "epsilon_min": config.epsilon_min,
            "epsilon_decay": config.epsilon_decay,
            "target_sync_interval": config.target_sync_interval,
            "log_window_steps": config.log_window_steps,
            "grad_clip_norm": config.grad_clip_norm,
            "loss_type": config.loss_type,
            "double_dqn": config.double_dqn, "dueling_dqn": config.dueling_dqn,
            "state_normalization_enabled": config.state_normalization_enabled,
            "state_normalization_min_count": config.state_normalization_min_count,
            "kan_grid_update_enabled": config.kan_grid_update_enabled,
            "kan_grid_update_interval_steps": config.kan_grid_update_interval_steps,
            "kan_grid_update_until_step": config.kan_grid_update_until_step,
            "kan_grid_update_samples": config.kan_grid_update_samples,
            **agent.network_info,
        },
    }
    with (output_root / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    norm_stats = agent.input_normalization_stats()
    if norm_stats is not None:
        with (output_root / "input_normalization_stats.json").open("w", encoding="utf-8") as f:
            json.dump(norm_stats, f, ensure_ascii=False, indent=2)
    with (output_root / "train_logs.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False)
    return summary


def train_simple_dqn(train_csv: str | Path, output_dir: str | Path, config: DQNTrainConfig | None = None) -> dict:
    config = config or DQNTrainConfig()
    set_seed(config.seed)

    env = make_simple_env(train_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    return _train_env(env, output_root, config)


def train_dual_side_dqn(
    train_csv: str | Path,
    output_dir: str | Path,
    config: DQNTrainConfig | None = None,
    episode_steps: int | None = None,
    random_reset: bool | None = None,
    full_data_pass: bool = False,
    num_episodes: int | None = None,
    episode_hours: float = 6.0,
    fc_trim_candidates: tuple[float, ...] | None = None,
    batt_trim_candidates: tuple[float, ...] | None = None,
    reward_mode: str = "legacy",
) -> dict:
    config = config or DQNTrainConfig()
    set_seed(config.seed)

    df_raw = pd.read_csv(train_csv)
    # Expand 30s → 5s by repeating each row 6 times (ZOH for MPC references)
    repeat = 6  # 30s / 5s
    df = df_raw.loc[df_raw.index.repeat(repeat)].reset_index(drop=True)
    df["dqn_substep"] = np.tile(np.arange(repeat), len(df_raw))
    df["dqn_time_seconds"] = np.arange(len(df)) * 5.0
    df["sample_time_seconds"] = 30.0
    dt = 5.0  # DQN control seconds
    ep_steps = int(episode_hours * 3600 / dt) if episode_hours > 0 else (episode_steps or 4320)

    # Build episode start indices: every ep_steps rows
    total_rows = len(df) - ep_steps - 1
    num_episodes_available = max(1, total_rows // ep_steps)
    episode_starts = tuple(range(0, num_episodes_available * ep_steps, ep_steps))

    # Use remaining rows to augment validation set
    used_rows = num_episodes_available * ep_steps + ep_steps  # last episode needs full window
    overflow_rows = max(0, len(df) - used_rows)

    # Resolve total training steps from num_episodes
    if num_episodes is not None and num_episodes > 0:
        config = replace(config, max_steps=num_episodes * ep_steps)
    elif full_data_pass:
        config = replace(config, max_steps=len(df) - 1)

    env_config = load_dual_side_env_config(Path.cwd())
    env_config.episode_steps = int(ep_steps)
    env_config.random_reset = True
    env_config.soc_random_init_enabled = True
    env_config.soc_random_init_low = 0.55
    env_config.soc_random_init_high = 0.75
    env_config.episode_start_indices = episode_starts
    env_config.reward_mode = str(reward_mode)
    if fc_trim_candidates is not None:
        env_config.fc_trim_candidates_kw = tuple(fc_trim_candidates)
    if batt_trim_candidates is not None:
        env_config.batt_trim_candidates_kw = tuple(batt_trim_candidates)

    env = ShipEnvDualSide(df, config=env_config)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = _train_env(env, output_root, config)
    summary.update({
        "training_data_rows": int(len(df)),
        "training_max_steps": int(config.max_steps),
        "episode_steps": int(ep_steps),
        "episode_hours": float(episode_hours),
        "num_episodes_available": int(num_episodes_available),
        "episode_start_indices_count": len(episode_starts),
        "soc_random_init_enabled": True,
        "overflow_rows": int(overflow_rows),
        "loss_type": config.loss_type,
        "target_sync_interval": config.target_sync_interval,
    })
    with (output_root / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
