from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_reward_curve(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    rewards = logs.get("episode_rewards", logs.get("episode_reward", []))
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    if rewards:
        plt.plot(rewards, marker="o" if len(rewards) == 1 else None)
    plt.title("Episode Reward / Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_training_metrics(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    metrics = logs.get("episode_metrics", {})
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tracking = metrics.get("tracking_error_kw", [])
    balance = metrics.get("total_balance_error_kw", []) or metrics.get("balance_error_kw", [])
    soc_error = metrics.get("soc_abs_error", [])

    plt.figure(figsize=(9, 4))
    if tracking:
        plt.plot(tracking, label="tracking_error_kw")
    if balance:
        plt.plot(balance, label="balance_error_kw")
    if soc_error:
        plt.plot(soc_error, label="soc_abs_error")
    plt.title("Episode Control Metrics")
    plt.xlabel("Episode")
    plt.ylabel("Mean Value")
    if tracking or balance or soc_error:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_loss_curve(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    losses = logs.get("loss", [])
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    if losses:
        plt.plot(losses, linewidth=0.8, label="Bellman loss")
        plt.legend()
    plt.title("DQN Bellman Loss")
    plt.xlabel("Training Update")
    plt.ylabel("Bellman Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_step_window_metrics(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    step_window = logs.get("step_window", {})
    steps = step_window.get("end_step", [])
    metrics = step_window.get("metrics", {})
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reward_mean = step_window.get("reward_mean", [])
    tracking = metrics.get("tracking_error_kw", [])
    balance = metrics.get("total_balance_error_kw", []) or metrics.get("balance_error_kw", [])
    soc_error = metrics.get("soc_abs_error", [])

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    if steps and reward_mean:
        axes[0].plot(steps, reward_mean, label="window_reward_mean")
    axes[0].set_title("Step-Window Reward")
    axes[0].set_ylabel("Mean Reward")
    if steps and reward_mean:
        axes[0].legend()

    if steps and tracking:
        axes[1].plot(steps, tracking, label="tracking_error_kw")
    if steps and balance:
        axes[1].plot(steps, balance, label="balance_error_kw")
    if steps and soc_error:
        axes[1].plot(steps, soc_error, label="soc_abs_error")
    axes[1].set_title("Step-Window Control Metrics")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Mean Value")
    if steps and (tracking or balance or soc_error):
        axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_epsilon_curve(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    step_window = logs.get("step_window", {})
    steps = step_window.get("end_step", [])
    epsilons = step_window.get("epsilon", [])
    xlabel = "Training Step"
    if not steps or not epsilons:
        epsilons = logs.get("episode_epsilon", [])
        steps = list(range(len(epsilons)))
        xlabel = "Episode"

    plt.figure(figsize=(9, 4))
    if steps and epsilons:
        plt.plot(steps, epsilons, label="epsilon", color="tab:orange")
        plt.legend()
    plt.title("Epsilon-Greedy Exploration Rate")
    plt.xlabel(xlabel)
    plt.ylabel("Epsilon")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_dqn_convergence(train_logs_json: str | Path, output_png: str | Path) -> Path:
    with Path(train_logs_json).open("r", encoding="utf-8") as f:
        logs = json.load(f)

    rewards = logs.get("episode_rewards", logs.get("episode_reward", []))
    losses = logs.get("loss", [])
    epsilons = logs.get("episode_epsilon", [])
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    if rewards:
        axes[0].plot(range(1, len(rewards) + 1), rewards, color="tab:blue", label="episode reward")
        window = min(10, len(rewards))
        if window >= 2:
            moving = [
                sum(rewards[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
                for i in range(len(rewards))
            ]
            axes[0].plot(range(1, len(moving) + 1), moving, color="tab:orange", label=f"{window}-episode mean")
        axes[0].legend()
    axes[0].set_title("DQN Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.2)

    if losses:
        axes[1].plot(range(1, len(losses) + 1), losses, color="tab:green", linewidth=0.8)
    axes[1].set_title("DQN Bellman Loss")
    axes[1].set_xlabel("Training Update")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.2)

    if epsilons:
        axes[2].plot(range(1, len(epsilons) + 1), epsilons, color="tab:red")
    axes[2].set_title("Epsilon-Greedy Exploration")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
