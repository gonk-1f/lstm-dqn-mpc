from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (
            candidate
            / "experiments"
            / "sb3_manual"
            / "td3_aggregate"
            / "economic_ems.py"
        ).exists():
            return candidate
    raise RuntimeError(
        "未找到仓库根目录。请先在 PyCharm Terminal 中进入 "
        "td3-economic-ems-v2 Worktree 根目录。"
    )


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

from experiments.sb3_manual.td3_aggregate.economic_ems import make_v7_env
from experiments.sb3_manual.td3_aggregate.huber_td3 import HuberTD3


DEFAULT_DATA = ROOT / "data" / "processed" / "operating_dataset_v6_1s"


def write_trace(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="TD3 v7随机预热对照实验：5000步随机探索后每步更新"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=14163,
        help="默认14163=5000随机预热+9163次Critic更新",
    )
    parser.add_argument(
        "--run-name",
        default="seed42_steps14163",
        help="独立输出目录名称，避免覆盖其他实验",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")
    if args.total_steps <= 5000:
        raise ValueError("--total-steps必须大于5000，否则没有足够的梯度训练步数。")
    if not args.run_name or any(ch in args.run_name for ch in r'\/:*?"<>|'):
        raise ValueError("--run-name必须是合法的单层目录名称。")

    out_dir = ROOT / "outputs" / "diagnostics" / "td3_v7" / "random_warmup" / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"输出目录已经存在且非空：{out_dir}\n"
            "请更换 --run-name，避免覆盖已有实验结果。"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "sb3_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = out_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    env = make_v7_env(data_root=data_root, split="train")

    # Confirm the fixed voyage identity before training.
    reset_obs, reset_info = env.reset(seed=args.seed)
    sample_id = reset_info.get("sample_id")
    if sample_id != "parent_005_seg01":
        env.close()
        raise RuntimeError(
            f"预期固定训练航段parent_005_seg01，实际为：{sample_id!r}。"
            "停止实验，避免使用错误航段。"
        )
    print("固定训练航段：", sample_id)
    print("初始SOC：", reset_info.get("soc"))
    print("数据目录：", data_root)
    print("输出目录：", out_dir)

    action_noise = NormalActionNoise(
        mean=np.zeros(1, dtype=np.float32),
        sigma=np.full(1, 0.05, dtype=np.float32),
    )

    # replay_buffer_class is intentionally omitted:
    # SB3 2.9.2 then constructs NStepReplayBuffer automatically from n_steps=32.
    model = HuberTD3(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.001,
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=64,
        tau=0.001,
        gamma=1.0,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        n_steps=32,
        log_td_errors=True,
        require_complete_n_step_batch=True,
        policy_kwargs={
            "net_arch": {
                "pi": [256, 256],
                "qf": [256, 256],
            }
        },
        seed=args.seed,
        device=args.device,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    replay = model.replay_buffer
    print("\n训练配置：")
    print("learning_rate=0.001")
    print("learning_starts=5000")
    print("train_freq=1 step")
    print("gradient_steps=1")
    print("n_steps=", getattr(replay, "n_steps", None))
    print("replay_buffer_gamma=", getattr(replay, "gamma", None))
    print("batch_size=64")
    print("tau=0.001")
    print("gamma=1.0")
    print("policy_delay=2")
    print("noise_std=0.05")
    print("total_steps=", args.total_steps)
    print("预期Critic更新次数约为：", args.total_steps - 5000)
    print("注意：这是随机初始化的新模型，不加载、不覆盖v7第五轮模型。")

    model.learn(
        total_timesteps=args.total_steps,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    model_path = out_dir / "td3_v7_random_warmup5000"
    model.save(str(model_path))

    training_summary = {
        "seed": args.seed,
        "training_voyage": sample_id,
        "total_training_steps": int(args.total_steps),
        "learning_starts": 5000,
        "critic_updates": int(model._n_updates),
        "actor_updates_approx": int(model._n_updates // model.policy_delay),
        "replay_buffer_class": type(model.replay_buffer).__name__,
        "replay_buffer_n_steps": int(model.replay_buffer.n_steps),
        "replay_buffer_gamma": float(model.replay_buffer.gamma),
        "learning_rate": 0.001,
        "train_freq": 1,
        "gradient_steps": 1,
        "batch_size": 64,
        "tau": 0.001,
        "gamma": 1.0,
        "policy_delay": 2,
        "noise_std": 0.05,
        "model_path": str(model_path.with_suffix(".zip")),
    }
    (out_dir / "warmup_training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Separate deterministic evaluation on the same fixed voyage.
    eval_env = make_v7_env(data_root=data_root, split="train")
    obs, eval_reset_info = eval_env.reset(seed=args.seed)
    eval_sample_id = eval_reset_info.get("sample_id")
    if eval_sample_id != "parent_005_seg01":
        eval_env.close()
        env.close()
        raise RuntimeError(
            f"评估航段不是parent_005_seg01：{eval_sample_id!r}"
        )

    done = False
    rewards = []
    actor_actions = []
    socs = []
    fc_powers = []
    battery_powers = []
    trace_rows = []
    final_info = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_scalar = float(np.asarray(action).reshape(-1)[0])
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = bool(terminated or truncated)

        rewards.append(float(reward))
        actor_actions.append(action_scalar)

        if info.get("step_executed", False):
            soc = float(info["soc"])
            fc_kw = float(info["fc_actual_kw"])
            battery_kw = float(info["battery_actual_kw"])
            socs.append(soc)
            fc_powers.append(fc_kw)
            battery_powers.append(battery_kw)
            trace_rows.append(
                {
                    "step": len(trace_rows),
                    "actor_action": action_scalar,
                    "soc": soc,
                    "fc_actual_kw": fc_kw,
                    "battery_actual_kw": battery_kw,
                    "reward": float(reward),
                }
            )
        final_info = info

    vessel_summary = (
        final_info.get("vessel_episode", {}) if final_info else {}
    )
    result = {
        "seed": args.seed,
        "evaluation_voyage": eval_sample_id,
        "evaluation_completed": bool(vessel_summary.get("completed", False)),
        "evaluation_steps": len(rewards),
        "mean_algorithm_reward_per_step": (
            float(np.mean(rewards)) if rewards else None
        ),
        "sum_algorithm_reward": float(np.sum(rewards)) if rewards else None,
        "soc_mean": float(np.mean(socs)) if socs else None,
        "soc_final": float(socs[-1]) if socs else None,
        "soc_upper_dwell_steps": vessel_summary.get("soc_upper_dwell_steps"),
        "soc_upper_dwell_fraction": (
            float(vessel_summary["soc_upper_dwell_steps"]) / len(socs)
            if socs and vessel_summary.get("soc_upper_dwell_steps") is not None
            else None
        ),
        "mean_fc_kw": float(np.mean(fc_powers)) if fc_powers else None,
        "mean_battery_kw": float(np.mean(battery_powers)) if battery_powers else None,
        "actor_action_mean": float(np.mean(actor_actions)) if actor_actions else None,
        "actor_action_std": float(np.std(actor_actions)) if actor_actions else None,
        "actor_action_saturation_fraction_abs_ge_099": (
            float(np.mean(np.abs(actor_actions) >= 0.99))
            if actor_actions else None
        ),
        "hydrogen_mass_kg": vessel_summary.get("hydrogen_mass_kg"),
        "physical_economic_cost_cny": vessel_summary.get("physical_economic_cost_cny"),
        "reward_total_cost_cny": vessel_summary.get("reward_total_cost_cny"),
        "evaluation_scope": "fixed_train_parent_005_seg01_deterministic",
        "model_path": str(model_path.with_suffix(".zip")),
    }

    write_trace(out_dir / "evaluation_trace.csv", trace_rows)
    (out_dir / "warmup_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n训练汇总：")
    print(json.dumps(training_summary, ensure_ascii=False, indent=2))
    print("\n确定性评估汇总：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n输出文件：")
    print("模型：", model_path.with_suffix(".zip"))
    print("训练汇总：", out_dir / "warmup_training_summary.json")
    print("评估汇总：", out_dir / "warmup_evaluation.json")
    print("评估轨迹：", out_dir / "evaluation_trace.csv")

    eval_env.close()
    env.close()


if __name__ == "__main__":
    main()
