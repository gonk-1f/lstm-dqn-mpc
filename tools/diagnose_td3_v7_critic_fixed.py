from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "experiments" / "sb3_manual" / "td3_aggregate" / "economic_ems.py").exists():
            return candidate
    raise RuntimeError(
        "未找到仓库根目录。请先在 PyCharm Terminal 中 cd 到 lstm-dqn-mpc 仓库根目录。"
    )


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sb3_manual.td3_aggregate.economic_ems import make_v7_env
from experiments.sb3_manual.td3_aggregate.economic_action_wrappers import (
    FeasibleIntervalActionWrapper,
)
from experiments.sb3_manual.td3_aggregate.huber_td3 import HuberTD3


DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "td3_v7_fixed_parent005_n32_prevfc_500ep"
    / "td3"
    / "AggregateShipTD3EconomicV7-v0_2"
    / "fixed_train_evaluation"
    / "episode_0005"
    / "model.zip"
)
DEFAULT_DATA = ROOT / "data" / "processed" / "operating_dataset_v6_1s"
OUT_ROOT = ROOT / "outputs" / "diagnostics" / "td3_v7"


def get_env(data_root: Path):
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")
    return make_v7_env(data_root=data_root, split="train")


def load_model(model_path: Path, device: str):
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{model_path}\n"
            "请用 --model 指定本地实际存在的 model.zip。"
        )
    model = HuberTD3.load(str(model_path), device=device)
    model.policy.set_training_mode(False)
    return model


def find_action_wrapper(env):
    current = env
    while current is not None:
        if isinstance(current, FeasibleIntervalActionWrapper):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("环境包装链中未找到 FeasibleIntervalActionWrapper")


def save_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def critic_scan(args):
    """Evaluate Q1/Q2 for candidate normalized actions at real observed states."""
    out_dir = OUT_ROOT / "critic_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, args.device)
    env = get_env(args.data_root)
    action_wrapper = find_action_wrapper(env)

    action_grid = np.array(
        [-1.0, -0.75, -0.5, 0.0, 0.5, 0.75, 1.0], dtype=np.float32
    )
    targets = [0.60, 0.70, 0.80]
    best = {target: None for target in targets}

    obs, _ = env.reset(seed=args.seed)
    done = False
    steps = 0

    while not done and steps < args.max_steps:
        obs_arr = np.asarray(obs, dtype=np.float32).copy()
        soc = float(env.unwrapped.physical.soc)

        mapped_kw = []
        mapping_metadata = []
        for a in action_grid:
            _, metadata = action_wrapper._map(np.array([a], dtype=np.float32))
            mapped_kw.append(metadata["action_mapped_fc_kw"])
            mapping_metadata.append(metadata)

        base = env.unwrapped
        load_kw = float(base.episode.load_kw[base.index])
        previous_fc_kw = float(base.physical.previous_fc_kw)
        # Read the exact feasible interval from the v7 action wrapper itself.
        # This avoids duplicating its onboard-load normalization and feasibility logic.
        lower_kw = float(mapping_metadata[0]["feasible_fc_lower_kw"])
        upper_kw = float(mapping_metadata[0]["feasible_fc_upper_kw"])

        for target in targets:
            diff = abs(soc - target)
            old = best[target]
            if old is None or diff < old["soc_diff"]:
                best[target] = {
                    "target_soc": target,
                    "soc": soc,
                    "soc_diff": diff,
                    "step": steps,
                    "obs": obs_arr,
                    "mapped_kw": mapped_kw,
                    "load_kw": load_kw,
                    "previous_fc_kw": previous_fc_kw,
                    "feasible_lower_kw": float(lower_kw),
                    "feasible_upper_kw": float(upper_kw),
                }

        action, _ = model.predict(obs_arr, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        steps += 1

    rows = []
    model.policy.set_training_mode(False)

    for target in targets:
        sample = best[target]
        if sample is None:
            print(f"[WARN] 未采集到 SOC={target:.2f} 附近的状态")
            continue

        obs_batch = np.repeat(sample["obs"][None, :], len(action_grid), axis=0)
        obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)
        action_tensor = torch.as_tensor(
            action_grid.reshape(-1, 1),
            dtype=torch.float32,
            device=model.device,
        )

        with torch.inference_mode():
            q_values = model.critic(obs_tensor, action_tensor)
            actor_action = float(model.actor(obs_tensor[:1]).detach().cpu().item())

        q1 = q_values[0].detach().cpu().numpy().reshape(-1)
        q2 = q_values[1].detach().cpu().numpy().reshape(-1)

        print("\n" + "=" * 105)
        print(
            f"目标SOC={target:.2f} | 实际SOC={sample['soc']:.6f} | "
            f"航段步={sample['step']} | 负载={sample['load_kw']:.3f} kW"
        )
        print(
            f"上一时刻FC={sample['previous_fc_kw']:.3f} kW | "
            f"可行区间=[{sample['feasible_lower_kw']:.3f}, "
            f"{sample['feasible_upper_kw']:.3f}] kW | "
            f"Actor输出={actor_action:.6f}"
        )
        print(
            f"{'action':>9} {'mapped_FC_kW':>15} {'Q1':>16} {'Q2':>16} {'Q_min':>16}"
        )

        for i, a in enumerate(action_grid):
            qmin = min(float(q1[i]), float(q2[i]))
            mapped = sample["mapped_kw"][i]
            print(
                f"{float(a):>9.2f} "
                f"{(float(mapped) if mapped is not None else float('nan')):>15.3f} "
                f"{float(q1[i]):>16.8f} "
                f"{float(q2[i]):>16.8f} "
                f"{qmin:>16.8f}"
            )
            rows.append(
                {
                    "target_soc": target,
                    "actual_soc": sample["soc"],
                    "step": sample["step"],
                    "load_kw": sample["load_kw"],
                    "previous_fc_kw": sample["previous_fc_kw"],
                    "feasible_lower_kw": sample["feasible_lower_kw"],
                    "feasible_upper_kw": sample["feasible_upper_kw"],
                    "actor_action": actor_action,
                    "candidate_action": float(a),
                    "mapped_fc_kw": mapped,
                    "q1": float(q1[i]),
                    "q2": float(q2[i]),
                    "q_min": qmin,
                }
            )

        best_q1 = int(np.argmax(q1))
        best_q2 = int(np.argmax(q2))
        best_min = int(np.argmax(np.minimum(q1, q2)))
        print(
            "Q1最优动作=", float(action_grid[best_q1]),
            "| Q2最优动作=", float(action_grid[best_q2]),
            "| min(Q1,Q2)最优动作=", float(action_grid[best_min]),
        )

    save_csv(out_dir / "critic_action_scan.csv", rows)
    env.close()
    print(f"\n结果已保存：{out_dir / 'critic_action_scan.csv'}")


def grad_norm(parameters) -> float:
    values = [
        p.grad.detach().pow(2).sum()
        for p in parameters
        if p.grad is not None
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu().item())


def parameter_delta(parameters, before) -> float:
    values = [
        (p.detach() - old).pow(2).sum()
        for p, old in zip(parameters, before)
    ]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu().item())


def grad_audit(args):
    """Continue a loaded checkpoint briefly, measuring real gradients and parameter deltas."""
    out_dir = OUT_ROOT / "gradient_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "sb3_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3.common.logger import configure

    model = load_model(args.model, args.device)
    env = get_env(args.data_root)

    replay = model.replay_buffer
    if replay is None:
        raise RuntimeError("加载模型后 replay_buffer=None，无法执行梯度审计。")
    if replay.size() != 0:
        raise RuntimeError(
            f"预期加载的模型没有恢复Replay Buffer，但当前size={replay.size()}。"
            "为避免污染诊断，请先确认模型文件和Replay Buffer状态。"
        )

    model.set_env(env, force_reset=True)
    model.set_logger(configure(str(log_dir), ["stdout"]))

    original_train = model.train
    rows = []

    def wrapped_train(self, gradient_steps: int, batch_size: int = 100):
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        actor_before = [p.detach().clone() for p in actor_params]
        critic_before = [p.detach().clone() for p in critic_params]
        updates_before = int(self._n_updates)

        original_train(gradient_steps, batch_size)

        updates_after = int(self._n_updates)
        if updates_after <= updates_before:
            return

        critic_g = grad_norm(critic_params)
        critic_d = parameter_delta(critic_params, critic_before)

        actor_updated = (
            updates_before // self.policy_delay
            < updates_after // self.policy_delay
        )
        if actor_updated:
            actor_g = grad_norm(actor_params)
            actor_d = parameter_delta(actor_params, actor_before)
        else:
            actor_g = None
            actor_d = None

        row = {
            "n_updates": updates_after,
            "critic_gradient_l2": critic_g,
            "critic_parameter_delta_l2": critic_d,
            "actor_updated": actor_updated,
            "actor_gradient_l2": actor_g,
            "actor_parameter_delta_l2": actor_d,
        }
        rows.append(row)

        if len(rows) <= 10 or len(rows) % 50 == 0:
            print(
                f"[UPDATE {updates_after}] "
                f"Critic grad={critic_g:.6e}, delta={critic_d:.6e}, "
                f"Actor updated={actor_updated}, "
                f"Actor grad={actor_g if actor_g is not None else 'N/A'}, "
                f"Actor delta={actor_d if actor_d is not None else 'N/A'}"
            )

    model.train = types.MethodType(wrapped_train, model)

    print("模型：", args.model)
    print("Replay Buffer类型：", type(model.replay_buffer).__name__)
    print("Replay Buffer初始size：", model.replay_buffer.size())
    print("n_steps：", model.n_steps)
    print("train_freq：", model.train_freq)
    print("gradient_steps：", model.gradient_steps)
    print("policy_delay：", model.policy_delay)
    print("开始额外采集环境步数：", args.steps)
    print("注意：本诊断只修改内存中的模型，不保存或覆盖原checkpoint。")

    updates_before = int(model._n_updates)
    timesteps_before = int(model.num_timesteps)
    model.learn(
        total_timesteps=args.steps,
        reset_num_timesteps=False,
        progress_bar=False,
    )
    updates_after = int(model._n_updates)

    save_csv(out_dir / "gradient_audit.csv", rows)

    actor_rows = [r for r in rows if r["actor_updated"]]
    result = {
        "model": str(args.model),
        "additional_env_steps": int(model.num_timesteps - timesteps_before),
        "critic_updates_before": updates_before,
        "critic_updates_after": updates_after,
        "critic_updates_during_audit": updates_after - updates_before,
        "actor_updates_during_audit": len(actor_rows),
        "first_update": rows[0] if rows else None,
        "last_update": rows[-1] if rows else None,
        "gradient_csv": str(out_dir / "gradient_audit.csv"),
        "checkpoint_saved": False,
    }
    (out_dir / "gradient_audit_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n审计汇总：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    env.close()


def warmup_experiment(args):
    """Fresh TD3 run: 5,000 random warm-up steps, then one update per step."""
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed

    out_dir = OUT_ROOT / "random_warmup_5000"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "sb3_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    env = get_env(args.data_root)

    action_noise = NormalActionNoise(
        mean=np.zeros(1, dtype=np.float32),
        sigma=np.full(1, 0.05, dtype=np.float32),
    )

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
        tensorboard_log=str(out_dir / "tensorboard"),
    )
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    print("\n随机预热对照实验配置：")
    print("learning_starts=5000")
    print("train_freq=1 step")
    print("gradient_steps=1")
    print("n_steps=32")
    print("batch_size=64")
    print("gamma=1.0")
    print("policy_delay=2")
    print("noise_std=0.05")
    print("总环境步数：", args.total_steps)
    print("预期梯度更新次数约为：", max(0, args.total_steps - 5000))
    print("这是从随机初始化网络开始的新实验，不会加载或覆盖v7 checkpoint。")

    model.learn(
        total_timesteps=args.total_steps,
        reset_num_timesteps=True,
        progress_bar=False,
    )

    model_path = out_dir / "td3_v7_warmup5000"
    model.save(str(model_path))

    # Separate deterministic evaluation on the same fixed voyage.
    eval_env = get_env(args.data_root)
    obs, _ = eval_env.reset(seed=args.seed)
    done = False
    rewards = []
    socs = []
    fc_powers = []
    battery_powers = []
    final_info = None

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = bool(terminated or truncated)
        rewards.append(float(reward))
        if info.get("step_executed", False):
            socs.append(float(info["soc"]))
            fc_powers.append(float(info["fc_actual_kw"]))
            battery_powers.append(float(info["battery_actual_kw"]))
        final_info = info

    summary = final_info.get("vessel_episode", {}) if final_info else {}
    result = {
        "seed": args.seed,
        "learning_starts": 5000,
        "total_training_steps": int(args.total_steps),
        "critic_updates": int(model._n_updates),
        "actor_updates_approx": int(model._n_updates // model.policy_delay),
        "evaluation_completed": bool(summary.get("completed", False)),
        "evaluation_steps": len(rewards),
        "mean_algorithm_reward_per_step": (
            float(np.mean(rewards)) if rewards else None
        ),
        "soc_mean": float(np.mean(socs)) if socs else None,
        "soc_final": float(socs[-1]) if socs else None,
        "soc_upper_dwell_steps": summary.get("soc_upper_dwell_steps"),
        "soc_upper_dwell_fraction": (
            float(summary["soc_upper_dwell_steps"]) / len(socs)
            if socs and summary.get("soc_upper_dwell_steps") is not None
            else None
        ),
        "mean_fc_kw": float(np.mean(fc_powers)) if fc_powers else None,
        "mean_battery_kw": float(np.mean(battery_powers)) if battery_powers else None,
        "hydrogen_mass_kg": summary.get("hydrogen_mass_kg"),
        "physical_economic_cost_cny": summary.get("physical_economic_cost_cny"),
        "reward_total_cost_cny": summary.get("reward_total_cost_cny"),
        "actor_action_mean": (
            float(np.mean([
                # actor outputs are separately scanned below on evaluation states
                0.0
            ]))
            if False else None
        ),
        "model_path": str(model_path.with_suffix(".zip")),
        "evaluation_scope": "fixed_train_parent_005_seg01_deterministic",
    }
    (out_dir / "warmup_evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n随机预热实验确定性评估：")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n模型已保存：{model_path}.zip")
    print(f"评估结果：{out_dir / 'warmup_evaluation.json'}")

    eval_env.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="TD3 v7 三项诊断：Critic Q扫描、梯度审计、随机预热对照"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="v7 model.zip 路径",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA,
        help="处理后的1秒采样数据目录",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto / cuda / cpu",
    )
    parser.add_argument("--seed", type=int, default=42)

    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("critic", help="诊断Critic对不同动作的Q值")
    p1.add_argument("--max-steps", type=int, default=9258)
    p1.set_defaults(func=critic_scan)

    p2 = sub.add_parser("grad", help="记录真实梯度范数和参数变化")
    p2.add_argument("--steps", type=int, default=500)
    p2.set_defaults(func=grad_audit)

    p3 = sub.add_parser("warmup", help="从头训练：随机预热5000步")
    p3.add_argument(
        "--total-steps",
        type=int,
        default=14163,
        help="默认14163步，约对应9163次Critic更新，与v7第一轮更新量接近",
    )
    p3.set_defaults(func=warmup_experiment)

    args = parser.parse_args()
    args.model = args.model.resolve()
    args.data_root = args.data_root.resolve()

    print("仓库根目录：", ROOT)
    print("数据目录：", args.data_root)
    print("设备：", args.device)
    print("命令：", args.command)

    if args.command in ("critic", "grad") and not args.model.exists():
        raise FileNotFoundError(
            f"默认模型路径不存在：{args.model}\n"
            "请通过 --model 指定实际的 model.zip。"
        )
    args.func(args)


if __name__ == "__main__":
    main()
