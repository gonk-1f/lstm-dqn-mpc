from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


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
        "未找到仓库根目录。请在 PyCharm Terminal 中先 cd 到 "
        "lstm-dqn-mpc 的 td3-economic-ems-v2 Worktree 根目录。"
    )


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sb3_manual.td3_aggregate.economic_ems import make_v7_env
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
OUT_DIR = ROOT / "outputs" / "diagnostics" / "td3_v7" / "actor_gradient_decomposition"


def load_model(path: Path, device: str):
    if not path.is_file():
        raise FileNotFoundError(f"模型不存在：{path}")
    model = HuberTD3.load(str(path), device=device)
    model.policy.set_training_mode(False)
    return model


def collect_states(model, data_root: Path, seed: int, max_steps: int):
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{data_root}")

    env = make_v7_env(data_root=data_root, split="train")
    obs, _ = env.reset(seed=seed)
    base = env.unwrapped

    samples = []
    done = False
    step = 0

    while not done and step < max_steps:
        soc = float(base.physical.soc)
        index = int(base.index)
        load_kw = float(base.episode.load_kw[index])
        previous_fc_kw = float(base.physical.previous_fc_kw)

        action, _ = model.predict(obs, deterministic=True)
        samples.append(
            {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "soc": soc,
                "step": step,
                "load_kw": load_kw,
                "previous_fc_kw": previous_fc_kw,
                "actor_action_from_predict": float(
                    np.asarray(action).reshape(-1)[0]
                ),
            }
        )

        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        step += 1

    env.close()

    if not samples:
        raise RuntimeError("没有采集到任何状态。")

    print(f"确定性轨迹采集完成：{len(samples)} 步")
    print("状态来源：固定训练航段 parent_005_seg01")

    return samples


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tensor_stats(x: torch.Tensor):
    y = x.detach().float().reshape(-1)
    return {
        "mean": float(y.mean().cpu()),
        "median": float(y.median().cpu()),
        "min": float(y.min().cpu()),
        "max": float(y.max().cpu()),
        "mean_abs": float(y.abs().mean().cpu()),
        "max_abs": float(y.abs().max().cpu()),
    }


def grad_norm(grads):
    vals = [g.detach().pow(2).sum() for g in grads if g is not None]
    if not vals:
        return 0.0
    return float(torch.sqrt(torch.stack(vals).sum()).cpu())


def analyze_target(model, samples, target_soc: float, max_states: int):
    # Choose real states nearest to the requested SOC. No synthetic state is created.
    selected = sorted(samples, key=lambda x: abs(x["soc"] - target_soc))[:max_states]
    if not selected:
        raise RuntimeError(f"SOC={target_soc}附近没有可用状态。")

    obs_batch = np.stack([x["obs"] for x in selected]).astype(np.float32)
    obs_tensor, _ = model.policy.obs_to_tensor(obs_batch)

    actor = model.actor
    critic = model.critic

    if not isinstance(actor.mu, nn.Sequential):
        raise RuntimeError(
            f"当前Actor的mu不是nn.Sequential：{type(actor.mu).__name__}。"
            "请先检查SB3 Actor网络结构，不能假设最后一层是Tanh。"
        )
    if len(actor.mu) < 2 or not isinstance(actor.mu[-1], nn.Tanh):
        raise RuntimeError(
            "当前Actor的mu最后一层不是nn.Tanh，已停止分析，避免错误拆解梯度。"
        )

    # Reconstruct the exact SB3 actor path before its final Tanh.
    with torch.enable_grad():
        features = actor.extract_features(obs_tensor, actor.features_extractor)
        pre_tanh = actor.mu[:-1](features)
        actions = torch.tanh(pre_tanh)

        # Verify that this manual decomposition exactly matches actor.forward().
        with torch.no_grad():
            normal_actions = actor(obs_tensor)
            reconstruction_error = float(
                (normal_actions - actions.detach()).abs().max().cpu()
            )
        if reconstruction_error > 1e-6:
            raise RuntimeError(
                "手动拆解的Actor输出与actor.forward()不一致，"
                f"最大差值={reconstruction_error:.3e}。停止分析。"
            )

        q1 = critic.q1_forward(obs_tensor, actions)

        # dQ1/da: gradient supplied by the critic.
        d_q_da = torch.autograd.grad(
            q1.sum(), actions, retain_graph=True, create_graph=False
        )[0]

        actor_loss = -q1.mean()

        # dL/da and dL/dz: isolate the action and Tanh stages.
        d_loss_da = torch.autograd.grad(
            actor_loss, actions, retain_graph=True, create_graph=False
        )[0]
        d_loss_dz = torch.autograd.grad(
            actor_loss, pre_tanh, retain_graph=True, create_graph=False
        )[0]

        # Parameter gradients, without optimizer.step() and without changing weights.
        named_params = [
            (name, param)
            for name, param in actor.named_parameters()
            if param.requires_grad
        ]
        parameter_grads = torch.autograd.grad(
            actor_loss,
            [param for _, param in named_params],
            allow_unused=True,
            retain_graph=False,
            create_graph=False,
        )

    tanh_derivative = 1.0 - actions.detach().pow(2)

    # Aggregate parameter-gradient norms by network layer.
    layer_squares: dict[str, float] = {}
    for (name, _), grad in zip(named_params, parameter_grads):
        if grad is None:
            continue
        parts = name.split(".")
        layer_name = ".".join(parts[:2]) if len(parts) >= 2 else name
        layer_squares[layer_name] = (
            layer_squares.get(layer_name, 0.0)
            + float(grad.detach().pow(2).sum().cpu())
        )
    layer_norms = {
        name: float(np.sqrt(value)) for name, value in layer_squares.items()
    }

    # Also probe Critic's local action derivative at fixed normalized actions.
    candidate_actions = [-1.0, -0.5, 0.0, 0.5, 0.9, 0.99, 0.999, 1.0]
    candidate_rows = []
    for candidate in candidate_actions:
        action_probe = torch.full(
            (len(selected), 1),
            float(candidate),
            dtype=torch.float32,
            device=model.device,
            requires_grad=True,
        )
        with torch.enable_grad():
            q_probe = critic.q1_forward(obs_tensor, action_probe)
            dq_probe = torch.autograd.grad(
                q_probe.sum(), action_probe, retain_graph=False
            )[0]
        candidate_rows.append(
            {
                "target_soc": target_soc,
                "candidate_action": candidate,
                "q1_mean": float(q_probe.detach().mean().cpu()),
                "dq1_da_mean": float(dq_probe.detach().mean().cpu()),
                "dq1_da_mean_abs": float(dq_probe.detach().abs().mean().cpu()),
                "dq1_da_min": float(dq_probe.detach().min().cpu()),
                "dq1_da_max": float(dq_probe.detach().max().cpu()),
            }
        )

    state_rows = []
    for i, sample in enumerate(selected):
        state_rows.append(
            {
                "target_soc": target_soc,
                "actual_soc": sample["soc"],
                "soc_abs_error": abs(sample["soc"] - target_soc),
                "step": sample["step"],
                "load_kw": sample["load_kw"],
                "previous_fc_kw": sample["previous_fc_kw"],
                "actor_action_from_predict": sample["actor_action_from_predict"],
                "actor_action_reconstructed": float(actions[i, 0].detach().cpu()),
                "pre_tanh_z": float(pre_tanh[i, 0].detach().cpu()),
                "tanh_derivative": float(tanh_derivative[i, 0].cpu()),
                "q1": float(q1[i, 0].detach().cpu()),
                "dq1_da": float(d_q_da[i, 0].detach().cpu()),
                "d_actor_loss_da": float(d_loss_da[i, 0].detach().cpu()),
                "d_actor_loss_dz": float(d_loss_dz[i, 0].detach().cpu()),
                "saturated_abs_action_ge_099": bool(
                    abs(float(actions[i, 0].detach().cpu())) >= 0.99
                ),
                "tanh_derivative_below_1e_6": bool(
                    float(tanh_derivative[i, 0].cpu()) < 1e-6
                ),
            }
        )

    summary = {
        "target_soc": target_soc,
        "number_of_states": len(selected),
        "max_soc_distance": float(
            max(abs(x["soc"] - target_soc) for x in selected)
        ),
        "actor_reconstruction_max_abs_error": reconstruction_error,
        "actor_action": tensor_stats(actions),
        "pre_tanh_z": tensor_stats(pre_tanh),
        "tanh_derivative": tensor_stats(tanh_derivative),
        "q1": tensor_stats(q1),
        "d_q1_da": tensor_stats(d_q_da),
        "d_actor_loss_da": tensor_stats(d_loss_da),
        "d_actor_loss_dz": tensor_stats(d_loss_dz),
        "actor_parameter_gradient_l2": grad_norm(parameter_grads),
        "actor_parameter_gradient_by_layer_l2": layer_norms,
        "action_saturation_fraction_abs_ge_099": float(
            (actions.detach().abs() >= 0.99).float().mean().cpu()
        ),
        "tanh_derivative_below_1e_6_fraction": float(
            (tanh_derivative < 1e-6).float().mean().cpu()
        ),
        "critic_candidate_action_scan": candidate_rows,
    }

    return state_rows, candidate_rows, summary


def main():
    parser = argparse.ArgumentParser(
        description="分解TD3 Actor梯度：Critic动作梯度、Tanh导数、Actor逐层参数梯度"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=9258)
    parser.add_argument("--max-states-per-target", type=int, default=32)
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_root = args.data_root.resolve()

    print("仓库根目录：", ROOT)
    print("模型：", model_path)
    print("数据：", data_root)
    print("设备：", args.device)
    print("注意：本程序不调用optimizer.step()，不修改、不保存模型参数。")

    model = load_model(model_path, args.device)
    samples = collect_states(model, data_root, args.seed, args.max_steps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_state_rows = []
    all_candidate_rows = []
    summaries = []

    for target in (0.60, 0.70, 0.80):
        state_rows, candidate_rows, summary = analyze_target(
            model, samples, target, args.max_states_per_target
        )
        all_state_rows.extend(state_rows)
        all_candidate_rows.extend(candidate_rows)
        summaries.append(summary)

        print("\n" + "=" * 88)
        print(f"目标SOC={target:.2f}，状态数={summary['number_of_states']}")
        print(
            "Actor动作均值/范围：",
            summary["actor_action"]["mean"],
            summary["actor_action"]["min"],
            summary["actor_action"]["max"],
        )
        print(
            "Tanh前z均值/范围：",
            summary["pre_tanh_z"]["mean"],
            summary["pre_tanh_z"]["min"],
            summary["pre_tanh_z"]["max"],
        )
        print(
            "Tanh导数均值/最小值：",
            summary["tanh_derivative"]["mean"],
            summary["tanh_derivative"]["min"],
        )
        print(
            "Critic dQ1/da均值/绝对值均值：",
            summary["d_q1_da"]["mean"],
            summary["d_q1_da"]["mean_abs"],
        )
        print(
            "Actor dL/dz绝对值均值：",
            summary["d_actor_loss_dz"]["mean_abs"],
        )
        print(
            "Actor参数梯度L2：",
            summary["actor_parameter_gradient_l2"],
        )
        print("逐层梯度L2：", summary["actor_parameter_gradient_by_layer_l2"])
        print(
            "动作饱和比例：",
            summary["action_saturation_fraction_abs_ge_099"],
        )
        print(
            "Tanh导数<1e-6比例：",
            summary["tanh_derivative_below_1e_6_fraction"],
        )

    write_csv(OUT_DIR / "actor_gradient_states.csv", all_state_rows)
    write_csv(OUT_DIR / "critic_action_derivative_scan.csv", all_candidate_rows)

    output = {
        "model": str(model_path),
        "data_root": str(data_root),
        "environment_steps_collected": len(samples),
        "targets": summaries,
        "optimizer_step_called": False,
        "model_parameters_modified": False,
        "checkpoint_saved": False,
    }
    (OUT_DIR / "actor_gradient_decomposition_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n诊断完成。")
    print("状态梯度分解：", OUT_DIR / "actor_gradient_states.csv")
    print("Critic动作导数扫描：", OUT_DIR / "critic_action_derivative_scan.csv")
    print("汇总：", OUT_DIR / "actor_gradient_decomposition_summary.json")


if __name__ == "__main__":
    main()
