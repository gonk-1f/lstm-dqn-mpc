"""Deterministic native-SB3 DDPG validation; run this module from the project root."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
# The repository uses an uninstalled src layout. No shell PYTHONPATH is required.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from . import zoo_register  # Register the same environment used by RL Zoo.
from .env import SOC_REFERENCE, SOC_WEIGHT, RAMP_WEIGHT, FAILURE_REWARD
from .validation_plots import save_validation_plots
from .economic_reward import evaluation_identity

ENV_ID = "AggregateShipDDPG-v0"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/processed/operating_dataset_v6_1s"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/ddpg_fixed_parent_005_validation_linear_v2"
ACCOUNTING_FIELDS = (
    "hydrogen_mass_kg", "battery_charge_bus_kwh", "battery_discharge_bus_kwh",
    "h2_cost_cny", "fuel_cell_degradation_cost_cny",
    "battery_degradation_cost_cny", "shore_cost_cny", "c_bat_eq_cny",
    "battery_equivalent_hydrogen_kg", "physical_economic_cost_cny", "total_cost_cny",
    "c_soc_cny", "reward_total_cost_cny",
)
STEP_FIELDS = (
    "episode", "sample_id", "step_index", "interval_start_s", "interval_end_s",
    "step_executed", "actor_action", "load_kw", "requested_fc_kw",
    "fc_actual_kw", "battery_actual_kw", "soc_before", "soc", "delta_fc_kw",
    "soc_penalty", "fc_ramp_penalty", "projected", "feasible_fc_lower_kw",
    "feasible_fc_upper_kw", "reward", "terminal_reason", "terminated",
    "truncated", "evaluation_truncated",
    "fc_efficiency_actual", "equivalent_fc_efficiency",
) + ACCOUNTING_FIELDS


def step_accounting(env, before, info) -> dict:
    """Export the environment's actual economic calculation without recomputing it."""
    if info.get("terminal_reason") == "no_feasible_physical_power":
        return dict.fromkeys(ACCOUNTING_FIELDS)
    battery = info["battery_actual_kw"]
    return {
        **{name: info[name] for name in ACCOUNTING_FIELDS if name not in (
            "battery_charge_bus_kwh", "battery_discharge_bus_kwh"
        )},
        "battery_charge_bus_kwh": max(-battery, 0.0) / 3600.0,
        "battery_discharge_bus_kwh": max(battery, 0.0) / 3600.0,
    }


def _fingerprint(value) -> str:
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for member in item:
                visit(member)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def _model_snapshot(model) -> dict:
    replay = model.replay_buffer
    return {
        "policy_sha256": _fingerprint(model.policy.state_dict()),
        "optimizer_sha256": _fingerprint({
            "actor": model.actor.optimizer.state_dict(),
            "critic": model.critic.optimizer.state_dict(),
        }),
        "num_timesteps": int(model.num_timesteps),
        "n_updates": int(model._n_updates),
        "replay": None if replay is None else {
            "size": int(replay.size()), "pos": int(replay.pos), "full": bool(replay.full)
        },
        "no_gradients": all(p.grad is None for p in model.policy.parameters()),
    }


def run_validation(
    model_path: Path, data_root: Path | None = None,
    output_root: Path | None = None, device: str = "cpu", seed: int = 42,
    episodes: int = 8, max_steps_per_episode: int | None = None,
    vecnormalize_path: Path | None = None, torch_threads: int = 1,
    output_dir: Path | None = None,
) -> dict:
    if not 1 <= episodes <= 8:
        raise ValueError("episodes must be between 1 and 8; the default is Validation8")
    if max_steps_per_episode is not None and max_steps_per_episode < 1:
        raise ValueError("max_steps_per_episode must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    started = time.perf_counter()
    model_path = Path(model_path).resolve()
    data_root = Path(data_root or DEFAULT_DATA_ROOT).resolve()
    output_root = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()
    model_file_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = DDPG.load(str(model_path), device=device, action_noise=None)
    model.policy.set_training_mode(False)
    before_model = _model_snapshot(model)
    env = gym.make(ENV_ID, data_root=data_root, split="validation")
    vessel = env.unwrapped
    normalizer = None
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        reward_metadata = evaluation_identity(model, vessel.reward_identity)
        if vecnormalize_path is None:
            for candidate in (model_path.parent / ENV_ID / "vecnormalize.pkl",
                              model_path.parent / "vecnormalize.pkl"):
                if candidate.is_file():
                    vecnormalize_path = candidate
                    break
        if vecnormalize_path is not None:
            vecnormalize_path = Path(vecnormalize_path).resolve()
            normalizer = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: env]))
            normalizer.training = False
            normalizer.norm_reward = False
        normalization_before = None if normalizer is None else pickle.dumps(
            (normalizer.obs_rms, normalizer.ret_rms)
        )
        expected_ids = [ep.sample_id for ep in vessel.episodes]
        if len(expected_ids) != 8 or len(set(expected_ids)) != 8:
            raise RuntimeError("The existing validation environment must contain eight distinct voyages")
        if output_dir is None:
            timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            output_dir = output_root / f"validation_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=False)
        else:
            # The periodic callback has already created a unique checkpoint node.
            output_dir = Path(output_dir).resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
        rows, plot_data = [], []
        with (output_dir / "validation_step_data.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
            writer.writeheader()
            for episode_index in range(1, episodes + 1):
                obs, reset_info = env.reset(seed=seed if episode_index == 1 else None)
                if reset_info["sample_id"] != expected_ids[episode_index - 1]:
                    raise RuntimeError("Validation voyage order changed")
                initial_soc = vessel.physical.soc
                totals = dict.fromkeys(ACCOUNTING_FIELDS, 0.0)
                trace = {
                    "sample_id": reset_info["sample_id"], "time_s": [0.0],
                    "soc": [initial_soc], "interval_start_s": [], "fc_kw": [],
                    "battery_kw": [], "load_kw": [],
                }
                step_index, episode_reward = 0, 0.0
                while True:
                    before = vessel.physical
                    interval_start = vessel.index
                    model_obs = obs if normalizer is None else normalizer.normalize_obs(obs)
                    action, _ = model.predict(model_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    step_index += 1
                    episode_reward += float(reward)
                    executed = vessel.index > interval_start
                    limited = bool(
                        not (terminated or truncated) and max_steps_per_episode is not None
                        and step_index >= max_steps_per_episode
                    )
                    accounting = step_accounting(vessel, before, info)
                    step = {key: info.get(key) for key in STEP_FIELDS}
                    step.update(
                        episode=episode_index, step_index=step_index,
                        interval_start_s=interval_start,
                        interval_end_s=vessel.index if executed else None,
                        step_executed=executed, actor_action=float(np.asarray(action).item()),
                        soc_before=before.soc, reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated),
                        evaluation_truncated=limited,
                        terminal_reason="validation_step_limit" if limited else info.get("terminal_reason"),
                        **accounting,
                    )
                    writer.writerow(step)
                    if executed:
                        for name in ACCOUNTING_FIELDS:
                            totals[name] += accounting[name]
                        trace["time_s"].append(float(vessel.index))
                        trace["soc"].append(float(vessel.physical.soc))
                        trace["interval_start_s"].append(float(interval_start))
                        trace["fc_kw"].append(float(info["fc_actual_kw"]))
                        trace["battery_kw"].append(float(info["battery_actual_kw"]))
                        trace["load_kw"].append(float(info["load_kw"]))
                    if terminated or truncated or limited:
                        summary = dict(info["vessel_episode"]) if not limited else vessel._episode_summary(
                            False, "validation_step_limit"
                        )
                        success = bool(summary["completed"])
                        summary.update(
                            episode=episode_index, episode_reward=episode_reward,
                            episode_length=step_index, success=success,
                            completion_rate=float(success), soc_initial=initial_soc,
                            soc_delta=float(vessel.physical.soc - initial_soc),
                            evaluation_truncated=limited, **totals,
                            full_voyage_economic_cost_cny=totals["total_cost_cny"] if success else None,
                        )
                        rows.append(summary)
                        plot_data.append(trace)
                        print(
                            f"Validation {episode_index}/{episodes} {summary['sample_id']}: "
                            f"{summary['terminal_reason']}, coverage={summary['coverage']:.3f}, "
                            f"return={episode_reward:.3f}, cost={totals['total_cost_cny']:.3f} CNY",
                            flush=True,
                        )
                        break
        after_model = _model_snapshot(model)
        integrity = {
            "parameters_unchanged": before_model["policy_sha256"] == after_model["policy_sha256"],
            "optimizer_state_unchanged": before_model["optimizer_sha256"] == after_model["optimizer_sha256"],
            "counters_unchanged": all(before_model[key] == after_model[key] for key in ("num_timesteps", "n_updates")),
            "replay_unchanged": before_model["replay"] == after_model["replay"],
            "no_gradients": before_model["no_gradients"] and after_model["no_gradients"],
            "normalization_statistics_unchanged": normalizer is None or normalization_before == pickle.dumps(
                (normalizer.obs_rms, normalizer.ret_rms)
            ),
            "model_file_unchanged": model_file_hash == hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        if not all(integrity.values()):
            raise RuntimeError(f"Validation unexpectedly modified model state: {integrity}")
        validation_set_complete = episodes == 8 and not any(r["evaluation_truncated"] for r in rows)
        if [r["sample_id"] for r in rows] != expected_ids[:episodes]:
            raise RuntimeError("Validation must cover the selected distinct voyages in fixed order")
        successes = sum(r["success"] for r in rows)
        metrics = {
            "output_dir": str(output_dir), "model_path": str(model_path),
            "model_sha256": model_file_hash, "data_root": str(data_root),
            "split": "validation", "deterministic": True, "action_noise": None,
            **reward_metadata,
            "vecnormalize_path": None if normalizer is None else str(vecnormalize_path),
            "vecnormalize_training": False if normalizer is not None else None,
            "vecnormalize_norm_reward": False if normalizer is not None else None,
            "expected_validation_voyages": 8, "validation_voyages": len(rows),
            "validation_set_complete": validation_set_complete,
            "validation_order": expected_ids, "completed": successes,
            "completion_rate": successes / len(rows),
            "mean_coverage": sum(r["coverage"] for r in rows) / len(rows),
            "mean_episode_reward": sum(r["episode_reward"] for r in rows) / len(rows),
            "total_environment_steps": sum(r["episode_length"] for r in rows),
            "total_executed_steps": sum(r["executed_steps"] for r in rows),
            "total_hydrogen_mass_kg": sum(r["hydrogen_mass_kg"] for r in rows),
            "total_executed_economic_cost_cny": sum(r["total_cost_cny"] for r in rows),
            "mean_full_voyage_economic_cost_cny": (
                sum(r["total_cost_cny"] for r in rows) / 8
                if validation_set_complete and successes == 8 else None
            ),
            "economic_cost_scope": "Four economic components including signed instantaneous-efficiency battery hydrogen equivalence. SOC penalty is separate in reward_total_cost_cny; no additional terminal SOC/shore correction or unexecuted failure costs.",
            "unavailable_metrics": {
                "terminal_soc_restoration_cost_cny": "Not performed by the existing environment.",
                "stack_resolved_degradation_cost_cny": "No per-stack physical/health model in this environment.",
            },
            "battery_power_sign": "Positive = bus discharge; negative = bus charge. Energy fields are positive gross bus kWh.",
            "environment_reward": {
                "soc_reference": SOC_REFERENCE, "soc_weight": SOC_WEIGHT,
                "ramp_weight": RAMP_WEIGHT, "failure_reward": FAILURE_REWARD,
            },
            "integrity": integrity, "model_before": before_model, "model_after": after_model,
            "per_voyage": rows,
        }
        with (output_dir / "validation_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_validation_plots(rows, plot_data, output_dir, validation_set_complete)
        metrics["elapsed_s"] = time.perf_counter() - started
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        print(f"Results: {output_dir}", flush=True)
        return metrics
    finally:
        if normalizer is not None:
            normalizer.close()
        else:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Exact periodic checkpoint directory; existing results are not overwritten")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=8, help="1..8; default: all Validation8")
    parser.add_argument("--max-steps-per-episode", type=int, help="Bounded check only; not a full validation")
    parser.add_argument("--vecnormalize-path", type=Path, help="Saved training VecNormalize statistics, if used")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    return run_validation(**vars(args))


if __name__ == "__main__":
    main()
