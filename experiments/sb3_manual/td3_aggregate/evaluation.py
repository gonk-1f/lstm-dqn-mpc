"""Native TD3 deterministic evaluation with unchanged physical CNY accounting.

Train/validation import no Test adapter. The Test adapter is imported only when
``split='test'`` is explicitly requested for final checkpoint evaluation.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.utils import check_for_correct_spaces
import torch

# Import cost field names and plotting only. Importing the existing DDPG
# evaluator would also register its Huber algorithm in RL Zoo's global registry.
from ..ddpg_aggregate.economic_reward import FC_DEGRADATION_COST_FIELDS
from ..ddpg_aggregate.validation_plots import save_validation_plots
from .reward_wrapper import ScaledEconomicReward, assert_model_identity, make_env

FIXED_TRAIN_ID = "parent_005_seg01"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/processed/operating_dataset_v6_1s"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/td3_evaluation"
ACCOUNTING_FIELDS = (
    "hydrogen_mass_kg", "battery_charge_bus_kwh", "battery_discharge_bus_kwh",
    "h2_cost_cny", "fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny",
    "shore_cost_cny", "c_bat_eq_cny", "battery_equivalent_hydrogen_kg",
    "physical_economic_cost_cny", "total_cost_cny", "reward_economic_cost_cny",
    "soc_deviation_penalty_cny", "reward_total_cost_cny",
) + FC_DEGRADATION_COST_FIELDS
SHUTDOWN_FIELDS = (
    "shutdown_executed", "shutdown_time_s", "shutdown_duration_s", "post_shutdown_fc_kw",
    "post_shutdown_battery_kw", "physical_fc_on", "shutdown_delta_fc_kw",
    "shutdown_fc_transient_cost_cny", "shutdown_fc_start_stop_cost_cny", "shutdown_economic_cost_cny",
)
STEP_FIELDS = (
    "episode", "sample_id", "step_index", "interval_start_s", "interval_end_s",
    "step_executed", "actor_action", "load_kw", "requested_fc_kw", "fc_actual_kw",
    "battery_actual_kw", "battery_current_a", "soc_before", "soc", "delta_fc_kw",
    "fc_ramp_penalty", "projected", "feasible_fc_lower_kw", "feasible_fc_upper_kw",
    "reward", "terminal_reason", "terminated", "truncated", "evaluation_truncated",
    "fc_efficiency_actual", "equivalent_fc_efficiency", "fc_start_stop_cycles",
    "fc_cumulative_voltage_loss_uv", "reward_is_failure_override",
) + ACCOUNTING_FIELDS + SHUTDOWN_FIELDS + ("raw_reward", "scaled_reward", "reward_scale")
ECONOMIC_COST_SCOPE = (
    "total_cost_cny includes complete physical economic costs and zero-duration shutdown/cycle settlement, "
    "but no SOC penalty. reward_economic_cost_cny excludes FC cycle cost; reward_total_cost_cny "
    "adds the SOC deviation penalty. Fixed failure reward does not enter the economic ledger. "
    "Shutdown consumes no physical time, hydrogen or battery energy. All costs remain unscaled CNY."
)
_CHECKPOINT_METADATA = {"checkpoint_metadata.json", "model_metadata.json", "reward_identity.json"}
REWARD_MODES = ("economic", "economic_ems", "simple", "control", "control_huber")


def step_accounting(vessel, before, info: dict, accounting_fields=ACCOUNTING_FIELDS) -> dict:
    """Export the original ledger; only derive gross battery bus energy in kWh."""
    battery = info["battery_actual_kw"] if info["step_executed"] else 0.0
    return {
        **{name: info[name] for name in accounting_fields if name not in (
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
            "actor": model.actor.optimizer.state_dict(), "critic": model.critic.optimizer.state_dict(),
        }),
        "num_timesteps": int(model.num_timesteps), "n_updates": int(model._n_updates),
        "replay": None if replay is None else {
            "size": int(replay.size()), "pos": int(replay.pos), "full": bool(replay.full),
        },
        "no_gradients": all(parameter.grad is None for parameter in model.policy.parameters()),
    }


def _check_output_dir(output_dir: Path | None, model_path: Path) -> None:
    """An existing periodic node may contain its checkpoint, never old results."""
    if output_dir is None or not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise FileExistsError(f"Evaluation output is not a directory: {output_dir}")
    unexpected = [
        entry.name for entry in output_dir.iterdir()
        if not entry.is_file() or not (
            entry.resolve() == model_path or entry.name in _CHECKPOINT_METADATA
        )
    ]
    if unexpected:
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {unexpected}")


def _reject_vecnormalize(model_path: Path) -> None:
    for directory in (model_path.parent, model_path.parent / "AggregateShipTD3-v0",
                      model_path.parent / "AggregateShipDDPG-v0"):
        if (directory / "vecnormalize.pkl").is_file():
            raise ValueError("VecNormalize is unsupported in this TD3 experiment")


def _model_integrity(before: dict, after: dict, model_path: Path, model_hash: str) -> dict:
    result = {
        "parameters_unchanged": before["policy_sha256"] == after["policy_sha256"],
        "optimizer_state_unchanged": before["optimizer_sha256"] == after["optimizer_sha256"],
        "counters_unchanged": all(before[name] == after[name] for name in ("num_timesteps", "n_updates")),
        "replay_unchanged": before["replay"] == after["replay"],
        "no_gradients": before["no_gradients"] and after["no_gradients"],
        "model_file_unchanged": model_hash == hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    if not all(result.values()):
        raise RuntimeError(f"TD3 evaluation changed model state: {result}")
    return result


def _rollout(model, env, expected_id: str, episode: int, seed: int | None,
             max_steps: int | None, split: str, writer,
             accounting_fields=ACCOUNTING_FIELDS, step_fields=STEP_FIELDS) -> tuple[dict, dict]:
    vessel = env.unwrapped
    obs, reset_info = env.reset(seed=seed)
    expected_split = "train" if split == "fixed_train" else split
    if reset_info["sample_id"] != expected_id or reset_info["split"] != expected_split:
        raise RuntimeError(f"{split} voyage identity/order changed")
    initial_soc = float(vessel.physical.soc)
    totals = dict.fromkeys(accounting_fields, 0.0)
    trace = {
        "sample_id": expected_id, "time_s": [0.0], "soc": [initial_soc],
        "interval_start_s": [], "fc_kw": [], "battery_kw": [], "load_kw": [],
    }
    calls, raw_sum, algorithm_sum = 0, 0.0, 0.0
    actions, requests, powers, battery_powers = [], [], [], []
    reason = "fixed_eval_step_limit" if split == "fixed_train" else f"{split}_step_limit"
    while True:
        before, start = vessel.physical, vessel.index
        action, _ = model.predict(obs, deterministic=True)
        obs, scaled, terminated, truncated, info = env.step(action)
        calls += 1
        raw = float(info["raw_reward"])
        scale = float(info["reward_scale"])
        if scale != float(env.reward_identity["reward_scale"]) or not math.isclose(float(scaled), raw * scale, rel_tol=1e-12, abs_tol=1e-15):
            raise RuntimeError("Evaluation reward scaling differs from the training contract")
        if float(info["scaled_reward"]) != float(scaled):
            raise RuntimeError("Wrapper info and algorithm reward differ")
        raw_sum += raw
        algorithm_sum += float(scaled)
        executed = vessel.index > start
        limited = bool(not (terminated or truncated) and max_steps is not None and calls >= max_steps)
        accounting = step_accounting(vessel, before, info, accounting_fields)
        row = {key: info.get(key) for key in step_fields}
        row.update(
            episode=episode, step_index=calls, interval_start_s=start,
            interval_end_s=vessel.index if executed else None,
            step_executed=executed, actor_action=float(np.asarray(action).item()),
            soc_before=float(before.soc), reward=raw, raw_reward=raw,
            scaled_reward=float(scaled), reward_scale=scale,
            terminated=bool(terminated), truncated=bool(truncated),
            evaluation_truncated=limited,
            terminal_reason=reason if limited else info.get("terminal_reason"),
            **accounting,
        )
        writer.writerow(row)
        actions.append(float(np.asarray(action).item()))
        for name in accounting_fields:
            totals[name] += accounting[name]
        if info.get("shutdown_executed"):
            trace["shutdown_event"] = {name: info[name] for name in SHUTDOWN_FIELDS}
        if executed:
            requests.append(float(info["requested_fc_kw"]))
            powers.append(float(info["fc_actual_kw"]))
            battery_powers.append(float(info["battery_actual_kw"]))
            trace["time_s"].append(float(vessel.index))
            trace["soc"].append(float(vessel.physical.soc))
            trace["interval_start_s"].append(float(start))
            trace["fc_kw"].append(float(info["fc_actual_kw"]))
            trace["battery_kw"].append(float(info["battery_actual_kw"]))
            trace["load_kw"].append(float(info["load_kw"]))
        if terminated or truncated or limited:
            summary = dict(info["vessel_episode"]) if not limited else vessel._episode_summary(False, reason)
            success = bool(summary["completed"])
            summary.update(
                episode=episode, episode_reward=raw_sum, reward_sum=raw_sum,
                raw_reward_sum=raw_sum, algorithm_reward_sum=algorithm_sum, reward_scale=float(env.reward_identity["reward_scale"]),
                algorithm_reward_version=env.reward_identity["reward_version"],
                mean_reward_per_step=raw_sum / calls,
                mean_algorithm_reward_per_step=algorithm_sum / calls,
                episode_length=calls, success=success, completion_rate=float(success),
                soc_initial=initial_soc, soc_delta=float(vessel.physical.soc - initial_soc),
                evaluation_truncated=limited,
                actor_action_mean=float(np.mean(actions)), actor_action_std=float(np.std(actions)),
                actor_action_saturation_fraction=float(np.mean(np.abs(actions) >= 0.99)),
                mean_requested_fc_kw=float(np.mean(requests)) if requests else None,
                mean_fc_kw=float(np.mean(powers)) if powers else None,
                mean_battery_kw=float(np.mean(battery_powers)) if battery_powers else None,
                **totals,
                full_voyage_economic_cost_cny=totals["total_cost_cny"] if success else None,
            )
            if "reward_fc_variation_penalty" in totals:
                summary["reward_version"] = env.reward_identity["reward_version"]
                summary["fc_ramp_penalty"] = totals["reward_fc_variation_penalty"]
            print(f"TD3 {split} {expected_id}: {summary['terminal_reason']}, "
                  f"coverage={summary['coverage']:.4f}, raw_return={raw_sum:.6f}, "
                  f"algorithm_return={algorithm_sum:.6f}, cost={totals['total_cost_cny']:.6f} CNY", flush=True)
            return summary, trace


def evaluate(
    model_path: Path, split: str = "validation", data_root: Path | None = None,
    output_root: Path | None = None, output_dir: Path | None = None,
    device: str = "cpu", seed: int = 42, max_steps: int | None = None,
    episodes: int | None = None, episode: int | None = None,
    torch_threads: int = 1, test_source_root: Path | None = None,
    reward_mode: str = "economic",
) -> dict:
    """Evaluate a saved model copy; only an explicit step limit shortens voyages.

    Default scopes are one complete fixed Train voyage, all Validation8, or all
    Test4. Physical failure is a legitimate voyage ending. ``episodes`` selects
    a Validation prefix for bounded diagnostics and never changes voyage order.
    Training callbacks invoke this function via an independent CPU subprocess.
    """
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unsupported TD3 reward mode {reward_mode!r}")
    if reward_mode != "economic" and split == "test":
        raise ValueError("The simple-reward experiment only uses Train and Validation")
    if split not in ("fixed_train", "validation", "test"):
        raise ValueError("split must be fixed_train, validation or test")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    if episodes is not None and (split != "validation" or not 1 <= episodes <= 8):
        raise ValueError("episodes is only supported for Validation8 and must be 1..8")
    if test_source_root is not None and split != "test":
        raise ValueError("test_source_root is only supported for final Test4 evaluation")
    model_path = Path(model_path).resolve()
    output_dir = None if output_dir is None else Path(output_dir).resolve()
    _check_output_dir(output_dir, model_path)
    _reject_vecnormalize(model_path)
    torch.set_num_threads(torch_threads)
    started = time.perf_counter()
    data_root = Path(data_root or DEFAULT_DATA_ROOT).resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model_class = TD3
    algorithm = "stable_baselines3.TD3"
    accounting_fields, step_fields = ACCOUNTING_FIELDS, STEP_FIELDS
    cost_scope = ECONOMIC_COST_SCOPE
    if reward_mode == "economic_ems":
        from .economic_ems import (
            make_economic_env, REWARD_COMPONENT_FIELDS as ECONOMIC_REWARD_FIELDS,
            assert_economic_model_identity,
        )
        factory = make_economic_env
        accounting_fields = ACCOUNTING_FIELDS + ECONOMIC_REWARD_FIELDS
        step_fields = STEP_FIELDS + ECONOMIC_REWARD_FIELDS + (
            "actor_reference_fc_kw", "action_mapped_fc_kw", "action_mapping",
            "feasible_interval_width_kw",
        )
        cost_scope = (
            "Training reward is the weighted sum of actual H2 cost, FC transient/low/high variable degradation, "
            "signed battery-equivalent hydrogen cost, battery degradation, per-step SOC deviation and successful "
            "terminal SOC deviation. Start-stop cycle settlement remains in the complete physical ledger only. "
            "All physical costs and rewards are reported in raw CNY; TD3 receives reward_scale=0.001."
        )
        model_class = TD3
        algorithm = "stable_baselines3.TD3"
    elif reward_mode != "economic":
        from .simple_reward import make_simple_env, make_control_env, REWARD_COMPONENT_FIELDS
        if reward_mode == "control_huber":
            from .huber_control import make_huber_control_env
            from .huber_td3 import HuberTD3

            factory = make_huber_control_env
            model_class = HuberTD3
            algorithm = f"{HuberTD3.__module__}.{HuberTD3.__name__}"
        else:
            factory = make_control_env if reward_mode == "control" else make_simple_env
        accounting_fields = ACCOUNTING_FIELDS + REWARD_COMPONENT_FIELDS
        step_fields = STEP_FIELDS + REWARD_COMPONENT_FIELDS
        cost_scope = (
            ("Training reward contains quadratic FC variation and SOC deviation penalties only; H2 cost is excluded. "
             if reward_mode != "simple" else
             "Training reward contains actual H2 cost, quadratic FC variation penalty and SOC deviation penalty only. ")
            +
            "The original complete economic ledger (including degradation and signed battery equivalent H2) "
            "is retained for reporting and is excluded from the simple training reward. "
            "Failure reward is a separate override; zero-duration shutdown adds no simple reward penalty."
        )
    else:
        factory = make_env
    model = model_class.load(str(model_path), device=device, action_noise=None)
    if model.get_vec_normalize_env() is not None:
        raise ValueError("VecNormalize is unsupported in this TD3 experiment")
    model.policy.set_training_mode(False)
    before_model = _model_snapshot(model)
    if split == "test":
        # Explicit final-only import: train and validation never open Test data.
        from ..ddpg_aggregate.test4_env import AggregateDDPGTestEnv
        env = ScaledEconomicReward(AggregateDDPGTestEnv(data_root, test_source_root=test_source_root))
    else:
        env = factory(data_root, split="train" if split == "fixed_train" else "validation",
                       fixed_train_id=FIXED_TRAIN_ID if split == "fixed_train" else None)
    try:
        vessel = env.unwrapped
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        if reward_mode == "economic_ems":
            assert_economic_model_identity(model, env.reward_identity)
        else:
            assert_model_identity(model, env.reward_identity)
        if split == "fixed_train":
            selected = [voyage for voyage in vessel.episodes if voyage.sample_id == FIXED_TRAIN_ID]
            if len(selected) != 1 or len(selected[0].load_kw) != 9258:
                raise RuntimeError("The fixed Train diagnostic requires parent_005_seg01 with 9258 seconds")
            expected_ids = [FIXED_TRAIN_ID]
        else:
            expected_ids = [voyage.sample_id for voyage in vessel.episodes]
            required = 8 if split == "validation" else 4
            if len(expected_ids) != required or len(set(expected_ids)) != required:
                raise RuntimeError(f"{split} requires {required} distinct voyages in fixed order")
        chosen_ids = expected_ids[:episodes] if episodes is not None else expected_ids
        if output_dir is None:
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            output_dir = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve() / f"{split}_{stamp}"
            output_dir.mkdir(parents=True, exist_ok=False)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        rows, traces = [], []
        with (output_dir / f"{split}_step_data.csv").open("x", encoding="utf-8-sig", newline="") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=step_fields)
            writer.writeheader()
            for index, expected_id in enumerate(chosen_ids, 1):
                summary, trace = _rollout(
                    model, env, expected_id,
                    episode if split == "fixed_train" and episode is not None else index,
                    seed if index == 1 else None, max_steps, split, writer,
                    accounting_fields=accounting_fields, step_fields=step_fields,
                )
                rows.append(summary)
                traces.append(trace)
        after_model = _model_snapshot(model)
        integrity = _model_integrity(before_model, after_model, model_path, model_hash)
        if [row["sample_id"] for row in rows] != chosen_ids:
            raise RuntimeError("Evaluation voyage order changed")
        complete = len(chosen_ids) == len(expected_ids) and not any(row["evaluation_truncated"] for row in rows)
        successes = sum(row["success"] for row in rows)
        metrics = {
            "algorithm": algorithm, "output_dir": str(output_dir),
            "model_path": str(model_path), "model_sha256": model_hash, "data_root": str(data_root),
            "split": "train" if split == "fixed_train" else split, "evaluation_split": split,
            "deterministic": True, "action_noise": None, "device": str(model.device), "seed": seed,
            "reward_scale": float(env.reward_identity["reward_scale"]),
            "model_reward_identity": dict(model._td3_reward_identity),
            "evaluation_reward_identity": dict(env.reward_identity),
            "vecnormalize_path": None,
            "reward_units": {"reward_sum": "raw negative economic/SOC reward", "episode_reward": "raw negative economic/SOC reward",
                             "raw_reward_sum": "raw negative economic/SOC reward", "algorithm_reward_sum": "raw_reward_sum * reward_scale",
                             "economic_costs": "unscaled CNY"},
            "completed": successes, "completion_rate": successes / len(rows),
            "mean_coverage": sum(row["coverage"] for row in rows) / len(rows),
            "mean_episode_reward": sum(row["episode_reward"] for row in rows) / len(rows),
            "mean_algorithm_reward_sum": sum(row["algorithm_reward_sum"] for row in rows) / len(rows),
            "raw_reward_sum": sum(row["raw_reward_sum"] for row in rows),
            "algorithm_reward_sum": sum(row["algorithm_reward_sum"] for row in rows),
            "total_environment_steps": sum(row["episode_length"] for row in rows),
            "total_executed_steps": sum(row["executed_steps"] for row in rows),
            "total_hydrogen_mass_kg": sum(row["hydrogen_mass_kg"] for row in rows),
            "total_executed_economic_cost_cny": sum(row["total_cost_cny"] for row in rows),
            "mean_full_voyage_economic_cost_cny": (
                sum(row["total_cost_cny"] for row in rows) / len(rows)
                if complete and successes == len(rows) else None
            ),
            "economic_cost_scope": cost_scope,
            "battery_power_sign": "Positive bus power = discharge; negative = charge. Energy fields are positive gross bus kWh.",
            "integrity": integrity, "model_before": before_model, "model_after": after_model,
        }
        if split == "fixed_train":
            metrics.update(summary=rows[0], evaluation_scope="fixed_train_learning_diagnostic_not_generalization")
        else:
            metrics.update(per_voyage=rows, **{
                f"{split}_voyages": len(rows), f"expected_{split}_voyages": len(expected_ids),
                f"{split}_set_complete": complete, f"{split}_order": expected_ids,
            })
        if split == "test":
            from ..ddpg_aggregate.test4_plots import save_test_plots
            metrics.update(test_data=vessel.test_data_identity, opened_test_payloads=vessel.opened_test_payloads,
                           purpose="Final frozen-checkpoint evaluation; excluded from training and model selection.")
            save_test_plots(rows, traces, output_dir)
        elif split == "validation":
            save_validation_plots(rows, traces, output_dir, complete)
        with (output_dir / f"{split}_summary.csv").open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metrics["elapsed_s"] = time.perf_counter() - started
        with (output_dir / f"{split}_metrics.json").open("x", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"TD3 evaluation results: {output_dir}", flush=True)
        return metrics
    finally:
        env.close()


def argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=REWARD_MODES, default="economic")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, help="Fresh result directory or existing checkpoint-only periodic node")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", "--max-steps-per-episode", dest="max_steps", type=int,
                        help="Bounded diagnostic only; default runs full voyages or true physical failure")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser
