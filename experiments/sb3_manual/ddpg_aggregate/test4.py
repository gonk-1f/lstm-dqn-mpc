"""Final deterministic DDPG evaluation on the four frozen Test voyages."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pickle
import time

# Import the existing entry's path bootstrap and accounting/integrity helpers.
from .validate import (
    ACCOUNTING_FIELDS, STEP_FIELDS, DEFAULT_DATA_ROOT, ENV_ID, PROJECT_ROOT,
    C_REF, RAMP_WEIGHT, FAILURE_REWARD,
    _model_snapshot, step_accounting,
)
from .test4_env import AggregateDDPGTestEnv
from .test4_plots import save_test_plots
from .economic_reward import evaluation_identity
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/ddpg_fixed_parent_005_test_economic_v3"


def run_test4(
    model_path: Path, data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT, device: str = "cpu", seed: int = 42,
    test_source_root: Path | None = None, vecnormalize_path: Path | None = None,
    torch_threads: int = 1,
) -> dict:
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    started = time.perf_counter()
    model_path = Path(model_path).resolve()
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = DDPG.load(str(model_path), device=device, action_noise=None)
    model.policy.set_training_mode(False)
    before_model = _model_snapshot(model)
    env = AggregateDDPGTestEnv(Path(data_root), test_source_root=test_source_root)
    normalizer = None
    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
        reward_metadata = evaluation_identity(model, env.reward_identity)
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
        expected_ids = [episode.sample_id for episode in env.episodes]
        if len(expected_ids) != 4 or len(set(expected_ids)) != 4:
            raise RuntimeError("Final evaluation requires exactly four distinct Test voyages")
        timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
        output_dir = Path(output_root).resolve() / f"test_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=False)
        identity = {
            "model_path": str(model_path), "model_sha256": model_hash,
            "model_train_timesteps": before_model["num_timesteps"],
            "model_adam_updates": before_model["n_updates"],
            "deterministic": True, "action_noise": None, "seed": seed,
            "test_order": expected_ids, "test_data": env.test_data_identity,
            "vecnormalize_path": None if normalizer is None else str(vecnormalize_path),
            "purpose": "Final evaluation of this frozen checkpoint; no parameter search or model selection.",
            **reward_metadata,
            "environment_reward": {
                "c_ref": C_REF,
                "c_ref_used_in_reward": False,
                "ramp_weight": RAMP_WEIGHT, "failure_reward": FAILURE_REWARD,
            },
        }
        identity_path = output_dir / "test_identity.json"
        with identity_path.open("x", encoding="utf-8") as handle:
            json.dump(identity, handle, ensure_ascii=False, indent=2, allow_nan=False)
        rows, traces = [], []
        with (output_dir / "test_step_data.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
            writer.writeheader()
            for episode_index, expected_id in enumerate(expected_ids, 1):
                obs, reset_info = env.reset(seed=seed if episode_index == 1 else None)
                if reset_info["sample_id"] != expected_id or reset_info["split"] != "test":
                    raise RuntimeError("Test voyage identity/order changed")
                initial_soc = env.physical.soc
                totals = dict.fromkeys(ACCOUNTING_FIELDS, 0.0)
                trace = {
                    "sample_id": expected_id, "time_s": [0.0], "soc": [initial_soc],
                    "interval_start_s": [], "fc_kw": [], "battery_kw": [], "load_kw": [],
                }
                episode_steps, episode_reward = 0, 0.0
                while True:
                    before = env.physical
                    interval_start = env.index
                    model_obs = obs if normalizer is None else normalizer.normalize_obs(obs)
                    action, _ = model.predict(model_obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_steps += 1
                    episode_reward += float(reward)
                    executed = env.index > interval_start
                    accounting = step_accounting(env, before, info)
                    step = {key: info.get(key) for key in STEP_FIELDS}
                    step.update(
                        episode=episode_index, step_index=episode_steps,
                        interval_start_s=interval_start,
                        interval_end_s=env.index if executed else None,
                        step_executed=executed, actor_action=float(np.asarray(action).item()),
                        soc_before=before.soc, reward=float(reward),
                        terminated=bool(terminated), truncated=bool(truncated),
                        evaluation_truncated=False, **accounting,
                    )
                    writer.writerow(step)
                    if executed:
                        for name in ACCOUNTING_FIELDS:
                            totals[name] += accounting[name]
                        trace["time_s"].append(float(env.index))
                        trace["soc"].append(float(env.physical.soc))
                        trace["interval_start_s"].append(float(interval_start))
                        trace["fc_kw"].append(float(info["fc_actual_kw"]))
                        trace["battery_kw"].append(float(info["battery_actual_kw"]))
                        trace["load_kw"].append(float(info["load_kw"]))
                    if terminated or truncated:
                        summary = dict(info["vessel_episode"])
                        success = bool(summary["completed"])
                        summary.update(
                            episode=episode_index, episode_reward=episode_reward,
                            episode_length=episode_steps, success=success,
                            completion_rate=float(success), soc_initial=initial_soc,
                            soc_delta=float(env.physical.soc - initial_soc),
                            mean_reward_per_step=episode_reward / episode_steps,
                            **totals,
                            full_voyage_economic_cost_cny=totals["total_cost_cny"] if success else None,
                        )
                        rows.append(summary)
                        traces.append(trace)
                        print(
                            f"Test {episode_index}/4 {expected_id}: {summary['terminal_reason']}, "
                            f"coverage={summary['coverage']:.3f}, return={episode_reward:.3f}, "
                            f"cost={totals['total_cost_cny']:.3f} CNY", flush=True,
                        )
                        break
        after_model = _model_snapshot(model)
        integrity = {
            "parameters_unchanged": before_model["policy_sha256"] == after_model["policy_sha256"],
            "optimizer_state_unchanged": before_model["optimizer_sha256"] == after_model["optimizer_sha256"],
            "counters_unchanged": all(before_model[key] == after_model[key]
                                      for key in ("num_timesteps", "n_updates")),
            "replay_unchanged": before_model["replay"] == after_model["replay"],
            "no_gradients": before_model["no_gradients"] and after_model["no_gradients"],
            "normalization_statistics_unchanged": normalizer is None or normalization_before == pickle.dumps(
                (normalizer.obs_rms, normalizer.ret_rms)
            ),
            "model_file_unchanged": model_hash == hashlib.sha256(model_path.read_bytes()).hexdigest(),
        }
        if not all(integrity.values()):
            raise RuntimeError(f"Test evaluation changed model state: {integrity}")
        if [row["sample_id"] for row in rows] != expected_ids or any(
            row["split"] != "test" for row in rows
        ):
            raise RuntimeError("Final Test4 coverage/order/split check failed")
        successes = sum(row["success"] for row in rows)
        metrics = {
            **identity, "output_dir": str(output_dir), "split": "test",
            "test_voyages": len(rows), "test_set_complete": True,
            "opened_test_payloads": env.opened_test_payloads,
            "completed": successes, "completion_rate": successes / 4,
            "mean_coverage": sum(row["coverage"] for row in rows) / 4,
            "mean_episode_reward": sum(row["episode_reward"] for row in rows) / 4,
            "mean_per_episode_step_reward": sum(row["mean_reward_per_step"] for row in rows) / 4,
            "total_environment_steps": sum(row["episode_length"] for row in rows),
            "total_executed_steps": sum(row["executed_steps"] for row in rows),
            "total_hydrogen_mass_kg": sum(row["hydrogen_mass_kg"] for row in rows),
            "total_executed_economic_cost_cny": sum(row["total_cost_cny"] for row in rows),
            "mean_full_voyage_economic_cost_cny": (
                sum(row["total_cost_cny"] for row in rows) / 4 if successes == 4 else None
            ),
            "economic_cost_scope": "total_cost_cny includes H2, fixed-average-efficiency signed battery equivalence, battery degradation and complete four-condition FC degradation. reward_total_cost_cny excludes only FC start/stop degradation; the other six costs are counted once. No independent SOC penalty or unexecuted failure costs.",
            "battery_power_sign": "Positive bus power = discharge; negative = charge. Energy fields are positive gross bus kWh.",
            "vecnormalize_training": False if normalizer is not None else None,
            "vecnormalize_norm_reward": False if normalizer is not None else None,
            "integrity": integrity, "model_before": before_model, "model_after": after_model,
            "per_voyage": rows,
        }
        with (output_dir / "test_summary.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_test_plots(rows, traces, output_dir)
        metrics["elapsed_s"] = time.perf_counter() - started
        (output_dir / "test_metrics.json").write_text(
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-source-root", type=Path)
    parser.add_argument("--vecnormalize-path", type=Path)
    parser.add_argument("--torch-threads", type=int, default=1)
    return run_test4(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
