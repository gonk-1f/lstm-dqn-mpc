"""Audit the native SB3 warm-up rollouts without gradient updates."""

from datetime import datetime
from math import isfinite

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from td3_config import (
    DIAGNOSTIC_VOYAGE_ID, SEED, TRAIN_WINDOW_STEPS, load_dataset_bundle, save_json, write_csv,
)
from td3_reference_config import DATA_ROOT, TD3_KWARGS
from td3_reference_model_config import (
    FAILURE_COST_PER_REMAINING_STEP, MODEL_PARAMETERS, MODEL_VERSION,
    NORMAL_STEP_COST_UPPER_BOUND, OUTPUT_ROOT, make_env, observation_scales,
)
from train_td3_reference_model import (
    ACCOUNTING_FIELDS, REWARD_COST_FIELDS, TRACE_FIELDS, build_model, runtime_versions,
)


WARMUP_STEPS = 25_000
COST_TOLERANCE = 1e-5  # SB3's VecEnv carries rewards as float32.
ENDPOINT_TOLERANCE = 1e-6
EQUIVALENT_HYDROGEN_COST = "battery_equivalent_hydrogen_cost"
AUDIT_COST_FIELDS = (EQUIVALENT_HYDROGEN_COST, "total_economic_cost")


def _finite(value) -> bool:
    return value is not None and isfinite(value)


def _step_peak(rows: list[dict], key: str, *, absolute: bool = False) -> dict | None:
    row = max(
        (row for row in rows if _finite(row.get(key))),
        key=lambda row: abs(row[key]) if absolute else row[key],
        default=None,
    )
    if row is None:
        return None
    return {
        "cost": row[key], "episode": row["episode"], "step": row["step"],
        "global_step": row["global_step"], "load_kw": row["load_kw"],
        "fc_actual_kw": row["fc_actual_kw"], "battery_actual_kw": row["battery_actual_kw"],
        "soc": row["soc"], "fc_efficiency": row["fc_efficiency"],
    }


def _endpoint_fractions(rows: list[dict]) -> dict:
    fractions = [row["actor_action_fraction"] for row in rows if not row["failure"]]
    if not fractions:
        return {"action_lower_endpoint_fraction": None, "action_upper_endpoint_fraction": None}
    return {
        "action_lower_endpoint_fraction": sum(a <= ENDPOINT_TOLERANCE for a in fractions) / len(fractions),
        "action_upper_endpoint_fraction": sum(a >= 1.0 - ENDPOINT_TOLERANCE for a in fractions) / len(fractions),
    }


class WarmupTrace(BaseCallback):
    def __init__(self, budget: int):
        super().__init__()
        self.budget = budget
        self.steps: list[dict] = []
        self.episodes: list[dict] = []
        self.episode_step = 0
        self.episode_reward = 0.0
        self.stop_reason: str | None = None

    def _inspect_step(self, row: dict) -> None:
        if row["failure"]:
            self.stop_reason = "physical_failure"
            return
        fields = dict.fromkeys((
            "reward", *REWARD_COST_FIELDS, *AUDIT_COST_FIELDS,
            *(key for key in ACCOUNTING_FIELDS if row.get(key) is not None),
            "hydrogen_rate_kg_per_hour", "battery_equivalent_rate_kg_per_hour",
        ))
        invalid = [key for key in fields if not _finite(row.get(key))]
        if invalid:
            row["invalid_accounting_fields"] = ",".join(invalid)
            self.stop_reason = "non_finite_reward_or_accounting"
            return
        cost = sum(row[key] for key in REWARD_COST_FIELDS)
        row["normal_step_cost"] = cost
        if not isfinite(cost):
            self.stop_reason = "non_finite_reward_or_accounting"
        elif abs(row["reward"] + cost) > COST_TOLERANCE:
            self.stop_reason = "four_term_reward_mismatch"
        elif cost > NORMAL_STEP_COST_UPPER_BOUND + COST_TOLERANCE:
            self.stop_reason = "normal_step_cost_above_bound"
        elif cost > FAILURE_COST_PER_REMAINING_STEP + COST_TOLERANCE:
            self.stop_reason = "normal_step_cost_exceeds_failure_step_cost"

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])
        reward = float(self.locals["rewards"][0])
        self.episode_step += 1
        self.episode_reward += reward
        row = {
            "episode": len(self.episodes) + 1, "step": self.episode_step,
            "global_step": self.num_timesteps, "sample_id": info["voyage_id"],
            "window_start_step": info["window_start_step"],
            "window_end_step": info["window_end_step"],
            **{key: info.get(key) for key in TRACE_FIELDS},
            "reward": reward, "terminated": info["terminated"], "truncated": info["truncated"],
        }
        self._inspect_step(row)
        row["diagnostic_stop_reason"] = self.stop_reason
        self.steps.append(row)
        if done or self.stop_reason is not None or self.num_timesteps == self.budget:
            window_rows = self.steps[-self.episode_step:]
            normal_cost_peak = _step_peak(window_rows, "normal_step_cost")
            equivalent_cost_peak = _step_peak(window_rows, EQUIVALENT_HYDROGEN_COST, absolute=True)
            self.episodes.append({
                "episode": row["episode"], "window_start_step": row["window_start_step"],
                "window_end_step": row["window_end_step"],
                "attempted_steps": self.episode_step,
                "executed_steps": self.episode_step - int(info["failure"]),
                "initial_soc": window_rows[0]["soc_before"], "final_soc": info["soc"],
                "total_reward": self.episode_reward, "failure": info["failure"],
                "terminal_reason": info["terminal_reason"] or self.stop_reason or "diagnostic_step_budget",
                "diagnostic_stop_reason": self.stop_reason,
                "largest_normal_step_cost": normal_cost_peak["cost"] if normal_cost_peak else None,
                "largest_absolute_battery_equivalent_hydrogen_cost": (
                    abs(equivalent_cost_peak["cost"]) if equivalent_cost_peak else None
                ),
                **_endpoint_fractions(window_rows),
            })
            self.episode_step = 0
            self.episode_reward = 0.0
        return self.stop_reason is None and (done or self.num_timesteps < self.budget)


def main() -> None:
    versions = runtime_versions()
    if (versions["stable-baselines3"], versions["gymnasium"]) != ("2.2.1", "0.29.1"):
        raise RuntimeError("Use the isolated environment from requirements-td3-reference.txt.")
    torch.set_num_threads(1)
    set_random_seed(SEED, using_cuda=torch.cuda.is_available())
    bundle = load_dataset_bundle(DATA_ROOT)
    train_id = DIAGNOSTIC_VOYAGE_ID
    profile = bundle.train_profiles[bundle.train_ids.index(train_id)]
    load_scale, delta_scale = observation_scales(bundle.train_profiles)
    env = Monitor(make_env(
        [profile], [train_id], load_scale_kw=load_scale, load_delta_scale_kw=delta_scale,
        train_window_steps=TRAIN_WINDOW_STEPS,
    ))
    run_dir = OUTPUT_ROOT / datetime.now().strftime("warmup_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        model = build_model(env)
        budget = min(WARMUP_STEPS, model.learning_starts)
        model.set_logger(configure(str(run_dir), []))
        trace = WarmupTrace(budget)
        # Use the same rollout setup as learn(), then collect without calling train().
        _, callback = model._setup_learn(budget, trace)
        callback.on_training_start(locals(), globals())
        vec_env = model.get_env()
        while model.num_timesteps < budget:
            rollout = model.collect_rollouts(
                vec_env, callback, model.train_freq, model.replay_buffer,
                action_noise=model.action_noise, learning_starts=model.learning_starts,
            )
            if not rollout.continue_training:
                break
        callback.on_training_end()
        write_csv(run_dir / "warmup_trajectory.csv", trace.steps)
        write_csv(run_dir / "warmup_episodes.csv", trace.episodes)
        executed = [row for row in trace.steps if not row["failure"]]
        stop_step = None
        if trace.stop_reason is not None:
            stop_step = {
                key: None if isinstance(value, float) and not isfinite(value) else value
                for key, value in trace.steps[-1].items()
            }
        completed = sum(
            row["terminal_reason"] == "training_window_end" and row["diagnostic_stop_reason"] is None
            for row in trace.episodes
        )
        normal_cost_peak = _step_peak(executed, "normal_step_cost")
        equivalent_cost_peak = _step_peak(executed, EQUIVALENT_HYDROGEN_COST, absolute=True)
        save_json(run_dir / "warmup_summary.json", {
            "model_version": MODEL_VERSION, "model_parameters": MODEL_PARAMETERS,
            "versions": versions, "device": str(model.device), "seed": SEED,
            "voyage_id": train_id, "data_root": str(DATA_ROOT.resolve()), "td3_parameters": TD3_KWARGS,
            "step_budget": budget, "attempted_steps": model.num_timesteps,
            "executed_steps": len(executed),
            "completed_windows": completed,
            "replay_size": model.replay_buffer.size(), "gradient_updates": model._n_updates,
            "diagnostic_passed": (
                model.num_timesteps == WARMUP_STEPS and completed == 5
                and model._n_updates == 0 and trace.stop_reason is None
            ),
            "diagnostic_stop_reason": trace.stop_reason,
            "first_diagnostic_stop": stop_step,
            "normal_step_cost_upper_bound": NORMAL_STEP_COST_UPPER_BOUND,
            "normal_step_cost_diagnostic_limit": FAILURE_COST_PER_REMAINING_STEP,
            "largest_normal_step_cost": normal_cost_peak["cost"] if normal_cost_peak else None,
            "largest_normal_step_cost_step": normal_cost_peak,
            "largest_absolute_battery_equivalent_hydrogen_cost_step": equivalent_cost_peak,
            "source_cost_peaks": {
                key: _step_peak(executed, key) for key in (*REWARD_COST_FIELDS, "total_economic_cost")
            },
            "fc_zero_output_steps": sum(row["fc_actual_kw"] == 0.0 for row in executed),
            "first_failure": stop_step if trace.stop_reason == "physical_failure" else None,
            **_endpoint_fractions(executed),
        })
        print(f"Warm-up diagnostic complete after {model.num_timesteps} steps: {run_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
