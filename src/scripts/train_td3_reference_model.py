"""Train native TD3 with the adapted SimulationCode vessel environment."""

from datetime import datetime

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import check_for_correct_spaces, set_random_seed

from callbacks import ShipEMSTrainingCallback
from td3_config import (
    DIAGNOSTIC_VOYAGE_ID, DT_S, SEED, TRAIN_WINDOW_STEPS, VALIDATION_INTERVAL,
    load_dataset_bundle, save_json, write_csv,
)
from td3_reference_config import (
    ACTION_NOISE_SIGMA, DATA_ROOT, DIAGNOSTIC_INTERVAL, EPISODES, TD3_KWARGS,
    TOTAL_TIMESTEPS,
)
from td3_reference_model_config import (
    MODEL_PARAMETERS, MODEL_VERSION, OUTPUT_ROOT, SOC_WEIGHT, make_env,
    observation_scales,
)
from train_td3_reference import ReferenceDiagnostics, build_model, runtime_versions


COST_FIELDS = (
    "hydrogen_cost", "battery_equivalent_hydrogen_cost", "battery_degradation_cost",
    "fc_degradation_cost_raw", "fc_degradation_cost", "soc_penalty",
    "soc_boundary_penalty", "total_economic_cost", "failure_penalty",
)
ACCOUNTING_FIELDS = (
    *COST_FIELDS, "fc_start_count", "hydrogen_kg", "battery_equivalent_hydrogen_kg",
    "fc_startup_uv", "fc_low_uv", "fc_high_uv", "fc_transient_uv", "battery_qloss",
)
TRACE_FIELDS = (
    "load_kw", "requested_fc_kw", "actor_action_fraction",
    "fc_actual_kw", "requested_battery_kw", "battery_actual_kw", "delta_fc_kw", "soc_before", "soc",
    "power_balance_error_kw", "unserved_load_kw", "excess_power_kw", "soc_boundary_violation",
    "fc_efficiency", "hydrogen_rate_kg_per_hour",
    "battery_equivalent_rate_kg_per_hour", "battery_current_a", "battery_c_rate",
    *ACCOUNTING_FIELDS, "failure", "terminal_reason",
)
LEGACY_ACCOUNTING_FIELDS = (
    "fc_start_stop_cycles", "total_economic_cost_rmb", "voyage_hydrogen_kg",
    "fc_total_degradation_cost_rmb", "voyage_battery_degradation_cost_rmb",
    "battery_equivalent_hydrogen_kg", "battery_equivalent_hydrogen_cost_rmb",
)


def evaluate_voyage(model, env, voyage_index: int, seed: int | None, *, capture_steps: bool):
    obs, reset_info = env.reset(seed=seed, options={"voyage_index": voyage_index})
    initial_soc = reset_info["initial_soc"]
    totals = dict.fromkeys(ACCOUNTING_FIELDS, 0.0)
    steps = []
    soc_values = [initial_soc]
    fc_values, battery_values, fc_changes = [], [], []
    total_reward = 0.0
    expected_steps = reset_info["episode_length"]
    for step in range(1, expected_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        for key in totals:
            totals[key] += info.get(key) or 0.0
        soc_values.append(info["soc"])
        if not info["failure"]:
            fc_values.append(info["fc_actual_kw"])
            battery_values.append(info["battery_actual_kw"])
            fc_changes.append(info["delta_fc_kw"])
        if capture_steps:
            steps.append({
                "sample_id": reset_info["voyage_id"], "step": step, "reward": reward,
                **{key: info.get(key) for key in TRACE_FIELDS},
            })
        if terminated or truncated:
            break
    if not (terminated or truncated):
        raise RuntimeError(f"Evaluation voyage did not terminate at step {expected_steps}.")

    completed = step == expected_steps and terminated and info["terminal_reason"] == "voyage_end"
    return {
        "sample_id": reset_info["voyage_id"],
        "expected_steps": expected_steps, "executed_steps": len(fc_values),
        "environment_steps": step, "coverage": len(fc_values) / expected_steps,
        "completed": completed, "failure": info["failure"], "terminal_reason": info["terminal_reason"],
        "episode_reward": total_reward, "mean_reward_per_step": total_reward / step,
        "initial_soc": initial_soc, "final_soc": soc_values[-1],
        "soc_min": min(soc_values), "soc_max": max(soc_values),
        "mean_fc_kw": float(np.mean(fc_values)) if fc_values else None,
        "mean_battery_kw": float(np.mean(battery_values)) if battery_values else None,
        "fc_ramp_rms_kw": float(np.sqrt(np.mean(np.square(fc_changes)))) if fc_changes else None,
        **{f"sum_{key}": value for key, value in totals.items()},
    }, steps


class ModelDiagnostics(ReferenceDiagnostics):
    def __init__(self, *args, observation_scales: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_scales = observation_scales

    def _on_training_start(self) -> None:
        ShipEMSTrainingCallback._on_training_start(self)
        self.episode_costs = dict.fromkeys(ACCOUNTING_FIELDS, 0.0)

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        for key in self.episode_costs:
            self.episode_costs[key] += info.get(key) or 0.0
        keep_training = ShipEMSTrainingCallback._on_step(self)
        if self.locals["dones"][0]:
            row = self.episode_rows[-1]
            for key in LEGACY_ACCOUNTING_FIELDS:
                del row[key]
            row.update({f"sum_{key}": value for key, value in self.episode_costs.items()})
            self.episode_costs = dict.fromkeys(ACCOUNTING_FIELDS, 0.0)
        return keep_training

    def _evaluate(self, profiles, sample_ids, *, capture_steps: bool = False):
        env = make_env(profiles, sample_ids, **self.observation_scales)
        was_training = self.model.policy.training
        try:
            check_for_correct_spaces(env, self.model.observation_space, self.model.action_space)
            return [
                evaluate_voyage(
                    self.model, env, index, self.seed if index == 0 else None,
                    capture_steps=capture_steps,
                )
                for index in range(len(sample_ids))
            ]
        finally:
            env.close()
            self.model.policy.set_training_mode(was_training)

    def _run_fixed_train_diagnostic(self, episode_number: int) -> None:
        (result, steps), = self._evaluate(
            [self.train_profile], [self.train_id], capture_steps=True,
        )
        actions = np.array([row["actor_action_fraction"] for row in steps])
        counts, edges = np.histogram(actions, bins=20, range=(0.0, 1.0))
        rated_kw = float(self.model.action_space.high[0])
        result.update({
            "training_episode": episode_number,
            "critic_updates": self.model._n_updates,
            "critic_loss": self.logger.name_to_value.get("train/critic_loss"),
            "actor_command_mean_kw": float(actions.mean()) * rated_kw,
            "actor_command_std_kw": float(actions.std()) * rated_kw,
            "actor_action_saturation_fraction": float(np.mean(np.abs(2 * actions - 1) >= 0.99)),
            **self._replay_q_metrics(),
        })
        self.train_diagnostic_rows.append(result)
        write_csv(self.output_dir / f"power_episode_{episode_number:04d}.csv", steps)
        write_csv(
            self.output_dir / f"actor_actions_episode_{episode_number:04d}.csv",
            [
                {"command_lower_kw": lo * rated_kw, "command_upper_kw": hi * rated_kw, "count": int(count)}
                for lo, hi, count in zip(edges[:-1], edges[1:], counts)
            ],
        )
        self._save_power_plot(episode_number, result, steps)

    def _run_validation(self, episode_number: int) -> None:
        rows = [row for row, _ in self._evaluate(self.validation_profiles, self.validation_ids)]
        for row in rows:
            row["training_episode"] = episode_number
        self.validation_rows.extend(rows)
        completed = [row for row in rows if row["completed"]]
        metrics = {
            "training_episode": episode_number,
            "completed_voyages": len(completed),
            "completion_rate": len(completed) / len(rows),
            "mean_duration_coverage": float(np.mean([row["coverage"] for row in rows])),
            "mean_episode_reward": float(np.mean([row["episode_reward"] for row in rows])),
            "mean_total_economic_cost_completed_only": (
                float(np.mean([row["sum_total_economic_cost"] for row in completed]))
                if completed else None
            ),
        }
        report = {"metrics": metrics, "voyages": rows}
        save_json(self.output_dir / f"validation_episode_{episode_number:04d}.json", report)
        score = (len(completed), metrics["mean_episode_reward"])
        if self.best_score is None or score > self.best_score:
            self.best_score, self.best_episode = score, episode_number
            self.model.save(self.output_dir / "best_model")
            save_json(self.output_dir / "best_validation.json", report)


def main() -> None:
    versions = runtime_versions()
    if (versions["stable-baselines3"], versions["gymnasium"]) != ("2.2.1", "0.29.1"):
        raise RuntimeError("Use the isolated environment from requirements-td3-reference.txt.")
    torch.set_num_threads(1)
    set_random_seed(SEED, using_cuda=torch.cuda.is_available())
    bundle = load_dataset_bundle(DATA_ROOT)
    train_id = DIAGNOSTIC_VOYAGE_ID
    train_profile = bundle.train_profiles[bundle.train_ids.index(train_id)]
    load_scale, delta_scale = observation_scales(bundle.train_profiles)
    scales = {"load_scale_kw": load_scale, "load_delta_scale_kw": delta_scale}
    run_dir = OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    env = Monitor(
        make_env([train_profile], [train_id], train_window_steps=TRAIN_WINDOW_STEPS, **scales),
        filename=str(run_dir / "monitor"),
        info_keywords=("window_start_step", "window_end_step"),
    )
    try:
        model = build_model(env)
        model.set_logger(configure(str(run_dir / "sb3_logs"), ["stdout", "csv"]))
        callback = ModelDiagnostics(
            train_profile, train_id, bundle.validation_profiles, bundle.validation_ids, run_dir,
            max_episodes=EPISODES, reward_plot_interval=EPISODES + 1,
            train_diagnostic_interval=EPISODES + 1, validation_interval=EPISODES + 1,
            seed=SEED, observation_scales=scales,
        )
        save_json(run_dir / "training_config.json", {
            **TD3_KWARGS,
            "algorithm": "native SB3 TD3", "critic_loss": "mse",
            "versions": versions, "device": str(model.device), "torch_cuda": torch.version.cuda,
            "data_root": str(DATA_ROOT.resolve()), "split_manifest": bundle.manifest_path,
            "model_version": MODEL_VERSION, "model_parameters": MODEL_PARAMETERS,
            "action_mapping": "absolute_fc_command_kw", "observation_normalization": scales,
            "soc_weight": SOC_WEIGHT, "seed": SEED, "action_noise_sigma": ACTION_NOISE_SIGMA,
            "train_voyage_ids": [train_id], "step_duration_s": DT_S,
            "train_window_steps": TRAIN_WINDOW_STEPS, "max_episodes": EPISODES,
            "total_timesteps": TOTAL_TIMESTEPS, "diagnostic_interval": DIAGNOSTIC_INTERVAL,
            "validation_interval": VALIDATION_INTERVAL, "validation_ids": bundle.validation_ids,
        })
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, log_interval=5)
        save_json(run_dir / "training_summary.json", {
            "completed_training_windows": sum(row["completed"] for row in callback.episode_rows),
            "physical_failure_episodes": sum(row["failure"] for row in callback.episode_rows),
            "total_timesteps": model.num_timesteps, "critic_updates": model._n_updates,
            "actor_and_target_updates": model._n_updates // model.policy_delay,
            "best_validation_episode": callback.best_episode,
            "last_model": str(run_dir / "last_model.zip"),
        })
    finally:
        env.close()


if __name__ == "__main__":
    main()
