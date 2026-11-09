"""TD3 economic EMS diagnostics and model selection for the fixed voyage."""

from collections import deque
import json
import math
import shutil
import subprocess
import sys

from stable_baselines3.common.buffers import NStepReplayBuffer

from .callbacks import (
    ECONOMIC_FIELDS, FIXED_TRAIN_ID, FIXED_TRAIN_STEPS, PROJECT_ROOT, TD3TrainingCallback,
)
from .economic_ems import (
    N_STEP_RETURN, REWARD_COMPONENT_FIELDS, V6_ENV_ID, V6_REWARD_COMPONENT_FIELDS, V6_REWARD_VERSION,
)
from .huber_td3 import HuberTD3

from v3.control import SOC_MAX, SOC_MIN


class EconomicEMSTrainingCallback(TD3TrainingCallback):
    economic_fields = tuple(
        name for name in ECONOMIC_FIELDS if name != "soc_deviation_penalty_cny"
    ) + REWARD_COMPONENT_FIELDS
    reward_component_fields = REWARD_COMPONENT_FIELDS
    environment_id = "AggregateShipTD3Economic-v0"
    algorithm_name = "experiments.sb3_manual.td3_aggregate.huber_td3.HuberTD3"
    evaluation_extra_args = ("--reward-mode", "economic_ems")
    reward_scale = 0.001
    require_fixed_train = True

    def _on_training_start(self):
        identity = self.training_env.get_attr("reward_identity")[0]
        if identity.get("env_id") != self.environment_id or not isinstance(self.model, HuberTD3):
            raise ValueError("Economic EMS training requires its registered environment and HuberTD3")
        replay = require_economic_n_step_replay(self.model)
        voyages = self.training_env.get_attr("episodes")[0]
        fixed = [voyage for voyage in voyages if voyage.sample_id == FIXED_TRAIN_ID]
        if len(fixed) != 1 or len(fixed[0].load_kw) != FIXED_TRAIN_STEPS:
            raise ValueError(f"Economic EMS requires {FIXED_TRAIN_ID} with {FIXED_TRAIN_STEPS} steps")
        super()._on_training_start()
        self.logger.info(f"Replay: {type(replay).__name__}, n_steps={replay.n_steps}, gamma={replay.gamma}")
        self.logger.record("replay/n_steps", replay.n_steps)
        self.logger.record("replay/gamma", replay.gamma)
        self.pause_request = self.directory / "pause_after_episode.request"
        self._recent_raw_rewards = deque(getattr(self.model, "_td3_recent_raw_rewards", ()), maxlen=20)
        self._previous_battery_kw = None
        self._max_fc_kw = None

    def _observe_step(self, info):
        if info["sample_id"] != FIXED_TRAIN_ID:
            raise ValueError(f"Unexpected fixed training voyage: {info['sample_id']}")
        super()._observe_step(info)
        action = float(info["actor_action"])
        self._accumulate("actor_action", action)
        self._accumulate("actor_action_saturated", float(abs(action) >= 0.99))
        self._accumulate("actor_reference_fc_kw", info["actor_reference_fc_kw"])
        self._accumulate("feasible_interval_width_kw", info["feasible_interval_width_kw"])
        if info["action_mapped_fc_kw"] is not None:
            self._accumulate("action_mapped_fc_kw", info["action_mapped_fc_kw"])
        self._accumulate("warmup_step", float(self.num_timesteps <= self.model.learning_starts))
        if info["step_executed"]:
            fc_kw, battery_kw = info["fc_actual_kw"], info["battery_actual_kw"]
            self._max_fc_kw = fc_kw if self._max_fc_kw is None else max(self._max_fc_kw, fc_kw)
            self._accumulate("fc_actual_kw_squared", fc_kw ** 2)
            if self._previous_battery_kw is not None:
                self._accumulate("battery_delta_kw_squared", (battery_kw - self._previous_battery_kw) ** 2)
            self._previous_battery_kw = battery_kw

    def _record_episode(self, summary):
        calls = self._episode_counts["raw_reward"]
        executed = summary["executed_steps"]
        reason = summary["terminal_reason"]
        sample_id = summary["sample_id"]
        expected_steps = FIXED_TRAIN_STEPS
        if summary["total_steps"] != expected_steps or summary["soc_initial"] != 0.6:
            raise ValueError(f"{sample_id} requires {expected_steps} steps and initial SOC=0.6")
        if summary["completed"]:
            valid_ending = reason == "completed" and calls == executed == expected_steps
        else:
            valid_ending = (
                reason == "no_feasible_physical_power"
                and 0 <= executed < expected_steps
                and calls == executed + 1
            )
        if not valid_ending:
            raise ValueError(
                f"Unexpected economic EMS episode ending: {reason}, {executed} physical steps, {calls} calls"
            )
        super()._record_episode(summary)
        self._previous_battery_kw = None
        self._max_fc_kw = None

    def _additional_episode_fields(self, summary):
        counts, sums = self._episode_counts, self._episode_sums

        def mean(name):
            return sums[name] / counts[name] if counts[name] else None

        raw_return = float(summary["raw_reward_sum"])
        self._recent_raw_rewards.append(raw_return)
        self.model._td3_recent_raw_rewards = list(self._recent_raw_rewards)
        fc_mean = mean("fc_actual_kw")
        fc_mean_square = mean("fc_actual_kw_squared")
        battery_delta_mean_square = mean("battery_delta_kw_squared")
        return {
            "mean_reward_per_step": raw_return / counts["raw_reward"],
            "raw_reward_moving_mean_20": (
                math.fsum(self._recent_raw_rewards) / 20 if len(self._recent_raw_rewards) == 20 else None
            ),
            "warmup_steps": int(sums["warmup_step"]),
            "max_fc_kw": self._max_fc_kw,
            "std_fc_kw": (
                math.sqrt(max(0.0, fc_mean_square - fc_mean ** 2)) if fc_mean is not None else None
            ),
            "fc_ramp_rms_kw_per_step": summary["fc_ramp_rms_kw_per_step"],
            # Consecutive real intervals only; the zero-time shutdown is outside this statistic.
            "battery_ramp_rms_kw_per_step": (
                math.sqrt(battery_delta_mean_square) if battery_delta_mean_square is not None else None
            ),
            # This is the action applied in training, including warm-up and exploration noise.
            # progress.csv actor/* separately measures the noise-free policy on recent states.
            "applied_action_saturation_fraction": mean("actor_action_saturated"),
            "mean_actor_action": mean("actor_action"),
            "mean_mapped_fc_kw": mean("action_mapped_fc_kw"),
            "mean_feasible_interval_width_kw": mean("feasible_interval_width_kw"),
            "soc_initial": summary["soc_initial"],
            "soc_delta": summary["soc_delta"],
            "mean_soc_equivalence_factor": summary["mean_soc_equivalence_factor"],
            "battery_equivalent_hydrogen_total_kg": summary["battery_equivalent_hydrogen_total_kg"],
            "battery_equivalent_hydrogen_total_cny": summary["battery_equivalent_hydrogen_total_cny"],
            "soc_upper_dwell_steps": summary["soc_upper_dwell_steps"],
            "soc_lower_dwell_steps": summary["soc_lower_dwell_steps"],
            "fc_transient_cost_per_step": summary["fc_transient_cost_per_step"],
        }

    def _on_step(self):
        previous_scheduled_episode = self.scheduled_episode
        keep_training = super()._on_step()
        if self.locals["dones"][0]:
            if self.pause_request.is_file():
                self.logger.info(f"Pausing economic TD3 after episode {self.completed_episodes}.")
                keep_training = False
            if (
                self.scheduled_episode != previous_scheduled_episode
                and self.scheduled_episode % self.fixed_eval_every_episodes == 0
            ):
                subprocess.run(
                    [sys.executable, "-m", "experiments.sb3_manual.td3_aggregate.economic_ems_plots",
                     "--run-dir", str(self.directory)],
                    cwd=PROJECT_ROOT,
                    check=True,
                )
            if not keep_training:
                # SB3 2.9.2 returns immediately when a callback stops a rollout.
                # Store this last executed transition once, before returning False.
                # Other transitions and all optimizer updates use the native loop.
                rollout = self.locals
                self.model._update_info_buffer(rollout["infos"], rollout["dones"])
                self.model._store_transition(
                    rollout["replay_buffer"], rollout["buffer_actions"], rollout["new_obs"],
                    rollout["rewards"], rollout["dones"], rollout["infos"],
                )
                self.model._episode_num += 1
        return keep_training

    def _evaluate(self, split: str) -> None:
        super()._evaluate(split)
        if split != "validation":
            return
        metrics_path = (
            self.directory / "periodic_validation"
            / f"episode_{self.scheduled_episode:04d}" / "validation_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        voyages = metrics["per_voyage"]
        # Full physical completion is required before comparing learning costs.
        eligible = (
            metrics.get("validation_set_complete") is True
            and len(voyages) == 8
            and all(
                voyage["success"] is True
                and voyage["executed_steps"] == voyage["total_steps"]
                and float(voyage["soc_min"]) >= SOC_MIN - 1e-9
                and float(voyage["soc_max"]) <= SOC_MAX + 1e-9
                for voyage in voyages
            )
        )
        if not eligible:
            return
        score = metrics["total_learning_economic_cost_cny"]
        if not math.isfinite(score):
            raise RuntimeError("Eligible Validation8 results lack a finite learning cost")
        best_path = self.directory / "best_validation_model.zip"
        current = getattr(self, "best_validation_cost_cny", float("inf"))
        if score < current:
            node = metrics_path.parent
            shutil.copy2(node / "model.zip", best_path)
            self.best_validation_cost_cny = score
            (self.directory / "best_validation_metadata.json").write_text(
                json.dumps({
                    "training_episode": self.scheduled_episode,
                    "total_validation_learning_economic_cost_cny": score,
                    "mean_validation_learning_economic_cost_cny": metrics["mean_full_voyage_learning_economic_cost_cny"],
                    "total_validation_physical_economic_cost_cny": metrics["total_physical_economic_cost_cny"],
                    "mean_validation_physical_economic_cost_cny": metrics["mean_full_voyage_physical_economic_cost_cny"],
                    "validation_voyages": [voyage["sample_id"] for voyage in voyages],
                    "checkpoint": str(best_path),
                    "selection_rule": (
                        "complete Validation8 within physical limits, "
                        "then minimize cumulative learning economic cost"
                    ),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def require_economic_n_step_replay(model):
    replay = model.replay_buffer
    if (
        model.n_steps != N_STEP_RETURN
        or not isinstance(replay, NStepReplayBuffer)
        or replay.n_steps != N_STEP_RETURN
        or replay.gamma != 1.0
        or model.gamma != 1.0
    ):
        raise ValueError(
            f"Economic EMS requires SB3 NStepReplayBuffer with n_steps={N_STEP_RETURN} and gamma=1.0"
        )
    return replay


class EconomicEMSV6TrainingCallback(EconomicEMSTrainingCallback):
    economic_fields = ECONOMIC_FIELDS + V6_REWARD_COMPONENT_FIELDS
    reward_component_fields = V6_REWARD_COMPONENT_FIELDS
    environment_id = V6_ENV_ID
    evaluation_extra_args = ("--reward-mode", "economic_ems_v6")
    count_successful_episodes = True

    def _on_training_start(self):
        identity = self.training_env.get_attr("reward_identity")[0]
        if identity.get("reward_version") != V6_REWARD_VERSION or not getattr(self.model, "log_td_errors", False):
            raise ValueError("Economic EMS v6 requires the v6 reward identity and Critic TD error logging")
        if self.model.device.type != "cuda":
            raise ValueError("Economic EMS v6 requires CUDA")
        if (
            self.model.train_freq.frequency != 32 or self.model.train_freq.unit.value != "step"
            or self.model.gradient_steps != 1 or self.model.policy_delay != 2
        ):
            raise ValueError("Economic EMS v6 requires one Critic update every 32 steps and policy_delay=2")
        replay = self.model.replay_buffer
        if replay is not None and replay.size() and not replay.dones[(replay.pos - 1) % replay.buffer_size, 0]:
            raise ValueError("Economic EMS v6 can only resume replay saved at a voyage boundary")
        super()._on_training_start()
        self.logger.info("algorithm: TD3")
        for name, value in self._additional_training_metadata().items():
            self.logger.info(f"{name}: {value}")

    def _additional_training_metadata(self):
        identity = self.model._td3_reward_identity
        replay = self.model.replay_buffer
        return {
            "reward_version": identity["reward_version"],
            "training_selection": identity["training_selection"],
            "episode_length": FIXED_TRAIN_STEPS,
            "episode_budget_unit": "successful_full_voyages",
            "initial_soc": 0.6,
            "initial_fc_kw": 0.0,
            "action_mapping": identity["action_mapping"],
            "soc_deviation_weight": identity["soc_deviation_weight"],
            "physical_environment_soc_weight": identity["physical_accounting_identity"]["soc_deviation_weight"],
            "n_steps": self.model.n_steps,
            "replay_buffer_class": type(replay).__name__,
            "replay_buffer_n_steps": replay.n_steps,
            "gamma": self.model.gamma,
            "train_freq": self.model.train_freq.frequency,
            "gradient_steps": self.model.gradient_steps,
            "policy_delay": self.model.policy_delay,
            "td_error_diagnostics": "both Critics pooled, absolute pre-Huber residuals in scaled reward units",
        }
