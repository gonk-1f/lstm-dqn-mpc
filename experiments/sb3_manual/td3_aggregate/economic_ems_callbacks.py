"""TD3 economic EMS diagnostics and selection for the fixed Train voyage."""

from collections import deque
import json
import math
import shutil
import subprocess
import sys

from .callbacks import (
    ECONOMIC_FIELDS, FIXED_TRAIN_ID, FIXED_TRAIN_STEPS, PROJECT_ROOT, TD3TrainingCallback,
)
from .economic_ems import REWARD_COMPONENT_FIELDS
from .huber_td3 import HuberTD3


class EconomicEMSTrainingCallback(TD3TrainingCallback):
    economic_fields = ECONOMIC_FIELDS + REWARD_COMPONENT_FIELDS
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
        super()._on_training_start()
        self._recent_raw_rewards = deque(getattr(self.model, "_td3_recent_raw_rewards", ()), maxlen=20)
        self._previous_battery_kw = None
        self._max_fc_kw = None

    def _observe_step(self, info):
        if info["sample_id"] != FIXED_TRAIN_ID:
            raise ValueError(f"Economic EMS training only accepts {FIXED_TRAIN_ID}")
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
        if summary["total_steps"] != FIXED_TRAIN_STEPS or summary["soc_initial"] != 0.6:
            raise ValueError("Economic EMS voyage requires 9258 steps and initial SOC=0.6")
        if summary["completed"]:
            valid_ending = reason == "completed" and calls == executed == FIXED_TRAIN_STEPS
        else:
            valid_ending = (
                reason == "no_feasible_physical_power"
                and 0 <= executed < FIXED_TRAIN_STEPS
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
            "terminal_soc_target": summary["terminal_soc_target"],
            "terminal_soc_error": summary["terminal_soc_error"],
        }

    def _on_step(self):
        keep_training = super()._on_step()
        if self.locals["dones"][0]:
            if self.completed_episodes % 20 == 0:
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
            / f"episode_{self.completed_episodes:04d}" / "validation_metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        voyages = metrics["per_voyage"]
        # A candidate must complete all Validation8 voyages and keep every
        # terminal SOC within 0.05 of the specified 0.5 target.
        eligible = (
            metrics.get("validation_set_complete") is True
            and len(voyages) == 8
            and all(voyage.get("success") is True for voyage in voyages)
            and max(abs(float(voyage["soc_final"]) - 0.5) for voyage in voyages) <= 0.05 + 1e-12
            and min(float(voyage["soc_min"]) for voyage in voyages) >= 0.2 - 1e-9
        )
        if not eligible:
            return
        score = metrics["mean_full_voyage_economic_cost_cny"]
        if score is None or not math.isfinite(score):
            raise RuntimeError("Eligible Validation8 results lack a finite complete-voyage cost")
        best_path = self.directory / "best_validation_model.zip"
        current = getattr(self, "best_validation_cost_cny", float("inf"))
        if score < current:
            node = metrics_path.parent
            shutil.copy2(node / "model.zip", best_path)
            self.best_validation_cost_cny = score
            (self.directory / "best_validation_metadata.json").write_text(
                json.dumps({
                    "training_episode": self.completed_episodes,
                    "mean_validation_total_cost_cny": score,
                    "max_abs_terminal_soc_error": max(
                        abs(float(voyage["soc_final"]) - 0.5) for voyage in voyages
                    ),
                    "validation_voyages": [voyage["sample_id"] for voyage in voyages],
                    "checkpoint": str(best_path),
                    "selection_rule": "complete Validation8; every terminal SOC within 0.05 of 0.5; minimize mean full-voyage physical CNY cost",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
