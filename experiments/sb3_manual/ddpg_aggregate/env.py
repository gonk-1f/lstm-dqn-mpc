"""Five-observation, one-action aggregate FC/battery ship environment.

Physics and voyage splits are reused from the existing project. Observations use
current/past loads and elapsed time, without forecasting or action masking.
"""
from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path

import gymnasium as gym
import numpy as np

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.models.battery_energy import battery_current_from_soc_change
from v3.control import AccountState, EconomicMPC, SOC_MAX, SOC_MIN
from v4.v6_dataset import V6TrainingDataset
from .economic_reward import (
    load_reward_config, reward_identity,
    interval_economic_components, economic_reward, FAILURE_REWARD,
    shutdown_economic_components, FC_DEGRADATION_COST_FIELDS,
    SOC_DEVIATION_WEIGHT, SOC_REFERENCE,
)
from .huber_ddpg import CRITIC_LOSS_ID


FC_RATED_KW = 600.0
REWARD_CONFIG = load_reward_config()
C_REF = REWARD_CONFIG["c_ref"]  # Historical calibration metadata; not used by reward.
RAMP_WEIGHT = 0.0  # Legacy diagnostic field; no independent ramp penalty.
ECONOMIC_SUM_FIELDS = (
    "hydrogen_mass_kg", "h2_cost_cny", "fuel_cell_degradation_cost_cny",
    "battery_degradation_cost_cny", "shore_cost_cny", "physical_economic_cost_cny",
    "c_bat_eq_cny", "battery_equivalent_hydrogen_kg", "total_cost_cny",
    "reward_economic_cost_cny", "soc_deviation_penalty_cny", "reward_total_cost_cny",
) + FC_DEGRADATION_COST_FIELDS


class AggregateDDPGEnv(gym.Env):
    """One complete train/validation voyage, with 1s actions and real constraints."""

    metadata = {"render_modes": []}

    def __init__(
        self, data_root: str | Path, split: str = "train",
        fixed_train_id: str | None = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation (Test is not accessible)")
        if fixed_train_id is not None and split != "train":
            raise ValueError("fixed_train_id is only allowed for split=train")
        self.split = split
        self.fixed_train_id = fixed_train_id
        self.data_root = Path(data_root).resolve()
        dataset = V6TrainingDataset.open(self.data_root)
        if dataset.split_sizes != (35, 8):
            raise ValueError(f"Expected immutable Train35/Validation8, got {dataset.split_sizes}")
        self.episodes = tuple(dataset.load_train() if split == "train" else dataset.load_validation())
        self._fixed_train_index = None
        if fixed_train_id is not None:
            matches = [i for i, ep in enumerate(self.episodes) if ep.sample_id == fixed_train_id]
            if len(matches) != 1:
                raise ValueError(f"fixed_train_id {fixed_train_id!r} must match exactly one Train35 voyage; got {len(matches)}")
            self._fixed_train_index = matches[0]
        self._validation_cursor = 0
        self.physics = EconomicMPC(nominal_cost_cny=1.0)
        self.plant = self.physics.plant
        self.reward_identity = reward_identity(REWARD_CONFIG)
        self.reward_identity["fc_ramp_limit_kw_per_s"] = None
        self.reward_identity["observation_version"] = "soc_load_prev_fc_load_delta_elapsed_hours_v1"
        self.reward_identity["critic_loss"] = CRITIC_LOSS_ID
        self.economic_totals = dict.fromkeys(ECONOMIC_SUM_FIELDS, 0.0)
        self.equivalent_eta_sum = 0.0
        self.equivalent_eta_min = None
        self.equivalent_eta_max = None
        if abs(self.plant.fuel_cell_rated_total_kw - FC_RATED_KW) > 1e-9:
            raise ValueError("Existing physical plant does not have 600kW FC rating")
        if abs(self.plant.battery_nominal_energy_kwh - 624.0) > 1e-9:
            raise ValueError("Existing physical plant does not have 624kWh battery capacity")
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32), dtype=np.float32,
        )
        # [SOC, current_load_kw/600, previous_actual_fc_kw/600,
        #  (current_load_kw - previous_load_kw)/600, elapsed_seconds/3600]
        # Load can exceed 600kW because the battery may supplement FC output.
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf, 0.0], dtype=np.float32),
            high=np.array([1.0, np.inf, 1.0, np.inf, np.inf], dtype=np.float32), dtype=np.float32,
        )
        self.episode = None
        self.physical = AccountState()
        self.index = 0
        self.terminated = True
        self.starts = 0
        self.physical_fc_on = False
        self._cycle_settled = False
        self.battery_power_kw = 0.0
        self.last_interval_fc_kw = 0.0
        self.last_interval_battery_kw = 0.0
        self.ramp_sq_sum = 0.0
        self.reward_sum = 0.0
        self.soc_min_seen = 0.6
        self.soc_max_seen = 0.6
        self.projection_steps = 0

    def _observation(self) -> np.ndarray:
        if self.episode is None:
            raise RuntimeError("reset() must precede observation")
        idx = min(self.index, len(self.episode.load_kw) - 1)
        current_load = normalize_onboard_load_kw(float(self.episode.load_kw[idx]))
        previous_load = normalize_onboard_load_kw(float(self.episode.load_kw[max(0, idx - 1)]))
        return np.array([
            self.physical.soc,
            current_load / FC_RATED_KW,
            self.physical.previous_fc_kw / FC_RATED_KW,
            (current_load - previous_load) / FC_RATED_KW,
            self.index / 3600.0,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Cannot override frozen Train/Validation voyage selection")
        if self._fixed_train_index is not None:
            pick = self._fixed_train_index
        elif self.split == "train":
            pick = int(self.np_random.integers(0, len(self.episodes)))
        else:
            pick = self._validation_cursor % len(self.episodes)
            self._validation_cursor += 1
        self.episode = self.episodes[pick]
        self.physical = AccountState()  # Project's established SOC=0.6, FC=0 initial state
        self.index = 0
        self.terminated = False
        # Starting the operating cycle is an environment event, not an Actor action.
        # Zero output while physically on is idle; settle the full cycle at shutdown.
        self.starts = 1
        self.physical_fc_on = True
        self._cycle_settled = False
        self.battery_power_kw = 0.0
        self.last_interval_fc_kw = 0.0
        self.last_interval_battery_kw = 0.0
        self.ramp_sq_sum = 0.0
        self.reward_sum = 0.0
        self.soc_min_seen = self.physical.soc
        self.soc_max_seen = self.physical.soc
        self.projection_steps = 0
        self.economic_totals = dict.fromkeys(ECONOMIC_SUM_FIELDS, 0.0)
        self.equivalent_eta_sum = 0.0
        self.equivalent_eta_min = None
        self.equivalent_eta_max = None
        return self._observation(), {
            "sample_id": self.episode.sample_id, "split": self.split,
            "physical_fc_on": self.physical_fc_on, "fc_starts": self.starts,
            "soc_deviation_weight": SOC_DEVIATION_WEIGHT, "soc_reference": SOC_REFERENCE,
        }

    def _shutdown(self):
        """Close one operating cycle without executing another physical interval."""
        if self._cycle_settled:
            raise RuntimeError("The voyage start-stop cycle has already been settled")
        final_interval_fc_kw = self.physical.previous_fc_kw
        self.physical, costs = shutdown_economic_components(self.physics, self.physical)
        self.physical_fc_on = False
        self.battery_power_kw = 0.0
        self._cycle_settled = True
        return costs, {
            "shutdown_executed": True,
            "shutdown_time_s": self.index,
            "shutdown_duration_s": 0.0,
            "post_shutdown_fc_kw": 0.0,
            "post_shutdown_battery_kw": 0.0,
            "shutdown_delta_fc_kw": -float(final_interval_fc_kw),
            "shutdown_fc_transient_cost_cny": costs["fuel_cell_transient_cost_cny"],
            "shutdown_fc_start_stop_cost_cny": costs["fuel_cell_start_stop_cost_cny"],
            "shutdown_economic_cost_cny": costs["total_cost_cny"],
        }

    def _physically_feasible_fc_range(self, load: float):
        """Continuous intersection of FC rating, battery power and SOC constraints.

        +battery bus power = discharge; negative = charge. Same 0.95/0.95
        charge/discharge efficiencies as the existing formal battery model.
        """
        eta_chg, eta_dis = self.physics.efficiency.require_calibrated()
        cap = self.plant.battery_nominal_energy_kwh
        max_discharge_soc_kw = max(0.0, (self.physical.soc - SOC_MIN) * cap * 3600.0 * eta_dis)
        max_charge_soc_kw = max(0.0, (SOC_MAX - self.physical.soc) * cap * 3600.0 / eta_chg)
        bat_lo = max(self.plant.battery_charge_min_kw, -max_charge_soc_kw)
        bat_hi = min(self.plant.battery_discharge_max_kw, max_discharge_soc_kw)
        fc_lo = max(0.0, load - bat_hi)
        fc_hi = min(FC_RATED_KW, load - bat_lo)
        return float(fc_lo), float(fc_hi)

    def _episode_summary(self, success: bool, reason: str):
        steps = self.index
        return {
            "sample_id": self.episode.sample_id,
            "split": self.split,
            "completed": bool(success),
            "terminal_reason": reason,
            "executed_steps": steps,
            "total_steps": len(self.episode.load_kw),
            "coverage": steps / len(self.episode.load_kw),
            "soc_final": float(self.physical.soc),
            "soc_min": float(self.soc_min_seen),
            "soc_max": float(self.soc_max_seen),
            "fc_final_kw": float(self.physical.previous_fc_kw),
            "battery_final_kw": float(self.battery_power_kw),
            "last_interval_fc_kw": float(self.last_interval_fc_kw),
            "last_interval_battery_kw": float(self.last_interval_battery_kw),
            "physical_fc_on": self.physical_fc_on,
            "fc_starts": self.starts,
            "fc_shutdowns": int(self._cycle_settled),
            "fc_start_stop_cycles": int(self._cycle_settled),
            "fc_cumulative_voltage_loss_uv": float(self.physical.fc_loss_uv),
            "fc_ramp_rms_kw_per_step": sqrt(self.ramp_sq_sum / steps) if steps else 0.0,
            "power_projection_fraction": self.projection_steps / steps if steps else 0.0,
            "reward_sum": self.reward_sum,
            **self.economic_totals,
            "mean_equivalent_fc_efficiency": self.equivalent_eta_sum / steps if steps else None,
            "min_equivalent_fc_efficiency": self.equivalent_eta_min,
            "max_equivalent_fc_efficiency": self.equivalent_eta_max,
            "reward_version": self.reward_identity["reward_version"],
            "soc_deviation_weight": SOC_DEVIATION_WEIGHT,
            "soc_reference": SOC_REFERENCE,
            "c_ref_cny": C_REF,
            "c_ref_used_in_reward": False,
        }

    def step(self, action):
        if self.terminated or self.episode is None:
            raise RuntimeError("Cannot call step() before reset or after episode end")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 1 or not isfinite(float(action[0])) or not -1.000001 <= float(action[0]) <= 1.000001:
            raise ValueError(f"Invalid continuous action: {action}")
        a = float(np.clip(action[0], -1.0, 1.0))
        target_kw = FC_RATED_KW * (a + 1.0) / 2.0
        prev_kw = float(self.physical.previous_fc_kw)
        load_kw = normalize_onboard_load_kw(float(self.episode.load_kw[self.index]))
        feasible_lo, feasible_hi = self._physically_feasible_fc_range(load_kw)
        if feasible_lo > feasible_hi + 1e-8:
            self.terminated = True
            shutdown_costs, shutdown_info = self._shutdown()
            for name in ECONOMIC_SUM_FIELDS:
                self.economic_totals[name] += shutdown_costs[name]
            # A failed, unexecuted action has no SOC penalty or physical supply.
            # Shutdown life losses remain in accounting; the fixed failure reward
            # overrides cost-based reward, without rewriting any earlier transition.
            reward = FAILURE_REWARD
            self.reward_sum += reward
            info = {
                "sample_id": self.episode.sample_id,
                "terminal_reason": "no_feasible_physical_power",
                "requested_fc_kw": target_kw,
                "load_kw": load_kw,
                "feasible_fc_lower_kw": feasible_lo,
                "feasible_fc_upper_kw": feasible_hi,
                "soc": self.physical.soc,
                "fc_actual_kw": prev_kw,
                "step_executed": False,
                **{name: shutdown_costs[name] for name in ECONOMIC_SUM_FIELDS},
                **shutdown_info,
                "physical_fc_on": self.physical_fc_on,
                "fc_start_stop_cycles": int(self._cycle_settled),
                "fc_cumulative_voltage_loss_uv": float(self.physical.fc_loss_uv),
                "reward_is_failure_override": True,
            }
            info["vessel_episode"] = self._episode_summary(False, info["terminal_reason"])
            return self._observation(), reward, True, False, info

        # The Actor requests a setpoint; real plant executes the nearest admissible power.
        # This is a deterministic physical actuator, not a learned action mask.
        fc_kw = float(np.clip(target_kw, feasible_lo, feasible_hi))
        self.projection_steps += int(abs(fc_kw - target_kw) > 1e-5)
        next_physical, ledger, battery_kw = self.physics.interval(
            self.physical, fc_kw, load_kw, duration_s=1.0,
            fc_is_on=True, fc_start_stop_cycles=0,
        )
        battery_current_a = battery_current_from_soc_change(
            self.physical.soc, next_physical.soc, 1.0,
            self.physics.battery_normalization.nominal_charge_capacity_ah,
        )
        if not (self.plant.battery_charge_min_kw - 1e-6 <= battery_kw <= self.plant.battery_discharge_max_kw + 1e-6):
            raise RuntimeError("Internal implementation error: actual battery power exceeded bounds")
        if not (SOC_MIN - 1e-9 <= next_physical.soc <= SOC_MAX + 1e-9):
            raise RuntimeError("Internal implementation error: actual SOC exceeded bounds")
        delta_fc = fc_kw - prev_kw
        costs = interval_economic_components(
            self.physics, self.physical, next_physical, fc_kw, ledger,
            fc_is_on=True, fc_start_stop_cycles=0,
        )
        self.ramp_sq_sum += delta_fc ** 2
        self.physical = next_physical
        self.battery_power_kw = float(battery_kw)
        self.last_interval_fc_kw = fc_kw
        self.last_interval_battery_kw = float(battery_kw)
        self.index += 1
        self.terminated = self.index >= len(self.episode.load_kw)
        shutdown_info = {"shutdown_executed": False}
        if self.terminated:
            shutdown_costs, shutdown_info = self._shutdown()
            # Merge only additive accounting fields. Preserve the real interval's
            # efficiency/power diagnostics and its single SOC penalty.
            for name in ECONOMIC_SUM_FIELDS:
                costs[name] += shutdown_costs[name]
        reward = economic_reward(costs["reward_total_cost_cny"])
        for name in ECONOMIC_SUM_FIELDS:
            self.economic_totals[name] += costs[name]
        eta = costs["equivalent_fc_efficiency"]
        self.equivalent_eta_sum += eta
        self.equivalent_eta_min = eta if self.equivalent_eta_min is None else min(self.equivalent_eta_min, eta)
        self.equivalent_eta_max = eta if self.equivalent_eta_max is None else max(self.equivalent_eta_max, eta)
        self.soc_min_seen = min(self.soc_min_seen, self.physical.soc)
        self.soc_max_seen = max(self.soc_max_seen, self.physical.soc)
        self.reward_sum += reward
        info = {
            "sample_id": self.episode.sample_id,
            "load_kw": load_kw,
            "requested_fc_kw": target_kw,
            "fc_actual_kw": fc_kw,
            "battery_actual_kw": float(battery_kw),
            "battery_current_a": battery_current_a,
            "soc": float(self.physical.soc),
            "delta_fc_kw": delta_fc,
            "fc_ramp_penalty": 0.0,
            "step_executed": True,
            **costs,
            **shutdown_info,
            "physical_fc_on": self.physical_fc_on,
            "fc_start_stop_cycles": int(self._cycle_settled),
            "fc_cumulative_voltage_loss_uv": float(self.physical.fc_loss_uv),
            "reward_is_failure_override": False,
            "projected": abs(fc_kw - target_kw) > 1e-5,
            "terminal_reason": "completed" if self.terminated else None,
        }
        if self.terminated:
            info["vessel_episode"] = self._episode_summary(True, "completed")
        return self._observation(), float(reward), self.terminated, False, info
