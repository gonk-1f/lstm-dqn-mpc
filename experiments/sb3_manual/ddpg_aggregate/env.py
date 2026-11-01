"""Minimal 3-observation, 1-action aggregate FC/battery ship environment.

Physics and voyage splits are reused from the existing project. No action masking,
forecasting, economic reward, additional observation features, or custom DDPG.
"""
from __future__ import annotations

from math import isfinite, sqrt
from pathlib import Path

import gymnasium as gym
import numpy as np

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v3.control import AccountState, EconomicMPC, SOC_MAX, SOC_MIN
from v4.v6_dataset import V6TrainingDataset


FC_RATED_KW = 600.0
FC_RAMP_KW_PER_S = 48.0
SOC_REFERENCE = 0.5
SOC_WEIGHT = 100.0
RAMP_WEIGHT = 0.0025
FAILURE_REWARD = -1000.0  # Provisional terminal penalty, NOT a proven safe choice.


class AggregateDDPGEnv(gym.Env):
    """One complete train/validation voyage, with 1s actions and real constraints."""

    metadata = {"render_modes": []}

    def __init__(self, data_root: str | Path, split: str = "train") -> None:
        super().__init__()
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation (Test is not accessible)")
        self.split = split
        dataset = V6TrainingDataset.open(Path(data_root))
        if dataset.split_sizes != (35, 8):
            raise ValueError(f"Expected immutable Train35/Validation8, got {dataset.split_sizes}")
        self.episodes = tuple(dataset.load_train() if split == "train" else dataset.load_validation())
        self._validation_cursor = 0
        self.physics = EconomicMPC(nominal_cost_cny=1.0)
        self.plant = self.physics.plant
        if abs(self.plant.fuel_cell_rated_total_kw - FC_RATED_KW) > 1e-9:
            raise ValueError("Existing physical plant does not have 600kW FC rating")
        if abs(self.plant.battery_nominal_energy_kwh - 624.0) > 1e-9:
            raise ValueError("Existing physical plant does not have 624kWh battery capacity")
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32), dtype=np.float32,
        )
        # Same 3-dimensional state as v4's `minimal3`:
        # [SOC, current_load_kw/600, previous_actual_fc_kw/600]
        # Load can exceed 600kW because the battery may supplement FC output.
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, np.inf, 1.0], dtype=np.float32), dtype=np.float32,
        )
        self.episode = None
        self.physical = AccountState()
        self.index = 0
        self.terminated = True
        self.starts = 0
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
        return np.array([
            self.physical.soc,
            current_load / FC_RATED_KW,
            self.physical.previous_fc_kw / FC_RATED_KW,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Cannot override frozen Train/Validation voyage selection")
        if self.split == "train":
            pick = int(self.np_random.integers(0, len(self.episodes)))
        else:
            pick = self._validation_cursor % len(self.episodes)
            self._validation_cursor += 1
        self.episode = self.episodes[pick]
        self.physical = AccountState()  # Project's established SOC=0.6, FC=0 initial state
        self.index = 0
        self.terminated = False
        self.starts = 0
        self.ramp_sq_sum = 0.0
        self.reward_sum = 0.0
        self.soc_min_seen = self.physical.soc
        self.soc_max_seen = self.physical.soc
        self.projection_steps = 0
        return self._observation(), {"sample_id": self.episode.sample_id, "split": self.split}

    def _physically_feasible_fc_range(self, load: float, prev_fc: float):
        """Continuous intersection of FC, 48kW ramp, battery power and SOC constraints.

        +battery bus power = discharge; negative = charge. Same 0.95/0.95
        charge/discharge efficiencies as the existing formal battery model.
        """
        eta_chg, eta_dis = self.physics.efficiency.require_calibrated()
        cap = self.plant.battery_nominal_energy_kwh
        max_discharge_soc_kw = max(0.0, (self.physical.soc - SOC_MIN) * cap * 3600.0 * eta_dis)
        max_charge_soc_kw = max(0.0, (SOC_MAX - self.physical.soc) * cap * 3600.0 / eta_chg)
        bat_lo = max(self.plant.battery_charge_min_kw, -max_charge_soc_kw)
        bat_hi = min(self.plant.battery_discharge_max_kw, max_discharge_soc_kw)
        fc_lo = max(0.0, prev_fc - FC_RAMP_KW_PER_S, load - bat_hi)
        fc_hi = min(FC_RATED_KW, prev_fc + FC_RAMP_KW_PER_S, load - bat_lo)
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
            "fc_starts": self.starts,
            "fc_ramp_rms_kw_per_step": sqrt(self.ramp_sq_sum / steps) if steps else 0.0,
            "power_projection_fraction": self.projection_steps / steps if steps else 0.0,
            "reward_sum": self.reward_sum,
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
        feasible_lo, feasible_hi = self._physically_feasible_fc_range(load_kw, prev_kw)
        if feasible_lo > feasible_hi + 1e-8:
            self.terminated = True
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
            }
            info["vessel_episode"] = self._episode_summary(False, info["terminal_reason"])
            return self._observation(), reward, True, False, info

        # The Actor requests a setpoint; real plant executes the nearest admissible power.
        # This is a deterministic physical actuator, not a learned action mask.
        fc_kw = float(np.clip(target_kw, feasible_lo, feasible_hi))
        self.projection_steps += int(abs(fc_kw - target_kw) > 1e-5)
        next_physical, _ledger, battery_kw = self.physics.interval(
            self.physical, fc_kw, load_kw, duration_s=1.0
        )
        if not (self.plant.battery_charge_min_kw - 1e-6 <= battery_kw <= self.plant.battery_discharge_max_kw + 1e-6):
            raise RuntimeError("Internal implementation error: actual battery power exceeded bounds")
        if not (SOC_MIN - 1e-9 <= next_physical.soc <= SOC_MAX + 1e-9):
            raise RuntimeError("Internal implementation error: actual SOC exceeded bounds")
        delta_fc = fc_kw - prev_kw
        j_soc = SOC_WEIGHT * (next_physical.soc - SOC_REFERENCE) ** 2
        j_ramp = RAMP_WEIGHT * delta_fc ** 2
        reward = -(j_soc + j_ramp)
        self.starts += int(prev_kw <= 0.0 < fc_kw)
        self.ramp_sq_sum += delta_fc ** 2
        self.physical = next_physical
        self.index += 1
        self.soc_min_seen = min(self.soc_min_seen, self.physical.soc)
        self.soc_max_seen = max(self.soc_max_seen, self.physical.soc)
        self.reward_sum += reward
        self.terminated = self.index >= len(self.episode.load_kw)
        info = {
            "sample_id": self.episode.sample_id,
            "load_kw": load_kw,
            "requested_fc_kw": target_kw,
            "fc_actual_kw": fc_kw,
            "battery_actual_kw": float(battery_kw),
            "soc": float(self.physical.soc),
            "delta_fc_kw": delta_fc,
            "soc_penalty": j_soc,
            "fc_ramp_penalty": j_ramp,
            "projected": abs(fc_kw - target_kw) > 1e-5,
            "terminal_reason": "completed" if self.terminated else None,
        }
        if self.terminated:
            info["vessel_episode"] = self._episode_summary(True, "completed")
        return self._observation(), float(reward), self.terminated, False, info
