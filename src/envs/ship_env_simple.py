from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

from dqn.utils.reward import FixedRewardWeights, RewardParams, RewardRefs, calculate_reward
from envs.ship_env_base import ShipEnvBase


@dataclass
class SimpleEnvConfig:
    dt_seconds: float = 60.0
    battery_capacity_kwh: float = 1067.0
    fuel_cell_min_kw: float = 0.0
    fuel_cell_max_kw: float = 240.0
    battery_charge_max_kw: float = 350.0
    battery_discharge_max_kw: float = 350.0
    soc_min: float = 0.2
    soc_max: float = 0.95
    fc_action_grid_kw: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    batt_action_grid_kw: tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    episode_steps: int = 180
    random_reset: bool = True
    soc_ref_default: float = 0.65
    reward_weights: FixedRewardWeights = field(default_factory=FixedRewardWeights)


class ShipEnvSimple(ShipEnvBase):
    """Phase-1 environment: single fuel cell + single battery + single load.

    Observation layout:
    [load_kw, speed_knots, soc, fuel_cell_power_kw, battery_power_kw,
     mpc_fuel_cell_ref_kw, mpc_battery_ref_kw, fuel_cell_error_kw,
     battery_error_kw, load_ramp_kw, soc_ref, soc_error]
    """

    def __init__(self, data: pd.DataFrame, config: SimpleEnvConfig | None = None):
        self.df = data.reset_index(drop=True).copy()
        self.config = config or SimpleEnvConfig()
        if "dqn_control_seconds" in self.df.columns and not self.df["dqn_control_seconds"].dropna().empty:
            self.config.dt_seconds = float(self.df["dqn_control_seconds"].dropna().iloc[0])
        self.dt_hours = self.config.dt_seconds / 3600.0
        self.actions = [
            (fc_delta, batt_delta)
            for fc_delta in self.config.fc_action_grid_kw
            for batt_delta in self.config.batt_action_grid_kw
        ]
        self.action_dim = len(self.actions)
        self.state_dim = 12
        self.pointer = 0
        self.soc = 0.6
        self.fc_power_kw = 0.0
        self.batt_power_kw = 0.0
        self.prev_fc_power_kw = 0.0
        self.prev_batt_power_kw = 0.0
        self.current_row = None
        self.episode_step = 0
        self.episode_start = 0

    def _row(self, idx: int) -> pd.Series:
        return self.df.iloc[idx]

    def _get_obs(self) -> np.ndarray:
        row = self.current_row
        fc_ref = float(row.get("mpc_fuel_cell_ref_kw", row["fuel_cell_power_total_kw"]))
        batt_ref = float(row.get("mpc_battery_ref_kw", row["battery_power_total_kw"]))
        soc_ref = float(row.get("mpc_soc_ref", row.get("soc_ref", self.config.soc_ref_default)))
        return np.asarray(
            [
                float(row["load_total_kw"]),
                float(row.get("speed_knots", 0.0)),
                float(self.soc),
                float(self.fc_power_kw),
                float(self.batt_power_kw),
                fc_ref,
                batt_ref,
                self.fc_power_kw - fc_ref,
                self.batt_power_kw - batt_ref,
                float(row.get("load_ramp_kw", 0.0)),
                soc_ref,
                float(self.soc - soc_ref),
            ],
            dtype=np.float32,
        )

    def reset(self, start_index: int | None = None):
        max_start = max(0, len(self.df) - max(2, self.config.episode_steps + 1))
        if start_index is None:
            if self.config.random_reset and max_start > 0:
                start_index = int(np.random.randint(0, max_start + 1))
            else:
                start_index = 0
        self.pointer = int(start_index)
        self.episode_start = self.pointer
        self.episode_step = 0
        self.current_row = self._row(self.pointer)
        self.soc = float(self.current_row.get("soc_mean", 0.6))
        self.fc_power_kw = float(self.current_row.get("fuel_cell_power_total_kw", 0.0))
        self.batt_power_kw = float(self.current_row.get("battery_power_total_kw", 0.0))
        self.prev_fc_power_kw = self.fc_power_kw
        self.prev_batt_power_kw = self.batt_power_kw
        return self._get_obs(), {"pointer": self.pointer, "episode_start": self.episode_start}

    def _clip_power(self, fc_power_kw: float, batt_power_kw: float) -> tuple[float, float, float]:
        violation = 0.0
        clipped_fc = float(np.clip(fc_power_kw, self.config.fuel_cell_min_kw, self.config.fuel_cell_max_kw))
        if clipped_fc != fc_power_kw:
            violation += abs(fc_power_kw - clipped_fc)

        clipped_batt = float(
            np.clip(
                batt_power_kw,
                -self.config.battery_charge_max_kw,
                self.config.battery_discharge_max_kw,
            )
        )
        if clipped_batt != batt_power_kw:
            violation += abs(batt_power_kw - clipped_batt)
        return clipped_fc, clipped_batt, violation

    def _update_soc(self, batt_power_kw: float) -> float:
        next_soc = self.soc - batt_power_kw * self.dt_hours / self.config.battery_capacity_kwh
        violation = 0.0
        clipped_soc = float(np.clip(next_soc, self.config.soc_min, self.config.soc_max))
        if clipped_soc != next_soc:
            violation = abs(next_soc - clipped_soc)
        self.soc = clipped_soc
        return violation

    def step(self, action):
        fc_delta, batt_delta = self.actions[int(action)]
        row = self.current_row
        load_kw = float(row["load_total_kw"])
        fc_ref = float(row.get("mpc_fuel_cell_ref_kw", row["fuel_cell_power_total_kw"]))
        batt_ref = float(row.get("mpc_battery_ref_kw", row["battery_power_total_kw"]))
        soc_ref = float(row.get("mpc_soc_ref", row.get("soc_ref", self.config.soc_ref_default)))

        prev_state = {
            "p_fc_kw": self.fc_power_kw,
            "p_bat_kw": self.batt_power_kw,
            "soc": self.soc,
            "soc_ref": soc_ref,
        }

        proposed_fc = self.fc_power_kw + fc_delta
        proposed_batt = self.batt_power_kw + batt_delta
        self.fc_power_kw, self.batt_power_kw, power_violation = self._clip_power(proposed_fc, proposed_batt)
        soc_violation = self._update_soc(self.batt_power_kw)

        supplied_kw = self.fc_power_kw + self.batt_power_kw
        tracking_error = abs(self.fc_power_kw - fc_ref) + abs(self.batt_power_kw - batt_ref)
        balance_error = abs(supplied_kw - load_kw)
        next_state = {
            "p_fc_kw": self.fc_power_kw,
            "p_bat_kw": self.batt_power_kw,
            "soc": self.soc,
            "soc_ref": soc_ref,
        }
        reward, reward_info = calculate_reward(
            state=prev_state,
            action=int(action),
            next_state=next_state,
            refs=RewardRefs(p_fc_ref_kw=fc_ref, soc_ref=soc_ref),
            params=RewardParams(
                p_fc_min_kw=self.config.fuel_cell_min_kw,
                p_fc_max_kw=self.config.fuel_cell_max_kw,
                p_bat_min_kw=-self.config.battery_charge_max_kw,
                p_bat_max_kw=self.config.battery_discharge_max_kw,
                soc_min=self.config.soc_min,
                soc_max=self.config.soc_max,
                weights=self.config.reward_weights,
            ),
        )

        self.prev_fc_power_kw = self.fc_power_kw
        self.prev_batt_power_kw = self.batt_power_kw

        self.pointer += 1
        self.episode_step += 1
        terminated = self.pointer >= len(self.df) - 1
        truncated = self.episode_step >= self.config.episode_steps
        if not terminated:
            self.current_row = self._row(self.pointer)

        done = terminated or truncated
        observation = self._get_obs() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "load_kw": load_kw,
            "supplied_kw": supplied_kw,
            "fuel_cell_ref_kw": fc_ref,
            "battery_ref_kw": batt_ref,
            "fuel_cell_power_kw": self.fc_power_kw,
            "battery_power_kw": self.batt_power_kw,
            "tracking_error_kw": tracking_error,
            "balance_error_kw": balance_error,
            "soc": self.soc,
            "soc_ref": soc_ref,
            "power_violation": power_violation,
            "soc_violation": soc_violation,
            "episode_step": self.episode_step,
            "global_index": self.pointer,
        }
        info.update(reward_info)
        return observation, float(reward), done, truncated, info

    def describe_state_space(self) -> dict:
        return {
            "meaning": "lower-layer single-source/single-load vessel microgrid state",
            "features": [
                "load_kw",
                "speed_knots",
                "soc",
                "fuel_cell_power_kw",
                "battery_power_kw",
                "fuel_cell_ref_kw",
                "battery_ref_kw",
                "fuel_cell_tracking_error_kw",
                "battery_tracking_error_kw",
                "load_ramp_kw",
                "soc_ref",
                "soc_tracking_error",
            ],
        }

    def describe_action_space(self) -> dict:
        return {
            "meaning": "discrete lower-layer power adjustment action",
            "action_count": self.action_dim,
            "action_table": self.actions,
        }

    def describe_reward(self) -> dict:
        return {
            "meaning": "four-group normalized lower-layer DQN reward",
            "terms": {
                "energy_norm": self.config.reward_weights.w1,
                "fc_dynamic_norm": self.config.reward_weights.w2,
                "soc_norm": self.config.reward_weights.w3,
                "battery_use_norm": self.config.reward_weights.w4,
            },
        }
