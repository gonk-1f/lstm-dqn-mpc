from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import pandas as pd

from components.battery_cluster import BatteryCluster, BatteryClusterConfig
from components.fuel_cell_stack import FuelCellStack, FuelCellStackConfig
from components.power_allocator import (
    BatterySoftBalanceConfig,
    allocate_soft_balanced_battery,
    allocate_symmetric_fuel_cell,
)
from components.ship_power_bus import ShipPowerBus, ShipSideState
from dqn.utils.reward import FixedRewardWeights, RewardParams, RewardRefs, calculate_reward
from envs.ship_env_base import ShipEnvBase


@dataclass
class DualSideEnvConfig:
    profile_name: str = "formal_p6_reduced_order"
    dt_seconds: float = 5.0         # DQN control step
    episode_steps: int = 180
    random_reset: bool = True
    reward_mode: str = "legacy"              # "legacy" or "soc_asymmetric"
    soc_random_init_enabled: bool = False     # randomize SOC at reset
    soc_random_init_low: float = 0.55         # SOC random range low
    soc_random_init_high: float = 0.75        # SOC random range high
    episode_start_indices: tuple[int, ...] = ()  # pre-computed episode starts
    soc_min: float = 0.2
    soc_max: float = 0.95
    fuel_cell_max_kw: float = 560.0
    fuel_cell_ramp_kw: float = 48.0
    battery_capacity_kwh: float = 1806.0
    battery_charge_max_kw: float = 350.0
    battery_discharge_max_kw: float = 350.0
    fc_max_left_kw: float = 280.0    # 氫舟一号 560kW total / 2
    fc_max_right_kw: float = 280.0
    fc_ramp_kw: float = 24.0
    batt_cap_left_kwh: float = 903.0  # 1806kWh total / 2
    batt_cap_right_kwh: float = 903.0
    batt_charge_max_kw: float = 175.0
    batt_discharge_max_kw: float = 175.0
    split_candidates: tuple[float, ...] = (0.4, 0.45, 0.48, 0.5, 0.52, 0.55, 0.6)
    fc_trim_candidates_kw: tuple[float, ...] = (0.0,)
    batt_trim_candidates_kw: tuple[float, ...] = (0.0,)
    reward_tracking: float = 6.0
    reward_balance: float = 5.0
    reward_split_balance: float = 1.0
    reward_soc_balance: float = 12.0
    reward_center_bias: float = 0.8
    reward_action_switch: float = 0.4
    reward_constraint: float = 4.0
    fc_command_gain_left: float = 0.96
    fc_command_gain_right: float = 1.04
    batt_command_gain_left: float = 1.03
    batt_command_gain_right: float = 0.97
    disturbance_from_ramp_gain: float = 0.05
    soc_ref_default: float = 0.65
    reward_weights: FixedRewardWeights = field(default_factory=FixedRewardWeights)
    enforce_power_balance: bool = True
    fc_soc_support_gain_kw: float = 120.0
    fc_soc_support_medium_gain_kw: float = 400.0
    fc_soc_support_deadband: float = 0.05
    fc_soc_support_start_soc: float = 0.0
    fc_soc_support_medium_start_soc: float = 0.60
    fc_soc_support_strong_start_soc: float = 0.55
    fc_soc_support_full_start_soc: float = 0.50
    fc_soc_support_light_max_kw: float = 20.0
    fc_soc_support_medium_max_kw: float = 40.0
    fc_soc_support_max_kw: float = 40.0
    fc_soc_support_hold_hysteresis_enabled: bool = False
    fc_soc_support_hold_disable_soc: float = 0.60
    fc_soc_support_hold_reenable_soc: float = 0.58
    fc_soc_support_hold_scale: float = 0.0
    fc_soc_support_piecewise: bool = True
    fc_soc_support_battery_need_aware: bool = False
    prevent_high_soc_fc_surplus_charge: bool = True
    fc_surplus_charge_guard_margin: float = 0.0
    fc_surplus_charge_guard_eps_kw: float = 0.1
    fuel_cell_allocator_mode: str = "symmetric_total"
    battery_allocator_mode: str = "soft_balance"
    battery_soc_balance_gain: float = 0.35
    battery_split_min: float = 0.30
    battery_split_max: float = 0.70
    soc_balance_deadband: float = 0.005
    soc_balance_bias_gain_kw_per_soc: float = 350.0
    max_balance_bias_kw: float = 8.0
    max_balance_bias_ratio: float = 0.08
    balance_bias_smoothing_alpha: float = 0.25
    balance_bias_rate_limit_kw_per_step: float = 2.0
    batt_ramp_kw_per_step: float = 360.0
    dominance_deadband_kw: float = 0.5
    fc_var_reward_alpha: float = 0.0
    fc_var_reward_ema_alpha: float = 0.0
    fc_var_reward_soc_gate: float = 0.85
    fc_var_reward_progress_gate: float = 0.20
    soc_balance_highsoc_scale: float = 0.0
    soc_direction_fc_scale: float = 2.0
    soc_battery_direction_scale: float = 2.0
    soc_correction_credit_scale: float = 0.20
    fc_trim_penalty_scale: float = 0.50
    battery_corrective_use_discount: float = 0.50
    battery_wrong_direction_use_penalty: float = 0.50
    battery_use_soc_factor_min: float = 0.35
    battery_use_soc_factor_max: float = 1.50
    reset_start_indices: tuple[int, ...] = ()


def _coerce_config_value(current_value, incoming_value):
    if isinstance(current_value, bool):
        if isinstance(incoming_value, str):
            lowered = incoming_value.strip().lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        return bool(incoming_value)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(incoming_value)
    if isinstance(current_value, float):
        return float(incoming_value)
    if isinstance(current_value, tuple):
        if isinstance(incoming_value, tuple):
            return incoming_value
        if isinstance(incoming_value, list):
            return tuple(incoming_value)
    return incoming_value


def _apply_env_values(config: DualSideEnvConfig, values: dict | None, source: str) -> None:
    if not values:
        return
    for key, incoming_value in values.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown env config key `{key}` from {source}.")
        current_value = getattr(config, key)
        setattr(config, key, _coerce_config_value(current_value, incoming_value))


def load_dual_side_env_config(
    project_root: str | Path | None = None,
    profile_name: str | None = None,
    overrides: dict | None = None,
) -> DualSideEnvConfig:
    config = DualSideEnvConfig()
    project_config = {}
    if project_root is not None:
        try:
            from utils.config_loader import load_project_config
        except ModuleNotFoundError:
            project_config = {}
        else:
            project_config = load_project_config(project_root)

    environment_config = project_config.get("environment", {})
    profiles = environment_config.get("profiles", {})
    resolved_profile = str(profile_name or environment_config.get("active_profile", config.profile_name)).strip()
    if resolved_profile:
        config.profile_name = resolved_profile
    _apply_env_values(config, profiles.get(config.profile_name), f"environment profile `{config.profile_name}`")

    dqn_config = project_config.get("dqn", {})
    reward_weights = dqn_config.get("reward_weights", {})
    if reward_weights:
        config.reward_weights = FixedRewardWeights(**{**config.reward_weights.__dict__, **reward_weights})

    reward_defaults = dqn_config.get("reward_defaults", {})
    if "soc_ref" in reward_defaults:
        config.soc_ref_default = float(reward_defaults["soc_ref"])
    for key in [
        "fc_var_reward_alpha",
        "fc_var_reward_ema_alpha",
        "fc_var_reward_soc_gate",
        "fc_var_reward_progress_gate",
        "soc_balance_highsoc_scale",
        "soc_direction_fc_scale",
        "soc_battery_direction_scale",
        "soc_correction_credit_scale",
        "fc_trim_penalty_scale",
        "battery_corrective_use_discount",
        "battery_wrong_direction_use_penalty",
        "battery_use_soc_factor_min",
        "battery_use_soc_factor_max",
    ]:
        if key in reward_defaults:
            setattr(config, key, float(reward_defaults[key]))

    control_correction = dqn_config.get("control_correction", {})
    for key in [
        "enforce_power_balance",
        "fc_soc_support_gain_kw",
        "fc_soc_support_medium_gain_kw",
        "fc_soc_support_deadband",
        "fc_soc_support_start_soc",
        "fc_soc_support_medium_start_soc",
        "fc_soc_support_strong_start_soc",
        "fc_soc_support_full_start_soc",
        "fc_soc_support_light_max_kw",
        "fc_soc_support_medium_max_kw",
        "fc_soc_support_max_kw",
        "fc_soc_support_hold_hysteresis_enabled",
        "fc_soc_support_hold_disable_soc",
        "fc_soc_support_hold_reenable_soc",
        "fc_soc_support_hold_scale",
        "fc_soc_support_piecewise",
        "fc_soc_support_battery_need_aware",
        "prevent_high_soc_fc_surplus_charge",
        "fc_surplus_charge_guard_margin",
        "fc_surplus_charge_guard_eps_kw",
        "fuel_cell_allocator_mode",
        "battery_allocator_mode",
        "battery_soc_balance_gain",
        "battery_split_min",
        "battery_split_max",
        "soc_balance_deadband",
        "soc_balance_bias_gain_kw_per_soc",
        "max_balance_bias_kw",
        "max_balance_bias_ratio",
        "balance_bias_smoothing_alpha",
        "balance_bias_rate_limit_kw_per_step",
        "batt_ramp_kw_per_step",
        "dominance_deadband_kw",
    ]:
        if key in control_correction:
            setattr(config, key, control_correction[key])
    _apply_env_values(config, overrides, "explicit env overrides")
    return config


class ShipEnvDualSide(ShipEnvBase):
    """Minimal dual-side vessel environment for the next implementation phase."""

    def __init__(self, data: pd.DataFrame, config: DualSideEnvConfig | None = None):
        self.df = data.reset_index(drop=True).copy()
        self.config = config or DualSideEnvConfig()
        if "dqn_control_seconds" in self.df.columns and not self.df["dqn_control_seconds"].dropna().empty:
            self.config.dt_seconds = float(self.df["dqn_control_seconds"].dropna().iloc[0])
        self.dt_hours = self.config.dt_seconds / 3600.0
        self.actions = [
            (fc_split, batt_split, fc_trim_kw, batt_trim_kw)
            for fc_split in self.config.split_candidates
            for batt_split in self.config.split_candidates
            for fc_trim_kw in self.config.fc_trim_candidates_kw
            for batt_trim_kw in self.config.batt_trim_candidates_kw
        ]
        self.action_dim = len(self.actions)
        self.state_dim = 23
        self.pointer = 0
        self.episode_step = 0
        self.current_row = None
        center_action = (0.5, 0.5, 0.0, 0.0)
        self.prev_action = self.actions.index(center_action) if center_action in self.actions else len(self.actions) // 2

        self.left_battery = BatteryCluster(
            BatteryClusterConfig(
                name="left_battery",
                capacity_kwh=self.config.batt_cap_left_kwh,
                soc_min=self.config.soc_min,
                soc_max=self.config.soc_max,
                charge_power_max_kw=self.config.batt_charge_max_kw,
                discharge_power_max_kw=self.config.batt_discharge_max_kw,
            )
        )
        self.right_battery = BatteryCluster(
            BatteryClusterConfig(
                name="right_battery",
                capacity_kwh=self.config.batt_cap_right_kwh,
                soc_min=self.config.soc_min,
                soc_max=self.config.soc_max,
                charge_power_max_kw=self.config.batt_charge_max_kw,
                discharge_power_max_kw=self.config.batt_discharge_max_kw,
            )
        )
        self.left_fc = FuelCellStack(
            FuelCellStackConfig(
                name="left_fc",
                power_min_kw=0.0,
                power_max_kw=self.config.fc_max_left_kw,
                ramp_rate_kw=self.config.fc_ramp_kw,
            )
        )
        self.right_fc = FuelCellStack(
            FuelCellStackConfig(
                name="right_fc",
                power_min_kw=0.0,
                power_max_kw=self.config.fc_max_right_kw,
                ramp_rate_kw=self.config.fc_ramp_kw,
            )
        )
        self.prev_battery_balance_bias_kw = 0.0
        self.fc_soc_support_hold_latched = False
        self.fc_power_ema_kw = 0.0

    def _get_obs(self) -> np.ndarray:
        row = self.current_row
        fc_ref_left = float(row.get("mpc_fuel_cell_ref_left_kw", row.get("mpc_fuel_cell_ref_kw", row["fuel_cell_power_total_kw"]) * 0.5))
        fc_ref_right = float(row.get("mpc_fuel_cell_ref_right_kw", row.get("mpc_fuel_cell_ref_kw", row["fuel_cell_power_total_kw"]) * 0.5))
        batt_ref_left = float(row.get("mpc_battery_ref_left_kw", row.get("mpc_battery_ref_kw", row["battery_power_total_kw"]) * 0.5))
        batt_ref_right = float(row.get("mpc_battery_ref_right_kw", row.get("mpc_battery_ref_kw", row["battery_power_total_kw"]) * 0.5))
        soc_ref = float(row.get("mpc_soc_ref", row.get("soc_ref", self.config.soc_ref_default)))
        return np.asarray(
            [
                float(row["load_left_kw"]),
                float(row["load_right_kw"]),
                float(row.get("speed_knots", 0.0)),
                float(self.left_battery.soc),
                float(self.right_battery.soc),
                float(self.left_fc.power_kw),
                float(self.right_fc.power_kw),
                fc_ref_left,
                fc_ref_right,
                batt_ref_left,
                batt_ref_right,
                float(row["load_total_kw"]),
                float(row.get("fuel_cell_power_total_kw", 0.0)),
                float(row.get("battery_power_total_kw", 0.0)),
                float(self.left_fc.power_kw - fc_ref_left),
                float(self.right_fc.power_kw - fc_ref_right),
                float(self.left_battery.power_kw - batt_ref_left),
                float(self.right_battery.power_kw - batt_ref_right),
                soc_ref,
                float(self.left_battery.soc - soc_ref),
                float(self.right_battery.soc - soc_ref),
                float(self.actions[self.prev_action][0] - 0.5),
                float(self.actions[self.prev_action][1] - 0.5),
            ],
            dtype=np.float32,
        )

    def reset(self, start_index: int | None = None):
        max_start = max(0, len(self.df) - max(2, self.config.episode_steps + 1))
        if start_index is None:
            ep_starts = list(self.config.episode_start_indices)
            if ep_starts:
                valid = [s for s in ep_starts if 0 <= s <= max_start]
                start_index = int(valid[np.random.randint(0, len(valid))]) if valid else 0
            else:
                start_index = int(np.random.randint(0, max_start + 1)) if self.config.random_reset and max_start > 0 else 0
        self.pointer = int(start_index)
        self.episode_step = 0
        self.current_row = self.df.iloc[self.pointer]
        # SOC init — random if enabled, else from CSV
        if self.config.soc_random_init_enabled:
            lo = self.config.soc_random_init_low
            hi = self.config.soc_random_init_high
            soc_init = float(np.random.uniform(lo, hi))
            self.left_battery.reset(soc_init)
            self.right_battery.reset(soc_init)
        else:
            self.left_battery.reset(float(self.current_row.get("soc_left", self.current_row.get("soc_mean", 0.6))))
            self.right_battery.reset(float(self.current_row.get("soc_right", self.current_row.get("soc_mean", 0.6))))
        self.left_fc.reset(float(self.current_row.get("fuel_cell_power_left_kw", 0.0)))
        self.right_fc.reset(float(self.current_row.get("fuel_cell_power_right_kw", 0.0)))
        center_action = (0.5, 0.5, 0.0, 0.0)
        self.prev_action = self.actions.index(center_action) if center_action in self.actions else len(self.actions) // 2
        self.prev_battery_balance_bias_kw = 0.0
        self.fc_soc_support_hold_latched = False
        self.fc_power_ema_kw = float(self.left_fc.power_kw + self.right_fc.power_kw)
        self.prev_soc = 0.5 * (self.left_battery.soc + self.right_battery.soc)
        return self._get_obs(), {"pointer": self.pointer}

    def step(self, action):
        action = int(action)
        fc_split, batt_split, fc_trim_kw, batt_trim_kw = self.actions[action]
        row = self.current_row
        fc_ref_total = float(row.get("mpc_fuel_cell_ref_kw", row["fuel_cell_power_total_kw"]))
        fc_base_ref_total = float(row.get("mpc_fuel_cell_base_ref_kw", fc_ref_total))
        batt_ref_total = float(row.get("mpc_battery_ref_kw", row["battery_power_total_kw"]))
        soc_ref = float(row.get("mpc_soc_ref", row.get("soc_ref", self.config.soc_ref_default)))
        prev_state = {
            "p_fc_kw": self.left_fc.power_kw + self.right_fc.power_kw,
            "p_bat_kw": self.left_battery.power_kw + self.right_battery.power_kw,
            "soc": 0.5 * (self.left_battery.soc + self.right_battery.soc),
            "soc_ref": soc_ref,
            "p_fc_ema_kw": float(self.fc_power_ema_kw),
        }
        fc_ref_total_adjusted = fc_ref_total + fc_trim_kw
        batt_ref_total_adjusted = batt_ref_total + batt_trim_kw
        fc_ref_left = float(row.get("mpc_fuel_cell_ref_left_kw", fc_ref_total_adjusted * fc_split))
        fc_ref_right = float(row.get("mpc_fuel_cell_ref_right_kw", fc_ref_total_adjusted * (1.0 - fc_split)))
        batt_ref_left = float(row.get("mpc_battery_ref_left_kw", batt_ref_total_adjusted * batt_split))
        batt_ref_right = float(row.get("mpc_battery_ref_right_kw", batt_ref_total_adjusted * (1.0 - batt_split)))

        # The action modulates the left/right split around the upper-layer side references.
        left_fc_ref_action = fc_ref_left + fc_ref_total_adjusted * (fc_split - 0.5)
        right_fc_ref_action = fc_ref_right - fc_ref_total_adjusted * (fc_split - 0.5)
        left_batt_ref_action = batt_ref_left + batt_ref_total_adjusted * (batt_split - 0.5)
        right_batt_ref_action = batt_ref_right - batt_ref_total_adjusted * (batt_split - 0.5)

        # The lower layer is valuable only if it compensates actuator mismatch and
        # short-term disturbance that the upper layer does not model explicitly.
        load_ramp_proxy = float(row.get("load_ramp_kw", 0.0))
        left_fc_command = (
            self.config.fc_command_gain_left * left_fc_ref_action
            + self.config.disturbance_from_ramp_gain * load_ramp_proxy
        )
        right_fc_command = (
            self.config.fc_command_gain_right * right_fc_ref_action
            - self.config.disturbance_from_ramp_gain * load_ramp_proxy
        )
        raw_left_batt_command = (
            self.config.batt_command_gain_left * left_batt_ref_action
            - self.config.disturbance_from_ramp_gain * load_ramp_proxy
        )
        raw_right_batt_command = (
            self.config.batt_command_gain_right * right_batt_ref_action
            + self.config.disturbance_from_ramp_gain * load_ramp_proxy
        )
        left_soc_before_step = float(self.left_battery.soc)
        right_soc_before_step = float(self.right_battery.soc)
        mean_soc_before_step = 0.5 * (left_soc_before_step + right_soc_before_step)
        fc_soc_support_trigger_soc = (
            float(self.config.fc_soc_support_start_soc)
            if self.config.fc_soc_support_start_soc > 0.0
            else soc_ref - self.config.fc_soc_support_deadband
        )
        fc_soc_support_medium_trigger_soc = float(self.config.fc_soc_support_medium_start_soc)
        fc_soc_support_strong_trigger_soc = float(self.config.fc_soc_support_strong_start_soc)
        fc_soc_support_full_trigger_soc = float(self.config.fc_soc_support_full_start_soc)

        def _smoothstep(edge_low: float, edge_high: float, value: float) -> float:
            if edge_high <= edge_low:
                return 1.0 if value <= edge_low else 0.0
            ratio = float(np.clip((edge_high - value) / (edge_high - edge_low), 0.0, 1.0))
            return ratio * ratio * (3.0 - 2.0 * ratio)

        def _side_fc_support_target_kw(side_soc: float) -> tuple[float, float]:
            if not self.config.fc_soc_support_piecewise:
                deficit = max(0.0, fc_soc_support_trigger_soc - side_soc)
                return min(0.5 * self.config.fc_soc_support_max_kw, self.config.fc_soc_support_gain_kw * deficit), deficit

            total_deficit = max(0.0, fc_soc_support_trigger_soc - side_soc)
            if total_deficit <= 0.0:
                return 0.0, 0.0
            light_side_limit_kw = 0.5 * self.config.fc_soc_support_light_max_kw
            medium_side_limit_kw = 0.5 * self.config.fc_soc_support_medium_max_kw
            strong_side_limit_kw = 0.5 * self.config.fc_soc_support_max_kw
            if side_soc >= fc_soc_support_medium_trigger_soc:
                fraction = _smoothstep(fc_soc_support_medium_trigger_soc, fc_soc_support_trigger_soc, side_soc)
                return light_side_limit_kw * fraction, total_deficit
            if side_soc >= fc_soc_support_strong_trigger_soc:
                fraction = _smoothstep(fc_soc_support_strong_trigger_soc, fc_soc_support_medium_trigger_soc, side_soc)
                support = light_side_limit_kw + (medium_side_limit_kw - light_side_limit_kw) * fraction
                return support, total_deficit
            if side_soc >= fc_soc_support_full_trigger_soc:
                fraction = _smoothstep(fc_soc_support_full_trigger_soc, fc_soc_support_strong_trigger_soc, side_soc)
                support = medium_side_limit_kw + (strong_side_limit_kw - medium_side_limit_kw) * fraction
                return support, total_deficit
            return strong_side_limit_kw, total_deficit

        def _limit_support_by_battery_need(target_support_kw: float, raw_battery_command_kw: float) -> tuple[float, bool]:
            if not self.config.fc_soc_support_battery_need_aware:
                return target_support_kw, False
            limited_support_kw = float(np.clip(target_support_kw + raw_battery_command_kw, 0.0, target_support_kw))
            return limited_support_kw, limited_support_kw + 1e-9 < target_support_kw

        fc_soc_support_scale = 1.0
        if self.config.fc_soc_support_hold_hysteresis_enabled:
            if self.fc_soc_support_hold_latched:
                if mean_soc_before_step < float(self.config.fc_soc_support_hold_reenable_soc):
                    self.fc_soc_support_hold_latched = False
            elif mean_soc_before_step >= float(self.config.fc_soc_support_hold_disable_soc):
                self.fc_soc_support_hold_latched = True
            if self.fc_soc_support_hold_latched:
                fc_soc_support_scale = float(self.config.fc_soc_support_hold_scale)

        left_fc_soc_support_target_kw, left_soc_support_deficit = _side_fc_support_target_kw(left_soc_before_step)
        right_fc_soc_support_target_kw, right_soc_support_deficit = _side_fc_support_target_kw(right_soc_before_step)
        left_fc_soc_support_target_kw *= fc_soc_support_scale
        right_fc_soc_support_target_kw *= fc_soc_support_scale
        left_fc_soc_support_kw, left_fc_soc_support_need_limited = _limit_support_by_battery_need(
            left_fc_soc_support_target_kw,
            raw_left_batt_command,
        )
        right_fc_soc_support_kw, right_fc_soc_support_need_limited = _limit_support_by_battery_need(
            right_fc_soc_support_target_kw,
            raw_right_batt_command,
        )
        fc_soc_support_kw = left_fc_soc_support_kw + right_fc_soc_support_kw
        left_fc_command += left_fc_soc_support_kw
        right_fc_command += right_fc_soc_support_kw
        raw_battery_total_kw = raw_left_batt_command + raw_right_batt_command
        fc_surplus_charge_guard_kw = 0.0
        fc_surplus_charge_guard_active = False
        if (
            self.config.prevent_high_soc_fc_surplus_charge
            and fc_soc_support_kw <= self.config.fc_surplus_charge_guard_eps_kw
            and mean_soc_before_step >= soc_ref + self.config.fc_surplus_charge_guard_margin
        ):
            total_fc_command_kw = left_fc_command + right_fc_command
            desired_battery_discharge_kw = max(0.0, raw_battery_total_kw)
            max_fc_without_charge_kw = max(0.0, float(row["load_total_kw"]) - desired_battery_discharge_kw)
            if total_fc_command_kw > max_fc_without_charge_kw + self.config.fc_surplus_charge_guard_eps_kw:
                scale = max_fc_without_charge_kw / max(total_fc_command_kw, 1e-6)
                original_total_fc_command_kw = total_fc_command_kw
                left_fc_command *= scale
                right_fc_command *= scale
                fc_surplus_charge_guard_kw = original_total_fc_command_kw - left_fc_command - right_fc_command
                fc_surplus_charge_guard_active = True

        fc_allocator_total_command_kw = left_fc_command + right_fc_command
        if str(self.config.fuel_cell_allocator_mode).lower() == "symmetric_total":
            fc_allocation = allocate_symmetric_fuel_cell(
                total_cmd_kw=fc_allocator_total_command_kw,
                previous_left_kw=self.left_fc.power_kw,
                previous_right_kw=self.right_fc.power_kw,
                left_min_kw=0.0,
                left_max_kw=self.config.fc_max_left_kw,
                right_min_kw=0.0,
                right_max_kw=self.config.fc_max_right_kw,
                ramp_kw_per_step=self.config.fc_ramp_kw,
            )
            left_fc_command = fc_allocation.left_kw
            right_fc_command = fc_allocation.right_kw
            fc_allocator_symmetric = fc_allocation.fully_symmetric
            fc_allocator_constraint_adjusted = fc_allocation.constraint_adjusted
            fc_allocator_correction_kw = fc_allocation.correction_kw
            fc_allocator_total_error_kw = fc_allocation.total_error_kw
        else:
            fc_allocator_symmetric = bool(np.isclose(left_fc_command, right_fc_command, atol=1e-9))
            fc_allocator_constraint_adjusted = False
            fc_allocator_correction_kw = 0.0
            fc_allocator_total_error_kw = 0.0

        left_fc = self.left_fc.step(left_fc_command)
        right_fc = self.right_fc.step(right_fc_command)

        # Close total bus power with the battery layer after fuel-cell ramp/bound
        # limits are applied. The DQN battery split still decides how the total
        # residual correction is shared between the two sides.
        required_battery_total_kw = float(row["load_total_kw"]) - left_fc["power_kw"] - right_fc["power_kw"]
        balance_correction_kw = required_battery_total_kw - raw_battery_total_kw
        if self.config.enforce_power_balance:
            battery_allocator_mode = str(self.config.battery_allocator_mode).lower()
            if battery_allocator_mode == "soft_balance":
                battery_allocation = allocate_soft_balanced_battery(
                    total_cmd_kw=required_battery_total_kw,
                    soc_left=left_soc_before_step,
                    soc_right=right_soc_before_step,
                    previous_left_kw=self.left_battery.power_kw,
                    previous_right_kw=self.right_battery.power_kw,
                    charge_max_kw=self.config.batt_charge_max_kw,
                    discharge_max_kw=self.config.batt_discharge_max_kw,
                    previous_balance_bias_kw=self.prev_battery_balance_bias_kw,
                    config=BatterySoftBalanceConfig(
                        soc_balance_deadband=self.config.soc_balance_deadband,
                        soc_balance_bias_gain_kw_per_soc=self.config.soc_balance_bias_gain_kw_per_soc,
                        max_balance_bias_kw=self.config.max_balance_bias_kw,
                        max_balance_bias_ratio=self.config.max_balance_bias_ratio,
                        balance_bias_smoothing_alpha=self.config.balance_bias_smoothing_alpha,
                        balance_bias_rate_limit_kw_per_step=self.config.balance_bias_rate_limit_kw_per_step,
                        batt_ramp_kw_per_step=self.config.batt_ramp_kw_per_step,
                        dominance_deadband_kw=self.config.dominance_deadband_kw,
                    ),
                )
                left_batt_command = battery_allocation.left_kw
                right_batt_command = battery_allocation.right_kw
                self.prev_battery_balance_bias_kw = battery_allocation.final_balance_bias_kw
                soc_adjusted_batt_split = (
                    0.5
                    if abs(required_battery_total_kw) <= 1e-9
                    else float(left_batt_command / required_battery_total_kw)
                )
                battery_balance_bias_kw = battery_allocation.final_balance_bias_kw
                battery_desired_balance_bias_kw = battery_allocation.desired_balance_bias_kw
                battery_smoothed_balance_bias_kw = battery_allocation.smoothed_balance_bias_kw
                battery_previous_balance_bias_kw = battery_allocation.previous_balance_bias_kw
                battery_soc_deadband_active = battery_allocation.soc_deadband_active
                battery_bias_limited = battery_allocation.bias_limited
                battery_bias_rate_limited = battery_allocation.bias_rate_limited
                battery_allocator_constraint_adjusted = battery_allocation.constraint_adjusted
                battery_allocator_compensation_kw = battery_allocation.compensation_kw
                battery_allocator_total_error_kw = battery_allocation.total_error_kw
                battery_allocator_feasible_total_kw = battery_allocation.feasible_total_kw
                battery_allocator_base_left_kw = battery_allocation.base_left_kw
                battery_allocator_base_right_kw = battery_allocation.base_right_kw
            else:
                soc_delta = left_soc_before_step - right_soc_before_step
                if required_battery_total_kw >= 0.0:
                    soc_adjusted_batt_split = batt_split + self.config.battery_soc_balance_gain * soc_delta
                else:
                    soc_adjusted_batt_split = batt_split - self.config.battery_soc_balance_gain * soc_delta
                soc_adjusted_batt_split = float(
                    np.clip(
                        soc_adjusted_batt_split,
                        self.config.battery_split_min,
                        self.config.battery_split_max,
                    )
                )
                left_batt_command = required_battery_total_kw * soc_adjusted_batt_split
                right_batt_command = required_battery_total_kw * (1.0 - soc_adjusted_batt_split)
                battery_balance_bias_kw = 0.5 * (left_batt_command - right_batt_command)
                battery_desired_balance_bias_kw = battery_balance_bias_kw
                battery_smoothed_balance_bias_kw = battery_balance_bias_kw
                battery_previous_balance_bias_kw = self.prev_battery_balance_bias_kw
                self.prev_battery_balance_bias_kw = battery_balance_bias_kw
                battery_soc_deadband_active = False
                battery_bias_limited = False
                battery_bias_rate_limited = False
                battery_allocator_constraint_adjusted = False
                battery_allocator_compensation_kw = 0.0
                battery_allocator_total_error_kw = 0.0
                battery_allocator_feasible_total_kw = required_battery_total_kw
                battery_allocator_base_left_kw = 0.5 * required_battery_total_kw
                battery_allocator_base_right_kw = 0.5 * required_battery_total_kw
        else:
            soc_adjusted_batt_split = batt_split
            left_batt_command = raw_left_batt_command
            right_batt_command = raw_right_batt_command
            battery_balance_bias_kw = 0.5 * (left_batt_command - right_batt_command)
            battery_desired_balance_bias_kw = battery_balance_bias_kw
            battery_smoothed_balance_bias_kw = battery_balance_bias_kw
            battery_previous_balance_bias_kw = self.prev_battery_balance_bias_kw
            self.prev_battery_balance_bias_kw = battery_balance_bias_kw
            battery_soc_deadband_active = False
            battery_bias_limited = False
            battery_bias_rate_limited = False
            battery_allocator_constraint_adjusted = False
            battery_allocator_compensation_kw = 0.0
            battery_allocator_total_error_kw = 0.0
            battery_allocator_feasible_total_kw = left_batt_command + right_batt_command
            battery_allocator_base_left_kw = 0.5 * (left_batt_command + right_batt_command)
            battery_allocator_base_right_kw = 0.5 * (left_batt_command + right_batt_command)

        left_batt = self.left_battery.step(left_batt_command, self.dt_hours)
        right_batt = self.right_battery.step(right_batt_command, self.dt_hours)

        bus = ShipPowerBus(
            sides=[
                ShipSideState(
                    name="left",
                    fuel_cell_power_kw=left_fc["power_kw"],
                    battery_power_kw=left_batt["power_kw"],
                    load_power_kw=float(row["load_left_kw"]),
                ),
                ShipSideState(
                    name="right",
                    fuel_cell_power_kw=right_fc["power_kw"],
                    battery_power_kw=right_batt["power_kw"],
                    load_power_kw=float(row["load_right_kw"]),
                ),
            ]
        )
        left_fc_tracking_error = abs(left_fc["power_kw"] - fc_ref_left)
        right_fc_tracking_error = abs(right_fc["power_kw"] - fc_ref_right)
        fc_tracking_error = left_fc_tracking_error + right_fc_tracking_error
        left_fc_support_adjusted_tracking_error = abs(left_fc["power_kw"] - (fc_ref_left + left_fc_soc_support_kw))
        right_fc_support_adjusted_tracking_error = abs(right_fc["power_kw"] - (fc_ref_right + right_fc_soc_support_kw))
        fc_support_adjusted_tracking_error = (
            left_fc_support_adjusted_tracking_error + right_fc_support_adjusted_tracking_error
        )
        left_battery_reference_tracking_error = abs(left_batt["power_kw"] - batt_ref_left)
        right_battery_reference_tracking_error = abs(right_batt["power_kw"] - batt_ref_right)
        battery_reference_tracking_error = (
            left_battery_reference_tracking_error + right_battery_reference_tracking_error
        )
        left_battery_command_tracking_error = abs(left_batt["power_kw"] - left_batt_command)
        right_battery_command_tracking_error = abs(right_batt["power_kw"] - right_batt_command)
        battery_command_tracking_error = left_battery_command_tracking_error + right_battery_command_tracking_error
        left_tracking_error = left_fc_tracking_error + left_battery_reference_tracking_error
        right_tracking_error = right_fc_tracking_error + right_battery_reference_tracking_error
        total_tracking_error = left_tracking_error + right_tracking_error
        split_penalty = abs(bus.sides[0].balance_error_kw - bus.sides[1].balance_error_kw)
        soc_balance_penalty = abs(self.left_battery.soc - self.right_battery.soc)
        total_balance_error = abs(bus.total_supply_kw() - float(row["load_total_kw"]))
        center_bias_penalty = abs(fc_split - 0.5) + abs(batt_split - 0.5)
        trim_penalty = abs(fc_trim_kw) + abs(batt_trim_kw)
        prev_fc_split, prev_batt_split, prev_fc_trim_kw, prev_batt_trim_kw = self.actions[self.prev_action]
        action_switch_penalty = (
            abs(fc_split - prev_fc_split)
            + abs(batt_split - prev_batt_split)
            + 0.05 * abs(fc_trim_kw - prev_fc_trim_kw)
            + 0.05 * abs(batt_trim_kw - prev_batt_trim_kw)
        )
        episode_progress = float(self.episode_step / max(1, self.config.episode_steps - 1))
        constraint_penalty = (
            left_fc["ramp_violation"]
            + left_fc["bound_violation"]
            + right_fc["ramp_violation"]
            + right_fc["bound_violation"]
            + left_batt["power_violation"]
            + left_batt["soc_violation"]
            + right_batt["power_violation"]
            + right_batt["soc_violation"]
        )
        next_state = {
            "p_fc_kw": left_fc["power_kw"] + right_fc["power_kw"],
            "p_bat_kw": left_batt["power_kw"] + right_batt["power_kw"],
            "p_bat_left_kw": left_batt["power_kw"],
            "p_bat_right_kw": right_batt["power_kw"],
            "soc": 0.5 * (self.left_battery.soc + self.right_battery.soc),
            "soc_left": self.left_battery.soc,
            "soc_right": self.right_battery.soc,
            "soc_ref": soc_ref,
            "action_switch_penalty": action_switch_penalty,
            "p_fc_ema_kw": float(self.fc_power_ema_kw),
            "episode_progress": episode_progress,
        }
        if self.config.reward_mode == "soc_asymmetric":
            fc_trim = fc_trim_kw
            soc_now = 0.5 * (self.left_battery.soc + self.right_battery.soc)
            # Main: asymmetric SOC safety penalty
            k1, k2 = 3.0, 2.0
            if soc_now < 0.4:
                r_soc = -k1 * (0.4 - soc_now) ** 2
            elif soc_now > 0.8:
                r_soc = -k2 * (soc_now - 0.8) ** 2
            else:
                r_soc = 0.0
            # Trim penalty
            lam2 = 0.5
            r_trim = -lam2 * (fc_trim ** 2)
            # Weak fuel penalty
            lam3 = 0.03
            r_fuel = -lam3 * float(left_fc["power_kw"] + right_fc["power_kw"])
            # SOC trend reward
            w_trend = 0.2
            dsoc = soc_now - self.prev_soc
            if soc_now < 0.5:
                r_trend = w_trend * dsoc
            elif soc_now > 0.7:
                r_trend = w_trend * (-dsoc)
            else:
                r_trend = 0.0
            reward = r_soc + r_trim + r_fuel + r_trend
            reward_info = {
                "r_soc": float(r_soc), "r_trim": float(r_trim),
                "r_fuel": float(r_fuel), "r_trend": float(r_trend),
                "total_reward": float(reward), "soc": float(soc_now),
                "soc_ref": float(soc_ref), "fc_trim_kw": float(fc_trim),
                "tracking_error_kw": 0.0, "balance_error_kw": 0.0,
                "total_balance_error_kw": 0.0, "soc_abs_error": abs(soc_now - soc_ref),
            }
            self.prev_soc = soc_now
        else:
            reward, reward_info = calculate_reward(
                state=prev_state,
                action=action,
                next_state=next_state,
                refs=RewardRefs(p_fc_ref_kw=fc_ref_total_adjusted, p_fc_base_ref_kw=fc_base_ref_total, soc_ref=soc_ref),
                params=RewardParams(
                    p_fc_min_kw=0.0,
                    p_fc_max_kw=self.config.fc_max_left_kw + self.config.fc_max_right_kw,
                    p_bat_min_kw=-2.0 * self.config.batt_charge_max_kw,
                    p_bat_max_kw=2.0 * self.config.batt_discharge_max_kw,
                    soc_min=self.config.soc_min,
                    soc_max=self.config.soc_max,
                    weights=self.config.reward_weights,
                    fc_var_reward_alpha=float(self.config.fc_var_reward_alpha),
                    fc_var_reward_soc_gate=float(self.config.fc_var_reward_soc_gate),
                    fc_var_reward_progress_gate=float(self.config.fc_var_reward_progress_gate),
                    soc_balance_highsoc_scale=float(self.config.soc_balance_highsoc_scale),
                    soc_direction_fc_scale=float(self.config.soc_direction_fc_scale),
                    soc_battery_direction_scale=float(self.config.soc_battery_direction_scale),
                    soc_correction_credit_scale=float(self.config.soc_correction_credit_scale),
                    fc_trim_penalty_scale=float(self.config.fc_trim_penalty_scale),
                    battery_corrective_use_discount=float(self.config.battery_corrective_use_discount),
                    battery_wrong_direction_use_penalty=float(self.config.battery_wrong_direction_use_penalty),
                    battery_use_soc_factor_min=float(self.config.battery_use_soc_factor_min),
                    battery_use_soc_factor_max=float(self.config.battery_use_soc_factor_max),
                ),
            )
        ema_alpha = float(np.clip(self.config.fc_var_reward_ema_alpha, 0.0, 1.0))
        next_fc_total_kw = float(next_state["p_fc_kw"])
        self.fc_power_ema_kw = (
            next_fc_total_kw
            if ema_alpha <= 0.0
            else ema_alpha * next_fc_total_kw + (1.0 - ema_alpha) * float(self.fc_power_ema_kw)
        )
        self.prev_action = action

        self.pointer += 1
        self.episode_step += 1
        terminated = self.pointer >= len(self.df) - 1
        truncated = self.episode_step >= self.config.episode_steps
        done = terminated or truncated
        if not done:
            self.current_row = self.df.iloc[self.pointer]

        obs = self._get_obs() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = bus.as_dict()
        info.update(
            {
                "left_fuel_cell_ref_kw": fc_ref_left,
                "right_fuel_cell_ref_kw": fc_ref_right,
                "left_battery_ref_kw": batt_ref_left,
                "right_battery_ref_kw": batt_ref_right,
                "left_fuel_cell_power_kw": left_fc["power_kw"],
                "right_fuel_cell_power_kw": right_fc["power_kw"],
                "left_battery_power_kw": left_batt["power_kw"],
                "right_battery_power_kw": right_batt["power_kw"],
                "fuel_cell_power_kw": left_fc["power_kw"] + right_fc["power_kw"],
                "battery_power_kw": left_batt["power_kw"] + right_batt["power_kw"],
                "soc": 0.5 * (self.left_battery.soc + self.right_battery.soc),
                "left_soc": self.left_battery.soc,
                "right_soc": self.right_battery.soc,
                "soc_ref": soc_ref,
                "tracking_error_kw": total_tracking_error,
                "episode_progress": episode_progress,
                "p_fc_ema_kw": float(next_state["p_fc_ema_kw"]),
                "fc_tracking_error_kw": fc_tracking_error,
                "fc_support_adjusted_tracking_error_kw": fc_support_adjusted_tracking_error,
                "battery_reference_tracking_error_kw": battery_reference_tracking_error,
                "battery_command_tracking_error_kw": battery_command_tracking_error,
                "left_tracking_error_kw": left_tracking_error,
                "right_tracking_error_kw": right_tracking_error,
                "left_fc_tracking_error_kw": left_fc_tracking_error,
                "right_fc_tracking_error_kw": right_fc_tracking_error,
                "left_fc_support_adjusted_tracking_error_kw": left_fc_support_adjusted_tracking_error,
                "right_fc_support_adjusted_tracking_error_kw": right_fc_support_adjusted_tracking_error,
                "left_battery_reference_tracking_error_kw": left_battery_reference_tracking_error,
                "right_battery_reference_tracking_error_kw": right_battery_reference_tracking_error,
                "left_battery_command_tracking_error_kw": left_battery_command_tracking_error,
                "right_battery_command_tracking_error_kw": right_battery_command_tracking_error,
                "total_balance_error_kw": total_balance_error,
                "split_penalty_kw": split_penalty,
                "soc_balance_penalty": soc_balance_penalty,
                "center_bias_penalty": center_bias_penalty,
                "action_switch_penalty": action_switch_penalty,
                "trim_penalty_kw": trim_penalty,
                "constraint_penalty": constraint_penalty,
                "selected_fc_split": fc_split,
                "selected_battery_split": batt_split,
                "selected_fc_trim_kw": fc_trim_kw,
                "selected_battery_trim_kw": batt_trim_kw,
                "fuel_cell_allocator_mode": self.config.fuel_cell_allocator_mode,
                "fc_allocator_requested_total_kw": fc_allocator_total_command_kw,
                "fc_allocator_fully_symmetric": fc_allocator_symmetric,
                "fc_allocator_constraint_adjusted": fc_allocator_constraint_adjusted,
                "fc_allocator_correction_kw": fc_allocator_correction_kw,
                "fc_allocator_total_error_kw": fc_allocator_total_error_kw,
                "battery_allocator_mode": self.config.battery_allocator_mode,
                "battery_allocator_base_left_kw": battery_allocator_base_left_kw,
                "battery_allocator_base_right_kw": battery_allocator_base_right_kw,
                "battery_balance_bias_kw": battery_balance_bias_kw,
                "battery_desired_balance_bias_kw": battery_desired_balance_bias_kw,
                "battery_smoothed_balance_bias_kw": battery_smoothed_balance_bias_kw,
                "battery_previous_balance_bias_kw": battery_previous_balance_bias_kw,
                "battery_soc_deadband_active": battery_soc_deadband_active,
                "battery_bias_limited": battery_bias_limited,
                "battery_bias_rate_limited": battery_bias_rate_limited,
                "battery_allocator_constraint_adjusted": battery_allocator_constraint_adjusted,
                "battery_allocator_compensation_kw": battery_allocator_compensation_kw,
                "battery_allocator_total_error_kw": battery_allocator_total_error_kw,
                "battery_allocator_feasible_total_kw": battery_allocator_feasible_total_kw,
                "left_fc_command_kw": left_fc_command,
                "right_fc_command_kw": right_fc_command,
                "raw_left_battery_command_kw": raw_left_batt_command,
                "raw_right_battery_command_kw": raw_right_batt_command,
                "left_battery_command_kw": left_batt_command,
                "right_battery_command_kw": right_batt_command,
                "battery_balance_correction_kw": balance_correction_kw,
                "required_battery_total_kw": required_battery_total_kw,
                "fc_soc_support_kw": fc_soc_support_kw,
                "fc_soc_support_trigger_soc": fc_soc_support_trigger_soc,
                "fc_soc_support_medium_trigger_soc": fc_soc_support_medium_trigger_soc,
                "fc_soc_support_strong_trigger_soc": fc_soc_support_strong_trigger_soc,
                "fc_soc_support_full_trigger_soc": fc_soc_support_full_trigger_soc,
                "fc_soc_support_scale": fc_soc_support_scale,
                "fc_soc_support_hold_latched": self.fc_soc_support_hold_latched,
                "fc_soc_support_piecewise": self.config.fc_soc_support_piecewise,
                "fc_soc_support_battery_need_aware": self.config.fc_soc_support_battery_need_aware,
                "fc_surplus_charge_guard_kw": fc_surplus_charge_guard_kw,
                "fc_surplus_charge_guard_active": fc_surplus_charge_guard_active,
                "left_fc_soc_support_kw": left_fc_soc_support_kw,
                "right_fc_soc_support_kw": right_fc_soc_support_kw,
                "left_fc_soc_support_target_kw": left_fc_soc_support_target_kw,
                "right_fc_soc_support_target_kw": right_fc_soc_support_target_kw,
                "left_fc_soc_support_need_limited": left_fc_soc_support_need_limited,
                "right_fc_soc_support_need_limited": right_fc_soc_support_need_limited,
                "left_soc_support_deficit": left_soc_support_deficit,
                "right_soc_support_deficit": right_soc_support_deficit,
                "soc_support_deficit": 0.5 * (left_soc_support_deficit + right_soc_support_deficit),
                "soc_adjusted_battery_split": soc_adjusted_batt_split,
            }
        )
        info.update(reward_info)
        info["legacy_tracking_error_kw"] = total_tracking_error
        info["legacy_total_balance_error_kw"] = total_balance_error
        info["legacy_constraint_penalty"] = constraint_penalty
        return obs, float(reward), done, truncated, info

    def describe_state_space(self) -> dict:
        return {
            "meaning": "dual-side vessel microgrid state for lower-layer DQN",
            "features": [
                "load_left_kw",
                "load_right_kw",
                "speed_knots",
                "left_soc",
                "right_soc",
                "left_fuel_cell_power_kw",
                "right_fuel_cell_power_kw",
                "left_fuel_cell_ref_kw",
                "right_fuel_cell_ref_kw",
                "left_battery_ref_kw",
                "right_battery_ref_kw",
                "load_total_kw",
                "fuel_cell_power_total_kw",
                "battery_power_total_kw",
                "left_fuel_cell_tracking_error_kw",
                "right_fuel_cell_tracking_error_kw",
                "left_battery_tracking_error_kw",
                "right_battery_tracking_error_kw",
                "soc_ref",
                "left_soc_tracking_error",
                "right_soc_tracking_error",
                "previous_fc_split_delta",
                "previous_battery_split_delta",
            ],
        }

    def describe_action_space(self) -> dict:
        return {
            "meaning": "discrete lower-layer action for side split and total power trim",
            "action_count": self.action_dim,
            "action_table": self.actions,
        }

    def describe_reward(self) -> dict:
        return {
            "meaning": "four-group normalized lower-layer DQN reward with aggregate dual-side placeholders",
            "terms": {
                "energy_norm": self.config.reward_weights.w1,
                "fc_dynamic_norm": self.config.reward_weights.w2,
                "soc_norm": self.config.reward_weights.w3,
                "battery_use_norm": self.config.reward_weights.w4,
            },
            "fc_variance_alignment": {
                "alpha": float(self.config.fc_var_reward_alpha),
                "ema_alpha": float(self.config.fc_var_reward_ema_alpha),
                "soc_gate": float(self.config.fc_var_reward_soc_gate),
                "progress_gate": float(self.config.fc_var_reward_progress_gate),
                "soc_balance_highsoc_scale": float(self.config.soc_balance_highsoc_scale),
            },
        }
