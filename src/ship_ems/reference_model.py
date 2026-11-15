"""SimulationCode ship dynamics adapted to the project's 1 s load profiles."""

from math import exp, isfinite
from typing import NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CONFIG


FC_RATED_KW = CONFIG.fc_total_rated_gross_kw
FC_MIN_KW = 0.0
# Preserve the source coordinate P_rated / 2560 = 0.3125 at the new rating.
FC_NORMALIZATION_KW = FC_RATED_KW / (800.0 / 2560.0)
FC_RUNNING_THRESHOLD_KW = 0.1  # Physical kW cutoff in shipenv.py.
FC_LOW_LOAD_FRACTION = 0.2
FC_HIGH_LOAD_FRACTION = 0.8
FC_STARTUP_UV = 13.79
FC_LOW_UV_PER_HOUR = 8.66
FC_HIGH_UV_PER_HOUR = 10.0
FC_TRANSIENT_UV_PER_NORMALIZED_UNIT = 0.0441
FC_EOL_UV = 60_000.0
FC_REPLACEMENT_COST_PER_KW = 7_500.0 / 7.0

BATTERY_KWH = CONFIG.battery_total_capacity_kwh
BATTERY_VOLTAGE_V = CONFIG.battery_rated_voltage_v
BATTERY_CHARGE_KW = CONFIG.battery_max_charge_kw
BATTERY_DISCHARGE_KW = CONFIG.battery_max_discharge_kw
BATTERY_COST_PER_QLOSS = 350.0 * BATTERY_KWH / 20.0 / 100.0
SOC_MIN, SOC_MAX = 0.30, 0.80
SOC_INITIAL = CONFIG.soc_initial
SOC_TARGET = 0.50
HYDROGEN_COST_PER_KG = 4.0  # SimulationCode cost units, currency unspecified.
LHV_KJ_PER_KG = CONFIG.hydrogen_lhv_mj_per_kg * 1_000.0
STEP_S = 1.0

FC_LOAD_REFERENCE = np.array([
    0, 0.227058, 0.454115, 0.681173, 1.13529, 2.04352, 2.87606, 4.01135,
    5.52507, 7.87133, 9.91485, 12.2611, 17.2564, 23.3869, 28.3822, 31.8638,
    34.8155, 36.7077, 39.6594, 42.3084, 44.9574, 49.1958, 53.6613, 58.5809,
    62.0624, 64.5601, 67.2091, 70.1608, 72.5071, 76.2157, 79.7729,
]) / 80.0
FC_EFFICIENCY_REFERENCE = np.array([
    1e-6, 10, 15, 25, 34.4484, 40.1423, 44.5552, 50.1068,
    53.5231, 57.5089, 60.2135, 61.637, 62.9181, 63.6299, 63.6299, 63.7722,
    63.7722, 63.0605, 62.4911, 61.9217, 61.4947, 60.2135, 59.0747, 57.6512,
    56.2278, 55.516, 53.9502, 52.3843, 50.8185, 47.5445, 44.5552,
]) / 100.0


def fuel_cell_efficiency(power_kw: float) -> float:
    efficiency = np.interp(power_kw / FC_RATED_KW, FC_LOAD_REFERENCE, FC_EFFICIENCY_REFERENCE)
    return max(float(efficiency), 1e-6)


def hydrogen_rate_kg_per_hour(power_kw: float, efficiency: float) -> float:
    """Convert kJ/s to kg/h. Signed power is used for battery equivalent hydrogen."""
    return power_kw / (efficiency * LHV_KJ_PER_KG) * 3_600.0


class FuelCellDegradation(NamedTuple):
    startup: float
    low_load: float
    high_load: float
    transient: float


def fuel_cell_degradation(previous_kw: float, actual_kw: float) -> FuelCellDegradation:
    running = actual_kw > FC_RUNNING_THRESHOLD_KW
    startup = FC_STARTUP_UV if previous_kw <= FC_RUNNING_THRESHOLD_KW and running else 0.0
    low = running and actual_kw <= FC_LOW_LOAD_FRACTION * FC_RATED_KW
    high = running and actual_kw >= FC_HIGH_LOAD_FRACTION * FC_RATED_KW
    return FuelCellDegradation(
        startup,
        FC_LOW_UV_PER_HOUR * STEP_S / 3_600.0 if low else 0.0,
        FC_HIGH_UV_PER_HOUR * STEP_S / 3_600.0 if high else 0.0,
        FC_TRANSIENT_UV_PER_NORMALIZED_UNIT * abs(actual_kw - previous_kw) / FC_NORMALIZATION_KW,
    )


def battery_capacity_loss(power_kw: float) -> float:
    absolute_kw = abs(power_kw)
    # At constant voltage, I / nominal_Ah equals |P_kW| / capacity_kWh.
    c_rate = absolute_kw / BATTERY_KWH
    qloss1 = (absolute_kw / 3_600.0) ** 0.55
    qloss2 = 32_350.0 * exp(-0.3375 * c_rate) + 4_041.0 * exp(0.1271 * c_rate)
    qloss3 = exp(-(63_400.0 - (c_rate + 1.0) * 370.3) / (8.31 * 350.0))
    return qloss1 * qloss2 * qloss3 * STEP_S


def maximum_step_cost(soc_weight: float) -> float:
    """Conservative bound for reporting how the fixed failure cost compares."""
    battery_kw = max(BATTERY_CHARGE_KW, BATTERY_DISCHARGE_KW)
    # In the first segment efficiency grows at least linearly with power.
    # Beyond it, use the smallest efficiency at a positive tabulated load.
    low_power_ratio = FC_RATED_KW * FC_LOAD_REFERENCE[1] / (
        FC_EFFICIENCY_REFERENCE[1] - FC_EFFICIENCY_REFERENCE[0]
    )
    rated_ratio = FC_RATED_KW / float(np.min(FC_EFFICIENCY_REFERENCE[1:]))
    fuel = max(low_power_ratio, rated_ratio) * STEP_S * HYDROGEN_COST_PER_KG / LHV_KJ_PER_KG
    c_rate = battery_kw / BATTERY_KWH
    battery = (
        (battery_kw / 3_600.0) ** 0.55 * (32_350.0 + 4_041.0 * exp(0.1271 * c_rate))
        * exp(-(63_400.0 - (c_rate + 1.0) * 370.3) / (8.31 * 350.0))
        * STEP_S * BATTERY_COST_PER_QLOSS
    )
    fc_degradation_uv = (
        FC_STARTUP_UV + (FC_LOW_UV_PER_HOUR + FC_HIGH_UV_PER_HOUR) * STEP_S / 3_600.0
        + FC_TRANSIENT_UV_PER_NORMALIZED_UNIT * (FC_RATED_KW - FC_MIN_KW) / FC_NORMALIZATION_KW
    )
    fc = FC_REPLACEMENT_COST_PER_KW * FC_RATED_KW * fc_degradation_uv / FC_EOL_UV
    soc_distance = max(SOC_TARGET - SOC_MIN, SOC_MAX - SOC_TARGET)
    return fuel + battery + fc + soc_weight * soc_distance**2


class SimulationCodeShipEnv(gym.Env):
    """Map a relative action into the current feasible FC power interval."""

    def __init__(
        self, load_profiles, sample_ids, *, load_scale_kw: float, load_delta_scale_kw: float,
        soc_weight: float, failure_cost_per_remaining_step: float,
        train_window_steps: int | None = None,
    ):
        super().__init__()
        self.load_profiles = tuple(np.asarray(profile, dtype=float) for profile in load_profiles)
        self.sample_ids = tuple(sample_ids)
        if len(self.load_profiles) != len(self.sample_ids) or not self.load_profiles:
            raise ValueError("Load profiles and voyage IDs must match.")
        minimum_points = min(map(len, self.load_profiles))
        if minimum_points < 2 or (
            train_window_steps is not None and not 0 < train_window_steps < minimum_points
        ):
            raise ValueError("Each window needs one load sample beyond its last executed step.")
        self.train_window_steps = train_window_steps
        self.soc_weight = soc_weight
        self.failure_cost_per_remaining_step = failure_cost_per_remaining_step
        self.load_scale_kw = load_scale_kw
        self.load_delta_scale_kw = load_delta_scale_kw
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def _observation(self) -> np.ndarray:
        load_kw = self.profile[self.step_index]
        next_load_kw = self.profile[min(self.step_index + 1, self.episode_length)]
        return np.clip(np.array([
            2.0 * (self.soc - SOC_MIN) / (SOC_MAX - SOC_MIN) - 1.0,
            2.0 * load_kw / self.load_scale_kw - 1.0,
            (next_load_kw - load_kw) / self.load_delta_scale_kw,
            2.0 * self.fc_kw / FC_RATED_KW - 1.0,
        ], dtype=np.float32), -1.0, 1.0)

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        index = (options or {}).get("voyage_index", 0)
        profile = self.load_profiles[index]
        self.window_info = {"voyage_id": self.sample_ids[index], "episode_scope": "full_voyage"}
        if self.train_window_steps is not None:
            start = int(self.np_random.integers(len(profile) - self.train_window_steps))
            end = start + self.train_window_steps
            profile = profile[start:end + 1]
            self.window_info.update(
                episode_scope="training_window", window_start_step=start, window_end_step=end,
            )
        self.profile = profile
        self.episode_length = len(profile) - 1
        self.step_index = 0
        self.soc = SOC_INITIAL
        self.fc_kw = float(np.clip(0.5 * profile[0], 10.0, FC_RATED_KW))
        self.episode_done = False
        return self._observation(), {
            **self.window_info, "episode_length": self.episode_length,
            "num_samples": len(profile), "initial_soc": self.soc, "initial_fc_kw": self.fc_kw,
        }

    def step(self, action):
        if self.episode_done:
            raise RuntimeError("Reset the environment before taking another step.")
        (action_value,) = np.asarray(action).reshape(-1)
        if not isfinite(action_value):
            raise ValueError("Fuel-cell action fraction must be finite.")
        action_fraction = float(np.clip(action_value, 0.0, 1.0))
        previous_fc_kw = self.fc_kw
        load_kw = float(self.profile[self.step_index])
        battery_energy_kw = BATTERY_KWH * 3_600.0 / STEP_S
        battery_min_kw = max(-BATTERY_CHARGE_KW, (self.soc - SOC_MAX) * battery_energy_kw)
        battery_max_kw = min(BATTERY_DISCHARGE_KW, (self.soc - SOC_MIN) * battery_energy_kw)
        fc_min_kw = max(FC_MIN_KW, load_kw - battery_max_kw)
        fc_max_kw = min(FC_RATED_KW, load_kw - battery_min_kw)
        info = {
            **self.window_info, "load_kw": load_kw, "soc_before": self.soc,
            "actor_action_fraction": action_fraction, "truncated": False,
            "feasible_fc_lower_kw": fc_min_kw, "feasible_fc_upper_kw": fc_max_kw,
            "battery_feasible_lower_kw": battery_min_kw, "battery_feasible_upper_kw": battery_max_kw,
        }
        if fc_min_kw > fc_max_kw + 1e-8:
            self.episode_done = True
            penalty = self.failure_cost_per_remaining_step * (self.episode_length - self.step_index)
            return self._observation(), -penalty, True, False, {
                **info, "soc": self.soc, "failure": True, "terminated": True,
                "fc_actual_kw": None, "battery_actual_kw": None,
                "unserved_load_kw": max(0.0, load_kw - FC_RATED_KW - battery_max_kw),
                "excess_power_kw": max(0.0, FC_MIN_KW + battery_min_kw - load_kw),
                "economic_cost_ready": False, "failure_penalty": penalty,
                "terminal_reason": "no_feasible_physical_power",
            }

        # A discrepancy below the infeasibility tolerance is an endpoint.
        fc_min_kw = min(fc_min_kw, fc_max_kw)
        info["feasible_fc_lower_kw"] = fc_min_kw
        fc_kw = fc_min_kw + action_fraction * (fc_max_kw - fc_min_kw)
        battery_kw = load_kw - fc_kw
        self.soc -= battery_kw / battery_energy_kw
        self.fc_kw = fc_kw
        self.step_index += 1
        self.episode_done = self.step_index == self.episode_length
        reason = None
        if self.episode_done:
            reason = "training_window_end" if self.train_window_steps is not None else "voyage_end"

        efficiency = fuel_cell_efficiency(fc_kw)
        hydrogen_rate = hydrogen_rate_kg_per_hour(fc_kw, efficiency)
        battery_equivalent_rate = hydrogen_rate_kg_per_hour(battery_kw, efficiency)
        hydrogen_kg = hydrogen_rate * STEP_S / 3_600.0
        battery_equivalent_kg = battery_equivalent_rate * STEP_S / 3_600.0
        hydrogen_cost = hydrogen_kg * HYDROGEN_COST_PER_KG
        battery_equivalent_cost = battery_equivalent_kg * HYDROGEN_COST_PER_KG
        qloss = battery_capacity_loss(battery_kw)
        battery_cost = qloss * BATTERY_COST_PER_QLOSS
        degradation = fuel_cell_degradation(previous_fc_kw, fc_kw)
        fc_cost = FC_REPLACEMENT_COST_PER_KW * FC_RATED_KW * sum(degradation) / FC_EOL_UV
        soc_penalty = self.soc_weight * (self.soc - SOC_TARGET) ** 2
        actual_cost = hydrogen_cost + battery_cost + fc_cost
        reward = -(actual_cost + soc_penalty)

        return self._observation(), reward, self.episode_done, False, {
            **info, "soc": self.soc, "fc_actual_kw": fc_kw, "battery_actual_kw": battery_kw,
            "delta_fc_kw": fc_kw - previous_fc_kw,
            "power_balance_error_kw": fc_kw + battery_kw - load_kw,
            "battery_current_a": battery_kw * 1_000.0 / BATTERY_VOLTAGE_V,
            "battery_c_rate": abs(battery_kw) / BATTERY_KWH, "battery_qloss": qloss,
            "fc_efficiency": efficiency, "hydrogen_rate_kg_per_hour": hydrogen_rate,
            "battery_equivalent_rate_kg_per_hour": battery_equivalent_rate,
            "hydrogen_kg": hydrogen_kg, "battery_equivalent_hydrogen_kg": battery_equivalent_kg,
            "hydrogen_cost": hydrogen_cost, "battery_equivalent_hydrogen_cost": battery_equivalent_cost,
            "battery_degradation_cost": battery_cost,
            "fc_degradation_cost": fc_cost, "fc_start_count": int(degradation.startup > 0.0),
            "fc_startup_uv": degradation.startup, "fc_low_uv": degradation.low_load,
            "fc_high_uv": degradation.high_load, "fc_transient_uv": degradation.transient,
            "soc_penalty": soc_penalty,
            "total_economic_cost": actual_cost + battery_equivalent_cost,
            "failure": False, "failure_penalty": 0.0, "economic_cost_ready": True,
            "terminated": self.episode_done, "terminal_reason": reason,
        }
