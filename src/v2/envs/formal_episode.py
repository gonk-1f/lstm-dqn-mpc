"""Formal event-driven execution over authenticated ONBOARD/shore modes."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from ..config import PlantConfig, TAU_LPF_SECONDS, TimeScaleConfig
from ..control.causal_base_load import CausalBaseLoadFilter
from ..control.nonlinear_mpc import (
    DELTA_P_FC_OBJECTIVE_SCALE_KW,
    P_FC_OBJECTIVE_SCALE_KW,
    SOC_HARD_MAX,
    SOC_HARD_MIN,
    SOC_OBJECTIVE_SCALE,
    SOC_WORKING_HIGH,
    SOC_WORKING_LOW,
    MPCConfig,
    NonlinearMPC,
)
from ..data.supervisory_rules import OperatingMode
from ..dqn.state import OperatingHistorySample, build_formal_operating_state
from ..economics import (
    ShoreEnergy,
    ShoreEnergyClassification,
    build_formal_interval_ledger,
    terminal_recharge_grid_energy,
)
from ..models.battery_degradation import (
    BATTERY_NOMINAL_VOLTAGE_V,
    BatteryThroughputAccount,
    battery_degradation_step,
    formal_battery_lifetime_normalization,
)
from ..models.battery_energy import formal_battery_efficiency
from ..models.fuel_cell_degradation import (
    FuelCellVoltageLossAccount,
    formal_aggregate_fc_power_mapping,
    formal_aggregate_fc_voltage_loss_step_uv,
)
from ..models.fuel_cell_efficiency import (
    formal_fuel_cell_efficiency_map,
    hydrogen_mass_from_map_kg,
)
from .multirate_weight_env import MPCExecutionResult


SHORE_MODES = frozenset(
    (OperatingMode.SHORE_PENDING, OperatingMode.SHORE_CHARGING)
)


def build_formal_nonlinear_mpc() -> NonlinearMPC:
    plant = PlantConfig.research_simulation()
    return NonlinearMPC(
        MPCConfig(
            timescale=TimeScaleConfig.formal_baseline(),
            fuel_cell_rated_kw=plant.fuel_cell_rated_total_kw,
            battery_capacity_kwh=plant.battery_nominal_energy_kwh,
            battery_efficiency=formal_battery_efficiency(),
            battery_charge_min_kw=plant.battery_charge_min_kw,
            battery_discharge_max_kw=plant.battery_discharge_max_kw,
            soc_min=SOC_HARD_MIN,
            soc_max=SOC_HARD_MAX,
            soc_deadband_low=SOC_WORKING_LOW,
            soc_deadband_high=SOC_WORKING_HIGH,
            p_fc_scale_kw=P_FC_OBJECTIVE_SCALE_KW,
            delta_p_fc_scale_kw=DELTA_P_FC_OBJECTIVE_SCALE_KW,
            soc_scale=SOC_OBJECTIVE_SCALE,
            fuel_cell_ramp_kw_per_step=None,
        )
    )


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _modes(values: tuple[str, ...], expected: int) -> tuple[OperatingMode, ...]:
    if type(values) is not tuple or len(values) != expected:
        raise ValueError("operating_mode must be an exact tuple matching the episode")
    try:
        return tuple(OperatingMode(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("operating_mode contains an unknown mode") from exc


class FormalEpisodeBackend:
    """Advance one authenticated 30 s physical interval per backend call."""

    INITIAL_SOC = 0.60
    SHORE_TARGET_SOC = 0.60

    def __init__(
        self,
        *,
        load_kw: np.ndarray,
        speed_kn: np.ndarray,
        fc_power_kw: np.ndarray,
        battery_bus_kw: np.ndarray,
        operating_mode: tuple[str, ...],
        mpc: object,
    ) -> None:
        load = np.asarray(load_kw, dtype=float)
        speed = np.asarray(speed_kn, dtype=float)
        fc = np.asarray(fc_power_kw, dtype=float)
        battery = np.asarray(battery_bus_kw, dtype=float)
        if (
            load.ndim != 1
            or len(load) == 0
            or speed.shape != load.shape
            or fc.shape != load.shape
            or battery.shape != load.shape
        ):
            raise ValueError("episode power/speed vectors must be equal and nonempty")
        if (
            not np.isfinite(load).all()
            or not np.isfinite(speed).all()
            or not np.isfinite(fc).all()
            or not np.isfinite(battery).all()
            or (speed < 0.0).any()
        ):
            raise ValueError("episode power/speed values must be finite and speed nonnegative")
        modes = _modes(operating_mode, len(load))
        solve = getattr(mpc, "solve", None)
        if not callable(solve):
            raise TypeError("mpc must expose callable solve")
        self.load_kw = load.copy()
        self.speed_kn = speed.copy()
        self.fc_evidence_kw = fc.copy()
        self.battery_capacity_profile_kw = battery.copy()
        self.operating_mode = modes
        self.mpc = mpc
        self.timescale = TimeScaleConfig.formal_baseline()
        self.plant = PlantConfig.research_simulation()
        self.efficiency = formal_battery_efficiency()
        self.battery_normalization = formal_battery_lifetime_normalization()
        self.fc_efficiency = formal_fuel_cell_efficiency_map()
        self.fc_mapping = formal_aggregate_fc_power_mapping()
        self.reset()

    def reset(self) -> None:
        self.index = 0
        self.soc = self.INITIAL_SOC
        self.previous_fc_kw = 0.0
        self._reset_onboard_history()
        self.fc_account = FuelCellVoltageLossAccount()
        self.battery_account = BatteryThroughputAccount()
        self.mode_counts = {mode: 0 for mode in OperatingMode}
        self.mpc_solve_count = 0
        self.executed_fc_power_kw: list[float] = []
        self.observed_states: list[tuple[float, ...]] = []
        self._terminal_state: tuple[float, ...] | None = None
        self._done = False

    def _reset_onboard_history(self) -> None:
        self._base_filter = CausalBaseLoadFilter(
            sample_seconds=self.timescale.ts_mpc_seconds,
            tau_seconds=TAU_LPF_SECONDS,
        )
        self._past_samples: list[OperatingHistorySample] = []

    def _current_values(self) -> tuple[float, float, float, OperatingMode]:
        if self.index >= len(self.load_kw):
            raise RuntimeError("episode is already complete")
        return (
            float(self.load_kw[self.index]),
            float(self.speed_kn[self.index]),
            float(self.index * self.timescale.ts_mpc_seconds),
            self.operating_mode[self.index],
        )

    def _current_base_kw(self, load: float) -> float:
        observed = self._base_filter.observed_base_kw
        if observed is None:
            return load
        return self._base_filter.alpha * observed + (1.0 - self._base_filter.alpha) * load

    def _current_sample(self) -> tuple[OperatingHistorySample, float, OperatingMode]:
        load, speed, time_s, mode = self._current_values()
        sample = OperatingHistorySample(
            time_s,
            float(self.soc),
            float(self.previous_fc_kw),
            0.0,
            load,
            float(self._current_base_kw(load) if mode is OperatingMode.ONBOARD else load),
        )
        return sample, speed, mode

    def state(self) -> tuple[float, ...]:
        if self._done:
            if self._terminal_state is None:
                raise RuntimeError("terminal state is unavailable")
            return self._terminal_state
        sample, speed, mode = self._current_sample()
        if mode is not OperatingMode.ONBOARD:
            raise ValueError("DQN state is available only at an ONBOARD decision boundary")
        return build_formal_operating_state(
            tuple(self._past_samples + [sample]),
            current_time_seconds=sample.timestamp_seconds,
            speed_kn=speed,
        )

    def _shore_step(self, profile_bus_kw: float) -> tuple[float, float, float, float]:
        eta_chg, _ = self.efficiency.require_calibrated()
        requested_battery_kw = min(
            max(0.0, -profile_bus_kw),
            abs(self.plant.battery_charge_min_kw),
        )
        target = min(self.SHORE_TARGET_SOC, SOC_HARD_MAX)
        room_kwh = max(0.0, target - self.soc) * self.plant.battery_nominal_energy_kwh
        room_battery_kw = room_kwh * 3600.0 / self.timescale.ts_mpc_seconds
        accepted_battery_kw = min(requested_battery_kw, room_battery_kw)
        battery_bus_kw = -accepted_battery_kw
        updated_soc = self.soc + (
            accepted_battery_kw
            * self.timescale.ts_mpc_seconds
            / 3600.0
            / self.plant.battery_nominal_energy_kwh
        )
        grid_kwh = (
            accepted_battery_kw
            * self.timescale.ts_mpc_seconds
            / 3600.0
            / eta_chg
        )
        return 0.0, battery_bus_kw, updated_soc, grid_kwh

    def execute_mpc_step(self, weights: object) -> MPCExecutionResult:
        if self._done:
            raise RuntimeError("episode is already complete")
        sample, speed, mode = self._current_sample()
        if mode is OperatingMode.UNRESOLVED:
            raise ValueError("unresolved operating mode blocks formal execution")
        load = sample.load_power_kw
        soc_before = self.soc
        previous_fc = self.previous_fc_kw
        shore_grid_kwh = 0.0
        mpc_executed = mode is OperatingMode.ONBOARD

        if mpc_executed:
            current_state = self.state()
            self.observed_states.append(current_state)
            plan = self.mpc.solve(
                observed_load_kw=load,
                current_soc=self.soc,
                previous_executed_p_fc_kw=self.previous_fc_kw,
                weights=weights,
                base_load_filter=self._base_filter,
            )
            command = plan.first_command()
            p_fc = _finite(command.p_fc_kw, "p_fc_kw")
            p_batt = _finite(command.p_batt_bus_kw, "p_batt_bus_kw")
            next_state = _finite(command.predicted_next_soc, "predicted_next_soc")
            if not 0.0 <= p_fc <= self.plant.fuel_cell_rated_total_kw:
                raise ValueError("executed FC command lies outside [0, 600]")
            self._base_filter.commit(load)
            self._past_samples.append(sample)
            self.mpc_solve_count += 1
        else:
            self._reset_onboard_history()
            p_fc, p_batt, next_state, shore_grid_kwh = self._shore_step(
                float(self.battery_capacity_profile_kw[self.index])
            )

        if not 0.0 <= next_state <= 1.0:
            raise ValueError("executed interval produced invalid SOC")
        fc_before = self.fc_account.total_uv
        cycle = int((previous_fc > 0.0) != (p_fc > 0.0))
        fc_step = formal_aggregate_fc_voltage_loss_step_uv(
            previous_fc,
            p_fc,
            self.timescale.ts_mpc_seconds,
            self.plant.fuel_cell_rated_total_kw,
            is_on=p_fc > 0.0,
            mapping=self.fc_mapping,
            aggregate_start_stop_cycles=cycle,
        )
        self.fc_account.add(fc_step)
        battery_before = self.battery_account.weighted_ah
        current_a = p_batt * 1000.0 / BATTERY_NOMINAL_VOLTAGE_V
        battery_step = battery_degradation_step(
            soc_before,
            current_a,
            self.timescale.ts_mpc_seconds,
            self.battery_normalization.nominal_charge_capacity_ah,
        )
        self.battery_account.add(battery_step)
        hydrogen = hydrogen_mass_from_map_kg(
            p_fc, self.timescale.ts_mpc_seconds, self.fc_efficiency
        )

        self.previous_fc_kw = p_fc
        self.soc = next_state
        self.index += 1
        self.mode_counts[mode] += 1
        self.executed_fc_power_kw.append(p_fc)
        done = self.index >= len(self.load_kw)
        if done:
            terminal = terminal_recharge_grid_energy(
                episode_initial_soc=self.INITIAL_SOC,
                episode_end_soc=self.soc,
                battery_capacity_kwh=self.plant.battery_nominal_energy_kwh,
                battery_efficiency=self.efficiency,
            )
            shore_grid_kwh += terminal.energy_kwh
            self.soc = self.INITIAL_SOC
            terminal_sample = OperatingHistorySample(
                sample.timestamp_seconds + self.timescale.ts_mpc_seconds,
                float(self.soc),
                0.0,
                0.0,
                float(load),
                float(load),
            )
            self._terminal_state = build_formal_operating_state(
                (terminal_sample,),
                current_time_seconds=terminal_sample.timestamp_seconds,
                speed_kn=speed,
            )
            self._done = True
        next_decision_ready = done or self.operating_mode[self.index] is OperatingMode.ONBOARD
        shore = (
            ShoreEnergy(shore_grid_kwh, ShoreEnergyClassification.MODELED)
            if shore_grid_kwh > 0.0
            else None
        )
        ledger = build_formal_interval_ledger(
            hydrogen_mass_kg=hydrogen,
            fuel_cell_cumulative_voltage_loss_before_uv=fc_before,
            fuel_cell_cumulative_voltage_loss_after_uv=self.fc_account.total_uv,
            fuel_cell_rated_kw=self.plant.fuel_cell_rated_total_kw,
            battery_cumulative_weighted_ah_before=battery_before,
            battery_cumulative_weighted_ah_after=self.battery_account.weighted_ah,
            battery_capacity_kwh=self.plant.battery_nominal_energy_kwh,
            battery_normalization=self.battery_normalization,
            shore_energy=shore,
        )
        return MPCExecutionResult(
            ledger,
            done,
            mpc_solve_executed=mpc_executed,
            next_decision_ready=next_decision_ready,
        )


__all__ = [
    "FormalEpisodeBackend",
    "SHORE_MODES",
    "build_formal_nonlinear_mpc",
]
