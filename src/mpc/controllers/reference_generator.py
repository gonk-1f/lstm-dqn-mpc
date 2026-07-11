from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mpc.solvers.casadi_solver import (
    CasadiMPCConfig,
    ShipCasadiMPC,
    ShipDualSideCasadiMPC,
    casadi_available,
)
from mpc.solvers.scipy_solver import ShipScipyMPC, ShipDualSideScipyMPC, scipy_available


@dataclass
class ReferenceConfig:
    horizon_steps: int = 8
    soc_target: float = 0.65
    fuel_cell_min_kw: float = 0.0
    fuel_cell_max_kw: float = 240.0
    battery_charge_max_kw: float = 350.0
    battery_discharge_max_kw: float = 350.0
    soc_feedback_gain: float = 160.0
    fuel_cell_base_ratio: float = 0.8
    side_balance_gain: float = 0.15


def compute_side_split(
    load_left_kw: float,
    load_right_kw: float,
    soc_left: float,
    soc_right: float,
    balance_gain: float = 0.15,
) -> tuple[float, float]:
    total_load = max(load_left_kw + load_right_kw, 1e-6)
    base_left_ratio = load_left_kw / total_load
    soc_bias = balance_gain * (soc_right - soc_left)
    left_ratio = float(np.clip(base_left_ratio + soc_bias, 0.2, 0.8))
    right_ratio = 1.0 - left_ratio
    return left_ratio, right_ratio


class HeuristicReferenceGenerator:
    """Fallback upper-layer reference generator used before full MPC is finalized."""

    def __init__(self, config: ReferenceConfig | None = None):
        self.config = config or ReferenceConfig()

    def generate(self, load_forecast_kw: np.ndarray, current_soc: float, prev_fc_kw: float) -> tuple[float, float]:
        mean_load = float(np.mean(load_forecast_kw))
        soc_correction = self.config.soc_feedback_gain * (self.config.soc_target - current_soc)
        # If SOC is below target, increase fuel-cell reference so the battery
        # is not asked to keep discharging. The previous sign did the opposite.
        fc_ref = self.config.fuel_cell_base_ratio * mean_load + soc_correction
        fc_ref = float(np.clip(fc_ref, self.config.fuel_cell_min_kw, self.config.fuel_cell_max_kw))

        batt_ref = float(load_forecast_kw[0] - fc_ref)
        batt_ref = float(
            np.clip(
                batt_ref,
                -self.config.battery_charge_max_kw,
                self.config.battery_discharge_max_kw,
            )
        )
        return fc_ref, batt_ref


class CasadiReferenceGenerator:
    """Reference generator backed by the actual upper-layer CasADi MPC."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not casadi_available():
            raise ModuleNotFoundError("casadi is not available for CasadiReferenceGenerator.")
        self.config = config or CasadiMPCConfig()
        self.mpc = ShipCasadiMPC(self.config)

    def generate_result(
        self,
        load_forecast_kw: np.ndarray,
        current_soc: float,
        prev_fc_kw: float,
        terminal_load_kw: float | None = None,
        soc_reference_value: float | None = None,
    ):
        return self.mpc.solve(
            current_soc=current_soc,
            prev_fc_kw=prev_fc_kw,
            load_forecast_kw=load_forecast_kw,
            terminal_load_kw=terminal_load_kw,
            soc_reference_value=soc_reference_value,
        )

    def generate(self, load_forecast_kw: np.ndarray, current_soc: float, prev_fc_kw: float) -> tuple[float, float]:
        result = self.generate_result(load_forecast_kw=load_forecast_kw, current_soc=current_soc, prev_fc_kw=prev_fc_kw)
        return result.fuel_cell_ref_kw, result.battery_ref_kw


class DualSideCasadiReferenceGenerator:
    """Side-aware CasADi MPC that directly optimizes left/right references."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not casadi_available():
            raise ModuleNotFoundError("casadi is not available for DualSideCasadiReferenceGenerator.")
        self.config = config or CasadiMPCConfig()
        self.mpc = ShipDualSideCasadiMPC(self.config)

    def generate_dual(
        self,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        soc_left: float,
        soc_right: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ):
        result = self.mpc.solve(
            soc_left_0=soc_left,
            soc_right_0=soc_right,
            prev_fc_left_kw=prev_fc_left_kw,
            prev_fc_right_kw=prev_fc_right_kw,
            load_left_forecast_kw=load_left_forecast_kw,
            load_right_forecast_kw=load_right_forecast_kw,
            terminal_load_left_kw=terminal_load_left_kw,
            terminal_load_right_kw=terminal_load_right_kw,
        )
        return result

    def generate_dual_tuple(
        self,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        soc_left: float,
        soc_right: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ) -> tuple[float, float, float, float]:
        result = self.generate_dual(
            load_left_forecast_kw=load_left_forecast_kw,
            load_right_forecast_kw=load_right_forecast_kw,
            soc_left=soc_left,
            soc_right=soc_right,
            prev_fc_left_kw=prev_fc_left_kw,
            prev_fc_right_kw=prev_fc_right_kw,
            terminal_load_left_kw=terminal_load_left_kw,
            terminal_load_right_kw=terminal_load_right_kw,
        )
        return (
            result.fuel_cell_ref_left_kw,
            result.fuel_cell_ref_right_kw,
            result.battery_ref_left_kw,
            result.battery_ref_right_kw,
        )


class ScipyReferenceGenerator:
    """Reference generator backed by strict SciPy/SLSQP MPC."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not scipy_available():
            raise ModuleNotFoundError("scipy is not available for ScipyReferenceGenerator.")
        self.config = config or CasadiMPCConfig()
        self.mpc = ShipScipyMPC(self.config)

    def generate_result(
        self,
        load_forecast_kw: np.ndarray,
        current_soc: float,
        prev_fc_kw: float,
        terminal_load_kw: float | None = None,
    ):
        return self.mpc.solve(
            current_soc=current_soc,
            prev_fc_kw=prev_fc_kw,
            load_forecast_kw=load_forecast_kw,
            terminal_load_kw=terminal_load_kw,
        )

    def generate(self, load_forecast_kw: np.ndarray, current_soc: float, prev_fc_kw: float) -> tuple[float, float]:
        result = self.generate_result(load_forecast_kw=load_forecast_kw, current_soc=current_soc, prev_fc_kw=prev_fc_kw)
        return result.fuel_cell_ref_kw, result.battery_ref_kw


class DualSideScipyReferenceGenerator:
    """Side-aware strict SciPy/SLSQP MPC for left/right references."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not scipy_available():
            raise ModuleNotFoundError("scipy is not available for DualSideScipyReferenceGenerator.")
        self.config = config or CasadiMPCConfig()
        self.mpc = ShipDualSideScipyMPC(self.config)

    def generate_dual(
        self,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        soc_left: float,
        soc_right: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ):
        return self.mpc.solve(
            soc_left_0=soc_left,
            soc_right_0=soc_right,
            prev_fc_left_kw=prev_fc_left_kw,
            prev_fc_right_kw=prev_fc_right_kw,
            load_left_forecast_kw=load_left_forecast_kw,
            load_right_forecast_kw=load_right_forecast_kw,
            terminal_load_left_kw=terminal_load_left_kw,
            terminal_load_right_kw=terminal_load_right_kw,
        )

    def generate_dual_tuple(
        self,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        soc_left: float,
        soc_right: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ) -> tuple[float, float, float, float]:
        result = self.generate_dual(
            load_left_forecast_kw=load_left_forecast_kw,
            load_right_forecast_kw=load_right_forecast_kw,
            soc_left=soc_left,
            soc_right=soc_right,
            prev_fc_left_kw=prev_fc_left_kw,
            prev_fc_right_kw=prev_fc_right_kw,
            terminal_load_left_kw=terminal_load_left_kw,
            terminal_load_right_kw=terminal_load_right_kw,
        )
        return (
            result.fuel_cell_ref_left_kw,
            result.fuel_cell_ref_right_kw,
            result.battery_ref_left_kw,
            result.battery_ref_right_kw,
        )


def attach_reference_columns(
    dataset: pd.DataFrame,
    generator: object | None = None,
    source_name: str | None = None,
    solve_stride: int = 1,
) -> pd.DataFrame:
    generator = generator or HeuristicReferenceGenerator()
    df = dataset.copy().reset_index(drop=True)
    solve_stride = max(1, int(solve_stride))
    fc_refs = []
    batt_refs = []
    fc_refs_left = []
    fc_refs_right = []
    batt_refs_left = []
    batt_refs_right = []
    soc_refs = []
    objective_values = []
    objective_fuel = []
    objective_smooth = []
    objective_soc = []
    objective_battery = []
    objective_terminal_soc = []
    objective_fuel_term = []
    objective_smooth_term = []
    objective_soc_term = []
    objective_battery_term = []
    objective_terminal_soc_term = []
    objective_total_cost = []
    objective_enable_battery_use = []
    solve_success = []
    solver_messages = []
    prev_fc = float(df["fuel_cell_power_total_kw"].iloc[0]) if "fuel_cell_power_total_kw" in df else 0.0
    prev_fc_left = float(df.get("fuel_cell_power_left_kw", df.get("fuel_cell_power_total_kw", pd.Series([0.0])) * 0.5).iloc[0])
    prev_fc_right = float(df.get("fuel_cell_power_right_kw", df.get("fuel_cell_power_total_kw", pd.Series([0.0])) * 0.5).iloc[0])
    last_fc_ref = prev_fc
    last_batt_ref = float(df["load_total_kw"].iloc[0] - prev_fc) if "load_total_kw" in df else 0.0
    last_fc_left = prev_fc_left
    last_fc_right = prev_fc_right
    last_batt_left = 0.0
    last_batt_right = 0.0
    config = getattr(generator, "config", None)
    last_soc_ref = float(getattr(config, "soc_target", 0.65)) if config is not None else 0.65
    last_objective_value = 0.0
    last_objective_fuel = 0.0
    last_objective_smooth = 0.0
    last_objective_soc = 0.0
    last_objective_battery = 0.0
    last_objective_terminal_soc = 0.0
    last_objective_fuel_term = 0.0
    last_objective_smooth_term = 0.0
    last_objective_soc_term = 0.0
    last_objective_battery_term = 0.0
    last_objective_terminal_soc_term = 0.0
    last_objective_total_cost = 0.0
    last_objective_enable_battery_use = bool(getattr(config, "enable_battery_use_in_mpc", True)) if config else True
    last_solve_success = True
    last_solver_message = "not_solved_yet"
    horizon = int(
        getattr(config, "horizon_steps", getattr(config, "prediction_horizon", 8))
    )

    for idx in range(len(df)):
        if idx % solve_stride == 0:
            current_soc = float(df["soc_mean"].iloc[idx])
            if f"load_pred_h{horizon}" in df.columns:
                forecast = np.array([float(df[f"load_pred_h{i+1}"].iloc[idx]) for i in range(horizon)])
            else:
                forecast = df["load_total_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
            if len(forecast) < horizon:
                if len(forecast) == 0:
                    forecast = np.zeros(horizon, dtype=float)
                else:
                    forecast = np.pad(forecast, (0, horizon - len(forecast)), mode="edge")
            dual_supported = hasattr(generator, "generate_dual") and "load_left_kw" in df.columns and "load_right_kw" in df.columns
            if dual_supported:
                left_forecast = df["load_left_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
                right_forecast = df["load_right_kw"].iloc[idx : idx + horizon].to_numpy(dtype=float)
                if len(left_forecast) < horizon:
                    left_forecast = np.pad(left_forecast, (0, horizon - len(left_forecast)), mode="edge")
                if len(right_forecast) < horizon:
                    right_forecast = np.pad(right_forecast, (0, horizon - len(right_forecast)), mode="edge")
                result = generator.generate_dual(
                    load_left_forecast_kw=left_forecast,
                    load_right_forecast_kw=right_forecast,
                    soc_left=float(df.get("soc_left", df["soc_mean"]).iloc[idx]),
                    soc_right=float(df.get("soc_right", df["soc_mean"]).iloc[idx]),
                    prev_fc_left_kw=prev_fc_left,
                    prev_fc_right_kw=prev_fc_right,
                )
                fc_left = result.fuel_cell_ref_left_kw
                fc_right = result.fuel_cell_ref_right_kw
                batt_left = result.battery_ref_left_kw
                batt_right = result.battery_ref_right_kw
                last_fc_left = fc_left
                last_fc_right = fc_right
                last_batt_left = batt_left
                last_batt_right = batt_right
                last_fc_ref = fc_left + fc_right
                last_batt_ref = batt_left + batt_right
                last_soc_ref = float(result.soc_ref)
                last_objective_value = float(result.objective_value)
                last_objective_fuel = float(result.objective_info.get("fuel_cost_norm_mean", 0.0))
                last_objective_smooth = float(result.objective_info.get("fc_smooth_norm_mean", 0.0))
                last_objective_soc = float(result.objective_info.get("soc_dev_norm_mean", 0.0))
                last_objective_battery = float(result.objective_info.get("battery_use_norm_mean", 0.0))
                last_objective_terminal_soc = float(result.objective_info.get("terminal_soc_norm", 0.0))
                last_objective_fuel_term = float(result.objective_info.get("fuel_cost_term", 0.0))
                last_objective_smooth_term = float(result.objective_info.get("fc_smooth_term", 0.0))
                last_objective_soc_term = float(result.objective_info.get("soc_dev_term", 0.0))
                last_objective_battery_term = float(result.objective_info.get("battery_use_term", 0.0))
                last_objective_terminal_soc_term = float(result.objective_info.get("terminal_soc_term", 0.0))
                last_objective_total_cost = float(result.objective_info.get("total_mpc_cost", last_objective_value))
                last_objective_enable_battery_use = bool(result.objective_info.get("enable_battery_use_in_mpc", True))
                last_solve_success = bool(getattr(result, "success", True))
                last_solver_message = str(result.objective_info.get("solver_message", ""))
                prev_fc_left = fc_left
                prev_fc_right = fc_right
            else:
                if hasattr(generator, "generate_result"):
                    result = generator.generate_result(forecast, current_soc=current_soc, prev_fc_kw=prev_fc)
                    fc_ref = result.fuel_cell_ref_kw
                    batt_ref = result.battery_ref_kw
                    last_soc_ref = float(result.soc_ref)
                    last_objective_value = float(result.objective_value)
                    last_objective_fuel = float(result.objective_info.get("fuel_cost_norm_mean", 0.0))
                    last_objective_smooth = float(result.objective_info.get("fc_smooth_norm_mean", 0.0))
                    last_objective_soc = float(result.objective_info.get("soc_dev_norm_mean", 0.0))
                    last_objective_battery = float(result.objective_info.get("battery_use_norm_mean", 0.0))
                    last_objective_terminal_soc = float(result.objective_info.get("terminal_soc_norm", 0.0))
                    last_objective_fuel_term = float(result.objective_info.get("fuel_cost_term", 0.0))
                    last_objective_smooth_term = float(result.objective_info.get("fc_smooth_term", 0.0))
                    last_objective_soc_term = float(result.objective_info.get("soc_dev_term", 0.0))
                    last_objective_battery_term = float(result.objective_info.get("battery_use_term", 0.0))
                    last_objective_terminal_soc_term = float(result.objective_info.get("terminal_soc_term", 0.0))
                    last_objective_total_cost = float(result.objective_info.get("total_mpc_cost", last_objective_value))
                    last_objective_enable_battery_use = bool(result.objective_info.get("enable_battery_use_in_mpc", True))
                    last_solve_success = bool(getattr(result, "success", True))
                    last_solver_message = str(result.objective_info.get("solver_message", ""))
                else:
                    fc_ref, batt_ref = generator.generate(forecast, current_soc=current_soc, prev_fc_kw=prev_fc)
                    last_soc_ref = float(getattr(config, "soc_target", 0.65)) if config is not None else 0.65
                    last_objective_value = 0.0
                    last_objective_fuel = 0.0
                    last_objective_smooth = 0.0
                    last_objective_soc = 0.0
                    last_objective_battery = 0.0
                    last_objective_terminal_soc = 0.0
                    last_objective_fuel_term = 0.0
                    last_objective_smooth_term = 0.0
                    last_objective_soc_term = 0.0
                    last_objective_battery_term = 0.0
                    last_objective_terminal_soc_term = 0.0
                    last_objective_total_cost = 0.0
                    last_objective_enable_battery_use = True
                    last_solve_success = True
                    last_solver_message = "heuristic_reference"
                if "load_left_kw" in df.columns and "load_right_kw" in df.columns:
                    left_ratio, right_ratio = compute_side_split(
                        load_left_kw=float(df["load_left_kw"].iloc[idx]),
                        load_right_kw=float(df["load_right_kw"].iloc[idx]),
                        soc_left=float(df.get("soc_left", df["soc_mean"]).iloc[idx]),
                        soc_right=float(df.get("soc_right", df["soc_mean"]).iloc[idx]),
                        balance_gain=float(getattr(config, "side_balance_gain", 0.15)),
                    )
                else:
                    left_ratio, right_ratio = 0.5, 0.5
                last_fc_ref = fc_ref
                last_batt_ref = batt_ref
                last_fc_left = fc_ref * left_ratio
                last_fc_right = fc_ref * right_ratio
                last_batt_left = batt_ref * left_ratio
                last_batt_right = batt_ref * right_ratio
            prev_fc = last_fc_ref
        else:
            prev_fc = last_fc_ref

        fc_refs.append(last_fc_ref)
        batt_refs.append(last_batt_ref)
        fc_refs_left.append(last_fc_left)
        fc_refs_right.append(last_fc_right)
        batt_refs_left.append(last_batt_left)
        batt_refs_right.append(last_batt_right)
        soc_refs.append(last_soc_ref)
        objective_values.append(last_objective_value)
        objective_fuel.append(last_objective_fuel)
        objective_smooth.append(last_objective_smooth)
        objective_soc.append(last_objective_soc)
        objective_battery.append(last_objective_battery)
        objective_terminal_soc.append(last_objective_terminal_soc)
        objective_fuel_term.append(last_objective_fuel_term)
        objective_smooth_term.append(last_objective_smooth_term)
        objective_soc_term.append(last_objective_soc_term)
        objective_battery_term.append(last_objective_battery_term)
        objective_terminal_soc_term.append(last_objective_terminal_soc_term)
        objective_total_cost.append(last_objective_total_cost)
        objective_enable_battery_use.append(last_objective_enable_battery_use)
        solve_success.append(last_solve_success)
        solver_messages.append(last_solver_message)

    df["mpc_fuel_cell_ref_kw"] = fc_refs
    df["mpc_battery_ref_kw"] = batt_refs
    df["mpc_fuel_cell_ref_left_kw"] = fc_refs_left
    df["mpc_fuel_cell_ref_right_kw"] = fc_refs_right
    df["mpc_battery_ref_left_kw"] = batt_refs_left
    df["mpc_battery_ref_right_kw"] = batt_refs_right
    df["mpc_soc_ref"] = soc_refs
    df["mpc_objective_value"] = objective_values
    df["mpc_fuel_cost_norm_mean"] = objective_fuel
    df["mpc_fc_smooth_norm_mean"] = objective_smooth
    df["mpc_soc_dev_norm_mean"] = objective_soc
    df["mpc_battery_use_norm_mean"] = objective_battery
    df["mpc_terminal_soc_norm"] = objective_terminal_soc
    df["mpc_fuel_cost_term"] = objective_fuel_term
    df["mpc_fc_smooth_term"] = objective_smooth_term
    df["mpc_soc_dev_term"] = objective_soc_term
    df["mpc_battery_use_term"] = objective_battery_term
    df["mpc_terminal_soc_term"] = objective_terminal_soc_term
    df["mpc_total_cost"] = objective_total_cost
    df["mpc_enable_battery_use_in_mpc"] = objective_enable_battery_use
    df["mpc_success"] = solve_success
    df["mpc_solver_message"] = solver_messages
    df["mpc_solve_stride"] = solve_stride
    if source_name:
        df["mpc_source"] = source_name
    return df
