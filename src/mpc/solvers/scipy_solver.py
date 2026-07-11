from __future__ import annotations

from dataclasses import replace

import numpy as np

from mpc.solvers.casadi_solver import (
    CasadiMPCConfig,
    CasadiMPCResult,
    DualSideCasadiMPCResult,
    _build_dual_objective_info,
    _build_single_objective_info,
)


def scipy_available() -> bool:
    try:
        import scipy.optimize  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def _clip_soc_for_feasibility(value: float, cfg: CasadiMPCConfig) -> float:
    return float(np.clip(value, cfg.soc_min, cfg.soc_max))


def _smooth_abs(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.square(value) + eps)


class ShipScipyMPC:
    """Strict upper-layer MPC solved with SciPy/SLSQP.

    This is used when CasADi is not installed. It still solves a constrained
    rolling optimization problem: power balance is enforced by deriving
    battery power from load and fuel-cell power, while SOC, device limits and
    fuel-cell ramp limits are hard SLSQP constraints.
    """

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not scipy_available():
            raise ModuleNotFoundError("scipy is not available in the current interpreter.")
        self.config = config or CasadiMPCConfig()
        self._last_solution: np.ndarray | None = None

    def _soc_traj(self, soc0: float, p_batt: np.ndarray) -> np.ndarray:
        cfg = self.config
        soc = np.empty(cfg.prediction_horizon + 1, dtype=float)
        soc[0] = _clip_soc_for_feasibility(soc0, cfg)
        for k in range(cfg.prediction_horizon):
            soc[k + 1] = soc[k] - p_batt[k] * cfg.dt_hours / cfg.battery_capacity_kwh
        return soc

    def solve(
        self,
        current_soc: float,
        prev_fc_kw: float,
        load_forecast_kw: np.ndarray,
        terminal_load_kw: float | None = None,
    ) -> CasadiMPCResult:
        from scipy.optimize import Bounds, minimize

        cfg = self.config
        load = np.asarray(load_forecast_kw, dtype=float).reshape(-1)
        if load.shape[0] != cfg.prediction_horizon:
            raise ValueError(f"Expected horizon {cfg.prediction_horizon}, got {load.shape[0]}")

        prev_fc = float(np.clip(prev_fc_kw, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw))
        current_soc = _clip_soc_for_feasibility(current_soc, cfg)
        lower_from_batt = load - cfg.battery_discharge_max_kw
        upper_from_batt = load + cfg.battery_charge_max_kw
        lb = np.maximum(cfg.fuel_cell_min_kw, lower_from_batt)
        ub = np.minimum(cfg.fuel_cell_max_kw, upper_from_batt)
        lb = np.minimum(lb, ub)

        if self._last_solution is None:
            soc_bias = 0.35 * cfg.fuel_cell_max_kw * (cfg.soc_target - current_soc)
            x0 = np.clip(0.8 * load + soc_bias, lb, ub)
        else:
            x0 = np.clip(np.roll(self._last_solution, -1), lb, ub)
            x0[-1] = x0[-2] if len(x0) > 1 else x0[-1]

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            p_fc = np.asarray(x, dtype=float)
            p_batt = load - p_fc
            soc = self._soc_traj(current_soc, p_batt)
            return p_fc, p_batt, soc

        terminal_load = float(load[-1] if terminal_load_kw is None else terminal_load_kw)

        def objective(x: np.ndarray) -> float:
            p_fc, p_batt, soc = unpack(x)
            delta_fc = np.diff(np.concatenate([[prev_fc], p_fc]))
            fuel_cost_norm = p_fc / max(cfg.fuel_cell_max_kw, 1e-6)
            fc_smooth_norm = _smooth_abs(delta_fc) / max(cfg.fuel_cell_max_kw, 1e-6)
            soc_dev_norm = _smooth_abs(soc[1:] - cfg.soc_target) / max(cfg.soc_max - cfg.soc_min, 1e-6)
            terminal_batt = terminal_load - p_fc[-1]
            terminal_soc = soc[-1] - terminal_batt * cfg.dt_hours / cfg.battery_capacity_kwh
            terminal_soc_norm = float(abs(terminal_soc - cfg.soc_target) / max(cfg.soc_max - cfg.soc_min, 1e-6))
            battery_use_norm = (p_batt / max(cfg.battery_discharge_max_kw, 1e-6)) ** 2
            battery_use_term = cfg.q4_battery_use * battery_use_norm if cfg.enable_battery_use_in_mpc else 0.0
            return float(
                np.sum(
                    cfg.q1_fuel_cost * fuel_cost_norm
                    + cfg.q2_fc_smooth * fc_smooth_norm
                    + cfg.q3_soc_dev * soc_dev_norm
                    + battery_use_term
                )
                + cfg.q5_soc_terminal * terminal_soc_norm
            )

        def soc_min_constraint(x: np.ndarray) -> np.ndarray:
            return unpack(x)[2][1:] - cfg.soc_min

        def soc_max_constraint(x: np.ndarray) -> np.ndarray:
            return cfg.soc_max - unpack(x)[2][1:]

        def ramp_up_constraint(x: np.ndarray) -> np.ndarray:
            delta = np.diff(np.concatenate([[prev_fc], x]))
            return cfg.fuel_cell_ramp_kw - delta

        def ramp_down_constraint(x: np.ndarray) -> np.ndarray:
            delta = np.diff(np.concatenate([[prev_fc], x]))
            return cfg.fuel_cell_ramp_kw + delta

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=Bounds(lb, ub),
            constraints=[
                {"type": "ineq", "fun": soc_min_constraint},
                {"type": "ineq", "fun": soc_max_constraint},
                {"type": "ineq", "fun": ramp_up_constraint},
                {"type": "ineq", "fun": ramp_down_constraint},
            ],
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )
        p_fc, p_batt, soc = unpack(result.x if result.success else x0)
        self._last_solution = p_fc.copy()
        objective_info = _build_single_objective_info(
            p_fc_traj=p_fc,
            p_bat_traj=p_batt,
            soc_traj=soc,
            prev_fc_kw=prev_fc,
            config=cfg,
            terminal_load_kw=terminal_load,
        )

        return CasadiMPCResult(
            fuel_cell_ref_kw=float(p_fc[0]),
            battery_ref_kw=float(p_batt[0]),
            predicted_soc=float(soc[1]),
            soc_ref=float(cfg.soc_target),
            fuel_cell_ref_traj_kw=[float(v) for v in p_fc],
            battery_ref_traj_kw=[float(v) for v in p_batt],
            soc_pred_traj=[float(v) for v in soc],
            objective_value=float(objective_info["objective_value"]),
            objective_info={**objective_info, "solver_message": str(result.message)},
            success=bool(result.success),
        )


class ShipDualSideScipyMPC:
    """Dual-side strict MPC solved with SciPy/SLSQP."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not scipy_available():
            raise ModuleNotFoundError("scipy is not available in the current interpreter.")
        self.config = config or CasadiMPCConfig()
        self._last_solution: np.ndarray | None = None

    def _soc_traj(self, soc0: float, p_batt: np.ndarray, capacity_kwh: float) -> np.ndarray:
        cfg = self.config
        soc = np.empty(cfg.prediction_horizon + 1, dtype=float)
        soc[0] = _clip_soc_for_feasibility(soc0, cfg)
        for k in range(cfg.prediction_horizon):
            soc[k + 1] = soc[k] - p_batt[k] * cfg.dt_hours / capacity_kwh
        return soc

    def solve(
        self,
        soc_left_0: float,
        soc_right_0: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ) -> DualSideCasadiMPCResult:
        from scipy.optimize import Bounds, minimize

        cfg = self.config
        N = cfg.prediction_horizon
        load_left = np.asarray(load_left_forecast_kw, dtype=float).reshape(-1)
        load_right = np.asarray(load_right_forecast_kw, dtype=float).reshape(-1)
        if load_left.shape[0] != N or load_right.shape[0] != N:
            raise ValueError("Dual-side load forecasts must match prediction horizon.")

        half_capacity = cfg.battery_capacity_kwh / 2.0
        half_fc_max = cfg.fuel_cell_max_kw / 2.0
        half_charge_max = cfg.battery_charge_max_kw / 2.0
        half_discharge_max = cfg.battery_discharge_max_kw / 2.0
        half_ramp = cfg.fuel_cell_ramp_kw / 2.0
        prev_left = float(np.clip(prev_fc_left_kw, cfg.fuel_cell_min_kw, half_fc_max))
        prev_right = float(np.clip(prev_fc_right_kw, cfg.fuel_cell_min_kw, half_fc_max))
        soc_left_0 = _clip_soc_for_feasibility(soc_left_0, cfg)
        soc_right_0 = _clip_soc_for_feasibility(soc_right_0, cfg)

        lb_left = np.maximum(cfg.fuel_cell_min_kw, load_left - half_discharge_max)
        ub_left = np.minimum(half_fc_max, load_left + half_charge_max)
        lb_right = np.maximum(cfg.fuel_cell_min_kw, load_right - half_discharge_max)
        ub_right = np.minimum(half_fc_max, load_right + half_charge_max)
        lb_left = np.minimum(lb_left, ub_left)
        lb_right = np.minimum(lb_right, ub_right)
        lb = np.concatenate([lb_left, lb_right])
        ub = np.concatenate([ub_left, ub_right])

        if self._last_solution is None:
            soc_bias_left = 0.35 * half_fc_max * (cfg.soc_target - soc_left_0)
            soc_bias_right = 0.35 * half_fc_max * (cfg.soc_target - soc_right_0)
            x0_left = np.clip(0.8 * load_left + soc_bias_left, lb_left, ub_left)
            x0_right = np.clip(0.8 * load_right + soc_bias_right, lb_right, ub_right)
            x0 = np.concatenate([x0_left, x0_right])
        else:
            shifted_left = np.roll(self._last_solution[:N], -1)
            shifted_right = np.roll(self._last_solution[N:], -1)
            if N > 1:
                shifted_left[-1] = shifted_left[-2]
                shifted_right[-1] = shifted_right[-2]
            x0 = np.clip(np.concatenate([shifted_left, shifted_right]), lb, ub)

        def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            p_fc_left = np.asarray(x[:N], dtype=float)
            p_fc_right = np.asarray(x[N:], dtype=float)
            p_batt_left = load_left - p_fc_left
            p_batt_right = load_right - p_fc_right
            soc_left = self._soc_traj(soc_left_0, p_batt_left, half_capacity)
            soc_right = self._soc_traj(soc_right_0, p_batt_right, half_capacity)
            return p_fc_left, p_fc_right, p_batt_left, p_batt_right, soc_left, soc_right

        terminal_load_left = float(load_left[-1] if terminal_load_left_kw is None else terminal_load_left_kw)
        terminal_load_right = float(load_right[-1] if terminal_load_right_kw is None else terminal_load_right_kw)

        def objective(x: np.ndarray) -> float:
            p_fc_left, p_fc_right, p_batt_left, p_batt_right, soc_left, soc_right = unpack(x)
            fc_scale = max(cfg.fuel_cell_max_kw, 1e-6)
            bat_scale = max(half_discharge_max, 1e-6)
            soc_span = max(cfg.soc_max - cfg.soc_min, 1e-6)
            delta_left = np.diff(np.concatenate([[prev_left], p_fc_left]))
            delta_right = np.diff(np.concatenate([[prev_right], p_fc_right]))
            fuel_cost_norm = (p_fc_left + p_fc_right) / fc_scale
            fc_smooth_norm = (_smooth_abs(delta_left) + _smooth_abs(delta_right)) / fc_scale
            soc_dev_norm = (_smooth_abs(soc_left[1:] - cfg.soc_target) + _smooth_abs(soc_right[1:] - cfg.soc_target)) / (
                2.0 * soc_span
            )
            terminal_batt_left = terminal_load_left - p_fc_left[-1]
            terminal_batt_right = terminal_load_right - p_fc_right[-1]
            terminal_soc_left = soc_left[-1] - terminal_batt_left * cfg.dt_hours / half_capacity
            terminal_soc_right = soc_right[-1] - terminal_batt_right * cfg.dt_hours / half_capacity
            terminal_soc_norm = (
                abs(terminal_soc_left - cfg.soc_target) + abs(terminal_soc_right - cfg.soc_target)
            ) / (2.0 * soc_span)
            battery_use_norm = ((p_batt_left / bat_scale) ** 2 + (p_batt_right / bat_scale) ** 2) / 2.0
            battery_use_term = cfg.q4_battery_use * battery_use_norm if cfg.enable_battery_use_in_mpc else 0.0
            return float(
                np.sum(
                    cfg.q1_fuel_cost * fuel_cost_norm
                    + cfg.q2_fc_smooth * fc_smooth_norm
                    + cfg.q3_soc_dev * soc_dev_norm
                    + battery_use_term
                )
                + cfg.q5_soc_terminal * terminal_soc_norm
            )

        def soc_min_constraint(x: np.ndarray) -> np.ndarray:
            *_, soc_left, soc_right = unpack(x)
            return np.concatenate([soc_left[1:] - cfg.soc_min, soc_right[1:] - cfg.soc_min])

        def soc_max_constraint(x: np.ndarray) -> np.ndarray:
            *_, soc_left, soc_right = unpack(x)
            return np.concatenate([cfg.soc_max - soc_left[1:], cfg.soc_max - soc_right[1:]])

        def ramp_up_constraint(x: np.ndarray) -> np.ndarray:
            p_fc_left, p_fc_right, *_ = unpack(x)
            delta_left = np.diff(np.concatenate([[prev_left], p_fc_left]))
            delta_right = np.diff(np.concatenate([[prev_right], p_fc_right]))
            return np.concatenate([half_ramp - delta_left, half_ramp - delta_right])

        def ramp_down_constraint(x: np.ndarray) -> np.ndarray:
            p_fc_left, p_fc_right, *_ = unpack(x)
            delta_left = np.diff(np.concatenate([[prev_left], p_fc_left]))
            delta_right = np.diff(np.concatenate([[prev_right], p_fc_right]))
            return np.concatenate([half_ramp + delta_left, half_ramp + delta_right])

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=Bounds(lb, ub),
            constraints=[
                {"type": "ineq", "fun": soc_min_constraint},
                {"type": "ineq", "fun": soc_max_constraint},
                {"type": "ineq", "fun": ramp_up_constraint},
                {"type": "ineq", "fun": ramp_down_constraint},
            ],
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )
        values = result.x if result.success else x0
        self._last_solution = np.asarray(values, dtype=float).copy()
        p_fc_left, p_fc_right, p_batt_left, p_batt_right, soc_left, soc_right = unpack(values)
        objective_info = _build_dual_objective_info(
            p_fc_left_traj=p_fc_left,
            p_fc_right_traj=p_fc_right,
            p_batt_left_traj=p_batt_left,
            p_batt_right_traj=p_batt_right,
            soc_left_traj=soc_left,
            soc_right_traj=soc_right,
            prev_fc_left_kw=prev_left,
            prev_fc_right_kw=prev_right,
            config=cfg,
            terminal_load_left_kw=terminal_load_left,
            terminal_load_right_kw=terminal_load_right,
        )

        return DualSideCasadiMPCResult(
            fuel_cell_ref_left_kw=float(p_fc_left[0]),
            fuel_cell_ref_right_kw=float(p_fc_right[0]),
            battery_ref_left_kw=float(p_batt_left[0]),
            battery_ref_right_kw=float(p_batt_right[0]),
            predicted_soc_left=float(soc_left[1]),
            predicted_soc_right=float(soc_right[1]),
            soc_ref=float(cfg.soc_target),
            fuel_cell_ref_left_traj_kw=[float(v) for v in p_fc_left],
            fuel_cell_ref_right_traj_kw=[float(v) for v in p_fc_right],
            battery_ref_left_traj_kw=[float(v) for v in p_batt_left],
            battery_ref_right_traj_kw=[float(v) for v in p_batt_right],
            soc_pred_left_traj=[float(v) for v in soc_left],
            soc_pred_right_traj=[float(v) for v in soc_right],
            objective_value=float(objective_info["objective_value"]),
            objective_info={**objective_info, "solver_message": str(result.message)},
            success=bool(result.success),
        )


def solve_upper_mpc_scipy(
    current_state: dict,
    load_prediction: list[float] | np.ndarray,
    refs: dict | None,
    params: CasadiMPCConfig | None,
) -> CasadiMPCResult:
    config = params or CasadiMPCConfig()
    if refs and "soc_ref" in refs:
        config = replace(config, soc_target=float(refs["soc_ref"]))
    solver = ShipScipyMPC(config)
    return solver.solve(
        current_soc=float(current_state["soc"]),
        prev_fc_kw=float(current_state.get("previous_fuel_cell_kw", 0.0)),
        load_forecast_kw=np.asarray(load_prediction, dtype=float),
    )
