from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from benchmark_mpc_qp_osqp_1s import (
    _h2_kg_for_step,
    _qp_bounds_for_step,
    _solve_with_persistent_osqp,
    _try_import_osqp,
    default_config,
)
from mpc_solvers.mpc_qp_formulation import (
    QpMpcConfig,
    QpProblem,
    build_qp_problem,
    resolved_ramp_kw_per_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = REPO_ROOT / "outputs" / "mpc_solver_benchmark_1s" / "data" / "test_voyages_spline_1s.parquet"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "mpc_1s_n6_weight_selection"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_SELECTION_CONFIG = REPO_ROOT / "configs" / "benchmarks" / "mpc_1s_n6_provisional.json"
EXPECTED_TEST_VOYAGES: tuple[str, ...] = tuple(f"voyage_{index:03d}" for index in range(60, 67))
N6_HORIZON = 6
N6_DT_SECONDS = 1.0
N6_OSQP_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "polishing": True,
    "warm_starting": True,
    "eps_abs": 1.0e-5,
    "eps_rel": 1.0e-5,
    "max_iter": 10000,
    "adaptive_rho_interval": 25,
}
N6_TOLERANCES: dict[str, float] = {
    "actual_balance_kw": 0.01,
    "qp_balance_kw": 0.1,
    "power_bound_kw": 0.1,
    "ramp_kw": 0.1,
    "soc": 1.0e-6,
    "soc_prediction": 1.0e-5,
    "fc_above_load_kw": 1.0,
    "near_limit_kw": 1.0,
}
REQUIRED_N6_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "total_steps",
        "solver_success_rate",
        "solved_inaccurate_fraction",
        "max_iter_count",
        "max_iter_fraction",
        "solve_time_ms_mean",
        "solve_time_ms_p95",
        "solve_time_ms_p99",
        "solve_time_ms_max",
        "max_plan_power_balance_residual_kw",
        "max_fc_bound_residual_kw",
        "max_battery_bound_residual_kw",
        "max_ramp_residual_kw",
        "initial_soc",
        "final_soc",
        "soc_net_change",
        "soc_min",
        "soc_max",
        "hydrogen_total_kg",
        "load_energy_mwh",
        "hydrogen_intensity_kg_per_mwh",
        "battery_charge_energy_kwh",
        "battery_discharge_energy_kwh",
        "battery_throughput_kwh",
        "fc_at_max_fraction",
        "battery_near_limit_fraction",
        "fc_above_load_fraction",
        "fc_surplus_energy_kwh",
    }
)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {"candidate_id": "A", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "B", "q_h2": 0.5, "q_soc": 1.5, "q_batt": 0.05, "soc_band": 0.05},
    {"candidate_id": "C", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.05, "soc_band": 0.075},
    {"candidate_id": "D", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.075, "soc_band": 0.05},
)


@dataclass(frozen=True)
class N6QpTransform:
    variable_scale: np.ndarray
    variable_offset: np.ndarray
    row_scale: np.ndarray
    constraint_offset: np.ndarray
    objective_constant: float

    def to_normalized(self, physical: np.ndarray) -> np.ndarray:
        values = np.asarray(physical, dtype=float).reshape(-1)
        return (values - self.variable_offset) / self.variable_scale

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=float).reshape(-1)
        return self.variable_offset + self.variable_scale * values

    def transform_bounds(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_values = np.asarray(lower, dtype=float).reshape(-1)
        upper_values = np.asarray(upper, dtype=float).reshape(-1)
        return (
            self.row_scale * (lower_values - self.constraint_offset),
            self.row_scale * (upper_values - self.constraint_offset),
        )


def scale_n6_qp_problem(
    problem: QpProblem,
    *,
    config: QpMpcConfig,
    soc_reference: float = 0.55,
) -> tuple[QpProblem, N6QpTransform]:
    horizon = int(config.horizon)
    if horizon != N6_HORIZON:
        raise ValueError(f"N=6 scaling requires horizon={N6_HORIZON}, got {horizon}")
    expected_variables = 3 * horizon + 1
    expected_constraints = 6 * horizon + 2
    if problem.P.shape != (expected_variables, expected_variables):
        raise ValueError("unexpected N=6 QP variable dimensions")
    if problem.A.shape != (expected_constraints, expected_variables):
        raise ValueError("unexpected N=6 QP constraint dimensions")

    fuel_cell_scale = float(config.fuel_cell_max_kw)
    battery_scale = max(
        float(config.battery_charge_max_kw),
        float(config.battery_discharge_max_kw),
    )
    soc_scale = float(config.soc_band)
    ramp_scale = max(float(resolved_ramp_kw_per_step(config)), 1.0)
    variable_scale = np.concatenate(
        [
            np.full(horizon, fuel_cell_scale),
            np.full(horizon, battery_scale),
            np.full(horizon + 1, soc_scale),
        ]
    )
    variable_offset = np.concatenate(
        [
            np.zeros(2 * horizon),
            np.full(horizon + 1, float(soc_reference)),
        ]
    )
    row_scale = np.concatenate(
        [
            np.full(horizon, 1.0 / fuel_cell_scale),
            np.full(horizon, 1.0 / battery_scale),
            np.full(horizon + 1, 1.0 / soc_scale),
            np.full(1 + horizon, 1.0 / soc_scale),
            np.full(horizon, 1.0 / fuel_cell_scale),
            np.full(horizon, 1.0 / ramp_scale),
        ]
    )
    scale_matrix = sparse.diags(variable_scale, format="csc")
    row_matrix = sparse.diags(row_scale, format="csc")
    constraint_offset = np.asarray(problem.A @ variable_offset, dtype=float).reshape(-1)
    scaled_p = (scale_matrix @ problem.P @ scale_matrix).tocsc()
    scaled_q = variable_scale * (
        np.asarray(problem.P @ variable_offset, dtype=float).reshape(-1) + problem.q
    )
    scaled_a = (row_matrix @ problem.A @ scale_matrix).tocsc()
    scaled_l = row_scale * (problem.l - constraint_offset)
    scaled_u = row_scale * (problem.u - constraint_offset)
    objective_constant = float(
        0.5 * variable_offset @ problem.P @ variable_offset + problem.q @ variable_offset
    )
    transform = N6QpTransform(
        variable_scale=variable_scale,
        variable_offset=variable_offset,
        row_scale=row_scale,
        constraint_offset=constraint_offset,
        objective_constant=objective_constant,
    )
    metadata = {
        **problem.metadata,
        "osqp_affine_scaling": True,
        "osqp_variable_scale": variable_scale.tolist(),
        "osqp_variable_offset": variable_offset.tolist(),
        "osqp_constraint_row_scale": row_scale.tolist(),
    }
    return (
        QpProblem(P=scaled_p, q=scaled_q, A=scaled_a, l=scaled_l, u=scaled_u, metadata=metadata),
        transform,
    )


def _setup_n6_osqp_solver(osqp_module: Any, problem: QpProblem) -> Any:
    solver = osqp_module.OSQP()
    solver.setup(
        P=problem.P,
        q=problem.q,
        A=problem.A,
        l=problem.l,
        u=problem.u,
        **N6_OSQP_SETTINGS,
    )
    return solver


def candidate_config(candidate_id: str) -> QpMpcConfig:
    selected = next(
        (candidate for candidate in CANDIDATES if candidate["candidate_id"] == str(candidate_id).upper()),
        None,
    )
    if selected is None:
        raise ValueError(f"candidate_id must be one of {[item['candidate_id'] for item in CANDIDATES]}")
    return default_config(
        horizon=N6_HORIZON,
        battery_capacity_kwh=693.0,
        q_h2=float(selected["q_h2"]),
        q_soc=float(selected["q_soc"]),
        q_batt=float(selected["q_batt"]),
        q_ramp=0.0,
        q_terminal_soc=0.0,
        battery_power_max_kw=346.5,
        battery_power_ref_kw=346.5,
        soc_band=float(selected["soc_band"]),
    )


def physical_h2_kg_step(config: QpMpcConfig, p_fc_kw: float) -> float:
    return float(_h2_kg_for_step(config, float(p_fc_kw)))


def ideal_future_window(loads_kw: np.ndarray, *, decision_index: int) -> np.ndarray:
    loads = np.asarray(loads_kw, dtype=float).reshape(-1)
    index = int(decision_index)
    if index < 0 or index >= len(loads) - 1:
        raise ValueError("decision_index must have a future execution sample")
    horizon = loads[index + 1 : index + 1 + N6_HORIZON]
    if len(horizon) < N6_HORIZON:
        horizon = np.pad(horizon, (0, N6_HORIZON - len(horizon)), mode="edge")
    return horizon.astype(float, copy=False)


def extract_first_step(
    solution: np.ndarray,
    *,
    config: QpMpcConfig,
    load_actual_kw: float,
    current_soc: float,
) -> dict[str, float]:
    horizon = int(config.horizon)
    if horizon != N6_HORIZON:
        raise ValueError(f"N=6 experiment requires horizon={N6_HORIZON}, got {horizon}")
    values = np.asarray(solution, dtype=float).reshape(-1)
    expected_size = 3 * horizon + 1
    if len(values) != expected_size:
        raise ValueError(f"N=6 solution must contain {expected_size} values")

    p_fc_plan = float(values[0])
    p_batt_plan = float(values[horizon])
    soc_predicted = float(values[2 * horizon + 1])
    p_fc_actual = p_fc_plan
    p_batt_actual = float(load_actual_kw) - p_fc_actual
    soc_actual = float(current_soc) - (
        p_batt_actual * float(config.dt_seconds) / 3600.0 / float(config.battery_capacity_kwh)
    )
    return {
        "P_fc_plan_kw": p_fc_plan,
        "P_batt_plan_kw": p_batt_plan,
        "SOC_predicted": soc_predicted,
        "P_fc_actual_kw": p_fc_actual,
        "P_batt_actual_kw": p_batt_actual,
        "SOC_actual": soc_actual,
    }


def run_voyage(
    *,
    voyage_id: str,
    loads_kw: np.ndarray,
    times_s: np.ndarray,
    candidate_id: str,
    max_steps: int | None = None,
    initial_soc: float = 0.55,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loads = np.asarray(loads_kw, dtype=float).reshape(-1)
    times = np.asarray(times_s, dtype=float).reshape(-1)
    if len(loads) != len(times):
        raise ValueError("loads_kw and times_s must have the same length")
    if len(loads) < 2:
        raise ValueError("a voyage must contain at least two samples")
    if not np.all(np.isfinite(loads)) or not np.all(np.isfinite(times)):
        raise ValueError("voyage loads and times must be finite")
    if not np.allclose(np.diff(times), N6_DT_SECONDS, rtol=0.0, atol=1.0e-9):
        raise ValueError("voyage times must be strictly spaced at 1 s")

    config = candidate_config(candidate_id)
    osqp_module, osqp_error = _try_import_osqp()
    if osqp_module is None:
        raise RuntimeError(f"Cannot import osqp: {osqp_error}")

    current_soc = float(initial_soc)
    prev_fc_actual = float(np.clip(loads[0], config.fuel_cell_min_kw, config.fuel_cell_max_kw))
    setup_horizon = ideal_future_window(loads, decision_index=0)
    setup_problem = build_qp_problem(
        config,
        load_forecast_kw=setup_horizon,
        current_soc=current_soc,
        prev_fc_kw=prev_fc_actual,
        soc_reference=0.55,
        include_diagnostics=False,
    )
    scaled_setup_problem, transform = scale_n6_qp_problem(setup_problem, config=config)
    solver = _setup_n6_osqp_solver(osqp_module, scaled_setup_problem)

    decision_count = len(loads) - 1
    if max_steps is not None:
        decision_count = min(decision_count, max(0, int(max_steps)))
    control_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []

    for decision_index in range(decision_count):
        execution_index = decision_index + 1
        load_horizon = ideal_future_window(loads, decision_index=decision_index)
        load_actual = float(loads[execution_index])
        soc_before = float(current_soc)
        prev_fc_before = float(prev_fc_actual)
        lower, upper = _qp_bounds_for_step(
            config,
            load_forecast_kw=load_horizon,
            current_soc=soc_before,
            prev_fc_kw=prev_fc_before,
        )
        scaled_lower, scaled_upper = transform.transform_bounds(lower, upper)
        result, solve_ms = _solve_with_persistent_osqp(
            solver,
            lower=scaled_lower,
            upper=scaled_upper,
        )
        initial_status = str(result.info.status)
        initial_status_lower = initial_status.lower()
        cold_restart_used = bool(
            "maximum iterations" in initial_status_lower or "max_iter" in initial_status_lower
        )
        attempt_count = 1
        if cold_restart_used:
            setup_start = time.perf_counter()
            recovery_problem = build_qp_problem(
                config,
                load_forecast_kw=load_horizon,
                current_soc=soc_before,
                prev_fc_kw=prev_fc_before,
                soc_reference=0.55,
                include_diagnostics=False,
            )
            scaled_recovery_problem, transform = scale_n6_qp_problem(
                recovery_problem,
                config=config,
            )
            solver = _setup_n6_osqp_solver(osqp_module, scaled_recovery_problem)
            recovery_setup_ms = (time.perf_counter() - setup_start) * 1000.0
            scaled_lower, scaled_upper = transform.transform_bounds(lower, upper)
            result, recovery_solve_ms = _solve_with_persistent_osqp(
                solver,
                lower=scaled_lower,
                upper=scaled_upper,
            )
            solve_ms = float(solve_ms) + float(recovery_setup_ms) + float(recovery_solve_ms)
            attempt_count = 2
        status = str(result.info.status)
        status_lower = status.lower()
        success = bool(status_lower.startswith("solved") and result.x is not None)
        solved_inaccurate = bool("solved inaccurate" in status_lower)
        max_iter_reached = bool(
            cold_restart_used
            or "maximum iterations" in status_lower
            or "max_iter" in status_lower
        )
        cold_restart_succeeded = bool(cold_restart_used and success)

        if success:
            applied = extract_first_step(
                transform.to_physical(np.asarray(result.x, dtype=float)),
                config=config,
                load_actual_kw=load_actual,
                current_soc=soc_before,
            )
            prev_fc_actual = float(applied["P_fc_actual_kw"])
            current_soc = float(applied["SOC_actual"])
        else:
            applied = {
                "P_fc_plan_kw": float("nan"),
                "P_batt_plan_kw": float("nan"),
                "SOC_predicted": float("nan"),
                "P_fc_actual_kw": float("nan"),
                "P_batt_actual_kw": float("nan"),
                "SOC_actual": float("nan"),
            }

        p_fc = float(applied["P_fc_actual_kw"])
        p_batt_actual = float(applied["P_batt_actual_kw"])
        p_batt_plan = float(applied["P_batt_plan_kw"])
        soc_actual = float(applied["SOC_actual"])
        fc_delta = p_fc - prev_fc_before if success else float("nan")
        plan_balance_residual = (
            abs(float(applied["P_fc_plan_kw"]) + p_batt_plan - float(load_horizon[0]))
            if success
            else float("nan")
        )
        actual_balance_residual = abs(p_fc + p_batt_actual - load_actual) if success else float("nan")
        fc_bound_residual = (
            max(0.0, float(config.fuel_cell_min_kw) - p_fc, p_fc - float(config.fuel_cell_max_kw))
            if success
            else float("nan")
        )
        batt_bound_residual = (
            max(
                0.0,
                -float(config.battery_charge_max_kw) - p_batt_actual,
                p_batt_actual - float(config.battery_discharge_max_kw),
            )
            if success
            else float("nan")
        )
        ramp_residual = (
            max(0.0, abs(fc_delta) - float(config.fuel_cell_ramp_rate_kw_per_s) * float(config.dt_seconds))
            if success
            else float("nan")
        )
        soc_bound_residual = (
            max(0.0, float(config.soc_min) - soc_actual, soc_actual - float(config.soc_max))
            if success
            else float("nan")
        )
        soc_prediction_residual = (
            abs(float(applied["SOC_predicted"]) - soc_actual) if success else float("nan")
        )

        control_row: dict[str, Any] = {
            "candidate_id": str(candidate_id).upper(),
            "voyage_id": str(voyage_id),
            "voyage_expected_steps": int(decision_count),
            "decision_index": int(decision_index),
            "execution_index": int(execution_index),
            "decision_time_s": float(times[decision_index]),
            "time_s": float(times[execution_index]),
            "load_actual_kw": load_actual,
            "SOC_before": soc_before,
            "prev_fc_actual_kw": prev_fc_before,
            **{f"load_h{index + 1}_kw": float(value) for index, value in enumerate(load_horizon)},
            **applied,
            "fc_delta_actual_kw": fc_delta,
            "plan_balance_residual_kw": plan_balance_residual,
            "actual_balance_residual_kw": actual_balance_residual,
            "fc_bound_residual_kw": fc_bound_residual,
            "battery_bound_residual_kw": batt_bound_residual,
            "ramp_residual_kw": ramp_residual,
            "soc_bound_residual": soc_bound_residual,
            "soc_prediction_residual": soc_prediction_residual,
            "h2_kg_step": physical_h2_kg_step(config, p_fc) if success else float("nan"),
            "success": success,
            "status": status,
        }
        control_rows.append(control_row)
        solver_rows.append(
            {
                "candidate_id": str(candidate_id).upper(),
                "voyage_id": str(voyage_id),
                "voyage_expected_steps": int(decision_count),
                "decision_index": int(decision_index),
                "execution_index": int(execution_index),
                "time_s": float(times[execution_index]),
                "status": status,
                "initial_status": initial_status,
                "success": success,
                "solved_inaccurate": solved_inaccurate,
                "max_iter_reached": max_iter_reached,
                "cold_restart_used": cold_restart_used,
                "cold_restart_succeeded": cold_restart_succeeded,
                "attempt_count": attempt_count,
                "iterations": int(getattr(result.info, "iter", -1)),
                "solve_ms": float(solve_ms),
                "primal_residual": float(getattr(result.info, "prim_res", np.nan)),
                "dual_residual": float(getattr(result.info, "dual_res", np.nan)),
            }
        )
        if not success:
            break

    return pd.DataFrame(control_rows), pd.DataFrame(solver_rows)


def _expected_step_count(frame: pd.DataFrame) -> int:
    if frame.empty or "voyage_expected_steps" not in frame.columns:
        return int(len(frame))
    expected = pd.to_numeric(frame["voyage_expected_steps"], errors="coerce")
    if "voyage_id" not in frame.columns:
        return int(expected.max()) if expected.notna().any() else int(len(frame))
    temporary = pd.DataFrame(
        {
            "voyage_id": frame["voyage_id"].astype(str),
            "expected": expected,
        }
    )
    return int(temporary.groupby("voyage_id", sort=False)["expected"].max().fillna(0).sum())


def _solver_statistics(frame: pd.DataFrame, *, scope: str, voyage_id: str) -> dict[str, Any]:
    count = int(len(frame))
    expected_count = _expected_step_count(frame)
    success = frame["success"].fillna(False).astype(bool) if count else pd.Series(dtype=bool)
    inaccurate = (
        frame["solved_inaccurate"].fillna(False).astype(bool) if count else pd.Series(dtype=bool)
    )
    max_iter = frame["max_iter_reached"].fillna(False).astype(bool) if count else pd.Series(dtype=bool)
    statuses = (
        frame["status"].fillna("missing").astype(str) if count and "status" in frame else pd.Series(dtype=str)
    )
    status_lower = statuses.str.lower()
    final_max_iter = status_lower.str.contains("maximum iterations|max_iter", regex=True)
    primal_infeasible = status_lower.str.contains("primal infeasible", regex=False)
    dual_infeasible = status_lower.str.contains("dual infeasible", regex=False)
    cold_restart = (
        frame["cold_restart_used"].fillna(False).astype(bool)
        if count and "cold_restart_used" in frame
        else pd.Series(False, index=frame.index, dtype=bool)
    )
    cold_restart_succeeded = (
        frame["cold_restart_succeeded"].fillna(False).astype(bool)
        if count and "cold_restart_succeeded" in frame
        else pd.Series(False, index=frame.index, dtype=bool)
    )
    solve_ms = pd.to_numeric(frame["solve_ms"], errors="coerce").dropna() if count else pd.Series(dtype=float)
    iterations = (
        pd.to_numeric(frame["iterations"], errors="coerce").dropna() if count else pd.Series(dtype=float)
    )
    primal = (
        pd.to_numeric(frame["primal_residual"], errors="coerce").abs().dropna()
        if count
        else pd.Series(dtype=float)
    )
    dual = (
        pd.to_numeric(frame["dual_residual"], errors="coerce").abs().dropna()
        if count
        else pd.Series(dtype=float)
    )
    return {
        "scope": scope,
        "voyage_id": voyage_id,
        "total_steps": expected_count,
        "solver_attempt_count": count,
        "solver_success_count": int(success.sum()) if count else 0,
        "solver_failure_count": int((~success).sum()) if count else 0,
        "solver_success_rate": float(success.mean()) if count else float("nan"),
        "closed_loop_coverage_fraction": (
            float(success.sum() / expected_count) if expected_count else float("nan")
        ),
        "unattempted_after_failure_count": max(0, expected_count - count),
        "closed_loop_complete": bool(
            expected_count > 0 and count == expected_count and bool(success.all())
        ),
        "solved_inaccurate_count": int(inaccurate.sum()) if count else 0,
        "solved_inaccurate_fraction": float(inaccurate.mean()) if count else float("nan"),
        "max_iter_count": int(max_iter.sum()) if count else 0,
        "max_iter_fraction": float(max_iter.mean()) if count else float("nan"),
        "final_max_iter_count": int(final_max_iter.sum()) if count else 0,
        "primal_infeasible_count": int(primal_infeasible.sum()) if count else 0,
        "dual_infeasible_count": int(dual_infeasible.sum()) if count else 0,
        "cold_restart_count": int(cold_restart.sum()) if count else 0,
        "cold_restart_success_count": int(cold_restart_succeeded.sum()) if count else 0,
        "final_status_counts": (
            json.dumps(statuses.value_counts().sort_index().to_dict(), ensure_ascii=False)
            if count
            else "{}"
        ),
        "solve_time_ms_mean": float(solve_ms.mean()) if len(solve_ms) else float("nan"),
        "solve_time_ms_p95": float(solve_ms.quantile(0.95)) if len(solve_ms) else float("nan"),
        "solve_time_ms_p99": float(solve_ms.quantile(0.99)) if len(solve_ms) else float("nan"),
        "solve_time_ms_max": float(solve_ms.max()) if len(solve_ms) else float("nan"),
        "iterations_mean": float(iterations.mean()) if len(iterations) else float("nan"),
        "iterations_max": int(iterations.max()) if len(iterations) else -1,
        "primal_residual_max_abs": float(primal.max()) if len(primal) else float("nan"),
        "dual_residual_max_abs": float(dual.max()) if len(dual) else float("nan"),
    }


def _physical_metrics(frame: pd.DataFrame, *, config: QpMpcConfig) -> dict[str, Any]:
    success = frame["success"].fillna(False).astype(bool)
    applied = frame.loc[success].copy()
    expected_count = _expected_step_count(frame)
    complete = bool(
        expected_count > 0 and len(frame) == expected_count and bool(success.all())
    )
    dt_hours = float(config.dt_seconds) / 3600.0
    load = pd.to_numeric(applied["load_actual_kw"], errors="coerce")
    p_fc = pd.to_numeric(applied["P_fc_actual_kw"], errors="coerce")
    p_batt = pd.to_numeric(applied["P_batt_actual_kw"], errors="coerce")
    soc = pd.to_numeric(applied["SOC_actual"], errors="coerce")
    h2 = pd.to_numeric(applied["h2_kg_step"], errors="coerce")

    load_energy_mwh = float(load.sum() * dt_hours / 1000.0)
    h2_total = float(h2.sum())
    charge = float((-p_batt.clip(upper=0.0)).sum() * dt_hours)
    discharge = float(p_batt.clip(lower=0.0).sum() * dt_hours)
    surplus = (p_fc - load).clip(lower=0.0)
    initial_soc = float(frame.iloc[0]["SOC_before"]) if len(frame) else float("nan")
    final_soc = float(soc.iloc[-1]) if len(soc) else initial_soc

    actual_balance = pd.to_numeric(applied["actual_balance_residual_kw"], errors="coerce")
    plan_balance = pd.to_numeric(applied["plan_balance_residual_kw"], errors="coerce")
    fc_bound = pd.to_numeric(applied["fc_bound_residual_kw"], errors="coerce")
    batt_bound = pd.to_numeric(applied["battery_bound_residual_kw"], errors="coerce")
    ramp = pd.to_numeric(applied["ramp_residual_kw"], errors="coerce")
    soc_bound = pd.to_numeric(applied["soc_bound_residual"], errors="coerce")
    soc_prediction = pd.to_numeric(
        applied.get("soc_prediction_residual", pd.Series(0.0, index=applied.index)),
        errors="coerce",
    )
    physical_bad = (
        (actual_balance > N6_TOLERANCES["actual_balance_kw"])
        | (plan_balance > N6_TOLERANCES["qp_balance_kw"])
        | (fc_bound > N6_TOLERANCES["power_bound_kw"])
        | (batt_bound > N6_TOLERANCES["power_bound_kw"])
        | (ramp > N6_TOLERANCES["ramp_kw"])
        | (soc_bound > N6_TOLERANCES["soc"])
        | (soc_prediction > N6_TOLERANCES["soc_prediction"])
    )

    return {
        "total_steps": expected_count,
        "attempted_steps": int(len(frame)),
        "applied_steps": int(success.sum()),
        "closed_loop_coverage_fraction": (
            float(success.sum() / expected_count) if expected_count else float("nan")
        ),
        "closed_loop_complete": complete,
        "aggregate_metrics_comparable": complete,
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "soc_net_change": float(final_soc - initial_soc),
        "soc_min": min(initial_soc, float(soc.min())) if len(soc) else initial_soc,
        "soc_max": max(initial_soc, float(soc.max())) if len(soc) else initial_soc,
        "hydrogen_total_kg": h2_total,
        "load_energy_mwh": load_energy_mwh,
        "hydrogen_intensity_kg_per_mwh": (
            h2_total / load_energy_mwh if load_energy_mwh > 0.0 else float("nan")
        ),
        "battery_charge_energy_kwh": charge,
        "battery_discharge_energy_kwh": discharge,
        "battery_throughput_kwh": charge + discharge,
        "fc_at_max_fraction": (
            float((p_fc >= float(config.fuel_cell_max_kw) - N6_TOLERANCES["near_limit_kw"]).mean())
            if len(p_fc)
            else float("nan")
        ),
        "battery_near_limit_fraction": (
            float(
                (
                    p_batt.abs()
                    >= max(float(config.battery_charge_max_kw), float(config.battery_discharge_max_kw))
                    - N6_TOLERANCES["near_limit_kw"]
                ).mean()
            )
            if len(p_batt)
            else float("nan")
        ),
        "fc_above_load_fraction": (
            float((p_fc > load + N6_TOLERANCES["fc_above_load_kw"]).mean())
            if len(p_fc)
            else float("nan")
        ),
        "fc_surplus_energy_kwh": float(surplus.sum() * dt_hours),
        "max_actual_power_balance_residual_kw": (
            float(actual_balance.max()) if len(actual_balance) else float("nan")
        ),
        "max_plan_power_balance_residual_kw": (
            float(plan_balance.max()) if len(plan_balance) else float("nan")
        ),
        "max_fc_bound_residual_kw": float(fc_bound.max()) if len(fc_bound) else float("nan"),
        "max_battery_bound_residual_kw": (
            float(batt_bound.max()) if len(batt_bound) else float("nan")
        ),
        "max_ramp_residual_kw": float(ramp.max()) if len(ramp) else float("nan"),
        "max_soc_bound_residual": float(soc_bound.max()) if len(soc_bound) else float("nan"),
        "max_soc_prediction_residual": (
            float(soc_prediction.max()) if len(soc_prediction) else float("nan")
        ),
        "physical_infeasible_point_count": int(physical_bad.sum()),
    }


def build_candidate_metrics(
    control_df: pd.DataFrame,
    solver_df: pd.DataFrame,
    *,
    config: QpMpcConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    voyage_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    for voyage_id, controls in control_df.groupby("voyage_id", sort=True):
        voyage_solver = solver_df[solver_df["voyage_id"].astype(str).eq(str(voyage_id))]
        solver_stats = _solver_statistics(
            voyage_solver,
            scope="voyage",
            voyage_id=str(voyage_id),
        )
        voyage_rows.append(
            {
                "voyage_id": str(voyage_id),
                **_physical_metrics(controls, config=config),
                **{key: value for key, value in solver_stats.items() if key not in {"scope", "voyage_id"}},
            }
        )
        solver_rows.append(solver_stats)

    overall_solver = _solver_statistics(solver_df, scope="overall", voyage_id="all")
    solver_statistics = pd.DataFrame([overall_solver, *solver_rows])
    voyage_metrics = pd.DataFrame(voyage_rows)
    overall = _physical_metrics(control_df, config=config)
    if not voyage_metrics.empty:
        overall["initial_soc"] = float(voyage_metrics["initial_soc"].mean())
        overall["final_soc"] = float(voyage_metrics["final_soc"].mean())
        overall["soc_net_change"] = float(voyage_metrics["soc_net_change"].mean())
        overall["worst_voyage_soc_net_change"] = float(voyage_metrics["soc_net_change"].min())
    else:
        overall["worst_voyage_soc_net_change"] = float("nan")
    overall.update(
        {key: value for key, value in overall_solver.items() if key not in {"scope", "voyage_id"}}
    )
    return overall, voyage_metrics, solver_statistics


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _candidate_metadata(candidate_id: str, *, input_path: Path) -> dict[str, Any]:
    normalized_id = str(candidate_id).upper()
    config = candidate_config(normalized_id)
    return {
        "candidate_id": normalized_id,
        "status": "raw_candidate",
        "solver": "OSQP",
        "problem_class": "convex_qp",
        "forecast_source": "offline ideal foresight",
        "input_data": input_path.as_posix(),
        "input_data_note": "natural-clipped cubic-spline 1 s reconstruction; not measured 1 s data",
        "lstm_used": False,
        "timing": {
            "decision_interval_seconds": N6_DT_SECONDS,
            "forecast_samples": "t+1..t+6",
            "prediction_horizon_steps": N6_HORIZON,
            "control_horizon_steps": N6_HORIZON,
            "applied_action": "first step only",
            "actual_battery_definition": "P_load(t+1) - P_fc(t+1)",
            "actual_soc_update": "SOC - P_batt/(3600*693)",
            "voyage_end_policy": "repeat the final sample within the same voyage; never cross voyages",
        },
        "initial_state": {
            "soc": 0.55,
            "fuel_cell_kw": "clip(first voyage load, 0, 560)",
            "reset_for_each_voyage": True,
        },
        "soc_reference": 0.55,
        "model": asdict(config),
        "tolerances": dict(N6_TOLERANCES),
        "osqp_settings": {
            "persistent_solver_per_voyage": True,
            **N6_OSQP_SETTINGS,
            "affine_variable_and_constraint_scaling": True,
            "max_iter_recovery": "one cold restart of the same QP; no control fallback",
        },
        "scope_note": "N=60 is historical benchmark only; this run does not search N=60 weights.",
    }


def _constraint_audit_markdown(candidate_id: str, summary: dict[str, Any]) -> str:
    metrics = (
        "max_actual_power_balance_residual_kw",
        "max_plan_power_balance_residual_kw",
        "max_fc_bound_residual_kw",
        "max_battery_bound_residual_kw",
        "max_ramp_residual_kw",
        "max_soc_bound_residual",
        "max_soc_prediction_residual",
        "physical_infeasible_point_count",
        "closed_loop_complete",
        "closed_loop_coverage_fraction",
    )
    lines = [
        f"# Candidate {str(candidate_id).upper()} constraint audit",
        "",
        "The audit uses actual closed-loop quantities after applying only the first QP action.",
        "",
        "## Numerical tolerance",
        "",
        "Small solver residuals are classified against explicit tolerances; a small value such as "
        "0.0154 kW is a numerical tolerance issue, not automatically a physical strategy failure.",
        "",
        "| tolerance | value |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value:.12g} |" for key, value in N6_TOLERANCES.items())
    lines.extend(["", "## Overall raw maxima", "", "| metric | value |", "|---|---:|"])
    for metric in metrics:
        value = summary.get(metric)
        display = "not available" if value is None else str(value)
        lines.append(f"| `{metric}` | {display} |")
    lines.extend(
        [
            "",
            "`physical_infeasible_point_count` counts only residuals beyond the configured physical "
            "tolerances. Solver failures remain separately visible in solver_statistics.csv.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_voyage_plot(frame: pd.DataFrame, destination: Path, *, candidate_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    voyage_id = str(frame.iloc[0]["voyage_id"])
    plotted = frame.loc[frame["success"].fillna(False).astype(bool)] if "success" in frame else frame
    failure_times = (
        pd.to_numeric(
            frame.loc[~frame["success"].fillna(False).astype(bool), "time_s"],
            errors="coerce",
        ).dropna()
        if "success" in frame
        else pd.Series(dtype=float)
    )
    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(plotted["time_s"], plotted["load_actual_kw"], label="load", linewidth=1.0)
    axes[0].plot(plotted["time_s"], plotted["P_fc_actual_kw"], label="fuel cell", linewidth=1.0)
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(loc="best")
    axes[1].plot(plotted["time_s"], plotted["P_batt_actual_kw"], color="tab:orange", linewidth=1.0)
    axes[1].axhline(346.5, color="0.6", linestyle="--", linewidth=0.8)
    axes[1].axhline(-346.5, color="0.6", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Battery (kW)")
    axes[2].plot(plotted["time_s"], plotted["SOC_actual"], color="tab:green", linewidth=1.0)
    axes[2].axhline(0.2, color="0.6", linestyle="--", linewidth=0.8)
    axes[2].axhline(0.8, color="0.6", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("SOC")
    axes[2].set_xlabel("Voyage time (s)")
    for failure_time in failure_times:
        for axis in axes:
            axis.axvline(float(failure_time), color="tab:red", linestyle=":", linewidth=1.0)
    figure.suptitle(f"N=6 candidate {candidate_id}: {voyage_id}")
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def write_candidate_artifacts(
    *,
    candidate_id: str,
    summary: dict[str, Any],
    voyage_metrics: pd.DataFrame,
    solver_statistics: pd.DataFrame,
    controls: pd.DataFrame,
    output_root: str | Path,
    input_path: str | Path,
    make_plots: bool = True,
) -> Path:
    normalized_id = str(candidate_id).upper()
    case_dir = Path(output_root) / f"candidate_{normalized_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    metadata = _candidate_metadata(normalized_id, input_path=Path(input_path))

    (case_dir / "config.json").write_text(
        json.dumps(_json_ready(metadata), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_payload = {"candidate_id": normalized_id, **summary}
    (case_dir / "summary_metrics.json").write_text(
        json.dumps(_json_ready(summary_payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    voyage_metrics.to_csv(case_dir / "voyage_metrics.csv", index=False)
    solver_statistics.to_csv(case_dir / "solver_statistics.csv", index=False)
    (case_dir / "constraint_audit.md").write_text(
        _constraint_audit_markdown(normalized_id, summary_payload),
        encoding="utf-8",
    )

    if make_plots and not controls.empty:
        plot_dir = case_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for voyage_id, frame in controls.groupby("voyage_id", sort=True):
            _write_voyage_plot(
                frame,
                plot_dir / f"{str(voyage_id)}_power_soc.png",
                candidate_id=normalized_id,
            )
    return case_dir


def load_spline_test_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    frame = pd.read_parquet(path)
    required = {"voyage_id", "split", "time_s", "load_total_kw", "dataset_version"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"spline input is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("spline input is empty")
    if set(frame["split"].astype(str).unique()) != {"test"}:
        raise ValueError("N=6 weight selection requires only the test split")
    versions = set(frame["dataset_version"].astype(str).unique())
    if versions != {"cubic_spline_1s_natural_clipped"}:
        raise ValueError(
            "N=6 weight selection requires dataset_version=cubic_spline_1s_natural_clipped"
        )
    frame = frame.copy()
    frame["voyage_id"] = frame["voyage_id"].astype(str)
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
    frame["load_total_kw"] = pd.to_numeric(frame["load_total_kw"], errors="raise")
    if not np.isfinite(frame[["time_s", "load_total_kw"]].to_numpy(dtype=float)).all():
        raise ValueError("spline input time and load values must be finite")
    if (frame["load_total_kw"] < -1.0e-9).any():
        raise ValueError("natural-clipped spline input must not contain negative load")
    frame = frame.sort_values(["voyage_id", "time_s"], kind="stable").reset_index(drop=True)
    for voyage_id, voyage in frame.groupby("voyage_id", sort=True):
        times = voyage["time_s"].to_numpy(dtype=float)
        if len(times) < 2:
            raise ValueError(f"voyage {voyage_id} must contain at least two samples")
        if not np.allclose(np.diff(times), N6_DT_SECONDS, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"voyage {voyage_id} is not strictly spaced at 1 s")
    return frame


def _portable_input_path(input_path: Path) -> Path:
    try:
        return input_path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        return input_path


def run_candidate(
    candidate_id: str,
    *,
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    make_plots: bool = True,
    max_steps_per_voyage: int | None = None,
    expected_voyage_count: int | None = None,
) -> dict[str, Any]:
    normalized_id = str(candidate_id).upper()
    config = candidate_config(normalized_id)
    data = load_spline_test_data(input_path)
    voyage_count = int(data["voyage_id"].nunique())
    if expected_voyage_count is not None and voyage_count != int(expected_voyage_count):
        raise ValueError(
            f"expected {int(expected_voyage_count)} test voyages, found {voyage_count}"
        )
    if expected_voyage_count == len(EXPECTED_TEST_VOYAGES):
        actual_voyages = tuple(sorted(data["voyage_id"].unique()))
        if actual_voyages != EXPECTED_TEST_VOYAGES:
            raise ValueError(
                f"formal run requires voyages {EXPECTED_TEST_VOYAGES}, found {actual_voyages}"
            )

    reference_energy_by_voyage = {
        str(voyage_id): float(
            voyage.iloc[1:]["load_total_kw"].sum() * N6_DT_SECONDS / 3600.0 / 1000.0
        )
        for voyage_id, voyage in data.groupby("voyage_id", sort=True)
    }

    control_frames: list[pd.DataFrame] = []
    solver_frames: list[pd.DataFrame] = []
    for voyage_id, voyage in data.groupby("voyage_id", sort=True):
        controls, solver = run_voyage(
            voyage_id=str(voyage_id),
            loads_kw=voyage["load_total_kw"].to_numpy(dtype=float),
            times_s=voyage["time_s"].to_numpy(dtype=float),
            candidate_id=normalized_id,
            max_steps=max_steps_per_voyage,
            initial_soc=0.55,
        )
        control_frames.append(controls)
        solver_frames.append(solver)

    control_df = pd.concat(control_frames, ignore_index=True)
    solver_df = pd.concat(solver_frames, ignore_index=True)
    summary, voyage_metrics, solver_statistics = build_candidate_metrics(
        control_df,
        solver_df,
        config=config,
    )
    summary = {
        "candidate_id": normalized_id,
        "voyage_count": voyage_count,
        "input_sample_count": int(len(data)),
        "is_partial_debug_run": max_steps_per_voyage is not None,
        "reference_load_energy_mwh": float(sum(reference_energy_by_voyage.values())),
        **summary,
    }
    voyage_metrics.insert(0, "candidate_id", normalized_id)
    voyage_metrics["reference_load_energy_mwh"] = voyage_metrics["voyage_id"].map(
        reference_energy_by_voyage
    )
    solver_statistics.insert(0, "candidate_id", normalized_id)
    case_dir = write_candidate_artifacts(
        candidate_id=normalized_id,
        summary=summary,
        voyage_metrics=voyage_metrics,
        solver_statistics=solver_statistics,
        controls=control_df,
        output_root=output_root,
        input_path=_portable_input_path(Path(input_path)),
        make_plots=make_plots,
    )
    return {
        "candidate_id": normalized_id,
        "summary": summary,
        "voyage_metrics": voyage_metrics,
        "solver_statistics": solver_statistics,
        "case_dir": case_dir,
    }


def _format_report_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return f"{number:.6g}" if np.isfinite(number) else "n/a"
    return str(value)


def _validate_explicit_selection(
    selection: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    status = str(selection.get("status", ""))
    valid_ids = {item["candidate_id"] for item in CANDIDATES}
    by_candidate = {str(item.get("candidate_id", "")).upper(): item for item in summaries}
    if status == "no_candidate_selected":
        if selection.get("selected_candidate") not in {None, ""}:
            raise ValueError("no_candidate_selected decision cannot name a selected candidate")
        if set(by_candidate) != valid_ids:
            raise ValueError("no_candidate_selected decision requires formal results for A, B, C, and D")
        for candidate_id, summary in by_candidate.items():
            if int(summary.get("voyage_count", -1)) != len(EXPECTED_TEST_VOYAGES):
                raise ValueError(f"candidate {candidate_id} must contain all seven formal voyages")
            if bool(summary.get("is_partial_debug_run", True)):
                raise ValueError(f"candidate {candidate_id} cannot be a partial debug run")
        decisions = selection.get("candidate_decisions", {})
        if set(decisions) != valid_ids:
            raise ValueError("no_candidate_selected decision requires a rejection reason for every candidate")
        if not selection.get("selection_reasons"):
            raise ValueError("no_candidate_selected decision requires explicit engineering reasons")
        return
    if status not in {"provisional", "accepted"}:
        raise ValueError("selection status must be provisional, accepted, or no_candidate_selected")
    if set(by_candidate) != valid_ids:
        raise ValueError("provisional/accepted selection requires formal results for A, B, C, and D")
    for candidate_id, candidate_summary in by_candidate.items():
        if int(candidate_summary.get("voyage_count", -1)) != len(EXPECTED_TEST_VOYAGES):
            raise ValueError(f"candidate {candidate_id} must contain all seven formal voyages")
        if bool(candidate_summary.get("is_partial_debug_run", True)):
            raise ValueError(f"candidate {candidate_id} cannot be a partial debug run")
    decisions = selection.get("candidate_decisions", {})
    if set(decisions) != valid_ids:
        raise ValueError("selection requires an engineering decision for every candidate")
    if not selection.get("selection_reasons"):
        raise ValueError("selection requires explicit engineering reasons")
    if not str(selection.get("selection_method", "")).strip():
        raise ValueError("selection requires an explicit manual selection method")
    selected_candidate = str(selection.get("selected_candidate", "")).upper()
    if selected_candidate not in valid_ids:
        raise ValueError(f"selected_candidate must be one of {sorted(valid_ids)}")
    if selected_candidate not in by_candidate:
        raise ValueError(f"selected candidate {selected_candidate} has no completed summary")
    summary = by_candidate[selected_candidate]
    if int(summary.get("voyage_count", -1)) != len(EXPECTED_TEST_VOYAGES):
        raise ValueError("selected candidate must contain all seven formal voyages")
    if bool(summary.get("is_partial_debug_run", True)):
        raise ValueError("selected candidate cannot be a partial debug run")
    if not bool(summary.get("closed_loop_complete", False)):
        raise ValueError("selected candidate must be closed-loop complete")
    if int(summary.get("solver_failure_count", 0)) != 0:
        raise ValueError("selected candidate must have zero final solver failures")
    if int(summary.get("physical_infeasible_point_count", -1)) != 0:
        raise ValueError("selected candidate must have zero physical infeasible points")
    if not bool(summary.get("aggregate_metrics_comparable", False)):
        raise ValueError("selected candidate must have complete comparable aggregate metrics")
    soc_min = float(summary.get("soc_min", float("nan")))
    soc_max = float(summary.get("soc_max", float("nan")))
    if not np.isfinite(soc_min) or not np.isfinite(soc_max):
        raise ValueError("selected candidate must report finite SOC extrema")
    if soc_min < 0.2 - N6_TOLERANCES["soc"] or soc_max > 0.8 + N6_TOLERANCES["soc"]:
        raise ValueError("selected candidate SOC must remain within physical bounds")
    worst_soc_change = float(summary.get("worst_voyage_soc_net_change", float("nan")))
    if not np.isfinite(worst_soc_change) or worst_soc_change < -0.03:
        raise ValueError("selected candidate violates the worst-voyage SOC net-change gate")
    if status == "accepted":
        if not bool(selection.get("engineering_review_complete", False)):
            raise ValueError("accepted selection requires a completed engineering review")
        verification = selection.get("verification", {})
        required_verification = {
            "compileall_passed",
            "full_test_suite_passed",
            "diff_check_passed",
            "report_review_complete",
        }
        if not isinstance(verification, dict) or not all(
            verification.get(key) is True for key in required_verification
        ):
            raise ValueError(
                "accepted selection requires verification of compileall, full tests, diff check, and report review"
            )
        solve_time_max = float(summary.get("solve_time_ms_max", float("nan")))
        if not np.isfinite(solve_time_max) or solve_time_max >= 1000.0:
            raise ValueError("accepted selection must satisfy the 1 s solver performance gate")


def write_combined_reports(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    reports_dir: str | Path = DEFAULT_REPORTS_DIR,
    selection: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    output_path = Path(output_root)
    summaries: list[dict[str, Any]] = []
    candidate_order = {item["candidate_id"]: index for index, item in enumerate(CANDIDATES)}
    for summary_path in output_path.glob("candidate_*/summary_metrics.json"):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        candidate_id = str(payload.get("candidate_id", summary_path.parent.name.removeprefix("candidate_"))).upper()
        payload["candidate_id"] = candidate_id
        summaries.append(payload)
    summaries.sort(key=lambda item: candidate_order.get(str(item["candidate_id"]), 999))
    if not summaries:
        raise ValueError(f"no candidate summary_metrics.json found under {output_path}")

    if selection:
        _validate_explicit_selection(selection, summaries)

    report_path = Path(reports_dir)
    report_path.mkdir(parents=True, exist_ok=True)
    table_path = report_path / "mpc_1s_n6_weight_selection_table.csv"
    pd.DataFrame(summaries).to_csv(table_path, index=False)

    selected = selection or {}
    selection_status = str(selected.get("status", "pending_manual_review"))
    selected_candidate = selected.get("selected_candidate")
    selection_method = str(selected.get("selection_method", "manual engineering review pending"))
    key_columns = (
        "candidate_id",
        "closed_loop_complete",
        "closed_loop_coverage_fraction",
        "solver_success_rate",
        "physical_infeasible_point_count",
        "final_soc",
        "worst_voyage_soc_net_change",
        "hydrogen_total_kg",
        "battery_throughput_kwh",
        "fc_above_load_fraction",
        "fc_surplus_energy_kwh",
        "solve_time_ms_p99",
    )
    lines = [
        "# 1 s N=6 OSQP-QP MPC fixed-weight selection",
        "",
        "## Experiment boundary",
        "",
        "This is an offline ideal-foresight experiment on the natural-clipped cubic-spline 1 s "
        "reconstruction; it is not measured 1 s data and it does not use LSTM predictions.",
        "At each second, the future six true samples (`t+1..t+6`) form the N=6 prediction window. "
        "The QP has a six-step control horizon, but the closed loop applies the first action only, "
        "then rolls forward by one second.",
        "N=60 remains a historical solver/performance benchmark and was not searched in this task.",
        "",
        "## Candidate summary",
        "",
        "| " + " | ".join(key_columns) + " |",
        "|" + "|".join("---:" for _ in key_columns) + "|",
    ]
    for summary in summaries:
        lines.append("| " + " | ".join(_format_report_value(summary.get(key)) for key in key_columns) + " |")
    lines.extend(
        [
            "",
            "All 24 requested aggregate metrics are preserved in the companion CSV and in each "
            "candidate's summary/voyage files.",
            "For a candidate that terminates on solver infeasibility, energy/economy values describe "
            "only the successfully applied prefix and are not comparable for selection.",
            "",
            "## Engineering selection",
            "",
            f"- Status: `{selection_status}`",
            (
                f"- Selected candidate: `{selected_candidate}`"
                if selected_candidate
                else (
                    "- Selected candidate: none"
                    if selection_status == "no_candidate_selected"
                    else "- Selected candidate: pending"
                )
            ),
            f"- Method: {selection_method}",
            "- Priority order: physical feasibility, long-term SOC, power allocation, economy/device use, solver performance.",
            "- The legacy automated \"least-bad\" conclusion is not used in place of engineering judgment.",
        ]
    )
    reasons = selected.get("selection_reasons", [])
    if reasons:
        lines.extend(["", "Selection reasons:"])
        lines.extend(f"- {reason}" for reason in reasons)
    decisions = selected.get("candidate_decisions", {})
    if decisions:
        lines.extend(["", "Candidate decisions:"])
        for candidate_id in sorted(decisions, key=lambda item: candidate_order.get(str(item), 999)):
            lines.append(f"- **{candidate_id}**: {decisions[candidate_id]}")
    lines.extend(
        [
            "",
            "## Numerical interpretation",
            "",
            "Residuals are interpreted using the tolerances recorded in each config.json. A small "
            "residual such as 0.0154 kW is a numerical tolerance observation, not by itself a physical "
            "strategy failure.",
            "",
        ]
    )
    markdown_path = report_path / "mpc_1s_n6_weight_selection_summary.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, table_path


def _load_selection(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_selection_path(explicit_path: Path | None, output_root: Path) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    adjacent_manual_decision = output_root / "manual_decision.json"
    if adjacent_manual_decision.exists():
        return adjacent_manual_decision
    if output_root.resolve() == DEFAULT_OUTPUT_ROOT.resolve() and DEFAULT_SELECTION_CONFIG.exists():
        return DEFAULT_SELECTION_CONFIG
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one N=6 ideal-foresight OSQP-QP fixed-weight candidate."
    )
    parser.add_argument("--candidate", choices=[item["candidate_id"] for item in CANDIDATES])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--selection-config", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-steps-per-voyage", type=int)
    parser.add_argument("--expected-voyages", type=int, default=7)
    args = parser.parse_args(argv)

    if not args.report_only:
        if args.candidate is None:
            parser.error("--candidate is required unless --report-only is used")
        result = run_candidate(
            args.candidate,
            input_path=args.input,
            output_root=args.output_root,
            make_plots=not args.no_plots,
            max_steps_per_voyage=args.max_steps_per_voyage,
            expected_voyage_count=args.expected_voyages,
        )
        print(json.dumps(_json_ready(result["summary"]), ensure_ascii=False))

    markdown_path, table_path = write_combined_reports(
        output_root=args.output_root,
        reports_dir=args.reports_dir,
        selection=_load_selection(_resolve_selection_path(args.selection_config, args.output_root)),
    )
    print(f"report={markdown_path}")
    print(f"table={table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
