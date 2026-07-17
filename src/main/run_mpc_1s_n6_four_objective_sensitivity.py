from __future__ import annotations

import time
from dataclasses import dataclass
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
)
from mpc_solvers.mpc_qp_formulation import (
    QpMpcConfig,
    QpProblem,
    build_qp_problem,
    resolved_ramp_kw_per_step,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_VARIANT = "n6_h2_batt_soc_fcvar_normalized_v1"
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "outputs"
    / "mpc_solver_benchmark_1s"
    / "data"
    / "test_voyages_spline_1s.parquet"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mpc_1s_n6_four_objective_sensitivity"
DEFAULT_SUMMARY_REPORT = (
    REPO_ROOT / "reports" / "mpc_1s_n6_four_objective_sensitivity_summary.md"
)
DEFAULT_TABLE_REPORT = (
    REPO_ROOT / "reports" / "mpc_1s_n6_four_objective_sensitivity_table.csv"
)
EXPECTED_TEST_VOYAGES: tuple[str, ...] = tuple(
    f"voyage_{index:03d}" for index in range(60, 67)
)
N6_HORIZON = 6
N6_DT_SECONDS = 1.0
FIXED_SOC_REFERENCE = 0.55
N6_OSQP_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "polishing": True,
    "warm_starting": True,
    "eps_abs": 1.0e-5,
    "eps_rel": 1.0e-5,
    "max_iter": 20000,
    "adaptive_rho": True,
}
N6_TOLERANCES: dict[str, float] = {
    "actual_balance_kw": 0.01,
    "qp_balance_kw": 0.1,
    "power_bound_kw": 0.1,
    "ramp_kw": 0.1,
    "soc": 1.0e-6,
    "soc_prediction": 1.0e-5,
    "fc_above_load_kw": 1.0e-6,
    "near_limit_kw": 1.0,
}

WEIGHT_NAMES: tuple[str, ...] = ("q_h2", "q_batt", "q_soc", "q_fc_var")
WEIGHT_VALUES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class SensitivityCase:
    config_id: str
    varied_weight: str | None
    weight_value: float
    q_h2: float
    q_batt: float
    q_soc: float
    q_fc_var: float


def _weight_token(value: float) -> str:
    return {0.25: "0p25", 0.5: "0p5", 2.0: "2", 4.0: "4"}[float(value)]


def build_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    cases = [
        SensitivityCase(
            config_id="baseline_1_1_1_1",
            varied_weight=None,
            weight_value=1.0,
            q_h2=1.0,
            q_batt=1.0,
            q_soc=1.0,
            q_fc_var=1.0,
        )
    ]
    for weight_name in WEIGHT_NAMES:
        for value in WEIGHT_VALUES:
            if value == 1.0:
                continue
            weights = dict.fromkeys(WEIGHT_NAMES, 1.0)
            weights[weight_name] = float(value)
            cases.append(
                SensitivityCase(
                    config_id=f"{weight_name}_{_weight_token(value)}",
                    varied_weight=weight_name,
                    weight_value=float(value),
                    q_h2=weights["q_h2"],
                    q_batt=weights["q_batt"],
                    q_soc=weights["q_soc"],
                    q_fc_var=weights["q_fc_var"],
                )
            )
    weight_tuples = {
        (case.q_h2, case.q_batt, case.q_soc, case.q_fc_var) for case in cases
    }
    if len(cases) != 17 or len(weight_tuples) != 17:
        raise RuntimeError("sensitivity case construction must yield exactly 17 unique cases")
    return tuple(cases)


def four_objective_config(case: SensitivityCase) -> QpMpcConfig:
    return QpMpcConfig(
        horizon=N6_HORIZON,
        dt_seconds=N6_DT_SECONDS,
        battery_capacity_kwh=693.0,
        battery_charge_max_kw=346.5,
        battery_discharge_max_kw=346.5,
        battery_power_ref_kw=346.5,
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=560.0,
        fuel_cell_ramp_rate_kw_per_s=48.0,
        fuel_cell_ramp_kw=None,
        soc_min=0.2,
        soc_max=0.8,
        soc_band=0.05,
        objective_variant=OBJECTIVE_VARIANT,
        q_h2=float(case.q_h2),
        q_fc_var=float(case.q_fc_var),
        q_soc=float(case.q_soc),
        q_batt=float(case.q_batt),
        q_ramp=0.0,
        q_terminal_soc=0.0,
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

    fuel_cell_scale = 560.0
    battery_scale = 346.5
    soc_scale = 0.05
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
            np.full(horizon + 1, FIXED_SOC_REFERENCE),
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
        0.5 * variable_offset @ problem.P @ variable_offset
        + problem.q @ variable_offset
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
        QpProblem(
            P=scaled_p,
            q=scaled_q,
            A=scaled_a,
            l=scaled_l,
            u=scaled_u,
            metadata=metadata,
        ),
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


def scaled_linear_for_previous_fc(
    base_scaled_linear: np.ndarray,
    *,
    config: QpMpcConfig,
    transform: N6QpTransform,
    base_previous_fc_kw: float,
    previous_fc_kw: float,
) -> np.ndarray:
    if str(config.objective_variant) != OBJECTIVE_VARIANT:
        raise ValueError(
            f"rolling linear refresh requires objective_variant={OBJECTIVE_VARIANT}"
        )
    linear = np.asarray(base_scaled_linear, dtype=float).copy()
    ramp_kw_per_step = float(resolved_ramp_kw_per_step(config))
    delta_physical = (
        -2.0
        * float(config.q_fc_var)
        * (float(previous_fc_kw) - float(base_previous_fc_kw))
        / ramp_kw_per_step**2
    )
    linear[0] += float(transform.variable_scale[0]) * delta_physical
    return linear


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
        p_batt_actual
        * float(config.dt_seconds)
        / 3600.0
        / float(config.battery_capacity_kwh)
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
    case: SensitivityCase,
    config: QpMpcConfig,
    max_steps: int | None = None,
    initial_soc: float = FIXED_SOC_REFERENCE,
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
    if max_steps is not None and (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or int(max_steps) < 1
    ):
        raise ValueError(
            f"max_steps must be a positive integer or None, got {max_steps!r}"
        )
    try:
        initial_soc_value = float(initial_soc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"initial_soc must be a finite number, got {initial_soc!r}") from exc
    if (
        not np.isfinite(initial_soc_value)
        or initial_soc_value < float(config.soc_min)
        or initial_soc_value > float(config.soc_max)
    ):
        raise ValueError(
            "initial_soc must be finite and within "
            f"[{config.soc_min}, {config.soc_max}], got {initial_soc!r}"
        )

    osqp_module, osqp_error = _try_import_osqp()
    if osqp_module is None:
        raise RuntimeError(f"Cannot import osqp: {osqp_error}")

    current_soc = initial_soc_value
    prev_fc_actual = float(
        np.clip(loads[0], config.fuel_cell_min_kw, config.fuel_cell_max_kw)
    )
    setup_horizon = ideal_future_window(loads, decision_index=0)
    setup_problem = build_qp_problem(
        config,
        load_forecast_kw=setup_horizon,
        current_soc=current_soc,
        prev_fc_kw=prev_fc_actual,
        soc_reference=FIXED_SOC_REFERENCE,
        include_diagnostics=False,
    )
    scaled_setup_problem, transform = scale_n6_qp_problem(
        setup_problem,
        config=config,
    )
    solver = _setup_n6_osqp_solver(osqp_module, scaled_setup_problem)
    setup_previous_fc = float(prev_fc_actual)
    base_scaled_linear = scaled_setup_problem.q.copy()

    decision_count = len(loads) - 1
    if max_steps is not None:
        decision_count = min(decision_count, int(max_steps))
    control_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    h2_reference = physical_h2_kg_step(config, float(config.fuel_cell_max_kw))

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
        scaled_linear = scaled_linear_for_previous_fc(
            base_scaled_linear,
            config=config,
            transform=transform,
            base_previous_fc_kw=setup_previous_fc,
            previous_fc_kw=prev_fc_before,
        )
        result, solve_ms = _solve_with_persistent_osqp(
            solver,
            lower=scaled_lower,
            upper=scaled_upper,
            linear=scaled_linear,
        )
        initial_status = str(result.info.status)
        initial_status_lower = initial_status.lower()
        cold_restart_used = bool(
            "maximum iterations" in initial_status_lower
            or "max_iter" in initial_status_lower
        )
        attempt_count = 1
        if cold_restart_used:
            setup_start = time.perf_counter()
            recovery_problem = build_qp_problem(
                config,
                load_forecast_kw=load_horizon,
                current_soc=soc_before,
                prev_fc_kw=prev_fc_before,
                soc_reference=FIXED_SOC_REFERENCE,
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
                linear=scaled_recovery_problem.q,
            )
            solve_ms = (
                float(solve_ms)
                + float(recovery_setup_ms)
                + float(recovery_solve_ms)
            )
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
            abs(
                float(applied["P_fc_plan_kw"])
                + p_batt_plan
                - float(load_horizon[0])
            )
            if success
            else float("nan")
        )
        actual_balance_residual = (
            abs(p_fc + p_batt_actual - load_actual) if success else float("nan")
        )
        fc_bound_residual = (
            max(
                0.0,
                float(config.fuel_cell_min_kw) - p_fc,
                p_fc - float(config.fuel_cell_max_kw),
            )
            if success
            else float("nan")
        )
        battery_bound_residual = (
            max(
                0.0,
                -float(config.battery_charge_max_kw) - p_batt_actual,
                p_batt_actual - float(config.battery_discharge_max_kw),
            )
            if success
            else float("nan")
        )
        ramp_residual = (
            max(0.0, abs(fc_delta) - float(resolved_ramp_kw_per_step(config)))
            if success
            else float("nan")
        )
        soc_bound_residual = (
            max(
                0.0,
                float(config.soc_min) - soc_actual,
                soc_actual - float(config.soc_max),
            )
            if success
            else float("nan")
        )
        soc_prediction_residual = (
            abs(float(applied["SOC_predicted"]) - soc_actual)
            if success
            else float("nan")
        )
        if success:
            raw_h2 = physical_h2_kg_step(config, p_fc)
            raw_batt = p_batt_actual**2
            raw_soc = (soc_actual - FIXED_SOC_REFERENCE) ** 2
            raw_fc_var = fc_delta**2
            j_h2 = raw_h2 / h2_reference
            j_batt = raw_batt / float(config.battery_power_ref_kw) ** 2
            j_soc = raw_soc / float(config.soc_band) ** 2
            j_fc_var = raw_fc_var / float(resolved_ramp_kw_per_step(config)) ** 2
            weighted_h2 = float(case.q_h2) * j_h2
            weighted_batt = float(case.q_batt) * j_batt
            weighted_soc = float(case.q_soc) * j_soc
            weighted_fc_var = float(case.q_fc_var) * j_fc_var
            objective_steps = {
                "h2_kg_step": raw_h2,
                "p_batt_sq_kw2_step": raw_batt,
                "soc_error_sq_step": raw_soc,
                "fc_delta_sq_kw2_step": raw_fc_var,
                "J_h2_norm_step": j_h2,
                "J_batt_norm_step": j_batt,
                "J_soc_norm_step": j_soc,
                "J_fc_var_norm_step": j_fc_var,
                "weighted_h2_contribution_step": weighted_h2,
                "weighted_batt_contribution_step": weighted_batt,
                "weighted_soc_contribution_step": weighted_soc,
                "weighted_fc_var_contribution_step": weighted_fc_var,
                "total_weighted_objective_step": (
                    weighted_h2 + weighted_batt + weighted_soc + weighted_fc_var
                ),
            }
        else:
            objective_steps = {
                name: float("nan")
                for name in (
                    "h2_kg_step",
                    "p_batt_sq_kw2_step",
                    "soc_error_sq_step",
                    "fc_delta_sq_kw2_step",
                    "J_h2_norm_step",
                    "J_batt_norm_step",
                    "J_soc_norm_step",
                    "J_fc_var_norm_step",
                    "weighted_h2_contribution_step",
                    "weighted_batt_contribution_step",
                    "weighted_soc_contribution_step",
                    "weighted_fc_var_contribution_step",
                    "total_weighted_objective_step",
                )
            }

        control_rows.append(
            {
                "config_id": case.config_id,
                "voyage_id": str(voyage_id),
                "voyage_expected_steps": int(decision_count),
                "decision_index": int(decision_index),
                "execution_index": int(execution_index),
                "decision_time_s": float(times[decision_index]),
                "time_s": float(times[execution_index]),
                "load_actual_kw": load_actual,
                "SOC_before": soc_before,
                "prev_fc_actual_kw": prev_fc_before,
                **{
                    f"load_h{index + 1}_kw": float(value)
                    for index, value in enumerate(load_horizon)
                },
                **applied,
                "fc_delta_actual_kw": fc_delta,
                "plan_balance_residual_kw": plan_balance_residual,
                "actual_balance_residual_kw": actual_balance_residual,
                "fc_bound_residual_kw": fc_bound_residual,
                "battery_bound_residual_kw": battery_bound_residual,
                "ramp_residual_kw": ramp_residual,
                "soc_bound_residual": soc_bound_residual,
                "soc_prediction_residual": soc_prediction_residual,
                **objective_steps,
                "success": success,
                "status": status,
            }
        )
        solver_rows.append(
            {
                "config_id": case.config_id,
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

    controls = pd.DataFrame(control_rows)
    if not controls.empty:
        successful = controls["success"].fillna(False).astype(bool)
        for step_column, cumulative_column in (
            ("J_h2_norm_step", "cum_J_h2_norm"),
            ("J_batt_norm_step", "cum_J_batt_norm"),
            ("J_soc_norm_step", "cum_J_soc_norm"),
            ("J_fc_var_norm_step", "cum_J_fc_var_norm"),
        ):
            controls[cumulative_column] = float("nan")
            controls.loc[successful, cumulative_column] = (
                controls.loc[successful, step_column].cumsum()
            )
    return controls, pd.DataFrame(solver_rows)


def _normalized_success_flags(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "1.0": True,
                "false": False,
                "0": False,
                "0.0": False,
            }
        )
        .astype("boolean")
    )


def _voyage_metric_rows_aligned(
    controls: pd.DataFrame,
    solver_rows: pd.DataFrame,
) -> bool:
    required = {"voyage_id", "decision_index", "execution_index", "success"}
    if (
        len(controls) != len(solver_rows)
        or not required.issubset(controls.columns)
        or not required.issubset(solver_rows.columns)
    ):
        return False

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "voyage_id": frame["voyage_id"].astype("string").str.strip(),
                "decision_index": pd.to_numeric(
                    frame["decision_index"], errors="coerce"
                ).astype("Float64"),
                "execution_index": pd.to_numeric(
                    frame["execution_index"], errors="coerce"
                ).astype("Float64"),
                "success": _normalized_success_flags(frame["success"]),
            }
        ).reset_index(drop=True)

    control_key = normalized(controls)
    solver_key = normalized(solver_rows)
    if (
        control_key.isna().any().any()
        or solver_key.isna().any().any()
        or control_key["voyage_id"].eq("").any()
        or solver_key["voyage_id"].eq("").any()
        or not np.isfinite(
            control_key[["decision_index", "execution_index"]].to_numpy(dtype=float)
        ).all()
        or not np.isfinite(
            solver_key[["decision_index", "execution_index"]].to_numpy(dtype=float)
        ).all()
    ):
        return False
    return bool(control_key.equals(solver_key))


def build_voyage_metrics(
    controls: pd.DataFrame,
    solver_rows: pd.DataFrame,
    *,
    case: SensitivityCase,
    config: QpMpcConfig,
) -> dict[str, Any]:
    success = _normalized_success_flags(controls["success"]).fillna(False).astype(bool)
    applied = controls.loc[success]
    solver_success = (
        _normalized_success_flags(solver_rows["success"]).fillna(False).astype(bool)
    )
    failed_solver = solver_rows.loc[~solver_success]
    status = solver_rows["status"].fillna("").astype(str)
    solve_ms = pd.to_numeric(solver_rows["solve_ms"], errors="coerce").dropna()
    successful_soc = pd.to_numeric(applied["SOC_actual"], errors="coerce").dropna()
    successful_fc = pd.to_numeric(applied["P_fc_actual_kw"], errors="coerce").dropna()
    successful_batt = pd.to_numeric(
        applied["P_batt_actual_kw"], errors="coerce"
    ).dropna()
    successful_fc_delta = pd.to_numeric(
        applied["fc_delta_actual_kw"], errors="coerce"
    ).dropna()
    successful_balance = pd.to_numeric(
        applied["actual_balance_residual_kw"], errors="coerce"
    ).dropna()

    expected_count = int(pd.to_numeric(controls["voyage_expected_steps"]).max())
    completed = bool(
        len(controls) == expected_count
        and len(solver_rows) == expected_count
        and _voyage_metric_rows_aligned(controls, solver_rows)
        and solver_success.all()
    )
    initial_soc = float(controls.iloc[0]["SOC_before"])
    final_soc = float(applied.iloc[-1]["SOC_actual"]) if len(applied) else initial_soc
    first_failure_time_s = (
        float(failed_solver.iloc[0]["time_s"]) if len(failed_solver) else float("nan")
    )
    first_failure_status = (
        str(failed_solver.iloc[0]["status"]) if len(failed_solver) else ""
    )
    j_h2_norm = float(applied["J_h2_norm_step"].sum())
    j_batt_norm = float(applied["J_batt_norm_step"].sum())
    j_soc_norm = float(applied["J_soc_norm_step"].sum())
    j_fc_var_norm = float(applied["J_fc_var_norm_step"].sum())
    weighted_h2 = float(case.q_h2) * j_h2_norm
    weighted_batt = float(case.q_batt) * j_batt_norm
    weighted_soc = float(case.q_soc) * j_soc_norm
    weighted_fc_var = float(case.q_fc_var) * j_fc_var_norm

    return {
        "config_id": case.config_id,
        "q_h2": float(case.q_h2),
        "q_batt": float(case.q_batt),
        "q_soc": float(case.q_soc),
        "q_fc_var": float(case.q_fc_var),
        "voyage_id": str(controls.iloc[0]["voyage_id"]),
        "completed": completed,
        "solver_failure_count": int((~solver_success).sum()),
        "primal_infeasible_count": int(
            status.str.contains("primal infeasible", case=False, regex=False).sum()
        ),
        "max_iter_count": int(solver_rows["max_iter_reached"].fillna(False).sum()),
        "first_failure_time_s": first_failure_time_s,
        "mean_solve_time_ms": float(solve_ms.mean()),
        "p95_solve_time_ms": float(solve_ms.quantile(0.95)),
        "max_solve_time_ms": float(solve_ms.max()),
        "initial_soc": initial_soc,
        "final_soc": final_soc,
        "delta_soc": final_soc - initial_soc,
        "min_soc": (
            min(initial_soc, float(successful_soc.min()))
            if len(successful_soc)
            else initial_soc
        ),
        "max_soc": (
            max(initial_soc, float(successful_soc.max()))
            if len(successful_soc)
            else initial_soc
        ),
        "max_power_balance_residual_kw": (
            float(successful_balance.max()) if len(successful_balance) else float("nan")
        ),
        "max_fc_ramp_kw_per_step": (
            float(successful_fc_delta.abs().max())
            if len(successful_fc_delta)
            else float("nan")
        ),
        "max_fc_kw": float(successful_fc.max()) if len(successful_fc) else float("nan"),
        "min_fc_kw": float(successful_fc.min()) if len(successful_fc) else float("nan"),
        "max_batt_discharge_kw": (
            float(successful_batt.clip(lower=0.0).max())
            if len(successful_batt)
            else float("nan")
        ),
        "max_batt_charge_kw": (
            float((-successful_batt.clip(upper=0.0)).max())
            if len(successful_batt)
            else float("nan")
        ),
        "total_h2_kg": float(applied["h2_kg_step"].sum()),
        "sum_p_batt_sq_kw2": float(applied["p_batt_sq_kw2_step"].sum()),
        "sum_soc_error_sq": float(applied["soc_error_sq_step"].sum()),
        "sum_fc_delta_sq_kw2": float(applied["fc_delta_sq_kw2_step"].sum()),
        "J_h2_norm": j_h2_norm,
        "J_batt_norm": j_batt_norm,
        "J_soc_norm": j_soc_norm,
        "J_fc_var_norm": j_fc_var_norm,
        "weighted_h2_contribution": weighted_h2,
        "weighted_batt_contribution": weighted_batt,
        "weighted_soc_contribution": weighted_soc,
        "weighted_fc_var_contribution": weighted_fc_var,
        "total_weighted_objective": (
            weighted_h2 + weighted_batt + weighted_soc + weighted_fc_var
        ),
        "expected_step_count": expected_count,
        "attempted_step_count": int(len(solver_rows)),
        "applied_step_count": int(success.sum()),
        "first_failure_status": first_failure_status,
        "metrics_comparable": completed,
    }


def build_configuration_summary(
    voyage_metrics: pd.DataFrame,
    solver_rows: pd.DataFrame,
    *,
    case: SensitivityCase,
) -> dict[str, Any]:
    completed = voyage_metrics["completed"].astype(bool)
    solve_ms = pd.to_numeric(solver_rows["solve_ms"], errors="coerce").dropna()
    sum_columns = [
        "solver_failure_count",
        "primal_infeasible_count",
        "max_iter_count",
        "total_h2_kg",
        "sum_p_batt_sq_kw2",
        "sum_soc_error_sq",
        "sum_fc_delta_sq_kw2",
        "J_h2_norm",
        "J_batt_norm",
        "J_soc_norm",
        "J_fc_var_norm",
        "weighted_h2_contribution",
        "weighted_batt_contribution",
        "weighted_soc_contribution",
        "weighted_fc_var_contribution",
        "total_weighted_objective",
    ]
    summary: dict[str, Any] = {
        "config_id": case.config_id,
        "varied_weight": case.varied_weight or "baseline",
        "weight_value": float(case.weight_value),
        "q_h2": float(case.q_h2),
        "q_batt": float(case.q_batt),
        "q_soc": float(case.q_soc),
        "q_fc_var": float(case.q_fc_var),
        "voyage_count": int(len(voyage_metrics)),
        "completed_voyage_count": int(completed.sum()),
        "completion_rate": float(completed.mean()),
        "initial_soc": float(voyage_metrics["initial_soc"].mean()),
        "final_soc_mean": float(voyage_metrics["final_soc"].mean()),
        "delta_soc_mean": float(voyage_metrics["delta_soc"].mean()),
        "min_soc": float(voyage_metrics["min_soc"].min()),
        "max_soc": float(voyage_metrics["max_soc"].max()),
        "max_power_balance_residual_kw": float(
            voyage_metrics["max_power_balance_residual_kw"].max()
        ),
        "max_fc_ramp_kw_per_step": float(
            voyage_metrics["max_fc_ramp_kw_per_step"].max()
        ),
        "max_fc_kw": float(voyage_metrics["max_fc_kw"].max()),
        "min_fc_kw": float(voyage_metrics["min_fc_kw"].min()),
        "max_batt_discharge_kw": float(
            voyage_metrics["max_batt_discharge_kw"].max()
        ),
        "max_batt_charge_kw": float(voyage_metrics["max_batt_charge_kw"].max()),
        "mean_solve_time_ms": float(solve_ms.mean()),
        "p95_solve_time_ms": float(solve_ms.quantile(0.95)),
        "max_solve_time_ms": float(solve_ms.max()),
        "metrics_comparable": bool(completed.all()),
    }
    summary.update(
        {name: float(pd.to_numeric(voyage_metrics[name]).sum()) for name in sum_columns}
    )
    for name in (
        "solver_failure_count",
        "primal_infeasible_count",
        "max_iter_count",
    ):
        summary[name] = int(summary[name])
    return summary


def evaluate_configuration(
    case: SensitivityCase,
    *,
    data: pd.DataFrame,
) -> dict[str, Any]:
    config = four_objective_config(case)
    control_frames: list[pd.DataFrame] = []
    solver_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for voyage_id, voyage in data.groupby("voyage_id", sort=True):
        controls, solver_rows = run_voyage(
            voyage_id=str(voyage_id),
            loads_kw=voyage["load_total_kw"].to_numpy(dtype=float),
            times_s=voyage["time_s"].to_numpy(dtype=float),
            case=case,
            config=config,
        )
        control_frames.append(controls)
        solver_frames.append(solver_rows)
        metric_rows.append(
            build_voyage_metrics(controls, solver_rows, case=case, config=config)
        )
    controls = pd.concat(control_frames, ignore_index=True)
    solver_rows = pd.concat(solver_frames, ignore_index=True)
    voyage_metrics = pd.DataFrame(metric_rows)
    return {
        "case": case,
        "config": config,
        "controls": controls,
        "solver_rows": solver_rows,
        "voyage_metrics": voyage_metrics,
        "summary": build_configuration_summary(
            voyage_metrics,
            solver_rows,
            case=case,
        ),
    }


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
    return int(
        temporary.groupby("voyage_id", sort=False)["expected"].max().fillna(0).sum()
    )


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
        raise ValueError("N=6 sensitivity requires only the test split")
    versions = set(frame["dataset_version"].astype(str).unique())
    if versions != {"cubic_spline_1s_natural_clipped"}:
        raise ValueError(
            "N=6 sensitivity requires "
            "dataset_version=cubic_spline_1s_natural_clipped"
        )
    frame = frame.copy()
    frame["voyage_id"] = frame["voyage_id"].astype(str)
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
    frame["load_total_kw"] = pd.to_numeric(frame["load_total_kw"], errors="raise")
    if not np.isfinite(frame[["time_s", "load_total_kw"]].to_numpy(dtype=float)).all():
        raise ValueError("spline input time and load values must be finite")
    if (frame["load_total_kw"] < -1.0e-9).any():
        raise ValueError("natural-clipped spline input must not contain negative load")
    frame = frame.sort_values(["voyage_id", "time_s"], kind="stable").reset_index(
        drop=True
    )
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
