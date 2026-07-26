from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from mpc_solvers.osqp_runtime import (
    _qp_bounds_for_step,
    _solve_with_persistent_osqp,
    _try_import_osqp,
)
from mpc_solvers.mpc_qp_formulation import (
    QpMpcConfig,
    QpProblem,
    build_qp_problem,
    h2_quadratic_kg_step_coefficients,
    resolved_ramp_kw_per_step,
)
from mpc_solvers.n6_qp_scaling import (
    N6_OSQP_SETTINGS,
    N6QpTransform,
    _setup_n6_osqp_solver,
    scale_n6_qp_problem,
    scaled_linear_for_previous_fc,
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "mpc_1s_n6_candidate_C"
DEFAULT_SPLIT_JSON = REPO_ROOT / "outputs" / "config" / "voyage_split_total_load_721.json"


def _load_expected_test_voyages(split_path: Path = DEFAULT_SPLIT_JSON) -> tuple[str, ...]:
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    values = payload.get("test_voyages")
    if not isinstance(values, list) or not values:
        raise ValueError(f"active split has no test_voyages list: {split_path}")
    voyages = tuple(str(value) for value in values)
    if len(voyages) != len(set(voyages)):
        raise ValueError("active test voyage list contains duplicates")
    excluded = {str(value) for value in payload.get("excluded_voyages", [])}
    if excluded.intersection(voyages):
        raise ValueError("active test voyage list contains an excluded voyage")
    return voyages


EXPECTED_TEST_VOYAGES: tuple[str, ...] = _load_expected_test_voyages()
N6_HORIZON = 6
N6_DT_SECONDS = 1.0
FIXED_SOC_REFERENCE = 0.55

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
N6_STATE_COMMIT_TOLERANCES: dict[str, float] = {
    "actual_balance_kw": 0.01,
    "qp_balance_kw": 0.1,
    "power_bound_kw": 0.1,
    "ramp_kw": 0.1,
    "soc": 1.0e-5,
    "soc_prediction": 1.0e-5,
}

WEIGHT_NAMES: tuple[str, ...] = ("q_h2", "q_batt", "q_soc", "q_fc_var")


@dataclass(frozen=True)
class SensitivityCase:
    config_id: str
    varied_weight: str | None
    weight_value: float
    q_h2: float
    q_batt: float
    q_soc: float
    q_fc_var: float


def build_sensitivity_cases() -> tuple[SensitivityCase, ...]:
    return (
        SensitivityCase(
            config_id="candidate_C_h2_0p25_batt_0p4_soc_12_fcvar_20",
            varied_weight=None,
            weight_value=1.0,
            q_h2=0.25,
            q_batt=0.4,
            q_soc=12.0,
            q_fc_var=20.0,
        ),
    )


def four_objective_config(case: SensitivityCase) -> QpMpcConfig:
    return QpMpcConfig(
        horizon=N6_HORIZON,
        dt_seconds=N6_DT_SECONDS,
        battery_capacity_kwh=624.0,
        battery_charge_max_kw=624.0,
        battery_discharge_max_kw=1248.0,
        battery_power_ref_kw=624.0,
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





def physical_h2_kg_step(config: QpMpcConfig, p_fc_kw: float) -> float:
    quad, linear, _, _ = h2_quadratic_kg_step_coefficients(config)
    p = float(p_fc_kw)
    return float(quad * p * p + linear * p)


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

        solver_status = str(result.info.status)
        solver_status_lower = solver_status.lower()
        solver_claimed_success = bool(
            solver_status_lower.startswith("solved") and result.x is not None
        )
        solved_inaccurate = bool("solved inaccurate" in solver_status_lower)
        max_iter_reached = bool(
            cold_restart_used
            or "maximum iterations" in solver_status_lower
            or "max_iter" in solver_status_lower
        )
        rejection_reason = ""
        candidate_applied: dict[str, float] | None = None
        if solver_claimed_success:
            solution = np.asarray(result.x, dtype=float).reshape(-1)
            if len(solution) != len(transform.variable_scale):
                rejection_reason = "invalid_solution_shape"
            elif not np.isfinite(solution).all():
                rejection_reason = "nonfinite_solution"
            else:
                candidate_applied = extract_first_step(
                    transform.to_physical(solution),
                    config=config,
                    load_actual_kw=load_actual,
                    current_soc=soc_before,
                )
                if not np.isfinite(
                    np.asarray(tuple(candidate_applied.values()), dtype=float)
                ).all():
                    rejection_reason = "nonfinite_candidate"
                    candidate_applied = None
        else:
            rejection_reason = "solver_not_solved"

        if candidate_applied is not None:
            candidate_p_fc = float(candidate_applied["P_fc_actual_kw"])
            candidate_p_batt_actual = float(candidate_applied["P_batt_actual_kw"])
            candidate_p_batt_plan = float(candidate_applied["P_batt_plan_kw"])
            candidate_soc_actual = float(candidate_applied["SOC_actual"])
            candidate_fc_delta = candidate_p_fc - prev_fc_before
            plan_balance_residual = abs(
                float(candidate_applied["P_fc_plan_kw"])
                + candidate_p_batt_plan
                - float(load_horizon[0])
            )
            actual_balance_residual = abs(
                candidate_p_fc + candidate_p_batt_actual - load_actual
            )
            fc_bound_residual = max(
                0.0,
                float(config.fuel_cell_min_kw) - candidate_p_fc,
                candidate_p_fc - float(config.fuel_cell_max_kw),
            )
            battery_bound_residual = max(
                0.0,
                -float(config.battery_charge_max_kw) - candidate_p_batt_actual,
                candidate_p_batt_actual - float(config.battery_discharge_max_kw),
            )
            ramp_residual = max(
                0.0,
                abs(candidate_fc_delta) - float(resolved_ramp_kw_per_step(config)),
            )
            soc_bound_residual = max(
                0.0,
                float(config.soc_min) - candidate_soc_actual,
                candidate_soc_actual - float(config.soc_max),
            )
            soc_prediction_residual = abs(
                float(candidate_applied["SOC_predicted"]) - candidate_soc_actual
            )
            commit_checks = {
                "actual_balance_kw": actual_balance_residual,
                "qp_balance_kw": plan_balance_residual,
                "fc_power_bound_kw": fc_bound_residual,
                "battery_power_bound_kw": battery_bound_residual,
                "ramp_kw": ramp_residual,
                "soc": soc_bound_residual,
                "soc_prediction": soc_prediction_residual,
            }
            commit_limits = {
                "actual_balance_kw": N6_STATE_COMMIT_TOLERANCES["actual_balance_kw"],
                "qp_balance_kw": N6_STATE_COMMIT_TOLERANCES["qp_balance_kw"],
                "fc_power_bound_kw": N6_STATE_COMMIT_TOLERANCES["power_bound_kw"],
                "battery_power_bound_kw": N6_STATE_COMMIT_TOLERANCES["power_bound_kw"],
                "ramp_kw": N6_STATE_COMMIT_TOLERANCES["ramp_kw"],
                "soc": N6_STATE_COMMIT_TOLERANCES["soc"],
                "soc_prediction": N6_STATE_COMMIT_TOLERANCES["soc_prediction"],
            }
            rejected_fields = [
                name for name, value in commit_checks.items() if value > commit_limits[name]
            ]
            if rejected_fields:
                rejection_reason = "commit tolerance gate: " + ",".join(rejected_fields)
        else:
            plan_balance_residual = float("nan")
            actual_balance_residual = float("nan")
            fc_bound_residual = float("nan")
            battery_bound_residual = float("nan")
            ramp_residual = float("nan")
            soc_bound_residual = float("nan")
            soc_prediction_residual = float("nan")

        success = bool(solver_claimed_success and not rejection_reason)
        status = (
            solver_status
            if success or not solver_claimed_success
            else f"{solver_status}; {rejection_reason}"
        )
        cold_restart_succeeded = bool(cold_restart_used and success)
        if success and candidate_applied is not None:
            applied = candidate_applied
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
        soc_actual = float(applied["SOC_actual"])
        fc_delta = p_fc - prev_fc_before if success else float("nan")
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
                "solver_status": solver_status,
                "rejection_reason": rejection_reason,
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
                "solver_status": solver_status,
                "rejection_reason": rejection_reason,
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
    if versions != {"device_channel_natural_spline_1s"}:
        raise ValueError(
            "N=6 sensitivity requires "
            "dataset_version=device_channel_natural_spline_1s"
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
    resolved = input_path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return Path("external") / resolved.name


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_sha256() -> str:
    dependencies = (
        Path(__file__),
        REPO_ROOT / "src/main/mpc_solvers/mpc_qp_formulation.py",
        REPO_ROOT / "src/main/mpc_solvers/osqp_runtime.py",
        REPO_ROOT / "src/mpc/solvers/fc_dp0_curve.py",
        REPO_ROOT / "data/fuel_cell/FC_Dp0_curve_for_Python.csv",
    )
    component_hashes = "".join(sha256_file(path) for path in dependencies)
    return hashlib.sha256(component_hashes.encode("ascii")).hexdigest()


def prepare_case_dir(
    output_dir: Path,
    case: SensitivityCase,
    *,
    overwrite: bool,
    diagnostic_voyage: str | None = None,
) -> Path:
    root = Path(output_dir).resolve()
    if diagnostic_voyage is None:
        parent = root
    else:
        diagnostics_root = (root / "diagnostics").resolve()
        parent = (diagnostics_root / str(diagnostic_voyage)).resolve()
        if parent.parent != diagnostics_root:
            raise ValueError("diagnostic path escaped output directory")
    parent.mkdir(parents=True, exist_ok=True)
    case_dir = (parent / case.config_id).resolve()
    if case_dir.parent != parent:
        raise ValueError("configuration path escaped output directory")
    if case_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"existing configuration requires validation or --overwrite: {case_dir}"
            )
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    return case_dir


def configuration_metadata(
    result: dict[str, Any],
    *,
    input_path: Path,
    formal_complete: bool,
) -> dict[str, Any]:
    case = result["case"]
    config = result["config"]
    qp_metadata = build_qp_problem(
        config,
        load_forecast_kw=np.zeros(N6_HORIZON),
        current_soc=FIXED_SOC_REFERENCE,
        prev_fc_kw=0.0,
        soc_reference=FIXED_SOC_REFERENCE,
    ).metadata
    implementation_sha256 = _implementation_sha256()
    return {
        "config_id": case.config_id,
        "weights": {
            name: float(getattr(case, name)) for name in WEIGHT_NAMES
        },
        "model": asdict(config),
        "soc_reference": FIXED_SOC_REFERENCE,
        "qp_metadata": qp_metadata,
        "input_path": _portable_input_path(Path(input_path)).as_posix(),
        "input_sha256": sha256_file(Path(input_path)),
        "split_path": _portable_input_path(DEFAULT_SPLIT_JSON).as_posix(),
        "split_sha256": sha256_file(DEFAULT_SPLIT_JSON),
        "source_sha256": implementation_sha256,
        "implementation_sha256": implementation_sha256,
        "voyages": result["voyage_metrics"]["voyage_id"].astype(str).tolist(),
        "formal_complete": bool(formal_complete),
        "lstm_used": False,
        "dqn_used": False,
        "forecast": "t+1..t+6 actual natural-clipped spline load where available",
        "forecast_tail_policy": (
            "same-voyage final sample edge-hold; never crosses voyage boundary"
        ),
        "audit_tolerances": dict(N6_TOLERANCES),
        "state_commit_tolerances": dict(N6_STATE_COMMIT_TOLERANCES),
        "first_move_only": True,
        "configuration_summary": result["summary"],
        "configuration_summary_provenance": {
            "recoverable_fields": "validated against voyage_metrics.csv on reuse",
            "p95_solve_time_ms": (
                "exact all-solver-step aggregate persisted because voyage-level "
                "p95 values cannot recover it exactly"
            ),
        },
    }


def write_voyage_plot(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    config_id: str,
) -> None:
    required = {
        "time_s",
        "load_actual_kw",
        "P_fc_actual_kw",
        "P_batt_actual_kw",
        "SOC_actual",
        "cum_J_h2_norm",
        "cum_J_batt_norm",
        "cum_J_soc_norm",
        "cum_J_fc_var_norm",
        "success",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"voyage plot is missing required columns: {missing}")
    ordered = frame.sort_values("time_s", kind="stable")
    time_s = pd.to_numeric(ordered["time_s"], errors="coerce")
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 12))
    try:
        axes = np.asarray(axes).reshape(-1)

        axes[0].plot(time_s, ordered["load_actual_kw"], label="Load", linewidth=1.2)
        axes[0].plot(time_s, ordered["P_fc_actual_kw"], label="Fuel cell", linewidth=1.2)
        axes[0].set_ylabel("Power (kW)")
        axes[0].legend(loc="best")

        axes[1].plot(time_s, ordered["P_batt_actual_kw"], label="Battery", linewidth=1.2)
        axes[1].axhline(1248.0, color="0.45", linestyle="--", linewidth=0.9)
        axes[1].axhline(-624.0, color="0.45", linestyle="--", linewidth=0.9)
        axes[1].set_ylabel("P_batt (kW)")
        axes[1].legend(loc="best")

        axes[2].plot(time_s, ordered["SOC_actual"], label="SOC", linewidth=1.2)
        axes[2].axhline(0.2, color="0.45", linestyle="--", linewidth=0.9)
        axes[2].axhline(FIXED_SOC_REFERENCE, color="tab:green", linestyle=":", linewidth=1.0)
        axes[2].axhline(0.8, color="0.45", linestyle="--", linewidth=0.9)
        axes[2].set_ylabel("SOC")
        axes[2].legend(loc="best")

        for column, label in (
            ("cum_J_h2_norm", "H2 norm"),
            ("cum_J_batt_norm", "Battery norm"),
            ("cum_J_soc_norm", "SOC norm"),
            ("cum_J_fc_var_norm", "FC variation norm"),
        ):
            axes[3].plot(time_s, ordered[column], label=label, linewidth=1.1)
        axes[3].set_ylabel("Cumulative objective")
        axes[3].set_xlabel("Time (s)")
        axes[3].legend(loc="best", ncol=2)

        success = _normalized_success_flags(ordered["success"]).fillna(False).astype(bool)
        failures = ordered.loc[~success]
        if not failures.empty:
            failure_time = float(failures.iloc[0]["time_s"])
            for axis in axes:
                axis.axvline(
                    failure_time,
                    color="tab:red",
                    linestyle=":",
                    linewidth=1.2,
                    label="First failure",
                )
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.suptitle(f"{config_id}: {ordered.iloc[0]['voyage_id']}")
        figure.tight_layout()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)


def write_configuration_artifacts(
    result: dict[str, Any],
    *,
    output_dir: Path,
    input_path: Path,
    overwrite: bool,
    diagnostic_voyage: str | None,
) -> Path:
    case = result["case"]
    case_dir = prepare_case_dir(
        Path(output_dir),
        case,
        overwrite=overwrite,
        diagnostic_voyage=diagnostic_voyage,
    )
    plot_dir = case_dir / "plots"
    plot_dir.mkdir()
    try:
        configuration_p95 = float(result["summary"]["p95_solve_time_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("configuration p95 solve time is missing or invalid") from exc
    if not np.isfinite(configuration_p95):
        raise ValueError("configuration p95 solve time must be finite")
    voyage_metrics = result["voyage_metrics"].copy()
    voyage_metrics["configuration_p95_solve_time_ms"] = configuration_p95
    voyage_metrics.to_csv(case_dir / "voyage_metrics.csv", index=False)
    for voyage_id, frame in result["controls"].groupby("voyage_id", sort=True):
        write_voyage_plot(
            frame,
            plot_dir / f"{voyage_id}_power_soc_objectives.png",
            config_id=case.config_id,
        )
    metadata = configuration_metadata(
        result,
        input_path=Path(input_path),
        formal_complete=diagnostic_voyage is None,
    )
    (case_dir / "config.json").write_text(
        json.dumps(_json_ready(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return case_dir


_SUMMARY_SUM_COLUMNS: tuple[str, ...] = (
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
)
_FORBIDDEN_FIELD_TOKENS: tuple[str, ...] = (
    "selected",
    "score",
    "rank",
    "winner",
    "best",
)


def _reject_forbidden_fields(columns: Any) -> None:
    forbidden = [
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in _FORBIDDEN_FIELD_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"automatic selection fields are forbidden: {forbidden}")


def _rebuild_configuration_summary(
    voyage_metrics: pd.DataFrame,
    *,
    case: SensitivityCase,
    persisted_summary: dict[str, Any],
) -> dict[str, Any]:
    conditionally_finite_physical = {
        "max_power_balance_residual_kw",
        "max_fc_ramp_kw_per_step",
        "max_fc_kw",
        "min_fc_kw",
        "max_batt_discharge_kw",
        "max_batt_charge_kw",
    }
    always_finite_numeric = {
        *WEIGHT_NAMES,
        "expected_step_count",
        "attempted_step_count",
        "applied_step_count",
        "mean_solve_time_ms",
        "p95_solve_time_ms",
        "max_solve_time_ms",
        "initial_soc",
        "final_soc",
        "delta_soc",
        "min_soc",
        "max_soc",
        "configuration_p95_solve_time_ms",
        *_SUMMARY_SUM_COLUMNS,
    }
    required = {
        "completed",
        *always_finite_numeric,
        *conditionally_finite_physical,
    }
    missing = sorted(required.difference(voyage_metrics.columns))
    if missing:
        raise ValueError(f"voyage metrics cannot rebuild summary; missing: {missing}")
    numeric_metrics = voyage_metrics.copy()
    for name in sorted(always_finite_numeric):
        values = pd.to_numeric(voyage_metrics[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite numeric values")
        numeric_metrics[name] = values
    count_columns = {
        "expected_step_count",
        "attempted_step_count",
        "applied_step_count",
        "solver_failure_count",
        "primal_infeasible_count",
        "max_iter_count",
    }
    for name in count_columns:
        if (numeric_metrics[name] < 0).any():
            raise ValueError(f"{name} must contain non-negative values")
    if float(numeric_metrics["attempted_step_count"].sum()) <= 0.0:
        raise ValueError("attempted_step_count must have a positive configuration total")
    applied = numeric_metrics["applied_step_count"]
    for name in sorted(conditionally_finite_physical):
        raw_values = voyage_metrics[name]
        values = pd.to_numeric(raw_values, errors="coerce")
        invalid_tokens = raw_values.notna() & values.isna()
        infinite = values.notna() & ~np.isfinite(values.to_numpy(dtype=float))
        missing_after_applied = values.isna() & applied.gt(0)
        if invalid_tokens.any() or infinite.any() or missing_after_applied.any():
            raise ValueError(
                f"{name} must be finite numeric when applied_step_count is positive; "
                "only NaN with zero applied steps is allowed"
            )
        numeric_metrics[name] = values

    configuration_p95_values = numeric_metrics[
        "configuration_p95_solve_time_ms"
    ].to_numpy(dtype=float)
    if not np.equal(configuration_p95_values, configuration_p95_values[0]).all():
        raise ValueError("configuration p95 values must be identical in every voyage row")
    configuration_p95 = float(configuration_p95_values[0])
    try:
        persisted_p95 = float(persisted_summary["p95_solve_time_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("persisted exact aggregate p95 is invalid") from exc
    if not np.isfinite(persisted_p95) or persisted_p95 != configuration_p95:
        raise ValueError("configuration p95 mismatch between CSV and persisted summary")

    completed = (
        _normalized_success_flags(voyage_metrics["completed"])
        .fillna(False)
        .astype(bool)
    )
    attempted = numeric_metrics["attempted_step_count"]
    voyage_means = numeric_metrics["mean_solve_time_ms"]
    mean_solve_ms = float((attempted * voyage_means).sum() / attempted.sum())
    rebuilt: dict[str, Any] = {
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
        "initial_soc": float(numeric_metrics["initial_soc"].mean()),
        "final_soc_mean": float(numeric_metrics["final_soc"].mean()),
        "delta_soc_mean": float(numeric_metrics["delta_soc"].mean()),
        "min_soc": float(numeric_metrics["min_soc"].min()),
        "max_soc": float(numeric_metrics["max_soc"].max()),
        "max_power_balance_residual_kw": float(
            numeric_metrics["max_power_balance_residual_kw"].max()
        ),
        "max_fc_ramp_kw_per_step": float(
            numeric_metrics["max_fc_ramp_kw_per_step"].max()
        ),
        "max_fc_kw": float(numeric_metrics["max_fc_kw"].max()),
        "min_fc_kw": float(numeric_metrics["min_fc_kw"].min()),
        "max_batt_discharge_kw": float(
            numeric_metrics["max_batt_discharge_kw"].max()
        ),
        "max_batt_charge_kw": float(
            numeric_metrics["max_batt_charge_kw"].max()
        ),
        "mean_solve_time_ms": mean_solve_ms,
        "p95_solve_time_ms": configuration_p95,
        "max_solve_time_ms": float(numeric_metrics["max_solve_time_ms"].max()),
        "metrics_comparable": bool(completed.all()),
    }
    rebuilt.update(
        {
            name: float(numeric_metrics[name].sum())
            for name in _SUMMARY_SUM_COLUMNS
        }
    )
    for name in ("solver_failure_count", "primal_infeasible_count", "max_iter_count"):
        rebuilt[name] = int(rebuilt[name])

    _reject_forbidden_fields(persisted_summary)
    if set(persisted_summary) != set(rebuilt):
        raise ValueError("persisted configuration summary schema does not match metrics")
    for name, rebuilt_value in rebuilt.items():
        persisted_value = persisted_summary[name]
        if isinstance(rebuilt_value, (int, float, np.integer, np.floating)) and not isinstance(
            rebuilt_value, (bool, np.bool_)
        ):
            if persisted_value is None and np.isnan(float(rebuilt_value)):
                matches = True
            else:
                try:
                    matches = bool(
                        np.isclose(
                            float(rebuilt_value),
                            float(persisted_value),
                            rtol=1.0e-12,
                            atol=1.0e-12,
                            equal_nan=True,
                        )
                    )
                except (TypeError, ValueError):
                    matches = False
        else:
            matches = rebuilt_value == persisted_value
        if not matches:
            raise ValueError(f"persisted summary mismatch for {name}")
    # CSV values are authoritative and are all checked above. Returning the JSON
    # serialization preserves the exact original row, including the persisted
    # all-step p95 that cannot be derived from seven voyage-level percentiles.
    reused_summary = dict(persisted_summary)
    for name, rebuilt_value in rebuilt.items():
        if reused_summary[name] is None and isinstance(rebuilt_value, float) and np.isnan(
            rebuilt_value
        ):
            reused_summary[name] = rebuilt_value
    return reused_summary


def load_matching_case(
    case_dir: Path,
    *,
    case: SensitivityCase,
    input_path: Path,
    expected_voyages: tuple[str, ...],
    formal_complete: bool,
) -> dict[str, Any]:
    directory = Path(case_dir)
    expected_root_entries = {"config.json", "voyage_metrics.csv", "plots"}
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected_root_entries:
        raise ValueError(f"configuration artifact set is incomplete or unexpected: {directory}")
    metadata_path = directory / "config.json"
    metrics_path = directory / "voyage_metrics.csv"
    plot_dir = directory / "plots"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid configuration metadata: {metadata_path}") from exc

    if metadata.get("config_id") != case.config_id:
        raise ValueError("configuration id mismatch")
    weights = metadata.get("weights")
    expected_weights = {name: float(getattr(case, name)) for name in WEIGHT_NAMES}
    if not isinstance(weights, dict) or set(weights) != set(WEIGHT_NAMES):
        raise ValueError("configuration weight schema mismatch")
    try:
        weight_matches = all(float(weights[name]) == value for name, value in expected_weights.items())
    except (TypeError, ValueError):
        weight_matches = False
    if not weight_matches:
        raise ValueError("configuration weights mismatch")
    if metadata.get("model", {}).get("objective_variant") != OBJECTIVE_VARIANT:
        raise ValueError("objective variant mismatch")
    if metadata.get("qp_metadata", {}).get("objective_variant") != OBJECTIVE_VARIANT:
        raise ValueError("QP objective variant mismatch")
    if metadata.get("input_sha256") != sha256_file(Path(input_path)):
        raise ValueError("input SHA256 mismatch")
    if metadata.get("split_sha256") != sha256_file(DEFAULT_SPLIT_JSON):
        raise ValueError("active split SHA256 mismatch")
    if metadata.get("source_sha256") != _implementation_sha256():
        raise ValueError("source SHA256 mismatch")
    if metadata.get("implementation_sha256") != _implementation_sha256():
        raise ValueError("implementation SHA256 mismatch")
    if metadata.get("formal_complete") is not bool(formal_complete):
        raise ValueError("configuration completeness mismatch")
    expected_voyage_list = [str(voyage) for voyage in expected_voyages]
    if metadata.get("voyages") != expected_voyage_list:
        raise ValueError("configuration voyage metadata mismatch")

    try:
        voyage_metrics = pd.read_csv(
            metrics_path,
            dtype={"voyage_id": "string"},
            float_precision="round_trip",
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"invalid voyage metrics: {metrics_path}") from exc
    if len(voyage_metrics) != len(expected_voyage_list):
        raise ValueError("voyage metrics row count mismatch")
    actual_voyages = voyage_metrics["voyage_id"].astype(str).tolist()
    if actual_voyages != expected_voyage_list:
        raise ValueError("voyage metrics IDs mismatch")
    if "config_id" not in voyage_metrics or not voyage_metrics["config_id"].astype(str).eq(case.config_id).all():
        raise ValueError("voyage metrics configuration mismatch")
    for name, expected_weight in expected_weights.items():
        if name not in voyage_metrics:
            raise ValueError(f"voyage metrics weight is missing: {name}")
        values = pd.to_numeric(voyage_metrics[name], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite numeric values")
        if not np.allclose(values, expected_weight, rtol=0.0, atol=0.0):
            raise ValueError(f"voyage metrics weight mismatch: {name}")
        voyage_metrics[name] = values

    expected_plot_names = {
        f"{voyage}_power_soc_objectives.png" for voyage in expected_voyage_list
    }
    if not plot_dir.is_dir() or {path.name for path in plot_dir.iterdir()} != expected_plot_names:
        raise ValueError("voyage plot set mismatch")
    if not all(path.is_file() for path in plot_dir.iterdir()):
        raise ValueError("voyage plot path is not a file")
    persisted_summary = metadata.get("configuration_summary")
    if not isinstance(persisted_summary, dict):
        raise ValueError("configuration summary metadata is missing")
    return _rebuild_configuration_summary(
        voyage_metrics,
        case=case,
        persisted_summary=persisted_summary,
    )


def run_experiment(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    voyage_id: str | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    cases = build_sensitivity_cases()
    data = load_spline_test_data(Path(input_path))
    if voyage_id is None:
        actual_voyages = tuple(sorted(data["voyage_id"].astype(str).unique()))
        if actual_voyages != EXPECTED_TEST_VOYAGES:
            raise ValueError(
                f"formal run requires {EXPECTED_TEST_VOYAGES}, found {actual_voyages}"
            )
    else:
        if voyage_id not in EXPECTED_TEST_VOYAGES:
            raise ValueError(f"unsupported voyage: {voyage_id}")
        data = data.loc[data["voyage_id"].astype(str).eq(voyage_id)].copy()
        if data.empty:
            raise ValueError(f"input does not contain diagnostic voyage: {voyage_id}")

    root = Path(output_dir)
    summaries: list[dict[str, Any]] = []
    for case in cases:
        case_dir = (
            root / case.config_id
            if voyage_id is None
            else root / "diagnostics" / voyage_id / case.config_id
        )
        if case_dir.exists() and not overwrite:
            expected = EXPECTED_TEST_VOYAGES if voyage_id is None else (voyage_id,)
            summary = load_matching_case(
                case_dir,
                case=case,
                input_path=Path(input_path),
                expected_voyages=expected,
                formal_complete=voyage_id is None,
            )
        else:
            result = evaluate_configuration(case, data=data)
            write_configuration_artifacts(
                result,
                output_dir=root,
                input_path=Path(input_path),
                overwrite=overwrite,
                diagnostic_voyage=voyage_id,
            )
            summary = result["summary"]
        summaries.append(summary)
    return pd.DataFrame(summaries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed candidate_C N=6 normalized four-objective MPC"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Compatibility flag; candidate_C is the only available configuration.",
    )
    parser.add_argument("--voyage", choices=EXPECTED_TEST_VOYAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    run_experiment(
        output_dir=arguments.output_dir,
        voyage_id=arguments.voyage,
        overwrite=arguments.overwrite,
    )


if __name__ == "__main__":
    main()
