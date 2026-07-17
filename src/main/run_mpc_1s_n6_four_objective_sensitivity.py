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

    osqp_module, osqp_error = _try_import_osqp()
    if osqp_module is None:
        raise RuntimeError(f"Cannot import osqp: {osqp_error}")

    current_soc = float(initial_soc)
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
                "h2_kg_step": (
                    physical_h2_kg_step(config, p_fc) if success else float("nan")
                ),
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
