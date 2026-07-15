from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpc_solvers.mpc_qp_formulation import QpMpcConfig
from run_mpc_1s_n6_qsoc_feasibility import qsoc_candidate_config
from run_mpc_1s_n6_weight_selection import build_candidate_metrics, run_voyage


SOC_REFERENCE = 0.55
ACTIVE_ERROR_LIMIT = 0.02
ACTIVE_CORRECTION_THRESHOLD_KW = 5.0
COMPARISON_TOLERANCE = 1.0e-12
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "mpc_1s_n6_soc_clamping_diagnostic"
REPORT_PATH = REPO_ROOT / "reports" / "mpc_1s_n6_soc_clamping_diagnostic.md"
METRICS_REPORT_PATH = REPO_ROOT / "reports" / "mpc_1s_n6_soc_clamping_metrics.csv"


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    profile_kind: str
    candidate_id: str
    q_soc: float
    initial_soc: float


def _soc_code(value: float) -> str:
    return f"{int(round(float(value) * 100)):03d}"


def build_constant_profile() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(3601, dtype=float)
    loads = np.full(times.shape, 300.0, dtype=float)
    return times, loads


def build_pulse_profile() -> tuple[np.ndarray, np.ndarray]:
    times, constant_loads = build_constant_profile()
    loads = constant_loads.copy()
    loads[(times >= 600.0) & (times < 720.0)] = 450.0
    return times, loads


def clamping_candidate_config(q_soc: float) -> QpMpcConfig:
    normalized_q_soc = float(q_soc)
    if not np.isfinite(normalized_q_soc) or normalized_q_soc not in (10.0, 20.0):
        raise ValueError("q_soc must be one of {10.0, 20.0}")
    candidate_id = {
        10.0: "QSOC_10",
        20.0: "QSOC_20",
    }[normalized_q_soc]
    return qsoc_candidate_config(candidate_id)


def build_case_matrix() -> list[SyntheticCase]:
    constant_cases = [
        SyntheticCase(
            case_id=f"constant_soc{_soc_code(initial_soc)}_qsoc{int(q_soc)}",
            profile_kind="constant",
            candidate_id=f"QSOC_{int(q_soc)}",
            q_soc=q_soc,
            initial_soc=initial_soc,
        )
        for q_soc in (10.0, 20.0)
        for initial_soc in (0.53, 0.55, 0.57)
    ]
    pulse_cases = [
        SyntheticCase(
            case_id=f"pulse_soc055_qsoc{int(q_soc)}",
            profile_kind="pulse",
            candidate_id=f"QSOC_{int(q_soc)}",
            q_soc=q_soc,
            initial_soc=0.55,
        )
        for q_soc in (10.0, 20.0)
    ]
    return constant_cases + pulse_cases


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"frame must contain column {column!r}")
    try:
        values = np.asarray(frame[column], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"column {column!r} must be numeric and finite") from exc
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"column {column!r} must be one-dimensional and finite")
    return values


def _boolean_vector(mask: object, name: str = "mask") -> np.ndarray:
    raw = np.asarray(mask)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if bool(np.any(pd.isna(raw))):
        raise ValueError(f"{name} must not contain missing values")
    if raw.dtype.kind != "b" and not all(
        isinstance(value, (bool, np.bool_)) for value in raw.tolist()
    ):
        raise ValueError(f"{name} must contain boolean values")
    return raw.astype(bool, copy=False)


def annotate_correction_power(
    frame: pd.DataFrame,
    soc_reference: float = SOC_REFERENCE,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    reference = _finite_float(soc_reference, "soc_reference")
    soc_before = _finite_column(frame, "SOC_before")
    battery_power = _finite_column(frame, "P_batt_actual_kw")

    annotated = frame.copy(deep=True)
    error = soc_before - reference
    correction = np.sign(error) * battery_power
    positive = np.maximum(correction, 0.0)
    active = (correction > ACTIVE_CORRECTION_THRESHOLD_KW) & (
        np.abs(error) <= ACTIVE_ERROR_LIMIT + COMPARISON_TOLERANCE
    )

    annotated["soc_error_before"] = error
    annotated["P_correction_kw"] = correction
    annotated["positive_P_correction_kw"] = positive
    annotated["active_near_reference_correction"] = active
    return annotated


def longest_true_run(mask: object) -> int:
    values = _boolean_vector(mask)
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _aligned_window_mask(frame: pd.DataFrame, mask: object) -> np.ndarray:
    if isinstance(mask, pd.Series) and not mask.index.equals(frame.index):
        raise ValueError("mask index must exactly match frame index")
    values = _boolean_vector(mask)
    if values.size != len(frame):
        raise ValueError("mask length must match frame length")
    return values


def summarize_window(
    frame: pd.DataFrame,
    mask: object,
    dt_seconds: float = 1.0,
) -> dict[str, float | int]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    dt = _finite_float(dt_seconds, "dt_seconds")
    if dt <= 0.0:
        raise ValueError("dt_seconds must be greater than zero")

    selected = _aligned_window_mask(frame, mask)
    if not np.any(selected):
        raise ValueError("empty analysis window")
    window = annotate_correction_power(frame.iloc[np.flatnonzero(selected)])

    soc = _finite_column(window, "SOC_actual")
    battery_power = _finite_column(window, "P_batt_actual_kw")
    fuel_cell_power = _finite_column(window, "P_fc_actual_kw")
    load_power = _finite_column(window, "load_actual_kw")
    hydrogen_per_step = _finite_column(window, "h2_kg_step")
    correction = _finite_column(window, "P_correction_kw")
    positive_correction = _finite_column(window, "positive_P_correction_kw")
    active = _boolean_vector(window["active_near_reference_correction"])
    active_at_original_positions = np.zeros(len(frame), dtype=bool)
    active_at_original_positions[selected] = active

    battery_charge = np.maximum(-battery_power, 0.0)
    battery_discharge = np.maximum(battery_power, 0.0)
    wrong_direction = np.maximum(-correction, 0.0)
    fuel_cell_surplus = np.maximum(fuel_cell_power - load_power, 0.0)
    energy_factor = dt / 3600.0
    sample_count = int(soc.size)

    return {
        "sample_count": sample_count,
        "duration_s": float(sample_count * dt),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "soc_range": float(np.max(soc) - np.min(soc)),
        "soc_std": float(np.std(soc, ddof=0)),
        "soc_final": float(soc[-1]),
        "mean_abs_soc_error": float(np.mean(np.abs(soc - SOC_REFERENCE))),
        "mean_positive_P_correction_kw": float(np.mean(positive_correction)),
        "p95_positive_P_correction_kw": float(
            np.percentile(positive_correction, 95.0)
        ),
        "max_positive_P_correction_kw": float(np.max(positive_correction)),
        "ratio_active_correction": float(np.mean(active)),
        "active_correction_s": float(np.count_nonzero(active) * dt),
        "longest_active_correction_s": float(
            longest_true_run(active_at_original_positions) * dt
        ),
        "corrective_energy_kwh": float(np.sum(positive_correction) * energy_factor),
        "wrong_direction_energy_kwh": float(np.sum(wrong_direction) * energy_factor),
        "E_fc_surplus_kwh": float(np.sum(fuel_cell_surplus) * energy_factor),
        "E_batt_charge_kwh": float(np.sum(battery_charge) * energy_factor),
        "E_batt_discharge_kwh": float(np.sum(battery_discharge) * energy_factor),
        "E_batt_throughput_kwh": float(
            np.sum(battery_charge + battery_discharge) * energy_factor
        ),
        "H2_total_kg": float(np.sum(hydrogen_per_step)),
        "mean_P_fc_actual_kw": float(np.mean(fuel_cell_power)),
        "mean_P_batt_actual_kw": float(np.mean(battery_power)),
        "mean_load_actual_kw": float(np.mean(load_power)),
    }


def recovery_milestone(
    state_times_s: object,
    soc_values: object,
    *,
    initial_soc: float,
    reduction_fraction: float,
    soc_reference: float = SOC_REFERENCE,
) -> tuple[float | None, bool]:
    try:
        times = np.asarray(state_times_s, dtype=float)
        soc = np.asarray(soc_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("state times and SOC values must be numeric arrays") from exc
    if times.ndim != 1 or soc.ndim != 1:
        raise ValueError("state times and SOC values must be one-dimensional")
    if times.size == 0 or times.size != soc.size:
        raise ValueError("state times and SOC values must have equal nonzero length")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(soc)):
        raise ValueError("state times and SOC values must be finite")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("state times must be strictly increasing")

    initial = _finite_float(initial_soc, "initial_soc")
    fraction = _finite_float(reduction_fraction, "reduction_fraction")
    reference = _finite_float(soc_reference, "soc_reference")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("reduction_fraction must be between 0 and 1")

    threshold = abs(initial - reference) * (1.0 - fraction)
    reached = np.abs(soc - reference) <= threshold + COMPARISON_TOLERANCE
    reached_indices = np.flatnonzero(reached)
    if reached_indices.size == 0:
        return None, False
    first = int(reached_indices[0])
    return float(times[first]), bool(np.all(reached[first:]))


def steady_state_mask(
    frame: pd.DataFrame,
    *,
    trailing_window_steps: int = 60,
    max_load_range_kw: float = 5.0,
    jump_threshold_kw: float = 10.0,
    post_jump_exclusion_s: float = 60.0,
    fc_saturation_kw: float = 560.0,
    start_exclusion_s: float = 60.0,
    end_exclusion_s: float = 60.0,
) -> pd.Series:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("frame must not be empty")
    if isinstance(trailing_window_steps, (bool, np.bool_)) or not isinstance(
        trailing_window_steps, (int, np.integer)
    ):
        raise ValueError("trailing_window_steps must be a positive integer")
    window_steps = int(trailing_window_steps)
    if window_steps <= 0:
        raise ValueError("trailing_window_steps must be a positive integer")

    max_range = _finite_float(max_load_range_kw, "max_load_range_kw")
    jump_threshold = _finite_float(jump_threshold_kw, "jump_threshold_kw")
    post_jump = _finite_float(post_jump_exclusion_s, "post_jump_exclusion_s")
    fc_saturation = _finite_float(fc_saturation_kw, "fc_saturation_kw")
    start_exclusion = _finite_float(start_exclusion_s, "start_exclusion_s")
    end_exclusion = _finite_float(end_exclusion_s, "end_exclusion_s")
    if max_range < 0.0 or jump_threshold < 0.0:
        raise ValueError("load thresholds must be nonnegative")
    if post_jump < 0.0 or start_exclusion < 0.0 or end_exclusion < 0.0:
        raise ValueError("time exclusions must be nonnegative")
    if fc_saturation <= 0.0:
        raise ValueError("fc_saturation_kw must be greater than zero")

    time_column = "state_time_s" if "state_time_s" in frame.columns else "time_s"
    if time_column not in frame.columns:
        raise ValueError("frame must contain state_time_s or time_s")
    times = _finite_column(frame, time_column)
    load = _finite_column(frame, "load_actual_kw")
    fuel_cell_power = _finite_column(frame, "P_fc_actual_kw")
    if times.size > 1 and not np.allclose(
        np.diff(times),
        1.0,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(f"{time_column} must be strictly increasing in 1 s rows")

    load_series = pd.Series(load, dtype=float)
    rolling_range = (
        load_series.rolling(window_steps, min_periods=window_steps).max()
        - load_series.rolling(window_steps, min_periods=window_steps).min()
    ).to_numpy()
    stable_load = np.isfinite(rolling_range) & (rolling_range <= max_range)

    jumps = np.zeros(times.size, dtype=bool)
    if times.size > 1:
        jumps[1:] = np.abs(np.diff(load)) > jump_threshold
    latest_jump = np.maximum.accumulate(np.where(jumps, times, -np.inf))
    outside_post_jump = times - latest_jump >= post_jump

    mask = (
        stable_load
        & outside_post_jump
        & (fuel_cell_power < fc_saturation)
        & (times - times[0] >= start_exclusion)
        & (times[-1] - times >= end_exclusion)
    )
    return pd.Series(mask, index=frame.index, dtype=bool)


def run_synthetic_case(case: SyntheticCase) -> tuple[pd.DataFrame, dict[str, object]]:
    if case.profile_kind == "constant":
        times, loads = build_constant_profile()
    elif case.profile_kind == "pulse":
        times, loads = build_pulse_profile()
    else:
        raise ValueError(f"unknown synthetic profile: {case.profile_kind}")

    config = clamping_candidate_config(case.q_soc)
    if case.candidate_id != f"QSOC_{int(case.q_soc)}":
        raise ValueError(f"case {case.case_id} has inconsistent q_soc identity")

    controls, solver = run_voyage(
        voyage_id=case.case_id,
        loads_kw=loads,
        times_s=times,
        candidate_id=case.candidate_id,
        initial_soc=case.initial_soc,
        config=config,
    )
    if len(controls) != 3600:
        raise ValueError(f"case {case.case_id} produced {len(controls)} of 3600 steps")
    if "success" not in controls or not controls["success"].fillna(False).astype(bool).all():
        raise ValueError(f"case {case.case_id} contains a failed control step")
    if not np.isclose(float(controls.iloc[0]["SOC_before"]), case.initial_soc, atol=1.0e-12):
        raise ValueError(f"case {case.case_id} initial SOC does not match the case contract")
    if not np.isclose(float(controls.iloc[0]["prev_fc_actual_kw"]), 300.0, atol=1.0e-9):
        raise ValueError(f"case {case.case_id} initial fuel-cell power is not 300 kW")

    physical, _, _ = build_candidate_metrics(controls, solver, config=config)
    if not bool(physical.get("closed_loop_complete", False)):
        raise ValueError(f"case {case.case_id} is not a complete closed-loop run")
    if int(physical.get("solver_failure_count", 0)) != 0:
        raise ValueError(f"case {case.case_id} contains solver failures")
    if int(physical.get("physical_infeasible_point_count", 0)) != 0:
        raise ValueError(f"case {case.case_id} contains physically infeasible points")

    trajectory = annotate_correction_power(controls)
    trajectory.insert(0, "profile_kind", case.profile_kind)
    trajectory.insert(0, "q_soc", case.q_soc)
    trajectory.insert(0, "case_id", case.case_id)
    trajectory["interval_start_s"] = trajectory["decision_time_s"].astype(float)
    trajectory["state_time_s"] = trajectory["time_s"].astype(float)

    full_metrics = summarize_window(trajectory, np.ones(len(trajectory), dtype=bool))
    metrics: dict[str, object] = {
        "case_id": case.case_id,
        "profile_kind": case.profile_kind,
        "q_soc": case.q_soc,
        "initial_soc": case.initial_soc,
        **full_metrics,
        "solver_failure_count": int(physical.get("solver_failure_count", 0)),
        "physical_infeasible_point_count": int(
            physical.get("physical_infeasible_point_count", 0)
        ),
    }

    for state_time in (300, 900, 1800, 3600):
        state = trajectory.loc[trajectory["state_time_s"].eq(float(state_time)), "SOC_actual"]
        if len(state) != 1:
            raise ValueError(f"case {case.case_id} is missing SOC at {state_time} s")
        metrics[f"soc_at_{state_time}s"] = float(state.iloc[0])

    state_times = np.concatenate(([0.0], trajectory["state_time_s"].to_numpy(dtype=float)))
    soc_states = np.concatenate(([case.initial_soc], trajectory["SOC_actual"].to_numpy(dtype=float)))
    for percentage, fraction in ((25, 0.25), (50, 0.50)):
        reached_s, sustained = recovery_milestone(
            state_times,
            soc_states,
            initial_soc=case.initial_soc,
            reduction_fraction=fraction,
        )
        metrics[f"recovery_{percentage}_time_s"] = reached_s
        metrics[f"recovery_{percentage}_sustained"] = sustained

    if case.profile_kind == "constant":
        analysis_mask = trajectory["state_time_s"].gt(60.0)
        prefix = "after_60s_"
    else:
        analysis_mask = trajectory["state_time_s"].ge(780.0)
        prefix = "post_transient_"
    for name, value in summarize_window(trajectory, analysis_mask).items():
        metrics[f"{prefix}{name}"] = value

    return trajectory, metrics


def _select_case(
    trajectories: pd.DataFrame,
    *,
    profile_kind: str,
    q_soc: float,
    initial_soc: float,
) -> pd.DataFrame:
    selected = trajectories[
        trajectories["profile_kind"].eq(profile_kind)
        & trajectories["q_soc"].eq(q_soc)
    ]
    case_id = (
        f"{profile_kind}_soc{_soc_code(initial_soc)}_qsoc{int(q_soc)}"
    )
    selected = selected[selected["case_id"].eq(case_id)]
    if len(selected) != 3600:
        raise ValueError(f"missing complete trajectory for {case_id}")
    return selected


def _plot_constant_comparison(
    trajectories: pd.DataFrame,
    initial_soc: float,
    destination: Path,
) -> None:
    q10 = _select_case(
        trajectories, profile_kind="constant", q_soc=10.0, initial_soc=initial_soc
    )
    q20 = _select_case(
        trajectories, profile_kind="constant", q_soc=20.0, initial_soc=initial_soc
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    time = q10["state_time_s"]
    axes[0].plot(time, q10["load_actual_kw"], color="black", lw=1.0, label="load")
    axes[0].plot(time, q10["P_fc_actual_kw"], label="q_soc=10")
    axes[0].plot(time, q20["P_fc_actual_kw"], label="q_soc=20")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(ncol=3)
    axes[1].plot(time, q10["P_batt_actual_kw"], label="q_soc=10")
    axes[1].plot(time, q20["P_batt_actual_kw"], label="q_soc=20")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_ylabel("Battery (kW)")
    axes[1].legend()
    axes[2].plot(time, q10["SOC_actual"], label="q_soc=10")
    axes[2].plot(time, q20["SOC_actual"], label="q_soc=20")
    axes[2].axhline(SOC_REFERENCE, color="black", ls="--", lw=0.8, label="SOC ref")
    axes[2].set_ylabel("SOC")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend()
    fig.suptitle(
        f"diagnostic synthetic profile | constant 300 kW | initial SOC={initial_soc:.2f}"
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_pulse_comparison(trajectories: pd.DataFrame, destination: Path) -> None:
    q10 = _select_case(
        trajectories, profile_kind="pulse", q_soc=10.0, initial_soc=0.55
    )
    q20 = _select_case(
        trajectories, profile_kind="pulse", q_soc=20.0, initial_soc=0.55
    )
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    time = q10["state_time_s"]
    axes[0].plot(time, q10["load_actual_kw"], color="black")
    axes[0].set_ylabel("Load (kW)")
    axes[1].plot(time, q10["P_fc_actual_kw"], label="q_soc=10")
    axes[1].plot(time, q20["P_fc_actual_kw"], label="q_soc=20")
    axes[1].set_ylabel("FC (kW)")
    axes[1].legend()
    axes[2].plot(time, q10["P_batt_actual_kw"], label="q_soc=10")
    axes[2].plot(time, q20["P_batt_actual_kw"], label="q_soc=20")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("Battery (kW)")
    axes[3].plot(time, q10["SOC_actual"], label="q_soc=10")
    axes[3].plot(time, q20["SOC_actual"], label="q_soc=20")
    axes[3].axhline(SOC_REFERENCE, color="black", ls="--", lw=0.8)
    axes[3].set_ylabel("SOC")
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        for boundary in (600.0, 720.0, 780.0):
            axis.axvline(boundary, color="gray", ls=":" if boundary != 780.0 else "--", lw=0.9)
    fig.suptitle("diagnostic synthetic profile | 300-450-300 kW pulse")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_metric_comparison(metrics: pd.DataFrame, destination: Path) -> None:
    constant = metrics[metrics["profile_kind"].eq("constant")].copy()
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.5))
    specifications = (
        ("recovery_50_time_s", "50% recovery (s)"),
        ("mean_positive_P_correction_kw", "Mean correction (kW)"),
        ("E_fc_surplus_kwh", "FC surplus (kWh)"),
        ("E_batt_throughput_kwh", "Battery throughput (kWh)"),
        ("H2_excess_kg", "H2 excess (kg)"),
    )
    colors = {10.0: "tab:blue", 20.0: "tab:orange"}
    for axis, (column, label) in zip(axes, specifications):
        for q_soc in (10.0, 20.0):
            group = constant[constant["q_soc"].eq(q_soc)].sort_values("initial_soc")
            if column == "recovery_50_time_s":
                group = group[group["initial_soc"].ne(0.55)]
            values = pd.to_numeric(group[column], errors="coerce")
            axis.plot(group["initial_soc"], values, "o-", color=colors[q_soc], label=f"q={int(q_soc)}")
            if column == "recovery_50_time_s":
                for x_value, y_value in zip(group["initial_soc"], values):
                    if not np.isfinite(y_value):
                        axis.scatter([x_value], [0.0], marker="x", color=colors[q_soc])
                        axis.annotate("not reached", (x_value, 0.0), rotation=90, fontsize=7)
        axis.set_title(label)
        axis.set_xlabel("Initial SOC")
        axis.grid(alpha=0.25)
    axes[0].set_xlim(0.525, 0.575)
    axes[0].legend()
    fig.suptitle("diagnostic synthetic profile | q_soc metric comparison")
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for case in build_case_matrix():
        trajectory, metrics = run_synthetic_case(case)
        trajectories.append(trajectory)
        metric_rows.append(metrics)

    trajectory_table = pd.concat(trajectories, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    if len(trajectory_table) != 8 * 3600 or len(metrics) != 8:
        raise ValueError("formal diagnostic must contain exactly 8 cases and 28,800 steps")

    metrics["load_energy_kwh"] = metrics["mean_load_actual_kw"] * metrics["duration_s"] / 3600.0
    for row_index, row in metrics.iterrows():
        baseline = metrics[
            metrics["profile_kind"].eq("constant")
            & metrics["q_soc"].eq(row["q_soc"])
            & metrics["initial_soc"].eq(0.55)
        ]
        if len(baseline) != 1:
            raise ValueError("each q_soc must have one constant SOC=0.55 baseline")
        reference = baseline.iloc[0]
        metrics.loc[row_index, "H2_excess_kg"] = row["H2_total_kg"] - reference["H2_total_kg"]
        metrics.loc[row_index, "battery_throughput_excess_kwh"] = (
            row["E_batt_throughput_kwh"] - reference["E_batt_throughput_kwh"]
        )
        metrics.loc[row_index, "fc_surplus_energy_excess_kwh"] = (
            row["E_fc_surplus_kwh"] - reference["E_fc_surplus_kwh"]
        )

    metrics = metrics.rename(
        columns={
            "soc_range": "SOC_range",
            "soc_std": "SOC_std",
            "soc_final": "SOC_final",
            "mean_abs_soc_error": "mean_abs_SOC_error",
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plot_root = OUTPUT_ROOT / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    trajectory_table.to_parquet(OUTPUT_ROOT / "synthetic_trajectories.parquet", index=False)
    metrics.to_csv(OUTPUT_ROOT / "case_metrics.csv", index=False)
    METRICS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_REPORT_PATH, index=False)

    for initial_soc in (0.53, 0.55, 0.57):
        _plot_constant_comparison(
            trajectory_table,
            initial_soc,
            plot_root / f"constant_soc{_soc_code(initial_soc)}_comparison.png",
        )
    _plot_pulse_comparison(trajectory_table, plot_root / "pulse_comparison.png")
    _plot_metric_comparison(metrics, plot_root / "metric_comparison.png")
    return trajectory_table, metrics


def main() -> int:
    trajectories, metrics = run_experiment()
    print(f"cases={len(metrics)} steps={len(trajectories)}")
    print(f"metrics={METRICS_REPORT_PATH}")
    print(f"output={OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
