from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    build_qp_problem,
    h2_quadratic_kg_step_coefficients,
    ramp_kw_per_step_from_rate,
    resolved_ramp_kw_per_step,
    write_qp_formulation_check,
)

DEFAULT_INPUT_PARQUET = ROOT / "outputs" / "mpc_solver_benchmark_1s" / "data" / "test_voyages_spline_1s.parquet"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mpc_solver_benchmark_1s" / "osqp_n60"
EBATT277P2_OUTPUT_DIR = ROOT / "outputs" / "mpc_solver_benchmark_1s" / "osqp_n60_Ebatt277p2"
DEFAULT_SENSITIVITY_OUTPUT_DIR = (
    ROOT / "outputs" / "mpc_solver_benchmark_1s" / "osqp_n60_Ebatt277p2_weight_sensitivity"
)
RAW_WEIGHT_RETUNE_OUTPUT_DIR = (
    ROOT / "outputs" / "mpc_solver_benchmark_1s" / "osqp_n60_Ebatt277p2_raw_weight_retune"
)
SIMPLIFIED_SPEC_NORM_OUTPUT_DIR = (
    ROOT / "outputs" / "mpc_solver_benchmark_1s" / "osqp_n60_Ebatt693_simplified_spec_norm"
)
DEFAULT_REPORT_FILENAME = "REPORT_OSQP_QP_MPC_1S_BENCHMARK.md"
EBATT277P2_REPORT_FILENAME = "REPORT_EBATT277P2_WEIGHT_VALIDITY.md"
RAW_WEIGHT_RETUNE_REPORT_FILENAME = "REPORT_RAW_WEIGHT_RETUNE_FOR_PHYSICAL_BASELINE.md"
SIMPLIFIED_SPEC_NORM_REPORT_FILENAME = "REPORT_SIMPLIFIED_SPEC_NORMALIZED_OBJECTIVE.md"
QP_CHECK_FILENAME = "qp_formulation_check.md"
OLD_BATTERY_CAPACITY_KWH = 1806.0
EBATT277P2_BATTERY_CAPACITY_KWH = 277.2
SPEC_BATTERY_CLUSTER_CAPACITY_KWH = 69.3
SPEC_BATTERY_CLUSTER_COUNT = 10
SPEC_BATTERY_CAPACITY_KWH = SPEC_BATTERY_CLUSTER_CAPACITY_KWH * SPEC_BATTERY_CLUSTER_COUNT
SPEC_BATTERY_C_RATE = 0.5
SPEC_BATTERY_POWER_MAX_KW = SPEC_BATTERY_CAPACITY_KWH * SPEC_BATTERY_C_RATE
SPEC_BATTERY_POWER_REF_KW = SPEC_BATTERY_POWER_MAX_KW
SPEC_SOC_BAND = 0.05
SPEC_OBJECTIVE_VARIANT = "simplified_normalized_literature_v1"
DEFAULT_SOC_REFERENCE = 0.55
POWER_BALANCE_TOL_KW = 1.0e-2
VIOLATION_TOL = 1.0e-6
POWER_BOUND_TOL_KW = 1.0e-1
BATTERY_UNUSED_TOL_KW = 1.0
BATTERY_UNUSED_FRACTION_LIMIT = 0.95
BATTERY_LIMIT_STICK_FRACTION_LIMIT = 0.05
SOC_FINAL_DROP_LIMIT = -0.02
REALTIME_THRESHOLDS_MS = {
    "mean": 50.0,
    "p95": 200.0,
    "p99": 500.0,
    "max": 800.0,
}
RAW_RETUNE_CASES: list[dict[str, Any]] = [
    {"case_name": "case_raw_base_old", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 0.03, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": True},
    {"case_name": "case_raw_batt_1e_5", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-5, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": True},
    {"case_name": "case_raw_batt_1e_6", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": True},
    {"case_name": "case_raw_batt_1e_7", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-7, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": True},
    {"case_name": "case_raw_batt_1e_6_soc_200", "q_h2": 1.0, "q_soc": 200.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": False},
    {"case_name": "case_raw_batt_1e_6_soc_800", "q_h2": 1.0, "q_soc": 800.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": False},
    {"case_name": "case_raw_batt_1e_6_terminal_20", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 20.0, "priority": True},
    {"case_name": "case_raw_batt_1e_6_terminal_50", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 50.0, "priority": False},
    {"case_name": "case_raw_batt_1e_6_no_ramp_cost", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 1.0e-6, "q_ramp": 0.0, "q_terminal_soc": 0.0, "priority": False},
    {"case_name": "case_raw_h2_0p5_batt_1e_6", "q_h2": 0.5, "q_soc": 400.0, "q_batt": 1.0e-6, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0, "priority": False},
]
SIMPLIFIED_SPEC_NORM_CASES: list[dict[str, Any]] = [
    {"case_name": "case_spec_norm_base", "q_h2": 1.0, "q_soc": 1.0, "q_batt": 0.05},
    {"case_name": "case_spec_norm_more_batt", "q_h2": 1.0, "q_soc": 1.0, "q_batt": 0.01},
    {"case_name": "case_spec_norm_batt_conservative", "q_h2": 1.0, "q_soc": 1.0, "q_batt": 0.10},
    {"case_name": "case_spec_norm_soc_safe", "q_h2": 1.0, "q_soc": 2.0, "q_batt": 0.05},
    {"case_name": "case_spec_norm_h2_low_fc_main", "q_h2": 0.5, "q_soc": 2.0, "q_batt": 0.05},
    {"case_name": "case_spec_norm_h2_high_economy", "q_h2": 2.0, "q_soc": 2.0, "q_batt": 0.05},
    {"case_name": "case_spec_norm_more_batt_soc_safe", "q_h2": 1.0, "q_soc": 2.0, "q_batt": 0.01},
    {"case_name": "case_spec_norm_soc_strong", "q_h2": 1.0, "q_soc": 5.0, "q_batt": 0.05},
]


def default_config(
    *,
    horizon: int = 60,
    battery_capacity_kwh: float = SPEC_BATTERY_CAPACITY_KWH,
    q_h2: float = 1.0,
    q_soc: float = 1.0,
    q_batt: float = 0.05,
    q_ramp: float = 0.0,
    q_terminal_soc: float = 0.0,
    fuel_cell_ramp_rate_kw_per_s: float = 48.0,
    battery_power_max_kw: float = SPEC_BATTERY_POWER_MAX_KW,
    battery_power_ref_kw: float = SPEC_BATTERY_POWER_REF_KW,
    soc_band: float = SPEC_SOC_BAND,
    objective_variant: str = SPEC_OBJECTIVE_VARIANT,
) -> QpMpcConfig:
    return QpMpcConfig(
        horizon=int(horizon),
        dt_seconds=1.0,
        battery_capacity_kwh=float(battery_capacity_kwh),
        battery_charge_max_kw=float(battery_power_max_kw),
        battery_discharge_max_kw=float(battery_power_max_kw),
        battery_power_ref_kw=float(battery_power_ref_kw),
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=560.0,
        fuel_cell_ramp_rate_kw_per_s=float(fuel_cell_ramp_rate_kw_per_s),
        fuel_cell_ramp_kw=None,
        soc_min=0.2,
        soc_max=0.8,
        soc_band=float(soc_band),
        objective_variant=str(objective_variant),
        q_h2=float(q_h2),
        q_soc=float(q_soc),
        q_batt=float(q_batt),
        q_ramp=float(q_ramp),
        q_terminal_soc=float(q_terminal_soc),
    )


def json_safe_config(
    config: QpMpcConfig,
    *,
    osqp_available: bool,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, Any]:
    ramp_step = ramp_kw_per_step_from_rate(config.fuel_cell_ramp_rate_kw_per_s, dt_seconds=config.dt_seconds)
    return {
        "solver": "OSQP",
        "problem_class": "convex_qp",
        "osqp_available": bool(osqp_available),
        "horizon": int(config.horizon),
        "dt_seconds": float(config.dt_seconds),
        "soc_reference": float(soc_reference),
        "old_battery_capacity_kwh": float(OLD_BATTERY_CAPACITY_KWH),
        "battery_capacity_kwh": float(config.battery_capacity_kwh),
        "battery_capacity_change_note": (
            "Only this 1 s OSQP-QP benchmark uses the configured battery capacity. "
            "The existing 30 s mainline is not modified by this script."
        ),
        "battery_charge_max_kw": float(config.battery_charge_max_kw),
        "battery_discharge_max_kw": float(config.battery_discharge_max_kw),
        "battery_power_ref_kw": float(config.battery_power_ref_kw),
        "battery_power_basis_note": (
            "Specification-book 1806 kWh and about 900 kW imply about 0.5C. "
            "For the current formal scaled pack, 10 clusters x 69.3 kWh = 693 kWh, "
            "so 0.5C gives 346.5 kW. The old 277.2 kWh / 138.6 kW basis is legacy only."
        ),
        "fuel_cell_ramp_rate_kw_per_s": float(config.fuel_cell_ramp_rate_kw_per_s),
        "fuel_cell_ramp_kw_per_step": float(ramp_step),
        "fuel_cell_ramp_unit_note": "48 kW/s is converted to 48 kW per 1 s MPC step in this benchmark.",
        "fuel_cell_min_kw": float(config.fuel_cell_min_kw),
        "fuel_cell_max_kw": float(config.fuel_cell_max_kw),
        "soc_min": float(config.soc_min),
        "soc_max": float(config.soc_max),
        "soc_band": float(config.soc_band),
        "objective_variant": str(config.objective_variant),
        "objective_note": (
            "Formal 1 s variant uses only normalized H2, SOC maintenance, and battery power terms. "
            "Fuel-cell ramp remains a hard constraint, not a soft objective term."
        ),
        "q_h2": float(config.q_h2),
        "q_soc": float(config.q_soc),
        "q_batt": float(config.q_batt),
        "q_ramp": float(config.q_ramp),
        "q_terminal_soc": float(config.q_terminal_soc),
        "hydrogen_model_note": "current forced-origin Dp0 quadratic in the local QP formulation; not refit here",
        "input_data_note": "offline natural-clipped cubic-spline 1 s reconstruction, not measured 1 s data",
        "forecast_source_note": (
            "offline benchmark uses the reconstructed future load horizon from parquet; "
            "it is not an LSTM forecast validation"
        ),
    }


_json_safe_config = json_safe_config


def _try_import_osqp() -> tuple[Any | None, str | None]:
    try:
        import osqp  # type: ignore

        return osqp, None
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, str(exc)


def _h2_kg_for_step(config: QpMpcConfig, p_fc_kw: float) -> float:
    quad, linear, _, _ = h2_quadratic_kg_step_coefficients(config)
    p = float(p_fc_kw)
    return float(quad * p * p + linear * p)


def write_objective_term_scale_audit(
    *,
    output_dir: str | Path = RAW_WEIGHT_RETUNE_OUTPUT_DIR,
    config: QpMpcConfig | None = None,
) -> dict[str, str]:
    """Write the raw, unnormalized objective term scale audit requested for retuning."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or default_config(horizon=60, battery_capacity_kwh=EBATT277P2_BATTERY_CAPACITY_KWH)
    h2_quad, h2_linear, dp0_a1, dp0_a2 = h2_quadratic_kg_step_coefficients(cfg)
    rows: list[dict[str, Any]] = []

    for p_fc in [50, 100, 200, 300, 400, 560]:
        value = float(cfg.q_h2) * (h2_quad * p_fc * p_fc + h2_linear * p_fc)
        rows.append(
            {
                "term_type": "h2",
                "input_name": "P_fc_kw",
                "input_value": float(p_fc),
                "q_name": "q_h2",
                "q_value": float(cfg.q_h2),
                "term_value": value,
                "unit_note": "kg_H2_per_1s_step_weighted_by_q_h2",
            }
        )

    for q_batt in [0.03, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]:
        for p_batt in [1, 5, 10, 20, 50, 100, 200, 350]:
            rows.append(
                {
                    "term_type": "battery",
                    "input_name": "P_batt_kw",
                    "input_value": float(p_batt),
                    "q_name": "q_batt",
                    "q_value": float(q_batt),
                    "term_value": float(q_batt * p_batt * p_batt),
                    "unit_note": "raw_q_batt_times_kw_squared",
                }
            )

    for q_soc in [100, 200, 400, 800]:
        for deviation in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
            rows.append(
                {
                    "term_type": "soc",
                    "input_name": "SOC_deviation",
                    "input_value": float(deviation),
                    "q_name": "q_soc",
                    "q_value": float(q_soc),
                    "term_value": float(q_soc * deviation * deviation),
                    "unit_note": "raw_q_soc_times_soc_deviation_squared",
                }
            )

    for q_ramp in [0.0, 1.0e-6, 1.0e-5, 2.0e-5, 1.0e-4]:
        for delta in [1, 5, 10, 20, 48]:
            rows.append(
                {
                    "term_type": "ramp",
                    "input_name": "delta_P_fc_kw",
                    "input_value": float(delta),
                    "q_name": "q_ramp",
                    "q_value": float(q_ramp),
                    "term_value": float(q_ramp * delta * delta),
                    "unit_note": "raw_q_ramp_times_kw_step_squared",
                }
            )

    audit = pd.DataFrame(rows)
    csv_path = out_dir / "objective_term_scale_audit.csv"
    md_path = out_dir / "objective_term_scale_audit.md"
    audit.to_csv(csv_path, index=False)

    batt_50_old = float(audit.loc[
        audit["term_type"].eq("battery") & audit["q_value"].eq(0.03) & audit["input_value"].eq(50.0),
        "term_value",
    ].iloc[0])
    batt_50_1e6 = float(audit.loc[
        audit["term_type"].eq("battery") & audit["q_value"].eq(1.0e-6) & audit["input_value"].eq(50.0),
        "term_value",
    ].iloc[0])
    h2_100 = float(audit.loc[
        audit["term_type"].eq("h2") & audit["input_value"].eq(100.0),
        "term_value",
    ].iloc[0])
    soc_002_400 = float(audit.loc[
        audit["term_type"].eq("soc") & audit["q_value"].eq(400.0) & audit["input_value"].eq(0.02),
        "term_value",
    ].iloc[0])
    ramp_20 = float(audit.loc[
        audit["term_type"].eq("ramp") & audit["q_value"].eq(2.0e-5) & audit["input_value"].eq(20.0),
        "term_value",
    ].iloc[0])
    lines = [
        "# Raw Objective Term Scale Audit",
        "",
        "Scope: unnormalized objective only. No normalized objective, denominator scaling, DQN, LSTM training, or 30 s mainline change is introduced here.",
        "",
        "## Fixed Configuration",
        "",
        f"- dt: `{cfg.dt_seconds} s`",
        f"- N: `{cfg.horizon}`",
        f"- E_batt: `{cfg.battery_capacity_kwh} kWh`",
        f"- P_fc_max: `{cfg.fuel_cell_max_kw} kW`",
        f"- P_batt bound: `[-{cfg.battery_charge_max_kw}, {cfg.battery_discharge_max_kw}] kW`",
        f"- SOC_ref: `{DEFAULT_SOC_REFERENCE}`",
        f"- ramp limit: `{ramp_kw_per_step_from_rate(cfg.fuel_cell_ramp_rate_kw_per_s, dt_seconds=cfg.dt_seconds)} kW/step`",
        f"- H2 forced-origin coefficients: alpha=`{h2_quad}`, beta=`{h2_linear}` for kg/1s-step on P_fc in kW; source Dp0 a1=`{dp0_a1}`, a2=`{dp0_a2}`",
        "",
        "## Findings",
        "",
        f"1. With the old raw `q_batt=0.03`, `P_batt=50 kW` gives `q_batt * P_batt^2 = {batt_50_old:.6f}`. This is many orders above the single-step H2 term scale, for example the `P_fc=100 kW` H2 term is `{h2_100:.9f}` with `q_h2=1`.",
        f"2. Reducing `q_batt` to `1e-6` makes the same `P_batt=50 kW` term `{batt_50_1e6:.6f}`, which is in a plausible raw-objective range instead of dominating the optimizer.",
        "3. In this raw objective, `q_batt` should be tested around `1e-5` to `1e-7`; `0.03` is too large for direct multiplication by `P_batt^2` in kW.",
        f"4. `q_soc=400` gives `{soc_002_400:.6f}` for `SOC` deviation `0.02`. With `E_batt=277.2 kWh`, SOC responds faster than the old 1806 kWh benchmark, so terminal behavior must be checked from closed-loop runs rather than assumed.",
        f"5. `q_ramp=2e-5` gives `{ramp_20:.6f}` for a `20 kW` FC step; it is a soft smoothing term while the hard `48 kW/step` ramp constraint remains active regardless of `q_ramp`.",
        "",
        "The full numeric table is in `objective_term_scale_audit.csv`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv_path": str(csv_path), "md_path": str(md_path)}


def write_simplified_normalized_objective_check(
    *,
    output_dir: str | Path = SIMPLIFIED_SPEC_NORM_OUTPUT_DIR,
    config: QpMpcConfig | None = None,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or default_config()
    h2_quad, h2_linear, dp0_a1, dp0_a2 = h2_quadratic_kg_step_coefficients(cfg)
    h2_ref = float(h2_quad * cfg.fuel_cell_max_kw * cfg.fuel_cell_max_kw + h2_linear * cfg.fuel_cell_max_kw)
    probe = build_qp_problem(
        cfg,
        load_forecast_kw=np.zeros(cfg.horizon, dtype=float),
        current_soc=soc_reference,
        prev_fc_kw=0.0,
        soc_reference=soc_reference,
    )
    h2_examples = []
    for p_fc in [0.0, 100.0, 280.0, 560.0]:
        h2 = float((h2_quad * p_fc * p_fc + h2_linear * p_fc) / h2_ref)
        h2_examples.append({"P_fc_kw": p_fc, "H2_norm": h2})
    batt_examples = []
    for p_batt in [0.0, 10.0, 50.0, cfg.battery_power_ref_kw]:
        batt_examples.append({"P_batt_kw": p_batt, "Batt_norm": float((p_batt / cfg.battery_power_ref_kw) ** 2)})
    soc_examples = []
    for deviation in [0.0, 0.01, 0.03, cfg.soc_band]:
        soc_examples.append({"SOC_deviation": deviation, "SOC_norm": float((deviation / cfg.soc_band) ** 2)})

    md_path = out_dir / "simplified_normalized_objective_check.md"
    lines = [
        "# Simplified Normalized Objective Check",
        "",
        "Scope: formal 1 s OSQP-QP objective variant for the offline natural-clipped spline benchmark. This does not modify the 30 s CasADi/IPOPT mainline, train LSTM, or train DQN.",
        "",
        "## Variant",
        "",
        f"- Name: `{cfg.objective_variant}`",
        "- Objective: `sum(q_h2 * H2_norm + q_soc * SOC_norm + q_batt * Batt_norm)`",
        "- Removed from the objective: fuel-cell ramp soft penalty and terminal SOC penalty.",
        f"- Ramp remains as a hard constraint: `48 kW/s = {ramp_kw_per_step_from_rate(cfg.fuel_cell_ramp_rate_kw_per_s, dt_seconds=cfg.dt_seconds)} kW/step`.",
        "",
        "## Fixed Physical Denominators",
        "",
        f"- `P_fc_max = {cfg.fuel_cell_max_kw} kW`.",
        f"- `P_batt_ref = {cfg.battery_power_ref_kw} kW`, matching the 0.5C power for `E_batt = {cfg.battery_capacity_kwh} kWh`.",
        f"- `SOC_band = {cfg.soc_band}`.",
        f"- `m_H2_ref = alpha * 560^2 + beta * 560 = {h2_ref:.12f} kg/step`.",
        "",
        "These denominators are fixed physical scales. They are not test set max, min, mean, voyage statistics, or spline-data-derived statistics, so this check does not introduce data leakage.",
        "",
        "## Why 346.5 kW Replaces 138.6 kW",
        "",
        "- The specification-book full system is about `1806 kWh` and about `900 kW`, which is about `0.5C`.",
        "- The current formal scaled pack uses `10 x 69.3 kWh = 693 kWh`.",
        "- Therefore `693 kWh * 0.5C = 346.5 kW` is used for both the battery bound and the battery normalization denominator.",
        "- The old `277.2 kWh / 138.6 kW` basis is retained only as a legacy assumption in historical outputs, not as the formal battery basis for this run.",
        "",
        "## Convexity",
        "",
        f"- Hessian minimum eigenvalue: `{probe.metadata['hessian_min_eigenvalue']}`.",
        f"- Convex QP flag: `{probe.metadata['convex_qp']}`.",
        "- The objective is a nonnegative weighted sum of convex quadratic terms and a linear H2 term. Linear equality/inequality constraints preserve convex QP form.",
        "",
        "## Example Normalized Ranges",
        "",
        "### H2_norm",
        "",
        "| P_fc_kw | H2_norm |",
        "|---:|---:|",
        *[f"| {row['P_fc_kw']:.1f} | {row['H2_norm']:.9f} |" for row in h2_examples],
        "",
        "### SOC_norm",
        "",
        "| SOC_deviation | SOC_norm |",
        "|---:|---:|",
        *[f"| {row['SOC_deviation']:.3f} | {row['SOC_norm']:.9f} |" for row in soc_examples],
        "",
        "### Batt_norm",
        "",
        "| P_batt_kw | Batt_norm |",
        "|---:|---:|",
        *[f"| {row['P_batt_kw']:.1f} | {row['Batt_norm']:.9f} |" for row in batt_examples],
        "",
        "## Difference From Legacy Raw Objective",
        "",
        "- Legacy raw objective multiplied physical units directly, for example `q_batt * P_batt^2` and optional `q_ramp * delta_P_fc^2`.",
        "- The formal variant uses fixed, dimensionless physical normalizers and keeps only H2, SOC, and battery-use penalty terms.",
        "- This makes fixed weights easier to interpret and more suitable as a baseline before future DQN dynamic weighting, without adding DQN training here.",
        "",
        "## Source Coefficients",
        "",
        f"- alpha/quad coefficient in kg per 1 s step: `{h2_quad}`.",
        f"- beta/linear coefficient in kg per 1 s step: `{h2_linear}`.",
        f"- Dp0 forced-origin source coefficients: a1=`{dp0_a1}`, a2=`{dp0_a2}`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"md_path": str(md_path)}


def write_load_feasibility_check(
    data: pd.DataFrame,
    config: QpMpcConfig,
    *,
    output_dir: str | Path = SIMPLIFIED_SPEC_NORM_OUTPUT_DIR,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = data.copy()
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str).str.lower().eq("test")].copy()
    if "load_total_kw" not in frame.columns:
        raise ValueError("load feasibility check requires load_total_kw")
    available = float(config.fuel_cell_max_kw + config.battery_discharge_max_kw)
    load = _numeric(frame["load_total_kw"]).dropna()
    exceed = frame.loc[_numeric(frame["load_total_kw"]) > available].copy()
    voyages = sorted(str(v) for v in exceed["voyage_id"].dropna().unique()) if "voyage_id" in exceed.columns else []
    rows = [
        {
            "P_fc_max_kw": float(config.fuel_cell_max_kw),
            "P_batt_max_kw": float(config.battery_discharge_max_kw),
            "P_available_max_kw": available,
            "P_load_max": float(load.max()) if not load.empty else float("nan"),
            "P_load_p99": float(load.quantile(0.99)) if not load.empty else float("nan"),
            "P_load_p95": float(load.quantile(0.95)) if not load.empty else float("nan"),
            "P_load_mean": float(load.mean()) if not load.empty else float("nan"),
            "num_steps_load_exceeds_available": int(len(exceed)),
            "fraction_load_exceeds_available": float(len(exceed) / len(load)) if len(load) else float("nan"),
            "voyages_with_exceedance": ";".join(voyages),
        }
    ]
    csv_path = out_dir / "load_feasibility_check.csv"
    md_path = out_dir / "load_feasibility_check.md"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    row = rows[0]
    lines = [
        "# Load Feasibility Check",
        "",
        "Scope: test split of the offline natural-clipped 1 s spline benchmark. This is not measured 1 s data.",
        "",
        f"- P_fc_max = {config.fuel_cell_max_kw} kW",
        f"- P_batt_max = {config.battery_discharge_max_kw} kW",
        f"- P_available_max = {available:.1f} kW",
        f"- P_load_max = {row['P_load_max']:.6f} kW",
        f"- P_load_p99 = {row['P_load_p99']:.6f} kW",
        f"- P_load_p95 = {row['P_load_p95']:.6f} kW",
        f"- P_load_mean = {row['P_load_mean']:.6f} kW",
        f"- num_steps_load_exceeds_available = {row['num_steps_load_exceeds_available']}",
        f"- fraction_load_exceeds_available = {row['fraction_load_exceeds_available']:.9f}",
        f"- voyages_with_exceedance = `{row['voyages_with_exceedance'] or 'none'}`",
        "",
        f"If load exceeds `{available:.1f} kW`, the benchmark must attribute any resulting infeasibility to insufficient physical power under the configured 0.5C battery limit.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"csv_path": str(csv_path), "md_path": str(md_path)}


def _solve_problem(osqp_module: Any, problem: Any) -> tuple[Any, float]:
    solver = osqp_module.OSQP()
    start = time.perf_counter()
    solver.setup(
        P=problem.P,
        q=problem.q,
        A=problem.A,
        l=problem.l,
        u=problem.u,
        verbose=False,
        polish=True,
        warm_start=True,
        eps_abs=1.0e-4,
        eps_rel=1.0e-4,
        max_iter=4000,
    )
    result = solver.solve()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, float(elapsed_ms)


def _qp_bounds_for_step(
    config: QpMpcConfig,
    *,
    load_forecast_kw: np.ndarray | list[float],
    current_soc: float,
    prev_fc_kw: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build only OSQP lower/upper vectors for the fixed 1 s QP structure."""
    horizon = int(config.horizon)
    load = np.asarray(load_forecast_kw, dtype=float).reshape(-1)
    if len(load) != horizon:
        raise ValueError(f"load_forecast_kw must contain exactly {horizon} points")
    if not np.all(np.isfinite(load)):
        raise ValueError("load_forecast_kw contains non-finite values")

    n_constraints = 6 * horizon + 2
    lower = np.empty(n_constraints, dtype=float)
    upper = np.empty(n_constraints, dtype=float)
    cursor = 0

    lower[cursor : cursor + horizon] = float(config.fuel_cell_min_kw)
    upper[cursor : cursor + horizon] = float(config.fuel_cell_max_kw)
    cursor += horizon

    lower[cursor : cursor + horizon] = -float(config.battery_charge_max_kw)
    upper[cursor : cursor + horizon] = float(config.battery_discharge_max_kw)
    cursor += horizon

    lower[cursor : cursor + horizon + 1] = float(config.soc_min)
    upper[cursor : cursor + horizon + 1] = float(config.soc_max)
    cursor += horizon + 1

    lower[cursor] = float(current_soc)
    upper[cursor] = float(current_soc)
    cursor += 1

    lower[cursor : cursor + horizon] = 0.0
    upper[cursor : cursor + horizon] = 0.0
    cursor += horizon

    lower[cursor : cursor + horizon] = load
    upper[cursor : cursor + horizon] = load
    cursor += horizon

    ramp_kw = resolved_ramp_kw_per_step(config)
    lower[cursor] = float(prev_fc_kw) - ramp_kw
    upper[cursor] = float(prev_fc_kw) + ramp_kw
    cursor += 1

    lower[cursor : cursor + horizon - 1] = -ramp_kw
    upper[cursor : cursor + horizon - 1] = ramp_kw
    cursor += horizon - 1

    if cursor != n_constraints:
        raise RuntimeError(f"internal QP bound length mismatch: {cursor} != {n_constraints}")
    return lower, upper


def _can_reuse_osqp_solver(config: QpMpcConfig) -> bool:
    return (
        str(config.objective_variant) == SPEC_OBJECTIVE_VARIANT
        and float(config.q_ramp) == 0.0
        and float(config.q_terminal_soc) == 0.0
    )


def _setup_persistent_osqp_solver(osqp_module: Any, problem: Any) -> Any:
    solver = osqp_module.OSQP()
    solver.setup(
        P=problem.P,
        q=problem.q,
        A=problem.A,
        l=problem.l,
        u=problem.u,
        verbose=False,
        polish=True,
        warm_start=True,
        eps_abs=1.0e-4,
        eps_rel=1.0e-4,
        max_iter=4000,
    )
    return solver


def _solve_with_persistent_osqp(solver: Any, *, lower: np.ndarray, upper: np.ndarray) -> tuple[Any, float]:
    start = time.perf_counter()
    solver.update(l=lower, u=upper)
    result = solver.solve()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, float(elapsed_ms)


def advance_soc_from_battery_power(
    config: QpMpcConfig,
    *,
    current_soc: float,
    battery_power_kw: float,
) -> float:
    dt_hours = float(config.dt_seconds) / 3600.0
    return float(current_soc) - float(battery_power_kw) * dt_hours / float(config.battery_capacity_kwh)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _q(series: pd.Series, quantile: float) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.quantile(float(quantile)))


def _max(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.max())


def _min(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.min())


def _mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _sum(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return 0.0
    return float(values.sum())


def _success_mask(df: pd.DataFrame) -> pd.Series:
    if "success" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["success"].fillna(False).astype(bool)


def add_derived_control_columns(control_df: pd.DataFrame, config: QpMpcConfig) -> pd.DataFrame:
    if control_df.empty:
        return control_df.copy()
    out = control_df.copy()
    out["P_fc_kw"] = _numeric(out["P_fc_kw"])
    out["P_batt_kw"] = _numeric(out["P_batt_kw"])
    out["SOC"] = _numeric(out["SOC"])
    if "prev_fc_before_kw" in out.columns:
        prev_fc = _numeric(out["prev_fc_before_kw"])
        delta = out["P_fc_kw"] - prev_fc
    else:
        delta = out.groupby("voyage_id", sort=False)["P_fc_kw"].diff().fillna(0.0)
    out["fc_delta_kw"] = delta
    out["fc_ramp_kw"] = delta.abs()
    out["fc_ramp_kw_per_s"] = out["fc_ramp_kw"] / float(config.dt_seconds)
    ramp_limit = ramp_kw_per_step_from_rate(config.fuel_cell_ramp_rate_kw_per_s, dt_seconds=config.dt_seconds)
    out["ramp_violation_kw"] = (out["fc_ramp_kw"] - ramp_limit).clip(lower=0.0)
    out["P_fc_violation_kw"] = np.maximum(
        0.0,
        np.maximum(
            float(config.fuel_cell_min_kw) - out["P_fc_kw"],
            out["P_fc_kw"] - float(config.fuel_cell_max_kw),
        ),
    )
    out["P_batt_violation_kw"] = np.maximum(
        0.0,
        np.maximum(
            -float(config.battery_charge_max_kw) - out["P_batt_kw"],
            out["P_batt_kw"] - float(config.battery_discharge_max_kw),
        ),
    )
    out["soc_violation"] = np.maximum(
        0.0,
        np.maximum(
            float(config.soc_min) - out["SOC"],
            out["SOC"] - float(config.soc_max),
        ),
    )
    return out


def build_voyage_metrics(
    *,
    voyage_id: Any,
    voyage_controls: pd.DataFrame,
    voyage_times: pd.DataFrame,
    config: QpMpcConfig,
    h2_total_kg: float,
    initial_soc: float,
    soc_reference: float,
) -> dict[str, Any]:
    controls = add_derived_control_columns(voyage_controls, config)
    dt_hours = float(config.dt_seconds) / 3600.0
    p_batt = _numeric(controls["P_batt_kw"]).dropna()
    p_fc = _numeric(controls["P_fc_kw"]).dropna()
    soc = _numeric(controls["SOC"]).dropna()
    times = _numeric(voyage_times["total_controller_ms"]).dropna() if not voyage_times.empty else pd.Series(dtype=float)
    success = _success_mask(voyage_times) if not voyage_times.empty else pd.Series(dtype=bool)
    status = voyage_times["status"].astype(str) if "status" in voyage_times.columns else pd.Series(dtype=str)
    discharge_kwh = float(p_batt.clip(lower=0.0).sum() * dt_hours) if not p_batt.empty else 0.0
    charge_kwh = float((-p_batt.clip(upper=0.0)).sum() * dt_hours) if not p_batt.empty else 0.0
    throughput_kwh = float(p_batt.abs().sum() * dt_hours) if not p_batt.empty else 0.0
    success_rate = float(success.mean()) if len(success) else 0.0

    return {
        "voyage_id": voyage_id,
        "steps": int(len(controls)),
        "success_rate": success_rate,
        "solver_success_rate": success_rate,
        "solver_failure_count": int((~success).sum()) if len(success) else 0,
        "infeasible_count": int(status.str.contains("infeasible", case=False, na=False).sum()) if len(status) else 0,
        "fallback_count": 0,
        "H2_total_kg": float(h2_total_kg),
        "total_h2_cost": float(h2_total_kg),
        "SOC_initial": float(initial_soc),
        "SOC_min": float(soc.min()) if not soc.empty else float("nan"),
        "SOC_max": float(soc.max()) if not soc.empty else float("nan"),
        "SOC_final": float(soc.iloc[-1]) if not soc.empty else float("nan"),
        "SOC_final_minus_initial": float(soc.iloc[-1] - initial_soc) if not soc.empty else float("nan"),
        "SOC_mean_abs_deviation_from_ref": float((soc - float(soc_reference)).abs().mean()) if not soc.empty else float("nan"),
        "battery_throughput_kwh": throughput_kwh,
        "battery_discharge_energy_kwh": discharge_kwh,
        "battery_charge_energy_kwh": charge_kwh,
        "P_batt_mean_abs": float(p_batt.abs().mean()) if not p_batt.empty else float("nan"),
        "P_batt_max": float(p_batt.max()) if not p_batt.empty else float("nan"),
        "P_batt_min": float(p_batt.min()) if not p_batt.empty else float("nan"),
        "P_fc_mean": float(p_fc.mean()) if not p_fc.empty else float("nan"),
        "P_fc_max": float(p_fc.max()) if not p_fc.empty else float("nan"),
        "P_fc_min": float(p_fc.min()) if not p_fc.empty else float("nan"),
        "fc_ramp_max": _max(controls["fc_ramp_kw"]),
        "fc_ramp_violation_count": int((_numeric(controls["ramp_violation_kw"]) > VIOLATION_TOL).sum()),
        "power_balance_violation_max": _max(controls["balance_violation_kw"]),
        "max_balance_violation_kw": _max(controls["balance_violation_kw"]),
        "max_ramp_violation_kw": _max(controls["ramp_violation_kw"]),
        "max_soc_violation": _max(controls["soc_violation"]),
        "battery_power_violation_max": _max(controls["P_batt_violation_kw"]),
        "fc_power_violation_max": _max(controls["P_fc_violation_kw"]),
        "SOC_violation_count": int((_numeric(controls["soc_violation"]) > VIOLATION_TOL).sum()),
        "battery_power_violation_count": int((_numeric(controls["P_batt_violation_kw"]) > POWER_BOUND_TOL_KW).sum()),
        "fc_power_violation_count": int((_numeric(controls["P_fc_violation_kw"]) > POWER_BOUND_TOL_KW).sum()),
        "mean_total_ms": float(times.mean()) if not times.empty else float("nan"),
        "median_total_ms": float(times.median()) if not times.empty else float("nan"),
        "p90_total_ms": float(times.quantile(0.90)) if not times.empty else float("nan"),
        "p95_total_ms": float(times.quantile(0.95)) if not times.empty else float("nan"),
        "p99_total_ms": float(times.quantile(0.99)) if not times.empty else float("nan"),
        "max_total_ms": float(times.max()) if not times.empty else float("nan"),
    }


def timing_stats(time_df: pd.DataFrame, config: QpMpcConfig) -> dict[str, Any]:
    times = _numeric(time_df["total_controller_ms"]).dropna() if "total_controller_ms" in time_df.columns else pd.Series(dtype=float)
    success = _success_mask(time_df) if not time_df.empty else pd.Series(dtype=bool)
    status = time_df["status"].astype(str) if "status" in time_df.columns else pd.Series(dtype=str)
    mean_ms = float(times.mean()) if not times.empty else float("nan")
    median_ms = float(times.median()) if not times.empty else float("nan")
    p90_ms = float(times.quantile(0.90)) if not times.empty else float("nan")
    p95_ms = float(times.quantile(0.95)) if not times.empty else float("nan")
    p99_ms = float(times.quantile(0.99)) if not times.empty else float("nan")
    max_ms = float(times.max()) if not times.empty else float("nan")
    sample_ms = float(config.dt_seconds) * 1000.0
    success_rate = float(success.mean()) if len(success) else 0.0
    stats = {
        "steps": int(len(time_df)),
        "success_rate": success_rate,
        "solver_success_rate": success_rate,
        "infeasible_count": int(status.str.contains("infeasible", case=False, na=False).sum()) if len(status) else 0,
        "fallback_count": 0,
        "solve_time_ms_mean": mean_ms,
        "solve_time_ms_median": median_ms,
        "solve_time_ms_p90": p90_ms,
        "solve_time_ms_p95": p95_ms,
        "solve_time_ms_p99": p99_ms,
        "solve_time_ms_max": max_ms,
        "real_time_factor_mean": mean_ms / sample_ms if np.isfinite(mean_ms) else float("nan"),
        "real_time_factor_p99": p99_ms / sample_ms if np.isfinite(p99_ms) else float("nan"),
    }
    stats["mean_gate_passed"] = bool(np.isfinite(mean_ms) and mean_ms < REALTIME_THRESHOLDS_MS["mean"])
    stats["p95_gate_passed"] = bool(np.isfinite(p95_ms) and p95_ms < REALTIME_THRESHOLDS_MS["p95"])
    stats["p99_gate_passed"] = bool(np.isfinite(p99_ms) and p99_ms < REALTIME_THRESHOLDS_MS["p99"])
    stats["max_gate_passed"] = bool(np.isfinite(max_ms) and max_ms < REALTIME_THRESHOLDS_MS["max"])
    stats["success_gate_passed"] = bool(success_rate >= 0.99)
    stats["realtime_gate_passed"] = bool(
        stats["success_gate_passed"]
        and stats["mean_gate_passed"]
        and stats["p95_gate_passed"]
        and stats["p99_gate_passed"]
        and stats["max_gate_passed"]
    )
    return stats


def control_performance_metrics(
    control_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: QpMpcConfig,
    *,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, Any]:
    controls = add_derived_control_columns(control_df, config)
    dt_hours = float(config.dt_seconds) / 3600.0
    p_batt = _numeric(controls["P_batt_kw"]).dropna() if "P_batt_kw" in controls.columns else pd.Series(dtype=float)
    p_fc = _numeric(controls["P_fc_kw"]).dropna() if "P_fc_kw" in controls.columns else pd.Series(dtype=float)
    soc = _numeric(controls["SOC"]).dropna() if "SOC" in controls.columns else pd.Series(dtype=float)
    if not metrics_df.empty and "H2_total_kg" in metrics_df.columns:
        h2_total = _sum(metrics_df["H2_total_kg"])
    else:
        h2_total = 0.0
    discharge_kwh = float(p_batt.clip(lower=0.0).sum() * dt_hours) if not p_batt.empty else 0.0
    charge_kwh = float((-p_batt.clip(upper=0.0)).sum() * dt_hours) if not p_batt.empty else 0.0
    throughput_kwh = float(p_batt.abs().sum() * dt_hours) if not p_batt.empty else 0.0
    near_zero_fraction = float((p_batt.abs() <= BATTERY_UNUSED_TOL_KW).mean()) if not p_batt.empty else float("nan")
    near_limit_fraction = (
        float((p_batt.abs() >= 0.95 * max(config.battery_charge_max_kw, config.battery_discharge_max_kw)).mean())
        if not p_batt.empty
        else float("nan")
    )

    if not metrics_df.empty and "SOC_initial" in metrics_df.columns:
        soc_initial = _mean(metrics_df["SOC_initial"])
        soc_final = _mean(metrics_df["SOC_final"])
        soc_final_minus_initial = _mean(metrics_df["SOC_final_minus_initial"])
        soc_final_minus_initial_min = _min(metrics_df["SOC_final_minus_initial"])
    else:
        soc_initial = float(soc.iloc[0]) if not soc.empty else float("nan")
        soc_final = float(soc.iloc[-1]) if not soc.empty else float("nan")
        soc_final_minus_initial = float(soc_final - soc_initial) if np.isfinite(soc_initial) and np.isfinite(soc_final) else float("nan")
        soc_final_minus_initial_min = soc_final_minus_initial
    soc_abs_dev = (soc - float(soc_reference)).abs() if not soc.empty else pd.Series(dtype=float)
    p_batt_abs = p_batt.abs() if not p_batt.empty else pd.Series(dtype=float)
    fc_ramp = _numeric(controls["fc_ramp_kw"]).dropna() if "fc_ramp_kw" in controls.columns else pd.Series(dtype=float)
    load = _numeric(controls["load_total_kw"]).dropna() if "load_total_kw" in controls.columns else pd.Series(dtype=float)
    load_energy_kwh = float(load.sum() * dt_hours) if not load.empty else float("nan")
    available_power_kw = float(config.fuel_cell_max_kw + config.battery_discharge_max_kw)
    load_exceeds_count = int((load > available_power_kw).sum()) if not load.empty else 0
    fc_energy_kwh = float(p_fc.clip(lower=0.0).sum() * dt_hours) if not p_fc.empty else float("nan")
    if "load_total_kw" in controls.columns and not p_fc.empty:
        aligned = pd.DataFrame({"p_fc": p_fc, "load": _numeric(controls.loc[p_fc.index, "load_total_kw"])})
        aligned = aligned[aligned["load"].abs() > 1.0e-9]
        fc_load_share_mean = float((aligned["p_fc"] / aligned["load"]).mean()) if not aligned.empty else float("nan")
    else:
        fc_load_share_mean = float("nan")

    return {
        "total_h2_cost": h2_total,
        "H2_total_kg": h2_total,
        "SOC_min": float(soc.min()) if not soc.empty else float("nan"),
        "SOC_max": float(soc.max()) if not soc.empty else float("nan"),
        "SOC_initial": soc_initial,
        "SOC_final": soc_final,
        "SOC_final_minus_initial": soc_final_minus_initial,
        "SOC_final_minus_initial_min": soc_final_minus_initial_min,
        "SOC_mean_abs_deviation": float((soc - float(soc_reference)).abs().mean()) if not soc.empty else float("nan"),
        "SOC_mean_abs_deviation_from_ref": float(soc_abs_dev.mean()) if not soc_abs_dev.empty else float("nan"),
        "SOC_max_abs_deviation_from_ref": float(soc_abs_dev.max()) if not soc_abs_dev.empty else float("nan"),
        "SOC_drop_max_by_voyage": float(max(0.0, -soc_final_minus_initial_min)) if np.isfinite(soc_final_minus_initial_min) else float("nan"),
        "battery_throughput_kwh": throughput_kwh,
        "battery_discharge_kwh": discharge_kwh,
        "battery_discharge_energy_kWh": discharge_kwh,
        "battery_charge_kwh": charge_kwh,
        "battery_charge_energy_kWh": charge_kwh,
        "P_batt_mean_abs": float(p_batt.abs().mean()) if not p_batt.empty else float("nan"),
        "P_batt_max_abs": float(p_batt_abs.max()) if not p_batt_abs.empty else float("nan"),
        "P_batt_max": float(p_batt.max()) if not p_batt.empty else float("nan"),
        "P_batt_min": float(p_batt.min()) if not p_batt.empty else float("nan"),
        "P_batt_near_zero_fraction_abs_le_1kw": near_zero_fraction,
        "battery_active_fraction_abs_gt_1kW": float((p_batt_abs > 1.0).mean()) if not p_batt_abs.empty else float("nan"),
        "battery_active_fraction_abs_gt_5kW": float((p_batt_abs > 5.0).mean()) if not p_batt_abs.empty else float("nan"),
        "battery_active_fraction_abs_gt_10kW": float((p_batt_abs > 10.0).mean()) if not p_batt_abs.empty else float("nan"),
        "battery_saturation_fraction_abs_gt_0p9Pmax": float((p_batt_abs > 0.90 * max(config.battery_charge_max_kw, config.battery_discharge_max_kw)).mean()) if not p_batt_abs.empty else float("nan"),
        "P_batt_near_limit_fraction_abs_ge_95pct": near_limit_fraction,
        "load_exceeds_power_limit_count": load_exceeds_count,
        "load_exceeds_power_limit_fraction": float(load_exceeds_count / len(load)) if len(load) else float("nan"),
        "P_available_max_kw": available_power_kw,
        "P_load_max": float(load.max()) if not load.empty else float("nan"),
        "P_fc_mean": float(p_fc.mean()) if not p_fc.empty else float("nan"),
        "P_fc_std": float(p_fc.std(ddof=0)) if not p_fc.empty else float("nan"),
        "P_fc_max": float(p_fc.max()) if not p_fc.empty else float("nan"),
        "P_fc_min": float(p_fc.min()) if not p_fc.empty else float("nan"),
        "P_fc_ramp_mean_abs": float(fc_ramp.mean()) if not fc_ramp.empty else float("nan"),
        "P_fc_ramp_p95_abs": float(fc_ramp.quantile(0.95)) if not fc_ramp.empty else float("nan"),
        "P_fc_ramp_max_abs": float(fc_ramp.max()) if not fc_ramp.empty else float("nan"),
        "fc_energy_share": fc_energy_kwh / load_energy_kwh if np.isfinite(load_energy_kwh) and load_energy_kwh > 0.0 else float("nan"),
        "fc_load_share_mean": fc_load_share_mean,
        "fc_ramp_max": _max(controls["fc_ramp_kw"]) if "fc_ramp_kw" in controls.columns else float("nan"),
        "fc_ramp_violation_count": int((_numeric(controls["ramp_violation_kw"]) > VIOLATION_TOL).sum())
        if "ramp_violation_kw" in controls.columns
        else 0,
        "power_balance_violation_max": _max(controls["balance_violation_kw"])
        if "balance_violation_kw" in controls.columns
        else float("nan"),
        "battery_power_violation_max": _max(controls["P_batt_violation_kw"])
        if "P_batt_violation_kw" in controls.columns
        else float("nan"),
        "fc_power_violation_max": _max(controls["P_fc_violation_kw"])
        if "P_fc_violation_kw" in controls.columns
        else float("nan"),
        "SOC_violation_count": int((_numeric(controls["soc_violation"]) > VIOLATION_TOL).sum())
        if "soc_violation" in controls.columns
        else 0,
        "battery_power_violation_count": int((_numeric(controls["P_batt_violation_kw"]) > POWER_BOUND_TOL_KW).sum())
        if "P_batt_violation_kw" in controls.columns
        else 0,
        "fc_power_violation_count": int((_numeric(controls["P_fc_violation_kw"]) > POWER_BOUND_TOL_KW).sum())
        if "P_fc_violation_kw" in controls.columns
        else 0,
    }


def objective_term_summary(
    control_df: pd.DataFrame,
    config: QpMpcConfig,
    *,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> pd.DataFrame:
    controls = add_derived_control_columns(control_df, config)
    if "success" in controls.columns:
        controls = controls[_success_mask(controls)].copy()
    h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(config)
    h2_ref = float(h2_quad * config.fuel_cell_max_kw * config.fuel_cell_max_kw + h2_linear * config.fuel_cell_max_kw)

    def term_row(term_name: str, q_value: float, values: pd.Series | np.ndarray | list[float], unit_note: str) -> dict[str, Any]:
        series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            return {
                "term_name": term_name,
                "q_value": float(q_value),
                "term_sum": 0.0,
                "term_mean_per_step": float("nan"),
                "term_max_per_step": float("nan"),
                "term_count": 0,
                "unit_note": unit_note,
            }
        return {
            "term_name": term_name,
            "q_value": float(q_value),
            "term_sum": float(series.sum()),
            "term_mean_per_step": float(series.mean()),
            "term_max_per_step": float(series.max()),
            "term_count": int(series.count()),
            "unit_note": unit_note,
        }

    p_fc = _numeric(controls["P_fc_kw"]) if "P_fc_kw" in controls.columns else pd.Series(dtype=float)
    p_batt = _numeric(controls["P_batt_kw"]) if "P_batt_kw" in controls.columns else pd.Series(dtype=float)
    soc = _numeric(controls["SOC"]) if "SOC" in controls.columns else pd.Series(dtype=float)

    rows = [
        term_row(
            "H2_norm",
            config.q_h2,
            (h2_quad * p_fc * p_fc + h2_linear * p_fc) / h2_ref,
            "forced_origin_Dp0_kg_H2_per_1s_step_divided_by_m_H2_ref",
        ),
        term_row(
            "SOC_norm",
            config.q_soc,
            ((soc - float(soc_reference)) / float(config.soc_band)) ** 2,
            "SOC_next_deviation_squared_divided_by_SOC_band_squared",
        ),
        term_row(
            "Batt_norm",
            config.q_batt,
            (p_batt / float(config.battery_power_ref_kw)) ** 2,
            "P_batt_kw_squared_divided_by_P_batt_ref_squared",
        ),
    ]
    return pd.DataFrame(rows)


def objective_term_totals(control_df: pd.DataFrame, config: QpMpcConfig, *, soc_reference: float = DEFAULT_SOC_REFERENCE) -> dict[str, float]:
    summary = objective_term_summary(control_df, config, soc_reference=soc_reference)
    values = {str(row["term_name"]): float(row["term_sum"]) for _, row in summary.iterrows()}
    means = {str(row["term_name"]): float(row["term_mean_per_step"]) for _, row in summary.iterrows()}
    return {
        "H2_norm_sum": values.get("H2_norm", 0.0),
        "SOC_norm_sum": values.get("SOC_norm", 0.0),
        "Batt_norm_sum": values.get("Batt_norm", 0.0),
        "H2_norm_mean": means.get("H2_norm", float("nan")),
        "SOC_norm_mean": means.get("SOC_norm", float("nan")),
        "Batt_norm_mean": means.get("Batt_norm", float("nan")),
    }


def objective_terms_by_voyage(
    control_df: pd.DataFrame,
    config: QpMpcConfig,
    *,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> pd.DataFrame:
    if control_df.empty or "voyage_id" not in control_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for voyage_id, group in control_df.groupby("voyage_id", sort=True):
        row = {"voyage_id": voyage_id}
        row.update(objective_term_totals(group, config, soc_reference=soc_reference))
        rows.append(row)
    return pd.DataFrame(rows)


def _row_value(row: dict[str, Any] | pd.Series, *names: str, default: float = float("nan")) -> Any:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    else:
        data = dict(row)
    lower = {str(key).lower(): value for key, value in data.items()}
    for name in names:
        if name in data:
            return data[name]
        value = lower.get(name.lower())
        if value is not None:
            return value
    return default


def _row_float(row: dict[str, Any] | pd.Series, *names: str, default: float = float("nan")) -> float:
    value = _row_value(row, *names, default=default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def classify_physical_baseline(row: dict[str, Any] | pd.Series) -> str:
    success_rate = _row_float(row, "solver_success_rate", "success_rate", default=0.0)
    p99_ms = _row_float(row, "solve_time_ms_p99", "p99_solve_time_ms", default=float("inf"))
    max_ms = _row_float(row, "solve_time_ms_max", "max_solve_time_ms", default=0.0)
    p95_ms = _row_float(row, "solve_time_ms_p95", "p95_solve_time_ms", default=0.0)
    mean_ms = _row_float(row, "solve_time_ms_mean", "mean_solve_time_ms", default=0.0)
    infeasible_count = _row_float(row, "infeasible_count", default=0.0)
    fallback_count = _row_float(row, "fallback_count", default=0.0)
    load_exceeds_count = _row_float(
        row,
        "load_exceeds_power_limit_count",
        "num_steps_load_exceeds_available",
        "num_steps_load_exceeds_698p6",
        default=0.0,
    )
    if load_exceeds_count > 0 and (success_rate < 0.99 or infeasible_count > 0 or fallback_count > 0):
        return "FAIL_POWER_LIMIT_INSUFFICIENT"
    if (
        success_rate < 0.99
        or infeasible_count > 0
        or fallback_count > 0
        or mean_ms >= REALTIME_THRESHOLDS_MS["mean"]
        or p95_ms >= REALTIME_THRESHOLDS_MS["p95"]
        or p99_ms >= REALTIME_THRESHOLDS_MS["p99"]
        or max_ms >= REALTIME_THRESHOLDS_MS["max"]
    ):
        return "FAIL_SOLVER"

    balance_max = _row_float(row, "power_balance_violation_max", "max_balance_violation_kw", default=0.0)
    ramp_count = _row_float(row, "fc_ramp_violation_count", default=0.0)
    battery_count = _row_float(row, "battery_power_violation_count", default=0.0)
    fc_count = _row_float(row, "fc_power_violation_count", default=0.0)
    if balance_max > POWER_BALANCE_TOL_KW or ramp_count > 0 or battery_count > 0 or fc_count > 0:
        return "FAIL_CONSTRAINT"

    soc_min = _row_float(row, "SOC_min", default=DEFAULT_SOC_REFERENCE)
    soc_max = _row_float(row, "SOC_max", default=DEFAULT_SOC_REFERENCE)
    soc_violation_count = _row_float(row, "SOC_violation_count", default=0.0)
    if soc_violation_count > 0 or soc_min < 0.2 - 1.0e-6 or soc_max > 0.8 + 1.0e-6:
        return "FAIL_SOC_VIOLATION"

    soc_final_drop = _row_float(row, "SOC_final_minus_initial_min", default=0.0)
    soc_drop_max = _row_float(row, "SOC_drop_max_by_voyage", default=max(0.0, -soc_final_drop))
    if soc_final_drop < -0.05 or soc_drop_max > 0.05:
        return "FAIL_SOC_DROP"

    near_zero = _row_float(row, "P_batt_near_zero_fraction_abs_le_1kw", default=float("nan"))
    active_5kw = _row_float(row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw", default=float("nan"))
    if (np.isfinite(near_zero) and near_zero > 0.95) or (np.isfinite(active_5kw) and active_5kw <= 0.01):
        return "FAIL_BATTERY_UNUSED"

    saturation = _row_float(
        row,
        "battery_saturation_fraction_abs_gt_0p9Pmax",
        "battery_saturation_fraction_abs_gt_300kW",
        "battery_saturation_fraction_abs_gt_300kw",
        "P_batt_near_limit_fraction_abs_ge_95pct",
        default=0.0,
    )
    if saturation > 0.05:
        return "FAIL_BATTERY_OVERUSE"

    fc_energy_share = _row_float(row, "fc_energy_share", default=1.0)
    if np.isfinite(fc_energy_share) and fc_energy_share < 0.70:
        return "FAIL_FC_NOT_MAIN_SOURCE"

    if (np.isfinite(active_5kw) and active_5kw < 0.05) or soc_final_drop < -0.03 or soc_drop_max > 0.03:
        return "BORDERLINE"
    return "PASS_PHYSICAL_BASELINE"


def select_recommended_simplified_baseline(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {
            "recommended_fixed_mpc_baseline_before_dqn": "NONE_ACCEPTED",
            "least_bad_diagnostic_case": "",
            "accepted": False,
            "label": "NO_CASES",
            "reason": "No simplified normalized candidate rows are available.",
        }
    df = summary_df.copy()
    if "physical_label" not in df.columns:
        df["physical_label"] = df.apply(classify_physical_baseline, axis=1)
    label_rank = {
        "PASS_PHYSICAL_BASELINE": 0,
        "BORDERLINE": 1,
        "FAIL_BATTERY_UNUSED": 2,
        "FAIL_SOC_DROP": 3,
        "FAIL_FC_NOT_MAIN_SOURCE": 4,
        "FAIL_BATTERY_OVERUSE": 5,
        "FAIL_CONSTRAINT": 6,
        "FAIL_POWER_LIMIT_INSUFFICIENT": 7,
        "FAIL_SOLVER": 8,
        "FAIL_SOC_VIOLATION": 9,
    }
    sort_rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        active = _row_float(row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw", default=0.0)
        soc_drop = _row_float(row, "SOC_drop_max_by_voyage", default=max(0.0, -_row_float(row, "SOC_final_minus_initial_min", default=0.0)))
        saturation = _row_float(row, "battery_saturation_fraction_abs_gt_0p9Pmax", "P_batt_near_limit_fraction_abs_ge_95pct", default=0.0)
        fc_share = _row_float(row, "fc_energy_share", default=1.0)
        sort_rows.append(
            {
                "_idx": idx,
                "_label_rank": label_rank.get(str(row.get("physical_label", "")), 99),
                "_soc_drop": soc_drop if np.isfinite(soc_drop) else 999.0,
                "_saturation": saturation if np.isfinite(saturation) else 999.0,
                "_active_distance": abs(active - 0.20) if np.isfinite(active) else 999.0,
                "_fc_share_distance": abs(fc_share - 0.85) if np.isfinite(fc_share) else 999.0,
            }
        )
    sort_df = pd.DataFrame(sort_rows).sort_values(
        ["_label_rank", "_soc_drop", "_saturation", "_active_distance", "_fc_share_distance"],
        kind="mergesort",
    )
    best = df.loc[sort_df.iloc[0]["_idx"]]
    best_label = str(best.get("physical_label", classify_physical_baseline(best)))
    best_case = str(best.get("case_name", ""))
    accepted = best_label == "PASS_PHYSICAL_BASELINE"
    return {
        "recommended_fixed_mpc_baseline_before_dqn": best_case if accepted else "NONE_ACCEPTED",
        "least_bad_diagnostic_case": best_case,
        "accepted": bool(accepted),
        "label": best_label,
        "reason": (
            "Selected because it passed all physical baseline gates."
            if accepted
            else f"No candidate passed all physical baseline gates; least-bad diagnostic label is {best_label}."
        ),
    }


def write_code_cleanup_report(output_dir: str | Path = SIMPLIFIED_SPEC_NORM_OUTPUT_DIR) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "code_cleanup_report.md"
    lines = [
        "# 1 s OSQP-QP MPC Code Cleanup Report",
        "",
        "Current formal objective variant: `simplified_normalized_literature_v1`.",
        "",
        "## Deprecated Experiment Entrypoints",
        "",
        "- `raw_weight_retune`: removed from the active CLI workflow for the formal benchmark; historical output files are preserved.",
        "- `weight_sensitivity`: removed from the active CLI workflow for the formal benchmark; historical output files are preserved.",
        "- `physical_baseline_v2`, `soc_reserve_slack`, `fc_lowfreq_reference`, `fc_reference_tracking`, `normalized_objective_v1`, `terminal_soc_penalty_experiment`, and `ramp_soft_penalty_experiment`: not introduced into the current formal 1 s OSQP-QP entrypoint.",
        "",
        "## Retained Core Code",
        "",
        "- `src/main/mpc_solvers/mpc_qp_formulation.py`: retained as the OSQP-QP formulation module.",
        "- `src/main/benchmark_mpc_qp_osqp_1s.py`: retained as the 1 s offline benchmark runner.",
        "- Fuel-cell ramp is retained only as a hard constraint.",
        "- Objective terms retained: normalized H2, normalized SOC maintenance, normalized battery power penalty.",
        "",
        "## Scope Checks",
        "",
        "- 30 s mainline: not modified.",
        "- CasADi/IPOPT baseline: not modified.",
        "- outputs: preserved.",
        "- DQN: not modified and not trained.",
        "- LSTM: not trained.",
        "- Existing historical reports: preserved.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def evaluate_weight_validity(
    control_df: pd.DataFrame,
    time_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    config: QpMpcConfig,
    *,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, Any]:
    timing = timing_stats(time_df, config)
    performance = control_performance_metrics(control_df, metrics_df, config, soc_reference=soc_reference)
    controls = add_derived_control_columns(control_df, config)
    categories: list[str] = []
    reasons: list[str] = []

    if performance.get("load_exceeds_power_limit_count", 0) > 0 and (
        timing["solver_success_rate"] < 0.99 or timing["infeasible_count"] > 0
    ):
        categories.append("power_limit_insufficient")
        reasons.append(
            f"load_exceeds_power_limit_count={performance['load_exceeds_power_limit_count']} "
            f"above P_available_max={performance['P_available_max_kw']:.6f} kW"
        )

    if timing["solver_success_rate"] < 0.99 or timing["infeasible_count"] > 0:
        categories.append("solver_feasibility_failed")
        reasons.append(
            f"solver_success_rate={timing['solver_success_rate']:.6f}, infeasible_count={timing['infeasible_count']}"
        )

    if not timing["realtime_gate_passed"]:
        categories.append("solver_too_slow")
        reasons.append(
            "timing gate failed: "
            f"mean={timing['solve_time_ms_mean']:.3f} ms, p95={timing['solve_time_ms_p95']:.3f} ms, "
            f"p99={timing['solve_time_ms_p99']:.3f} ms, max={timing['solve_time_ms_max']:.3f} ms"
        )

    if (
        not np.isfinite(performance["power_balance_violation_max"])
        or performance["power_balance_violation_max"] > POWER_BALANCE_TOL_KW
        or performance["SOC_violation_count"] > 0
        or performance["battery_power_violation_count"] > 0
        or performance["fc_power_violation_count"] > 0
        or performance["fc_ramp_violation_count"] > 0
    ):
        categories.append("constraint_violation_failed")
        reasons.append(
            "constraint check failed: "
            f"balance_max={performance['power_balance_violation_max']:.6f} kW, "
            f"SOC_count={performance['SOC_violation_count']}, "
            f"battery_count={performance['battery_power_violation_count']}, "
            f"FC_count={performance['fc_power_violation_count']}, "
            f"ramp_count={performance['fc_ramp_violation_count']}"
        )

    final_drop_min = performance["SOC_final_minus_initial_min"]
    if np.isfinite(final_drop_min) and final_drop_min < SOC_FINAL_DROP_LIMIT:
        categories.append("SOC_sustain_failed")
        reasons.append(f"SOC_final_minus_initial_min={final_drop_min:.6f} < {SOC_FINAL_DROP_LIMIT}")

    p_batt_near_zero = performance["P_batt_near_zero_fraction_abs_le_1kw"]
    if np.isfinite(p_batt_near_zero) and p_batt_near_zero > BATTERY_UNUSED_FRACTION_LIMIT:
        categories.append("over_conservative_battery_unused")
        reasons.append(
            f"fraction(abs(P_batt)<=1 kW)={p_batt_near_zero:.6f} > {BATTERY_UNUSED_FRACTION_LIMIT}"
        )

    p_batt_near_limit = performance["P_batt_near_limit_fraction_abs_ge_95pct"]
    if np.isfinite(p_batt_near_limit) and p_batt_near_limit > BATTERY_LIMIT_STICK_FRACTION_LIMIT:
        categories.append("battery_power_limit_sticking")
        reasons.append(
            f"fraction(abs(P_batt)>=95% limit)={p_batt_near_limit:.6f} > {BATTERY_LIMIT_STICK_FRACTION_LIMIT}"
        )

    if not controls.empty:
        fc_delta = _numeric(controls.get("fc_delta_kw", pd.Series(dtype=float))).dropna()
        if not fc_delta.empty:
            near_ramp_fraction = float((fc_delta.abs() >= 0.90 * ramp_kw_per_step_from_rate(
                config.fuel_cell_ramp_rate_kw_per_s,
                dt_seconds=config.dt_seconds,
            )).mean())
            sign_changes = np.sign(fc_delta.to_numpy(dtype=float))
            sign_changes = sign_changes[sign_changes != 0.0]
            if len(sign_changes) > 2:
                jitter_fraction = float(np.mean(sign_changes[1:] != sign_changes[:-1]))
            else:
                jitter_fraction = 0.0
            if near_ramp_fraction > 0.10 and jitter_fraction > 0.60:
                categories.append("fc_high_frequency_jitter")
                reasons.append(
                    f"near_ramp_fraction={near_ramp_fraction:.6f}, fc_delta_sign_change_fraction={jitter_fraction:.6f}"
                )

    categories = list(dict.fromkeys(categories))
    weights_valid = len(categories) == 0
    return {
        "weights_valid": bool(weights_valid),
        "failure_categories": categories,
        "invalid_reasons": reasons,
        "timing": timing,
        "performance": performance,
    }


def _timing_distribution_table(time_df: pd.DataFrame, config: QpMpcConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames: list[tuple[str, Any, pd.DataFrame]] = [("overall", "all", time_df)]
    if not time_df.empty and "voyage_id" in time_df.columns:
        frames.extend(("voyage", voyage_id, group) for voyage_id, group in time_df.groupby("voyage_id", sort=True))
    sample_ms = float(config.dt_seconds) * 1000.0
    for scope, voyage_id, frame in frames:
        stats = timing_stats(frame, config)
        rows.append(
            {
                "scope": scope,
                "voyage_id": voyage_id,
                "steps": stats["steps"],
                "success_rate": stats["success_rate"],
                "mean_ms": stats["solve_time_ms_mean"],
                "median_ms": stats["solve_time_ms_median"],
                "p90_ms": stats["solve_time_ms_p90"],
                "p95_ms": stats["solve_time_ms_p95"],
                "p99_ms": stats["solve_time_ms_p99"],
                "max_ms": stats["solve_time_ms_max"],
                "real_time_factor_mean": stats["solve_time_ms_mean"] / sample_ms
                if np.isfinite(stats["solve_time_ms_mean"])
                else float("nan"),
                "real_time_factor_p99": stats["solve_time_ms_p99"] / sample_ms
                if np.isfinite(stats["solve_time_ms_p99"])
                else float("nan"),
                "realtime_gate_passed": stats["realtime_gate_passed"],
            }
        )
    return pd.DataFrame(rows)


def _failure_cases_table(control_df: pd.DataFrame, time_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "voyage_id",
        "step",
        "time_s",
        "status",
        "success",
        "total_controller_ms",
        "iterations",
        "load_total_kw",
        "P_fc_kw",
        "P_batt_kw",
        "SOC",
        "balance_violation_kw",
        "ramp_violation_kw",
        "soc_violation",
    ]
    if time_df.empty:
        return pd.DataFrame(columns=columns)
    mask = ~_success_mask(time_df)
    if "status" in time_df.columns:
        mask = mask | ~time_df["status"].astype(str).str.lower().str.startswith("solved")
    failures = time_df.loc[mask].copy()
    if failures.empty:
        return pd.DataFrame(columns=columns)
    merge_cols = [
        "voyage_id",
        "step",
        "load_total_kw",
        "P_fc_kw",
        "P_batt_kw",
        "SOC",
        "balance_violation_kw",
        "ramp_violation_kw",
        "soc_violation",
    ]
    available = [col for col in merge_cols if col in control_df.columns]
    failures = failures.merge(control_df[available], on=["voyage_id", "step"], how="left", suffixes=("", "_control"))
    for col in columns:
        if col not in failures.columns:
            failures[col] = np.nan
    return failures[columns]


def _constraint_violation_summary(performance: dict[str, Any], config: QpMpcConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "overall",
                "power_balance_violation_max_kw": performance["power_balance_violation_max"],
                "power_balance_tolerance_kw": POWER_BALANCE_TOL_KW,
                "SOC_violation_count": performance["SOC_violation_count"],
                "SOC_violation_tolerance": VIOLATION_TOL,
                "battery_power_violation_max_kw": performance["battery_power_violation_max"],
                "battery_power_violation_count": performance["battery_power_violation_count"],
                "fc_power_violation_max_kw": performance["fc_power_violation_max"],
                "fc_power_violation_count": performance["fc_power_violation_count"],
                "power_bound_violation_tolerance_kw": POWER_BOUND_TOL_KW,
                "fc_ramp_max_kw_per_step": performance["fc_ramp_max"],
                "fc_ramp_limit_kw_per_step": ramp_kw_per_step_from_rate(
                    config.fuel_cell_ramp_rate_kw_per_s,
                    dt_seconds=config.dt_seconds,
                ),
                "fc_ramp_violation_count": performance["fc_ramp_violation_count"],
            }
        ]
    )


def make_benchmark_plots(
    *,
    output_dir: Path,
    control_df: pd.DataFrame,
    time_df: pd.DataFrame,
    config: QpMpcConfig,
) -> None:
    if control_df.empty:
        return
    import matplotlib.pyplot as plt

    controls = add_derived_control_columns(control_df, config)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    voyages = list(controls["voyage_id"].drop_duplicates())
    n = max(len(voyages), 1)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))

    def subplots() -> tuple[Any, np.ndarray]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 3.8 * nrows), squeeze=False)
        return fig, axes.reshape(-1)

    def finish(fig: Any, axes: np.ndarray, filename: str) -> None:
        for ax in axes[len(voyages) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=160)
        plt.close(fig)

    fig, axes = subplots()
    for ax, voyage_id in zip(axes, voyages):
        group = controls[controls["voyage_id"].eq(voyage_id)]
        x = group["time_s"] / 60.0
        ax.plot(x, group["SOC"], linewidth=1.2)
        ax.axhline(config.soc_min, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(config.soc_max, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(DEFAULT_SOC_REFERENCE, color="tab:green", linestyle=":", linewidth=0.8)
        ax.set_title(str(voyage_id))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("SOC")
    finish(fig, axes, "soc_trajectory_by_voyage.png")

    def plot_power_overlay(filename: str) -> None:
        fig, axes = subplots()
        for ax, voyage_id in zip(axes, voyages):
            group = controls[controls["voyage_id"].eq(voyage_id)]
            x = group["time_s"] / 60.0
            ax.plot(x, group["load_total_kw"], label="load", linewidth=1.0, color="black")
            ax.plot(x, group["P_fc_kw"], label="P_fc", linewidth=1.0, color="tab:blue")
            ax.plot(x, group["P_batt_kw"], label="P_batt", linewidth=0.9, color="tab:orange")
            ax.axhline(0.0, color="0.7", linewidth=0.6)
            ax.set_title(str(voyage_id))
            ax.set_xlabel("time (min)")
            ax.set_ylabel("power (kW)")
            ax.legend(loc="best", fontsize=8)
        finish(fig, axes, filename)

    plot_power_overlay("power_split_by_voyage.png")
    plot_power_overlay("load_fc_batt_overlay_by_voyage.png")

    fig, axes = subplots()
    for ax, voyage_id in zip(axes, voyages):
        group = controls[controls["voyage_id"].eq(voyage_id)]
        ax.plot(group["time_s"] / 60.0, group["P_fc_kw"], linewidth=1.0, color="tab:blue")
        ax.axhline(config.fuel_cell_max_kw, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(config.fuel_cell_min_kw, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(str(voyage_id))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("P_fc (kW)")
    finish(fig, axes, "fc_power_by_voyage.png")

    fig, axes = subplots()
    for ax, voyage_id in zip(axes, voyages):
        group = controls[controls["voyage_id"].eq(voyage_id)]
        ax.plot(group["time_s"] / 60.0, group["P_batt_kw"], linewidth=1.0, color="tab:orange")
        ax.axhline(config.battery_discharge_max_kw, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(-config.battery_charge_max_kw, color="tab:red", linestyle="--", linewidth=0.8)
        ax.axhline(0.0, color="0.7", linewidth=0.6)
        ax.set_title(str(voyage_id))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("P_batt (kW)")
    finish(fig, axes, "batt_power_by_voyage.png")

    fig, axes = subplots()
    ramp_limit = ramp_kw_per_step_from_rate(config.fuel_cell_ramp_rate_kw_per_s, dt_seconds=config.dt_seconds)
    for ax, voyage_id in zip(axes, voyages):
        group = controls[controls["voyage_id"].eq(voyage_id)]
        ax.plot(group["time_s"] / 60.0, group["fc_ramp_kw"], linewidth=1.0, color="tab:purple")
        ax.axhline(ramp_limit, color="tab:red", linestyle="--", linewidth=0.8)
        ax.set_title(str(voyage_id))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("|delta P_fc| (kW/step)")
    finish(fig, axes, "fc_ramp_by_voyage.png")

    h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(config)
    h2_ref = float(h2_quad * config.fuel_cell_max_kw * config.fuel_cell_max_kw + h2_linear * config.fuel_cell_max_kw)
    fig, axes = subplots()
    for ax, voyage_id in zip(axes, voyages):
        group = controls[controls["voyage_id"].eq(voyage_id)]
        x = group["time_s"] / 60.0
        p_fc = _numeric(group["P_fc_kw"])
        p_batt = _numeric(group["P_batt_kw"])
        soc = _numeric(group["SOC"])
        h2_norm = (h2_quad * p_fc * p_fc + h2_linear * p_fc) / h2_ref
        soc_norm = ((soc - DEFAULT_SOC_REFERENCE) / float(config.soc_band)) ** 2
        batt_norm = (p_batt / float(config.battery_power_ref_kw)) ** 2
        ax.plot(x, h2_norm, label="H2_norm", linewidth=0.9)
        ax.plot(x, soc_norm, label="SOC_norm", linewidth=0.9)
        ax.plot(x, batt_norm, label="Batt_norm", linewidth=0.9)
        ax.set_title(str(voyage_id))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("normalized term")
        ax.legend(loc="best", fontsize=8)
    finish(fig, axes, "objective_terms_by_voyage.png")

    solve_times = _numeric(time_df["total_controller_ms"]).dropna() if "total_controller_ms" in time_df.columns else pd.Series(dtype=float)
    if not solve_times.empty:
        sorted_times = np.sort(solve_times.to_numpy(dtype=float))
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        plt.figure(figsize=(7, 4))
        plt.plot(sorted_times, cdf, linewidth=1.2)
        plt.axvline(REALTIME_THRESHOLDS_MS["p99"], color="tab:red", linestyle="--", linewidth=0.8)
        plt.xlabel("total controller time (ms)")
        plt.ylabel("CDF")
        plt.tight_layout()
        plt.savefig(fig_dir / "solve_time_cdf.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        if "voyage_id" in time_df.columns:
            data = [
                _numeric(group["total_controller_ms"]).dropna().to_numpy(dtype=float)
                for _, group in time_df.groupby("voyage_id", sort=True)
            ]
            labels = [str(v) for v, _ in time_df.groupby("voyage_id", sort=True)]
            plt.boxplot(data, tick_labels=labels, showfliers=False)
            plt.xticks(rotation=20, ha="right")
        else:
            plt.boxplot([sorted_times], tick_labels=["overall"], showfliers=False)
        plt.ylabel("total controller time (ms)")
        plt.tight_layout()
        plt.savefig(fig_dir / "solve_time_boxplot.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7, 4))
        solve_times.hist(bins=40)
        plt.xlabel("total controller time (ms)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(fig_dir / "solve_time_hist.png", dpi=160)
        plt.close()


def _build_report_lines(
    *,
    output_dir: Path,
    input_path: Path,
    config: QpMpcConfig,
    timing: dict[str, Any],
    performance: dict[str, Any],
    validity: dict[str, Any],
    max_steps_per_voyage: int | None,
    sensitivity_dir: Path | None = None,
) -> list[str]:
    ramp_limit = ramp_kw_per_step_from_rate(config.fuel_cell_ramp_rate_kw_per_s, dt_seconds=config.dt_seconds)
    capacity_ratio = OLD_BATTERY_CAPACITY_KWH / float(config.battery_capacity_kwh)
    valid_text = "valid" if validity["weights_valid"] else "invalid"
    reasons = validity["invalid_reasons"] or ["No validity failure was detected by the current gates."]
    continue_n180 = (
        "N=60 passes the 1 s timing gate, so N=180 can be benchmarked as the next solver-scaling test."
        if timing["realtime_gate_passed"] and validity["weights_valid"]
        else "Do not promote directly to N=180 as the next formal result until the current invalid reasons are handled; N=180 may still be run only as a solver-scaling diagnostic."
    )
    dqn_prep = (
        "The fixed QP-MPC baseline is acceptable for entering DQN-MPC preparation."
        if validity["weights_valid"]
        else "Do not enter DQN-MPC preparation from this run; fixed QP-MPC behavior is not accepted yet."
    )
    need_changes = (
        "No weight change is required by these gates."
        if validity["weights_valid"]
        else "Weight changes are required before accepting the fixed baseline; this report does not auto-select new final weights."
    )
    sensitivity_text = (
        f"Sensitivity output: `{sensitivity_dir}`" if sensitivity_dir is not None else "Sensitivity output: not run in this base report."
    )
    return [
        "# 1 s OSQP-QP MPC Benchmark Report",
        "",
        "Scope: this is only the 1 s OSQP-QP MPC benchmark on the offline reconstructed load profile. It does not modify the 30 s LSTM-MPC mainline, train LSTM, or train DQN.",
        "",
        "## Configuration",
        "",
        f"- Input parquet: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        "- Data source: natural-clipped cubic-spline reconstructed 1 s load profile from original 30 s real-vessel voyages.",
        "- Data caveat: this is not measured 1 s data and not online prediction evidence.",
        f"- Horizon: `{config.horizon}` steps",
        f"- Sample time: `{config.dt_seconds} s`",
        f"- SOC_ref: `{DEFAULT_SOC_REFERENCE}`",
        f"- Objective variant: `{config.objective_variant}`",
        f"- Battery capacity: `{config.battery_capacity_kwh} kWh`",
        f"- Battery power bound: `[-{config.battery_charge_max_kw}, {config.battery_discharge_max_kw}] kW`",
        f"- Battery normalization denominator P_batt_ref: `{config.battery_power_ref_kw} kW`",
        f"- Previous benchmark capacity for comparison: `{OLD_BATTERY_CAPACITY_KWH} kWh`",
        f"- SOC response scale versus 1806 kWh: `{capacity_ratio:.6f}x` faster for the same battery power and step time.",
        f"- Fuel-cell ramp: `{config.fuel_cell_ramp_rate_kw_per_s} kW/s = {ramp_limit} kW/step`",
        "- Fuel-cell ramp is a hard constraint in the formal simplified normalized variant.",
        f"- SOC band denominator: `{config.soc_band}`",
        f"- Weights: q_h2=`{config.q_h2}`, q_soc=`{config.q_soc}`, q_batt=`{config.q_batt}`, q_ramp=`{config.q_ramp}`, q_terminal=`{config.q_terminal_soc}`",
        f"- Max steps per voyage: `{max_steps_per_voyage}`",
        "",
        "## Solver Timing",
        "",
        f"- success_rate: `{timing['success_rate']:.6f}`",
        f"- infeasible_count: `{timing['infeasible_count']}`",
        f"- fallback_count: `{timing['fallback_count']}`",
        f"- mean / median / p90 / p95 / p99 / max solve time: `{timing['solve_time_ms_mean']:.3f}` / `{timing['solve_time_ms_median']:.3f}` / `{timing['solve_time_ms_p90']:.3f}` / `{timing['solve_time_ms_p95']:.3f}` / `{timing['solve_time_ms_p99']:.3f}` / `{timing['solve_time_ms_max']:.3f}` ms",
        f"- real_time_factor_mean: `{timing['real_time_factor_mean']:.6f}`",
        f"- real_time_factor_p99: `{timing['real_time_factor_p99']:.6f}`",
        f"- 1 s real-time gate passed: `{timing['realtime_gate_passed']}`",
        "",
        "## Control Metrics",
        "",
        f"- total_h2_cost / H2_total_kg: `{performance['total_h2_cost']:.6f}`",
        f"- SOC min/max/initial/final/final_minus_initial_mean: `{performance['SOC_min']:.6f}` / `{performance['SOC_max']:.6f}` / `{performance['SOC_initial']:.6f}` / `{performance['SOC_final']:.6f}` / `{performance['SOC_final_minus_initial']:.6f}`",
        f"- SOC final_minus_initial_min by voyage: `{performance['SOC_final_minus_initial_min']:.6f}`",
        f"- SOC mean_abs_deviation: `{performance['SOC_mean_abs_deviation']:.6f}`",
        f"- battery throughput/discharge/charge: `{performance['battery_throughput_kwh']:.6f}` / `{performance['battery_discharge_kwh']:.6f}` / `{performance['battery_charge_kwh']:.6f}` kWh",
        f"- P_batt mean_abs/max/min: `{performance['P_batt_mean_abs']:.6f}` / `{performance['P_batt_max']:.6f}` / `{performance['P_batt_min']:.6f}` kW",
        f"- P_batt abs<=1 kW fraction: `{performance['P_batt_near_zero_fraction_abs_le_1kw']:.6f}`",
        f"- P_fc mean/max/min: `{performance['P_fc_mean']:.6f}` / `{performance['P_fc_max']:.6f}` / `{performance['P_fc_min']:.6f}` kW",
        f"- fc_ramp_max / ramp_violation_count: `{performance['fc_ramp_max']:.6f}` kW/step / `{performance['fc_ramp_violation_count']}`",
        f"- power_balance_violation_max: `{performance['power_balance_violation_max']:.6f}` kW",
        f"- SOC/battery/FC power violation counts: `{performance['SOC_violation_count']}` / `{performance['battery_power_violation_count']}` / `{performance['fc_power_violation_count']}`",
        f"- battery/FC power violation max residuals: `{performance['battery_power_violation_max']:.6f}` / `{performance['fc_power_violation_max']:.6f}` kW; count tolerance `{POWER_BOUND_TOL_KW} kW`",
        "",
        "## Validity Decision",
        "",
        f"- Current weights are `{valid_text}`.",
        f"- Failure categories: `{';'.join(validity['failure_categories']) if validity['failure_categories'] else 'none'}`",
        f"- Need q_soc/q_batt/q_h2/terminal changes: {need_changes}",
        f"- Continue N=180: {continue_n180}",
        f"- Enter DQN-MPC preparation: {dqn_prep}",
        f"- {sensitivity_text}",
        "",
        "Invalid reasons:",
        *[f"- {reason}" for reason in reasons],
        "",
        "## Output Files",
        "",
        "- `solver_benchmark_summary.csv`",
        "- `solver_benchmark_by_voyage.csv`",
        "- `solver_timing_distribution.csv`",
        "- `solver_failure_cases.csv`",
        "- `constraint_violation_summary.csv`",
        "- `control_performance_summary.csv`",
        "- `objective_term_summary.csv`",
        "- `solver_config.json`",
        "- `figures/soc_trajectory_by_voyage.png`",
        "- `figures/power_split_by_voyage.png`",
        "- `figures/load_fc_batt_overlay_by_voyage.png`",
        "- `figures/fc_power_by_voyage.png`",
        "- `figures/batt_power_by_voyage.png`",
        "- `figures/fc_ramp_by_voyage.png`",
        "- `figures/objective_terms_by_voyage.png`",
        "- `figures/solve_time_cdf.png`",
        "- `figures/solve_time_boxplot.png`",
        "",
    ]


def write_benchmark_artifacts(
    *,
    output_dir: str | Path,
    config: QpMpcConfig,
    control_df: pd.DataFrame,
    time_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    input_path: str | Path,
    max_steps_per_voyage: int | None,
    make_plots: bool,
    report_filename: str = DEFAULT_REPORT_FILENAME,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
    osqp_available: bool = True,
    sensitivity_dir: str | Path | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_path)
    control_df = add_derived_control_columns(control_df, config)
    validity = evaluate_weight_validity(
        control_df,
        time_df,
        metrics_df,
        config,
        soc_reference=soc_reference,
    )
    timing = validity["timing"]
    performance = validity["performance"]

    control_path = out_dir / "control_timeseries.csv"
    time_path = out_dir / "solve_times.csv"
    metrics_path = out_dir / "voyage_metrics.csv"
    summary_path = out_dir / "solver_benchmark_summary.csv"
    by_voyage_path = out_dir / "solver_benchmark_by_voyage.csv"
    timing_dist_path = out_dir / "solver_timing_distribution.csv"
    failure_path = out_dir / "solver_failure_cases.csv"
    constraint_path = out_dir / "constraint_violation_summary.csv"
    control_perf_path = out_dir / "control_performance_summary.csv"
    objective_term_path = out_dir / "objective_term_summary.csv"
    objective_terms_by_voyage_path = out_dir / "objective_terms_by_voyage.csv"
    config_path = out_dir / "solver_config.json"
    qp_check_path = out_dir / QP_CHECK_FILENAME
    report_path = out_dir / report_filename
    objective_summary = objective_term_summary(control_df, config, soc_reference=soc_reference)
    objective_totals = objective_term_totals(control_df, config, soc_reference=soc_reference)
    objective_voyage = objective_terms_by_voyage(control_df, config, soc_reference=soc_reference)

    control_df.to_csv(control_path, index=False)
    time_df.to_csv(time_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    metrics_df.to_csv(by_voyage_path, index=False)
    _timing_distribution_table(time_df, config).to_csv(timing_dist_path, index=False)
    _failure_cases_table(control_df, time_df).to_csv(failure_path, index=False)
    _constraint_violation_summary(performance, config).to_csv(constraint_path, index=False)
    pd.DataFrame([performance]).to_csv(control_perf_path, index=False)
    objective_summary.to_csv(objective_term_path, index=False)
    objective_voyage.to_csv(objective_terms_by_voyage_path, index=False)

    summary = {
        "input_path": str(input_path),
        "output_dir": str(out_dir),
        "objective_variant": str(config.objective_variant),
        "horizon": int(config.horizon),
        "dt_seconds": float(config.dt_seconds),
        "old_battery_capacity_kwh": float(OLD_BATTERY_CAPACITY_KWH),
        "battery_capacity_kwh": float(config.battery_capacity_kwh),
        "battery_charge_max_kw": float(config.battery_charge_max_kw),
        "battery_discharge_max_kw": float(config.battery_discharge_max_kw),
        "battery_power_ref_kw": float(config.battery_power_ref_kw),
        "fuel_cell_ramp_rate_kw_per_s": float(config.fuel_cell_ramp_rate_kw_per_s),
        "fuel_cell_ramp_kw_per_step": ramp_kw_per_step_from_rate(
            config.fuel_cell_ramp_rate_kw_per_s,
            dt_seconds=config.dt_seconds,
        ),
        "soc_reference": float(soc_reference),
        "soc_band": float(config.soc_band),
        "q_h2": float(config.q_h2),
        "q_soc": float(config.q_soc),
        "q_batt": float(config.q_batt),
        "q_ramp": float(config.q_ramp),
        "q_terminal": float(config.q_terminal_soc),
        "max_steps_per_voyage": max_steps_per_voyage,
        "weights_valid": bool(validity["weights_valid"]),
        "failure_categories": ";".join(validity["failure_categories"]),
        "invalid_reasons": " | ".join(validity["invalid_reasons"]),
    }
    summary.update(timing)
    summary.update(
        {
            "mean_solve_time_ms": timing["solve_time_ms_mean"],
            "median_solve_time_ms": timing["solve_time_ms_median"],
            "p90_solve_time_ms": timing["solve_time_ms_p90"],
            "p95_solve_time_ms": timing["solve_time_ms_p95"],
            "p99_solve_time_ms": timing["solve_time_ms_p99"],
            "max_solve_time_ms": timing["solve_time_ms_max"],
        }
    )
    summary.update(performance)
    summary.update(objective_totals)
    summary["physical_label"] = classify_physical_baseline(summary)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    config_path.write_text(
        json.dumps(json_safe_config(config, osqp_available=osqp_available, soc_reference=soc_reference), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    probe_problem = build_qp_problem(
        config,
        load_forecast_kw=np.zeros(config.horizon, dtype=float),
        current_soc=soc_reference,
        prev_fc_kw=0.0,
        soc_reference=soc_reference,
    )
    write_qp_formulation_check(probe_problem, qp_check_path)

    if make_plots:
        make_benchmark_plots(output_dir=out_dir, control_df=control_df, time_df=time_df, config=config)

    sensitivity_path = Path(sensitivity_dir) if sensitivity_dir is not None else None
    report_path.write_text(
        "\n".join(
            _build_report_lines(
                output_dir=out_dir,
                input_path=input_path,
                config=config,
                timing=timing,
                performance=performance,
                validity=validity,
                max_steps_per_voyage=max_steps_per_voyage,
                sensitivity_dir=sensitivity_path,
            )
        ),
        encoding="utf-8",
    )

    return {
        "status": "completed",
        "report_path": str(report_path),
        "control_timeseries_path": str(control_path),
        "solve_times_path": str(time_path),
        "voyage_metrics_path": str(metrics_path),
        "solver_benchmark_summary_path": str(summary_path),
        "solver_benchmark_by_voyage_path": str(by_voyage_path),
        "solver_timing_distribution_path": str(timing_dist_path),
        "solver_failure_cases_path": str(failure_path),
        "constraint_violation_summary_path": str(constraint_path),
        "control_performance_summary_path": str(control_perf_path),
        "objective_term_summary_path": str(objective_term_path),
        "objective_terms_by_voyage_path": str(objective_terms_by_voyage_path),
        "solver_config_path": str(config_path),
        "qp_check_path": str(qp_check_path),
        "overall_success_rate": timing["success_rate"],
        "p99_total_ms": timing["solve_time_ms_p99"],
        "realtime_gate_passed": timing["realtime_gate_passed"],
        "weights_valid": validity["weights_valid"],
        "validity": validity,
    }


def _write_skip_report(
    *,
    output_dir: Path,
    config: QpMpcConfig,
    input_parquet: Path,
    reason: str,
    report_filename: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "solver_config.json"
    skip_path = output_dir / "SKIP_OSQP_NOT_INSTALLED.txt"
    report_path = output_dir / report_filename
    qp_check_path = output_dir / QP_CHECK_FILENAME
    config_path.write_text(
        json.dumps(json_safe_config(config, osqp_available=False), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    skip_path.write_text(reason.strip() + "\n", encoding="utf-8")

    probe_problem = build_qp_problem(
        config,
        load_forecast_kw=np.zeros(config.horizon, dtype=float),
        current_soc=DEFAULT_SOC_REFERENCE,
        prev_fc_kw=0.0,
        soc_reference=DEFAULT_SOC_REFERENCE,
    )
    write_qp_formulation_check(probe_problem, qp_check_path)

    lines = [
        "# OSQP QP MPC 1 s Benchmark Report",
        "",
        "Status: skipped because the Python `osqp` package is not installed in the current runtime.",
        "",
        f"- Input parquet expected at: `{input_parquet}`",
        f"- Output directory: `{output_dir}`",
        f"- Skip marker: `{skip_path}`",
        f"- QP formulation check: `{qp_check_path}`",
        "- Data definition: offline natural-clipped cubic-spline 1 s reconstruction; not measured 1 s data.",
        f"- Fuel-cell ramp source: `48 kW/s`; solver bound at dt=1 s: `{ramp_kw_per_step_from_rate(48.0, dt_seconds=1.0)} kW/step`.",
        "- Current 30 s CasADi/IPOPT baseline was not modified.",
        "",
        "No solve-time, H2, SOC, throughput, or violation metrics are reported from this skipped run.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "skipped_osqp_missing",
        "solver_config_path": str(config_path),
        "skip_path": str(skip_path),
        "report_path": str(report_path),
        "qp_check_path": str(qp_check_path),
    }


def run_benchmark(
    *,
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    horizon: int = 60,
    max_steps_per_voyage: int | None = None,
    make_plots: bool = True,
    battery_capacity_kwh: float = SPEC_BATTERY_CAPACITY_KWH,
    q_h2: float = 1.0,
    q_soc: float = 1.0,
    q_batt: float = 0.05,
    q_ramp: float = 0.0,
    q_terminal_soc: float = 0.0,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
    report_filename: str = DEFAULT_REPORT_FILENAME,
    sensitivity_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = default_config(
        horizon=horizon,
        battery_capacity_kwh=battery_capacity_kwh,
        q_h2=q_h2,
        q_soc=q_soc,
        q_batt=q_batt,
        q_ramp=q_ramp,
        q_terminal_soc=q_terminal_soc,
    )
    input_path = Path(input_parquet)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    osqp_module, osqp_error = _try_import_osqp()
    if osqp_module is None:
        return _write_skip_report(
            output_dir=out_dir,
            config=config,
            input_parquet=input_path,
            reason=f"Cannot import osqp: {osqp_error}",
            report_filename=report_filename,
        )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing benchmark input parquet: {input_path}")
    stale_skip = out_dir / "SKIP_OSQP_NOT_INSTALLED.txt"
    if stale_skip.exists():
        stale_skip.unlink()

    data = pd.read_parquet(input_path)
    required = {"voyage_id", "split", "time_s", "load_total_kw"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns: {sorted(missing)}")
    data = data[data["split"].astype(str).str.lower().eq("test")].copy()
    if data.empty:
        raise ValueError("Input parquet contains no split == test rows")

    control_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    dt_hours = config.dt_seconds / 3600.0
    ramp_limit = ramp_kw_per_step_from_rate(config.fuel_cell_ramp_rate_kw_per_s, dt_seconds=config.dt_seconds)

    for voyage_id, group in data.groupby("voyage_id", sort=True):
        group = group.sort_values("time_s", kind="mergesort").reset_index(drop=True)
        loads = group["load_total_kw"].to_numpy(dtype=float)
        times = group["time_s"].to_numpy(dtype=float)
        if len(loads) == 0:
            continue
        initial_soc = float(soc_reference)
        current_soc = float(initial_soc)
        prev_fc = float(np.clip(loads[0], config.fuel_cell_min_kw, config.fuel_cell_max_kw))
        voyage_h2 = 0.0
        steps = len(loads)
        if max_steps_per_voyage is not None:
            steps = min(steps, int(max_steps_per_voyage))

        persistent_solver = None
        use_persistent_solver = _can_reuse_osqp_solver(config)
        if use_persistent_solver:
            setup_forecast = loads[: config.horizon]
            if len(setup_forecast) < config.horizon:
                setup_forecast = np.pad(setup_forecast, (0, config.horizon - len(setup_forecast)), mode="edge")
            setup_problem = build_qp_problem(
                config,
                load_forecast_kw=setup_forecast,
                current_soc=current_soc,
                prev_fc_kw=prev_fc,
                soc_reference=soc_reference,
                include_diagnostics=False,
            )
            persistent_solver = _setup_persistent_osqp_solver(osqp_module, setup_problem)

        voyage_control_start = len(control_rows)
        voyage_time_start = len(time_rows)
        for step in range(steps):
            forecast = loads[step : step + config.horizon]
            if len(forecast) < config.horizon:
                forecast = np.pad(forecast, (0, config.horizon - len(forecast)), mode="edge")
            prev_fc_before = float(prev_fc)
            build_start = time.perf_counter()
            if persistent_solver is not None:
                lower, upper = _qp_bounds_for_step(
                    config,
                    load_forecast_kw=forecast,
                    current_soc=current_soc,
                    prev_fc_kw=prev_fc_before,
                )
                build_ms = (time.perf_counter() - build_start) * 1000.0
                result, solve_total_ms = _solve_with_persistent_osqp(
                    persistent_solver,
                    lower=lower,
                    upper=upper,
                )
            else:
                problem = build_qp_problem(
                    config,
                    load_forecast_kw=forecast,
                    current_soc=current_soc,
                    prev_fc_kw=prev_fc_before,
                    soc_reference=soc_reference,
                    include_diagnostics=False,
                )
                build_ms = (time.perf_counter() - build_start) * 1000.0
                result, solve_total_ms = _solve_problem(osqp_module, problem)
            total_ms = float(build_ms + solve_total_ms)
            status = str(result.info.status)
            solved = status.lower().startswith("solved")
            p_fc = np.nan
            p_batt = np.nan
            soc_next = np.nan
            if solved and result.x is not None:
                x = np.asarray(result.x, dtype=float)
                p_fc = float(x[0])
                p_batt = float(x[config.horizon])
                soc_next = advance_soc_from_battery_power(
                    config,
                    current_soc=current_soc,
                    battery_power_kw=p_batt,
                )
                current_soc = float(np.clip(soc_next, config.soc_min, config.soc_max))
                prev_fc = p_fc
                voyage_h2 += _h2_kg_for_step(config, p_fc)

            load_now = float(loads[step])
            balance_violation = float(abs(p_fc + p_batt - load_now)) if solved else np.nan
            fc_delta = float(p_fc - prev_fc_before) if solved else np.nan
            ramp_violation = float(max(0.0, abs(fc_delta) - ramp_limit)) if solved else np.nan
            soc_violation = (
                float(max(0.0, config.soc_min - current_soc, current_soc - config.soc_max))
                if solved
                else np.nan
            )
            p_fc_violation = (
                float(max(0.0, config.fuel_cell_min_kw - p_fc, p_fc - config.fuel_cell_max_kw))
                if solved
                else np.nan
            )
            p_batt_violation = (
                float(max(0.0, -config.battery_charge_max_kw - p_batt, p_batt - config.battery_discharge_max_kw))
                if solved
                else np.nan
            )
            time_rows.append(
                {
                    "voyage_id": voyage_id,
                    "step": int(step),
                    "time_s": float(times[step]),
                    "status": status,
                    "success": bool(solved),
                    "build_ms": float(build_ms),
                    "setup_plus_solve_ms": float(solve_total_ms),
                    "total_controller_ms": total_ms,
                    "iterations": int(getattr(result.info, "iter", -1)),
                    "objective": float(getattr(result.info, "obj_val", np.nan)),
                    "pri_res": float(getattr(result.info, "prim_res", np.nan)),
                    "dua_res": float(getattr(result.info, "dual_res", np.nan)),
                }
            )
            control_rows.append(
                {
                    "voyage_id": voyage_id,
                    "step": int(step),
                    "time_s": float(times[step]),
                    "load_total_kw": load_now,
                    "P_fc_kw": p_fc,
                    "P_batt_kw": p_batt,
                    "SOC": current_soc,
                    "status": status,
                    "success": bool(solved),
                    "prev_fc_before_kw": prev_fc_before,
                    "fc_delta_kw": fc_delta,
                    "fc_ramp_kw": abs(fc_delta) if solved else np.nan,
                    "fc_ramp_kw_per_s": abs(fc_delta) / float(config.dt_seconds) if solved else np.nan,
                    "balance_violation_kw": balance_violation,
                    "ramp_violation_kw": ramp_violation,
                    "soc_violation": soc_violation,
                    "P_fc_violation_kw": p_fc_violation,
                    "P_batt_violation_kw": p_batt_violation,
                    "h2_kg_step": _h2_kg_for_step(config, p_fc) if solved else np.nan,
                    "battery_discharge_kwh_step": max(p_batt, 0.0) * dt_hours if solved else np.nan,
                    "battery_charge_kwh_step": max(-p_batt, 0.0) * dt_hours if solved else np.nan,
                }
            )

        voyage_controls = pd.DataFrame(control_rows[voyage_control_start:])
        voyage_times = pd.DataFrame(time_rows[voyage_time_start:])
        if not voyage_controls.empty:
            metrics_rows.append(
                build_voyage_metrics(
                    voyage_id=voyage_id,
                    voyage_controls=voyage_controls,
                    voyage_times=voyage_times,
                    config=config,
                    h2_total_kg=voyage_h2,
                    initial_soc=initial_soc,
                    soc_reference=soc_reference,
                )
            )

    control_df = pd.DataFrame(control_rows)
    time_df = pd.DataFrame(time_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    return write_benchmark_artifacts(
        output_dir=out_dir,
        config=config,
        control_df=control_df,
        time_df=time_df,
        metrics_df=metrics_df,
        input_path=input_path,
        max_steps_per_voyage=max_steps_per_voyage,
        make_plots=make_plots,
        report_filename=report_filename,
        soc_reference=soc_reference,
        osqp_available=True,
        sensitivity_dir=sensitivity_dir,
    )


def _recompute_voyage_metrics_from_controls(
    control_df: pd.DataFrame,
    time_df: pd.DataFrame,
    config: QpMpcConfig,
    *,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> pd.DataFrame:
    if control_df.empty or "voyage_id" not in control_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    controls = add_derived_control_columns(control_df, config)
    times = time_df.copy()
    for voyage_id, group in controls.groupby("voyage_id", sort=True):
        voyage_times = times[times["voyage_id"].eq(voyage_id)].copy() if "voyage_id" in times.columns else pd.DataFrame()
        h2_total = _sum(group["h2_kg_step"]) if "h2_kg_step" in group.columns else float(
            sum(_h2_kg_for_step(config, value) for value in _numeric(group["P_fc_kw"]).dropna())
        )
        rows.append(
            build_voyage_metrics(
                voyage_id=voyage_id,
                voyage_controls=group,
                voyage_times=voyage_times,
                config=config,
                h2_total_kg=h2_total,
                initial_soc=soc_reference,
                soc_reference=soc_reference,
            )
        )
    return pd.DataFrame(rows)


def _format_float(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def select_recommended_raw_baseline(summary_df: pd.DataFrame) -> dict[str, Any]:
    if summary_df.empty:
        return {
            "recommended_fixed_mpc_baseline_before_dqn": "NONE_ACCEPTED",
            "least_bad_diagnostic_case": "",
            "accepted": False,
            "label": "NO_CASES",
            "reason": "No candidate summary rows are available.",
        }
    df = summary_df.copy()
    if "physical_label" not in df.columns:
        df["physical_label"] = df.apply(classify_physical_baseline, axis=1)
    label_rank = {
        "PASS_PHYSICAL_BASELINE": 0,
        "BORDERLINE": 1,
        "FAIL_BATTERY_UNUSED": 2,
        "FAIL_SOC_DROP": 3,
        "FAIL_BATTERY_OVERUSE": 4,
        "FAIL_FC_NOT_MAIN_SOURCE": 5,
        "FAIL_CONSTRAINT": 6,
        "FAIL_SOC_VIOLATION": 7,
        "FAIL_SOLVER": 8,
    }
    df["_label_rank"] = df["physical_label"].map(label_rank).fillna(99)
    for col in [
        "battery_saturation_fraction_abs_gt_300kW",
        "battery_saturation_fraction_abs_gt_300kw",
        "SOC_drop_max_by_voyage",
        "battery_active_fraction_abs_gt_5kW",
        "battery_active_fraction_abs_gt_5kw",
        "H2_total_kg",
        "solve_time_ms_p99",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    active_col = "battery_active_fraction_abs_gt_5kW" if "battery_active_fraction_abs_gt_5kW" in df.columns else "battery_active_fraction_abs_gt_5kw"
    saturation_col = (
        "battery_saturation_fraction_abs_gt_300kW"
        if "battery_saturation_fraction_abs_gt_300kW" in df.columns
        else "battery_saturation_fraction_abs_gt_300kw"
    )
    sort_cols = ["_label_rank"]
    ascending = [True]
    if saturation_col in df.columns:
        sort_cols.append(saturation_col)
        ascending.append(True)
    if "SOC_drop_max_by_voyage" in df.columns:
        sort_cols.append("SOC_drop_max_by_voyage")
        ascending.append(True)
    if active_col in df.columns:
        df["_active_sort"] = -df[active_col].fillna(-1.0)
        sort_cols.append("_active_sort")
        ascending.append(True)
    if "H2_total_kg" in df.columns:
        sort_cols.append("H2_total_kg")
        ascending.append(True)
    if "solve_time_ms_p99" in df.columns:
        sort_cols.append("solve_time_ms_p99")
        ascending.append(True)
    ranked = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    acceptable = ranked[ranked["physical_label"].isin(["PASS_PHYSICAL_BASELINE", "BORDERLINE"])]
    if not acceptable.empty:
        row = acceptable.iloc[0]
        return {
            "recommended_fixed_mpc_baseline_before_dqn": str(row["case_name"]),
            "least_bad_diagnostic_case": str(row["case_name"]),
            "accepted": bool(row["physical_label"] == "PASS_PHYSICAL_BASELINE"),
            "label": str(row["physical_label"]),
            "reason": "Selected from PASS/BORDERLINE candidates using saturation, SOC drop, battery activity, H2, and timing as tie breakers.",
        }
    row = ranked.iloc[0]
    return {
        "recommended_fixed_mpc_baseline_before_dqn": "NONE_ACCEPTED",
        "least_bad_diagnostic_case": str(row["case_name"]),
        "accepted": False,
        "label": str(row["physical_label"]),
        "reason": "No candidate passed the physical baseline gates; the listed case is diagnostic only, not a promoted baseline.",
    }


def _raw_candidate_table_lines(summary_df: pd.DataFrame) -> list[str]:
    lines = [
        "| case | label | q_batt | q_soc | q_terminal | p99 ms | H2 kg | batt active >5kW | batt sat >300kW | SOC drop max | FC energy share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if summary_df.empty:
        lines.append("| none | NO_CASES |  |  |  |  |  |  |  |  |  |")
        return lines
    for _, row in summary_df.iterrows():
        active = _row_float(row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw")
        saturation = _row_float(row, "battery_saturation_fraction_abs_gt_300kW", "battery_saturation_fraction_abs_gt_300kw")
        lines.append(
            f"| {row.get('case_name', '')} | {row.get('physical_label', '')} | "
            f"{_format_float(row.get('q_batt'), 8)} | {_format_float(row.get('q_soc'), 3)} | {_format_float(row.get('q_terminal'), 3)} | "
            f"{_format_float(row.get('solve_time_ms_p99'), 3)} | {_format_float(row.get('H2_total_kg'), 6)} | "
            f"{_format_float(active, 6)} | {_format_float(saturation, 6)} | "
            f"{_format_float(row.get('SOC_drop_max_by_voyage'), 6)} | {_format_float(row.get('fc_energy_share'), 6)} |"
        )
    return lines


def _write_raw_weight_candidate_decision(
    *,
    path: Path,
    summary_df: pd.DataFrame,
    recommendation: dict[str, Any],
    audit_paths: dict[str, str],
    priority_only: bool,
) -> None:
    lines = [
        "# Raw Weight Candidate Decision",
        "",
        "Scope: unnormalized raw-objective retune for the 1 s OSQP-QP MPC benchmark. No 30 s mainline, LSTM, or DQN code is changed.",
        "",
        f"- Priority-only run: `{priority_only}`",
        f"- Objective scale audit: `{audit_paths['md_path']}`",
        f"- recommended_fixed_mpc_baseline_before_dqn: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}`",
        f"- least_bad_diagnostic_case: `{recommendation['least_bad_diagnostic_case']}`",
        f"- selected_label: `{recommendation['label']}`",
        f"- decision_reason: {recommendation['reason']}",
        "",
        "## Candidate Table",
        "",
        *_raw_candidate_table_lines(summary_df),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_raw_weight_retune_report(
    *,
    path: Path,
    output_dir: Path,
    input_parquet: Path,
    summary_df: pd.DataFrame,
    by_voyage_df: pd.DataFrame,
    recommendation: dict[str, Any],
    audit_paths: dict[str, str],
    priority_only: bool,
) -> None:
    old_row = summary_df[summary_df["case_name"].eq("case_raw_base_old")].iloc[0] if not summary_df.empty and summary_df["case_name"].eq("case_raw_base_old").any() else None
    rec_case = recommendation["recommended_fixed_mpc_baseline_before_dqn"]
    rec_row = None
    if rec_case != "NONE_ACCEPTED" and not summary_df.empty and summary_df["case_name"].eq(rec_case).any():
        rec_row = summary_df[summary_df["case_name"].eq(rec_case)].iloc[0]
    elif not summary_df.empty and recommendation.get("least_bad_diagnostic_case") and summary_df["case_name"].eq(recommendation["least_bad_diagnostic_case"]).any():
        rec_row = summary_df[summary_df["case_name"].eq(recommendation["least_bad_diagnostic_case"])].iloc[0]

    old_active = _row_float(old_row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw") if old_row is not None else float("nan")
    rec_active = _row_float(rec_row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw") if rec_row is not None else float("nan")
    rec_sat = _row_float(rec_row, "battery_saturation_fraction_abs_gt_300kW", "battery_saturation_fraction_abs_gt_300kw") if rec_row is not None else float("nan")
    rec_case_label = str(rec_row.get("case_name", "")) if rec_row is not None else ""

    lines = [
        "# Raw Weight Retune For Physical Baseline",
        "",
        "Scope: this report covers only the 1 s OSQP-QP MPC benchmark on offline natural-clipped cubic-spline reconstructed load. It is not measured 1 s data and it is not an online LSTM forecast validation.",
        "",
        "## Run Context",
        "",
        f"- Input parquet: `{input_parquet}`",
        f"- Output directory: `{output_dir}`",
        f"- Priority-only run: `{priority_only}`",
        "- Fixed physical settings: `dt=1 s`, `N=60`, `E_batt=277.2 kWh`, `P_fc in [0,560] kW`, `P_batt in [-350,350] kW`, `SOC in [0.2,0.8]`, `ramp=48 kW/s = 48 kW/step`.",
        "- Objective form is raw and unnormalized; no denominator scaling or normalized surrogate was introduced.",
        "",
        "## Recommendation",
        "",
        f"- recommended_fixed_mpc_baseline_before_dqn: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}`",
        f"- least_bad_diagnostic_case: `{recommendation['least_bad_diagnostic_case']}`",
        f"- selected_label: `{recommendation['label']}`",
        f"- decision_reason: {recommendation['reason']}",
        "",
        "## Candidate Summary",
        "",
        *_raw_candidate_table_lines(summary_df),
        "",
        "## Required Questions",
        "",
        "1. Why no normalization: the requested experiment is a raw-objective retune. Changing to a normalized objective would be a different controller formulation, so it is deliberately not done here.",
        f"2. Term scales: the numeric audit is in `{audit_paths['csv_path']}` and `{audit_paths['md_path']}`; it compares H2, battery, SOC, and ramp terms on the raw units.",
        f"3. Why old `q_batt=0.03` is bad: the audit shows `50 kW` battery power costs `75` in one step, while the single-step H2 term is much smaller; closed-loop old-case active fraction >5 kW is `{_format_float(old_active, 6)}`.",
        "4. Approximate `q_batt` magnitude: the tested raw range is `1e-5` to `1e-7`; this is the order where `P_batt^2` stops dominating the objective by construction.",
        "5. Whether `q_soc=400` is too strong: it is not decided by coefficient size alone. With `E_batt=277.2 kWh`, the closed-loop SOC drop and deviation metrics are the evidence.",
        "6. Whether terminal SOC is needed: terminal candidates are included; if their physical label and SOC-drop metrics improve without suppressing battery participation, they are preferred.",
        "7. Whether `q_ramp` should remain: the hard `48 kW/step` ramp constraint enforces feasibility. The soft `q_ramp` term only smooths FC motion and can be removed only if closed-loop ramp plots remain acceptable.",
        "8. Lower `q_batt` and battery participation: compare `battery_active_fraction_abs_gt_5kW`, `P_batt_mean_abs`, and throughput across the table; the answer is based on these closed-loop outputs, not expectation.",
        f"9. Solver realtime: the report uses mean, median, p90, p95, p99, max, success rate, infeasible count, fallback count, and real-time factors; see `raw_weight_candidate_summary.csv`.",
        "10. SOC safety: accepted candidates must keep `SOC` within `[0.2,0.8]` and avoid excessive voyage-level SOC drop.",
        "11. FC as main source: accepted candidates must retain high `fc_energy_share`; otherwise they fail `FAIL_FC_NOT_MAIN_SOURCE`.",
        f"12. Battery overuse: the selected/diagnostic case `{rec_case_label}` has saturation fraction >300 kW of `{_format_float(rec_sat, 6)}` and active fraction >5 kW of `{_format_float(rec_active, 6)}`.",
        "13. Recommended case: the report field `recommended_fixed_mpc_baseline_before_dqn` is only written here; no global config is changed.",
        "14. Why this is not optimal: this is a limited fixed-weight benchmark, not a global optimization or automated tuning study.",
        "15. Future DQN dynamic weights: DQN remains future work after a fixed physical baseline is accepted; this run does not train or modify DQN.",
        "",
        "## Output Files",
        "",
        "- `objective_term_scale_audit.md`",
        "- `objective_term_scale_audit.csv`",
        "- `raw_weight_candidate_summary.csv`",
        "- `raw_weight_candidate_by_voyage.csv`",
        "- `raw_weight_candidate_decision.md`",
        "- each case directory: `solver_benchmark_summary.csv`, `solver_benchmark_by_voyage.csv`, `control_performance_summary.csv`, `constraint_violation_summary.csv`, `objective_term_summary.csv`, `solver_config.json`, and required figures.",
        "",
        f"Voyage rows included in aggregate by-voyage table: `{len(by_voyage_df)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _simplified_candidate_table_lines(summary_df: pd.DataFrame) -> list[str]:
    lines = [
        "| case | label | success | mean ms | p99 ms | H2 kg | batt throughput kWh | active >5kW | SOC drop max | FC share |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if summary_df.empty:
        lines.append("| none | NO_CASES |  |  |  |  |  |  |  |  |")
        return lines
    for _, row in summary_df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case_name", "")),
                    str(row.get("physical_label", "")),
                    _format_float(_row_float(row, "solver_success_rate", "success_rate"), 6),
                    _format_float(_row_float(row, "solve_time_ms_mean", "mean_solve_time_ms"), 3),
                    _format_float(_row_float(row, "solve_time_ms_p99", "p99_solve_time_ms"), 3),
                    _format_float(_row_float(row, "H2_total_kg", "total_h2_cost"), 6),
                    _format_float(_row_float(row, "battery_throughput_kwh", "battery_throughput_kWh"), 6),
                    _format_float(_row_float(row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw"), 6),
                    _format_float(_row_float(row, "SOC_drop_max_by_voyage"), 6),
                    _format_float(_row_float(row, "fc_energy_share"), 6),
                ]
            )
            + " |"
        )
    return lines


def _write_simplified_spec_norm_candidate_decision(
    *,
    path: Path,
    summary_df: pd.DataFrame,
    recommendation: dict[str, Any],
    load_check_paths: dict[str, str],
    objective_check_paths: dict[str, str],
    cleanup_report_path: str,
) -> None:
    lines = [
        "# Simplified Spec-Normalized Candidate Decision",
        "",
        "Scope: fixed-weight 1 s OSQP-QP MPC benchmark on offline natural-clipped spline load reconstruction. No DQN, LSTM, 30 s mainline, or CasADi/IPOPT baseline changes.",
        "",
        f"- Formal objective variant: `{SPEC_OBJECTIVE_VARIANT}`",
        f"- Battery capacity: `{SPEC_BATTERY_CAPACITY_KWH} kWh`",
        f"- Battery power bound and denominator: `{SPEC_BATTERY_POWER_MAX_KW} kW`",
        f"- Fuel-cell ramp hard constraint: `48 kW/s = {ramp_kw_per_step_from_rate(48.0, dt_seconds=1.0)} kW/step`",
        f"- Objective check: `{objective_check_paths['md_path']}`",
        f"- Load feasibility check: `{load_check_paths['md_path']}`",
        f"- Code cleanup report: `{cleanup_report_path}`",
        "",
        "## Candidate Table",
        "",
        *_simplified_candidate_table_lines(summary_df),
        "",
        "## Decision",
        "",
        f"- recommended_fixed_mpc_baseline_before_dqn: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}`",
        f"- least_bad_diagnostic_case: `{recommendation['least_bad_diagnostic_case']}`",
        f"- accepted: `{recommendation['accepted']}`",
        f"- selected_label: `{recommendation['label']}`",
        f"- reason: {recommendation['reason']}",
        "",
        "No global config is modified by this decision file.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_simplified_spec_norm_report(
    *,
    path: Path,
    output_dir: Path,
    input_parquet: Path,
    summary_df: pd.DataFrame,
    by_voyage_df: pd.DataFrame,
    recommendation: dict[str, Any],
    load_check_paths: dict[str, str],
    objective_check_paths: dict[str, str],
    cleanup_report_path: str,
) -> None:
    load_check = pd.read_csv(load_check_paths["csv_path"]).iloc[0].to_dict()
    rec_case = str(recommendation.get("recommended_fixed_mpc_baseline_before_dqn", "NONE_ACCEPTED"))
    diagnostic_case = str(recommendation.get("least_bad_diagnostic_case", ""))
    selected_case = rec_case if rec_case != "NONE_ACCEPTED" else diagnostic_case
    rec_row = None
    if selected_case and not summary_df.empty and "case_name" in summary_df.columns:
        matches = summary_df[summary_df["case_name"].astype(str).eq(selected_case)]
        if not matches.empty:
            rec_row = matches.iloc[0]
    rec_label = str(rec_row.get("physical_label", "NO_CASES")) if rec_row is not None else "NO_CASES"
    rec_active = _row_float(rec_row, "battery_active_fraction_abs_gt_5kW", "battery_active_fraction_abs_gt_5kw") if rec_row is not None else float("nan")
    rec_soc_min = _row_float(rec_row, "SOC_min") if rec_row is not None else float("nan")
    rec_soc_max = _row_float(rec_row, "SOC_max") if rec_row is not None else float("nan")
    rec_fc_share = _row_float(rec_row, "fc_energy_share") if rec_row is not None else float("nan")
    rec_success = _row_float(rec_row, "solver_success_rate", "success_rate") if rec_row is not None else float("nan")
    rec_p99 = _row_float(rec_row, "solve_time_ms_p99", "p99_solve_time_ms") if rec_row is not None else float("nan")
    p_available = float(load_check.get("P_available_max_kw", SPEC_BATTERY_POWER_MAX_KW + 560.0))
    load_exceeds = int(load_check.get("num_steps_load_exceeds_available", load_check.get("num_steps_load_exceeds_698p6", 0)))
    lines = [
        "# Simplified Spec-Normalized Objective 1 s OSQP-QP MPC Report",
        "",
        "Scope: offline 1 s natural-clipped cubic-spline benchmark only. This is not measured 1 s data and not online LSTM forecasting evidence.",
        "",
        "## Files",
        "",
        f"- Input parquet: `{input_parquet}`",
        f"- Output directory: `{output_dir}`",
        f"- Objective check: `{objective_check_paths['md_path']}`",
        f"- Load feasibility check: `{load_check_paths['md_path']}`",
        f"- Cleanup report: `{cleanup_report_path}`",
        f"- Candidate summary: `{output_dir / 'simplified_spec_norm_candidate_summary.csv'}`",
        f"- Candidate by-voyage summary: `{output_dir / 'simplified_spec_norm_candidate_by_voyage.csv'}`",
        f"- Candidate decision: `{output_dir / 'simplified_spec_norm_candidate_decision.md'}`",
        "",
        "## Candidate Table",
        "",
        *_simplified_candidate_table_lines(summary_df),
        "",
        "## Required Answers",
        "",
        "1. `P_batt_ref` uses `346.5 kW` because the current formal pack basis is `10 x 69.3 kWh = 693 kWh` and `0.5C` gives `346.5 kW`.",
        "2. The old `138.6 kW` denominator came from the previous `277.2 kWh` scaled pack and is legacy only for this formal run.",
        "3. `P_batt_max` is also `346.5 kW` so that the physical bound and the normalized battery denominator use the same scaled battery system basis.",
        "4. The fuel-cell ramp soft penalty is deleted because ramp is already enforced by hard constraints and the requested formal objective keeps only H2, SOC, and battery-use terms.",
        "5. The terminal SOC penalty is deleted because this formal baseline uses stage SOC maintenance only; no terminal SOC soft penalty or terminal SOC constraint is introduced.",
        "6. Ramp is retained through hard constraints: `|P_fc[0] - P_fc_prev| <= 48` and `|P_fc[k] - P_fc[k-1]| <= 48`.",
        "7. The objective matches a common normalized three-term literature form: H2 term, SOC maintenance term, and battery power penalty.",
        "8. Denominators are fixed physical scales: `P_fc_max=560 kW`, `P_batt_ref=346.5 kW`, `SOC_band=0.05`, and `m_H2_ref=alpha*560^2+beta*560`.",
        "9. The normalized problem remains a convex QP because all quadratic weights and denominators are nonnegative fixed constants and all constraints are linear.",
        f"10. Load feasibility under `P_fc_max + P_batt_max = {_format_float(p_available, 1)} kW`: max load `{_format_float(load_check.get('P_load_max'), 6)} kW`, p99 `{_format_float(load_check.get('P_load_p99'), 6)} kW`, exceedance count `{load_exceeds}`.",
        f"11. Most suitable case by the current gates: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}`; diagnostic case if none accepted: `{diagnostic_case}`.",
        f"12. Selected/diagnostic battery active fraction `abs(P_batt)>5 kW`: `{_format_float(rec_active, 6)}`.",
        f"13. Selected/diagnostic SOC min/max: `{_format_float(rec_soc_min, 6)}` / `{_format_float(rec_soc_max, 6)}`.",
        f"14. Selected/diagnostic FC energy share: `{_format_float(rec_fc_share, 6)}`.",
        f"15. Selected/diagnostic OSQP success and p99 solve time: `{_format_float(rec_success, 6)}` / `{_format_float(rec_p99, 3)} ms`.",
        f"16. Recommended fixed MPC baseline: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}` with label `{recommendation['label']}`.",
        f"17. Usable before DQN dynamic weighting: `{recommendation['accepted']}`. If false, do not proceed to DQN from this benchmark.",
        "18. The result is still based on offline spline 1 s reconstruction, not true measured 1 s data.",
        "",
        "## Decision",
        "",
        f"- recommended_fixed_mpc_baseline_before_dqn: `{recommendation['recommended_fixed_mpc_baseline_before_dqn']}`",
        f"- least_bad_diagnostic_case: `{diagnostic_case}`",
        f"- accepted: `{recommendation['accepted']}`",
        f"- selected_label: `{rec_label}`",
        f"- reason: {recommendation['reason']}",
        "",
        f"Voyage rows included in aggregate by-voyage table: `{len(by_voyage_df)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_simplified_spec_norm_benchmark(
    *,
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    output_dir: str | Path = SIMPLIFIED_SPEC_NORM_OUTPUT_DIR,
    horizon: int = 60,
    max_steps_per_voyage: int | None = None,
    make_plots: bool = True,
    battery_capacity_kwh: float = SPEC_BATTERY_CAPACITY_KWH,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    input_path = Path(input_parquet)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config = default_config(horizon=horizon, battery_capacity_kwh=battery_capacity_kwh)
    cleanup_report_path = write_code_cleanup_report(out_dir)
    objective_check_paths = write_simplified_normalized_objective_check(
        output_dir=out_dir,
        config=base_config,
        soc_reference=soc_reference,
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Missing benchmark input parquet: {input_path}")
    data = pd.read_parquet(input_path)
    load_check_paths = write_load_feasibility_check(data, base_config, output_dir=out_dir)

    summary_frames: list[pd.DataFrame] = []
    by_voyage_frames: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    for case in SIMPLIFIED_SPEC_NORM_CASES:
        case_name = str(case["case_name"])
        case_dir = out_dir / case_name
        print(f"[simplified_spec_norm] running {case_name}", flush=True)
        result = run_benchmark(
            input_parquet=input_path,
            output_dir=case_dir,
            horizon=horizon,
            max_steps_per_voyage=max_steps_per_voyage,
            make_plots=make_plots,
            battery_capacity_kwh=battery_capacity_kwh,
            q_h2=float(case["q_h2"]),
            q_soc=float(case["q_soc"]),
            q_batt=float(case["q_batt"]),
            q_ramp=0.0,
            q_terminal_soc=0.0,
            soc_reference=soc_reference,
            report_filename=f"REPORT_{case_name}.md",
        )
        results.append({"case_name": case_name, **result})
        summary_path = case_dir / "solver_benchmark_summary.csv"
        by_voyage_path = case_dir / "solver_benchmark_by_voyage.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "physical_label" not in summary.columns:
                summary["physical_label"] = summary.apply(classify_physical_baseline, axis=1)
            summary.insert(0, "case_name", case_name)
            summary_frames.append(summary)
        if by_voyage_path.exists():
            by_voyage = pd.read_csv(by_voyage_path)
            by_voyage.insert(0, "case_name", case_name)
            by_voyage_frames.append(by_voyage)
        print(f"[simplified_spec_norm] finished {case_name}", flush=True)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_voyage_df = pd.concat(by_voyage_frames, ignore_index=True) if by_voyage_frames else pd.DataFrame()
    if not summary_df.empty:
        summary_df["physical_label"] = summary_df.apply(classify_physical_baseline, axis=1)
    recommendation = select_recommended_simplified_baseline(summary_df)

    summary_path = out_dir / "simplified_spec_norm_candidate_summary.csv"
    by_voyage_path = out_dir / "simplified_spec_norm_candidate_by_voyage.csv"
    decision_path = out_dir / "simplified_spec_norm_candidate_decision.md"
    report_path = out_dir / SIMPLIFIED_SPEC_NORM_REPORT_FILENAME
    summary_df.to_csv(summary_path, index=False)
    by_voyage_df.to_csv(by_voyage_path, index=False)
    _write_simplified_spec_norm_candidate_decision(
        path=decision_path,
        summary_df=summary_df,
        recommendation=recommendation,
        load_check_paths=load_check_paths,
        objective_check_paths=objective_check_paths,
        cleanup_report_path=cleanup_report_path,
    )
    _write_simplified_spec_norm_report(
        path=report_path,
        output_dir=out_dir,
        input_parquet=input_path,
        summary_df=summary_df,
        by_voyage_df=by_voyage_df,
        recommendation=recommendation,
        load_check_paths=load_check_paths,
        objective_check_paths=objective_check_paths,
        cleanup_report_path=cleanup_report_path,
    )
    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "code_cleanup_report_path": cleanup_report_path,
        "objective_check_path": objective_check_paths["md_path"],
        "load_feasibility_check_csv_path": load_check_paths["csv_path"],
        "load_feasibility_check_md_path": load_check_paths["md_path"],
        "candidate_summary_path": str(summary_path),
        "candidate_by_voyage_path": str(by_voyage_path),
        "candidate_decision_path": str(decision_path),
        "report_path": str(report_path),
        "recommendation": recommendation,
        "case_results": results,
    }


def run_raw_weight_retune(
    *,
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    output_dir: str | Path = RAW_WEIGHT_RETUNE_OUTPUT_DIR,
    base_output_dir: str | Path | None = EBATT277P2_OUTPUT_DIR,
    horizon: int = 60,
    max_steps_per_voyage: int | None = None,
    battery_capacity_kwh: float = EBATT277P2_BATTERY_CAPACITY_KWH,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
    priority_only: bool = False,
    make_plots: bool = True,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    input_path = Path(input_parquet)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_config = default_config(horizon=horizon, battery_capacity_kwh=battery_capacity_kwh)
    audit_paths = write_objective_term_scale_audit(output_dir=out_dir, config=audit_config)

    cases = [case for case in RAW_RETUNE_CASES if (not priority_only or bool(case.get("priority")))]
    summary_frames: list[pd.DataFrame] = []
    by_voyage_frames: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []

    for case in cases:
        case_name = str(case["case_name"])
        case_dir = out_dir / case_name
        case_config = default_config(
            horizon=horizon,
            battery_capacity_kwh=battery_capacity_kwh,
            q_h2=float(case["q_h2"]),
            q_soc=float(case["q_soc"]),
            q_batt=float(case["q_batt"]),
            q_ramp=float(case["q_ramp"]),
            q_terminal_soc=float(case["q_terminal_soc"]),
        )
        reused_base = False
        if case_name == "case_raw_base_old" and base_output_dir is not None:
            base_dir = Path(base_output_dir)
            base_control_path = base_dir / "control_timeseries.csv"
            base_time_path = base_dir / "solve_times.csv"
            if base_control_path.exists() and base_time_path.exists():
                control_df = pd.read_csv(base_control_path)
                time_df = pd.read_csv(base_time_path)
                metrics_df = _recompute_voyage_metrics_from_controls(
                    control_df,
                    time_df,
                    case_config,
                    soc_reference=soc_reference,
                )
                result = write_benchmark_artifacts(
                    output_dir=case_dir,
                    config=case_config,
                    control_df=control_df,
                    time_df=time_df,
                    metrics_df=metrics_df,
                    input_path=input_path,
                    max_steps_per_voyage=max_steps_per_voyage,
                    make_plots=make_plots,
                    report_filename=f"REPORT_{case_name}.md",
                    soc_reference=soc_reference,
                    osqp_available=True,
                )
                result["status"] = "reused_existing_base_recomputed_artifacts"
                reused_base = True
        if not reused_base:
            result = run_benchmark(
                input_parquet=input_path,
                output_dir=case_dir,
                horizon=horizon,
                max_steps_per_voyage=max_steps_per_voyage,
                make_plots=make_plots,
                battery_capacity_kwh=battery_capacity_kwh,
                q_h2=float(case["q_h2"]),
                q_soc=float(case["q_soc"]),
                q_batt=float(case["q_batt"]),
                q_ramp=float(case["q_ramp"]),
                q_terminal_soc=float(case["q_terminal_soc"]),
                soc_reference=soc_reference,
                report_filename=f"REPORT_{case_name}.md",
            )
        results.append({"case_name": case_name, **result})

        summary_path = case_dir / "solver_benchmark_summary.csv"
        by_voyage_path = case_dir / "solver_benchmark_by_voyage.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            if "physical_label" not in summary.columns:
                summary["physical_label"] = summary.apply(classify_physical_baseline, axis=1)
            summary.insert(0, "case_name", case_name)
            summary_frames.append(summary)
        if by_voyage_path.exists():
            by_voyage = pd.read_csv(by_voyage_path)
            by_voyage.insert(0, "case_name", case_name)
            by_voyage_frames.append(by_voyage)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_voyage_df = pd.concat(by_voyage_frames, ignore_index=True) if by_voyage_frames else pd.DataFrame()
    if not summary_df.empty and "physical_label" not in summary_df.columns:
        summary_df["physical_label"] = summary_df.apply(classify_physical_baseline, axis=1)
    recommendation = select_recommended_raw_baseline(summary_df)

    summary_path = out_dir / "raw_weight_candidate_summary.csv"
    by_voyage_path = out_dir / "raw_weight_candidate_by_voyage.csv"
    decision_path = out_dir / "raw_weight_candidate_decision.md"
    report_path = out_dir / RAW_WEIGHT_RETUNE_REPORT_FILENAME
    summary_df.to_csv(summary_path, index=False)
    by_voyage_df.to_csv(by_voyage_path, index=False)
    _write_raw_weight_candidate_decision(
        path=decision_path,
        summary_df=summary_df,
        recommendation=recommendation,
        audit_paths=audit_paths,
        priority_only=priority_only,
    )
    _write_raw_weight_retune_report(
        path=report_path,
        output_dir=out_dir,
        input_parquet=input_path,
        summary_df=summary_df,
        by_voyage_df=by_voyage_df,
        recommendation=recommendation,
        audit_paths=audit_paths,
        priority_only=priority_only,
    )
    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "objective_term_scale_audit_csv_path": audit_paths["csv_path"],
        "objective_term_scale_audit_md_path": audit_paths["md_path"],
        "raw_weight_candidate_summary_path": str(summary_path),
        "raw_weight_candidate_by_voyage_path": str(by_voyage_path),
        "raw_weight_candidate_decision_path": str(decision_path),
        "report_path": str(report_path),
        "recommendation": recommendation,
        "priority_only": bool(priority_only),
        "case_results": results,
    }


def run_weight_sensitivity(
    *,
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    output_dir: str | Path = DEFAULT_SENSITIVITY_OUTPUT_DIR,
    base_output_dir: str | Path | None = None,
    horizon: int = 60,
    max_steps_per_voyage: int | None = None,
    battery_capacity_kwh: float = EBATT277P2_BATTERY_CAPACITY_KWH,
    soc_reference: float = DEFAULT_SOC_REFERENCE,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        {"case_name": "case_base", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 0.03, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0},
        {"case_name": "case_soc_high", "q_h2": 1.0, "q_soc": 800.0, "q_batt": 0.03, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0},
        {"case_name": "case_batt_high", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 0.10, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0},
        {"case_name": "case_h2_lower", "q_h2": 0.5, "q_soc": 400.0, "q_batt": 0.03, "q_ramp": 2.0e-5, "q_terminal_soc": 0.0},
        {"case_name": "case_terminal_on", "q_h2": 1.0, "q_soc": 400.0, "q_batt": 0.03, "q_ramp": 2.0e-5, "q_terminal_soc": 50.0},
    ]
    summary_frames: list[pd.DataFrame] = []
    by_voyage_frames: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    for case in cases:
        case_name = str(case["case_name"])
        case_dir = out_dir / case_name
        if case_name == "case_base" and base_output_dir is not None:
            base_dir = Path(base_output_dir)
            base_summary_path = base_dir / "solver_benchmark_summary.csv"
            base_by_voyage_path = base_dir / "solver_benchmark_by_voyage.csv"
            if base_summary_path.exists() and base_by_voyage_path.exists():
                summary = pd.read_csv(base_summary_path)
                summary.insert(0, "case_name", case_name)
                summary_frames.append(summary)
                by_voyage = pd.read_csv(base_by_voyage_path)
                by_voyage.insert(0, "case_name", case_name)
                by_voyage_frames.append(by_voyage)
                results.append(
                    {
                        "case_name": case_name,
                        "status": "reused_existing_base",
                        "solver_benchmark_summary_path": str(base_summary_path),
                        "solver_benchmark_by_voyage_path": str(base_by_voyage_path),
                    }
                )
                continue
        result = run_benchmark(
            input_parquet=input_parquet,
            output_dir=case_dir,
            horizon=horizon,
            max_steps_per_voyage=max_steps_per_voyage,
            make_plots=False,
            battery_capacity_kwh=battery_capacity_kwh,
            q_h2=float(case["q_h2"]),
            q_soc=float(case["q_soc"]),
            q_batt=float(case["q_batt"]),
            q_ramp=float(case["q_ramp"]),
            q_terminal_soc=float(case["q_terminal_soc"]),
            soc_reference=soc_reference,
            report_filename=f"REPORT_{case_name}.md",
        )
        results.append({"case_name": case_name, **result})
        summary = pd.read_csv(case_dir / "solver_benchmark_summary.csv")
        summary.insert(0, "case_name", case_name)
        summary_frames.append(summary)
        by_voyage = pd.read_csv(case_dir / "solver_benchmark_by_voyage.csv")
        by_voyage.insert(0, "case_name", case_name)
        by_voyage_frames.append(by_voyage)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_voyage_df = pd.concat(by_voyage_frames, ignore_index=True) if by_voyage_frames else pd.DataFrame()
    summary_path = out_dir / "weight_sensitivity_summary.csv"
    by_voyage_path = out_dir / "weight_sensitivity_by_voyage.csv"
    decision_path = out_dir / "weight_sensitivity_decision.md"
    summary_df.to_csv(summary_path, index=False)
    by_voyage_df.to_csv(by_voyage_path, index=False)

    lines = [
        "# Ebatt 277.2 kWh Weight Sensitivity Decision",
        "",
        "Scope: direction-only sensitivity around the current 1 s OSQP-QP weights. No final weights are auto-selected and no mainline config is changed.",
        "",
        f"- Input parquet: `{input_parquet}`",
        f"- Output directory: `{out_dir}`",
        f"- Battery capacity: `{battery_capacity_kwh} kWh`",
        f"- Fuel-cell ramp remains: `48 kW/s = 48 kW/step`",
        "",
        "## Cases",
        "",
        "| case | q_h2 | q_soc | q_batt | q_ramp | q_terminal | weights_valid | failure_categories | H2_total_kg | battery_throughput_kwh | SOC_final_minus_initial_min | p99_ms |",
        "|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['case_name']} | {row['q_h2']} | {row['q_soc']} | {row['q_batt']} | {row['q_ramp']} | {row['q_terminal']} | "
            f"{row['weights_valid']} | {row.get('failure_categories', '')} | {row['H2_total_kg']:.6f} | "
            f"{row['battery_throughput_kwh']:.6f} | {row['SOC_final_minus_initial_min']:.6f} | {row['solve_time_ms_p99']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This file is only a small sensitivity scan. Use it to identify direction; do not treat it as a tuned final MPC design.",
            "",
        ]
    )
    if not summary_df.empty and "weights_valid" in summary_df.columns:
        valid_mask = summary_df["weights_valid"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        valid_mask = pd.Series(False, index=summary_df.index)
    if not summary_df.empty and valid_mask.any():
        valid_cases = summary_df[valid_mask]["case_name"].tolist()
        lines.append(f"Validity gate passed in: `{', '.join(valid_cases)}`. A separate confirmation run is still required before promotion.")
    else:
        lines.append("No sensitivity case passed all validity gates. Broader objective redesign is required before promoting fixed QP-MPC.")
    decision_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "status": "completed",
        "output_dir": str(out_dir),
        "weight_sensitivity_summary_path": str(summary_path),
        "weight_sensitivity_by_voyage_path": str(by_voyage_path),
        "weight_sensitivity_decision_path": str(decision_path),
        "case_results": results,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_parquet", default=str(DEFAULT_INPUT_PARQUET))
    parser.add_argument("--output_dir", default=str(SIMPLIFIED_SPEC_NORM_OUTPUT_DIR))
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--max_steps_per_voyage", type=int, default=None)
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--battery_capacity_kwh", type=float, default=SPEC_BATTERY_CAPACITY_KWH)
    parser.add_argument("--q_h2", type=float, default=1.0)
    parser.add_argument("--q_soc", type=float, default=1.0)
    parser.add_argument("--q_batt", type=float, default=0.05)
    parser.add_argument("--q_ramp", type=float, default=0.0)
    parser.add_argument("--q_terminal_soc", type=float, default=0.0)
    parser.add_argument("--soc_reference", type=float, default=DEFAULT_SOC_REFERENCE)
    parser.add_argument("--report_filename", default=DEFAULT_REPORT_FILENAME)
    parser.add_argument("--run_simplified_spec_norm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.run_simplified_spec_norm:
        result = run_simplified_spec_norm_benchmark(
            input_parquet=args.input_parquet,
            output_dir=args.output_dir,
            horizon=args.horizon,
            max_steps_per_voyage=args.max_steps_per_voyage,
            make_plots=not args.no_plots,
            battery_capacity_kwh=args.battery_capacity_kwh,
            soc_reference=args.soc_reference,
        )
    else:
        result = run_benchmark(
            input_parquet=args.input_parquet,
            output_dir=args.output_dir,
            horizon=args.horizon,
            max_steps_per_voyage=args.max_steps_per_voyage,
            make_plots=not args.no_plots,
            battery_capacity_kwh=args.battery_capacity_kwh,
            q_h2=args.q_h2,
            q_soc=args.q_soc,
            q_batt=args.q_batt,
            q_ramp=args.q_ramp,
            q_terminal_soc=args.q_terminal_soc,
            soc_reference=args.soc_reference,
            report_filename=args.report_filename,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
