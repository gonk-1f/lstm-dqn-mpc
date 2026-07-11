"""One-step-ahead LSTM-MPC closed-loop test on the 7-2-1 test voyages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forecasting.feature_pipeline import prepare_lstm_features as _prepare_features  # noqa: E402
from forecasting.lstm_load_predictor import inverse_target, load_checkpoint, transform  # noqa: E402
from mpc.controllers.reference_generator import CasadiReferenceGenerator  # noqa: E402
from mpc.solvers.casadi_solver import CasadiMPCConfig, resolve_soc_reference  # noqa: E402
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic  # noqa: E402


LSTM_CKPT = PROJ / "outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt"
SPLIT_JSON = PROJ / "outputs/config/voyage_split_721.json"
SOURCE_CSV = PROJ / "data/processed/aligned_timeseries.csv"
WEIGHT_SETS_JSON = PROJ / "outputs/config/mpc_weight_sets.json"
ACTION_TABLE_JSON = PROJ / "outputs/config/dqn_mpc_action_table.json"
OUT_DIR = PROJ / "outputs/lstm_mpc_test"

DT_SECONDS = 30
MPC_HORIZON = 6
FORECAST_MODE = "LSTM"
CONTROL_TIMING = "one_step_ahead_lstm_mpc"
CONTROL_APPLY_TIMING = "execute_cached_previous_mpc_command"
CONTROL_LAYER_SCOPE = "total_power"
LOAD_DEFINITION = "left_plus_right_propulsion_inverter_active_power"
CAPACITY_BASIS = "propulsion_inverter_load_scope_equivalent_battery_capacity"
INITIAL_FC_MODE = "initial_current_load"
REFERENCE_GENERATOR_CLASS = CasadiReferenceGenerator
P6_WEIGHT_SOURCE = "dp0_raw_h2_soc_batt_ramp_nextstep_v1"
CURRENT_FIXED_WEIGHT_SET = "dp0_raw_h2_soc_batt_ramp_nextstep_v1"
TOTAL_LOAD_WEIGHT_SET = "dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1"
TOTAL_LOAD_QSOC400_QBATT0035_QRAMP3E5_WEIGHT_SET = "dp0_total_load_qsoc400_qbatt0035_qramp3e-5"
TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET = "dp0_total_load_qsoc400_qbatt003_qramp2e-5"
TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET = "dp0_total_load_qsoc500_qbatt003_qramp3e-5"
TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET = "dp0_total_load_qsoc500_qbatt0035_qramp3e-5"
TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET = "dp0_total_load_no_reserve_no_terminal_qsoc600"
NO_RESERVE_NO_TERMINAL_WEIGHT_SETS = {
    TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET,
    TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET,
    TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET,
    TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET,
}
FORBIDDEN_RULE_BASED_CONFIG_KEYS = {
    "low_load_fc_suppression",
    "soc_recovery_power_limit",
    "sustained_load_battery_discharge_limit",
    "fc_overproduction_limit",
    "low_load_fc_suppression_enabled",
    "soc_recovery_power_limit_enabled",
    "sustained_load_battery_discharge_limit_enabled",
    "fc_overproduction_limit_enabled",
}
DISABLED_WEIGHT_SET_CONFIG_KEYS = {
    "soc_reserve",
    "enable_soc_reserve_soft_penalty",
    "q_reserve",
    "reserve_penalty_total",
    "terminal_soc_band",
    "enable_terminal_soc_soft_penalty",
}

OBJECTIVE_INFO_EXPORT_FIELDS = [
    "effective_q_h2",
    "effective_q_soc",
    "effective_q_batt",
    "effective_q_ramp",
    "effective_q_terminal_soc",
    "effective_battery_capacity_kwh",
    "effective_fuel_cell_ramp_constraint_enabled",
    "effective_use_dimensionless_objective",
    "effective_normalize_h2_cost",
    "effective_soc_band",
    "effective_terminal_soc_band",
    "h2_mass_kg",
    "batt_throughput_kwh",
    "weighted_h2_cost",
    "soc_cost_raw",
    "weighted_soc_cost",
    "weighted_batt_cost",
    "ramp_cost_raw",
    "weighted_ramp_cost",
    "total_objective",
    "raw_h2_cost",
    "raw_soc_cost",
    "battery_use_term",
    "total_mpc_cost",
    "H2_step_kg_sum",
    "average_eta_fc",
    "min_eta_fc",
    "max_eta_fc",
    "raw_fc_cost_mode",
    "objective_scale_mode",
    "objective_mode",
    "use_h2_mass_cost",
    "normalize_h2_cost",
    "fuel_cell_ramp_constraint_enabled",
    "fuel_cell_ramp_kw",
    "battery_throughput_penalty_enabled",
    "battery_throughput_penalty_type",
    "battery_throughput_normalization_kw",
    "enable_fc_post_filter",
    "m_H2_ref_kg_per_step",
    "fc_efficiency_curve_source",
    "soc_reference_mode",
    "soc_penalty_type",
    "soc_ref_value",
    "soc_reserve",
    "terminal_soc_band",
    "P_fc_upper_bound",
    "P_fc_lower_bound",
    "battery_discharge_upper_bound",
    "battery_charge_upper_bound",
    "battery_discharge_limit_active",
    "battery_charge_limit_active",
]

DEBUG_INITIAL_DISPATCH_FIELDS = [
    "voyage_id",
    "t_index",
    "time_h",
    "load_kw",
    "pred_h1",
    "pred_h2",
    "pred_h3",
    "pred_h4",
    "pred_h5",
    "mpc_load_ref_0",
    "mpc_load_ref_max",
    "soc",
    "soc_ref",
    "soc_reserve",
    "soc_error",
    "soc_band",
    "terminal_soc_band",
    "P_fc_cmd",
    "P_batt_cmd",
    "P_fc_prev",
    "P_fc_upper_bound",
    "P_fc_lower_bound",
    "battery_discharge_upper_bound",
    "battery_charge_upper_bound",
    "mpc_solved",
    "fallback_control_used",
    "fallback_reason",
    "history_available",
    "lstm_available",
    "battery_discharge_limit_active",
    "battery_charge_limit_active",
    "weighted_h2_cost",
    "weighted_soc_cost",
    "weighted_terminal_soc_cost",
    "weighted_batt_cost",
    "weighted_ramp_cost",
    "total_objective",
]

RESERVE_REPORTING_COLUMN_FRAGMENTS = (
    "reserve",
)
TERMINAL_REPORTING_COLUMN_FRAGMENTS = (
    "terminal_soc",
)
TERMINAL_REPORTING_COLUMNS = {
    "q_terminal_soc",
    "weighted_terminal_soc_cost",
    "terminal_soc_term",
}
DISABLED_RESERVE_TERMINAL_CONFIG_KEYS = {
    "soc_reserve",
    "enable_soc_reserve_soft_penalty",
    "q_reserve",
    "reserve_penalty_total",
    "terminal_soc_band",
    "enable_terminal_soc_soft_penalty",
}

OBJECTIVE_BREAKDOWN_FIELDS = [
    "h2_mass_kg",
    "batt_throughput_kwh",
    "weighted_h2_cost",
    "soc_cost_raw",
    "weighted_soc_cost",
    "weighted_batt_cost",
    "ramp_cost_raw",
    "weighted_ramp_cost",
    "total_objective",
    "raw_h2_cost",
    "raw_soc_cost",
]

MPC_BASE_KWARGS: dict[str, Any] = {
    "prediction_horizon": MPC_HORIZON,
    "dt_hours": DT_SECONDS / 3600.0,
    "battery_capacity_kwh": 277.2,
    "fuel_cell_max_kw": 560.0,
    "fuel_cell_ramp_kw": 48.0,
    "fuel_cell_ramp_constraint_enabled": False,
    "initial_fc_mode": INITIAL_FC_MODE,
    "soc_target": 0.65,
    "soc_reference_mode": "initial_soc",
    "soc_reserve": 0.55,
    "soc_band": 0.0,
    "terminal_soc_band": 0.0,
    "soc_min": 0.2,
    "soc_max": 0.8,
    "objective_mode": "raw_physical",
    "use_raw_objective": False,
    "use_dimensionless_objective": False,
    "use_h2_mass_cost": True,
    "normalize_h2_cost": False,
    "raw_soc_squared": False,
    "enable_terminal_soc_soft_penalty": False,
    "ipopt_max_iter": 100,
    "ipopt_tol": 1e-4,
}

DP0_COMMON_CONFIG: dict[str, Any] = {
    "objective_mode": "raw_physical",
    "use_raw_objective": False,
    "use_dimensionless_objective": False,
    "use_h2_mass_cost": True,
    "normalize_h2_cost": False,
    "enable_fc_post_filter": False,
    "q_fc": 0.0,
    "soc_target": 0.65,
    "soc_reference_mode": "initial_soc",
    "soc_band": 0.0,
}

DEFAULT_WEIGHT_SETS: dict[str, dict[str, Any]] = {
    "dp0_raw_h2_soc_batt_ramp_nextstep_v1": {
        **DP0_COMMON_CONFIG,
        "q_h2": 1.0,
        "q_soc": 50.0,
        "q_batt": 0.025,
        "q_ramp": 0.0001,
        "q_terminal_soc": 0.0,
        "battery_capacity_kwh": 277.2,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": P6_WEIGHT_SOURCE,
    },
    TOTAL_LOAD_WEIGHT_SET: {
        **DP0_COMMON_CONFIG,
        "q_h2": 1.0,
        "q_soc": 300.0,
        "q_batt": 0.020,
        "q_ramp": 0.00010,
        "q_terminal_soc": 0.0,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
    TOTAL_LOAD_QSOC400_QBATT0035_QRAMP3E5_WEIGHT_SET: {
        **DP0_COMMON_CONFIG,
        "q_h2": 1.0,
        "q_soc": 400.0,
        "q_batt": 0.035,
        "q_ramp": 0.00003,
        "q_terminal_soc": 0.0,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_QSOC400_QBATT0035_QRAMP3E5_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
    TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET: {
        "objective_mode": "raw_physical",
        "use_raw_objective": False,
        "use_dimensionless_objective": False,
        "use_h2_mass_cost": True,
        "normalize_h2_cost": False,
        "enable_fc_post_filter": False,
        "q_h2": 1.0,
        "q_soc": 400.0,
        "q_fc": 0.0,
        "q_batt": 0.030,
        "q_ramp": 0.00002,
        "q_terminal_soc": 0.0,
        "soc_reference_mode": "initial_soc",
        "soc_target": 0.65,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc_band": 0.0,
        "raw_soc_squared": True,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
    TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET: {
        "objective_mode": "raw_physical",
        "use_raw_objective": False,
        "use_dimensionless_objective": False,
        "use_h2_mass_cost": True,
        "normalize_h2_cost": False,
        "enable_fc_post_filter": False,
        "q_h2": 1.0,
        "q_soc": 600.0,
        "q_fc": 0.0,
        "q_batt": 0.035,
        "q_ramp": 0.00003,
        "q_terminal_soc": 0.0,
        "soc_reference_mode": "initial_soc",
        "soc_target": 0.65,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc_band": 0.0,
        "raw_soc_squared": True,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
    TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET: {
        "objective_mode": "raw_physical",
        "use_raw_objective": False,
        "use_dimensionless_objective": False,
        "use_h2_mass_cost": True,
        "normalize_h2_cost": False,
        "enable_fc_post_filter": False,
        "q_h2": 1.0,
        "q_soc": 500.0,
        "q_fc": 0.0,
        "q_batt": 0.030,
        "q_ramp": 0.00003,
        "q_terminal_soc": 0.0,
        "soc_reference_mode": "initial_soc",
        "soc_target": 0.65,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc_band": 0.0,
        "raw_soc_squared": True,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
    TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET: {
        "objective_mode": "raw_physical",
        "use_raw_objective": False,
        "use_dimensionless_objective": False,
        "use_h2_mass_cost": True,
        "normalize_h2_cost": False,
        "enable_fc_post_filter": False,
        "q_h2": 1.0,
        "q_soc": 500.0,
        "q_fc": 0.0,
        "q_batt": 0.035,
        "q_ramp": 0.00003,
        "q_terminal_soc": 0.0,
        "soc_reference_mode": "initial_soc",
        "soc_target": 0.65,
        "soc_min": 0.20,
        "soc_max": 0.80,
        "soc_band": 0.0,
        "raw_soc_squared": True,
        "battery_capacity_kwh": 1806.0,
        "fuel_cell_ramp_constraint_enabled": False,
        "initial_fc_mode": INITIAL_FC_MODE,
        "P6_WEIGHT_SOURCE": TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET,
        "capacity_basis": "full ship energy storage capacity for energy-side equivalent total load",
        "control_timing": CONTROL_TIMING,
        "load_definition": "fuel_cell_total_kw + battery_total_kw",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe_value(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) and not isinstance(value, (bool, str)):
        return None
    return value


def compact_float_for_run_id(value: Any) -> str:
    text = f"{float(value):.8g}"
    return text.replace("+", "").replace("-", "m").replace(".", "p")


def make_run_id(weight_set: str, weights: dict[str, Any], *, timestamp: str | None = None) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [
        stamp,
        str(weight_set),
        f"qh2{compact_float_for_run_id(weights.get('q_h2', 0.0))}",
        f"qsoc{compact_float_for_run_id(weights.get('q_soc', 0.0))}",
        f"qbatt{compact_float_for_run_id(weights.get('q_batt', 0.0))}",
        f"qramp{compact_float_for_run_id(weights.get('q_ramp', 0.0))}",
    ]
    safe = "_".join(parts)
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe)


def file_md5(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_file_hashes(output_dir: Path, run_id: str) -> dict[str, Any]:
    output_dir = Path(output_dir)
    tracked = [
        "fixed_mpc_p6_comparison.png",
        "fixed_mpc_p6_timeseries.csv",
        "fixed_mpc_p6_metrics.csv",
        "fixed_mpc_p6_config.json",
        "fixed_mpc_p6_objective_breakdown.csv",
        "timing_debug_first_steps.csv",
        "solver_horizon_debug.json",
        "horizon_sensitivity_debug.csv",
        "effective_config_debug.json",
        "baseline_summary.md",
    ]
    payload: dict[str, Any] = {
        "run_id": run_id,
        "output_dir": str(output_dir.resolve()),
    }
    for name in tracked:
        path = output_dir / name
        stat = path.stat() if path.exists() else None
        payload[name] = {
            "path": str(path.resolve()),
            "exists": bool(path.exists()),
            "size_bytes": int(stat.st_size) if stat else None,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds") if stat else None,
            "md5": file_md5(path),
        }
    return payload


def effective_mpc_diagnostics(config: CasadiMPCConfig) -> dict[str, Any]:
    return {
        "effective_q_h2": float(config.q_h2),
        "effective_q_soc": float(config.q_soc),
        "effective_q_fc": float(config.q_fc),
        "effective_q_batt": float(config.q_batt),
        "effective_q_ramp": float(config.q_ramp),
        "effective_q_terminal_soc": float(config.q_terminal_soc) if config.q_terminal_soc is not None else 0.0,
        "effective_battery_capacity_kwh": float(config.battery_capacity_kwh),
        "effective_fuel_cell_ramp_constraint_enabled": bool(config.fuel_cell_ramp_constraint_enabled),
        "effective_fuel_cell_ramp_kw": float(config.fuel_cell_ramp_kw),
        "effective_enable_fc_ramp_hard_constraint": bool(config.fuel_cell_ramp_constraint_enabled),
        "effective_fc_ramp_limit_kw_per_step": float(config.fuel_cell_ramp_kw),
        "effective_use_dimensionless_objective": bool(config.use_dimensionless_objective),
        "effective_normalize_h2_cost": bool(config.normalize_h2_cost),
        "effective_objective_mode": str(config.objective_mode),
        "effective_use_h2_mass_cost": bool(config.use_h2_mass_cost),
        "effective_soc_reference_mode": str(config.soc_reference_mode),
        "effective_soc_band": float(config.soc_band),
        "effective_terminal_soc_band": float(config.terminal_soc_band),
        "effective_enable_terminal_soc_soft_penalty": bool(config.enable_terminal_soc_soft_penalty),
    }


def add_run_id_column(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "run_id" in out.columns:
        out["run_id"] = run_id
    else:
        out.insert(0, "run_id", run_id)
    return out


def uses_no_reserve_no_terminal_reporting(weight_set: str) -> bool:
    key = str(weight_set)
    return key in NO_RESERVE_NO_TERMINAL_WEIGHT_SETS


def clean_reporting_frame_for_weight_set(df: pd.DataFrame, *, weight_set: str) -> pd.DataFrame:
    if df.empty or not uses_no_reserve_no_terminal_reporting(weight_set):
        return df.copy()
    drop_cols = [
        col
        for col in df.columns
        if any(fragment in col.lower() for fragment in RESERVE_REPORTING_COLUMN_FRAGMENTS)
        or any(fragment in col.lower() for fragment in TERMINAL_REPORTING_COLUMN_FRAGMENTS)
        or col in TERMINAL_REPORTING_COLUMNS
    ]
    return df.drop(columns=drop_cols, errors="ignore").copy()


def clean_reporting_dict_for_weight_set(values: dict[str, Any], *, weight_set: str) -> dict[str, Any]:
    if not uses_no_reserve_no_terminal_reporting(weight_set):
        return dict(values)
    row = pd.DataFrame([values])
    cleaned = clean_reporting_frame_for_weight_set(row, weight_set=weight_set)
    return dict(cleaned.iloc[0]) if not cleaned.empty else {}


def clean_config_payload_for_weight_set(value: Any, *, weight_set: str) -> Any:
    if not uses_no_reserve_no_terminal_reporting(weight_set):
        return value
    if isinstance(value, dict):
        return {
            str(key): clean_config_payload_for_weight_set(item, weight_set=weight_set)
            for key, item in value.items()
            if str(key) not in DISABLED_RESERVE_TERMINAL_CONFIG_KEYS and "reserve" not in str(key).lower()
        }
    if isinstance(value, list):
        return [clean_config_payload_for_weight_set(item, weight_set=weight_set) for item in value]
    return value


def load_weight_sets(path: Path = WEIGHT_SETS_JSON) -> dict[str, dict[str, Any]]:
    raw = load_json(path) if path.exists() else {}
    merged: dict[str, dict[str, Any]] = {}

    def clean_weight_values(values: dict[str, Any]) -> dict[str, Any]:
        clean_values: dict[str, Any] = {}
        for subkey, value in values.items():
            if subkey in FORBIDDEN_RULE_BASED_CONFIG_KEYS or subkey in DISABLED_WEIGHT_SET_CONFIG_KEYS:
                continue
            clean_values[str(subkey)] = value
        return clean_values

    for key, defaults in DEFAULT_WEIGHT_SETS.items():
        clean_values = dict(defaults)
        raw_values = dict(raw.get(key, {})) if isinstance(raw, dict) and isinstance(raw.get(key, {}), dict) else {}
        clean_values.update(clean_weight_values(raw_values))
        merged[key] = clean_values
    if isinstance(raw, dict):
        for key, values in raw.items():
            if key in merged or not isinstance(values, dict):
                continue
            clean_values = clean_weight_values(dict(values))
            if clean_values:
                merged[str(key)] = clean_values
    if raw != merged:
        write_json(path, merged)
    raw = merged
    normalized: dict[str, dict[str, Any]] = {}
    for key, values in dict(raw).items():
        normalized[str(key)] = {}
        for subkey, value in dict(values).items():
            if isinstance(value, bool):
                normalized[str(key)][str(subkey)] = bool(value)
            elif isinstance(value, (int, float)):
                normalized[str(key)][str(subkey)] = float(value)
            else:
                normalized[str(key)][str(subkey)] = value
    return normalized


def build_action_table(base: dict[str, Any]) -> list[dict[str, float | int | bool]]:
    q_soc_delta = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]
    q_ramp_delta = [-0.04, -0.02, 0.0, 0.02, 0.04]
    actions: list[dict[str, float | int]] = []
    action_id = 0
    for dsoc in q_soc_delta:
        for dramp in q_ramp_delta:
            actions.append(
                {
                    "action_id": action_id,
                    "q_soc": max(10.0, float(base["q_soc"]) + dsoc),
                    **({"q_h2": float(base["q_h2"])} if "q_h2" in base else {}),
                    "q_fc": float(base["q_fc"]),
                    "q_batt": float(base["q_batt"]),
                    "q_ramp": max(0.0, float(base["q_ramp"]) + dramp),
                    "q_terminal_soc": float(base["q_terminal_soc"]),
                    **({"soc_band": float(base["soc_band"])} if "soc_band" in base else {}),
                    **(
                        {"use_h2_mass_cost": bool(base["use_h2_mass_cost"])}
                        if "use_h2_mass_cost" in base
                        else {}
                    ),
                    **(
                        {"use_dimensionless_objective": bool(base["use_dimensionless_objective"])}
                        if "use_dimensionless_objective" in base
                        else {}
                    ),
                }
            )
            action_id += 1
    return actions


def write_action_table(
    path: Path = ACTION_TABLE_JSON,
    base: dict[str, Any] | None = None,
    base_weight_set: str = CURRENT_FIXED_WEIGHT_SET,
) -> list[dict[str, float | int | bool]]:
    base_weights = dict(DEFAULT_WEIGHT_SETS[CURRENT_FIXED_WEIGHT_SET] if base is None else base)
    actions = build_action_table(base_weights)
    write_json(
        path,
        {
            "base_weight_set": base_weight_set,
            "control_timing": CONTROL_TIMING,
            "control_apply_timing": CONTROL_APPLY_TIMING,
            "actions": actions,
        },
    )
    return actions


def mpc_config_from_weights(weights: dict[str, Any]) -> CasadiMPCConfig:
    kwargs = dict(MPC_BASE_KWARGS)
    for key in [
        "q_h2",
        "q_soc",
        "q_fc",
        "q_batt",
        "q_ramp",
        "q_terminal_soc",
        "soc_band",
        "terminal_soc_band",
        "soc_target",
        "soc_reserve",
        "soc_min",
        "soc_max",
        "battery_throughput_normalization_kw",
        "battery_capacity_kwh",
        "fuel_cell_ramp_kw",
    ]:
        if key in weights:
            kwargs[key] = float(weights[key])
    for key in [
        "use_raw_objective",
        "use_dimensionless_objective",
        "use_h2_mass_cost",
        "normalize_h2_cost",
        "enable_terminal_soc_soft_penalty",
        "raw_soc_squared",
        "raw_fc_energy_linear",
        "enable_fc_post_filter",
        "battery_throughput_penalty_enabled",
        "fuel_cell_ramp_constraint_enabled",
    ]:
        if key in weights:
            kwargs[key] = bool(weights[key])
    if "soc_reference_mode" in weights:
        kwargs["soc_reference_mode"] = str(weights["soc_reference_mode"])
    if "objective_mode" in weights:
        kwargs["objective_mode"] = str(weights["objective_mode"])
    if "soc_penalty_type" in weights:
        kwargs["soc_penalty_type"] = str(weights["soc_penalty_type"])
    if "battery_throughput_penalty_type" in weights:
        kwargs["battery_throughput_penalty_type"] = str(weights["battery_throughput_penalty_type"])
    if "initial_fc_mode" in weights:
        kwargs["initial_fc_mode"] = str(weights["initial_fc_mode"])
    if "fc_ramp_limit_kw_per_step" in weights:
        kwargs["fuel_cell_ramp_kw"] = float(weights["fc_ramp_limit_kw_per_step"])
    if "enable_fc_ramp_hard_constraint" in weights:
        kwargs["fuel_cell_ramp_constraint_enabled"] = bool(weights["enable_fc_ramp_hard_constraint"])
    batt_cfg = weights.get("battery_throughput_penalty")
    if isinstance(batt_cfg, dict):
        if "enabled" in batt_cfg:
            kwargs["battery_throughput_penalty_enabled"] = bool(batt_cfg["enabled"])
        if "type" in batt_cfg:
            kwargs["battery_throughput_penalty_type"] = str(batt_cfg["type"])
        if "normalization_kw" in batt_cfg:
            kwargs["battery_throughput_normalization_kw"] = float(batt_cfg["normalization_kw"])
    return CasadiMPCConfig(**kwargs)


def objective_info_fields(objective_info: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in OBJECTIVE_INFO_EXPORT_FIELDS:
        value = objective_info.get(key, np.nan)
        if isinstance(value, (bool, str)):
            row[key] = value
        elif value is None:
            row[key] = np.nan
        else:
            row[key] = float(value)
    return row


def bool01(value: Any) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes"})
    try:
        return int(float(value) > 0.5)
    except (TypeError, ValueError):
        return 0


def build_mpc_load_ref(
    current_load_t: float,
    lstm_pred: np.ndarray,
    pred_horizon: int = MPC_HORIZON,
    mpc_horizon: int = MPC_HORIZON,
    *,
    lstm_available: bool = True,
) -> np.ndarray:
    pred = np.asarray(lstm_pred, dtype=float).reshape(-1)
    assert pred_horizon == 6
    assert mpc_horizon == 6
    if lstm_available and len(pred) >= 6 and np.all(np.isfinite(pred[:6])):
        ref = pred[:6]
    else:
        ref = np.full(mpc_horizon, float(current_load_t), dtype=float)
    assert len(ref) == 6
    return np.maximum(ref, 0.0)


def forecast_source_name(*, history_available: bool, lstm_available: bool) -> str:
    return "lstm_h1_to_h6" if history_available and lstm_available else "current_load_hold"


def make_debug_timing_row(
    voyage_id: int,
    decision_index_t: int,
    history_len: int,
    pred_horizon: int,
    mpc_load_ref: np.ndarray,
    actual_load_t: float,
    pred_h1: float,
    actual_load_t_plus_1: float | None,
    *,
    history_available: bool,
    lstm_available: bool,
    forecast_source: str,
) -> dict[str, Any]:
    mpc_ref = np.asarray(mpc_load_ref, dtype=float).reshape(-1)
    row = {
        "voyage_id": int(voyage_id),
        "decision_index_t": int(decision_index_t),
        "apply_index": int(decision_index_t) + 1,
        "history_start_index": int(decision_index_t) - int(history_len) + 1,
        "history_end_index": int(decision_index_t),
        "lstm_forecast_start_index": int(decision_index_t) + 1,
        "lstm_forecast_end_index": int(decision_index_t) + int(pred_horizon),
        "mpc_ref_len": int(len(mpc_ref)),
        "mpc_stage0_source": str(forecast_source),
        "mpc_stage1_source": str(forecast_source),
        "first_mpc_ref_load": float(mpc_ref[0]) if len(mpc_ref) else float("nan"),
        "actual_load_t": float(actual_load_t),
        "pred_h1": float(pred_h1),
        "actual_load_t_plus_1": float("nan") if actual_load_t_plus_1 is None else float(actual_load_t_plus_1),
        "history_available": bool(history_available),
        "lstm_available": bool(lstm_available),
        "forecast_source": str(forecast_source),
    }
    if row["history_end_index"] != row["decision_index_t"]:
        raise ValueError("Invalid timing: history_end_index must equal decision_index_t.")
    if row["lstm_forecast_start_index"] != row["decision_index_t"] + 1:
        raise ValueError("Invalid timing: LSTM forecast must start at t+1.")
    if row["lstm_forecast_end_index"] != row["decision_index_t"] + int(pred_horizon):
        raise ValueError(f"Invalid timing: LSTM forecast must end at t+{int(pred_horizon)}.")
    if row["mpc_ref_len"] != 6:
        raise ValueError("Invalid timing: MPC load reference length must equal 6.")
    if row["apply_index"] != row["decision_index_t"] + 1:
        raise ValueError("Invalid timing: first newly solved MPC control must be applied at t+1.")
    if row["mpc_stage0_source"] == "lstm_h1_to_h6":
        if not np.isclose(row["first_mpc_ref_load"], row["pred_h1"], atol=1e-8):
            raise ValueError("Invalid timing: MPC stage 0 must use LSTM h1 when LSTM is available.")
    elif row["mpc_stage0_source"] == "current_load_hold":
        if not np.isclose(row["first_mpc_ref_load"], row["actual_load_t"], atol=1e-8):
            raise ValueError("Invalid timing: current-load hold must repeat the measured load.")
    else:
        raise ValueError(f"Invalid timing: unsupported forecast source {row['mpc_stage0_source']!r}.")
    return row


def _feature_matrix(df_voyage: pd.DataFrame, payload: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_set = payload.get("feature_set", "rolling")
    pdf = _prepare_features(df_voyage, feature_set)
    features = list(payload["features"])
    for col in ["time_sin", "time_cos"]:
        if col in pdf.columns and col not in features:
            features.append(col)
    for col in features:
        if col not in pdf.columns:
            pdf[col] = 0.0
    vals = pdf[features].astype(float).ffill().fillna(0.0).to_numpy()
    vals = transform(vals, payload["feature_scaler"])
    actual = pdf["load_total_kw"].astype(float).ffill().fillna(0.0).to_numpy()
    return pdf, vals, actual


def forecast_voyage(
    df_voyage: pd.DataFrame,
    model: torch.nn.Module,
    payload: dict[str, Any],
    device: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = payload["config"]
    history_len = int(cfg["history_len"])
    pred_horizon = int(cfg["pred_horizon"])
    assert pred_horizon == 6
    pdf, vals, actual = _feature_matrix(df_voyage, payload)
    if len(pdf) < history_len:
        raise ValueError("Voyage is shorter than LSTM history length.")
    decision_indices = np.arange(history_len - 1, len(pdf), dtype=int)
    pred = np.full((len(decision_indices), pred_horizon), np.nan, dtype=float)
    true = np.full((len(decision_indices), pred_horizon), np.nan, dtype=float)
    model.eval()
    with torch.no_grad():
        for row_idx, decision_t in enumerate(decision_indices):
            hist = vals[decision_t - history_len + 1 : decision_t + 1]
            if hist.shape[0] != history_len:
                raise ValueError("Internal error: invalid LSTM history length.")
            x = torch.as_tensor(hist[None, :, :], dtype=torch.float32, device=device)
            raw = model(x).detach().cpu().numpy()[0]
            pred[row_idx] = np.maximum(inverse_target(raw, payload["target_scaler"]), 0.0)
            for h in range(1, pred_horizon + 1):
                true_idx = decision_t + h
                if true_idx < len(actual):
                    true[row_idx, h - 1] = actual[true_idx]
    return pdf, decision_indices, pred, true, actual


def _fallback_controls(
    load_total: float,
    prev_fc: float,
    cfg: CasadiMPCConfig,
) -> tuple[float, float, float]:
    fc = float(np.clip(prev_fc, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw))
    if cfg.fuel_cell_ramp_constraint_enabled:
        fc = float(np.clip(fc, prev_fc - cfg.fuel_cell_ramp_kw, prev_fc + cfg.fuel_cell_ramp_kw))
    batt = float(load_total - fc)
    unserved = 0.0
    if batt > cfg.battery_discharge_max_kw:
        unserved = batt - cfg.battery_discharge_max_kw
        batt = cfg.battery_discharge_max_kw
    batt = float(np.clip(batt, -cfg.battery_charge_max_kw, cfg.battery_discharge_max_kw))
    return fc, batt, unserved


def execute_cached_fc_step(
    *,
    actual_load_kw: float,
    fc_command_kw: float,
    soc_before: float,
    cfg: CasadiMPCConfig,
) -> dict[str, float | bool]:
    fc_executed = float(np.clip(fc_command_kw, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw))
    batt_actual = float(actual_load_kw - fc_executed)
    unserved_kw = float(max(batt_actual - cfg.battery_discharge_max_kw, 0.0))
    curtailed_kw = float(max(-cfg.battery_charge_max_kw - batt_actual, 0.0))
    discharge_limited = bool(unserved_kw > 1e-9)
    charge_limited = bool(curtailed_kw > 1e-9)
    batt_for_soc = float(np.clip(batt_actual, -cfg.battery_charge_max_kw, cfg.battery_discharge_max_kw))
    soc_after = float(soc_before - batt_for_soc * cfg.dt_hours / cfg.battery_capacity_kwh)
    return {
        "P_fc_executed_kw": fc_executed,
        "P_batt_actual_kw": batt_actual,
        "P_batt_for_soc_kw": batt_for_soc,
        "unserved_power_kw": unserved_kw,
        "curtailed_power_kw": curtailed_kw,
        "battery_discharge_limit_active": discharge_limited,
        "battery_charge_limit_active": charge_limited,
        "SOC_after": soc_after,
    }


def run_single_voyage(
    *,
    voyage_id: int,
    voyage_name: str,
    df_voyage: pd.DataFrame,
    model: torch.nn.Module,
    payload: dict[str, Any],
    device: str,
    weight_set: str,
    weights: dict[str, Any],
    output_dir: Path,
    init_soc: float = 0.55,
    make_plots: bool = True,
    write_outputs: bool = True,
    max_steps: int | None = None,
    run_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    cfg = mpc_config_from_weights(weights)
    generator = REFERENCE_GENERATOR_CLASS(cfg)
    pdf, lstm_decision_indices, pred, true, actual = forecast_voyage(df_voyage, model, payload, device)
    pred_by_decision = {int(idx): pred[row_idx] for row_idx, idx in enumerate(lstm_decision_indices)}
    true_by_decision = {int(idx): true[row_idx] for row_idx, idx in enumerate(lstm_decision_indices)}
    decision_indices = np.arange(len(pdf), dtype=int)
    if max_steps is not None:
        decision_indices = decision_indices[: int(max_steps)]
    history_len = int(payload["config"]["history_len"])
    pred_horizon = int(payload["config"]["pred_horizon"])
    assert pred_horizon == 6
    assert cfg.prediction_horizon == 6

    dt_h = cfg.dt_hours
    soc = float(init_soc)
    voyage_soc_ref = float(init_soc)
    initial_fc_cmd = float(np.clip(actual[0], cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw))
    cached_fc_cmd = initial_fc_cmd
    rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for decision_t in decision_indices:
        current_load = float(actual[int(decision_t)])
        history_available = int(decision_t) - history_len + 1 >= 0
        raw_lstm_pred = pred_by_decision.get(int(decision_t), np.full(pred_horizon, np.nan, dtype=float))
        true_row = true_by_decision.get(int(decision_t), np.full(pred_horizon, np.nan, dtype=float))
        lstm_available = bool(history_available and np.all(np.isfinite(raw_lstm_pred[:pred_horizon])))
        forecast_source = forecast_source_name(history_available=history_available, lstm_available=lstm_available)
        mpc_ref_total = build_mpc_load_ref(
            current_load,
            raw_lstm_pred,
            pred_horizon=pred_horizon,
            mpc_horizon=cfg.prediction_horizon,
            lstm_available=lstm_available,
        )
        if forecast_source == "lstm_h1_to_h6" and not np.allclose(mpc_ref_total, raw_lstm_pred[:6], atol=1e-8):
            raise AssertionError("Invalid MPC timing: mpc_load_ref must equal LSTM h1-h6 when LSTM is available.")
        actual_next = float(actual[int(decision_t) + 1]) if int(decision_t) + 1 < len(actual) else None
        debug_rows.append(
            make_debug_timing_row(
                voyage_id=voyage_id,
                decision_index_t=int(decision_t),
                history_len=history_len,
                pred_horizon=pred_horizon,
                mpc_load_ref=mpc_ref_total,
                actual_load_t=current_load,
                pred_h1=float(raw_lstm_pred[0]) if np.isfinite(raw_lstm_pred[0]) else float("nan"),
                actual_load_t_plus_1=actual_next,
                history_available=history_available,
                lstm_available=lstm_available,
                forecast_source=forecast_source,
            )
        )
        soc_before = soc
        executed = execute_cached_fc_step(
            actual_load_kw=current_load,
            fc_command_kw=cached_fc_cmd,
            soc_before=soc_before,
            cfg=cfg,
        )
        fc_executed = float(executed["P_fc_executed_kw"])
        batt_actual = float(executed["P_batt_actual_kw"])
        batt_for_soc = float(executed["P_batt_for_soc_kw"])
        soc = float(executed["SOC_after"])
        unserved_kw = float(executed["unserved_power_kw"])
        curtailed_kw = float(executed["curtailed_power_kw"])
        solve_start = time.perf_counter()
        success = True
        solver_message = ""
        objective_value = np.nan
        objective_info: dict[str, Any] = {}
        fallback_control_used = False
        fallback_reason = ""
        forecast_mode = forecast_source
        fc_next_cmd = float(fc_executed)
        batt_plan_stage0 = float(mpc_ref_total[0] - fc_next_cmd)
        fc_plan_traj = np.full(cfg.prediction_horizon, np.nan, dtype=float)
        batt_plan_traj = np.full(cfg.prediction_horizon, np.nan, dtype=float)
        soc_plan_traj = np.full(cfg.prediction_horizon + 1, np.nan, dtype=float)
        try:
            result = generator.generate_result(
                load_forecast_kw=mpc_ref_total,
                current_soc=soc,
                prev_fc_kw=fc_executed,
                soc_reference_value=voyage_soc_ref,
            )
            fc_next_cmd = float(result.fuel_cell_ref_kw)
            batt_plan_stage0 = float(result.battery_ref_kw)
            success = bool(result.success)
            objective_value = float(result.objective_value)
            objective_info = dict(result.objective_info)
            solver_message = str(objective_info.get("solver_message", ""))
            fc_plan_traj = np.asarray(result.fuel_cell_ref_traj_kw, dtype=float).reshape(-1)
            batt_plan_traj = np.asarray(result.battery_ref_traj_kw, dtype=float).reshape(-1)
            soc_plan_traj = np.asarray(result.soc_pred_traj, dtype=float).reshape(-1)
        except Exception as exc:  # pragma: no cover - solver failure path is environment dependent
            success = False
            solver_message = f"{type(exc).__name__}: {exc}"
            fallback_control_used = True
            fallback_reason = solver_message
            forecast_mode = f"{forecast_source}_fallback_after_solver_exception"
            fc_next_cmd, batt_plan_stage0, _ = _fallback_controls(mpc_ref_total[0], fc_executed, cfg)
            fc_plan_traj[0] = fc_next_cmd
            batt_plan_traj[0] = batt_plan_stage0
            soc_plan_traj[0] = soc
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        cached_fc_cmd = float(fc_next_cmd)
        served_kw = max(0.0, current_load - unserved_kw)
        row_soc_ref = float(objective_info.get("soc_ref_value", resolve_soc_reference(cfg, soc, voyage_soc_ref)))
        timestamp_value = (
            str(pdf["timestamp"].iloc[int(decision_t)])
            if "timestamp" in pdf.columns and int(decision_t) < len(pdf)
            else f"relative_time_h={float(decision_t) * DT_SECONDS / 3600.0:.6f}"
        )

        row: dict[str, Any] = {
            "run_id": run_id,
            "voyage_id": int(voyage_id),
            "voyage_name": voyage_name,
            "file_name": str(df_voyage["file_name"].iloc[0]) if "file_name" in df_voyage.columns else voyage_name,
            "decision_index_t": int(decision_t),
            "timestamp": timestamp_value,
            "time_h": float(decision_t) * DT_SECONDS / 3600.0,
            "weight_set": weight_set,
            "load_total_kw": current_load,
            "P_fc_kw": fc_executed,
            "P_batt_kw": batt_actual,
            "P_fc_executed_kw": fc_executed,
            "P_fc_next_cmd_kw": fc_next_cmd,
            "P_batt_actual_kw": batt_actual,
            "P_batt_for_soc_kw": batt_for_soc,
            "P_batt_plan_stage0_kw": batt_plan_stage0,
            "P_fc_prev_kw": fc_executed,
            "initial_fc_cmd_kw": initial_fc_cmd,
            "SOC_before": soc_before,
            "SOC": soc,
            "control_timing": CONTROL_TIMING,
            "control_apply_timing": CONTROL_APPLY_TIMING,
            "forecast_source": forecast_source,
            "load_definition": LOAD_DEFINITION,
            "capacity_basis": CAPACITY_BASIS,
            "battery_capacity_kwh": float(cfg.battery_capacity_kwh),
            "fuel_cell_ramp_constraint_enabled": bool(cfg.fuel_cell_ramp_constraint_enabled),
            "fuel_cell_ramp_kw": float(cfg.fuel_cell_ramp_kw),
            "enable_fc_ramp_hard_constraint": bool(cfg.fuel_cell_ramp_constraint_enabled),
            "fc_ramp_limit_kw_per_step": float(cfg.fuel_cell_ramp_kw),
            "initial_fc_mode": str(cfg.initial_fc_mode),
            "soc_reference_mode": str(cfg.soc_reference_mode),
            "soc_penalty_type": str(cfg.soc_penalty_type),
            "soc_ref_value": row_soc_ref,
            "soc_reserve": float(cfg.soc_reserve),
            "soc_band": float(cfg.soc_band),
            "terminal_soc_band": float(cfg.terminal_soc_band),
            "q_h2": float(cfg.q_h2),
            "q_soc": float(cfg.q_soc),
            "q_batt": float(cfg.q_batt),
            "q_ramp": float(cfg.q_ramp),
            "q_terminal_soc": float(cfg.q_terminal_soc) if cfg.q_terminal_soc is not None else 0.0,
            "solver_success": bool(success),
            "solver_message": solver_message,
            "mpc_solved": bool(success),
            "forecast_mode": forecast_mode,
            "fallback_control_used": bool(fallback_control_used),
            "fallback_reason": fallback_reason,
            "history_available": bool(history_available),
            "lstm_available": bool(lstm_available),
            "solve_ms": solve_ms,
            "objective_value": objective_value,
            "unserved_power_kw": unserved_kw,
            "curtailed_power_kw": curtailed_kw,
            "battery_discharge_limit_active": bool(executed["battery_discharge_limit_active"]),
            "battery_charge_limit_active": bool(executed["battery_charge_limit_active"]),
            "load_served_kw": served_kw,
            "objective_terminal_soc_term": float(objective_info.get("terminal_soc_term", 0.0)),
            "objective_total_mpc_cost": float(objective_info.get("total_mpc_cost", objective_value if np.isfinite(objective_value) else 0.0)),
        }
        row.update(objective_info_fields(objective_info))
        for h in range(1, pred_horizon + 1):
            row[f"pred_h{h}"] = float(raw_lstm_pred[h - 1]) if lstm_available else np.nan
            row[f"true_h{h}"] = float(true_row[h - 1]) if np.isfinite(true_row[h - 1]) else np.nan
        for k in range(cfg.prediction_horizon):
            row[f"mpc_ref_load_stage{k}"] = float(mpc_ref_total[k])
            row[f"P_fc_plan_stage{k}"] = (
                float(fc_plan_traj[k]) if k < len(fc_plan_traj) and np.isfinite(fc_plan_traj[k]) else np.nan
            )
            row[f"P_batt_plan_stage{k}"] = (
                float(batt_plan_traj[k]) if k < len(batt_plan_traj) and np.isfinite(batt_plan_traj[k]) else np.nan
            )
        for k in range(cfg.prediction_horizon + 1):
            row[f"SOC_plan_stage{k}"] = (
                float(soc_plan_traj[k]) if k < len(soc_plan_traj) and np.isfinite(soc_plan_traj[k]) else np.nan
            )
        rows.append(row)

    ts = pd.DataFrame(rows)
    debug = pd.DataFrame(debug_rows)
    horizon_metrics = compute_horizon_metrics(ts, label=str(voyage_id))
    horizon_metrics["run_id"] = run_id
    summary = compute_closed_loop_metrics(ts, cfg, voyage_id=voyage_id, voyage_name=voyage_name, weight_set=weight_set)
    summary_for_output = clean_reporting_dict_for_weight_set(summary, weight_set=weight_set)
    summary_for_output["run_id"] = run_id
    ts_for_output = clean_reporting_frame_for_weight_set(ts, weight_set=weight_set)
    no_reserve_no_terminal = uses_no_reserve_no_terminal_reporting(weight_set)

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_for_output.to_csv(output_dir / f"voyage_{voyage_id}_timeseries.csv", index=False, encoding="utf-8-sig")
        write_json(output_dir / f"voyage_{voyage_id}_summary.json", summary_for_output)
        if make_plots:
            title = (
                f"voyage_id={voyage_id} run_id={run_id} duration_h={summary_for_output['duration_h']:.1f} weight_set={weight_set} "
                f"forecast_mode={FORECAST_MODE} mpc_horizon=6 control_timing={CONTROL_TIMING}"
            )
            plot_load_forecast(ts_for_output, actual, title, output_dir / f"voyage_{voyage_id}_load_forecast.png")
            plot_power_split(ts_for_output, title, output_dir / f"voyage_{voyage_id}_power_split.png")
            plot_soc(
                ts_for_output,
                cfg,
                title,
                output_dir / f"voyage_{voyage_id}_soc.png",
                show_soc_reserve=not no_reserve_no_terminal,
                show_soc_bounds=not no_reserve_no_terminal,
            )
    return summary_for_output, horizon_metrics, ts, debug


def compute_horizon_metrics(ts: pd.DataFrame, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"voyage_id": label}
    for h in range(1, 7):
        pred = ts[f"pred_h{h}"].to_numpy(dtype=float)
        true = ts[f"true_h{h}"].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(true)
        if not np.any(mask):
            out[f"RMSE_h{h}"] = np.nan
            out[f"MAE_h{h}"] = np.nan
            out[f"WAPE_h{h}"] = np.nan
            out[f"count_h{h}"] = 0
            continue
        err = pred[mask] - true[mask]
        out[f"RMSE_h{h}"] = float(np.sqrt(np.mean(err**2)))
        out[f"MAE_h{h}"] = float(np.mean(np.abs(err)))
        out[f"WAPE_h{h}"] = float(np.sum(np.abs(err)) / (np.sum(np.abs(true[mask])) + 1e-6) * 100.0)
        out[f"count_h{h}"] = int(np.sum(mask))
    return out


def compute_low_load_fc_metrics(
    *,
    load: np.ndarray,
    fc: np.ndarray,
    h2_step_kg: np.ndarray,
    cfg: CasadiMPCConfig,
) -> dict[str, float]:
    threshold = 5.0
    idle_upper = 2.0
    dt_h = float(cfg.dt_hours)
    dt_s = dt_h * 3600.0
    min_steps = max(1, int(np.ceil(300.0 / max(dt_s, 1e-9))))
    low_mask = np.asarray(load, dtype=float) < threshold
    shutdown_times_s: list[float] = []
    idle_h2_kg = 0.0
    start: int | None = None

    for idx, is_low in enumerate(np.r_[low_mask, False]):
        if is_low and start is None:
            start = int(idx)
        if (not is_low) and start is not None:
            end = int(idx)
            if end - start >= min_steps:
                segment_fc = fc[start:end]
                segment_h2 = h2_step_kg[start:end]
                idle_h2_kg += float(np.sum(segment_h2[segment_fc > idle_upper]))
                hit = np.where(segment_fc < idle_upper)[0]
                if len(hit):
                    shutdown_times_s.append(float(hit[0]) * dt_s)
                else:
                    shutdown_times_s.append(float("nan"))
            start = None

    finite_shutdown = [value for value in shutdown_times_s if np.isfinite(value)]
    if finite_shutdown:
        shutdown_s = float(max(finite_shutdown))
    elif shutdown_times_s:
        shutdown_s = float("nan")
    else:
        shutdown_s = 0.0
    return {
        "fc_shutdown_time_after_load_zero_s": shutdown_s,
        "fc_shutdown_time_after_load_zero_min": shutdown_s / 60.0 if np.isfinite(shutdown_s) else float("nan"),
        "fc_idle_h2_consumption_kg": float(idle_h2_kg),
        "low_load_duration_h": float(np.sum(low_mask) * dt_h),
    }


def max_continuous_true_seconds(mask: np.ndarray, dt_seconds: float) -> float:
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    max_run = 0
    run = 0
    for value in arr:
        if value:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return float(max_run * float(dt_seconds))


def soc_delta_at_elapsed(ts: pd.DataFrame, soc_start: float, elapsed_min: float) -> float:
    if ts.empty or "time_h" not in ts.columns:
        return 0.0
    time_h = pd.to_numeric(ts["time_h"], errors="coerce").to_numpy(dtype=float)
    soc = pd.to_numeric(ts["SOC"], errors="coerce").to_numpy(dtype=float)
    if len(time_h) == 0 or not np.isfinite(time_h[0]):
        return 0.0
    target_h = float(time_h[0]) + float(elapsed_min) / 60.0
    mask = np.isfinite(time_h) & np.isfinite(soc) & (time_h <= target_h + 1e-12)
    if not np.any(mask):
        return 0.0
    return float(soc[np.where(mask)[0][-1]] - float(soc_start))


def compute_closed_loop_metrics(
    ts: pd.DataFrame,
    cfg: CasadiMPCConfig,
    *,
    voyage_id: int,
    voyage_name: str,
    weight_set: str,
) -> dict[str, Any]:
    fc = ts["P_fc_kw"].to_numpy(dtype=float)
    batt = ts["P_batt_kw"].to_numpy(dtype=float)
    soc = ts["SOC"].to_numpy(dtype=float)
    load = ts["load_total_kw"].to_numpy(dtype=float)
    unserved = ts["unserved_power_kw"].to_numpy(dtype=float)
    curtailed = (
        ts["curtailed_power_kw"].to_numpy(dtype=float)
        if "curtailed_power_kw" in ts.columns
        else np.zeros(len(ts), dtype=float)
    )
    ramp = np.diff(fc)
    abs_ramp = np.abs(ramp)
    ramp_limit_kw = float(cfg.fuel_cell_ramp_kw)
    dt_h = float(cfg.dt_hours)
    h2_step_kg = h2_kg_step_dp0_quadratic(
        fc,
        dt_seconds=dt_h * 3600.0,
        p_rated_total_kw=float(cfg.fuel_cell_max_kw),
    )
    low_load_metrics = compute_low_load_fc_metrics(load=load, fc=fc, h2_step_kg=h2_step_kg, cfg=cfg)
    h2_total_kg = float(np.sum(h2_step_kg))
    fc_energy_kwh = float(np.sum(fc) * dt_h)
    h2_per_fc_kwh = h2_total_kg / max(fc_energy_kwh, 1e-12)
    soc_start = float(ts["SOC_before"].iloc[0])
    soc_end = float(soc[-1])
    if "soc_ref_value" in ts.columns:
        soc_ref_series = pd.to_numeric(ts["soc_ref_value"], errors="coerce").dropna()
        soc_ref_value = float(soc_ref_series.iloc[0]) if not soc_ref_series.empty else resolve_soc_reference(cfg, soc_start, soc_start)
    else:
        soc_ref_value = resolve_soc_reference(cfg, soc_start, soc_start)
    soc_spending_kwh = float((soc_start - soc_end) * cfg.battery_capacity_kwh)
    charge_sustaining_h2_adjustment_kg = soc_spending_kwh * h2_per_fc_kwh
    if str(cfg.soc_reference_mode) == "reserve_only":
        soc_low_threshold = float(cfg.soc_reserve)
        soc_terminal_error = float(max(cfg.soc_reserve - soc_end, 0.0))
        soc_tracking_mae = float(np.mean(np.maximum(cfg.soc_reserve - soc, 0.0)))
    else:
        soc_low_threshold = float(soc_ref_value - cfg.soc_band)
        soc_terminal_error = float(abs(soc_end - soc_ref_value))
        soc_tracking_mae = float(np.mean(np.abs(soc - soc_ref_value)))
    first_10_steps = max(1, int(np.ceil((10.0 / 60.0) / max(dt_h, 1e-12))))
    first_10_soc = soc[: min(len(soc), first_10_steps)]
    soc_rise_first_10min = float(np.max(first_10_soc) - soc_start) if len(first_10_soc) else 0.0
    charge_power = np.maximum(-batt, 0.0)
    discharge_power = np.maximum(batt, 0.0)
    dt_seconds = dt_h * 3600.0
    soc_reserve_gap = soc - float(cfg.soc_reserve)
    fc_minus_load = fc - load
    battery_covers_load_gt_80pct = (load > 20.0) & (fc < 0.2 * load) & (batt > 0.8 * load)
    fc_off_under_load = (load > 20.0) & (fc < 5.0)
    if "time_h" in ts.columns and len(ts):
        t = pd.to_numeric(ts["time_h"], errors="coerce").to_numpy(dtype=float)
        initial_mask = t <= float(t[0]) + 1.5 + 1e-12
    else:
        initial_steps = min(len(ts), int(np.ceil(1.5 / max(dt_h, 1e-12))))
        initial_mask = np.zeros(len(ts), dtype=bool)
        initial_mask[:initial_steps] = True
    initial_battery_only_time_s = float(np.sum(battery_covers_load_gt_80pct & initial_mask) * dt_seconds)
    initial_window_s = float(max(np.sum(initial_mask), 1) * dt_seconds)
    time_batt_covers_load_gt_80pct_s = float(np.sum(battery_covers_load_gt_80pct) * dt_seconds)
    time_fc_below_5kw_while_load_above_20kw_s = float(np.sum(fc_off_under_load) * dt_seconds)
    max_continuous_fc_off_under_load_s = max_continuous_true_seconds(fc_off_under_load, dt_seconds)
    fallback_values = (
        ts["fallback_control_used"].map(bool01).to_numpy(dtype=float)
        if "fallback_control_used" in ts.columns
        else np.zeros(len(ts), dtype=float)
    )
    metrics = {
        "weight_set": weight_set,
        "voyage_id": int(voyage_id),
        "voyage_name": voyage_name,
        "file_name": str(ts["file_name"].iloc[0]) if "file_name" in ts.columns else voyage_name,
        "duration_h": float(len(ts) * dt_h),
        "control_timing": CONTROL_TIMING,
        "control_apply_timing": CONTROL_APPLY_TIMING,
        "load_definition": LOAD_DEFINITION,
        "capacity_basis": CAPACITY_BASIS,
        "battery_capacity_kwh": float(cfg.battery_capacity_kwh),
        "fuel_cell_ramp_constraint_enabled": bool(cfg.fuel_cell_ramp_constraint_enabled),
        "fuel_cell_ramp_kw": float(cfg.fuel_cell_ramp_kw),
        "enable_fc_ramp_hard_constraint": bool(cfg.fuel_cell_ramp_constraint_enabled),
        "fc_ramp_limit_kw_per_step": float(cfg.fuel_cell_ramp_kw),
        "initial_fc_mode": str(cfg.initial_fc_mode),
        "q_h2": float(cfg.q_h2),
        "q_soc": float(cfg.q_soc),
        "q_batt": float(cfg.q_batt),
        "q_ramp": float(cfg.q_ramp),
        "q_terminal_soc": float(cfg.q_terminal_soc) if cfg.q_terminal_soc is not None else 0.0,
        "forecast_source_current_load_hold_steps": int((ts["forecast_source"] == "current_load_hold").sum())
        if "forecast_source" in ts.columns
        else 0,
        "forecast_source_lstm_steps": int((ts["forecast_source"] == "lstm_h1_to_h6").sum())
        if "forecast_source" in ts.columns
        else 0,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_delta": float(soc_end - soc_start),
        "SOC_start": soc_start,
        "SOC_end": soc_end,
        "SOC_delta": float(soc_end - soc_start),
        "soc_reference_mode": str(cfg.soc_reference_mode),
        "soc_ref_value": float(soc_ref_value),
        "soc_reserve": float(cfg.soc_reserve),
        "soc_min": float(np.min(soc)),
        "soc_max": float(np.max(soc)),
        "SOC_min": float(np.min(soc)),
        "SOC_max": float(np.max(soc)),
        "soc_mean": float(np.mean(soc)),
        "soc_std": float(np.std(soc, ddof=1)) if len(soc) > 1 else 0.0,
        "soc_slope_std": float(np.std(np.diff(soc), ddof=1)) if len(soc) > 2 else 0.0,
        "soc_step_max_abs": float(np.max(np.abs(np.diff(soc)))) if len(soc) > 1 else 0.0,
        "soc_low_threshold": soc_low_threshold,
        "time_below_SOC_low_h": float(np.sum(soc < soc_low_threshold) * dt_h),
        "time_below_SOC_min_h": float(np.sum(soc < cfg.soc_min) * dt_h),
        "soc_below_reserve_duration_s": float(np.sum(soc < cfg.soc_reserve) * dt_seconds),
        "soc_below_reserve_min_gap": float(np.min(soc_reserve_gap)) if len(soc_reserve_gap) else 0.0,
        "soc_terminal_error": soc_terminal_error,
        "soc_target_tracking_mae": soc_tracking_mae,
        "soc_rise_first_10min": soc_rise_first_10min,
        "soc_drop_first_10min": soc_delta_at_elapsed(ts, soc_start, 10.0),
        "soc_drop_first_30min": soc_delta_at_elapsed(ts, soc_start, 30.0),
        "H2_total_kg": h2_total_kg,
        "h2_consumption_kg": h2_total_kg,
        "charge_sustaining_adjusted_H2": float(h2_total_kg + charge_sustaining_h2_adjustment_kg),
        "charge_sustaining_adjusted_h2_kg": float(h2_total_kg + charge_sustaining_h2_adjustment_kg),
        "charge_sustaining_h2_adjustment_kg": float(charge_sustaining_h2_adjustment_kg),
        "soc_spending_kwh": soc_spending_kwh,
        "fc_energy_kwh": fc_energy_kwh,
        "max_fc_kw": float(np.max(fc)),
        "load_mean_kw": float(np.mean(load)),
        "load_max_kw": float(np.max(load)),
        "load_min_kw": float(np.min(load)),
        "fc_mean_kw": float(np.mean(fc)),
        "fc_std_kw": float(np.std(fc, ddof=1)) if len(fc) > 1 else 0.0,
        "fc_above_load_energy_kwh": float(np.sum(np.maximum(fc_minus_load, 0.0)) * dt_h),
        "fc_below_load_energy_kwh": float(np.sum(np.maximum(-fc_minus_load, 0.0)) * dt_h),
        "fc_load_tracking_mae": float(np.mean(np.abs(fc_minus_load))) if len(fc_minus_load) else 0.0,
        "fc_load_tracking_bias": float(np.mean(fc_minus_load)) if len(fc_minus_load) else 0.0,
        "P_fc_std": float(np.std(fc, ddof=1)) if len(fc) > 1 else 0.0,
        "fc_ramp_mean_kw": float(np.mean(abs_ramp)) if len(abs_ramp) else 0.0,
        "fc_ramp_p95_kw": float(np.percentile(abs_ramp, 95)) if len(abs_ramp) else 0.0,
        "fc_ramp_max_kw": float(np.max(abs_ramp)) if len(abs_ramp) else 0.0,
        "fc_ramp_violation_count": int(np.sum(abs_ramp > ramp_limit_kw + 1e-6)) if len(abs_ramp) else 0,
        "P_fc_startup_slope_max": float(np.max(np.maximum(ramp, 0.0))) if len(ramp) else 0.0,
        "P_fc_shutdown_slope_max": float(np.max(np.maximum(-ramp, 0.0))) if len(ramp) else 0.0,
        "P_fc_ramp_mean": float(np.mean(abs_ramp)) if len(abs_ramp) else 0.0,
        "P_fc_ramp_max": float(np.max(abs_ramp)) if len(abs_ramp) else 0.0,
        "P_batt_std": float(np.std(batt, ddof=1)) if len(batt) > 1 else 0.0,
        "battery_discharge_kwh": float(np.sum(np.maximum(batt, 0.0)) * dt_h),
        "battery_charge_kwh": float(np.sum(np.maximum(-batt, 0.0)) * dt_h),
        "battery_net_discharge_kwh": float(np.sum(np.maximum(batt, 0.0)) * dt_h - np.sum(np.maximum(-batt, 0.0)) * dt_h),
        "battery_discharge_energy": float(np.sum(discharge_power) * dt_h),
        "battery_charge_energy": float(np.sum(charge_power) * dt_h),
        "battery_throughput_kwh": float(np.sum(np.abs(batt)) * dt_h),
        "max_charge_power_kw_observed": float(np.max(charge_power)),
        "max_discharge_power_kw_observed": float(np.max(discharge_power)),
        "max_batt_charge_kw": float(np.max(charge_power)),
        "max_batt_discharge_kw": float(np.max(discharge_power)),
        "time_fc_above_300kw_s": float(np.sum(fc > 300.0) * dt_h * 3600.0),
        "time_fc_above_400kw_s": float(np.sum(fc > 400.0) * dt_h * 3600.0),
        "time_batt_charge_above_200kw_s": float(np.sum(charge_power > 200.0) * dt_h * 3600.0),
        "time_batt_charge_above_300kw_s": float(np.sum(charge_power > 300.0) * dt_h * 3600.0),
        "initial_battery_only_time_s": initial_battery_only_time_s,
        "initial_battery_only_time_min": initial_battery_only_time_s / 60.0,
        "initial_battery_only_ratio": float(initial_battery_only_time_s / max(initial_window_s, 1e-12)),
        "time_batt_covers_load_gt_80pct_s": time_batt_covers_load_gt_80pct_s,
        "time_batt_covers_load_gt_80pct_min": time_batt_covers_load_gt_80pct_s / 60.0,
        "time_fc_below_5kw_while_load_above_20kw_s": time_fc_below_5kw_while_load_above_20kw_s,
        "time_fc_below_5kw_while_load_above_20kw_min": time_fc_below_5kw_while_load_above_20kw_s / 60.0,
        "max_continuous_fc_off_under_load_s": max_continuous_fc_off_under_load_s,
        "max_continuous_fc_off_under_load_min": max_continuous_fc_off_under_load_s / 60.0,
        "fallback_control_used_count": int(np.sum(fallback_values > 0.5)),
        "fallback_count": int(np.sum(fallback_values > 0.5)),
        "infeasible_count": int(np.sum(ts["solver_success"].astype(float).to_numpy(dtype=float) < 0.5)),
        "fallback_control_used_ratio": float(np.mean(fallback_values > 0.5)) if len(fallback_values) else 0.0,
        "battery_rms_kw": float(np.sqrt(np.mean(batt**2))) if len(batt) else 0.0,
        "battery_abs_mean_kw": float(np.mean(np.abs(batt))) if len(batt) else 0.0,
        "load_served_kwh": float(np.sum(np.maximum(load - unserved, 0.0)) * dt_h),
        "unserved_energy_kwh": float(np.sum(np.maximum(unserved, 0.0)) * dt_h),
        "curtailed_energy_kwh": float(np.sum(np.maximum(curtailed, 0.0)) * dt_h),
        "solver_success_rate": float(ts["solver_success"].astype(float).mean()),
        "objective_value_mean": float(np.nanmean(ts["objective_value"].to_numpy(dtype=float))),
        "solve_ms_mean": float(np.mean(ts["solve_ms"].to_numpy(dtype=float))),
        "solve_ms_p95": float(np.percentile(ts["solve_ms"].to_numpy(dtype=float), 95)),
    }
    metrics.update(low_load_metrics)
    for key in [
        "weighted_h2_cost",
        "weighted_soc_cost",
        "weighted_ramp_cost",
        "weighted_batt_cost",
        "weighted_terminal_soc_cost",
        "total_objective",
    ]:
        if key in ts.columns:
            values = pd.to_numeric(ts[key], errors="coerce").to_numpy(dtype=float)
            metrics[f"{key}_mean"] = float(np.nanmean(values))
        else:
            metrics[f"{key}_mean"] = 0.0
    return metrics


def compute_objective_breakdown(ts: pd.DataFrame) -> dict[str, Any]:
    row: dict[str, Any] = {
        "weight_set": str(ts["weight_set"].iloc[0]),
        "voyage_id": int(ts["voyage_id"].iloc[0]),
        "voyage_name": str(ts["voyage_name"].iloc[0]),
    }
    for key in OBJECTIVE_BREAKDOWN_FIELDS:
        if key not in ts.columns:
            values = np.zeros(len(ts), dtype=float)
        else:
            values = pd.to_numeric(ts[key], errors="coerce").to_numpy(dtype=float)
        row[key] = float(np.nanmean(values))
        row[f"{key}_sum"] = float(np.nansum(values))
        row[f"{key}_max"] = float(np.nanmax(values)) if len(values) else 0.0
    for key in ["H2_step_kg_sum", "total_mpc_cost"]:
        values = pd.to_numeric(ts[key], errors="coerce").to_numpy(dtype=float)
        row[f"{key}_mean"] = float(np.nanmean(values))
        row[f"{key}_sum"] = float(np.nansum(values))
    total = float(row.get("total_objective_sum", row.get("total_mpc_cost_sum", 0.0)))
    denom = total if abs(total) > 1e-12 else float("nan")
    for key in [
        "weighted_h2_cost",
        "weighted_soc_cost",
        "weighted_batt_cost",
        "weighted_ramp_cost",
    ]:
        row[f"{key}_ratio"] = float(row.get(f"{key}_sum", 0.0) / denom) if np.isfinite(denom) else 0.0
    return row


def plot_load_forecast(ts: pd.DataFrame, actual: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    t_actual = np.arange(len(actual)) * DT_SECONDS / 3600.0
    ax.plot(t_actual, actual, "k-", lw=0.7, alpha=0.8, label="actual load")
    base_t = ts["decision_index_t"].to_numpy(dtype=float) * DT_SECONDS / 3600.0
    for h, color, lw in [(1, "tab:blue", 0.65), (5, "tab:orange", 0.55), (6, "tab:green", 0.55)]:
        ax.plot(base_t + h * DT_SECONDS / 3600.0, ts[f"pred_h{h}"], color=color, lw=lw, alpha=0.85, label=f"pred h={h}")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Load (kW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_power_split(ts: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    t = ts["time_h"].to_numpy(dtype=float)
    ax.plot(t, ts["load_total_kw"], "k-", lw=0.8, label="load")
    ax.plot(t, ts["P_fc_kw"], "tab:red", lw=0.75, label="P_fc")
    ax.plot(t, ts["P_batt_kw"], "tab:blue", lw=0.75, label="P_batt")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_soc(
    ts: pd.DataFrame,
    cfg: CasadiMPCConfig,
    title: str,
    out_path: Path,
    *,
    show_soc_reserve: bool = True,
    show_soc_bounds: bool = True,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    t = ts["time_h"].to_numpy(dtype=float)
    ax.plot(t, ts["SOC"], "tab:green", lw=0.8, label="SOC")
    if "soc_ref_value" in ts.columns:
        ax.plot(t, ts["soc_ref_value"], color="gray", ls="--", lw=0.8, label="SOC_ref")
    else:
        ax.axhline(cfg.soc_target, color="gray", ls="--", lw=0.8, label="SOC_ref")
    if show_soc_reserve and "soc_reserve" in ts.columns:
        ax.plot(t, ts["soc_reserve"], color="tab:orange", ls="-.", lw=0.8, label="SOC_reserve")
    if show_soc_bounds:
        ax.axhline(cfg.soc_min, color="tab:red", ls=":", lw=0.8, label="SOC_min")
        ax.axhline(cfg.soc_max, color="tab:red", ls=":", lw=0.8, label="SOC_max")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SOC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fixed_mpc_comparison(
    all_ts: list[pd.DataFrame],
    out_path: Path,
    *,
    show_soc_reserve: bool = True,
    plot_metadata: dict[str, Any] | None = None,
) -> None:
    if not all_ts:
        return
    fig, axes = plt.subplots(len(all_ts), 1, figsize=(16, max(4, 3.2 * len(all_ts))), sharex=False)
    if len(all_ts) == 1:
        axes = [axes]
    for ax, ts in zip(axes, all_ts):
        t = ts["time_h"].to_numpy(dtype=float)
        voyage_id = int(ts["voyage_id"].iloc[0])
        ax.plot(t, ts["load_total_kw"], color="black", lw=0.75, label="Load")
        ax.plot(t, ts["P_fc_kw"], color="tab:red", lw=0.75, label="P_fc")
        ax.plot(t, ts["P_batt_kw"], color="tab:blue", lw=0.65, label="P_batt")
        ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax.set_ylabel("Power (kW)")
        ax.set_title(f"Voyage {voyage_id}", fontsize=9)
        ax.grid(alpha=0.15)
        ax_soc = ax.twinx()
        ax_soc.plot(t, ts["SOC"], color="tab:green", lw=0.75, alpha=0.85, label="SOC")
        if "soc_ref_value" in ts.columns:
            ax_soc.plot(t, ts["soc_ref_value"], color="gray", ls="--", lw=0.7, alpha=0.8, label="SOC_ref")
        if show_soc_reserve and "soc_reserve" in ts.columns:
            ax_soc.plot(t, ts["soc_reserve"], color="tab:orange", ls="-.", lw=0.7, alpha=0.8, label="SOC_reserve")
        ax_soc.set_ylabel("SOC")
        ax_soc.set_ylim(0.2, 0.8)
        lines, labels = ax.get_legend_handles_labels()
        soc_lines, soc_labels = ax_soc.get_legend_handles_labels()
        ax.legend(lines + soc_lines, labels + soc_labels, loc="upper right", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Time (hours)")
    if plot_metadata:
        title = (
            f"run_id={plot_metadata.get('run_id', '')} | weight_set={plot_metadata.get('weight_set', '')} | "
            f"q_soc={plot_metadata.get('q_soc', '')} q_batt={plot_metadata.get('q_batt', '')} "
            f"q_ramp={plot_metadata.get('q_ramp', '')}"
        )
        fig.suptitle(title, fontsize=9)
        fig.text(
            0.995,
            0.004,
            f"plot_input={plot_metadata.get('plot_input_source', '')}",
            ha="right",
            va="bottom",
            fontsize=6,
            color="dimgray",
        )
        fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.965])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_initial_dispatch_debug(ts: pd.DataFrame, window_min: float = 90.0) -> pd.DataFrame:
    if ts.empty:
        return pd.DataFrame(columns=DEBUG_INITIAL_DISPATCH_FIELDS)
    work = ts.copy()
    t0 = float(pd.to_numeric(work["time_h"], errors="coerce").min())
    mask = pd.to_numeric(work["time_h"], errors="coerce") <= t0 + float(window_min) / 60.0 + 1e-12
    work = work.loc[mask].copy()
    if work.empty:
        return pd.DataFrame(columns=DEBUG_INITIAL_DISPATCH_FIELDS)
    stage_cols = [f"mpc_ref_load_stage{k}" for k in range(MPC_HORIZON) if f"mpc_ref_load_stage{k}" in work.columns]
    debug = pd.DataFrame(index=work.index)
    debug["voyage_id"] = work["voyage_id"].astype(int)
    debug["t_index"] = work["decision_index_t"].astype(int)
    debug["time_h"] = pd.to_numeric(work["time_h"], errors="coerce")
    debug["load_kw"] = pd.to_numeric(work["load_total_kw"], errors="coerce")
    for h in range(1, 6):
        debug[f"pred_h{h}"] = pd.to_numeric(work.get(f"pred_h{h}", np.nan), errors="coerce")
    debug["mpc_load_ref_0"] = pd.to_numeric(work.get("mpc_ref_load_stage0", np.nan), errors="coerce")
    debug["mpc_load_ref_max"] = (
        work[stage_cols].apply(pd.to_numeric, errors="coerce").max(axis=1) if stage_cols else np.nan
    )
    debug["soc"] = pd.to_numeric(work["SOC_before"], errors="coerce")
    debug["soc_ref"] = pd.to_numeric(work["soc_ref_value"], errors="coerce")
    debug["soc_reserve"] = pd.to_numeric(work["soc_reserve"], errors="coerce")
    debug["soc_error"] = debug["soc"] - debug["soc_ref"]
    debug["soc_band"] = pd.to_numeric(work["soc_band"], errors="coerce")
    debug["terminal_soc_band"] = pd.to_numeric(work["terminal_soc_band"], errors="coerce")
    debug["P_fc_cmd"] = pd.to_numeric(work["P_fc_kw"], errors="coerce")
    debug["P_batt_cmd"] = pd.to_numeric(work["P_batt_kw"], errors="coerce")
    debug["P_fc_prev"] = pd.to_numeric(work["P_fc_prev_kw"], errors="coerce")
    for col in [
        "P_fc_upper_bound",
        "P_fc_lower_bound",
        "battery_discharge_upper_bound",
        "battery_charge_upper_bound",
        "weighted_h2_cost",
        "weighted_soc_cost",
        "weighted_terminal_soc_cost",
        "weighted_batt_cost",
        "weighted_ramp_cost",
        "total_objective",
    ]:
        debug[col] = pd.to_numeric(work.get(col, np.nan), errors="coerce")
    for col in [
        "mpc_solved",
        "fallback_control_used",
        "history_available",
        "lstm_available",
        "battery_discharge_limit_active",
        "battery_charge_limit_active",
    ]:
        debug[col] = work.get(col, False).map(bool01) if col in work.columns else 0
    debug["fallback_reason"] = work.get("fallback_reason", "").fillna("").astype(str)
    return debug.reindex(columns=DEBUG_INITIAL_DISPATCH_FIELDS)


def aggregate_horizon_metrics(all_ts: list[pd.DataFrame]) -> dict[str, Any]:
    combined = pd.concat(all_ts, ignore_index=True)
    return compute_horizon_metrics(combined, label="all")


def json_safe_config(config: CasadiMPCConfig) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in vars(config).items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            safe[key] = value
    return safe


def build_solver_horizon_debug(config: CasadiMPCConfig) -> dict[str, Any]:
    n = int(config.prediction_horizon)
    raw_physical = str(config.objective_mode).strip().lower() == "raw_physical"
    return {
        "solver": "CasADi nlpsol IPOPT",
        "solver_class": "ShipCasadiMPC via CasadiReferenceGenerator",
        "N": n,
        "mpc_horizon": n,
        "decision_variable_length": n,
        "p_fc_decision_length": n,
        "p_batt_decision_length": n,
        "soc_state_length": n + 1,
        "soc_prediction_steps": n,
        "p_batt_abs_decision_length": n if raw_physical else 0,
        "total_nlp_variable_length": (n + 1) + n + n + (n if raw_physical else 0),
        "objective_loop_range": f"0..{n - 1}",
        "uses_stages": list(range(n)),
        "uses_stage_0_to_5": n == 6,
        "returns_only_first_control": True,
        "returned_control_fields": ["fuel_cell_ref_kw", "battery_ref_kw", "predicted_soc"],
        "forecast_parameter_length": n,
        "control_timing": CONTROL_TIMING,
        "control_apply_timing": CONTROL_APPLY_TIMING,
        "fuel_cell_ramp_constraint_enabled": bool(config.fuel_cell_ramp_constraint_enabled),
        "enable_fc_ramp_hard_constraint": bool(config.fuel_cell_ramp_constraint_enabled),
        "fuel_cell_ramp_kw": float(config.fuel_cell_ramp_kw),
        "fc_ramp_limit_kw_per_step": float(config.fuel_cell_ramp_kw),
        "objective_mode": str(config.objective_mode),
        "use_h2_mass_cost": bool(config.use_h2_mass_cost),
    }


def build_timing_debug_first_steps(ts: pd.DataFrame, rows_per_source: int = 30) -> pd.DataFrame:
    columns = [
        "voyage_id",
        "global_step",
        "local_step",
        "timestamp",
        "history_available",
        "lstm_available",
        "forecast_source",
        "actual_load_t",
        *[f"lstm_pred_h{h}" for h in range(1, MPC_HORIZON + 1)],
        *[f"mpc_ref_load_stage{k}" for k in range(MPC_HORIZON)],
        "P_fc_executed_t",
        "P_batt_actual_t",
        *[f"P_fc_plan_stage{k}" for k in range(MPC_HORIZON)],
        "P_fc_next_cmd",
        "SOC_before",
        "SOC_after",
        "all_mpc_ref_match_lstm_h1_h6",
        "all_mpc_ref_equal_current_load",
        "stage0_equals_lstm_h1",
        "stage0_equals_actual_load_t",
    ]
    if ts.empty:
        return pd.DataFrame(columns=columns)

    work = ts.copy().reset_index(drop=True)
    work["global_step"] = np.arange(len(work), dtype=int)
    selected_parts: list[pd.DataFrame] = []
    for _, group in work.groupby("voyage_id", sort=True):
        hold = group[group["forecast_source"].astype(str) == "current_load_hold"].head(rows_per_source)
        lstm = group[group["forecast_source"].astype(str) == "lstm_h1_to_h6"].head(rows_per_source)
        selected_parts.extend([hold, lstm])
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else work.head(0)
    if selected.empty:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(index=selected.index)
    out["voyage_id"] = selected["voyage_id"].astype(int)
    out["global_step"] = selected["global_step"].astype(int)
    out["local_step"] = selected["decision_index_t"].astype(int)
    out["timestamp"] = selected.get("timestamp", "").astype(str)
    out["history_available"] = selected.get("history_available", False).map(bool01)
    out["lstm_available"] = selected.get("lstm_available", False).map(bool01)
    out["forecast_source"] = selected.get("forecast_source", "").astype(str)
    out["actual_load_t"] = pd.to_numeric(selected["load_total_kw"], errors="coerce")
    for h in range(1, MPC_HORIZON + 1):
        out[f"lstm_pred_h{h}"] = pd.to_numeric(selected.get(f"pred_h{h}", np.nan), errors="coerce")
    for k in range(MPC_HORIZON):
        out[f"mpc_ref_load_stage{k}"] = pd.to_numeric(
            selected.get(f"mpc_ref_load_stage{k}", np.nan), errors="coerce"
        )
        out[f"P_fc_plan_stage{k}"] = pd.to_numeric(selected.get(f"P_fc_plan_stage{k}", np.nan), errors="coerce")
    out["P_fc_executed_t"] = pd.to_numeric(selected.get("P_fc_executed_kw", selected.get("P_fc_kw", np.nan)), errors="coerce")
    out["P_batt_actual_t"] = pd.to_numeric(
        selected.get("P_batt_actual_kw", selected.get("P_batt_kw", np.nan)), errors="coerce"
    )
    out["P_fc_next_cmd"] = pd.to_numeric(selected.get("P_fc_next_cmd_kw", np.nan), errors="coerce")
    out["SOC_before"] = pd.to_numeric(selected.get("SOC_before", np.nan), errors="coerce")
    out["SOC_after"] = pd.to_numeric(selected.get("SOC", np.nan), errors="coerce")

    stage_cols = [f"mpc_ref_load_stage{k}" for k in range(MPC_HORIZON)]
    pred_cols = [f"lstm_pred_h{h}" for h in range(1, MPC_HORIZON + 1)]
    stage_values = out[stage_cols].to_numpy(dtype=float)
    pred_values = out[pred_cols].to_numpy(dtype=float)
    actual_values = out["actual_load_t"].to_numpy(dtype=float)[:, None]
    all_match_lstm = np.all(np.isclose(stage_values, pred_values, atol=1e-8, equal_nan=False), axis=1)
    all_equal_current = np.all(np.isclose(stage_values, actual_values, atol=1e-8, equal_nan=False), axis=1)
    stage0_match_lstm = np.isclose(out["mpc_ref_load_stage0"], out["lstm_pred_h1"], atol=1e-8, equal_nan=False)
    stage0_equal_actual = np.isclose(out["mpc_ref_load_stage0"], out["actual_load_t"], atol=1e-8, equal_nan=False)
    out["all_mpc_ref_match_lstm_h1_h6"] = all_match_lstm
    out["all_mpc_ref_equal_current_load"] = all_equal_current
    out["stage0_equals_lstm_h1"] = stage0_match_lstm
    out["stage0_equals_actual_load_t"] = stage0_equal_actual

    lstm_mask = out["forecast_source"].eq("lstm_h1_to_h6")
    hold_mask = out["forecast_source"].eq("current_load_hold")
    if bool(lstm_mask.any()) and not bool(out.loc[lstm_mask, "all_mpc_ref_match_lstm_h1_h6"].all()):
        raise ValueError("Invalid timing debug: LSTM rows must map h1-h6 to MPC stages 0-5.")
    if bool(hold_mask.any()) and not bool(out.loc[hold_mask, "all_mpc_ref_equal_current_load"].all()):
        raise ValueError("Invalid timing debug: current-load hold rows must repeat actual_load_t.")
    return out.reindex(columns=columns).reset_index(drop=True)


def build_effective_config_debug(
    *,
    output_dir: Path,
    weight_set: str,
    weights: dict[str, Any],
    effective_cfg: CasadiMPCConfig,
    run_id: str,
) -> dict[str, Any]:
    weight_jsons = sorted(PROJ.rglob("mpc_weight_sets.json"))
    residual_rule_keys = sorted(str(key) for key in weights if str(key) in FORBIDDEN_RULE_BASED_CONFIG_KEYS)
    return {
        "run_id": run_id,
        "cwd": str(Path.cwd().resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "actual_weight_sets_json_path": str(WEIGHT_SETS_JSON.resolve()),
        "actual_weight_sets_json_md5": file_md5(WEIGHT_SETS_JSON),
        "mpc_weight_sets_json_candidates": [str(path.resolve()) for path in weight_jsons],
        "multiple_mpc_weight_sets_json_found": len(weight_jsons) > 1,
        "active_weight_set": str(weight_set),
        "P6_WEIGHT_SOURCE": str(weights.get("P6_WEIGHT_SOURCE", "")),
        "run_layer_weights": {
            "q_h2": float(weights.get("q_h2", 0.0)),
            "q_soc": float(weights.get("q_soc", 0.0)),
            "q_fc": float(weights.get("q_fc", 0.0)),
            "q_batt": float(weights.get("q_batt", 0.0)),
            "q_ramp": float(weights.get("q_ramp", 0.0)),
            "q_terminal_soc": float(weights.get("q_terminal_soc", 0.0)),
        },
        "solver_effective_weights": effective_mpc_diagnostics(effective_cfg),
        "battery_capacity_kwh": float(effective_cfg.battery_capacity_kwh),
        "fuel_cell_ramp_constraint_enabled": bool(effective_cfg.fuel_cell_ramp_constraint_enabled),
        "enable_fc_ramp_hard_constraint": bool(effective_cfg.fuel_cell_ramp_constraint_enabled),
        "fuel_cell_ramp_kw": float(effective_cfg.fuel_cell_ramp_kw),
        "fc_ramp_limit_kw_per_step": float(effective_cfg.fuel_cell_ramp_kw),
        "objective_mode": str(effective_cfg.objective_mode),
        "use_dimensionless_objective": bool(effective_cfg.use_dimensionless_objective),
        "normalize_h2_cost": bool(effective_cfg.normalize_h2_cost),
        "soc_band": float(effective_cfg.soc_band),
        "terminal_soc_band": float(effective_cfg.terminal_soc_band),
        "enable_terminal_soc_soft_penalty": bool(effective_cfg.enable_terminal_soc_soft_penalty),
        "soc_reference_mode": str(effective_cfg.soc_reference_mode),
        "soc_min": float(effective_cfg.soc_min),
        "soc_max": float(effective_cfg.soc_max),
        "residual_rule_based_config_keys": residual_rule_keys,
        "residual_soc_reserve_key_in_active_weights": "soc_reserve" in weights,
        "residual_q_reserve_key_in_active_weights": "q_reserve" in weights,
        "residual_terminal_penalty_enabled": bool(effective_cfg.enable_terminal_soc_soft_penalty),
    }


def build_horizon_sensitivity_debug(
    weights: dict[str, Any],
    *,
    current_soc: float = 0.55,
    prev_fc_kw: float = 100.0,
) -> pd.DataFrame:
    cases = {
        "A_flat_100": np.full(MPC_HORIZON, 100.0, dtype=float),
        "B_last_step_500": np.asarray([100.0, 100.0, 100.0, 100.0, 100.0, 500.0], dtype=float),
        "C_h2_to_h6_500": np.asarray([100.0, 500.0, 500.0, 500.0, 500.0, 500.0], dtype=float),
    }
    rows: list[dict[str, Any]] = []
    for case_id, forecast in cases.items():
        cfg = mpc_config_from_weights(weights)
        row: dict[str, Any] = {
            "case_id": case_id,
            "current_soc": float(current_soc),
            "prev_fc_kw": float(prev_fc_kw),
            "success": False,
        }
        for k, value in enumerate(forecast):
            row[f"load_stage{k}"] = float(value)
        try:
            generator = REFERENCE_GENERATOR_CLASS(cfg)
            result = generator.generate_result(
                load_forecast_kw=forecast,
                current_soc=float(current_soc),
                prev_fc_kw=float(prev_fc_kw),
                soc_reference_value=float(current_soc),
            )
            info = dict(result.objective_info)
            row.update(
                {
                    "success": bool(result.success),
                    "P_fc_next_cmd": float(result.fuel_cell_ref_kw),
                    "P_batt_stage0": float(result.battery_ref_kw),
                    "objective_total": float(result.objective_value),
                    "weighted_h2_cost": float(info.get("weighted_h2_cost", np.nan)),
                    "weighted_soc_cost": float(info.get("weighted_soc_cost", np.nan)),
                    "weighted_batt_cost": float(info.get("weighted_batt_cost", np.nan)),
                    "weighted_ramp_cost": float(info.get("weighted_ramp_cost", np.nan)),
                    "raw_h2_cost": float(info.get("raw_h2_cost", np.nan)),
                    "raw_soc_cost": float(info.get("raw_soc_cost", np.nan)),
                    "batt_throughput_kwh": float(info.get("batt_throughput_kwh", np.nan)),
                    "ramp_cost_raw": float(info.get("ramp_cost_raw", np.nan)),
                    "solver_message": str(info.get("solver_message", "")),
                }
            )
            for k in range(MPC_HORIZON):
                row[f"P_fc_plan_stage{k}"] = float(result.fuel_cell_ref_traj_kw[k])
                row[f"P_batt_plan_stage{k}"] = float(result.battery_ref_traj_kw[k])
            for k in range(MPC_HORIZON + 1):
                row[f"SOC_plan_stage{k}"] = float(result.soc_pred_traj[k])
        except Exception as exc:  # pragma: no cover - depends on local solver availability
            row["solver_message"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def write_baseline_summary(
    output_dir: Path,
    *,
    weight_set: str,
    weights: dict[str, Any],
    voyage_names: list[str],
    metrics_df: pd.DataFrame,
    horizon_df: pd.DataFrame,
    objective_df: pd.DataFrame,
    config_payload: dict[str, Any],
) -> None:
    metrics_cols = [
        "voyage_id",
        "duration_h",
        "SOC_start",
        "SOC_end",
        "SOC_min",
        "SOC_max",
        "SOC_delta",
        "H2_total_kg",
        "charge_sustaining_adjusted_H2",
        "battery_throughput_kwh",
        "fc_above_load_energy_kwh",
        "fc_below_load_energy_kwh",
        "fc_load_tracking_mae",
        "fc_ramp_mean_kw",
        "fc_ramp_max_kw",
        "solver_success_rate",
    ]
    horizon_cols = [
        "voyage_id",
        "RMSE_h1",
        "MAE_h1",
        "WAPE_h1",
        "RMSE_h6",
        "MAE_h6",
        "WAPE_h6",
    ]
    objective_cols = [
        "voyage_id",
        "h2_mass_kg_sum",
        "batt_throughput_kwh_sum",
        "weighted_h2_cost_sum",
        "weighted_soc_cost_sum",
        "weighted_batt_cost_sum",
        "weighted_ramp_cost_sum",
        "total_objective_sum",
    ]
    lines = [
        "# Clean Fixed LSTM-H2-MPC Baseline v1",
        "",
        "## Configuration",
        "",
        f"- Weight set: `{weight_set}`",
        f"- Weights: `{json.dumps({k: weights.get(k) for k in ['q_h2', 'q_soc', 'q_fc', 'q_batt', 'q_ramp', 'q_terminal_soc']}, ensure_ascii=False)}`",
        f"- Load definition: `{config_payload.get('load_definition')}`",
        f"- LSTM checkpoint: `{config_payload.get('lstm_checkpoint')}`",
        f"- Split file: `{config_payload.get('split_json')}`",
        f"- Battery capacity: `{config_payload.get('battery_capacity_kwh')} kWh`",
        f"- Fuel-cell hard ramp constraint enabled: `{config_payload.get('fuel_cell_ramp_constraint_enabled')}`",
        f"- SOC limits: `{config_payload.get('effective_mpc_config', {}).get('soc_min')}` to `{config_payload.get('effective_mpc_config', {}).get('soc_max')}`",
        f"- SOC reference mode: `{config_payload.get('effective_mpc_config', {}).get('soc_reference_mode')}`",
        "",
        "## Timing",
        "",
        f"- Control timing: `{CONTROL_TIMING}`",
        f"- Control apply timing: `{CONTROL_APPLY_TIMING}`",
        "- When LSTM is available, MPC load reference is exactly `[pred_h1, pred_h2, pred_h3, pred_h4, pred_h5, pred_h6]`.",
        "- Before LSTM history is available, MPC uses a six-step current-load hold.",
        "- The measured current load is used for cached-command execution and SOC update, not as MPC stage 0 when LSTM is available.",
        "",
        "## Objective",
        "",
        "```text",
        "J = sum_k [",
        "    q_h2   * mH2_kg[k]",
        "  + q_soc  * (SOC[k+1] - SOC_ref)^2",
        "  + q_batt * E_batt_kwh[k]",
        "  + q_ramp * (P_fc[k] - P_fc[k-1])^2",
        "]",
        "```",
        "",
        "The Dp0 hydrogen term uses the imported fresh fuel-cell curve. No reserve penalty, terminal SOC penalty, SOC deadband, normalized objective, or rule-based load/SOC limits are active in this fixed baseline.",
        "",
        "## Test Voyages",
        "",
    ]
    lines.extend(f"- `{name}`" for name in voyage_names)
    lines += [
        "",
        "## Forecast Metrics",
        "",
        "```text",
        horizon_df[[col for col in horizon_cols if col in horizon_df.columns]].to_string(index=False)
        if not horizon_df.empty
        else "No forecast metrics.",
        "```",
        "",
        "## Closed-Loop Metrics",
        "",
        "```text",
        metrics_df[[col for col in metrics_cols if col in metrics_df.columns]].to_string(index=False)
        if not metrics_df.empty
        else "No closed-loop metrics.",
        "```",
        "",
        "## Objective Breakdown",
        "",
        "```text",
        objective_df[[col for col in objective_cols if col in objective_df.columns]].to_string(index=False)
        if not objective_df.empty
        else "No objective breakdown.",
        "```",
        "",
        "## Fixed Baseline Selection Note",
        "",
        "`q_soc=400, q_batt=0.030, q_ramp=0.00002` is retained as the current fixed baseline because it reduced the excessive fuel-cell share observed in higher-SOC-weight runs while preserving charge-sustaining behavior better than lower battery-throughput penalties. This is a fixed-weight baseline, not evidence for dynamic-weight KAN-DQN.",
        "",
        "## Limitations",
        "",
        "Fixed weights cannot adapt to voyage-dependent operating phases. Remaining fuel-cell over/under-production and SOC drift must be handled by objective-weight design or the future dynamic weighting layer, not by if-load or if-SOC rules.",
        "",
        "## Future Dynamic Weighting",
        "",
        "The future SineKAN-DQN stage should adjust MPC objective weights online while keeping this same physical MPC interface: LSTM h1-h6 forecast in, total P_fc/P_batt reference out.",
        "",
        "A clean literature-consistent LSTM-H2-MPC baseline has been restored, using only physical constraints and objective-function-based optimization terms.",
        "",
    ]
    output_dir.joinpath("baseline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    output_dir: Path,
    *,
    weight_set: str,
    weights: dict[str, Any],
    voyage_names: list[str],
    summaries: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    recommended_weight_set: str | None = None,
) -> None:
    metrics_df = pd.DataFrame(summaries)
    horizon_df = pd.DataFrame(horizon_rows)
    lines = [
        "# LSTM-MPC Fixed-Weight Baseline",
        "",
        f"- LSTM checkpoint: `{LSTM_CKPT}`",
        f"- Split file: `{SPLIT_JSON}`",
        f"- Weight set: `{weight_set}`",
        f"- Weights: `{json.dumps(weights, ensure_ascii=False)}`",
        f"- Control timing: `{CONTROL_TIMING}`",
        f"- Control apply timing: `{CONTROL_APPLY_TIMING}`",
        f"- Control layer scope: `{CONTROL_LAYER_SCOPE}`",
        f"- Forecast mode: `{FORECAST_MODE}`",
        "- MPC horizon: `6`",
        "- MPC load reference: `[pred_h1, pred_h2, pred_h3, pred_h4, pred_h5, pred_h6]` when LSTM history is available; otherwise six-step current-load hold.",
        "- Closed-loop execution: cached previous `P_fc` command is applied first; actual `P_batt` is computed from current measured load.",
        "- MPC output: total `P_fc_next_cmd_kw`; closed-loop actual fields are total `P_fc_kw`/`P_batt_kw`; no left/right device-level energy management in this entrypoint.",
        "",
        "## Test Voyages",
        "",
    ]
    for i, name in enumerate(voyage_names, start=1):
        lines.append(f"{i}. `{name}`")
    lines += [
        "",
        "## Forecast Metrics",
        "",
        "```text",
        horizon_df.to_string(index=False) if not horizon_df.empty else "No forecast metrics.",
        "```",
        "",
        "## Closed-Loop Metrics",
        "",
        "```text",
        metrics_df.to_string(index=False) if not metrics_df.empty else "No closed-loop metrics.",
        "```",
        "",
        "## Weight Sweep",
        "",
        f"- Recommended fixed MPC baseline: `{recommended_weight_set or weight_set}`",
        "",
        "## Current Conclusion",
        "",
        "LSTM-MPC fixed-weight baseline has been established on four unseen real voyage segments using one-step-ahead receding-horizon MPC.",
        "",
        "Do not claim KAN-DQN effectiveness until DQN training and test comparisons are complete.",
        "",
    ]
    output_dir.joinpath("README_LSTM_MPC_TEST.md").write_text("\n".join(lines), encoding="utf-8")


def run_all(
    *,
    weight_set: str = CURRENT_FIXED_WEIGHT_SET,
    output_dir: Path = OUT_DIR,
    voyage: str | None = None,
    init_soc: float = 0.55,
    make_plots: bool = True,
    write_outputs: bool = True,
    max_steps: int | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weight_sets = load_weight_sets()
    if weight_set not in weight_sets:
        raise KeyError(f"Unknown weight set: {weight_set}")
    weights = weight_sets[weight_set]
    run_id = make_run_id(weight_set, weights)
    effective_cfg = mpc_config_from_weights(weights)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, payload = load_checkpoint(LSTM_CKPT, device=device)
    if int(payload["config"]["pred_horizon"]) != MPC_HORIZON:
        raise ValueError(f"Expected LSTM pred_horizon={MPC_HORIZON}.")
    split = load_json(SPLIT_JSON)
    voyage_names = [voyage] if voyage else list(split["test_voyages"])
    df_all = pd.read_csv(SOURCE_CSV)

    summaries: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    all_debug: list[pd.DataFrame] = []
    all_ts: list[pd.DataFrame] = []
    for idx, voyage_name in enumerate(voyage_names, start=1):
        df_voyage = df_all[df_all["voyage_name"] == voyage_name].reset_index(drop=True)
        if df_voyage.empty:
            raise ValueError(f"Voyage not found in source CSV: {voyage_name}")
        summary, horizon_metrics, ts, debug = run_single_voyage(
            voyage_id=idx,
            voyage_name=voyage_name,
            df_voyage=df_voyage,
            model=model,
            payload=payload,
            device=device,
            weight_set=weight_set,
            weights=weights,
            output_dir=output_dir,
            init_soc=init_soc,
            make_plots=make_plots,
            write_outputs=write_outputs,
            max_steps=max_steps,
            run_id=run_id,
        )
        summaries.append(summary)
        horizon_rows.append(horizon_metrics)
        all_ts.append(ts)
        all_debug.append(debug)

    if all_ts:
        horizon_rows.append(aggregate_horizon_metrics(all_ts))
    metrics_df = add_run_id_column(pd.DataFrame(summaries), run_id)
    horizon_df = add_run_id_column(pd.DataFrame(horizon_rows), run_id)
    objective_df = pd.DataFrame([compute_objective_breakdown(ts) for ts in all_ts]) if all_ts else pd.DataFrame()
    debug_df = pd.concat(all_debug, ignore_index=True) if all_debug else pd.DataFrame()
    initial_debug_df = (
        pd.concat([build_initial_dispatch_debug(ts) for ts in all_ts], ignore_index=True) if all_ts else pd.DataFrame()
    )
    no_reserve_no_terminal = uses_no_reserve_no_terminal_reporting(weight_set)
    objective_df = add_run_id_column(objective_df, run_id)
    debug_df = add_run_id_column(debug_df, run_id)
    initial_debug_df = add_run_id_column(initial_debug_df, run_id)
    metrics_df = clean_reporting_frame_for_weight_set(metrics_df, weight_set=weight_set)
    objective_df = clean_reporting_frame_for_weight_set(objective_df, weight_set=weight_set)
    debug_df = clean_reporting_frame_for_weight_set(debug_df, weight_set=weight_set)
    initial_debug_df = clean_reporting_frame_for_weight_set(initial_debug_df, weight_set=weight_set)
    all_ts_for_output = [
        clean_reporting_frame_for_weight_set(add_run_id_column(ts, run_id), weight_set=weight_set) for ts in all_ts
    ]

    if not objective_df.empty:
        ratio_cols = [
            "weighted_h2_cost_ratio",
            "weighted_soc_cost_ratio",
            "weighted_batt_cost_ratio",
            "weighted_ramp_cost_ratio",
        ]
        print_cols = ["voyage_id", *[col for col in ratio_cols if col in objective_df.columns]]
        print("Objective contribution ratios:")
        print(objective_df[print_cols].to_string(index=False))

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_input_path = output_dir / "fixed_mpc_p6_timeseries.csv"
        plot_output_path = output_dir / "fixed_mpc_p6_comparison.png"
        config_payload = {
            "run_id": run_id,
            "lstm_checkpoint": str(LSTM_CKPT),
            "split_json": str(SPLIT_JSON),
            "source_csv": str(SOURCE_CSV),
            "weight_sets_json": str(WEIGHT_SETS_JSON.resolve()),
            "weight_sets_json_md5": file_md5(WEIGHT_SETS_JSON),
            "available_weight_sets": sorted(weight_sets.keys()),
            "weight_set": weight_set,
            "weights": weights,
            "P6_WEIGHT_SOURCE": str(weights.get("P6_WEIGHT_SOURCE", "")),
            "mpc_base_kwargs_template": MPC_BASE_KWARGS,
            "effective_mpc_config": json_safe_config(effective_cfg),
            "solver_effective_weights": effective_mpc_diagnostics(effective_cfg),
            "plot_input_source": "in_memory_all_ts_for_output_from_current_run",
            "plot_input_timeseries_path": str(plot_input_path.resolve()),
            "plot_output_png_path": str(plot_output_path.resolve()),
            "control_timing": CONTROL_TIMING,
            "control_apply_timing": CONTROL_APPLY_TIMING,
            "mpc_load_reference_when_lstm_available": [
                "pred_h1",
                "pred_h2",
                "pred_h3",
                "pred_h4",
                "pred_h5",
                "pred_h6",
            ],
            "mpc_load_reference_when_lstm_unavailable": "current_load_hold_repeated_6_steps",
            "actual_load_t_usage": "execution_feedback_and_SOC_update_only_when_lstm_available",
            "lstm_prediction_horizon_used": 6,
            "mpc_stage_mapping": {f"stage{k}": f"pred_h{k + 1}" for k in range(MPC_HORIZON)},
            "control_layer_scope": CONTROL_LAYER_SCOPE,
            "forecast_mode": FORECAST_MODE,
            "load_definition": LOAD_DEFINITION,
            "capacity_basis": CAPACITY_BASIS,
            "initial_fc_mode": INITIAL_FC_MODE,
            "battery_capacity_kwh": float(effective_cfg.battery_capacity_kwh),
            "fuel_cell_ramp_constraint_enabled": bool(effective_cfg.fuel_cell_ramp_constraint_enabled),
            "enable_fc_ramp_hard_constraint": bool(effective_cfg.fuel_cell_ramp_constraint_enabled),
            "fuel_cell_ramp_kw": float(effective_cfg.fuel_cell_ramp_kw),
            "fc_ramp_limit_kw_per_step": float(effective_cfg.fuel_cell_ramp_kw),
        }
        config_payload = clean_config_payload_for_weight_set(config_payload, weight_set=weight_set)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        horizon_df.to_csv(output_dir / "forecast_horizon_metrics.csv", index=False, encoding="utf-8-sig")
        objective_df.to_csv(output_dir / "objective_breakdown.csv", index=False, encoding="utf-8-sig")
        objective_df.to_csv(output_dir / "objective_decomposition.csv", index=False, encoding="utf-8-sig")
        debug_df.to_csv(output_dir / "debug_timing.csv", index=False, encoding="utf-8-sig")
        initial_debug_df.to_csv(output_dir / "debug_initial_dispatch.csv", index=False, encoding="utf-8-sig")
        metrics_df.to_csv(output_dir / "fixed_mpc_p6_metrics.csv", index=False, encoding="utf-8-sig")
        objective_df.to_csv(output_dir / "fixed_mpc_p6_objective_breakdown.csv", index=False, encoding="utf-8-sig")
        objective_df.to_csv(output_dir / "fixed_mpc_p6_objective_decomposition.csv", index=False, encoding="utf-8-sig")
        if all_ts_for_output:
            combined_ts = pd.concat(all_ts_for_output, ignore_index=True)
            combined_ts.to_csv(plot_input_path, index=False, encoding="utf-8-sig")
        else:
            combined_ts = pd.DataFrame()
        timing_debug_first_steps = build_timing_debug_first_steps(combined_ts)
        timing_debug_first_steps.to_csv(output_dir / "timing_debug_first_steps.csv", index=False, encoding="utf-8-sig")
        solver_horizon_debug = build_solver_horizon_debug(effective_cfg)
        write_json(output_dir / "solver_horizon_debug.json", solver_horizon_debug)
        effective_config_debug = build_effective_config_debug(
            output_dir=output_dir,
            weight_set=weight_set,
            weights=weights,
            effective_cfg=effective_cfg,
            run_id=run_id,
        )
        write_json(output_dir / "effective_config_debug.json", effective_config_debug)
        horizon_sensitivity_debug = build_horizon_sensitivity_debug(weights)
        horizon_sensitivity_debug.to_csv(output_dir / "horizon_sensitivity_debug.csv", index=False, encoding="utf-8-sig")
        write_json(output_dir / "run_config.json", config_payload)
        write_json(output_dir / "fixed_mpc_p6_config.json", config_payload)
        if make_plots:
            plot_fixed_mpc_comparison(
                all_ts_for_output,
                plot_output_path,
                show_soc_reserve=not no_reserve_no_terminal,
                plot_metadata={
                    "run_id": run_id,
                    "weight_set": weight_set,
                    "q_soc": float(effective_cfg.q_soc),
                    "q_batt": float(effective_cfg.q_batt),
                    "q_ramp": float(effective_cfg.q_ramp),
                    "plot_input_source": "in_memory_all_ts_for_output_from_current_run",
                },
            )
        write_baseline_summary(
            output_dir,
            weight_set=weight_set,
            weights=weights,
            voyage_names=voyage_names,
            metrics_df=metrics_df,
            horizon_df=horizon_df,
            objective_df=objective_df,
            config_payload=config_payload,
        )
        file_hashes = build_run_file_hashes(output_dir=output_dir, run_id=run_id)
        write_json(output_dir / "run_file_hashes.json", file_hashes)
        write_json(output_dir / "fixed_mpc_p6_file_hashes.json", file_hashes)
        write_action_table(base=weights, base_weight_set=weight_set)
        write_readme(
            output_dir,
            weight_set=weight_set,
            weights=weights,
            voyage_names=voyage_names,
            summaries=summaries,
            horizon_rows=horizon_rows,
        )
    return metrics_df, horizon_df, debug_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-step-ahead LSTM-MPC on 7-2-1 test voyages.")
    parser.add_argument("--weight_set", default=CURRENT_FIXED_WEIGHT_SET)
    parser.add_argument("--voyage", default=None)
    parser.add_argument("--soc", type=float, default=0.55)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Optional debug limit; default runs full voyages.")
    parser.add_argument("--no_plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics_df, horizon_df, _ = run_all(
        weight_set=str(args.weight_set),
        output_dir=Path(args.output_dir),
        voyage=args.voyage,
        init_soc=float(args.soc),
        make_plots=not bool(args.no_plots),
        write_outputs=True,
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"Saved: {Path(args.output_dir)}")
    if not metrics_df.empty:
        print(metrics_df[["voyage_id", "duration_h", "soc_end", "soc_min", "fc_energy_kwh", "solver_success_rate"]].to_string(index=False))
    if not horizon_df.empty:
        print(horizon_df[["voyage_id", "MAE_h1", "WAPE_h1", "MAE_h6", "WAPE_h6"]].to_string(index=False))


if __name__ == "__main__":
    main()
