from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mpc.solvers.fc_dp0_curve import CURVE_CSV_PATH, dp0_quadratic_coefficients  # noqa: E402


def ramp_kw_per_step_from_rate(rate_kw_per_s: float, *, dt_seconds: float) -> float:
    """Convert a physical ramp rate into the discrete MPC step limit."""
    rate = float(rate_kw_per_s)
    dt = float(dt_seconds)
    if rate < 0.0:
        raise ValueError("rate_kw_per_s must be nonnegative")
    if dt <= 0.0:
        raise ValueError("dt_seconds must be positive")
    return rate * dt


@dataclass(frozen=True)
class QpMpcConfig:
    horizon: int = 60
    dt_seconds: float = 1.0
    battery_capacity_kwh: float = 693.0
    battery_charge_max_kw: float = 346.5
    battery_discharge_max_kw: float = 346.5
    battery_power_ref_kw: float = 346.5
    fuel_cell_min_kw: float = 0.0
    fuel_cell_max_kw: float = 560.0
    fuel_cell_ramp_rate_kw_per_s: float = 48.0
    fuel_cell_ramp_kw: float | None = None
    soc_min: float = 0.2
    soc_max: float = 0.8
    soc_band: float = 0.05
    objective_variant: str = "simplified_normalized_literature_v1"
    q_h2: float = 1.0
    q_fc_var: float = 0.0
    q_soc: float = 1.0
    q_batt: float = 0.05
    q_ramp: float = 0.0
    q_terminal_soc: float = 0.0


@dataclass(frozen=True)
class QpProblem:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    l: np.ndarray
    u: np.ndarray
    metadata: dict[str, Any]


def resolved_ramp_kw_per_step(config: QpMpcConfig) -> float:
    if config.fuel_cell_ramp_kw is not None:
        ramp = float(config.fuel_cell_ramp_kw)
        if ramp < 0.0:
            raise ValueError("fuel_cell_ramp_kw must be nonnegative")
        return ramp
    return ramp_kw_per_step_from_rate(
        config.fuel_cell_ramp_rate_kw_per_s,
        dt_seconds=config.dt_seconds,
    )


def h2_quadratic_kg_step_coefficients(config: QpMpcConfig) -> tuple[float, float, float, float]:
    """Return ``quad, linear, dp0_a1, dp0_a2`` for kg H2 per MPC step.

    The project Dp0 fit is expressed on the 0..1 relative-load axis:
    ``m_dot_gps = (rated_kw / 100) * (a1 * r + a2 * r^2)``.
    """
    a1, a2 = dp0_quadratic_coefficients()
    rated = max(float(config.fuel_cell_max_kw), 1.0e-12)
    dt = float(config.dt_seconds)
    linear = dt * a1 / 100000.0
    quad = dt * a2 / (100000.0 * rated)
    return float(quad), float(linear), float(a1), float(a2)


def _add_normalized_h2_cost(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    fc0: int,
    horizon: int,
    weight: float,
    quadratic_kg_per_step: float,
    linear_kg_per_step: float,
    reference_kg_per_step: float,
) -> None:
    for k in range(horizon):
        index = fc0 + k
        hessian[index, index] += 2.0 * weight * quadratic_kg_per_step / reference_kg_per_step
        linear[index] += weight * linear_kg_per_step / reference_kg_per_step


def _add_normalized_battery_power_cost(
    hessian: np.ndarray,
    *,
    batt0: int,
    horizon: int,
    weight: float,
    reference_kw: float,
) -> None:
    for k in range(horizon):
        hessian[batt0 + k, batt0 + k] += 2.0 * weight / reference_kw**2


def _add_normalized_soc_tracking_cost(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    soc0: int,
    horizon: int,
    weight: float,
    soc_reference: float,
    soc_band: float,
) -> None:
    for k in range(1, horizon + 1):
        index = soc0 + k
        hessian[index, index] += 2.0 * weight / soc_band**2
        linear[index] += -2.0 * weight * soc_reference / soc_band**2


def _add_normalized_fc_variation_cost(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    fc0: int,
    horizon: int,
    weight: float,
    prev_fc_kw: float,
    reference_kw_per_step: float,
) -> None:
    if reference_kw_per_step <= 0.0:
        raise ValueError("fuel-cell variation reference must be positive")
    scaled_weight = weight / reference_kw_per_step**2
    hessian[fc0, fc0] += 2.0 * scaled_weight
    linear[fc0] += -2.0 * scaled_weight * prev_fc_kw
    for k in range(1, horizon):
        current = fc0 + k
        previous = current - 1
        hessian[current, current] += 2.0 * scaled_weight
        hessian[previous, previous] += 2.0 * scaled_weight
        hessian[current, previous] += -2.0 * scaled_weight
        hessian[previous, current] += -2.0 * scaled_weight


def _validate_config(config: QpMpcConfig) -> None:
    if int(config.horizon) <= 0:
        raise ValueError("horizon must be positive")
    if float(config.dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be positive")
    if float(config.battery_capacity_kwh) <= 0.0:
        raise ValueError("battery_capacity_kwh must be positive")
    if float(config.battery_power_ref_kw) <= 0.0:
        raise ValueError("battery_power_ref_kw must be positive")
    if float(config.fuel_cell_max_kw) <= float(config.fuel_cell_min_kw):
        raise ValueError("fuel_cell_max_kw must be greater than fuel_cell_min_kw")
    if float(config.soc_max) <= float(config.soc_min):
        raise ValueError("soc_max must be greater than soc_min")
    if float(config.soc_band) <= 0.0:
        raise ValueError("soc_band must be positive")
    for name in ("q_h2", "q_fc_var", "q_soc", "q_batt", "q_ramp", "q_terminal_soc"):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if str(config.objective_variant) not in {
        "n6_h2_batt_soc_fcvar_normalized_v1",
        "simplified_normalized_literature_v1",
        "legacy_raw_h2_soc_batt_ramp_terminal",
    }:
        raise ValueError(f"unsupported objective_variant: {config.objective_variant}")


def build_qp_problem(
    config: QpMpcConfig,
    *,
    load_forecast_kw: np.ndarray | list[float],
    current_soc: float,
    prev_fc_kw: float,
    soc_reference: float,
    include_diagnostics: bool = True,
) -> QpProblem:
    """Build a sparse convex QP in OSQP form: min 0.5*x'Px + q'x."""
    _validate_config(config)
    horizon = int(config.horizon)
    load = np.asarray(load_forecast_kw, dtype=float).reshape(-1)
    if len(load) != horizon:
        raise ValueError(f"load_forecast_kw must contain exactly {horizon} points")
    if not np.all(np.isfinite(load)):
        raise ValueError("load_forecast_kw contains non-finite values")

    n_fc = horizon
    n_batt = horizon
    n_soc = horizon + 1
    fc0 = 0
    batt0 = n_fc
    soc0 = n_fc + n_batt
    n_var = n_fc + n_batt + n_soc

    hessian = np.zeros((n_var, n_var), dtype=float)
    linear = np.zeros(n_var, dtype=float)

    h2_quad, h2_linear, dp0_a1, dp0_a2 = h2_quadratic_kg_step_coefficients(config)
    objective_variant = str(config.objective_variant)
    h2_reference = float(h2_quad * config.fuel_cell_max_kw * config.fuel_cell_max_kw + h2_linear * config.fuel_cell_max_kw)
    if h2_reference <= 0.0:
        raise ValueError("h2 reference denominator must be positive")
    objective_terms: list[str]
    if objective_variant == "n6_h2_batt_soc_fcvar_normalized_v1":
        objective_terms = [
            "H2_norm",
            "Batt_power_sq_norm",
            "SOC_tracking_sq_norm",
            "FC_variation_sq_norm",
        ]
        _add_normalized_h2_cost(
            hessian,
            linear,
            fc0=fc0,
            horizon=horizon,
            weight=float(config.q_h2),
            quadratic_kg_per_step=h2_quad,
            linear_kg_per_step=h2_linear,
            reference_kg_per_step=h2_reference,
        )
        _add_normalized_battery_power_cost(
            hessian,
            batt0=batt0,
            horizon=horizon,
            weight=float(config.q_batt),
            reference_kw=float(config.battery_power_ref_kw),
        )
        _add_normalized_soc_tracking_cost(
            hessian,
            linear,
            soc0=soc0,
            horizon=horizon,
            weight=float(config.q_soc),
            soc_reference=float(soc_reference),
            soc_band=float(config.soc_band),
        )
        _add_normalized_fc_variation_cost(
            hessian,
            linear,
            fc0=fc0,
            horizon=horizon,
            weight=float(config.q_fc_var),
            prev_fc_kw=float(prev_fc_kw),
            reference_kw_per_step=float(resolved_ramp_kw_per_step(config)),
        )
    elif objective_variant == "simplified_normalized_literature_v1":
        objective_terms = ["H2_norm", "SOC_norm", "Batt_norm"]
        _add_normalized_h2_cost(
            hessian,
            linear,
            fc0=fc0,
            horizon=horizon,
            weight=float(config.q_h2),
            quadratic_kg_per_step=h2_quad,
            linear_kg_per_step=h2_linear,
            reference_kg_per_step=h2_reference,
        )
        _add_normalized_battery_power_cost(
            hessian,
            batt0=batt0,
            horizon=horizon,
            weight=float(config.q_batt),
            reference_kw=float(config.battery_power_ref_kw),
        )
        _add_normalized_soc_tracking_cost(
            hessian,
            linear,
            soc0=soc0,
            horizon=horizon,
            weight=float(config.q_soc),
            soc_reference=float(soc_reference),
            soc_band=float(config.soc_band),
        )
    else:
        objective_terms = ["h2", "soc", "battery", "ramp", "terminal_soc"]
        for k in range(horizon):
            idx = fc0 + k
            hessian[idx, idx] += 2.0 * float(config.q_h2) * h2_quad
            linear[idx] += float(config.q_h2) * h2_linear

        for k in range(horizon):
            idx = batt0 + k
            hessian[idx, idx] += 2.0 * float(config.q_batt)

        for k in range(1, horizon + 1):
            idx = soc0 + k
            hessian[idx, idx] += 2.0 * float(config.q_soc)
            linear[idx] += -2.0 * float(config.q_soc) * float(soc_reference)

        terminal_weight = float(config.q_terminal_soc)
        if terminal_weight > 0.0:
            idx = soc0 + horizon
            hessian[idx, idx] += 2.0 * terminal_weight
            linear[idx] += -2.0 * terminal_weight * float(soc_reference)

        ramp_weight = float(config.q_ramp)
        if ramp_weight > 0.0:
            idx = fc0
            hessian[idx, idx] += 2.0 * ramp_weight
            linear[idx] += -2.0 * ramp_weight * float(prev_fc_kw)
            for k in range(1, horizon):
                i = fc0 + k
                j = fc0 + k - 1
                hessian[i, i] += 2.0 * ramp_weight
                hessian[j, j] += 2.0 * ramp_weight
                hessian[i, j] += -2.0 * ramp_weight
                hessian[j, i] += -2.0 * ramp_weight

    rows: list[dict[int, float]] = []
    lowers: list[float] = []
    uppers: list[float] = []

    def add_row(coeffs: dict[int, float], lower: float, upper: float) -> None:
        rows.append(coeffs)
        lowers.append(float(lower))
        uppers.append(float(upper))

    for k in range(horizon):
        add_row({fc0 + k: 1.0}, config.fuel_cell_min_kw, config.fuel_cell_max_kw)
    for k in range(horizon):
        add_row({batt0 + k: 1.0}, -config.battery_charge_max_kw, config.battery_discharge_max_kw)
    for k in range(horizon + 1):
        add_row({soc0 + k: 1.0}, config.soc_min, config.soc_max)

    add_row({soc0: 1.0}, current_soc, current_soc)
    dt_hours = float(config.dt_seconds) / 3600.0
    soc_coeff = dt_hours / float(config.battery_capacity_kwh)
    for k in range(horizon):
        add_row(
            {
                soc0 + k + 1: 1.0,
                soc0 + k: -1.0,
                batt0 + k: soc_coeff,
            },
            0.0,
            0.0,
        )

    for k in range(horizon):
        add_row({fc0 + k: 1.0, batt0 + k: 1.0}, load[k], load[k])

    ramp_kw = resolved_ramp_kw_per_step(config)
    add_row({fc0: 1.0}, float(prev_fc_kw) - ramp_kw, float(prev_fc_kw) + ramp_kw)
    for k in range(1, horizon):
        add_row({fc0 + k: 1.0, fc0 + k - 1: -1.0}, -ramp_kw, ramp_kw)

    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []
    for r, coeffs in enumerate(rows):
        for c, value in coeffs.items():
            row_idx.append(r)
            col_idx.append(c)
            data.append(float(value))

    P = sparse.csc_matrix(hessian)
    A = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(len(rows), n_var)).tocsc()
    l = np.asarray(lowers, dtype=float)
    u = np.asarray(uppers, dtype=float)

    if include_diagnostics:
        min_eig = float(np.linalg.eigvalsh((hessian + hessian.T) * 0.5).min())
        convex_qp: bool | None = bool(min_eig >= -1.0e-10)
    else:
        min_eig = float("nan")
        convex_qp = None
    fuel_cell_max_reference = str(float(config.fuel_cell_max_kw)).removesuffix(".0")
    dt_reference = str(float(config.dt_seconds)).removesuffix(".0")
    battery_power_reference = str(float(config.battery_power_ref_kw)).removesuffix(".0")
    soc_band_reference = str(float(config.soc_band)).removesuffix(".0")
    fc_variation_reference = str(float(ramp_kw)).removesuffix(".0")
    metadata = {
        "horizon": horizon,
        "dt_seconds": float(config.dt_seconds),
        "variable_order": "P_fc[0:N], P_batt[0:N], SOC[0:N+1]",
        "n_variables": int(n_var),
        "n_constraints": int(A.shape[0]),
        "fuel_cell_ramp_rate_kw_per_s": float(config.fuel_cell_ramp_rate_kw_per_s),
        "fuel_cell_ramp_kw_per_step": float(ramp_kw),
        "fuel_cell_ramp_kw_explicit_override": config.fuel_cell_ramp_kw is not None,
        "fuel_cell_ramp_source": (
            "explicit fuel_cell_ramp_kw per discrete step"
            if config.fuel_cell_ramp_kw is not None
            else "fuel_cell_ramp_rate_kw_per_s multiplied by dt_seconds"
        ),
        "objective_variant": objective_variant,
        "objective_terms": objective_terms,
        "objective_uses_term_normalization": objective_variant
        in {
            "n6_h2_batt_soc_fcvar_normalized_v1",
            "simplified_normalized_literature_v1",
        },
        "soc_cost_in_objective": bool(
            {"SOC_tracking_sq_norm", "SOC_norm", "soc"}.intersection(objective_terms)
        ),
        "battery_power_ref_kw": float(config.battery_power_ref_kw),
        "fuel_cell_variation_ref_kw_per_step": float(
            ramp_kw
            if objective_variant == "n6_h2_batt_soc_fcvar_normalized_v1"
            else config.fuel_cell_ramp_rate_kw_per_s * config.dt_seconds
        ),
        "soc_band": float(config.soc_band),
        "h2_reference_kg_per_step": h2_reference,
        "h2_curve_csv": str(CURVE_CSV_PATH),
        "dp0_forced_origin_a1": float(dp0_a1),
        "dp0_forced_origin_a2": float(dp0_a2),
        "h2_kg_step_quad_coeff": float(h2_quad),
        "h2_kg_step_linear_coeff": float(h2_linear),
        "hessian_min_eigenvalue": min_eig,
        "convex_qp": convex_qp,
        "diagnostics_computed": bool(include_diagnostics),
        "battery_cost_form": (
            "normalized (P_batt / P_batt_ref)^2"
            if objective_variant
            in {
                "n6_h2_batt_soc_fcvar_normalized_v1",
                "simplified_normalized_literature_v1",
            }
            else "legacy raw quadratic P_batt^2"
        ),
        "fuel_cell_variation_cost_form": (
            f"((P_fc[0] - P_fc_prev) / {fc_variation_reference})^2 and "
            f"((P_fc[k] - P_fc[k-1]) / {fc_variation_reference})^2"
            if objective_variant == "n6_h2_batt_soc_fcvar_normalized_v1"
            else "not active in this objective variant"
        ),
    }
    if objective_variant == "n6_h2_batt_soc_fcvar_normalized_v1":
        metadata.update(
            {
                "q_h2": float(config.q_h2),
                "q_batt": float(config.q_batt),
                "q_soc": float(config.q_soc),
                "q_fc_var": float(config.q_fc_var),
                "soc_reference": float(soc_reference),
                "objective_term_descriptions": {
                    "H2_norm": (
                        "sum(k=0..N-1) m_H2(P_fc[k]) / "
                        f"m_H2({fuel_cell_max_reference} kW, {dt_reference} s)"
                    ),
                    "Batt_power_sq_norm": (
                        "sum(k=0..N-1) "
                        f"(P_batt[k] / {battery_power_reference} kW)^2"
                    ),
                    "SOC_tracking_sq_norm": (
                        "sum(k=1..N) "
                        f"((SOC[k] - SOC_ref) / {soc_band_reference})^2"
                    ),
                    "FC_variation_sq_norm": (
                        f"((P_fc[0] - P_fc_prev) / {fc_variation_reference} kW)^2 + "
                        "sum(k=1..N-1) "
                        f"((P_fc[k] - P_fc[k-1]) / {fc_variation_reference} kW)^2"
                    ),
                },
                "terminal_soc_cost_in_objective": False,
                "slack_cost_in_objective": False,
                "extra_ramp_cost_in_objective": False,
                "ignored_objective_weight_fields": ["q_ramp", "q_terminal_soc"],
            }
        )
    return QpProblem(P=P, q=linear, A=A, l=l, u=u, metadata=metadata)


def hessian_min_eigenvalue(problem: QpProblem) -> float:
    dense = problem.P.toarray()
    return float(np.linalg.eigvalsh((dense + dense.T) * 0.5).min())


def write_qp_formulation_check(problem: QpProblem, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    meta = problem.metadata
    lines = [
        "# 1 s QP Formulation Check",
        "",
        "Scope: parallel benchmark formulation only. This file does not modify the current 30 s CasADi/IPOPT mainline.",
        "",
        "## Dimensions",
        "",
        f"- Horizon: `{meta['horizon']}`",
        f"- Sample time: `{meta['dt_seconds']} s`",
        f"- Variables: `{meta['n_variables']}`",
        f"- Constraints: `{meta['n_constraints']}`",
        f"- Variable order: `{meta['variable_order']}`",
        "",
        "## Ramp Unit",
        "",
        f"- Source ramp rate: `{meta['fuel_cell_ramp_rate_kw_per_s']} kW/s`",
        f"- Solver ramp bound: `{meta['fuel_cell_ramp_kw_per_step']} kW/step`",
        f"- Conversion rule: `{meta['fuel_cell_ramp_source']}`",
        "",
        "## Hydrogen Quadratic",
        "",
        f"- Dp0 CSV: `{meta['h2_curve_csv']}`",
        f"- Forced-origin fit a1: `{meta['dp0_forced_origin_a1']}`",
        f"- Forced-origin fit a2: `{meta['dp0_forced_origin_a2']}`",
        f"- kg/step quadratic coefficient: `{meta['h2_kg_step_quad_coeff']}`",
        f"- kg/step linear coefficient: `{meta['h2_kg_step_linear_coeff']}`",
        "",
        "## Convexity",
        "",
        f"- Hessian minimum eigenvalue: `{meta['hessian_min_eigenvalue']}`",
        f"- Convex QP flag: `{meta['convex_qp']}`",
        "",
        "## JSON Metadata",
        "",
        "```json",
        json.dumps(meta, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
