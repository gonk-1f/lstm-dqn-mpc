"""Controller-independent existence audit for the frozen 1 s physical model.

The first input sample initializes previous FC power; only ``loads_kw[1:]``
consumes energy, matching ``DqnMpcWeightEnv.reset/step``. Positive battery
power discharges it. Dynamics are lossless: SOC decreases by P_batt/(624*3600).
No controller, reward, forecast or terminal SOC objective is imported or run.

Feasible means an explicit allocation passed independent residual checks.
Infeasible means HiGHS returned its infeasibility status for the full linear
constraint system, within floating point solver tolerances; it is not a formal
exact-arithmetic certificate. Other solver outcomes remain unknown.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags, eye, hstack, vstack

FC_MIN_KW = 0.0
FC_MAX_KW = 600.0
BATT_MIN_KW = -624.0
BATT_MAX_KW = 1248.0
FC_RAMP_KW_PER_S = 48.0
BATTERY_CAPACITY_KWH = 624.0
SOC_MIN = 0.2
SOC_MAX = 0.8
INITIAL_SOC = 0.55
ENERGY_KW_S = BATTERY_CAPACITY_KWH * 3600.0
POWER_TOLERANCE_KW = 1e-6
SOC_TOLERANCE = 1e-9


def _validate_witness(load: np.ndarray, previous_fc: float, fc: np.ndarray):
    """Recompute power balance and every cumulative SOC from the proposed FC."""
    if fc.shape != load.shape or not np.all(np.isfinite(fc)):
        return {"valid": False, "reason": "nonfinite or wrong-sized FC witness"}, None
    batt = load - fc
    soc = INITIAL_SOC - np.cumsum(batt) / ENERGY_KW_S
    ramp = np.abs(np.diff(np.r_[previous_fc, fc]))
    checks = {
        "power_balance_max_abs_kw": float(np.max(np.abs(fc + batt - load))),
        "fc_bound_max_violation_kw": float(max(0., np.max(FC_MIN_KW - fc), np.max(fc - FC_MAX_KW))),
        "battery_bound_max_violation_kw": float(max(0., np.max(BATT_MIN_KW - batt), np.max(batt - BATT_MAX_KW))),
        "fc_ramp_max_violation_kw": float(max(0., np.max(ramp) - FC_RAMP_KW_PER_S)),
        "soc_bound_max_violation": float(max(0., SOC_MIN - np.min(soc), np.max(soc) - SOC_MAX)),
        "soc_dynamics_max_abs": float(np.max(np.abs(np.diff(np.r_[INITIAL_SOC, soc]) + batt / ENERGY_KW_S))),
        "power_tolerance_kw": POWER_TOLERANCE_KW,
        "soc_tolerance": SOC_TOLERANCE,
    }
    checks["valid"] = bool(
        all(np.isfinite(v) for v in checks.values())
        and all(checks[k] <= POWER_TOLERANCE_KW for k in (
            "power_balance_max_abs_kw", "fc_bound_max_violation_kw",
            "battery_bound_max_violation_kw", "fc_ramp_max_violation_kw"))
        and checks["soc_bound_max_violation"] <= SOC_TOLERANCE
        and checks["soc_dynamics_max_abs"] <= SOC_TOLERANCE
    )
    return checks, {"p_fc_kw": fc, "p_batt_kw": batt, "soc": soc}


def _envelope_witness(load: np.ndarray, previous_fc: float) -> np.ndarray | None:
    """Try an O(n) convex combination of ramp-feasible power envelopes.

    Backward/forward propagation gives the greatest and least sequences
    satisfying per-point FC/battery and neighboring ramp bounds (if these
    bounds intersect). A convex combination satisfies all those linear
    constraints. Intersecting its allowable mixing-coefficient intervals over
    every cumulative energy bound yields a sufficient existence witness.
    Failure here says nothing about existence; the full LP then decides.
    """
    lower = np.maximum(FC_MIN_KW, load - BATT_MAX_KW)
    upper = np.minimum(FC_MAX_KW, load - BATT_MIN_KW)
    lower[0] = max(lower[0], previous_fc - FC_RAMP_KW_PER_S)
    upper[0] = min(upper[0], previous_fc + FC_RAMP_KW_PER_S)
    for i in range(load.size - 2, -1, -1):
        lower[i] = max(lower[i], lower[i + 1] - FC_RAMP_KW_PER_S)
        upper[i] = min(upper[i], upper[i + 1] + FC_RAMP_KW_PER_S)
    for i in range(1, load.size):
        lower[i] = max(lower[i], lower[i - 1] - FC_RAMP_KW_PER_S)
        upper[i] = min(upper[i], upper[i - 1] + FC_RAMP_KW_PER_S)
    if np.any(lower > upper + POWER_TOLERANCE_KW):
        return None
    energy_lower_fc = np.cumsum(load - lower)
    energy_span = np.cumsum(upper - lower)
    energy_min = (INITIAL_SOC - SOC_MAX) * ENERGY_KW_S
    energy_max = (INITIAL_SOC - SOC_MIN) * ENERGY_KW_S
    variable = energy_span > 1e-8
    fixed = ~variable
    if np.any(energy_lower_fc[fixed] < energy_min - 1e-6) or np.any(energy_lower_fc[fixed] > energy_max + 1e-6):
        return None
    alpha_min, alpha_max = 0., 1.
    if np.any(variable):
        alpha_min = max(alpha_min, float(np.max((energy_lower_fc[variable] - energy_max) / energy_span[variable])))
        alpha_max = min(alpha_max, float(np.min((energy_lower_fc[variable] - energy_min) / energy_span[variable])))
    if alpha_min > alpha_max + 1e-12:
        return None
    target = energy_lower_fc[-1] / energy_span[-1] if energy_span[-1] > 1e-8 else 0.
    alpha = float(np.clip(target, alpha_min, alpha_max))
    return lower + alpha * (upper - lower)


def _solve_full_lp(load: np.ndarray, previous_fc: float):
    """O(n) sparse storage: variables are FC[kW] and cumulative battery[kW s]."""
    n = load.size
    identity = eye(n, format="csr")
    difference = diags([np.ones(n), -np.ones(n - 1)], [0, -1], shape=(n, n), format="csr")
    # q[t] - q[t-1] + FC[t] = load[t], q[-1] = 0.
    equality = hstack([identity, difference], format="csr")
    ramp_upper = np.full(n, FC_RAMP_KW_PER_S)
    ramp_lower = ramp_upper.copy()
    ramp_upper[0] += previous_fc
    ramp_lower[0] -= previous_fc
    power_rows = vstack([difference, -difference, identity, -identity], format="csr")
    inequalities = hstack([power_rows, power_rows * 0.], format="csr")
    inequalities.eliminate_zeros()
    limits = np.r_[ramp_upper, ramp_lower, load - BATT_MIN_KW, BATT_MAX_KW - load]
    bounds = np.empty((2 * n, 2))
    bounds[:n] = [FC_MIN_KW, FC_MAX_KW]
    bounds[n:] = [(INITIAL_SOC - SOC_MAX) * ENERGY_KW_S, (INITIAL_SOC - SOC_MIN) * ENERGY_KW_S]
    return linprog(
        np.zeros(2 * n), A_ub=inequalities, b_ub=limits,
        A_eq=equality, b_eq=load, bounds=bounds, method="highs",
        options={"primal_feasibility_tolerance": 1e-8,
                 "dual_feasibility_tolerance": 1e-8, "time_limit": 120.},
    )


def audit_feasibility(loads_kw, *, return_witness: bool = False) -> dict[str, Any]:
    """Audit an unmodified 1 s load sequence; invalid input raises ValueError.

    Returns status (feasible/infeasible/unknown), feasible (bool/None), method,
    solver_status/message, witness_validation residuals, min/max/final_soc,
    initialization and physical constraints. SOC summaries include the initial
    state. With return_witness, feasible results additionally contain ``witness``
    with NumPy arrays p_fc_kw, p_batt_kw and post-step soc, each length len(load)-1.
    No terminal SOC target is imposed. Solver resource failures remain unknown.
    """
    try:
        loads = np.asarray(loads_kw, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError("loads_kw must be a finite one-dimensional numeric series") from error
    if loads.ndim != 1 or loads.size < 2 or not np.all(np.isfinite(loads)):
        raise ValueError("loads_kw must contain at least two finite 1 s samples in one dimension")
    previous_fc = float(np.clip(loads[0], FC_MIN_KW, FC_MAX_KW))
    load = loads[1:]
    result = {
        "status": "unknown", "feasible": None,
        "method": "checked ramp-envelope convex allocation; full sparse feasibility LP on failure",
        "solver_status": None, "solver_message": None,
        "witness_validation": None, "min_soc": None, "max_soc": None, "final_soc": None,
        "sample_count": int(loads.size), "executed_step_count": int(load.size),
        "initialization": {"initial_soc": INITIAL_SOC, "previous_fc_kw": previous_fc,
                           "first_executed_load_index": 1, "dt_seconds": 1.},
        "constraints": {"fc_kw": [FC_MIN_KW, FC_MAX_KW],
                        "battery_kw": [BATT_MIN_KW, BATT_MAX_KW],
                        "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
                        "soc": [SOC_MIN, SOC_MAX], "fc_ramp_kw_per_s": FC_RAMP_KW_PER_S,
                        "battery_model": "lossless", "terminal_soc_target": None},
    }
    fc = _envelope_witness(load, previous_fc)
    checks, witness = (None, None) if fc is None else _validate_witness(load, previous_fc, fc)
    if checks is not None and checks["valid"]:
        result["method"] = "O(n) ramp-envelope convex allocation with independently checked physical witness"
    else:
        result["method"] = "full sparse linear feasibility problem (SciPy HiGHS); independently checked physical witness"
        try:
            solved = _solve_full_lp(load, previous_fc)
        except Exception as error:
            result["solver_message"] = f"{type(error).__name__}: {error}"
            return result
        result["solver_status"] = int(solved.status)
        result["solver_message"] = str(solved.message)
        if solved.status == 2:
            result.update(status="infeasible", feasible=False)
            return result
        if solved.status != 0 or solved.x is None:
            return result
        checks, witness = _validate_witness(load, previous_fc, np.asarray(solved.x[:load.size]))
    result["witness_validation"] = checks
    if not checks["valid"]:
        return result
    soc = witness["soc"]
    result.update(status="feasible", feasible=True,
                  min_soc=float(min(INITIAL_SOC, np.min(soc))),
                  max_soc=float(max(INITIAL_SOC, np.max(soc))), final_soc=float(soc[-1]))
    if return_witness:
        result["witness"] = witness
    return result
