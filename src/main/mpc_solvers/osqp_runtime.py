from __future__ import annotations

import time
from typing import Any

import numpy as np

from .formal_config import FIXED_SOC_REFERENCE
from .mpc_qp_formulation import QpMpcConfig, resolved_ramp_kw_per_step


def _try_import_osqp() -> tuple[Any | None, str | None]:
    try:
        import osqp  # type: ignore

        return osqp, None
    except Exception as exc:  # pragma: no cover - depends on local environment
        return None, str(exc)


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

    n_constraints = 8 * horizon + 2
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

    deficit_mode = str(config.soc_penalty_mode) == "deficit_only"
    lower[cursor : cursor + horizon] = 0.0
    upper[cursor : cursor + horizon] = (
        max(FIXED_SOC_REFERENCE - float(config.soc_min), 0.0)
        if deficit_mode
        else 0.0
    )
    cursor += horizon

    lower[cursor : cursor + horizon] = (
        FIXED_SOC_REFERENCE if deficit_mode else -np.inf
    )
    upper[cursor : cursor + horizon] = np.inf
    cursor += horizon

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


def _solve_with_persistent_osqp(
    solver: Any,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    linear: np.ndarray | None = None,
) -> tuple[Any, float]:
    start = time.perf_counter()
    if linear is None:
        solver.update(l=lower, u=upper)
    else:
        solver.update(q=linear, l=lower, u=upper)
    # Preserve OSQP's status/result object for explicit failure handling.
    # The library plans to change its default to raise on non-solved status.
    result = solver.solve(raise_error=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, float(elapsed_ms)
