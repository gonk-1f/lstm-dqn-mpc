"""Forecast errors over aligned origin-by-horizon load arrays."""

from __future__ import annotations

import numpy as np


def persistence_forecast(histories_kw: np.ndarray, *, horizon: int = 5) -> np.ndarray:
    """Repeat L_k for every future step at each forecast origin k."""
    histories = np.asarray(histories_kw, dtype=np.float64)
    if histories.ndim != 3 or histories.shape[2] != 1 or histories.shape[1] == 0:
        raise ValueError("histories must have shape [origins, history, 1]")
    if type(horizon) is not int or horizon <= 0 or not np.isfinite(histories).all():
        raise ValueError("horizon and histories must be valid")
    return np.repeat(histories[:, -1, 0, None], horizon, axis=1)


def _one_scope(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    error = actual - predicted
    absolute = np.abs(error)
    denominator = float(np.abs(actual).sum())
    zero_count = int(np.count_nonzero(actual == 0.0))
    return {
        "mae_kw": float(absolute.mean()),
        "rmse_kw": float(np.sqrt(np.mean(error**2))),
        "bias_kw": float(error.mean()),
        "wape_percent": float(100.0 * absolute.sum() / denominator) if denominator > 0 else None,
        "mape_percent": float(100.0 * np.mean(absolute / np.abs(actual))) if zero_count == 0 else None,
        "zero_actual_count": zero_count,
        "case_count": int(actual.size),
    }


def forecast_metrics(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> dict[str, object]:
    """MAPE is null if any actual is zero; no epsilon or case deletion."""
    actual = np.asarray(actual_kw, dtype=np.float64)
    predicted = np.asarray(predicted_kw, dtype=np.float64)
    if actual.ndim != 2 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must have equal nonempty [origins,horizon] shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted must be finite")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("load powers must be nonnegative")
    result: dict[str, object] = _one_scope(actual, predicted)
    result["horizon"] = [
        _one_scope(actual[:, step], predicted[:, step])
        for step in range(actual.shape[1])
    ]
    return result
