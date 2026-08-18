from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np


REQUIRED_COLUMNS = ("P_sys_kW", "mH2_g_s_Dp0", "eta_percent_Dp0")
CURVE_CSV_PATH = Path(__file__).resolve().parents[3] / "data" / "fuel_cell" / "FC_Dp0_curve_for_Python.csv"
CURVE_SOURCE_LABEL = "matlab_exported_dp0_curve"


def _as_1d(values: np.ndarray | float | Iterable[float]) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(values, dtype=float)
    return np.ravel(arr), tuple(arr.shape)


@lru_cache(maxsize=1)
def load_dp0_curve() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CURVE_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing D_p=0 curve CSV: {CURVE_CSV_PATH}")

    with CURVE_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {CURVE_CSV_PATH}: {tuple(reader.fieldnames or ())}; "
                f"expected {REQUIRED_COLUMNS}"
            )
        rows: list[tuple[float, float, float]] = []
        for line_no, row in enumerate(reader, start=2):
            try:
                rows.append(
                    (
                        float(row["P_sys_kW"]),
                        float(row["mH2_g_s_Dp0"]),
                        float(row["eta_percent_Dp0"]),
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid row at line {line_no} in {CURVE_CSV_PATH}: {row}") from exc

    if not rows:
        raise ValueError(f"No data rows found in {CURVE_CSV_PATH}")

    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    data = data[order]
    p_sys = data[:, 0]
    m_h2 = data[:, 1]
    eta_percent = data[:, 2]

    if np.any(np.diff(p_sys) < -1.0e-12):
        raise ValueError(f"P_sys_kW must be nondecreasing in {CURVE_CSV_PATH}")
    if not np.isclose(p_sys[0], 0.0):
        raise ValueError(f"Expected the curve to start at 0 kW in {CURVE_CSV_PATH}")
    if not np.isclose(p_sys[-1], 100.0):
        raise ValueError(f"Expected the curve to end at 100 kW in {CURVE_CSV_PATH}")
    if np.any(m_h2 < -1.0e-12):
        raise ValueError(f"Hydrogen rate must be nonnegative in {CURVE_CSV_PATH}")
    if np.any(eta_percent < -1.0e-12):
        raise ValueError(f"Efficiency must be nonnegative in {CURVE_CSV_PATH}")
    return p_sys, m_h2, eta_percent


def _map_to_curve(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    *,
    p_rated_total_kw: float,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, float]:
    p_fc, shape = _as_1d(p_fc_total_kw)
    rated = max(float(p_rated_total_kw), 1.0e-12)
    ratio = np.clip(p_fc / rated, 0.0, 1.0)
    mapped_kw = 100.0 * ratio
    scale = rated / 100.0
    return p_fc, shape, mapped_kw, scale


def _interp_curve(
    values_kw: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    interpolated = np.interp(values_kw, x, y)
    return np.asarray(interpolated, dtype=float).reshape(shape)


def h2_rate_gps_dp0(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    p_rated_total_kw: float = 600.0,
) -> np.ndarray:
    p_sys, m_h2, _ = load_dp0_curve()
    _, shape, mapped_kw, scale = _map_to_curve(p_fc_total_kw, p_rated_total_kw=p_rated_total_kw)
    return scale * _interp_curve(mapped_kw, p_sys, m_h2, shape)


def eta_dp0(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    p_rated_total_kw: float = 600.0,
) -> np.ndarray:
    p_sys, _, eta_percent = load_dp0_curve()
    _, shape, mapped_kw, _ = _map_to_curve(p_fc_total_kw, p_rated_total_kw=p_rated_total_kw)
    return _interp_curve(mapped_kw, p_sys, eta_percent / 100.0, shape)


def h2_kg_step_dp0(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    *,
    dt_seconds: float = 30.0,
    p_rated_total_kw: float = 600.0,
) -> np.ndarray:
    return h2_rate_gps_dp0(p_fc_total_kw, p_rated_total_kw=p_rated_total_kw) * float(dt_seconds) / 1000.0


def dp0_quadratic_coefficients() -> tuple[float, float]:
    p_sys, m_h2, _ = load_dp0_curve()
    r = p_sys / 100.0
    design = np.column_stack([r, r**2])
    a1, a2 = np.linalg.lstsq(design, m_h2, rcond=None)[0]
    return float(a1), float(a2)


def h2_rate_gps_dp0_quadratic(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    p_rated_total_kw: float = 600.0,
) -> np.ndarray:
    a1, a2 = dp0_quadratic_coefficients()
    p_fc, shape = _as_1d(p_fc_total_kw)
    rated = max(float(p_rated_total_kw), 1.0e-12)
    ratio = np.clip(p_fc / rated, 0.0, 1.0)
    scale = rated / 100.0
    rate = scale * (a1 * ratio + a2 * ratio**2)
    return np.asarray(rate, dtype=float).reshape(shape)


def h2_kg_step_dp0_quadratic(
    p_fc_total_kw: np.ndarray | float | Iterable[float],
    *,
    dt_seconds: float = 30.0,
    p_rated_total_kw: float = 600.0,
) -> np.ndarray:
    return h2_rate_gps_dp0_quadratic(p_fc_total_kw, p_rated_total_kw=p_rated_total_kw) * float(dt_seconds) / 1000.0
