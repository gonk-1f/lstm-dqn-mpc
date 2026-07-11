"""Validate the MATLAB-exported fresh D_p=0 fuel-cell curve import."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _bootstrap_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


PROJ = _bootstrap_path()

from mpc.solvers.fc_dp0_curve import (  # noqa: E402
    CURVE_CSV_PATH,
    dp0_quadratic_coefficients,
    eta_dp0,
    h2_rate_gps_dp0,
    h2_rate_gps_dp0_quadratic,
    load_dp0_curve,
)


DEFAULT_OUTPUT_DIR = PROJ / "outputs/fc_dp0_curve_validation"
DEFAULT_MATLAB_PNG = Path(r"C:\Users\20883\OneDrive\Desktop\氢耗\FC_Dp0_hydrogen_efficiency.png")
DEFAULT_RATED_TOTAL_KW = 560.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate imported fresh D_p=0 FC hydrogen/efficiency curve.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--matlab-png", default=str(DEFAULT_MATLAB_PNG))
    parser.add_argument("--p-rated-total-kw", type=float, default=DEFAULT_RATED_TOTAL_KW)
    return parser


def _resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJ / path


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_points_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_validation(output_path: Path, matlab_png: Path) -> None:
    p_sys, m_h2, eta_percent = load_dp0_curve()
    fig = plt.figure(figsize=(13.5, 6.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.1])
    ax_h2 = fig.add_subplot(grid[0, 0])
    ax_eta = fig.add_subplot(grid[1, 0])
    ax_ref = fig.add_subplot(grid[:, 1])

    ax_h2.plot(p_sys, h2_rate_gps_dp0(p_sys, p_rated_total_kw=100.0), color="black", linewidth=1.8)
    ax_h2.scatter(p_sys[:: max(len(p_sys) // 12, 1)], m_h2[:: max(len(p_sys) // 12, 1)], s=15, color="tab:blue")
    ax_h2.set_ylabel("mH2 (g/s)")
    ax_h2.set_title("Python import: hydrogen rate")
    ax_h2.grid(True, alpha=0.25)

    ax_eta.plot(p_sys, eta_dp0(p_sys, p_rated_total_kw=100.0) * 100.0, color="black", linewidth=1.8)
    ax_eta.scatter(
        p_sys[:: max(len(p_sys) // 12, 1)],
        eta_percent[:: max(len(p_sys) // 12, 1)],
        s=15,
        color="tab:orange",
    )
    ax_eta.set_xlabel("P_sys (kW)")
    ax_eta.set_ylabel("eta (%)")
    ax_eta.set_title("Python import: efficiency")
    ax_eta.grid(True, alpha=0.25)

    if matlab_png.exists():
        image = plt.imread(str(matlab_png))
        ax_ref.imshow(image)
        ax_ref.set_title("MATLAB exported PNG reference")
    else:
        ax_ref.text(0.5, 0.5, f"MATLAB PNG not found:\n{matlab_png}", ha="center", va="center", fontsize=9)
    ax_ref.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matlab_png = _resolve(args.matlab_png)
    p_rated = float(args.p_rated_total_kw)
    ratios = np.array([0.0, 0.1, 0.5, 1.0], dtype=float)
    p_total = p_rated * ratios
    h2_gps = h2_rate_gps_dp0(p_total, p_rated_total_kw=p_rated)
    eta = eta_dp0(p_total, p_rated_total_kw=p_rated)
    rows = [
        {
            "load_ratio": float(ratio),
            "P_fc_total_kw": float(power),
            "P_map_kW": float(100.0 * ratio),
            "mH2_total_g_s": float(rate),
            "eta": float(eff),
        }
        for ratio, power, rate, eff in zip(ratios, p_total, h2_gps, eta)
    ]

    p_sys, m_h2, eta_percent = load_dp0_curve()
    a1, a2 = dp0_quadratic_coefficients()
    exact_560 = h2_rate_gps_dp0(p_rated * (p_sys / 100.0), p_rated_total_kw=p_rated)
    fit_560 = h2_rate_gps_dp0_quadratic(p_rated * (p_sys / 100.0), p_rated_total_kw=p_rated)
    peak_idx = int(np.argmax(eta_percent))
    checks = {
        "p0_h2_is_zero": bool(abs(rows[0]["mH2_total_g_s"]) <= 1.0e-12),
        "p0_eta_is_zero": bool(abs(rows[0]["eta"]) <= 1.0e-12),
        "eta_rises_then_declines": bool(0 < peak_idx < len(eta_percent) - 1 and eta_percent[peak_idx] > eta_percent[-1]),
        "eta_peak_percent": float(eta_percent[peak_idx]),
        "eta_peak_P_sys_kW": float(p_sys[peak_idx]),
        "quadratic_a1": float(a1),
        "quadratic_a2": float(a2),
        "quadratic_rmse_gps_at_rated_total": float(np.sqrt(np.mean((fit_560 - exact_560) ** 2))),
    }

    plot_path = output_dir / "fc_dp0_curve_python_vs_matlab.png"
    points_csv = output_dir / "fc_dp0_curve_check_points.csv"
    summary_json = output_dir / "fc_dp0_curve_validation.json"
    _plot_validation(plot_path, matlab_png)
    _write_points_csv(points_csv, rows)
    payload = {
        "curve_csv": CURVE_CSV_PATH,
        "matlab_png": matlab_png,
        "python_validation_plot": plot_path,
        "points_csv": points_csv,
        "p_rated_total_kw": p_rated,
        "points": rows,
        "checks": checks,
    }
    summary_json.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
