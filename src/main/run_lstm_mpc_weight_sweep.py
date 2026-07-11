"""Weight sweep for the standard zero-delay LSTM-MPC baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[2]
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lstm_mpc_test import (  # noqa: E402
    OUT_DIR as LSTM_MPC_OUT_DIR,
    WEIGHT_SETS_JSON,
    load_weight_sets,
    run_all,
    write_action_table,
    write_json,
)


OUT_DIR = PROJ / "outputs/lstm_mpc_weight_sweep"
DEFAULT_SWEEP_WEIGHT_SETS = [
    "dp0_h2_only_diag",
    "dp0_h2_soc_v1",
    "dp0_h2_soc_batt_v1",
    "dp0_h2_soc_batt_ramp_v1",
    "dp0_h2_soc_batt_terminal_v1",
]
CHARGE_SUSTAINING_TERMINAL_ERROR_MAX = 0.10
WEIGHTED_OBJECTIVE_COLUMNS = [
    "weighted_h2_cost",
    "weighted_soc_cost",
    "weighted_ramp_cost",
    "weighted_batt_cost",
    "weighted_terminal_soc_cost",
]


def aggregate_weight_metrics(weight_set: str, metrics_df: pd.DataFrame) -> dict[str, Any]:
    h2_col = "h2_consumption_kg" if "h2_consumption_kg" in metrics_df.columns else "H2_total_kg"
    adjusted_h2_col = (
        "charge_sustaining_adjusted_h2_kg"
        if "charge_sustaining_adjusted_h2_kg" in metrics_df.columns
        else "charge_sustaining_adjusted_H2"
    )
    first_mode = str(metrics_df["soc_reference_mode"].iloc[0]) if "soc_reference_mode" in metrics_df.columns else ""
    return {
        "weight_set": weight_set,
        "soc_reference_mode": first_mode,
        "mean_soc_ref_value": float(metrics_df["soc_ref_value"].mean()) if "soc_ref_value" in metrics_df.columns else np.nan,
        "mean_soc_reserve": float(metrics_df["soc_reserve"].mean()) if "soc_reserve" in metrics_df.columns else np.nan,
        "mean_soc_terminal_error": float(metrics_df["soc_terminal_error"].mean()),
        "mean_soc_min": float(metrics_df["soc_min"].mean()),
        "mean_soc_max": float(metrics_df["soc_max"].mean()),
        "max_fc_kw": float(metrics_df["max_fc_kw"].max()) if "max_fc_kw" in metrics_df.columns else np.nan,
        "max_charge_power_kw_observed": float(metrics_df["max_charge_power_kw_observed"].max())
        if "max_charge_power_kw_observed" in metrics_df.columns
        else np.nan,
        "max_discharge_power_kw_observed": float(metrics_df["max_discharge_power_kw_observed"].max())
        if "max_discharge_power_kw_observed" in metrics_df.columns
        else np.nan,
        "total_time_fc_above_300kw_s": float(metrics_df["time_fc_above_300kw_s"].sum())
        if "time_fc_above_300kw_s" in metrics_df.columns
        else 0.0,
        "total_time_fc_above_400kw_s": float(metrics_df["time_fc_above_400kw_s"].sum())
        if "time_fc_above_400kw_s" in metrics_df.columns
        else 0.0,
        "total_time_batt_charge_above_200kw_s": float(metrics_df["time_batt_charge_above_200kw_s"].sum())
        if "time_batt_charge_above_200kw_s" in metrics_df.columns
        else 0.0,
        "total_time_batt_charge_above_300kw_s": float(metrics_df["time_batt_charge_above_300kw_s"].sum())
        if "time_batt_charge_above_300kw_s" in metrics_df.columns
        else 0.0,
        "max_initial_battery_only_time_min": float(metrics_df["initial_battery_only_time_min"].max())
        if "initial_battery_only_time_min" in metrics_df.columns
        else 0.0,
        "max_continuous_fc_off_under_load_min": float(metrics_df["max_continuous_fc_off_under_load_min"].max())
        if "max_continuous_fc_off_under_load_min" in metrics_df.columns
        else 0.0,
        "max_time_batt_covers_load_gt_80pct_min": float(metrics_df["time_batt_covers_load_gt_80pct_min"].max())
        if "time_batt_covers_load_gt_80pct_min" in metrics_df.columns
        else 0.0,
        "total_time_batt_covers_load_gt_80pct_min": float(metrics_df["time_batt_covers_load_gt_80pct_min"].sum())
        if "time_batt_covers_load_gt_80pct_min" in metrics_df.columns
        else 0.0,
        "min_soc_drop_first_10min": float(metrics_df["soc_drop_first_10min"].min())
        if "soc_drop_first_10min" in metrics_df.columns
        else 0.0,
        "min_soc_drop_first_30min": float(metrics_df["soc_drop_first_30min"].min())
        if "soc_drop_first_30min" in metrics_df.columns
        else 0.0,
        "max_fallback_control_used_ratio": float(metrics_df["fallback_control_used_ratio"].max())
        if "fallback_control_used_ratio" in metrics_df.columns
        else 0.0,
        "max_soc_rise_first_10min": float(metrics_df["soc_rise_first_10min"].max())
        if "soc_rise_first_10min" in metrics_df.columns
        else 0.0,
        "min_soc_start_minus_reserve": float((metrics_df["soc_start"] - metrics_df["soc_reserve"]).min())
        if {"soc_start", "soc_reserve"}.issubset(metrics_df.columns)
        else 0.0,
        "mean_h2_consumption_kg": float(metrics_df[h2_col].mean()),
        "total_h2_consumption_kg": float(metrics_df[h2_col].sum()),
        "mean_charge_sustaining_adjusted_h2_kg": float(metrics_df[adjusted_h2_col].mean())
        if adjusted_h2_col in metrics_df.columns
        else np.nan,
        "total_charge_sustaining_adjusted_h2_kg": float(metrics_df[adjusted_h2_col].sum())
        if adjusted_h2_col in metrics_df.columns
        else np.nan,
        "mean_abs_soc_delta": float(metrics_df["soc_delta"].abs().mean()) if "soc_delta" in metrics_df.columns else np.nan,
        "mean_fc_energy_kwh": float(metrics_df["fc_energy_kwh"].mean()),
        "mean_fc_ramp_mean_kw": float(metrics_df["fc_ramp_mean_kw"].mean()),
        "mean_battery_throughput_kwh": float(metrics_df["battery_throughput_kwh"].mean()),
        "mean_fc_shutdown_time_after_load_zero_min": float(
            metrics_df["fc_shutdown_time_after_load_zero_min"].replace([np.inf, -np.inf], np.nan).mean()
        ),
        "total_fc_idle_h2_consumption_kg": float(metrics_df["fc_idle_h2_consumption_kg"].sum()),
        "total_unserved_energy_kwh": float(metrics_df["unserved_energy_kwh"].sum()),
        "mean_solver_success_rate": float(metrics_df["solver_success_rate"].mean()),
    }


def recommend_weight_set(summary_rows: list[dict[str, Any]]) -> str:
    def _finite_or_large(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else 1.0e9

    feasible = [
        row
        for row in summary_rows
        if "_diag" not in str(row.get("weight_set", ""))
        and float(row["total_unserved_energy_kwh"]) <= 1e-9
        and float(row["mean_soc_min"]) >= 0.2 - 1e-9
        and float(row["mean_soc_max"]) <= 0.8 + 1e-9
        and float(row["mean_solver_success_rate"]) >= 0.99
    ]
    if not feasible:
        raise ValueError("No physically feasible clean H2-MPC weight set passed the sweep filters.")
    charge_sustaining = [
        row
        for row in feasible
        if _finite_or_large(row, "mean_soc_terminal_error") <= CHARGE_SUSTAINING_TERMINAL_ERROR_MAX + 1e-9
    ]
    ranking_pool = charge_sustaining if charge_sustaining else feasible
    candidates = sorted(
        ranking_pool,
        key=lambda row: (
            _finite_or_large(row, "mean_soc_terminal_error"),
            _finite_or_large(row, "mean_charge_sustaining_adjusted_h2_kg"),
            _finite_or_large(row, "mean_h2_consumption_kg"),
            _finite_or_large(row, "mean_battery_throughput_kwh"),
            _finite_or_large(row, "mean_fc_ramp_mean_kw"),
            _finite_or_large(row, "mean_abs_soc_delta"),
        ),
    )
    if not candidates:
        raise ValueError("No weight sweep rows are available.")
    return str(candidates[0]["weight_set"])


def _read_effective_config(weight_set_dir: Path) -> dict[str, Any]:
    path = weight_set_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("effective_mpc_config", {})


def _plot_power_soc(ts: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    t = ts["time_h"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(t, ts["load_total_kw"], color="black", lw=0.75, label="Load")
    axes[0].plot(t, ts["P_fc_kw"], color="tab:red", lw=0.75, label="P_fc")
    axes[0].plot(t, ts["P_batt_kw"], color="tab:blue", lw=0.65, label="P_batt")
    axes[0].axhline(0.0, color="gray", lw=0.5)
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.15)
    axes[1].plot(t, ts["SOC"], color="tab:green", lw=0.8, label="SOC")
    if "soc_ref_value" in ts.columns:
        axes[1].plot(t, ts["soc_ref_value"], color="gray", ls="--", lw=0.8, label="SOC_ref")
    else:
        axes[1].axhline(float(cfg.get("soc_target", 0.65)), color="gray", ls="--", lw=0.8, label="SOC_ref")
    if "soc_reserve" in ts.columns:
        axes[1].plot(t, ts["soc_reserve"], color="tab:orange", ls="-.", lw=0.8, label="SOC_reserve")
    else:
        axes[1].axhline(float(cfg.get("soc_reserve", 0.55)), color="tab:orange", ls="-.", lw=0.8, label="SOC_reserve")
    axes[1].axhline(float(cfg.get("soc_min", 0.2)), color="tab:red", ls=":", lw=0.8, label="SOC_min")
    axes[1].axhline(float(cfg.get("soc_max", 0.8)), color="tab:red", ls=":", lw=0.8, label="SOC_max")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("SOC")
    axes[1].legend(fontsize=8, ncol=4)
    axes[1].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_objective_decomposition(ts: pd.DataFrame, out_path: Path) -> None:
    t = ts["time_h"].to_numpy(dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    for col in WEIGHTED_OBJECTIVE_COLUMNS:
        values = pd.to_numeric(ts[col], errors="coerce").fillna(0.0) if col in ts.columns else pd.Series(0.0, index=ts.index)
        ax.plot(t, values.to_numpy(dtype=float), lw=0.75, label=col)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Weighted objective term")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_soc_reference_behavior(ts: pd.DataFrame, out_path: Path) -> None:
    t = ts["time_h"].to_numpy(dtype=float)
    t0 = float(t[0]) if len(t) else 0.0
    mask = t <= t0 + (20.0 / 60.0)
    if not np.any(mask):
        mask = np.ones(len(ts), dtype=bool)
    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(t[mask], ts.loc[mask, "P_fc_kw"], color="tab:red", lw=0.8, label="P_fc")
    axes[0].plot(t[mask], ts.loc[mask, "P_batt_kw"], color="tab:blue", lw=0.8, label="P_batt")
    axes[0].axhline(300.0, color="tab:red", ls=":", lw=0.7, label="300 kW")
    axes[0].axhline(-80.0, color="tab:blue", ls=":", lw=0.7, label="-80 kW charge limit")
    axes[0].axhline(0.0, color="gray", lw=0.5)
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(fontsize=8, ncol=4)
    axes[0].grid(alpha=0.15)

    axes[1].plot(t[mask], ts.loc[mask, "SOC"], color="tab:green", lw=0.8, label="SOC")
    if "soc_ref_value" in ts.columns:
        axes[1].plot(t[mask], ts.loc[mask, "soc_ref_value"], color="gray", ls="--", lw=0.8, label="SOC_ref")
    if "soc_reserve" in ts.columns:
        axes[1].plot(t[mask], ts.loc[mask, "soc_reserve"], color="tab:orange", ls="-.", lw=0.8, label="SOC_reserve")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("SOC")
    axes[1].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_low_load_behavior(ts: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    threshold = float(cfg.get("low_load_threshold_kw", 5.0))
    t = ts["time_h"].to_numpy(dtype=float)
    low = ts["load_total_kw"].to_numpy(dtype=float) < threshold
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    if np.any(low):
        for col, color, label in [
            ("load_total_kw", "black", "Load"),
            ("P_fc_kw", "tab:red", "P_fc"),
            ("P_batt_kw", "tab:blue", "P_batt"),
        ]:
            values = ts[col].to_numpy(dtype=float)
            ax.plot(t, np.where(low, values, np.nan), lw=0.8, color=color, label=label)
        ax2 = ax.twinx()
        ax2.plot(t, np.where(low, ts["SOC"].to_numpy(dtype=float), np.nan), color="tab:green", lw=0.8, label="SOC")
        ax2.set_ylabel("SOC")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, ncol=4)
    else:
        ax.text(0.5, 0.5, f"No load < {threshold:.1f} kW interval", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Power (kW)")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_initial_dispatch_zoom(ts: pd.DataFrame, cfg: dict[str, Any], out_path: Path) -> None:
    if ts.empty:
        return
    t = ts["time_h"].to_numpy(dtype=float)
    t0 = float(t[0])
    mask = t <= t0 + 1.0 + 1e-12
    if not np.any(mask):
        mask = np.ones(len(ts), dtype=bool)
    weight_set = str(ts["weight_set"].iloc[0]) if "weight_set" in ts.columns else ""
    title = (
        f"{weight_set} | mode={cfg.get('soc_reference_mode', '')} | penalty={cfg.get('soc_penalty_type', 'symmetric_tracking')} | "
        f"q_h2={cfg.get('q_h2', '')} q_soc={cfg.get('q_soc', '')} q_batt={cfg.get('q_batt', '')} "
        f"q_terminal={cfg.get('q_terminal_soc', '')} | soc_band={cfg.get('soc_band', '')} "
        f"terminal_band={cfg.get('terminal_soc_band', '')}"
    )
    fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
    axes[0].plot(t[mask], ts.loc[mask, "load_total_kw"], color="black", lw=0.85, label="Load")
    axes[0].plot(t[mask], ts.loc[mask, "P_fc_kw"], color="tab:red", lw=0.85, label="P_fc")
    axes[0].plot(t[mask], ts.loc[mask, "P_batt_kw"], color="tab:blue", lw=0.75, label="P_batt")
    axes[0].axhline(0.0, color="gray", lw=0.5)
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title(title, fontsize=8)
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(alpha=0.15)

    axes[1].plot(t[mask], ts.loc[mask, "SOC"], color="tab:green", lw=0.85, label="SOC")
    if "soc_ref_value" in ts.columns:
        axes[1].plot(t[mask], ts.loc[mask, "soc_ref_value"], color="gray", ls="--", lw=0.8, label="SOC_ref")
    if "soc_reserve" in ts.columns:
        axes[1].plot(t[mask], ts.loc[mask, "soc_reserve"], color="tab:orange", ls="-.", lw=0.8, label="SOC_reserve")
    axes[1].set_ylabel("SOC")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(alpha=0.15)

    values = (
        pd.to_numeric(ts.loc[mask, "fallback_control_used"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "fallback_control_used" in ts.columns
        else np.zeros(int(np.sum(mask)), dtype=float)
    )
    axes[2].step(t[mask], values, where="post", lw=0.8, label="fallback")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_ylabel("Active")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(fontsize=8, ncol=3)
    axes[2].grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_weight_set_figures(weight_set_dir: Path, figures_dir: Path) -> None:
    cfg = _read_effective_config(weight_set_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(weight_set_dir.glob("voyage_*_timeseries.csv")):
        stem = path.stem.replace("_timeseries", "")
        ts = pd.read_csv(path)
        _plot_power_soc(ts, cfg, figures_dir / f"{stem}_power_soc.png")
        _plot_initial_dispatch_zoom(ts, cfg, figures_dir / f"{stem}_initial_dispatch_zoom.png")
        _plot_objective_decomposition(ts, figures_dir / f"{stem}_objective_decomposition.png")
        _plot_soc_reference_behavior(ts, figures_dir / f"{stem}_soc_reference_behavior.png")
        _plot_low_load_behavior(ts, cfg, figures_dir / f"{stem}_low_load_fc_behavior.png")


def write_diagnosis_report(
    output_dir: Path,
    summary_df: pd.DataFrame,
    objective_df: pd.DataFrame,
    recommended: str,
) -> None:
    chosen = summary_df[summary_df["weight_set"] == recommended]
    objective_summary = "No objective decomposition rows."
    if not objective_df.empty:
        grouped = objective_df.groupby("weight_set", as_index=False)[
            [f"{col}_sum" for col in WEIGHTED_OBJECTIVE_COLUMNS if f"{col}_sum" in objective_df.columns]
            + (["total_objective_sum"] if "total_objective_sum" in objective_df.columns else [])
        ].sum()
        objective_summary = grouped.to_string(index=False)
    high_fc_problem = bool((summary_df.get("max_fc_kw", pd.Series(dtype=float)).astype(float) > 300.0).any())
    initial_batt_problem = bool(
        (summary_df.get("max_initial_battery_only_time_min", pd.Series(dtype=float)).astype(float) > 2.0).any()
    )
    lines = [
        "# Clean H2-MPC Baseline Report",
        "",
        "This run restores the fixed LSTM-H2-MPC baseline to a clean literature-consistent form.",
        "",
        "## Removed Or Disabled Rule-Based Controls",
        "",
        "- `low_load_fc_suppression` is removed from active weights and solver bound construction.",
        "- `soc_recovery_power_limit` is removed from active weights and solver bound construction.",
        "- `sustained_load_battery_discharge_limit` is removed from active weights and solver bound construction.",
        "- `fc_overproduction_limit` is removed from active weights and solver bound construction.",
        "- Initial battery-only and battery-only dispatch exclusion rules are not used in baseline control or recommendation.",
        "- No if-load-then-limit or if-SOC-then-force-FC/battery rule is active.",
        "",
        "## Current Baseline Scope",
        "",
        "- DQN was not trained in this run.",
        "- Timing remains standard zero-delay: `mpc_load_ref = [actual_load_t] + lstm_pred[:5]`.",
        "- The controller remains total-power LSTM-H2-MPC, not left/right device-level EMS.",
        "- H2 cost uses the imported fresh Dp=0 fuel-cell curve through the CasADi quadratic fit.",
        "- Forecasts entering MPC remain nonnegative projected LSTM predictions.",
        "",
        "## Physical Constraints",
        "",
        "- Fuel cell: `0 <= P_fc <= fuel_cell_max_kw`.",
        "- Fuel-cell ramp: `|P_fc[k] - P_fc[k-1]| <= fuel_cell_ramp_kw`.",
        "- Battery: `-battery_charge_max_kw <= P_batt <= battery_discharge_max_kw`.",
        "- SOC: `SOC_min <= SOC <= SOC_max`.",
        "- No load-dependent or SOC-dependent bound rewrite is used.",
        "",
        "## Objective Function Terms",
        "",
        "- Hydrogen mass cost: Dp0 curve-based stage hydrogen mass.",
        "- SOC cost: tracking/reserve/charge-sustaining cost selected by `soc_reference_mode` and `soc_band`.",
        "- Battery throughput/degradation proxy: absolute battery power term controlled by `q_batt`.",
        "- Fuel-cell ramp cost: `Delta P_fc` smoothness term controlled by `q_ramp`.",
        "- Optional terminal SOC cost: controlled by `q_terminal_soc` and `terminal_soc_band`.",
        "",
        "## Objective Decomposition",
        "",
        "```text",
        objective_summary,
        "```",
        "",
        "## Weight Sweep Summary",
        "",
        "```text",
        summary_df.to_string(index=False) if not summary_df.empty else "No sweep rows.",
        "```",
        "",
        "## Recommendation",
        "",
        f"- Recommended baseline: `{recommended}`",
        "- Selection first requires physical feasibility: no unserved energy, SOC within bounds, and solver success rate >= 0.99.",
        "- Diagnostic weight sets ending in `_diag` are reported but not eligible as the recommended baseline.",
        f"- Recommendation requires charge-sustaining behavior when available: mean terminal SOC error <= {CHARGE_SUSTAINING_TERMINAL_ERROR_MAX:.2f}.",
        "- Ranking then uses objective-related metrics: terminal SOC error, charge-sustaining adjusted H2, H2, battery throughput, FC ramp mean, and absolute SOC delta.",
        "- Initial battery-only behavior and high FC power are reported as diagnostics, not hard-coded exclusion rules.",
        "",
        "```text",
        chosen.to_string(index=False) if not chosen.empty else "Recommended row not found.",
        "```",
        "",
        "## Remaining Behavior Diagnostics",
        "",
        f"- Any candidate with `max_fc_kw > 300 kW`: `{high_fc_problem}`.",
        f"- Any candidate with initial battery-only time > 2 min: `{initial_batt_problem}`.",
        "- If these remain in the selected baseline, correction must use objective weights, cost normalization, SOC reference, terminal SOC cost, or battery throughput/degradation proxy cost.",
        "- Do not add if-load or if-SOC control rules to mask these behaviors.",
        "",
        "## Clean Active Weight Sets",
        "",
        "- `dp0_h2_only_diag`",
        "- `dp0_h2_soc_v1`",
        "- `dp0_h2_soc_batt_v1`",
        "- `dp0_h2_soc_batt_ramp_v1`",
        "- `dp0_h2_soc_batt_terminal_v1`",
        "",
        "A clean literature-consistent LSTM-H2-MPC baseline has been restored, using only physical constraints and objective-function-based optimization terms.",
    ]
    output_dir.joinpath("README_CLEAN_H2_MPC_BASELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_sweep_result_to_main_readme(recommended: str, summary_df: pd.DataFrame) -> None:
    readme = LSTM_MPC_OUT_DIR / "README_LSTM_MPC_TEST.md"
    if not readme.exists():
        return
    section = [
        "",
        "## Weight Sweep Result",
        "",
        f"- Recommended fixed MPC baseline: `{recommended}`",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
        "",
    ]
    text = readme.read_text(encoding="utf-8")
    marker = "\n## Weight Sweep Result\n"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    readme.write_text(text.rstrip() + "\n" + "\n".join(section), encoding="utf-8")


def run_sweep(
    *,
    output_dir: Path = OUT_DIR,
    init_soc: float = 0.55,
    max_steps: int | None = None,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    loaded_weight_sets = load_weight_sets(WEIGHT_SETS_JSON)
    weight_sets = {key: loaded_weight_sets[key] for key in DEFAULT_SWEEP_WEIGHT_SETS if key in loaded_weight_sets}
    all_voyage_rows: list[pd.DataFrame] = []
    objective_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for weight_set in weight_sets:
        print(f"Running weight_set={weight_set}")
        weight_dir = output_dir / weight_set
        metrics_df, _, _ = run_all(
            weight_set=weight_set,
            output_dir=weight_dir,
            init_soc=init_soc,
            make_plots=False,
            write_outputs=True,
            max_steps=max_steps,
            device=device,
        )
        plot_weight_set_figures(weight_dir, output_dir / "figures" / weight_set)
        objective_path = weight_dir / "objective_decomposition.csv"
        if objective_path.exists():
            objective_part = pd.read_csv(objective_path)
            objective_part["weight_set"] = weight_set
            objective_rows.append(objective_part)
        summary = aggregate_weight_metrics(weight_set, metrics_df)
        summary_rows.append(summary)
        metrics_with_summary = metrics_df.copy()
        for key, value in summary.items():
            if key != "weight_set":
                metrics_with_summary[key] = value
        all_voyage_rows.append(metrics_with_summary)
    voyage_df = pd.concat(all_voyage_rows, ignore_index=True) if all_voyage_rows else pd.DataFrame()
    objective_df = pd.concat(objective_rows, ignore_index=True) if objective_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    recommended = recommend_weight_set(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    voyage_df.to_csv(output_dir / "sweep_metrics.csv", index=False, encoding="utf-8-sig")
    objective_df.to_csv(output_dir / "sweep_objective_decomposition.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(output_dir / "sweep_summary.csv", index=False, encoding="utf-8-sig")
    write_json(
        output_dir / "sweep_recommendation.json",
        {
            "recommended_weight_set": recommended,
            "selection_rule": [
                "total_unserved_energy_kwh == 0",
                "soc_min >= 0.2",
                "soc_max <= 0.8",
                "mean_solver_success_rate >= 0.99",
                "exclude diagnostic weight sets ending in _diag",
                f"prefer charge-sustaining candidates with mean_soc_terminal_error <= {CHARGE_SUSTAINING_TERMINAL_ERROR_MAX:.2f}",
                "sort by terminal SOC error, charge-sustaining adjusted H2, H2, battery throughput, FC ramp mean, abs SOC delta",
                "initial battery-only time and high FC power are diagnostics only, not hard exclusion rules",
            ],
        },
    )
    write_json(output_dir / "recommended_weight_set.json", {"recommended_weight_set": recommended})
    write_action_table(base=weight_sets[recommended], base_weight_set=recommended)
    write_diagnosis_report(output_dir, summary_df, objective_df, recommended)
    append_sweep_result_to_main_readme(recommended, summary_df)
    return voyage_df, summary_df, recommended


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LSTM-MPC fixed-weight sweep on the 7-2-1 test voyages.")
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--soc", type=float, default=0.55)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="Optional debug limit; default runs full voyages.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, summary_df, recommended = run_sweep(
        output_dir=Path(args.output_dir),
        init_soc=float(args.soc),
        max_steps=args.max_steps,
        device=args.device,
    )
    print(f"Saved: {Path(args.output_dir)}")
    print(summary_df.to_string(index=False))
    print(f"Recommended fixed MPC baseline: {recommended}")


if __name__ == "__main__":
    main()
