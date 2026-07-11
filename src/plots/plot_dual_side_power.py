from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dual_side_power(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(df["total_load_kw"], label="Total Load", color="tab:blue")
    axes[0].plot(df["total_supply_kw"], label="Total Supply", color="tab:orange")
    if "left_fuel_cell_ref_kw" in df.columns:
        axes[0].plot(df["left_fuel_cell_ref_kw"], label="Left FC Ref", color="tab:green", linestyle="--")
    if "right_fuel_cell_ref_kw" in df.columns:
        axes[0].plot(df["right_fuel_cell_ref_kw"], label="Right FC Ref", color="tab:red", linestyle="--")
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend(ncol=2, fontsize=8)

    if "left_soc" in df.columns:
        axes[1].plot(df["left_soc"], label="Left SOC", color="tab:green")
    if "right_soc" in df.columns:
        axes[1].plot(df["right_soc"], label="Right SOC", color="tab:red")
    if "left_tracking_error_kw" in df.columns:
        axes[1].plot(df["left_tracking_error_kw"], label="Left Track Err", color="tab:purple", linestyle=":")
    if "right_tracking_error_kw" in df.columns:
        axes[1].plot(df["right_tracking_error_kw"], label="Right Track Err", color="tab:brown", linestyle=":")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("SOC / Error")
    axes[1].legend(ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_power_balance(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    total_fc = df.get("left_fuel_cell_power_kw", 0.0) + df.get("right_fuel_cell_power_kw", 0.0)
    total_batt = df.get("left_battery_power_kw", 0.0) + df.get("right_battery_power_kw", 0.0)

    axes[0].plot(df["total_load_kw"], label="Load", color="tab:blue", linewidth=1.5)
    axes[0].plot(total_fc, label="Fuel Cell Power", color="tab:green", linewidth=1.2)
    axes[0].plot(total_batt, label="Battery Power", color="tab:orange", linewidth=1.2)
    axes[0].plot(df["total_supply_kw"], label="Total Supply", color="tab:red", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Power (kW)")
    axes[0].set_title("Power Allocation and Balance")
    axes[0].legend(ncol=2, fontsize=8)

    axes[1].plot(df["total_balance_error_kw"], label="Balance Error", color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Error (kW)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
