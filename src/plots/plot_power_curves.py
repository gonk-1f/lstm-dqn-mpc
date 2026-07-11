from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_power_curves(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(df["load_kw"], label="Load")
    plt.plot(df["fuel_cell_ref_kw"], label="FC Ref")
    plt.plot(df["battery_ref_kw"], label="Battery Ref")
    plt.plot(df["fuel_cell_power_kw"], label="FC Actual")
    plt.plot(df["battery_power_kw"], label="Battery Actual")
    plt.plot(df["supplied_kw"], label="Supplied")
    plt.xlabel("Step")
    plt.ylabel("Power (kW)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
