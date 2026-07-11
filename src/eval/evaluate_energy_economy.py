from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_energy_economy(result_csv: str | Path) -> dict:
    df = pd.read_csv(result_csv)
    return {
        "fuel_cell_energy_proxy": float(df["fuel_cell_power_kw"].abs().sum()),
        "battery_energy_proxy": float(df["battery_power_kw"].abs().sum()),
        "tracking_error_proxy": float(df["tracking_error_kw"].abs().mean()),
        "balance_error_proxy": float(df["balance_error_kw"].abs().mean()),
    }
