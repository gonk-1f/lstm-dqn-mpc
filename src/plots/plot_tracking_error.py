from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_tracking_error(result_csv: str | Path, output_png: str | Path) -> Path:
    df = pd.read_csv(result_csv)
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(df["tracking_error_kw"], label="Tracking Error")
    plt.plot(df["balance_error_kw"], label="Balance Error")
    plt.xlabel("Step")
    plt.ylabel("kW")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
