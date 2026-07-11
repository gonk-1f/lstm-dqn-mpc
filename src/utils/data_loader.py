from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class VoyageFiles:
    root: Path
    bms_dir: Path
    ems_dir: Path
    fuel_cell_dir: Path
    propulsion_dir: Path


def discover_voyages(raw_root: str | Path) -> list[VoyageFiles]:
    root = Path(raw_root)
    voyages = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        voyage = VoyageFiles(
            root=path,
            bms_dir=path / "BMS",
            ems_dir=path / "EMS",
            fuel_cell_dir=path / "燃料电池系统",
            propulsion_dir=path / "推进系统",
        )
        if voyage.bms_dir.exists() and voyage.ems_dir.exists() and voyage.fuel_cell_dir.exists():
            voyages.append(voyage)
    return voyages


def load_processed_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    return pd.read_csv(csv_path, parse_dates=["timestamp"])


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
