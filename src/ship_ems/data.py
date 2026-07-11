from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


FUEL_CELL_PATTERN = "*氢燃料电池#*.csv"
INVERTER_PATTERN = "*逆变电源*.csv"
DC_PATTERN = "*配*DC*.csv"
SPEED_PATTERN = "*AIS航速*.csv"


def _read_csv(path: Path, usecols: Iterable[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=list(usecols), encoding="utf-8-sig")


def _clean_speed(value: object) -> float:
    text = str(value).strip().lower().replace("kn", "").strip()
    return float(text) if text else 0.0


def _series_from_file(path: Path, value_columns: dict[str, str], transforms: dict[str, callable] | None = None) -> pd.DataFrame:
    df = _read_csv(path, ["Time", *value_columns.keys()]).rename(columns={"Time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    transforms = transforms or {}
    for original, renamed in value_columns.items():
        if original in transforms:
            df[renamed] = df[original].map(transforms[original])
        else:
            df[renamed] = pd.to_numeric(df[original], errors="coerce")
    return df[["timestamp", *value_columns.values()]].dropna(subset=["timestamp"]).sort_values("timestamp")


def _sum_series(frames: list[pd.DataFrame], column_name: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["timestamp", column_name])
    combined = pd.concat(frames, ignore_index=True)
    return combined.groupby("timestamp", as_index=False)[column_name].sum().sort_values("timestamp")


def _merge_frames(base: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming.empty:
        return base
    return pd.merge_asof(
        base.sort_values("timestamp"),
        incoming.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=30),
    )


def _load_battery_side(side_name: str, bms_dir: Path) -> pd.DataFrame:
    files = sorted(bms_dir.glob(f"{side_name}电池系统BDM*.csv"))
    if not files:
        raise FileNotFoundError(f"Cannot find battery system file for {side_name} in {bms_dir}")
    df = _series_from_file(
        files[0],
        {
            "SOC(%)": f"soc_{side_name}",
            "总电压(V)": f"voltage_{side_name}",
            "总电流(A)": f"current_{side_name}",
            "剩余瓦时(kWh)": f"energy_kwh_{side_name}",
        },
    )
    df[f"battery_power_{side_name}"] = -(df[f"voltage_{side_name}"] * df[f"current_{side_name}"]) / 1000.0
    return df


def _load_fuel_cells(fc_dir: Path) -> pd.DataFrame:
    left_frames = []
    right_frames = []
    for path in sorted(fc_dir.glob(FUEL_CELL_PATTERN)):
        if "左" in path.name:
            left_frames.append(_series_from_file(path, {"发电功率(kW)": "fuel_cell_power_left"}))
        elif "右" in path.name:
            right_frames.append(_series_from_file(path, {"发电功率(kW)": "fuel_cell_power_right"}))
    left = _sum_series(left_frames, "fuel_cell_power_left")
    right = _sum_series(right_frames, "fuel_cell_power_right")
    return pd.merge_asof(left, right, on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=30)).fillna(0.0)


def _load_inverter_power(ems_dir: Path) -> pd.DataFrame:
    left_frames = []
    right_frames = []
    for path in sorted(ems_dir.glob(INVERTER_PATTERN)):
        if "左" in path.name:
            left_frames.append(_series_from_file(path, {"输出有功功率(kW)": "inverter_power_left"}))
        elif "右" in path.name:
            right_frames.append(_series_from_file(path, {"输出有功功率(kW)": "inverter_power_right"}))
    left = _sum_series(left_frames, "inverter_power_left")
    right = _sum_series(right_frames, "inverter_power_right")
    return pd.merge_asof(left, right, on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=30)).fillna(0.0)


def _load_dc_power(ems_dir: Path) -> pd.DataFrame:
    left_frames = []
    right_frames = []
    for path in sorted(ems_dir.glob(DC_PATTERN)):
        frame = _series_from_file(path, {"充放电流(A)": "dc_current", "电池侧直流电压(V)": "dc_voltage"})
        if "左" in path.name:
            left_frames.append(frame)
        elif "右" in path.name:
            right_frames.append(frame)

    def finalize(frames: list[pd.DataFrame], output_name: str) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["timestamp", output_name])
        combined = pd.concat(frames, ignore_index=True)
        combined[output_name] = (combined["dc_current"] * combined["dc_voltage"]).abs() / 1000.0
        return combined.groupby("timestamp", as_index=False)[output_name].sum().sort_values("timestamp")

    left = finalize(left_frames, "dc_power_left")
    right = finalize(right_frames, "dc_power_right")
    return pd.merge_asof(left, right, on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=30)).fillna(0.0)


def _load_speed(nav_dir: Path) -> pd.DataFrame:
    files = sorted(nav_dir.glob(SPEED_PATTERN))
    if not files:
        return pd.DataFrame(columns=["timestamp", "speed_knots"])
    return _series_from_file(files[0], {"航速(节)": "speed_knots"}, transforms={"航速(节)": _clean_speed})


def load_single_voyage(voyage_dir: str | Path, resample_seconds: int = 60) -> pd.DataFrame:
    voyage_path = Path(voyage_dir)
    df = _merge_frames(_load_battery_side("左", voyage_path / "BMS"), _load_battery_side("右", voyage_path / "BMS"))
    for frame in [
        _load_fuel_cells(voyage_path / "燃料电池系统"),
        _load_inverter_power(voyage_path / "EMS"),
        _load_dc_power(voyage_path / "EMS"),
        _load_speed(voyage_path / "推进系统"),
    ]:
        df = _merge_frames(df, frame)

    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df["voyage_name"] = voyage_path.name
    for name in [
        "fuel_cell_power_left",
        "fuel_cell_power_right",
        "inverter_power_left",
        "inverter_power_right",
        "dc_power_left",
        "dc_power_right",
    ]:
        df[name] = df[name].fillna(0.0)
    df["speed_knots"] = df["speed_knots"].ffill().fillna(0.0)
    df["load_power_left"] = df["inverter_power_left"] + df["dc_power_left"]
    df["load_power_right"] = df["inverter_power_right"] + df["dc_power_right"]
    df["load_power_total"] = df["load_power_left"] + df["load_power_right"]
    df["fuel_cell_power_total"] = df["fuel_cell_power_left"] + df["fuel_cell_power_right"]
    df["battery_power_total"] = df["battery_power_左"] + df["battery_power_右"]
    df["soc_mean"] = (df["soc_左"] + df["soc_右"]) / 200.0
    df["soc_left"] = df["soc_左"] / 100.0
    df["soc_right"] = df["soc_右"] / 100.0

    numeric_cols = [c for c in df.columns if c not in {"timestamp", "voyage_name"}]
    resampled = (
        df.set_index("timestamp")[numeric_cols]
        .resample(f"{resample_seconds}s")
        .mean()
        .interpolate(method="time", limit_direction="both")
        .reset_index()
    )
    resampled["voyage_name"] = voyage_path.name
    resampled["time_step_seconds"] = float(resample_seconds)
    return resampled


def build_master_dataset(data_root: str | Path, resample_seconds: int = 60) -> pd.DataFrame:
    root = Path(data_root)
    datasets = [load_single_voyage(path, resample_seconds=resample_seconds) for path in sorted(root.iterdir()) if path.is_dir()]
    if not datasets:
        raise FileNotFoundError(f"No voyage folders found in {root}")
    df = pd.concat(datasets, ignore_index=True)
    df["next_load_power_total"] = df.groupby("voyage_name")["load_power_total"].shift(-1)
    df["next_speed_knots"] = df.groupby("voyage_name")["speed_knots"].shift(-1)
    df["load_rolling_mean_5"] = (
        df.groupby("voyage_name")["load_power_total"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["speed_rolling_mean_5"] = (
        df.groupby("voyage_name")["speed_knots"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    return df.dropna(subset=["next_load_power_total"]).reset_index(drop=True)


def save_dataset(dataset: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def train_eval_split(dataset: pd.DataFrame, eval_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    voyage_names = sorted(dataset["voyage_name"].unique())
    split_index = max(1, int(round(len(voyage_names) * (1 - eval_ratio))))
    train_voyages = set(voyage_names[:split_index])
    train_df = dataset[dataset["voyage_name"].isin(train_voyages)].reset_index(drop=True)
    eval_df = dataset[~dataset["voyage_name"].isin(train_voyages)].reset_index(drop=True)
    if eval_df.empty:
        eval_df = train_df.copy()
    return train_df, eval_df


@dataclass
class ShipDatasetSummary:
    rows: int
    voyages: int
    load_max_kw: float
    fuel_cell_max_kw: float
    battery_power_abs_max_kw: float


def summarize_dataset(dataset: pd.DataFrame) -> ShipDatasetSummary:
    return ShipDatasetSummary(
        rows=len(dataset),
        voyages=dataset["voyage_name"].nunique(),
        load_max_kw=float(dataset["load_power_total"].max()),
        fuel_cell_max_kw=float(dataset["fuel_cell_power_total"].max()),
        battery_power_abs_max_kw=float(dataset["battery_power_total"].abs().max()),
    )
