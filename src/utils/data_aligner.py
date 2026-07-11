from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.data_loader import VoyageFiles


def _read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=columns, encoding="utf-8-sig")


def _build_series(path: Path, mapping: dict[str, str], transform_map: dict[str, callable] | None = None) -> pd.DataFrame:
    df = _read_csv(path, ["Time", *mapping.keys()]).rename(columns={"Time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    transform_map = transform_map or {}
    for source, target in mapping.items():
        if source in transform_map:
            df[target] = df[source].map(transform_map[source])
        else:
            df[target] = pd.to_numeric(df[source], errors="coerce")
    return df[["timestamp", *mapping.values()]].sort_values("timestamp")


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, tolerance_seconds: int = 30) -> pd.DataFrame:
    if right.empty:
        return left
    return pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )


def _sum_frames(frames: list[pd.DataFrame], value_column: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["timestamp", value_column])
    merged = pd.concat(frames, ignore_index=True)
    return merged.groupby("timestamp", as_index=False)[value_column].sum().sort_values("timestamp")


def _rename_value_column(frame: pd.DataFrame, source_column: str, target_column: str) -> pd.DataFrame:
    return frame.rename(columns={source_column: target_column})[["timestamp", target_column]]


def _clean_speed(raw: object) -> float:
    text = str(raw).replace("kn", "").strip()
    return float(text) if text else 0.0


def align_single_voyage(voyage: VoyageFiles, resample_seconds: int = 60) -> pd.DataFrame:
    left_battery = _build_series(
        next(voyage.bms_dir.glob("左电池系统BDM*.csv")),
        {
            "SOC(%)": "soc_left_pct",
            "总电压(V)": "battery_voltage_left_v",
            "总电流(A)": "battery_current_left_a",
        },
    )
    right_battery = _build_series(
        next(voyage.bms_dir.glob("右电池系统BDM*.csv")),
        {
            "SOC(%)": "soc_right_pct",
            "总电压(V)": "battery_voltage_right_v",
            "总电流(A)": "battery_current_right_a",
        },
    )

    fuel_cell_left = []
    fuel_cell_right = []
    for path in sorted(voyage.fuel_cell_dir.glob("*氢燃料电池#*.csv")):
        frame = _build_series(path, {"发电功率(kW)": "fuel_cell_power_kw"})
        if "左" in path.name:
            fuel_cell_left.append(frame)
        elif "右" in path.name:
            fuel_cell_right.append(frame)

    inverter_left = []
    inverter_right = []
    for path in sorted(voyage.ems_dir.glob("*逆变电源*.csv")):
        frame = _build_series(path, {"输出有功功率(kW)": "inverter_power_kw"})
        if "左" in path.name:
            inverter_left.append(frame)
        elif "右" in path.name:
            inverter_right.append(frame)

    speed_files = sorted(voyage.propulsion_dir.glob("*AIS航速*.csv"))
    speed_frame = pd.DataFrame(columns=["timestamp", "speed_knots"])
    if speed_files:
        speed_frame = _build_series(
            speed_files[0],
            {"航速(节)": "speed_knots"},
            transform_map={"航速(节)": _clean_speed},
        )

    df = _merge_asof(left_battery, right_battery)
    fuel_cell_left = [_rename_value_column(frame, "fuel_cell_power_kw", "fuel_cell_power_left_kw") for frame in fuel_cell_left]
    fuel_cell_right = [_rename_value_column(frame, "fuel_cell_power_kw", "fuel_cell_power_right_kw") for frame in fuel_cell_right]
    inverter_left = [_rename_value_column(frame, "inverter_power_kw", "load_left_kw") for frame in inverter_left]
    inverter_right = [_rename_value_column(frame, "inverter_power_kw", "load_right_kw") for frame in inverter_right]

    df = _merge_asof(df, _sum_frames(fuel_cell_left, "fuel_cell_power_left_kw"))
    df = _merge_asof(df, _sum_frames(fuel_cell_right, "fuel_cell_power_right_kw"))
    df = _merge_asof(df, _sum_frames(inverter_left, "load_left_kw"))
    df = _merge_asof(df, _sum_frames(inverter_right, "load_right_kw"))
    df = _merge_asof(df, speed_frame)

    for column in [
        "fuel_cell_power_left_kw",
        "fuel_cell_power_right_kw",
        "load_left_kw",
        "load_right_kw",
        "speed_knots",
    ]:
        if column not in df:
            df[column] = 0.0
        df[column] = df[column].fillna(0.0)

    df["battery_power_left_kw"] = -(df["battery_voltage_left_v"] * df["battery_current_left_a"]) / 1000.0
    df["battery_power_right_kw"] = -(df["battery_voltage_right_v"] * df["battery_current_right_a"]) / 1000.0
    df["fuel_cell_power_total_kw"] = df["fuel_cell_power_left_kw"] + df["fuel_cell_power_right_kw"]
    df["battery_power_total_kw"] = df["battery_power_left_kw"] + df["battery_power_right_kw"]
    df["load_total_kw"] = df["load_left_kw"] + df["load_right_kw"]
    df["soc_mean"] = (df["soc_left_pct"] + df["soc_right_pct"]) / 200.0
    df["soc_left"] = df["soc_left_pct"] / 100.0
    df["soc_right"] = df["soc_right_pct"] / 100.0
    df["voyage_name"] = voyage.root.name

    numeric_columns = [column for column in df.columns if column not in {"timestamp", "voyage_name"}]
    aligned = (
        df.set_index("timestamp")[numeric_columns]
        .resample(f"{resample_seconds}s")
        .mean()
        .interpolate(method="time", limit_direction="both")
        .reset_index()
    )
    aligned["voyage_name"] = voyage.root.name
    aligned["sample_time_seconds"] = float(resample_seconds)
    return aligned
