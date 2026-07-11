"""Build the 66-voyage energy-side total-load dataset and chronological 7:2:1 split."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJ / "total_load_excels"
DEFAULT_OUTPUT_DIR = PROJ / "outputs" / "total_load_dataset_build"
DEFAULT_CONFIG_DIR = PROJ / "outputs" / "config"
AIS_ROOT_DIRNAME = "\u6c22\u821f\u4e00\u53f7"
AIS_PROPULSION_DIRNAME = "\u63a8\u8fdb\u7cfb\u7edf"
AIS_SPEED_COLUMN = "\u822a\u901f(\u8282)"
DEFAULT_AIS_ROOT = Path.home() / "OneDrive" / "Desktop" / AIS_ROOT_DIRNAME
LOAD_DEFINITION = "fuel_cell_total_kw + battery_total_kw"
LOAD_SCOPE = "energy_side_equivalent_total_load"
TARGET_LOAD_COLUMN = "total_load_fc_plus_batt_kw"
DT_SECONDS = 30.0
REQUIRED_COLUMNS = ["timestamp", "fuel_cell_total_kw", "battery_total_kw", TARGET_LOAD_COLUMN]
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


@dataclass
class VoyageRecord:
    path: Path
    frame: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    speed_source_file: str | None = None
    speed_coverage_ratio: float | None = None
    speed_missing_after_fill_count: int | None = None
    speed_max_lag_s: float | None = None


def _col_ref_to_index(ref: str) -> int:
    letters = re.sub(r"[^A-Z]", "", ref.upper())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _text_content(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext())


def _read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [_text_content(si) for si in root.findall("x:si", NS)]


def read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    """Read simple xlsx sheets using the standard library.

    The generated total-load workbooks contain one plain worksheet with scalar
    cells, so a full Excel engine is unnecessary for this build step.
    """
    with ZipFile(path) as zf:
        shared_strings = _read_shared_strings(zf)
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in zf.namelist():
            raise ValueError(f"{path} does not contain {sheet_path}")
        root = ET.fromstring(zf.read(sheet_path))
    rows: list[list[Any]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        values: dict[int, Any] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "")
            if not ref:
                continue
            idx = _col_ref_to_index(ref)
            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("x:v", NS)
            if cell_type == "s":
                raw = _text_content(value_node)
                value = shared_strings[int(raw)] if raw else ""
            elif cell_type == "inlineStr":
                value = _text_content(cell.find("x:is", NS))
            else:
                value = _text_content(value_node)
            values[idx] = value
        if values:
            width = max(values) + 1
            rows.append([values.get(i, "") for i in range(width)])
    if not rows:
        raise ValueError(f"{path} has no worksheet rows")
    header = [str(value).strip() for value in rows[0]]
    data = []
    for raw_row in rows[1:]:
        padded = raw_row + [""] * (len(header) - len(raw_row))
        data.append(padded[: len(header)])
    return pd.DataFrame(data, columns=header)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _speed_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace("kn", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _read_ais_speed_csv(voyage_excel: Path, ais_root: Path) -> tuple[pd.DataFrame, Path]:
    speed_dir = Path(ais_root) / voyage_excel.stem / AIS_PROPULSION_DIRNAME
    candidates = sorted(speed_dir.glob("AIS*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No AIS speed CSV found for {voyage_excel.name} under {speed_dir}")
    speed_path = candidates[0]
    raw = pd.read_csv(speed_path)
    required = ["Time", AIS_SPEED_COLUMN]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"{speed_path} missing AIS speed columns: {missing}")
    speed = pd.DataFrame(
        {
            "ais_time": pd.to_datetime(raw["Time"], errors="coerce"),
            "speed_knots": _speed_numeric(raw[AIS_SPEED_COLUMN]),
        }
    )
    speed = speed.dropna(subset=["ais_time"]).sort_values("ais_time").drop_duplicates("ais_time").reset_index(drop=True)
    if speed.empty:
        raise ValueError(f"{speed_path} has no valid AIS speed rows")
    return speed, speed_path


def _merge_ais_speed(
    frame: pd.DataFrame,
    *,
    voyage_excel: Path,
    ais_root: Path,
    tolerance_s: float,
) -> tuple[pd.DataFrame, str, float, int, float]:
    speed, speed_path = _read_ais_speed_csv(voyage_excel, ais_root)
    base = frame.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        base,
        speed,
        left_on="timestamp",
        right_on="ais_time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=float(tolerance_s)),
    )
    lag = (merged["timestamp"] - merged["ais_time"]).dt.total_seconds()
    matched = merged["speed_knots"].notna()
    coverage = float(matched.mean()) if len(merged) else 0.0
    max_lag = float(lag[matched].max()) if matched.any() else float("nan")
    merged["speed_match_lag_s"] = lag
    merged["speed_source_file"] = speed_path.name
    merged["speed_knots"] = merged["speed_knots"].ffill().fillna(0.0)
    missing_after_fill = int(merged["speed_knots"].isna().sum())
    return merged.drop(columns=["ais_time"]), speed_path.name, coverage, missing_after_fill, max_lag


def _prepare_voyage_frame(
    path: Path,
    *,
    include_ais_speed: bool = False,
    include_existing_speed: bool = False,
    ais_root: Path = DEFAULT_AIS_ROOT,
    ais_tolerance_s: float = 90.0,
) -> VoyageRecord:
    raw = read_xlsx_first_sheet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "file_name": path.name,
            "timestamp": pd.to_datetime(raw["timestamp"], errors="coerce"),
            "fuel_cell_total_kw": _numeric(raw["fuel_cell_total_kw"]),
            "battery_total_kw": _numeric(raw["battery_total_kw"]),
            "load_total_kw": _numeric(raw[TARGET_LOAD_COLUMN]),
        }
    )
    if "propulsion_load_kw" in raw.columns:
        out["propulsion_load_kw"] = _numeric(raw["propulsion_load_kw"])
    elif "propulsion_inverter_total_kw" in raw.columns:
        out["propulsion_load_kw"] = _numeric(raw["propulsion_inverter_total_kw"])
    else:
        out["propulsion_load_kw"] = np.nan
    out["load_gap_kw"] = out["load_total_kw"] - out["propulsion_load_kw"]
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{path.name} has no valid timestamp rows")
    identity_error = (out["load_total_kw"] - (out["fuel_cell_total_kw"] + out["battery_total_kw"])).abs().max()
    if pd.notna(identity_error) and float(identity_error) > 1e-3:
        raise ValueError(f"{path.name} load identity error {identity_error:.6f} kW exceeds tolerance")
    speed_source_file = None
    speed_coverage_ratio = None
    speed_missing_after_fill_count = None
    speed_max_lag_s = None
    if include_existing_speed and "speed_knots" in raw.columns and not include_ais_speed:
        out["speed_knots"] = _numeric(raw["speed_knots"]).ffill().fillna(0.0)
        out["speed_match_lag_s"] = 0.0
        out["speed_source_file"] = "existing_input_column:speed_knots"
        speed_source_file = "existing_input_column:speed_knots"
        speed_coverage_ratio = float(out["speed_knots"].notna().mean()) if len(out) else 0.0
        speed_missing_after_fill_count = int(out["speed_knots"].isna().sum())
        speed_max_lag_s = 0.0
    if include_ais_speed:
        out, speed_source_file, speed_coverage_ratio, speed_missing_after_fill_count, speed_max_lag_s = _merge_ais_speed(
            out,
            voyage_excel=path,
            ais_root=Path(ais_root),
            tolerance_s=float(ais_tolerance_s),
        )
    return VoyageRecord(
        path=path,
        frame=out,
        start_time=out["timestamp"].iloc[0],
        end_time=out["timestamp"].iloc[-1],
        speed_source_file=speed_source_file,
        speed_coverage_ratio=speed_coverage_ratio,
        speed_missing_after_fill_count=speed_missing_after_fill_count,
        speed_max_lag_s=speed_max_lag_s,
    )


def _timestamp_gap_count(ts: pd.Series, expected_dt_seconds: float = DT_SECONDS) -> int:
    diffs = pd.to_datetime(ts, errors="coerce").sort_values().diff().dt.total_seconds().dropna()
    tolerance = max(1e-6, min(1.5, float(expected_dt_seconds) * 0.1))
    return int(np.sum(np.abs(diffs.to_numpy(dtype=float) - float(expected_dt_seconds)) > tolerance))


def _summary_row(record: VoyageRecord, voyage_id: str, *, sample_interval_seconds: float = DT_SECONDS) -> dict[str, Any]:
    df = record.frame
    required = ["timestamp", "fuel_cell_total_kw", "battery_total_kw", "load_total_kw"]
    duration_h = (record.end_time - record.start_time).total_seconds() / 3600.0 if len(df) > 1 else 0.0
    identity_error = (df["load_total_kw"] - (df["fuel_cell_total_kw"] + df["battery_total_kw"])).abs().max()
    row = {
        "voyage_id": voyage_id,
        "file_name": record.path.name,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "duration_h": float(duration_h),
        "num_samples": int(len(df)),
        "mean_load_kw": float(df["load_total_kw"].mean()),
        "max_load_kw": float(df["load_total_kw"].max()),
        "min_load_kw": float(df["load_total_kw"].min()),
        "negative_load_count": int((df["load_total_kw"] < -1e-9).sum()),
        "missing_value_count": int(df[required].isna().sum().sum()),
        "timestamp_gap_count": _timestamp_gap_count(df["timestamp"], float(sample_interval_seconds)),
        "timestamp_monotonic_increasing": bool(df["timestamp"].is_monotonic_increasing),
        "load_identity_max_abs_error_kw": float(identity_error) if pd.notna(identity_error) else np.nan,
        "extreme_load_gt_1000kw_count": int((df["load_total_kw"].abs() > 1000.0).sum()),
        "load_definition": LOAD_DEFINITION,
        "load_scope": LOAD_SCOPE,
        "sample_interval_seconds": float(sample_interval_seconds),
    }
    if "speed_knots" in df.columns:
        row.update(
            {
                "speed_source_file": record.speed_source_file,
                "speed_coverage_ratio": float(record.speed_coverage_ratio) if record.speed_coverage_ratio is not None else np.nan,
                "speed_missing_after_fill_count": int(record.speed_missing_after_fill_count or 0),
                "speed_max_lag_s": float(record.speed_max_lag_s) if record.speed_max_lag_s is not None else np.nan,
                "speed_mean_knots": float(df["speed_knots"].mean()),
                "speed_max_knots": float(df["speed_knots"].max()),
            }
        )
    return row


def _split_records(records: list[VoyageRecord], split_counts: tuple[int, int, int]) -> dict[str, list[str]]:
    train_n, val_n, test_n = split_counts
    if train_n + val_n + test_n != len(records):
        raise ValueError(f"split_counts={split_counts} do not sum to {len(records)} voyages")
    voyage_ids = [f"voyage_{idx:03d}" for idx in range(1, len(records) + 1)]
    return {
        "train_voyages": voyage_ids[:train_n],
        "validation_voyages": voyage_ids[train_n : train_n + val_n],
        "test_voyages": voyage_ids[train_n + val_n :],
    }


def _write_split_files(
    *,
    records: list[VoyageRecord],
    split: dict[str, list[str]],
    config_dir: Path,
    include_ais_speed: bool = False,
    include_existing_speed: bool = False,
    sample_interval_seconds: float = DT_SECONDS,
    split_json_name: str | None = None,
    split_txt_name: str | None = None,
) -> None:
    voyage_ids = [f"voyage_{idx:03d}" for idx in range(1, len(records) + 1)]
    file_by_id = {voyage_id: record.path.name for voyage_id, record in zip(voyage_ids, records)}
    start_by_id = {voyage_id: record.start_time.isoformat() for voyage_id, record in zip(voyage_ids, records)}
    payload: dict[str, Any] = {
        **split,
        "train": split["train_voyages"],
        "validation": split["validation_voyages"],
        "test": split["test_voyages"],
        "file_by_voyage": file_by_id,
        "start_time_by_voyage": start_by_id,
        "total_voyages": len(records),
        "split_basis": "chronological_by_voyage_start_time",
        "target_load": TARGET_LOAD_COLUMN,
        "load_definition": LOAD_DEFINITION,
        "load_scope": LOAD_SCOPE,
        "no_window_crossing_voyage_boundary": True,
        "scaler_fit_scope": "train_voyages_only",
        "sample_interval_seconds": float(sample_interval_seconds),
    }
    if include_ais_speed or include_existing_speed:
        payload["auxiliary_features"] = ["speed_knots", "delta_speed"]
        payload["speed_source"] = (
            "AIS speed CSV merged by backward-asof timestamp alignment"
            if include_ais_speed
            else "existing speed_knots column from input workbook"
        )
    config_dir.mkdir(parents=True, exist_ok=True)
    if split_json_name is None:
        split_json_name = "voyage_split_total_load_speed_721.json" if include_ais_speed else "voyage_split_total_load_721.json"
    if split_txt_name is None:
        split_txt_name = "SPLIT_TOTAL_LOAD_SPEED_721.txt" if include_ais_speed else "SPLIT_TOTAL_LOAD_721.txt"
    (config_dir / split_json_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"总航段数：{len(records)}",
        f"Train：{len(split['train_voyages'])}",
        f"Val：{len(split['validation_voyages'])}",
        f"Test：{len(split['test_voyages'])}",
        "划分依据：按航段起始时间排序的 7:2:1 航段级划分",
        f"目标负荷：{TARGET_LOAD_COLUMN}",
        f"load_definition：{LOAD_DEFINITION}",
        f"load_scope：{LOAD_SCOPE}",
        "LSTM 窗口：不跨航段边界",
        "归一化器：仅使用 Train 航段拟合",
    ]
    if include_ais_speed or include_existing_speed:
        lines.append("Auxiliary feature: AIS speed_knots, delta_speed")
    lines.append(f"sample_interval_seconds: {float(sample_interval_seconds)}")
    (config_dir / split_txt_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_parquet_if_available(df: pd.DataFrame, path: Path) -> str:
    try:
        df.to_parquet(path, index=False)
        return "written"
    except Exception as exc:
        marker = path.with_suffix(path.suffix + ".unavailable.txt")
        marker.write_text(f"Parquet was not written because no parquet engine is available: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        return f"unavailable: {type(exc).__name__}: {exc}"


def build_dataset(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    expected_count: int = 66,
    split_counts: tuple[int, int, int] = (46, 13, 7),
    include_ais_speed: bool = False,
    include_existing_speed: bool = False,
    ais_root: Path = DEFAULT_AIS_ROOT,
    ais_tolerance_s: float = 90.0,
    sample_interval_seconds: float = DT_SECONDS,
    split_json_name: str | None = None,
    split_txt_name: str | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config_dir = Path(config_dir)
    excel_paths = sorted(input_dir.glob("*.xlsx"))
    if expected_count is not None and len(excel_paths) != int(expected_count):
        raise ValueError(f"Expected {expected_count} Excel files in {input_dir}, found {len(excel_paths)}")
    records = [
        _prepare_voyage_frame(
            path,
            include_ais_speed=include_ais_speed,
            include_existing_speed=include_existing_speed,
            ais_root=Path(ais_root),
            ais_tolerance_s=float(ais_tolerance_s),
        )
        for path in excel_paths
    ]
    records.sort(key=lambda item: (item.start_time, item.path.name))

    summary_rows = []
    segment_frames = []
    for idx, record in enumerate(records, start=1):
        voyage_id = f"voyage_{idx:03d}"
        frame = record.frame.copy()
        frame.insert(0, "voyage_name", voyage_id)
        frame.insert(0, "voyage_id", voyage_id)
        frame["load_definition"] = LOAD_DEFINITION
        frame["load_scope"] = LOAD_SCOPE
        segment_frames.append(frame)
        summary_rows.append(_summary_row(record, voyage_id, sample_interval_seconds=float(sample_interval_seconds)))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary_total_load_66.csv", index=False, encoding="utf-8-sig")
    combined = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    combined_columns = [
        "voyage_id",
        "voyage_name",
        "file_name",
        "timestamp",
        "load_total_kw",
        "fuel_cell_total_kw",
        "battery_total_kw",
        "propulsion_load_kw",
        "load_gap_kw",
    ]
    has_speed_column = any("speed_knots" in frame.columns for frame in segment_frames)
    if has_speed_column:
        combined_columns.extend(["speed_knots", "speed_match_lag_s", "speed_source_file"])
    combined_columns.extend(["load_definition", "load_scope"])
    combined = combined[combined_columns]
    combined.to_csv(output_dir / "total_load_66_segments.csv", index=False, encoding="utf-8-sig")
    parquet_status = _write_parquet_if_available(combined, output_dir / "total_load_66_segments.parquet")

    split = _split_records(records, split_counts)
    _write_split_files(
        records=records,
        split=split,
        config_dir=config_dir,
        include_ais_speed=include_ais_speed,
        include_existing_speed=include_existing_speed,
        sample_interval_seconds=float(sample_interval_seconds),
        split_json_name=split_json_name,
        split_txt_name=split_txt_name,
    )
    if split_json_name is None:
        split_json_name = "voyage_split_total_load_speed_721.json" if include_ais_speed else "voyage_split_total_load_721.json"
    if split_txt_name is None:
        split_txt_name = "SPLIT_TOTAL_LOAD_SPEED_721.txt" if include_ais_speed else "SPLIT_TOTAL_LOAD_721.txt"
    return {
        "excel_count": len(excel_paths),
        "processed_voyages": len(records),
        "train_count": len(split["train_voyages"]),
        "validation_count": len(split["validation_voyages"]),
        "test_count": len(split["test_voyages"]),
        "summary_csv": str((output_dir / "summary_total_load_66.csv").resolve()),
        "segments_csv": str((output_dir / "total_load_66_segments.csv").resolve()),
        "segments_parquet": str((output_dir / "total_load_66_segments.parquet").resolve()),
        "parquet_status": parquet_status,
        "split_json": str((config_dir / split_json_name).resolve()),
        "split_txt": str((config_dir / split_txt_name).resolve()),
        "include_ais_speed": bool(include_ais_speed),
        "include_existing_speed": bool(include_existing_speed),
        "sample_interval_seconds": float(sample_interval_seconds),
        "ais_root": str(Path(ais_root).resolve()) if include_ais_speed else None,
        "ais_tolerance_s": float(ais_tolerance_s) if include_ais_speed else None,
        "load_definition": LOAD_DEFINITION,
        "load_scope": LOAD_SCOPE,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build 66-voyage total-load dataset and chronological 7:2:1 split.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config_dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--expected_count", type=int, default=66)
    parser.add_argument("--train_count", type=int, default=46)
    parser.add_argument("--val_count", type=int, default=13)
    parser.add_argument("--test_count", type=int, default=7)
    parser.add_argument("--include_ais_speed", action="store_true")
    parser.add_argument("--include_existing_speed", action="store_true")
    parser.add_argument("--ais_root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--ais_tolerance_s", type=float, default=90.0)
    parser.add_argument("--sample_interval_seconds", type=float, default=DT_SECONDS)
    parser.add_argument("--split_json_name", default=None)
    parser.add_argument("--split_txt_name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_dir=args.config_dir,
        expected_count=args.expected_count,
        split_counts=(args.train_count, args.val_count, args.test_count),
        include_ais_speed=bool(args.include_ais_speed),
        include_existing_speed=bool(args.include_existing_speed),
        ais_root=args.ais_root,
        ais_tolerance_s=float(args.ais_tolerance_s),
        sample_interval_seconds=float(args.sample_interval_seconds),
        split_json_name=args.split_json_name,
        split_txt_name=args.split_txt_name,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()
