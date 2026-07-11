"""DEPRECATED: non-causal linear 30 s to 1 s reconstruction.

DEPRECATED: linear interpolation from 30 s to 1 s creates non-causal
reconstructed data and must not be used as valid online forecasting evidence.
This file is retained only as an archived implementation reference.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from build_total_load_dataset_721 import read_xlsx_first_sheet


PROJ = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJ / "total_load_excels"
DEFAULT_OUTPUT_DIR = PROJ / "total_load_excels_1s"
DEFAULT_SUMMARY_DIR = PROJ / "outputs" / "total_load_1s_build"
DEFAULT_AIS_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
AIS_PROPULSION_DIRNAME = "推进系统"
AIS_SPEED_COLUMN = "航速(节)"
POWER_COLUMNS = ["fuel_cell_total_kw", "battery_total_kw", "total_load_fc_plus_batt_kw"]
OUTPUT_COLUMNS = ["timestamp", "time_s", "time_h", *POWER_COLUMNS, "speed_knots"]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _speed_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("kn", "", regex=False).str.strip(), errors="coerce")


def resolve_ais_speed_file(ais_root: Path, voyage_name: str) -> Path:
    speed_dir = Path(ais_root) / voyage_name / AIS_PROPULSION_DIRNAME
    candidates = sorted(speed_dir.glob("AIS*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No AIS speed CSV found for voyage {voyage_name!r} under {speed_dir}")
    return candidates[0]


def read_ais_speed(ais_root: Path, voyage_name: str) -> tuple[pd.DataFrame, Path]:
    speed_path = resolve_ais_speed_file(Path(ais_root), voyage_name)
    raw = pd.read_csv(speed_path, encoding="utf-8-sig")
    required = ["Time", AIS_SPEED_COLUMN]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"{speed_path} missing AIS speed columns: {missing}")
    speed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["Time"], errors="coerce"),
            "speed_knots": _speed_numeric(raw[AIS_SPEED_COLUMN]),
        }
    )
    speed = speed.dropna(subset=["timestamp", "speed_knots"]).sort_values("timestamp")
    speed = speed.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    if speed.empty:
        raise ValueError(f"{speed_path} has no valid AIS speed rows")
    return speed, speed_path


def _seconds_since_start(timestamps: pd.Series, start: pd.Timestamp) -> np.ndarray:
    return (pd.to_datetime(timestamps, errors="coerce") - start).dt.total_seconds().to_numpy(dtype=float)


def _interpolate_series(source_t_s: np.ndarray, source_values: np.ndarray, target_t_s: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_t_s) & np.isfinite(source_values)
    if not valid.any():
        return np.zeros_like(target_t_s, dtype=float)
    x = source_t_s[valid]
    y = source_values[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]
    if len(unique_x) == 1:
        return np.full_like(target_t_s, float(unique_y[0]), dtype=float)
    return np.interp(target_t_s, unique_x, unique_y, left=float(unique_y[0]), right=float(unique_y[-1]))


def build_one_voyage_1s(
    source: pd.DataFrame,
    *,
    voyage_name: str,
    ais_root: Path,
    source_file_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = [col for col in ["timestamp", *POWER_COLUMNS] if col not in source.columns]
    if missing:
        raise ValueError(f"{source_file_name} missing required columns: {missing}")

    work = pd.DataFrame({"timestamp": pd.to_datetime(source["timestamp"], errors="coerce")})
    for col in POWER_COLUMNS:
        work[col] = _numeric(source[col])
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="first")
    if work.empty:
        raise ValueError(f"{source_file_name} has no valid timestamp rows")

    for col in POWER_COLUMNS:
        if work[col].isna().all():
            raise ValueError(f"{source_file_name} column {col} has no numeric values")
        work[col] = work[col].interpolate(method="linear", limit_direction="both")

    source_identity_error = (work["total_load_fc_plus_batt_kw"] - work["fuel_cell_total_kw"] - work["battery_total_kw"]).abs().max()
    if pd.notna(source_identity_error) and float(source_identity_error) > 1e-3:
        raise ValueError(f"{source_file_name} load identity error {source_identity_error:.6f} kW exceeds tolerance")

    start = pd.Timestamp(work["timestamp"].iloc[0]).floor("s")
    end = pd.Timestamp(work["timestamp"].iloc[-1]).floor("s")
    if end < start:
        raise ValueError(f"{source_file_name} invalid timestamp range")
    target_timestamps = pd.date_range(start=start, end=end, freq="1s")
    target_t_s = (target_timestamps - start).total_seconds().to_numpy(dtype=float)
    source_t_s = _seconds_since_start(work["timestamp"], start)

    out = pd.DataFrame({"timestamp": target_timestamps})
    out["time_s"] = target_t_s
    out["time_h"] = target_t_s / 3600.0
    out["fuel_cell_total_kw"] = _interpolate_series(
        source_t_s, work["fuel_cell_total_kw"].to_numpy(dtype=float), target_t_s
    )
    out["battery_total_kw"] = _interpolate_series(source_t_s, work["battery_total_kw"].to_numpy(dtype=float), target_t_s)
    out["total_load_fc_plus_batt_kw"] = out["fuel_cell_total_kw"] + out["battery_total_kw"]

    speed, speed_path = read_ais_speed(Path(ais_root), voyage_name)
    speed_t_s = _seconds_since_start(speed["timestamp"], start)
    out["speed_knots"] = _interpolate_series(speed_t_s, speed["speed_knots"].to_numpy(dtype=float), target_t_s)

    out = out[OUTPUT_COLUMNS]
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    identity_error = (out["total_load_fc_plus_batt_kw"] - out["fuel_cell_total_kw"] - out["battery_total_kw"]).abs().max()
    diffs = pd.to_datetime(out["timestamp"]).diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    timestamp_gap_count = int(np.sum(np.abs(diffs - 1.0) > 1e-9))
    meta = {
        "file_name": source_file_name,
        "voyage_name": voyage_name,
        "start_time": str(out["timestamp"].iloc[0]),
        "end_time": str(out["timestamp"].iloc[-1]),
        "num_samples_1s": int(len(out)),
        "duration_s": float(target_t_s[-1]) if len(target_t_s) else 0.0,
        "source_samples": int(len(work)),
        "ais_source_file": speed_path.name,
        "ais_samples": int(len(speed)),
        "timestamp_gap_count": timestamp_gap_count,
        "load_identity_max_abs_error_kw": float(identity_error) if pd.notna(identity_error) else math.nan,
        "negative_load_count": int((out["total_load_fc_plus_batt_kw"] < -1e-9).sum()),
        "extreme_load_gt_1000kw_count": int((out["total_load_fc_plus_batt_kw"].abs() > 1000.0).sum()),
        "speed_missing_count": int(out["speed_knots"].isna().sum()),
        "speed_min_knots": float(out["speed_knots"].min()),
        "speed_max_knots": float(out["speed_knots"].max()),
        "mean_load_kw": float(out["total_load_fc_plus_batt_kw"].mean()),
        "max_load_kw": float(out["total_load_fc_plus_batt_kw"].max()),
        "min_load_kw": float(out["total_load_fc_plus_batt_kw"].min()),
    }
    return out, meta


def _excel_col_name(index_1_based: int) -> str:
    name = ""
    idx = int(index_1_based)
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(value: Any, cell_ref: str) -> str:
    if value is None:
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return f'<c r="{cell_ref}"/>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{float(value):.12g}</v></c>'
    text = escape(str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_xlsx(df: pd.DataFrame, path: Path, *, sheet_name: str = "Sheet1") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_xml: list[str] = []
    headers = list(df.columns)
    header_cells = [_cell_xml(col, f"{_excel_col_name(i + 1)}1") for i, col in enumerate(headers)]
    rows_xml.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_idx, row in enumerate(df.itertuples(index=False, name=None), start=2):
        cells = [_cell_xml(value, f"{_excel_col_name(col_idx + 1)}{row_idx}") for col_idx, value in enumerate(row)]
        rows_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')
    dimension = f"A1:{_excel_col_name(len(headers))}{len(df) + 1}"
    worksheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dimension}"/><sheetData>{"".join(rows_xml)}</sheetData></worksheet>'
    )
    safe_sheet = escape(sheet_name[:31] or "Sheet1")
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{safe_sheet}" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        "</Relationships>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        "</Types>"
    )
    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/styles.xml", styles_xml)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def build_total_load_1s_excels(input_dir: Path, ais_root: Path, output_dir: Path, summary_dir: Path) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    summary_dir = Path(summary_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    files = sorted(input_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No .xlsx files found under {input_dir}")
    for path in files:
        raw = read_xlsx_first_sheet(path)
        built, meta = build_one_voyage_1s(raw, voyage_name=path.stem, ais_root=Path(ais_root), source_file_name=path.name)
        out_path = output_dir / path.name
        write_xlsx(built, out_path)
        meta["output_file"] = str(out_path)
        rows.append(meta)
    summary = pd.DataFrame(rows)
    summary_path = summary_dir / "summary_total_load_1s.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--ais_root", type=Path, default=DEFAULT_AIS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_total_load_1s_excels(
        input_dir=args.input_dir,
        ais_root=args.ais_root,
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
    )
    payload = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "ais_root": str(Path(args.ais_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "summary_csv": str((Path(args.summary_dir) / "summary_total_load_1s.csv").resolve()),
        "processed_voyages": int(len(summary)),
        "max_identity_error_kw": float(summary["load_identity_max_abs_error_kw"].max()),
        "total_timestamp_gap_count": int(summary["timestamp_gap_count"].sum()),
        "total_speed_missing_count": int(summary["speed_missing_count"].sum()),
        "total_negative_load_count": int(summary["negative_load_count"].sum()),
        "total_extreme_load_gt_1000kw_count": int(summary["extreme_load_gt_1000kw_count"].sum()),
    }
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
