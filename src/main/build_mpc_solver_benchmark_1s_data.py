from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "spline_1s_diagnostics" / "data" / "natural_clipped_by_voyage"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "mpc_solver_benchmark_1s" / "data"
SKIP_MARKER = ROOT / "outputs" / "mpc_solver_benchmark_1s" / "SKIP_BENCHMARK_DATA_NOT_READY.txt"

REQUIRED_COLUMNS = (
    "voyage_id",
    "split",
    "time_s",
    "load_total_kw",
    "is_original_30s_point",
    "online_feasible",
    "uses_future_endpoint",
)
OPTIONAL_COLUMNS = (
    "dataset_version",
    "time_original_or_reconstructed",
    "timestamp",
    "source_interval_start_time",
    "source_interval_end_time",
    "file_name",
)


def _parse_bool_series(values: pd.Series, *, default: bool) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default).astype(bool)
    mapped = values.map(
        lambda value: default
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    return mapped.astype(bool)


def _write_skip(reason: str) -> None:
    SKIP_MARKER.parent.mkdir(parents=True, exist_ok=True)
    SKIP_MARKER.write_text(reason.strip() + "\n", encoding="utf-8")


def _read_voyage_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("voyage_id", "split", "time_s", "load_total_kw"):
        if column not in frame.columns:
            raise ValueError(f"{path} missing required column: {column}")
    if "is_original_30s_point" not in frame.columns:
        frame["is_original_30s_point"] = False
    if "online_feasible" not in frame.columns:
        frame["online_feasible"] = False
    if "uses_future_endpoint" not in frame.columns:
        frame["uses_future_endpoint"] = True
    for column in OPTIONAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame["voyage_id"] = frame["voyage_id"].astype(str)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["time_s"] = pd.to_numeric(frame["time_s"], errors="coerce")
    frame["load_total_kw"] = pd.to_numeric(frame["load_total_kw"], errors="coerce")
    frame["is_original_30s_point"] = _parse_bool_series(frame["is_original_30s_point"], default=False)
    frame["online_feasible"] = _parse_bool_series(frame["online_feasible"], default=False)
    frame["uses_future_endpoint"] = _parse_bool_series(frame["uses_future_endpoint"], default=True)
    frame["source_csv"] = str(path)
    return frame


def _quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for voyage_id, group in frame.groupby("voyage_id", sort=True):
        group = group.sort_values("time_s")
        diffs = group["time_s"].diff().dropna()
        rows.append(
            {
                "voyage_id": voyage_id,
                "split": ",".join(sorted(group["split"].dropna().unique())),
                "rows": int(len(group)),
                "time_s_start": float(group["time_s"].min()),
                "time_s_end": float(group["time_s"].max()),
                "median_dt_s": float(diffs[diffs > 0].median()) if not diffs[diffs > 0].empty else np.nan,
                "duplicate_time_s": int(group.duplicated(["time_s"]).sum()),
                "nan_core_values": int(group[["time_s", "load_total_kw"]].isna().sum().sum()),
                "negative_load_rows": int((group["load_total_kw"] < 0.0).sum()),
                "min_load_total_kw": float(group["load_total_kw"].min()),
                "max_load_total_kw": float(group["load_total_kw"].max()),
                "mean_load_total_kw": float(group["load_total_kw"].mean()),
                "original_30s_rows": int(group["is_original_30s_point"].sum()),
                "online_feasible_any": bool(group["online_feasible"].any()),
                "uses_future_endpoint_all": bool(group["uses_future_endpoint"].all()),
            }
        )
    return pd.DataFrame(rows)


def _write_check_report(
    *,
    output_path: Path,
    source_dir: Path,
    parquet_path: Path,
    split_json_path: Path,
    all_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    sample_interval_seconds: float,
) -> None:
    split_counts = all_summary["split"].value_counts().to_dict() if not all_summary.empty else {}
    lines = [
        "# 1 s Spline Benchmark Data Check",
        "",
        "Scope: offline solver benchmark input only. These rows are natural cubic-spline reconstructions with negative load clipped to zero; they are not measured 1 s data.",
        "",
        f"- Source directory: `{source_dir}`",
        f"- Output parquet: `{parquet_path}`",
        f"- Split JSON: `{split_json_path}`",
        f"- Inferred sample interval: `{sample_interval_seconds} s`",
        f"- Source voyage split count summary: `{split_counts}`",
        "- `online_feasible`: `false`",
        "- `uses_future_endpoint`: `true`",
        "- `not_measured_1s`: `true`",
        "",
        "## Test Voyage Checks",
        "",
        "| voyage_id | rows | median_dt_s | duplicate_time_s | nan_core_values | negative_load_rows | min_load_total_kw | max_load_total_kw | original_30s_rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_summary.to_dict("records"):
        lines.append(
            "| {voyage_id} | {rows} | {median_dt_s:.3f} | {duplicate_time_s} | {nan_core_values} | {negative_load_rows} | {min_load_total_kw:.6f} | {max_load_total_kw:.6f} | {original_30s_rows} |".format(
                **row
            )
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_benchmark_dataset(
    *,
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str | int | float | list[str]]:
    source_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(path for path in source_dir.glob("voyage_*.csv") if path.name != "manifest.csv")
    if not csv_files:
        reason = f"No per-voyage CSV files found under {source_dir}"
        _write_skip(reason)
        raise RuntimeError(reason)

    frames = [_read_voyage_csv(path) for path in csv_files]
    all_data = pd.concat(frames, ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in all_data.columns]
    if missing:
        reason = f"Missing required columns after load: {missing}"
        _write_skip(reason)
        raise ValueError(reason)

    all_summary = _quality_summary(all_data)
    test_data = all_data[all_data["split"].eq("test")].copy()
    if test_data.empty:
        reason = "No split == test rows found in natural_clipped_by_voyage"
        _write_skip(reason)
        raise RuntimeError(reason)

    test_data = test_data.sort_values(["voyage_id", "time_s"], kind="mergesort").reset_index(drop=True)
    test_summary = _quality_summary(test_data)
    bad_rows = test_summary[
        (test_summary["duplicate_time_s"] > 0)
        | (test_summary["nan_core_values"] > 0)
        | (test_summary["negative_load_rows"] > 0)
    ]
    if not bad_rows.empty:
        reason = "1 s benchmark data quality check failed:\n" + bad_rows.to_string(index=False)
        _write_skip(reason)
        raise ValueError(reason)

    positive_diffs = test_data.groupby("voyage_id")["time_s"].diff().dropna()
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    sample_interval_seconds = float(positive_diffs.median()) if not positive_diffs.empty else 1.0
    if not np.isclose(sample_interval_seconds, 1.0, atol=1.0e-9):
        reason = f"Expected 1 s median interval, found {sample_interval_seconds}"
        _write_skip(reason)
        raise ValueError(reason)

    selected_columns = list(REQUIRED_COLUMNS) + [column for column in OPTIONAL_COLUMNS if column in test_data.columns]
    selected_columns.append("source_csv")
    parquet_path = out_dir / "test_voyages_spline_1s.parquet"
    test_data[selected_columns].to_parquet(parquet_path, index=False)

    split_payload = {
        "source_dir": str(source_dir),
        "source_dataset_version": "cubic_spline_1s_natural_clipped_by_voyage",
        "data_definition": "offline natural cubic-spline 1 s reconstruction, load_total_kw clipped at 0",
        "sample_interval_seconds": 1.0,
        "online_feasible": False,
        "uses_future_endpoint": True,
        "not_measured_1s": True,
        "train_voyages": sorted(all_data.loc[all_data["split"].eq("train"), "voyage_id"].unique().tolist()),
        "validation_voyages": sorted(
            all_data.loc[all_data["split"].isin(["validation", "val"]), "voyage_id"].unique().tolist()
        ),
        "test_voyages": sorted(test_data["voyage_id"].unique().tolist()),
        "test_rows": int(len(test_data)),
        "required_columns_preserved": list(REQUIRED_COLUMNS),
    }
    split_json_path = out_dir / "voyage_split_spline_1s_total_load_721.json"
    split_json_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = out_dir / "spline_1s_benchmark_data_check.md"
    _write_check_report(
        output_path=report_path,
        source_dir=source_dir,
        parquet_path=parquet_path,
        split_json_path=split_json_path,
        all_summary=all_summary,
        test_summary=test_summary,
        sample_interval_seconds=sample_interval_seconds,
    )
    if SKIP_MARKER.exists():
        SKIP_MARKER.unlink()

    return {
        "parquet_path": str(parquet_path),
        "split_json_path": str(split_json_path),
        "check_report_path": str(report_path),
        "test_rows": int(len(test_data)),
        "test_voyages": sorted(test_data["voyage_id"].unique().tolist()),
        "sample_interval_seconds": 1.0,
    }


def main() -> None:
    result = build_benchmark_dataset()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

