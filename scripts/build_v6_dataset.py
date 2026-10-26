"""Reuse the frozen independent dataset's canonical 1 s Train/Validation files.

The current dataset is already on a complete one-second grid. No reconstruction,
resampling or interpolation is performed. Test remains metadata only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "v5-economic4-ddqn/data/processed/operating_dataset_independent_v2"
DEFAULT_OUTPUT = ROOT / "data/processed/operating_dataset_v6_1s"


def build(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    active = json.loads((source / "metadata/active_split.json").read_text(encoding="utf-8"))
    with (source / active["manifest_relative_path"]).open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    selected = [row for row in source_rows if row["split"] in ("train", "validation")]
    reports = []
    for row in selected:
        original = source / row["relative_path"]
        destination = output / row["relative_path"]
        before = pd.read_csv(original, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        duration = int(row["duration_s"])
        time = before["time_s"].to_numpy(dtype=float)
        source_time = before["source_time_s"].to_numpy(dtype=float)
        source_rows_1s = before["source_1s_row"].to_numpy(dtype=np.int64)
        stamps = pd.to_datetime(before["timestamp"])
        timestamp_dt = stamps.diff().iloc[1:].dt.total_seconds().to_numpy()
        irregular = bool(np.any(np.diff(time) != 1) or np.any(timestamp_dt != 1)
                         or np.any(np.diff(source_time) != 1) or np.any(np.diff(source_rows_1s) != 1))
        missing = bool(np.any(np.diff(time) > 1) or np.any(timestamp_dt > 1)
                       or np.any(np.diff(source_time) > 1) or np.any(np.diff(source_rows_1s) > 1))
        if irregular or not np.array_equal(time, np.arange(duration + 1, dtype=float)):
            raise ValueError(f"{row['candidate_id']}: source is not the expected complete canonical 1 s grid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        after = pd.read_csv(destination, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        if not before.equals(after):
            raise ValueError(f"{row['candidate_id']}: copied canonical fields changed")
        max_deviation = float(np.max(np.abs(before["load_total_kw"].to_numpy(dtype=float)
                                             - after["load_total_kw"].to_numpy(dtype=float))))
        reports.append({
            "sample_id": row["candidate_id"], "split": row["split"],
            "parent_sample_id": row["parent_sample_id"],
            "point_count_before": len(before), "point_count_after": len(after),
            "duration_s": duration, "original_anchor_count": len(before),
            "added_points": 0, "max_anchor_abs_load_deviation_kw": max_deviation,
            "irregular_time_intervals": irregular, "missing_intervals": missing,
            "start_timestamp": before["timestamp"].iloc[0],
            "end_timestamp": before["timestamp"].iloc[-1],
            "source_start_time_s": before["source_time_s"].iloc[0],
            "source_end_time_s": before["source_time_s"].iloc[-1],
            "source_start_1s_row": before["source_1s_row"].iloc[0],
            "source_end_1s_row": before["source_1s_row"].iloc[-1],
            "all_payload_fields_text_exact": True,
            "derivation": "unchanged_existing_canonical_1s",
            "source_path": str(original), "destination_path": str(destination),
        })

    metadata = output / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    manifest_columns = [column for column in source_rows[0] if "sha256" not in column]
    manifest_columns += ["source_dataset_path", "source_payload_relative_path", "payload_copied", "v6_derivation"]
    with (metadata / "sample_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_columns)
        writer.writeheader()
        for row in source_rows:
            copied = row["split"] in ("train", "validation")
            derived = {column: row[column] for column in manifest_columns if column in row}
            derived.update(source_dataset_path=str(source), source_payload_relative_path=row["relative_path"],
                           payload_copied=copied,
                           v6_derivation="unchanged_existing_canonical_1s" if copied else "test_metadata_only")
            writer.writerow(derived)
    v6_active = {
        "dataset_version": "operating_dataset_v6_1s", "split_version": active["split_version"],
        "manifest_relative_path": "metadata/sample_manifest.csv", "split_counts": active["split_counts"],
        "source_dataset_path": str(source), "source_manifest_relative_path": active["manifest_relative_path"],
        "model_step_s": 1, "copied_payloads": len(selected), "copied_test_payloads": 0,
        "derivation": "unchanged_existing_canonical_1s", "status": "SPLIT_FROZEN",
    }
    (metadata / "active_split.json").write_text(json.dumps(v6_active, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    frame = pd.DataFrame(reports)
    frame.to_csv(metadata / "data_processing_report.csv", index=False, encoding="utf-8-sig")
    split_summary = {
        split: {
            "episodes": int(frame["split"].eq(split).sum()),
            "nodes_before": int(frame.loc[frame["split"].eq(split), "point_count_before"].sum()),
            "nodes_after": int(frame.loc[frame["split"].eq(split), "point_count_after"].sum()),
            "control_intervals": int(frame.loc[frame["split"].eq(split), "duration_s"].sum()),
        } for split in ("train", "validation")
    }
    summary = {
        "source_path": str(source), "destination_path": str(output),
        "frozen_split_counts": active["split_counts"], "processed_splits": split_summary,
        "reused_existing_1s_payloads": len(selected), "added_interpolation_points": 0,
        "original_anchor_count": int(frame["original_anchor_count"].sum()),
        "max_anchor_abs_load_deviation_kw": float(frame["max_anchor_abs_load_deviation_kw"].max()),
        "all_payload_fields_text_exact": bool(frame["all_payload_fields_text_exact"].all()),
        "irregular_episodes": int(frame["irregular_time_intervals"].sum()),
        "missing_interval_episodes": int(frame["missing_intervals"].sum()),
        "opened_test_payloads": 0, "copied_test_payloads": 0,
        "terminal_boundary": "last node is retained metadata; actions only at t=0..duration_s-1",
        "interpretation": "Reuse of existing processed 1 s trajectories; no new real 1 Hz measurements.",
    }
    (metadata / "data_processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# V6 1 秒数据处理报告", "",
        "直接复用冻结数据集已有的 canonical 1 秒轨迹，43 个 gzip 文件原样复制；新增插值点 0。",
        "Train/Validation/Test 保持 35/8/4；只打开并复制 Train 和 Validation，Test 仅保留冻结元数据。",
        "逐点对照全部字段的原始文本，时间戳、负载、source_time_s、source_1s_row 全部完全一致。",
        "所有原始锚点负载最大绝对偏差为 0；无不规则间隔或缺失区间。",
        "已有处理轨迹不属于新增的真实 1 Hz 实测数据。终点节点保留，但不产生额外动作。", "",
        f"源目录：`{source}`", f"目标目录：`{output}`", "",
        "每条航段的完整来源与目标路径、源行号、起止时间戳见同目录 data_processing_report.csv。", "",
        "| 航段 | 划分 | 处理前节点 | 处理后节点 | 时长(s) | 原始锚点 | 最大偏差(kW) | 不规则/缺失 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in reports:
        lines.append(f"| {row['sample_id']} | {row['split']} | {row['point_count_before']} | "
                     f"{row['point_count_after']} | {row['duration_s']} | {row['original_anchor_count']} | "
                     f"{row['max_anchor_abs_load_deviation_kw']:g} | 无/无 |")
    (metadata / "data_processing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
