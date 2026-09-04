"""Rebuild a traceable operating-segment dataset from the untouched raw telemetry."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.data_loader import discover_voyages  # noqa: E402
from utils.rebuilt_operating_dataset import (  # noqa: E402
    collapse_battery_records,
    collapse_fc_records,
    match_nearest_without_reuse,
    pchip_to_one_second,
)


RAW_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "operating_segments_1s_rebuilt"
MATCH_TOLERANCE_S = 5.0


def _sort_key(path: Path) -> tuple[int, int, int, str]:
    match = re.match(r"(\d+)月(\d+)日(\d+)_", path.name)
    if not match:
        raise ValueError(f"unrecognised voyage folder: {path.name}")
    return (*map(int, match.groups()), path.name)


def _files(directory: Path, prefix: str) -> list[Path]:
    paths = sorted(path for path in directory.glob("*.csv") if path.name.startswith(f"{prefix}_"))
    if not paths:
        raise ValueError(f"missing channel {prefix} in {directory}")
    return paths


def _read_many(paths: list[Path], columns: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(path, usecols=columns, encoding="utf-8-sig") for path in paths]
    return pd.concat(frames, ignore_index=True)


def _read_fc(directory: Path, side: str, number: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = _read_many(_files(directory, f"{side}氢燃料电池#{number}"), ["Time", "发电功率(kW)"])
    return collapse_fc_records(raw.rename(columns={"Time": "timestamp", "发电功率(kW)": "power_kw"}))


def _read_battery(directory: Path, side: str, number: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = _read_many(_files(directory, f"{side}电池簇{number}"), ["Time", "总电压(V)", "总电流(A)"])
    return collapse_battery_records(raw.rename(columns={"Time": "timestamp", "总电压(V)": "voltage_v", "总电流(A)": "current_a"}))


def _read_ais(voyage_root: Path) -> pd.DataFrame:
    paths = sorted((voyage_root / "推进系统").glob("AIS航速_*.csv"))
    if len(paths) != 1:
        return pd.DataFrame(columns=["timestamp", "ais_speed_kn"])
    raw = pd.read_csv(paths[0], usecols=["Time", "航速(节)"], encoding="utf-8-sig")
    speed = pd.to_numeric(raw["航速(节)"].astype(str).str.replace(r"\s*kn$", "", regex=True), errors="coerce")
    return pd.DataFrame({"timestamp": pd.to_datetime(raw["Time"], errors="coerce"), "ais_speed_kn": speed.mask(speed.eq(-9999.0))}).dropna(subset=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp")


def _align_voyage(voyage_root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    fc_channels: list[pd.DataFrame] = []
    battery_channels: list[pd.DataFrame] = []
    qa: dict[str, object] = {"source_folder": voyage_root.name, "channels": {}, "matched_rows": 0, "fully_aligned_rows": 0}
    for side in ("左", "右"):
        for number in range(1, 5):
            channel, stats = _read_fc(voyage_root / "燃料电池系统", side, number)
            fc_channels.append(channel)
            qa["channels"][f"fc_{side}_{number}"] = stats
        for number in range(1, 7):
            channel, stats = _read_battery(voyage_root / "BMS", side, number)
            battery_channels.append(channel)
            qa["channels"][f"battery_{side}_{number}"] = stats
    reference = fc_channels[0][["timestamp"]].copy()
    output = reference.copy()
    for index, channel in enumerate(fc_channels):
        matched = match_nearest_without_reuse(reference, channel, tolerance_s=MATCH_TOLERANCE_S)
        output[f"fc_{index + 1}_kw"] = matched["power_kw"]
    for index, channel in enumerate(battery_channels):
        matched = match_nearest_without_reuse(reference, channel, tolerance_s=MATCH_TOLERANCE_S)
        output[f"battery_{index + 1}_kw"] = matched["power_kw"]
    fc_columns = [column for column in output if column.startswith("fc_")]
    battery_columns = [column for column in output if column.startswith("battery_")]
    output["fc_total_kw"] = output[fc_columns].sum(axis=1, min_count=8)
    output["battery_total_kw"] = output[battery_columns].sum(axis=1, min_count=12)
    output["load_total_kw"] = output[["fc_total_kw", "battery_total_kw"]].sum(axis=1, min_count=2)
    ais = _read_ais(voyage_root)
    if not ais.empty:
        output = pd.merge_asof(output.sort_values("timestamp"), ais, on="timestamp", direction="nearest", tolerance=pd.Timedelta(seconds=MATCH_TOLERANCE_S))
    else:
        output["ais_speed_kn"] = np.nan
    output["parent_voyage"] = voyage_root.name
    output["aligned"] = output[["fc_total_kw", "battery_total_kw"]].notna().all(axis=1)
    qa["matched_rows"] = int(len(output))
    qa["fully_aligned_rows"] = int(output["aligned"].sum())
    return output, qa


def _run_ids(frame: pd.DataFrame, break_mask: pd.Series) -> pd.Series:
    return break_mask.astype(bool).cumsum()


def _derive_shore_policy(raw: pd.DataFrame) -> dict[str, float | int]:
    finite = raw.loc[raw["aligned"]].copy()
    cadence = finite.sort_values("timestamp").groupby("parent_voyage")["timestamp"].diff().dt.total_seconds().dropna()
    median_cadence = float(cadence.loc[cadence.between(20.0, 40.0)].median())
    speed = finite["ais_speed_kn"].dropna()
    fc = finite["fc_total_kw"].abs()
    negative_battery = -finite.loc[finite["battery_total_kw"].lt(0.0), "battery_total_kw"]
    return {
        "median_raw_cadence_s": median_cadence,
        "fc_idle_threshold_kw": float(max(0.1, min(5.0, fc.quantile(0.05)))),
        "speed_idle_threshold_kn": float(max(0.05, min(0.3, speed.quantile(0.10)))) if not speed.empty else 0.0,
        "battery_charge_threshold_kw": float(max(0.1, negative_battery.quantile(0.05))) if not negative_battery.empty else np.inf,
        "minimum_shore_points": int(max(3, np.ceil(90.0 / median_cadence))),
        "long_gap_threshold_s": float(max(120.0, 4.0 * median_cadence)),
    }


def _mark_shore_and_segments(frame: pd.DataFrame, policy: dict[str, float | int]) -> pd.DataFrame:
    result = frame.sort_values("timestamp", kind="stable").copy()
    shore_point = (
        result["aligned"]
        & result["fc_total_kw"].le(float(policy["fc_idle_threshold_kw"]))
        & result["ais_speed_kn"].le(float(policy["speed_idle_threshold_kn"]))
        & result["battery_total_kw"].lt(-float(policy["battery_charge_threshold_kw"]))
    )
    shore_group = shore_point.ne(shore_point.shift()).cumsum()
    shore_size = shore_point.groupby(shore_group).transform("sum")
    result["shore"] = shore_point & shore_size.ge(int(policy["minimum_shore_points"]))
    gap = result["timestamp"].diff().dt.total_seconds().gt(float(policy["long_gap_threshold_s"]))
    invalid = ~result["aligned"]
    boundary = gap | invalid | result["shore"] | result["shore"].shift(fill_value=False)
    result["candidate_segment"] = _run_ids(result, boundary)
    result["usable"] = result["aligned"] & ~result["shore"]
    return result


def _split_voyages(voyages: list[str]) -> dict[str, list[str]]:
    count = len(voyages)
    train_count = int(np.floor(0.70 * count))
    validation_count = int(np.floor(0.20 * count))
    return {
        "train": voyages[:train_count],
        "validation": voyages[train_count:train_count + validation_count],
        "test": voyages[train_count + validation_count:],
    }


def build_dataset(raw_root: Path = RAW_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    raw_root, output_root = Path(raw_root), Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    voyages = sorted(discover_voyages(raw_root), key=lambda voyage: _sort_key(voyage.root))
    staging = output_root.with_name(f"{output_root.name}.staging")
    if staging.exists():
        raise FileExistsError(f"stale staging directory: {staging}")
    raw_by_voyage: dict[str, pd.DataFrame] = {}
    alignment_qa: list[dict[str, object]] = []
    try:
        for index, voyage in enumerate(voyages, start=1):
            aligned, qa = _align_voyage(voyage.root)
            raw_by_voyage[voyage.root.name] = aligned
            alignment_qa.append(qa)
            print(f"[raw alignment] {index}/{len(voyages)} {voyage.root.name}", flush=True)
        policy = _derive_shore_policy(pd.concat(raw_by_voyage.values(), ignore_index=True))
        staging.mkdir(parents=True)
        raw_dir = staging / "raw_30s_total_load_by_voyage"
        clean_dir = staging / "cleaned_30s_segments"
        one_second_dir = staging / "operating_segments_1s_rebuilt"
        raw_dir.mkdir(); clean_dir.mkdir(); one_second_dir.mkdir()
        manifest_rows: list[dict[str, object]] = []
        exclusion_rows: list[dict[str, object]] = []
        segment_id = 0
        for index, (voyage_name, raw) in enumerate(raw_by_voyage.items(), start=1):
            marked = _mark_shore_and_segments(raw, policy)
            raw_cols = ["timestamp", "fc_total_kw", "battery_total_kw", "load_total_kw", "ais_speed_kn", "aligned", "shore"]
            target = raw_dir / voyage_name; target.mkdir()
            marked[raw_cols].to_csv(target / "power_30s.csv", index=False, encoding="utf-8-sig")
            exclusion_rows.append({"parent_voyage": voyage_name, "unaligned_rows": int((~marked.aligned).sum()), "shore_rows": int(marked.shore.sum())})
            for _, candidate in marked.loc[marked.usable].groupby("candidate_segment", sort=False):
                if len(candidate) < 2:
                    continue
                segment_id += 1
                identifier = f"operating_segment_{segment_id:04d}"
                cleaned = candidate[["timestamp", "fc_total_kw", "battery_total_kw", "load_total_kw", "ais_speed_kn"]].copy()
                cleaned.to_csv(clean_dir / f"{identifier}.csv", index=False, encoding="utf-8-sig")
                one_second, pchip_qa = pchip_to_one_second(cleaned)
                one_second.to_csv(one_second_dir / f"{identifier}.csv", index=False, encoding="utf-8-sig")
                manifest_rows.append({"segment_id": identifier, "parent_voyage": voyage_name, "start_time": cleaned.timestamp.iloc[0].isoformat(), "end_time": cleaned.timestamp.iloc[-1].isoformat(), "raw_points": len(cleaned), "num_1s_points": len(one_second), "raw_csv": f"cleaned_30s_segments/{identifier}.csv", "one_second_csv": f"operating_segments_1s_rebuilt/{identifier}.csv", **pchip_qa})
            print(f"[segments] {index}/{len(raw_by_voyage)} {voyage_name}", flush=True)
        manifest = pd.DataFrame(manifest_rows)
        effective = sorted(manifest.parent_voyage.unique(), key=lambda name: _sort_key(raw_root / name)) if not manifest.empty else []
        splits = _split_voyages(effective)
        manifest["split"] = np.select([manifest.parent_voyage.isin(splits["train"]), manifest.parent_voyage.isin(splits["validation"])], ["train", "validation"], default="test")
        manifest.to_csv(staging / "split_manifest.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(exclusion_rows).to_csv(staging / "exclusion_qa.csv", index=False, encoding="utf-8-sig")
        pd.json_normalize(alignment_qa, sep=".").to_csv(staging / "alignment_qa_by_voyage.csv", index=False, encoding="utf-8-sig")
        raw_values = pd.concat([frame[["fc_total_kw", "battery_total_kw", "load_total_kw"]] for frame in raw_by_voyage.values()], ignore_index=True)
        one_values = pd.concat([pd.read_csv(one_second_dir / Path(row.one_second_csv).name, usecols=["load_total_kw"]) for row in manifest.itertuples(index=False)], ignore_index=True) if not manifest.empty else pd.DataFrame({"load_total_kw": []})
        summary = {"raw_voyage_count": len(voyages), "raw_total_rows": int(sum(len(frame) for frame in raw_by_voyage.values())), "alignment": alignment_qa, "shore_policy": policy, "segment_count": int(len(manifest)), "effective_voyage_count": len(effective), "split_voyages": splits, "split_segment_counts": manifest.split.value_counts().to_dict(), "raw_load_stats": raw_values.load_total_kw.describe().to_dict(), "one_second_load_counts": {"zero": int(one_values.load_total_kw.eq(0).sum()), "negative": int(one_values.load_total_kw.lt(0).sum()), "under_1_kw": int(one_values.load_total_kw.lt(1).sum())}, "exclusions": exclusion_rows}
        (staging / "qa_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        staging.replace(output_root)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(build_dataset(), ensure_ascii=False, indent=2, default=float))
