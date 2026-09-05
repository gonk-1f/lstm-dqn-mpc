"""Build a traceable, parent-safe 1 s PCHIP load dataset from read-only raw telemetry."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from io import StringIO
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
    align_ais_to_power, chronological_parent_splits, collapse_battery_records,
    collapse_fc_records, collapse_scalar_records, find_contiguous_intervals,
    battery_state, match_nearest_without_reuse, pchip_to_one_second, select_shore_intervals,
)
from zero_residual_numerical_negative_loads import zero_residual_numerical_negatives  # noqa: E402

RAW_ROOT = Path.home() / "OneDrive" / "Desktop" / "氢舟一号"
OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "operating_segments_1s_rebuilt"
LONG_GAP_THRESHOLD_S = 120.0
BASELINE_COMMIT = "dc9af36d6705d24bd1fa2adde46cd3b07a301c15"


def _sort_key(path: Path) -> tuple[int, int, int, str]:
    match = re.match(r"(\d+)月(\d+)日(\d+)_", path.name)
    if not match:
        raise ValueError(f"unrecognised voyage folder: {path.name}")
    return (*map(int, match.groups()), path.name)


def _files(directory: Path, prefix: str) -> list[Path]:
    paths = sorted(path for path in directory.glob("*.csv") if path.name.startswith(f"{prefix}_"))
    if not paths:
        raise ValueError(f"missing raw channel {prefix} in {directory}")
    return paths


def _read_many(paths: list[Path], wanted: list[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        headers = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        frames.append(pd.read_csv(path, usecols=[name for name in wanted if name in headers], encoding="utf-8-sig"))
    return pd.concat(frames, ignore_index=True)


def _read_fc(directory: Path, side: str, number: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = _read_many(_files(directory, f"{side}氢燃料电池#{number}"), ["Time", "发电功率(kW)"])
    if "发电功率(kW)" not in raw:
        raise ValueError(f"{side} fuel-cell {number} has no power field")
    return collapse_fc_records(raw.rename(columns={"Time": "timestamp", "发电功率(kW)": "power_kw"}))


def _read_battery(directory: Path, side: str, number: int) -> tuple[pd.DataFrame, dict[str, int]]:
    raw = _read_many(_files(directory, f"{side}电池簇{number}"), ["Time", "总电压(V)", "总电流(A)", "SOC(%)"])
    if not {"Time", "总电压(V)", "总电流(A)"}.issubset(raw.columns):
        raise ValueError(f"{side} battery {number} lacks voltage/current")
    power, stats = collapse_battery_records(raw.rename(columns={"Time": "timestamp", "总电压(V)": "voltage_v", "总电流(A)": "current_a"}))
    if "SOC(%)" not in raw:
        power["soc_pct"] = np.nan
        return power, stats
    soc, soc_stats = collapse_scalar_records(raw.rename(columns={"Time": "timestamp"}), value_column="SOC(%)", output_column="soc_pct")
    stats["soc_exact_duplicate_records_removed"] = soc_stats["exact_duplicate_records_removed"]
    return power.merge(soc, on="timestamp", how="left"), stats


def _read_optional_scalar(directory: Path, prefix: str, raw_name: str, output_name: str) -> pd.DataFrame:
    paths = sorted(path for path in directory.glob("*.csv") if path.name.startswith(f"{prefix}_"))
    if not paths:
        return pd.DataFrame(columns=["timestamp", output_name])
    raw = _read_many(paths, ["Time", raw_name])
    if raw_name not in raw:
        return pd.DataFrame(columns=["timestamp", output_name])
    return collapse_scalar_records(raw.rename(columns={"Time": "timestamp"}), value_column=raw_name, output_column=output_name)[0]


def _read_ais(voyage_root: Path) -> pd.DataFrame:
    paths = sorted((voyage_root / "推进系统").glob("AIS航速_*.csv"))
    if len(paths) != 1:
        return pd.DataFrame(columns=["timestamp", "ais_speed_kn"])
    raw = pd.read_csv(paths[0], encoding="utf-8-sig")
    if "Time" not in raw or "航速(节)" not in raw:
        return pd.DataFrame(columns=["timestamp", "ais_speed_kn"])
    speed = pd.to_numeric(raw["航速(节)"].astype(str).str.replace(r"\s*kn$", "", regex=True), errors="coerce").mask(lambda value: value.eq(-9999.0))
    return collapse_scalar_records(pd.DataFrame({"timestamp": raw["Time"], "speed": speed}), value_column="speed", output_column="ais_speed_kn")[0]


def _nearest_offsets(reference: pd.Series, source: pd.Series) -> np.ndarray:
    ref = pd.to_datetime(reference, errors="coerce").dropna().map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    src = pd.to_datetime(source, errors="coerce").dropna().map(lambda value: pd.Timestamp(value).value).to_numpy(dtype=np.int64)
    if not len(ref) or not len(src):
        return np.array([], dtype=float)
    src = np.sort(src.copy()); right = np.searchsorted(src, ref, side="left"); left = right - 1
    delta = np.full(len(ref), np.inf)
    left_ok, right_ok = left >= 0, right < len(src)
    delta[left_ok] = np.minimum(delta[left_ok], np.abs(ref[left_ok] - src[left[left_ok]]))
    delta[right_ok] = np.minimum(delta[right_ok], np.abs(src[right[right_ok]] - ref[right_ok]))
    return delta[np.isfinite(delta)] / 1.0e9


def _build_cache(voyages: list[object]) -> tuple[dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, object]]], list[pd.DataFrame]]:
    cache: dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, object]]] = {}
    ais_frames: list[pd.DataFrame] = []
    for voyage in voyages:
        fc, battery = [], []
        qa: dict[str, object] = {"parent_voyage": voyage.root.name, "channels": {}}
        for side in ("左", "右"):
            for number in range(1, 5):
                frame, stats = _read_fc(voyage.fuel_cell_dir, side, number); fc.append(frame); qa["channels"][f"fc_{side}_{number}"] = stats
            for number in range(1, 7):
                frame, stats = _read_battery(voyage.bms_dir, side, number); battery.append(frame); qa["channels"][f"battery_{side}_{number}"] = stats
        cache[voyage.root.name] = (fc, battery, qa)
        ais_frames.append(_read_ais(voyage.root))
    return cache, ais_frames


def _policies(cache: dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, object]]], ais_frames: list[pd.DataFrame]) -> dict[str, object]:
    offsets: list[float] = []
    for fc, battery, _ in cache.values():
        for channel in [*fc[1:], *battery]:
            values = _nearest_offsets(fc[0]["timestamp"], channel["timestamp"])
            offsets.extend(values[values <= 15.0].tolist())
    if not offsets:
        raise RuntimeError("cannot derive a device timestamp matching tolerance")
    offset = pd.Series(offsets)
    gaps = pd.concat(
        [pd.to_datetime(frame["timestamp"], errors="coerce").sort_values().diff().dt.total_seconds() for frame in ais_frames],
        ignore_index=True,
    ).dropna()
    normal = gaps[gaps.between(5.0, 60.0)]
    if normal.empty:
        ais_median, ais_q99 = 20.0, 30.0
    else:
        ais_median, ais_q99 = float(normal.median()), float(normal.quantile(0.99))
    ais_max_gap = float(min(120.0, max(2.0 * ais_median, math.ceil(1.5 * ais_q99))))
    return {"device_offset_q95_s": float(offset.quantile(.95)), "device_offset_q99_s": float(offset.quantile(.99)), "device_match_tolerance_s": float(max(1, min(15, math.ceil(float(offset.quantile(.99)))))), "ais_median_interval_s": ais_median, "ais_q99_interval_s": ais_q99, "ais_max_normal_gap_s": ais_max_gap, "ais_max_nearest_s": float(math.ceil(ais_max_gap / 2.0)), "long_gap_threshold_s": LONG_GAP_THRESHOLD_S}


def _align_voyage(voyage: object, cached: tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, object]], policy: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    fc, battery, qa = cached
    reference = fc[0][["timestamp"]].copy(); output = reference.copy()
    tolerance = float(policy["device_match_tolerance_s"])
    for number, channel in enumerate(fc, 1):
        output[f"fc_{number}_kw"] = match_nearest_without_reuse(reference, channel, tolerance_s=tolerance)["power_kw"]
    for number, channel in enumerate(battery, 1):
        matched = match_nearest_without_reuse(reference, channel, tolerance_s=tolerance)
        output[f"battery_{number}_kw"] = matched["power_kw"]
        output[f"battery_{number}_soc_pct"] = matched["soc_pct"]
    fc_columns = [name for name in output if name.startswith("fc_")]
    batt_columns = [name for name in output if name.startswith("battery_") and name.endswith("_kw")]
    soc_columns = [name for name in output if name.endswith("_soc_pct")]
    output[fc_columns] = output[fc_columns].apply(pd.to_numeric, errors="coerce")
    output[batt_columns] = output[batt_columns].apply(pd.to_numeric, errors="coerce")
    output[soc_columns] = output[soc_columns].apply(pd.to_numeric, errors="coerce")
    output["fc_total_kw"] = output[fc_columns].sum(axis=1, min_count=8)
    output["battery_total_kw"] = output[batt_columns].sum(axis=1, min_count=12)
    output["load_total_kw"] = output[["fc_total_kw", "battery_total_kw"]].sum(axis=1, min_count=2)
    output["soc_mean_pct"] = output[soc_columns].mean(axis=1)
    output["aligned"] = output[["fc_total_kw", "battery_total_kw"]].notna().all(axis=1)
    ais = align_ais_to_power(output, _read_ais(voyage.root), max_normal_gap_s=float(policy["ais_max_normal_gap_s"]), max_nearest_s=float(policy["ais_max_nearest_s"]))
    output[["speed_aligned_kn", "speed_source"]] = ais[["speed_aligned_kn", "speed_source"]]
    inverter = []
    for side in ("左", "右"):
        source = _read_optional_scalar(voyage.ems_dir, f"{side}逆变电源", "输出有功功率(kW)", "inverter_kw")
        if not source.empty:
            inverter.append(match_nearest_without_reuse(reference, source, tolerance_s=tolerance)["inverter_kw"])
    output["propulsion_inverter_kw"] = pd.concat(inverter, axis=1).sum(axis=1, min_count=2) if len(inverter) == 2 else np.nan
    output["parent_voyage"] = voyage.root.name
    qa.update({"matched_rows": int(len(output)), "fully_aligned_rows": int(output.aligned.sum()), "unaligned_rows": int((~output.aligned).sum()), "ais_sources": output.speed_source.value_counts().to_dict(), "soc_available_rows": int(output.soc_mean_pct.notna().sum()), "propulsion_available_rows": int(output.propulsion_inverter_kw.notna().sum())})
    return output, qa


def _shore_policy(raw: pd.DataFrame, timing: dict[str, object]) -> dict[str, object]:
    valid = raw[raw.aligned]
    positive_fc = valid.loc[valid.fc_total_kw.gt(0), "fc_total_kw"]
    positive_speed = valid.loc[valid.speed_aligned_kn.gt(0), "speed_aligned_kn"]
    fc_idle = float(positive_fc.min() / 2) if not positive_fc.empty else 0.0
    speed_idle = float(positive_speed.quantile(0.01)) if not positive_speed.empty else 0.0
    stationary_charge = -valid.loc[(valid.speed_aligned_kn.le(speed_idle)) & valid.battery_total_kw.lt(0), "battery_total_kw"]
    return {"fc_idle_threshold_kw": fc_idle, "speed_idle_threshold_kn": speed_idle, "battery_charge_threshold_kw": 1.0, "battery_deadband_kw": 1.0, "minimum_shore_points": int(max(3, math.ceil(90.0 / float(timing["median_power_cadence_s"])))), "long_gap_threshold_s": LONG_GAP_THRESHOLD_S, "stationary_charge_quantiles_kw": stationary_charge.quantile([.05, .6, .7]).to_dict()}


def _zero_small_stationary_load_drift(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Zero only derived stationary load drift; never alter measured channels."""
    result = frame.copy()
    load = pd.to_numeric(result["load_total_kw"], errors="coerce")
    battery = pd.to_numeric(result["battery_total_kw"], errors="coerce")
    fc = pd.to_numeric(result["fc_total_kw"], errors="coerce")
    speed = pd.to_numeric(result["speed_aligned_kn"], errors="coerce")
    propulsion = pd.to_numeric(result.get("propulsion_inverter_kw"), errors="coerce")
    candidate = (
        result["aligned"].astype(bool)
        & load.ge(-1.0) & load.lt(0.0)
        & battery.abs().le(1.0) & fc.le(1.0) & speed.le(0.1)
    )
    if propulsion.notna().any():
        candidate &= propulsion.abs().le(1.0)
    result["is_load_zero_drift"] = candidate.fillna(False)
    result.loc[result["is_load_zero_drift"], "load_total_kw"] = 0.0
    selected = result.loc[result["is_load_zero_drift"]]
    return result, {
        "point_count": int(len(selected)),
        "minimum_original_load_kw": float(load[selected.index].min()) if not selected.empty else 0.0,
        "duration_s": float((pd.Timestamp(selected["timestamp"].iloc[-1]) - pd.Timestamp(selected["timestamp"].iloc[0])).total_seconds()) if len(selected) >= 2 else 0.0,
    }


def _mark_abnormal(frame: pd.DataFrame, shore_policy: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    result = frame.copy(); result["is_abnormal"] = False
    result["fc_total_kw"] = pd.to_numeric(result["fc_total_kw"], errors="coerce")
    result["battery_total_kw"] = pd.to_numeric(result["battery_total_kw"], errors="coerce")
    result["propulsion_inverter_kw"] = pd.to_numeric(result["propulsion_inverter_kw"], errors="coerce")
    result["soc_mean_pct"] = pd.to_numeric(result["soc_mean_pct"], errors="coerce")
    available = result.propulsion_inverter_kw.notna() & result.soc_mean_pct.notna()
    active = result.loc[available & result.propulsion_inverter_kw.gt(0), "propulsion_inverter_kw"]
    if active.empty:
        return result, pd.DataFrame(), "unavailable: no positive propulsion-inverter evidence with SOC"
    prop_threshold = float(active.min() / 2)
    battery_idle = float(max(1.0, result.battery_total_kw.abs().quantile(.10)))
    candidate = available & result.fc_total_kw.le(float(shore_policy["fc_idle_threshold_kw"])) & result.battery_total_kw.abs().le(battery_idle) & result.propulsion_inverter_kw.gt(prop_threshold)
    if "is_external_charging" in result:
        candidate &= ~result["is_external_charging"].astype(bool)
    intervals = find_contiguous_intervals(result, long_gap_threshold_s=LONG_GAP_THRESHOLD_S)
    group = (candidate.ne(candidate.shift(fill_value=False)) | intervals.ne(intervals.shift(fill_value=intervals.iloc[0]))).cumsum()
    rows=[]
    for _, run in result.loc[candidate].groupby(group[candidate], sort=False):
        drop = float(run.soc_mean_pct.iloc[0] - run.soc_mean_pct.iloc[-1])
        if len(run) >= 3 and drop > .05:
            result.loc[run.index, "is_abnormal"] = True
            rows.append({"start_time": run.timestamp.iloc[0].isoformat(), "end_time": run.timestamp.iloc[-1].isoformat(), "abnormal_rows": int(len(run)), "duration_s": float((run.timestamp.iloc[-1]-run.timestamp.iloc[0]).total_seconds()), "reason": "fc_and_battery_near_zero_with_positive_propulsion_and_soc_drop", "soc_drop_pct": drop})
    return result, pd.DataFrame(rows), "available"


def _classify_negative_intervals(
    frame: pd.DataFrame,
    *,
    speed_idle_threshold_kn: float,
    battery_charging_threshold_kw: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove sustained negative FC-plus-battery balances without clipping them.

    A negative net balance with stationary, continuously charging batteries is
    external charging even when the FC channels remain non-zero.  Other
    sustained negative balances are physical inconsistencies, because this
    dataset deliberately contains only FC and battery energy sources.
    """
    result = frame.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["load_total_kw"] = pd.to_numeric(result["load_total_kw"], errors="coerce")
    result["battery_total_kw"] = pd.to_numeric(result["battery_total_kw"], errors="coerce")
    result["fc_total_kw"] = pd.to_numeric(result["fc_total_kw"], errors="coerce")
    result["speed_aligned_kn"] = pd.to_numeric(result["speed_aligned_kn"], errors="coerce")
    result["propulsion_inverter_kw"] = pd.to_numeric(result["propulsion_inverter_kw"], errors="coerce")
    result["is_external_charging"] = result["is_shore"].astype(bool) if "is_shore" in result else False
    result["is_physical_inconsistency"] = False

    # This runs after _zero_small_stationary_load_drift.  Therefore every
    # remaining negative balance is unexplained by the permitted zero-drift
    # rule and cannot enter a formal non-negative load dataset.
    candidate = result["aligned"].astype(bool) & result["load_total_kw"].lt(0.0) & ~result["is_external_charging"]
    interval = find_contiguous_intervals(result, long_gap_threshold_s=LONG_GAP_THRESHOLD_S)
    starts = candidate.ne(candidate.shift(fill_value=False)) | interval.ne(interval.shift(fill_value=interval.iloc[0]))
    groups = starts.cumsum()
    external_rows: list[dict[str, object]] = []
    physical_rows: list[dict[str, object]] = []
    for _, run in result.loc[candidate].groupby(groups[candidate], sort=False):
        known_speed = run["speed_aligned_kn"].dropna()
        stationary = not known_speed.empty and bool(known_speed.le(speed_idle_threshold_kn).all())
        battery_charging = bool(run["battery_total_kw"].lt(-battery_charging_threshold_kw).all())
        propulsion_active = bool(run["propulsion_inverter_kw"].gt(0).any())
        shared = {
            "start_time": run["timestamp"].iloc[0].isoformat(),
            "end_time": run["timestamp"].iloc[-1].isoformat(),
            "raw_points": int(len(run)),
            "duration_s": float((run["timestamp"].iloc[-1] - run["timestamp"].iloc[0]).total_seconds()),
            "min_load_kw": float(run["load_total_kw"].min()),
            "mean_load_kw": float(run["load_total_kw"].mean()),
            "mean_fc_kw": float(run["fc_total_kw"].mean()),
            "mean_battery_kw": float(run["battery_total_kw"].mean()),
            "speed_min_kn": float(known_speed.min()) if not known_speed.empty else np.nan,
            "speed_max_kn": float(known_speed.max()) if not known_speed.empty else np.nan,
            "speed_unavailable_points": int(run["speed_aligned_kn"].isna().sum()),
            "propulsion_positive_points": int(run["propulsion_inverter_kw"].gt(0).sum()),
        }
        if stationary and battery_charging and not propulsion_active:
            result.loc[run.index, "is_external_charging"] = True
            external_rows.append({**shared, "reason": "external_charging_or_shore_power: stationary continuous battery charging with negative FC_plus_battery balance"})
        else:
            result.loc[run.index, "is_physical_inconsistency"] = True
            physical_rows.append({**shared, "reason": "physical_inconsistency: sustained negative FC_plus_battery balance while moving, propelling, or without stationary evidence"})
    return result, pd.DataFrame(external_rows), pd.DataFrame(physical_rows)


def _pure_idle_summary(segment: pd.DataFrame) -> dict[str, object]:
    """Classify an already continuous candidate segment without cutting it apart."""
    frame = segment.copy()
    speed = pd.to_numeric(frame["speed_aligned_kn"], errors="coerce")
    fc = pd.to_numeric(frame["fc_total_kw"], errors="coerce")
    load = pd.to_numeric(frame["load_total_kw"], errors="coerce")
    propulsion = pd.to_numeric(frame["propulsion_inverter_kw"], errors="coerce")
    propulsion_available = bool(propulsion.notna().any())
    pure_idle = speed.le(0.1) & fc.le(1.0) & load.abs().le(1.0)
    if propulsion_available:
        pure_idle &= propulsion.abs().le(1.0)
    pure_idle = pure_idle.fillna(False)
    total_points = int(len(frame))
    pure_points = int(pure_idle.sum())
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    gaps = timestamps.shift(-1).sub(timestamps).dt.total_seconds()
    valid_gap = gaps.gt(0) & gaps.le(LONG_GAP_THRESHOLD_S)
    normal_gaps = gaps[valid_gap]
    representative_last_gap = float(normal_gaps.median()) if not normal_gaps.empty else 0.0
    weights = gaps.where(valid_gap, 0.0)
    if total_points:
        weights.iloc[-1] = representative_last_gap
    total_duration_weight = float(weights.sum())
    pure_idle_duration_s = float(weights[pure_idle].sum())
    ratio = pure_idle_duration_s / total_duration_weight if total_duration_weight else 0.0
    duration_s = float((pd.Timestamp(frame.timestamp.iloc[-1]) - pd.Timestamp(frame.timestamp.iloc[0])).total_seconds()) if total_points >= 2 else 0.0
    return {
        "pure_idle_points": pure_points,
        "total_valid_points": total_points,
        "pure_idle_ratio": ratio,
        "pure_idle_duration_s": pure_idle_duration_s,
        "propulsion_unavailable": not propulsion_available,
        "remove": ratio >= 0.50,
        "kept_point_count": total_points if ratio < 0.50 else 0,
    }


def _longest_stationary_run_s(segment: pd.DataFrame) -> float:
    frame = segment.sort_values("timestamp", kind="stable").reset_index(drop=True)
    stationary = pd.to_numeric(frame["speed_aligned_kn"], errors="coerce").le(0.1)
    interval = find_contiguous_intervals(frame, long_gap_threshold_s=LONG_GAP_THRESHOLD_S)
    groups = (stationary.ne(stationary.shift(fill_value=False)) | interval.ne(interval.shift(fill_value=interval.iloc[0]))).cumsum()
    durations = [float((run.timestamp.iloc[-1] - run.timestamp.iloc[0]).total_seconds()) for _, run in frame.loc[stationary].groupby(groups[stationary], sort=False)]
    return max(durations, default=0.0)


def _baseline_snapshot(output_root: Path) -> tuple[dict[str, object], pd.DataFrame]:
    relative_root = output_root.resolve().relative_to(REPO_ROOT).as_posix()
    def baseline_text(name: str) -> str:
        result = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{relative_root}/{name}"],
            cwd=REPO_ROOT, capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"cannot read required baseline {BASELINE_COMMIT}:{name}: {result.stderr.decode(errors='replace').strip()}")
        return result.stdout.decode("utf-8-sig")
    summary = json.loads(baseline_text("qa_summary.json"))
    shore = pd.read_csv(StringIO(baseline_text("shore_intervals.csv")))
    baseline_manifest = pd.read_csv(StringIO(baseline_text("split_manifest.csv")))
    summary.setdefault("split", {})["point_counts"] = baseline_manifest.groupby("split")["num_1s_points"].sum().to_dict()
    if shore.empty or "mean_battery_kw" not in shore:
        return summary, pd.DataFrame()
    return summary, shore.loc[pd.to_numeric(shore["mean_battery_kw"], errors="coerce").abs().lt(1.0)].copy()


def _load_current_aligned_30s(output_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, object], list[dict[str, object]]]:
    """Load the frozen baseline aligned 30 s input without device CSV access.

    Reading from the named baseline commit makes repeated builds idempotent:
    a prior rebuilt output can never become this run's input.
    """
    relative_root = output_root.resolve().relative_to(REPO_ROOT).as_posix()
    raw_prefix = f"{relative_root}/raw_30s_total_load_by_voyage/"
    listing = subprocess.run(
        ["git", "-c", "core.quotePath=false", "ls-tree", "-r", "--name-only", BASELINE_COMMIT, "--", f"{relative_root}/raw_30s_total_load_by_voyage"],
        cwd=REPO_ROOT, capture_output=True, check=False,
    )
    if listing.returncode:
        raise RuntimeError(f"cannot list baseline aligned 30 s input: {listing.stderr.decode(errors='replace').strip()}")
    source_paths = sorted(
        [line.strip() for line in listing.stdout.decode("utf-8").splitlines() if line.strip().startswith(raw_prefix) and line.strip().endswith("/power_30s.csv")],
        key=lambda value: _sort_key(Path(value).parent),
    )
    if not source_paths:
        raise RuntimeError(f"missing baseline aligned 30 s input under {raw_prefix}")
    required = {"timestamp", "fc_total_kw", "battery_total_kw", "load_total_kw", "speed_aligned_kn", "aligned"}
    frames: dict[str, pd.DataFrame] = {}
    alignment: list[dict[str, object]] = []
    for source_path in source_paths:
        parent_name = Path(source_path).parent.name
        content = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{source_path}"], cwd=REPO_ROOT, capture_output=True, check=False,
        )
        if content.returncode:
            raise RuntimeError(f"cannot read baseline aligned power file: {source_path}")
        frame = pd.read_csv(StringIO(content.stdout.decode("utf-8-sig")))
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(f"aligned power file missing {sorted(missing)}: {source_path}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        for column in ("fc_total_kw", "battery_total_kw", "load_total_kw", "speed_aligned_kn", "soc_mean_pct", "propulsion_inverter_kw"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["aligned"] = frame["aligned"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
        frame["parent_voyage"] = parent_name
        frames[parent_name] = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        alignment.append({"parent_voyage": parent_name, "matched_rows": int(len(frame)), "fully_aligned_rows": int(frame["aligned"].sum()), "unaligned_rows": int((~frame["aligned"]).sum()), "source": f"baseline_commit:{BASELINE_COMMIT}"})
    gaps = pd.concat([frame["timestamp"].diff().dt.total_seconds() for frame in frames.values()], ignore_index=True).dropna()
    normal = gaps[gaps.between(20.0, 40.0)]
    if normal.empty:
        raise RuntimeError("current aligned 30 s input has no normal 20-40 s cadence")
    return frames, {"median_power_cadence_s": float(normal.median()), "long_gap_threshold_s": LONG_GAP_THRESHOLD_S, "source": f"baseline_commit:{BASELINE_COMMIT}"}, alignment


def _segments(frame: pd.DataFrame) -> list[pd.DataFrame]:
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    starts=[]; current=[]; previous=None
    for index, row in frame.iterrows():
        gap = previous is not None and (pd.Timestamp(row.timestamp)-previous).total_seconds() > LONG_GAP_THRESHOLD_S
        usable = (
            bool(row.aligned)
            and not bool(row.is_shore)
            and not bool(row.is_external_charging)
            and not bool(row.is_abnormal)
            and not bool(row.is_physical_inconsistency)
        )
        if current and (gap or not usable): starts.append(frame.loc[current].copy()); current=[]
        if usable: current.append(index)
        previous=pd.Timestamp(row.timestamp)
    if current: starts.append(frame.loc[current].copy())
    return [item for item in starts if len(item) >= 2]


def _stats(frame: pd.DataFrame) -> dict[str, object]:
    values = pd.to_numeric(frame.load_total_kw, errors="coerce")
    return {"count": int(values.notna().sum()), "min_kw": float(values.min()), "max_kw": float(values.max()), "mean_kw": float(values.mean()), "zero_count": int(values.eq(0).sum()), "negative_count": int(values.lt(0).sum()), "under_1_kw_count": int(values.lt(1).sum())}


def _negative_rows(one_second: pd.DataFrame, cleaned: pd.DataFrame, parent: str, identifier: str) -> list[dict[str, object]]:
    negative=one_second.load_total_kw.lt(0); groups=negative.ne(negative.shift(fill_value=False)).cumsum(); rows=[]
    for _, run in one_second.loc[negative].groupby(groups[negative], sort=False):
        source=cleaned.loc[cleaned.timestamp.between(run.timestamp.iloc[0], run.timestamp.iloc[-1])]
        source_negative=source.loc[source.load_total_kw.lt(0)]
        tolerated = len(source_negative) <= 2 and source_negative.load_total_kw.abs().le(1.0).all()
        explanation = (
            "numerical_tolerance: at most two source samples and absolute load at most 1 kW"
            if tolerated else "unexpected_remaining_negative: investigate source interval before formal acceptance"
        )
        rows.append({"parent_voyage":parent,"segment_id":identifier,"start_time":run.timestamp.iloc[0].isoformat(),"end_time":run.timestamp.iloc[-1].isoformat(),"duration_s":int(len(run)-1),"min_load_kw":float(run.load_total_kw.min()),"source_negative_points":int(len(source_negative)),"source_min_load_kw":float(source_negative.load_total_kw.min()) if not source_negative.empty else np.nan,"mean_fc_kw":float(source.fc_total_kw.mean()) if not source.empty else np.nan,"mean_battery_kw":float(source.battery_total_kw.mean()) if not source.empty else np.nan,"speed_min_kn":float(source.speed_aligned_kn.min()) if source.speed_aligned_kn.notna().any() else np.nan,"speed_max_kn":float(source.speed_aligned_kn.max()) if source.speed_aligned_kn.notna().any() else np.nan,"explanation":explanation})
    return rows


def build_dataset(raw_root: Path = RAW_ROOT, output_root: Path = OUTPUT_ROOT) -> dict[str, object]:
    raw_root, output_root = Path(raw_root), Path(output_root)
    baseline_summary, baseline_low_shore = _baseline_snapshot(output_root)
    raw_by_parent, timing, alignment = _load_current_aligned_30s(output_root)
    parent_names = sorted(raw_by_parent, key=lambda name: _sort_key(Path(name)))
    shore_policy=_shore_policy(pd.concat(raw_by_parent.values(),ignore_index=True),timing)
    staging=output_root.with_name(f"{output_root.name}.staging")
    if staging.exists(): shutil.rmtree(staging)
    if output_root.exists():
        if output_root.resolve() != (REPO_ROOT/"data"/"processed"/"operating_segments_1s_rebuilt").resolve(): raise ValueError("unexpected output path")
    staging.mkdir(parents=True); raw_dir=staging/"raw_30s_total_load_by_voyage"; clean_dir=staging/"cleaned_30s_segments"; one_dir=staging/"operating_segments_1s"; raw_dir.mkdir(); clean_dir.mkdir(); one_dir.mkdir()
    shore_rows=[]; physical_rows=[]; negative_rows=[]; manifest_rows=[]; exclusion_rows=[]; pure_idle_rows=[]; retained_idle_rows=[]; kept_by_parent={}; marked_by_parent={}; segment=0; candidate_segment=0
    try:
        zero_drift_audits=[]
        for parent in parent_names:
            marked, zero_drift = _zero_small_stationary_load_drift(raw_by_parent[parent])
            zero_drift["parent_voyage"] = parent
            zero_drift_audits.append(zero_drift)
            marked,shore=select_shore_intervals(marked,**{key:shore_policy[key] for key in ("fc_idle_threshold_kw","speed_idle_threshold_kn","battery_charge_threshold_kw","minimum_shore_points","long_gap_threshold_s")})
            if not shore.empty:
                shore["reason"] = "external_charging_or_shore_power: stationary battery charging beyond 1 kW deadband"
            marked,external,physical=_classify_negative_intervals(marked,speed_idle_threshold_kn=float(shore_policy["speed_idle_threshold_kn"]),battery_charging_threshold_kw=float(shore_policy["battery_deadband_kw"]))
            marked,abnormal,status=_mark_abnormal(marked,shore_policy)
            if (marked["is_load_zero_drift"] & (marked["is_shore"] | marked["is_external_charging"] | marked["is_physical_inconsistency"] | marked["is_abnormal"])).any():
                raise RuntimeError(f"zero load drift was incorrectly classified as excluded: {parent}")
            marked["battery_state"] = battery_state(marked["battery_total_kw"], deadband_kw=float(shore_policy["battery_deadband_kw"]))
            marked_by_parent[parent] = marked.copy()
            if not shore.empty: shore.insert(0,"parent_voyage",parent); shore_rows.extend(shore.to_dict("records"))
            if not external.empty: external.insert(0,"parent_voyage",parent); shore_rows.extend(external.to_dict("records"))
            if not physical.empty: physical.insert(0,"parent_voyage",parent); physical_rows.extend(physical.to_dict("records"))
            if not abnormal.empty:
                abnormal.insert(0,"parent_voyage",parent)
                abnormal["raw_points"] = abnormal["abnormal_rows"]
                physical_rows.extend(abnormal.to_dict("records"))
            target=raw_dir/parent; target.mkdir()
            raw_cols=["timestamp","fc_total_kw","battery_total_kw","load_total_kw","speed_aligned_kn","speed_source","soc_mean_pct","propulsion_inverter_kw","battery_state","aligned","is_load_zero_drift","is_shore","is_external_charging","is_abnormal","is_physical_inconsistency","parent_voyage"]
            marked[raw_cols].to_csv(target/"power_30s.csv",index=False,encoding="utf-8-sig")
            exclusion_rows.append({"parent_voyage":parent,"unaligned_rows":int((~marked.aligned).sum()),"shore_rows":int(marked.is_shore.sum()),"external_charging_rows":int(marked.is_external_charging.sum()),"physical_inconsistency_rows":int(marked.is_physical_inconsistency.sum()),"abnormal_rows":int(marked.is_abnormal.sum()),"abnormal_status":status})
            kept_by_parent[parent] = []
            for cleaned in _segments(marked):
                candidate_segment += 1
                candidate_id = f"candidate_segment_{candidate_segment:04d}"
                idle = _pure_idle_summary(cleaned)
                duration_s = float((cleaned.timestamp.iloc[-1] - cleaned.timestamp.iloc[0]).total_seconds())
                if bool(idle["remove"]):
                    pure_idle_rows.append({"parent_voyage":parent,"segment_id":candidate_id,"start_time":cleaned.timestamp.iloc[0].isoformat(),"end_time":cleaned.timestamp.iloc[-1].isoformat(),"duration_s":duration_s,"pure_idle_duration_s":idle["pure_idle_duration_s"],"pure_idle_ratio":idle["pure_idle_ratio"],"mean_speed":float(pd.to_numeric(cleaned.speed_aligned_kn,errors="coerce").mean()),"mean_fc_kw":float(cleaned.fc_total_kw.mean()),"mean_load_kw":float(cleaned.load_total_kw.mean()),"propulsion_unavailable":idle["propulsion_unavailable"],"reason":"predominantly_pure_idle"})
                    continue
                segment+=1; identifier=f"operating_segment_{segment:04d}"; cleaned=cleaned[["timestamp","fc_total_kw","battery_total_kw","load_total_kw","speed_aligned_kn","speed_source","soc_mean_pct","propulsion_inverter_kw","parent_voyage"]].copy(); cleaned["segment_id"]=identifier
                kept_by_parent[parent].append(cleaned[["timestamp"]].copy())
                stationary_run_s = _longest_stationary_run_s(cleaned)
                if duration_s > 3600.0 and stationary_run_s >= 600.0 and float(idle["pure_idle_ratio"]) < 0.50:
                    retained_idle_rows.append({"parent_voyage":parent,"segment_id":identifier,"duration_s":duration_s,"longest_stationary_run_s":stationary_run_s,"pure_idle_ratio":idle["pure_idle_ratio"],"mean_load_kw":float(cleaned.load_total_kw.mean())})
                cleaned.to_csv(clean_dir/f"{identifier}.csv",index=False,encoding="utf-8-sig")
                one,qa=pchip_to_one_second(cleaned); one.to_csv(one_dir/f"{identifier}.csv",index=False,encoding="utf-8-sig"); negative_rows.extend(_negative_rows(one,cleaned,parent,identifier))
                manifest_rows.append({"parent_voyage":parent,"segment_id":identifier,"start_time":cleaned.timestamp.iloc[0].isoformat(),"end_time":cleaned.timestamp.iloc[-1].isoformat(),"duration_s":int((cleaned.timestamp.iloc[-1]-cleaned.timestamp.iloc[0]).total_seconds()),"num_1s_points":len(one),"raw_min_kw":float(cleaned.load_total_kw.min()),"raw_max_kw":float(cleaned.load_total_kw.max()),"pchip_min_kw":qa["pchip_min_kw"],"pchip_max_kw":float(one.load_total_kw.max()),"raw_csv":f"cleaned_30s_segments/{identifier}.csv","one_second_csv":f"operating_segments_1s/{identifier}.csv"})
        manifest=pd.DataFrame(manifest_rows)
        if manifest.empty: raise RuntimeError("no operating segments")
        parents=sorted(manifest.parent_voyage.unique(),key=lambda name:_sort_key(Path(name))); splits=chronological_parent_splits(parents)
        manifest["split"]=np.select([manifest.parent_voyage.isin(splits["train"]),manifest.parent_voyage.isin(splits["validation"])],["train","validation"],default="test")
        recovery_rows=[]
        for _, baseline in baseline_low_shore.iterrows():
            parent=str(baseline["parent_voyage"]); start=pd.Timestamp(baseline["start_time"]); end=pd.Timestamp(baseline["end_time"])
            marked=marked_by_parent.get(parent,pd.DataFrame()); kept=pd.concat(kept_by_parent.get(parent,[]),ignore_index=True) if kept_by_parent.get(parent) else pd.DataFrame(columns=["timestamp"])
            source=marked.loc[marked.timestamp.between(start,end)] if not marked.empty else pd.DataFrame()
            entered=kept.loc[kept.timestamp.between(start,end)] if not kept.empty else pd.DataFrame()
            recovered=not source.empty and not bool(source.is_external_charging.any())
            entered_duration=float((entered.timestamp.iloc[-1]-entered.timestamp.iloc[0]).total_seconds()) if len(entered)>=2 else 0.0
            recovery_rows.append({"parent_voyage":parent,"start_time":start.isoformat(),"end_time":end.isoformat(),"baseline_duration_s":float(baseline["duration_s"]),"baseline_mean_battery_kw":float(baseline["mean_battery_kw"]),"recovered_from_shore":recovered,"final_normal_points":int(len(entered)),"final_normal_duration_s":entered_duration})
        recovery=pd.DataFrame(recovery_rows)
        pure_columns=["parent_voyage","segment_id","start_time","end_time","duration_s","pure_idle_duration_s","pure_idle_ratio","mean_speed","mean_fc_kw","mean_load_kw","propulsion_unavailable","reason"]
        audit_columns=["parent_voyage","segment_id","duration_s","longest_stationary_run_s","pure_idle_ratio","mean_load_kw"]
        manifest.to_csv(staging/"split_manifest.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(shore_rows).to_csv(staging/"shore_intervals.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(physical_rows).to_csv(staging/"abnormal_intervals.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(negative_rows).to_csv(staging/"negative_load_intervals.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(exclusion_rows).to_csv(staging/"exclusion_qa.csv",index=False,encoding="utf-8-sig"); pd.json_normalize(alignment,sep=".").to_csv(staging/"alignment_qa_by_voyage.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(pure_idle_rows,columns=pure_columns).to_csv(staging/"pure_idle_removed_segments.csv",index=False,encoding="utf-8-sig"); pd.DataFrame(retained_idle_rows,columns=audit_columns).sample(n=min(5,len(retained_idle_rows)),random_state=0).to_csv(staging/"retained_midsegment_idle_audit.csv",index=False,encoding="utf-8-sig"); recovery.to_csv(staging/"shore_deadband_recovery.csv",index=False,encoding="utf-8-sig")
        pd.DataFrame(zero_drift_audits).to_csv(staging/"load_zero_drift_qa.csv",index=False,encoding="utf-8-sig")
        raw_values=pd.concat([value[["load_total_kw"]] for value in marked_by_parent.values()],ignore_index=True); clean_values=pd.concat([pd.read_csv(path,usecols=["load_total_kw"]) for path in clean_dir.glob("*.csv")],ignore_index=True); one_values=pd.concat([pd.read_csv(path,usecols=["load_total_kw"]) for path in one_dir.glob("*.csv")],ignore_index=True)
        counts=manifest.groupby("parent_voyage").segment_id.count()
        remaining_substantive=[row for row in negative_rows if row["min_load_kw"] < -1.0 and row["duration_s"] > 1]
        recovered = recovery.loc[recovery.recovered_from_shore] if not recovery.empty else recovery
        baseline_pchip=baseline_summary.get("pchip_1s",{})
        split_points=manifest.groupby("split").num_1s_points.sum().to_dict()
        baseline_split_points=baseline_summary.get("split",{}).get("point_counts",{})
        removed_durations=[float(row["duration_s"]) for row in pure_idle_rows]
        summary={"raw_voyage_count":len(parent_names),"raw_channel_count":len(parent_names)*20,"raw_total_rows":int(sum(len(value) for value in raw_by_parent.values())),"timestamp_policy":timing,"shore_policy":shore_policy,"load_zero_drift":{"point_count":int(sum(int(row["point_count"]) for row in zero_drift_audits)),"by_parent_file":"load_zero_drift_qa.csv","excluded_point_count":0},"battery_deadband":{"deadband_kw":1.0,"baseline_low_battery_shore_interval_count":int(len(baseline_low_shore)),"recovered_interval_count":int(len(recovered)),"recovered_total_duration_s":float(recovered.baseline_duration_s.sum()) if not recovered.empty else 0.0,"recovered_final_normal_duration_s":float(recovered.final_normal_duration_s.sum()) if not recovered.empty else 0.0},"alignment":alignment,"raw_load":_stats(raw_values),"cleaned_30s":{**_stats(clean_values),"segment_count":len(manifest),"total_duration_s":int(manifest.duration_s.sum())},"pchip_1s":{**_stats(one_values),"segment_count":len(manifest),"total_points":len(one_values),"total_duration_s":int(len(one_values)-len(manifest)),"nonphysical_overshoot":False},"external_charging":{"interval_count":len(shore_rows),"deleted_rows":int(sum(row["raw_points"] for row in shore_rows)),"total_duration_s":float(sum(row["duration_s"] for row in shore_rows))},"physical_inconsistency":{"interval_count":len(physical_rows),"deleted_rows":int(sum(row["raw_points"] for row in physical_rows)),"total_duration_s":float(sum(row["duration_s"] for row in physical_rows)),"unavailable_statuses":sorted(set(row["abnormal_status"] for row in exclusion_rows if str(row["abnormal_status"]).startswith("unavailable")))},"shore":{"interval_count":len(shore_rows),"deleted_rows":int(sum(row["raw_points"] for row in shore_rows)),"total_duration_s":float(sum(row["duration_s"] for row in shore_rows))},"abnormal":{"interval_count":len(physical_rows),"deleted_rows":int(sum(row["raw_points"] for row in physical_rows)),"total_duration_s":float(sum(row["duration_s"] for row in physical_rows))},"pure_idle":{"removed_segment_count":len(pure_idle_rows),"removed_total_duration_s":float(sum(removed_durations)),"maximum_removed_ratio":float(max((row["pure_idle_ratio"] for row in pure_idle_rows),default=0.0)),"removed_duration_quantiles_s":pd.Series(removed_durations,dtype=float).quantile([0,.5,.9,1]).to_dict(),"retained_long_midsegment_idle_audit_count":min(5,len(retained_idle_rows))},"negative_load":{"interval_count":len(negative_rows),"point_count":int(one_values.load_total_kw.lt(0).sum()),"under_minus_1_point_count":int(one_values.load_total_kw.lt(-1).sum()),"remaining_substantive_interval_count":len(remaining_substantive),"remaining_substantive_intervals":remaining_substantive,"all_intervals_written":True},"baseline_comparison":{"baseline_commit":BASELINE_COMMIT,"delta_1s_points":int(len(one_values)-int(baseline_pchip.get("total_points",0))),"delta_duration_s":int((len(one_values)-len(manifest))-int(baseline_pchip.get("total_duration_s",0))),"delta_segment_count":int(len(manifest)-int(baseline_summary.get("cleaned_30s",{}).get("segment_count",0))),"delta_shore_interval_count":int(len(shore_rows)-int(baseline_summary.get("shore",{}).get("interval_count",0))),"delta_abnormal_interval_count":int(len(physical_rows)-int(baseline_summary.get("abnormal",{}).get("interval_count",0))),"delta_pure_idle_removed_segment_count":int(len(pure_idle_rows)-int(baseline_summary.get("pure_idle",{}).get("removed_segment_count",0))),"delta_split_1s_points":{key:int(split_points.get(key,0)-int(baseline_split_points.get(key,0))) for key in ("train","validation","test")}},"segment_fragmentation":{"per_parent":counts.to_dict(),"more_than_5":counts[counts.gt(5)].to_dict(),"under_5_min":manifest.loc[manifest.duration_s.lt(300),["parent_voyage","segment_id","duration_s"]].to_dict("records")},"split":{"parent_voyage_counts":{key:len(value) for key,value in splits.items()},"segment_counts":manifest.split.value_counts().to_dict(),"point_counts":split_points,"duration_s":manifest.groupby("split").duration_s.sum().to_dict(),"parent_voyage_overlap_count":0}}
        (staging/"qa_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float)+"\n",encoding="utf-8"); zero_residual_numerical_negatives(staging, require_nonnegative_sources=True)
        if output_root.exists(): shutil.rmtree(output_root)
        staging.replace(output_root); return json.loads((output_root/"qa_summary.json").read_text(encoding="utf-8"))
    except Exception:
        shutil.rmtree(staging,ignore_errors=True); raise


if __name__ == "__main__":
    print(json.dumps(build_dataset(),ensure_ascii=False,indent=2,default=float))
