"""Run the small strict Train-only v2 objective-scale audit from raw parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.final_dataset_source import _read, collapse_channel
from v2.analysis.action_screening import DataSplit, DatasetProvenance
from v2.analysis.train_objective_scale_runner import (
    AuditReadyState,
    run_train_objective_scale_audit,
    select_representative_cases,
)
from v2.contracts import DATASET_VERSION
from v2.data.supervisory_rules import (
    FC_ZERO_TOLERANCE_KW,
    FRESHNESS_CAP_SECONDS,
    LONG_GAP_SECONDS,
    SPEED_ZERO_TOLERANCE_KN,
    OperatingMode,
)
from v2.data.train_supervisory_audit import (
    ParentRawChannels,
    RawChannel,
    RawRecord,
    build_parent_supervisory_states,
)


LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _aware(value: object) -> object:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(LOCAL_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(LOCAL_TIMEZONE)
    return timestamp.to_pydatetime()


def _raw_channel(
    channel_id: str,
    frame: pd.DataFrame,
    value_columns: tuple[str, ...],
    source_paths: tuple[str, ...],
) -> RawChannel:
    records = []
    for row_number, row in frame.iterrows():
        values = tuple(float(row[column]) for column in value_columns)
        if not all(np.isfinite(values)):
            continue
        timestamp = _aware(row["timestamp"])
        records.append(
            RawRecord(
                timestamp,
                values,
                f"{'|'.join(source_paths)}@{timestamp.isoformat()}#collapsed-row-{row_number}",
            )
        )
    return RawChannel(channel_id, tuple(records))


def _read_parent(raw_root: Path, parent: str) -> ParentRawChannels:
    parent_root = raw_root / parent
    if not parent_root.is_dir():
        raise FileNotFoundError(f"missing Train parent directory: {parent_root}")
    fc_channels = []
    battery_channels = []
    for side in ("左", "右"):
        for number in range(1, 5):
            channel_id = f"{side}氢燃料电池#{number}"
            source_rows: list[dict[str, object]] = []
            frame, _ = _read(
                parent_root / "燃料电池系统",
                channel_id,
                {"发电功率(kW)": "power_kw"},
                source_rows,
            )
            fc_channels.append(
                _raw_channel(
                    channel_id,
                    frame,
                    ("power_kw",),
                    tuple(str(row["path"]) for row in source_rows),
                )
            )
        for number in range(1, 7):
            channel_id = f"{side}电池簇{number}"
            source_rows = []
            frame, _ = _read(
                parent_root / "BMS",
                channel_id,
                {
                    "总电压(V)": "voltage_v",
                    "总电流(A)": "current_a",
                    "SOC(%)": "soc_pct",
                },
                source_rows,
            )
            frame["power_kw"] = -(frame["voltage_v"] * frame["current_a"]) / 1000.0
            frame["soc"] = frame["soc_pct"] / 100.0
            battery_channels.append(
                _raw_channel(
                    channel_id,
                    frame,
                    ("power_kw", "soc"),
                    tuple(str(row["path"]) for row in source_rows),
                )
            )

    ais_paths = sorted((parent_root / "推进系统").glob("AIS航速_*.csv"))
    if not ais_paths:
        speed = RawChannel("ais-speed", ())
    else:
        frames = tuple(pd.read_csv(path, encoding="utf-8-sig") for path in ais_paths)
        raw = pd.concat(frames, ignore_index=True).rename(
            columns={"Time": "timestamp", "航速(节)": "speed_kn"}
        )
        raw["speed_kn"] = pd.to_numeric(
            raw["speed_kn"].astype(str).str.replace(r"\s*kn$", "", regex=True),
            errors="coerce",
        )
        collapsed, _ = collapse_channel(raw, ["speed_kn"])
        speed = _raw_channel(
            "ais-speed",
            collapsed,
            ("speed_kn",),
            tuple(str(path) for path in ais_paths),
        )
    return ParentRawChannels(
        parent,
        tuple(fc_channels),
        tuple(battery_channels),
        speed,
    )


def _provenance_digest(
    train_manifest: pd.DataFrame,
    source_inventory: pd.DataFrame,
) -> str:
    parents = tuple(train_manifest["parent"].astype(str))
    sources = source_inventory.loc[
        source_inventory["path"].astype(str).map(
            lambda value: any(f"\\{parent}\\" in value for parent in parents)
        ),
        ["path", "sha256"],
    ].sort_values("path")
    payload = {
        "dataset_version": DATASET_VERSION,
        "parents": parents,
        "sources": sources.to_dict("records"),
        "rules": {
            "freshness_cap_seconds": FRESHNESS_CAP_SECONDS,
            "speed_zero_tolerance_kn": SPEED_ZERO_TOLERANCE_KN,
            "fc_zero_tolerance_kw": FC_ZERO_TOLERANCE_KW,
            "long_gap_seconds": LONG_GAP_SECONDS,
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def run(raw_root: Path, metadata_root: Path) -> dict[str, object]:
    manifest = pd.read_csv(metadata_root / "parent_split_manifest.csv")
    train_manifest = manifest.loc[
        manifest["split"].astype(str).str.casefold().eq("train")
    ].copy()
    if train_manifest.empty:
        raise ValueError("no manifest-declared Train parents")

    all_states = []
    ready_states: list[AuditReadyState] = []
    exact_duplicates = 0
    conflicts = 0
    for parent in train_manifest["parent"].astype(str):
        parent_result = build_parent_supervisory_states(_read_parent(raw_root, parent))
        exact_duplicates += parent_result.exact_duplicate_rows_removed
        conflicts += parent_result.conflicting_duplicate_count
        all_states.extend(parent_result.states)

        history: list[float] = []
        previous_state = None
        for state in parent_result.states:
            contiguous_sailing = (
                state.mode is OperatingMode.SAILING_ISLAND
                and state.p_load_kw is not None
                and previous_state is not None
                and previous_state.mode is OperatingMode.SAILING_ISLAND
                and previous_state.p_load_kw is not None
                and not state.long_gap_contaminated
            )
            if not contiguous_sailing:
                history = []
            if contiguous_sailing and state.audit_eligible and history:
                ready_states.append(
                    AuditReadyState(
                        parent,
                        state.timestamp,
                        state.p_load_kw,
                        state.soc_system,
                        state.previous_p_fc_total_kw,
                        tuple(history),
                        abs(state.p_load_kw - previous_state.p_load_kw),
                    )
                )
            if state.mode is OperatingMode.SAILING_ISLAND and state.p_load_kw is not None:
                history.append(state.p_load_kw)
            previous_state = state

    source_inventory = pd.read_csv(metadata_root / "source_files.csv")
    provenance = DatasetProvenance(
        DATASET_VERSION,
        f"sha256:{_provenance_digest(train_manifest, source_inventory)}",
        DataSplit.TRAIN,
    )
    cases = select_representative_cases(tuple(ready_states))
    if not cases:
        raise ValueError("strict Train rules produced no representative MPC cases")
    result = run_train_objective_scale_audit(tuple(ready_states), provenance)

    mode_counts = {
        mode.value: sum(state.mode is mode for state in all_states)
        for mode in OperatingMode
    }
    soc_band_counts = {
        "below_0.4": sum(state.soc_system < 0.4 for state in ready_states),
        "within_0.4_0.6": sum(0.4 <= state.soc_system <= 0.6 for state in ready_states),
        "above_0.6": sum(state.soc_system > 0.6 for state in ready_states),
    }
    return {
        "readiness": "YES",
        "train_parent_count": int(len(train_manifest)),
        "supervisory_state_count": len(all_states),
        "mode_counts": mode_counts,
        "audit_ready_state_count": len(ready_states),
        "audit_ready_parent_count": len({state.parent_id for state in ready_states}),
        "soc_band_counts": soc_band_counts,
        "representative_case_count": len(cases),
        "representative_case_ids": tuple(case.case_id for case in cases),
        "exact_duplicate_rows_removed": exact_duplicates,
        "conflicting_duplicate_timestamps": conflicts,
        "provenance_id": provenance.provenance_id,
        "objective_statistics": result.objective_statistics,
        "soc_active": result.soc_active,
        "active_p95": result.active_p95,
        "scale_ratio": result.scale_ratio,
        "status": result.status,
        "dominance_statistics": result.dominance_statistics,
        "result_digest": result.digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "operating_dataset_final" / "metadata",
    )
    arguments = parser.parse_args()
    summary = run(arguments.raw_root.resolve(), arguments.metadata_root.resolve())
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
