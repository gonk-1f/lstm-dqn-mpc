"""Train-only evidence construction for the v2 DQN state audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from ..data.formal_training_dataset import FormalEpisode
from ..data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from ..data.train_supervisory_audit import ParentSupervisoryState
from ..data.train_supervisory_audit import (
    ParentRawChannels,
    causal_align_latest_without_reuse,
    resolve_duplicates,
)
from ..dqn.state import (
    FORMAL_STATE_FEATURE_NAMES,
    FORMAL_STATE_SCHEMA_DIGEST,
    FORMAL_STATE_SCHEMA_VERSION,
    FORMAL_STATE_SPEED_SCALE_KN,
)
from ..models.battery_degradation import (
    BATTERY_LIFETIME_Q_AH,
    BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
    BATTERY_NOMINAL_VOLTAGE_V,
    battery_degradation_step,
)
from ..models.fuel_cell_degradation import (
    FC_AGGREGATE_TO_REFERENCE_POWER_RATIO,
    FC_EOL_VOLTAGE_LOSS_UV,
    FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR,
    FC_START_STOP_LOSS_UV_PER_CYCLE,
    FC_TRANSIENT_LOSS_UV_PER_DELTA_KW,
)


ACTIVE_DATASET_VERSION = "operating_dataset_zero_boundary_v2"
AUDIT_SAMPLE_SECONDS = 30.0
AUDIT_HISTORY_SECONDS = 150.0
AUDIT_TAU_LPF_SECONDS = 90.0
AUDIT_POWER_SCALE_KW = 600.0
AUDIT_BATTERY_POWER_SCALE_KW = 1248.0
NEAR_ZERO_VARIANCE_STD = 1.0e-8

CANDIDATE_NORMALIZED_FEATURES = (
    "soc",
    "fuel_cell_power_fraction",
    "previous_fuel_cell_power_fraction",
    "battery_power_fraction",
    "load_power_fraction",
    "recent_load_mean_fraction",
    "recent_load_population_std_fraction",
    "recent_load_window_trend_fraction",
    "causal_base_load_fraction",
    "recent_delta_soc",
)
PROPOSED_NORMALIZED_FEATURES = FORMAL_STATE_FEATURE_NAMES


@dataclass(frozen=True)
class TrainSegment:
    parent: str
    sample_id: str
    relative_path: str
    start_timestamp: datetime
    end_timestamp: datetime
    sha256: str
    dataset_version: str


@dataclass(frozen=True)
class AuditFeatureRow:
    parent: str
    sample_id: str
    timestamp: datetime
    soc: float
    fc_power_kw: float
    previous_fc_power_kw: float
    battery_power_kw: float
    load_power_kw: float
    recent_load_mean_kw: float
    recent_load_population_std_kw: float
    recent_load_trend_kw_per_s: float
    base_load_kw: float
    recent_delta_soc: float
    delta_load_kw: float
    delta_fc_kw: float
    measured_power_balance_residual_kw: float
    history_sample_count: int
    speed_kn: float = 0.0


@dataclass(frozen=True)
class AuditSupervisorySample:
    """One complete FC/BMS snapshot observed at its latest channel arrival."""

    parent_id: str
    timestamp: datetime
    fc_power_kw: float
    battery_power_kw: float
    soc: float
    previous_fc_power_kw: float | None

    @property
    def audit_eligible(self) -> bool:
        return self.previous_fc_power_kw is not None

    @property
    def p_fc_total_kw(self) -> float:
        return self.fc_power_kw

    @property
    def p_batt_total_kw(self) -> float:
        return self.battery_power_kw

    @property
    def soc_system(self) -> float:
        return self.soc

    @property
    def previous_p_fc_total_kw(self) -> float | None:
        return self.previous_fc_power_kw


def assemble_audit_supervisory_samples(
    channels: ParentRawChannels,
) -> tuple[AuditSupervisorySample, ...]:
    """Align complete FC/BMS cycles and timestamp them at latest arrival.

    A channel selected up to ten seconds after the first-FC reference is not
    future information at the returned snapshot: the snapshot time is the
    latest selected channel arrival. AIS is intentionally excluded because it
    is not an input to the audited state and formal Train boundaries already
    define the operating interval.
    """

    if type(channels) is not ParentRawChannels:
        raise TypeError("channels must be an exact ParentRawChannels")
    from ..data.segment_power_source import (
        ALIGNMENT_TOLERANCE_SECONDS,
        align_near_synchronous_cycles,
    )

    raw_channels = channels.fuel_cell_channels + channels.battery_channels
    resolved = tuple(resolve_duplicates(channel.records) for channel in raw_channels)
    reference = tuple(record.timestamp for record in resolved[0].records)
    if not reference:
        return ()
    alignment = align_near_synchronous_cycles(
        reference,
        tuple(
            tuple(record.timestamp for record in resolution.records)
            for resolution in resolved
        ),
        tolerance_seconds=ALIGNMENT_TOLERANCE_SECONDS,
        max_channel_span_seconds=ALIGNMENT_TOLERANCE_SECONDS,
    )
    samples: list[AuditSupervisorySample] = []
    previous_fc: float | None = None
    previous_timestamp: datetime | None = None
    for position, snapshot in zip(
        alignment.valid_reference_positions,
        alignment.snapshot_timestamps,
    ):
        selected = tuple(
            resolution.records[alignment.channel_indices[index][position]]
            for index, resolution in enumerate(resolved)
            if alignment.channel_indices[index][position] is not None
        )
        if len(selected) != 20:
            continue
        fc = math.fsum(record.values[0] for record in selected[:8])
        battery = math.fsum(record.values[0] for record in selected[8:20])
        soc = math.fsum(record.values[1] for record in selected[8:20]) / 12.0
        if not (math.isfinite(fc) and math.isfinite(battery) and 0.0 <= soc <= 1.0):
            continue
        usable_previous = previous_fc
        if (
            previous_timestamp is None
            or (snapshot - previous_timestamp).total_seconds() > 45.0
        ):
            usable_previous = None
        samples.append(
            AuditSupervisorySample(
                parent_id=channels.parent_id,
                timestamp=snapshot,
                fc_power_kw=float(fc),
                battery_power_kw=float(battery),
                soc=float(soc),
                previous_fc_power_kw=usable_previous,
            )
        )
        previous_fc = float(fc)
        previous_timestamp = snapshot
    return tuple(samples)


def align_battery_soc_to_timestamps(
    channels: ParentRawChannels,
    target_timestamps: Sequence[datetime],
) -> tuple[float | None, ...]:
    """Causally align all twelve cluster SOC values without source-row reuse."""

    if type(channels) is not ParentRawChannels:
        raise TypeError("channels must be an exact ParentRawChannels")
    targets = tuple(target_timestamps)
    if any(type(value) is not datetime for value in targets):
        raise TypeError("target_timestamps must contain exact datetime values")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("target_timestamps must be strictly increasing")
    resolved = tuple(
        resolve_duplicates(channel.records) for channel in channels.battery_channels
    )
    aligned = tuple(
        causal_align_latest_without_reuse(
            targets,
            tuple(record.timestamp for record in resolution.records),
        )
        for resolution in resolved
    )
    values: list[float | None] = []
    for position in range(len(targets)):
        indices = tuple(mapping[position] for mapping in aligned)
        if any(index is None for index in indices):
            values.append(None)
            continue
        records = tuple(
            resolution.records[index]
            for resolution, index in zip(resolved, indices)
            if index is not None
        )
        if len(records) != 12 or any(len(record.values) != 2 for record in records):
            values.append(None)
            continue
        soc = math.fsum(record.values[1] for record in records) / 12.0
        values.append(float(soc) if math.isfinite(soc) and 0.0 <= soc <= 1.0 else None)
    return tuple(values)


def build_formal_episode_feature_rows(
    episode: FormalEpisode,
    soc_values: Sequence[float | None],
) -> tuple[AuditFeatureRow, ...]:
    """Build measured proxy rows on the exact shore-aware formal S8 axis."""

    if type(episode) is not FormalEpisode or episode.split != "train":
        raise TypeError("episode must be an exact Train FormalEpisode")
    soc = tuple(soc_values)
    if len(soc) != episode.step_count:
        raise ValueError("soc_values must match the formal episode axis")
    alpha = math.exp(-AUDIT_SAMPLE_SECONDS / AUDIT_TAU_LPF_SECONDS)
    base_load: float | None = None
    history: list[dict[str, object]] = []
    rows: list[AuditFeatureRow] = []
    for index, mode_value in enumerate(episode.operating_mode):
        mode = OperatingMode(mode_value)
        if mode is not OperatingMode.ONBOARD:
            base_load = None
            history = []
            continue
        timestamp = episode.timestamp[index]
        if timestamp.tzinfo is None:
            raise ValueError("formal episode timestamps must be timezone-aware")
        raw_load = float(episode.load_kw[index])
        load = normalize_onboard_load_kw(raw_load)
        base_load = (
            load
            if base_load is None
            else alpha * base_load + (1.0 - alpha) * load
        )
        entry = {
            "timestamp": timestamp.to_pydatetime(),
            "load": load,
            "fc": float(episode.fc_power_kw[index]),
            "soc": soc[index],
            "base": base_load,
        }
        history.append(entry)
        cutoff = timestamp.timestamp() - AUDIT_HISTORY_SECONDS
        history = [
            item
            for item in history
            if item["timestamp"].timestamp() >= cutoff
        ]
        current_soc = soc[index]
        if current_soc is None:
            continue
        loads = tuple(float(item["load"]) for item in history)
        timestamps = tuple(item["timestamp"] for item in history)
        fc = float(entry["fc"])
        previous_fc = float(history[-2]["fc"]) if len(history) >= 2 else fc
        first_soc = next(
            (
                float(item["soc"])
                for item in history
                if item["soc"] is not None
            ),
            float(current_soc),
        )
        trend = _least_squares_trend(timestamps, loads) if len(history) >= 2 else 0.0
        battery = float(episode.battery_bus_kw[index])
        rows.append(
            AuditFeatureRow(
                parent=episode.parent,
                sample_id=episode.sample_id,
                timestamp=timestamp.to_pydatetime(),
                soc=float(current_soc),
                fc_power_kw=fc,
                previous_fc_power_kw=previous_fc,
                battery_power_kw=battery,
                load_power_kw=load,
                recent_load_mean_kw=float(np.mean(np.asarray(loads, dtype=float))),
                recent_load_population_std_kw=float(
                    np.std(np.asarray(loads, dtype=float), ddof=0)
                ),
                recent_load_trend_kw_per_s=trend,
                base_load_kw=float(base_load),
                recent_delta_soc=float(current_soc) - first_soc,
                delta_load_kw=load - float(base_load),
                delta_fc_kw=fc - previous_fc,
                measured_power_balance_residual_kw=raw_load - fc - battery,
                history_sample_count=len(history),
                speed_kn=float(episode.speed_kn[index]),
            )
        )
    return tuple(rows)


def episode_life_upper_bounds(episode: FormalEpisode) -> dict[str, float | int]:
    """Conservatively bound hidden cumulative life within one reset episode."""

    if type(episode) is not FormalEpisode or episode.split != "train":
        raise TypeError("episode must be an exact Train FormalEpisode")
    modes = tuple(OperatingMode(value) for value in episode.operating_mode)
    onboard_steps = sum(mode is OperatingMode.ONBOARD for mode in modes)
    shore_runs = sum(
        mode is not OperatingMode.ONBOARD
        and (index == 0 or modes[index - 1] is OperatingMode.ONBOARD)
        for index, mode in enumerate(modes)
    )
    max_reference_delta_kw = (
        AUDIT_POWER_SCALE_KW * FC_AGGREGATE_TO_REFERENCE_POWER_RATIO
    )
    max_dynamic_uv = (
        FC_START_STOP_LOSS_UV_PER_CYCLE
        + FC_TRANSIENT_LOSS_UV_PER_DELTA_KW * max_reference_delta_kw
    )
    fc_upper_uv = (
        onboard_steps
        * (
            FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR * AUDIT_SAMPLE_SECONDS / 3600.0
            + max_dynamic_uv
        )
        + shore_runs * max_dynamic_uv
    )
    max_current_a = AUDIT_BATTERY_POWER_SCALE_KW * 1000.0 / BATTERY_NOMINAL_VOLTAGE_V
    battery_step = battery_degradation_step(
        0.0,
        max_current_a,
        AUDIT_SAMPLE_SECONDS,
        BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
    )
    battery_upper_ah = episode.step_count * battery_step.weighted_ah
    return {
        "supervisory_step_count": episode.step_count,
        "onboard_step_count": onboard_steps,
        "shore_run_count": shore_runs,
        "fc_voltage_loss_uv_upper_bound": float(fc_upper_uv),
        "fc_raw_life_fraction_upper_bound": float(
            fc_upper_uv / FC_EOL_VOLTAGE_LOSS_UV
        ),
        "battery_weighted_ah_upper_bound": float(battery_upper_ah),
        "battery_raw_life_fraction_upper_bound": float(
            battery_upper_ah / BATTERY_LIFETIME_Q_AH
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative.parts[0] != "train"
        or ".." in relative.parts
    ):
        raise ValueError(f"Train path must remain below train/: {relative_path!r}")
    resolved = (root / Path(*relative.parts)).resolve()
    train_root = (root / "train").resolve()
    try:
        resolved.relative_to(train_root)
    except ValueError as error:
        raise ValueError(
            f"Train path escapes the formal Train directory: {relative_path!r}"
        ) from error
    return resolved


def load_train_segments(dataset_root: str | Path) -> tuple[TrainSegment, ...]:
    """Load and authenticate only the active formal Train segment whitelist."""

    root = Path(dataset_root).resolve()
    metadata = root / "metadata"
    qa_path = metadata / "qa_summary.json"
    manifest_path = metadata / "sample_manifest.csv"
    if not qa_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("formal dataset metadata is incomplete")
    qa = json.loads(qa_path.read_text(encoding="utf-8-sig"))
    version = qa.get("dataset_version")
    if version != ACTIVE_DATASET_VERSION:
        raise ValueError(
            f"state audit requires {ACTIVE_DATASET_VERSION}, received {version!r}"
        )

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {
        "parent",
        "sample_id",
        "relative_path",
        "split",
        "start_timestamp",
        "end_timestamp",
        "sha256",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"formal manifest is missing columns: {sorted(missing)}")
    train = manifest.loc[manifest["split"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("formal manifest has no Train segments")
    if train["sample_id"].isna().any() or train["sample_id"].duplicated().any():
        raise ValueError("Train segment identifiers must be unique and nonempty")

    segments: list[TrainSegment] = []
    for row in train.itertuples(index=False):
        relative_path = str(row.relative_path)
        path = _train_path(root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"formal Train segment is missing: {path}")
        expected = str(row.sha256).lower()
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Train segment SHA-256 mismatch: {relative_path}")
        start = pd.Timestamp(row.start_timestamp)
        end = pd.Timestamp(row.end_timestamp)
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError(f"Train segment has invalid timestamps: {row.sample_id}")
        segments.append(
            TrainSegment(
                parent=str(row.parent),
                sample_id=str(row.sample_id),
                relative_path=relative_path,
                start_timestamp=start.to_pydatetime(),
                end_timestamp=end.to_pydatetime(),
                sha256=actual,
                dataset_version=str(version),
            )
        )
    return tuple(
        sorted(segments, key=lambda item: (item.start_timestamp, item.sample_id))
    )


def _load_lookup(load_frame: pd.DataFrame) -> dict[int, float]:
    required = {"timestamp", "load_total_kw"}
    missing = required.difference(load_frame.columns)
    if missing:
        raise ValueError(f"Train load frame is missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(load_frame["timestamp"], utc=True, errors="raise")
    if timestamps.duplicated().any():
        raise ValueError("Train load timestamps must be unique")
    loads = pd.to_numeric(load_frame["load_total_kw"], errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(loads).all():
        raise ValueError("Train loads must be finite")
    return {
        int(timestamp.value): float(load)
        for timestamp, load in zip(timestamps, loads)
    }


def _timestamp_key(value: datetime) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("supervisory timestamps must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _least_squares_trend(
    timestamps: Sequence[datetime],
    loads: Sequence[float],
) -> float:
    origin = timestamps[0]
    elapsed = np.asarray(
        [(timestamp - origin).total_seconds() for timestamp in timestamps],
        dtype=float,
    )
    values = np.asarray(loads, dtype=float)
    centered_time = elapsed - float(np.mean(elapsed))
    denominator = float(np.dot(centered_time, centered_time))
    if denominator <= 0.0:
        raise ValueError("load trend requires distinct timestamps")
    centered_load = values - float(np.mean(values))
    return float(np.dot(centered_time, centered_load) / denominator)


def build_causal_feature_rows(
    segment: TrainSegment,
    states: Sequence[ParentSupervisoryState | AuditSupervisorySample],
    load_frame: pd.DataFrame,
) -> tuple[AuditFeatureRow, ...]:
    """Build fully observed Train rows using only a segment-local causal window."""

    if type(segment) is not TrainSegment:
        raise TypeError("segment must be an exact TrainSegment")
    checked = tuple(states)
    allowed = (ParentSupervisoryState, AuditSupervisorySample)
    if any(type(state) not in allowed for state in checked):
        raise TypeError("states must contain supported exact supervisory values")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(checked, checked[1:])
    ):
        raise ValueError("supervisory timestamps must be strictly increasing")
    lookup = _load_lookup(load_frame)
    alpha = math.exp(-AUDIT_SAMPLE_SECONDS / AUDIT_TAU_LPF_SECONDS)
    base_load: float | None = None
    history: list[tuple[ParentSupervisoryState, float, float]] = []
    rows: list[AuditFeatureRow] = []
    for state in checked:
        if state.timestamp < segment.start_timestamp:
            continue
        if state.timestamp > segment.end_timestamp:
            break
        if state.parent_id != segment.parent:
            raise ValueError("supervisory parent does not match Train segment")
        key = _timestamp_key(state.timestamp)
        if key not in lookup:
            continue
        load = lookup[key]
        if load <= 0.0:
            continue
        base_load = load if base_load is None else alpha * base_load + (1.0 - alpha) * load
        if not state.audit_eligible:
            continue
        if (
            state.p_fc_total_kw is None
            or state.p_batt_total_kw is None
            or state.soc_system is None
            or state.previous_p_fc_total_kw is None
        ):
            continue
        history.append((state, load, base_load))
        cutoff = state.timestamp.timestamp() - AUDIT_HISTORY_SECONDS
        history = [
            item for item in history if item[0].timestamp.timestamp() >= cutoff
        ]
        if len(history) < 6:
            continue
        elapsed = (history[-1][0].timestamp - history[0][0].timestamp).total_seconds()
        if elapsed < AUDIT_HISTORY_SECONDS:
            continue
        window_states = tuple(item[0] for item in history)
        window_loads = tuple(item[1] for item in history)
        current = window_states[-1]
        current_base = history[-1][2]
        mean = float(np.mean(np.asarray(window_loads, dtype=float)))
        std = float(np.std(np.asarray(window_loads, dtype=float), ddof=0))
        trend = _least_squares_trend(
            tuple(item.timestamp for item in window_states),
            window_loads,
        )
        fc = float(current.p_fc_total_kw)
        previous_fc = float(current.previous_p_fc_total_kw)
        battery = float(current.p_batt_total_kw)
        soc = float(current.soc_system)
        rows.append(
            AuditFeatureRow(
                parent=segment.parent,
                sample_id=segment.sample_id,
                timestamp=current.timestamp,
                soc=soc,
                fc_power_kw=fc,
                previous_fc_power_kw=previous_fc,
                battery_power_kw=battery,
                load_power_kw=window_loads[-1],
                recent_load_mean_kw=mean,
                recent_load_population_std_kw=std,
                recent_load_trend_kw_per_s=trend,
                base_load_kw=current_base,
                recent_delta_soc=soc - float(window_states[0].soc_system),
                delta_load_kw=window_loads[-1] - current_base,
                delta_fc_kw=fc - previous_fc,
                measured_power_balance_residual_kw=(
                    window_loads[-1] - fc - battery
                ),
                history_sample_count=len(history),
            )
        )
    return tuple(rows)


def build_train_feature_rows(
    dataset_root: str | Path,
    segments: Sequence[TrainSegment],
    state_loader: Callable[
        [str], Sequence[ParentSupervisoryState | AuditSupervisorySample]
    ],
) -> tuple[AuditFeatureRow, ...]:
    """Build rows for an already authenticated Train whitelist only."""

    if not callable(state_loader):
        raise TypeError("state_loader must be callable")
    root = Path(dataset_root).resolve()
    checked = tuple(segments)
    if not checked or any(type(segment) is not TrainSegment for segment in checked):
        raise TypeError("segments must contain TrainSegment values")
    state_cache: dict[
        str, tuple[ParentSupervisoryState | AuditSupervisorySample, ...]
    ] = {}
    rows: list[AuditFeatureRow] = []
    for segment in checked:
        path = _train_path(root, segment.relative_path)
        if segment.parent not in state_cache:
            state_cache[segment.parent] = tuple(state_loader(segment.parent))
        load_frame = pd.read_csv(
            path,
            usecols=["timestamp", "time_s", "load_total_kw"],
            encoding="utf-8-sig",
        )
        rows.extend(
            build_causal_feature_rows(
                segment,
                state_cache[segment.parent],
                load_frame,
            )
        )
    return tuple(rows)


def feature_frame(rows: Sequence[AuditFeatureRow]) -> pd.DataFrame:
    """Convert immutable physical rows into a stable tabular representation."""

    checked = tuple(rows)
    if not checked or any(type(row) is not AuditFeatureRow for row in checked):
        raise TypeError("rows must contain AuditFeatureRow values")
    frame = pd.DataFrame(asdict(row) for row in checked)
    ordered = [field.name for field in AuditFeatureRow.__dataclass_fields__.values()]
    return frame.loc[:, ordered]


def normalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply frozen physical denominators without data-fitted min-max scaling."""

    required = {
        "soc",
        "fc_power_kw",
        "previous_fc_power_kw",
        "battery_power_kw",
        "load_power_kw",
        "recent_load_mean_kw",
        "recent_load_population_std_kw",
        "recent_load_trend_kw_per_s",
        "base_load_kw",
        "recent_delta_soc",
        "delta_load_kw",
        "delta_fc_kw",
        "speed_kn",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature frame is missing columns: {sorted(missing)}")
    result = pd.DataFrame(index=frame.index)
    result["soc"] = frame["soc"].astype(float)
    result["fuel_cell_power_fraction"] = (
        frame["fc_power_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["previous_fuel_cell_power_fraction"] = (
        frame["previous_fc_power_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["battery_power_fraction"] = (
        frame["battery_power_kw"].astype(float) / AUDIT_BATTERY_POWER_SCALE_KW
    )
    result["load_power_fraction"] = (
        frame["load_power_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["recent_load_mean_fraction"] = (
        frame["recent_load_mean_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["recent_load_population_std_fraction"] = (
        frame["recent_load_population_std_kw"].astype(float)
        / AUDIT_POWER_SCALE_KW
    )
    result["recent_load_window_trend_fraction"] = (
        frame["recent_load_trend_kw_per_s"].astype(float)
        * AUDIT_HISTORY_SECONDS
        / AUDIT_POWER_SCALE_KW
    )
    result["causal_base_load_fraction"] = (
        frame["base_load_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["recent_delta_soc"] = frame["recent_delta_soc"].astype(float)
    result["load_residual_fraction"] = (
        frame["delta_load_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["fuel_cell_delta_fraction"] = (
        frame["delta_fc_kw"].astype(float) / AUDIT_POWER_SCALE_KW
    )
    result["speed_fraction"] = (
        frame["speed_kn"].astype(float) / FORMAL_STATE_SPEED_SCALE_KN
    )
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("normalized feature frame must be finite")
    return result


def formal_s8_frame(normalized: pd.DataFrame) -> pd.DataFrame:
    """Select the frozen production S8 in its exact schema order."""

    missing = set(FORMAL_STATE_FEATURE_NAMES).difference(normalized.columns)
    if missing:
        raise ValueError(f"normalized frame is missing S8 columns: {sorted(missing)}")
    return normalized.loc[:, list(FORMAL_STATE_FEATURE_NAMES)].copy()


def descriptive_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable Train-only descriptive statistics for numeric features."""

    rows: list[dict[str, object]] = []
    for feature in frame.columns:
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(dtype=float))]
        if finite.empty:
            raise ValueError(f"feature has no finite observations: {feature}")
        array = finite.to_numpy(dtype=float)
        std = float(np.std(array, ddof=0))
        rows.append(
            {
                "feature": str(feature),
                "count": int(array.size),
                "missing_count": int(len(values) - array.size),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "mean": float(np.mean(array)),
                "std": std,
                "p01": float(np.quantile(array, 0.01)),
                "p05": float(np.quantile(array, 0.05)),
                "p50": float(np.quantile(array, 0.50)),
                "p95": float(np.quantile(array, 0.95)),
                "p99": float(np.quantile(array, 0.99)),
                "near_zero_variance": bool(std <= NEAR_ZERO_VARIANCE_STD),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrices(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise-complete Pearson and Spearman matrices."""

    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="pearson"), numeric.corr(method="spearman")


def _pair_correlations(
    frame: pd.DataFrame,
    left: str,
    right: str,
) -> tuple[float, float]:
    pair = frame[[left, right]].dropna()
    if len(pair) < 2 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return math.nan, math.nan
    return (
        float(pair[left].corr(pair[right], method="pearson")),
        float(pair[left].corr(pair[right], method="spearman")),
    )


def redundancy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize model identities separately from measured correlations."""

    required = {
        "measured_power_balance_residual_kw",
        "recent_load_mean_kw",
        "base_load_kw",
        "fc_power_kw",
        "previous_fc_power_kw",
        "delta_fc_kw",
        "recent_delta_soc",
        "battery_power_kw",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"redundancy frame is missing columns: {sorted(missing)}")
    residual = frame["measured_power_balance_residual_kw"].to_numpy(dtype=float)
    relationships = [
        (
            "recent_load_mean_vs_causal_base",
            "recent_load_mean_kw",
            "base_load_kw",
            "EMPIRICAL_CORRELATION",
        ),
        (
            "current_fc_vs_previous_fc",
            "fc_power_kw",
            "previous_fc_power_kw",
            "LINEAR_REPARAMETERIZATION_WITH_DELTA",
        ),
        (
            "recent_delta_soc_vs_battery_power",
            "recent_delta_soc",
            "battery_power_kw",
            "EMPIRICAL_CORRELATION",
        ),
    ]
    rows: list[dict[str, object]] = [
        {
            "relationship": "environment_power_balance",
            "left_feature": "battery_power_kw",
            "right_feature": "load_power_kw-fc_power_kw",
            "model_status": "EXACT_IDENTITY",
            "pearson": math.nan,
            "spearman": math.nan,
            "measured_max_abs_residual_kw": float(np.max(np.abs(residual))),
            "interpretation": (
                "Battery power is deterministic in the simulated environment; "
                "measured residual reflects telemetry/alignment mismatch."
            ),
        }
    ]
    for name, left, right, status in relationships:
        pearson, spearman = _pair_correlations(frame, left, right)
        rows.append(
            {
                "relationship": name,
                "left_feature": left,
                "right_feature": right,
                "model_status": status,
                "pearson": pearson,
                "spearman": spearman,
                "measured_max_abs_residual_kw": math.nan,
                "interpretation": "Train-only descriptive evidence",
            }
        )
    return pd.DataFrame(rows)


def regime_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe physically relevant Train regimes without freezing policy thresholds."""

    required = {
        "soc",
        "fc_power_kw",
        "load_power_kw",
        "recent_load_population_std_kw",
        "recent_load_trend_kw_per_s",
        "delta_load_kw",
        "delta_fc_kw",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"regime frame is missing columns: {sorted(missing)}")
    trend = frame["recent_load_trend_kw_per_s"].astype(float)
    volatility = frame["recent_load_population_std_kw"].astype(float)
    fc = frame["fc_power_kw"].astype(float)
    trend_deadband = float(np.quantile(np.abs(trend), 0.50))
    volatility_high = float(np.quantile(volatility, 0.75))
    fc_low = float(np.quantile(fc, 0.25))
    fc_high = float(np.quantile(fc, 0.75))
    masks = (
        ("steady_load", np.abs(trend) <= trend_deadband, trend_deadband, "kW/s"),
        ("load_rise", trend > trend_deadband, trend_deadband, "kW/s"),
        ("load_fall", trend < -trend_deadband, -trend_deadband, "kW/s"),
        ("high_volatility", volatility >= volatility_high, volatility_high, "kW"),
        ("low_soc", frame["soc"].astype(float) < 0.4, 0.4, "fraction"),
        ("high_soc", frame["soc"].astype(float) > 0.6, 0.6, "fraction"),
        ("fc_low_load", fc <= fc_low, fc_low, "kW"),
        ("fc_high_load", fc >= fc_high, fc_high, "kW"),
    )
    rows: list[dict[str, object]] = []
    for name, mask, threshold, unit in masks:
        subset = frame.loc[mask]
        rows.append(
            {
                "regime": name,
                "count": int(len(subset)),
                "fraction": float(len(subset) / len(frame)) if len(frame) else 0.0,
                "threshold": threshold,
                "threshold_unit": unit,
                "threshold_status": (
                    "FROZEN_CONTROLLER_BOUND" if name in {"low_soc", "high_soc"}
                    else "DESCRIPTIVE_TRAIN_ONLY"
                ),
                "mean_soc": float(subset["soc"].mean()) if len(subset) else math.nan,
                "mean_load_kw": (
                    float(subset["load_power_kw"].mean()) if len(subset) else math.nan
                ),
                "mean_fc_kw": (
                    float(subset["fc_power_kw"].mean()) if len(subset) else math.nan
                ),
                "mean_delta_load_kw": (
                    float(subset["delta_load_kw"].mean()) if len(subset) else math.nan
                ),
                "mean_delta_fc_kw": (
                    float(subset["delta_fc_kw"].mean()) if len(subset) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def state_comparison() -> pd.DataFrame:
    """Return the frozen audit definitions of requested state ablations."""

    s8 = PROPOSED_NORMALIZED_FEATURES
    definitions = (
        ("S10", CANDIDATE_NORMALIZED_FEATURES, "current candidate"),
        ("S8", s8, "frozen formal baseline"),
        (
            "S7-NO-SPEED",
            tuple(feature for feature in s8 if feature != "speed_fraction"),
            "ablation without AIS speed",
        ),
        (
            "S7-NO-STD",
            tuple(
                feature
                for feature in s8
                if feature != "recent_load_population_std_fraction"
            ),
            "remove load standard deviation",
        ),
        (
            "MINIMUM",
            (
                "soc",
                "causal_base_load_fraction",
                "load_residual_fraction",
                "fuel_cell_power_fraction",
                "fuel_cell_delta_fraction",
            ),
            "minimum defensible physical state",
        ),
    )
    return pd.DataFrame(
        {
            "state_id": name,
            "dimension": len(features),
            "features": "|".join(features),
            "purpose": purpose,
        }
        for name, features, purpose in definitions
    )


def markov_inventory() -> pd.DataFrame:
    """Return the code-audited memory inventory for the current v2 design."""

    rows = (
        (
            "battery_soc",
            "PHYSICAL_TRANSITION_STATE",
            "KEEP soc",
            "src/v2/models/battery_energy.py:next_soc",
            "SOC directly changes the next physical state and q_soc trade-off.",
        ),
        (
            "causal_lpf_state",
            "CONTROLLER_DYNAMIC_STATE",
            "KEEP causal_base_load_fraction",
            "src/v2/control/causal_base_load.py:CausalBaseLoadFilter._observed_base_kw",
            "The next base reference depends on the committed LPF state.",
        ),
        (
            "previous_executed_fc_power",
            "CONTROLLER_DYNAMIC_STATE",
            "COVER with current FC plus delta FC",
            "src/v2/control/nonlinear_mpc.py:NonlinearMPC.solve",
            "P_fc(k) and delta P_fc(k) recover P_fc(k-1) exactly.",
        ),
        (
            "fc_on_off_state",
            "DEGRADATION_DYNAMIC_STATE",
            "COVER by current FC operating state",
            "src/v2/models/fuel_cell_degradation.py:FuelCellVoltageLossTracker.is_on",
            "Start/stop status is observable from the executed FC on/off condition.",
        ),
        (
            "cumulative_fc_voltage_loss",
            "REWARD_CLIPPING_STATE",
            "CONDITIONAL: omit only with reset and far-from-EOL invariant",
            "src/v2/models/fuel_cell_degradation.py:fuel_cell_economic_life_increment",
            "Below EOL, marginal interval cost is independent of the cumulative level; clipping changes it near EOL.",
        ),
        (
            "cumulative_battery_weighted_ah",
            "REWARD_CLIPPING_STATE",
            "CONDITIONAL: omit only with reset and far-from-EOL invariant",
            "src/v2/models/battery_degradation.py:battery_economic_life_increment",
            "Below EOL, marginal interval cost is independent of the cumulative level; clipping changes it near EOL.",
        ),
        (
            "cumulative_start_stop_counts",
            "DIAGNOSTIC_ACCOUNTING",
            "OMIT",
            "src/v2/models/fuel_cell_degradation.py:FuelCellVoltageLossTracker",
            "Counts diagnose accumulated loss; the next increment depends on current on/off transition, not the count.",
        ),
        (
            "previous_dqn_action",
            "POLICY_HISTORY",
            "OMIT while no switching penalty or action-rate constraint exists",
            "src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment.step",
            "The environment holds the selected action for M steps but defines no reward or transition term from the prior action.",
        ),
        (
            "mpc_warm_start",
            "NUMERICAL_SOLVER_STATE",
            "OMIT; enforce solver robustness separately",
            "src/v2/control/nonlinear_mpc.py:shifted_warm_start",
            "Warm start is an optional initial guess, not a physical state; local-solution sensitivity remains a solver audit concern.",
        ),
        (
            "terminal_recharge_initial_soc",
            "EPISODE_ACCOUNTING_STATE",
            "OMIT only when initial SOC is frozen per episode",
            "src/v2/economics.py:terminal_recharge_grid_energy",
            "Terminal shore cost depends on initial minus final SOC; a variable initial SOC would need state or explicit episode context.",
        ),
        (
            "macro_step_position",
            "MULTIRATE_EXECUTION_STATE",
            "OMIT at DQN decision boundaries",
            "src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment.step",
            "The DQN is queried only at macro boundaries; the M-step countdown is internal during action execution.",
        ),
        (
            "environment_done_failed_replay_history",
            "SOFTWARE_BOOKKEEPING",
            "OMIT",
            "src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment",
            "Done/failure gates and replay history do not define a continuing physical state presented for another action.",
        ),
    )
    return pd.DataFrame(
        rows,
        columns=(
            "memory_id",
            "classification",
            "state_treatment",
            "code_evidence",
            "reason",
        ),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", "<br>").replace("\n", " ")

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join((header, rule, *body))


def _feature_statistics_bundle(
    physical: pd.DataFrame,
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    physical_columns = (
        "soc",
        "fc_power_kw",
        "previous_fc_power_kw",
        "battery_power_kw",
        "load_power_kw",
        "recent_load_mean_kw",
        "recent_load_population_std_kw",
        "recent_load_trend_kw_per_s",
        "base_load_kw",
        "recent_delta_soc",
        "delta_load_kw",
        "delta_fc_kw",
        "measured_power_balance_residual_kw",
        "speed_kn",
    )
    physical_stats = descriptive_statistics(physical.loc[:, physical_columns])
    physical_stats.insert(1, "representation", "physical")
    normalized_stats = descriptive_statistics(normalized)
    normalized_stats.insert(1, "representation", "normalized")
    return pd.concat((physical_stats, normalized_stats), ignore_index=True)


def _render_plots(
    output_root: Path,
    normalized: pd.DataFrame,
    pearson: pd.DataFrame,
    regimes: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 9))
    image = axis.imshow(pearson.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [str(value) for value in pearson.columns]
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    axis.set_title("Train-only Pearson correlation of normalized state features")
    figure.colorbar(image, ax=axis, label="Pearson r")
    figure.tight_layout()
    figure.savefig(output_root / "candidate_correlation_heatmap.png", dpi=180)
    plt.close(figure)

    distribution_features = PROPOSED_NORMALIZED_FEATURES
    figure, axes = plt.subplots(4, 2, figsize=(11, 13))
    for axis, feature in zip(axes.flat, distribution_features):
        axis.hist(normalized[feature].to_numpy(dtype=float), bins=40, color="#2b6f9f")
        axis.set_title(feature)
        axis.set_ylabel("Train observations")
    axes.flat[-1].axis("off")
    figure.suptitle("Train-only distributions for frozen S8", y=0.995)
    figure.tight_layout()
    figure.savefig(output_root / "feature_distributions.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(regimes["regime"], regimes["count"], color="#4c8f6f")
    axis.set_ylabel("Eligible Train observations")
    axis.set_title("Train-only operating-regime coverage")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(output_root / "regime_discrimination.png", dpi=180)
    plt.close(figure)


def _render_report(
    *,
    physical: pd.DataFrame,
    statistics: pd.DataFrame,
    redundancy: pd.DataFrame,
    regimes: pd.DataFrame,
    comparison: pd.DataFrame,
    inventory: pd.DataFrame,
    segment_count: int,
    row_count: int,
    measured_proxy_coverage: dict[str, object],
    hidden_life_state_upper_bounds: dict[str, object],
) -> str:
    from ..control.nonlinear_mpc import SOC_HARD_MAX, SOC_HARD_MIN

    relationship = redundancy.set_index("relationship")
    mean_base = relationship.loc["recent_load_mean_vs_causal_base"]
    delta_soc = relationship.loc["recent_delta_soc_vs_battery_power"]
    balance = relationship.loc["environment_power_balance"]
    decisions = pd.DataFrame(
        (
            ("soc", "KEEP", "q_soc", "直接描述电池能量裕量。"),
            ("fuel_cell_power_fraction", "KEEP", "q_base / q_smooth / economic reward", "FC 工作点影响功率分配、效率与退化。"),
            ("previous_fuel_cell_power_fraction", "REPLACE", "q_smooth", "改为 delta P_fc；与当前 P_fc 联合可精确恢复前一时刻功率，且控制语义更直接。"),
            ("battery_power_fraction", "REMOVE", "none independently", "环境中 P_batt=P_load-P_fc，是确定性冗余。"),
            ("load_power_fraction", "REPLACE", "q_base / q_smooth", "用 P_base 与 P_load-P_base 分离低频基础负荷和瞬时峰谷。"),
            ("recent_load_mean_fraction", "REMOVE", "q_base already covered", f"P_base 是下层控制器真实动态状态；Train Pearson={mean_base['pearson']:.4f}，Spearman={mean_base['spearman']:.4f}。"),
            ("recent_load_population_std_fraction", "KEEP", "q_smooth", "因果波动强度提供瞬时 residual 之外的信息。"),
            ("recent_load_window_trend_fraction", "KEEP", "q_base / q_smooth", "区分增载、减载与稳态，决定 FC 跟随和电池缓冲需求。"),
            ("causal_base_load_fraction", "KEEP", "q_base", "它既是 LPF 必要记忆，也是 J_base 的直接参考。"),
            ("recent_delta_soc", "REMOVE", "q_soc already covered by SOC", f"它主要是电池功率的时间积分结果；与当前电池功率的 Train Pearson={delta_soc['pearson']:.4f}。"),
            ("speed_fraction", "ADD", "shore interlock / operating context", "AIS 航速区分在航与靠泊上下文；岸电区间不进入 DQN 决策。"),
        ),
        columns=("current_feature", "decision", "weight_link", "reason"),
    )
    schema = pd.DataFrame(
        (
            (1, "soc", "SOC(k)", "保持原始 fraction", "当前值"),
            (2, "causal_base_load_fraction", "P_base(k)", "除以 600 kW", "LPF 内部状态"),
            (3, "load_residual_fraction", "P_load(k)-P_base(k)", "除以 600 kW", "当前值"),
            (4, "recent_load_population_std_fraction", "[k-150 s,k] 内 P_load 的总体标准差", "除以 600 kW", "150 s 因果历史"),
            (5, "recent_load_window_trend_fraction", "[k-150 s,k] 内负荷最小二乘趋势", "trend*150 s/600 kW", "150 s 因果历史"),
            (6, "fuel_cell_power_fraction", "P_fc(k)", "除以 600 kW", "当前值"),
            (7, "fuel_cell_delta_fraction", "P_fc(k)-P_fc(k-1)", "除以 600 kW", "前一执行 FC 功率"),
            (8, "speed_fraction", "v(k)", "除以 20 kn", "当前 AIS 航速"),
        ),
        columns=("order", "feature", "definition", "normalization", "memory"),
    )
    normalized_stats = statistics.loc[
        statistics["representation"].eq("normalized")
        & statistics["feature"].isin(PROPOSED_NORMALIZED_FEATURES)
    ].copy()
    outside_soc = (
        (physical["soc"].astype(float) < SOC_HARD_MIN)
        | (physical["soc"].astype(float) > SOC_HARD_MAX)
    )
    outside_count = int(outside_soc.sum())
    outside_fraction = float(outside_soc.mean())
    sections = [
        "# V2 DQN 状态空间审核\n",
        "## 审核范围\n",
        f"本审核只使用 `operating_dataset_zero_boundary_v2/train` 的 {segment_count} 个航段和 {row_count} 个 eligible 30 s 状态点。Validation/Test 航段 CSV 打开数为 0。原始 FC/BMS 遥测由 Train parent 与时间边界双重白名单限制。本轮未运行 DQN 训练或动作筛选。\n",
        (
            f"formal ONBOARD 轴共有 {measured_proxy_coverage['formal_onboard_row_count']} 个点；"
            f"其中 {measured_proxy_coverage['measured_proxy_row_count']} 个点（"
            f"{float(measured_proxy_coverage['measured_proxy_row_fraction']):.2%}）具备严格因果、"
            "12 簇齐全且不复用原始行的实测 SOC proxy。"
            f"无可用 proxy 行的航段为 {measured_proxy_coverage['segments_without_measured_proxy_rows']}。"
            "这只限制实测分布证据覆盖率，不会删除 formal 训练轴上的 ONBOARD 点；正式环境 SOC 由模型递推。\n"
        ),
        "## 当前 10 维候选状态\n",
        _markdown_table(decisions),
        "\n## 冻结 S8 schema\n",
        _markdown_table(schema),
        "\n所有尺度都是固定物理尺度，不使用 Train min-max。600 kW 是冻结的 v2 research-simulation plant rating，不是实船技术规格中的 560 kW；归一化结果允许超出 [-1,1]，不得裁剪。\n",
        "## Train-only 数值证据\n",
        _markdown_table(normalized_stats),
        f"\n仿真环境中的功率平衡是精确恒等式。实测重建残差最大绝对值为 {float(balance['measured_max_abs_residual_kw']):.6g} kW；该量来自同源功率重建，不能作为 battery_power 独立信息的证据。\n",
        "## 冗余分析\n",
        _markdown_table(redundancy),
        "\n## 运行工况区分能力\n",
        _markdown_table(regimes),
        "\n表中的 trend、volatility 和 FC 阈值仅用于 Train 描述，不是生产策略阈值，也没有利用 held-out 数据拟合。\n",
        "## 状态结构消融比较\n",
        _markdown_table(comparison),
        "\n这些比较是结构与因果信息消融。S8 是已冻结正式 baseline；无航速版本仅作为消融，不参与当前正式训练。\n",
        "## Markov 性审核\n",
        _markdown_table(inventory),
        "\n只有在每个训练 episode 都重置累计退化、且 preflight 上界证明 episode 远离 EOL clipping 时，累计 FC/Battery lifetime fraction 才可省略；否则必须增加两项 clipped lifetime state。MPC warm start 不进入 DQN state，但 integrated solver robustness 必须证明不同初值不会导致实质不同的执行命令。terminal recharge 还要求冻结 episode initial SOC。\n",
        "## Minimum defensible state\n",
        "最小可辩护状态为 `[SOC, P_base, P_load-P_base, P_fc, delta_P_fc]`（S5）。它保留电池能量裕量、LPF 记忆、瞬时峰谷、FC 工作点与 FC 动态，但删除显式 volatility 和 trend，因此只适合作为论文消融基线，不应作为首选正式状态。\n",
        "## 证据边界与局限性\n",
        f"实船 Train SOC 中有 {outside_count}/{row_count}（{outside_fraction:.2%}）位于 v2 仿真硬区间 [{SOC_HARD_MIN:.2f}, {SOC_HARD_MAX:.2f}] 之外。这些实测 SOC/FC 数据用于判断特征覆盖与区分力，不代表未来仿真策略的 state-visitation distribution。正式环境仍将依据模型转移生成 SOC 与 FC 轨迹。功率平衡残差接近零是因为 formal load 与 FC/BMS 功率同源构造，不是独立传感器验证。\n",
        "## S8 最终结论\n",
        (
            "冻结的八维 S8 与生产 `FORMAL_STATE_FEATURE_NAMES` 完全一致。前七维保留 SOC、LPF 记忆、负荷残差/波动/趋势及 FC 工作点动态；`speed_fraction` 提供 AIS 在航上下文。DQN 只在 ONBOARD 决策边界读取 S8，shore_pending/shore_charging 会重置控制历史并暂停 DQN/MPC。"
            f"累计退化账户逐 episode 重置；保守上界为 FC={float(hidden_life_state_upper_bounds.get('max_fc_raw_life_fraction_upper_bound', float('nan'))):.6f}、battery={float(hidden_life_state_upper_bounds.get('max_battery_raw_life_fraction_upper_bound', float('nan'))):.6f}，均低于 EOL=1，因此 clipped lifetime 在当前 formal episode 内不可达，累计退化无需进入 S8。\n"
        ),
    ]
    return "\n".join(sections)


def write_audit_artifacts(
    *,
    rows: Sequence[AuditFeatureRow],
    segments: Sequence[TrainSegment],
    output_root: str | Path,
    report_path: str | Path,
    dataset_root: str | Path,
    raw_root: str | Path,
    power_manifest_path: str | Path | None = None,
    ais_manifest_path: str | Path | None = None,
    mode_manifest_path: str | Path | None = None,
    formal_onboard_row_count: int | None = None,
    hidden_life_state_upper_bounds: dict[str, object] | None = None,
) -> dict[str, object]:
    """Write one immutable Train-only evidence bundle and Markdown report."""

    checked_rows = tuple(rows)
    checked_segments = tuple(segments)
    if not checked_rows:
        raise ValueError("state audit requires at least one eligible Train row")
    if not checked_segments:
        raise ValueError("state audit requires at least one Train segment")
    audited_segment_ids = {row.sample_id for row in checked_rows}
    formal_onboard = (
        len(checked_rows)
        if formal_onboard_row_count is None
        else int(formal_onboard_row_count)
    )
    if formal_onboard < len(checked_rows):
        raise ValueError("formal ONBOARD count cannot be below measured proxy rows")
    measured_proxy_coverage = {
        "formal_onboard_row_count": formal_onboard,
        "measured_proxy_row_count": len(checked_rows),
        "measured_proxy_row_fraction": len(checked_rows) / formal_onboard,
        "measured_proxy_segment_count": len(audited_segment_ids),
        "segments_without_measured_proxy_rows": [
            segment.sample_id
            for segment in checked_segments
            if segment.sample_id not in audited_segment_ids
        ],
    }
    life_bounds = hidden_life_state_upper_bounds or {
        "status": "NOT_EVALUATED_LEGACY_FIXTURE"
    }
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"audit output already exists: {destination}")
    destination.mkdir(parents=True)
    report = Path(report_path).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)

    physical = feature_frame(checked_rows)
    normalized = normalize_feature_frame(physical)
    statistics = _feature_statistics_bundle(physical, normalized)
    pearson, spearman = correlation_matrices(normalized)
    redundancy = redundancy_summary(physical)
    regimes = regime_summary(physical)
    comparison = state_comparison()
    inventory = markov_inventory()

    physical.to_csv(destination / "feature_rows.csv", index=False, encoding="utf-8")
    statistics.to_csv(
        destination / "feature_statistics.csv", index=False, encoding="utf-8"
    )
    pearson.to_csv(destination / "pearson_correlation.csv", encoding="utf-8")
    spearman.to_csv(destination / "spearman_correlation.csv", encoding="utf-8")
    physical.loc[
        :,
        [
            "parent",
            "sample_id",
            "timestamp",
            "load_power_kw",
            "fc_power_kw",
            "battery_power_kw",
            "measured_power_balance_residual_kw",
        ],
    ].to_csv(
        destination / "power_balance_residuals.csv",
        index=False,
        encoding="utf-8",
    )
    redundancy.to_csv(
        destination / "redundancy_summary.csv", index=False, encoding="utf-8"
    )
    regimes.to_csv(destination / "regime_summary.csv", index=False, encoding="utf-8")
    comparison.to_csv(
        destination / "state_comparison.csv", index=False, encoding="utf-8"
    )
    inventory.to_csv(
        destination / "markov_inventory.csv", index=False, encoding="utf-8"
    )
    _render_plots(destination, normalized, pearson, regimes)

    report.write_text(
        _render_report(
            physical=physical,
            statistics=statistics,
            redundancy=redundancy,
            regimes=regimes,
            comparison=comparison,
            inventory=inventory,
            segment_count=len(checked_segments),
            row_count=len(checked_rows),
            measured_proxy_coverage=measured_proxy_coverage,
            hidden_life_state_upper_bounds=life_bounds,
        ),
        encoding="utf-8",
    )
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(destination.iterdir())
        if path.is_file() and path.name != "audit_manifest.json"
    }
    versions = {segment.dataset_version for segment in checked_segments}
    if versions != {ACTIVE_DATASET_VERSION}:
        raise ValueError("all audit segments must use the active dataset version")
    input_manifest_paths = {
        "power": power_manifest_path,
        "ais": ais_manifest_path,
        "modes": mode_manifest_path,
    }
    supplied_input_manifests = {
        name: path for name, path in input_manifest_paths.items() if path is not None
    }
    if supplied_input_manifests and len(supplied_input_manifests) != 3:
        raise ValueError("power, AIS, and mode manifests must be supplied together")
    input_manifest_sha256: dict[str, str] = {}
    if supplied_input_manifests:
        resolved_manifests = {
            name: Path(path).resolve()
            for name, path in supplied_input_manifests.items()
        }
        missing_manifests = [
            str(path) for path in resolved_manifests.values() if not path.is_file()
        ]
        if missing_manifests:
            raise FileNotFoundError(
                f"state-audit input manifests are missing: {missing_manifests}"
            )
        input_manifest_sha256 = {
            name: _sha256(path) for name, path in resolved_manifests.items()
        }

    manifest: dict[str, object] = {
        "dataset_version": ACTIVE_DATASET_VERSION,
        "dataset_root": str(Path(dataset_root).resolve()),
        "raw_root": str(Path(raw_root).resolve()),
        "split": "train",
        "train_segment_count": len(checked_segments),
        "train_parent_count": len({segment.parent for segment in checked_segments}),
        "eligible_row_count": len(checked_rows),
        "sample_seconds": AUDIT_SAMPLE_SECONDS,
        "history_seconds": AUDIT_HISTORY_SECONDS,
        "tau_lpf_seconds": AUDIT_TAU_LPF_SECONDS,
        "power_scale_kw": AUDIT_POWER_SCALE_KW,
        "battery_power_scale_kw": AUDIT_BATTERY_POWER_SCALE_KW,
        "formal_state_schema_version": FORMAL_STATE_SCHEMA_VERSION,
        "formal_state_schema_digest": FORMAL_STATE_SCHEMA_DIGEST,
        "segment_ids": [segment.sample_id for segment in checked_segments],
        "segment_sha256": {
            segment.sample_id: segment.sha256 for segment in checked_segments
        },
        "input_manifest_sha256": input_manifest_sha256,
        "measured_proxy_coverage": measured_proxy_coverage,
        "hidden_life_state_upper_bounds": life_bounds,
        "artifact_sha256": artifact_hashes,
        "held_out_segment_files_opened": 0,
        "formal_training_started": False,
        "action_catalog_modified": False,
    }
    (destination / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "ACTIVE_DATASET_VERSION",
    "AUDIT_BATTERY_POWER_SCALE_KW",
    "AUDIT_HISTORY_SECONDS",
    "AUDIT_POWER_SCALE_KW",
    "AUDIT_SAMPLE_SECONDS",
    "AUDIT_TAU_LPF_SECONDS",
    "AuditFeatureRow",
    "AuditSupervisorySample",
    "CANDIDATE_NORMALIZED_FEATURES",
    "PROPOSED_NORMALIZED_FEATURES",
    "TrainSegment",
    "align_battery_soc_to_timestamps",
    "assemble_audit_supervisory_samples",
    "build_causal_feature_rows",
    "build_formal_episode_feature_rows",
    "build_train_feature_rows",
    "correlation_matrices",
    "episode_life_upper_bounds",
    "descriptive_statistics",
    "feature_frame",
    "formal_s8_frame",
    "load_train_segments",
    "markov_inventory",
    "normalize_feature_frame",
    "redundancy_summary",
    "regime_summary",
    "state_comparison",
    "write_audit_artifacts",
]
