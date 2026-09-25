"""Fail-closed loader for the frozen power dataset and 30 s AIS sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .ais_speed_sidecar import AIS_GAP_PROVENANCE, AIS_RAW_PROVENANCE
from .supervisory_rules import OperatingMode


EXPECTED_SPLIT_COUNTS = {"train": 38, "validation": 10, "test": 5}
EXPECTED_TRAIN_STEPS = 30_909


def _onboard_macro_transition_count(values: tuple[str, ...]) -> int:
    """Count M=5 decisions across contiguous ONBOARD runs only."""

    total = 0
    run = 0
    for value in values:
        mode = OperatingMode(value)
        if mode is OperatingMode.ONBOARD:
            run += 1
            continue
        total += math.ceil(run / 5) if run else 0
        run = 0
    total += math.ceil(run / 5) if run else 0
    return int(total)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes dataset root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class FormalEpisode:
    parent: str
    sample_id: str
    split: str
    timestamp: tuple[pd.Timestamp, ...]
    time_s: np.ndarray
    load_kw: np.ndarray
    speed_kn: np.ndarray
    speed_provenance: tuple[str, ...]
    fc_power_kw: np.ndarray
    battery_bus_kw: np.ndarray
    operating_mode: tuple[str, ...]
    mode_reason: tuple[str, ...]

    @property
    def step_count(self) -> int:
        return len(self.time_s)

    @property
    def macro_transition_count(self) -> int:
        return _onboard_macro_transition_count(self.operating_mode)


class FormalTrainingDataset:
    def __init__(
        self,
        power_root: Path,
        ais_root: Path,
        mode_root: Path,
        power: pd.DataFrame,
        ais: pd.DataFrame,
        modes: pd.DataFrame,
    ) -> None:
        self._power_root = power_root
        self._ais_root = ais_root
        self._mode_root = mode_root
        self._power = power
        self._ais = ais
        self._modes = modes
        self._cache: dict[str, tuple[FormalEpisode, ...]] = {}
        self._opened_test_payloads = 0

    @classmethod
    def open(
        cls, power_root: Path, ais_root: Path, mode_root: Path
    ) -> "FormalTrainingDataset":
        power = Path(power_root).resolve()
        ais = Path(ais_root).resolve()
        modes = Path(mode_root).resolve()
        power_manifest = power / "metadata" / "sample_manifest.csv"
        ais_manifest = ais / "metadata" / "sample_manifest.csv"
        mode_manifest = modes / "metadata" / "sample_manifest.csv"
        if not power_manifest.is_file() or not ais_manifest.is_file() or not mode_manifest.is_file():
            raise FileNotFoundError("formal power/AIS/mode manifest is missing")
        p = pd.read_csv(power_manifest, encoding="utf-8-sig")
        a = pd.read_csv(ais_manifest, encoding="utf-8-sig")
        m = pd.read_csv(mode_manifest, encoding="utf-8-sig")
        required_power = {
            "parent", "sample_id", "relative_path", "split", "duration_s", "sha256"
        }
        required_ais = {
            "parent", "sample_id", "relative_path", "split",
            "supervisory_row_count", "sha256",
        }
        required_modes = {
            "parent", "sample_id", "relative_path", "split",
            "supervisory_row_count", "unresolved_row_count", "sha256",
        }
        if (
            required_power.difference(p.columns)
            or required_ais.difference(a.columns)
            or required_modes.difference(m.columns)
        ):
            raise ValueError("formal power/AIS/mode manifest schema mismatch")
        if (
            p["sample_id"].duplicated().any()
            or a["sample_id"].duplicated().any()
            or m["sample_id"].duplicated().any()
        ):
            raise ValueError("formal manifests require unique sample IDs")
        counts = {
            split: int((p["split"] == split).sum())
            for split in ("train", "validation", "test")
        }
        if counts != EXPECTED_SPLIT_COUNTS:
            raise ValueError(f"formal split counts differ: {counts}")
        identities = p[["parent", "sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
        ais_identities = a[["parent", "sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
        mode_identities = m[["parent", "sample_id", "split"]].sort_values("sample_id").reset_index(drop=True)
        if not identities.equals(ais_identities) or not identities.equals(mode_identities):
            raise ValueError("power, AIS, and mode identities/splits differ")
        for row in p[p["split"].eq("test")].itertuples(index=False):
            _contained(power, str(row.relative_path))
        for row in a[a["split"].eq("test")].itertuples(index=False):
            _contained(ais, str(row.relative_path))
        for row in m[m["split"].eq("test")].itertuples(index=False):
            _contained(modes, str(row.relative_path))
        result = cls(power, ais, modes, p, a, m)
        if result.train_supervisory_steps != EXPECTED_TRAIN_STEPS:
            raise ValueError("formal Train supervisory-step count differs")
        if not 0 < result.train_macro_transitions <= math.ceil(EXPECTED_TRAIN_STEPS / 5):
            raise ValueError("formal Train ONBOARD macro-transition count is invalid")
        return result

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            split: int((self._power["split"] == split).sum())
            for split in ("train", "validation", "test")
        }

    @property
    def train_supervisory_steps(self) -> int:
        return int(
            self._ais.loc[
                self._ais["split"].eq("train"), "supervisory_row_count"
            ].sum()
        )

    @property
    def train_macro_transitions(self) -> int:
        total = 0
        for row in self._modes[self._modes["split"].eq("train")].itertuples(index=False):
            path = _contained(self._mode_root, str(row.relative_path))
            if _sha256(path) != str(row.sha256):
                raise ValueError(f"{row.sample_id}: mode SHA-256 mismatch")
            values = tuple(pd.read_csv(path, encoding="utf-8-sig")["mode"].astype(str))
            total += _onboard_macro_transition_count(values)
        return int(total)

    @property
    def unresolved_mode_counts(self) -> dict[str, int]:
        return {
            split: int(
                self._modes.loc[
                    self._modes["split"].eq(split), "unresolved_row_count"
                ].sum()
            )
            for split in ("train", "validation", "test")
        }

    @property
    def opened_test_payloads(self) -> int:
        return self._opened_test_payloads

    def _load_episode(
        self, power_row: object, ais_row: object, mode_row: object
    ) -> FormalEpisode:
        power_path = _contained(self._power_root, str(power_row.relative_path))
        ais_path = _contained(self._ais_root, str(ais_row.relative_path))
        mode_path = _contained(self._mode_root, str(mode_row.relative_path))
        if _sha256(power_path) != str(power_row.sha256):
            raise ValueError(f"{power_row.sample_id}: power SHA-256 mismatch")
        if _sha256(ais_path) != str(ais_row.sha256):
            raise ValueError(f"{power_row.sample_id}: AIS SHA-256 mismatch")
        if _sha256(mode_path) != str(mode_row.sha256):
            raise ValueError(f"{power_row.sample_id}: mode SHA-256 mismatch")
        power = pd.read_csv(power_path, encoding="utf-8-sig")
        speed = pd.read_csv(ais_path, encoding="utf-8-sig")
        modes = pd.read_csv(mode_path, encoding="utf-8-sig")
        if not {"timestamp", "time_s", "load_total_kw"}.issubset(power.columns):
            raise ValueError("power payload schema mismatch")
        if not {"timestamp", "time_s", "speed_kn", "speed_provenance"}.issubset(speed.columns):
            raise ValueError("AIS payload schema mismatch")
        required_mode_columns = {
            "timestamp", "time_s", "load_total_kw", "p_fc_total_kw",
            "p_batt_bus_kw", "mode", "mode_reason",
        }
        if required_mode_columns.difference(modes.columns):
            raise ValueError("mode payload schema mismatch")
        duration = float(power_row.duration_s)
        time = pd.to_numeric(power["time_s"], errors="coerce")
        selected = power.loc[
            time.mod(30.0).eq(0.0) & time.add(30.0).le(duration),
            ["timestamp", "time_s", "load_total_kw"],
        ].reset_index(drop=True)
        if (
            len(selected) != int(ais_row.supervisory_row_count)
            or len(selected) != int(mode_row.supervisory_row_count)
            or len(speed) != len(selected)
            or len(modes) != len(selected)
        ):
            raise ValueError("power/AIS/mode supervisory row counts differ")
        selected_time = selected["time_s"].to_numpy(dtype=float)
        speed_time = pd.to_numeric(speed["time_s"], errors="coerce").to_numpy(dtype=float)
        mode_time = pd.to_numeric(modes["time_s"], errors="coerce").to_numpy(dtype=float)
        if not np.array_equal(selected_time, speed_time) or not np.array_equal(selected_time, mode_time):
            raise ValueError("power/AIS/mode supervisory axes differ")
        power_timestamp = pd.to_datetime(selected["timestamp"], errors="coerce")
        speed_timestamp = pd.to_datetime(speed["timestamp"], errors="coerce")
        mode_timestamp = pd.to_datetime(modes["timestamp"], errors="coerce")
        if (
            power_timestamp.isna().any()
            or speed_timestamp.isna().any()
            or mode_timestamp.isna().any()
            or not np.array_equal(power_timestamp.astype(str).to_numpy(), speed_timestamp.astype(str).to_numpy())
            or not np.array_equal(power_timestamp.astype(str).to_numpy(), mode_timestamp.astype(str).to_numpy())
        ):
            raise ValueError("power/AIS/mode timestamps differ")
        load = pd.to_numeric(selected["load_total_kw"], errors="coerce").to_numpy(dtype=float)
        velocity = pd.to_numeric(speed["speed_kn"], errors="coerce").to_numpy(dtype=float)
        provenance = tuple(str(value) for value in speed["speed_provenance"])
        fc = pd.to_numeric(modes["p_fc_total_kw"], errors="coerce").to_numpy(dtype=float)
        battery = pd.to_numeric(modes["p_batt_bus_kw"], errors="coerce").to_numpy(dtype=float)
        operating_mode = tuple(str(value) for value in modes["mode"])
        mode_reason = tuple(str(value) for value in modes["mode_reason"])
        if (
            not np.isfinite(load).all()
            or not np.isfinite(velocity).all()
            or not np.isfinite(fc).all()
            or not np.isfinite(battery).all()
            or (velocity < 0.0).any()
        ):
            raise ValueError("formal power/AIS/mode payload contains invalid values")
        allowed = {AIS_RAW_PROVENANCE, AIS_GAP_PROVENANCE}
        if not set(provenance).issubset(allowed):
            raise ValueError("AIS provenance is not formal")
        allowed_modes = {mode.value for mode in OperatingMode}
        if not set(operating_mode).issubset(allowed_modes):
            raise ValueError("mode payload contains an unknown mode")
        return FormalEpisode(
            str(power_row.parent), str(power_row.sample_id), str(power_row.split),
            tuple(power_timestamp), selected_time, load, velocity, provenance,
            fc, battery, operating_mode, mode_reason,
        )

    def load_split(self, split: str) -> tuple[FormalEpisode, ...]:
        if type(split) is not str:
            raise TypeError("split must be an exact str")
        if split == "test":
            raise PermissionError("Test payload access is forbidden during training/model selection")
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        if split in self._cache:
            return self._cache[split]
        power = self._power[self._power["split"].eq(split)].sort_values("sample_id")
        ais_by_id = self._ais.set_index("sample_id")
        modes_by_id = self._modes.set_index("sample_id")
        episodes = tuple(
            self._load_episode(
                row,
                ais_by_id.loc[str(row.sample_id)],
                modes_by_id.loc[str(row.sample_id)],
            )
            for row in power.itertuples(index=False)
        )
        self._cache[split] = episodes
        return episodes

    def load_train(self) -> tuple[FormalEpisode, ...]:
        return self.load_split("train")

    def load_validation(self) -> tuple[FormalEpisode, ...]:
        return self.load_split("validation")


__all__ = ["FormalEpisode", "FormalTrainingDataset"]
