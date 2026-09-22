"""Strict post-alignment aggregation for twelve equal-capacity BMS clusters."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real
from typing import Sequence


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _timestamp(value: object, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be an exact datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class AlignedBatteryClusterSample:
    """One deduplicated, causal cluster match for one supervisory timestamp."""

    cluster_id: str
    supervisory_timestamp: datetime
    source_timestamp: datetime
    soc: float
    power_kw: float
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _text(self.cluster_id, "cluster_id"))
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))
        supervisory = _timestamp(self.supervisory_timestamp, "supervisory_timestamp")
        source = _timestamp(self.source_timestamp, "source_timestamp")
        if source > supervisory:
            raise ValueError("source_timestamp must not be later than supervisory_timestamp")
        soc = _finite_scalar(self.soc, "soc")
        if not 0.0 <= soc <= 1.0:
            raise ValueError("soc must lie in [0, 1]")
        object.__setattr__(self, "soc", soc)
        object.__setattr__(self, "power_kw", _finite_scalar(self.power_kw, "power_kw"))


@dataclass(frozen=True)
class AggregatedBatteryState:
    supervisory_timestamp: datetime
    soc_system: float
    p_batt_total_kw: float
    cluster_provenance: tuple[tuple[str, datetime, str], ...]


def aggregate_battery_clusters(
    samples: Sequence[AlignedBatteryClusterSample],
    *,
    expected_cluster_ids: Sequence[str],
    max_age_seconds: float,
) -> AggregatedBatteryState:
    """Aggregate one strict state after per-cluster deduplication and alignment.

    The function fails closed unless exactly twelve distinct expected clusters
    are present at one supervisory timestamp and every source match is causal
    and fresh. It intentionally has no default freshness tolerance.
    """

    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence")
    if isinstance(expected_cluster_ids, (str, bytes)) or not isinstance(
        expected_cluster_ids, Sequence
    ):
        raise TypeError("expected_cluster_ids must be a sequence")
    expected = tuple(_text(value, "expected_cluster_ids item") for value in expected_cluster_ids)
    if len(expected) != 12 or len(set(expected)) != 12:
        raise ValueError("expected_cluster_ids must contain exactly twelve unique IDs")
    age_limit = _finite_scalar(max_age_seconds, "max_age_seconds")
    if age_limit < 0.0:
        raise ValueError("max_age_seconds must be nonnegative")
    if len(samples) != 12 or any(type(sample) is not AlignedBatteryClusterSample for sample in samples):
        raise ValueError("exactly twelve aligned cluster samples are required")

    by_id: dict[str, AlignedBatteryClusterSample] = {}
    for sample in samples:
        if sample.cluster_id in by_id:
            raise ValueError("duplicate cluster sample at supervisory timestamp")
        by_id[sample.cluster_id] = sample
    if set(by_id) != set(expected):
        raise ValueError("cluster sample IDs do not match the required twelve clusters")

    ordered = tuple(by_id[cluster_id] for cluster_id in expected)
    supervisory = ordered[0].supervisory_timestamp
    if any(sample.supervisory_timestamp != supervisory for sample in ordered):
        raise ValueError("all cluster samples must share one supervisory timestamp")
    for sample in ordered:
        age = (supervisory - sample.source_timestamp).total_seconds()
        if age < 0.0:
            raise ValueError("future source samples are prohibited")
        if age > age_limit:
            raise ValueError("cluster sample exceeds max_age_seconds")

    return AggregatedBatteryState(
        supervisory_timestamp=supervisory,
        soc_system=math.fsum(sample.soc for sample in ordered) / 12.0,
        p_batt_total_kw=math.fsum(sample.power_kw for sample in ordered),
        cluster_provenance=tuple(
            (sample.cluster_id, sample.source_timestamp, sample.provenance)
            for sample in ordered
        ),
    )
