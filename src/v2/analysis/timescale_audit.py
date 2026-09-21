"""Train-only statistical diagnostics for the provisional v2 time scales.

This module deliberately does not select a formal time scale.  It produces a
sealed diagnostic snapshot for the fixed provisional comparison ``M in
{5, 10}`` while keeping the lower-controller horizon fixed at ``N = 5``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import TypeVar

from ..contracts import DATASET_VERSION
from .action_screening import (
    DataSplit,
    DatasetProvenance,
    HeldOutSelectionError,
)


FORMAL_TIMESCALE_SELECTION_STATUS = "NO-GO"
PROVISIONAL_N_MPC = 5
PROVISIONAL_SWITCH_CANDIDATES = (5, 10)


class SelectionParameter(Enum):
    N_MPC = "n_mpc"
    DQN_SWITCH_STEPS = "dqn_switch_steps"
    TAU_LPF_SECONDS = "tau_lpf_seconds"
    SOC_DEADBAND = "soc_deadband"
    STATE_SCHEMA = "state_schema"
    ACTION_CATALOG = "action_catalog"
    REWARD_SCALE = "reward_scale"
    OBJECTIVE_NORMALIZATION = "objective_normalization"


PayloadT = TypeVar("PayloadT")


def _nonempty_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


def _exact_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _canonical_train_provenance(value: object) -> DatasetProvenance:
    if type(value) is not DatasetProvenance:
        raise TypeError("an exact DatasetProvenance is required")
    if type(value.dataset_version) is not str or type(value.provenance_id) is not str:
        raise TypeError("provenance strings must remain exact strings")
    if type(value.split) is not DataSplit:
        raise TypeError("provenance split must remain an exact DataSplit")
    canonical = DatasetProvenance(
        value.dataset_version,
        value.provenance_id,
        value.split,
    )
    if vars(value) != vars(canonical):
        raise ValueError("provenance contains mutated or injected fields")
    if canonical.dataset_version != DATASET_VERSION:
        raise ValueError(f"dataset_version must equal {DATASET_VERSION}")
    if canonical.split is not DataSplit.TRAIN:
        raise HeldOutSelectionError(
            "selection and calibration of N, M, tau_LPF, deadband, state schema, "
            "action catalog, reward scale, and objective normalization are Train-only"
        )
    return canonical


def require_train_selection(
    provenance: object,
    parameter: object,
) -> DatasetProvenance:
    """Validate a centralized Train-only method-selection boundary."""

    checked = _canonical_train_provenance(provenance)
    if type(parameter) is not SelectionParameter:
        raise TypeError("parameter must be an exact SelectionParameter")
    return checked


def load_train_selection_payload(
    *,
    provenance: object,
    parameter: object,
    payload_loader: Callable[[], PayloadT],
) -> PayloadT:
    """Run the provenance guard before invoking a lazy payload reader."""

    require_train_selection(provenance, parameter)
    if not callable(payload_loader):
        raise TypeError("payload_loader must be callable")
    return payload_loader()


def _finite_series(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("time-series payload must be an iterable of real scalars")
    try:
        source = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("time-series payload must be iterable") from exc
    if len(source) < max(PROVISIONAL_SWITCH_CANDIDATES):
        raise ValueError("time-series payload needs at least ten samples")
    return tuple(_finite(item, f"time-series sample {index}") for index, item in enumerate(source))


def _scaled_mean(values: tuple[float, ...]) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized_mean = math.fsum(value / scale for value in values) / len(values)
    result = normalized_mean * scale
    if not math.isfinite(result):
        raise ValueError("mean result must be finite")
    return result


def _variance(values: tuple[float, ...]) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = tuple(value / scale for value in values)
    mean = math.fsum(normalized) / len(normalized)
    unit_variance = math.fsum((value - mean) ** 2 for value in normalized) / len(normalized)
    if unit_variance == 0.0:
        return 0.0
    standard_deviation = math.sqrt(unit_variance) * scale
    result = standard_deviation * standard_deviation
    if not math.isfinite(result):
        raise ValueError("variance result must be finite")
    return result


def _autocorrelation(values: tuple[float, ...], lag: int) -> float:
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = tuple(value / scale for value in values)
    mean = math.fsum(normalized) / len(normalized)
    centered = tuple(value - mean for value in normalized)
    denominator = math.fsum(value * value for value in centered)
    if denominator == 0.0:
        return 0.0
    numerator = math.fsum(
        centered[index] * centered[index - lag]
        for index in range(lag, len(centered))
    )
    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError("autocorrelation result must be finite")
    return result


@dataclass(frozen=True)
class TimeSeriesDiagnostics:
    autocorrelation_lags: tuple[int, ...]
    autocorrelation: tuple[float, ...]
    rolling_window_samples: int
    rolling_variance: tuple[float, ...]
    change_threshold: float
    change_point_indices: tuple[int, ...]
    regime_durations_samples: tuple[int, ...]
    regime_durations_seconds: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.autocorrelation_lags) is not tuple or not self.autocorrelation_lags:
            raise TypeError("autocorrelation_lags must remain a nonempty tuple")
        lags = tuple(
            _exact_nonnegative_int(value, "autocorrelation lag")
            for value in self.autocorrelation_lags
        )
        if any(value == 0 for value in lags) or len(set(lags)) != len(lags):
            raise ValueError("autocorrelation lags must be positive and unique")
        if type(self.autocorrelation) is not tuple or len(self.autocorrelation) != len(lags):
            raise TypeError("autocorrelation must be a matching tuple")
        autocorrelation = tuple(
            _finite(value, "autocorrelation value") for value in self.autocorrelation
        )
        window = _exact_nonnegative_int(self.rolling_window_samples, "rolling window")
        if window < 2:
            raise ValueError("rolling window must be at least two samples")
        if type(self.rolling_variance) is not tuple or not self.rolling_variance:
            raise TypeError("rolling_variance must remain a nonempty tuple")
        rolling = tuple(_finite(value, "rolling variance") for value in self.rolling_variance)
        if any(value < 0.0 for value in rolling):
            raise ValueError("rolling variances must be nonnegative")
        threshold = _finite(self.change_threshold, "change_threshold", positive=True)
        if type(self.change_point_indices) is not tuple:
            raise TypeError("change_point_indices must remain a tuple")
        change_points = tuple(
            _exact_nonnegative_int(value, "change-point index")
            for value in self.change_point_indices
        )
        if any(value == 0 for value in change_points) or tuple(sorted(set(change_points))) != change_points:
            raise ValueError("change points must be strictly increasing positive indices")
        if type(self.regime_durations_samples) is not tuple or not self.regime_durations_samples:
            raise TypeError("regime sample durations must remain a nonempty tuple")
        durations = tuple(
            _exact_nonnegative_int(value, "regime duration")
            for value in self.regime_durations_samples
        )
        if any(value == 0 for value in durations):
            raise ValueError("regime durations must be positive")
        if type(self.regime_durations_seconds) is not tuple or len(self.regime_durations_seconds) != len(durations):
            raise TypeError("regime second durations must be a matching tuple")
        duration_seconds = tuple(
            _finite(value, "regime duration seconds", positive=True)
            for value in self.regime_durations_seconds
        )
        object.__setattr__(self, "autocorrelation_lags", lags)
        object.__setattr__(self, "autocorrelation", autocorrelation)
        object.__setattr__(self, "rolling_window_samples", window)
        object.__setattr__(self, "rolling_variance", rolling)
        object.__setattr__(self, "change_threshold", threshold)
        object.__setattr__(self, "change_point_indices", change_points)
        object.__setattr__(self, "regime_durations_samples", durations)
        object.__setattr__(self, "regime_durations_seconds", duration_seconds)


@dataclass(frozen=True)
class SwitchScaleDiagnostic:
    dqn_switch_steps: int
    complete_interval_count: int
    discarded_tail_samples: int
    macro_means: tuple[float, ...]
    mean_within_interval_variance: float

    def __post_init__(self) -> None:
        steps = _exact_nonnegative_int(self.dqn_switch_steps, "dqn_switch_steps")
        count = _exact_nonnegative_int(self.complete_interval_count, "complete_interval_count")
        discarded = _exact_nonnegative_int(
            self.discarded_tail_samples,
            "discarded_tail_samples",
        )
        if steps not in PROVISIONAL_SWITCH_CANDIDATES:
            raise ValueError("dqn_switch_steps must be one of the fixed provisional candidates")
        if count <= 0:
            raise ValueError("at least one complete macro interval is required")
        if type(self.macro_means) is not tuple or len(self.macro_means) != count:
            raise TypeError("macro_means must be a tuple matching interval count")
        means = tuple(_finite(value, "macro mean") for value in self.macro_means)
        within = _finite(
            self.mean_within_interval_variance,
            "mean_within_interval_variance",
        )
        if within < 0.0:
            raise ValueError("mean within-interval variance must be nonnegative")
        object.__setattr__(self, "dqn_switch_steps", steps)
        object.__setattr__(self, "complete_interval_count", count)
        object.__setattr__(self, "discarded_tail_samples", discarded)
        object.__setattr__(self, "macro_means", means)
        object.__setattr__(self, "mean_within_interval_variance", within)


_AUDIT_SEAL = object()


@dataclass(frozen=True, init=False)
class TimeScaleAuditResult:
    provenance: DatasetProvenance
    n_mpc: int
    sample_seconds: float
    sample_count: int
    candidate_switch_steps: tuple[int, int]
    diagnostics: TimeSeriesDiagnostics
    switch_sensitivity: tuple[SwitchScaleDiagnostic, SwitchScaleDiagnostic]
    formal_selection_status: str
    selected_dqn_switch_steps: None
    digest: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("time-scale audit results can only be created by run_timescale_audit")

    @classmethod
    def _create(
        cls,
        seal: object,
        *,
        provenance: DatasetProvenance,
        sample_seconds: float,
        sample_count: int,
        diagnostics: TimeSeriesDiagnostics,
        switch_sensitivity: tuple[SwitchScaleDiagnostic, SwitchScaleDiagnostic],
    ) -> "TimeScaleAuditResult":
        if seal is not _AUDIT_SEAL:
            raise TypeError("invalid time-scale audit construction seal")
        instance = object.__new__(cls)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "n_mpc", PROVISIONAL_N_MPC)
        object.__setattr__(instance, "sample_seconds", sample_seconds)
        object.__setattr__(instance, "sample_count", sample_count)
        object.__setattr__(instance, "candidate_switch_steps", PROVISIONAL_SWITCH_CANDIDATES)
        object.__setattr__(instance, "diagnostics", diagnostics)
        object.__setattr__(instance, "switch_sensitivity", switch_sensitivity)
        object.__setattr__(instance, "formal_selection_status", FORMAL_TIMESCALE_SELECTION_STATUS)
        object.__setattr__(instance, "selected_dqn_switch_steps", None)
        object.__setattr__(instance, "digest", _result_digest(instance))
        return instance

    def validate(self) -> "TimeScaleAuditResult":
        return _validate_result(self)


def _provenance_payload(value: DatasetProvenance) -> list[str]:
    return [value.dataset_version, value.provenance_id, value.split.value]


def _diagnostics_payload(value: TimeSeriesDiagnostics) -> dict[str, object]:
    return {
        "autocorrelation_lags": value.autocorrelation_lags,
        "autocorrelation": value.autocorrelation,
        "rolling_window_samples": value.rolling_window_samples,
        "rolling_variance": value.rolling_variance,
        "change_threshold": value.change_threshold,
        "change_point_indices": value.change_point_indices,
        "regime_durations_samples": value.regime_durations_samples,
        "regime_durations_seconds": value.regime_durations_seconds,
    }


def _switch_payload(value: SwitchScaleDiagnostic) -> dict[str, object]:
    return {
        "dqn_switch_steps": value.dqn_switch_steps,
        "complete_interval_count": value.complete_interval_count,
        "discarded_tail_samples": value.discarded_tail_samples,
        "macro_means": value.macro_means,
        "mean_within_interval_variance": value.mean_within_interval_variance,
    }


def _result_digest(value: TimeScaleAuditResult) -> str:
    encoded = json.dumps(
        {
            "kind": "train_only_timescale_audit_v1",
            "provenance": _provenance_payload(value.provenance),
            "n_mpc": value.n_mpc,
            "sample_seconds": value.sample_seconds,
            "sample_count": value.sample_count,
            "candidate_switch_steps": value.candidate_switch_steps,
            "diagnostics": _diagnostics_payload(value.diagnostics),
            "switch_sensitivity": tuple(
                _switch_payload(item) for item in value.switch_sensitivity
            ),
            "formal_selection_status": value.formal_selection_status,
            "selected_dqn_switch_steps": value.selected_dqn_switch_steps,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_result(value: object) -> TimeScaleAuditResult:
    if type(value) is not TimeScaleAuditResult:
        raise TypeError("result must be an exact TimeScaleAuditResult")
    expected_fields = {
        "provenance",
        "n_mpc",
        "sample_seconds",
        "sample_count",
        "candidate_switch_steps",
        "diagnostics",
        "switch_sensitivity",
        "formal_selection_status",
        "selected_dqn_switch_steps",
        "digest",
    }
    if set(vars(value)) != expected_fields:
        raise ValueError("time-scale audit result contains injected or missing fields")
    provenance = _canonical_train_provenance(value.provenance)
    if value.provenance != provenance or vars(value.provenance) != vars(provenance):
        raise ValueError("result provenance is not canonical")
    if type(value.n_mpc) is not int or value.n_mpc != PROVISIONAL_N_MPC:
        raise ValueError("result N must remain fixed at five")
    seconds = _finite(value.sample_seconds, "result sample_seconds", positive=True)
    sample_count = _exact_nonnegative_int(value.sample_count, "result sample_count")
    if sample_count < max(PROVISIONAL_SWITCH_CANDIDATES):
        raise ValueError("result sample_count must support both M candidates")
    if (
        type(value.candidate_switch_steps) is not tuple
        or any(type(item) is not int for item in value.candidate_switch_steps)
        or value.candidate_switch_steps != PROVISIONAL_SWITCH_CANDIDATES
    ):
        raise ValueError("result M candidates must remain exactly (5, 10)")
    if type(value.diagnostics) is not TimeSeriesDiagnostics:
        raise TypeError("result diagnostics must remain exact TimeSeriesDiagnostics")
    diagnostics = TimeSeriesDiagnostics(**_diagnostics_payload(value.diagnostics))
    if vars(value.diagnostics) != vars(diagnostics):
        raise ValueError("time-series diagnostics contain injected or mutated fields")
    if any(lag >= sample_count for lag in diagnostics.autocorrelation_lags):
        raise ValueError("result autocorrelation lag exceeds sample_count")
    if len(diagnostics.rolling_variance) != sample_count - diagnostics.rolling_window_samples + 1:
        raise ValueError("result rolling variance length does not match sample_count")
    if any(index >= sample_count for index in diagnostics.change_point_indices):
        raise ValueError("result change point exceeds sample_count")
    if sum(diagnostics.regime_durations_samples) != sample_count:
        raise ValueError("result regime durations do not cover sample_count")
    expected_duration_seconds = tuple(
        seconds * count for count in diagnostics.regime_durations_samples
    )
    if (
        not all(math.isfinite(item) for item in expected_duration_seconds)
        or diagnostics.regime_durations_seconds != expected_duration_seconds
    ):
        raise ValueError("result regime seconds do not match sample count and sample time")
    if type(value.switch_sensitivity) is not tuple or len(value.switch_sensitivity) != 2:
        raise TypeError("result switch_sensitivity must remain a two-item tuple")
    rebuilt = tuple(SwitchScaleDiagnostic(**_switch_payload(item)) for item in value.switch_sensitivity)
    if any(vars(actual) != vars(expected) for actual, expected in zip(value.switch_sensitivity, rebuilt)):
        raise ValueError("switch sensitivity contains injected or mutated fields")
    if tuple(item.dqn_switch_steps for item in rebuilt) != PROVISIONAL_SWITCH_CANDIDATES:
        raise ValueError("switch sensitivity ordering must remain (5, 10)")
    if any(
        item.complete_interval_count * item.dqn_switch_steps + item.discarded_tail_samples
        != sample_count
        or item.discarded_tail_samples >= item.dqn_switch_steps
        for item in rebuilt
    ):
        raise ValueError("switch sensitivity coverage does not match sample_count")
    if (
        type(value.formal_selection_status) is not str
        or value.formal_selection_status != FORMAL_TIMESCALE_SELECTION_STATUS
    ):
        raise ValueError("formal time-scale selection must remain NO-GO")
    if value.selected_dqn_switch_steps is not None:
        raise ValueError("a formal switch time must not be selected")
    if type(value.digest) is not str or value.digest != _result_digest(value):
        raise ValueError("time-scale audit digest mismatch")
    return value


def _make_diagnostics(
    values: tuple[float, ...],
    *,
    sample_seconds: float,
    autocorrelation_lags: object,
    rolling_window_samples: object,
    change_threshold: object,
) -> TimeSeriesDiagnostics:
    if type(autocorrelation_lags) is not tuple or not autocorrelation_lags:
        raise TypeError("autocorrelation_lags must be a nonempty tuple")
    lags = tuple(
        _exact_nonnegative_int(value, "autocorrelation lag")
        for value in autocorrelation_lags
    )
    if any(value == 0 or value >= len(values) for value in lags) or len(set(lags)) != len(lags):
        raise ValueError("lags must be positive, unique, and shorter than the series")
    window = _exact_nonnegative_int(rolling_window_samples, "rolling_window_samples")
    if not 2 <= window <= len(values):
        raise ValueError("rolling window must lie between two and the series length")
    threshold = _finite(change_threshold, "change_threshold", positive=True)
    rolling = tuple(
        _variance(values[start : start + window])
        for start in range(len(values) - window + 1)
    )
    changes = tuple(
        index
        for index in range(1, len(values))
        if abs(values[index] - values[index - 1]) >= threshold
    )
    boundaries = (0,) + changes + (len(values),)
    regime_samples = tuple(
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
    )
    return TimeSeriesDiagnostics(
        autocorrelation_lags=lags,
        autocorrelation=tuple(_autocorrelation(values, lag) for lag in lags),
        rolling_window_samples=window,
        rolling_variance=rolling,
        change_threshold=threshold,
        change_point_indices=changes,
        regime_durations_samples=regime_samples,
        regime_durations_seconds=tuple(sample_seconds * count for count in regime_samples),
    )


def _switch_diagnostic(values: tuple[float, ...], steps: int) -> SwitchScaleDiagnostic:
    count = len(values) // steps
    intervals = tuple(
        values[index * steps : (index + 1) * steps]
        for index in range(count)
    )
    means = tuple(_scaled_mean(interval) for interval in intervals)
    variances = tuple(_variance(interval) for interval in intervals)
    return SwitchScaleDiagnostic(
        dqn_switch_steps=steps,
        complete_interval_count=count,
        discarded_tail_samples=len(values) - count * steps,
        macro_means=means,
        mean_within_interval_variance=math.fsum(variances) / count,
    )


def run_timescale_audit(
    *,
    provenance: object,
    payload_loader: Callable[[], Iterable[Real]],
    sample_seconds: object,
    autocorrelation_lags: object,
    rolling_window_samples: object,
    change_threshold: object,
    n_mpc: object = PROVISIONAL_N_MPC,
    candidate_switch_steps: object = PROVISIONAL_SWITCH_CANDIDATES,
) -> TimeScaleAuditResult:
    """Build a deterministic diagnostic snapshot without making a selection."""

    canonical_provenance = require_train_selection(
        provenance,
        SelectionParameter.DQN_SWITCH_STEPS,
    )
    if type(n_mpc) is not int:
        raise TypeError("n_mpc must be an exact integer")
    if n_mpc != PROVISIONAL_N_MPC:
        raise ValueError("the provisional audit keeps n_mpc fixed at five")
    if type(candidate_switch_steps) is not tuple:
        raise TypeError("candidate_switch_steps must be an exact tuple")
    if (
        any(type(item) is not int for item in candidate_switch_steps)
        or candidate_switch_steps != PROVISIONAL_SWITCH_CANDIDATES
    ):
        raise ValueError("the only permitted M sensitivity is exactly (5, 10)")
    seconds = _finite(sample_seconds, "sample_seconds", positive=True)
    payload = load_train_selection_payload(
        provenance=canonical_provenance,
        parameter=SelectionParameter.DQN_SWITCH_STEPS,
        payload_loader=payload_loader,
    )
    values = _finite_series(payload)
    diagnostics = _make_diagnostics(
        values,
        sample_seconds=seconds,
        autocorrelation_lags=autocorrelation_lags,
        rolling_window_samples=rolling_window_samples,
        change_threshold=change_threshold,
    )
    sensitivity = tuple(
        _switch_diagnostic(values, steps)
        for steps in PROVISIONAL_SWITCH_CANDIDATES
    )
    result = TimeScaleAuditResult._create(
        _AUDIT_SEAL,
        provenance=DatasetProvenance(
            canonical_provenance.dataset_version,
            canonical_provenance.provenance_id,
            canonical_provenance.split,
        ),
        sample_seconds=seconds,
        sample_count=len(values),
        diagnostics=diagnostics,
        switch_sensitivity=sensitivity,  # type: ignore[arg-type]
    )
    return _validate_result(result)


__all__ = [
    "FORMAL_TIMESCALE_SELECTION_STATUS",
    "PROVISIONAL_N_MPC",
    "PROVISIONAL_SWITCH_CANDIDATES",
    "SelectionParameter",
    "SwitchScaleDiagnostic",
    "TimeScaleAuditResult",
    "TimeSeriesDiagnostics",
    "load_train_selection_payload",
    "require_train_selection",
    "run_timescale_audit",
]
