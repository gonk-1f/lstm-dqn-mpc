"""Train-only, tamper-evident interfaces for offline action screening.

The algorithms operate only on caller-supplied evidence. They do not claim
that the repository's current data gate has passed or that a final catalog
exists. Every selection stage consumes the sealed result of its predecessor.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from typing import Any

from ..control.nonlinear_mpc import MPCWeights
from ..dqn.action_space import ActionCandidate, generate_candidate_action_bank


class HeldOutSelectionError(PermissionError):
    """Raised before held-out or unknown data can affect method selection."""


class ScreeningLineageError(RuntimeError):
    """Raised when a sealed screening result has been forged or mutated."""


class CatalogFinalizationError(RuntimeError):
    """Raised when final-catalog evidence is incomplete or has not passed."""


class DataSplit(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"
    UNKNOWN = "Unknown"


class MetricDirection(Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class DistanceThresholdRule(Enum):
    ZERO = "zero"
    MIN_POSITIVE_PAIRWISE = "min_positive_pairwise"
    MEDIAN_PAIRWISE = "median_pairwise"


def _nonempty_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _repeated_runs(value: object) -> int:
    if type(value) is not int:
        raise TypeError("repeated_runs must be an exact integer")
    if value < 2:
        raise ValueError("a reproducibility audit requires at least two runs")
    return value


@dataclass(frozen=True)
class DatasetProvenance:
    dataset_version: str
    provenance_id: str
    split: DataSplit

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dataset_version", _nonempty_text(self.dataset_version, "dataset_version")
        )
        object.__setattr__(
            self, "provenance_id", _nonempty_text(self.provenance_id, "provenance_id")
        )
        if type(self.split) is not DataSplit:
            raise TypeError("split must be an exact DataSplit; string coercion is forbidden")


def _validate_provenance(value: object) -> DatasetProvenance:
    if type(value) is not DatasetProvenance:
        raise TypeError("an exact DatasetProvenance is required")
    _nonempty_text(value.dataset_version, "dataset_version")
    _nonempty_text(value.provenance_id, "provenance_id")
    if type(value.split) is not DataSplit:
        raise TypeError("split must remain an exact DataSplit")
    return value


def _require_train(provenance: object) -> DatasetProvenance:
    checked = _validate_provenance(provenance)
    if checked.split is not DataSplit.TRAIN:
        raise HeldOutSelectionError(
            "candidate removal, thresholds, clustering, medoids, K, and final "
            "catalog selection are Train-only"
        )
    return checked


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: MetricDirection
    scale: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "metric name"))
        if type(self.direction) is not MetricDirection:
            raise TypeError("metric direction must be an exact MetricDirection")
        scale = _nonnegative_finite(self.scale, "metric scale")
        if scale == 0.0:
            raise ValueError("metric scale must be positive")
        object.__setattr__(self, "scale", scale)


def _validate_metric(value: object) -> MetricDefinition:
    if type(value) is not MetricDefinition:
        raise TypeError("metrics must contain exact MetricDefinition values")
    _nonempty_text(value.name, "metric name")
    if type(value.direction) is not MetricDirection:
        raise TypeError("metric direction must remain an exact MetricDirection")
    if type(value.scale) is not float or not math.isfinite(value.scale) or value.scale <= 0.0:
        raise ValueError("metric scale must remain a finite positive float")
    return value


@dataclass(frozen=True)
class BehaviorFingerprint:
    candidate_id: str
    provenance: DatasetProvenance
    metrics: tuple[MetricDefinition, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _nonempty_text(self.candidate_id, "candidate_id")
        )
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("fingerprint provenance must be exact DatasetProvenance")
        _validate_provenance(self.provenance)
        if isinstance(self.metrics, (str, bytes)):
            raise TypeError("metrics must be a sequence of MetricDefinition values")
        try:
            metrics = tuple(self.metrics)
        except TypeError as exc:
            raise TypeError("metrics must be iterable") from exc
        if not metrics or any(type(metric) is not MetricDefinition for metric in metrics):
            raise TypeError("metrics must contain exact MetricDefinition values")
        names = tuple(metric.name for metric in metrics)
        if len(set(names)) != len(names):
            raise ValueError("fingerprint metric names must be unique")
        if isinstance(self.values, (str, bytes)):
            raise TypeError("fingerprint values must be a numeric sequence")
        try:
            source_values = tuple(self.values)
        except TypeError as exc:
            raise TypeError("fingerprint values must be iterable") from exc
        if len(source_values) != len(metrics):
            raise ValueError("fingerprint values must match the metric schema length")
        values: list[float] = []
        for index, value in enumerate(source_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"fingerprint value {index} must be a real non-bool scalar")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError("fingerprint values must all be finite")
            if not math.isfinite(converted / metrics[index].scale):
                raise ValueError("fingerprint normalized values must all be finite")
            values.append(converted)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "values", tuple(values))

    @property
    def normalized_minimization_vector(self) -> tuple[float, ...]:
        return tuple(
            value / metric.scale
            if metric.direction is MetricDirection.MINIMIZE
            else -(value / metric.scale)
            for metric, value in zip(self.metrics, self.values)
        )


def _validate_fingerprint(value: object) -> BehaviorFingerprint:
    if type(value) is not BehaviorFingerprint:
        raise TypeError("fingerprint must be an exact BehaviorFingerprint")
    _nonempty_text(value.candidate_id, "candidate_id")
    _validate_provenance(value.provenance)
    if type(value.metrics) is not tuple or not value.metrics:
        raise TypeError("fingerprint metrics must remain a nonempty tuple")
    for metric in value.metrics:
        _validate_metric(metric)
    if len({metric.name for metric in value.metrics}) != len(value.metrics):
        raise ValueError("fingerprint metric names must remain unique")
    if type(value.values) is not tuple or len(value.values) != len(value.metrics):
        raise ValueError("fingerprint values must remain a matching tuple")
    for metric, item in zip(value.metrics, value.values):
        if type(item) is not float or not math.isfinite(item):
            raise ValueError("fingerprint values must remain finite floats")
        if not math.isfinite(item / metric.scale):
            raise ValueError("fingerprint normalized values must remain finite")
    return value


@dataclass(frozen=True)
class FeasibilityResult:
    candidate_id: str
    provenance: DatasetProvenance
    passed: bool
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _nonempty_text(self.candidate_id, "candidate_id")
        )
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("feasibility provenance must be exact DatasetProvenance")
        _validate_provenance(self.provenance)
        object.__setattr__(self, "passed", _exact_bool(self.passed, "feasibility passed"))
        object.__setattr__(self, "reason", _nonempty_text(self.reason, "feasibility reason"))


@dataclass(frozen=True)
class SolverReproducibilityResult:
    candidate_id: str
    provenance: DatasetProvenance
    passed: bool
    repeated_runs: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _nonempty_text(self.candidate_id, "candidate_id")
        )
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("solver provenance must be exact DatasetProvenance")
        _validate_provenance(self.provenance)
        object.__setattr__(self, "passed", _exact_bool(self.passed, "solver passed"))
        object.__setattr__(self, "repeated_runs", _repeated_runs(self.repeated_runs))
        object.__setattr__(self, "reason", _nonempty_text(self.reason, "solver reason"))


@dataclass(frozen=True)
class CandidateScreeningRecord:
    candidate_id: str
    feasibility: FeasibilityResult
    reproducibility: SolverReproducibilityResult
    fingerprint: BehaviorFingerprint

    def __post_init__(self) -> None:
        candidate_id = _nonempty_text(self.candidate_id, "candidate_id")
        nested = (
            (self.feasibility, FeasibilityResult, "feasibility"),
            (self.reproducibility, SolverReproducibilityResult, "reproducibility"),
            (self.fingerprint, BehaviorFingerprint, "fingerprint"),
        )
        for value, expected, name in nested:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}")
            if value.candidate_id != candidate_id:
                raise ValueError("all screening evidence must identify the same candidate")
        provenances = (
            _require_train(self.feasibility.provenance),
            _require_train(self.reproducibility.provenance),
            _require_train(self.fingerprint.provenance),
        )
        if provenances[1:] != (provenances[0], provenances[0]):
            raise ValueError("all screening evidence must share the same exact provenance")
        object.__setattr__(self, "candidate_id", candidate_id)

    @property
    def provenance(self) -> DatasetProvenance:
        return self.feasibility.provenance


def _validate_record(value: object) -> CandidateScreeningRecord:
    if type(value) is not CandidateScreeningRecord:
        raise TypeError("records must contain exact CandidateScreeningRecord values")
    candidate_id = _nonempty_text(value.candidate_id, "candidate_id")
    feasibility = value.feasibility
    reproducibility = value.reproducibility
    fingerprint = _validate_fingerprint(value.fingerprint)
    if type(feasibility) is not FeasibilityResult:
        raise TypeError("feasibility must remain exact FeasibilityResult")
    if type(reproducibility) is not SolverReproducibilityResult:
        raise TypeError("reproducibility must remain exact SolverReproducibilityResult")
    _nonempty_text(feasibility.candidate_id, "candidate_id")
    _nonempty_text(reproducibility.candidate_id, "candidate_id")
    _exact_bool(feasibility.passed, "feasibility passed")
    _exact_bool(reproducibility.passed, "solver passed")
    _repeated_runs(reproducibility.repeated_runs)
    _nonempty_text(feasibility.reason, "feasibility reason")
    _nonempty_text(reproducibility.reason, "solver reason")
    if candidate_id != feasibility.candidate_id or candidate_id != reproducibility.candidate_id:
        raise ValueError("screening candidate identity has been altered")
    if candidate_id != fingerprint.candidate_id:
        raise ValueError("fingerprint candidate identity has been altered")
    provenances = (
        _require_train(feasibility.provenance),
        _require_train(reproducibility.provenance),
        _require_train(fingerprint.provenance),
    )
    if provenances[1:] != (provenances[0], provenances[0]):
        raise ValueError("all screening evidence must share the same exact provenance")
    return value


@dataclass(frozen=True)
class DataReadinessEvidence:
    provenance: DatasetProvenance
    passed: bool
    audit_id: str
    reason: str
    digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("data readiness requires exact DatasetProvenance")
        _validate_provenance(self.provenance)
        object.__setattr__(self, "passed", _exact_bool(self.passed, "data readiness passed"))
        object.__setattr__(self, "audit_id", _nonempty_text(self.audit_id, "data audit_id"))
        object.__setattr__(self, "reason", _nonempty_text(self.reason, "data audit reason"))
        object.__setattr__(self, "digest", _readiness_digest(self))

    def validate(self) -> DataReadinessEvidence:
        return _validate_readiness(self)


@dataclass(frozen=True)
class SolverReproducibilityAudit:
    provenance: DatasetProvenance
    passed: bool
    candidate_ids: tuple[str, ...]
    repeated_runs: int
    audit_id: str
    reason: str
    digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("solver audit requires exact DatasetProvenance")
        _validate_provenance(self.provenance)
        object.__setattr__(self, "passed", _exact_bool(self.passed, "solver audit passed"))
        if isinstance(self.candidate_ids, (str, bytes)):
            raise TypeError("candidate_ids must be a sequence")
        try:
            ids = tuple(_nonempty_text(value, "candidate_id") for value in self.candidate_ids)
        except TypeError as exc:
            raise TypeError("candidate_ids must be iterable exact strings") from exc
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("solver audit candidate_ids must be nonempty and unique")
        object.__setattr__(self, "candidate_ids", ids)
        object.__setattr__(self, "repeated_runs", _repeated_runs(self.repeated_runs))
        object.__setattr__(self, "audit_id", _nonempty_text(self.audit_id, "solver audit_id"))
        object.__setattr__(self, "reason", _nonempty_text(self.reason, "solver audit reason"))
        object.__setattr__(self, "digest", _solver_audit_digest(self))

    def validate(self) -> SolverReproducibilityAudit:
        return _validate_solver_audit(self)


def _validated_records(
    records: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    if isinstance(records, (str, bytes)):
        raise TypeError("records must be an iterable of screening records")
    try:
        result = tuple(records)
    except TypeError as exc:
        raise TypeError("records must be iterable") from exc
    if not result:
        raise ValueError("screening records cannot be empty")
    for record in result:
        _validate_record(record)
    provenances = tuple(record.provenance for record in result)
    if any(provenance != provenances[0] for provenance in provenances[1:]):
        raise ValueError("all screening records must share the same exact provenance")
    ids = tuple(record.candidate_id for record in result)
    if len(set(ids)) != len(ids):
        raise ValueError("screening candidate IDs must be unique")
    schema = result[0].fingerprint.metrics
    if any(record.fingerprint.metrics != schema for record in result[1:]):
        raise ValueError("all behavior fingerprints must share one exact metric schema")
    return tuple(sorted(result, key=lambda record: record.candidate_id))


def _passed_hard_gates(
    records: tuple[CandidateScreeningRecord, ...],
) -> tuple[CandidateScreeningRecord, ...]:
    return tuple(
        record
        for record in records
        if record.feasibility.passed and record.reproducibility.passed
    )


def _provenance_payload(value: DatasetProvenance) -> list[str]:
    return [value.dataset_version, value.provenance_id, value.split.value]


def _record_payload(value: CandidateScreeningRecord) -> dict[str, Any]:
    return {
        "candidate_id": value.candidate_id,
        "provenance": _provenance_payload(value.provenance),
        "feasibility": [value.feasibility.passed, value.feasibility.reason],
        "reproducibility": [
            value.reproducibility.passed,
            value.reproducibility.repeated_runs,
            value.reproducibility.reason,
        ],
        "metrics": [
            [metric.name, metric.direction.value, metric.scale]
            for metric in value.fingerprint.metrics
        ],
        "values": list(value.fingerprint.values),
    }


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readiness_digest(value: DataReadinessEvidence) -> str:
    return _digest(
        {
            "kind": "data_readiness",
            "provenance": _provenance_payload(value.provenance),
            "passed": value.passed,
            "audit_id": value.audit_id,
            "reason": value.reason,
        }
    )


def _solver_audit_digest(value: SolverReproducibilityAudit) -> str:
    return _digest(
        {
            "kind": "solver_reproducibility_audit",
            "provenance": _provenance_payload(value.provenance),
            "passed": value.passed,
            "candidate_ids": value.candidate_ids,
            "repeated_runs": value.repeated_runs,
            "audit_id": value.audit_id,
            "reason": value.reason,
        }
    )


_STAGE_SEAL = object()


class _SealedStage:
    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("screening stage results can only be created by pipeline functions")


@dataclass(frozen=True, init=False)
class HardGateResult(_SealedStage):
    source_records: tuple[CandidateScreeningRecord, ...]
    records: tuple[CandidateScreeningRecord, ...]
    provenance: DatasetProvenance
    audit_id: str
    digest: str

    @classmethod
    def _create(cls, seal: object, source_records: tuple[CandidateScreeningRecord, ...],
                records: tuple[CandidateScreeningRecord, ...], provenance: DatasetProvenance,
                audit_id: str) -> HardGateResult:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, value in (("source_records", source_records), ("records", records),
                            ("provenance", provenance), ("audit_id", audit_id)):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _hard_digest(instance))
        return instance


@dataclass(frozen=True, init=False)
class ParetoResult(_SealedStage):
    parent: HardGateResult
    records: tuple[CandidateScreeningRecord, ...]
    provenance: DatasetProvenance
    audit_id: str
    digest: str

    @classmethod
    def _create(cls, seal: object, parent: HardGateResult,
                records: tuple[CandidateScreeningRecord, ...], audit_id: str) -> ParetoResult:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, value in (("parent", parent), ("records", records),
                            ("provenance", parent.provenance), ("audit_id", audit_id)):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _pareto_digest(instance))
        return instance


@dataclass(frozen=True, init=False)
class DistanceThresholdEvidence(_SealedStage):
    parent: object
    provenance: DatasetProvenance
    rule: DistanceThresholdRule
    value: float
    audit_id: str
    digest: str

    @classmethod
    def _create(
        cls,
        seal: object,
        parent: object,
        rule: DistanceThresholdRule,
        value: float,
        audit_id: str,
    ) -> DistanceThresholdEvidence:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, item in (
            ("parent", parent),
            ("provenance", parent.provenance),
            ("rule", rule),
            ("value", value),
            ("audit_id", audit_id),
        ):
            object.__setattr__(instance, name, item)
        object.__setattr__(instance, "digest", _threshold_digest(instance))
        return instance


@dataclass(frozen=True, init=False)
class NearDuplicateResult(_SealedStage):
    parent: ParetoResult
    records: tuple[CandidateScreeningRecord, ...]
    provenance: DatasetProvenance
    threshold_evidence: DistanceThresholdEvidence
    audit_id: str
    digest: str

    @classmethod
    def _create(cls, seal: object, parent: ParetoResult,
                records: tuple[CandidateScreeningRecord, ...],
                threshold_evidence: DistanceThresholdEvidence,
                audit_id: str) -> NearDuplicateResult:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, value in (("parent", parent), ("records", records),
                            ("provenance", parent.provenance),
                            ("threshold_evidence", threshold_evidence),
                            ("audit_id", audit_id)):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _near_digest(instance))
        return instance

    @property
    def threshold(self) -> float:
        return self.threshold_evidence.value


@dataclass(frozen=True, init=False)
class ClusteringResult(_SealedStage):
    parent: NearDuplicateResult
    records: tuple[CandidateScreeningRecord, ...]
    assignments: tuple[tuple[str, ...], ...]
    provenance: DatasetProvenance
    threshold_evidence: DistanceThresholdEvidence
    audit_id: str
    digest: str

    @classmethod
    def _create(cls, seal: object, parent: NearDuplicateResult,
                assignments: tuple[tuple[str, ...], ...],
                threshold_evidence: DistanceThresholdEvidence,
                audit_id: str) -> ClusteringResult:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, value in (("parent", parent), ("records", parent.records),
                            ("assignments", assignments), ("provenance", parent.provenance),
                            ("threshold_evidence", threshold_evidence),
                            ("audit_id", audit_id)):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _cluster_digest(instance))
        return instance

    @property
    def threshold(self) -> float:
        return self.threshold_evidence.value


@dataclass(frozen=True, init=False)
class MedoidSelectionResult(_SealedStage):
    parent: ClusteringResult
    records: tuple[CandidateScreeningRecord, ...]
    provenance: DatasetProvenance
    audit_id: str
    digest: str

    @classmethod
    def _create(cls, seal: object, parent: ClusteringResult,
                records: tuple[CandidateScreeningRecord, ...],
                audit_id: str) -> MedoidSelectionResult:
        if seal is not _STAGE_SEAL:
            raise TypeError("invalid stage-construction seal")
        instance = object.__new__(cls)
        for name, value in (("parent", parent), ("records", records),
                            ("provenance", parent.provenance), ("audit_id", audit_id)):
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _medoid_digest(instance))
        return instance


def _record_ids(records: tuple[CandidateScreeningRecord, ...]) -> list[str]:
    return [record.candidate_id for record in records]


def _hard_digest(value: HardGateResult) -> str:
    return _digest({"stage": "hard_gate",
                    "source": [_record_payload(record) for record in value.source_records],
                    "records": _record_ids(value.records),
                    "provenance": _provenance_payload(value.provenance),
                    "audit_id": value.audit_id})


def _pareto_digest(value: ParetoResult) -> str:
    return _digest({"stage": "pareto", "parent": value.parent.digest,
                    "records": _record_ids(value.records),
                    "provenance": _provenance_payload(value.provenance),
                    "audit_id": value.audit_id})


def _threshold_digest(value: DistanceThresholdEvidence) -> str:
    return _digest(
        {
            "kind": "distance_threshold",
            "parent": value.parent.digest,
            "provenance": _provenance_payload(value.provenance),
            "rule": value.rule.value,
            "value": value.value,
            "audit_id": value.audit_id,
        }
    )


def _near_digest(value: NearDuplicateResult) -> str:
    return _digest({"stage": "near_duplicate", "parent": value.parent.digest,
                    "records": _record_ids(value.records),
                    "provenance": _provenance_payload(value.provenance),
                    "threshold_evidence": value.threshold_evidence.digest,
                    "audit_id": value.audit_id})


def _cluster_digest(value: ClusteringResult) -> str:
    return _digest({"stage": "clustering", "parent": value.parent.digest,
                    "records": _record_ids(value.records), "assignments": value.assignments,
                    "provenance": _provenance_payload(value.provenance),
                    "threshold_evidence": value.threshold_evidence.digest,
                    "audit_id": value.audit_id})


def _medoid_digest(value: MedoidSelectionResult) -> str:
    return _digest({"stage": "medoid_selection", "parent": value.parent.digest,
                    "records": _record_ids(value.records),
                    "provenance": _provenance_payload(value.provenance),
                    "audit_id": value.audit_id})


def _same_record_objects(actual: tuple[CandidateScreeningRecord, ...],
                         expected: tuple[CandidateScreeningRecord, ...]) -> bool:
    return len(actual) == len(expected) and all(
        left is right for left, right in zip(actual, expected)
    )


def _pareto_records(records: tuple[CandidateScreeningRecord, ...],
                    ) -> tuple[CandidateScreeningRecord, ...]:
    vectors = {record.candidate_id: record.fingerprint.normalized_minimization_vector
               for record in records}
    kept: list[CandidateScreeningRecord] = []
    for candidate in records:
        target = vectors[candidate.candidate_id]
        dominated = any(
            all(left <= right for left, right in zip(vectors[other.candidate_id], target))
            and any(left < right for left, right in zip(vectors[other.candidate_id], target))
            for other in records if other.candidate_id != candidate.candidate_id
        )
        if not dominated:
            kept.append(candidate)
    return tuple(kept)


def _distance(left: CandidateScreeningRecord, right: CandidateScreeningRecord) -> float:
    try:
        result = math.dist(left.fingerprint.normalized_minimization_vector,
                           right.fingerprint.normalized_minimization_vector)
    except OverflowError:
        return math.inf
    return result if math.isfinite(result) else math.inf


def _derived_threshold(
    records: tuple[CandidateScreeningRecord, ...], rule: DistanceThresholdRule
) -> float:
    if rule is DistanceThresholdRule.ZERO:
        return 0.0
    distances = sorted(
        distance
        for left in range(len(records))
        for right in range(left + 1, len(records))
        if math.isfinite(distance := _distance(records[left], records[right]))
    )
    if rule is DistanceThresholdRule.MIN_POSITIVE_PAIRWISE:
        return next((distance for distance in distances if distance > 0.0), 0.0)
    if rule is DistanceThresholdRule.MEDIAN_PAIRWISE:
        if not distances:
            return 0.0
        middle = len(distances) // 2
        if len(distances) % 2:
            return distances[middle]
        return distances[middle - 1] / 2.0 + distances[middle] / 2.0
    raise TypeError("threshold rule must be an exact DistanceThresholdRule")


def _near_records(records: tuple[CandidateScreeningRecord, ...], threshold: float,
                  ) -> tuple[CandidateScreeningRecord, ...]:
    representatives: list[CandidateScreeningRecord] = []
    for candidate in records:
        if all(_distance(candidate, kept) > threshold for kept in representatives):
            representatives.append(candidate)
    return tuple(representatives)


def _cluster_assignments(records: tuple[CandidateScreeningRecord, ...], threshold: float,
                         ) -> tuple[tuple[str, ...], ...]:
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if _distance(records[left], records[right]) <= threshold:
                union(left, right)
    members: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        members.setdefault(find(index), []).append(record.candidate_id)
    groups = tuple(tuple(sorted(group)) for group in members.values())
    return tuple(sorted(groups, key=lambda group: group[0]))


def _medoid_records(value: ClusteringResult) -> tuple[CandidateScreeningRecord, ...]:
    by_id = {record.candidate_id: record for record in value.records}
    selected: list[CandidateScreeningRecord] = []
    for cluster in value.assignments:
        pair_distances = {
            (left, right): _distance(by_id[left], by_id[right])
            for left in cluster
            for right in cluster
        }
        finite_positive = tuple(
            distance
            for distance in pair_distances.values()
            if math.isfinite(distance) and distance > 0.0
        )
        common_scale = max(finite_positive, default=1.0)

        def score(candidate_id: str) -> float:
            distances = tuple(
                pair_distances[candidate_id, other] for other in cluster
            )
            if any(not math.isfinite(distance) for distance in distances):
                return math.inf
            return math.fsum(distance / common_scale for distance in distances)

        medoid_id = min(
            cluster,
            key=lambda candidate_id: (score(candidate_id), candidate_id),
        )
        selected.append(by_id[medoid_id])
    return tuple(selected)


def _lineage_failure(message: str) -> ScreeningLineageError:
    return ScreeningLineageError(f"invalid screening lineage: {message}")


def _validate_hard(value: HardGateResult) -> None:
    try:
        if type(value.source_records) is not tuple or type(value.records) is not tuple:
            raise ValueError("hard-gate records are not tuples")
        source = _validated_records(value.source_records)
        if not _same_record_objects(value.source_records, source):
            raise ValueError("hard-gate source ordering or identity changed")
        if not _same_record_objects(value.records, _passed_hard_gates(source)):
            raise ValueError("hard-gate output changed")
        if value.provenance != source[0].provenance:
            raise ValueError("hard-gate provenance changed")
        _nonempty_text(value.audit_id, "hard-gate audit_id")
        if type(value.digest) is not str or value.digest != _hard_digest(value):
            raise ValueError("hard-gate digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def _validate_pareto(value: ParetoResult) -> None:
    try:
        if type(value.parent) is not HardGateResult:
            raise ValueError("Pareto parent has wrong type")
        _validate_hard(value.parent)
        expected = _pareto_records(value.parent.records)
        if type(value.records) is not tuple or not _same_record_objects(value.records, expected):
            raise ValueError("Pareto output changed")
        if value.provenance != value.parent.provenance:
            raise ValueError("Pareto provenance changed")
        _nonempty_text(value.audit_id, "Pareto audit_id")
        if type(value.digest) is not str or value.digest != _pareto_digest(value):
            raise ValueError("Pareto digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def _validate_threshold_evidence(
    value: DistanceThresholdEvidence, expected_parent: object
) -> None:
    try:
        if type(value) is not DistanceThresholdEvidence:
            raise TypeError("threshold evidence has wrong type")
        if value.parent is not expected_parent:
            raise ValueError("threshold evidence belongs to a different parent")
        if type(expected_parent) is ParetoResult:
            _validate_pareto(expected_parent)
        elif type(expected_parent) is NearDuplicateResult:
            _validate_near(expected_parent)
        else:
            raise TypeError("threshold evidence parent has wrong type")
        if value.provenance != expected_parent.provenance:
            raise ValueError("threshold evidence provenance changed")
        if type(value.rule) is not DistanceThresholdRule:
            raise TypeError("threshold rule must remain exact DistanceThresholdRule")
        if type(value.value) is not float:
            raise TypeError("derived threshold must remain a canonical float")
        expected_value = _derived_threshold(expected_parent.records, value.rule)
        if value.value != expected_value:
            raise ValueError("derived threshold value changed")
        _nonempty_text(value.audit_id, "threshold audit_id")
        if type(value.digest) is not str or value.digest != _threshold_digest(value):
            raise ValueError("threshold evidence digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def _validate_near(value: NearDuplicateResult) -> None:
    try:
        if type(value.parent) is not ParetoResult:
            raise ValueError("near-duplicate parent has wrong type")
        _validate_pareto(value.parent)
        _validate_threshold_evidence(value.threshold_evidence, value.parent)
        threshold = value.threshold_evidence.value
        expected = _near_records(value.parent.records, threshold)
        if type(value.records) is not tuple or not _same_record_objects(value.records, expected):
            raise ValueError("near-duplicate output changed")
        if value.provenance != value.parent.provenance:
            raise ValueError("near-duplicate provenance changed")
        _nonempty_text(value.audit_id, "near-duplicate audit_id")
        if type(value.digest) is not str or value.digest != _near_digest(value):
            raise ValueError("near-duplicate digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def _validate_cluster(value: ClusteringResult) -> None:
    try:
        if type(value.parent) is not NearDuplicateResult:
            raise ValueError("clustering parent has wrong type")
        _validate_near(value.parent)
        _validate_threshold_evidence(value.threshold_evidence, value.parent)
        threshold = value.threshold_evidence.value
        if type(value.records) is not tuple or not _same_record_objects(value.records,
                                                                         value.parent.records):
            raise ValueError("clustering input records changed")
        expected = _cluster_assignments(value.records, threshold)
        if type(value.assignments) is not tuple or value.assignments != expected:
            raise ValueError("cluster assignments changed")
        if any(type(group) is not tuple for group in value.assignments):
            raise ValueError("cluster assignments are not canonical tuples")
        if value.provenance != value.parent.provenance:
            raise ValueError("clustering provenance changed")
        _nonempty_text(value.audit_id, "clustering audit_id")
        if type(value.digest) is not str or value.digest != _cluster_digest(value):
            raise ValueError("clustering digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def _validate_medoid(value: MedoidSelectionResult) -> None:
    try:
        if type(value.parent) is not ClusteringResult:
            raise ValueError("medoid parent has wrong type")
        _validate_cluster(value.parent)
        expected = _medoid_records(value.parent)
        if type(value.records) is not tuple or not _same_record_objects(value.records, expected):
            raise ValueError("medoid output changed")
        if value.provenance != value.parent.provenance:
            raise ValueError("medoid provenance changed")
        _nonempty_text(value.audit_id, "medoid audit_id")
        if type(value.digest) is not str or value.digest != _medoid_digest(value):
            raise ValueError("medoid digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError, HeldOutSelectionError, OverflowError) as exc:
        raise _lineage_failure(str(exc)) from exc


def apply_hard_gates(records: Iterable[CandidateScreeningRecord], *, audit_id: str,
                     ) -> HardGateResult:
    """Validate all Train evidence and seal the explicit hard-gate output."""
    source = _validated_records(records)
    audit = _nonempty_text(audit_id, "hard-gate audit_id")
    result = HardGateResult._create(
        _STAGE_SEAL, source, _passed_hard_gates(source), source[0].provenance, audit
    )
    _validate_hard(result)
    return result


def feasibility_gate(records: Iterable[CandidateScreeningRecord],
                     ) -> tuple[CandidateScreeningRecord, ...]:
    """Return Train records passing the diagnostic physical-feasibility gate."""
    checked = _validated_records(records)
    return tuple(record for record in checked if record.feasibility.passed)


def solver_reproducibility_gate(records: Iterable[CandidateScreeningRecord],
                                ) -> tuple[CandidateScreeningRecord, ...]:
    """Return Train records passing the diagnostic repeated-solve gate."""
    checked = _validated_records(records)
    return tuple(record for record in checked if record.reproducibility.passed)


def pareto_front(parent: HardGateResult, *, audit_id: str) -> ParetoResult:
    """Seal the deterministic non-dominated set from a valid hard-gate result."""
    if type(parent) is not HardGateResult:
        raise TypeError("pareto_front requires an exact HardGateResult")
    _validate_hard(parent)
    result = ParetoResult._create(
        _STAGE_SEAL, parent, _pareto_records(parent.records),
        _nonempty_text(audit_id, "Pareto audit_id")
    )
    _validate_pareto(result)
    return result


def derive_distance_threshold(
    parent: ParetoResult | NearDuplicateResult,
    *,
    rule: DistanceThresholdRule,
    audit_id: str,
) -> DistanceThresholdEvidence:
    """Derive a sealed numeric threshold solely from the parent's Train records."""
    if type(parent) is ParetoResult:
        _validate_pareto(parent)
    elif type(parent) is NearDuplicateResult:
        _validate_near(parent)
    else:
        raise TypeError(
            "derive_distance_threshold requires an exact ParetoResult or "
            "NearDuplicateResult"
        )
    if type(rule) is not DistanceThresholdRule:
        raise TypeError("rule must be an exact DistanceThresholdRule")
    audit = _nonempty_text(audit_id, "threshold audit_id")
    result = DistanceThresholdEvidence._create(
        _STAGE_SEAL, parent, rule, _derived_threshold(parent.records, rule), audit
    )
    _validate_threshold_evidence(result, parent)
    return result


def remove_near_duplicates(
    parent: ParetoResult,
    threshold_evidence: DistanceThresholdEvidence,
    *,
    audit_id: str,
) -> NearDuplicateResult:
    """Seal deterministic ID-first representatives within a normalized radius."""
    if type(parent) is not ParetoResult:
        raise TypeError("remove_near_duplicates requires an exact ParetoResult")
    if type(threshold_evidence) is not DistanceThresholdEvidence:
        raise TypeError(
            "remove_near_duplicates requires exact DistanceThresholdEvidence"
        )
    _validate_pareto(parent)
    _validate_threshold_evidence(threshold_evidence, parent)
    result = NearDuplicateResult._create(
        _STAGE_SEAL,
        parent,
        _near_records(parent.records, threshold_evidence.value),
        threshold_evidence,
        _nonempty_text(audit_id, "near-duplicate audit_id")
    )
    _validate_near(result)
    return result


def cluster_by_distance(
    parent: NearDuplicateResult,
    threshold_evidence: DistanceThresholdEvidence,
    *,
    audit_id: str,
) -> ClusteringResult:
    """Seal deterministic single-linkage components at a declared threshold."""
    if type(parent) is not NearDuplicateResult:
        raise TypeError("cluster_by_distance requires an exact NearDuplicateResult")
    if type(threshold_evidence) is not DistanceThresholdEvidence:
        raise TypeError("cluster_by_distance requires exact DistanceThresholdEvidence")
    _validate_near(parent)
    _validate_threshold_evidence(threshold_evidence, parent)
    result = ClusteringResult._create(
        _STAGE_SEAL,
        parent,
        _cluster_assignments(parent.records, threshold_evidence.value),
        threshold_evidence,
        _nonempty_text(audit_id, "clustering audit_id")
    )
    _validate_cluster(result)
    return result


def select_cluster_medoids(parent: ClusteringResult, *, audit_id: str,
                           ) -> MedoidSelectionResult:
    """Seal minimum-total-distance representatives; IDs break exact ties."""
    if type(parent) is not ClusteringResult:
        raise TypeError("select_cluster_medoids requires an exact ClusteringResult")
    _validate_cluster(parent)
    result = MedoidSelectionResult._create(
        _STAGE_SEAL, parent, _medoid_records(parent),
        _nonempty_text(audit_id, "medoid audit_id")
    )
    _validate_medoid(result)
    return result


def _validate_readiness(value: object) -> DataReadinessEvidence:
    if type(value) is not DataReadinessEvidence:
        raise TypeError("data_readiness must be exact DataReadinessEvidence")
    try:
        _validate_provenance(value.provenance)
        _exact_bool(value.passed, "data readiness passed")
        _nonempty_text(value.audit_id, "data audit_id")
        _nonempty_text(value.reason, "data audit reason")
        if type(value.digest) is not str or value.digest != _readiness_digest(value):
            raise ValueError("data readiness digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise _lineage_failure(str(exc)) from exc
    _require_train(value.provenance)
    return value


def _validate_solver_audit(value: object) -> SolverReproducibilityAudit:
    if type(value) is not SolverReproducibilityAudit:
        raise TypeError("solver_audit must be exact SolverReproducibilityAudit")
    try:
        _validate_provenance(value.provenance)
        _exact_bool(value.passed, "solver audit passed")
        if type(value.candidate_ids) is not tuple:
            raise TypeError("solver audit candidate_ids must remain a tuple")
        ids = tuple(_nonempty_text(item, "candidate_id") for item in value.candidate_ids)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("solver audit candidate_ids must remain unique")
        _repeated_runs(value.repeated_runs)
        _nonempty_text(value.audit_id, "solver audit_id")
        _nonempty_text(value.reason, "solver audit reason")
        if type(value.digest) is not str or value.digest != _solver_audit_digest(value):
            raise ValueError("solver audit digest mismatch")
    except ScreeningLineageError:
        raise
    except (TypeError, ValueError, AttributeError) as exc:
        raise _lineage_failure(str(exc)) from exc
    _require_train(value.provenance)
    return value


def _validated_canonical_action_bank(
    candidates: Iterable[ActionCandidate],
) -> tuple[ActionCandidate, ...]:
    try:
        supplied = tuple(candidates)
    except TypeError as exc:
        raise TypeError("candidates must be iterable") from exc
    if not supplied or any(type(candidate) is not ActionCandidate for candidate in supplied):
        raise TypeError("candidates must contain exact ActionCandidate values")

    for candidate in supplied:
        try:
            reconstructed = ActionCandidate(
                candidate.n_base, candidate.n_smooth, candidate.n_soc
            )
            expected_id = (
                f"w_{reconstructed.n_base}_{reconstructed.n_smooth}_"
                f"{reconstructed.n_soc}"
            )
            if candidate.action_id != expected_id or candidate != reconstructed:
                raise ValueError("candidate identity is not canonical")
            if vars(candidate) != vars(reconstructed):
                raise ValueError("candidate instance contains injected attributes")
            weights = ActionCandidate.to_mpc_weights(candidate)
            if (
                type(weights) is not MPCWeights
                or type(weights.q_base) is not float
                or type(weights.q_smooth) is not float
                or type(weights.q_soc) is not float
                or (weights.q_base, weights.q_smooth, weights.q_soc)
                != reconstructed.as_tuple()
            ):
                raise ValueError("candidate MPC weights are not canonical")
        except (TypeError, ValueError, AttributeError) as exc:
            raise CatalogFinalizationError(
                "candidate bank contains mutated or invalid action evidence"
            ) from exc

    canonical = generate_candidate_action_bank()
    if supplied != canonical:
        raise CatalogFinalizationError(
            "finalization requires the complete canonical 36-candidate bank"
        )
    return canonical


def finalize_action_catalog(
    candidates: Iterable[ActionCandidate], selection: MedoidSelectionResult, *,
    data_readiness: DataReadinessEvidence, solver_audit: SolverReproducibilityAudit,
) -> tuple[ActionCandidate, ...]:
    """Freeze only a complete, sealed, Train-derived screening result."""
    bank = _validated_canonical_action_bank(candidates)
    if type(selection) is not MedoidSelectionResult:
        raise TypeError("selection must be an exact MedoidSelectionResult")
    _validate_medoid(selection)

    root = selection.parent.parent.parent.parent
    source = root.source_records
    bank_ids = tuple(candidate.action_id for candidate in bank)
    source_ids = tuple(record.candidate_id for record in source)
    if len(source) != 36 or source_ids != tuple(sorted(bank_ids)):
        raise CatalogFinalizationError(
            "complete screening evidence is required for all 36 candidates"
        )
    readiness = _validate_readiness(data_readiness)
    audit = _validate_solver_audit(solver_audit)
    provenance = selection.provenance
    if readiness.provenance != provenance or audit.provenance != provenance:
        raise CatalogFinalizationError("all finalization evidence must share provenance")
    if not readiness.passed:
        raise CatalogFinalizationError("usable Train data audit has not passed")
    if (not audit.passed or audit.repeated_runs < 2 or len(audit.candidate_ids) != 36
            or set(audit.candidate_ids) != set(bank_ids)):
        raise CatalogFinalizationError(
            "a passed complete repeated-solve audit covering all 36 candidates is required"
        )
    selected_ids = tuple(record.candidate_id for record in selection.records)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise CatalogFinalizationError("sealed selection must be nonempty and unique")
    source_by_id = {record.candidate_id: record for record in source}
    if any(candidate_id not in source_by_id for candidate_id in selected_ids):
        raise CatalogFinalizationError("selected candidates must come from the canonical bank")
    if any(not source_by_id[candidate_id].feasibility.passed
           or not source_by_id[candidate_id].reproducibility.passed
           for candidate_id in selected_ids):
        raise CatalogFinalizationError(
            "every selected candidate must pass feasibility and reproducibility"
        )
    selected_set = set(selected_ids)
    return tuple(candidate for candidate in bank if candidate.action_id in selected_set)


__all__ = [
    "BehaviorFingerprint", "CandidateScreeningRecord", "CatalogFinalizationError",
    "ClusteringResult", "DataReadinessEvidence", "DataSplit", "DatasetProvenance",
    "DistanceThresholdEvidence", "DistanceThresholdRule",
    "FeasibilityResult", "HardGateResult", "HeldOutSelectionError",
    "MedoidSelectionResult", "MetricDefinition", "MetricDirection",
    "NearDuplicateResult", "ParetoResult", "ScreeningLineageError",
    "SolverReproducibilityAudit", "SolverReproducibilityResult", "apply_hard_gates",
    "cluster_by_distance", "derive_distance_threshold", "feasibility_gate",
    "finalize_action_catalog", "pareto_front",
    "remove_near_duplicates", "select_cluster_medoids", "solver_reproducibility_gate",
]
