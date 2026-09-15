"""Train-only deterministic interfaces for offline action screening.

The algorithms in this module operate only on evidence supplied by a caller.
They do not assert that the repository's current data gate has passed or that a
formal action catalog exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from collections.abc import Iterable, Sequence

from ..dqn.action_space import ActionCandidate, CANDIDATE_ACTION_BANK


class HeldOutSelectionError(PermissionError):
    """Raised before held-out or unknown data can affect method selection."""


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


def _require_train(provenance: object) -> DatasetProvenance:
    if type(provenance) is not DatasetProvenance:
        raise TypeError("an exact DatasetProvenance is required")
    if provenance.split is not DataSplit.TRAIN:
        raise HeldOutSelectionError(
            "candidate removal, thresholds, clustering, medoids, K, and final "
            "catalog selection are Train-only"
        )
    return provenance


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


@dataclass(frozen=True)
class BehaviorFingerprint:
    candidate_id: str
    metrics: tuple[MetricDefinition, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _nonempty_text(self.candidate_id, "candidate_id")
        )
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
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError("fingerprint values must all be finite")
            values.append(normalized)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "values", tuple(values))

    @property
    def normalized_minimization_vector(self) -> tuple[float, ...]:
        return tuple(
            (value / metric.scale)
            if metric.direction is MetricDirection.MINIMIZE
            else -(value / metric.scale)
            for metric, value in zip(self.metrics, self.values)
        )


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
        nested_types = (
            (self.feasibility, FeasibilityResult, "feasibility"),
            (self.reproducibility, SolverReproducibilityResult, "reproducibility"),
            (self.fingerprint, BehaviorFingerprint, "fingerprint"),
        )
        for value, expected, name in nested_types:
            if type(value) is not expected:
                raise TypeError(f"{name} must be an exact {expected.__name__}")
            if value.candidate_id != candidate_id:
                raise ValueError("all screening evidence must identify the same candidate")
        if self.feasibility.provenance != self.reproducibility.provenance:
            raise ValueError("feasibility and solver evidence must share provenance")
        object.__setattr__(self, "candidate_id", candidate_id)

    @property
    def provenance(self) -> DatasetProvenance:
        return self.feasibility.provenance


@dataclass(frozen=True)
class DataReadinessEvidence:
    provenance: DatasetProvenance
    passed: bool
    audit_id: str

    def __post_init__(self) -> None:
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("data readiness requires exact DatasetProvenance")
        object.__setattr__(self, "passed", _exact_bool(self.passed, "data readiness passed"))
        object.__setattr__(self, "audit_id", _nonempty_text(self.audit_id, "data audit_id"))


@dataclass(frozen=True)
class SolverReproducibilityAudit:
    provenance: DatasetProvenance
    passed: bool
    candidate_ids: tuple[str, ...]
    repeated_runs: int
    audit_id: str

    def __post_init__(self) -> None:
        if type(self.provenance) is not DatasetProvenance:
            raise TypeError("solver audit requires exact DatasetProvenance")
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
    if any(type(record) is not CandidateScreeningRecord for record in result):
        raise TypeError("records must contain exact CandidateScreeningRecord values")

    # Validate every split before applying any filtering.  Thus a failed or
    # otherwise ignorable held-out row cannot influence the selection path.
    provenances = tuple(_require_train(record.provenance) for record in result)
    if any(provenance != provenances[0] for provenance in provenances[1:]):
        raise ValueError("all screening records must share the same exact provenance")
    ids = tuple(record.candidate_id for record in result)
    if len(set(ids)) != len(ids):
        raise ValueError("screening candidate IDs must be unique")
    schema = result[0].fingerprint.metrics
    if any(record.fingerprint.metrics != schema for record in result[1:]):
        raise ValueError("all behavior fingerprints must share one exact metric schema")
    return tuple(sorted(result, key=lambda record: record.candidate_id))


def _matching_train_provenance(
    value: object,
    expected: DatasetProvenance,
    name: str,
) -> DatasetProvenance:
    provenance = _require_train(value)
    if provenance != expected:
        raise ValueError(f"{name} must match the screening-record provenance")
    return provenance


def _passed_hard_gates(
    checked: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    return tuple(
        record
        for record in checked
        if record.feasibility.passed and record.reproducibility.passed
    )


def apply_hard_gates(
    records: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    """Retain only candidates passing both explicit non-objective gates."""

    checked = _validated_records(records)
    return _passed_hard_gates(checked)


def feasibility_gate(
    records: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    """Retain records whose explicit physical-feasibility result passed."""

    checked = _validated_records(records)
    return tuple(record for record in checked if record.feasibility.passed)


def solver_reproducibility_gate(
    records: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    """Retain records whose explicit repeated-solve result passed."""

    checked = _validated_records(records)
    return tuple(record for record in checked if record.reproducibility.passed)


def pareto_front(
    records: Iterable[CandidateScreeningRecord],
) -> tuple[CandidateScreeningRecord, ...]:
    """Return the deterministic non-dominated set after both hard gates."""

    gated = apply_hard_gates(records)
    vectors = {
        record.candidate_id: record.fingerprint.normalized_minimization_vector
        for record in gated
    }
    kept: list[CandidateScreeningRecord] = []
    for candidate in gated:
        target = vectors[candidate.candidate_id]
        dominated = any(
            all(left <= right for left, right in zip(vectors[other.candidate_id], target))
            and any(left < right for left, right in zip(vectors[other.candidate_id], target))
            for other in gated
            if other.candidate_id != candidate.candidate_id
        )
        if not dominated:
            kept.append(candidate)
    return tuple(kept)


def _distance(left: CandidateScreeningRecord, right: CandidateScreeningRecord) -> float:
    lhs = left.fingerprint.normalized_minimization_vector
    rhs = right.fingerprint.normalized_minimization_vector
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(lhs, rhs)))


def remove_near_duplicates(
    records: Iterable[CandidateScreeningRecord],
    distance_threshold: float,
    *,
    threshold_provenance: DatasetProvenance,
) -> tuple[CandidateScreeningRecord, ...]:
    """Greedily keep the lexicographically first point within each radius."""

    threshold = _nonnegative_finite(distance_threshold, "distance_threshold")
    checked = _validated_records(records)
    _matching_train_provenance(
        threshold_provenance, checked[0].provenance, "threshold_provenance"
    )
    gated = _passed_hard_gates(checked)
    representatives: list[CandidateScreeningRecord] = []
    for candidate in gated:
        if all(_distance(candidate, kept) > threshold for kept in representatives):
            representatives.append(candidate)
    return tuple(representatives)


def cluster_by_distance(
    records: Iterable[CandidateScreeningRecord],
    distance_threshold: float,
    *,
    threshold_provenance: DatasetProvenance,
) -> tuple[tuple[str, ...], ...]:
    """Build deterministic single-linkage components at an explicit threshold."""

    threshold = _nonnegative_finite(distance_threshold, "distance_threshold")
    checked = _validated_records(records)
    _matching_train_provenance(
        threshold_provenance, checked[0].provenance, "threshold_provenance"
    )
    gated = _passed_hard_gates(checked)
    parents = list(range(len(gated)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for left in range(len(gated)):
        for right in range(left + 1, len(gated)):
            if _distance(gated[left], gated[right]) <= threshold:
                union(left, right)

    members: dict[int, list[str]] = {}
    for index, record in enumerate(gated):
        members.setdefault(find(index), []).append(record.candidate_id)
    clusters = tuple(tuple(sorted(group)) for group in members.values())
    return tuple(sorted(clusters, key=lambda group: group[0]))


def select_cluster_medoids(
    records: Iterable[CandidateScreeningRecord],
    clusters: Sequence[Sequence[str]],
    *,
    cluster_provenance: DatasetProvenance,
) -> tuple[CandidateScreeningRecord, ...]:
    """Select minimum-total-distance representatives; IDs break exact ties."""

    checked = _validated_records(records)
    _matching_train_provenance(
        cluster_provenance, checked[0].provenance, "cluster_provenance"
    )
    gated = _passed_hard_gates(checked)
    by_id = {record.candidate_id: record for record in gated}
    if isinstance(clusters, (str, bytes)):
        raise TypeError("clusters must be a sequence of candidate-ID sequences")
    normalized_clusters: list[tuple[str, ...]] = []
    for cluster in clusters:
        if isinstance(cluster, (str, bytes)):
            raise TypeError("each cluster must be a candidate-ID sequence")
        group = tuple(_nonempty_text(value, "cluster candidate_id") for value in cluster)
        if not group:
            raise ValueError("clusters cannot be empty")
        normalized_clusters.append(tuple(sorted(group)))
    flattened = tuple(value for cluster in normalized_clusters for value in cluster)
    if len(set(flattened)) != len(flattened) or set(flattened) != set(by_id):
        raise ValueError("clusters must partition exactly the gated candidate records")

    selected: list[CandidateScreeningRecord] = []
    for cluster in sorted(normalized_clusters, key=lambda group: group[0]):
        medoid_id = min(
            cluster,
            key=lambda candidate_id: (
                sum(_distance(by_id[candidate_id], by_id[other]) for other in cluster),
                candidate_id,
            ),
        )
        selected.append(by_id[medoid_id])
    return tuple(selected)


def finalize_action_catalog(
    candidates: Iterable[ActionCandidate],
    records: Iterable[CandidateScreeningRecord],
    *,
    selected_candidate_ids: Sequence[str],
    selection_provenance: DatasetProvenance,
    data_readiness: DataReadinessEvidence,
    solver_audit: SolverReproducibilityAudit,
) -> tuple[ActionCandidate, ...]:
    """Freeze a supplied selection only after complete, matching Train evidence.

    This generic boundary is usable by a future audited pipeline.  It does not
    change the repository's current NO-GO status or publish a module-level
    catalog.
    """

    try:
        bank = tuple(candidates)
    except TypeError as exc:
        raise TypeError("candidates must be iterable") from exc
    if not bank or any(type(candidate) is not ActionCandidate for candidate in bank):
        raise TypeError("candidates must contain exact ActionCandidate values")
    if bank != CANDIDATE_ACTION_BANK:
        raise CatalogFinalizationError(
            "finalization requires the complete canonical 36-candidate bank"
        )
    bank_ids = tuple(candidate.action_id for candidate in bank)
    if len(set(bank_ids)) != len(bank_ids):
        raise CatalogFinalizationError("candidate bank IDs must be unique")

    checked = _validated_records(records)
    evidence_ids = tuple(record.candidate_id for record in checked)
    if set(evidence_ids) != set(bank_ids) or len(evidence_ids) != len(bank_ids):
        raise CatalogFinalizationError(
            "complete screening evidence is required for every candidate"
        )
    provenance = checked[0].provenance

    _matching_train_provenance(
        selection_provenance, provenance, "selection_provenance"
    )

    if type(data_readiness) is not DataReadinessEvidence:
        raise TypeError("data_readiness must be exact DataReadinessEvidence")
    if type(solver_audit) is not SolverReproducibilityAudit:
        raise TypeError("solver_audit must be exact SolverReproducibilityAudit")
    _require_train(data_readiness.provenance)
    _require_train(solver_audit.provenance)
    if data_readiness.provenance != provenance or solver_audit.provenance != provenance:
        raise CatalogFinalizationError("all finalization evidence must share provenance")
    if not data_readiness.passed:
        raise CatalogFinalizationError("usable Train data audit has not passed")
    if not solver_audit.passed or set(solver_audit.candidate_ids) != set(bank_ids):
        raise CatalogFinalizationError(
            "a passed complete solver reproducibility audit is required"
        )

    if isinstance(selected_candidate_ids, (str, bytes)):
        raise TypeError("selected_candidate_ids must be a sequence")
    selected = tuple(
        _nonempty_text(candidate_id, "selected candidate_id")
        for candidate_id in selected_candidate_ids
    )
    if not selected or len(set(selected)) != len(selected):
        raise CatalogFinalizationError("selected candidate IDs must be nonempty and unique")
    if not set(selected).issubset(bank_ids):
        raise CatalogFinalizationError("selected candidate IDs must come from the bank")
    selected_set = set(selected)
    records_by_id = {record.candidate_id: record for record in checked}
    if any(
        not records_by_id[candidate_id].feasibility.passed
        or not records_by_id[candidate_id].reproducibility.passed
        for candidate_id in selected
    ):
        raise CatalogFinalizationError(
            "every selected candidate must pass feasibility and reproducibility"
        )
    return tuple(candidate for candidate in bank if candidate.action_id in selected_set)


__all__ = [
    "BehaviorFingerprint",
    "CandidateScreeningRecord",
    "CatalogFinalizationError",
    "DataReadinessEvidence",
    "DataSplit",
    "DatasetProvenance",
    "FeasibilityResult",
    "HeldOutSelectionError",
    "MetricDefinition",
    "MetricDirection",
    "SolverReproducibilityAudit",
    "SolverReproducibilityResult",
    "apply_hard_gates",
    "cluster_by_distance",
    "feasibility_gate",
    "finalize_action_catalog",
    "pareto_front",
    "remove_near_duplicates",
    "select_cluster_medoids",
    "solver_reproducibility_gate",
]
