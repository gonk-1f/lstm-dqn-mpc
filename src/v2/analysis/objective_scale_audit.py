"""Train-only MPC objective magnitude, contribution, and response audit."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real

from ..control.nonlinear_mpc import (
    MPCPlan,
    ObjectiveComponents,
    SolverDiagnostics,
    weighted_objective,
)
from ..dqn.action_space import (
    CANDIDATE_ACTION_BANK,
    ActionCandidate,
)
from .action_screening import DatasetProvenance
from .timescale_audit import SelectionParameter, require_train_selection


FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS = "GO"
OBJECTIVE_TERMS = ("base", "smooth", "soc")
_RESULT_SEAL = object()


def _nonempty_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _finite_tuple(value: object, name: str, *, nonempty: bool = True) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if nonempty and not value:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


class ObjectiveScaleStatus(Enum):
    GO = "GO"
    WARNING = "WARNING"
    NO_GO = "NO-GO"


def classify_objective_scale_ratio(value: object) -> ObjectiveScaleStatus:
    ratio = _finite(value, "scale_ratio", nonnegative=True)
    if ratio < 1.0:
        raise ValueError("scale_ratio cannot be less than one")
    if ratio <= 5.0:
        return ObjectiveScaleStatus.GO
    if ratio < 10.0:
        return ObjectiveScaleStatus.WARNING
    return ObjectiveScaleStatus.NO_GO


@dataclass(frozen=True)
class ObjectiveAuditCase:
    case_id: str
    order: int
    payload: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty_text(self.case_id, "case_id"))
        object.__setattr__(self, "order", _exact_int(self.order, "order"))
        object.__setattr__(self, "payload", _finite_tuple(self.payload, "case payload"))


@dataclass(frozen=True)
class BehaviorTolerance:
    objective_abs: float
    power_kw_abs: float
    soc_abs: float

    def __post_init__(self) -> None:
        for name in ("objective_abs", "power_kw_abs", "soc_abs"):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name, nonnegative=True),
            )


@dataclass(frozen=True)
class ObjectiveAuditObservation:
    case_id: str
    case_order: int
    action_id: str
    weights: tuple[float, float, float]
    components: tuple[float, float, float]
    p_fc_first_kw: float
    p_batt_first_kw: float
    predicted_soc_path: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty_text(self.case_id, "case_id"))
        object.__setattr__(self, "case_order", _exact_int(self.case_order, "case_order"))
        object.__setattr__(self, "action_id", _nonempty_text(self.action_id, "action_id"))
        weights = _finite_tuple(self.weights, "weights")
        if len(weights) != 3 or any(value <= 0.0 for value in weights):
            raise ValueError("weights must contain three positive values")
        if not math.isclose(math.fsum(weights), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("weights must sum to one")
        components = _finite_tuple(self.components, "components")
        if len(components) != 3 or any(value < 0.0 for value in components):
            raise ValueError("components must contain three nonnegative values")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "components", components)
        object.__setattr__(
            self, "p_fc_first_kw", _finite(self.p_fc_first_kw, "p_fc_first_kw")
        )
        object.__setattr__(
            self, "p_batt_first_kw", _finite(self.p_batt_first_kw, "p_batt_first_kw")
        )
        object.__setattr__(
            self,
            "predicted_soc_path",
            _finite_tuple(self.predicted_soc_path, "predicted_soc_path"),
        )


@dataclass(frozen=True)
class TermStatistics:
    term: str
    count: int
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float

    def __post_init__(self) -> None:
        if type(self.term) is not str:
            raise TypeError("term must be an exact string")
        if self.term not in OBJECTIVE_TERMS:
            raise ValueError("term must be base, smooth, or soc")
        object.__setattr__(self, "count", _exact_int(self.count, "count", minimum=1))
        for name in ("mean", "std", "p50", "p90", "p95", "p99", "max"):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, nonnegative=True)
            )


@dataclass(frozen=True)
class SocActiveStatistics:
    positive_count: int
    probability_positive: float
    p50_positive: float | None
    p90_positive: float | None
    p95_positive: float | None
    p99_positive: float | None

    def __post_init__(self) -> None:
        count = _exact_int(self.positive_count, "positive_count")
        probability = _finite(
            self.probability_positive,
            "probability_positive",
            nonnegative=True,
        )
        if probability > 1.0:
            raise ValueError("probability_positive cannot exceed one")
        percentiles = (
            self.p50_positive,
            self.p90_positive,
            self.p95_positive,
            self.p99_positive,
        )
        if count == 0:
            if probability != 0.0 or any(value is not None for value in percentiles):
                raise ValueError("zero positive SOC count requires zero probability and no percentiles")
        else:
            if probability <= 0.0 or any(value is None for value in percentiles):
                raise ValueError("positive SOC count requires probability and percentiles")
            checked = tuple(
                _finite(value, "positive SOC percentile", nonnegative=True)
                for value in percentiles
            )
            if tuple(sorted(checked)) != checked:
                raise ValueError("positive SOC percentiles must be nondecreasing")
            for name, checked_value in zip(
                ("p50_positive", "p90_positive", "p95_positive", "p99_positive"),
                checked,
            ):
                object.__setattr__(self, name, checked_value)
        object.__setattr__(self, "positive_count", count)
        object.__setattr__(self, "probability_positive", probability)


@dataclass(frozen=True)
class ContributionDominance:
    dominant_term: str
    dominated_term: str
    count: int
    rate: float
    case_action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.dominant_term) is not str or type(self.dominated_term) is not str:
            raise TypeError("dominance terms must be exact strings")
        if (
            self.dominant_term not in OBJECTIVE_TERMS
            or self.dominated_term not in OBJECTIVE_TERMS
            or self.dominant_term == self.dominated_term
        ):
            raise ValueError("dominance requires two distinct objective terms")
        count = _exact_int(self.count, "dominance count")
        rate = _finite(self.rate, "dominance rate", nonnegative=True)
        if rate > 1.0:
            raise ValueError("dominance rate cannot exceed one")
        if type(self.case_action_ids) is not tuple:
            raise TypeError("case_action_ids must be an exact tuple")
        ids = tuple(_nonempty_text(value, "case/action ID") for value in self.case_action_ids)
        if len(ids) != count or len(set(ids)) != len(ids):
            raise ValueError("dominance IDs must be unique and match count")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "case_action_ids", ids)


@dataclass(frozen=True)
class BehavioralRedundancy:
    case_id: str
    action_pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty_text(self.case_id, "case_id"))
        if type(self.action_pairs) is not tuple or not self.action_pairs:
            raise TypeError("action_pairs must be a nonempty exact tuple")
        pairs: list[tuple[str, str]] = []
        for value in self.action_pairs:
            if type(value) is not tuple or len(value) != 2:
                raise TypeError("each redundant action pair must be an exact two-item tuple")
            pair = (
                _nonempty_text(value[0], "left action_id"),
                _nonempty_text(value[1], "right action_id"),
            )
            if pair[0] == pair[1]:
                raise ValueError("a redundant action pair must contain distinct actions")
            pairs.append(pair)
        if len(set(pairs)) != len(pairs):
            raise ValueError("redundant action pairs must be unique")
        object.__setattr__(self, "action_pairs", tuple(pairs))


def _snapshot_case(value: object) -> ObjectiveAuditCase:
    if type(value) is not ObjectiveAuditCase:
        raise TypeError("cases must contain exact ObjectiveAuditCase values")
    snapshot = ObjectiveAuditCase(value.case_id, value.order, value.payload)
    if vars(value) != vars(snapshot):
        raise ValueError("objective audit case contains injected or mutated fields")
    return snapshot


def _snapshot_action(value: object) -> ActionCandidate:
    if type(value) is not ActionCandidate:
        raise TypeError("actions must contain exact ActionCandidate values")
    snapshot = ActionCandidate(value.n_base, value.n_smooth, value.n_soc)
    if vars(value) != vars(snapshot) or value.action_id != snapshot.action_id:
        raise ValueError("objective audit action is not canonical")
    if snapshot not in CANDIDATE_ACTION_BANK:
        raise ValueError("action must belong to the canonical candidate bank")
    return snapshot


def _snapshot_plan(value: object) -> MPCPlan:
    if type(value) is not MPCPlan:
        raise TypeError("solver_runner must return an exact MPCPlan")
    if set(vars(value)) != {
        "p_fc_kw",
        "p_batt_bus_kw",
        "soc_path",
        "load_forecast_kw",
        "base_reference_kw",
        "components",
        "objective_value",
        "diagnostics",
    }:
        raise ValueError("solver plan contains injected or missing fields")
    if type(value.components) is not ObjectiveComponents or set(vars(value.components)) != {
        "j_base", "j_smooth", "j_soc"
    }:
        raise ValueError("solver objective components are not canonical")
    if type(value.diagnostics) is not SolverDiagnostics or set(vars(value.diagnostics)) != {
        "success", "status", "message", "iterations", "objective_value"
    }:
        raise ValueError("solver diagnostics are not canonical")
    components = ObjectiveComponents(
        value.components.j_base,
        value.components.j_smooth,
        value.components.j_soc,
    )
    diagnostics = SolverDiagnostics(
        value.diagnostics.success,
        value.diagnostics.status,
        value.diagnostics.message,
        value.diagnostics.iterations,
        value.diagnostics.objective_value,
    )
    return MPCPlan(
        value.p_fc_kw,
        value.p_batt_bus_kw,
        value.soc_path,
        value.load_forecast_kw,
        value.base_reference_kw,
        components,
        value.objective_value,
        diagnostics,
    )


def _snapshot_observation(value: object) -> ObjectiveAuditObservation:
    if type(value) is not ObjectiveAuditObservation:
        raise TypeError("observations must remain exact ObjectiveAuditObservation values")
    snapshot = ObjectiveAuditObservation(
        value.case_id,
        value.case_order,
        value.action_id,
        value.weights,
        value.components,
        value.p_fc_first_kw,
        value.p_batt_first_kw,
        value.predicted_soc_path,
    )
    if vars(value) != vars(snapshot):
        raise ValueError("objective observation contains injected or mutated fields")
    return snapshot


def _percentile(sorted_values: tuple[float, ...], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _statistics(term: str, values: tuple[float, ...]) -> TermStatistics:
    ordered = tuple(sorted(values))
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return TermStatistics(
        term,
        len(values),
        mean,
        math.sqrt(variance),
        _percentile(ordered, 0.50),
        _percentile(ordered, 0.90),
        _percentile(ordered, 0.95),
        _percentile(ordered, 0.99),
        ordered[-1],
    )


def _soc_active(values: tuple[float, ...]) -> SocActiveStatistics:
    positive = tuple(sorted(value for value in values if value > 0.0))
    if not positive:
        return SocActiveStatistics(0, 0.0, None, None, None, None)
    return SocActiveStatistics(
        len(positive),
        len(positive) / len(values),
        _percentile(positive, 0.50),
        _percentile(positive, 0.90),
        _percentile(positive, 0.95),
        _percentile(positive, 0.99),
    )


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(left - right) <= tolerance


def _same_behavior(
    left: ObjectiveAuditObservation,
    right: ObjectiveAuditObservation,
    tolerance: BehaviorTolerance,
) -> bool:
    return (
        all(
            _close(a, b, tolerance.objective_abs)
            for a, b in zip(left.components, right.components)
        )
        and _close(left.p_fc_first_kw, right.p_fc_first_kw, tolerance.power_kw_abs)
        and _close(left.p_batt_first_kw, right.p_batt_first_kw, tolerance.power_kw_abs)
        and len(left.predicted_soc_path) == len(right.predicted_soc_path)
        and all(
            _close(a, b, tolerance.soc_abs)
            for a, b in zip(left.predicted_soc_path, right.predicted_soc_path)
        )
    )


def _derive_summary(
    observations: tuple[ObjectiveAuditObservation, ...],
    tolerance: BehaviorTolerance,
) -> dict[str, object]:
    component_values = tuple(
        tuple(observation.components[index] for observation in observations)
        for index in range(3)
    )
    objective_statistics = tuple(
        _statistics(term, values)
        for term, values in zip(OBJECTIVE_TERMS, component_values)
    )
    soc_active = _soc_active(component_values[2])
    active_p95 = (
        objective_statistics[0].p95,
        objective_statistics[1].p95,
        soc_active.p95_positive,
    )
    usable = tuple(value for value in active_p95 if value is not None and value > 0.0)
    if len(usable) != 3:
        scale_ratio = None
        status = ObjectiveScaleStatus.NO_GO
        recommended = None
    else:
        scale_ratio = max(usable) / min(usable)
        status = classify_objective_scale_ratio(scale_ratio)
        recommended = active_p95 if status is ObjectiveScaleStatus.NO_GO else None

    contribution_values = tuple(
        tuple(
            observation.weights[index] * observation.components[index]
            for observation in observations
        )
        for index in range(3)
    )
    contributions = tuple(
        _statistics(term, values)
        for term, values in zip(OBJECTIVE_TERMS, contribution_values)
    )

    dominance: list[ContributionDominance] = []
    for dominant_index, dominant_term in enumerate(OBJECTIVE_TERMS):
        for dominated_index, dominated_term in enumerate(OBJECTIVE_TERMS):
            if dominant_index == dominated_index:
                continue
            matched = tuple(
                f"{observation.case_id}/{observation.action_id}"
                for observation in observations
                if 0.1 * observation.components[dominant_index]
                > 0.7 * observation.components[dominated_index]
            )
            dominance.append(
                ContributionDominance(
                    dominant_term,
                    dominated_term,
                    len(matched),
                    len(matched) / len(observations),
                    matched,
                )
            )

    redundancies: list[BehavioralRedundancy] = []
    case_ids = tuple(dict.fromkeys(item.case_id for item in observations))
    for case_id in case_ids:
        case_observations = tuple(item for item in observations if item.case_id == case_id)
        pairs = tuple(
            (case_observations[left].action_id, case_observations[right].action_id)
            for left in range(len(case_observations))
            for right in range(left + 1, len(case_observations))
            if _same_behavior(case_observations[left], case_observations[right], tolerance)
        )
        if pairs:
            redundancies.append(BehavioralRedundancy(case_id, pairs))

    return {
        "objective_statistics": objective_statistics,
        "soc_active": soc_active,
        "active_p95": active_p95,
        "scale_ratio": scale_ratio,
        "status": status,
        "weighted_contribution_statistics": contributions,
        "dominance_statistics": tuple(dominance),
        "behavioral_redundancy": tuple(redundancies),
        "recommended_fixed_normalization_constants": recommended,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _result_digest(value: "ObjectiveScaleAuditResult") -> str:
    payload = {
        name: _jsonable(getattr(value, name))
        for name in vars(value)
        if name != "digest"
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, init=False)
class ObjectiveScaleAuditResult:
    provenance: DatasetProvenance
    observations: tuple[ObjectiveAuditObservation, ...]
    behavior_tolerance: BehaviorTolerance
    objective_statistics: tuple[TermStatistics, ...]
    soc_active: SocActiveStatistics
    active_p95: tuple[float, float, float | None]
    scale_ratio: float | None
    status: ObjectiveScaleStatus
    weighted_contribution_statistics: tuple[TermStatistics, ...]
    dominance_statistics: tuple[ContributionDominance, ...]
    behavioral_redundancy: tuple[BehavioralRedundancy, ...]
    recommended_fixed_normalization_constants: tuple[float, float, float] | None
    digest: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("objective scale results can only be created by the audit")

    @classmethod
    def _create(
        cls,
        seal: object,
        *,
        provenance: DatasetProvenance,
        observations: tuple[ObjectiveAuditObservation, ...],
        behavior_tolerance: BehaviorTolerance,
    ) -> "ObjectiveScaleAuditResult":
        if seal is not _RESULT_SEAL:
            raise TypeError("invalid objective scale result construction seal")
        summary = _derive_summary(observations, behavior_tolerance)
        instance = object.__new__(cls)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "observations", observations)
        object.__setattr__(instance, "behavior_tolerance", behavior_tolerance)
        for name, value in summary.items():
            object.__setattr__(instance, name, value)
        object.__setattr__(instance, "digest", _result_digest(instance))
        return instance

    def validate(self) -> "ObjectiveScaleAuditResult":
        return _validate_result(self)


def _validate_result(value: object) -> ObjectiveScaleAuditResult:
    if type(value) is not ObjectiveScaleAuditResult:
        raise TypeError("result must be an exact ObjectiveScaleAuditResult")
    expected_fields = set(ObjectiveScaleAuditResult.__dataclass_fields__)
    if set(vars(value)) != expected_fields:
        raise ValueError("objective scale result contains injected or missing fields")
    provenance = require_train_selection(
        value.provenance, SelectionParameter.OBJECTIVE_NORMALIZATION
    )
    if vars(value.provenance) != vars(provenance):
        raise ValueError("objective scale provenance is not canonical")
    if type(value.observations) is not tuple or not value.observations:
        raise TypeError("observations must remain a nonempty tuple")
    observations = tuple(_snapshot_observation(item) for item in value.observations)
    if type(value.behavior_tolerance) is not BehaviorTolerance:
        raise TypeError("behavior_tolerance must remain exact BehaviorTolerance")
    tolerance = BehaviorTolerance(
        value.behavior_tolerance.objective_abs,
        value.behavior_tolerance.power_kw_abs,
        value.behavior_tolerance.soc_abs,
    )
    if vars(value.behavior_tolerance) != vars(tolerance):
        raise ValueError("behavior tolerance contains injected fields")
    for collection_name, collection, item_type in (
        ("objective_statistics", value.objective_statistics, TermStatistics),
        (
            "weighted_contribution_statistics",
            value.weighted_contribution_statistics,
            TermStatistics,
        ),
        ("dominance_statistics", value.dominance_statistics, ContributionDominance),
        ("behavioral_redundancy", value.behavioral_redundancy, BehavioralRedundancy),
    ):
        if type(collection) is not tuple:
            raise TypeError(f"{collection_name} must remain an exact tuple")
        for item in collection:
            if type(item) is not item_type:
                raise TypeError(f"{collection_name} contains a noncanonical item")
            if set(vars(item)) != set(item_type.__dataclass_fields__):
                raise ValueError(f"{collection_name} contains injected or missing fields")
    if type(value.soc_active) is not SocActiveStatistics or set(
        vars(value.soc_active)
    ) != set(SocActiveStatistics.__dataclass_fields__):
        raise ValueError("soc_active contains injected, missing, or invalid fields")
    expected = _derive_summary(observations, tolerance)
    for name, result in expected.items():
        if getattr(value, name) != result:
            raise ValueError(f"objective scale derived field mismatch: {name}")
    if type(value.digest) is not str or value.digest != _result_digest(value):
        raise ValueError("objective scale audit digest mismatch")
    return value


def run_objective_scale_audit(
    *,
    provenance: object,
    cases_loader: Callable[[], Iterable[ObjectiveAuditCase]],
    actions: tuple[ActionCandidate, ...],
    solver_runner: Callable[[ObjectiveAuditCase, ActionCandidate], MPCPlan],
    behavior_tolerance: BehaviorTolerance,
) -> ObjectiveScaleAuditResult:
    """Solve every canonical action for each Train case and audit raw terms."""

    canonical_provenance = require_train_selection(
        provenance, SelectionParameter.OBJECTIVE_NORMALIZATION
    )
    if not callable(cases_loader):
        raise TypeError("cases_loader must be callable")
    if not callable(solver_runner):
        raise TypeError("solver_runner must be callable")
    if type(actions) is not tuple or not actions:
        raise TypeError("actions must be a nonempty exact tuple")
    canonical_actions = tuple(_snapshot_action(action) for action in actions)
    if canonical_actions != CANDIDATE_ACTION_BANK:
        raise ValueError(
            "objective scale audit requires the complete canonical 36-action bank"
        )
    if type(behavior_tolerance) is not BehaviorTolerance:
        raise TypeError("behavior_tolerance must be exact BehaviorTolerance")
    tolerance = BehaviorTolerance(
        behavior_tolerance.objective_abs,
        behavior_tolerance.power_kw_abs,
        behavior_tolerance.soc_abs,
    )
    if vars(behavior_tolerance) != vars(tolerance):
        raise ValueError("behavior tolerance contains injected fields")

    raw_cases = cases_loader()
    if isinstance(raw_cases, (str, bytes)):
        raise TypeError("cases_loader must return an iterable of audit cases")
    try:
        cases = tuple(_snapshot_case(case) for case in raw_cases)
    except TypeError as exc:
        raise TypeError("cases_loader must return iterable exact audit cases") from exc
    if not cases:
        raise ValueError("objective scale audit requires at least one Train case")
    if tuple(case.order for case in cases) != tuple(range(len(cases))):
        raise ValueError("audit case order must be contiguous and canonical")
    if len(set(case.case_id for case in cases)) != len(cases):
        raise ValueError("audit case IDs must be unique")

    observations: list[ObjectiveAuditObservation] = []
    for case in cases:
        for action in canonical_actions:
            runner_case = ObjectiveAuditCase(case.case_id, case.order, case.payload)
            runner_action = ActionCandidate(action.n_base, action.n_smooth, action.n_soc)
            plan = _snapshot_plan(solver_runner(runner_case, runner_action))
            if len(plan.p_fc_kw) != 5:
                raise ValueError("objective scale audit requires the provisional N=5 MPC plan")
            weights = action.to_mpc_weights()
            expected_objective = weighted_objective(plan.components, weights)
            if not math.isclose(
                plan.objective_value, expected_objective, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("solver plan objective does not match the requested action")
            observations.append(
                ObjectiveAuditObservation(
                    case.case_id,
                    case.order,
                    action.action_id,
                    action.as_tuple(),
                    (
                        plan.components.j_base,
                        plan.components.j_smooth,
                        plan.components.j_soc,
                    ),
                    plan.p_fc_kw[0],
                    plan.p_batt_bus_kw[0],
                    plan.soc_path,
                )
            )

    result = ObjectiveScaleAuditResult._create(
        _RESULT_SEAL,
        provenance=DatasetProvenance(
            canonical_provenance.dataset_version,
            canonical_provenance.provenance_id,
            canonical_provenance.split,
        ),
        observations=tuple(observations),
        behavior_tolerance=tolerance,
    )
    return _validate_result(result)


__all__ = [
    "FORMAL_OBJECTIVE_SCALE_AUDIT_STATUS",
    "BehaviorTolerance",
    "BehavioralRedundancy",
    "ContributionDominance",
    "ObjectiveAuditCase",
    "ObjectiveAuditObservation",
    "ObjectiveScaleAuditResult",
    "ObjectiveScaleStatus",
    "SocActiveStatistics",
    "TermStatistics",
    "classify_objective_scale_ratio",
    "run_objective_scale_audit",
]
