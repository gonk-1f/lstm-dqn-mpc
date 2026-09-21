"""Paired Train-only warm/cold solver reliability diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real
from statistics import median

from .action_screening import DatasetProvenance
from .timescale_audit import (
    PROVISIONAL_N_MPC,
    SelectionParameter,
    require_train_selection,
)


FORMAL_SOLVER_DECISION_STATUS = "NO-GO"


def _nonempty_text(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    if not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


def _exact_int(value: object, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_tuple(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))


class SolverStartMode(Enum):
    COLD = "cold"
    WARM = "warm"


@dataclass(frozen=True)
class SolverAuditCase:
    case_id: str
    seed: int
    order: int
    payload: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty_text(self.case_id, "case_id"))
        object.__setattr__(self, "seed", _exact_int(self.seed, "seed", minimum=0))
        object.__setattr__(self, "order", _exact_int(self.order, "order", minimum=0))
        object.__setattr__(self, "payload", _finite_tuple(self.payload, "case payload"))


@dataclass(frozen=True)
class SolverRunObservation:
    case_id: str
    seed: int
    order: int
    mode: SolverStartMode
    success: bool
    solve_seconds: float
    iterations: int
    status: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty_text(self.case_id, "case_id"))
        object.__setattr__(self, "seed", _exact_int(self.seed, "seed", minimum=0))
        object.__setattr__(self, "order", _exact_int(self.order, "order", minimum=0))
        if type(self.mode) is not SolverStartMode:
            raise TypeError("mode must be an exact SolverStartMode")
        if type(self.success) is not bool:
            raise TypeError("success must be an exact bool")
        object.__setattr__(
            self,
            "solve_seconds",
            _finite(self.solve_seconds, "solve_seconds", positive=True),
        )
        object.__setattr__(
            self,
            "iterations",
            _exact_int(self.iterations, "iterations", minimum=0),
        )
        object.__setattr__(self, "status", _exact_int(self.status, "status"))


@dataclass(frozen=True)
class PairedSolverCaseResult:
    case: SolverAuditCase
    cold: SolverRunObservation
    warm: SolverRunObservation

    def __post_init__(self) -> None:
        if type(self.case) is not SolverAuditCase:
            raise TypeError("case must be an exact SolverAuditCase")
        if type(self.cold) is not SolverRunObservation or type(self.warm) is not SolverRunObservation:
            raise TypeError("paired observations must be exact SolverRunObservation values")
        identity = (self.case.case_id, self.case.seed, self.case.order)
        if (self.cold.case_id, self.cold.seed, self.cold.order) != identity:
            raise ValueError("cold observation does not match case/seed/order")
        if (self.warm.case_id, self.warm.seed, self.warm.order) != identity:
            raise ValueError("warm observation does not match case/seed/order")
        if self.cold.mode is not SolverStartMode.COLD:
            raise ValueError("cold observation has the wrong start mode")
        if self.warm.mode is not SolverStartMode.WARM:
            raise ValueError("warm observation has the wrong start mode")


_AUDIT_SEAL = object()


@dataclass(frozen=True, init=False)
class PairedSolverAuditResult:
    provenance: DatasetProvenance
    n_mpc: int
    pairs: tuple[PairedSolverCaseResult, ...]
    cold_success_count: int
    warm_success_count: int
    both_success_count: int
    cold_success_rate: float
    warm_success_rate: float
    speed_comparison_case_ids: tuple[str, ...]
    median_cold_over_warm_solve_time: float | None
    formal_decision_status: str
    digest: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("paired solver audit results can only be created by the audit function")

    @classmethod
    def _create(
        cls,
        seal: object,
        *,
        provenance: DatasetProvenance,
        pairs: tuple[PairedSolverCaseResult, ...],
    ) -> "PairedSolverAuditResult":
        if seal is not _AUDIT_SEAL:
            raise TypeError("invalid paired solver audit construction seal")
        cold_success = sum(pair.cold.success for pair in pairs)
        warm_success = sum(pair.warm.success for pair in pairs)
        comparable = tuple(pair for pair in pairs if pair.cold.success and pair.warm.success)
        speed_ids = tuple(pair.case.case_id for pair in comparable)
        ratios = tuple(pair.cold.solve_seconds / pair.warm.solve_seconds for pair in comparable)
        instance = object.__new__(cls)
        object.__setattr__(instance, "provenance", provenance)
        object.__setattr__(instance, "n_mpc", PROVISIONAL_N_MPC)
        object.__setattr__(instance, "pairs", pairs)
        object.__setattr__(instance, "cold_success_count", cold_success)
        object.__setattr__(instance, "warm_success_count", warm_success)
        object.__setattr__(instance, "both_success_count", len(comparable))
        object.__setattr__(instance, "cold_success_rate", cold_success / len(pairs))
        object.__setattr__(instance, "warm_success_rate", warm_success / len(pairs))
        object.__setattr__(instance, "speed_comparison_case_ids", speed_ids)
        object.__setattr__(
            instance,
            "median_cold_over_warm_solve_time",
            float(median(ratios)) if ratios else None,
        )
        object.__setattr__(instance, "formal_decision_status", FORMAL_SOLVER_DECISION_STATUS)
        object.__setattr__(instance, "digest", _result_digest(instance))
        return instance

    def validate(self) -> "PairedSolverAuditResult":
        return _validate_result(self)


def _case_payload(value: SolverAuditCase) -> dict[str, object]:
    return {
        "case_id": value.case_id,
        "seed": value.seed,
        "order": value.order,
        "payload": value.payload,
    }


def _observation_payload(value: SolverRunObservation) -> dict[str, object]:
    return {
        "case_id": value.case_id,
        "seed": value.seed,
        "order": value.order,
        "mode": value.mode.value,
        "success": value.success,
        "solve_seconds": value.solve_seconds,
        "iterations": value.iterations,
        "status": value.status,
    }


def _pair_payload(value: PairedSolverCaseResult) -> dict[str, object]:
    return {
        "case": _case_payload(value.case),
        "cold": _observation_payload(value.cold),
        "warm": _observation_payload(value.warm),
    }


def _result_digest(value: PairedSolverAuditResult) -> str:
    encoded = json.dumps(
        {
            "kind": "paired_warm_cold_solver_audit_v1",
            "provenance": [
                value.provenance.dataset_version,
                value.provenance.provenance_id,
                value.provenance.split.value,
            ],
            "n_mpc": value.n_mpc,
            "pairs": tuple(_pair_payload(pair) for pair in value.pairs),
            "cold_success_count": value.cold_success_count,
            "warm_success_count": value.warm_success_count,
            "both_success_count": value.both_success_count,
            "cold_success_rate": value.cold_success_rate,
            "warm_success_rate": value.warm_success_rate,
            "speed_comparison_case_ids": value.speed_comparison_case_ids,
            "median_cold_over_warm_solve_time": value.median_cold_over_warm_solve_time,
            "formal_decision_status": value.formal_decision_status,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_case(value: object) -> SolverAuditCase:
    if type(value) is not SolverAuditCase:
        raise TypeError("cases must contain exact SolverAuditCase values")
    snapshot = SolverAuditCase(value.case_id, value.seed, value.order, value.payload)
    if vars(value) != vars(snapshot):
        raise ValueError("solver case contains mutated or injected fields")
    return snapshot


def _snapshot_observation(value: object) -> SolverRunObservation:
    if type(value) is not SolverRunObservation:
        raise TypeError("solver_runner must return exact SolverRunObservation values")
    snapshot = SolverRunObservation(
        value.case_id,
        value.seed,
        value.order,
        value.mode,
        value.success,
        value.solve_seconds,
        value.iterations,
        value.status,
    )
    if vars(value) != vars(snapshot):
        raise ValueError("solver observation contains mutated or injected fields")
    return snapshot


def _snapshot_pair(value: object) -> PairedSolverCaseResult:
    if type(value) is not PairedSolverCaseResult:
        raise TypeError("pairs must remain exact PairedSolverCaseResult values")
    snapshot = PairedSolverCaseResult(
        _snapshot_case(value.case),
        _snapshot_observation(value.cold),
        _snapshot_observation(value.warm),
    )
    if vars(value) != vars(snapshot):
        raise ValueError("paired solver result contains mutated or injected fields")
    return snapshot


def _validate_result(value: object) -> PairedSolverAuditResult:
    if type(value) is not PairedSolverAuditResult:
        raise TypeError("result must be an exact PairedSolverAuditResult")
    expected_fields = {
        "provenance",
        "n_mpc",
        "pairs",
        "cold_success_count",
        "warm_success_count",
        "both_success_count",
        "cold_success_rate",
        "warm_success_rate",
        "speed_comparison_case_ids",
        "median_cold_over_warm_solve_time",
        "formal_decision_status",
        "digest",
    }
    if set(vars(value)) != expected_fields:
        raise ValueError("paired solver audit result contains injected or missing fields")
    provenance = require_train_selection(value.provenance, SelectionParameter.DQN_SWITCH_STEPS)
    if vars(value.provenance) != vars(provenance):
        raise ValueError("result provenance is not canonical")
    if type(value.n_mpc) is not int or value.n_mpc != PROVISIONAL_N_MPC:
        raise ValueError("paired solver audit N must remain fixed at five")
    if type(value.pairs) is not tuple or not value.pairs:
        raise TypeError("pairs must remain a nonempty tuple")
    pairs = tuple(_snapshot_pair(pair) for pair in value.pairs)
    count = len(pairs)
    expected_cold = sum(pair.cold.success for pair in pairs)
    expected_warm = sum(pair.warm.success for pair in pairs)
    comparable = tuple(pair for pair in pairs if pair.cold.success and pair.warm.success)
    expected_ids = tuple(pair.case.case_id for pair in comparable)
    expected_median = (
        float(median(pair.cold.solve_seconds / pair.warm.solve_seconds for pair in comparable))
        if comparable
        else None
    )
    exact_counts = (
        type(value.cold_success_count) is int
        and type(value.warm_success_count) is int
        and type(value.both_success_count) is int
        and value.cold_success_count == expected_cold
        and value.warm_success_count == expected_warm
        and value.both_success_count == len(comparable)
    )
    if not exact_counts:
        raise ValueError("solver success counts do not match paired observations")
    if (
        type(value.cold_success_rate) is not float
        or type(value.warm_success_rate) is not float
        or value.cold_success_rate != expected_cold / count
        or value.warm_success_rate != expected_warm / count
    ):
        raise ValueError("solver success rates do not match paired observations")
    if type(value.speed_comparison_case_ids) is not tuple or value.speed_comparison_case_ids != expected_ids:
        raise ValueError("speed comparison must include only both-success cases")
    if value.median_cold_over_warm_solve_time != expected_median:
        raise ValueError("median speed ratio does not match both-success pairs")
    if (
        type(value.formal_decision_status) is not str
        or value.formal_decision_status != FORMAL_SOLVER_DECISION_STATUS
    ):
        raise ValueError("formal solver decision must remain NO-GO")
    if type(value.digest) is not str or value.digest != _result_digest(value):
        raise ValueError("paired solver audit digest mismatch")
    return value


def run_paired_solver_audit(
    *,
    provenance: object,
    cases_loader: Callable[[], Iterable[SolverAuditCase]],
    solver_runner: Callable[[SolverAuditCase, SolverStartMode], SolverRunObservation],
    n_mpc: object = PROVISIONAL_N_MPC,
) -> PairedSolverAuditResult:
    """Run cold then warm for each identical case in one declared order.

    The Train provenance check intentionally precedes both lazy case loading and
    any solver invocation.
    """

    canonical_provenance = require_train_selection(
        provenance,
        SelectionParameter.DQN_SWITCH_STEPS,
    )
    if type(n_mpc) is not int:
        raise TypeError("n_mpc must be an exact integer")
    if n_mpc != PROVISIONAL_N_MPC:
        raise ValueError("the provisional solver audit keeps n_mpc fixed at five")
    if not callable(cases_loader):
        raise TypeError("cases_loader must be callable")
    if not callable(solver_runner):
        raise TypeError("solver_runner must be callable")
    raw_cases = cases_loader()
    if isinstance(raw_cases, (str, bytes)):
        raise TypeError("cases_loader must return an iterable of cases")
    try:
        cases = tuple(_snapshot_case(case) for case in raw_cases)
    except TypeError as exc:
        raise TypeError("cases_loader must return an iterable of exact cases") from exc
    if not cases:
        raise ValueError("paired solver audit requires at least one case")
    if tuple(case.order for case in cases) != tuple(range(len(cases))):
        raise ValueError("case order must be unique, contiguous, and already canonical")
    ids = tuple(case.case_id for case in cases)
    if len(set(ids)) != len(ids):
        raise ValueError("solver audit case IDs must be unique")

    pairs: list[PairedSolverCaseResult] = []
    for case in cases:
        observations: list[SolverRunObservation] = []
        for mode in (SolverStartMode.COLD, SolverStartMode.WARM):
            runner_case = SolverAuditCase(case.case_id, case.seed, case.order, case.payload)
            observation = _snapshot_observation(solver_runner(runner_case, mode))
            expected_identity = (case.case_id, case.seed, case.order, mode)
            actual_identity = (
                observation.case_id,
                observation.seed,
                observation.order,
                observation.mode,
            )
            if actual_identity != expected_identity:
                raise ValueError("solver observation does not match requested case/seed/order/mode")
            observations.append(observation)
        pairs.append(PairedSolverCaseResult(case, observations[0], observations[1]))

    result = PairedSolverAuditResult._create(
        _AUDIT_SEAL,
        provenance=DatasetProvenance(
            canonical_provenance.dataset_version,
            canonical_provenance.provenance_id,
            canonical_provenance.split,
        ),
        pairs=tuple(pairs),
    )
    return _validate_result(result)


__all__ = [
    "FORMAL_SOLVER_DECISION_STATUS",
    "PairedSolverAuditResult",
    "PairedSolverCaseResult",
    "SolverAuditCase",
    "SolverRunObservation",
    "SolverStartMode",
    "run_paired_solver_audit",
]
