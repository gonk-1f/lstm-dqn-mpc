"""Frozen project policy for learnable terminal physical infeasibility."""

from __future__ import annotations

from dataclasses import dataclass
import math


FORMAL_TERMINAL_FAILURE_PENALTY_SCORE = 50_000.0
FORMAL_FAILURE_KIND = "physical_mpc_infeasibility"
FAILURE_EVIDENCE_STATUS = "DERIVED_TRAIN_ONLY / PROJECT_DESIGN"
FAILURE_CALIBRATION_ID = "v2_train_w_8_1_1_v1"


@dataclass(frozen=True)
class FormalFailurePolicy:
    """Exact identity-bound failure score; never part of the CNY ledger."""

    penalty_score: float
    failure_kind: str
    evidence_status: str
    calibration_id: str

    def __post_init__(self) -> None:
        if type(self.penalty_score) is not float:
            raise TypeError("penalty_score must be an exact float")
        if not math.isfinite(self.penalty_score) or self.penalty_score <= 0.0:
            raise ValueError("penalty_score must be finite and positive")
        if self.penalty_score != FORMAL_TERMINAL_FAILURE_PENALTY_SCORE:
            raise ValueError("penalty_score differs from the frozen formal value")
        expected = (
            FORMAL_FAILURE_KIND,
            FAILURE_EVIDENCE_STATUS,
            FAILURE_CALIBRATION_ID,
        )
        actual = (self.failure_kind, self.evidence_status, self.calibration_id)
        if any(type(value) is not str for value in actual):
            raise TypeError("failure policy text fields must be exact strings")
        if actual != expected:
            raise ValueError("failure policy identity differs from the frozen baseline")

    @classmethod
    def formal_baseline(cls) -> "FormalFailurePolicy":
        return cls(
            FORMAL_TERMINAL_FAILURE_PENALTY_SCORE,
            FORMAL_FAILURE_KIND,
            FAILURE_EVIDENCE_STATUS,
            FAILURE_CALIBRATION_ID,
        )


FORMAL_FAILURE_POLICY = FormalFailurePolicy.formal_baseline()


__all__ = [
    "FAILURE_CALIBRATION_ID",
    "FAILURE_EVIDENCE_STATUS",
    "FORMAL_FAILURE_KIND",
    "FORMAL_FAILURE_POLICY",
    "FORMAL_TERMINAL_FAILURE_PENALTY_SCORE",
    "FormalFailurePolicy",
]
