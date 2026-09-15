"""Canonical candidate weights for the v2 three-term MPC objective."""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import ACTION_TABLE_VERSION
from ..control.nonlinear_mpc import MPCWeights


class ActionCatalogUnavailableError(RuntimeError):
    """Raised when code requests a formal catalog before screening is complete."""


@dataclass(frozen=True)
class ActionCandidate:
    """One positive tenth-grid point, represented canonically by integers."""

    n_base: int
    n_smooth: int
    n_soc: int

    def __post_init__(self) -> None:
        numerators = (self.n_base, self.n_smooth, self.n_soc)
        if any(type(value) is not int for value in numerators):
            raise TypeError("action numerators must be exact integers")
        if any(value < 1 for value in numerators) or sum(numerators) != 10:
            raise ValueError("action numerators must be positive and sum to ten")

    @property
    def numerators(self) -> tuple[int, int, int]:
        return (self.n_base, self.n_smooth, self.n_soc)

    @property
    def action_id(self) -> str:
        return f"w_{self.n_base}_{self.n_smooth}_{self.n_soc}"

    @property
    def q_base(self) -> float:
        return self.n_base / 10.0

    @property
    def q_smooth(self) -> float:
        return self.n_smooth / 10.0

    @property
    def q_soc(self) -> float:
        return self.n_soc / 10.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.q_base, self.q_smooth, self.q_soc)

    def to_mpc_weights(self) -> MPCWeights:
        return MPCWeights(*self.as_tuple())


def generate_candidate_action_bank() -> tuple[ActionCandidate, ...]:
    """Return all 36 positive integer compositions of ten, lexicographically."""

    return tuple(
        ActionCandidate(n_base, n_smooth, 10 - n_base - n_smooth)
        for n_base in range(1, 9)
        for n_smooth in range(1, 10 - n_base)
    )


CANDIDATE_ACTION_BANK = generate_candidate_action_bank()
CANDIDATE_ACTIONS = CANDIDATE_ACTION_BANK

# This is deliberately not the candidate bank.  It remains unset until a
# complete Train-only screen and solver audit pass the explicit finalization
# boundary in v2.analysis.action_screening.
ACTION_CATALOG_STATUS = "NO-GO"
FINAL_DQN_ACTION_CATALOG: None = None
FINAL_DQN_ACTIONS: None = None


def get_final_dqn_action_catalog() -> tuple[ActionCandidate, ...]:
    raise ActionCatalogUnavailableError(
        "NO-GO: usable Train operating-cycle data and a passed solver "
        "reproducibility audit are absent; the candidate bank is not a final "
        "DQN action catalog"
    )


__all__ = [
    "ACTION_CATALOG_STATUS",
    "ACTION_TABLE_VERSION",
    "CANDIDATE_ACTIONS",
    "CANDIDATE_ACTION_BANK",
    "FINAL_DQN_ACTIONS",
    "FINAL_DQN_ACTION_CATALOG",
    "ActionCandidate",
    "ActionCatalogUnavailableError",
    "generate_candidate_action_bank",
    "get_final_dqn_action_catalog",
]
