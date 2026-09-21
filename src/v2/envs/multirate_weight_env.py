"""One-DQN-action-per-macro-interval execution for v2.

The formal action catalog and economic calibration are not closed yet.  This
module therefore exposes no production factory: callers must explicitly opt
into the sealed synthetic-test path until those gates become GO.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol

from ..config import TimeScaleConfig
from ..control.nonlinear_mpc import MPCWeights
from ..dqn.action_space import ActionCandidate
from ..economics import RawCnyIntervalLedger


TRAINING_READINESS_STATUS = "NO-GO"


def _validate_timescale(value: object) -> TimeScaleConfig:
    if type(value) is not TimeScaleConfig:
        raise TypeError("timescale must be an exact TimeScaleConfig")
    if type(value.ts_mpc_seconds) is not float:
        raise TypeError("stored ts_mpc_seconds must remain an exact float")
    if not math.isfinite(value.ts_mpc_seconds) or value.ts_mpc_seconds <= 0.0:
        raise ValueError("stored ts_mpc_seconds must remain finite and positive")
    if type(value.n_mpc) is not int or type(value.dqn_switch_steps) is not int:
        raise TypeError("stored N and M must remain exact integers")
    if value.n_mpc <= 0 or value.dqn_switch_steps <= 0:
        raise ValueError("stored N and M must remain positive")
    try:
        durations = (value.prediction_seconds, value.switch_seconds)
    except OverflowError as exc:
        raise ValueError("derived prediction and switch durations must remain finite") from exc
    if any(type(item) is not float or not math.isfinite(item) or item <= 0.0 for item in durations):
        raise ValueError("derived prediction and switch durations must remain finite and positive")
    return TimeScaleConfig(
        value.ts_mpc_seconds,
        value.n_mpc,
        value.dqn_switch_steps,
    )


def _validate_state(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable exact tuple")
    if not value:
        raise ValueError(f"{name} must not be empty")
    for item in value:
        if type(item) is not float:
            raise TypeError(f"{name} must contain exact floats")
        if not math.isfinite(item):
            raise ValueError(f"{name} must contain only finite floats")
    return value


def _validate_candidate(value: object) -> ActionCandidate:
    if type(value) is not ActionCandidate:
        raise TypeError("action catalog must contain exact ActionCandidate values")
    numerators = (value.n_base, value.n_smooth, value.n_soc)
    if any(type(item) is not int for item in numerators):
        raise TypeError("stored action numerators must remain exact integers")
    if any(item < 1 for item in numerators) or sum(numerators) != 10:
        raise ValueError("stored action numerators must remain positive and sum to ten")
    canonical = ActionCandidate(*numerators)
    if vars(value) != vars(canonical):
        raise ValueError("stored action contains injected or altered attributes")
    if value.action_id != canonical.action_id:
        raise ValueError("stored action identity is not canonical")
    weights = ActionCandidate.to_mpc_weights(value)
    if type(weights) is not MPCWeights:
        # MPCWeights is an alias of the exact frozen ObjectiveWeights class.
        raise TypeError("candidate did not produce exact MPCWeights")
    if (weights.q_base, weights.q_smooth, weights.q_soc) != canonical.as_tuple():
        raise ValueError("candidate weights are not canonical")
    return canonical


def _validate_ledger(value: object) -> RawCnyIntervalLedger:
    if type(value) is not RawCnyIntervalLedger:
        raise TypeError("ledger must be an exact RawCnyIntervalLedger")
    # Accessing components revalidates values in case frozen storage was forged.
    value.components_cny
    return value


class MPCStepBackend(Protocol):
    """Backend boundary: one call means one solve and first-command execution."""

    def execute_mpc_step(self, weights: MPCWeights) -> "MPCExecutionResult": ...


@dataclass(frozen=True)
class MPCExecutionResult:
    """Economic result of exactly one physically executed MPC command."""

    ledger: RawCnyIntervalLedger
    done: bool

    def __post_init__(self) -> None:
        _validate_ledger(self.ledger)
        if type(self.done) is not bool:
            raise TypeError("done must be an exact bool")


@dataclass(frozen=True)
class MacroTransition:
    """The sole replay item emitted at a DQN macro boundary."""

    state: tuple[float, ...]
    action: ActionCandidate
    reward_cny: float
    next_state: tuple[float, ...]
    done: bool
    executed_mpc_steps: int
    ledger: RawCnyIntervalLedger

    def __post_init__(self) -> None:
        _validate_state(self.state, "state")
        _validate_candidate(self.action)
        _validate_state(self.next_state, "next_state")
        _validate_ledger(self.ledger)
        if type(self.reward_cny) is not float or not math.isfinite(self.reward_cny):
            raise TypeError("reward_cny must be an exact finite float")
        if self.reward_cny != self.ledger.reward_cny:
            raise ValueError("reward_cny must equal the negative raw-CNY ledger total")
        if type(self.done) is not bool:
            raise TypeError("done must be an exact bool")
        if type(self.executed_mpc_steps) is not int:
            raise TypeError("executed_mpc_steps must be an exact integer")
        if self.executed_mpc_steps <= 0:
            raise ValueError("executed_mpc_steps must be positive")

    @property
    def action_id(self) -> str:
        return self.action.action_id


class MacroStepExecutionError(RuntimeError):
    """A fail-closed macro step error after zero or more physical executions.

    Already executed plant commands are not rolled back.  The count is exposed
    so a caller can reconcile external plant state before explicitly resetting.
    """

    def __init__(self, message: str, *, executed_mpc_steps: int) -> None:
        if type(executed_mpc_steps) is not int or executed_mpc_steps < 0:
            raise ValueError("executed_mpc_steps must be an exact non-negative integer")
        super().__init__(
            f"{message}; {executed_mpc_steps} MPC step(s) were already executed; "
            "no replay transition was emitted"
        )
        self.executed_mpc_steps = executed_mpc_steps


class MultiRateWeightEnvironment:
    """Hold one immutable weight action for ``dqn_switch_steps`` executions."""

    def __init__(
        self,
        *,
        timescale: TimeScaleConfig,
        action_catalog: tuple[ActionCandidate, ...],
        backend: MPCStepBackend,
        state_provider: Callable[[], tuple[float, ...]],
        synthetic_test_mode: bool = False,
        replay_sink: Callable[[MacroTransition], None] | None = None,
    ) -> None:
        if type(synthetic_test_mode) is not bool:
            raise TypeError("synthetic_test_mode must be an exact bool")
        if not synthetic_test_mode:
            raise PermissionError(
                "formal v2 training is NO-GO; synthetic_test_mode=True is required "
                "for an explicitly injected test backend"
            )
        checked_timescale = _validate_timescale(timescale)
        if type(action_catalog) is not tuple:
            raise TypeError("action_catalog must be an immutable exact tuple")
        if not action_catalog:
            raise ValueError("action_catalog must not be empty")
        checked_catalog = tuple(_validate_candidate(item) for item in action_catalog)
        action_ids = tuple(item.action_id for item in checked_catalog)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action_catalog contains duplicate action IDs")
        execute = getattr(backend, "execute_mpc_step", None)
        if not callable(execute):
            raise TypeError("backend must expose callable execute_mpc_step")
        if not callable(state_provider):
            raise TypeError("state_provider must be callable")
        if replay_sink is not None and not callable(replay_sink):
            raise TypeError("replay_sink must be callable or None")

        self._timescale = checked_timescale
        self._actions = dict(zip(action_ids, checked_catalog))
        self._execute_mpc_step = execute
        self._state_provider = state_provider
        self._replay_sink = replay_sink
        self._current_state: tuple[float, ...] | None = None
        self._transitions: list[MacroTransition] = []
        self._done = False
        self._failed = False

    @property
    def transitions(self) -> tuple[MacroTransition, ...]:
        return tuple(self._transitions)

    @property
    def current_state(self) -> tuple[float, ...] | None:
        return self._current_state

    def reset(self) -> tuple[float, ...]:
        state = _validate_state(self._state_provider(), "reset state")
        self._current_state = state
        self._transitions.clear()
        self._done = False
        self._failed = False
        return state

    def step(self, action_id: str) -> MacroTransition:
        if self._current_state is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("episode is done; reset before another step")
        if self._failed:
            raise RuntimeError("environment failed after plant execution; reset required")
        if type(action_id) is not str:
            raise TypeError("action_id must be an exact str")
        try:
            action = self._actions[action_id]
        except KeyError as exc:
            raise ValueError(f"unknown action_id: {action_id}") from exc
        _validate_candidate(action)
        canonical_weight_values = action.as_tuple()

        ledgers: list[RawCnyIntervalLedger] = []
        done = False
        executed = 0
        try:
            # N is intentionally absent here: it belongs inside each backend
            # solve.  M alone determines actual receding-horizon executions.
            for _ in range(self._timescale.dqn_switch_steps):
                # Backends are untrusted mutable boundaries.  A fresh value
                # prevents one call from poisoning a later MPC execution.
                weights = MPCWeights(*canonical_weight_values)
                result = self._execute_mpc_step(weights)
                if type(result) is not MPCExecutionResult:
                    raise TypeError("backend result must be an exact MPCExecutionResult")
                _validate_ledger(result.ledger)
                if type(result.done) is not bool:
                    raise TypeError("stored backend done flag must remain an exact bool")
                # Snapshot before the next backend call can mutate an object it
                # previously returned.
                ledger = RawCnyIntervalLedger(*result.ledger.components_cny)
                done_snapshot = result.done
                ledgers.append(ledger)
                executed += 1
                done = done_snapshot
                if done:
                    break

            components = tuple(
                math.fsum(ledger.components_cny[index] for ledger in ledgers)
                for index in range(4)
            )
            if not all(math.isfinite(value) for value in components):
                raise ValueError("macro raw-CNY component accumulation must remain finite")
            macro_ledger = RawCnyIntervalLedger(*components)
            next_state = _validate_state(self._state_provider(), "boundary next_state")
            transition = MacroTransition(
                state=self._current_state,
                action=action,
                reward_cny=macro_ledger.reward_cny,
                next_state=next_state,
                done=done,
                executed_mpc_steps=executed,
                ledger=macro_ledger,
            )
            if self._replay_sink is not None:
                self._replay_sink(transition)
        except Exception as exc:
            self._failed = True
            if isinstance(exc, MacroStepExecutionError):
                raise
            raise MacroStepExecutionError(
                "macro-step execution failed",
                executed_mpc_steps=executed,
            ) from exc

        self._transitions.append(transition)
        self._current_state = next_state
        self._done = done
        return transition


__all__ = [
    "MPCExecutionResult",
    "MPCStepBackend",
    "MacroStepExecutionError",
    "MacroTransition",
    "MultiRateWeightEnvironment",
    "TRAINING_READINESS_STATUS",
]
