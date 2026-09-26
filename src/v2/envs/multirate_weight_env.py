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
from ..control.nonlinear_mpc import MPCWeights, PhysicalInfeasibilityError
from ..dqn.action_space import ActionCandidate
from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG
from ..dqn.state import FORMAL_STATE_DIMENSION
from ..economics import RawCnyIntervalLedger
from ..failure_policy import FORMAL_FAILURE_KIND, FORMAL_FAILURE_POLICY, FormalFailurePolicy


TRAINING_READINESS_STATUS = "READY_FOR_INTEGRATED_PREFLIGHT"


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
    """Backend boundary: one call advances one physical control interval."""

    def execute_mpc_step(self, weights: MPCWeights) -> "MPCExecutionResult": ...


@dataclass(frozen=True)
class MPCExecutionResult:
    """Economic result of one physical interval, with explicit MPC semantics."""

    ledger: RawCnyIntervalLedger
    done: bool
    mpc_solve_executed: bool = True
    next_decision_ready: bool = True

    def __post_init__(self) -> None:
        _validate_ledger(self.ledger)
        if type(self.done) is not bool:
            raise TypeError("done must be an exact bool")
        if type(self.mpc_solve_executed) is not bool:
            raise TypeError("mpc_solve_executed must be an exact bool")
        if type(self.next_decision_ready) is not bool:
            raise TypeError("next_decision_ready must be an exact bool")
        if self.done and not self.next_decision_ready:
            raise ValueError("terminal interval must expose a decision boundary")


@dataclass(frozen=True)
class MacroTransition:
    """The sole replay item emitted at a DQN macro boundary."""

    state: tuple[float, ...]
    action: ActionCandidate
    learning_reward: float
    next_state: tuple[float, ...]
    done: bool
    executed_mpc_steps: int
    ledger: RawCnyIntervalLedger
    failure_penalty_score: float = 0.0
    failure_kind: str | None = None

    def __post_init__(self) -> None:
        _validate_state(self.state, "state")
        _validate_candidate(self.action)
        _validate_state(self.next_state, "next_state")
        _validate_ledger(self.ledger)
        if type(self.learning_reward) is not float or not math.isfinite(self.learning_reward):
            raise TypeError("learning_reward must be an exact finite float")
        if (
            type(self.failure_penalty_score) is not float
            or not math.isfinite(self.failure_penalty_score)
            or self.failure_penalty_score < 0.0
        ):
            raise TypeError("failure_penalty_score must be an exact finite nonnegative float")
        if type(self.done) is not bool:
            raise TypeError("done must be an exact bool")
        if type(self.executed_mpc_steps) is not int:
            raise TypeError("executed_mpc_steps must be an exact integer")
        if self.executed_mpc_steps < 0:
            raise ValueError("executed_mpc_steps must be nonnegative")
        if self.failure_kind is None:
            if self.failure_penalty_score != 0.0:
                raise ValueError("successful transition cannot carry a failure penalty")
            if self.executed_mpc_steps <= 0:
                raise ValueError("successful transition must execute at least one MPC step")
            if self.learning_reward != self.ledger.reward_cny:
                raise ValueError("successful learning reward must equal raw economic reward")
        else:
            if type(self.failure_kind) is not str:
                raise TypeError("failure_kind must be an exact string or None")
            if self.failure_kind != FORMAL_FAILURE_KIND:
                raise ValueError("failure_kind differs from the formal physical failure kind")
            if not self.done:
                raise ValueError("failed transition must terminate the episode")
            if self.failure_penalty_score != FORMAL_FAILURE_POLICY.penalty_score:
                raise ValueError("failed transition must use the frozen penalty score")
            expected = self.ledger.reward_cny - self.failure_penalty_score
            if self.learning_reward != expected:
                raise ValueError("failed learning reward must include the separate penalty")

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def raw_economic_cost_cny(self) -> float:
        return self.ledger.total_cost_cny

    @property
    def failed(self) -> bool:
        return self.failure_kind is not None

    @property
    def episode_completed(self) -> bool:
        return self.done and not self.failed


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


class ReplaySinkNotificationError(RuntimeError):
    """An observer failed after the authoritative transition was committed."""

    def __init__(self, transition: MacroTransition) -> None:
        if type(transition) is not MacroTransition:
            raise TypeError("transition must be an exact MacroTransition")
        MacroTransition(**transition.__dict__)
        super().__init__(
            "replay observer notification failed after the transition was committed"
        )
        self.transition_committed = True
        self.transition = transition


def _transition_snapshot(value: MacroTransition) -> MacroTransition:
    """Return a deeply detached canonical value for an external observer."""
    if type(value) is not MacroTransition:
        raise TypeError("transition must be an exact MacroTransition")
    MacroTransition(**value.__dict__)
    return MacroTransition(
        state=tuple(item for item in value.state),
        action=ActionCandidate(*value.action.numerators),
        learning_reward=value.learning_reward,
        next_state=tuple(item for item in value.next_state),
        done=value.done,
        executed_mpc_steps=value.executed_mpc_steps,
        ledger=RawCnyIntervalLedger(*value.ledger.components_cny),
        failure_penalty_score=value.failure_penalty_score,
        failure_kind=value.failure_kind,
    )


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
        formal_training_mode: bool = False,
        replay_sink: Callable[[MacroTransition], None] | None = None,
        failure_policy: FormalFailurePolicy | None = None,
    ) -> None:
        if type(synthetic_test_mode) is not bool:
            raise TypeError("synthetic_test_mode must be an exact bool")
        if type(formal_training_mode) is not bool:
            raise TypeError("formal_training_mode must be an exact bool")
        if synthetic_test_mode == formal_training_mode:
            raise PermissionError(
                "exactly one of synthetic_test_mode or formal_training_mode is required"
            )
        checked_timescale = _validate_timescale(timescale)
        if type(action_catalog) is not tuple:
            raise TypeError("action_catalog must be an immutable exact tuple")
        if not action_catalog:
            raise ValueError("action_catalog must not be empty")
        checked_catalog = tuple(_validate_candidate(item) for item in action_catalog)
        if formal_training_mode and checked_catalog != FINAL_DQN_ACTION_CATALOG:
            raise ValueError("formal mode requires the exact frozen 36-action catalog")
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
        if formal_training_mode and failure_policy is None:
            failure_policy = FORMAL_FAILURE_POLICY
        if failure_policy is not None:
            if type(failure_policy) is not FormalFailurePolicy:
                raise TypeError("failure_policy must be an exact FormalFailurePolicy or None")
            if failure_policy != FORMAL_FAILURE_POLICY:
                raise ValueError("failure_policy differs from the frozen formal policy")

        self._timescale = checked_timescale
        self._actions = dict(zip(action_ids, checked_catalog))
        self._execute_mpc_step = execute
        self._backend = backend
        self._formal_training_mode = formal_training_mode
        self._state_provider = state_provider
        self._replay_sink = replay_sink
        self._failure_policy = failure_policy
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
        if self._formal_training_mode:
            reset_backend = getattr(self._backend, "reset", None)
            if not callable(reset_backend):
                raise TypeError("formal backend must expose callable reset")
            reset_backend()
        state = _validate_state(self._state_provider(), "reset state")
        if self._formal_training_mode and len(state) != FORMAL_STATE_DIMENSION:
            raise ValueError("formal reset state must use frozen S8 dimension")
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
            encountered_pause = False
            while True:
                # Backends are untrusted mutable boundaries.  A fresh value
                # prevents one call from poisoning a later MPC execution.
                weights = MPCWeights(*canonical_weight_values)
                result = self._execute_mpc_step(weights)
                # Returning from this boundary means one physical command was
                # executed even if the backend's result object is malformed.
                if type(result) is not MPCExecutionResult:
                    executed += 1
                    raise TypeError("backend result must be an exact MPCExecutionResult")
                _validate_ledger(result.ledger)
                if type(result.done) is not bool:
                    raise TypeError("stored backend done flag must remain an exact bool")
                # Snapshot before the next backend call can mutate an object it
                # previously returned.
                ledger = RawCnyIntervalLedger(*result.ledger.components_cny)
                done_snapshot = result.done
                ledgers.append(ledger)
                if result.mpc_solve_executed:
                    executed += 1
                else:
                    encountered_pause = True
                done = done_snapshot
                if done:
                    break
                if result.next_decision_ready and (
                    encountered_pause
                    or executed >= self._timescale.dqn_switch_steps
                ):
                    break

            macro_ledger = _aggregate_ledgers(ledgers)
            next_state = _validate_state(self._state_provider(), "boundary next_state")
            transition = MacroTransition(
                state=self._current_state,
                action=action,
                learning_reward=macro_ledger.reward_cny,
                next_state=next_state,
                done=done,
                executed_mpc_steps=executed,
                ledger=macro_ledger,
            )
        except PhysicalInfeasibilityError as exc:
            if self._failure_policy is None:
                self._failed = True
                raise MacroStepExecutionError(
                    "macro-step execution failed",
                    executed_mpc_steps=executed,
                ) from exc
            try:
                macro_ledger = _aggregate_ledgers(ledgers)
                next_state = _validate_state(
                    self._state_provider(), "physical failure next_state"
                )
                transition = MacroTransition(
                    state=self._current_state,
                    action=action,
                    learning_reward=(
                        macro_ledger.reward_cny - self._failure_policy.penalty_score
                    ),
                    next_state=next_state,
                    done=True,
                    executed_mpc_steps=executed,
                    ledger=macro_ledger,
                    failure_penalty_score=self._failure_policy.penalty_score,
                    failure_kind=self._failure_policy.failure_kind,
                )
                done = True
            except Exception as transition_error:
                self._failed = True
                raise MacroStepExecutionError(
                    "physical-failure transition construction failed",
                    executed_mpc_steps=executed,
                ) from transition_error
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
        if self._replay_sink is not None:
            observer_value = _transition_snapshot(transition)
            try:
                self._replay_sink(observer_value)
            except Exception as exc:
                raise ReplaySinkNotificationError(transition) from exc
        return transition


def _aggregate_ledgers(
    ledgers: list[RawCnyIntervalLedger],
) -> RawCnyIntervalLedger:
    components = tuple(
        math.fsum(ledger.components_cny[index] for ledger in ledgers)
        for index in range(4)
    )
    if not all(math.isfinite(value) for value in components):
        raise ValueError("macro raw-CNY component accumulation must remain finite")
    return RawCnyIntervalLedger(*components)


__all__ = [
    "MPCExecutionResult",
    "MPCStepBackend",
    "MacroStepExecutionError",
    "MacroTransition",
    "MultiRateWeightEnvironment",
    "ReplaySinkNotificationError",
    "TRAINING_READINESS_STATUS",
]
