"""Shared no-learning evaluation for formal v2 policies."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Protocol

import numpy as np

from ..config import TimeScaleConfig
from ..data.formal_training_dataset import FormalEpisode
from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG
from ..dqn.state import FORMAL_STATE_DIMENSION
from ..envs.formal_episode import FormalEpisodeBackend, build_formal_nonlinear_mpc
from ..envs.multirate_weight_env import MacroTransition, MultiRateWeightEnvironment


_ACTION_INDEX_BY_ID = {
    action.action_id: index
    for index, action in enumerate(FINAL_DQN_ACTION_CATALOG)
}
_ACTION_ORDER = tuple(_ACTION_INDEX_BY_ID)


def _exact_finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be an exact finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _validated_action_counts(value: object) -> tuple[tuple[str, int], ...]:
    if type(value) is not tuple:
        raise TypeError("action_counts must be an immutable exact tuple")
    previous_index = -1
    checked: list[tuple[str, int]] = []
    for item in value:
        if type(item) is not tuple or len(item) != 2:
            raise TypeError("each action count must be an exact (str, int) tuple")
        action_id, count = item
        if type(action_id) is not str or action_id not in _ACTION_INDEX_BY_ID:
            raise ValueError("action_counts contains an unknown action")
        if type(count) is not int or count <= 0:
            raise ValueError("action counts must be exact positive integers")
        index = _ACTION_INDEX_BY_ID[action_id]
        if index <= previous_index:
            raise ValueError("action_counts must follow canonical action-catalog order")
        previous_index = index
        checked.append((action_id, count))
    return tuple(checked)


def _validated_s8(state: object) -> np.ndarray:
    try:
        values = np.asarray(state, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("state must be one finite S8 vector") from exc
    if values.shape != (FORMAL_STATE_DIMENSION,) or not np.isfinite(values).all():
        raise ValueError("state must be one finite S8 vector")
    return values.copy()


@dataclass(frozen=True)
class EpisodeEvaluation:
    sample_id: str
    completed: bool
    failure_kind: str | None
    transition_count: int
    executed_mpc_steps: int
    h2_cost_cny: float
    fc_degradation_cost_cny: float
    battery_degradation_cost_cny: float
    shore_cost_cny: float
    raw_economic_cost_cny: float
    failure_penalty_score: float
    learning_reward: float
    soc_min: float
    soc_max: float
    action_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be a nonempty exact string")
        if type(self.completed) is not bool:
            raise TypeError("completed must be an exact bool")
        if self.failure_kind is not None and (
            type(self.failure_kind) is not str or not self.failure_kind
        ):
            raise TypeError("failure_kind must be a nonempty exact string or None")
        if self.completed == (self.failure_kind is not None):
            raise ValueError("episode must have exactly one terminal outcome")
        if type(self.transition_count) is not int or self.transition_count <= 0:
            raise ValueError("transition_count must be an exact positive integer")
        if type(self.executed_mpc_steps) is not int or self.executed_mpc_steps < 0:
            raise ValueError("executed_mpc_steps must be an exact nonnegative integer")
        components = (
            _exact_finite_float(self.h2_cost_cny, "h2_cost_cny", nonnegative=True),
            _exact_finite_float(
                self.fc_degradation_cost_cny,
                "fc_degradation_cost_cny",
                nonnegative=True,
            ),
            _exact_finite_float(
                self.battery_degradation_cost_cny,
                "battery_degradation_cost_cny",
                nonnegative=True,
            ),
            _exact_finite_float(self.shore_cost_cny, "shore_cost_cny", nonnegative=True),
        )
        raw_cost = _exact_finite_float(
            self.raw_economic_cost_cny,
            "raw_economic_cost_cny",
            nonnegative=True,
        )
        penalty = _exact_finite_float(
            self.failure_penalty_score,
            "failure_penalty_score",
            nonnegative=True,
        )
        reward = _exact_finite_float(self.learning_reward, "learning_reward")
        if raw_cost != math.fsum(components):
            raise ValueError("raw economic cost must equal its four components")
        if reward != -raw_cost - penalty:
            raise ValueError("learning reward must equal negative raw cost and penalty")
        if self.completed and penalty != 0.0:
            raise ValueError("completed episode cannot carry a failure penalty")
        soc_min = _exact_finite_float(self.soc_min, "soc_min")
        soc_max = _exact_finite_float(self.soc_max, "soc_max")
        if not 0.0 <= soc_min <= soc_max <= 1.0:
            raise ValueError("SOC diagnostics must satisfy 0 <= min <= max <= 1")
        counts = _validated_action_counts(self.action_counts)
        if sum(count for _, count in counts) != self.transition_count:
            raise ValueError("action counts must equal transition_count")


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_id: str
    episodes: tuple[EpisodeEvaluation, ...]

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be a nonempty exact string")
        if type(self.episodes) is not tuple or not self.episodes:
            raise ValueError("episodes must be a nonempty immutable exact tuple")
        expected_fields = {item.name for item in fields(EpisodeEvaluation)}
        seen_ids: set[str] = set()
        for episode in self.episodes:
            if type(episode) is not EpisodeEvaluation:
                raise TypeError("episodes must contain exact EpisodeEvaluation values")
            if set(vars(episode)) != expected_fields:
                raise ValueError("episode evaluation contains altered stored fields")
            EpisodeEvaluation(**vars(episode))
            if episode.sample_id in seen_ids:
                raise ValueError("episode sample IDs must be unique")
            seen_ids.add(episode.sample_id)

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(episode.sample_id for episode in self.episodes)

    @property
    def completed_episodes(self) -> int:
        return sum(episode.completed for episode in self.episodes)

    @property
    def failed_episodes(self) -> int:
        return len(self.episodes) - self.completed_episodes

    @property
    def completion_rate(self) -> float:
        return float(self.completed_episodes / len(self.episodes))

    @property
    def transition_count(self) -> int:
        return sum(episode.transition_count for episode in self.episodes)

    @property
    def transitions(self) -> int:
        return self.transition_count

    @property
    def executed_mpc_steps(self) -> int:
        return sum(episode.executed_mpc_steps for episode in self.episodes)

    @property
    def h2_cost_cny(self) -> float:
        return math.fsum(episode.h2_cost_cny for episode in self.episodes)

    @property
    def fc_degradation_cost_cny(self) -> float:
        return math.fsum(
            episode.fc_degradation_cost_cny for episode in self.episodes
        )

    @property
    def battery_degradation_cost_cny(self) -> float:
        return math.fsum(
            episode.battery_degradation_cost_cny for episode in self.episodes
        )

    @property
    def shore_cost_cny(self) -> float:
        return math.fsum(episode.shore_cost_cny for episode in self.episodes)

    @property
    def raw_economic_cost_cny(self) -> float:
        return math.fsum(
            (
                self.h2_cost_cny,
                self.fc_degradation_cost_cny,
                self.battery_degradation_cost_cny,
                self.shore_cost_cny,
            )
        )

    @property
    def failure_penalty_score(self) -> float:
        return math.fsum(episode.failure_penalty_score for episode in self.episodes)

    @property
    def learning_reward(self) -> float:
        return -self.raw_economic_cost_cny - self.failure_penalty_score

    @property
    def action_counts(self) -> tuple[tuple[str, int], ...]:
        totals = {action_id: 0 for action_id in _ACTION_ORDER}
        for episode in self.episodes:
            for action_id, count in episode.action_counts:
                totals[action_id] += count
        return tuple(
            (action_id, totals[action_id])
            for action_id in _ACTION_ORDER
            if totals[action_id] > 0
        )


class _FormalPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def action_index(self, state: object) -> int: ...


@dataclass(frozen=True)
class FixedActionPolicy:
    action_id: str

    def __post_init__(self) -> None:
        if type(self.action_id) is not str:
            raise TypeError("action_id must be an exact string")
        if self.action_id not in _ACTION_INDEX_BY_ID:
            raise ValueError(f"unknown action: {self.action_id}")

    @property
    def policy_id(self) -> str:
        return self.action_id

    def action_index(self, state: object) -> int:
        _validated_s8(state)
        return _ACTION_INDEX_BY_ID[self.action_id]


@dataclass(frozen=True)
class GreedyDqnPolicy:
    agent: object
    policy_id: str = "greedy_dqn"

    def __post_init__(self) -> None:
        if type(self.policy_id) is not str or not self.policy_id:
            raise ValueError("policy_id must be a nonempty exact string")
        if not callable(getattr(self.agent, "greedy_action", None)):
            raise TypeError("agent must expose callable greedy_action")

    def action_index(self, state: object) -> int:
        values = _validated_s8(state)
        index = self.agent.greedy_action(values)
        if type(index) is not int or not 0 <= index < len(FINAL_DQN_ACTION_CATALOG):
            raise ValueError("greedy_action must return a canonical action index")
        return index


def build_formal_environment(
    episode: FormalEpisode,
) -> tuple[FormalEpisodeBackend, MultiRateWeightEnvironment]:
    if type(episode) is not FormalEpisode:
        raise TypeError("episode must be an exact FormalEpisode")
    backend = FormalEpisodeBackend(
        load_kw=episode.load_kw,
        speed_kn=episode.speed_kn,
        fc_power_kw=episode.fc_power_kw,
        battery_bus_kw=episode.battery_bus_kw,
        operating_mode=episode.operating_mode,
        mpc=build_formal_nonlinear_mpc(),
    )
    environment = MultiRateWeightEnvironment(
        timescale=TimeScaleConfig.formal_baseline(),
        action_catalog=FINAL_DQN_ACTION_CATALOG,
        backend=backend,
        state_provider=backend.state,
        formal_training_mode=True,
    )
    return backend, environment


def _episode_evaluation(
    episode: FormalEpisode,
    policy: _FormalPolicy,
) -> EpisodeEvaluation:
    backend, environment = build_formal_environment(episode)
    state = environment.reset()
    transitions: list[MacroTransition] = []
    selected_actions: list[str] = []
    while True:
        action_index = policy.action_index(state)
        if type(action_index) is not int or not 0 <= action_index < len(FINAL_DQN_ACTION_CATALOG):
            raise ValueError("policy returned an invalid action index")
        action_id = FINAL_DQN_ACTION_CATALOG[action_index].action_id
        transition = environment.step(action_id)
        if type(transition) is not MacroTransition:
            raise TypeError("formal environment must return exact MacroTransition values")
        if transition.action_id != action_id:
            raise ValueError("formal environment returned a different action")
        transitions.append(transition)
        selected_actions.append(action_id)
        state = transition.next_state
        if transition.done:
            break

    terminal = transitions[-1]
    if not terminal.episode_completed and not terminal.failed:
        raise RuntimeError("evaluation episode ended without a terminal outcome")
    soc_snapshot = tuple(backend.executed_soc)
    if not soc_snapshot:
        raise ValueError("formal backend did not expose an initial SOC snapshot")
    soc_values = tuple(
        _exact_finite_float(value, "executed SOC") for value in soc_snapshot
    )
    if any(not 0.0 <= value <= 1.0 for value in soc_values):
        raise ValueError("executed SOC snapshot lies outside [0, 1]")

    h2_cost = math.fsum(item.ledger.h2_cost_cny for item in transitions)
    fc_cost = math.fsum(
        item.ledger.fuel_cell_degradation_cost_cny for item in transitions
    )
    battery_cost = math.fsum(
        item.ledger.battery_degradation_cost_cny for item in transitions
    )
    shore_cost = math.fsum(item.ledger.shore_cost_cny for item in transitions)
    raw_cost = math.fsum((h2_cost, fc_cost, battery_cost, shore_cost))
    penalty = math.fsum(item.failure_penalty_score for item in transitions)
    counts = {action_id: 0 for action_id in _ACTION_ORDER}
    for action_id in selected_actions:
        counts[action_id] += 1
    action_counts = tuple(
        (action_id, counts[action_id])
        for action_id in _ACTION_ORDER
        if counts[action_id] > 0
    )
    return EpisodeEvaluation(
        sample_id=episode.sample_id,
        completed=terminal.episode_completed,
        failure_kind=terminal.failure_kind,
        transition_count=len(transitions),
        executed_mpc_steps=sum(item.executed_mpc_steps for item in transitions),
        h2_cost_cny=h2_cost,
        fc_degradation_cost_cny=fc_cost,
        battery_degradation_cost_cny=battery_cost,
        shore_cost_cny=shore_cost,
        raw_economic_cost_cny=raw_cost,
        failure_penalty_score=penalty,
        learning_reward=-raw_cost - penalty,
        soc_min=min(soc_values),
        soc_max=max(soc_values),
        action_counts=action_counts,
    )


def evaluate_formal_policy(
    *,
    episodes: tuple[FormalEpisode, ...],
    policy: FixedActionPolicy | GreedyDqnPolicy,
) -> PolicyEvaluation:
    """Evaluate ordered formal episodes without training or exploration."""

    if type(episodes) is not tuple or not episodes:
        raise ValueError("episodes must be a nonempty immutable exact tuple")
    if type(policy) not in (FixedActionPolicy, GreedyDqnPolicy):
        raise TypeError("policy must be an exact formal evaluation policy")
    for episode in episodes:
        if type(episode) is not FormalEpisode:
            raise TypeError("episodes must contain exact FormalEpisode values")
    results = tuple(_episode_evaluation(episode, policy) for episode in episodes)
    return PolicyEvaluation(policy.policy_id, results)


__all__ = [
    "EpisodeEvaluation",
    "FixedActionPolicy",
    "GreedyDqnPolicy",
    "PolicyEvaluation",
    "build_formal_environment",
    "evaluate_formal_policy",
]
