"""Deterministic per-round Train episode permutations."""

from __future__ import annotations

import random


class EpisodeShuffleSchedule:
    def __init__(self, episode_ids: tuple[str, ...], *, seed: int) -> None:
        if type(episode_ids) is not tuple or not episode_ids:
            raise TypeError("episode_ids must be a nonempty exact tuple")
        if any(type(value) is not str or not value for value in episode_ids):
            raise ValueError("episode_ids must contain nonempty exact strings")
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError("episode_ids must be unique")
        if type(seed) is not int:
            raise TypeError("seed must be an exact int")
        self._episode_ids = episode_ids
        self._rng = random.Random(seed)
        self._rounds_emitted = 0

    @property
    def rounds_emitted(self) -> int:
        return self._rounds_emitted

    def next_round(self) -> tuple[str, ...]:
        values = list(self._episode_ids)
        self._rng.shuffle(values)
        self._rounds_emitted += 1
        return tuple(values)

    def state_dict(self) -> dict[str, object]:
        return {
            "episode_ids": self._episode_ids,
            "rng_state": self._rng.getstate(),
            "rounds_emitted": self._rounds_emitted,
        }

    def load_state_dict(self, state: object) -> None:
        if type(state) is not dict:
            raise TypeError("schedule state must be an exact dict")
        if state.get("episode_ids") != self._episode_ids:
            raise ValueError("schedule episode identities differ")
        rounds = state.get("rounds_emitted")
        if type(rounds) is not int or rounds < 0:
            raise ValueError("schedule rounds_emitted is invalid")
        self._rng.setstate(state["rng_state"])
        self._rounds_emitted = rounds


__all__ = ["EpisodeShuffleSchedule"]
