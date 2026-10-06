"""Masked MLP Double-DQN for direct FC power and raw-CNY rewards."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
from torch import nn

from dqn.networks.mlp_qnet import MLPQNetwork

from .control import ACTION_KW, DirectTransition


STATE_DIM = 8


def masked_double_dqn_targets(
    rewards: torch.Tensor, online_next_q: torch.Tensor,
    target_next_q: torch.Tensor, done: torch.Tensor, *,
    next_action_masks: torch.Tensor, gamma: float,
) -> torch.Tensor:
    """Choose the next feasible action online; evaluate it with the target net."""
    if online_next_q.shape != target_next_q.shape or online_next_q.shape != next_action_masks.shape:
        raise ValueError("next-Q arrays and action mask must share a shape")
    if rewards.shape != done.shape or rewards.shape != (online_next_q.shape[0], 1):
        raise ValueError("rewards and done must be column vectors")
    if ((done == 0).flatten() & ~next_action_masks.any(dim=1)).any():
        raise ValueError("a nonterminal next state needs a feasible action")
    selected = online_next_q.masked_fill(~next_action_masks, -torch.inf).argmax(dim=1, keepdim=True)
    return rewards + gamma * (1.0 - done) * target_next_q.gather(1, selected)


@dataclass(frozen=True)
class ReplayItem:
    state: tuple[float, ...]
    action_index: int
    reward_cny: float
    next_state: tuple[float, ...]
    done: bool
    next_feasible_indices: tuple[int, ...]


class DirectPowerDDQN:
    def __init__(
        self, *, seed: int = 42, gamma: float = 1.0,
        learning_rate: float = 1e-4, replay_capacity: int = 100_000,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        if not 0 <= gamma <= 1 or learning_rate <= 0 or replay_capacity < 1:
            raise ValueError("invalid Double-DQN hyperparameters")
        self.gamma = float(gamma)
        self.random = random.Random(seed)
        torch.manual_seed(seed)
        self.online = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.target = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[ReplayItem] = deque(maxlen=replay_capacity)

    @staticmethod
    def _state(value: Sequence[float]) -> tuple[float, ...]:
        state = tuple(float(item) for item in value)
        if len(state) != STATE_DIM or not all(math.isfinite(item) for item in state):
            raise ValueError("state must have eight finite features")
        return state

    @staticmethod
    def _indices(actions: Sequence[int]) -> tuple[int, ...]:
        values = tuple(actions)
        if any(type(value) is not int or value not in ACTION_KW for value in values):
            raise ValueError("feasible actions must use the 10 kW grid")
        if len(set(values)) != len(values):
            raise ValueError("feasible actions contain duplicates")
        return tuple(ACTION_KW.index(value) for value in values)

    def select_power(
        self, state: Sequence[float], feasible_actions: Sequence[int], *, epsilon: float = 0.0,
    ) -> int:
        values = self._state(state)
        indices = self._indices(feasible_actions)
        if not indices:
            raise ValueError("no feasible FC action")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.random.random() < epsilon:
            return ACTION_KW[self.random.choice(indices)]
        with torch.no_grad():
            q = self.online(torch.tensor([values], dtype=torch.float32))[0]
        mask = torch.full_like(q, -torch.inf)
        mask[list(indices)] = q[list(indices)]
        return ACTION_KW[int(mask.argmax().item())]

    def remember(
        self, state: Sequence[float], action_kw: int, reward_cny: float,
        next_state: Sequence[float], *, done: bool,
        next_feasible_actions: Sequence[int],
    ) -> None:
        if type(action_kw) is not int or action_kw not in ACTION_KW:
            raise ValueError("action must use the FC grid")
        if not math.isfinite(reward_cny) or type(done) is not bool:
            raise ValueError("reward must be finite and done must be bool")
        next_indices = self._indices(next_feasible_actions)
        if (done and next_indices) or (not done and not next_indices):
            raise ValueError("next feasible actions do not match terminal flag")
        self.replay.append(ReplayItem(
            self._state(state), ACTION_KW.index(action_kw), float(reward_cny),
            self._state(next_state), done, next_indices,
        ))

    def remember_transition(self, transition: DirectTransition) -> None:
        self.remember(
            transition.state, transition.action_kw, transition.reward_cny,
            transition.next_state, done=transition.done,
            next_feasible_actions=transition.next_feasible_actions,
        )

    def learn(self, *, batch_size: int = 64) -> float | None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.replay) < batch_size:
            return None
        batch = self.random.sample(tuple(self.replay), batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32)
        actions = torch.tensor([[item.action_index] for item in batch], dtype=torch.long)
        rewards = torch.tensor([[item.reward_cny] for item in batch], dtype=torch.float32)
        done = torch.tensor([[float(item.done)] for item in batch], dtype=torch.float32)
        masks = torch.zeros((batch_size, len(ACTION_KW)), dtype=torch.bool)
        for row, item in enumerate(batch):
            masks[row, list(item.next_feasible_indices)] = True
        q = self.online(states).gather(1, actions)
        with torch.no_grad():
            target = masked_double_dqn_targets(
                rewards, self.online(next_states), self.target(next_states), done,
                next_action_masks=masks, gamma=self.gamma,
            )
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
