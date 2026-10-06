"""Double DQN for persistence or forecast-aware control states."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .control import DecisionTransition, MPCWeights


def double_dqn_targets(
    rewards: torch.Tensor, online_next_q: torch.Tensor,
    target_next_q: torch.Tensor, done: torch.Tensor, *, gamma: float,
) -> torch.Tensor:
    """Online chooses a'; target evaluates it; terminal rows do not bootstrap."""
    if online_next_q.shape != target_next_q.shape:
        raise ValueError("online and target Q arrays must have equal shape")
    chosen = online_next_q.argmax(dim=1, keepdim=True)
    return rewards + gamma * (1.0 - done) * target_next_q.gather(1, chosen)


@dataclass(frozen=True)
class ReplayItem:
    state: tuple[float, ...]
    action_index: int
    reward_cny: float
    next_state: tuple[float, ...]
    done: bool


class _QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DoubleDQNAgent:
    """Caller supplies Train-screened independent (lambda_ref, lambda_soc) pairs."""

    def __init__(
        self, actions: Sequence[MPCWeights], *, state_dim: int = 10,
        reward_scale_cny: float, seed: int = 42, gamma: float = 1.0,
        learning_rate: float = 1e-4, replay_capacity: int = 200_000,
    ):
        self.actions = tuple(actions)
        if not self.actions or any(type(item) is not MPCWeights for item in self.actions):
            raise ValueError("actions must contain MPCWeights")
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("action catalog contains duplicates")
        if state_dim not in (10, 15):
            raise ValueError("DQN state must have 10 persistence or 15 forecast-aware values")
        if not math.isfinite(reward_scale_cny) or reward_scale_cny <= 0:
            raise ValueError("reward scale must come from positive Train-only cost")
        if not 0 < gamma <= 1 or replay_capacity <= 0:
            raise ValueError("invalid DQN training configuration")
        self.state_dim = state_dim
        self.reward_scale_cny = float(reward_scale_cny)
        self.gamma = gamma
        self.random = random.Random(seed)
        torch.manual_seed(seed)
        self.online = _QNetwork(state_dim, len(self.actions))
        self.target = _QNetwork(state_dim, len(self.actions))
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[ReplayItem] = deque(maxlen=replay_capacity)

    def _state(self, value: Sequence[float]) -> tuple[float, ...]:
        state = tuple(float(item) for item in value)
        if len(state) != self.state_dim or not np.isfinite(state).all():
            raise ValueError(f"DQN state must contain {self.state_dim} finite values")
        return state

    def select_weights(self, state: Sequence[float], *, epsilon: float = 0.0) -> MPCWeights:
        values = self._state(state)
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0, 1]")
        if self.random.random() < epsilon:
            index = self.random.randrange(len(self.actions))
        else:
            with torch.no_grad():
                q = self.online(torch.tensor([values], dtype=torch.float32))
            index = int(q.argmax(dim=1).item())
        return self.actions[index]

    def remember(
        self, state: Sequence[float], action: MPCWeights, reward_cny: float,
        next_state: Sequence[float], *, done: bool,
    ) -> None:
        if action not in self.actions:
            raise ValueError("action was not selected from this catalog")
        if not math.isfinite(reward_cny) or type(done) is not bool:
            raise ValueError("reward and done must be finite and valid")
        self.replay.append(ReplayItem(
            self._state(state), self.actions.index(action), float(reward_cny),
            self._state(next_state), done,
        ))

    def remember_transition(self, transition: DecisionTransition, *, done: bool | None = None) -> None:
        if transition.done and done is False:
            raise ValueError("a terminal transition cannot be stored as nonterminal")
        self.remember(
            transition.state, transition.action, transition.reward_cny,
            transition.next_state, done=transition.done if done is None else done,
        )

    def learn(self, *, batch_size: int = 64) -> float | None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.replay) < batch_size:
            return None
        sample = self.random.sample(tuple(self.replay), batch_size)
        states = torch.tensor([item.state for item in sample], dtype=torch.float32)
        next_states = torch.tensor([item.next_state for item in sample], dtype=torch.float32)
        actions = torch.tensor([[item.action_index] for item in sample])
        rewards = torch.tensor(
            [[item.reward_cny / self.reward_scale_cny] for item in sample], dtype=torch.float32
        )
        done = torch.tensor([[float(item.done)] for item in sample], dtype=torch.float32)
        q = self.online(states).gather(1, actions)
        with torch.no_grad():
            targets = double_dqn_targets(
                rewards, self.online(next_states), self.target(next_states),
                done, gamma=self.gamma,
            )
        loss = nn.functional.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
