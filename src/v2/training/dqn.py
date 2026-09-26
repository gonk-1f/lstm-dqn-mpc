"""Independent v2 Double-DQN primitives with explicit frozen defaults."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from numbers import Real

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..dqn.action_space import FINAL_DQN_ACTION_CATALOG
from ..dqn.state import FORMAL_STATE_DIMENSION


EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 150_000


@dataclass(frozen=True)
class DqnTrainingConfig:
    state_dim: int
    action_dim: int
    hidden_dims: tuple[int, int]
    gamma: float
    learning_rate: float
    batch_size: int
    replay_capacity: int
    warmup_steps: int
    target_sync_steps: int
    gradient_clip_norm: float
    rounds: int

    def __post_init__(self) -> None:
        for name in (
            "state_dim", "action_dim", "batch_size", "replay_capacity",
            "warmup_steps", "target_sync_steps", "rounds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.hidden_dims) is not tuple or len(self.hidden_dims) != 2 or any(
            type(value) is not int or value <= 0 for value in self.hidden_dims
        ):
            raise ValueError("hidden_dims must contain two positive exact integers")
        for name in ("gamma", "learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real scalar")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if self.learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clip must be positive")
        if self.state_dim != FORMAL_STATE_DIMENSION or self.action_dim != len(FINAL_DQN_ACTION_CATALOG):
            raise ValueError("DQN dimensions must match frozen S8/36 contracts")

    @classmethod
    def formal_baseline(cls) -> "DqnTrainingConfig":
        return cls(
            state_dim=FORMAL_STATE_DIMENSION,
            action_dim=len(FINAL_DQN_ACTION_CATALOG),
            hidden_dims=(128, 128),
            gamma=1.0,
            learning_rate=1.0e-4,
            batch_size=256,
            replay_capacity=200_000,
            warmup_steps=5_000,
            target_sync_steps=1_000,
            gradient_clip_norm=10.0,
            rounds=40,
        )


def epsilon_at_global_step(global_macro_step: int) -> float:
    if type(global_macro_step) is not int or global_macro_step < 0:
        raise ValueError("global_macro_step must be a nonnegative exact integer")
    if global_macro_step >= EPSILON_DECAY_STEPS:
        return EPSILON_END
    fraction = global_macro_step / EPSILON_DECAY_STEPS
    return float(EPSILON_START + fraction * (EPSILON_END - EPSILON_START))


class QNetwork(nn.Module):
    def __init__(self, config: DqnTrainingConfig) -> None:
        super().__init__()
        if type(config) is not DqnTrainingConfig:
            raise TypeError("config must be an exact DqnTrainingConfig")
        h1, h2 = config.hidden_dims
        self.layers = nn.Sequential(
            nn.Linear(config.state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, config.action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, *, state_dim: int, seed: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive exact integer")
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive exact integer")
        if type(seed) is not int:
            raise TypeError("seed must be an exact int")
        self.capacity = capacity
        self.state_dim = state_dim
        self.states = np.empty((capacity, state_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        left = np.asarray(state, dtype=np.float32)
        right = np.asarray(next_state, dtype=np.float32)
        if left.shape != (self.state_dim,) or right.shape != (self.state_dim,):
            raise ValueError("replay states must match state_dim")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("replay states must be finite")
        if type(action) is not int or not 0 <= action < len(FINAL_DQN_ACTION_CATALOG):
            raise ValueError("replay action index is outside frozen catalog")
        if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
            raise ValueError("replay reward must be finite")
        if type(done) is not bool:
            raise TypeError("replay done must be an exact bool")
        index = self.position
        self.states[index] = left
        self.actions[index] = action
        self.rewards[index] = float(reward)
        self.next_states[index] = right
        self.dones[index] = float(done)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if type(batch_size) is not int or batch_size <= 0 or batch_size > self.size:
            raise ValueError("batch_size must be positive and no larger than replay")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            self.states[indices].copy(), self.actions[indices].copy(),
            self.rewards[indices].copy(), self.next_states[indices].copy(),
            self.dones[indices].copy(),
        )

    def state_dict(self) -> dict[str, object]:
        length = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "state_dim": self.state_dim,
            "position": self.position,
            "size": self.size,
            "states": self.states[:length].copy(),
            "actions": self.actions[:length].copy(),
            "rewards": self.rewards[:length].copy(),
            "next_states": self.next_states[:length].copy(),
            "dones": self.dones[:length].copy(),
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: object) -> None:
        if type(state) is not dict:
            raise TypeError("replay state must be an exact dict")
        if state.get("capacity") != self.capacity or state.get("state_dim") != self.state_dim:
            raise ValueError("replay dimensions differ from formal configuration")
        size = state.get("size")
        position = state.get("position")
        if type(size) is not int or not 0 <= size <= self.capacity:
            raise ValueError("replay size is invalid")
        if type(position) is not int or not 0 <= position < self.capacity:
            raise ValueError("replay position is invalid")
        length = self.capacity if size == self.capacity else size
        arrays = (
            ("states", self.states, (length, self.state_dim)),
            ("actions", self.actions, (length,)),
            ("rewards", self.rewards, (length,)),
            ("next_states", self.next_states, (length, self.state_dim)),
            ("dones", self.dones, (length,)),
        )
        for name, destination, shape in arrays:
            source = np.asarray(state.get(name))
            if source.shape != shape:
                raise ValueError(f"replay {name} shape differs")
            destination[:length] = source
        self.size = size
        self.position = position
        self.rng.bit_generator.state = state["rng_state"]


class DqnAgent:
    def __init__(self, config: DqnTrainingConfig, *, seed: int, device: str) -> None:
        if type(config) is not DqnTrainingConfig:
            raise TypeError("config must be an exact DqnTrainingConfig")
        if type(seed) is not int or type(device) is not str:
            raise TypeError("seed and device must be exact int/str values")
        self.config = config
        self.device = torch.device(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.rng = random.Random(seed)
        self.online = QNetwork(config).to(self.device)
        self.target = QNetwork(config).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.replay_capacity, state_dim=config.state_dim, seed=seed)
        self.optimizer_steps = 0

    def select_action(self, state: np.ndarray, *, epsilon: float) -> int:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != (self.config.state_dim,) or not np.isfinite(values).all():
            raise ValueError("state must be one finite formal state vector")
        if not 0.0 <= float(epsilon) <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if self.rng.random() < float(epsilon):
            return self.rng.randrange(self.config.action_dim)
        return self.greedy_action(values)

    def greedy_action(self, state: np.ndarray) -> int:
        """Select the online-network argmax without consuming exploration RNG."""

        values = np.asarray(state, dtype=np.float32)
        if values.shape != (self.config.state_dim,) or not np.isfinite(values).all():
            raise ValueError("state must be one finite formal state vector")
        with torch.no_grad():
            tensor = torch.as_tensor(values, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def optimize(self) -> float:
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(batch.states, device=self.device)
        actions = torch.as_tensor(batch.actions, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = torch.as_tensor(batch.next_states, device=self.device)
        dones = torch.as_tensor(batch.dones, device=self.device)
        predicted = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_values = self.target(next_states).gather(1, next_actions).squeeze(1)
            expected = rewards + self.config.gamma * (1.0 - dones) * next_values
        loss = F.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.optimizer_steps += 1
        if self.optimizer_steps % self.config.target_sync_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.detach().cpu().item())

    def state_dict(self) -> dict[str, object]:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_steps": self.optimizer_steps,
            "replay": self.replay.state_dict(),
            "agent_rng_state": self.rng.getstate(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

    def load_state_dict(self, state: object) -> None:
        if type(state) is not dict:
            raise TypeError("agent state must be an exact dict")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        steps = state["optimizer_steps"]
        if type(steps) is not int or steps < 0:
            raise ValueError("optimizer_steps is invalid")
        self.optimizer_steps = steps
        self.replay.load_state_dict(state["replay"])
        self.rng.setstate(state["agent_rng_state"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])


__all__ = [
    "DqnAgent", "DqnTrainingConfig", "EPSILON_DECAY_STEPS", "EPSILON_END",
    "EPSILON_START", "QNetwork", "ReplayBatch", "ReplayBuffer",
    "epsilon_at_global_step",
]
