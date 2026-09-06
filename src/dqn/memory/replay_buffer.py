from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.common_types import DQNTransition


@dataclass
class ReplayBuffer:
    """Experience replay memory for off-policy ship Q-learning.

    The buffer stores transitions `(s_t, a_t, r_t, s_{t+1}, done_t)` gathered
    from the lower-layer environment. Random minibatch sampling breaks the
    strong temporal correlation in sequential vessel data and is therefore a
    key part of stable DQN training.
    """

    max_size: int
    size: int = 0
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    next_states: list[np.ndarray] = field(default_factory=list)

    def push(self, state, action, reward, done, next_state) -> None:
        transition = DQNTransition(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=bool(done),
        )
        if self.size < self.max_size:
            self.states.append(transition.state)
            self.actions.append(transition.action)
            self.rewards.append(transition.reward)
            self.dones.append(transition.done)
            self.next_states.append(transition.next_state)
        else:
            idx = self.size % self.max_size
            self.states[idx] = transition.state
            self.actions[idx] = transition.action
            self.rewards[idx] = transition.reward
            self.dones[idx] = transition.done
            self.next_states[idx] = transition.next_state
        self.size += 1

    def __len__(self) -> int:
        return min(self.size, self.max_size)

    @property
    def write_position(self) -> int:
        """Return the next circular write index for checkpoint diagnostics."""

        return int(self.size % self.max_size)

    def state_dict(self) -> dict[str, object]:
        """Serialize the complete replay state required for exact resume."""

        return {
            "max_size": int(self.max_size),
            "size": int(self.size),
            "write_position": int(self.write_position),
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.bool_),
            "next_states": np.asarray(self.next_states, dtype=np.float32),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore a replay state after validating capacity and circular order."""

        required = {
            "max_size", "size", "write_position", "states", "actions",
            "rewards", "dones", "next_states",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"replay checkpoint is missing keys: {sorted(missing)}")
        if int(state["max_size"]) != int(self.max_size):
            raise ValueError("replay checkpoint buffer capacity does not match config")
        size = int(state["size"])
        if size < 0:
            raise ValueError("replay checkpoint size must be nonnegative")
        length = min(size, self.max_size)
        arrays = {
            "states": np.asarray(state["states"], dtype=np.float32),
            "actions": np.asarray(state["actions"], dtype=np.int64),
            "rewards": np.asarray(state["rewards"], dtype=np.float32),
            "dones": np.asarray(state["dones"], dtype=np.bool_),
            "next_states": np.asarray(state["next_states"], dtype=np.float32),
        }
        if any(len(values) != length for values in arrays.values()):
            raise ValueError("replay checkpoint arrays do not match stored size")
        if int(state["write_position"]) != size % self.max_size:
            raise ValueError("replay checkpoint write position is inconsistent")
        self.size = size
        self.states = [np.asarray(value, dtype=np.float32).copy() for value in arrays["states"]]
        self.actions = [int(value) for value in arrays["actions"]]
        self.rewards = [float(value) for value in arrays["rewards"]]
        self.dones = [bool(value) for value in arrays["dones"]]
        self.next_states = [np.asarray(value, dtype=np.float32).copy() for value in arrays["next_states"]]

    def sample(self, batch_size: int):
        total = len(self)
        indices = np.random.randint(total, size=batch_size)
        states = np.asarray([self.states[i] for i in indices], dtype=np.float32)
        actions = np.asarray([self.actions[i] for i in indices], dtype=np.int64)
        rewards = np.asarray([self.rewards[i] for i in indices], dtype=np.float32)
        dones = np.asarray([self.dones[i] for i in indices], dtype=np.float32)
        next_states = np.asarray([self.next_states[i] for i in indices], dtype=np.float32)
        return states, actions, rewards, dones, next_states

    def sample_states(self, batch_size: int) -> np.ndarray:
        total = len(self)
        indices = np.random.randint(total, size=min(int(batch_size), total))
        return np.asarray([self.states[i] for i in indices], dtype=np.float32)
