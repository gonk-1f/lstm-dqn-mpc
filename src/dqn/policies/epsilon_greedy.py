from __future__ import annotations

import numpy as np


class EpsilonGreedyPolicy:
    def __init__(self, epsilon_start: float, epsilon_min: float, epsilon_decay: float):
        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    def select_action(self, greedy_action: int, action_dim: int, warmup: bool = False) -> int:
        if warmup or np.random.rand() < self.epsilon:
            return int(np.random.randint(action_dim))
        return int(greedy_action)

    def step(self) -> float:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return self.epsilon
