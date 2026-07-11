from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvStepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class ShipEnvBase:
    """Base class for ship microgrid environments.

    Later implementations should provide:
    - observation builder
    - action mapping
    - constraint checks
    - state transition
    - reward calculation
    """

    def reset(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def describe_state_space(self) -> dict:
        raise NotImplementedError

    def describe_action_space(self) -> dict:
        raise NotImplementedError

    def describe_reward(self) -> dict:
        raise NotImplementedError
