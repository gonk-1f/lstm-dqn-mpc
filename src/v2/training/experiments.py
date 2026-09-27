"""Frozen H1-H4 history-state DQN study profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..economics import (
    RewardScaleCalibration,
    _validate_reward_scale,
    scale_learning_reward,
)
from ..envs.multirate_weight_env import MacroTransition
from .dqn import DqnTrainingConfig


RAW_REWARD_SCALING_IDENTITY = "raw_learning_reward_v1"
_PROFILE_MATRIX = {
    "H1": ("raw", 1.0e-4),
    "H2": ("scaled", 1.0e-4),
    "H3": ("scaled", 3.0e-4),
    "H4": ("scaled", 1.0e-3),
}


@dataclass(frozen=True)
class DqnExperimentProfile:
    experiment_id: str
    reward_mode: str
    learning_rate: float
    reward_scaling_identity: str

    def __post_init__(self) -> None:
        if self.experiment_id not in _PROFILE_MATRIX:
            raise ValueError("experiment_id must be one of H1, H2, H3, H4")
        expected_mode, expected_rate = _PROFILE_MATRIX[self.experiment_id]
        if self.reward_mode != expected_mode:
            raise ValueError("reward_mode differs from the frozen experiment matrix")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate != expected_rate
        ):
            raise ValueError("learning_rate differs from the frozen experiment matrix")
        if (
            type(self.reward_scaling_identity) is not str
            or not self.reward_scaling_identity
        ):
            raise ValueError("reward_scaling_identity must be a nonempty exact string")
        if (
            self.reward_mode == "raw"
            and self.reward_scaling_identity != RAW_REWARD_SCALING_IDENTITY
        ):
            raise ValueError("raw profile must use the raw reward identity")

    def dqn_config(self, *, rounds: int) -> DqnTrainingConfig:
        if type(rounds) is not int or rounds <= 0:
            raise ValueError("rounds must be a positive exact integer")
        return replace(
            DqnTrainingConfig.formal_baseline(),
            learning_rate=self.learning_rate,
            rounds=rounds,
            experiment_id=self.experiment_id,
            reward_mode=self.reward_mode,
            reward_scaling_identity=self.reward_scaling_identity,
        )

    def replay_reward(
        self,
        transition: MacroTransition,
        calibration: RewardScaleCalibration | None,
    ) -> float:
        if type(transition) is not MacroTransition:
            raise TypeError("transition must be an exact MacroTransition")
        if self.reward_mode == "raw":
            if calibration is not None:
                raise ValueError("raw H1 must not receive a reward calibration")
            return transition.learning_reward
        if calibration is None:
            raise ValueError("scaled profiles require a reward calibration")
        checked = _validate_reward_scale(calibration)
        if checked.digest != self.reward_scaling_identity:
            raise ValueError("reward calibration differs from the profile identity")
        return scale_learning_reward(
            transition.learning_reward,
            calibration=checked,
        )


def history_study_profiles(
    calibration: RewardScaleCalibration,
) -> dict[str, DqnExperimentProfile]:
    checked = _validate_reward_scale(calibration)
    return {
        experiment_id: DqnExperimentProfile(
            experiment_id=experiment_id,
            reward_mode=reward_mode,
            learning_rate=learning_rate,
            reward_scaling_identity=(
                RAW_REWARD_SCALING_IDENTITY
                if reward_mode == "raw"
                else checked.digest
            ),
        )
        for experiment_id, (reward_mode, learning_rate) in _PROFILE_MATRIX.items()
    }


__all__ = [
    "DqnExperimentProfile",
    "RAW_REWARD_SCALING_IDENTITY",
    "history_study_profiles",
]
