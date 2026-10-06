"""Frozen H1-H4 history-state DQN study profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ..config import TimeScaleConfig
from ..economics import (
    RewardScaleCalibration,
    _validate_reward_scale,
    scale_learning_reward,
)
from ..envs.multirate_weight_env import MacroTransition
from .dqn import DqnTrainingConfig


RAW_REWARD_SCALING_IDENTITY = "raw_learning_reward_v1"
_HISTORY_PROFILE_IDS = ("H1", "H2", "H3", "H4")
M10_EXPERIMENT_ID = "M10"
_PROFILE_MATRIX = {
    "H1": ("raw", 1.0e-4, 200_000, 150_000, 5_000, 5),
    "H2": ("scaled", 1.0e-4, 200_000, 150_000, 5_000, 5),
    "H3": ("scaled", 3.0e-4, 200_000, 150_000, 5_000, 5),
    "H4": ("scaled", 1.0e-3, 200_000, 150_000, 5_000, 5),
    M10_EXPERIMENT_ID: ("scaled", 1.0e-3, 100_000, 75_000, 2_500, 10),
}


@dataclass(frozen=True)
class DqnExperimentProfile:
    experiment_id: str
    reward_mode: str
    learning_rate: float
    reward_scaling_identity: str
    replay_capacity: int = 200_000
    epsilon_decay_steps: int = 150_000
    warmup_steps: int = 5_000
    dqn_switch_steps: int = 5

    def __post_init__(self) -> None:
        if self.experiment_id not in _PROFILE_MATRIX:
            raise ValueError("experiment_id must be one of H1, H2, H3, H4, M10")
        (
            expected_mode,
            expected_rate,
            expected_capacity,
            expected_decay,
            expected_warmup,
            expected_switch,
        ) = (
            _PROFILE_MATRIX[self.experiment_id]
        )
        if self.reward_mode != expected_mode:
            raise ValueError("reward_mode differs from the frozen experiment matrix")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate != expected_rate
        ):
            raise ValueError("learning_rate differs from the frozen experiment matrix")
        if self.replay_capacity != expected_capacity:
            raise ValueError("replay_capacity differs from the frozen experiment matrix")
        if self.epsilon_decay_steps != expected_decay:
            raise ValueError("epsilon_decay_steps differs from the frozen experiment matrix")
        if self.warmup_steps != expected_warmup:
            raise ValueError("warmup_steps differs from the frozen experiment matrix")
        if self.dqn_switch_steps != expected_switch:
            raise ValueError("dqn_switch_steps differs from the frozen experiment matrix")
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
            replay_capacity=self.replay_capacity,
            warmup_steps=self.warmup_steps,
            epsilon_decay_steps=self.epsilon_decay_steps,
            rounds=rounds,
            experiment_id=self.experiment_id,
            reward_mode=self.reward_mode,
            reward_scaling_identity=self.reward_scaling_identity,
        )

    @property
    def timescale(self) -> TimeScaleConfig:
        return TimeScaleConfig(30.0, 5, self.dqn_switch_steps)

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
            replay_capacity=replay_capacity,
            epsilon_decay_steps=epsilon_decay_steps,
            warmup_steps=warmup_steps,
            dqn_switch_steps=dqn_switch_steps,
        )
        for experiment_id in _HISTORY_PROFILE_IDS
        for (
            reward_mode,
            learning_rate,
            replay_capacity,
            epsilon_decay_steps,
            warmup_steps,
            dqn_switch_steps,
        ) in (
            _PROFILE_MATRIX[experiment_id],
        )
    }


def history_study_profile(
    experiment_id: str,
    calibration: RewardScaleCalibration | None,
) -> DqnExperimentProfile:
    if experiment_id not in _PROFILE_MATRIX:
        raise ValueError("experiment_id must be one of H1, H2, H3, H4")
    (
        reward_mode,
        learning_rate,
        replay_capacity,
        epsilon_decay_steps,
        warmup_steps,
        dqn_switch_steps,
    ) = (
        _PROFILE_MATRIX[experiment_id]
    )
    if reward_mode == "raw":
        if calibration is not None:
            raise ValueError("raw H1 must not receive a reward calibration")
        identity = RAW_REWARD_SCALING_IDENTITY
    else:
        if calibration is None:
            raise ValueError("scaled H2-H4 require a reward calibration")
        identity = _validate_reward_scale(calibration).digest
    return DqnExperimentProfile(
        experiment_id=experiment_id,
        reward_mode=reward_mode,
        learning_rate=learning_rate,
        reward_scaling_identity=identity,
        replay_capacity=replay_capacity,
        epsilon_decay_steps=epsilon_decay_steps,
        warmup_steps=warmup_steps,
        dqn_switch_steps=dqn_switch_steps,
    )


def m10_ablation_profile(
    calibration: RewardScaleCalibration,
) -> DqnExperimentProfile:
    checked = _validate_reward_scale(calibration)
    (
        reward_mode,
        learning_rate,
        replay_capacity,
        epsilon_decay_steps,
        warmup_steps,
        dqn_switch_steps,
    ) = _PROFILE_MATRIX[M10_EXPERIMENT_ID]
    return DqnExperimentProfile(
        experiment_id=M10_EXPERIMENT_ID,
        reward_mode=reward_mode,
        learning_rate=learning_rate,
        reward_scaling_identity=checked.digest,
        replay_capacity=replay_capacity,
        epsilon_decay_steps=epsilon_decay_steps,
        warmup_steps=warmup_steps,
        dqn_switch_steps=dqn_switch_steps,
    )


__all__ = [
    "DqnExperimentProfile",
    "M10_EXPERIMENT_ID",
    "RAW_REWARD_SCALING_IDENTITY",
    "history_study_profiles",
    "history_study_profile",
    "m10_ablation_profile",
]
