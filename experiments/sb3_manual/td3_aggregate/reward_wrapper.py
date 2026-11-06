"""Scale algorithm rewards without changing the physical/economic environment."""
from __future__ import annotations

from copy import deepcopy
from math import isfinite
from pathlib import Path

import gymnasium as gym

from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv

REWARD_SCALE = 0.0001
TD3_REWARD_VERSION = "td3_economic_scaled_1e4_v1"


class ScaledEconomicReward(gym.RewardWrapper):
    """Keep raw CNY and reward_sum intact; pass raw_reward / 10000 to TD3."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        base_identity = deepcopy(env.unwrapped.reward_identity)
        self.reward_identity = {
            **base_identity,
            "base_reward_version": base_identity["reward_version"],
            "reward_version": TD3_REWARD_VERSION,
            "algorithm": "TD3",
            "critic_loss": "sb3_native_mse",
            "reward_scale": REWARD_SCALE,
        }
        self.raw_reward_sum = 0.0
        self.algorithm_reward_sum = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.raw_reward_sum = 0.0
        self.algorithm_reward_sum = 0.0
        return observation, {**info, "reward_scale": REWARD_SCALE}

    def reward(self, reward):
        raw_reward = float(reward)
        if not isfinite(raw_reward):
            raise ValueError("Non-finite raw economic reward")
        return raw_reward / 10000.0

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        scaled_reward = self.reward(raw_reward)
        self.raw_reward_sum += float(raw_reward)
        self.algorithm_reward_sum += scaled_reward
        info = {
            **info,
            "raw_reward": float(raw_reward),
            "scaled_reward": scaled_reward,
            "reward_scale": REWARD_SCALE,
        }
        if "vessel_episode" in info:
            info["vessel_episode"] = {
                **info["vessel_episode"],
                "raw_reward_sum": self.raw_reward_sum,
                "algorithm_reward_sum": self.algorithm_reward_sum,
                "reward_scale": REWARD_SCALE,
                "algorithm_reward_version": TD3_REWARD_VERSION,
            }
        return observation, scaled_reward, terminated, truncated, info


def make_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> ScaledEconomicReward:
    """The same environment factory is used by Zoo and independent evaluation."""
    return ScaledEconomicReward(AggregateDDPGEnv(
        data_root=data_root, split=split, fixed_train_id=fixed_train_id,
    ))


def assert_model_identity(model, identity: dict) -> None:
    """Reject DDPG/legacy/differently-scaled checkpoints, even with equal spaces."""
    if identity.get("algorithm") != "TD3" or identity.get("reward_scale") != REWARD_SCALE:
        raise ValueError("Expected the TD3 scaled economic reward identity")
    if getattr(model, "_td3_reward_identity", None) != identity:
        raise ValueError("TD3 checkpoint reward identity does not match this environment")
    if getattr(model, "reward_scale", None) != REWARD_SCALE:
        raise ValueError("TD3 checkpoint reward_scale must be 0.0001")
