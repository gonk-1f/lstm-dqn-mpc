"""Economic TD3 reward: physical CNY costs plus calibrated SOC shaping."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym

from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv

from .economic_action_wrappers import (
    ACTION_MAPPING_VERSION,
    OBSERVATION_VERSION,
    FeasibleIntervalActionWrapper,
    VoyageProgressObservationWrapper,
)
from .huber_td3 import CRITIC_LOSS_ID

DEFAULT_CONFIG_PATH = Path(__file__).with_name("economic_ems_config.json")
REWARD_COMPONENT_FIELDS = (
    "reward_h2_cost_cny",
    "reward_fc_degradation_cost_cny",
    "reward_battery_equivalent_hydrogen_cost_cny",
    "reward_battery_degradation_cost_cny",
    "reward_soc_tracking_penalty_cny",
    "reward_terminal_soc_penalty_cny",
    "reward_failure_penalty_cny",
)
FC_VARIABLE_DEGRADATION_FIELDS = (
    "fuel_cell_transient_cost_cny",
    "fuel_cell_low_load_cost_cny",
    "fuel_cell_high_load_cost_cny",
)


def load_economic_ems_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("reward_version") != "td3_economic_cost_soc_terminal_feasible_interval_v1":
        raise ValueError("Unsupported TD3 economic reward version")
    if config.get("reward_scale") != 0.001:
        raise ValueError("TD3 economic reward scale must be 0.001")
    for key in (
        "hydrogen_cost_weight",
        "fuel_cell_variable_degradation_weight",
        "battery_equivalent_hydrogen_weight",
        "battery_degradation_weight",
        "soc_step_weight_cny",
        "soc_terminal_weight_cny",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["failure_reward"] >= 0:
        raise ValueError("failure_reward must be negative")
    if not 0.0 <= config["soc_reference"] <= 1.0:
        raise ValueError("soc_reference must lie in [0, 1]")
    return config


class EconomicEMSTD3Reward(gym.RewardWrapper):
    """Expose a negative economic cost while preserving the original CNY ledger."""

    def __init__(self, env: gym.Env, config: dict | None = None):
        super().__init__(env)
        self.config = load_economic_ems_config() if config is None else dict(config)
        self.reward_scale = float(self.config["reward_scale"])
        self.soc_reference = float(self.config["soc_reference"])
        self.soc_step_weight = float(self.config["soc_step_weight_cny"])
        self.soc_terminal_weight = float(self.config["soc_terminal_weight_cny"])
        self.failure_reward = float(self.config["failure_reward"])
        base_identity = deepcopy(env.unwrapped.reward_identity)
        config_hash = hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest()
        self.reward_identity = {
            **base_identity,
            "base_reward_version": base_identity["reward_version"],
            "reward_version": self.config["reward_version"],
            "env_id": "AggregateShipTD3Economic-v0",
            "algorithm": "TD3",
            "critic_loss": CRITIC_LOSS_ID,
            "critic_huber_delta": 1.0,
            "reward_scale": self.reward_scale,
            "formula": self.config["reward_formula"],
            "economic_cost_weights": {
                "hydrogen": self.config["hydrogen_cost_weight"],
                "fuel_cell_variable_degradation": self.config["fuel_cell_variable_degradation_weight"],
                "battery_equivalent_hydrogen": self.config["battery_equivalent_hydrogen_weight"],
                "battery_degradation": self.config["battery_degradation_weight"],
            },
            "soc_reference": self.soc_reference,
            "soc_step_weight_cny": self.soc_step_weight,
            "soc_terminal_weight_cny": self.soc_terminal_weight,
            "failure_reward": self.failure_reward,
            "fc_degradation_reward_scope": list(FC_VARIABLE_DEGRADATION_FIELDS),
            "fc_start_stop_reward_scope": self.config["fc_start_stop_reward_scope"],
            "battery_equivalent_hydrogen_sign": self.config["battery_equivalent_hydrogen_sign"],
            "action_mapping": ACTION_MAPPING_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "training_selection": self.config["training_selection"],
            "economic_ems_config_sha256": config_hash,
            "physical_accounting_identity": deepcopy(base_identity),
        }
        self.raw_reward_sum = 0.0
        self.algorithm_reward_sum = 0.0
        self.component_totals = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
        self.initial_soc = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.raw_reward_sum = 0.0
        self.algorithm_reward_sum = 0.0
        self.component_totals = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
        self.initial_soc = float(self.env.unwrapped.physical.soc)
        return observation, {
            **info,
            "reward_version": self.reward_identity["reward_version"],
            "reward_scale": self.reward_scale,
            "soc_reference": self.soc_reference,
            "soc_step_weight_cny": self.soc_step_weight,
            "soc_terminal_weight_cny": self.soc_terminal_weight,
        }

    def step(self, action):
        observation, _, terminated, truncated, original = self.env.step(action)
        failed = bool(original.get("reward_is_failure_override", False))
        components = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
        if failed:
            components["reward_failure_penalty_cny"] = -self.failure_reward
            raw_reward = self.failure_reward
        else:
            weights = self.config
            components["reward_h2_cost_cny"] = (
                float(original["h2_cost_cny"]) * float(weights["hydrogen_cost_weight"])
            )
            components["reward_fc_degradation_cost_cny"] = (
                math.fsum(float(original[name]) for name in FC_VARIABLE_DEGRADATION_FIELDS)
                * float(weights["fuel_cell_variable_degradation_weight"])
            )
            components["reward_battery_equivalent_hydrogen_cost_cny"] = (
                float(original["c_bat_eq_cny"]) * float(weights["battery_equivalent_hydrogen_weight"])
            )
            components["reward_battery_degradation_cost_cny"] = (
                float(original["battery_degradation_cost_cny"])
                * float(weights["battery_degradation_weight"])
            )
            components["reward_soc_tracking_penalty_cny"] = (
                self.soc_step_weight * (float(original["soc"]) - self.soc_reference) ** 2
            )
            if terminated and not truncated and original.get("terminal_reason") == "completed":
                components["reward_terminal_soc_penalty_cny"] = (
                    self.soc_terminal_weight * (float(original["soc"]) - self.soc_reference) ** 2
                )
            raw_reward = -math.fsum(components.values())

        total_reward_cost = math.fsum(components.values())
        economic_reward_cost = math.fsum((
            components["reward_h2_cost_cny"],
            components["reward_fc_degradation_cost_cny"],
            components["reward_battery_equivalent_hydrogen_cost_cny"],
            components["reward_battery_degradation_cost_cny"],
        ))
        soc_penalty = (
            components["reward_soc_tracking_penalty_cny"]
            + components["reward_terminal_soc_penalty_cny"]
        )
        scaled_reward = raw_reward * self.reward_scale
        self.raw_reward_sum += raw_reward
        self.algorithm_reward_sum += scaled_reward
        for name, value in components.items():
            self.component_totals[name] += value

        info = {
            **original,
            **components,
            "reward_economic_cost_cny": economic_reward_cost,
            "soc_deviation_penalty_cny": soc_penalty,
            "reward_total_cost_cny": total_reward_cost,
            "raw_reward": raw_reward,
            "scaled_reward": scaled_reward,
            "reward_scale": self.reward_scale,
            "reward_version": self.reward_identity["reward_version"],
            "reward_is_failure_override": failed,
        }
        if "vessel_episode" in original:
            totals = self.component_totals
            episode_economic_cost = math.fsum((
                totals["reward_h2_cost_cny"],
                totals["reward_fc_degradation_cost_cny"],
                totals["reward_battery_equivalent_hydrogen_cost_cny"],
                totals["reward_battery_degradation_cost_cny"],
            ))
            episode_soc_penalty = (
                totals["reward_soc_tracking_penalty_cny"]
                + totals["reward_terminal_soc_penalty_cny"]
            )
            episode_reward_cost = math.fsum(totals.values())
            summary = {
                **original["vessel_episode"],
                **totals,
                "reward_sum": self.raw_reward_sum,
                "raw_reward_sum": self.raw_reward_sum,
                "algorithm_reward_sum": self.algorithm_reward_sum,
                "reward_scale": self.reward_scale,
                "reward_version": self.reward_identity["reward_version"],
                "algorithm_reward_version": self.reward_identity["reward_version"],
                "reward_economic_cost_cny": episode_economic_cost,
                "soc_deviation_penalty_cny": episode_soc_penalty,
                "reward_total_cost_cny": episode_reward_cost,
                "soc_initial": self.initial_soc,
                "soc_delta": float(self.env.unwrapped.physical.soc - self.initial_soc),
                "terminal_soc_target": self.soc_reference,
                "terminal_soc_error": float(self.env.unwrapped.physical.soc - self.soc_reference),
            }
            info["vessel_episode"] = summary
        return observation, float(scaled_reward), terminated, truncated, info


def wrap_economic_env(base_env: gym.Env) -> EconomicEMSTD3Reward:
    """Apply progress, feasible-interval action mapping and economic reward in order."""
    observation_env = VoyageProgressObservationWrapper(base_env)
    action_env = FeasibleIntervalActionWrapper(observation_env)
    return EconomicEMSTD3Reward(action_env)


def make_economic_env(
    data_root: str | Path,
    split: str = "train",
    fixed_train_id: str | None = None,
) -> EconomicEMSTD3Reward:
    """Zoo/evaluation factory. Train defaults to random Train35 sampling."""
    base_env = AggregateDDPGEnv(
        data_root=data_root,
        split=split,
        fixed_train_id=fixed_train_id,
    )
    return wrap_economic_env(base_env)


def assert_economic_model_identity(model, identity: dict) -> None:
    if identity.get("algorithm") != "TD3" or identity.get("reward_scale") != 0.001:
        raise ValueError("Expected the versioned TD3 economic reward at scale 0.001")
    if getattr(model, "_td3_reward_identity", None) != identity:
        raise ValueError("TD3 checkpoint reward identity does not match economic evaluation")
    if getattr(model, "reward_scale", None) != 0.001:
        raise ValueError("TD3 checkpoint reward_scale must be 0.001")
