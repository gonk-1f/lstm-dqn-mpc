"""Independent hydrogen/quadratic-FC/SOC experiment over unchanged ship physics.

The inner environment continues recording its complete original economic ledger.
Only the reward exposed to native TD3 and the matching reward summaries change.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv

from .reward_wrapper import REWARD_SCALE, ScaledEconomicReward

SIMPLE_REWARD_VERSION = "td3_hydrogen_quadratic_fc_soc_scaled_1e4_v1"
CONTROL_REWARD_VERSION = "td3_quadratic_fc_soc_scaled_1e4_v1"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("simple_reward_config.json")
REWARD_COMPONENT_FIELDS = (
    "reward_h2_cost_cny", "reward_fc_variation_penalty",
    "reward_soc_penalty", "failure_reward_total",
)


def load_simple_reward_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read the fixed historical coefficients; never calibrate during training."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SimpleHydrogenReward(ScaledEconomicReward):
    """Apply the confirmed quadratic reward, preserving the existing 1e-4 scale."""

    def __init__(self, env, include_hydrogen: bool = True):
        super().__init__(env)
        self.include_hydrogen = include_hydrogen
        config = load_simple_reward_config()
        self.lambda_fc = float(config["lambda_fc"])
        self.lambda_soc = float(config["lambda_soc"])
        self.soc_reference = float(config["soc_reference"])
        self.failure_reward = float(config["failure_reward"])
        self.reward_identity = {
            **self.reward_identity,
            "reward_version": SIMPLE_REWARD_VERSION,
            "env_id": "AggregateShipTD3Simple-v0",
            "formula": config["formula"],
            "fc_variation_weight": self.lambda_fc,
            "fc_variation_power": 2,
            "soc_deviation_weight": self.lambda_soc,
            "soc_reference": self.soc_reference,
            "failure_reward": self.failure_reward,
            "simple_reward_config_sha256": hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest(),
            "physical_accounting_identity": deepcopy(env.unwrapped.reward_identity),
            "training_cost_scope": "hydrogen plus quadratic actual FC variation and SOC deviation only",
            "physical_accounting_scope": config["accounting_scope"],
            "terminal_reward_rule": config["terminal_rule"],
        }
        if not self.include_hydrogen:
            # Keep the default three-term identity byte-for-byte compatible
            # with saved checkpoints; only the new control experiment differs.
            self.reward_identity.update(
                reward_version=CONTROL_REWARD_VERSION,
                env_id="AggregateShipTD3Control-v0",
                include_hydrogen=False,
                formula="-(lambda_fc*(P_FC_actual-P_FC_previous_actual)^2 + lambda_soc*(SOC_next-0.5)^2)",
                training_cost_scope="quadratic actual FC variation and SOC deviation only; no hydrogen or degradation costs",
                physical_accounting_scope="Actual hydrogen, FC/battery degradation and signed battery-equivalent hydrogen remain in the complete economic ledger and are excluded from the control training reward.",
            )
        self.component_totals = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.component_totals = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
        self.initial_soc = float(self.env.unwrapped.physical.soc)
        return observation, {
            **info, "reward_version": self.reward_identity["reward_version"],
            "fc_variation_weight": self.lambda_fc, "fc_variation_power": 2,
            "soc_deviation_weight": self.lambda_soc,
            "soc_reference": self.soc_reference,
        }

    def step(self, action):
        observation, _, terminated, truncated, original = self.env.step(action)
        failed = bool(original.get("reward_is_failure_override", False))
        if failed:
            # An infeasible request executes no load interval: no fabricated
            # hydrogen, FC change, or additional SOC penalty enters its reward.
            components = dict.fromkeys(REWARD_COMPONENT_FIELDS, 0.0)
            components["failure_reward_total"] = self.failure_reward
        else:
            components = {
                "reward_h2_cost_cny": float(original["h2_cost_cny"]) if self.include_hydrogen else 0.0,
                "reward_fc_variation_penalty": self.lambda_fc * float(original["delta_fc_kw"]) ** 2,
                "reward_soc_penalty": self.lambda_soc * (float(original["soc"]) - self.soc_reference) ** 2,
                "failure_reward_total": 0.0,
            }
        reward_cost = math.fsum(components[name] for name in REWARD_COMPONENT_FIELDS[:3])
        raw_reward = -reward_cost + components["failure_reward_total"]
        scaled_reward = self.reward(raw_reward)
        self.raw_reward_sum += raw_reward
        self.algorithm_reward_sum += scaled_reward
        for name, value in components.items():
            self.component_totals[name] += value
        info = {
            **original,
            **components,
            "reward_economic_cost_cny": components["reward_h2_cost_cny"],
            "soc_deviation_penalty_cny": components["reward_soc_penalty"],
            "fc_ramp_penalty": components["reward_fc_variation_penalty"],
            "reward_total_cost_cny": reward_cost,
            "raw_reward": raw_reward, "scaled_reward": scaled_reward,
            "reward_scale": REWARD_SCALE, "reward_version": self.reward_identity["reward_version"],
        }
        if "vessel_episode" in original:
            totals = self.component_totals
            info["vessel_episode"] = {
                **original["vessel_episode"],
                **totals,
                "reward_sum": self.raw_reward_sum,
                "raw_reward_sum": self.raw_reward_sum,
                "algorithm_reward_sum": self.algorithm_reward_sum,
                "reward_economic_cost_cny": totals["reward_h2_cost_cny"],
                "soc_deviation_penalty_cny": totals["reward_soc_penalty"],
                "fc_ramp_penalty": totals["reward_fc_variation_penalty"],
                "reward_total_cost_cny": math.fsum(totals[name] for name in REWARD_COMPONENT_FIELDS[:3]),
                "reward_scale": REWARD_SCALE, "reward_version": self.reward_identity["reward_version"],
                "algorithm_reward_version": self.reward_identity["reward_version"],
                "soc_initial": self.initial_soc,
                "soc_delta": float(self.env.unwrapped.physical.soc - self.initial_soc),
            }
        return observation, scaled_reward, terminated, truncated, info


def make_simple_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> SimpleHydrogenReward:
    """Reuse the exact same physical environment and immutable voyage selection."""
    return SimpleHydrogenReward(AggregateDDPGEnv(
        data_root=data_root, split=split, fixed_train_id=fixed_train_id,
    ))


def make_control_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> SimpleHydrogenReward:
    """Run only the FC/SOC penalties while retaining the actual economic ledger."""
    return SimpleHydrogenReward(AggregateDDPGEnv(
        data_root=data_root, split=split, fixed_train_id=fixed_train_id,
    ), include_hydrogen=False)
