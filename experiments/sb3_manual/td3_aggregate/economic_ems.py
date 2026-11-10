"""Versioned economic TD3 rewards with a separate physical cost ledger."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym

from experiments.sb3_manual.ddpg_aggregate.env import AggregateDDPGEnv
from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
from v3.control import SOC_MAX, SOC_MIN

from ..ddpg_aggregate.economic_reward import ETA_FC_AVG, battery_equivalent_cost
from .callbacks import FIXED_TRAIN_ID, FIXED_TRAIN_STEPS
from .economic_action_wrappers import (
    ACTION_MAPPING_VERSION,
    OBSERVATION_VERSION,
    FeasibleIntervalActionWrapper,
    VoyageProgressObservationWrapper,
)
from .huber_td3 import CRITIC_LOSS_ID

DEFAULT_CONFIG_PATH = Path(__file__).with_name("economic_ems_config.json")
V6_CONFIG_PATH = Path(__file__).with_name("economic_ems_v6_config.json")
V7_CONFIG_PATH = Path(__file__).with_name("economic_ems_v7_config.json")
V8_CONFIG_PATH = Path(__file__).with_name("economic_ems_v8_config.json")
ECONOMIC_EMS_ENV_ID = "AggregateShipTD3Economic-v0"
ECONOMIC_EMS_REWARD_VERSION = "td3_economic_cost_bat_eq_stepwise_soc_factor_mu03_n32_fixed_parent005_v3"
ECONOMIC_EMS_OUTPUT_NAME = "td3_economic_ems_v31_n32_fixed_parent005_500ep"
V6_ENV_ID = "AggregateShipTD3EconomicV6-v0"
V6_REWARD_VERSION = "td3_v6_fixed_parent005_n32_prevfc"
V6_OUTPUT_NAME = "td3_v6_fixed_parent005_n32_prevfc_500ep"
V7_ENV_ID = "AggregateShipTD3EconomicV7-v0"
V7_REWARD_VERSION = "td3_v7_fixed_parent005_n32_prevfc"
V7_OUTPUT_NAME = "td3_v7_fixed_parent005_n32_prevfc_500ep"
V8_ENV_ID = "AggregateShipTD3EconomicV8-v0"
V8_REWARD_VERSION = "td3_v8_fixed_parent005_n32_delta60_soc_potential"
V8_OUTPUT_NAME = "td3_v8_fixed_parent005_n32_delta60_soc_potential_500ep"
ECONOMIC_REWARD_VERSIONS = {
    ECONOMIC_EMS_ENV_ID: ECONOMIC_EMS_REWARD_VERSION,
    V6_ENV_ID: V6_REWARD_VERSION,
    V7_ENV_ID: V7_REWARD_VERSION,
    V8_ENV_ID: V8_REWARD_VERSION,
}
N_STEP_RETURN = 32
REWARD_SCALE = 0.001
SOC_DWELL_TOLERANCE = 0.001
LEGACY_SOC_FIELDS = ("soc_deviation_penalty_cny", "soc_deviation_weight", "soc_reference")
BATTERY_EQUIVALENCE_STEP_FIELDS = (
    "battery_equivalent_hydrogen_step_kg",
    "battery_equivalent_hydrogen_step_cny",
    "soc_equivalence_factor",
)
REWARD_COMPONENT_FIELDS = (
    "reward_h2_cost_cny",
    "reward_fc_degradation_cost_cny",
    "reward_battery_equivalent_hydrogen_cost_cny",
    "reward_battery_degradation_cost_cny",
    "reward_failure_penalty_cny",
)
V6_REWARD_COMPONENT_FIELDS = (
    "reward_h2_cost_cny",
    "reward_fc_degradation_cost_cny",
    "reward_battery_degradation_cost_cny",
    "reward_soc_tracking_penalty_cny",
    "reward_failure_penalty_cny",
)
V7_REWARD_COMPONENT_FIELDS = V6_REWARD_COMPONENT_FIELDS
V8_REWARD_COMPONENT_FIELDS = (*V6_REWARD_COMPONENT_FIELDS, "reward_soc_potential_cost_cny")
FC_VARIABLE_DEGRADATION_FIELDS = (
    "fuel_cell_transient_cost_cny",
    "fuel_cell_low_load_cost_cny",
    "fuel_cell_high_load_cost_cny",
)


def load_economic_ems_config(
    path: str | Path = DEFAULT_CONFIG_PATH, *, reward_version: str = ECONOMIC_EMS_REWARD_VERSION,
) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("reward_version") != reward_version:
        raise ValueError("Unsupported TD3 economic reward version")
    if config.get("reward_scale") != REWARD_SCALE:
        raise ValueError("TD3 economic reward scale must be 0.001")
    for key in (
        "hydrogen_cost_weight",
        "fuel_cell_variable_degradation_weight",
        "battery_degradation_weight",
    ):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not math.isfinite(config["failure_reward"]) or config["failure_reward"] >= 0:
        raise ValueError("failure_reward must be negative")
    if reward_version == V8_REWARD_VERSION:
        if (
            config["soc_step_weight_cny"] != 1.0 or config["soc_reference"] != 0.5
            or config["soc_potential_weight_cny"] != 20000.0
        ):
            raise ValueError("Economic TD3 V8 requires SOC path weight 1, reference 0.5 and potential weight 20000")
    elif reward_version in (V6_REWARD_VERSION, V7_REWARD_VERSION):
        if config["soc_step_weight_cny"] != 5.0 or config["soc_reference"] != 0.5:
            raise ValueError("Economic TD3 requires SOC weight 5.0 and reference 0.5")
    elif not 0.0 <= config["soc_equivalence_mu"] <= 1.0:
        raise ValueError("soc_equivalence_mu must lie in [0, 1]")
    return config


class EconomicEMSTD3Reward(gym.RewardWrapper):
    """Expose a negative economic cost while preserving the original CNY ledger."""

    def __init__(self, env: gym.Env, config: dict | None = None):
        super().__init__(env)
        self.config = load_economic_ems_config() if config is None else dict(config)
        version = self.config["reward_version"]
        self.uses_soc_tracking = version in (V6_REWARD_VERSION, V7_REWARD_VERSION, V8_REWARD_VERSION)
        self.soc_potential_weight = self.config.get("soc_potential_weight_cny") if version == V8_REWARD_VERSION else None
        self.component_fields = {
            V6_REWARD_VERSION: V6_REWARD_COMPONENT_FIELDS,
            V7_REWARD_VERSION: V7_REWARD_COMPONENT_FIELDS,
            V8_REWARD_VERSION: V8_REWARD_COMPONENT_FIELDS,
        }.get(version, REWARD_COMPONENT_FIELDS)
        self.fc_reward_fields = (
            ("fuel_cell_transient_cost_cny",) if version == V7_REWARD_VERSION else FC_VARIABLE_DEGRADATION_FIELDS
        )
        self.reward_scale = float(self.config["reward_scale"])
        self.soc_equivalence_mu = None if self.uses_soc_tracking else float(self.config["soc_equivalence_mu"])
        self.eta_chg, self.eta_dis = env.unwrapped.physics.efficiency.require_calibrated()
        self.failure_reward = float(self.config["failure_reward"])
        base_identity = deepcopy(env.unwrapped.reward_identity)
        physical_identity = deepcopy(base_identity) if self.uses_soc_tracking else None
        for name in (*LEGACY_SOC_FIELDS, "formula", "battery_equivalence"):
            base_identity.pop(name, None)
        config_hash = hashlib.sha256(
            json.dumps(self.config, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        self.reward_identity = {
            **base_identity,
            "base_reward_version": base_identity["reward_version"],
            "reward_version": self.config["reward_version"],
            "env_id": ECONOMIC_EMS_ENV_ID,
            "algorithm": "TD3",
            "critic_loss": CRITIC_LOSS_ID,
            "critic_huber_delta": 1.0,
            "n_step_return": N_STEP_RETURN,
            "reward_scale": self.reward_scale,
            "formula": self.config["reward_formula"],
            "economic_cost_weights": {
                "hydrogen": self.config["hydrogen_cost_weight"],
                "fuel_cell_variable_degradation": self.config["fuel_cell_variable_degradation_weight"],
                "battery_equivalent_hydrogen": 1.0,
                "battery_degradation": self.config["battery_degradation_weight"],
            },
            "soc_equivalence_mu": self.soc_equivalence_mu,
            "soc_min": SOC_MIN,
            "soc_max": SOC_MAX,
            "eta_fc_avg": ETA_FC_AVG,
            "eta_chg": self.eta_chg,
            "eta_dis": self.eta_dis,
            "hydrogen_price_cny_per_kg": HYDROGEN_PRICE_CNY_PER_KG,
            "hydrogen_lhv_kwh_per_kg": LHV_H2_KWH_PER_KG,
            "interval_duration_s": 1.0,
            "failure_reward": self.failure_reward,
            "fc_degradation_reward_scope": list(self.fc_reward_fields),
            "fc_start_stop_reward_scope": self.config["fc_start_stop_reward_scope"],
            "battery_equivalent_hydrogen_sign": self.config["battery_equivalent_hydrogen_sign"],
            "battery_equivalence": self.config["battery_equivalence"],
            "action_mapping": ACTION_MAPPING_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "training_selection": self.config["training_selection"],
            "economic_ems_config_sha256": config_hash,
            "physical_accounting_identity": deepcopy(base_identity),
        }
        if self.uses_soc_tracking:
            self.reward_identity.pop("soc_equivalence_mu")
            self.reward_identity["economic_cost_weights"].pop("battery_equivalent_hydrogen")
            self.reward_identity.update(
                env_id={V6_REWARD_VERSION: V6_ENV_ID, V7_REWARD_VERSION: V7_ENV_ID, V8_REWARD_VERSION: V8_ENV_ID}[version],
                action_mapping=self.config["action_mapping"],
                soc_deviation_weight=self.config["soc_step_weight_cny"],
                soc_step_weight_cny=self.config["soc_step_weight_cny"],
                soc_reference=self.config["soc_reference"],
                physical_accounting_identity=physical_identity,
            )
        if self.soc_potential_weight is not None:
            self.reward_identity["soc_potential_weight_cny"] = self.soc_potential_weight
            self.reward_identity["soc_potential_reward_coefficient"] = self.soc_potential_weight * self.reward_scale
        self.initial_soc = None
        self._reset_accounting()

    def _reset_accounting(self):
        self.raw_reward_sum = 0.0
        self.algorithm_reward_sum = 0.0
        self.component_totals = dict.fromkeys(self.component_fields, 0.0)
        self.battery_equivalent_mass_kg = 0.0
        self.battery_equivalent_cost_cny = 0.0
        self.soc_equivalence_factor_sum = 0.0
        self.soc_upper_dwell_steps = 0
        self.soc_lower_dwell_steps = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._reset_accounting()
        base = self.env.unwrapped
        self.initial_soc = float(base.physical.soc)
        if info.get("split") == "train" and (
            info["sample_id"] != FIXED_TRAIN_ID
            or base.index != 0
            or not math.isclose(self.initial_soc, 0.6, abs_tol=1e-9)
            or base.physical.previous_fc_kw != 0.0
        ):
            raise RuntimeError("Economic TD3 must reset parent_005_seg01 at t=0, SOC=0.6 and FC=0 kW")
        reward_details = (
            {key: self.reward_identity[key] for key in ("soc_reference", "soc_deviation_weight", "soc_step_weight_cny")}
            if self.uses_soc_tracking else {"soc_equivalence_mu": self.soc_equivalence_mu}
        )
        return observation, {
            **{name: value for name, value in info.items() if name not in LEGACY_SOC_FIELDS},
            "reward_version": self.reward_identity["reward_version"],
            "reward_scale": self.reward_scale,
            **reward_details,
        }

    def _battery_equivalent_hydrogen(self, battery_kw: float, soc_before: float) -> tuple[float, float]:
        midpoint = (SOC_MIN + SOC_MAX) / 2.0
        factor = 1.0 - 2.0 * self.soc_equivalence_mu * (soc_before - midpoint) / (SOC_MAX - SOC_MIN)
        bus_energy_kwh = battery_kw / 3600.0
        cell_energy_kwh = (
            bus_energy_kwh / self.eta_dis if battery_kw >= 0.0 else bus_energy_kwh * self.eta_chg
        )
        mass_kg = factor * cell_energy_kwh / (ETA_FC_AVG * LHV_H2_KWH_PER_KG)
        return mass_kg, factor

    def voyage_diagnostics(self) -> dict:
        base = self.env.unwrapped
        steps = base.index
        return {
            "mean_soc_equivalence_factor": self.soc_equivalence_factor_sum / steps if steps and not self.uses_soc_tracking else None,
            "battery_equivalent_hydrogen_total_kg": self.battery_equivalent_mass_kg,
            "battery_equivalent_hydrogen_total_cny": self.battery_equivalent_cost_cny,
            "soc_upper_dwell_steps": self.soc_upper_dwell_steps,
            "soc_lower_dwell_steps": self.soc_lower_dwell_steps,
            "fc_transient_cost_per_step": (
                base.economic_totals["fuel_cell_transient_cost_cny"] / steps if steps else None
            ),
        }

    def step(self, action):
        base = self.env.unwrapped
        before = base.physical
        observation, _, terminated, truncated, original = self.env.step(action)
        failed = bool(original.get("reward_is_failure_override", False))
        components = dict.fromkeys(self.component_fields, 0.0)
        battery_mass_kg, battery_eq_cost, factor = 0.0, 0.0, None
        if failed:
            components["reward_failure_penalty_cny"] = -self.failure_reward
            raw_reward = self.failure_reward
        else:
            weights = self.config
            components["reward_h2_cost_cny"] = (
                float(original["h2_cost_cny"]) * float(weights["hydrogen_cost_weight"])
            )
            components["reward_fc_degradation_cost_cny"] = (
                math.fsum(float(original[name]) for name in self.fc_reward_fields)
                * float(weights["fuel_cell_variable_degradation_weight"])
            )
            if self.uses_soc_tracking:
                equivalent = battery_equivalent_cost(base.physics, before, base.physical, original["fc_actual_kw"])
                battery_mass_kg = equivalent["battery_equivalent_hydrogen_kg"]
                battery_eq_cost = equivalent["c_bat_eq_cny"]
                reference = self.config["soc_reference"]
                next_deviation_sq = (float(original["soc"]) - reference) ** 2
                components["reward_soc_tracking_penalty_cny"] = self.config["soc_step_weight_cny"] * next_deviation_sq
                if self.soc_potential_weight is not None:
                    before_deviation_sq = (float(before.soc) - reference) ** 2
                    components["reward_soc_potential_cost_cny"] = (
                        self.soc_potential_weight * (next_deviation_sq - before_deviation_sq)
                    )
            else:
                battery_mass_kg, factor = self._battery_equivalent_hydrogen(
                    float(original["battery_actual_kw"]), float(before.soc),
                )
                battery_eq_cost = HYDROGEN_PRICE_CNY_PER_KG * battery_mass_kg
                components["reward_battery_equivalent_hydrogen_cost_cny"] = battery_eq_cost
            components["reward_battery_degradation_cost_cny"] = (
                float(original["battery_degradation_cost_cny"])
                * float(weights["battery_degradation_weight"])
            )
            raw_reward = -math.fsum(components.values())

        if not math.isfinite(raw_reward):
            raise ValueError("Nonfinite economic TD3 reward")
        total_reward_cost = math.fsum(components.values())
        economic_reward_cost = math.fsum((
            components["reward_h2_cost_cny"],
            components["reward_fc_degradation_cost_cny"],
            components.get("reward_battery_equivalent_hydrogen_cost_cny", 0.0),
            components["reward_battery_degradation_cost_cny"],
        ))
        if not failed:
            self.battery_equivalent_mass_kg += battery_mass_kg
            self.battery_equivalent_cost_cny += battery_eq_cost
            if factor is not None:
                self.soc_equivalence_factor_sum += factor
            soc_after = float(original["soc"])
            self.soc_upper_dwell_steps += int(soc_after >= SOC_MAX - SOC_DWELL_TOLERANCE)
            self.soc_lower_dwell_steps += int(soc_after <= SOC_MIN + SOC_DWELL_TOLERANCE)
        scaled_reward = raw_reward * self.reward_scale
        self.raw_reward_sum += raw_reward
        self.algorithm_reward_sum += scaled_reward
        for name, value in components.items():
            self.component_totals[name] += value

        info = {
            **original,
            **components,
            "battery_equivalent_hydrogen_step_kg": battery_mass_kg,
            "battery_equivalent_hydrogen_step_cny": battery_eq_cost,
            "soc_equivalence_factor": factor,
            "c_bat_eq_cny": battery_eq_cost,
            "battery_equivalent_hydrogen_kg": battery_mass_kg,
            "total_cost_cny": original["physical_economic_cost_cny"],
            "reward_economic_cost_cny": economic_reward_cost,
            "reward_total_cost_cny": total_reward_cost,
            "raw_reward": raw_reward,
            "scaled_reward": scaled_reward,
            "reward_scale": self.reward_scale,
            "reward_version": self.reward_identity["reward_version"],
            "reward_is_failure_override": failed,
        }
        if self.uses_soc_tracking:
            info["soc_deviation_penalty_cny"] = components["reward_soc_tracking_penalty_cny"]
        else:
            info.pop("soc_deviation_penalty_cny", None)
        if "vessel_episode" in original:
            totals = self.component_totals
            episode_economic_cost = math.fsum((
                totals["reward_h2_cost_cny"],
                totals["reward_fc_degradation_cost_cny"],
                totals.get("reward_battery_equivalent_hydrogen_cost_cny", 0.0),
                totals["reward_battery_degradation_cost_cny"],
            ))
            episode_reward_cost = math.fsum(totals.values())
            base_summary = original["vessel_episode"]
            summary = {
                **{name: value for name, value in base_summary.items() if name not in LEGACY_SOC_FIELDS},
                **totals,
                **self.voyage_diagnostics(),
                "reward_sum": self.raw_reward_sum,
                "raw_reward_sum": self.raw_reward_sum,
                "algorithm_reward_sum": self.algorithm_reward_sum,
                "reward_scale": self.reward_scale,
                "reward_version": self.reward_identity["reward_version"],
                "algorithm_reward_version": self.reward_identity["reward_version"],
                "reward_economic_cost_cny": episode_economic_cost,
                "c_bat_eq_cny": self.battery_equivalent_cost_cny,
                "battery_equivalent_hydrogen_kg": self.battery_equivalent_mass_kg,
                "total_cost_cny": base_summary["physical_economic_cost_cny"],
                "reward_total_cost_cny": episode_reward_cost,
                "soc_initial": self.initial_soc,
                "soc_delta": float(self.env.unwrapped.physical.soc - self.initial_soc),
            }
            if self.uses_soc_tracking:
                summary["soc_deviation_penalty_cny"] = totals["reward_soc_tracking_penalty_cny"]
            info["vessel_episode"] = summary
        return observation, float(scaled_reward), terminated, truncated, info


def wrap_economic_env(base_env: gym.Env) -> EconomicEMSTD3Reward:
    """Apply progress, feasible-interval action mapping and economic reward in order."""
    base_env.physical_accounting_only = True
    observation_env = VoyageProgressObservationWrapper(base_env)
    action_env = FeasibleIntervalActionWrapper(observation_env)
    return EconomicEMSTD3Reward(action_env)


def _wrap_soc_tracking_env(base_env: gym.Env, config_path: Path, reward_version: str) -> EconomicEMSTD3Reward:
    config = load_economic_ems_config(config_path, reward_version=reward_version)
    base_env.physical_accounting_only = True
    observation_env = VoyageProgressObservationWrapper(base_env)
    action_env = FeasibleIntervalActionWrapper(observation_env, action_mapping=config["action_mapping"])
    return EconomicEMSTD3Reward(action_env, config=config)


def wrap_v6_env(base_env: gym.Env) -> EconomicEMSTD3Reward:
    return _wrap_soc_tracking_env(base_env, V6_CONFIG_PATH, V6_REWARD_VERSION)


def wrap_v7_env(base_env: gym.Env) -> EconomicEMSTD3Reward:
    return _wrap_soc_tracking_env(base_env, V7_CONFIG_PATH, V7_REWARD_VERSION)


def wrap_v8_env(base_env: gym.Env) -> EconomicEMSTD3Reward:
    return _wrap_soc_tracking_env(base_env, V8_CONFIG_PATH, V8_REWARD_VERSION)


def make_economic_env(
    data_root: str | Path,
    split: str = "train",
    fixed_train_id: str | None = None,
    *,
    wrapper=wrap_economic_env,
) -> EconomicEMSTD3Reward:
    """Build the fixed Train voyage or the separate ordered Validation8."""
    if fixed_train_id not in (None, FIXED_TRAIN_ID):
        raise ValueError(f"The fixed Train diagnostic requires {FIXED_TRAIN_ID}")
    if split == "train":
        fixed_train_id = FIXED_TRAIN_ID
    base_env = AggregateDDPGEnv(
        data_root=data_root,
        split=split,
        fixed_train_id=fixed_train_id,
    )
    if fixed_train_id is not None:
        voyage = next(episode for episode in base_env.episodes if episode.sample_id == FIXED_TRAIN_ID)
        if len(voyage.load_kw) != FIXED_TRAIN_STEPS:
            base_env.close()
            raise ValueError(f"{FIXED_TRAIN_ID} must contain {FIXED_TRAIN_STEPS} one-second steps")
    return wrapper(base_env)


def make_v6_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> EconomicEMSTD3Reward:
    return make_economic_env(data_root, split=split, fixed_train_id=fixed_train_id, wrapper=wrap_v6_env)


def make_v7_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> EconomicEMSTD3Reward:
    return make_economic_env(data_root, split=split, fixed_train_id=fixed_train_id, wrapper=wrap_v7_env)


def make_v8_env(
    data_root: str | Path, split: str = "train", fixed_train_id: str | None = None,
) -> EconomicEMSTD3Reward:
    return make_economic_env(data_root, split=split, fixed_train_id=fixed_train_id, wrapper=wrap_v8_env)


def assert_economic_model_identity(model, identity: dict) -> None:
    if (
        identity.get("algorithm") != "TD3"
        or identity.get("env_id") not in ECONOMIC_REWARD_VERSIONS
        or identity.get("reward_version") != ECONOMIC_REWARD_VERSIONS[identity["env_id"]]
        or identity.get("n_step_return") != N_STEP_RETURN
        or identity.get("reward_scale") != REWARD_SCALE
    ):
        raise ValueError("Expected the versioned TD3 economic reward at scale 0.001")
    if getattr(model, "_td3_reward_identity", None) != identity:
        raise ValueError("TD3 checkpoint reward identity does not match economic evaluation")
    if getattr(model, "reward_scale", None) != REWARD_SCALE:
        raise ValueError("TD3 checkpoint reward_scale must be 0.001")
    if getattr(model, "n_steps", None) != N_STEP_RETURN:
        raise ValueError("TD3 checkpoint must use a 32-step return")
