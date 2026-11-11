"""Negative economic/SOC reward with environment-managed FC cycle accounting.

SOC differences already include the formal battery charge/discharge efficiencies.
Battery hydrogen equivalence uses one curve-calibrated FC average efficiency,
read once from the fixed configuration and independent of the current FC power.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from numbers import Real
from pathlib import Path

from v2.economics import HYDROGEN_PRICE_CNY_PER_KG, build_formal_interval_ledger
from v2.models.battery_energy import BATTERY_CURRENT_ACCOUNTING_VERSION
from v2.models.fuel_cell_degradation import (
    FC_DEGRADATION_MODEL_VERSION, FC_EOL_VOLTAGE_LOSS_UV,
    FC_TRANSIENT_LOSS_UV_PER_DELTA_KW, FC_START_STOP_LOSS_UV_PER_CYCLE,
    FuelCellVoltageLoss,
    aggregate_start_cycle_count, formal_aggregate_fc_voltage_loss_step_uv,
)
from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG

REWARD_VERSION = "ddpg_economic_env_cycle_soc_v6"
FAILURE_REWARD = -20000.0
SOC_REFERENCE = 0.5
DEFAULT_CONFIG_PATH = Path(__file__).with_name("economic_reward_config.json")
FC_DEGRADATION_COST_FIELDS = (
    "fuel_cell_transient_cost_cny", "fuel_cell_low_load_cost_cny",
    "fuel_cell_high_load_cost_cny", "fuel_cell_start_stop_cost_cny",
)
REWARD_COST_FIELDS = (
    "h2_cost_cny", "c_bat_eq_cny", "battery_degradation_cost_cny",
) + FC_DEGRADATION_COST_FIELDS[:3]


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _check_config(config: dict) -> None:
    if config["reward_version"] != REWARD_VERSION:
        raise ValueError("Economic reward config version does not match the implementation")
    reference = _finite(config["c_ref"], "c_ref")
    failure = _finite(config["failure_reward"], "failure_reward")
    average_efficiency = _finite(config["eta_fc_avg"], "eta_fc_avg")
    soc_weight = _finite(config["soc_deviation_weight"], "soc_deviation_weight")
    if reference <= 0.0:
        raise ValueError("c_ref must be positive")
    if failure != FAILURE_REWARD:
        raise ValueError("Economic reward config failure_reward does not match the implementation")
    if not 0.0 < average_efficiency <= 1.0:
        raise ValueError("eta_fc_avg must lie in (0, 1]")
    if soc_weight < 0.0:
        raise ValueError("soc_deviation_weight must be nonnegative")


def load_reward_config(path=DEFAULT_CONFIG_PATH) -> dict:
    """Read fixed Train35 calibration explicitly; never calibrate at training time."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_config(config)
    return config


_FIXED_REWARD_CONFIG = load_reward_config()
ETA_FC_AVG = _FIXED_REWARD_CONFIG["eta_fc_avg"]
SOC_DEVIATION_WEIGHT = _FIXED_REWARD_CONFIG["soc_deviation_weight"]


def peak_fc_efficiency(curve) -> float:
    """Compute the formal PCHIP peak over positive-power source support knots."""
    powers = [float(power) for power in curve.power_kw if power > 0.0]
    if not powers:
        raise ValueError("The FC efficiency curve has no positive-power support")
    # PCHIP preserves local monotonicity, so no interior value exceeds the knots.
    peak = max(_finite(value, "positive-power FC efficiency") for value in curve.eta(powers))
    if not 0.0 < peak <= 1.0:
        raise ValueError("The computed positive-power FC efficiency peak must lie in (0, 1]")
    return peak


def battery_equivalent_cost(physics, before, after, fc_kw) -> dict:
    """Zhou Eq. (25), k=1: price signed cell energy using fixed FC efficiency."""
    power = _finite(fc_kw, "fc_kw")
    actual_efficiency = _finite(physics.fc_efficiency.eta(power), "actual FC efficiency")
    equivalent_efficiency = ETA_FC_AVG
    cell_energy_kwh = (
        _finite(before.soc, "SOC before") - _finite(after.soc, "SOC after")
    ) * physics.plant.battery_nominal_energy_kwh
    # The original SOC update already includes 0.95/0.95. Do not apply it again.
    mass_kg = cell_energy_kwh / (equivalent_efficiency * LHV_H2_KWH_PER_KG)
    cost_cny = mass_kg * HYDROGEN_PRICE_CNY_PER_KG
    return {
        "c_bat_eq_cny": _finite(cost_cny, "battery equivalent hydrogen cost"),
        "battery_equivalent_hydrogen_kg": _finite(mass_kg, "battery equivalent hydrogen mass"),
        "equivalent_fc_efficiency": equivalent_efficiency,
        "fc_efficiency_actual": actual_efficiency,
    }


def _fc_degradation_components(fc_loss, ledger) -> dict:
    """Allocate the existing EOL-capped cost without charging degradation twice."""
    cost_per_uv = ledger.fuel_cell_degradation_cost_cny / fc_loss.total_uv if fc_loss.total_uv else 0.0
    return dict(zip(FC_DEGRADATION_COST_FIELDS, (
        fc_loss.transient_uv * cost_per_uv,
        fc_loss.low_runtime_uv * cost_per_uv,
        fc_loss.high_runtime_uv * cost_per_uv,
        fc_loss.start_stop_uv * cost_per_uv,
    )))


def interval_economic_components(
    physics, before, after, fc_kw, ledger, *,
    fc_is_on: bool | None = None, fc_start_stop_cycles: int | None = None,
    physical_accounting_only: bool = False,
) -> dict:
    """Reuse the physical ledger, optionally adding the legacy DDPG reward terms."""
    if ledger.shore_cost_cny != 0.0:
        raise ValueError("The aggregate onboard reward does not include shore settlement")
    # Reuse the physical model's four raw losses for this executed 1-second step.
    # Allocate the existing ledger cost rather than charging degradation again.
    fc_loss = formal_aggregate_fc_voltage_loss_step_uv(
        before.previous_fc_kw, fc_kw, 1.0, physics.plant.fuel_cell_rated_total_kw,
        is_on=fc_kw > 0.0 if fc_is_on is None else fc_is_on,
        mapping=physics.fc_mapping,
        aggregate_start_stop_cycles=(
            aggregate_start_cycle_count(before.previous_fc_kw, fc_kw)
            if fc_start_stop_cycles is None else fc_start_stop_cycles
        ),
    )
    fc_components = _fc_degradation_components(fc_loss, ledger)
    values = {
        "hydrogen_mass_kg": ledger.h2_cost_cny / HYDROGEN_PRICE_CNY_PER_KG,
        "h2_cost_cny": ledger.h2_cost_cny,
        "fuel_cell_degradation_cost_cny": ledger.fuel_cell_degradation_cost_cny,
        "battery_degradation_cost_cny": ledger.battery_degradation_cost_cny,
        "shore_cost_cny": ledger.shore_cost_cny,
        "physical_economic_cost_cny": ledger.total_cost_cny,
        **fc_components,
    }
    if physical_accounting_only:
        values.update(
            c_bat_eq_cny=0.0,
            battery_equivalent_hydrogen_kg=0.0,
            equivalent_fc_efficiency=ETA_FC_AVG,
            fc_efficiency_actual=float(physics.fc_efficiency.eta(fc_kw)),
        )
    else:
        values.update(battery_equivalent_cost(physics, before, after, fc_kw))
    for name, value in values.items():
        _finite(value, name)
    total = math.fsum((
        values["h2_cost_cny"], values["fuel_cell_degradation_cost_cny"],
        values["battery_degradation_cost_cny"],
        values["c_bat_eq_cny"],
    ))
    if not math.isfinite(total) or total < 0.0:
        raise ValueError(
            f"Economic total is negative or nonfinite ({total!r}); "
            f"check cost signs and duplicate pricing; components={values!r}"
        )
    values["total_cost_cny"] = total
    values["reward_economic_cost_cny"] = math.fsum(values[name] for name in REWARD_COST_FIELDS)
    values["soc_deviation_penalty_cny"] = (
        0.0 if physical_accounting_only else SOC_DEVIATION_WEIGHT * (after.soc - SOC_REFERENCE) ** 2
    )
    values["reward_total_cost_cny"] = math.fsum((
        values["reward_economic_cost_cny"], values["soc_deviation_penalty_cny"],
    ))
    return values


def shutdown_economic_components(physics, state) -> tuple:
    """Settle one zero-duration shutdown; the environment calls this exactly once.

    Only FC transient loss and one complete start-stop cycle consume life. No
    physical interval, hydrogen consumption, battery throughput or SOC update runs.
    """
    fc_loss = FuelCellVoltageLoss(
        low_runtime_uv=0.0,
        high_runtime_uv=0.0,
        transient_uv=(
            FC_TRANSIENT_LOSS_UV_PER_DELTA_KW
            * abs(float(state.previous_fc_kw) * physics.fc_mapping.aggregate_to_reference_power_ratio)
        ),
        start_stop_uv=FC_START_STOP_LOSS_UV_PER_CYCLE,
    )
    closed = replace(
        state, previous_fc_kw=0.0, fc_loss_uv=state.fc_loss_uv + fc_loss.total_uv,
    )
    ledger = build_formal_interval_ledger(
        hydrogen_mass_kg=0.0,
        fuel_cell_cumulative_voltage_loss_before_uv=state.fc_loss_uv,
        fuel_cell_cumulative_voltage_loss_after_uv=closed.fc_loss_uv,
        fuel_cell_rated_kw=physics.plant.fuel_cell_rated_total_kw,
        battery_cumulative_weighted_ah_before=state.battery_weighted_ah,
        battery_cumulative_weighted_ah_after=closed.battery_weighted_ah,
        battery_capacity_kwh=physics.plant.battery_nominal_energy_kwh,
        battery_normalization=physics.battery_normalization,
        shore_energy=None,
    )
    fc_components = _fc_degradation_components(fc_loss, ledger)
    terminal_transient_cost = fc_components["fuel_cell_transient_cost_cny"]
    return closed, {
        "hydrogen_mass_kg": 0.0,
        "h2_cost_cny": 0.0,
        "fuel_cell_degradation_cost_cny": ledger.fuel_cell_degradation_cost_cny,
        "battery_degradation_cost_cny": 0.0,
        "shore_cost_cny": 0.0,
        "physical_economic_cost_cny": ledger.total_cost_cny,
        **fc_components,
        "c_bat_eq_cny": 0.0,
        "battery_equivalent_hydrogen_kg": 0.0,
        "equivalent_fc_efficiency": ETA_FC_AVG,
        "fc_efficiency_actual": 0.0,
        "total_cost_cny": ledger.total_cost_cny,
        "reward_economic_cost_cny": terminal_transient_cost,
        "soc_deviation_penalty_cny": 0.0,
        "reward_total_cost_cny": terminal_transient_cost,
    }


def evaluation_identity(model, current_identity: dict) -> dict:
    """Require models trained with the same supported reward identity."""
    if not isinstance(current_identity, dict) or current_identity.get("reward_version") != REWARD_VERSION:
        raise ValueError("Evaluation reward identity is missing or unsupported")
    saved = getattr(model, "_ddpg_reward_identity", None)
    if not isinstance(saved, dict) or saved.get("reward_version") != REWARD_VERSION:
        raise ValueError("Model reward identity is missing or unsupported; retrain with the current reward")
    if saved != current_identity:
        raise ValueError("Model and evaluation reward identities differ")
    return {
        "model_reward_identity": saved,
        "evaluation_reward_identity": current_identity,
        "legacy_policy_evaluated_with_current_reward": False,
    }


def economic_reward(reward_total_cost_cny) -> float:
    """Return the negative signed reward cost in CNY, without scaling or shaping."""
    return -_finite(reward_total_cost_cny, "reward total cost")


def reward_identity(config: dict) -> dict:
    """Expose the fixed reward/calibration identity without loading a model."""
    _check_config(config)
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "reward_version": REWARD_VERSION,
        "c_ref": float(config["c_ref"]),
        "c_ref_used_in_reward": False,
        "failure_reward": FAILURE_REWARD,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "formula": "-(C_H2+C_bat_eq+C_LB_deg+C_FC_transient+C_FC_low+C_FC_high+soc_deviation_weight*(SOC_next-0.5)^2)",
        "soc_deviation_weight": float(config["soc_deviation_weight"]),
        "soc_reference": SOC_REFERENCE,
        "fc_lifecycle": "environment-managed running includes zero-output idle; one complete start-stop cycle is settled only at shutdown",
        "terminal_shutdown": "zero duration; FC/LB outputs zero; SOC and battery Ah unchanged; terminal transient enters the final normal reward; cycle cost stays in physical accounting only",
        "fc_degradation_cost_allocation": "raw interval voltage-loss proportions of the original EOL-capped ledger cost; start/stop excluded only from reward",
        "fuel_cell_degradation_model": FC_DEGRADATION_MODEL_VERSION,
        "fuel_cell_eol_voltage_loss_uv": FC_EOL_VOLTAGE_LOSS_UV,
        "battery_current_accounting": BATTERY_CURRENT_ACCOUNTING_VERSION,
        "eta_fc_avg": float(config["eta_fc_avg"]),
        "battery_equivalence": "signed SOC cell-energy delta / fixed eta_fc_avg / LHV; Zhou Eq. (25), k=1; no extra battery/DC-DC efficiency",
    }
