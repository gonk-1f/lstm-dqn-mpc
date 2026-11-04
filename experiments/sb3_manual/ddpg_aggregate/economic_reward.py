"""Reference-normalized four-cost reward with signed battery hydrogen equivalence.

SOC differences already include the formal battery charge/discharge efficiencies.
FC-off battery supply is valued at the computed positive-power curve peak, as
explicitly selected for this reward version. No model or config is loaded here.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
from v2.models.battery_energy import BATTERY_CURRENT_ACCOUNTING_VERSION
from v2.models.fuel_cell_degradation import (
    FC_DEGRADATION_MODEL_VERSION, FC_EOL_VOLTAGE_LOSS_UV,
)
from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG

REWARD_VERSION = "ddpg_economic_normalized_tanh_v3"
FAILURE_REWARD = -10000.0
DEFAULT_CONFIG_PATH = Path(__file__).with_name("economic_reward_config.json")


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
    if reference <= 0.0:
        raise ValueError("c_ref must be positive")
    if failure != FAILURE_REWARD:
        raise ValueError("Economic reward config failure_reward does not match the implementation")


def load_reward_config(path=DEFAULT_CONFIG_PATH) -> dict:
    """Read fixed Train35 calibration explicitly; never calibrate at training time."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    _check_config(config)
    return config


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
    """Price the signed cell-energy change once, with the selected FC efficiency."""
    power = _finite(fc_kw, "fc_kw")
    actual_efficiency = _finite(physics.fc_efficiency.eta(power), "actual FC efficiency")
    equivalent_efficiency = (
        actual_efficiency if power > 0.0 else peak_fc_efficiency(physics.fc_efficiency)
    )
    if equivalent_efficiency <= 0.0:
        raise ValueError("Positive FC power must have positive formal-curve efficiency")
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


def interval_economic_components(physics, before, after, fc_kw, ledger) -> dict:
    """Reuse the executed physical ledger and add exactly one signed SOC value."""
    if ledger.shore_cost_cny != 0.0:
        raise ValueError("The aggregate onboard reward does not include shore settlement")
    values = {
        "hydrogen_mass_kg": ledger.h2_cost_cny / HYDROGEN_PRICE_CNY_PER_KG,
        "h2_cost_cny": ledger.h2_cost_cny,
        "fuel_cell_degradation_cost_cny": ledger.fuel_cell_degradation_cost_cny,
        "battery_degradation_cost_cny": ledger.battery_degradation_cost_cny,
        "shore_cost_cny": ledger.shore_cost_cny,
        "physical_economic_cost_cny": ledger.total_cost_cny,
        **battery_equivalent_cost(physics, before, after, fc_kw),
    }
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
    return values


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


def economic_reward(total_cost_cny, c_ref) -> float:
    """Return tanh(C_ref/C) in CNY; the exact zero-cost limit is one."""
    cost = _finite(total_cost_cny, "four-item economic cost")
    reference = _finite(c_ref, "c_ref")
    if cost < 0.0:
        raise ValueError("Negative economic cost: check cost signs and duplicate pricing; no abs/clip is allowed")
    if reference <= 0.0:
        raise ValueError("c_ref must be positive")
    return 1.0 if cost == 0.0 else math.tanh(reference / cost)


def reward_identity(config: dict) -> dict:
    """Expose the fixed reward/calibration identity without loading a model."""
    _check_config(config)
    encoded = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "reward_version": REWARD_VERSION,
        "c_ref": float(config["c_ref"]),
        "failure_reward": FAILURE_REWARD,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "formula": "tanh(C_ref/(C_H2+C_FC_deg+C_LB_deg+C_bat_eq)); C=0 -> 1",
        "fuel_cell_degradation_model": FC_DEGRADATION_MODEL_VERSION,
        "fuel_cell_eol_voltage_loss_uv": FC_EOL_VOLTAGE_LOSS_UV,
        "battery_current_accounting": BATTERY_CURRENT_ACCOUNTING_VERSION,
        "battery_equivalence": "signed SOC cell-energy delta; actual eta(P_fc) when on, computed positive-power curve peak when off; no extra battery/DC-DC efficiency",
    }
