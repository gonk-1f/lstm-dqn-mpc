from __future__ import annotations


def battery_degradation_cost(power_kw: float, coefficient: float = 0.02) -> float:
    """Quadratic proxy for battery usage and equivalent aging."""
    return float(coefficient * (power_kw ** 2))


def fuel_cell_operating_cost(power_kw: float, coefficient: float = 0.08) -> float:
    """Quadratic proxy for hydrogen consumption and stack burden."""
    return float(coefficient * (power_kw ** 2))


def ramp_penalty(current_kw: float, previous_kw: float, coefficient: float = 0.3) -> float:
    delta = current_kw - previous_kw
    return float(coefficient * (delta ** 2))


def soc_tracking_cost(current_soc: float, target_soc: float, coefficient: float = 5.0) -> float:
    return float(coefficient * ((current_soc - target_soc) ** 2))
