from __future__ import annotations


def power_balance_residual(load_kw: float, fuel_cell_kw: float, battery_kw: float) -> float:
    return float(fuel_cell_kw + battery_kw - load_kw)


def split_balance_residual(left_supply_kw: float, right_supply_kw: float, left_load_kw: float, right_load_kw: float) -> float:
    return float((left_supply_kw - left_load_kw) - (right_supply_kw - right_load_kw))
