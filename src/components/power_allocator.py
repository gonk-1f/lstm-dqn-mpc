from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SideFeasibleRange:
    min_kw: float
    max_kw: float


@dataclass
class FuelCellAllocationResult:
    left_kw: float
    right_kw: float
    requested_total_kw: float
    feasible_total_kw: float
    total_error_kw: float
    fully_symmetric: bool
    constraint_adjusted: bool
    correction_kw: float


@dataclass
class BatterySoftBalanceConfig:
    soc_balance_deadband: float = 0.005
    soc_balance_bias_gain_kw_per_soc: float = 350.0
    max_balance_bias_kw: float = 8.0
    max_balance_bias_ratio: float = 0.08
    balance_bias_smoothing_alpha: float = 0.25
    balance_bias_rate_limit_kw_per_step: float = 2.0
    batt_ramp_kw_per_step: float = 360.0
    dominance_deadband_kw: float = 0.5


@dataclass
class BatteryAllocationResult:
    left_kw: float
    right_kw: float
    requested_total_kw: float
    feasible_total_kw: float
    total_error_kw: float
    base_left_kw: float
    base_right_kw: float
    desired_balance_bias_kw: float
    smoothed_balance_bias_kw: float
    final_balance_bias_kw: float
    previous_balance_bias_kw: float
    soc_diff: float
    soc_deadband_active: bool
    bias_limited: bool
    bias_rate_limited: bool
    constraint_adjusted: bool
    compensation_kw: float


def _feasible_range(power_min_kw: float, power_max_kw: float, previous_kw: float, ramp_kw: float) -> SideFeasibleRange:
    lower = max(float(power_min_kw), float(previous_kw) - float(ramp_kw))
    upper = min(float(power_max_kw), float(previous_kw) + float(ramp_kw))
    return SideFeasibleRange(min_kw=lower, max_kw=upper)


def _allocate_two_sides_with_total(
    total_cmd_kw: float,
    desired_left_kw: float,
    left_range: SideFeasibleRange,
    right_range: SideFeasibleRange,
) -> tuple[float, float, float, float]:
    requested_total = float(total_cmd_kw)
    min_total = left_range.min_kw + right_range.min_kw
    max_total = left_range.max_kw + right_range.max_kw
    feasible_total = float(np.clip(requested_total, min_total, max_total))

    left_min_for_total = max(left_range.min_kw, feasible_total - right_range.max_kw)
    left_max_for_total = min(left_range.max_kw, feasible_total - right_range.min_kw)
    left_kw = float(np.clip(float(desired_left_kw), left_min_for_total, left_max_for_total))
    right_kw = float(feasible_total - left_kw)
    return left_kw, right_kw, feasible_total, requested_total - feasible_total


def allocate_symmetric_fuel_cell(
    total_cmd_kw: float,
    previous_left_kw: float,
    previous_right_kw: float,
    left_min_kw: float,
    left_max_kw: float,
    right_min_kw: float,
    right_max_kw: float,
    ramp_kw_per_step: float,
) -> FuelCellAllocationResult:
    """Allocate total FC command symmetrically, then minimally compensate constraints."""

    left_range = _feasible_range(left_min_kw, left_max_kw, previous_left_kw, ramp_kw_per_step)
    right_range = _feasible_range(right_min_kw, right_max_kw, previous_right_kw, ramp_kw_per_step)
    requested_total = float(total_cmd_kw)
    symmetric_left = 0.5 * requested_total
    left_kw, right_kw, feasible_total, total_error = _allocate_two_sides_with_total(
        requested_total,
        symmetric_left,
        left_range,
        right_range,
    )
    symmetric_after_feasibility = 0.5 * feasible_total
    correction = abs(left_kw - symmetric_after_feasibility) + abs(right_kw - symmetric_after_feasibility)
    fully_symmetric = bool(
        np.isclose(total_error, 0.0, atol=1e-9)
        and np.isclose(left_kw, right_kw, atol=1e-9)
        and np.isclose(left_kw + right_kw, requested_total, atol=1e-9)
    )
    return FuelCellAllocationResult(
        left_kw=left_kw,
        right_kw=right_kw,
        requested_total_kw=requested_total,
        feasible_total_kw=feasible_total,
        total_error_kw=float(total_error),
        fully_symmetric=fully_symmetric,
        constraint_adjusted=not fully_symmetric,
        correction_kw=float(correction),
    )


def _raw_balance_bias_kw(total_cmd_kw: float, soc_diff: float, config: BatterySoftBalanceConfig) -> tuple[float, bool]:
    deadband = abs(float(config.soc_balance_deadband))
    if abs(soc_diff) < deadband:
        return 0.0, True
    effective_diff = float(np.sign(soc_diff) * max(0.0, abs(soc_diff) - deadband))
    raw_bias = float(config.soc_balance_bias_gain_kw_per_soc) * effective_diff
    bias_cap = min(
        abs(float(config.max_balance_bias_kw)),
        abs(float(total_cmd_kw)) * max(0.0, float(config.max_balance_bias_ratio)),
    )
    return float(np.clip(raw_bias, -bias_cap, bias_cap)), False


def allocate_soft_balanced_battery(
    total_cmd_kw: float,
    soc_left: float,
    soc_right: float,
    previous_left_kw: float,
    previous_right_kw: float,
    charge_max_kw: float,
    discharge_max_kw: float,
    previous_balance_bias_kw: float,
    config: BatterySoftBalanceConfig,
) -> BatteryAllocationResult:
    """Allocate total battery power by symmetric split plus small, smooth SOC-balance bias.

    Positive battery power means discharge. Negative power means charge.
    The bias sign follows SOC_left - SOC_right:
    - left SOC higher -> left receives a small positive bias, so it discharges
      slightly more or charges slightly less;
    - left SOC lower -> left receives a small negative bias.
    """

    requested_total = float(total_cmd_kw)
    soc_diff = float(soc_left) - float(soc_right)
    base_left = 0.5 * requested_total
    base_right = 0.5 * requested_total
    desired_bias, deadband_active = _raw_balance_bias_kw(requested_total, soc_diff, config)

    bias_cap = min(
        abs(float(config.max_balance_bias_kw)),
        abs(requested_total) * max(0.0, float(config.max_balance_bias_ratio)),
    )
    bias_limited = abs(desired_bias) >= max(0.0, bias_cap) - 1e-9 and abs(desired_bias) > 0.0

    if deadband_active:
        smoothed_bias = 0.0
        rate_limited_bias = 0.0
        rate_limited = False
    else:
        alpha = float(np.clip(config.balance_bias_smoothing_alpha, 0.0, 1.0))
        smoothed_bias = float(previous_balance_bias_kw + alpha * (desired_bias - previous_balance_bias_kw))
        rate_limit = abs(float(config.balance_bias_rate_limit_kw_per_step))
        delta = smoothed_bias - float(previous_balance_bias_kw)
        if rate_limit > 0.0 and abs(delta) > rate_limit:
            rate_limited_bias = float(previous_balance_bias_kw + np.sign(delta) * rate_limit)
            rate_limited = True
        else:
            rate_limited_bias = smoothed_bias
            rate_limited = False
        rate_limited_bias = float(np.clip(rate_limited_bias, -bias_cap, bias_cap))

    desired_left = base_left + rate_limited_bias

    left_range = _feasible_range(
        -float(charge_max_kw),
        float(discharge_max_kw),
        float(previous_left_kw),
        float(config.batt_ramp_kw_per_step),
    )
    right_range = _feasible_range(
        -float(charge_max_kw),
        float(discharge_max_kw),
        float(previous_right_kw),
        float(config.batt_ramp_kw_per_step),
    )
    left_kw, right_kw, feasible_total, total_error = _allocate_two_sides_with_total(
        requested_total,
        desired_left,
        left_range,
        right_range,
    )
    final_bias = 0.5 * (left_kw - right_kw)
    compensation = abs(left_kw - desired_left) + abs(right_kw - (base_right - rate_limited_bias))
    constraint_adjusted = bool(compensation > 1e-9 or abs(total_error) > 1e-9)

    return BatteryAllocationResult(
        left_kw=float(left_kw),
        right_kw=float(right_kw),
        requested_total_kw=requested_total,
        feasible_total_kw=float(feasible_total),
        total_error_kw=float(total_error),
        base_left_kw=float(base_left),
        base_right_kw=float(base_right),
        desired_balance_bias_kw=float(desired_bias),
        smoothed_balance_bias_kw=float(smoothed_bias),
        final_balance_bias_kw=float(final_bias),
        previous_balance_bias_kw=float(previous_balance_bias_kw),
        soc_diff=float(soc_diff),
        soc_deadband_active=bool(deadband_active),
        bias_limited=bool(bias_limited),
        bias_rate_limited=bool(rate_limited),
        constraint_adjusted=constraint_adjusted,
        compensation_kw=float(compensation),
    )
