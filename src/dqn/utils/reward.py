from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class FixedRewardWeights:
    # Clean lower-layer reward weights:
    # w1 energy use, w2 fuel-cell dynamics, w3 SOC objective,
    # w4 battery-use objective. Older fine-grained weights are accepted for
    # compatibility but are no longer independent reward groups.
    w1: float = 0.30
    w2: float = 0.25
    w3: float = 0.80
    w4: float = 0.35
    w5: float = 0.00
    w6_battery_smooth: float = 0.00
    w7_soc_balance: float = 0.00
    # Deprecated compatibility weights. They are accepted from older configs
    # but are no longer used as independent reward terms.
    w7_soc_boundary: float = 0.0
    w8_action_smooth: float = 0.0
    w9_soc_balance: float = 0.0
    w10_battery_balance: float = 0.10


@dataclass
class RewardRefs:
    p_fc_ref_kw: float = 0.0
    p_fc_base_ref_kw: float | None = None
    soc_ref: float = 0.65


@dataclass
class RewardParams:
    p_fc_min_kw: float
    p_fc_max_kw: float
    p_bat_min_kw: float
    p_bat_max_kw: float
    soc_min: float
    soc_max: float
    weights: FixedRewardWeights
    soc_guard_low: float = 0.40
    soc_guard_high: float = 0.85
    fc_var_reward_alpha: float = 0.0
    fc_var_reward_soc_gate: float = 0.85
    fc_var_reward_progress_gate: float = 0.20
    soc_balance_highsoc_scale: float = 0.0
    soc_direction_fc_scale: float = 2.0
    soc_battery_direction_scale: float = 2.0
    soc_correction_credit_scale: float = 0.20
    fc_trim_penalty_scale: float = 0.50
    battery_corrective_use_discount: float = 0.50
    battery_wrong_direction_use_penalty: float = 0.50
    battery_use_soc_factor_min: float = 0.35
    battery_use_soc_factor_max: float = 1.50


def calc_soc_protect_norm(
    soc: float,
    soc_ref: float,
    soc_guard_low: float,
    soc_guard_high: float,
    soc_min: float,
    soc_max: float,
) -> float:
    """Compact SOC protection term.

    Inside the guard band it behaves like a normalized SOC tracking error.
    Near the guard limits it adds a soft quadratic guard penalty. Hard
    violations remain the job of `constraint_penalty_norm`.
    """

    soc_span = max(float(soc_max) - float(soc_min), 1e-6)
    soc_dev_norm = abs(float(soc) - float(soc_ref)) / soc_span
    low_margin = max(float(soc_ref) - float(soc_guard_low), 1e-6)
    high_margin = max(float(soc_guard_high) - float(soc_ref), 1e-6)
    low_guard_norm = (max(0.0, float(soc_ref) - float(soc)) / low_margin) ** 2
    high_guard_norm = (max(0.0, float(soc) - float(soc_ref)) / high_margin) ** 2
    if soc_guard_low <= soc <= soc_guard_high:
        guard_norm = 0.25 * (low_guard_norm + high_guard_norm)
    else:
        # Keep this soft; hard out-of-bounds penalties live in constraints.
        guard_norm = low_guard_norm + high_guard_norm
    return float(soc_dev_norm + guard_norm)


def calculate_reward(state: dict, action: int | None, next_state: dict, refs: RewardRefs, params: RewardParams):
    """Four-group normalized lower-layer reward for the clean KAN-DQN stage.

    Notes:
    - `SOC_ref` is a placeholder when the real MPC layer does not yet export it.
      The environments currently use `mpc_soc_ref` if present, otherwise a
      configurable default such as 0.65.
    - In the dual-side environment, `P_fc` and `P_bat` are currently aggregated
      total powers so the reward can be shared by both the simple and dual-side
      lower-layer DQN code paths with minimal refactoring.
    - Physical boundary projection is reported as a diagnostic term. It is not
      part of the learned objective unless a caller deliberately sets `w5`.
    """

    del action  # Reserved for future reward variants that depend on it explicitly.

    p_fc = float(next_state["p_fc_kw"])
    p_fc_prev = float(state["p_fc_kw"])
    p_bat = float(next_state["p_bat_kw"])
    soc = float(next_state["soc"])
    soc_ref = float(next_state.get("soc_ref", 0.65))

    p_fc_max = max(float(params.p_fc_max_kw), 1e-6)
    p_bat_scale = max(abs(float(params.p_bat_max_kw)), abs(float(params.p_bat_min_kw)), 1e-6)
    soc_span = max(float(params.soc_max) - float(params.soc_min), 1e-6)

    fuel_cost_norm = max(p_fc, 0.0) / p_fc_max
    p_fc_ref = max(float(refs.p_fc_ref_kw), 0.0)
    p_fc_base_ref = p_fc_ref if refs.p_fc_base_ref_kw is None else max(float(refs.p_fc_base_ref_kw), 0.0)
    fc_surplus_norm = max(0.0, p_fc - p_fc_ref) / p_fc_max
    fc_deficit_norm = max(0.0, p_fc_ref - p_fc) / p_fc_max
    fc_correction_surplus_norm = max(0.0, p_fc_ref - p_fc_base_ref) / p_fc_max
    fc_correction_deficit_norm = max(0.0, p_fc_base_ref - p_fc_ref) / p_fc_max
    fc_trim_norm = abs(p_fc_ref - p_fc_base_ref) / p_fc_max
    fc_smooth_norm = abs(p_fc - p_fc_prev) / p_fc_max
    p_fc_ema_kw = float(next_state.get("p_fc_ema_kw", state.get("p_fc_ema_kw", p_fc_prev)))
    fc_var_proxy_norm = abs(p_fc - p_fc_ema_kw) / p_fc_max
    episode_progress = float(next_state.get("episode_progress", 1.0))
    highsoc_early_gate = float(
        soc >= float(params.fc_var_reward_soc_gate)
        and episode_progress <= float(params.fc_var_reward_progress_gate)
    )
    fc_var_gate = float(highsoc_early_gate and float(params.fc_var_reward_alpha) > 0.0)
    fc_trim_penalty_norm = float(params.fc_trim_penalty_scale) * fc_trim_norm
    fc_smooth_aligned_norm = (
        fc_smooth_norm
        + fc_trim_penalty_norm
        + float(params.fc_var_reward_alpha) * fc_var_gate * fc_var_proxy_norm
    )
    soc_dev_norm = abs(soc - soc_ref) / soc_span
    battery_deg_norm = (p_bat / p_bat_scale) ** 2
    p_bat_prev = float(state.get("p_bat_kw", p_bat))
    battery_smooth_norm = abs(p_bat - p_bat_prev) / p_bat_scale
    soc_guard_low = float(params.soc_guard_low)
    soc_guard_high = float(params.soc_guard_high)
    soc_boundary_norm = calc_soc_protect_norm(
        soc=soc,
        soc_ref=soc_ref,
        soc_guard_low=soc_guard_low,
        soc_guard_high=soc_guard_high,
        soc_min=float(params.soc_min),
        soc_max=float(params.soc_max),
    ) - soc_dev_norm
    soc_protect_norm = soc_dev_norm + soc_boundary_norm
    action_smooth_norm = float(next_state.get("action_switch_penalty", 0.0))
    smooth_norm = battery_smooth_norm
    soc_balance_norm = abs(float(next_state.get("soc_left", soc)) - float(next_state.get("soc_right", soc))) / soc_span
    soc_balance_aligned_norm = soc_balance_norm * (
        1.0 + float(params.soc_balance_highsoc_scale) * highsoc_early_gate
    )
    p_bat_left = float(next_state.get("p_bat_left_kw", p_bat * 0.5))
    p_bat_right = float(next_state.get("p_bat_right_kw", p_bat * 0.5))
    battery_balance_norm = abs(p_bat_left - p_bat_right) / p_bat_scale

    penalty = 0.0
    if soc < params.soc_min or soc > params.soc_max:
        penalty += 1.0
    if p_fc < params.p_fc_min_kw or p_fc > params.p_fc_max_kw:
        penalty += 1.0
    if p_bat < params.p_bat_min_kw or p_bat > params.p_bat_max_kw:
        penalty += 1.0
    constraint_penalty_norm = penalty

    energy_norm = fuel_cost_norm
    fc_dynamic_norm = fc_smooth_aligned_norm
    soc_deficit_norm = max(0.0, soc_ref - soc) / soc_span
    soc_surplus_norm = max(0.0, soc - soc_ref) / soc_span
    battery_discharge_norm = max(p_bat, 0.0) / p_bat_scale
    battery_charge_norm = max(-p_bat, 0.0) / p_bat_scale
    wrong_battery_direction_norm = (
        soc_deficit_norm * battery_discharge_norm
        + soc_surplus_norm * battery_charge_norm
    )
    corrective_battery_direction_norm = (
        soc_deficit_norm * battery_charge_norm
        + soc_surplus_norm * battery_discharge_norm
    )
    fc_soc_direction_norm = float(params.soc_direction_fc_scale) * (
        soc_deficit_norm * fc_correction_deficit_norm
        + soc_surplus_norm * fc_correction_surplus_norm
    )
    soc_recovery_norm = wrong_battery_direction_norm
    soc_correction_credit_norm = float(params.soc_correction_credit_scale) * corrective_battery_direction_norm
    terminal_soc_norm = (episode_progress ** 2) * soc_dev_norm
    soc_battery_direction_norm = float(params.soc_battery_direction_scale) * wrong_battery_direction_norm
    soc_direction_norm = soc_battery_direction_norm + fc_soc_direction_norm
    soc_norm_raw = (
        soc_protect_norm
        + soc_balance_aligned_norm
        + soc_direction_norm
        + terminal_soc_norm
    )
    soc_norm = max(0.0, soc_norm_raw - soc_correction_credit_norm)

    battery_direction_den = max(
        (soc_deficit_norm + soc_surplus_norm) * (battery_discharge_norm + battery_charge_norm),
        1e-6,
    )
    corrective_battery_use_alignment = corrective_battery_direction_norm / battery_direction_den
    wrong_battery_use_alignment = wrong_battery_direction_norm / battery_direction_den
    battery_use_soc_factor = (
        1.0
        - float(params.battery_corrective_use_discount) * corrective_battery_use_alignment
        + float(params.battery_wrong_direction_use_penalty) * wrong_battery_use_alignment
    )
    battery_use_soc_factor = min(
        float(params.battery_use_soc_factor_max),
        max(float(params.battery_use_soc_factor_min), battery_use_soc_factor),
    )
    battery_use_raw_norm = (battery_deg_norm + battery_smooth_norm + battery_balance_norm) / 3.0
    battery_use_norm = battery_use_raw_norm * battery_use_soc_factor
    safety_projection_norm = constraint_penalty_norm

    reward = -(
        params.weights.w1 * energy_norm
        + params.weights.w2 * fc_dynamic_norm
        + params.weights.w3 * soc_norm
        + params.weights.w4 * battery_use_norm
        + params.weights.w5 * safety_projection_norm
    )

    reward_info = {
        "energy_norm": float(energy_norm),
        "fc_dynamic_norm": float(fc_dynamic_norm),
        "soc_norm": float(soc_norm),
        "soc_norm_raw": float(soc_norm_raw),
        "soc_recovery_norm": float(soc_recovery_norm),
        "soc_direction_norm": float(soc_direction_norm),
        "soc_battery_direction_norm": float(soc_battery_direction_norm),
        "wrong_battery_direction_norm": float(wrong_battery_direction_norm),
        "corrective_battery_direction_norm": float(corrective_battery_direction_norm),
        "fc_soc_direction_norm": float(fc_soc_direction_norm),
        "soc_correction_credit_norm": float(soc_correction_credit_norm),
        "terminal_soc_norm": float(terminal_soc_norm),
        "battery_use_norm": float(battery_use_norm),
        "battery_use_raw_norm": float(battery_use_raw_norm),
        "battery_use_soc_factor": float(battery_use_soc_factor),
        "corrective_battery_use_alignment": float(corrective_battery_use_alignment),
        "wrong_battery_use_alignment": float(wrong_battery_use_alignment),
        "safety_projection_norm": float(safety_projection_norm),
        "fuel_cost_norm": float(fuel_cost_norm),
        "fc_ref_norm": float(p_fc_ref / p_fc_max),
        "fc_base_ref_norm": float(p_fc_base_ref / p_fc_max),
        "fc_trim_norm": float(fc_trim_norm),
        "fc_trim_penalty_norm": float(fc_trim_penalty_norm),
        "fc_surplus_norm": float(fc_surplus_norm),
        "fc_deficit_norm": float(fc_deficit_norm),
        "fc_correction_surplus_norm": float(fc_correction_surplus_norm),
        "fc_correction_deficit_norm": float(fc_correction_deficit_norm),
        "fc_smooth_norm": float(fc_smooth_norm),
        "fc_var_proxy_norm": float(fc_var_proxy_norm),
        "highsoc_early_gate": float(highsoc_early_gate),
        "fc_var_gate": float(fc_var_gate),
        "fc_smooth_aligned_norm": float(fc_smooth_aligned_norm),
        "soc_protect_norm": float(soc_protect_norm),
        "soc_dev_norm": float(soc_dev_norm),
        "battery_deg_norm": float(battery_deg_norm),
        "battery_smooth_norm": float(battery_smooth_norm),
        "smooth_norm": float(smooth_norm),
        "soc_boundary_norm": float(soc_boundary_norm),
        "action_smooth_norm": float(action_smooth_norm),
        "soc_balance_norm": float(soc_balance_norm),
        "soc_balance_aligned_norm": float(soc_balance_aligned_norm),
        "battery_balance_norm": float(battery_balance_norm),
        "constraint_penalty_norm": float(constraint_penalty_norm),
        "total_reward": float(reward),
        "reward_weights": asdict(params.weights),
    }
    return float(reward), reward_info
