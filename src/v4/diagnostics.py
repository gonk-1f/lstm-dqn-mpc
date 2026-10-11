"""Read-only reward, physical-trajectory and fixed-state Q diagnostics."""
from math import fsum, isfinite, sqrt

import torch

from v3.control import AccountState
from .control import ACTION_KW, build_state, feasible_fc_actions


COMPONENT_NAMES = ('h2_cost_cny', 'fuel_cell_degradation_cost_cny',
                   'battery_degradation_cost_cny', 'shore_cost_cny')


def ledger_components(ledger) -> dict:
    return {name: getattr(ledger, name) if ledger is not None else 0.0
            for name in COMPONENT_NAMES}


def reward_totals(transitions) -> dict:
    values = tuple(transitions)
    return {
        'original_reward': fsum(t.reward_cny if t.original_reward is None else t.original_reward for t in values),
        'immediate_battery_energy_adjustment': fsum(t.immediate_battery_energy_adjustment for t in values),
        'terminal_correction': fsum(t.terminal_correction for t in values),
        'new_reward': fsum(t.reward_cny for t in values),
        'failure_penalty_equivalent_cny':fsum(t.failure_penalty_equivalent_cny for t in values),
    }


def _diagnostic_reward_scale(value):
    scale = float(value)
    if not isfinite(scale) or scale <= 0:
        raise ValueError('reward_scale must be finite and positive')
    return scale


def trajectory_record(sample_id, transitions, *, completed, failure=None, reward_scale=1.0) -> dict:
    scale = _diagnostic_reward_scale(reward_scale)
    transitions = tuple(transitions)
    rows, voyage = [], 0
    for item in transitions:
        physical_actions = getattr(item, 'physical_feasible_actions', ())
        candidate_actions = getattr(item, 'policy_candidate_actions', ())
        rows.append({
            'voyage_index': voyage, 'soc_before': item.state[0], 'soc_after': item.actual_soc,
            'fc_kw': item.action_kw, 'battery_bus_kw': item.actual_battery_kw,
            'load_kw': item.actual_battery_kw + item.action_kw,
            'original_reward': item.reward_cny if item.original_reward is None else item.original_reward,
            'immediate_battery_energy_adjustment': item.immediate_battery_energy_adjustment,
            'terminal_correction': item.terminal_correction, 'new_reward': item.reward_cny,
            'scaled_training_reward': item.reward_cny * scale,
            'soc_soft_penalty': item.soc_soft_penalty_cny,
            'original_economic_ledger': ledger_components(item.original_economic_ledger),
            'modeled_fixed_target_shore_ledger': ledger_components(item.shore_ledger),
            'shore_settlement_basis': (
                'modeled_fixed_target_soc_0.6' if item.shore_ledger is not None else None),
            'modeled_terminal_settlement': ledger_components(item.modeled_terminal_ledger),
            'done': item.done, 'next_feasible_actions_kw': list(item.next_feasible_actions),
            'physical_feasible_actions_kw': list(physical_actions),
            'policy_candidate_actions_kw': list(candidate_actions),
            'fc_stop_reason': (
                'forced_no_positive_action' if item.action_kw == 0 and item.state[1] > 0
                and physical_actions == (0,) else
                'delayed_or_continued_off' if item.action_kw == 0 else None
            ),
            'terminal_reason':item.terminal_reason or ('completed' if item.done else None),
            'successful_terminal':item.is_successful_terminal,
            'failure_penalty_equivalent_cny':item.failure_penalty_equivalent_cny,
        })
        voyage += int(item.done)
    return {'sample_id': str(sample_id), 'completed': completed, 'failure': failure,
            'transitions': rows, 'reward_feedback_totals': reward_totals(transitions),
            'reward_scale': scale, 'training_reward_unit': 'scaled reward-equivalent CNY',
            'economic_ledger_unit': 'CNY',
            'scaled_training_reward_total': fsum(item.reward_cny * scale for item in transitions)}


def executed_transition_statistics(transitions) -> dict:
    """Executed ONBOARD summaries, including failures, without crossing SHORE.

    Each FC change compares with that action's observed prior FC state. Reset
    states therefore count an actual departure transition from zero correctly.
    No automatic SHORE shutdown or modeled recharge control point is fabricated.
    """
    values = tuple(transitions)
    power = [item.action_kw for item in values]
    raw_change = [item.action_kw-item.state[1]*ACTION_KW[-1] for item in values]
    # Undo only machine roundoff from normalized FC state, in diagnostic kW.
    change = [0.0 if abs(value) <= 1e-9 else value for value in raw_change]
    n = len(values)
    return {
        'onboard_soc_max_including_failed_prefix': max((item.actual_soc for item in values),default=None),
        'fc_power_statistics_kw': {
            'sample_count': n, 'mean': fsum(power)/n if n else None,
            'minimum': min(power) if n else None, 'maximum': max(power) if n else None,
            'root_mean_square': sqrt(fsum(item*item for item in power)/n) if n else None,
            'distribution_step_counts': {str(action):sum(item==action for item in power) for action in ACTION_KW},
            'scope': 'All executed ONBOARD actions including failed prefixes; no SHORE steps',
        },
        'fc_change_statistics_kw': {
            'mean_signed': fsum(change)/n if n else None,
            'mean_absolute': fsum(abs(item) for item in change)/n if n else None,
            'root_mean_square': sqrt(fsum(item*item for item in change)/n) if n else None,
            'max_absolute': max((abs(item) for item in change),default=None),
            'nonzero_fraction': sum(item!=0 for item in change)/n if n else None,
            'scope': 'Commanded FC minus current observed previous FC; reset boundaries preserved',
        },
    }


def fixed_q_diagnostics(agent, accountant) -> list[dict]:
    """Three synthetic, fixed observable states; never samples a policy action."""
    probes = []
    scale = _diagnostic_reward_scale(getattr(agent,'reward_scale',1.))
    for label, soc, load in (('low_soc_load300', .35, 300.),
                              ('working_soc_load600', .5, 600.),
                              ('high_soc_load100', .75, 100.)):
        physical = AccountState(soc=soc, previous_fc_kw=300.)
        state = build_state(physical, (300., 300., 300.), (load, load, load))
        feasible = feasible_fc_actions(physical, load, accountant)
        with torch.no_grad():
            q = agent.online(torch.tensor([state], dtype=torch.float32))[0].tolist()
        order = sorted(ACTION_KW, key=lambda power: (-q[ACTION_KW.index(power)], power))
        feasible_order = [power for power in order if power in feasible]
        probes.append({
            'label': label, 'state': list(state), 'action_kw': list(ACTION_KW),
            'q_values': q, 'action_order_kw': order, 'feasible_actions_kw': list(feasible),
            'reward_scale': scale, 'q_unit': 'scaled reward-equivalent CNY',
            'q_values_original_reward_units': [value/scale for value in q],
            'q_original_unit': 'original reward-equivalent CNY',
            'greedy_feasible_action_kw': feasible_order[0], 'fc_zero_q': q[0],
            'fc_zero_q_original_reward_units': q[0]/scale,
            'fc_zero_rank_among_61': order.index(0)+1,
            'fc_zero_feasible_rank': feasible_order.index(0)+1 if 0 in feasible else None,
            'scope': 'fixed synthetic probes, not an estimate of the training-state distribution',
        })
    return probes
