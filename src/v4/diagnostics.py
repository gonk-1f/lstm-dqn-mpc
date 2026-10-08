"""Read-only reward, physical-trajectory and fixed-state Q diagnostics."""
from math import fsum

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
    }


def trajectory_record(sample_id, transitions, *, completed, failure=None) -> dict:
    rows, voyage = [], 0
    for item in transitions:
        rows.append({
            'voyage_index': voyage, 'soc_before': item.state[0], 'soc_after': item.actual_soc,
            'fc_kw': item.action_kw, 'battery_bus_kw': item.actual_battery_kw,
            'load_kw': item.actual_battery_kw + item.action_kw,
            'original_reward': item.reward_cny if item.original_reward is None else item.original_reward,
            'immediate_battery_energy_adjustment': item.immediate_battery_energy_adjustment,
            'terminal_correction': item.terminal_correction, 'new_reward': item.reward_cny,
            'soc_soft_penalty': item.soc_soft_penalty_cny,
            'original_economic_ledger': ledger_components(item.original_economic_ledger),
            'actual_shore_ledger': ledger_components(item.shore_ledger),
            'modeled_terminal_settlement': ledger_components(item.modeled_terminal_ledger),
            'done': item.done, 'next_feasible_actions_kw': list(item.next_feasible_actions),
        })
        voyage += int(item.done)
    return {'sample_id': str(sample_id), 'completed': completed, 'failure': failure,
            'transitions': rows, 'reward_feedback_totals': reward_totals(transitions)}


def fixed_q_diagnostics(agent, accountant) -> list[dict]:
    """Three synthetic, fixed observable states; never samples a policy action."""
    probes = []
    for label, soc, load in (('low_soc_load300', .35, 300.),
                              ('working_soc_load600', .5, 600.),
                              ('high_soc_load100', .75, 100.)):
        physical = AccountState(soc=soc, previous_fc_kw=300.)
        state = build_state(physical, (load, load, load), accountant, departure=False)
        feasible = feasible_fc_actions(physical, load, accountant)
        with torch.no_grad():
            q = agent.online(torch.tensor([state], dtype=torch.float32))[0].tolist()
        order = sorted(ACTION_KW, key=lambda power: (-q[ACTION_KW.index(power)], power))
        feasible_order = [power for power in order if power in feasible]
        probes.append({
            'label': label, 'state': list(state), 'action_kw': list(ACTION_KW),
            'q_values': q, 'action_order_kw': order, 'feasible_actions_kw': list(feasible),
            'greedy_feasible_action_kw': feasible_order[0], 'fc_zero_q': q[0],
            'fc_zero_rank_among_61': order.index(0)+1,
            'fc_zero_feasible_rank': feasible_order.index(0)+1 if 0 in feasible else None,
            'scope': 'fixed synthetic probes, not an estimate of the training-state distribution',
        })
    return probes
