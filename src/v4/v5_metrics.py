"""Bounded evaluation-only V5 economic and exact action-pair diagnostics.

No replay insertions, RNG draws or optimizer calls. Counterfactual transitions
use the original interval accountant, physical mask and failure finalizer.
"""
from dataclasses import replace
from math import fsum

import torch

from v2.data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from v3.control import AccountState
from .actions import candidate_fc_actions
from .control import (build_state, feasible_fc_actions, infeasibility_cause, state_feature_value,
                      ReplayExecutionError, NoFeasibleFCActionError)
from .dqn import masked_double_dqn_targets
from .failure_replay import FailurePenalty, prepare_failure_replay
from .independent_economics import soc_equivalent_hydrogen_cost_cny
from .reward_feedback import BatteryEnergyValue


def economic_action_pair_statistics(transitions, agent, accountant, config,
                                   failure_penalty, *, episode):
    if episode is None or not hasattr(episode, 'dt_s'):
        raise ValueError('exact episode timing is required for action-pair diagnostics')
    values = tuple(transitions)
    if len(values) > len(episode.dt_s):
        raise ValueError('trajectory exceeds episode timing')
    penalty = failure_penalty or FailurePenalty.from_reference()
    sums = {name: 0. for name in ('actual_h2', 'signed_soc_equivalent_h2',
                                 'fc_degradation', 'battery_degradation')}
    # Deterministic chronological coverage plus last executed state; no RNG.
    selected = {round(i * (len(values)-1) / 15) for i in range(16)} if values else set()
    physical, rows = AccountState(), []
    max_state_error = max_reward_error = 0.
    failure_sum = 0.
    def mask(p, j):
        full = feasible_fc_actions(p, float(episode.load_kw[j]), accountant,
                                   duration_s=float(episode.dt_s[j]))
        return full, candidate_fc_actions(full, p.previous_fc_kw, 'increment7')
    def candidate(before, item, j, power):
        after, ledger, battery = accountant.interval(before, power,
            normalize_onboard_load_kw(float(episode.load_kw[j])), duration_s=float(episode.dt_s[j]))
        soc_cost = soc_equivalent_hydrogen_cost_cny(before.soc, after.soc, config)
        raw = ledger.reward_cny - soc_cost
        terminal = j + 1 == len(episode.dt_s)
        next_j = j if terminal else j + 1
        next_load = float(episode.load_kw[next_j])
        prior_load = float(episode.load_kw[max(0, next_j-1)])
        ns = build_state(after, (power,)*3, (prior_load,prior_load,next_load),
                         state_mode='load_delta4')
        full, feasible = ((),()) if terminal else mask(after, next_j)
        delta = int(power-before.previous_fc_kw)
        transition = replace(item, action_kw=int(power), action_delta_kw=delta,
            action_index=agent.action_values_kw.index(delta), reward_cny=raw, original_reward=raw,
            actual_soc=after.soc, actual_battery_kw=battery, executed_ledger=ledger,
            soc_equivalent_hydrogen_cost_cny=soc_cost, soc_soft_penalty_cny=0., fc_ramp_penalty_cny=0.,
            failure_penalty_equivalent_cny=0., terminal_correction=0.,
            done=terminal, terminal_reason='completed' if terminal else None,
            next_state=ns, next_feasible_actions=feasible)
        if not terminal and not feasible:
            cause = 'increment_action_limited' if full else infeasibility_cause(next_load, accountant)
            error = ReplayExecutionError(next_j, OperatingMode.ONBOARD,
                NoFeasibleFCActionError('exact candidate successor has no legal action'),
                executed_transitions=(transition,), failure_kind='no_feasible_action',
                failure_cause=cause, failed_state=ns)
            transition = prepare_failure_replay(error, penalty=penalty,
                energy_value=BatteryEnergyValue.from_accountant(accountant),
                redistribute_battery_energy=False).transitions[-1]
        with torch.inference_mode():
            successor = torch.tensor([transition.next_state], dtype=torch.float32)
            next_mask = torch.zeros((1,agent.action_dim),dtype=torch.bool)
            indices = [agent.action_values_kw.index(int(fc-power)) for fc in transition.next_feasible_actions]
            next_mask[0,indices] = True
            y = masked_double_dqn_targets(
                torch.tensor([[transition.reward_cny*agent.reward_scale]],dtype=torch.float32),
                agent.online(successor), agent.target(successor),
                torch.tensor([[float(transition.done)]],dtype=torch.float32),
                next_action_masks=next_mask,gamma=agent.gamma).item()
        return transition, y
    for j,item in enumerate(values):
        if len(item.state)!=4 or item.action_mode!='increment7':
            raise ValueError('economic action-pair metrics require V5 four-dimensional increment states')
        after, ledger, _ = accountant.interval(physical,item.action_kw,
            normalize_onboard_load_kw(float(episode.load_kw[j])),duration_s=float(episode.dt_s[j]))
        max_state_error = max(max_state_error, abs(physical.soc-item.state[0]),
                              abs(physical.previous_fc_kw-state_feature_value(
                                  item.state, 'previous_fc_kw_over_600')*600),
                              abs(after.soc-item.actual_soc))
        economic = {'actual_h2':ledger.h2_cost_cny,
            'signed_soc_equivalent_h2':soc_equivalent_hydrogen_cost_cny(physical.soc,after.soc,config),
            'fc_degradation':ledger.fuel_cell_degradation_cost_cny,
            'battery_degradation':ledger.battery_degradation_cost_cny}
        for name,value in economic.items():
            sums[name]+=value
        failure_sum += item.failure_penalty_equivalent_cny
        max_reward_error = max(max_reward_error,abs(item.reward_cny+fsum(economic.values())
                                                   +item.failure_penalty_equivalent_cny))
        if j in selected and item.action_delta_kw:
            _, legal = mask(physical,j)
            if physical.previous_fc_kw in legal:
                actual, actual_y = candidate(physical,item,j,item.action_kw)
                hold, hold_y = candidate(physical,item,j,int(physical.previous_fc_kw))
                if abs(actual.reward_cny-item.reward_cny)>1e-8 or actual.done!=item.done:
                    raise ValueError('actual candidate does not reproduce formal reward/termination')
                with torch.inference_mode():
                    q = agent.online(torch.tensor([item.state],dtype=torch.float32))[0].tolist()
                hold_index=agent.action_values_kw.index(0)
                dq=q[item.action_index]-q[hold_index]
                dy=actual_y-hold_y
                rows.append({'step_1based':j+1,'soc':item.state[0],
                    'actual_action_index':item.action_index,'actual_delta_kw':item.action_delta_kw,
                    'q_actual':q[item.action_index],'q_hold':q[hold_index],
                    'target_actual':actual_y,'target_hold':hold_y,
                    'delta_q':dq,'delta_y':dy,
                    'delta_reward_scaled':agent.reward_scale*(actual.reward_cny-hold.reward_cny),
                    'ordering_sign_agrees':(dq>0)-(dq<0)==(dy>0)-(dy<0),
                    'counterfactual_done':hold.done,'counterfactual_terminal_reason':hold.terminal_reason})
        physical=after
    if max_state_error>1e-8 or max_reward_error>1e-8:
        raise ValueError('economic/physical diagnostic reconstruction differs from executed trajectory')
    return {'visited_steps':len(values),'scope':'at most 16 deterministic states per full episode at diagnostic rounds; exact physical candidate transfers, not true Q values',
        'economic_reward_components_cny':sums,'failure_penalty_reward_units':failure_sum,
        'max_physical_reconstruction_error':max_state_error,
        'max_actual_reward_reconstruction_error':max_reward_error,
        'compared_change_hold_states':len(rows),'ordering_agreement_count':sum(r['ordering_sign_agrees'] for r in rows),
        'online_change_prefers_but_target_hold_count':sum(r['delta_q']>0 and r['delta_y']<0 for r in rows),
        'action_pairs':rows}
