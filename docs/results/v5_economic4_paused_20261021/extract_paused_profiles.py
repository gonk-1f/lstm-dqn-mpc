"""Read archived round-32 Greedy profiles, one at a time; no environment replay."""
from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from math import fsum
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dqn.networks.mlp_qnet import MLPQNetwork
from v4.trajectory_io import iter_trajectory_records

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/v5_economic4_delta7_td1_per_interleaved_seed42_100r_20261021_165354_083'
ROUND = 32
ACTIONS = (-60, -30, -10, 0, 10, 30, 60)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def longest(mask, durations):
    current = maximum = 0.
    for flag, duration in zip(mask, durations):
        current = current + duration if flag else 0.
        maximum = max(maximum, current)
    return maximum


def summary(profile, split):
    transitions = profile['transitions']
    durations = [float(t['duration_s']) for t in transitions]
    changes = [int(t['action_delta_kw']) for t in transitions]
    soc_after = [float(t['soc_after']) for t in transitions]
    soc_all = [float(t['soc_before']) for t in transitions] + soc_after
    reversals = sum(a * b < 0 for a, b in zip(changes, changes[1:]))
    direction_pairs = sum(a != 0 and b != 0 for a, b in zip(changes, changes[1:]))
    nonzero_changes = [value for value in changes if value]
    dt = fsum(durations)
    components = {
        'h2_actual_cost_cny': fsum(t['original_economic_ledger']['h2_cost_cny'] for t in transitions),
        'h2_battery_equivalent_signed_cost_cny': fsum(t['soc_equivalent_hydrogen_cost_cny'] for t in transitions),
        'fc_degradation_cost_cny': fsum(t['original_economic_ledger']['fuel_cell_degradation_cost_cny'] for t in transitions),
        'battery_degradation_cost_cny': fsum(t['original_economic_ledger']['battery_degradation_cost_cny'] for t in transitions),
    }
    result = {'round': ROUND, 'split': split, 'sample_id': profile['sample_id'],
        'completed': bool(profile['completed']), 'failure_cause': profile.get('failure_cause', ''),
        'failure_message': profile.get('failure') or '', 'executed_steps': len(transitions),
        'executed_duration_s': dt, 'failed_row_0based': len(transitions) if not profile['completed'] else '',
        'failed_observed_state': json.dumps(profile.get('failed_observed_state')),
        'soc_min': min(soc_all), 'soc_max': max(soc_all), 'soc_terminal': soc_after[-1],
        'fc_mean_abs_delta_kw_per_step': fsum(map(abs, changes))/len(changes),
        'fc_duration_weighted_abs_delta_kw': fsum(abs(a)*d for a,d in zip(changes,durations))/dt,
        'fc_max_abs_delta_kw': max(map(abs, changes)),
        'fc_nonzero_changes': len(nonzero_changes),
        'fc_nonzero_step_fraction': len(nonzero_changes)/len(changes),
        'fc_nonzero_time_fraction': fsum(d for a,d in zip(changes,durations) if a)/dt,
        'fc_adjacent_direction_pairs': direction_pairs,
        'fc_adjacent_reversals': reversals,
        'fc_adjacent_reversal_fraction': reversals/direction_pairs if direction_pairs else None,
        'fc_nonzero_sequence_reversals': sum(a*b<0 for a,b in zip(nonzero_changes,nonzero_changes[1:])),
        'fc_starts': sum(t['fc_kw']>0 and abs(t['state'][2]*600)<1e-9 for t in transitions),
        'fc_stops': sum(t['fc_kw']==0 and t['state'][2]*600>1e-9 for t in transitions),
        **components,
        'executed_economic_cost_cny': fsum(components.values()),
        'failure_penalty_reward_units': fsum(t['failure_penalty_equivalent_cny'] for t in transitions),
        'failure_penalty_occurrences': sum(t['failure_penalty_equivalent_cny'] != 0 for t in transitions),
        'scaled_training_reward': fsum(t['scaled_training_reward'] for t in transitions),
        'complete_zhou_cost_cny': (profile.get('economic_evaluation') or {}).get('total_cost_cny'),
    }
    result['reward_economy_reconciliation_abs_error_raw'] = abs(
        result['scaled_training_reward']/profile['reward_scale']
        +result['executed_economic_cost_cny']+result['failure_penalty_reward_units'])
    result['complete_zhou_reconciliation_abs_error_cny'] = (abs(result['executed_economic_cost_cny']-result['complete_zhou_cost_cny'])
        if result['complete_zhou_cost_cny'] is not None else None)
    for label, mask in (
            ('soc_le021', [s<=.21 for s in soc_after]), ('soc_ge079', [s>=.79 for s in soc_after]),
            ('soc_le025', [s<=.25 for s in soc_after]), ('soc_ge075', [s>=.75 for s in soc_after])):
        duration = fsum(d for flag,d in zip(mask,durations) if flag)
        result[label+'_duration_s'] = duration
        result[label+'_time_fraction'] = duration/dt
        result[label+'_longest_s'] = longest(mask, durations)
    return result


def compact_step(t, index, profile, split, online, target):
    state = t['state']
    previous = round(state[2]*600)
    candidate = t['policy_candidate_actions_kw']
    indices = [ACTIONS.index(round(power-previous)) for power in candidate]
    with torch.inference_mode():
        tensor = torch.tensor([state],dtype=torch.float32)
        q = online(tensor)[0].tolist()
        qt = target(tensor)[0].tolist()
    order = sorted(indices,key=lambda i:(-q[i],ACTIONS[i]))
    target_order = sorted(indices,key=lambda i:(-qt[i],ACTIONS[i]))
    actual = t['action_index']
    ledger = t['original_economic_ledger']
    row = {
        'round': ROUND, 'split': split, 'sample_id': profile['sample_id'], 'completed': profile['completed'],
        'failure_cause': profile.get('failure_cause',''), 'step_1based': index+1,
        'relative_time_s': t['relative_time_s'], 'duration_s': t['duration_s'],
        'soc_before': t['soc_before'], 'soc_after': t['soc_after'],
        'raw_load_kw': t['raw_load_kw'], 'load_kw': t['load_kw'],
        'fc_previous_kw': previous, 'fc_kw': t['fc_kw'], 'battery_bus_kw': t['battery_bus_kw'],
        'action_index': actual, 'action_delta_kw': t['action_delta_kw'],
        'legal_increment_bitmask': sum(1<<i for i in indices),
        'hold_legal': 3 in indices, 'greedy_action_index':order[0],
        'greedy_matches_archive': order[0]==actual, 'online_target_best_agree':order[0]==target_order[0],
        'q_actual':q[actual], 'q_hold':q[3], 'q_actual_minus_hold':q[actual]-q[3] if 3 in indices else None,
        'q_best_minus_second':q[order[0]]-q[order[1]] if len(order)>1 else None,
        'h2_actual_cost_cny':ledger['h2_cost_cny'],
        'h2_battery_equivalent_signed_cost_cny':t['soc_equivalent_hydrogen_cost_cny'],
        'fc_degradation_cost_cny':ledger['fuel_cell_degradation_cost_cny'],
        'battery_degradation_cost_cny':ledger['battery_degradation_cost_cny'],
        'normal_economic_cost_cny':ledger['h2_cost_cny']+t['soc_equivalent_hydrogen_cost_cny']+ledger['fuel_cell_degradation_cost_cny']+ledger['battery_degradation_cost_cny'],
        'failure_penalty_reward_units':t['failure_penalty_equivalent_cny'],
        'new_reward_raw':t['new_reward'], 'scaled_training_reward':t['scaled_training_reward'],
        'done':t['done'],'terminal_reason':t['terminal_reason'],
    }
    for i in range(4):
        row[f'state_{i}']=state[i]
        row[f'next_state_{i}']=t['next_state'][i]
    for i in range(7):
        row[f'q_online_{i}']=q[i]
        row[f'q_target_{i}']=qt[i]
    return row


def main():
    torch.set_num_threads(1)
    checkpoint = torch.load(RUN/'pause_analysis_20261021/round_032_weights_readonly.pt',map_location='cpu',weights_only=False)
    assert checkpoint['round']==ROUND
    online, target = MLPQNetwork(4,7,(128,128)),MLPQNetwork(4,7,(128,128))
    online.load_state_dict(checkpoint['online_state'],strict=True)
    target.load_state_dict(checkpoint['target_state'],strict=True)
    online.eval();target.eval()
    metrics, validation_steps = [], {}
    path=OUT/'round032_greedy_steps.csv.gz'
    q_counts=Counter()
    with gzip.open(path,'wt',encoding='utf-8',newline='') as handle:
        writer=None
        for split in ('train','validation'):
            source=RUN/f'round_{ROUND:03d}_{split}_trajectories.json'
            count=0
            for section, profile in iter_trajectory_records(source):
                count+=1
                metrics.append(summary(profile,split))
                steps=[]
                for j,t in enumerate(profile['transitions']):
                    row=compact_step(t,j,profile,split,online,target)
                    if writer is None:
                        writer=csv.DictWriter(handle,list(row));writer.writeheader()
                    writer.writerow(row)
                    if split=='validation':
                        steps.append(row)
                        q_counts['states']+=1
                        q_counts['greedy_match']+=row['greedy_matches_archive']
                        if row['action_delta_kw'] and row['hold_legal']:
                            q_counts['changed_hold_legal']+=1
                            q_counts['online_target_disagree_change_hold_legal']+=not row['online_target_best_agree']
                if split=='validation':
                    validation_steps[profile['sample_id']]=steps
            assert count=={'train':35,'validation':8}[split],(split,count)
    write_csv(OUT/'sample_metrics.csv',metrics)
    selected=[]
    val=[r for r in metrics if r['split']=='validation']
    completed=[r for r in val if r['completed']]
    failed=[r for r in val if r['failure_cause']=='soc_limited']
    for label,pool,key in (
        ('most_adjacent_reversals',val,lambda r:r['fc_adjacent_reversals']),
        ('soc_limited_failure',failed,lambda r:r['executed_steps']),
        ('completed_without_boundary_contact',completed,lambda r:-r['fc_mean_abs_delta_kw_per_step'])):
        pool=[r for r in pool if r['sample_id'] not in [p['sample_id'] for p in selected]]
        if pool:
            chosen=dict(max(pool,key=key));chosen['selection_reason']=label;selected.append(chosen)
    write_csv(OUT/'representative_profile_selection.csv',selected)
    fig,axes=plt.subplots(len(selected),2,figsize=(15,3.1*len(selected)),squeeze=False,constrained_layout=True)
    for j,profile in enumerate(selected):
        steps=validation_steps[profile['sample_id']]
        write_csv(OUT/f"round032_validation_{profile['sample_id']}_profile.csv",steps)
        t=np.array([s['relative_time_s'] for s in steps])/3600
        ax=axes[j,0]
        for field,label,color in (('load_kw','Load','black'),('fc_kw','FC','tab:blue'),('battery_bus_kw','Battery','tab:orange')):
            ax.step(t,[s[field] for s in steps],where='post',label=label,color=color,linewidth=1)
        ax.set(ylabel='Power (kW)',xlabel='Elapsed time (h)',title=f"{profile['sample_id']} | {profile['selection_reason']}")
        ax.grid(alpha=.2);ax.legend(loc='best',ncol=3,fontsize=8)
        ax=axes[j,1]
        ax.plot(t,[s['soc_after'] for s in steps],color='tab:green',linewidth=1.3)
        for y in (.2,.8):ax.axhline(y,color='red',linestyle='--',linewidth=.8)
        for y in (.21,.79):ax.axhline(y,color='grey',linestyle=':',linewidth=.6)
        ax.set(ylim=(.18,.82),xlabel='Elapsed time (h)',ylabel='SOC',title=f"{'Completed' if profile['completed'] else profile['failure_cause']} | reversals={profile['fc_adjacent_reversals']}")
        ax.grid(alpha=.2)
        if not profile['completed']:
            ax.plot(t[-1],steps[-1]['soc_after'],'rx',markersize=8,label='Last valid control')
            ax.legend(fontsize=8)
    fig.savefig(OUT/'representative_profiles.png',dpi=160);plt.close(fig)
    aggregate={}
    for split in ('train','validation'):
        group=[r for r in metrics if r['split']==split]
        duration=fsum(r['executed_duration_s'] for r in group)
        count=sum(r['executed_steps'] for r in group)
        a={'samples':len(group),'completed':sum(r['completed'] for r in group),
           'failure_causes':dict(Counter(r['failure_cause'] for r in group if not r['completed'])),
           'executed_steps':count,'executed_duration_s':duration,
           'mean_abs_fc_change_kw_per_step':fsum(r['fc_mean_abs_delta_kw_per_step']*r['executed_steps'] for r in group)/count,
           'nonzero_fc_step_fraction':sum(r['fc_nonzero_changes'] for r in group)/count,
           'adjacent_reversals':sum(r['fc_adjacent_reversals'] for r in group),
           'adjacent_direction_pairs':sum(r['fc_adjacent_direction_pairs'] for r in group),
           'fc_starts':sum(r['fc_starts'] for r in group),'fc_stops':sum(r['fc_stops'] for r in group),
           'soc_min':min(r['soc_min'] for r in group),'soc_max':max(r['soc_max'] for r in group),
           'max_reward_reconciliation_error_raw':max(r['reward_economy_reconciliation_abs_error_raw'] for r in group),
           'max_complete_zhou_reconciliation_error_cny':max((r['complete_zhou_reconciliation_abs_error_cny'] for r in group if r['completed']),default=None)}
        for key in ('soc_le021','soc_ge079','soc_le025','soc_ge075'):
            a[key+'_duration_s']=fsum(r[key+'_duration_s'] for r in group)
            a[key+'_time_fraction']=a[key+'_duration_s']/duration
            a[key+'_longest_s']=max(r[key+'_longest_s'] for r in group)
        a['cost_scope']='Executed observed prefixes; incomplete sample costs are not complete-voyage comparable costs'
        a['completed_sample_ids']=[r['sample_id'] for r in group if r['completed']]
        a['failure_penalty_occurrences']=sum(r['failure_penalty_occurrences'] for r in group)
        a['economic_components_executed_cny']={k:fsum(r[k] for r in group) for k in ('h2_actual_cost_cny','h2_battery_equivalent_signed_cost_cny','fc_degradation_cost_cny','battery_degradation_cost_cny')}
        aggregate[split]=a
    changed_hold=[s for steps in validation_steps.values() for s in steps if s['action_delta_kw'] and s['hold_legal']]
    q_diagnostics={'changed_hold_legal_count':len(changed_hold),
        'online_target_best_disagree_fraction':sum(not s['online_target_best_agree'] for s in changed_hold)/len(changed_hold),
        'q_chosen_minus_hold_quantiles':{str(p):float(np.quantile([s['q_actual_minus_hold'] for s in changed_hold],p)) for p in (0,.05,.5,.95,1)},
        'best_minus_second_quantiles':{str(p):float(np.quantile([s['q_best_minus_second'] for s in changed_hold if s['q_best_minus_second'] is not None],p)) for p in (0,.05,.5,.95,1)},
        'best_minus_second_below_0_001_count':sum(s['q_best_minus_second']<.001 for s in changed_hold if s['q_best_minus_second'] is not None),
        'near_tie_threshold_note':'0.001 is a descriptive Q-gap cut, not proof of instability; target is also an estimated network, not truth.'}
    findings={'run_dir':str(RUN),'experiment_id':checkpoint['experiment_id'],'checkpoint_round':ROUND,
        'source_commit':checkpoint['source_commit'],'source_tree_sha256':checkpoint['source_tree_sha256'],
        'method':'Only stream archived round032 Train/Validation; no environment/policy replay; frozen CPU/eval single-state Online/Target forward; no optimizer or Replay load',
        'soc_time_definition':'SOC_after assigned to each executed interval duration; includes failed prefixes. This is sampled control-interval exposure, not a continuous-time crossing interpolation.',
        'reversal_definition':'Adjacent executed decisions with product(delta_prev,delta_current)<0. Holds break adjacency; sample boundaries never connect. Also report nonzero-sequence reversals separately.',
        'q_scope':'Round32 frozen checkpoint; Online Q only establishes predicted ranking, not Bellman target or counterfactual realized benefit',
        'aggregate':aggregate,'validation_q_counts':dict(q_counts),'validation_q_summary':q_diagnostics,'representative_profiles':selected,
        'evidence_preserved':'All 43 latest Greedy profiles preserved as compact per-step gzip CSV with exact archived states/next_states, action mask, 7 Online/Target Q values, costs and rewards. Earlier raw trajectories are not reconstructed here.'}
    (OUT/'profile_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'aggregate':aggregate,'q_counts':dict(q_counts),'selected':[r['sample_id'] for r in selected]},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
