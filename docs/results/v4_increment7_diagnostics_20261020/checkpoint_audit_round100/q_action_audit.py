"""CPU/eval-only reconstruction of R100 archived decision Q values."""
import csv
import json
import math
from statistics import mean, median

import audit_support as a


def distribution(values):
    values = sorted(float(x) for x in values if x is not None)
    if not values:
        return {'count': 0}
    def quantile(q):
        p = (len(values)-1)*q
        lo = int(p)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(p-lo)
    return {'count':len(values), 'min':values[0], 'p10':quantile(.1),
            'median':median(values), 'mean':mean(values),
            'p90':quantile(.9), 'max':values[-1]}


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_profile(policy, profile, accountant, steps):
    records = profile['transitions']
    rows=[]
    for index in steps:
        item=records[index]
        previous = item['fc_kw']-item['action_delta_kw']
        state=(item['soc_before'],item['load_kw']/600,previous/600)
        physical=a.AccountState(soc=state[0], previous_fc_kw=previous)
        absolute,candidate=a.physical_mask(physical,item['load_kw'],accountant,item['duration_s'])
        assert list(absolute)==item['physical_feasible_actions_kw'], (profile['sample_id'],index,'physical')
        assert list(candidate)==item['policy_candidate_actions_kw'], (profile['sample_id'],index,'candidate')
        if index:
            assert previous==records[index-1]['fc_kw'], (profile['sample_id'],index,'actual-history')
            assert state[0]==records[index-1]['soc_after'], (profile['sample_id'],index,'SOC-history')
        q=policy.q(state)
        legal=[i for i,d in enumerate(a.FC_INCREMENT_KW) if previous+d in candidate]
        ranked=sorted(legal,key=lambda i:(-q[i],i))
        best=ranked[0]
        actual=item['action_index']
        assert a.FC_INCREMENT_KW[actual]==item['action_delta_kw']
        assert previous+a.FC_INCREMENT_KW[actual]==item['fc_kw']
        prevdelta=records[index-1]['action_delta_kw'] if index else 0
        hold=q[3] if 3 in legal else None
        row={
            'sample_id':profile['sample_id'],'archive_transition_index':index,
            'step_1based':index+1,'relative_time_s':item['relative_time_s'],
            'duration_s':item['duration_s'],'soc_before':state[0],
            'load_kw':item['load_kw'],'previous_fc_kw':previous,
            'state_soc':state[0],'state_load_over600':state[1],'state_previous_fc_over600':state[2],
            'physical_feasible_actions_kw':json.dumps(absolute),
            'physical_action_count':len(absolute),'policy_candidate_actions_kw':json.dumps(candidate),
            'legal_action_indices':json.dumps(legal),'legal_action_deltas_kw':json.dumps([a.FC_INCREMENT_KW[i] for i in legal]),
            'actual_action_index':actual,'actual_delta_kw':item['action_delta_kw'],
            'actual_fc_kw':item['fc_kw'],'actual_legal_rank':ranked.index(actual)+1,
            'actual_q':q[actual],'best_action_index':best,'best_delta_kw':a.FC_INCREMENT_KW[best],
            'best_fc_kw':previous+a.FC_INCREMENT_KW[best],'best_q':q[best],
            'second_action_index':ranked[1] if len(ranked)>1 else None,
            'second_q':q[ranked[1]] if len(ranked)>1 else None,
            'best_minus_second_q':q[best]-q[ranked[1]] if len(ranked)>1 else None,
            'hold_legal':3 in legal,'hold_q':hold,
            'best_minus_hold_q':q[best]-hold if hold is not None else None,
            'actual_minus_hold_q':q[actual]-hold if hold is not None else None,
            'greedy_matches_archive':actual==best,
            'previous_actual_delta_kw':prevdelta,
            'reversal_against_previous_executed_action':prevdelta*item['action_delta_kw']<0,
            'reversal_within_selected_window':bool(rows and rows[-1]['actual_delta_kw']*item['action_delta_kw']<0),
            'next_soc':item['soc_after'],'next_battery_kw':item['battery_bus_kw'],
            'raw_reward':item['new_reward'],'scaled_training_reward':item['scaled_training_reward'],
            'failure_penalty_equivalent_cny':item['failure_penalty_equivalent_cny'],
            'q_unit':'scaled_reward_equivalent_cny',
        }
        for delta,qi in zip(a.FC_INCREMENT_KW,q):
            row[f'q_delta_{delta:+d}_kw']=qi
        rows.append(row)
    return rows


def summarize(rows):
    rows=list(rows)
    within_reversal=[bool(i and rows[i-1]['actual_delta_kw']*r['actual_delta_kw']<0)
                     for i,r in enumerate(rows)]
    def group(selection):
        return {'count':len(selection),
                'top2_margin':distribution([r['best_minus_second_q'] for r in selection]),
                'best_minus_hold_q':distribution([r['best_minus_hold_q'] for r in selection]),
                'actual_q':distribution([r['actual_q'] for r in selection]),
                'hold_q':distribution([r['hold_q'] for r in selection]),
                'greedy_matches':sum(r['greedy_matches_archive'] for r in selection)}
    summary=group(rows)
    summary.update({'step_first':rows[0]['step_1based'],'step_last':rows[-1]['step_1based'],
                    'all_seven_actions_legal':sum(r['physical_action_count']>0 and len(json.loads(r['legal_action_indices']))==7 for r in rows),
                    'hold_legal_count':sum(r['hold_legal'] for r in rows),
                    'within_window_reversal_count':sum(within_reversal),
                    'against_previous_actual_reversal_count':sum(r['reversal_against_previous_executed_action'] for r in rows),
                    'mean_absolute_delta_kw':mean(abs(r['actual_delta_kw']) for r in rows),
                    'actual_delta_counts':{str(d):sum(r['actual_delta_kw']==d for r in rows) for d in a.FC_INCREMENT_KW},
                    'reversal_group':group([r for r,flag in zip(rows,within_reversal) if flag]),
                    'non_reversal_group':group([r for r,flag in zip(rows,within_reversal) if not flag]),
                    'top2_margin_below_001_count':sum(r['best_minus_second_q'] is not None and r['best_minus_second_q']<.001 for r in rows),
                    'top2_margin_below_01_count':sum(r['best_minus_second_q'] is not None and r['best_minus_second_q']<.01 for r in rows)})
    return summary


def add_target_reference(rows, cp):
    with a.torch.random.fork_rng(devices=[]):
        model=a.MLPQNetwork(3,7,(128,128))
    model.load_state_dict(cp['target_state'],strict=True)
    model.eval().requires_grad_(False)
    target=a.FrozenPolicy(model)
    for row in rows:
        q=target.q((row['state_soc'],row['state_load_over600'],row['state_previous_fc_over600']))
        legal=json.loads(row['legal_action_indices'])
        ranked=sorted(legal,key=lambda i:(-q[i],i))
        row['target_best_action_index']=ranked[0]
        row['target_best_delta_kw']=a.FC_INCREMENT_KW[ranked[0]]
        row['target_best_q']=q[ranked[0]]
        row['target_best_minus_second_q']=q[ranked[0]]-q[ranked[1]] if len(ranked)>1 else None
        row['target_q_actual']=q[row['actual_action_index']]
        row['target_q_hold']=q[3] if row['hold_legal'] else None
        row['online_target_best_agrees']=ranked[0]==row['best_action_index']
        for delta,value in zip(a.FC_INCREMENT_KW,q):
            row[f'target_q_delta_{delta:+d}_kw']=value
    return {'online_target_same_best_count':sum(r['online_target_best_agrees'] for r in rows),
            'count':len(rows),'target_top2_margin':distribution([r['target_best_minus_second_q'] for r in rows]),
            'target_actual_minus_hold':distribution([r['target_q_actual']-r['target_q_hold'] for r in rows if r['hold_legal']]),
            'interpretation':'Same R100 target network reference only; not a historical checkpoint or proof of correct Q ranking.'}


def add_fixed_reference(rows, profile, cp, policy, accountant):
    """Frozen-snapshot DDQN residual and recorded-policy return; no rollout."""
    with a.torch.random.fork_rng(devices=[]):
        model=a.MLPQNetwork(3,7,(128,128))
    model.load_state_dict(cp['target_state'],strict=True)
    model.eval().requires_grad_(False)
    target=a.FrozenPolicy(model)
    transitions=profile['transitions']
    tails=[0.]*len(transitions)
    running=0.
    gamma=cp['hyperparameters']['gamma']
    assert gamma==1. and cp['hyperparameters']['n_step']==1
    scale=cp['hyperparameters']['reward_scale']
    for index in range(len(transitions)-1,-1,-1):
        running=transitions[index]['scaled_training_reward']+gamma*running
        tails[index]=running
    for row in rows:
        index=row['archive_transition_index']
        item=transitions[index]
        assert math.isclose(item['scaled_training_reward'],item['new_reward']*scale,rel_tol=1e-13,abs_tol=1e-13)
        value=item['scaled_training_reward']
        if item['done']:
            assert not item['next_feasible_actions_kw']
            row['fixed_ddqn_next_best_delta_kw']=None
            row['fixed_ddqn_target_next_q']=0.
        else:
            next_item=transitions[index+1]
            next_state=(item['soc_after'],next_item['load_kw']/600,item['fc_kw']/600)
            assert next_state[0]==next_item['soc_before']
            assert item['fc_kw']==next_item['fc_kw']-next_item['action_delta_kw']
            physical=a.AccountState(soc=next_state[0],previous_fc_kw=item['fc_kw'])
            _,candidate=a.physical_mask(physical,next_item['load_kw'],accountant,next_item['duration_s'])
            assert list(candidate)==item['next_feasible_actions_kw'],(profile['sample_id'],index,'successor-mask')
            next_online=policy.q(next_state)
            next_target=target.q(next_state)
            legal=[i for i,d in enumerate(a.FC_INCREMENT_KW) if item['fc_kw']+d in candidate]
            best=max(legal,key=lambda i:next_online[i])
            value+=gamma*next_target[best]
            row['fixed_ddqn_next_best_delta_kw']=a.FC_INCREMENT_KW[best]
            row['fixed_ddqn_target_next_q']=next_target[best]
        row['fixed_ddqn_target']=value
        row['fixed_ddqn_target_minus_actual_q']=value-row['actual_q']
        row['actual_greedy_remaining_return']=tails[index]
        row['actual_q_minus_recorded_greedy_return']=row['actual_q']-tails[index]
    return {'fixed_snapshot_ddqn_signed_residual':distribution([r['fixed_ddqn_target_minus_actual_q'] for r in rows]),
            'fixed_snapshot_ddqn_absolute_residual':distribution([abs(r['fixed_ddqn_target_minus_actual_q']) for r in rows]),
            'recorded_greedy_remaining_return':distribution([r['actual_greedy_remaining_return'] for r in rows]),
            'q_minus_recorded_greedy_return':distribution([r['actual_q_minus_recorded_greedy_return'] for r in rows]),
            'all_successor_masks_match_archived':True,
            'interpretation':'Frozen R100 1-step target is recomputed with the same parameter snapshot, not the historical optimizer-time target. Recorded remaining return is the actual same-R100 Greedy continuation, not optimal Q ground truth; hidden remaining voyage context may differ across equal states.'}


def main():
    policy,cp,metadata=a.load_policy()
    accountant,_=a.accountant_config(cp)
    checkpoints=sorted(p.name for p in (a.RUN/'diagnostic_checkpoints').glob('*.pt'))
    identity={k:v for k,v in cp.items() if k not in ('online_state','target_state')}
    identity.update({'checkpoint_path':str(a.RUN/'diagnostic_checkpoints/round_100.pt'),
                     'checkpoint_sha256':a.sha256(a.RUN/'diagnostic_checkpoints/round_100.pt'),
                     'metadata_identity_keys_match':True,
                     'network_layers':str(policy.model),
                     'online_parameter_shapes':{k:list(v.shape) for k,v in cp['online_state'].items()},
                     'target_parameter_shapes':{k:list(v.shape) for k,v in cp['target_state'].items()},
                     'diagnostic_checkpoint_files':checkpoints,
                     'exact_round038_present':'round_038.pt' in checkpoints,
                     'exact_round062_present':'round_062.pt' in checkpoints,
                     'evaluation_mode':not policy.model.training,
                     'model_parameters_require_grad':any(p.requires_grad for p in policy.model.parameters()),
                     'device':'cpu','batch_size_for_every_query':1,
                     'not_eligible_best_model':True})
    (a.OUT/'checkpoint_audit_identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
    zoompath=a.ROOT/'docs/results/v4_increment7_zhou_td1_per_100r_20261018/round100_validation_parent_020_seg01_zoom.csv'
    with zoompath.open(encoding='utf-8-sig',newline='') as f:
        zoom=list(csv.DictReader(f))
    window=[int(r['step_1based'])-1 for r in zoom]
    assert len(window)==64 and window==list(range(window[0],window[-1]+1))
    profile020=a.load_profile('parent_020_seg01','validation')
    rows020=inspect_profile(policy,profile020,accountant,window)
    for row,arch in zip(rows020,zoom):
        for ours,theirs in [('soc_before','soc_before'),('next_soc','soc_after'),('actual_fc_kw','fc_kw'),('actual_delta_kw','action_delta_kw')]:
            assert math.isclose(float(row[ours]),float(arch[theirs]),rel_tol=1e-12,abs_tol=1e-12)
    profile015=a.load_profile('parent_015_seg01','train')
    rows015=inspect_profile(policy,profile015,accountant,range(len(profile015['transitions'])))
    target020=add_target_reference(rows020,cp)
    target015=add_target_reference(rows015,cp)
    fixed020=add_fixed_reference(rows020,profile020,cp,policy,accountant)
    fixed015=add_fixed_reference(rows015,profile015,cp,policy,accountant)
    write_csv(a.OUT/'q_action_audit_parent020.csv',rows020)
    write_csv(a.OUT/'q_action_audit_parent015.csv',rows015)
    d=json.loads((a.RUN/'diagnostic_checkpoints/round_100_decisions.json').read_text())
    checks=[]
    for record in d['decisions']:
        if record['sample_id']!='parent_015_seg01':
            continue
        row=rows015[record['archive_transition_index']]
        qs=[row[f'q_delta_{delta:+d}_kw'] for delta in a.FC_INCREMENT_KW]
        checks.append({'archive_transition_index':record['archive_transition_index'],
                       'max_abs_q_difference':max(abs(x-y) for x,y in zip(qs,record['q_values'])),
                       'selected_action_matches':row['best_action_index']==record['selected_action_index'],
                       'state_matches':all(x==y for x,y in zip(record['state'],(row['state_soc'],row['state_load_over600'],row['state_previous_fc_over600'])))})
    failed=profile015.get('failed_observed_state')
    failure_state=None
    if failed:
        physical=a.AccountState(soc=failed[0],previous_fc_kw=round(failed[2]*600))
        absolute,candidate=a.physical_mask(physical,failed[1]*600,accountant,30.)
        failure_state={'state':failed,'q_values':policy.q(failed),
                       'absolute_physical_actions_kw':absolute,'increment_candidate_actions_kw':candidate,
                       'failure':profile015['failure'],'failure_cause':profile015.get('failure_cause')}
    summary={'window_source':str(zoompath),'window_note':'Archived actual 64 rows are step1based54..117, not55..118; no old file modified.',
             'parent020':summarize(rows020),'parent015':summarize(rows015),
             'parent015_last8':summarize(rows015[-8:]),'parent015_last32':summarize(rows015[-32:]),
             'parent015_last100':summarize(rows015[-100:]),
             'target_reference_parent020':target020,'target_reference_parent015':target015,
             'fixed_references_parent020':fixed020,'fixed_references_parent015':fixed015,
             'archived_exact_q_checks_parent015':checks,'parent015_failed_observed_state':failure_state,
             'all_recomputed_masks_match_archived':True,
             'all_actual_actions_greedy':all(r['greedy_matches_archive'] for r in rows020+rows015),
             'gradient_optimizer_training_calls':0,
             'comparison_interpretation':'Q differences are frozen R100 learned long-term estimates, not realized return or reward difference.'}
    (a.OUT/'QSummary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
