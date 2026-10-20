"""Read-only frozen 1-step DDQN candidate-target comparison, not a rollout."""
import csv
import json
import math

import audit_support as a
from q_action_audit import distribution


def main():
    policy,cp,_=a.load_policy()
    with a.torch.random.fork_rng(devices=[]):
        model=a.MLPQNetwork(3,7,(128,128))
    model.load_state_dict(cp['target_state'],strict=True)
    model.eval().requires_grad_(False)
    target=a.FrozenPolicy(model)
    accountant,_=a.accountant_config(cp)
    profile=a.load_profile('parent_020_seg01','validation')
    actual=profile['transitions']
    with (a.OUT/'q_action_audit_parent020.csv').open(encoding='utf-8-sig',newline='') as f:
        qrows=list(csv.DictReader(f))
    with (a.OUT/'reward_decomposition.csv').open(encoding='utf-8-sig',newline='') as f:
        rewards={(int(r['step_1based']),role):r for r in csv.DictReader(f)
                 for role in r['candidate_role'].split('|')
                 if r['sample_id']=='parent_020_seg01' and role in ('actual','hold')}
    rows=[]
    for qrow in qrows:
        step=int(qrow['step_1based'])
        index=step-1
        assert not actual[index]['done'] and index+1<len(actual)
        next_actual=actual[index+1]
        row={'sample_id':'parent_020_seg01','step_1based':step,'relative_time_s':qrow['relative_time_s'],
             'current_soc':float(qrow['soc_before']),'current_load_kw':float(qrow['load_kw']),
             'current_previous_fc_kw':float(qrow['previous_fc_kw']),
             'actual_delta_kw':int(qrow['actual_delta_kw']),
             'next_actual_observed_load_kw':next_actual['load_kw'],
             'current_online_q_actual':float(qrow['actual_q']),
             'current_online_q_hold':float(qrow['hold_q']),
             'current_q_actual_minus_hold':float(qrow['actual_minus_hold_q']),
             'reverse_against_previous_actual':qrow['reversal_against_previous_executed_action']}
        for role in ('actual','hold'):
            reward=rewards[step,role]
            fc=float(reward['candidate_fc_kw'])
            soc=float(reward['candidate_soc_after'])
            physical=a.AccountState(soc=soc,previous_fc_kw=fc,
                fc_loss_uv=float(reward['candidate_fc_loss_after_uv']),
                battery_weighted_ah=float(reward['candidate_battery_weighted_ah_after']))
            full,candidate=a.physical_mask(physical,next_actual['load_kw'],accountant,next_actual['duration_s'])
            next_state=(soc,next_actual['load_kw']/600,fc/600)
            online=policy.q(next_state)
            tq=target.q(next_state)
            legal=[i for i,d in enumerate(a.FC_INCREMENT_KW) if fc+d in candidate]
            immediate=float(reward['normal_training_reward_scaled_once'])
            if legal:
                best=max(legal,key=lambda i:online[i])
                bootstrap=tq[best]
                y=immediate+cp['hyperparameters']['gamma']*bootstrap
                row[f'{role}_next_best_delta_kw']=a.FC_INCREMENT_KW[best]
            else:
                # No fabricated continuation or counterfactual failure penalty.
                bootstrap=None
                y=None
                row[f'{role}_next_best_delta_kw']=None
            row.update({f'{role}_candidate_fc_kw':fc,f'{role}_candidate_next_soc':soc,
                        f'{role}_immediate_normal_scaled_reward':immediate,
                        f'{role}_next_physical_actions_kw':json.dumps(full),
                        f'{role}_next_increment_actions_kw':json.dumps(candidate),
                        f'{role}_next_legal_action_indices':json.dumps(legal),
                        f'{role}_next_mask_empty':not bool(legal),
                        f'{role}_target_next_q':bootstrap,f'{role}_frozen_ddqn_target':y,
                        f'{role}_target_minus_current_q':None if y is None else y-row[f'current_online_q_{role}']})
            if role=='actual':
                assert list(candidate)==actual[index]['next_feasible_actions_kw']
                assert math.isclose(y,float(qrow['fixed_ddqn_target']),abs_tol=1e-10)
        row['immediate_actual_minus_hold']=row['actual_immediate_normal_scaled_reward']-row['hold_immediate_normal_scaled_reward']
        row['bootstrap_actual_minus_hold']=(row['actual_target_next_q']-row['hold_target_next_q']
            if row['actual_target_next_q'] is not None and row['hold_target_next_q'] is not None else None)
        row['frozen_target_actual_minus_hold']=(row['actual_frozen_ddqn_target']-row['hold_frozen_ddqn_target']
            if row['actual_frozen_ddqn_target'] is not None and row['hold_frozen_ddqn_target'] is not None else None)
        row['current_q_gap_minus_frozen_target_gap']=(row['current_q_actual_minus_hold']-row['frozen_target_actual_minus_hold']
            if row['frozen_target_actual_minus_hold'] is not None else None)
        rows.append(row)
    with (a.OUT/'bootstrap_action_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete=[r for r in rows if r['frozen_target_actual_minus_hold'] is not None]
    def pack(values):
        return {key:distribution([r[key] for r in values]) for key in
            ('current_q_actual_minus_hold','immediate_actual_minus_hold','bootstrap_actual_minus_hold',
             'frozen_target_actual_minus_hold','current_q_gap_minus_frozen_target_gap',
             'actual_target_minus_current_q','hold_target_minus_current_q')}
    summary={'sample_id':'parent_020_seg01','step_first':rows[0]['step_1based'],'step_last':rows[-1]['step_1based'],
             'count':len(rows),'valid_both_successors':len(complete),
             'actual_no_legal_successor_count':sum(r['actual_next_mask_empty'] for r in rows),
             'hold_no_legal_successor_count':sum(r['hold_next_mask_empty'] for r in rows),
             'current_q_prefers_actual_count':sum(r['current_q_actual_minus_hold']>0 for r in complete),
             'immediate_reward_prefers_actual_count':sum(r['immediate_actual_minus_hold']>0 for r in complete),
             'frozen_1step_target_prefers_actual_count':sum(r['frozen_target_actual_minus_hold']>0 for r in complete),
             'frozen_1step_target_prefers_hold_count':sum(r['frozen_target_actual_minus_hold']<0 for r in complete),
             'statistics':pack(complete),
             'actual_delta_groups':{str(d):{'count':len(g:=[r for r in complete if r['actual_delta_kw']==d]),
                                           'target_prefers_actual_count':sum(r['frozen_target_actual_minus_hold']>0 for r in g),
                                           'statistics':pack(g)} for d in (-30,60)},
             'interpretation':'Exact one-step candidate SOC and rewards come from formal physical reward reconstruction, using the same archived actual next observed load. Frozen Online chooses the next feasible increment and frozen Target evaluates it. Candidate target comparisons are not true long-horizon returns, not proof of optimal action ordering, and not an independent rollout; no failure award was assumed for impossible successors.',
             'modified_formal_weights_replay_or_optimizer':False}
    (a.OUT/'bootstrap_action_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
