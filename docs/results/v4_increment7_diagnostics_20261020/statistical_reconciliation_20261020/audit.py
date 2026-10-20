"""Reconcile archived evidence. No training or policy simulation."""
from pathlib import Path
import csv,json,sys,hashlib,math
from collections import Counter,defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent;RUN=OUT.parent;ROOT=RUN.parent.parent
sys.path[:0]=[str(ROOT/'src'),str(RUN/'reward_scale_audit_20261019')]
from cpu_checkpoint import load
from v4.trajectory_io import iter_trajectory_records,resolve_trajectory_path
from v4.independent_dataset import IndependentTrainingDataset
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()
def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def table(n,rows):
    with (OUT/n).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    meta=json.loads((RUN/'run_metadata.json').read_text(encoding='utf-8'));report=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
    protect=[RUN/'run_metadata.json',RUN/'report.json',RUN/'training_state_latest.pt']+list((ROOT/'src/v4').glob('*.py'))
    before={str(p.relative_to(ROOT)):sha(p) for p in protect};dump('protected_hashes.json',before)
    counts=report['execution_counts'];total=counts['training_environment_transitions']+counts['bootstrap_environment_transitions']
    assert total==counts['economic_replay_insertions']==1113841
    assert 32*counts['economic_optimizer_updates']+counts['remaining_transition_credit']==total
    scope=[]
    for name,env,insert,updates,soft in [('bootstrap',14995,14995,468,468),('formal_rounds1_100',1098846,1098846,34339,34339),('training_including_bootstrap',total,total,34807,34807),('all_greedy_evaluation',1445749,0,0,0),('r100_exploratory',12517,12517,391,391),('r100_greedy_train',13931,0,0,0),('r100_greedy_validation',3726,0,0,0),('r100_three_phases',30174,12517,391,391)]:
        scope.append(dict(experiment_id=meta['experiment_id'],scope=name,executed_environment_actions=env,replay_insertions=insert,adam_updates=updates,target_soft_updates=soft,
            utd_updates_over_environment_actions=updates/env,
            comparable_training_utd=name not in ('r100_three_phases','all_greedy_evaluation','r100_greedy_train','r100_greedy_validation')))
    table('count_scopes.csv',scope)
    # Recalculate each Validation failure's G_t from original rewards, point-by-point.
    prior=read(RUN/'state_information_audit_20261020/actual_states_q_returns.csv');qmap={(r['sample_id'],int(r['step'])):r for r in prior if r['context']=='validation'}
    values=[];max_g_error=0.
    for sec,p in iter_trajectory_records(RUN/'round_100_validation_trajectories.json'):
        if p['completed']:continue
        ts=p['transitions'];reward=np.array([t['scaled_training_reward'] for t in ts]);g=np.cumsum(reward[::-1])[::-1]
        assert sum(t['failure_penalty_equivalent_cny']>0 for t in ts)==1
        assert ts[-1]['done'] and all(not t['done'] for t in ts[:-1])
        for i,(t,gi) in enumerate(zip(ts,g)):
            assert abs(t['new_reward']*.001-t['scaled_training_reward'])<1e-12
            prev=qmap[(p['sample_id'],i)];qi=float(prev['q_actual']);max_g_error=max(max_g_error,abs(gi-float(prev['realized_greedy_return'])))
            values.append(dict(experiment_id=meta['experiment_id'],round=100,sample_id=p['sample_id'],step_0based=i,time_s=t['relative_time_s'],steps_until_final_executed_action=len(ts)-1-i,
                q_same_state_action=qi,g_from_same_time=float(gi),q_minus_g=float(qi-gi),abs_q_minus_g=float(abs(qi-gi)),
                reward_scaled=reward[i],reward_raw=t['new_reward'],soc=t['soc_before'],load_kw=t['load_kw'],fc_kw=t['fc_kw'],
                failure_penalty_raw=t['failure_penalty_equivalent_cny'],gamma=1,reward_scale=.001,terminal_reason=ts[-1]['terminal_reason']))
    assert max_g_error<1e-12
    table('validation_failure_paired_q_g.csv',values)
    errstats=[]
    for sid in sorted(set(r['sample_id'] for r in values)):
        for lo,hi in ((0,7),(8,31),(32,99),(100,100000)):
            rr=[r for r in values if r['sample_id']==sid and lo<=r['steps_until_final_executed_action']<=hi]
            if not rr:continue
            e=np.array([r['q_minus_g'] for r in rr]);errstats.append(dict(sample_id=sid,distance_lo=lo,distance_hi=hi,n=len(e),signed_mean=float(e.mean()),absolute_median=float(np.median(abs(e))),absolute_p95=float(np.quantile(abs(e),.95)),absolute_p99=float(np.quantile(abs(e),.99))))
    table('validation_failure_q_g_by_distance.csv',errstats)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for j,sid in enumerate(sorted(set(r['sample_id'] for r in values))):
        rr=[r for r in values if r['sample_id']==sid];x=[r['steps_until_final_executed_action'] for r in rr]
        axes[0,j].plot(x,[r['q_same_state_action'] for r in rr],label='Q(s_t,a_t)');axes[0,j].plot(x,[r['g_from_same_time'] for r in rr],label='G_t');axes[0,j].set_title(sid);axes[0,j].legend()
        axes[1,j].plot(x,[r['q_minus_g'] for r in rr]);axes[1,j].axhline(0,color='gray');axes[1,j].set_xlabel('Steps before final executed action (0 = terminal)');axes[1,j].set_ylabel('Q_t - G_t, scaled units')
        for ax in axes[:,j]:ax.invert_xaxis();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(OUT/'validation_failure_paired_q_g.png',dpi=150);plt.close(fig)
    failure=read(RUN/'action_mask_audit_20261019/replay_empty_mask_states.csv')
    assert len(failure)==210
    states=Counter((float(r['soc']),float(r['load_kw']),float(r['previous_fc_kw'])) for r in failure)
    locations=Counter((r['sample_id'],r['failed_row_0based']) for r in failure)
    assert len(set((r['round'],r['sample_id'],r['failed_row_0based']) for r in failure))==210
    table('replay_failure_state_multiplicity.csv',[dict(soc=s[0],load_kw=s[1],previous_fc_kw=s[2],exact_occurrences=n) for s,n in states.items()])
    table('replay_failure_location_multiplicity.csv',[dict(sample_id=s[0],failed_row_0based=s[1],occurrences=n) for s,n in locations.items()])
    cp=load(RUN/'training_state_latest.pt');assert cp['metadata']==meta
    replay=cp['agent']['replay'];lookup={(t.source_round,t.source_sample_id,t.source_row_index):t for t in replay};assert len(lookup)==len(replay)
    dataset=IndependentTrainingDataset.open(ROOT/'data/processed/operating_dataset_independent_v2');eps={e.sample_id:e for e in dataset.load_train()}
    terms=('h2_actual','soc_equivalent','fc_degradation','battery_degradation','soc_aux','fc_squared_ramp_aux','failure')
    components=[];labels=[];matched=0;max_reward_error=0.;failed_archives_checked=0
    for rnd in sorted(set(t.source_round for t in replay)):
        for sec,p in iter_trajectory_records(RUN/f'round_{rnd:03d}_exploratory_trajectories.json'):
            ep=eps[p['sample_id']]
            for i,t in enumerate(p['transitions']):
                key=(rnd,p['sample_id'],int(ep.source_1s_row[i]))
                if key not in lookup:continue
                item=lookup[key];l=t['original_economic_ledger']
                c=[l['h2_cost_cny'],t['soc_equivalent_hydrogen_cost_cny'],l['fuel_cell_degradation_cost_cny'],l['battery_degradation_cost_cny'],t['soc_soft_penalty'],t['fc_ramp_penalty_cny'],t['failure_penalty_equivalent_cny']]
                error=abs(-math.fsum(c)*.001-item.reward_cny);max_reward_error=max(max_reward_error,error);assert error<1e-10
                if str(item.terminal_reason).startswith('failure_'):
                    assert i==len(p['transitions'])-1 and item.done and not item.next_feasible_indices
                    assert t['failure_penalty_equivalent_cny']>0;failed_archives_checked+=1
                components.append(c);labels.append(item.experience_outcome);matched+=1
        print('Matched archived Replay source round',rnd,matched,flush=True)
    assert matched==300000 and failed_archives_checked==210
    c=np.array(components);labels=np.array(labels);stats=[]
    for label in ('all','success','failure'):
        select=np.ones(len(c),bool) if label=='all' else labels==label;a=c[select];maximum=abs(a).max(1);denom=abs(a).sum()
        for j,term in enumerate(terms):
            v=a[:,j];stats.append(dict(scope='final_retained_train_replay_unweighted_entries',outcome=label,component=term,n=len(v),mean=float(v.mean()),absolute_median=float(np.median(abs(v))),absolute_p95=float(np.quantile(abs(v),.95)),signed_sum=float(v.sum()),absolute_sum=float(abs(v).sum()),absolute_component_share=float(abs(v).sum()/denom),dominant_frequency=float(((abs(v)>=maximum-1e-12)&(maximum>0)).mean())))
    table('replay_reward_component_statistics.csv',stats)
    # Existing verified per-step reward statistics cover Greedy Train/Validation separately.
    original=read(RUN/'reward_scale_audit_20261019/reward_component_statistics.csv')
    table('greedy_reward_statistics_separate_scopes.csv',[r for r in original if r['context'] in ('train','validation')])
    dump('verification.json',dict(experiment_id=meta['experiment_id'],old_counts=counts,old_training_environment_steps_including_bootstrap=total,
        q_g_pairs=len(values),max_paired_return_reconstruction_error=max_g_error,replay_failures=210,unique_exact_failed_states=len(states),duplicate_state_occurrences=210-len(states),unique_failure_locations=len(locations),maximum_state_multiplicity=max(states.values()),
        exact_causes=dict(Counter(r['exact_diagnosis'] for r in failure)),replay_rows_reward_matched=matched,max_replay_reward_error=max_reward_error,
        replay_failure_terminal_semantics_checked=failed_archives_checked,
        protected_files_unchanged=all(sha(ROOT/p)==h for p,h in before.items()),test_payloads_read=0,policy_executions=0,optimizer_updates=0))
    assert all(sha(ROOT/p)==h for p,h in before.items());print('DONE',flush=True)
if __name__=='__main__':main()
