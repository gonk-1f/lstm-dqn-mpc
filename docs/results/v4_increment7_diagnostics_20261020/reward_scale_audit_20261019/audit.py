"""Read-only R100 reward/scale audit. No policy execution or optimizer calls."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=RUN.parent.parent
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from v3.control import AccountState,EconomicMPC
from v4.control import ledger_training_reward,soc_deviation_squared,FC_SQUARED_RAMP_REWARD_PER_KW2
from v4.independent_economics import ZhouEvaluationConfig,soc_equivalent_hydrogen_cost_cny
from v4.trajectory_io import iter_trajectory_records
from v2.models.fuel_cell_degradation import (
    formal_aggregate_fc_voltage_loss_step_uv,aggregate_start_cycle_count,
    FC_AGGREGATE_REPLACEMENT_COST_CNY,FC_EOL_VOLTAGE_LOSS_UV)
from cpu_checkpoint import load as load_cpu_checkpoint,forward

COMP=('h2_actual','soc_equivalent','fc_degradation','battery_degradation','soc_aux',
      'fc_squared_ramp_aux','failure','shore','legacy_adjustment_correction')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def dump(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def dist(v):
    v=np.asarray(v,dtype=float)
    return dict(count=len(v),mean=float(v.mean()),min=float(v.min()),max=float(v.max()),
        abs_mean=float(abs(v).mean()),abs_median=float(np.median(abs(v))),
        abs_p90=float(np.quantile(abs(v),.9)),abs_p95=float(np.quantile(abs(v),.95)),
        abs_p99=float(np.quantile(abs(v),.99)),signed_sum=float(v.sum()),abs_sum=float(abs(v).sum()))


def main():
    metadata=json.loads((RUN/'run_metadata.json').read_text(encoding='utf-8'))
    hp=metadata['hyperparameters']
    assert hp['n_step']==1 and hp['gamma']==1 and hp['reward_cost_mode']=='zhou_joint'
    paths=[RUN/n for n in ('run_metadata.json','report.json','round_metrics.csv','training_state_latest.pt')]
    paths+=list((ROOT/'src/v4').glob('*.py'))+list((ROOT/'src/v2/models').glob('*.py'))
    paths+=[ROOT/'src/v3/control.py',ROOT/'src/v2/economics.py']
    paths+=[RUN/f'round_100_{kind}_trajectories.json.gz' for kind in ('exploratory','train','validation')]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    if (OUT/'protected_hashes_before.json').exists():
        assert json.loads((OUT/'protected_hashes_before.json').read_text(encoding='utf-8'))==before
    else:dump('protected_hashes_before.json',before)
    config=ZhouEvaluationConfig(hp['economic_evaluation']['fc_average_efficiency'])
    accountant=EconomicMPC(nominal_cost_cny=1.)
    rows=[];voyages=[];checks=defaultdict(float);raw_train={}
    print('Stream-read three R100 archives and reconstruct executed interval ledgers',flush=True)
    for context in ('exploratory','train','validation'):
        records=iter_trajectory_records(RUN/f'round_100_{context}_trajectories.json')
        for section,profile in records:
            status='completed' if profile['completed'] else 'failed_prefix'
            assert (section=='completed')==profile['completed']
            state=AccountState();values=[];fc_sub=[]
            for i,t in enumerate(profile['transitions']):
                assert abs(state.soc-t['soc_before'])<1e-9
                assert abs(state.previous_fc_kw-(t['fc_kw']-t['action_delta_kw']))<1e-9
                nxt,ledger,batt=accountant.interval(state,t['fc_kw'],t['load_kw'],duration_s=t['duration_s'])
                for name in ('h2_cost_cny','fuel_cell_degradation_cost_cny','battery_degradation_cost_cny','shore_cost_cny'):
                    checks[name]=max(checks[name],abs(getattr(ledger,name)-t['original_economic_ledger'][name]))
                checks['soc']=max(checks['soc'],abs(nxt.soc-t['soc_after']))
                checks['battery_kw']=max(checks['battery_kw'],abs(batt-t['battery_bus_kw']))
                soc_eq=soc_equivalent_hydrogen_cost_cny(state.soc,nxt.soc,config)
                soc_aux=hp['beta_soc']*soc_deviation_squared(nxt.soc)
                ramp=FC_SQUARED_RAMP_REWARD_PER_KW2*t['action_delta_kw']**2
                c=np.array([ledger.h2_cost_cny,soc_eq,ledger.fuel_cell_degradation_cost_cny,
                    ledger.battery_degradation_cost_cny,soc_aux,ramp,t['failure_penalty_equivalent_cny'],
                    ledger.shore_cost_cny,-t['immediate_battery_energy_adjustment']-t['terminal_correction']])
                reward=-math.fsum(c.tolist())
                checks['raw_reward']=max(checks['raw_reward'],abs(reward-t['new_reward']))
                checks['scaled_reward']=max(checks['scaled_reward'],abs(hp['reward_scale']*reward-t['scaled_training_reward']))
                for field,val in [('soc_equivalent_hydrogen_cost_cny',soc_eq),('soc_soft_penalty',soc_aux),('fc_ramp_penalty_cny',ramp)]:
                    checks[field]=max(checks[field],abs(t[field]-val))
                for field in ('modeled_fixed_target_shore_ledger','modeled_terminal_settlement'):
                    assert t[field] is None or not any(t[field].values())
                assert not t['immediate_battery_energy_adjustment'] and not t['terminal_correction']
                if t['done']:
                    assert i==len(profile['transitions'])-1 and not t['next_feasible_actions_kw']
                else:assert t['next_feasible_actions_kw']
                raw_fc=formal_aggregate_fc_voltage_loss_step_uv(state.previous_fc_kw,t['fc_kw'],t['duration_s'],600.,
                    is_on=t['fc_kw']>0,mapping=accountant.fc_mapping,
                    aggregate_start_stop_cycles=aggregate_start_cycle_count(state.previous_fc_kw,t['fc_kw']))
                # Verify no EOL clipping before assigning the exact linear four components.
                assert nxt.fc_loss_uv<FC_EOL_VOLTAGE_LOSS_UV
                sub=[getattr(raw_fc,k)*FC_AGGREGATE_REPLACEMENT_COST_CNY/FC_EOL_VOLTAGE_LOSS_UV
                     for k in ('low_runtime_uv','high_runtime_uv','transient_uv','start_stop_uv')]
                checks['fc_four_subterms']=max(checks['fc_four_subterms'],abs(sum(sub)-ledger.fuel_cell_degradation_cost_cny))
                row=dict(context=context,status=status,sample_id=profile['sample_id'],step_1based=i+1,
                    duration_s=t['duration_s'],load_kw=t['load_kw'],soc_before=state.soc,soc_after=nxt.soc,
                    fc_kw=t['fc_kw'],delta_fc_kw=t['action_delta_kw'],done=t['done'],
                    terminal_reason=t['terminal_reason'],raw_reward=reward,scaled_reward=reward*hp['reward_scale'],
                    **dict(zip(COMP,c)),**dict(zip(('fc_low','fc_high','fc_transient','fc_start'),sub)))
                rows.append(row);values.append(c);fc_sub.append(sub)
                if context=='exploratory':
                    raw_train[(profile['sample_id'],i)]=(tuple((state.soc,t['load_kw']/600,state.previous_fc_kw/600)),
                        t['action_index'],reward*hp['reward_scale'])
                state=nxt
            matrix=np.array(values)
            sums=matrix.sum(0);abss=abs(matrix).sum(0)
            assert np.count_nonzero(matrix[:,6])==(not profile['completed'])
            telescope=soc_equivalent_hydrogen_cost_cny(.6,state.soc,config)
            checks['soc_telescoping']=max(checks['soc_telescoping'],abs(sums[1]-telescope))
            economic=math.fsum(sums[:4])
            if profile['completed']:
                checks['zhou_complete_cost']=max(checks['zhou_complete_cost'],abs(economic-profile['economic_evaluation']['total_cost_cny']))
            voyages.append(dict(context=context,status=status,sample_id=profile['sample_id'],steps=len(values),
                terminal_soc=state.soc,prefix_not_complete_cost=not profile['completed'],
                zhou_cost_or_observed_prefix_only=economic,raw_return=-float(sums.sum()),
                scaled_return=-float(sums.sum())*hp['reward_scale'],
                **{f'{k}_signed_sum':float(sums[j]) for j,k in enumerate(COMP)},
                **{f'{k}_abs_sum':float(abss[j]) for j,k in enumerate(COMP)},
                **dict(zip(('fc_low','fc_high','fc_transient','fc_start'),np.array(fc_sub).sum(0)))))
            print(context,profile['sample_id'],len(values),status,flush=True)
    assert max(checks.values())<1e-7,dict(checks)
    table('reward_step_components.csv',rows);table('reward_voyage_totals.csv',voyages)
    # Per-step distributions, signed and absolute totals, and ties-inclusive dominance.
    vals=np.array([[r[c] for c in COMP] for r in rows]);absvals=abs(vals)
    contexts=np.array([r['context'] for r in rows]);statuses=np.array([r['status'] for r in rows])
    load=np.array([r['load_kw'] for r in rows]);soc=np.array([r['soc_after'] for r in rows])
    summary=[]
    for context in ('exploratory','train','validation'):
        for status in ('completed','failed_prefix'):
            base=(contexts==context)&(statuses==status)
            groups=[('all','all',base)]
            for lo,hi in ((0,200),(200,600),(600,1000),(1000,float('inf'))):
                groups.append(('load_kw',f'[{lo},{hi})',base&(load>=lo)&(load<hi)))
            for lo,hi in ((.2,.4),(.4,.6),(.6,.79),(.79,.80000001)):
                groups.append(('soc_after',f'[{lo},{hi})',base&(soc>=lo)&(soc<hi)))
            for group,label,mask in groups:
                if not mask.any():continue
                total=float(absvals[mask].sum());maximum=absvals[mask].max(1)
                for j,c in enumerate(COMP):
                    d=dist(vals[mask,j]);d.update(context=context,status=status,group=group,bin=label,
                        component=c,absolute_component_share=float(absvals[mask,j].sum()/total) if total else 0.,
                        largest_absolute_component_fraction=float(((absvals[mask,j]>=maximum-1e-12)&(maximum>0)).mean()),
                        raw_mean=d['mean'],scaled_mean=d['mean']*hp['reward_scale'],
                        scaled_abs_median=d['abs_median']*hp['reward_scale'],
                        scaled_abs_p95=d['abs_p95']*hp['reward_scale'])
                    summary.append(d)
    table('reward_component_statistics.csv',summary)
    rewardscale=[]
    for c in ('all','completed','failed_prefix'):
        mask=np.ones(len(rows),dtype=bool) if c=='all' else statuses==c
        reward=np.array([r['raw_reward'] for r in rows])[mask]
        for label,scale,beta in [('A',.001,1.),('B',.01,1.),('C',1.,1.),('D',1.,1000.)]:
            rewardscale.append(dict(arm=label,status=c,reward_scale=scale,smooth_l1_beta=beta,
                **dist(reward*scale)))
    table('reward_scale_magnitudes.csv',rewardscale)
    print('Recompute one frozen R100 Train-Replay snapshot on CPU; no optimizer restore or step',flush=True)
    cp=load_cpu_checkpoint(RUN/'training_state_latest.pt')
    assert cp['completed_round']==100 and cp['metadata']==metadata
    agent=cp['agent'];replay=agent['replay']
    online,target=agent['online'],agent['target']
    x=np.array([t.state for t in replay],dtype=np.float32);act=np.array([t.action_index for t in replay])
    def category(t):
        if (t.terminal_reason or '').startswith('failure_'):return 'failure_terminal'
        if t.done:return 'normal_terminal'
        return 'failure_prefix' if t.experience_outcome=='failure' else 'ordinary_nonterminal'
    cats=np.array([category(t) for t in replay]);q=np.empty(len(replay));y=q.copy()
    # Match every R100 exploratory archive reward directly against its real Replay item.
    bysource=defaultdict(list)
    for t in replay:
        if t.source_round==100:bysource[t.source_sample_id].append(t)
    replay_errors=defaultdict(float);matched=0
    for source,items in bysource.items():
        for i,t in enumerate(sorted(items,key=lambda t:t.source_row_index)):
            sx,action,reward=raw_train[(source,i)]
            replay_errors['state']=max(replay_errors['state'],max(abs(v-w) for v,w in zip(sx,t.state)))
            assert action==t.action_index
            replay_errors['scaled_reward']=max(replay_errors['scaled_reward'],abs(reward-t.reward_cny));matched+=1
    assert matched==len(raw_train) and max(replay_errors.values())<1e-9
    for start in range(0,len(replay),2048):
        b=replay[start:start+2048];ns=np.array([t.next_state for t in b],dtype=np.float32)
        mask=np.zeros((len(b),7),dtype=bool)
        for j,t in enumerate(b):mask[j,list(t.next_feasible_indices)]=True
        q[start:start+len(b)]=forward(online,x[start:start+len(b)])[np.arange(len(b)),act[start:start+len(b)]]
        chosen=np.where(mask,forward(online,ns),-np.inf).argmax(1)
        nxt=forward(target,ns)[np.arange(len(b)),chosen]
        reward=np.array([t.reward_cny for t in b],dtype=np.float32)
        done=np.array([t.done for t in b])
        y[start:start+len(b)]=reward+np.where(done,np.float32(0),nxt)
    residual=y-q;tdrows=[];rescale=[]
    # Validate numerical fallback against earlier true-PyTorch snapshot summaries
    # and all seven archived Q values on actual Train015 states.
    old=json.loads((RUN/'checkpoint_audit_round100/replay_root_cause_review.json').read_text(encoding='utf-8'))
    inference_errors={}
    for cat,ref in old['category_summary'].items():
        sel=cats==cat
        assert sel.sum()==ref['count']
        inference_errors[cat]=abs(float(abs(residual[sel]).mean())-ref['absolute_td']['mean'])
    probes=list(csv.DictReader((RUN/'checkpoint_audit_round100/q_action_audit_parent015.csv').open(encoding='utf-8-sig')))
    px=np.array([[float(r[k]) for k in ('state_soc','state_load_over600','state_previous_fc_over600')] for r in probes],dtype=np.float32)
    pq=forward(online,px)
    refq=np.array([[float(r['q_delta_'+f'{d:+d}'+'_kw']) for d in agent['action_values_kw']] for r in probes])
    inference_errors['archived_seven_q_max']=float(abs(pq-refq).max())
    assert max(inference_errors.values())<2e-5,inference_errors
    for cat in ('all','ordinary_nonterminal','normal_terminal','failure_prefix','failure_terminal'):
        mask=np.ones(len(q),dtype=bool) if cat=='all' else cats==cat
        for kind,v in [('q_selected',q[mask]),('target',y[mask]),('td',residual[mask])]:
            tdrows.append(dict(category=cat,quantity=kind,**dist(v),
                quadratic_fraction=float((abs(residual[mask])<1).mean()),
                linear_fraction=float((abs(residual[mask])>=1).mean())))
        for label,scale,beta in [('A',.001,1.),('B',.01,1.),('C',1.,1.),('D',1.,1000.)]:
            v=residual[mask]*(scale/.001);av=abs(v)
            loss=np.where(av<beta,.5*v*v/beta,av-.5*beta)
            rescale.append(dict(arm=label,category=cat,scale=scale,beta=beta,
                scope='Analytical unit conversion of BOTH Q and target, not a new model or training result',
                td_mae=float(av.mean()),quadratic_fraction=float((av<beta).mean()),
                linear_fraction=float((av>=beta).mean()),unweighted_smooth_l1=float(loss.mean())))
    table('snapshot_td_q_target_statistics.csv',tdrows)
    table('analytical_td_rescaling.csv',rescale)
    # Actual final Adam update reconstructed from saved post-update moments.
    opt=agent['optimizer'];groups=opt['param_groups'];assert len(groups)==1
    pg=groups[0];assert pg['weight_decay']==0 and not pg['amsgrad'] and not pg.get('maximize',False)
    b1,b2=pg['betas'];updates=[];parameter_count=0
    parameters=[(k,v) for k,v in online.items() if k.startswith('layers.')]
    for (name,param),idx in zip(parameters,pg['params']):
        st=opt['state'][idx];step=int(st['step'].item())
        m=st['exp_avg'].astype(float)/(1-b1**step);v=st['exp_avg_sq'].astype(float)/(1-b2**step)
        delta=-pg['lr']*m/(np.sqrt(v)+pg['eps'])
        parameter_count+=param.size
        updates.append(dict(parameter=name,numel=param.size,adam_step=step,
            update_l2=float(np.linalg.norm(delta)),parameter_l2=float(np.linalg.norm(param.astype(float))),
            abs_update_mean=float(abs(delta).mean()),abs_update_max=float(abs(delta).max())))
    table('adam_final_step_reconstruction.csv',updates)
    update_norm=math.sqrt(sum(t['update_l2']**2 for t in updates))
    param_norm=math.sqrt(sum(t['parameter_l2']**2 for t in updates))
    dump('verification.json',dict(identity=metadata,rows_reconstructed=len(rows),voyages=len(voyages),
        reconstruction_max_errors=dict(checks),r100_exploratory_replay_items_matched=matched,
        replay_match_max_errors=dict(replay_errors),replay_count=len(replay),
        q_inference_method='Read-only restricted NumPy float32 inference because Torch CUDA DLL import hit WinError1455; validated against prior real-Torch evidence, not bitwise identical',
        q_inference_crosscheck_errors=inference_errors,
        td_within_2e_minus5_of_huber_boundary=int((abs(abs(residual)-1)<2e-5).sum()),
        adam=dict(lr=pg['lr'],betas=pg['betas'],epsilon=pg['eps'],last_step_l2=update_norm,
            last_step_relative_parameter_l2=update_norm/param_norm,parameter_count=parameter_count,
            method='Analytical reconstruction from actual last Adam moments; subject to floating-point application rounding. No optimizer step executed.',
            historical_mean_parameter_update='not logged; cannot reconstruct whole training from final moments'),
        no_training=True,no_optimizer_calls=True,test_payloads_read=0,
        protected_files_unchanged=all(sha(ROOT/k)==v for k,v in before.items())))
    assert all(sha(ROOT/k)==v for k,v in before.items())
    print('DONE: source/checkpoints/archives unchanged',flush=True)


if __name__=='__main__':main()
