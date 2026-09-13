"""Cached-only RMS/L2 audit. No MPC, environment or DQN module is imported."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'outputs/ideal_reference_84_train_20260912'
FIXED = ROOT/'outputs/fixed_physical_l2_84_train_20260912'
OUT = ROOT/'outputs/rms_l2_84_train_20260912'
TERMS = ('H','B','S','F')
READS = {}


def record(path):
    READS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def csv(path):
    return pd.read_csv(record(path))


def arrays(path):
    with np.load(record(path)) as z:
        return {k:z[k] for k in z.files}


def write(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')


def dist(v):
    v=np.asarray(v,dtype=float).ravel()
    if not len(v) or not np.isfinite(v).all(): raise ValueError('missing/nonfinite audit values')
    return dict(count=len(v),mean=float(v.mean()),**dict(zip(('min','p01','p50','p99','max'),
                map(float,np.quantile(v,[0,.01,.5,.99,1])))))


def score(values):
    values=np.asarray(values,dtype=float)
    if values.shape[-1]!=4 or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('four finite, nonnegative physical metrics required; no clipping')
    contributions=values/6.
    contributions=contributions.copy()
    contributions[...,0] **= 2
    d2=contributions.sum(axis=-1)
    shares=np.divide(contributions,d2[...,None],out=np.zeros_like(contributions),where=d2[...,None]!=0)
    return 1./(1.+np.sqrt(d2)),np.sqrt(contributions),contributions,shares


def metrics(values):
    r,x,g,c=score(values)
    return {'reward':dist(r),'coordinates':{t:dist(x[...,i]) for i,t in enumerate(TERMS)},
            'd2_terms':{t:dist(g[...,i]) for i,t in enumerate(TERMS)},
            'shares':{t:dist(c[...,i]) for i,t in enumerate(TERMS)},
            'share_over_90pct':{t:float(np.mean(c[...,i]>.9)) for i,t in enumerate(TERMS)},
            'share_over_99pct':{t:float(np.mean(c[...,i]>.99)) for i,t in enumerate(TERMS)}}


def compare(base,other,fc,other_fc):
    ra=score(base)[0];rb=score(other)[0];wa=ra.argmax(-1);wb=rb.argmax(-1);ix=np.arange(len(wa))
    return pd.DataFrame({'winner_changed':wa!=wb,'max_reward_shift':abs(ra-rb).max(-1),
                         'regret_under_other':rb.max(-1)-rb[ix,wa],
                         'first_fc_change_kw':abs(fc[ix,wa,0]-other_fc[ix,wb,0]),
                         'plan_rms_change_kw':np.sqrt(np.mean((fc[ix,wa]-other_fc[ix,wb])**2,axis=-1))})


def comparison_summary(df):
    return {'count':len(df),'winner_changes':int(df.winner_changed.sum()),
            **{c:dist(df[c]) for c in ('max_reward_shift','regret_under_other','first_fc_change_kw','plan_rms_change_kw')}}


def run():
    states=csv(OLD/'states.csv');data=arrays(OLD/'state_physical_results.npz')
    assert states.segment_id.str.startswith('train_').all() and data['valid'][:,4:].all()
    assert np.array_equal(states.audit_id,np.arange(1440))
    values=data['f'][:,4:];fc=data['fc_plans'][:,4:];r,x,g,c=score(values)
    q=np.array([v for v in itertools.product(range(1,8),repeat=4) if sum(v)==10])/10.
    assert q.shape==(84,4)
    write(OUT/'actions.json',[dict(action_id=i,weights=q[i]) for i in range(84)])
    w=r.argmax(1);ix=np.arange(len(states));rank=np.sort(r,axis=1)
    old=csv(FIXED/'state_summary.csv').set_index('audit_id').loc[states.audit_id]
    df=states.copy();df['winner']=w;df['winner_reward']=r[ix,w];df['gap']=rank[:,-1]-rank[:,-2]
    df['relative_gap']=df.gap/df.winner_reward;df['reward_range']=r.max(1)-r.min(1)
    df['winner_fc']=fc[ix,w,0];df['max_grid_fc']=fc[:,:,0].max(1);df['old_winner']=old.winner.to_numpy()
    df['old_winner_fc']=old.winner_fc.to_numpy();df['winner_changed_vs_old']=w!=df.old_winner
    df['winner_soc1']=df.soc-(df.current_load_kw-df.winner_fc)/(3600.*624.)
    df['max_grid_fc_deficit']=df.max_grid_fc-df.winner_fc
    for j,t in enumerate(TERMS):
        df['winner_q'+t]=q[w,j];df['mean_share_'+t]=c[:,:,j].mean(1);df['winner_share_'+t]=c[ix,w,j]
        df['winner_coordinate_'+t]=x[ix,w,j]
    df['load_bin']=pd.cut(df.current_load_kw,[-np.inf,300,600,900,np.inf],labels=['<=300','300-600','600-900','>900']).astype(str)
    df.to_csv(OUT/'state_summary.csv',index=False)
    action=pd.DataFrame({'audit_id':np.repeat(states.audit_id,84),'action_id':np.tile(np.arange(84),len(states)),
                         'reward':r.ravel(),'p_fc0':fc[:,:,0].ravel()})
    for j,t in enumerate(TERMS):
        action[t+'_coordinate']=x[:,:,j].ravel();action[t+'_d2_term']=g[:,:,j].ravel();action[t+'_share']=c[:,:,j].ravel()
    action.to_csv(OUT/'action_scores.csv',index=False)
    paired=[]
    for pair_id,a in df[df.cohort=='matched_soc'].groupby('pair_id'):
        a=a.set_index('soc').loc[[.55,.45,.35,.25,.22]];ids=a.audit_id.to_numpy(dtype=int)
        assert a.current_load_kw.nunique()==a.previous_fc_kw.nunique()==1
        delta=np.diff(a.winner_fc)
        paired.append({'pair_id':pair_id,'all_actions_reward_decreasing':bool((np.diff(r[ids],axis=0)<0).all()),
                       'best_reward_decreasing':bool((np.diff(a.winner_reward)<0).all()),
                       'fc_nondecreasing_01kw':bool((delta>=-.1).all()),'minimum_adjacent_fc_change_kw':float(delta.min()),
                       'fc_022_minus_055':float(a.winner_fc.iloc[-1]-a.winner_fc.iloc[0]),
                       'old_fc_022_minus_055':float(a.old_winner_fc.iloc[-1]-a.old_winner_fc.iloc[0])})
    pd.DataFrame(paired).to_csv(OUT/'paired_soc_checks.csv',index=False)
    socs=[]
    for soc,a in df[df.cohort=='matched_soc'].groupby('soc',sort=False):
        ids=a.audit_id.to_numpy(dtype=int)
        socs.append({'soc':soc,'all_actions':metrics(values[ids]),'winners':metrics(values[ids,w[ids]]),
                     'winner_fc':dist(a.winner_fc),'old_winner_fc':dist(a.old_winner_fc),'gap':dist(a.gap),
                     'max_grid_fc_deficit':dist(a.max_grid_fc_deficit),'winners_count':a.winner.value_counts().to_dict()})
    write(OUT/'matched_soc_summary.json',socs)
    groups=[]
    for keys,a in df.groupby(['cohort','soc','load_bin']):
        groups.append(dict(cohort=keys[0],soc=keys[1],load_bin=keys[2],count=len(a),winner_counts=a.winner.value_counts().to_dict(),
                           q_mean={t:float(a['winner_q'+t].mean()) for t in TERMS},gap=dist(a.gap)))
    write(OUT/'winner_groups.json',groups)
    # Each omission is an offline sensitivity check; all MPC trajectories stay fixed.
    influence=[]
    for j,t in enumerate(TERMS):
        omitted=g.copy();omitted[:,:,j]=0;wo=omitted.sum(-1).argmin(1)
        d=states[['audit_id','cohort','soc']].copy();d['omitted_scoring_term']=t
        d['winner_changed']=w!=wo;d['fc_change_kw']=fc[ix,wo,0]-fc[ix,w,0]
        d['plan_rms_change_kw']=np.sqrt(np.mean((fc[ix,wo]-fc[ix,w])**2,axis=1));influence.append(d)
    influence=pd.concat(influence,ignore_index=True);influence.to_csv(OUT/'term_influence_diagnostic.csv',index=False)
    infl=[]
    for (soc,t),a in influence[influence.cohort=='matched_soc'].groupby(['soc','omitted_scoring_term']):
        infl.append(dict(soc=soc,omitted_scoring_term=t,fc_change=dist(a.fc_change_kw),abs_fc_change=dist(abs(a.fc_change_kw)),
                         fraction_fc_changes_over_01kw=float(np.mean(abs(a.fc_change_kw)>.1))))
    write(OUT/'term_influence_summary.json',infl)
    exact=compare(values,data['exact_f'][:,4:],fc,fc);exact.to_csv(OUT/'exact_soc_sensitivity.csv',index=False)
    perturb=arrays(FIXED/'stability_results.npz');ids=perturb['audit_ids'];sensitivity={}
    for phase in ('warm_forward','warm_reverse','tight_reverse','cold_default'):
        assert perturb[phase+'_valid'].all()
        check=compare(values[ids],perturb[phase+'_f'],fc[ids],perturb[phase+'_fc'])
        sensitivity[phase]=comparison_summary(check);check['audit_id']=ids;check.to_csv(OUT/f'stability_{phase}.csv',index=False)
    write(OUT/'stability_summary.json',sensitivity)
    fp32=r.astype(np.float32);w32=fp32.argmax(1)
    rounding=pd.DataFrame({'audit_id':ix,'winner_changed':w!=w32,'regret':r.max(1)-r[ix,w32],
                           'fc_change_kw':abs(fc[ix,w,0]-fc[ix,w32,0]),
                           'plan_rms_change_kw':np.sqrt(np.mean((fc[ix,w]-fc[ix,w32])**2,axis=1))})
    rounding.to_csv(OUT/'float32_sensitivity.csv',index=False)
    summary={'states':len(df),'actions':84,'all_actions':metrics(values),'winners':metrics(values[ix,w]),
              'winner_counts':df.winner.value_counts().to_dict(),'winner_changes_vs_old':int(df.winner_changed_vs_old.sum()),
              'gap':dist(df.gap),'relative_gap':dist(df.relative_gap),'reward_range':dist(df.reward_range),
              'paired_checks':{key:sum(a[key] for a in paired) for key in ('all_actions_reward_decreasing','best_reward_decreasing','fc_nondecreasing_01kw')},
              'fc_soc_response':dist([a['fc_022_minus_055'] for a in paired]),
              'old_fc_soc_response':dist([a['old_fc_022_minus_055'] for a in paired]),
              'exact_soc':comparison_summary(exact),
              'float32':{'winner_changes':int(rounding.winner_changed.sum()),'regret':dist(rounding.regret),
                         'fc_change_kw':dist(rounding.fc_change_kw),'plan_rms_change_kw':dist(rounding.plan_rms_change_kw)}}
    write(OUT/'summary.json',summary)
    saved=csv(FIXED/'rescored_old_policy_traces.csv');sv=saved[list(TERMS)].to_numpy();rr,xx,gg,cc=score(sv)
    saved['rms_reward']=rr
    for j,t in enumerate(TERMS):saved['rms_share_'+t]=cc[:,j]
    saved.to_csv(OUT/'rescored_old_policy_traces.csv',index=False)
    trace_groups=[]
    for (win,pol),a in saved.groupby(['window','policy']):
        trace_groups.append(dict(window=win,policy=pol,steps=len(a),metrics=metrics(a[list(TERMS)].to_numpy())))
    write(OUT/'old_policy_trace_summary.json',dict(steps=len(saved),windows_with_saved_metrics=int(saved.window.nunique()),
          pooled=metrics(sv),groups=trace_groups,new_policy_rollout=False))
    wd=arrays(FIXED/'window_supplement_results.npz');ws=csv(FIXED/'window_supplement_states.csv')
    wr,wx,wg,wc=score(wd['f']);ww=wr.argmax(1);wi=np.arange(len(ws));rank=np.sort(wr,axis=1)
    ws['winner']=ww;ws['winner_reward']=wr[wi,ww];ws['winner_fc']=wd['fc'][wi,ww,0]
    ws['max_grid_fc']=wd['fc'][:,:,0].max(1);ws['gap']=rank[:,-1]-rank[:,-2]
    ws['old_l2_winner_fc']=csv(FIXED/'window_supplement_summary.csv').winner_fc
    for j,t in enumerate(TERMS):ws['mean_share_'+t]=wc[:,:,j].mean(1)
    ws.to_csv(OUT/'window_supplement_summary.csv',index=False)
    write(OUT/'window_supplement_summary.json',dict(all_actions=metrics(wd['f']),gap=dist(ws.gap),winner_counts=ws.winner.value_counts().to_dict()))
    write(OUT/'stage3_gate.json',{'soc_scale':.05,'persistent_high_S_share_in_old_low_soc_traces':True,
          'H_B_F_control_influence_suppressed':False,'scale_sensitivity_triggered':False,
          'reason':'High score contribution alone does not establish control dominance. At SOC .22, omitting H or F changes all 120 first FC controls by >0.1 kW; omitting B changes 119. New SOC responsiveness is weaker than old, so enlarging the denominator is not supported here.'})
    write(OUT/'provenance.json',dict(reads=READS,mpc_solves=0,soc_scale_candidates_evaluated=[.05],
          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('RESCORED 120960 original + 3360 supplement + 6927 trace scores; reused 40320 perturbed solves; new solves=0',flush=True)


def verify():
    data=arrays(OLD/'state_physical_results.npz');v=data['f'][:,4:];r,coords,terms,c=score(v)
    # Explicit physical examples and independent component-by-component identity.
    assert score(np.array([6.,24.,96.,6.]))[0]==1/(1+np.sqrt(22.))
    assert score(np.zeros(4))[0]==1.
    np.testing.assert_allclose(score([3.,0.,0.,0.])[0],2/3)
    np.testing.assert_allclose(score([0.,0.,96.,0.])[0],.2)
    np.testing.assert_allclose(r,1/(1+np.sqrt((v[:,:,0]/6)**2+v[:,:,1]/6+v[:,:,2]/6+v[:,:,3]/6)),rtol=1e-15)
    np.testing.assert_allclose(coords**2,terms,rtol=1e-14)
    np.testing.assert_allclose(c.sum(-1),1,atol=1e-15)
    scores=csv(OUT/'action_scores.csv');states=csv(OUT/'state_summary.csv')
    np.testing.assert_allclose(scores.reward.to_numpy().reshape(1440,84),r,rtol=1e-14)
    np.testing.assert_array_equal(states.winner,r.argmax(1))
    traces=csv(OUT/'rescored_old_policy_traces.csv')
    assert len(traces)==6927
    tv=traces[list(TERMS)].to_numpy()
    expected=1/(1+np.sqrt((tv[:,0]/6)**2+tv[:,1:].sum(axis=1)/6))
    np.testing.assert_allclose(traces.rms_reward,expected,rtol=1e-14)
    extra=arrays(FIXED/'window_supplement_results.npz')['f']
    er=1/(1+np.sqrt((extra[:,:,0]/6)**2+extra[:,:,1:].sum(axis=2)/6))
    es=csv(OUT/'window_supplement_summary.csv')
    np.testing.assert_allclose(es.winner_reward,er.max(axis=1),rtol=1e-14)
    np.testing.assert_array_equal(es.winner,er.argmax(axis=1))
    influence=csv(OUT/'term_influence_diagnostic.csv')
    low=influence[(influence.cohort=='matched_soc')&(influence.soc==.22)]
    counts={t:int((abs(low[low.omitted_scoring_term==t].fc_change_kw)>.1).sum()) for t in ('H','B','F')}
    assert counts==dict(H=120,B=119,F=120)
    paired=csv(OUT/'paired_soc_checks.csv');assert paired.all_actions_reward_decreasing.all()
    previous=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    changed=[p for p,h in previous.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    write(OUT/'verification.json',dict(protected_files_unchanged=len(previous),scores_verified=120960,
          extra_action_scores_verified=3360,old_policy_trace_scores_verified=6927,stage3_gate_influence_counts=counts,
          physical_hand_checks=True,new_mpc_solves=0,formal_reward_changed=False,dqn_trained=False,
          validation_test_dataset_read=False,soc_denominator_changed=False,stage3_executed=False))
    provenance=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
    for p,h in provenance['reads'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    provenance['verified_final_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(OUT/'provenance.json',provenance)
    print('VERIFIED',len(previous),'protected files unchanged; 120960 original + 3360 supplement + 6927 trace scores checked',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if not args.verify_only:run()
    verify()
