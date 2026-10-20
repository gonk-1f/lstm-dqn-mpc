"""Offline state-information audit. No Q-network fitting or policy execution."""
from pathlib import Path
import sys,json,csv,hashlib
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
OUT=Path(__file__).resolve().parent;RUN=OUT.parent;ROOT=RUN.parent.parent
sys.path[:0]=[str(ROOT/'src'),str(RUN/'reward_scale_audit_20261019')]
from cpu_checkpoint import load,forward
from v4.independent_dataset import IndependentTrainingDataset
from v4.trajectory_io import iter_trajectory_records,resolve_trajectory_path
from v2.data.supervisory_rules import normalize_onboard_load_kw

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def table(n,rows):
    with (OUT/n).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(n,v):
    with (OUT/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def stats(v):
    v=np.asarray(v);return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p95=float(np.quantile(v,.95)),maximum=float(v.max()))

def main():
    meta=json.loads((RUN/'run_metadata.json').read_text(encoding='utf-8'))
    paths=list((ROOT/'src/v4').glob('*.py'))+[RUN/'training_state_latest.pt',RUN/'report.json',RUN/'run_metadata.json']
    paths += [resolve_trajectory_path(RUN/f'round_100_{c}_trajectories.json') for c in ('train','validation')]
    before={str(p.relative_to(ROOT)):sha(p) for p in paths};dump('protected_hashes_before.json',before)
    ds=IndependentTrainingDataset.open(ROOT/'data/processed/operating_dataset_independent_v2')
    assert ds.identity_hashes()==meta['manifest_sha256']
    eps={e.sample_id:e for e in (*ds.load_train(),*ds.load_validation())}
    cp=load(RUN/'training_state_latest.pt');assert cp['completed_round']==100 and cp['metadata']==meta
    agent=cp['agent'];assert meta['hyperparameters']['n_step']==1 and meta['hyperparameters']['gamma']==1
    rows=[];matches=0;max_q_error=0.
    for context in ('train','validation'):
        for section,p in iter_trajectory_records(RUN/f'round_100_{context}_trajectories.json'):
            ts=p['transitions'];ep=eps[p['sample_id']];loads=np.array([normalize_onboard_load_kw(float(l)) for l in ep.load_kw]);diff=np.r_[0.,np.diff(loads)]
            rtg=np.cumsum(np.array([t['scaled_training_reward'] for t in ts])[::-1])[::-1]
            x=np.array([[t['soc_before'],t['load_kw']/600,(t['fc_kw']-t['action_delta_kw'])/600] for t in ts],dtype=np.float32)
            q=forward(agent['online'],x);ns=np.array([[t['soc_after'],loads[min(i+1,len(loads)-1)]/600,t['fc_kw']/600] for i,t in enumerate(ts)],dtype=np.float32)
            nq=forward(agent['online'],ns);nt=forward(agent['target'],ns)
            for i,t in enumerate(ts):
                legal=[j for j,d in enumerate(agent['action_values_kw']) if t['fc_kw']-t['action_delta_kw']+d in t['policy_candidate_actions_kw']]
                order=sorted(legal,key=lambda j:(-q[i,j],j));a=t['action_index'];matches+=order[0]==a
                if t['done']:y=t['scaled_training_reward']
                else:
                    nextlegal=[j for j,d in enumerate(agent['action_values_kw']) if t['fc_kw']+d in t['next_feasible_actions_kw']]
                    b=max(nextlegal,key=lambda j:nq[i,j]);y=t['scaled_training_reward']+float(nt[i,b])
                cat=('normal_terminal' if t['done'] else 'ordinary_nonterminal') if p['completed'] else ('failure_terminal' if t['done'] else 'failure_prefix')
                row=dict(context=context,sample_id=ep.sample_id,parent=ep.parent_sample_id,step=i,soc=t['soc_before'],
                    load_kw=t['load_kw'],previous_fc_kw=t['fc_kw']-t['action_delta_kw'],dload_kw=diff[i],
                    action_index=a,category=cat,observed_suffix_steps=len(ts)-i,
                    phase_fraction_diagnostic_only=float(ep.time_s[i]/ep.duration_s),
                    source_steps_remaining_diagnostic_only=ep.step_count-i,
                    q_actual=float(q[i,a]),target=y,td=y-float(q[i,a]),
                    realized_greedy_return=float(rtg[i]),q_minus_realized=float(q[i,a]-rtg[i]),
                    best_minus_second=float(q[i,order[0]]-q[i,order[1]]) if len(order)>1 else None,
                    greedy_matches_archive=order[0]==a)
                for h in (1,8,32):
                    row[f'future_mean_load_{h}']=float(loads[i+1:i+1+h].mean()) if i+h<len(loads) else None
                rows.append(row)
    table('actual_states_q_returns.csv',rows)
    print('Read actual Train/Validation states',len(rows),'Q matches',matches,flush=True)
    # Check fallback arithmetic against previous actual Torch probes.
    with (RUN/'checkpoint_audit_round100/q_action_audit_parent015.csv').open(encoding='utf-8-sig') as f:probes=list(csv.DictReader(f))
    pq=forward(agent['online'],np.array([[float(r[k]) for k in ('state_soc','state_load_over600','state_previous_fc_over600')] for r in probes],dtype=np.float32))
    ref=np.array([[float(r['q_delta_'+f'{d:+d}'+'_kw']) for d in agent['action_values_kw']] for r in probes]);max_q_error=float(abs(pq-ref).max());assert max_q_error<2e-5
    # No Q update: these fixed diagnostic regressions predict only observed load labels.
    held=set(json.loads((RUN/'checkpoint_audit_round100/jointfit_history_20261019/design.json').read_text(encoding='utf-8'))['heldout_parents'])
    results=[];uncertainty=[]
    for h in (1,8,32):
        rr=[r for r in rows if r[f'future_mean_load_{h}'] is not None]
        x=np.array([[r['soc'],r['load_kw']/600,r['previous_fc_kw']/600,r['dload_kw']/600] for r in rr]);y=np.array([r[f'future_mean_load_{h}'] for r in rr]);par=np.array([r['parent'] for r in rr]);context=np.array([r['context'] for r in rr])
        for split in ('heldout_train_parent','external_validation'):
            fit=(context=='train') & (~np.isin(par,sorted(held)) if split=='heldout_train_parent' else True)
            test=(context=='train')&np.isin(par,sorted(held)) if split=='heldout_train_parent' else context=='validation'
            predictions={}
            for feat,cols in [('load_only',[1]),('load_plus_causal_delta',[1,3]),('minimal3',[0,1,2]),('minimal3_plus_causal_delta',[0,1,2,3])]:
                z=x[:,cols];z=(z-z[fit].mean(0))/np.maximum(z[fit].std(0),1e-6)
                xt=np.c_[np.ones(fit.sum()),z[fit]];xh=np.c_[np.ones(test.sum()),z[test]];reg=np.eye(xt.shape[1]);reg[0,0]=0
                ridge=xh@np.linalg.solve(xt.T@xt+reg,xt.T@y[fit])
                _,idx=cKDTree(z[fit]).query(z[test],k=32,workers=1);neighbor=y[fit][idx];knn=neighbor.mean(1)
                for method,pred in [('ridge_alpha1',ridge),('knn32',knn)]:
                    er=pred-y[test];predictions[(feat,method)]=abs(er)
                    results.append(dict(split=split,horizon_steps=h,label='next load kW' if h==1 else 'future mean load kW',method=method,features=feat,fit_rows=int(fit.sum()),evaluation_rows=int(test.sum()),mae=float(abs(er).mean()),rmse=float(np.sqrt((er**2).mean())),residual_variance=float(er.var()),parent_macro_mae=float(np.mean([abs(er)[par[test]==p].mean() for p in set(par[test])]))))
                uncertainty.append(dict(split=split,horizon_steps=h,features=feat,
                    mean_neighbor_label_std=float(neighbor.std(1).mean()),
                    empirical_neighbor_p90_p10_width=float((np.quantile(neighbor,.9,axis=1)-np.quantile(neighbor,.1,axis=1)).mean()),
                    heldout_in_neighbor_p10_p90=float(((y[test]>=np.quantile(neighbor,.1,axis=1))&(y[test]<=np.quantile(neighbor,.9,axis=1))).mean()),
                    scope='local empirical label spread; not calibrated conditional entropy'))
    table('causal_history_prediction.csv',results);table('conditional_neighbor_dispersion.csv',uncertainty)
    # Validation states matched to nearest states of different Train parents.
    tr=[r for r in rows if r['context']=='train'];va=[r for r in rows if r['context']=='validation']
    feature=lambda rr:np.array([[r['soc'],r['load_kw']/600,r['previous_fc_kw']/600] for r in rr])
    tree=cKDTree(feature(tr));distance,index=tree.query(feature(va),k=32,workers=1)
    pairs=[]
    for v,dd,ii in zip(va,distance,index):
        for kind,k in [('nearest',0),('nearest_same_action',next((j for j,z in enumerate(ii) if tr[z]['action_index']==v['action_index']),None))]:
            if k is None:continue
            t=tr[ii[k]]
            row=dict(pair_kind=kind,validation_sample=v['sample_id'],validation_step=v['step'],train_sample=t['sample_id'],train_step=t['step'],
                normalized_euclidean_distance=float(dd[k]),delta_soc=v['soc']-t['soc'],delta_load_kw=v['load_kw']-t['load_kw'],delta_previous_fc_kw=v['previous_fc_kw']-t['previous_fc_kw'],delta_causal_load_change_kw=v['dload_kw']-t['dload_kw'],
                validation_category=v['category'],train_category=t['category'],validation_action=v['action_index'],train_action=t['action_index'],
                delta_phase_diagnostic_only=v['phase_fraction_diagnostic_only']-t['phase_fraction_diagnostic_only'],
                delta_observed_suffix_steps=v['observed_suffix_steps']-t['observed_suffix_steps'],
                delta_realized_return=v['realized_greedy_return']-t['realized_greedy_return'],delta_target=v['target']-t['target'])
            for h in (1,8,32):row[f'delta_future_mean_load_{h}']=v[f'future_mean_load_{h}']-t[f'future_mean_load_{h}'] if v[f'future_mean_load_{h}'] is not None and t[f'future_mean_load_{h}'] is not None else None
            pairs.append(row)
    table('near_state_pairs.csv',pairs)
    grouped=[]
    for context in ('train','validation'):
        for cat in ('ordinary_nonterminal','normal_terminal','failure_prefix','failure_terminal'):
            rr=[r for r in rows if r['context']==context and r['category']==cat]
            if not rr:continue
            for field in ('q_actual','target','td','q_minus_realized'):
                a=np.array([r[field] for r in rr]);grouped.append(dict(context=context,category=cat,field=field,**stats(a),abs_mae=float(abs(a).mean())))
        for lo,hi in ((1,8),(9,32),(33,100),(101,100000)):
            rr=[r for r in rows if r['context']==context and r['category']=='failure_prefix' and lo<=r['observed_suffix_steps']<=hi]
            if rr:grouped.append(dict(context=context,category=f'failure_observed_suffix_{lo}_{hi}',field='q_minus_realized',**stats([r['q_minus_realized'] for r in rr]),abs_mae=float(np.mean([abs(r['q_minus_realized']) for r in rr]))))
    table('category_q_return_diagnostics.csv',grouped)
    hist=json.loads((RUN/'report.json').read_text(encoding='utf-8'));history=[]
    for r in hist['rounds']:
        history.append(dict(round=r['round'],td_mae=r['td_statistics']['mean_absolute_td_error'],train_completed=r['greedy_train']['completed'],validation_completed=r['greedy_validation']['completed']))
    table('td_vs_completion.csv',history)
    assert all(sha(ROOT/p)==v for p,v in before.items()) and ds.identity_hashes()==meta['manifest_sha256']
    dump('verification.json',dict(metadata=meta,state_rows=len(rows),q_greedy_matches=matches,real_torch_probe_max_difference=max_q_error,
        protected_files_unchanged=True,test_payloads_read=0,policy_replays=0,q_updates=0,
        diagnostics='Fixed Ridge alpha1/kNN32 regressions only; future loads and phase are labels/stratifiers, never policy inputs',
        validation_use='external descriptive diagnostic only; not fit or model selection, no hyperparameter search'))
    print('DONE',flush=True)
if __name__=='__main__':main()
