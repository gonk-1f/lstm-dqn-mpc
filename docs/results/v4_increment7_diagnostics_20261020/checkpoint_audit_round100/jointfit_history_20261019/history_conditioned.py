"""Train-only predictive information check, conditioned on actual minimal3 states."""
from diagnose import *


def main():
    design=json.loads((HERE/'design.json').read_text(encoding='utf-8'))
    before=json.loads((HERE/'protected_hashes_before.json').read_text(encoding='utf-8'))
    cp=torch.load(RUN/'training_state_latest.pt',map_location='cpu',weights_only=False)
    # Latest formal exploratory path only, one actual state per source row. No Greedy replay.
    states={}
    for t in cp['agent']['replay']:
        if t.source_round==100:
            key=(t.source_sample_id,t.source_row_index)
            assert key not in states
            states[key]=t.state
    del cp
    dataset=IndependentTrainingDataset.open(ROOT/'data/processed/operating_dataset_independent_v2')
    parent_map=dict(zip(dataset._manifest.candidate_id,dataset._manifest.parent_sample_id))
    held=set(design['heldout_parents'])
    records=[]
    matched=0
    for ep in dataset.load_train():
        loads=np.array([normalize_onboard_load_kw(float(p)) for p in ep.load_kw])
        diff=np.r_[0.,np.diff(loads)]
        prev=np.r_[0.,diff[:-1]]
        for row in range(len(loads)):
            # Formal provenance stores original canonical 1 s rows, not local 30 s indices.
            key=(ep.sample_id,int(ep.source_1s_row[row]))
            if key not in states:
                continue
            s=states[key]
            assert abs(s[1]*600-loads[row])<1e-7
            matched+=1
            for h in (1,8):
                if row+h>=len(loads):
                    continue
                label=float(loads[row+1]) if h==1 else float(np.sum(
                    loads[row+1:row+1+h]*ep.dt_s[row+1:row+1+h])/3600)
                records.append((ep.sample_id,parent_map[ep.sample_id],h,*s,diff[row]/600,prev[row]/600,label))
    assert matched==len(states)
    output=[]; pervoyage=[]; bootstrap=[]
    for h in (1,8):
        r=[v for v in records if v[2]==h]
        names=np.array([v[0] for v in r]); parents=np.array([v[1] for v in r])
        f=np.array([v[3:8] for v in r]); y=np.array([v[8] for v in r])
        fit=~np.isin(parents,sorted(held)); heldparents=parents[~fit]
        predictions={}
        for mode,cols in (('actual_minimal3',[0,1,2]),('minimal3_plus_two_load_differences',[0,1,2,3,4])):
            z=f[:,cols]
            z=(z-z[fit].mean(0))/np.maximum(z[fit].std(0),1e-6)
            xt=np.column_stack((np.ones(fit.sum()),z[fit])); xh=np.column_stack((np.ones((~fit).sum()),z[~fit]))
            reg=np.eye(xt.shape[1]); reg[0,0]=0
            predictions[('ridge_alpha1',mode)]=xh@np.linalg.solve(xt.T@xt+reg,xt.T@y[fit])
            _, idx=cKDTree(z[fit]).query(z[~fit],k=32,workers=1)
            predictions[('knn_k32',mode)]=y[fit][idx].mean(1)
        yh=y[~fit]; nh=names[~fit]
        for (method,mode),p in predictions.items():
            errors=np.abs(p-yh)
            macro=[errors[heldparents==g].mean() for g in sorted(set(heldparents))]
            output.append(dict(horizon_steps=h,target_unit='kW' if h==1 else 'kWh',method=method,
                features=mode, fit_rows=int(fit.sum()),heldout_rows=int((~fit).sum()),
                heldout_mae=float(errors.mean()),heldout_rmse=float(np.sqrt(np.mean((p-yh)**2))),
                heldout_parent_macro_mae=float(np.mean(macro))))
            for name in sorted(set(nh)):
                j=nh==name
                pervoyage.append(dict(horizon_steps=h,method=method,features=mode,sample=name,
                    count=int(j.sum()),mae=float(errors[j].mean())))
        for method in ('ridge_alpha1','knn_k32'):
            e0=np.abs(predictions[(method,'actual_minimal3')]-yh)
            e1=np.abs(predictions[(method,'minimal3_plus_two_load_differences')]-yh)
            # Paired parent bootstrap: uncertainty across 7 heldout parents, not IID intervals.
            diffs=np.array([(e1-e0)[heldparents==g].mean() for g in sorted(set(heldparents))])
            rng=np.random.default_rng(SEED)
            means=diffs[rng.integers(0,len(diffs),size=(2000,len(diffs)))].mean(1)
            bootstrap.append(dict(horizon_steps=h,method=method,heldout_parents=len(diffs),
                macro_delta_mae_history_minus_minimal3=float(diffs.mean()),
                paired_parent_bootstrap_p025=float(np.quantile(means,.025)),
                paired_parent_bootstrap_p975=float(np.quantile(means,.975)),
                parents_improved=int((diffs<0).sum()),parents_worsened=int((diffs>0).sum())))
    table('history_conditioned_metrics.csv',output)
    table('history_conditioned_per_voyage.csv',pervoyage)
    table('history_conditioned_parent_bootstrap.csv',bootstrap)
    after={p:sha256(ROOT/p) for p in before}
    assert before==after
    dump('history_conditioned_verification.json',dict(protected_files_unchanged=True,
        actual_r100_train_states_matched=matched,validation_payloads_read=0,test_payloads_read=0,
        models='Ridge alpha1 and kNN32 only, no Q-network optimization',
        caveat='R100 exploratory path selection; incomplete prefixes are selection-biased. No counterfactual full-voyage states.'))
    print('Conditional history check complete',flush=True)


if __name__=='__main__':
    main()
