"""Read-only statistics of current PER weights; no sampling or optimizer calls."""
from pathlib import Path
import sys,csv,json,hashlib,math
import numpy as np
OUT=Path(__file__).resolve().parent;RUN=OUT.parent;ROOT=RUN.parent.parent
sys.path.insert(0,str(RUN/'reward_scale_audit_20261019'))
from cpu_checkpoint import load
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=list((ROOT/'src/v4').glob('*.py'))+[RUN/'training_state_latest.pt',RUN/'run_metadata.json',RUN/'report.json']
before={str(p.relative_to(ROOT)):sha(p) for p in paths}
cp=load(RUN/'training_state_latest.pt');a=cp['agent'];ps=a['priority_index'];entries=ps['entries']
raw=np.array([v for _,v in entries]);mass=(raw+ps['epsilon'])**ps['alpha'];prob=mass/mass.sum();n=len(raw)
beta=a['sampling_config']['beta'];normalizer=max(1.,(n*prob.min())**(-beta));w=(n*prob)**(-beta)/normalizer
failure_n=sum(str(t.terminal_reason).startswith('failure_') for t in a['replay'])
batch_values=np.r_[w,1/normalizer];batch_prob=np.r_[prob*62/64,2/64]
pool_values=np.r_[w,np.full(failure_n,1/normalizer)]
def quantile(values,weights,q):
    order=np.argsort(values);return float(values[order][np.searchsorted(np.cumsum(weights[order])/weights.sum(),q,side='left')])
rows=[]
for scope,values,p in [('live_pool_uniform_entries',pool_values,np.ones(len(pool_values))/len(pool_values)),('current_expected_quota_batch',batch_values,batch_prob)]:
    rows.append(dict(scope=scope,n=len(values),mean=float(np.dot(values,p)),minimum=float(values.min()),
        p10=quantile(values,p,.1),median=quantile(values,p,.5),p90=quantile(values,p,.9),
        p95=quantile(values,p,.95),p99=quantile(values,p,.99),maximum=float(values.max()),
        note='snapshot analytic distribution, not actual historical batch quantiles'))
with (OUT/'is_weight_snapshot_quantiles.csv').open('x',encoding='utf-8-sig',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
assert all(sha(ROOT/p)==h for p,h in before.items())
with (OUT/'verification.json').open('x',encoding='utf-8') as f:json.dump(dict(protected_hashes=before,protected_files_unchanged=True,
    sampling_config=a['sampling_config'],normalizer=normalizer,minimum_conditional_probability=float(prob.min()),
    min_raw_priority=float(raw.min()),ordinary_pool_count=n,failure_pool_count=failure_n,
    source_round=cp['completed_round'],training_steps=0,optimizer_calls=0,test_payloads=0,
    tau_half_life_updates=math.log(.5)/math.log(.999),tau_half_life_environment_steps=32*math.log(.5)/math.log(.999),
    tau_average_environment_retention_change=1-.999**(1/32)),f,indent=2)
print(rows)
