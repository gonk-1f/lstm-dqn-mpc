"""Read-only action/mask audit of archived states, not policy replay or training."""
from pathlib import Path
import sys,json,csv,hashlib,math
from collections import Counter,defaultdict
import numpy as np

OUT=Path(__file__).resolve().parent
RUN=OUT.parent
ROOT=RUN.parent.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(RUN/'reward_scale_audit_20261019'))
from cpu_checkpoint import load as load_checkpoint
from v4.actions import FC_INCREMENT_KW,FC_ABSOLUTE_KW,candidate_fc_actions,previous_grid_power_kw
from v4.control import feasible_fc_actions,infeasibility_cause,SOC_MIN,SOC_MAX
from v4.trajectory_io import iter_trajectory_records,resolve_trajectory_path
from v4.independent_dataset import IndependentTrainingDataset
from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.models.battery_energy import next_soc
from v3.control import EconomicMPC,AccountState

SETS={'increment7':FC_INCREMENT_KW,'dense13_max60':tuple(range(-60,61,10)),
      'dense25_max120':tuple(range(-120,121,10)),'absolute61':None}
accountant=EconomicMPC(nominal_cost_cny=1.)
plant=accountant.plant
eta_c,eta_d=accountant.efficiency.require_calibrated()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def dump(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def table(name,rows):
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def continuous(soc,load,dt):
    lower_bat=max(plant.battery_charge_min_kw,-(SOC_MAX-soc)*624*3600/(eta_c*dt))
    upper_bat=min(plant.battery_discharge_max_kw,(soc-SOC_MIN)*624*eta_d*3600/dt)
    lo,hi=max(0.,load-upper_bat),min(600.,load-lower_bat)
    return (lo,hi) if lo<=hi else None

def official(soc,load,previous,dt):
    state=AccountState(soc=soc,previous_fc_kw=previous)
    absolute=feasible_fc_actions(state,load,accountant,duration_s=dt)
    choices={k:(absolute if deltas is None else tuple(p for p in absolute if p-previous in deltas))
             for k,deltas in SETS.items()}
    assert choices['increment7']==candidate_fc_actions(absolute,previous,'increment7')
    return absolute,choices

def vector_masks(soc,load,previous,dt):
    power=np.asarray(FC_ABSOLUTE_KW)[None,:]
    bat=load[:,None]-power
    after=soc[:,None]-np.where(bat>0,bat/eta_d,eta_c*bat)*(dt[:,None]/3600)/624
    mask=(bat>=plant.battery_charge_min_kw)&(bat<=plant.battery_discharge_max_kw)&(after>=SOC_MIN)&(after<=SOC_MAX)
    return {k:mask if ds is None else mask&np.isin(power-previous[:,None],ds) for k,ds in SETS.items()}

def diagnosis(soc,load,previous,dt,choices):
    cont=continuous(soc,load,dt)
    if not choices['absolute61']:
        if cont is not None:return 'absolute_grid_limited'
        return 'structural_power' if infeasibility_cause(load,accountant)=='structural_power' else 'soc_limited'
    if choices['increment7']:return 'feasible'
    return 'sparse_increment_hole' if choices['dense13_max60'] else 'max60_reach_limited'

def failure_record(scope,context,sample,round_,idx,episode,soc,previous,load,recorded,choices):
    cont=continuous(soc,load,float(episode.dt_s[idx]))
    cause=diagnosis(soc,load,previous,float(episode.dt_s[idx]),choices)
    nonempty=choices['absolute61']
    nearest=min(nonempty,key=lambda p:abs(p-previous)) if nonempty else None
    return dict(scope=scope,context=context,round=round_,sample_id=sample,
        failed_row_0based=idx,failed_decision_1based=idx+1,
        time_s=float(episode.time_s[idx]),timestamp=str(episode.timestamp[idx]),
        source_1s_row=int(episode.source_1s_row[idx]),duration_s=float(episode.dt_s[idx]),
        soc=soc,load_kw=load,previous_fc_kw=previous,recorded_cause=recorded,
        exact_diagnosis=cause,continuous_fc_min_kw=None if cont is None else cont[0],
        continuous_fc_max_kw=None if cont is None else cont[1],
        current_legal_fc_kw=json.dumps(choices['increment7']),
        absolute_legal_fc_kw=json.dumps(nonempty),
        missing_feasible_delta_kw=json.dumps([p-previous for p in nonempty if p-previous not in FC_INCREMENT_KW]),
        dense13_legal_delta_kw=json.dumps([p-previous for p in choices['dense13_max60']]),
        dense25_legal_delta_kw=json.dumps([p-previous for p in choices['dense25_max120']]),
        nearest_feasible_fc_kw=nearest,nearest_feasible_delta_kw=None if nearest is None else nearest-previous,
        recorded_matches_absolute_grid_reason=recorded==('increment_action_limited' if nonempty else cause))

def main():
    metadata=json.loads((RUN/'run_metadata.json').read_text(encoding='utf-8'))
    report=json.loads((RUN/'report.json').read_text(encoding='utf-8'))
    dataset=IndependentTrainingDataset.open(ROOT/'data/processed/operating_dataset_independent_v2')
    assert dataset.identity_hashes()==metadata['manifest_sha256']
    episodes={e.sample_id:e for e in (*dataset.load_train(),*dataset.load_validation())}
    source_rows={sid:{int(row):idx for idx,row in enumerate(e.source_1s_row)} for sid,e in episodes.items()}
    protected=[RUN/n for n in ('run_metadata.json','report.json','training_state_latest.pt')]
    protected+=list((ROOT/'src/v4').glob('*.py'))
    protected+=[ROOT/'src/v3/control.py',ROOT/'src/v2/config.py',ROOT/'src/v2/models/battery_energy.py']
    protected+=[resolve_trajectory_path(RUN/f'round_100_{c}_trajectories.json') for c in ('exploratory','train','validation')]
    before={str(p.relative_to(ROOT)):sha(p) for p in protected}
    if (OUT/'protected_hashes_before.json').exists():
        assert json.loads((OUT/'protected_hashes_before.json').read_text(encoding='utf-8'))==before
    else:dump('protected_hashes_before.json',before)
    maps=[]
    for previous in (0,20,560,600):
        for idx,delta in enumerate(FC_INCREMENT_KW):
            power=previous+delta
            maps.append(dict(previous_fc_kw=previous,action_index=idx,delta_kw=delta,
                raw_mapped_power_kw=power,in_fc_range=power in FC_ABSOLUTE_KW,
                handling='evaluate physical mask' if power in FC_ABSOLUTE_KW else 'excluded; not clipped',
                duplicates_at_this_previous_power=False))
    if not (OUT/'action_mapping.csv').exists():table('action_mapping.csv',maps)
    # All 61 possible previous grid powers; this is mapping enumeration, not simulation.
    for previous in FC_ABSOLUTE_KW:
        assert previous_grid_power_kw(previous/600*600)==previous
        raw=[previous+d for d in FC_INCREMENT_KW]
        assert len(set(raw))==7
        assert candidate_fc_actions(FC_ABSOLUTE_KW,previous,'increment7')==tuple(p for p in FC_ABSOLUTE_KW if p-previous in FC_INCREMENT_KW)
    rows=[];failures=[];checks=Counter();r100_actual={}
    for context in ('exploratory','train','validation'):
        n=0
        for section,profile in iter_trajectory_records(RUN/f'round_100_{context}_trajectories.json'):
            sid=profile['sample_id'];episode=episodes[sid];ts=profile['transitions']
            previous=0.;soc=.6
            for i,t in enumerate(ts):
                assert abs(t['soc_before']-soc)<1e-12
                assert t['fc_kw']-t['action_delta_kw']==previous
                assert abs(t['load_kw']-normalize_onboard_load_kw(float(episode.load_kw[i])))<1e-9
                assert t['duration_s']==episode.dt_s[i]
                absolute,choices=official(soc,t['load_kw'],previous,t['duration_s'])
                checks['archived_physical_mask_mismatch']+=list(absolute)!=t['physical_feasible_actions_kw']
                checks['archived_increment_mask_mismatch']+=list(choices['increment7'])!=t['policy_candidate_actions_kw']
                assert previous+FC_INCREMENT_KW[t['action_index']]==t['fc_kw']
                assert t['fc_kw'] in choices['increment7']
                ns=next_soc(soc,t['battery_bus_kw'],t['duration_s'],624.,efficiency=accountant.efficiency)
                assert ns==t['soc_after']
                # Verify vectorized exact equations against the official scalar function.
                vm=vector_masks(np.array([soc]),np.array([t['load_kw']]),np.array([previous]),np.array([t['duration_s']]))
                for key in SETS:assert tuple(np.array(FC_ABSOLUTE_KW)[vm[key][0]])==choices[key]
                if not t['done']:
                    _,nc=official(ns,normalize_onboard_load_kw(float(episode.load_kw[i+1])),t['fc_kw'],float(episode.dt_s[i+1]))
                    checks['archived_next_mask_mismatch']+=list(nc['increment7'])!=t['next_feasible_actions_kw']
                else:assert not t['next_feasible_actions_kw']
                row=dict(context=context,status='completed' if profile['completed'] else 'failed_prefix',
                    state_kind='executed',sample_id=sid,decision_1based=i+1,time_s=t['relative_time_s'],
                    soc=soc,load_kw=t['load_kw'],previous_fc_kw=previous,
                    **{k+'_legal_count':len(v) for k,v in choices.items()},
                    continuous_feasible=continuous(soc,t['load_kw'],t['duration_s']) is not None)
                rows.append(row)
                if context=='exploratory':r100_actual[(sid,int(episode.source_1s_row[i]))]=(t['action_index'],t['fc_kw'],t['action_delta_kw'])
                previous=t['fc_kw'];soc=ns;n+=1
            if not profile['completed']:
                i=len(ts);load=normalize_onboard_load_kw(float(episode.load_kw[i]))
                state=(soc,load/600,previous/600)
                assert np.max(abs(np.array(state)-profile['failed_observed_state']))<1e-12
                _,choices=official(soc,load,previous,float(episode.dt_s[i]))
                assert not choices['increment7']
                failures.append(failure_record('r100_archives',context,sid,100,i,episode,soc,previous,load,profile['failure_cause'],choices))
                rows.append(dict(context=context,status='failed_prefix',state_kind='failed_observation',sample_id=sid,
                    decision_1based=i+1,time_s=float(episode.time_s[i]),soc=soc,load_kw=load,previous_fc_kw=previous,
                    **{k+'_legal_count':len(v) for k,v in choices.items()},continuous_feasible=continuous(soc,load,float(episode.dt_s[i])) is not None))
        print('Verified official masks:',context,n,flush=True)
    assert not any(checks.values()),checks
    table('r100_visited_state_masks.csv',rows)
    table('r100_empty_mask_states.csv',failures)
    coverage=[]
    def coverage_rows(scope,maskdict,contexts=None):
        for key,mask in maskdict.items():
            counts=mask.sum(1)
            coverage.append(dict(scope=scope,action_set=key,nominal_action_count=61 if SETS[key] is None else len(SETS[key]),
                max_delta_kw=600 if SETS[key] is None else max(abs(d) for d in SETS[key]),
                observations=len(counts),nonempty=int((counts>0).sum()),empty=int((counts==0).sum()),
                coverage_fraction=float((counts>0).mean()),mean_legal_actions=float(counts.mean())))
    for context in ('exploratory','train','validation'):
        for kind in ('executed','failed_observation'):
            chosen=[r for r in rows if r['context']==context and r['state_kind']==kind]
            for key in SETS:
                counts=np.array([r[key+'_legal_count'] for r in chosen])
                coverage.append(dict(scope=f'r100_{context}_{kind}',action_set=key,
                    nominal_action_count=61 if SETS[key] is None else len(SETS[key]),
                    max_delta_kw=600 if SETS[key] is None else max(abs(d) for d in SETS[key]),
                    observations=len(counts),nonempty=int((counts>0).sum()),empty=int((counts==0).sum()),
                    coverage_fraction=float((counts>0).mean()),mean_legal_actions=float(counts.mean())))
    print('Read-only Replay action/mask audit',flush=True)
    cp=load_checkpoint(RUN/'training_state_latest.pt')
    assert cp['metadata']==metadata and cp['completed_round']==100
    agent=cp['agent'];items=agent['replay']
    assert tuple(agent['action_values_kw'])==FC_INCREMENT_KW
    counts_all={k:[] for k in SETS};replay_fails=[];matched=0;rounds=Counter()
    for start in range(0,len(items),4096):
        batch=items[start:start+4096]
        indices=[source_rows[t.source_sample_id][t.source_row_index] for t in batch]
        duration=np.array([episodes[t.source_sample_id].dt_s[i] for t,i in zip(batch,indices)])
        soc=np.array([t.state[0] for t in batch]);load=np.array([t.state[1]*600 for t in batch])
        previous=np.array([previous_grid_power_kw(t.state[2]*600) for t in batch])
        vm=vector_masks(soc,load,previous,duration)
        for key in SETS:counts_all[key].extend(vm[key].sum(1).tolist())
        nduration=np.array([episodes[t.source_sample_id].dt_s[min(i+1,episodes[t.source_sample_id].step_count-1)] for t,i in zip(batch,indices)])
        nvm=vector_masks(np.array([t.next_state[0] for t in batch]),np.array([t.next_state[1]*600 for t in batch]),
            np.array([previous_grid_power_kw(t.next_state[2]*600) for t in batch]),nduration)
        for j,(t,i) in enumerate(zip(batch,indices)):
            power=previous[j]+FC_INCREMENT_KW[t.action_index]
            assert power in FC_ABSOLUTE_KW and vm['increment7'][j,int(power/10)]
            assert abs(t.next_state[2]*600-power)<1e-9
            if not t.done:
                expected=tuple(k for k,d in enumerate(FC_INCREMENT_KW) if 0<=power+d<=600 and nvm['increment7'][j,int((power+d)/10)])
                assert tuple(t.next_feasible_indices)==expected
            else:assert not t.next_feasible_indices
            if t.source_round==100:
                assert (t.action_index,int(power),FC_INCREMENT_KW[t.action_index])==r100_actual[(t.source_sample_id,t.source_row_index)]
                matched+=1
            rounds[t.source_round]+=1
            if str(t.terminal_reason).startswith('failure_'):
                episode=episodes[t.source_sample_id];n=i+1
                rawload=normalize_onboard_load_kw(float(episode.load_kw[n]))
                assert abs(t.next_state[1]*600-rawload)<1e-9
                ns=t.next_state[0];_,choices=official(ns,rawload,float(power),float(episode.dt_s[n]))
                assert not choices['increment7']
                replay_fails.append(failure_record('final_train_replay','train',t.source_sample_id,t.source_round,
                    n,episode,ns,float(power),rawload,t.terminal_reason.removeprefix('failure_'),choices))
    assert matched==len(r100_actual)
    for key,counts in counts_all.items():
        counts=np.asarray(counts)
        coverage.append(dict(scope='final_train_replay_executed_states',action_set=key,
            nominal_action_count=61 if SETS[key] is None else len(SETS[key]),
            max_delta_kw=600 if SETS[key] is None else max(abs(d) for d in SETS[key]),observations=len(counts),
            nonempty=int((counts>0).sum()),empty=int((counts==0).sum()),coverage_fraction=float((counts>0).mean()),
            mean_legal_actions=float(counts.mean())))
    for key in SETS:
        field={'increment7':'current_legal_fc_kw','absolute61':'absolute_legal_fc_kw',
               'dense13_max60':'dense13_legal_delta_kw','dense25_max120':'dense25_legal_delta_kw'}[key]
        counts=np.array([len(json.loads(r[field])) for r in replay_fails])
        coverage.append(dict(scope='final_train_replay_failure_observations',action_set=key,
            nominal_action_count=61 if SETS[key] is None else len(SETS[key]),
            max_delta_kw=600 if SETS[key] is None else max(abs(d) for d in SETS[key]),observations=len(counts),
            nonempty=int((counts>0).sum()),empty=int((counts==0).sum()),coverage_fraction=float((counts>0).mean()),
            mean_legal_actions=float(counts.mean())))
    table('replay_empty_mask_states.csv',replay_fails)
    table('candidate_coverage.csv',coverage)
    allround=[]
    for r in report['rounds']:
        for context in ('exploratory_train','greedy_train','greedy_validation'):
            for reason,n in r[context].get('failure_reason_counts',{}).items():
                allround.append(dict(round=r['round'],context=context,recorded_reason=reason,count=n,
                    scope='existing report counts; older exact failed states not re-read'))
    table('all_round_logged_failure_reasons.csv',allround)
    assert dataset.identity_hashes()==metadata['manifest_sha256']
    unchanged=all(sha(ROOT/p)==digest for p,digest in before.items());assert unchanged
    dump('verification.json',dict(metadata=metadata,head='dc3c1947bc4dc400710896dfe3acca2e9ae3f54d',
        actual_action_list=list(FC_INCREMENT_KW),plant=dict(fc_max=600,battery_kwh=624,soc_min=SOC_MIN,soc_max=SOC_MAX,
            battery_charge_min_kw=plant.battery_charge_min_kw,battery_discharge_max_kw=plant.battery_discharge_max_kw,
            eta_charge=eta_c,eta_discharge=eta_d),r100_executed_states=len(rows)-len(failures),
        r100_failed_observations=len(failures),archive_mismatch_counts=dict(checks),replay_items=len(items),
        replay_source_round_counts=dict(rounds),replay_failure_observations=len(replay_fails),
        r100_replay_action_matches=matched,mapping_pairs_enumerated=61*7,
        protected_files_unchanged=unchanged,protected_file_count=len(before),test_payloads_opened=dataset.opened_test_payloads,
        no_optimizer_or_policy_execution=True,
        mask_method='R100 and all failed observations call official scalar masks; Replay uses vectorized identical formal SOC/power equations, crosschecked on all R100 executed states',
        witness_scope='existing 43-episode reachability CSV retained as offline whole-voyage existence evidence; not rerun in this audit'))
    print('DONE',len(rows),len(items),len(replay_fails),flush=True)

if __name__=='__main__':main()
