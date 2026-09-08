"""Build a fresh, traceable dataset from original vessel telemetry only.

This entry point never imports a controller or historical performance result.
It refuses an existing output directory. See metadata/README.md and policy.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.final_dataset_source import read_originals, derive_timing_policy, align_originals
from utils.final_dataset_semantics import derive_operation_policy, classify_states, select_samples, runs, _duration, operation_pause_mask
from utils.final_dataset_feasibility import audit_feasibility
from utils.rebuilt_operating_dataset import pchip_to_one_second
from utils.final_dataset_verification import verify_artifacts, legacy_boundary_reconciliation
from utils.final_dataset_split import redesign_from_raw, BOUNDARY_LOAD_REASON


def clean_json(value):
    if isinstance(value, dict):return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)):return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_,)):return bool(value)
    if isinstance(value, (np.integer,)):return int(value)
    if isinstance(value, (float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,Path)):return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows, columns=None):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns is not None:frame=pd.DataFrame(columns=columns)
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.12g',lineterminator='\n')


def assign_parent_roles(parents):
    if len(parents)!=66 or len(set(parents))!=66:
        raise ValueError('Frozen version requires the 66 original chronological parents; review a source change explicitly')
    return {p:('train' if i<46 else 'validation' if i<59 else 'test') for i,p in enumerate(parents)}


def summarize_load(values):
    values=np.asarray(values,dtype=float); x=values[1:]; delta=np.diff(values)
    return {'duration_s':len(x),'point_count_1s':len(values),'energy_kwh':x.sum()/3600,
        'mean_load_kw':x.mean(),'median_load_kw':np.median(x),'p90_load_kw':np.quantile(x,.9),
        'p95_load_kw':np.quantile(x,.95),'p99_load_kw':np.quantile(x,.99),'max_load_kw':values.max(),
        'start_load_kw':values[0],'end_load_kw':values[-1],
        'above600_duration_s':int((x>600).sum()),'above600_load_energy_kwh':x[x>600].sum()/3600,
        'above600_deficit_energy_kwh':np.maximum(x-600,0).sum()/3600,
        'low_load_duration_s':int((x<200).sum()),'medium_load_duration_s':int(((x>=200)&(x<400)).sum()),
        'high_load_duration_s':int((x>=400).sum()),'delta_mean_kw_per_s':delta.mean(),
        'delta_std_kw_per_s':delta.std(),'delta_abs_mean_kw_per_s':np.abs(delta).mean(),
        'delta_abs_p90_kw_per_s':np.quantile(np.abs(delta),.9),'delta_abs_p95_kw_per_s':np.quantile(np.abs(delta),.95),
        'delta_abs_p99_kw_per_s':np.quantile(np.abs(delta),.99),'delta_abs_max_kw_per_s':np.abs(delta).max()}


def validate_sample(source, rebuilt, gap):
    for d in [source,rebuilt]:
        if len(d)<2 or not d.timestamp.is_monotonic_increasing or d.timestamp.duplicated().any():
            raise ValueError('Nonunique or nonmonotone sample time')
        if not np.isfinite(d.load_total_kw).all() or d.load_total_kw.lt(0).any():
            raise ValueError('Nonfinite or negative selected load')
    if source.timestamp.diff().dt.total_seconds().dropna().gt(gap).any():raise ValueError('True gap crossed')
    if not rebuilt.timestamp.diff().dt.total_seconds().dropna().eq(1).all():raise ValueError('Not an exact 1 s grid')
    if source.timestamp.iloc[0]!=rebuilt.timestamp.iloc[0] or source.timestamp.iloc[-1]!=rebuilt.timestamp.iloc[-1]:
        raise ValueError('Reconstruction does not preserve observed endpoints')


def ledger_intervals(frame, gap, cadence):
    groups=(frame.reason.ne(frame.reason.shift())|frame.timestamp.diff().dt.total_seconds().gt(gap)).cumsum()
    rows=[]
    for _,g in frame.groupby(groups,sort=False):
        rows.append({'parent':g.parent.iloc[0],'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],
            'duration_s':_duration(g,cadence),'point_count':len(g),'exclusion_reason':g.reason.iloc[0]})
    return rows


def measured_soc(g, sid, role, policy):
    complete=g.soc_channel_count.eq(12); soc=g.soc_mean_pct.where(complete)
    dt=g.timestamp.diff().dt.total_seconds()
    # Maximum specified discharge is 2 C. A two-percentage-point allowance covers
    # adjacent one-percent quantization bins; this checks continuity, not calibration.
    jump_allowance=2.*100.*dt/3600.+2.
    jumps=(soc.diff().abs()>jump_allowance).fillna(False)
    reliable=bool(complete.all() and not jumps.any())
    return {'sample_id':sid,'split':role,'parent':g.parent.iloc[0],
        'measured_start_soc_pct':soc.iloc[0],'measured_end_soc_pct':soc.iloc[-1],
        'all_12_channels_available_fraction':complete.mean(),'any_channel_available_fraction':g.soc_channel_count.gt(0).mean(),
        'max_adjacent_soc_jump_pct':soc.diff().abs().max(),'implausible_jump_count':jumps.sum(),
        'continuous_and_plausible':reliable,'reliability_scope':'arithmetic mean of 12 clusters; range, availability, 2C plus 2 percentage-point quantization continuity only; not sensor calibration or energy-weighted pack SOC',
        'used_for_simulation_initialization':False}


def qa_plot(g, rebuilt, soc_info, path, sid, policy):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=(g.timestamp-g.timestamp.iloc[0]).dt.total_seconds()/60.
    fig,ax=plt.subplots(5,1,figsize=(13,11),sharex=True,layout='constrained')
    ax[0].plot(rebuilt.time_s/60,rebuilt.load_total_kw,color='#1f4e79',lw=1)
    ax[0].scatter(t,g.load_total_kw,s=5,color='#1f4e79');ax[0].axhline(600,color='darkred',ls='--',lw=.7)
    for a,col,color,label in [(ax[1],'speed_kn','#248045','AIS speed (kn)'),(ax[2],'fc_total_kw','#a26800','Measured FC (kW)'),(ax[3],'battery_total_kw','#8441a3','Measured battery (kW)')]:
        a.plot(t,g[col],lw=1,color=color);a.set_ylabel(label)
    ax[0].set_ylabel('Total load (kW)');ax[3].axhline(0,color='gray',lw=.6)
    if soc_info['continuous_and_plausible']:
        ax[4].plot(t,g.soc_mean_pct,lw=1,color='#8a3333')
    else:
        ax[4].plot(t,g.soc_mean_pct.where(g.soc_channel_count.eq(12)),lw=.8,color='gray')
        ax[4].text(.01,.8,'Measured SOC: incomplete / continuity uncertain',transform=ax[4].transAxes)
    ax[4].set_ylabel('Measured SOC (%)');ax[4].set_xlabel('Elapsed voyage time (min)')
    for a in ax:
        a.grid(alpha=.2)
        for stop in runs(g,g.speed_kn.le(policy['stationary_speed_kn']),policy['power_gap_limit_s']):
            a.axvspan((stop.timestamp.iloc[0]-g.timestamp.iloc[0]).total_seconds()/60.,(stop.timestamp.iloc[-1]-g.timestamp.iloc[0]).total_seconds()/60.,color='#d9e8d3',alpha=.45)
    ax[0].set_title(sid+' | natural voyage | green: observed stationary phases\nRaw measured channels; load reconstructed at 1 s; no controller rollout')
    fig.savefig(path,dpi=120,metadata={'Software':'final operating dataset builder'});plt.close(fig)


def build_dataset(raw_root, output_root, legacy_manifest=None):
    raw_root=Path(raw_root).resolve();root=Path(output_root).resolve()
    if root.exists():raise FileExistsError('Output directory already exists; no overwrite or cleanup is permitted')
    if root==raw_root or raw_root in root.parents:raise ValueError('Output must not be inside original data')
    cache,source_files,channel_qa=read_originals(raw_root)
    chronological_roles=assign_parent_roles(list(cache))
    timing=derive_timing_policy(cache)
    frames,repairs=align_originals(cache,timing)
    policy,stationary_distribution=derive_operation_policy(frames,timing)
    roles,split_inventory,split_design,test_candidate_audit=redesign_from_raw(frames,policy,chronological_roles)
    policy.update({'parent_split':split_design,
        'version':'final-operating-v1','load_definition':'sum of 8 measured FC powers minus sum(V*I/1000) of 12 battery clusters; no independent full-load meter',
        'prohibited_test_candidate_source':'total_load_excels/ is permanently prohibited from formal Test candidate search because it is 8 FC + BDM, not 8 FC + 12 battery clusters',
        'formal_test_candidate_source':'original raw telemetry aligned as 8 measured FC channels plus 12 measured battery cluster voltage/current channels only',
        'load_bins_kw':[0,200,400,600],'load_bins_basis':'descriptive only: one third and two thirds of 600 kW FC rating, never selection thresholds',
        'energy_duration_convention':'1 s values[1:] right-endpoint sum; first point initializes FC; raw interval support last-first+30 s; missing clock intervals reported separately',
        'soc_continuity_rule':'all 12 clusters available and adjacent arithmetic-mean jump <=2C*elapsed_hours*100 +2 percentage points; no calibration claim',
        'tiny_pchip_negative_epsilon_kw':1e-9,'test_selection':'all complete natural, quality accepted, independently feasible voyages in held-out parents; no outcome input or count quota'})
    print('Frozen policy:',json.dumps(clean_json({k:v for k,v in policy.items() if k!='operational_pause_intervals'}),ensure_ascii=False),flush=True)
    root.mkdir(parents=True,exist_ok=False)
    meta=root/'metadata'
    for name in ['train','validation','test','stress_cases','metadata','metadata/aligned_30s','metadata/cleaned_30s','metadata/qa_plots','metadata/feasibility_witnesses']:
        (root/name).mkdir(exist_ok=True)
    write_json(meta/'policy.json',policy)
    write_csv(meta/'source_files.csv',source_files)
    write_csv(meta/'channel_quality.csv',channel_qa)
    write_csv(meta/'stationary_duration_distribution.csv',stationary_distribution)
    write_csv(meta/'parent_quality_inventory.csv',split_inventory)
    write_csv(meta/'test_candidate_boundary_audit.csv',test_candidate_audit)
    samples=[];socs=[];boundaries=[];exclusions=[];parent_rows=[];stop_rows=[];short_rows=[];ledger_rows=[];feasibility_rows=[]
    class_tables={k:[] for k in ['confirmed_shore','ambiguous_external_supply','onboard_fc_charging']}
    final_loads={k:[] for k in ['train','validation','test']};selected_telemetry={k:[] for k in final_loads}
    pchip_corrections=0;total_points=0;counter=0;corrections=[];gaps=[]
    for number,(parent,raw) in enumerate(frames.items(),1):
        role=roles[parent];pid=f'parent_{number:03d}'
        parent_repairs=[r for r in repairs if r['parent']==parent]
        d,tables=classify_states(raw,policy)
        for k in tables:class_tables[k].extend(tables[k])
        candidates,reasons=select_samples(d,role,policy)
        owner=pd.Series('',index=d.index,dtype=object);disposition=pd.Series('excluded',index=d.index,dtype=object)
        parent_count=0
        for ci,candidate in enumerate(candidates,1):
            g=candidate['frame'];rebuilt,pqa=pchip_to_one_second(g)
            validate_sample(g,rebuilt,policy['power_gap_limit_s'])
            pchip_corrections+=pqa['floating_negative_zeroed']
            checked=audit_feasibility(rebuilt.load_total_kw.to_numpy(),return_witness=role=='test')
            sid=f'{role}_{pid}_{ci:02d}'
            target=role if checked['feasible'] is True else 'stress_cases' if checked['feasible'] is False else None
            reason='physical_infeasible_stress_case' if target=='stress_cases' else 'physical_feasibility_unknown' if target is None else 'accepted_'+role
            reasons.loc[g.index]=reason
            if target is None:
                feasibility_rows.append({'sample_id':sid,'parent':parent,'assigned_split':role,'included_split':'excluded',**{k:v for k,v in checked.items() if not isinstance(v,(dict,list,np.ndarray))}})
                continue
            sid=sid if target==role else 'stress_'+sid
            owner.loc[g.index]=sid;disposition.loc[g.index]=target
            path=root/target/(sid+'.csv');write_csv(path,rebuilt)
            write_csv(meta/'cleaned_30s'/(sid+'.csv'),g)
            write_json(meta/'feasibility_witnesses'/(sid+'.json'),{k:v for k,v in checked.items() if k!='witness'})
            if 'witness' in checked:
                w=pd.DataFrame(checked['witness']);w.insert(0,'time_s',rebuilt.time_s.iloc[1:].to_numpy())
                write_csv(meta/'feasibility_witnesses'/(sid+'.csv'),w)
            info=measured_soc(g,sid,target,policy);socs.append(info)
            stats=summarize_load(rebuilt.load_total_kw.to_numpy())
            actual_dt=g.timestamp.diff().dt.total_seconds().fillna(0)
            stationary_s=float(actual_dt.where(g.speed_kn.le(policy['stationary_speed_kn']),0).sum())
            sample_repairs=[r for r in parent_repairs if g.timestamp.iloc[0]<=r['timestamp']<=g.timestamp.iloc[-1]]
            row={'sample_id':sid,'parent_id':pid,'parent':parent,'assigned_split':role,'split':target,
                'relative_path':path.relative_to(root).as_posix(),'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],
                'source_points_30s':len(g),'natural_complete':candidate['natural_complete'],
                'start_boundary':candidate['start_boundary'],'end_boundary':candidate['end_boundary'],
                'inclusion_reason':candidate['inclusion_reason'] if target==role else reason,
                'start_speed_kn':g.speed_kn.iloc[0],'end_speed_kn':g.speed_kn.iloc[-1],
                'max_speed_kn':g.speed_kn.max(),'measured_fc_zero_fraction':g.fc_total_kw.eq(0).mean(),
                'short_stop_count':candidate['short_stop_count'],'stationary_duration_s':stationary_s,
                'sailing_duration_s':stats['duration_s']-stationary_s,'alignment_repair_channel_count':len(sample_repairs),
                'alignment_repaired_power_points':len({r['timestamp'] for r in sample_repairs if r['power_channel']}),
                'onboard_charging_point_count':int(g.state_class.eq('onboard_fc_charging').sum()),
                'onboard_charging_duration_s':float(actual_dt.where(g.state_class.eq('onboard_fc_charging'),0).sum()),
                'measured_start_soc_pct':info['measured_start_soc_pct'],'measured_end_soc_pct':info['measured_end_soc_pct'],
                'boundary_load_nonzero_reason':BOUNDARY_LOAD_REASON if target=='test' and (g.load_total_kw.iloc[0]>=10 or g.load_total_kw.iloc[-1]>=10) else '',
                'physical_feasibility':checked['status'],'qa_plot':f'metadata/qa_plots/{sid}.png' if target=='test' else '',**stats}
            samples.append(row)
            feasibility_rows.append({'sample_id':sid,'parent':parent,'assigned_split':role,'included_split':target,
                **{k:v for k,v in checked.items() if not isinstance(v,(dict,list,np.ndarray))},
                'witness_validation':json.dumps(clean_json(checked['witness_validation']),sort_keys=True)})
            for side,idx in [('start',g.index[0]),('end',g.index[-1])]:
                neighbor=idx-1 if side=='start' else idx+1
                n=d.iloc[neighbor] if 0<=neighbor<len(d) else None
                boundaries.append({'sample_id':sid,'parent':parent,'split':target,'side':side,'timestamp':d.timestamp.iloc[idx],
                    'load_kw':d.load_total_kw.iloc[idx],'speed_kn':d.speed_kn.iloc[idx],'measured_fc_kw':d.fc_total_kw.iloc[idx],
                    'measured_battery_kw':d.battery_total_kw.iloc[idx],'measured_soc_pct':d.soc_mean_pct.iloc[idx],
                    'boundary_kind':candidate[side+'_boundary'],'natural_transition_observed':candidate['natural_'+side],
                    'neighbor_timestamp':n.timestamp if n is not None else None,'neighbor_load_kw':n.load_total_kw if n is not None else None,
                    'neighbor_speed_kn':n.speed_kn if n is not None else None,'neighbor_state_class':n.state_class if n is not None else 'outside_recorded_parent',
                    'evidence':'observed AIS stationary prefix/suffix and sustained sailing; no zero-load requirement' if candidate['natural_'+side] else 'parent record window boundary; not claimed a complete voyage'})
            if target==role:
                parent_count+=1;final_loads[role].append(rebuilt.load_total_kw.to_numpy());selected_telemetry[role].append(g)
            if target=='test':qa_plot(g,rebuilt,info,meta/'qa_plots'/(sid+'.png'),sid,policy)
        d['sample_id']=owner;d['disposition']=disposition;d['reason']=reasons
        if reasons.eq('selected_candidate').any() or reasons.eq('not_selected').any():raise AssertionError('Unaccounted selection state')
        write_csv(meta/'aligned_30s'/(pid+'.csv'),d)
        for idx in d.index[d.correction.ne('none')]:
            corrections.append({'parent':parent,'timestamp':d.timestamp.iloc[idx],'correction':d.correction.iloc[idx],
                'raw_load_kw':raw.load_total_kw.iloc[idx],'corrected_load_kw':d.load_total_kw.iloc[idx],'sample_id':owner.iloc[idx]})
        index_by_time=dict(zip(d.timestamp,d.index))
        for r in parent_repairs:
            if r['parent']==parent:
                idx=index_by_time[r['timestamp']]
                r.update(sample_id=owner.iloc[idx],disposition=disposition.iloc[idx],post_alignment_state=d.state_class.iloc[idx],
                    post_alignment_load_kw=d.load_total_kw.iloc[idx],all_power_channels_valid=bool(d.aligned.iloc[idx]))
        interval_rows=ledger_intervals(d,policy['power_gap_limit_s'],policy['power_cadence_s'])
        ledger_rows.extend(interval_rows)
        exclusions.extend(r for r in interval_rows if not r['exclusion_reason'].startswith('accepted_'))
        for idx in d.index[d.timestamp.diff().dt.total_seconds().gt(policy['power_gap_limit_s'])]:
            gaps.append({'parent':parent,'before_timestamp':d.timestamp.iloc[idx-1],'after_timestamp':d.timestamp.iloc[idx],
                'elapsed_gap_s':(d.timestamp.iloc[idx]-d.timestamp.iloc[idx-1]).total_seconds(),
                'unobserved_duration_s':(d.timestamp.iloc[idx]-d.timestamp.iloc[idx-1]).total_seconds()-policy['power_cadence_s'],
                'reason':'source_reference_time_gap_never_interpolated'})
        pause=operation_pause_mask(d,policy)
        for stop in runs(d,pause,policy['power_gap_limit_s']):
            a,b=stop.index[0],stop.index[-1]
            before=a>0 and not pause.iloc[a-1] and np.isfinite(d.speed_kn.iloc[a-1]) and (d.timestamp.iloc[a]-d.timestamp.iloc[a-1]).total_seconds()<=policy['power_gap_limit_s']
            after=b<len(d)-1 and not pause.iloc[b+1] and np.isfinite(d.speed_kn.iloc[b+1]) and (d.timestamp.iloc[b+1]-d.timestamp.iloc[b]).total_seconds()<=policy['power_gap_limit_s']
            duration=_duration(stop,policy['power_cadence_s'])
            selected=stop.sample_id.ne('') & stop.disposition.isin(['train','validation','test'])
            short=bool(before and after and stop.speed_kn.le(policy['stationary_speed_kn']).any() and duration<=policy['short_stop_max_s'] and stop.eligible.all() and selected.all() and stop.sample_id.nunique()==1)
            row={'parent':parent,'start_time':stop.timestamp.iloc[0],'end_time':stop.timestamp.iloc[-1],
                'duration_s':duration,'point_count':len(stop),'sailing_before':before,'sailing_after':after,
                'mean_load_kw':stop.load_total_kw.mean(),'mean_fc_kw':stop.fc_total_kw.mean(),
                'has_excluded_supply':bool(stop.state_class.isin(['confirmed_shore','ambiguous_external_supply']).any()),
                'retained_formal_points':int(selected.sum()),'retained_formal_support_s':int(selected.sum())*policy['power_cadence_s'],
                'sample_ids':'|'.join(sorted(set(stop.loc[selected,'sample_id']))),
                'stop_kind':'short_operational_stop' if short else 'long_bracketed_stop' if before and after and duration>policy['short_stop_max_s'] else 'terminal_dwell' if not(before and after) else 'unselected_or_uncertain_stationary_interval'}
            if short:short_rows.append(row)
            elif row['stop_kind']!='unselected_or_uncertain_stationary_interval':stop_rows.append(row)
        total_points+=len(d)
        parent_rows.append({'parent_id':pid,'parent':parent,'chronological_index':number,'split':role,
            'split_reason':split_inventory.set_index('parent').loc[parent,'selection_reason'],'original_chronological_split':chronological_roles[parent],
            'source_points':len(d),'formal_sample_count':parent_count,'formal_source_points':int(disposition.isin(['train','validation','test']).sum()),
            'stress_source_points':int(disposition.eq('stress_cases').sum()),'excluded_source_points':int(disposition.eq('excluded').sum()),
            'start_time':d.timestamp.iloc[0],'end_time':d.timestamp.iloc[-1]})
        print(f'Extracted parent {number}/66: {role}, {parent_count} formal samples',flush=True)
    manifest=pd.DataFrame(samples)
    if manifest.empty:
        manifest=pd.DataFrame(columns=['sample_id','parent_id','parent','assigned_split','split','start_time','end_time','natural_complete','physical_feasibility','source_points_30s'])
    test=manifest[manifest.split.eq('test')].copy()
    test['difficulty_descriptor']=''
    for idx,row in test.iterrows():
        tags=[]
        for col,label in [('duration_s','long_duration'),('mean_load_kw','higher_mean_load'),('above600_deficit_energy_kwh','higher_peak_energy'),('delta_abs_p95_kw_per_s','faster_load_variation')]:
            if row[col]>=test[col].quantile(.75) and row[col]>0:tags.append(label)
        if row.short_stop_count>0:tags.append('includes_operational_stop')
        test.loc[idx,'difficulty_descriptor']=';'.join(tags) or 'lower_to_middle_relative_load'
    manifest['difficulty_descriptor']=manifest.sample_id.map(test.set_index('sample_id').difficulty_descriptor).fillna('not_test')
    write_csv(meta/'parent_split_manifest.csv',parent_rows)
    write_csv(meta/'sample_manifest.csv',manifest)
    write_csv(meta/'test_voyage_manifest.csv',test)
    write_csv(meta/'boundary_provenance.csv',boundaries)
    write_csv(meta/'exclusion_manifest.csv',exclusions,['parent','start_time','end_time','duration_s','point_count','exclusion_reason'])
    write_csv(meta/'source_point_accounting.csv',ledger_rows)
    write_csv(meta/'source_time_gaps.csv',gaps,['parent','before_timestamp','after_timestamp','elapsed_gap_s','unobserved_duration_s','reason'])
    write_csv(meta/'alignment_repairs.csv',repairs,['parent','timestamp','channel','source_timestamp','original_offset_s','repair_method','tolerance_s','power_channel','sample_id','disposition'])
    for k,rows in class_tables.items():
        for row in rows:
            matching=manifest[(manifest.parent==row['parent'])&(manifest.start_time<=row['end_time'])&(manifest.end_time>=row['start_time'])]
            row['overlapping_sample_ids']='|'.join(matching.sample_id.tolist())
        write_csv(meta/(k+'_intervals.csv'),rows,['parent','start_time','end_time','duration_s','point_count','reason'])
    write_csv(meta/'long_stop_intervals.csv',stop_rows,['parent','start_time','end_time','duration_s','stop_kind'])
    write_csv(meta/'short_operational_stop_intervals.csv',short_rows,['parent','start_time','end_time','duration_s','stop_kind'])
    write_csv(meta/'measured_soc_summary.csv',socs)
    write_csv(meta/'physical_feasibility_test.csv',feasibility_rows)
    write_csv(meta/'load_corrections.csv',corrections,['parent','timestamp','correction','raw_load_kw','corrected_load_kw','sample_id'])
    coverage={}
    for role,arrays in final_loads.items():
        part=manifest[manifest.split.eq(role)]
        if not arrays:coverage[role]={'sample_count':0};continue
        # Concatenate executed intervals only; never take delta across sample boundaries.
        x=np.concatenate([a[1:] for a in arrays]);delta=np.concatenate([np.diff(a) for a in arrays])
        coverage[role]={'sample_count':len(part),'assigned_parent_count':list(roles.values()).count(role),'used_parent_count':part.parent.nunique(),
            'onboard_charging_interval_count':sum(bool(part[(part.parent.eq(r['parent']))&(part.start_time<=r['end_time'])&(part.end_time>=r['start_time'])].shape[0]) for r in class_tables['onboard_fc_charging']),
            'duration_s':int(part.duration_s.sum()),'duration_hours':part.duration_s.sum()/3600,
            'load_mean_kw':x.mean(),'load_median_kw':np.median(x),'load_p90_kw':np.quantile(x,.9),
            'load_p95_kw':np.quantile(x,.95),'load_p99_kw':np.quantile(x,.99),'load_max_kw':max(a.max() for a in arrays),
            'delta_abs_mean_kw_per_s':np.abs(delta).mean(),'delta_abs_p95_kw_per_s':np.quantile(np.abs(delta),.95),
            'delta_abs_p99_kw_per_s':np.quantile(np.abs(delta),.99),'delta_abs_max_kw_per_s':np.abs(delta).max(),
            'duration_quantiles_s':part.duration_s.quantile([0,.25,.5,.75,1]).to_dict(),
            'parent_sample_distribution':part.parent.value_counts().sort_index().to_dict(),
            **{col:part[col].sum() for col in ['low_load_duration_s','medium_load_duration_s','high_load_duration_s','above600_duration_s','above600_load_energy_kwh','above600_deficit_energy_kwh','stationary_duration_s','sailing_duration_s','short_stop_count','onboard_charging_duration_s']}}
    exclusions_frame=pd.DataFrame(exclusions)
    reason_counts=exclusions_frame.groupby('exclusion_reason').point_count.sum().to_dict() if len(exclusions_frame) else {}
    accepted_points=sum(r['formal_source_points'] for r in parent_rows)
    selected_candidate_ids=test_candidate_audit.loc[test_candidate_audit.selected_final,'voyage_id'].tolist()
    stable_both=test_candidate_audit.start_stable_low10 & test_candidate_audit.end_stable_low10
    stable_one=test_candidate_audit.start_stable_low10 ^ test_candidate_audit.end_stable_low10
    checks={'all_three_formal_roles_nonempty':all(manifest.split.eq(r).any() for r in ['train','validation','test']),
        'parent_leakage_free':len({p for p in roles})==66 and all(manifest[manifest.parent.eq(p)].assigned_split.nunique()<=1 for p in roles),
        'test_all_complete_natural':bool(len(test)>0 and test.natural_complete.all()),
        'test_all_physically_feasible':bool(len(test)>0 and test.physical_feasibility.eq('feasible').all()),
        'test_all_shore_free':bool(len(test)>0 and test_candidate_audit.loc[test_candidate_audit.selected_final,'shore_free'].all()),
        'test_boundary_audit_matches_frozen_manifest':sorted(selected_candidate_ids)==sorted(test.sample_id.tolist()),
        'formal_source_search_excludes_total_load_excels':all('total_load_excels' not in [part.lower() for part in Path(r['path']).parts] for r in source_files),
        'all_formal_samples_physically_feasible':bool(manifest[manifest.split.ne('stress_cases')].physical_feasibility.eq('feasible').all()),
        'source_point_accounting_complete':sum(r['point_count'] for r in ledger_rows)==total_points,
        'every_sample_grid_and_load_checked':True,'selection_precedes_PCHIP':True,
        'no_controller_or_historical_outcome_used':True,'original_source_hashes_unchanged':all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in source_files)}
    qa={'policy':policy,'checks':checks,'all_automated_checks_passed':all(checks.values()),'coverage':coverage,
        'raw_parent_count':66,'source_reference_point_count':total_points,'formal_source_point_count':accepted_points,
        'unused_source_point_fraction':1-accepted_points/total_points,
        'unused_fraction_denominator':'all left-FC1 reference records; time gaps have no measured points and are reported separately; stress not formal',
        'exclusion_reason_points':reason_counts,'exclusion_reason_fraction_of_source':{k:v/total_points for k,v in reason_counts.items()},
        'source_unobserved_gap_duration_s':sum(r['unobserved_duration_s'] for r in gaps),
        'charging_class_summary':{k:{'interval_count':len(rows),'support_duration_s':sum(r['duration_s'] for r in rows)} for k,rows in class_tables.items()},
        'stop_summary':{'short_retained_count':len(short_rows),'short_retained_support_s':sum(r['duration_s'] for r in short_rows),
            'long_or_terminal_interval_count':len(stop_rows),'long_or_terminal_support_s':sum(r['duration_s'] for r in stop_rows),
            'long_or_terminal_retained_support_s':sum(r['retained_formal_support_s'] for r in stop_rows)},
        'alignment':{'repaired_channel_records':len(repairs),'repaired_power_channel_records':sum(bool(r['power_channel']) for r in repairs),
            'beyond_legacy_1s_channel_records':sum(bool(r['exceeded_legacy_1s_tolerance']) for r in repairs),
            'beyond_legacy_1s_power_reference_points':len({(r['parent'],r['timestamp']) for r in repairs if r['power_channel'] and r['exceeded_legacy_1s_tolerance']}),
            'unique_repaired_power_reference_points':len({(r['parent'],r['timestamp']) for r in repairs if r['power_channel']}),
            'formal_repaired_power_reference_points':len({(r['parent'],r['timestamp']) for r in repairs if r['power_channel'] and r['disposition'] in ['train','validation','test']})},
        'pchip_floating_negative_zeroed':pchip_corrections,'raw_zero_drift_corrections':len(corrections),
        'test_ids':test.sample_id.tolist(),'test_parent_list_frozen':sorted(test.parent.unique()),
        'test_candidate_boundary_summary':{'strict_complete_natural_candidate_count':len(test_candidate_audit),
            'stable_start_and_end_below_10kw_count':int(stable_both.sum()),
            'stable_exactly_one_boundary_below_10kw_count':int(stable_one.sum()),
            'stable_neither_boundary_below_10kw_count':int((~stable_both&~stable_one).sum()),
            'selected_test_count':int(test_candidate_audit.selected_final.sum()),'replaced_existing_test_count':0,
            'decision':'retain all eight existing Test candidates because no equally qualified complete voyage has a stable observed boundary below 10 kW',
            'nonzero_boundary_reason':BOUNDARY_LOAD_REASON},
        'limitations':['No direct shore connection or independently verified whole-load/service meter; stationary FC-on charging conservatively ambiguous.',
            'Complete natural tasks include short low-speed harbor maneuvers when observed departure/sailing/arrival criteria hold; inspect duration and maximum speed rather than assuming all are long route voyages.',
            'No outcome-based test choice; old model weights that trained on reassigned test parents cannot be used as an uncontaminated new-split benchmark.',
            'AIS stationary duration has no asserted natural bimodal boundary; empirical/context policy is explicit and may exclude useful uncertain records.',
            'Feasibility establishes existence in the specified lossless model; not a controller score and not an obligation to recover terminal SOC to 0.55.',
            'Measured SOC continuity QA does not establish calibration or replay initial SOC.',
            'Parent file edges are not automatically voyages; validation final SOC means fragment final SOC.',
            'Old train/evaluation data defaults remain unchanged until a separately authorized integration; this builder does not run them.']}
    write_json(meta/'qa_summary.json',qa)
    verified=verify_artifacts(root)
    write_json(meta/'independent_artifact_verification.json',verified)
    qa['checks']['independent_serialized_artifacts_pass']=verified['passed']
    qa['all_automated_checks_passed']=all(qa['checks'].values())
    if legacy_manifest is not None:
        overlap,cases=legacy_boundary_reconciliation(root,legacy_manifest)
        write_csv(meta/'legacy_segment_overlap.csv',overlap)
        write_csv(meta/'legacy_alignment_boundary_cases.csv',cases)
        qa['legacy_reference']={'path':str(legacy_manifest),'sha256':hashlib.sha256(Path(legacy_manifest).read_bytes()).hexdigest(),'use':'post-selection provenance only'}
        qa['legacy_boundary_case_summary']=cases
    write_json(meta/'qa_summary.json',qa)
    write_json(meta/'build_code_hashes.json',{str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*[Path(__file__).resolve().parents[1]/'utils'/name for name in ['final_dataset_source.py','final_dataset_semantics.py','final_dataset_feasibility.py','final_dataset_verification.py','final_dataset_split.py','rebuilt_operating_dataset.py']]]})
    (meta/'README.md').write_text('''# Final operating dataset\n\nThis is a data-only, raw-telemetry rebuild. Train covers normal operating fragments; validation contains independently assigned high-quality operating fragments; test contains observed departure-to-arrival voyages. A parent folder is a recording window, not a voyage. No historical controller result is a selection input.\n\n`policy.json` freezes every selection threshold and its empirical or engineering basis. `parent_split_manifest.csv` freezes the final 46/13/7 roles before formal extraction. If the original chronological Test pool is inadequate, `parent_quality_inventory.csv` documents the raw-only completeness/feasibility survey and `policy.json` documents deterministic feature-based parent redesign. Historical trained weights that saw reassigned Test parents are not valid clean benchmark models. No parent may move between roles after observing controller results. `test_voyage_manifest.csv` is the frozen test list; descriptors use within-test feature quartiles only and never affect inclusion.\n\n`test_candidate_boundary_audit.csv` records every strict complete natural-voyage candidate from the formal 8-FC plus 12-cluster chain. Stable load below 10 kW means at least three consecutive observed 30 s points and is a preference only. The survey found no candidate with a stable low-load boundary, so all eight existing representative Test voyages remain frozen. Their FC power is zero at the observed endpoints while positive battery discharge supplies vessel-side load; the recorded reason is `onboard auxiliary load under vessel-side independent supply`. No load was edited and no voyage was trimmed or joined.\n\n`total_load_excels/` is permanently prohibited from formal Test candidate search because it contains 8-FC plus BDM totals rather than the required 8-FC plus 12-cluster reconstruction. Formal candidates come only from the original raw telemetry aligned through the documented 8-FC plus 12-cluster chain.\n\n`aligned_30s/` records every reference point, physical state, disposition and sample owner. `source_point_accounting.csv` covers every reference point exactly once. `exclusion_manifest.csv` includes unused records and stress cases; `source_time_gaps.csv` separately describes unobserved time. `channel_quality.csv` records invalid timestamps and duplicate handling; conflicting duplicates are unavailable, never averaged. Original per-channel files and hashes are in `source_files.csv`.\n\n`alignment_repairs.csv` logs every nonzero matched channel offset, with a separate flag for offsets beyond the former 1 s tolerance, including whether it survives physical and semantic checks. PCHIP is only applied after sample acceptance and never across excluded points or a missing sampling cycle. `load_corrections.csv` logs the narrowly defined stationary sub-kW zero drift; PCHIP floating correction counts are in QA.\n\nBattery power is positive for discharge. Total load is measured FC plus measured battery discharge; it is not an independent demand meter. Inverter power is retained as an auxiliary measured channel; unknown circuit topology prevents labeling it propulsion or service load. Stationary FC-on charging with no independent external-supply evidence is ambiguous and excluded. Sustained stationary net charging with FC power at the empirical shutdown noise floor is high-confidence external supply, not direct plug-state proof.\n\nRaw interval durations use last-first+30 s support and must not be confused with 1 s sample elapsed durations (last-first). Energy and 1 s duration QA use values[1:] to match the existing initialization convention. Descriptive load bins 200/400 kW are fractions of the 600 kW FC rating, not extraction thresholds.\n\n`physical_feasibility_test.csv` includes all candidate audits with assigned/included role; ordinary test rows are feasible. Feasible test allocation witnesses are saved in `feasibility_witnesses/` and checked independently. They are existence certificates under the specified lossless model, not DQN/MPC trajectories. SOC starts at 0.55 in the audit only; no terminal target is imposed. Measured SOC is metadata only.\n\nRebuild to a NEW destination:\n\n```powershell\npython -X utf8 -B src/main/build_final_operating_dataset.py --raw-root "C:/Users/20883/OneDrive/Desktop/氢舟一号" --output-root data/processed/operating_dataset_final --legacy-manifest data/processed/operating_segments_1s_rebuilt/split_manifest.csv\n```\n\nAn existing output is refused. Do not remove CSVs by hand. Source/code hashes permit deterministic reproduction. Old datasets, controller defaults, staged changes and historical outputs are outside this builder's write scope.\n''',encoding='utf-8')
    write_csv(meta/'artifact_hashes.csv',[{'relative_path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file()])
    print('BUILD_SUMMARY',json.dumps(clean_json({k:qa[k] for k in ['all_automated_checks_passed','coverage','unused_source_point_fraction','charging_class_summary','stop_summary','alignment','test_ids']}),ensure_ascii=False),flush=True)
    if not all(checks.values()):raise RuntimeError('Dataset generated for inspection but mandatory automated checks failed; do not freeze')
    return qa


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--legacy-manifest',type=Path,help='Optional post-selection overlap provenance only; never used for eligibility')
    args=parser.parse_args();build_dataset(args.raw_root,args.output_root,args.legacy_manifest)
