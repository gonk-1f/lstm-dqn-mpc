"""Read-only artifact checks independent of the sample extractor."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def verify_artifacts(root):
    root=Path(root);meta=root/'metadata'
    samples=pd.read_csv(meta/'sample_manifest.csv');parents=pd.read_csv(meta/'parent_split_manifest.csv')
    candidates=pd.read_csv(meta/'test_candidate_boundary_audit.csv')
    source_files=pd.read_csv(meta/'source_files.csv')
    policy=pd.read_json(meta/'policy.json',typ='series')
    parent_roles=parents.set_index('parent').split.to_dict()
    errors=[];seen=set();reference_count=0;formal_count=0;checked_witnesses=0
    formal=samples[samples.split.isin(['train','validation','test'])]
    if not samples.sample_id.is_unique:errors.append('Duplicate sample IDs')
    if not candidates.voyage_id.is_unique:errors.append('Duplicate strict natural-voyage candidate IDs')
    selected_candidates=candidates[candidates.selected_final]
    selected_tests=samples[samples.split.eq('test')]
    if sorted(selected_candidates.voyage_id)!=sorted(selected_tests.sample_id):errors.append('Boundary candidate audit and frozen Test manifest differ')
    required_candidate_truth=['complete_natural_voyage','shore_free']
    if selected_candidates.empty or not selected_candidates[required_candidate_truth].all(axis=None):errors.append('Selected Test boundary candidate lacks complete or shore-free evidence')
    if not selected_candidates.physical_feasibility.eq('feasible').all():errors.append('Selected Test boundary candidate is not physically feasible')
    if not policy.prohibited_test_candidate_source.startswith('total_load_excels/ is permanently prohibited'):
        errors.append('Formal policy does not prohibit the known 8-FC plus BDM source')
    if source_files.path.astype(str).str.contains('total_load_excels',case=False,regex=False).any():errors.append('Prohibited total_load_excels source entered formal source inventory')
    expected_paths={row.relative_path for row in samples.itertuples(index=False)}
    actual_paths={p.relative_to(root).as_posix() for role in ['train','validation','test','stress_cases'] for p in (root/role).glob('*.csv')}
    if expected_paths!=actual_paths:errors.append('Manifest and physical sample files differ')
    for row in samples.itertuples(index=False):
        if row.assigned_split!=parent_roles[row.parent]:errors.append(row.sample_id+': parent assignment mismatch')
        if row.split!='stress_cases' and row.split!=row.assigned_split:errors.append(row.sample_id+': parent leakage')
        d=pd.read_csv(root/row.relative_path,parse_dates=['timestamp'])
        if not np.array_equal(d.time_s.to_numpy(),(d.timestamp-d.timestamp.iloc[0]).dt.total_seconds().to_numpy()):errors.append(row.sample_id+': relative time axis differs from timestamps')
        g=pd.read_csv(meta/'cleaned_30s'/(row.sample_id+'.csv'),parse_dates=['timestamp'])
        if not np.isfinite(d.load_total_kw).all() or d.load_total_kw.lt(0).any():errors.append(row.sample_id+': load invalid')
        if not d.timestamp.diff().dt.total_seconds().iloc[1:].eq(1).all():errors.append(row.sample_id+': 1s grid invalid')
        if g.timestamp.diff().dt.total_seconds().iloc[1:].gt(policy.power_gap_limit_s).any():errors.append(row.sample_id+': source gap crossed')
        if not g.eligible.all() or not np.isfinite(g.speed_kn).all():errors.append(row.sample_id+': excluded state inside sample')
        if g.timestamp.iloc[[0,-1]].tolist()!=d.timestamp.iloc[[0,-1]].tolist():errors.append(row.sample_id+': invented endpoints')
        x=(g.timestamp-g.timestamp.iloc[0]).dt.total_seconds().to_numpy()
        expected=PchipInterpolator(x,g.load_total_kw.to_numpy())(d.time_s.to_numpy())
        if not np.allclose(d.load_total_kw,expected,rtol=0,atol=1e-6):errors.append(row.sample_id+': load differs from source PCHIP')
        pid=parents.loc[parents.parent.eq(row.parent),'parent_id'].iloc[0]
        aligned=pd.read_csv(meta/'aligned_30s'/(pid+'.csv'),parse_dates=['timestamp']).set_index('timestamp')
        original=aligned.reindex(g.timestamp)
        for col in ['load_total_kw','fc_total_kw','battery_total_kw','speed_kn']:
            if not np.allclose(g[col],original[col],rtol=0,atol=1e-8,equal_nan=False):errors.append(row.sample_id+': cleaned source differs from parent '+col)
        if int(row.point_count_1s)!=len(d) or int(row.source_points_30s)!=len(g):errors.append(row.sample_id+': manifest point count mismatch')
        for stamp in g.timestamp:
            key=(row.parent,stamp)
            if key in seen:errors.append(row.sample_id+': overlapping sample ownership')
            seen.add(key)
        if row.split=='test':
            if not row.natural_complete or not g.speed_kn.iloc[[0,-1]].le(policy.stationary_speed_kn).all():errors.append(row.sample_id+': incomplete test')
            if (row.start_load_kw>=10 or row.end_load_kw>=10) and row.boundary_load_nonzero_reason!='onboard auxiliary load under vessel-side independent supply':
                errors.append(row.sample_id+': nonzero Test boundary lacks frozen vessel-side auxiliary-load reason')
            w=pd.read_csv(meta/'feasibility_witnesses'/(row.sample_id+'.csv'))
            load=d.load_total_kw.to_numpy()[1:];fc=w.p_fc_kw.to_numpy();batt=w.p_batt_kw.to_numpy();soc=w.soc.to_numpy()
            if len(w)!=len(load):errors.append(row.sample_id+': witness length mismatch');continue
            if not np.isfinite(w[['p_fc_kw','p_batt_kw','soc']].to_numpy()).all():
                errors.append(row.sample_id+': nonfinite witness');continue
            previous=np.clip(d.load_total_kw.iloc[0],0,600)
            residuals=[np.max(np.abs(fc+batt-load)),max(0,-fc.min(),fc.max()-600),
                max(0,-624-batt.min(),batt.max()-1248),max(0,np.abs(np.diff(np.r_[previous,fc])).max()-48)]
            if max(residuals)>1e-6 or not np.allclose(soc,.55-np.cumsum(batt)/(624*3600),rtol=0,atol=1e-9) or soc.min()<.2-1e-9 or soc.max()>.8+1e-9:
                errors.append(row.sample_id+': serialized witness violates physics')
            checked_witnesses+=1
    for path in sorted((meta/'aligned_30s').glob('*.csv')):
        d=pd.read_csv(path,parse_dates=['timestamp']);reference_count+=len(d)
        accepted=d.disposition.isin(['train','validation','test']);formal_count+=int(accepted.sum())
        if not d.timestamp.is_unique or not d.timestamp.is_monotonic_increasing:errors.append(path.name+': reference time invalid')
        if d.reason.isna().any() or d.reason.isin(['not_selected','selected_candidate']).any():errors.append(path.name+': missing disposition')
        if not d.loc[accepted,'eligible'].all():errors.append(path.name+': ineligible formal source row')
        for sid,g in d[d.sample_id.notna()].groupby('sample_id'):
            if sid not in set(samples.sample_id):errors.append(path.name+': absent sample owner '+sid)
            else:
                m=samples.set_index('sample_id').loc[sid]
                if len(g)!=m.source_points_30s:errors.append(sid+': incomplete ownership ledger')
    accounting=pd.read_csv(meta/'source_point_accounting.csv')
    excluded=pd.read_csv(meta/'exclusion_manifest.csv')
    if accounting.point_count.sum()!=reference_count:errors.append('Reference point accounting mismatch')
    if reference_count-formal_count!=excluded.point_count.sum():errors.append('Exclusion accounting mismatch')
    if formal.source_points_30s.sum()!=formal_count:errors.append('Formal source ownership count mismatch')
    return {'passed':not errors,'errors':errors,'samples_checked':len(samples),'formal_samples_checked':len(formal),
        'test_witnesses_independently_recomputed':checked_witnesses,'reference_points_checked':reference_count,
        'formal_reference_points_checked':formal_count,'checks_include':['parent role consistency','no overlapping source ownership','all formal source states eligible',
            'finite nonnegative load','1 s grid','source gaps','observed endpoints','full source ledger','strict Test candidate boundary audit',
            'prohibited 8-FC plus BDM source absence','serialized test power/SOC/ramp witnesses']}


def legacy_boundary_reconciliation(root, legacy_path):
    """Post-selection references only. Legacy rows never influence eligibility."""
    root=Path(root);meta=root/'metadata'
    old=pd.read_csv(legacy_path,parse_dates=['start_time','end_time'])
    samples=pd.read_csv(meta/'sample_manifest.csv',parse_dates=['start_time','end_time'])
    parents=pd.read_csv(meta/'parent_split_manifest.csv').set_index('parent')
    overlap=[]
    for row in old.itertuples(index=False):
        found=samples[(samples.parent==row.parent_voyage)&(samples.start_time<=row.end_time)&(samples.end_time>=row.start_time)]
        for new in found.itertuples(index=False):
            overlap.append({'legacy_segment_id':row.segment_id,'parent':row.parent_voyage,'new_sample_id':new.sample_id,
                'new_split':new.split,'overlap_start':max(row.start_time,new.start_time),'overlap_end':min(row.end_time,new.end_time),
                'use':'post-selection provenance only, not selection input'})
        if found.empty:overlap.append({'legacy_segment_id':row.segment_id,'parent':row.parent_voyage,'new_sample_id':'','new_split':'excluded','use':'post-selection provenance only'})
    cases=[]
    for a,b in [(148,149),(153,154),(154,155),(157,158),(159,160)]:
        x=old[old.segment_id.eq(f'operating_segment_{a:04d}')].iloc[0]
        y=old[old.segment_id.eq(f'operating_segment_{b:04d}')].iloc[0]
        parent=x.parent_voyage
        if parent!=y.parent_voyage:raise ValueError('Legacy pair is not in one parent')
        d=pd.read_csv(meta/'aligned_30s'/(parents.loc[parent,'parent_id']+'.csv'),parse_dates=['timestamp'])
        middle=d[d.timestamp.gt(x.end_time)&d.timestamp.lt(y.start_time)]
        cover=samples[(samples.parent==parent)&(samples.start_time<=x.end_time)&(samples.end_time>=y.start_time)]
        cases.append({'legacy_A':x.segment_id,'legacy_B':y.segment_id,'parent':parent,'A_end_time':x.end_time,'B_start_time':y.start_time,
            'old_elapsed_cut_s':(y.start_time-x.end_time).total_seconds(),'interior_reference_points':len(middle),
            'interior_all_power_aligned':bool(len(middle)>0 and middle.aligned.all()),
            'interior_state_classes':'|'.join(sorted(middle.state_class.unique())),
            'interior_disposition_reasons':'|'.join(sorted(middle.reason.unique())),
            'same_new_sample_ids':'|'.join(cover.sample_id),'restored_in_formal_sample':bool(cover.split.isin(['train','validation','test']).any()),
            'interior_min_load_kw':middle.load_total_kw.min(),'interior_max_load_kw':middle.load_total_kw.max(),
            'interior_min_speed_kn':middle.speed_kn.min(),'interior_max_speed_kn':middle.speed_kn.max()})
    return overlap,cases
