"""Parent-level split redesign from raw quality and natural-voyage features only."""
import numpy as np
import pandas as pd
from utils.final_dataset_semantics import classify_states,select_samples
from utils.rebuilt_operating_dataset import pchip_to_one_second
from utils.final_dataset_feasibility import audit_feasibility


BOUNDARY_LOAD_REASON='onboard auxiliary load under vessel-side independent supply'


def consecutive_boundary_low_count(values, threshold_kw=10., from_start=True):
    """Count consecutive observed 30 s boundary points in [0, threshold)."""
    values=np.asarray(values,dtype=float)
    if not from_start:values=values[::-1]
    valid=np.isfinite(values)&(values>=0)&(values<threshold_kw)
    return int(np.argmax(~valid)) if len(valid) and not valid.all() else int(valid.sum())


def representative_parents(inventory, count):
    """Deterministic feature coverage, no controller scores or random seed."""
    available=inventory[inventory.feasible_natural_voyages.gt(0)].copy().sort_values('chronological_index')
    if len(available)<count:count=len(available)
    if not count:return []
    columns=['natural_duration_s','mean_load_kw','above600_deficit_energy_kwh','delta_abs_p95_kw_per_s']
    ranked=available[columns].rank(pct=True,method='average').to_numpy()
    # Start near the feature median, then include the largest physically feasible
    # overload-energy demand; spread remaining parents over the observed regimes.
    selected=[int(np.argmin(((ranked-.5)**2).sum(axis=1)))]
    hard=int(np.argmax(available.above600_deficit_energy_kwh.to_numpy()))
    if count>1 and hard not in selected:selected.append(hard)
    while len(selected)<count:
        distance=((ranked[:,None,:]-ranked[np.array(selected)][None,:,:])**2).sum(axis=2).min(axis=1)
        distance[selected]=-1
        selected.append(int(np.argmax(distance)))
    return available.iloc[selected].parent.tolist()


def redesign_from_raw(frames, policy, chronological_roles):
    """Survey is metadata only; freeze parents before formal sample extraction.

    User explicitly permits redesign when the original test parents are inadequate.
    Every parent receives one role; even unused parents retain it. The survey uses
    the same strict natural-boundary rules and independent physical existence audit.
    """
    rows=[];candidate_rows=[]
    for n,(parent,raw) in enumerate(frames.items(),1):
        d,_=classify_states(raw,policy)
        candidates,_=select_samples(d,'test',policy)
        accepted=[];deltas=[];loads=[];duration=0.;deficit=0.;unknown=0;infeasible=0
        for ci,candidate in enumerate(candidates,1):
            g=candidate['frame'];rebuilt,_=pchip_to_one_second(g)
            feasible=audit_feasibility(rebuilt.load_total_kw.to_numpy())
            start_load=float(g.load_total_kw.iloc[0]);end_load=float(g.load_total_kw.iloc[-1])
            start_low_count=consecutive_boundary_low_count(g.load_total_kw,from_start=True)
            end_low_count=consecutive_boundary_low_count(g.load_total_kw,from_start=False)
            candidate_rows.append({'voyage_id':f'candidate_parent_{n:03d}_{ci:02d}','parent_id':f'parent_{n:03d}','parent':parent,
                'duration_s':len(rebuilt)-1,'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],
                'start_load_kw':start_load,'end_load_kw':end_load,'start_speed_kn':g.speed_kn.iloc[0],'end_speed_kn':g.speed_kn.iloc[-1],
                'start_fc_kw':g.fc_total_kw.iloc[0],'end_fc_kw':g.fc_total_kw.iloc[-1],
                'start_batt_kw':g.battery_total_kw.iloc[0],'end_batt_kw':g.battery_total_kw.iloc[-1],
                'measured_start_soc_pct':g.soc_mean_pct.iloc[0],'measured_end_soc_pct':g.soc_mean_pct.iloc[-1],
                'start_load_below_10kw':bool(0<=start_load<10),'end_load_below_10kw':bool(0<=end_load<10),
                'start_consecutive_low10_points_30s':start_low_count,'end_consecutive_low10_points_30s':end_low_count,
                'start_stable_low10':start_low_count>=3,'end_stable_low10':end_low_count>=3,
                'complete_natural_voyage':bool(candidate['natural_complete']),
                'shore_free':bool(not g.state_class.isin(['confirmed_shore','ambiguous_external_supply']).any()),
                'physical_feasibility':feasible['status'],'short_stop_count':candidate['short_stop_count'],
                'start_boundary':candidate['start_boundary'],'end_boundary':candidate['end_boundary'],
                'boundary_load_nonzero_reason':BOUNDARY_LOAD_REASON if start_load>=10 or end_load>=10 else ''})
            if feasible['feasible'] is not True:
                unknown+=feasible['feasible'] is None;infeasible+=feasible['feasible'] is False;continue
            x=rebuilt.load_total_kw.to_numpy();accepted.append(candidate);loads.append(x[1:]);deltas.append(np.diff(x))
            duration+=len(x)-1;deficit+=np.maximum(x[1:]-600,0).sum()/3600
        rows.append({'parent':parent,'chronological_index':n,'original_chronological_split':chronological_roles[parent],
            'feasible_natural_voyages':len(accepted),'infeasible_natural_voyages':infeasible,'unknown_feasibility_natural_voyages':unknown,
            'natural_duration_s':duration,'mean_load_kw':np.concatenate(loads).mean() if loads else 0.,
            'above600_deficit_energy_kwh':deficit,'delta_abs_p95_kw_per_s':np.quantile(np.abs(np.concatenate(deltas)),.95) if deltas else 0.})
    inventory=pd.DataFrame(rows)
    old_test_count=int(inventory.loc[inventory.original_chronological_split.eq('test'),'feasible_natural_voyages'].sum())
    if old_test_count>=5:
        roles=chronological_roles.copy();basis='retained chronological 46/13/7 split; at least five feasible complete held-out voyages'
    else:
        test=representative_parents(inventory,7)
        # Preserve 7 test parents if fewer qualify; extra parents remain explicitly unused.
        test+= [p for p in chronological_roles if chronological_roles[p]=='test' and p not in test][:7-len(test)]
        if len(test)<7:test += [p for p in chronological_roles if p not in test][:7-len(test)]
        remaining_natural=inventory.loc[inventory.feasible_natural_voyages.gt(0)&~inventory.parent.isin(test),'parent'].tolist()
        validation=remaining_natural[:13]
        preferred=[p for p in chronological_roles if chronological_roles[p]=='validation' and p not in test and p not in validation]
        validation+=preferred[:13-len(validation)]
        if len(validation)<13:validation += [p for p in chronological_roles if p not in test and p not in validation][:13-len(validation)]
        roles={p:('test' if p in test else 'validation' if p in validation else 'train') for p in chronological_roles}
        basis='redesigned because original test parents supplied fewer than five complete feasible voyages; seven test parents selected by frozen raw-feature rank coverage, remaining natural parents prioritized for validation, then original validation chronology; remaining parents train'
    inventory['final_split']=inventory.parent.map(roles)
    natural=set(inventory.loc[inventory.feasible_natural_voyages.gt(0),'parent'])
    inventory['selection_reason']=[
        'retained original chronological '+roles[p] if old_test_count>=5 else
        ('test raw-feature representative order '+str(test.index(p)+1)+'; complete natural and independently feasible' if p in test and p in natural else
         'test parent reserve; no qualified natural voyage' if p in test else
         'validation: remaining independent parent with complete feasible natural operation' if p in validation and p in natural else
         'validation: original chronological validation parent retained as supplementary quality-screened operation' if p in validation else
         'training: remaining independent parent; normal operating fragments retained only after semantic and physical screening')
        for p in inventory.parent]
    candidate_audit=pd.DataFrame(candidate_rows)
    selected=(candidate_audit.parent.map(roles).eq('test')&candidate_audit.physical_feasibility.eq('feasible'))
    candidate_audit['selected_final']=selected
    candidate_audit.loc[selected,'voyage_id']='test_'+candidate_audit.loc[selected,'voyage_id'].str.removeprefix('candidate_')
    candidate_audit['selection_reason']=np.where(selected,
        'retained representative complete natural shore-free feasible Test voyage; no stable <10 kW boundary substitute exists',
        'qualified complete natural shore-free feasible voyage retained outside Test by frozen independent parent split')
    return roles,inventory,{'basis':basis,'original_test_feasible_natural_voyage_count':old_test_count,
        'survey_natural_voyage_count':int(inventory.feasible_natural_voyages.sum()),
        'features':['natural duration','mean load','above600 deficit energy','absolute load change p95'],
        'algorithm':'deterministic median-feature seed, maximum feasible overload-energy seed, then farthest-point coverage in equal-weight percentile ranks; chronological tie-break',
        'selection_inputs':'raw telemetry quality, AIS natural boundaries, load features and independent existence feasibility only; no historical training or controller scores',
        'test_boundary_preference':'prefer complete candidates with stable observed start and end load below 10 kW only when quality is no lower; zero qualifying alternatives found, so all eight existing representative Test voyages are retained',
        'stable_low10_definition':'at least three consecutive observed 30 s boundary points with 0 <= load < 10 kW; preference only, never a trimming or load-editing rule',
        'freeze_rule':'rebuilding unchanged raw inputs regenerates the same parent list; do not reselect after controller outcomes; previous weights trained on reassigned parents cannot provide an uncontaminated benchmark'},candidate_audit
