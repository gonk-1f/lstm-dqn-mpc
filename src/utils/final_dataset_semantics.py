"""AIS-led operation semantics; conservative separation of external supply."""
from __future__ import annotations
import numpy as np
import pandas as pd


def runs(frame, mask, gap_limit):
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    cut = frame.timestamp.diff().dt.total_seconds().gt(gap_limit)
    groups = (mask.ne(mask.shift(fill_value=False)) | cut).cumsum()
    return [g for _, g in frame.loc[mask].groupby(groups[mask], sort=False)]


def _duration(group, cadence):
    return float((group.timestamp.iloc[-1] - group.timestamp.iloc[0]).total_seconds() + cadence)


def operation_pause_mask(d,policy):
    """Near-static jitter and isolated speed spikes do not resume an operation."""
    sailing=pd.Series(False,index=d.index)
    for g in runs(d,d.speed_kn.ge(policy.get('sustained_sailing_speed_kn',1.)),policy['power_gap_limit_s']):
        if _duration(g,policy['power_cadence_s'])>=policy['minimum_sailing_s']:
            sailing.loc[g.index]=True
    return np.isfinite(d.speed_kn) & ~sailing


def derive_operation_policy(frames, timing):
    dt = timing['power_cadence_s']; gap = timing['power_gap_limit_s']
    stationary = .1  # AIS native resolution: 0.1 knot; freeze actual observed distribution below.
    positives = np.concatenate([d.loc[d.speed_kn.gt(0), 'speed_kn'].to_numpy() for d in frames.values()])
    distributions, bracketed = [], []
    for parent, d in frames.items():
        for g in runs(d, d.speed_kn.le(stationary), gap):
            a,b = g.index[0],g.index[-1]
            before = a>0 and d.speed_kn.iloc[a-1]>stationary and (d.timestamp.iloc[a]-d.timestamp.iloc[a-1]).total_seconds()<=gap
            after = b<len(d)-1 and d.speed_kn.iloc[b+1]>stationary and (d.timestamp.iloc[b+1]-d.timestamp.iloc[b]).total_seconds()<=gap
            dur = _duration(g,dt)
            distributions.append({'parent':parent,'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],
                'duration_s':dur,'sailing_before':before,'sailing_after':after,'bracketed':before and after})
    pause_policy={**timing,'minimum_sailing_s':6*dt,'sustained_sailing_speed_kn':1.}
    pause_durations=[]
    for parent,d in frames.items():
        pause=operation_pause_mask(d,pause_policy)
        for g in runs(d,pause,gap):
            a,b=g.index[0],g.index[-1]
            before=a>0 and not pause.iloc[a-1] and np.isfinite(d.speed_kn.iloc[a-1]) and (d.timestamp.iloc[a]-d.timestamp.iloc[a-1]).total_seconds()<=gap
            after=b<len(d)-1 and not pause.iloc[b+1] and np.isfinite(d.speed_kn.iloc[b+1]) and (d.timestamp.iloc[b+1]-d.timestamp.iloc[b]).total_seconds()<=gap
            dur=_duration(g,dt)
            pause_durations.append({'parent':parent,'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],'duration_s':dur,'bracketed_by_sustained_sailing':bool(before and after)})
            if before and after and g.speed_kn.le(stationary).any():bracketed.append(dur)
    if not bracketed:raise ValueError('No bracketed stationary intervals to derive stop timescale')
    # An empirical upper-tail cutoff is used with context, not presented as a natural bimodal boundary.
    stop_max = float(np.ceil(np.quantile(bracketed,.9)/dt)*dt)
    settle = float(np.clip(np.ceil(np.median(bracketed)/dt)*dt,2*dt,10*dt))
    return {**timing,'stationary_speed_kn':stationary,'sustained_sailing_speed_kn':1.,'battery_deadband_kw':1.,'zero_drift_kw':1.,
        'charging_min_duration_s':3*dt,'short_stop_max_s':max(stop_max,settle),
        'terminal_settle_s':settle,'minimum_sailing_s':6*dt,'minimum_sample_s':20*dt,
        'measured_fc_limit_kw':600.,'measured_battery_charge_limit_kw':-624.,'measured_battery_discharge_limit_kw':1248.,
        'measured_power_limit_basis':'specified installed system ratings; FC tolerance is observed aggregate shutdown noise, battery tolerance is 1 kW meter deadband; severe measured violations excluded before net-load feasibility',
        'stationary_duration_quantiles_s':{str(q):float(np.quantile([x['duration_s'] for x in distributions],q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        'bracketed_stop_duration_quantiles_s':{str(q):float(np.quantile(bracketed,q)) for q in [0,.25,.5,.75,.9,.95,.99,1]},
        'stationary_interval_count':len(distributions),'bracketed_stationary_interval_count':len(bracketed),
        'operational_pause_intervals':pause_durations,
        'positive_aligned_speed_q01_kn':float(np.quantile(positives,.01)),
        'stop_basis':'No bimodality asserted. Pause means absence of >=1kn sailing sustained for six measured cycles; isolated AIS jitter cannot restart a task. Bracketed-pause empirical p90 requires sustained sailing on both sides and actual stationary evidence. Terminal dwell retains empirical median pause duration bounded to 2-10 cycles; all thresholds frozen before selection.',
        'engineering_basis':{'stationary_speed_kn':'0.1 kn AIS resolution/near-static deadband; not a load threshold',
            'sustained_sailing_speed_kn':'1 kn = ten AIS resolution increments; excludes near-static drift from positive sailing/source evidence',
            'battery_deadband_kw':'1 kW existing aggregate meter deadband, <0.2% of rated FC; not an external-supply proof',
            'minimum_sample_s':'20 measured cycles (10 min at 30 s), to reject isolated residual fragments',
            'minimum_sailing_s':'6 measured cycles of sustained task evidence',
            'charging_min_duration_s':'3 measured cycles avoids single transient source labels'},
        'charging_certainty_limit':'No independent shore-connected sensor or verified service-load meter. Stationary FC-on charging remains ambiguous. Onboard charging can be confirmed operationally while sustained sailing; no stationary FC-on interval is restored by energy balance alone.'}, distributions


def classify_states(frame, policy):
    d=frame.sort_values('timestamp').reset_index(drop=True).copy()
    d['state_class']='normal_operation'; d['correction']='none'
    valid=d.aligned & np.isfinite(d.load_total_kw) & d.fc_total_kw.ge(0)
    d.loc[~valid,'state_class']='invalid_power_alignment'
    impossible=valid & ((d.fc_total_kw>policy.get('measured_fc_limit_kw',600.)+policy['fc_noise_total_kw']) |
        (d.battery_total_kw<policy.get('measured_battery_charge_limit_kw',-624.)-policy['battery_deadband_kw']) |
        (d.battery_total_kw>policy.get('measured_battery_discharge_limit_kw',1248.)+policy['battery_deadband_kw']))
    d.loc[impossible,'state_class']='measured_power_outside_equipment_envelope'
    valid &= ~impossible
    d.loc[valid & ~np.isfinite(d.speed_kn),'state_class']='unknown_operation_state'
    static=d.speed_kn.le(policy['stationary_speed_kn'])
    off=d.fc_total_kw.le(policy['fc_noise_total_kw'])
    drift=valid & static & off & d.load_total_kw.between(-policy['zero_drift_kw'],0,inclusive='left') & d.battery_total_kw.abs().le(policy['battery_deadband_kw']) & d.inverter_total_kw.abs().le(policy['zero_drift_kw'])
    d.loc[drift,'load_total_kw']=0.;d.loc[drift,'correction']='stationary_sub_kw_zero_drift'
    tables={'confirmed_shore':[],'ambiguous_external_supply':[],'onboard_fc_charging':[]}
    charge=valid & d.battery_total_kw.lt(-policy['battery_deadband_kw']) & np.isfinite(d.speed_kn)
    underway=d.speed_kn.ge(policy.get('sustained_sailing_speed_kn',1.))
    # Near-static drift cannot establish absence of shore supply.
    for is_static in [True,False]:
        for g in runs(d,charge & (~underway if is_static else underway),policy['power_gap_limit_s']):
            enough=_duration(g,policy['power_cadence_s'])>=policy['charging_min_duration_s']
            soc=g.loc[g.soc_channel_count.eq(12),'soc_mean_pct'].dropna()
            rising=len(soc)>=3 and soc.iloc[-1]>soc.iloc[0]
            source_ok=g.fc_total_kw.gt(policy['fc_noise_total_kw']).all() and g.load_total_kw.ge(0).all()
            if is_static and static.loc[g.index].all() and enough and off.loc[g.index].all():
                kind='confirmed_shore';reason='sustained_static_net_charge_FC_total_at_empirical_shutdown_noise_floor'
            elif not is_static and enough and rising and source_ok:
                kind='onboard_fc_charging';reason='sustained_sailing_FC_generation_net_battery_charge_SOC_rise_positive_balance'
            elif is_static:
                kind='ambiguous_external_supply';reason='stationary_charge_without_independent_external_supply_disambiguation'
            else:
                # Transient charging during motion with valid nonnegative supply is normal operation.
                continue
            d.loc[g.index,'state_class']=kind
            tables[kind].append({'parent':g.parent.iloc[0],'start_time':g.timestamp.iloc[0],'end_time':g.timestamp.iloc[-1],
                'duration_s':_duration(g,policy['power_cadence_s']),'point_count':len(g),'reason':reason,
                'mean_fc_kw':g.fc_total_kw.mean(),'mean_battery_kw':g.battery_total_kw.mean(),
                'all_FC_stopped_status_fraction':g.fc_all_stopped.mean(),
                'mean_load_kw':g.load_total_kw.mean(),'start_soc_pct':soc.iloc[0] if len(soc) else np.nan,
                'end_soc_pct':soc.iloc[-1] if len(soc) else np.nan,'speed_min_kn':g.speed_kn.min(),'speed_max_kn':g.speed_kn.max()})
    normal=d.state_class.isin(['normal_operation','onboard_fc_charging'])
    d.loc[normal & d.load_total_kw.lt(0),'state_class']='physical_inconsistency_negative_balance'
    unexplained=normal & off & d.battery_total_kw.abs().le(policy['battery_deadband_kw']) & d.inverter_total_kw.gt(d.load_total_kw.clip(lower=0)+policy['battery_deadband_kw'])
    d.loc[unexplained,'state_class']='unexplained_source_load_inconsistency'
    d['eligible']=d.state_class.isin(['normal_operation','onboard_fc_charging'])
    return d,tables


def _boundary_kind(d,idx,side,policy):
    neighbor=idx-1 if side=='start' else idx+1
    if neighbor<0 or neighbor>=len(d):return 'parent_record_edge'
    if abs((d.timestamp.iloc[idx]-d.timestamp.iloc[neighbor]).total_seconds())>policy['power_gap_limit_s']:
        return 'source_time_gap'
    if not d.eligible.iloc[neighbor]:return str(d.state_class.iloc[neighbor])
    return 'contextual_long_stop_boundary'


def select_samples(frame, role, policy):
    """Return accepted in-memory intervals and a reason for every unselected point."""
    d=frame.copy(); dt=policy['power_cadence_s'];gap=policy['power_gap_limit_s']
    reasons=pd.Series(np.where(d.eligible,'not_selected',d.state_class),index=d.index,dtype=object)
    candidates=[]
    for block in runs(d,d.eligible,gap):
        active=~operation_pause_mask(block,policy)
        if not active.any():
            reasons.loc[block.index]='stationary_only_without_sailing_task';continue
        # First remove dwell cores; keep observed arrival/departure settling points.
        keep=pd.Series(True,index=block.index)
        for stop in runs(block,operation_pause_mask(block,policy),gap):
            a,b=stop.index[0],stop.index[-1]
            before=a>block.index[0] and bool(active.loc[a-1])
            after=b<block.index[-1] and bool(active.loc[b+1])
            if before and after and _duration(stop,dt)<=policy['short_stop_max_s']:
                continue
            retain=pd.Series(False,index=stop.index)
            if before:retain |= stop.timestamp.le(stop.timestamp.iloc[0]+pd.Timedelta(seconds=policy['terminal_settle_s']))
            if after:retain |= stop.timestamp.ge(stop.timestamp.iloc[-1]-pd.Timedelta(seconds=policy['terminal_settle_s']))
            discard=stop.index[~retain];keep.loc[discard]=False;reasons.loc[discard]='long_stop_dwell_core'
        for piece in runs(d,d.index.isin(keep.index[keep]),gap):
            a,b=piece.index[0],piece.index[-1]
            duration=(piece.timestamp.iloc[-1]-piece.timestamp.iloc[0]).total_seconds()
            moving=piece.speed_kn.gt(policy['stationary_speed_kn'])
            start_kind=_boundary_kind(d,a,'start',policy);end_kind=_boundary_kind(d,b,'end',policy)
            # The samples adjacent to a trimmed dwell core have a semantic boundary, not a quality hole.
            if a>0 and reasons.iloc[a-1]=='long_stop_dwell_core':start_kind='departure_after_long_stop'
            if b<len(d)-1 and reasons.iloc[b+1]=='long_stop_dwell_core':end_kind='arrival_before_long_stop'
            front=[];back=[]
            for idx in piece.index:
                if d.speed_kn.iloc[idx]>policy['stationary_speed_kn']:break
                front.append(idx)
            for idx in piece.index[::-1]:
                if d.speed_kn.iloc[idx]>policy['stationary_speed_kn']:break
                back.append(idx)
            natural_start=len(front)>=2 and _duration(d.loc[front],dt)>=2*dt
            natural_end=len(back)>=2 and _duration(d.loc[sorted(back)],dt)>=2*dt
            sailing_runs=runs(piece,piece.speed_kn.ge(policy.get('sustained_sailing_speed_kn',1.)),gap)
            sustained=max((_duration(r,dt) for r in sailing_runs),default=0)>=policy['minimum_sailing_s']
            quality_kinds={'invalid_power_alignment','unknown_operation_state','source_time_gap',
                'physical_inconsistency_negative_balance','unexplained_source_load_inconsistency',
                'measured_power_outside_equipment_envelope','confirmed_shore','ambiguous_external_supply'}
            quality_cut=(start_kind in quality_kinds and not natural_start) or (end_kind in quality_kinds and not natural_end)
            # A short static prefix/suffix next to a quality cut can be an INTERNAL
            # stop. It does not establish a completed task. Test must not use it.
            test_quality_cut=start_kind in quality_kinds or end_kind in quality_kinds
            natural=natural_start and natural_end and sustained and not test_quality_cut
            if duration<policy['minimum_sample_s'] or not sustained:
                reason='insufficient_continuous_operation_duration'
            elif role!='train' and (quality_cut or (role=='validation' and test_quality_cut)):reason='quality_cut_without_natural_transition'
            elif role=='test' and not natural:reason='test_incomplete_natural_voyage'
            else:reason=None
            if reason:
                reasons.loc[piece.index]=reason;continue
            short_count=0
            for stop in runs(piece,operation_pause_mask(piece,policy),gap):
                if stop.index[0]>a and stop.index[-1]<b and stop.speed_kn.le(policy['stationary_speed_kn']).any() and _duration(stop,dt)<=policy['short_stop_max_s']:short_count+=1
            candidates.append({'frame':piece.copy(),'natural_complete':natural,'natural_start':natural_start,
                'natural_end':natural_end,'start_boundary':start_kind,'end_boundary':end_kind,
                'short_stop_count':short_count,'inclusion_reason': 'complete_natural_voyage' if natural else 'quality_verified_parent_window_operation'})
            reasons.loc[piece.index]='selected_candidate'
    return candidates,reasons
