"""Read-only small-load evidence over the frozen 35 Train voyages.

Bins describe the data; they do not define a new operational deadband.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def bin_name(value):
    value = abs(float(value))
    if not isfinite(value):
        raise ValueError('load must be finite')
    if value == 0:
        return 'exact_zero'
    for limit, label in ((1e-6, '(0,1e-6]'), (1e-4, '(1e-6,1e-4]'),
                         (.001, '(1e-4,0.001]'), (.01, '(0.001,0.01]'),
                         (.1, '(0.01,0.1]'), (1., '(0.1,1]')):
        if value <= limit:
            return label
    return '>1'


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_target(manifest_row, raw_root, source_rows):
    """Replay preprocessing only for the explicitly identified Train parent."""
    import numpy as np
    import pandas as pd
    from scipy.interpolate import PchipInterpolator
    from v2.data.segment_power_source import load_parent_power_series
    from v2.data.power_gap_interpolation import interpolate_power_gaps
    from v2.data.zero_boundary_dataset import trim_to_zero_boundaries, reconstruct_one_second
    if manifest_row['split'] != 'train' or manifest_row['candidate_id'] != 'parent_015_seg01':
        raise ValueError('source trace requires the selected Train parent identity')
    parent = manifest_row['parent_name']
    measured = load_parent_power_series(raw_root, parent)
    filled = interpolate_power_gaps(measured.timestamps, measured.fc_total_kw,
                                   measured.battery_raw_total_kw, step_seconds=30.)
    anchor = pd.DataFrame(dict(timestamp=filled.timestamps, fc_total_kw=filled.fc_kw,
        battery_raw_total_kw=filled.battery_raw_kw, source_total_kw=filled.source_kw,
        is_cubic_imputed=filled.is_interpolated))
    trimmed = trim_to_zero_boundaries(anchor)
    rebuilt, qa = reconstruct_one_second(trimmed.frame)
    original = np.array([float(row['load_total_kw']) for row in source_rows])
    values = rebuilt['load_total_kw'].to_numpy(dtype=float)
    first_three = trimmed.frame.iloc[:3].copy()
    x = (first_three.timestamp-first_three.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    interp = PchipInterpolator(x, first_three.source_total_kw.to_numpy())
    # Component telemetry is aligned and summed; it is not an independent load sensor.
    boundary_stamp = trimmed.frame.timestamp.iloc[0]
    measured_positions = [i for i, stamp in enumerate(measured.timestamps) if pd.Timestamp(stamp) == boundary_stamp]
    raw_at_boundary = []
    for i in measured_positions:
        raw_at_boundary.append(dict(timestamp=pd.Timestamp(measured.timestamps[i]).isoformat(),
            fc_kw=float(measured.fc_total_kw[i]), battery_raw_kw=float(measured.battery_raw_total_kw[i]),
            battery_bus_kw=-float(measured.battery_raw_total_kw[i]),
            derived_total_kw=float(measured.source_total_kw[i])))
    return dict(parent=parent, sample_id=manifest_row['candidate_id'],
        reconstructed_nodes=len(values), source_nodes=len(original),
        source_reconstruction_max_abs_difference_kw=float(np.max(abs(values-original))),
        target_timestamp=rebuilt.timestamp.iloc[1].isoformat(),
        target_pchip_kw=float(values[1]), first_three_anchor_pchip_t1_kw=float(interp(1.)),
        first_interval_coefficients_descending=interp.c[:,0].tolist(),
        aligned_derived_sample_exactly_at_target=any(pd.Timestamp(stamp)==rebuilt.timestamp.iloc[1] for stamp in measured.timestamps),
        aligned_measured_points=len(measured.timestamps), cubic_gap_count=filled.interpolation_gap_count,
        cubic_imputed_points=filled.interpolated_point_count,
        trimmed_anchor_points=len(trimmed.frame), first_three_anchors=[dict(timestamp=pd.Timestamp(row.timestamp).isoformat(),
            relative_time_s=float((pd.Timestamp(row.timestamp)-boundary_stamp).total_seconds()),
            canonical_source_total_kw=float(row.source_total_kw), fc_kw=float(row.fc_total_kw),
            battery_raw_kw=float(row.battery_raw_total_kw), cubic_imputed=bool(row.is_cubic_imputed))
            for row in first_three.itertuples(index=False)],
        aligned_source_at_start=raw_at_boundary,
        start_original_derived_kw=trimmed.start.original_boundary_load_kw,
        start_canonical_kw=trimmed.start.canonical_boundary_load_kw, start_boundary_kind=trimmed.start.kind,
        end_boundary_kind=trimmed.end.kind, pchip_qa=qa,
        measurement_scope='Only the current Train parent raw channels were read; 1s total load is derived/interpolated, not independently measured.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--raw-root', type=Path, default=Path.home()/'OneDrive/Desktop/氢舟一号')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/audits/v6_one_plus_reward_20261029')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args(argv)
    self_check()
    if args.self_check:
        print('small-load audit self-check passed')
        return
    from v2.data.supervisory_rules import normalize_onboard_load_kw
    from v3.control import AccountState, EconomicMPC
    from v4.independent_economics import ZhouEvaluationConfig, soc_equivalent_hydrogen_cost_cny
    metadata = json.loads((args.data_root/'metadata/active_split.json').read_text(encoding='utf-8'))
    manifest = read_csv(args.data_root/metadata['manifest_relative_path'])
    train = sorted((row for row in manifest if row['split']=='train'), key=lambda row: row['candidate_id'])
    if len(train) != 35:
        raise ValueError('requires frozen 35 Train voyages')
    train_ids = {row['candidate_id'] for row in train}
    diagnostics = defaultdict(list)
    diagnostic_files = []
    for name in ('train_prefix_steps.csv', 'train_full_steps.csv'):
        path = ROOT/'docs/audits/v6_inverse_reward_20261029'/name
        diagnostic_files.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
        for row in read_csv(path):
            if row['sample_id'] not in train_ids:
                raise ValueError('fixed diagnostic contains a non-Train ID')
            diagnostics[(row['sample_id'], float(row['time_s']))].append(dict(row, diagnostic_source=name))
    bins, per_voyage, small, matches, sources = Counter(), [], [], [], []
    power_cache, component_cache = {}, {}
    accountant = EconomicMPC(nominal_cost_cny=1.)
    config = ZhouEvaluationConfig.from_formal_curve()
    fixed_state = AccountState()
    counters = Counter()
    target_source = None
    for manifest_row in train:
        sid = manifest_row['candidate_id']
        path = args.data_root/manifest_row['relative_path']
        rows = read_csv(path)
        source_path = ROOT/'data/processed/operating_dataset_zero_boundary_v2'/manifest_row['source_file_relative_path']
        if source_path not in power_cache:
            power_cache[source_path] = read_csv(source_path)
            sources.append(dict(path=source_path.relative_to(ROOT).as_posix(), sha256=sha(source_path),
                                role='current_Train_parent_canonical_1s'))
        source_rows = power_cache[source_path]
        component_path = ROOT/'data/processed/operating_dataset_zero_boundary_v2_modes'/manifest_row['source_file_relative_path']
        if component_path not in component_cache:
            component_cache[component_path] = {float(row['time_s']):row for row in read_csv(component_path)}
            sources.append(dict(path=component_path.relative_to(ROOT).as_posix(), sha256=sha(component_path),
                                role='current_Train_parent_30s_component_sidecar_not_new_1s_telemetry'))
        component_rows = component_cache[component_path]
        sources.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), role='current_Train_v6_canonical_1s'))
        local = Counter()
        for row in rows:
            raw, time = float(row['load_total_kw']), float(row['time_s'])
            source = source_rows[int(row['source_1s_row'])]
            if row['load_total_kw'] != source['load_total_kw'] or row['timestamp'] != source['timestamp']:
                raise ValueError('V6 canonical node differs from mapped parent source text')
            scope = 'control' if time < float(manifest_row['duration_s']) else 'terminal_metadata'
            bins[(scope, bin_name(raw), 'negative' if raw<0 else 'positive' if raw>0 else 'zero')] += 1
            counters['all_nodes'] += 1
            if abs(raw)>1:
                continue
            normalized = normalize_onboard_load_kw(raw)
            counters['small_nodes'] += 1
            counters['small_'+scope] += 1
            local['small_'+scope] += 1
            component = component_rows.get(float(row['source_time_s']))
            observed = diagnostics.get((sid,time), [])
            if observed:
                counters['small_with_diagnostic'] += 1
            else:
                counters['small_without_diagnostic'] += 1
            # Every counterfactual is a separate fixed fresh state, never a voyage.
            cf = None
            if scope == 'control':
                after, ledger, batt = accountant.interval(fixed_state, 0., normalized, duration_s=1.)
                soc_cost = soc_equivalent_hydrogen_cost_cny(.6, after.soc, config)
                cf = dict(normalized_load_kw=normalized, fc_kw=0., battery_kw=batt,
                    soc_before=.6, soc_after=after.soc, soc_delta=after.soc-.6,
                    h2_cost_cny=ledger.h2_cost_cny, fc_degradation_cost_cny=ledger.fuel_cell_degradation_cost_cny,
                    battery_degradation_cost_cny=ledger.battery_degradation_cost_cny,
                    soc_equivalent_cost_cny=soc_cost, total_cost_cny=fsum((ledger.total_cost_cny,soc_cost)))
                counters['counterfactual_steps'] += 1
            best = min(observed, key=lambda value:(value['policy']!='load_follow', value['diagnostic_source']!='train_prefix_steps.csv')) if observed else None
            diagnostic_fields = ('fc_kw','battery_kw','soc_before','soc_after','h2_cost_cny',
                'fc_degradation_cost_cny','battery_degradation_cost_cny','soc_equivalent_cost_cny','total_cost_cny')
            record = dict(sample_id=sid, parent_sample_id=manifest_row['parent_sample_id'], parent_name=manifest_row['parent_name'],
                split='train', node_scope=scope, time_s=time, timestamp=row['timestamp'],
                raw_load_text=row['load_total_kw'], raw_load_kw=raw, normalized_load_kw=normalized,
                abs_load_bin=bin_name(raw), source_time_s=float(row['source_time_s']), source_1s_row=int(row['source_1s_row']),
                canonical_path=path.relative_to(ROOT).as_posix(), source_parent_path=source_path.relative_to(ROOT).as_posix(),
                source_load_text=source['load_total_kw'], source_text_exact=True,
                source_record_fc_kw=component.get('p_fc_total_kw') if component else None,
                source_record_battery_bus_kw=component.get('p_batt_bus_kw') if component else None,
                source_record_power_provenance=component.get('power_provenance') if component else None,
                source_record_kind='existing_30s_component_sidecar_at_exact_timestamp' if component else 'no_exact_1s_component_record',
                diagnostic_match_count=len(observed), diagnostic_policy=best['policy'] if best else None,
                diagnostic_source=best['diagnostic_source'] if best else None,
                **{'diagnostic_'+key:best[key] if best else None for key in diagnostic_fields},
                counterfactual_scenario='independent_fresh_SOC_0.6_FC_off_dt1s' if cf else None,
                counterfactual_assumptions='each_row_fresh_ageing_FC0_SOC0.6_fixed_no_learning_not_a_trajectory' if cf else 'terminal_metadata_no_action',
                **{'counterfactual_'+key:cf[key] if cf else None for key in ('normalized_load_kw','fc_kw','battery_kw','soc_before','soc_after','soc_delta',
                    'h2_cost_cny','fc_degradation_cost_cny','battery_degradation_cost_cny','soc_equivalent_cost_cny','total_cost_cny')})
            small.append(record)
            for match in observed:
                matches.append(dict(sample_id=sid,time_s=time,raw_load_kw=raw,**{'diagnostic_'+key:value for key,value in match.items()}))
        per_voyage.append(dict(sample_id=sid,nodes=len(rows),control_intervals=int(float(manifest_row['duration_s'])),**local))
        if sid == 'parent_015_seg01':
            target_source = source_rows
    if counters['counterfactual_steps'] > 5382:
        raise ValueError('approved independent one-step audit budget exceeded')
    target = next(row for row in train if row['candidate_id']=='parent_015_seg01')
    provenance = trace_target(target, args.raw_root, target_source)
    # Record the mistaken legacy numeric ID lookup; current split metadata
    # identifies its displayed parent as Train parent_012, and it is excluded.
    scope_event = dict(path='data/processed/operating_dataset_final/metadata/aligned_30s/parent_015.csv',
        action='read_header_and_first_seven_rows_during_initial_source_search',
        reason_excluded='legacy numeric ID refers to 4月18 parent, not current parent_015 (4月28)',
        current_split_metadata_match=[dict(sample_id=row['candidate_id'],split=row['split']) for row in manifest
            if row['parent_name']=='4月18日12_00_4月18日18_00'],
        used_for_statistics_or_parameters=False, reopened=False)
    result = dict(scope='35 Train full canonical curves; descriptive bins only; no data/production change',
        counts=dict(counters), train_voyages=len(train), per_voyage=per_voyage,
        bins=[dict(node_scope=key[0],abs_load_bin=key[1],sign=key[2],count=count) for key,count in sorted(bins.items())],
        small_negative=sum(row['raw_load_kw']<0 for row in small), small_zero=sum(row['raw_load_kw']==0 for row in small),
        small_positive=sum(row['raw_load_kw']>0 for row in small),
        raw_zero_terminal=sum(row['raw_load_kw']==0 and row['node_scope']=='terminal_metadata' for row in small),
        small_with_exact_component_sidecar=sum(row['source_record_fc_kw'] is not None for row in small),
        target_trace=provenance, normalize_contract=dict(negative_inclusive_range_kw=[-1.,0.],negative_result_kw=0.,
            positive_policy='all strictly positive finite values retained',below_minus_one='ValueError'),
        diagnostic_files=diagnostic_files, source_files=sources,
        read_scope_events=[scope_event], current_Validation_payloads_opened=0,current_Test_payloads_opened=0,
        read_scope_basis='Legacy mistaken lookup was subsequently identified by current manifest parent_name as Train parent_012; excluded, not reopened.',
        raw_source_parent_payloads_opened=[target['parent_name']], new_voyage_control_executions=0,
        counterfactual_assumptions='5382 independent 1-second fixed fresh SOC .6 / FC0 states; terminal nodes excluded; no learning or state carryover',
        unknowns=['No independent load sensor establishes true 1.741 W at t1.',
            'CSV precision does not establish instrument accuracy or physical relevance of sub-kW values.',
            'Most small nodes lack an existing full policy trajectory; diagnostic fields stay null.',
            '30s component sidecar may contain aligned/cubic/PCHIP values; source provenance is retained, never called independent 1s measurement.'])
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'load_small_nodes.csv',small)
    write_csv(args.output/'load_diagnostic_matches.csv',matches)
    write_csv(args.output/'load_bins.csv',result['bins'])
    write_csv(args.output/'load_per_voyage.csv',per_voyage)
    (args.output/'load_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(counts=result['counts'],target_trace=provenance),ensure_ascii=False))


def self_check():
    assert callable(globals().get('bin_name')), 'small-load classification must exist'
    assert bin_name(0.) == 'exact_zero'
    assert bin_name(.0017413563970556567) == '(0.001,0.01]'
    assert bin_name(-.0017413563970556567) == '(0.001,0.01]'
    assert bin_name(1.) == '(0.1,1]'
    assert bin_name(1.0001) == '>1'
    from v2.data.supervisory_rules import normalize_onboard_load_kw
    assert normalize_onboard_load_kw(-.0017413563970556567) == 0.
    assert normalize_onboard_load_kw(.0017413563970556567) == .0017413563970556567
    assert normalize_onboard_load_kw(-1.) == 0.
    try:
        normalize_onboard_load_kw(-1.001)
    except ValueError:
        pass
    else:
        raise AssertionError('load below -1 kW must reject')


if __name__ == '__main__':
    main()
