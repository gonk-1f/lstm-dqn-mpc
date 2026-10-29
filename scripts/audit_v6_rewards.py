"""Read archived Train/Validation logs and compare frozen reward formulas.

No training, policy inference, dataset payload loading or environment execution.
Sparse Train prefixes remain partial observations: missing degradation ledgers
and unknown cumulative EOL state are never replaced with a complete-voyage cost.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from v2.models.fuel_cell_degradation import (
    FC_AGGREGATE_REPLACEMENT_COST_CNY, FC_AGGREGATE_TO_REFERENCE_POWER_RATIO,
    FC_EOL_VOLTAGE_LOSS_UV, FC_HIGH_LOAD_FRACTION,
    FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR, FC_LOW_RUNTIME_LOSS_UV_PER_HOUR,
    FC_START_STOP_LOSS_UV_PER_CYCLE, FC_TRANSIENT_LOSS_UV_PER_DELTA_KW,
)
from v4.independent_economics import ZhouEvaluationConfig, soc_equivalent_hydrogen_cost_cny

MODES = ('full_economic', 'zhou_economic_only', 'zhou_joint', 'v6_energy_soc_ramp')
TRANSIENT_CNY_PER_KW = (FC_TRANSIENT_LOSS_UV_PER_DELTA_KW
    * FC_AGGREGATE_TO_REFERENCE_POWER_RATIO
    * FC_AGGREGATE_REPLACEMENT_COST_CNY / FC_EOL_VOLTAGE_LOSS_UV)


def step_reward_comparison(*, h2_cost, fc_degradation, battery_degradation,
        soc_before, soc_after, delta_kw, config, failure_penalty=0.,
        zhou_beta_soc=0., v6_soc_weight=.2, v6_ramp_weight=.05):
    """Independent arithmetic for one frozen action, with failure cost separate.

    Zhou-joint beta is explicit; its default is zero here. Full-economic also
    uses beta=0 and no battery-energy redistribution for a fair comparison.
    """
    inputs = (h2_cost, fc_degradation, battery_degradation, soc_before,
              soc_after, delta_kw, failure_penalty, zhou_beta_soc,
              v6_soc_weight, v6_ramp_weight)
    if not all(math.isfinite(float(v)) for v in inputs):
        raise ValueError('reward comparison inputs must be finite')
    if min(h2_cost, fc_degradation, battery_degradation, failure_penalty,
           zhou_beta_soc, v6_soc_weight, v6_ramp_weight) < 0.:
        raise ValueError('ledger costs, penalties and auxiliary weights must be nonnegative')
    correction = soc_equivalent_hydrogen_cost_cny(soc_before, soc_after, config)
    ledger = math.fsum((h2_cost, fc_degradation, battery_degradation))
    zhou_soc = zhou_beta_soc * (soc_after - .6) ** 2
    zhou_ramp = TRANSIENT_CNY_PER_KW / 60. * delta_kw ** 2
    v6_soc = v6_soc_weight * ((soc_after - .5) / .1) ** 2
    v6_ramp = v6_ramp_weight * delta_kw ** 2 / 600.
    return dict(full_economic=-math.fsum((ledger, failure_penalty)),
        zhou_economic_only=-math.fsum((ledger, correction, failure_penalty)),
        zhou_joint=-math.fsum((ledger, correction, zhou_soc, zhou_ramp, failure_penalty)),
        v6_energy_soc_ramp=-math.fsum((h2_cost, correction, v6_soc, v6_ramp, failure_penalty)),
        soc_equivalent_cost=correction, ledger_cost=ledger,
        zhou_soc_penalty=zhou_soc, zhou_ramp_penalty=zhou_ramp,
        v6_soc_penalty=v6_soc, v6_ramp_penalty=v6_ramp,
        failure_penalty=failure_penalty)


def fc_cost_components(previous_kw, power_kw, duration_s, cumulative_loss_before_uv=0.):
    """Independently expose raw components and capped cumulative-difference cost.

    Components before EOL are unambiguous. No per-component allocation of an
    EOL-clipped total is invented; the difference is reported as cap reduction.
    """
    if (not all(math.isfinite(float(v)) for v in
                (previous_kw, power_kw, duration_s, cumulative_loss_before_uv))
            or min(previous_kw, power_kw, cumulative_loss_before_uv) < 0.
            or max(previous_kw, power_kw) > 600. or duration_s <= 0.):
        raise ValueError('invalid formal FC trace domain')
    rate = FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR if power_kw >= FC_HIGH_LOAD_FRACTION * 600. else FC_LOW_RUNTIME_LOSS_UV_PER_HOUR
    runtime = rate * duration_s / 3600. if power_kw > 0. else 0.
    transient = FC_TRANSIENT_LOSS_UV_PER_DELTA_KW * abs(power_kw - previous_kw) * FC_AGGREGATE_TO_REFERENCE_POWER_RATIO
    start = FC_START_STOP_LOSS_UV_PER_CYCLE if previous_kw == 0. and power_kw > 0. else 0.
    raw_uv = math.fsum((runtime, transient, start))
    conversion = FC_AGGREGATE_REPLACEMENT_COST_CNY / FC_EOL_VOLTAGE_LOSS_UV
    raw_cost = raw_uv * conversion
    charged = FC_AGGREGATE_REPLACEMENT_COST_CNY * (
        min((cumulative_loss_before_uv + raw_uv) / FC_EOL_VOLTAGE_LOSS_UV, 1.)
        - min(cumulative_loss_before_uv / FC_EOL_VOLTAGE_LOSS_UV, 1.))
    return dict(runtime_raw_cny=runtime * conversion,
        transient_raw_cny=transient * conversion, start_raw_cny=start * conversion,
        raw_total_cny=raw_cost, charged_total_cny=charged,
        eol_cap_reduction_cny=raw_cost - charged,
        cumulative_loss_after_uv=cumulative_loss_before_uv + raw_uv)


def validate_source_path(path):
    """Reject Test payload paths before opening any files."""
    path = Path(path)
    if any(p.lower() == 'test' or p.lower().startswith('test_round_')
           or p.lower().endswith('_test') for p in path.parts):
        raise ValueError(f'Test payload is outside this offline audit: {path}')
    return path


def _number(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value is not None and value != '':
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f'nonfinite {key} in archived log')
            return result
    if default is not None:
        return default
    raise ValueError(f'missing archived field: {keys}')


def summarize_logged_rows(rows, *, config, source_kind, step_seconds=1., zhou_beta_soc=0.):
    """Compare the same realized actions, never infer a new policy's behavior."""
    rows = list(rows)
    if not rows:
        raise ValueError('archived trajectory must contain at least one row')
    sparse = source_kind == 'train_sparse_prefix'
    times = [_number(r, 'time_s') for r in rows]
    contiguous = (not sparse and times[0] == 0.
        and math.isclose(_number(rows[0], 'soc_before'), .6, abs_tol=1e-12)
        and math.isclose(_number(rows[0], 'fc_kw') - _number(rows[0], 'fc_delta_kw', 'delta_kw'), 0., abs_tol=1e-9)
        and all(math.isclose(b-a, step_seconds, abs_tol=1e-8) for a,b in zip(times, times[1:])))
    has_degradation = all('fc_degradation_cost_cny' in r and 'battery_degradation_cost_cny' in r for r in rows)
    completed = (contiguous and rows[-1].get('terminal_reason') == 'completed'
                 and str(rows[-1].get('done')).lower() == 'true')
    terms = defaultdict(list)
    cumulative_loss = 0.
    logged_fc_residual = []
    for row in rows:
        before = _number(row, 'soc_before')
        after = _number(row, 'soc', 'soc_after')
        power = _number(row, 'fc_kw')
        delta = _number(row, 'fc_delta_kw', 'delta_kw')
        parts = fc_cost_components(power-delta, power, step_seconds, cumulative_loss if contiguous else 0.)
        if contiguous:
            cumulative_loss = parts['cumulative_loss_after_uv']
        reward = step_reward_comparison(h2_cost=_number(row, 'fc_h2'),
            fc_degradation=_number(row, 'fc_degradation_cost_cny', default=0.),
            battery_degradation=_number(row, 'battery_degradation_cost_cny', default=0.),
            soc_before=before, soc_after=after, delta_kw=delta, config=config,
            failure_penalty=_number(row, 'failure_penalty', default=0.), zhou_beta_soc=zhou_beta_soc)
        for key, value in reward.items():
            terms[key].append(value)
        for key in ('runtime_raw_cny', 'transient_raw_cny', 'start_raw_cny', 'raw_total_cny'):
            terms['fc_' + key].append(parts[key])
        if contiguous:
            terms['fc_charged_reconstructed_cny'].append(parts['charged_total_cny'])
        if has_degradation:
            terms['fc_degradation_logged_cny'].append(_number(row, 'fc_degradation_cost_cny'))
            terms['battery_degradation_logged_cny'].append(_number(row, 'battery_degradation_cost_cny'))
            if contiguous:
                logged_fc_residual.append(abs(parts['charged_total_cny'] - _number(row, 'fc_degradation_cost_cny')))
        terms['h2_cost_cny'].append(_number(row, 'fc_h2'))
        terms['soc_correction_residual_abs'].append(abs(reward['soc_equivalent_cost']-_number(row, 'battery_h2eq')))
        terms['v6_reward_residual_abs'].append(abs(reward['v6_energy_soc_ramp']-_number(row, 'reward', 'immediate_reward')))
        terms['v6_soc_residual_abs'].append(abs(reward['v6_soc_penalty']-_number(row, 'soc_penalty')))
        terms['v6_ramp_residual_abs'].append(abs(reward['v6_ramp_penalty']-_number(row, 'fc_ramp', 'fc_ramp_penalty')))
    totals = {key:math.fsum(values) for key, values in terms.items()}
    correction_endpoint = (soc_equivalent_hydrogen_cost_cny(
        _number(rows[0], 'soc_before'), _number(rows[-1], 'soc', 'soc_after'), config)
        if contiguous else None)
    economic = math.fsum((totals['ledger_cost'], correction_endpoint)) if contiguous and has_degradation else None
    result = dict(source_kind=source_kind, recorded_steps=len(rows),
        first_time_s=times[0], last_time_s=times[-1], contiguous_from_reset=contiguous,
        completed=completed, terminal_reason=rows[-1].get('terminal_reason', rows[-1].get('failure_reason')),
        complete_economic_cost_cny=economic if completed else None,
        executed_contiguous_prefix_economic_cost_cny=economic,
        soc_initial=_number(rows[0], 'soc_before'), soc_terminal=_number(rows[-1], 'soc', 'soc_after'),
        failure_penalty_cny=totals['failure_penalty'], h2_cost_cny=totals['h2_cost_cny'],
        soc_equivalent_cost_cny=totals['soc_equivalent_cost'],
        soc_telescoping_residual_cny=(totals['soc_equivalent_cost']-correction_endpoint) if contiguous else None,
        fc_degradation_logged_cny=totals.get('fc_degradation_logged_cny'),
        battery_degradation_logged_cny=totals.get('battery_degradation_logged_cny'),
        fc_runtime_raw_cny=totals['fc_runtime_raw_cny'],
        fc_transient_raw_cny=totals['fc_transient_raw_cny'], fc_start_raw_cny=totals['fc_start_raw_cny'],
        fc_raw_total_cny=totals['fc_raw_total_cny'],
        fc_charged_reconstructed_cny=totals.get('fc_charged_reconstructed_cny'),
        fc_logged_max_step_residual_cny=max(logged_fc_residual, default=None),
        v6_soc_penalty=totals['v6_soc_penalty'], v6_ramp_penalty=totals['v6_ramp_penalty'],
        zhou_soc_penalty=totals['zhou_soc_penalty'], zhou_ramp_penalty=totals['zhou_ramp_penalty'],
        v6_recorded_rows_return=totals['v6_energy_soc_ramp'],
        v6_reward_max_step_residual=max(terms['v6_reward_residual_abs']),
        soc_correction_max_step_residual=max(terms['soc_correction_residual_abs']),
        v6_soc_max_step_residual=max(terms['v6_soc_residual_abs']),
        v6_ramp_max_step_residual=max(terms['v6_ramp_residual_abs']))
    for mode in MODES:
        result[mode + '_return'] = totals[mode] if contiguous and has_degradation else None
    result['zhou_return_plus_economic_plus_failure'] = (
        math.fsum((totals['zhou_economic_only'], economic, totals['failure_penalty']))
        if economic is not None else None)
    return result


def _read_rows(path):
    path = validate_source_path(path)
    with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT/'docs/results/sb3_dqn_1s_20261028/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/audits/v6_optuna_20261028')
    parser.add_argument('--zhou-beta-soc', type=float, default=0.)
    args = parser.parse_args(argv)
    validate_source_path(args.source_root)
    config = ZhouEvaluationConfig.from_formal_curve()
    manifest = []
    summaries = []
    paths = sorted(args.source_root.glob('round_[0-9][0-9]_validation/*.csv.gz'))
    paths += sorted(args.source_root.glob('round_[0-9][0-9]_train_failure_prefixes.csv.gz'))
    if not paths:
        raise ValueError('no archived Train/Validation step logs found')
    for path in paths:
        rows = _read_rows(path)
        relative = path.relative_to(args.source_root).as_posix()
        manifest.append(dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=len(rows)))
        is_train = path.name.endswith('_train_failure_prefixes.csv.gz')
        groups = defaultdict(list)
        for row in rows:
            key = row.get('sample_id', path.name.removesuffix('.csv.gz'))
            groups[key].append(row)
        for sample_id, group in groups.items():
            summaries.append(dict(source_file=relative, round=int(path.parts[-1 if is_train else -2][6:8]),
                sample_id=sample_id, **summarize_logged_rows(group, config=config,
                    source_kind='train_sparse_prefix' if is_train else 'validation_trajectory',
                    zhou_beta_soc=args.zhou_beta_soc)))
    scales = []
    for delta in (0., 10., 20., 30., 40., 50., 60.):
        parts = fc_cost_components(100., 100.+delta, 1.)
        rewards = step_reward_comparison(h2_cost=0., fc_degradation=0., battery_degradation=0.,
            soc_before=.5, soc_after=.5, delta_kw=delta, config=config)
        scales.append(dict(delta_kw=delta, formal_transient_cny=parts['transient_raw_cny'],
            v6_squared_ramp= rewards['v6_ramp_penalty'], zhou_squared_ramp=rewards['zhou_ramp_penalty']))
    by_round = []
    for round_index in sorted({r['round'] for r in summaries}):
        subset = [r for r in summaries if r['round']==round_index and r['source_kind']=='validation_trajectory']
        if not subset:
            continue
        summable = ('h2_cost_cny', 'soc_equivalent_cost_cny', 'fc_degradation_logged_cny',
            'battery_degradation_logged_cny', 'fc_runtime_raw_cny', 'fc_transient_raw_cny',
            'fc_start_raw_cny', 'v6_soc_penalty', 'v6_ramp_penalty', 'failure_penalty_cny')
        by_round.append(dict(round=round_index, trajectories=len(subset),
            completed=sum(r['completed'] for r in subset),
            **{k:math.fsum(r[k] for r in subset) for k in summable},
            all_complete=all(r['completed'] for r in subset),
            complete_fixed_set_economic_cost_cny=(math.fsum(r['complete_economic_cost_cny'] for r in subset)
                if all(r['completed'] for r in subset) else None),
            **{m+'_recorded_prefix_return':math.fsum(r[m+'_return'] for r in subset) for m in MODES}))
    validation = [r for r in summaries if r['source_kind']=='validation_trajectory']
    report = dict(scope='archived Train/Validation logs only; fixed realized actions; no policy improvement claim',
        configuration=dict(gamma=1., zhou_beta_soc=args.zhou_beta_soc, full_economic_beta_soc=0.,
            redistribution=False, archived_step_seconds=1., fc_average_efficiency=config.fc_average_efficiency),
        source_file_count=len(manifest), validation_trajectory_count=len(validation),
        sparse_train_group_count=len(summaries)-len(validation),
        validation_recorded_rows=sum(r['recorded_steps'] for r in validation),
        validation_max_abs_zhou_return_identity_residual_cny=max(abs(r['zhou_return_plus_economic_plus_failure']) for r in validation),
        validation_max_abs_soc_telescoping_residual_cny=max(abs(r['soc_telescoping_residual_cny']) for r in validation),
        validation_max_fc_step_reconstruction_residual_cny=max(r['fc_logged_max_step_residual_cny'] for r in validation),
        validation_max_v6_reward_step_residual=max(r['v6_reward_max_step_residual'] for r in validation),
        by_round=by_round,
        limitations=['Sparse Train rows omit intervening actions and physical degradation ledgers; economic returns are null.',
            'Raw FC start/runtime/transient components are pre-EOL; clipped component allocation is not defined.',
            'Comparisons score the same logged actions and do not estimate trained-policy performance.',
            'Failure prefix cost is not the missing complete-voyage cost. Test payloads are never opened.'])
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output/'offline_reward_comparison.csv', summaries)
    _write_csv(args.output/'fc_component_scales.csv', scales)
    for name, data in (('offline_reward_comparison.json', report), ('reward_source_manifest.json', manifest)):
        (args.output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('by_round', 'limitations')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
