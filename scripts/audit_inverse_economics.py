"""Audit unchanged economics with Train-only records and bounded new execution.

No learning, Validation/Test payload, production reward, or economic definition
is changed. Offline reciprocals are undefined (JSON null) for C <= 0.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from math import fsum, isfinite
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
from v2.models.fuel_cell_efficiency import LHV_H2_KWH_PER_KG
from v3.control import AccountState, EconomicMPC
from v4.actions import candidate_fc_actions
from v4.control import _v6_feasible_fc_actions
from v4.independent_economics import ZhouEvaluationConfig, soc_equivalent_hydrogen_cost_cny
from v4.v6_dataset import V6TrainingDataset
from v6_sb3.direct_environment import DirectVesselEnv
from v6_sb3.environment import ACTION_DELTAS, InfeasibleReset

COMPONENTS = ('total_cost_cny', 'h2_cost_cny', 'fc_degradation_cost_cny',
              'battery_degradation_cost_cny', 'soc_equivalent_cost_cny', 'inverse_reward')


def distribution(values):
    """Linear-interpolated quantiles; explicit nulls never become zero costs."""
    raw = list(values)
    finite = [float(value) for value in raw if value is not None]
    if any(not isfinite(value) for value in finite):
        raise ValueError('audit distributions require finite observed values')
    result = dict(count=len(finite), missing=len(raw)-len(finite),
                  min=None, p1=None, median=None, p99=None, max=None, mean=None,
                  zero=0, negative=0)
    if finite:
        result.update(min=min(finite), p1=float(np.quantile(finite, .01)),
                      median=float(np.median(finite)), p99=float(np.quantile(finite, .99)),
                      max=max(finite), mean=fsum(finite)/len(finite),
                      zero=sum(value == 0. for value in finite),
                      negative=sum(value < 0. for value in finite))
    return result


def _cost_fields(ledger, soc_cost):
    return dict(h2_cost_cny=ledger.h2_cost_cny,
        fc_degradation_cost_cny=ledger.fuel_cell_degradation_cost_cny,
        battery_degradation_cost_cny=ledger.battery_degradation_cost_cny,
        soc_equivalent_cost_cny=soc_cost,
        total_cost_cny=fsum((ledger.total_cost_cny, soc_cost)))


def boundary_cases():
    """Model-legal states, distinguishing fresh resets from EOL boundaries."""
    accountant = EconomicMPC(nominal_cost_cny=1.)
    config = ZhouEvaluationConfig.from_formal_curve()
    qlife = accountant.battery_normalization.q_lifetime_ah
    warm = AccountState()
    for power in (60., 70.):
        warm, _, _ = accountant.interval(warm, power, power, duration_s=1.)
    specifications = [
        ('idle_zero', AccountState(), 0., 0., 'fresh_reset_reachable'),
        ('efficient_charge_fresh', warm, 70., 0., 'reachable_after_legal_60_then_70_kW_warmup_load_matches_FC'),
        ('efficient_charge_fc_eol', AccountState(previous_fc_kw=70., fc_loss_uv=70000.), 70., 0., 'EOL_model_boundary_not_observed_Train'),
        ('efficient_charge_both_eol', AccountState(previous_fc_kw=70., fc_loss_uv=70000., battery_weighted_ah=qlife), 70., 0., 'EOL_model_boundary_not_observed_Train'),
        ('off_discharge', AccountState(), 0., 100., 'fresh_reset_reachable'),
        ('on_discharge', warm, 70., 100., 'reachable_after_legal_60_then_70_kW_warmup_load_matches_FC'),
        ('low_power', AccountState(previous_fc_kw=10.), 10., 10., 'positive_low_efficiency'),
        ('rated_power', AccountState(previous_fc_kw=600.), 600., 600., 'rated_low_efficiency'),
        ('soc_min_idle', AccountState(soc=.2), 0., 0., 'SOC_boundary'),
        ('soc_max_idle', AccountState(soc=.8), 0., 0., 'SOC_boundary'),
        ('soc_min_charge', AccountState(soc=.2, previous_fc_kw=70.), 70., 0., 'SOC_boundary'),
        ('soc_max_discharge', AccountState(soc=.8), 0., 100., 'SOC_boundary'),
    ]
    rows = []
    for name, before, fc, load, reachability in specifications:
        legal = _v6_feasible_fc_actions(before, load, accountant, duration_s=1.)
        candidates = candidate_fc_actions(legal, before.previous_fc_kw, 'increment13')
        after, ledger, battery = accountant.interval(before, fc, load, duration_s=1.)
        costs = _cost_fields(ledger, soc_equivalent_hydrogen_cost_cny(before.soc, after.soc, config))
        rows.append(dict(case=name, reachability=reachability, physical_legal=fc in legal,
            increment13_legal=fc in candidates, soc_before=before.soc, soc_after=after.soc,
            fc_loss_before_uv=before.fc_loss_uv, battery_weighted_before_ah=before.battery_weighted_ah,
            load_kw=load, fc_kw=fc, previous_fc_kw=before.previous_fc_kw, battery_kw=battery,
            eta_fc=float(accountant.fc_efficiency.eta(fc)), **costs,
            inverse_reward=1./costs['total_cost_cny'] if costs['total_cost_cny'] > 0 else None,
            charge_credit_to_actual_h2=(-costs['soc_equivalent_cost_cny']/ledger.h2_cost_cny
                                        if battery < 0 and ledger.h2_cost_cny else None)))
    return rows


def round_trip_cycle():
    """Two legal increments restore stored energy after a .95*.95 bus cycle."""
    accountant = EconomicMPC(nominal_cost_cny=1.)
    config = ZhouEvaluationConfig.from_formal_curve()
    state = AccountState()
    for power in (60., 70.):
        state, _, _ = accountant.interval(state, power, power, duration_s=1.)
    initial = state.soc
    charged, charge_ledger, _ = accountant.interval(state, 70., 0., duration_s=1.)
    eta_chg, eta_dis = accountant.efficiency.require_calibrated()
    discharge_bus = 70. * eta_chg * eta_dis
    discharged, discharge_ledger, _ = accountant.interval(charged, 10., 10.+discharge_bus, duration_s=1.)
    corrections = (soc_equivalent_hydrogen_cost_cny(initial, charged.soc, config),
                   soc_equivalent_hydrogen_cost_cny(charged.soc, discharged.soc, config))
    actual_cost = fsum((charge_ledger.total_cost_cny, discharge_ledger.total_cost_cny))
    return dict(soc_before=initial, soc_after=discharged.soc,
        charge_bus_kwh=70./3600., discharge_bus_kwh=discharge_bus/3600.,
        soc_correction_sum_cny=fsum(corrections), actual_ledger_cost_cny=actual_cost,
        total_cost_cny=fsum((actual_cost, *corrections)),
        commands_kw=[70., 10.], loads_kw=[0., 10.+discharge_bus],
        note='Equal SOC cancels fixed shadow value; actual fuel and degradation remain positive. Warmup excluded from cycle costs.')


def run_trajectory(episode, *, policy, max_steps):
    if getattr(episode, 'split', None) != 'train':
        raise ValueError('new economic audit executes Train only')
    if policy not in ('load_follow', 'hold_300'):
        raise ValueError('unknown fixed audit policy')
    env = DirectVesselEnv(episode, reward_mode='zhou_economic_only')
    rows = []
    begin = perf_counter()
    try:
        env.reset(seed=20261029)
        for _ in range(min(max_steps, len(episode.load_kw))):
            state = env.current_decision_state
            legal_indices = np.flatnonzero(env.action_masks())
            previous_fc = round(state[2]*600)
            target = normalize_onboard_load_kw(state[1]*600) if policy == 'load_follow' else 300.
            action = min(legal_indices, key=lambda a: (abs(previous_fc+ACTION_DELTAS[a]-target), int(a)))
            _, _, done, _, info = env.step(int(action))
            transition = info['transition']
            ledger = transition.executed_ledger
            costs = _cost_fields(ledger, transition.soc_equivalent_hydrogen_cost_cny)
            fc, battery = transition.action_kw, transition.actual_battery_kw
            eta = float(env.accountant.fc_efficiency.eta(fc))
            # A charge-only comparison: value the bus charging input at its
            # actual generating H2 cost. No discharge model is silently chosen.
            charge_generation_value = (-(-battery)/3600./(eta*LHV_H2_KWH_PER_KG)*HYDROGEN_PRICE_CNY_PER_KG
                                       if battery < 0 else None)
            rows.append(dict(sample_id=episode.sample_id, policy=policy,
                evidence='new_bounded_audit_execution_not_archived', time_s=transition.relative_time_s,
                raw_load_kw=transition.raw_load_kw, fc_kw=fc, battery_kw=battery,
                soc_before=state[0], soc_after=transition.actual_soc, eta_fc=eta, **costs,
                inverse_reward=1./costs['total_cost_cny'] if costs['total_cost_cny'] > 0 else None,
                charge_generation_value_cny=charge_generation_value,
                charge_only_value_difference_cny=(charge_generation_value-costs['soc_equivalent_cost_cny']
                                                  if charge_generation_value is not None else 0.),
                power_balance_residual_kw=fc+battery-normalize_onboard_load_kw(transition.raw_load_kw),
                failure_penalty_equivalent_cny=transition.failure_penalty_equivalent_cny,
                done=done, terminal_reason=transition.terminal_reason))
            if done:
                break
    except InfeasibleReset:
        pass
    finally:
        status = env.episode_status or 'audit_prefix_cutoff'
        env.close()
    soc_final = rows[-1]['soc_after'] if rows else .6
    correction = soc_equivalent_hydrogen_cost_cny(.6, soc_final, ZhouEvaluationConfig.from_formal_curve())
    valid_inverse = all(row['inverse_reward'] is not None for row in rows)
    inverse_sum = fsum(row['inverse_reward'] for row in rows) if valid_inverse else None
    return rows, dict(sample_id=episode.sample_id, policy=policy, steps=len(rows),
        status=status, completed=status == 'completed', duration_s=episode.duration_s,
        soc_final=soc_final, economic_cost_cny=fsum(row['total_cost_cny'] for row in rows),
        fixed_soc_correction_cny=correction,
        soc_telescope_residual_cny=fsum(row['soc_equivalent_cost_cny'] for row in rows)-correction,
        offline_inverse_sum_before_terminal_replacement=inverse_sum,
        undefined_reciprocal_steps=sum(row['inverse_reward'] is None for row in rows),
        charge_branch_only_alternative_value_difference_cny=fsum(row['charge_only_value_difference_cny'] for row in rows),
        full_realtime_economic_cost_cny=None,
        note='No complete realtime discharge valuation selected. Audit cutoff is not a physical failure.',
        elapsed_s=perf_counter()-begin)


def grouped_distributions(rows):
    predicates = dict(all=lambda r: True, fc_on=lambda r: r['fc_kw'] > 0,
        fc_off=lambda r: r['fc_kw'] == 0, charge=lambda r: r['battery_kw'] < 0,
        discharge=lambda r: r['battery_kw'] > 0, battery_idle=lambda r: r['battery_kw'] == 0)
    return {group: {component: distribution(row[component] for row in rows if predicate(row))
                    for component in COMPONENTS} for group, predicate in predicates.items()}


def summarize_sparse_rows(rows):
    if any(row.get('source_split') != 'train' for row in rows):
        raise ValueError('sparse audit accepts Train rows only')
    times = [float(row['time_s']) for row in rows]
    return dict(recorded_steps=len(rows),
        contiguous_from_reset=bool(times) and times[0] == 0 and all(b-a == 1 for a,b in zip(times,times[1:])),
        complete_step_economic_distribution_cny=None,
        fc_degradation_distribution_cny=None, battery_degradation_distribution_cny=None,
        h2_distribution_cny=distribution(float(row['fc_h2']) for row in rows),
        soc_equivalent_distribution_cny=distribution(float(row['battery_h2eq']) for row in rows))


def scan_train_archive(source_root, train_ids):
    summary_files = sorted(source_root.glob('round_*_train_samples.csv'))
    summary_rows, sparse_rows, sparse_groups = 0, [], []
    sources = []
    def source(path, kind):
        sources.append(dict(path=path.relative_to(ROOT).as_posix(), kind=kind,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for path in summary_files:
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 35 or {r['sample_id'] for r in rows} != train_ids:
            raise ValueError('archived summary violates frozen 35-Train identity')
        summary_rows += len(rows)
        source(path, 'Train_cumulative_summary_not_step_ledger')
    for path in sorted(source_root.glob('round_*_train_failure_prefixes.csv.gz')):
        groups = defaultdict(list)
        with gzip.open(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['sample_id'] not in train_ids:
                    raise ValueError('sparse record contains non-Train ID')
                groups[row['sample_id']].append(row)
                sparse_rows.append(row)
        for sid, rows in groups.items():
            sparse_groups.append(dict(source=path.name, sample_id=sid, **summarize_sparse_rows(rows)))
        source(path, 'Train_sparse_observation_missing_degradation')
    return dict(summary_files=len(summary_files), summary_rows=summary_rows,
        existing_full_35Train_step_cost_distribution_cny=None,
        sparse_observed_rows=len(sparse_rows), sparse_group_count=len(sparse_groups),
        sparse_observations=summarize_sparse_rows(sparse_rows), sparse_groups=sparse_groups,
        limitation='Cumulative summaries have no within-voyage step distributions. Sparse rows omit both degradation ledgers and intermediate actions.',
        sources=sources)


def write_csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--source-root', type=Path, default=ROOT/'docs/results/sb3_dqn_1s_20261028/raw')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/audits/v6_inverse_reward_20261029')
    parser.add_argument('--prefix-steps', type=int, default=32, choices=range(1,33))
    parser.add_argument('--full-shortest', action='store_true', help='Two fixed policies, at most 3272 steps each; no learning')
    args = parser.parse_args(argv)
    dataset = V6TrainingDataset.open(args.data_root)
    train = dataset.load_train()  # Never call load_validation or load_split('test').
    if len(train) != 35 or any(episode.split != 'train' for episode in train):
        raise ValueError('requires frozen 35-Train dataset')
    ids = {episode.sample_id for episode in train}
    archive = scan_train_archive(args.source_root, ids)
    prefix_rows, prefix_runs, full_rows, full_runs = [], [], [], []
    for episode in train:
        rows, result = run_trajectory(episode, policy='load_follow', max_steps=args.prefix_steps)
        prefix_rows.extend(rows)
        prefix_runs.append(result)
    if args.full_shortest:
        shortest = min(train, key=lambda episode: (episode.duration_s, episode.sample_id))
        if len(shortest.load_kw) > 3272:
            raise ValueError('shortest Train exceeds approved bounded-audit step budget')
        for policy in ('load_follow', 'hold_300'):
            rows, result = run_trajectory(shortest, policy=policy, max_steps=len(shortest.load_kw))
            full_rows.extend(rows)
            full_runs.append(result)
    boundaries = boundary_cases()
    result = dict(audit_version='inverse_pre_change_v1', production_economics_changed=False,
        production_reward_changed=False, official_training_allowed=False,
        blocking_reason='Fresh physically legal idle has C=0; reciprocal undefined without changing requested design.',
        fc_average_efficiency=ZhouEvaluationConfig.from_formal_curve().fc_average_efficiency,
        fc_average_method='fsum(eta(P) for P in 10..600 step10)/60; equal action weights, zero excluded',
        new_payloads_opened=dict(train=len(train), validation=0, test=0),
        executed_step_budget=dict(prefix_max=35*args.prefix_steps, full_max=6544 if args.full_shortest else 0,
            actual_prefix=len(prefix_rows), actual_full=len(full_rows)),
        boundary_cases=boundaries, round_trip_cycle=round_trip_cycle(), archived_evidence=archive,
        prefix_runs=prefix_runs, new_prefix_distributions=grouped_distributions(prefix_rows),
        full_runs=full_runs, new_full_distributions=grouped_distributions(full_rows),
        train_duration_s=distribution(episode.duration_s for episode in train),
        max_abs_power_balance_residual_kw=max(abs(row['power_balance_residual_kw']) for row in prefix_rows+full_rows),
        quantile_method='numpy.quantile default linear; exact-zero/negative comparisons, no epsilon',
        reciprocal_scope='Offline mathematical diagnostic only; no production inverse reward or failure -1 enabled')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'boundary_cases.csv', boundaries)
    write_csv(args.output/'train_prefix_steps.csv', prefix_rows)
    write_csv(args.output/'train_full_steps.csv', full_rows)
    (args.output/'pre_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), prefix_steps=len(prefix_rows), full_steps=len(full_rows),
        full_runs=full_runs, zero_boundary_cases=sum(row['total_cost_cny']==0 for row in boundaries),
        negative_boundary_cases=sum(row['total_cost_cny']<0 for row in boundaries)), ensure_ascii=False))


if __name__ == '__main__':
    main()
