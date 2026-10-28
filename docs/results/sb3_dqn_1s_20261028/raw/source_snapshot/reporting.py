"""Bounded, quiet reports from actual SB3/V6 Train and Validation records."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v3.control import AccountState
from v4.actions import FC_INCREMENT_13_KW, candidate_fc_actions
from v4.control import _v6_feasible_fc_actions
from v4.v6_training import execute, plot_profile, sample_record, split_summary, write_csv, write_json


def describe_failure(episode, values, status, round_index, phase, epsilon, accountant):
    """Describe the unexecuted failure decision F, not its last executed F-1 row.

    The physical interval fields are extrema of the existing 10 kW absolute
    grid, rather than a newly calculated continuous feasibility boundary.
    """
    if status == 'completed':
        return None
    index = len(values)
    soc = float(values[-1].actual_soc) if values else .6
    previous_soc = float(values[-1].state[0]) if values else None
    previous_fc = int(values[-1].action_kw) if values else 0
    load = float(episode.load_kw[index])
    physical = _v6_feasible_fc_actions(
        AccountState(soc=soc, previous_fc_kw=previous_fc), load,
        accountant, duration_s=float(episode.dt_s[index]))
    allowed = candidate_fc_actions(physical, previous_fc, 'increment13')
    candidates = [dict(action_index=i, delta_kw=delta, fc_kw=previous_fc+delta,
                       in_fc_bounds=0 <= previous_fc+delta <= 600,
                       physical_legal=previous_fc+delta in physical,
                       allowed=previous_fc+delta in allowed)
                  for i, delta in enumerate(FC_INCREMENT_13_KW)]
    times = getattr(episode, 'source_time_s', None)
    timestamps = getattr(episode, 'timestamp', None)
    source_rows = getattr(episode, 'source_1s_row', None)
    return dict(round=round_index, phase=phase, sample_id=episode.sample_id,
        failure_reason=status, epsilon=float(epsilon), executed_steps=index,
        failed_elapsed_time_s=float(episode.time_s[index]),
        failed_source_time_s=float(times[index]) if times is not None else None,
        failed_timestamp=str(timestamps[index]) if timestamps is not None else None,
        source_1s_row=int(source_rows[index]) if source_rows is not None else None,
        soc=soc, previous_soc=previous_soc, raw_load_kw=load,
        physical_load_kw=normalize_onboard_load_kw(load), observed_load_over_600=load/600.,
        previous_fc_kw=previous_fc, physical_legal_fc_kw=list(physical),
        physical_legal_fc_min_kw=min(physical) if physical else None,
        physical_legal_fc_max_kw=max(physical) if physical else None,
        minimum_grid_increase_kw=max(0, min(physical)-previous_fc) if physical else None,
        minimum_grid_decrease_kw=max(0, previous_fc-max(physical)) if physical else None,
        increment_legal_fc_kw=list(allowed),
        all13_legal_mask=[row['allowed'] for row in candidates],
        all13_candidates=candidates,
        grid_interval_note='min/max of legal absolute 10 kW grid powers, not exact continuous bounds',
        last_executed_time_s=float(values[-1].relative_time_s) if values else None,
        state_provenance='Float64 executed transition SOC/FC; original episode row at F; no new rollout')


def _profile_rows(values, agent):
    cumulative = dict(fc_h2=0., battery_h2eq=0., soc_penalty=0., fc_ramp=0.,
                      failure_penalty=0., reward=0.)
    for start in range(0, len(values), 2048):
        batch = values[start:start+2048]
        with torch.inference_mode():
            q_batch = agent.online(torch.tensor(
                [t.state for t in batch], dtype=torch.float32, device=agent.device)).cpu().numpy()
        if not np.isfinite(q_batch).all():
            raise FloatingPointError('non-finite Validation Q')
        for t, q in zip(batch, q_batch):
            legal = [agent.action_index_for_power(p, t.state) for p in t.policy_candidate_actions]
            ranked = sorted(legal, key=lambda i: float(q[i]), reverse=True)
            components = dict(fc_h2=t.executed_ledger.h2_cost_cny,
                battery_h2eq=t.soc_equivalent_hydrogen_cost_cny,
                soc_penalty=t.soc_soft_penalty_cny, fc_ramp=t.fc_ramp_penalty_cny,
                failure_penalty=t.failure_penalty_equivalent_cny, reward=t.reward_cny)
            for key, value in components.items():
                cumulative[key] += value
            yield dict(time_s=t.relative_time_s, load_kw=t.raw_load_kw, fc_kw=t.action_kw,
                battery_kw=t.actual_battery_kw, soc_before=t.state[0], soc=t.actual_soc,
                fc_delta_kw=t.action_delta_kw, done=t.done, terminal_reason=t.terminal_reason,
                hold_q=float(q[6]), hold_legal=6 in legal,
                hold_rank=1+sum(q[i] > q[6] for i in legal) if 6 in legal else None,
                legal_indices=json.dumps(legal),
                legal_action_ranking_delta_kw=json.dumps([agent.action_values_kw[i] for i in ranked]),
                best_minus_second_q=(float(q[ranked[0]]-q[ranked[1]]) if len(ranked)>1 else None),
                selected_q=float(q[t.action_index]), selected_minus_hold=float(q[t.action_index]-q[6]),
                fc_degradation_cost_cny=t.executed_ledger.fuel_cell_degradation_cost_cny,
                battery_degradation_cost_cny=t.executed_ledger.battery_degradation_cost_cny,
                **components, **{f'cumulative_{key}': value for key, value in cumulative.items()},
                **{f'q_delta_{delta}': float(q[i]) for i, delta in enumerate(agent.action_values_kw)})


def quiet_evaluate(validation, agent, accountant, economics, penalty, output, round_index):
    """Greedy evaluation only; stream CSV rows, never append to Train Replay."""
    output = Path(output)
    folder = output / f'round_{round_index:02d}_validation'
    folder.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for episode in validation:
        values, status = execute(episode, lambda s, a: agent.select_power(s, a, epsilon=0.),
                                 accountant, economics, penalty)
        rows.append(sample_record(episode, values, status))
        if status != 'completed':
            failures.append(describe_failure(episode, values, status, round_index,
                                             'greedy_validation', 0., accountant))
        if values:
            with gzip.open(folder/f'{episode.sample_id}.csv.gz', 'wt',
                           encoding='utf-8', newline='') as handle:
                writer = None
                for row in _profile_rows(values, agent):
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
            plot_profile(folder/f'{episode.sample_id}.png', episode.sample_id, values)
    write_csv(folder/'sample_metrics.csv', rows)
    return split_summary(rows), rows, failures


def _number(row, key):
    value = row.get(key)
    return None if value in (None, '') else float(value)


def _is_complete(row):
    return row['completed'] is True or row['completed'] == 'True'


def _normalize_reference(row, current, *, v5=False):
    duration = float(row['executed_duration_s'])
    return dict(sample_id=row['sample_id'], completed=_is_complete(row),
        status=(row.get('failure_cause') or 'completed') if v5 else row['status'],
        executed_duration_s=duration,
        duration_coverage=(duration/current['voyage_duration_s'] if v5 else float(row['duration_coverage'])),
        soc_min=float(row['soc_min']), soc_max=float(row['soc_max']),
        soc_terminal=float(row['soc_terminal']),
        fc_nonzero_changes=int(row['fc_nonzero_changes']),
        fc_adjacent_reversals=int(row['fc_adjacent_reversals']),
        fc_changes_per_hour=float(row['fc_nonzero_changes'])*3600./duration if duration else 0.,
        fc_reversals_per_hour=float(row['fc_adjacent_reversals'])*3600./duration if duration else 0.,
        complete_economic_cost_cny=_number(row, 'complete_zhou_cost_cny' if v5 else 'complete_economic_cost_cny'),
        fc_degradation_cost_cny=float(row['fc_degradation_cost_cny']))


def write_comparison(output, round_index, rows, reference_root):
    """Compare existing V6 R2/R8 and V5 R32 on jointly completed voyages only.

    ``reference_root`` is the project root. Different references can yield
    different common subsets; every aggregate explicitly lists its IDs.
    """
    output, reference_root = Path(output), Path(reference_root)
    with (reference_root/'data/reference/v6_run02/validation_samples.csv').open(
            encoding='utf-8-sig', newline='') as handle:
        historical = list(csv.DictReader(handle))
    references = [(f'v6_round_{ri:02d}', ri,
                   {r['sample_id']: r for r in historical if int(r['round']) == ri}, False)
                  for ri in (2, 8)]
    v5_path = reference_root/'docs/results/v5_economic4_paused_20261021/sample_metrics.csv'
    with v5_path.open(encoding='utf-8-sig', newline='') as handle:
        v5 = {r['sample_id']: r for r in csv.DictReader(handle)
              if r['split'] == 'validation' and int(r['round']) == 32}
    references.append(('v5_round_32', 32, v5, True))
    comparisons, aggregates = [], []
    for label, prior_round, source, v5 in references:
        common, old_complete, new_complete = [], [], []
        for current in rows:
            previous = source.get(current['sample_id'])
            prior = _normalize_reference(previous, current, v5=v5) if previous is not None else None
            jointly_complete = bool(prior is not None and prior['completed'] and current['completed'])
            if jointly_complete:
                common.append(current['sample_id'])
                old_complete.append(prior)
                new_complete.append(current)
            comparisons.append(dict(reference=label, reference_round=prior_round,
                sb3_round=round_index, sample_id=current['sample_id'], reference_available=prior is not None,
                reference_completed=prior['completed'] if prior else None,
                sb3_completed=bool(current['completed']),
                reference_status=prior['status'] if prior else None, sb3_status=current['status'],
                reference_duration_coverage=prior['duration_coverage'] if prior else None,
                sb3_duration_coverage=current['duration_coverage'],
                reference_fc_changes_per_hour=prior['fc_changes_per_hour'] if prior else None,
                sb3_fc_changes_per_hour=current['fc_changes_per_hour'],
                reference_fc_reversals_per_hour=prior['fc_reversals_per_hour'] if prior else None,
                sb3_fc_reversals_per_hour=current['fc_reversals_per_hour'],
                reference_soc_min=prior['soc_min'] if prior else None,
                reference_soc_max=prior['soc_max'] if prior else None,
                reference_soc_terminal=prior['soc_terminal'] if prior else None,
                sb3_soc_min=current['soc_min'], sb3_soc_max=current['soc_max'],
                sb3_soc_terminal=current['soc_terminal'], same_complete_sample=jointly_complete,
                reference_common_complete_cost_cny=prior['complete_economic_cost_cny'] if jointly_complete else None,
                sb3_common_complete_cost_cny=current['complete_economic_cost_cny'] if jointly_complete else None))
        old_duration = math.fsum(r['executed_duration_s'] for r in old_complete)
        new_duration = math.fsum(r['executed_duration_s'] for r in new_complete)
        aggregates.append(dict(reference=label, reference_round=prior_round, sb3_round=round_index,
            common_completed_ids=common, common_completed_count=len(common),
            reference_common_executed_duration_s=old_duration,
            sb3_common_executed_duration_s=new_duration,
            reference_common_complete_cost_cny=(math.fsum(r['complete_economic_cost_cny'] for r in old_complete)
                                                 if common else None),
            sb3_common_complete_cost_cny=(math.fsum(r['complete_economic_cost_cny'] for r in new_complete)
                                           if common else None),
            reference_common_fc_degradation_cost_cny=(math.fsum(r['fc_degradation_cost_cny'] for r in old_complete)
                                                      if common else None),
            sb3_common_fc_degradation_cost_cny=(math.fsum(r['fc_degradation_cost_cny'] for r in new_complete)
                                                if common else None),
            reference_common_fc_changes_per_hour=(sum(r['fc_nonzero_changes'] for r in old_complete)*3600./old_duration
                                                   if old_duration else None),
            sb3_common_fc_changes_per_hour=(sum(r['fc_nonzero_changes'] for r in new_complete)*3600./new_duration
                                            if new_duration else None),
            reference_common_fc_reversals_per_hour=(sum(r['fc_adjacent_reversals'] for r in old_complete)*3600./old_duration
                                                     if old_duration else None),
            sb3_common_fc_reversals_per_hour=(sum(r['fc_adjacent_reversals'] for r in new_complete)*3600./new_duration
                                              if new_duration else None)))
    write_csv(output/f'round_{round_index:02d}_reference_comparison.csv', comparisons)
    serializable_aggregates = [dict(r, common_completed_ids=json.dumps(r['common_completed_ids'])) for r in aggregates]
    write_csv(output/f'round_{round_index:02d}_common_completed_comparison.csv', serializable_aggregates)
    result = dict(round=round_index, comparisons=comparisons, common_completed=aggregates,
        cost_scope='Complete costs only on each explicitly listed jointly completed subset; no failed prefix cost substitution.',
        reference_root=str(reference_root),
        v5_coverage_note='V5 executed duration divided by the same sample ID canonical voyage duration; 30-second and 1-second action counts are not compared as equivalent counts.')
    write_json(output/f'round_{round_index:02d}_reference_comparison.json', result)
    return result
