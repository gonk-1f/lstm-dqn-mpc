"""Formal, frozen R100 replay and two explicitly offline interventions only."""
from dataclasses import asdict
import csv
import json
from math import fsum

from audit_support import (
    ROOT, OUT, load_policy, load_episode, load_profile, accountant_config,
    AccountState, physical_mask,
)
from v4.control import replay_episode, ReplayExecutionError
from v3.episode_replay import _ledger_total
from v4.independent_economics import evaluate_zhou_economics


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def motion(transitions):
    ds = [int(t.action_delta_kw) for t in transitions]
    return {
        'steps': len(ds), 'mean_abs_delta_kw': fsum(map(abs, ds))/len(ds) if ds else None,
        'max_abs_delta_kw': max(map(abs, ds), default=None),
        'adjacent_direction_reversals': sum(a*b < 0 for a, b in zip(ds, ds[1:])),
        'adjacent_pairs': max(0, len(ds)-1),
        'starts': sum(t.action_kw > 0 and t.state[2] == 0 for t in transitions),
        'stops': sum(t.action_kw == 0 and t.state[2] > 0 for t in transitions),
    }


def compare_archive(transitions, profile, error):
    rows = profile['transitions']
    if len(rows) != len(transitions):
        raise AssertionError(f'length mismatch: {len(rows)} != {len(transitions)}')
    maxima = {'soc_before': 0., 'soc_after': 0., 'load_kw': 0., 'battery_bus_kw': 0.,
              'reward_before_failure_penalty': 0., 'ledger': 0.}
    for i, (t, old) in enumerate(zip(transitions, rows)):
        if int(old['fc_kw']) != t.action_kw or old['action_index'] != t.action_index:
            raise AssertionError(f'archive action mismatch at {i}')
        if old['action_delta_kw'] != t.action_delta_kw:
            raise AssertionError(f'archive increment mismatch at {i}')
        if tuple(old['physical_feasible_actions_kw']) != t.physical_feasible_actions:
            raise AssertionError(f'archive physical mask mismatch at {i}')
        if tuple(old['policy_candidate_actions_kw']) != t.policy_candidate_actions:
            raise AssertionError(f'archive increment mask mismatch at {i}')
        for key, value in (('soc_before', t.state[0]), ('soc_after', t.actual_soc),
                           ('load_kw', t.raw_load_kw), ('battery_bus_kw', t.actual_battery_kw)):
            maxima[key] = max(maxima[key], abs(value-old[key]))
        old_reward_without_failure = old['new_reward'] + old['failure_penalty_equivalent_cny']
        maxima['reward_before_failure_penalty'] = max(maxima['reward_before_failure_penalty'],
                                                    abs(t.reward_cny-old_reward_without_failure))
        for key, value in old['original_economic_ledger'].items():
            maxima['ledger'] = max(maxima['ledger'], abs(getattr(t.executed_ledger, key)-value))
    if max(maxima.values()) > 1e-8:
        raise AssertionError(f'archive numerical mismatch: {maxima}')
    if bool(error) == bool(profile['completed']):
        raise AssertionError('completion mismatch')
    return {'exact_actions_masks_and_length': True, 'maximum_absolute_differences': maxima,
            'last_original_failure_penalty': rows[-1]['failure_penalty_equivalent_cny'],
            'note': 'Formal replay returns executed actions before failure-penalty attachment by training; '
                    'the archived single failure penalty is excluded only for reward equality checking.'}


def run(episode, frozen, accountant, config, variant, window_start, window_stop):
    physical = AccountState()
    decision = 0
    shadows = []

    def policy(state, feasible):
        nonlocal physical, decision
        if abs(state[0]-physical.soc) > 1e-12 or abs(state[2]*600-physical.previous_fc_kw) > 1e-9:
            raise AssertionError('shadow physical state diverged')
        load = state[1]*600
        duration = float(episode.dt_s[decision])
        full, expected = physical_mask(physical, load, accountant, duration)
        if feasible != expected:
            raise AssertionError('formal mask mismatch')
        if variant == 'target150_window' and window_start <= decision < window_stop:
            chosen = min(feasible, key=lambda fc: (abs(fc-150), fc))
        elif variant == 'max_next_soc_before_window' and decision < window_start:
            chosen = max(feasible, key=lambda fc: (accountant.interval(
                physical, fc, load, duration_s=duration)[0].soc, -fc))
        else:
            chosen = frozen(state, feasible)
        shadows.append(asdict(physical))
        physical = accountant.interval(physical, chosen, load, duration_s=duration)[0]
        decision += 1
        return chosen

    try:
        result = replay_episode(episode, policy, accountant=accountant, beta_soc=200.,
                                state_mode='minimal3', action_mode='increment7',
                                reward_cost_mode='zhou_joint', economic_evaluation_config=config)
        transitions, error = result.transitions, None
        evaluation = result.economic_evaluation
    except ReplayExecutionError as exc:
        result, transitions, error = None, exc.executed_transitions, exc
        ledger = _ledger_total([t.executed_ledger for t in transitions])
        evaluation = evaluate_zhou_economics(ledger, soc_initial=.6,
                                            soc_terminal=physical.soc, config=config)
    if len(shadows) != len(transitions):
        raise AssertionError('shadow action count mismatch')
    e = asdict(evaluation)
    selected = transitions[window_start:window_stop]
    summary = {
        'sample_id': episode.sample_id, 'split': episode.split, 'variant': variant,
        'completed': error is None, 'executed_steps': len(transitions),
        'total_episode_intervals': len(episode.dt_s),
        'failed_row_index_0based': error.row_index if error else None,
        'failure_kind': error.failure_kind if error else None,
        'failure_cause': error.failure_cause if error else None,
        'cost_scope': 'complete_voyage' if error is None else 'executed_prefix_not_comparable_full_cost',
        'soc_min_including_initial': min([.6] + [t.actual_soc for t in transitions]),
        'soc_max_including_initial': max([.6] + [t.actual_soc for t in transitions]),
        'soc_terminal_or_failure': physical.soc,
        'window_start_index_0based': window_start,
        'window_stop_index_exclusive': window_stop,
        'soc_entering_window': (transitions[window_start].state[0]
                                if len(transitions) > window_start else None),
        'fc_entering_window_kw': (transitions[window_start].state[2]*600
                                 if len(transitions) > window_start else None),
        **{key: e[key] for key in ('actual_hydrogen_mass_kg', 'soc_equivalent_hydrogen_mass_kg',
                                  'equivalent_hydrogen_cost_cny', 'fuel_cell_degradation_cost_cny',
                                  'battery_degradation_cost_cny', 'total_cost_cny')},
        'soc_auxiliary_penalty_sum': fsum(t.soc_soft_penalty_cny for t in transitions),
        'quadratic_ramp_penalty_sum': fsum(t.fc_ramp_penalty_cny for t in transitions),
        'normal_raw_training_reward_sum_without_failure_penalty': fsum(t.reward_cny for t in transitions),
        'normal_scaled_training_reward_sum_without_failure_penalty': fsum(t.reward_cny*.001 for t in transitions),
        'step_equivalent_h2_cost_sum': fsum(t.soc_equivalent_hydrogen_cost_cny for t in transitions),
        **{'full_'+k: v for k, v in motion(transitions).items()},
        **{'window_'+k: v for k, v in motion(selected).items()},
    }
    if error:
        failed_load = float(episode.load_kw[error.row_index])
        full, increment = physical_mask(physical, failed_load, accountant,
                                       float(episode.dt_s[error.row_index]))
        summary['failed_load_kw'] = failed_load
        summary['failed_previous_fc_kw'] = physical.previous_fc_kw
        summary['failed_physical_fc_min_kw'] = min(full, default=None)
        summary['failed_physical_fc_max_kw'] = max(full, default=None)
        summary['failed_physical_candidate_count'] = len(full)
        summary['failed_increment_candidates_kw'] = json.dumps(increment)
    else:
        for key in ('failed_load_kw', 'failed_previous_fc_kw', 'failed_physical_fc_min_kw',
                    'failed_physical_fc_max_kw', 'failed_physical_candidate_count',
                    'failed_increment_candidates_kw'):
            summary[key] = None
    identity_difference = abs(-summary['normal_raw_training_reward_sum_without_failure_penalty']
                              - summary['total_cost_cny'] - summary['soc_auxiliary_penalty_sum']
                              - summary['quadratic_ramp_penalty_sum'])
    summary['reward_economic_identity_abs_error'] = identity_difference
    if identity_difference > 1e-8:
        raise AssertionError('reward and whole-voyage economic identity mismatch')
    rows = []
    for i, (t, shadow) in enumerate(zip(transitions, shadows)):
        rows.append({'sample_id': episode.sample_id, 'variant': variant, 'step_1based': i+1,
                     'relative_time_s': t.relative_time_s, 'duration_s': t.duration_s,
                     'load_kw': t.raw_load_kw, 'fc_kw': t.action_kw,
                     'previous_fc_kw': shadow['previous_fc_kw'], 'delta_fc_kw': t.action_delta_kw,
                     'soc_before': t.state[0], 'soc_after': t.actual_soc,
                     'battery_bus_kw': t.actual_battery_kw,
                     'fc_loss_before_uv': shadow['fc_loss_uv'],
                     'battery_weighted_ah_before': shadow['battery_weighted_ah'],
                     'h2_cost_cny': t.executed_ledger.h2_cost_cny,
                     'fc_degradation_cost_cny': t.executed_ledger.fuel_cell_degradation_cost_cny,
                     'battery_degradation_cost_cny': t.executed_ledger.battery_degradation_cost_cny,
                     'soc_equivalent_hydrogen_cost_cny': t.soc_equivalent_hydrogen_cost_cny,
                     'soc_penalty': t.soc_soft_penalty_cny,
                     'quadratic_ramp_penalty': t.fc_ramp_penalty_cny,
                     'raw_training_reward_without_failure_attachment': t.reward_cny,
                     'policy_candidates_kw': json.dumps(t.policy_candidate_actions),
                     'intervention': ((variant == 'target150_window' and window_start <= i < window_stop)
                                      or (variant == 'max_next_soc_before_window' and i < window_start))})
    write_csv(OUT / f'counterfactual_{episode.sample_id}_{variant}.csv', rows)
    return summary, transitions, error


def main():
    frozen, cp, metadata = load_policy()
    accountant, config = accountant_config(cp)
    zoom = list(csv.DictReader((ROOT / 'docs/results/v4_increment7_zhou_td1_per_100r_20261018/'
                                'round100_validation_parent_020_seg01_zoom.csv').open(encoding='utf-8-sig')))
    steps = [int(r['step_1based']) for r in zoom]
    assert steps == list(range(steps[0], steps[0]+64))
    bounds = {'parent_020_seg01': (steps[0]-1, steps[-1]), 'parent_015_seg01': (123,155)}
    summaries, verification = [], {}
    for sid, split, variant in [('parent_020_seg01', 'validation', 'target150_window'),
                                ('parent_015_seg01', 'train', 'max_next_soc_before_window')]:
        episode, hashes = load_episode(sid, split)
        profile = load_profile(sid, split)
        begin, end = bounds[sid]
        if sid == 'parent_015_seg01':
            assert len(profile['transitions']) == end and begin == end-32
        baseline, transitions, error = run(episode, frozen, accountant, config, 'greedy_r100', begin, end)
        verification[sid] = compare_archive(transitions, profile, error)
        summaries.append(baseline)
        counterfactual, _, _ = run(episode, frozen, accountant, config, variant, begin, end)
        summaries.append(counterfactual)
    write_csv(OUT / 'counterfactual_summary.csv', summaries)
    payload = {'checkpoint_round': cp['round'], 'experiment_id': cp['experiment_id'],
               'source_commit': cp['source_commit'], 'archive_reproduction': verification,
               'interventions': {
                   'target150_window': 'CSV actual 1-based steps 54..117 inclusive (zero-based 53..116); '
                                       'legal candidate nearest 150 kW, ties lower FC; outside window frozen Greedy.',
                   'max_next_soc_before_window': 'Before zero-based step 123 choose legal candidate giving '
                                                'largest formal next SOC, ties lower FC; then frozen Greedy. '
                                                'Uses post hoc original failure-window location; this local '
                                                'greedy-SOC diagnostic is not a proven global reachable-SOC upper bound.'},
               'summaries': summaries}
    (OUT / 'counterfactual_details.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                   encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
