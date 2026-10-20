"""Read-only exact interval-accounting audit; never trains or saves a model."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from math import fsum

from audit_support import OUT, RUN, load_profile, accountant_config, AccountState, physical_mask
from v4.control import (
    ledger_training_reward, soc_deviation_squared, FC_SQUARED_RAMP_REWARD_PER_KW2,
)
from v4.independent_economics import soc_equivalent_hydrogen_cost_cny


def average(values):
    return fsum(values) / len(values) if values else None


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0}
    def quantile(p):
        x = p * (len(values) - 1)
        i = int(x)
        return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (x - i)
    return {'n': len(values), 'mean': average(values), 'min': values[0],
            'p05': quantile(.05), 'median': quantile(.5), 'p95': quantile(.95), 'max': values[-1]}


def main():
    # Read identity-only metadata. All physics/config paths are the formal implementations.
    metadata = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
    hp = metadata['hyperparameters']
    assert hp['state_mode'] == 'minimal3' and hp['action_mode'] == 'increment7'
    assert hp['reward_cost_mode'] == 'zhou_joint'
    accountant, evaluation = accountant_config(metadata)
    beta_soc = hp['beta_soc']
    reward_scale = hp['reward_scale']
    rows, summary, snapshots = [], {}, {}
    ledger_names = ('h2_cost_cny', 'fuel_cell_degradation_cost_cny',
                    'battery_degradation_cost_cny', 'shore_cost_cny')
    for sample_id, split, first, last in (
        ('parent_020_seg01', 'validation', 54, 117),
        ('parent_015_seg01', 'train', 1, 100000),
    ):
        profile = load_profile(sample_id, split)
        transitions = profile['transitions']
        state = AccountState()
        errors = defaultdict(float)
        sample_rows, all_actual_rows = [], []
        selected_steps = []
        snapshots[sample_id] = []
        for step, tr in enumerate(transitions, 1):
            dt, load, actual_fc = tr['duration_s'], tr['load_kw'], tr['fc_kw']
            previous_fc = actual_fc - tr['action_delta_kw']
            errors['previous_fc_kw'] = max(errors['previous_fc_kw'], abs(previous_fc - state.previous_fc_kw))
            errors['soc_before'] = max(errors['soc_before'], abs(tr['soc_before'] - state.soc))
            physical, candidates = physical_mask(state, load, accountant, dt)
            assert list(physical) == tr['physical_feasible_actions_kw'], (sample_id, step, 'physical mask')
            assert list(candidates) == tr['policy_candidate_actions_kw'], (sample_id, step, 'increment mask')
            assert actual_fc in candidates
            snapshots[sample_id].append({'step_1based': step, 'before': asdict(state)})
            next_actual, actual_ledger, actual_battery = accountant.interval(state, actual_fc, load, duration_s=dt)
            errors['soc_after'] = max(errors['soc_after'], abs(tr['soc_after'] - next_actual.soc))
            errors['battery_bus_kw'] = max(errors['battery_bus_kw'], abs(tr['battery_bus_kw'] - actual_battery))
            for name in ledger_names:
                errors[name] = max(errors[name], abs(getattr(actual_ledger, name) - tr['original_economic_ledger'][name]))

            requested = {actual_fc: ['actual']}
            for delta, label in ((0, 'hold'), (-60, 'minus60'), (60, 'plus60')):
                fc = int(previous_fc + delta)
                if fc in candidates:
                    requested.setdefault(fc, []).append(label)
            actual_row = None
            for fc, labels in requested.items():
                successor, ledger, battery = accountant.interval(state, fc, load, duration_s=dt)
                delta = fc - state.previous_fc_kw
                soc_energy = soc_equivalent_hydrogen_cost_cny(state.soc, successor.soc, evaluation)
                soc_aux = beta_soc * soc_deviation_squared(successor.soc)
                ramp_aux = FC_SQUARED_RAMP_REWARD_PER_KW2 * delta ** 2
                normal = ledger_training_reward(ledger, 'zhou_joint') - soc_energy - soc_aux - ramp_aux
                penalty = tr['failure_penalty_equivalent_cny'] if 'actual' in labels else 0.0
                reward_with_actual_terminal_penalty = normal - penalty
                row = {
                    'sample_id': sample_id, 'split': split, 'step_1based': step,
                    'relative_time_s': tr['relative_time_s'], 'duration_s': dt,
                    'candidate_role': '|'.join(labels), 'is_actual': 'actual' in labels,
                    'counterfactual_is_single_step_only': 'actual' not in labels,
                    'load_kw': load, 'soc_before': state.soc,
                    'previous_fc_kw': state.previous_fc_kw,
                    'fc_loss_before_uv': state.fc_loss_uv,
                    'battery_weighted_ah_before': state.battery_weighted_ah,
                    'candidate_delta_kw': delta, 'candidate_fc_kw': fc,
                    'candidate_battery_kw': battery, 'candidate_soc_after': successor.soc,
                    'candidate_fc_loss_after_uv': successor.fc_loss_uv,
                    'candidate_battery_weighted_ah_after': successor.battery_weighted_ah,
                    'physical_feasible_actions_kw': '|'.join(map(str, physical)),
                    'increment_candidate_powers_kw': '|'.join(map(str, candidates)),
                    **{name: getattr(ledger, name) for name in ledger_names},
                    'signed_soc_equivalent_hydrogen_cost_cny': soc_energy,
                    'zhou_three_cost_cny': fsum((ledger.h2_cost_cny, soc_energy,
                                                ledger.fuel_cell_degradation_cost_cny,
                                                ledger.battery_degradation_cost_cny)),
                    'soc_auxiliary_reward_equivalent_penalty': soc_aux,
                    'squared_ramp_auxiliary_reward_equivalent_penalty': ramp_aux,
                    'normal_training_reward_unscaled': normal,
                    'normal_training_reward_scaled_once': normal * reward_scale,
                    'actual_failure_penalty_unscaled': penalty,
                    'candidate_current_normal_plus_actual_recorded_penalty_scaled': reward_with_actual_terminal_penalty * reward_scale,
                    'actual_recorded_training_reward_unscaled': tr['new_reward'] if 'actual' in labels else '',
                    'actual_recorded_training_reward_scaled_once': tr['scaled_training_reward'] if 'actual' in labels else '',
                    'actual_archived_action_delta_kw': tr['action_delta_kw'],
                    'actual_archived_terminal_reason': tr['terminal_reason'],
                }
                if 'actual' in labels:
                    errors['soc_aux'] = max(errors['soc_aux'], abs(soc_aux - tr['soc_soft_penalty']))
                    errors['ramp_aux'] = max(errors['ramp_aux'], abs(ramp_aux - tr['fc_ramp_penalty_cny']))
                    errors['soc_energy'] = max(errors['soc_energy'], abs(soc_energy - tr['soc_equivalent_hydrogen_cost_cny']))
                    errors['reward_unscaled'] = max(errors['reward_unscaled'], abs(reward_with_actual_terminal_penalty - tr['new_reward']))
                    errors['reward_scaled'] = max(errors['reward_scaled'], abs(reward_with_actual_terminal_penalty * reward_scale - tr['scaled_training_reward']))
                    actual_row = row
                if first <= step <= last:
                    rows.append(row)
                    sample_rows.append(row)
            assert actual_row is not None
            all_actual_rows.append(actual_row)
            if first <= step <= last:
                selected_steps.append(step)
            state = next_actual
        assert max(errors.values()) < 1e-8, (sample_id, dict(errors))
        by_step = defaultdict(list)
        for row in sample_rows:
            by_step[row['step_1based']].append(row)
        comparisons, hold_comparisons = [], []
        for step, values in by_step.items():
            actual = next(row for row in values if row['is_actual'])
            best = max(values, key=lambda r: r['normal_training_reward_unscaled'])
            comparison = {
                'step_1based': step,
                'actual_delta_kw': actual['candidate_delta_kw'],
                'best_tested_immediate_delta_kw': best['candidate_delta_kw'],
                'actual_minus_best_tested_normal_reward_scaled':
                    actual['normal_training_reward_scaled_once'] - best['normal_training_reward_scaled_once'],
            }
            hold = next((row for row in values if 'hold' in row['candidate_role'].split('|')), None)
            if hold:
                comparison['actual_minus_hold_normal_reward_scaled'] = (
                    actual['normal_training_reward_scaled_once'] - hold['normal_training_reward_scaled_once'])
                comparison['actual_minus_hold_three_cost_cny'] = actual['zhou_three_cost_cny'] - hold['zhou_three_cost_cny']
                comparison['actual_minus_hold_soc_aux'] = actual['soc_auxiliary_reward_equivalent_penalty'] - hold['soc_auxiliary_reward_equivalent_penalty']
                comparison['actual_minus_hold_ramp_aux'] = actual['squared_ramp_auxiliary_reward_equivalent_penalty'] - hold['squared_ramp_auxiliary_reward_equivalent_penalty']
                hold_comparisons.append(comparison)
            comparisons.append(comparison)
        totals = {name: fsum(row[name] for row in all_actual_rows) for name in (
            *ledger_names, 'signed_soc_equivalent_hydrogen_cost_cny', 'zhou_three_cost_cny',
            'soc_auxiliary_reward_equivalent_penalty',
            'squared_ramp_auxiliary_reward_equivalent_penalty',
            'normal_training_reward_unscaled', 'normal_training_reward_scaled_once',
            'actual_failure_penalty_unscaled')}
        totals['actual_recorded_training_reward_scaled_once'] = fsum(row['actual_recorded_training_reward_scaled_once'] for row in all_actual_rows)
        telescope = soc_equivalent_hydrogen_cost_cny(.6, state.soc, evaluation)
        assert abs(telescope - totals['signed_soc_equivalent_hydrogen_cost_cny']) < 1e-8
        if profile['completed']:
            assert abs(totals['zhou_three_cost_cny'] - profile['economic_evaluation']['total_cost_cny']) < 1e-8
        summary[sample_id] = {
            'completed': profile['completed'], 'failure': profile['failure'],
            'actual_executed_steps': len(transitions), 'audited_window_first_last_1based': [min(selected_steps), max(selected_steps)],
            'audited_window_steps': len(selected_steps), 'candidate_rows': len(sample_rows),
            'initial_state': asdict(AccountState()), 'reconstructed_final_state': asdict(state),
            'max_reconstruction_absolute_errors': dict(errors), 'full_executed_prefix_totals': totals,
            'prefix_cost_is_complete_voyage_evaluation': profile['completed'],
            'soc_equivalent_telescoping_error_cny': totals['signed_soc_equivalent_hydrogen_cost_cny'] - telescope,
            'hold_legal_steps': len(hold_comparisons),
            'nonzero_actual_delta_steps_when_hold_legal': sum(c['actual_delta_kw'] != 0 for c in hold_comparisons),
            'actual_normal_reward_strictly_below_hold_steps': sum(c['actual_minus_hold_normal_reward_scaled'] < -1e-12 for c in hold_comparisons),
            'actual_normal_reward_strictly_above_hold_steps': sum(c['actual_minus_hold_normal_reward_scaled'] > 1e-12 for c in hold_comparisons),
            'actual_minus_hold_normal_reward_scaled_distribution': distribution([c['actual_minus_hold_normal_reward_scaled'] for c in hold_comparisons]),
            'actual_below_best_tested_immediate_steps': sum(c['actual_minus_best_tested_normal_reward_scaled'] < -1e-12 for c in comparisons),
            'actual_minus_best_tested_normal_reward_scaled_distribution': distribution([c['actual_minus_best_tested_normal_reward_scaled'] for c in comparisons]),
            'representative_comparisons': sorted(hold_comparisons, key=lambda c: c['actual_minus_hold_normal_reward_scaled'])[:5],
            'disclaimer': ('Candidates share each observed exact pre-action AccountState; they are one-step counterfactuals only. '
                           'No future failure penalty is invented for alternate actions. Actual terminal penalty is recorded once separately. '
                           'Candidate set is actual, hold, -60, +60 when legal, not necessarily all seven actions.'),
        }
    with (OUT / 'reward_decomposition.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'reward_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'reconstructed_account_states.json').write_text(json.dumps(snapshots, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
