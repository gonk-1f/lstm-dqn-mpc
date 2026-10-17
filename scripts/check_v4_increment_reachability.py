"""Bounded offline Train/Validation reachability check; never opens Test.

The SOC hull for each of the 61 physical FC levels is an outer relaxation:
an empty hull proves infeasibility, a nonempty hull alone proves nothing.
Separately keep two actually reached SOC points per FC level and predecessor
indices. A recovered full sequence is checked by the formal replay environment.
Witnesses use the full observed load record offline; they are noncausal
existence witnesses, not an online control policy or economic optimum.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from v2.data.supervisory_rules import normalize_onboard_load_kw
from v3.control import EconomicMPC, SOC_MAX, SOC_MIN
from v4.actions import FC_INCREMENT_KW
from v4.control import replay_episode
from v4.independent_dataset import IndependentTrainingDataset
from v4.independent_economics import ZhouEvaluationConfig


DELTAS = np.array(FC_INCREMENT_KW, dtype=int)
POWERS = np.arange(0, 601, 10)
PREVIOUS = np.arange(61)[:, None] - DELTAS[None, :] // 10
VALID_PREVIOUS = (PREVIOUS >= 0) & (PREVIOUS < 61)
PREVIOUS = np.clip(PREVIOUS, 0, 60)  # Array lookup only; never clips FC commands.


def search(episode, accountant):
    """Return certified outer failure or a real finite-path candidate."""
    low = np.full(61, np.inf)
    high = np.full(61, -np.inf)
    low[0] = high[0] = .6
    points = np.full((2, 61), np.nan)
    points[:, 0] = .6
    parents = np.full((episode.step_count, 2, 61), -1, dtype=np.int16)
    eta_chg, eta_dis = accountant.efficiency.require_calibrated()
    plant = accountant.plant
    hull_failure = None
    point_failure = None
    for row, (raw_load, duration) in enumerate(zip(episode.load_kw, episode.dt_s)):
        load = normalize_onboard_load_kw(raw_load)
        battery = load - POWERS
        power_ok = ((battery >= plant.battery_charge_min_kw)
                    & (battery <= plant.battery_discharge_max_kw))
        # Same piecewise bus-to-cell equation as formal next_soc; all final
        # witnesses are executed again by the formal replay, not trusted here.
        cell = np.where(battery > 0, battery / eta_dis, eta_chg * battery)
        shift = -cell * (float(duration) / 3600) / plant.battery_nominal_energy_kwh
        old_low, old_high = low.copy(), high.copy()
        candidates_low = np.where(VALID_PREVIOUS, low[PREVIOUS], np.inf)
        candidates_high = np.where(VALID_PREVIOUS, high[PREVIOUS], -np.inf)
        low = np.maximum(SOC_MIN, candidates_low.min(axis=1) + shift)
        high = np.minimum(SOC_MAX, candidates_high.max(axis=1) + shift)
        hull_valid = power_ok & (low <= high)
        low = np.where(hull_valid, low, np.inf)
        high = np.where(hull_valid, high, -np.inf)
        if not hull_valid.any():
            reachable = np.isfinite(old_low)
            attempted = (reachable[PREVIOUS] & VALID_PREVIOUS)
            fc_candidates = POWERS[attempted.any(axis=1)]
            hull_failure = {
                'failure_row': row, 'failure_time_s': float(episode.time_s[row]),
                'failure_timestamp': str(episode.timestamp[row]), 'failure_load_kw': load,
                'preceding_soc_min': float(old_low[reachable].min()),
                'preceding_soc_max': float(old_high[reachable].max()),
                'reachable_previous_fc_kw': '|'.join(map(str, POWERS[reachable])),
                'attempted_fc_kw': '|'.join(map(str, fc_candidates)),
                'attempted_next_soc_max': float((candidates_high.max(axis=1) + shift)[
                    attempted.any(axis=1) & power_ok].max(initial=-np.inf)),
            }
            break
        if point_failure is not None:
            continue
        candidates = points[:, PREVIOUS] + shift[None, :, None]
        valid = (VALID_PREVIOUS[None, :, :] & power_ok[None, :, None]
                 & np.isfinite(candidates)
                 & (candidates >= SOC_MIN) & (candidates <= SOC_MAX))
        # Flatten (previous side, delta) while preserving the destination FC.
        candidate_matrix = candidates.transpose(1, 0, 2).reshape(61, 14)
        valid_matrix = valid.transpose(1, 0, 2).reshape(61, 14)
        minimum_index = np.where(valid_matrix, candidate_matrix, np.inf).argmin(axis=1)
        maximum_index = np.where(valid_matrix, candidate_matrix, -np.inf).argmax(axis=1)
        destination_valid = valid_matrix.any(axis=1)
        for side, chosen in enumerate((minimum_index, maximum_index)):
            points[side] = np.where(destination_valid,
                                   candidate_matrix[np.arange(61), chosen], np.nan)
            previous_side, delta_index = np.divmod(chosen, 7)
            parent = previous_side * 61 + PREVIOUS[np.arange(61), delta_index]
            parents[row, side] = np.where(destination_valid, parent, -1)
        if not destination_valid.any():
            point_failure = row
    if hull_failure is not None:
        return 'proven_infeasible_outer_hull', None, hull_failure
    if point_failure is not None:
        return 'unresolved_nonempty_hull_no_witness', None, {'failure_row': point_failure}
    # One actual reached terminal point, then exact predecessor backtracking.
    final = int(np.nanargmax(points))
    sequence = np.empty(episode.step_count, dtype=int)
    for row in range(episode.step_count - 1, -1, -1):
        side, fc_index = divmod(final, 61)
        sequence[row] = int(POWERS[fc_index])
        final = int(parents[row, side, fc_index])
        if final < 0:
            raise RuntimeError('actual-point witness has a broken predecessor chain')
    if final % 61 != 0 or not np.isin(np.diff(np.r_[0, sequence]), DELTAS).all():
        raise RuntimeError('witness starts from a nonzero FC or uses an illegal increment')
    return 'candidate_witness', sequence, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('data/processed/operating_dataset_independent_v2'))
    parser.add_argument('--output', type=Path,
                        default=Path('docs/results/v4_increment7_reachability_20261017.csv'))
    args = parser.parse_args()
    dataset = IndependentTrainingDataset.open(args.data_root)
    accountant = EconomicMPC(nominal_cost_cny=1)
    config = ZhouEvaluationConfig.from_formal_curve()
    results = []
    for episode in dataset.load_train() + dataset.load_validation():
        status, sequence, evidence = search(episode, accountant)
        result = dict(split=episode.split, sample_id=episode.sample_id,
                      steps=episode.step_count, duration_s=episode.duration_s,
                      first_raw_load_kw=float(episode.load_kw[0]),
                      peak_raw_load_kw=float(np.max(episode.load_kw)), status=status,
                      method='SOC hull relaxation plus two-actual-points-per-FC search',
                      witness_kind='noncausal offline existence witness', **evidence)
        if sequence is not None:
            commands = iter(map(int, sequence))
            previous = 0

            def witness_policy(_state, physical_actions):
                nonlocal previous
                command = next(commands)
                if command - previous not in DELTAS or command not in physical_actions:
                    raise RuntimeError('offline witness violates increment or formal physical mask')
                previous = command
                return command

            replay = replay_episode(episode, witness_policy, accountant=accountant,
                                    state_mode='minimal3', action_mode='increment7',
                                    reward_cost_mode='zhou_joint', beta_soc=200,
                                    economic_evaluation_config=config)
            if (len(replay.transitions) != episode.step_count
                    or not replay.transitions[-1].done
                    or replay.modeled_terminal_settlement is not None
                    or replay.shore_blocks):
                raise RuntimeError('witness did not complete a pure independent voyage')
            result.update(status='formal_replay_verified_witness',
                          formal_verification_api='increment7_zhou_joint',
                          first_fc_kw=int(sequence[0]),
                          maximum_absolute_delta_kw=int(np.abs(np.diff(np.r_[0, sequence])).max()),
                          minimum_soc=min(.6, *[t.actual_soc for t in replay.transitions]),
                          maximum_soc=max(.6, *[t.actual_soc for t in replay.transitions]),
                          terminal_soc=replay.final_state.soc,
                          zhou_total_cost_cny=replay.comparable_cost_cny,
                          sequence_sha256=hashlib.sha256(sequence.astype('<i2').tobytes()).hexdigest())
        results.append(result)
        print(episode.split, episode.sample_id, result['status'],
              result.get('failure_row', ''), flush=True)
    if dataset.opened_test_payloads:
        raise RuntimeError('sealed Test payload was opened')
    if dataset.identity_hashes() != dataset._initial_hashes:
        raise RuntimeError('dataset metadata changed during offline reachability check')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in results for key in row))
    with args.output.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    print('wrote', args.output, 'Test payloads opened:', dataset.opened_test_payloads, flush=True)


if __name__ == '__main__':
    main()
