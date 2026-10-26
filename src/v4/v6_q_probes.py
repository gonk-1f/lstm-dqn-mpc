"""Fixed physical states for Q reporting; no transitions enter training Replay."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from v3.control import AccountState
from .actions import FC_INCREMENT_13_KW, candidate_fc_actions
from .control import _causal_load_history, build_state, feasible_fc_actions


def build_probe_states(source, validation, old_run, accountant):
    """Freeze four V5 physical probes and five states around each old Val failure."""
    episodes = {episode.sample_id: episode for episode in validation}
    history = {name: _causal_load_history(episode.load_kw, episode.time_s)
               for name, episode in episodes.items()}
    probes = []

    def add(*, probe_id, group, sample_id, time_s, soc, previous_fc_kw,
            original_delta_kw=None, failure_time_s=None, offset_s=None):
        episode = episodes[sample_id]
        index = int(np.searchsorted(episode.time_s, time_s))
        load = float(episode.load_kw[index])
        physical = AccountState(soc=float(soc), previous_fc_kw=float(previous_fc_kw))
        lag, mean = history[sample_id]
        state = build_state(
            physical, (physical.previous_fc_kw,) * 3, (load,) * 3,
            state_mode='history5', load_30s_ago_kw=float(lag[index]),
            mean_load_300s_kw=float(mean[index]))
        physical_legal = feasible_fc_actions(
            physical, load, accountant, duration_s=float(episode.dt_s[index]))
        legal = candidate_fc_actions(physical_legal, physical.previous_fc_kw, 'increment13')
        indices = [FC_INCREMENT_13_KW.index(power - physical.previous_fc_kw) for power in legal]
        probes.append(dict(
            probe_id=probe_id, probe_group=group, sample_id=sample_id, split='validation',
            time_s=float(time_s), failure_time_s=failure_time_s,
            offset_from_failure_s=offset_s, state=list(state),
            previous_fc_kw=physical.previous_fc_kw, raw_load_kw=load,
            physical_legal_fc_kw=list(physical_legal), legal_indices=indices,
            legal_mask=[index in indices for index in range(13)],
            original_chosen_delta_kw=original_delta_kw,
            source=('V5 physical SOC/FC/time with V6 causal observed-load history'
                    if group == 'v5_fixed_physical' else
                    'old V6 run01 Validation decision state; F uses recorded failure SOC/FC; '
                    'earlier states use trace soc_before and FC minus delta; causal load history'),
        ))

    for item in source:
        instant = (int(item['step_1based']) - 1) * 30
        add(probe_id=f"v5_fixed/{item['sample_id']}/{instant}", group='v5_fixed_physical',
            sample_id=item['sample_id'], time_s=instant,
            soc=item['state'][0], previous_fc_kw=float(item['state'][2]) * 600.0,
            original_delta_kw=int(item['chosen_delta_kw']))

    old_run = Path(old_run)
    with (old_run / 'round_01_failure_details.csv').open(encoding='utf-8-sig', newline='') as handle:
        failures = [row for row in csv.DictReader(handle) if row['phase'] == 'greedy_validation']
    for failure in failures:
        sample_id = failure['sample_id']
        failed_time = float(failure['failed_elapsed_time_s'])
        requested = {failed_time - seconds for seconds in (60, 30, 5, 1)}
        with gzip.open(old_run / 'round_01_validation' / f'{sample_id}.csv.gz',
                       'rt', encoding='utf-8-sig', newline='') as handle:
            trace = {float(row['time_s']): row for row in csv.DictReader(handle)
                     if float(row['time_s']) in requested}
        for seconds in (60, 30, 5, 1, 0):
            instant = failed_time - seconds
            label = f'F-{seconds}' if seconds else 'F'
            row = trace[instant] if seconds else failure
            soc = float(row['soc_before']) if seconds else float(row['soc'])
            previous = (float(row['fc_kw']) - float(row['fc_delta_kw'])
                        if seconds else float(row['previous_fc_kw']))
            add(probe_id=f'v6_run01_failure/{sample_id}/{label}',
                group='v6_run01_validation_failure', sample_id=sample_id, time_s=instant,
                soc=soc, previous_fc_kw=previous,
                original_delta_kw=int(float(row['fc_delta_kw'])) if seconds else None,
                failure_time_s=failed_time, offset_s=-seconds)
    return probes


def write_probe_snapshot(agent, probes, round_index, output):
    """Write every Q, both rankings and changes on exactly the supplied fixed states."""
    folder = Path(output) / 'q_probes'
    folder.mkdir(parents=True, exist_ok=True)
    previous_paths = sorted(path for path in folder.glob('round_*.json')
                            if int(path.stem.split('_')[1]) < round_index)
    previous = ({row['probe_id']: row for row in json.loads(previous_paths[-1].read_text(encoding='utf-8'))}
                if previous_paths else {})
    with torch.inference_mode():
        batch = torch.tensor([item['state'] for item in probes], dtype=torch.float32, device=agent.device)
        all_q = agent.online(batch).cpu().tolist()
    rows = []
    for probe, q in zip(probes, all_q):
        legal = probe['legal_indices']
        raw_order = sorted(range(13), key=lambda index: (-q[index], index))
        legal_order = [index for index in raw_order if index in legal]
        raw_rank = [1 + sum(value > q[index] for value in q) for index in range(13)]
        legal_rank = [1 + sum(q[other] > q[index] for other in legal) if index in legal else None
                      for index in range(13)]
        best = legal_order[0] if legal_order else None
        hold_legal = 6 in legal
        up = next((index for index in legal_order if index > 6), None)
        down = next((index for index in legal_order if index < 6), None)
        raw_up = next(index for index in raw_order if index > 6)
        raw_down = next(index for index in raw_order if index < 6)
        prior = previous.get(probe['probe_id'])
        original = probe['original_chosen_delta_kw']
        row = dict(
            **probe, round=round_index, q_values=q,
            raw_order_indices=raw_order, raw_rank_by_action=raw_rank,
            legal_order_indices=legal_order, legal_rank_by_action=legal_rank,
            hold_legal=hold_legal, hold_rank=legal_rank[6], hold_raw_rank=raw_rank[6], q_hold=q[6],
            greedy_delta_kw=FC_INCREMENT_13_KW[best] if best is not None else None,
            q_greedy=q[best] if best is not None else None,
            greedy_minus_hold=q[best] - q[6] if best is not None and hold_legal else None,
            raw_greedy_delta_kw=FC_INCREMENT_13_KW[raw_order[0]],
            original_action_minus_hold=(q[FC_INCREMENT_13_KW.index(original)] - q[6]
                                       if original is not None else None),
            best_up_delta_kw=FC_INCREMENT_13_KW[up] if up is not None else None,
            best_down_delta_kw=FC_INCREMENT_13_KW[down] if down is not None else None,
            q_best_up=q[up] if up is not None else None,
            q_best_down=q[down] if down is not None else None,
            best_up_minus_hold=q[up] - q[6] if up is not None and hold_legal else None,
            best_down_minus_hold=q[down] - q[6] if down is not None and hold_legal else None,
            best_up_minus_best_down=q[up] - q[down] if up is not None and down is not None else None,
            raw_best_up_delta_kw=FC_INCREMENT_13_KW[raw_up],
            raw_best_down_delta_kw=FC_INCREMENT_13_KW[raw_down],
            raw_best_up_minus_hold=q[raw_up] - q[6], raw_best_down_minus_hold=q[raw_down] - q[6],
            raw_best_up_minus_best_down=q[raw_up] - q[raw_down],
            top1_minus_top2_legal=q[legal_order[0]] - q[legal_order[1]] if len(legal_order) > 1 else None,
            top1_minus_top2_raw=q[raw_order[0]] - q[raw_order[1]],
            previous_round=prior['round'] if prior else None,
            raw_rank_change_from_previous=([new - old for new, old in zip(raw_rank, prior['raw_rank_by_action'])]
                                           if prior else None),
            legal_rank_change_from_previous=([new - old if new is not None else None
                                             for new, old in zip(legal_rank, prior['legal_rank_by_action'])]
                                             if prior else None),
            q_change_from_previous=([new - old for new, old in zip(q, prior['q_values'])] if prior else None),
            greedy_changed_from_previous=(
                (FC_INCREMENT_13_KW[best] if best is not None else None) != prior['greedy_delta_kw']
                if prior else None),
        )
        for index, delta in enumerate(FC_INCREMENT_13_KW):
            row[f'q_delta_{delta}'] = q[index]
            row[f'legal_delta_{delta}'] = index in legal
            row[f'raw_rank_delta_{delta}'] = raw_rank[index]
            row[f'legal_rank_delta_{delta}'] = legal_rank[index]
        rows.append(row)
    (folder / 'definitions.json').write_text(
        json.dumps(probes, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (folder / f'round_{round_index:02d}.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (folder / f'round_{round_index:02d}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    return rows
