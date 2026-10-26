"""Read-only value reports from actual formal Train failure trajectories."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .dqn import masked_dqn_targets


class TrainFailureValues:
    """Retain failure tails and freeze the first upper/lower Train examples."""

    def __init__(self, output):
        self.output = Path(output)
        self.fixed_events = []
        self.round_prefixes = []

    def capture(self, episode, values, status, round_index, replay_start):
        if status == 'completed' or not values:
            return
        terminal_soc = values[-1].actual_soc
        side = 'upper' if terminal_soc >= .79 else 'lower' if terminal_soc <= .21 else 'interior'
        rewards = np.asarray([t.reward_cny for t in values], dtype=np.float64)
        returns = np.cumsum(rewards[::-1])[::-1]
        count = len(values)
        selected = sorted(set(range(max(0, count - 300), count)) |
                          {max(0, count - k) for k in (1800, 600)} | {0})
        rows = []
        for index in selected:
            t = values[index]
            legal = [(p - round(t.state[2] * 600 / 10) * 10 + 60) // 10
                     for p in t.policy_candidate_actions]
            rows.append(dict(source_split='train', origin_round=round_index,
                sample_id=episode.sample_id, boundary_side=side, failure_reason=status,
                replay_id=replay_start + index, time_s=t.relative_time_s,
                seconds_to_failure=count - index, state=list(t.state),
                next_state=list(t.next_state), action_index=t.action_index,
                fc_kw=t.action_kw, delta_kw=t.action_delta_kw, load_kw=t.raw_load_kw,
                soc_before=t.state[0], soc_after=t.actual_soc,
                legal_indices=legal,
                next_legal_indices=[(p - t.action_kw + 60) // 10 for p in t.next_feasible_actions],
                immediate_reward=t.reward_cny,
                observed_behavior_return_to_failure=float(returns[index]), done=t.done,
                fc_h2=t.executed_ledger.h2_cost_cny,
                battery_h2eq=t.soc_equivalent_hydrogen_cost_cny,
                soc_penalty=t.soc_soft_penalty_cny,
                fc_ramp_penalty=t.fc_ramp_penalty_cny,
                failure_penalty=t.failure_penalty_equivalent_cny))
        self.round_prefixes.extend(rows)
        if side in ('upper', 'lower') and not any(e['boundary_side'] == side for e in self.fixed_events):
            event = dict(origin_round=round_index, sample_id=episode.sample_id,
                boundary_side=side, failure_reason=status,
                terminal_replay_id=replay_start + count - 1,
                failed_state=list(values[-1].next_state), rows=rows)
            self.fixed_events.append(event)
            (self.output / 'train_failure_fixed_states.json').write_text(
                json.dumps(self.fixed_events, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    @staticmethod
    def _write_csv(path, rows):
        if not rows:
            return
        with gzip.open(path, 'wt', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                              for k, v in row.items()} for row in rows)

    def snapshot(self, agent, round_index, terminal_draw_counts):
        prefix_file = f'round_{round_index:02d}_train_failure_prefixes.csv.gz'
        self._write_csv(self.output / prefix_file, self.round_prefixes)
        prefix_count = len(self.round_prefixes)
        self.round_prefixes.clear()
        rows = [r for event in self.fixed_events for r in event['rows']]
        evaluated = []
        events = []
        if rows:
            device = agent.device
            states = torch.tensor([r['state'] for r in rows], dtype=torch.float32, device=device)
            successors = torch.tensor([r['next_state'] for r in rows], dtype=torch.float32, device=device)
            masks = torch.zeros((len(rows), 13), dtype=torch.bool, device=device)
            for index, row in enumerate(rows):
                masks[index, row['next_legal_indices']] = True
            with torch.inference_mode():
                all_q = agent.online(states).cpu().numpy()
                targets = masked_dqn_targets(
                    torch.tensor([[r['immediate_reward']] for r in rows], dtype=torch.float32, device=device),
                    agent.target(successors),
                    torch.tensor([[float(r['done'])] for r in rows], dtype=torch.float32, device=device),
                    next_action_masks=masks, gamma=agent.gamma).cpu().numpy().ravel()
            for row, q, target in zip(rows, all_q, targets):
                ranked = sorted(row['legal_indices'], key=lambda i: float(q[i]), reverse=True)
                taken = float(q[row['action_index']])
                best_up = max((float(q[i]) for i in ranked if i > 6), default=None)
                best_down = max((float(q[i]) for i in ranked if i < 6), default=None)
                evaluated.append(dict(checkpoint_round=round_index, **row,
                    q_taken=taken, current_one_step_td_target=float(target),
                    signed_td_residual=float(target) - taken,
                    observed_behavior_return_minus_q=row['observed_behavior_return_to_failure'] - taken,
                    q_hold=float(q[6]), hold_legal=6 in ranked,
                    best_up_minus_hold=None if best_up is None else best_up - float(q[6]),
                    best_down_minus_hold=None if best_down is None else best_down - float(q[6]),
                    legal_action_ranking_delta_kw=[agent.action_values_kw[i] for i in ranked],
                    best_minus_second_q=float(q[ranked[0]] - q[ranked[1]]) if len(ranked) > 1 else None,
                    **{f'q_delta_{d}': float(q[i]) for i, d in enumerate(agent.action_values_kw)}))
            for event in self.fixed_events:
                group = [r for r in evaluated if r['origin_round'] == event['origin_round']
                         and r['sample_id'] == event['sample_id']]
                terminal = group[-1]
                replay_id = event['terminal_replay_id']
                events.append(dict(boundary_side=event['boundary_side'], sample_id=event['sample_id'],
                    origin_round=event['origin_round'], terminal_replay_id=replay_id,
                    actual_training_draws=terminal_draw_counts.get(str(replay_id), terminal_draw_counts.get(replay_id, 0)),
                    terminal_q=terminal['q_taken'], terminal_target=terminal['current_one_step_td_target'],
                    terminal_signed_residual=terminal['signed_td_residual'],
                    terminal_mae=abs(terminal['signed_td_residual']),
                    prefix_td_mae=(float(np.mean([abs(r['signed_td_residual']) for r in group[:-1]]))
                                   if len(group) > 1 else None),
                    full_q_rows=len(group)))
        values_file = f'round_{round_index:02d}_train_failure_values.csv.gz'
        self._write_csv(self.output / values_file, evaluated)
        result = dict(file=values_file if evaluated else None,
            prefix_file=prefix_file if prefix_count else None, recorded_prefix_rows=prefix_count,
            probes_count=len(evaluated), events=events,
            interpretation='Only actual Train transitions; read-only checkpoint inference. '
            'Observed return is the realized epsilon-greedy trajectory return, not Q* or a counterfactual benefit. '
            'TD targets use current target-network masked max; they are moving learning targets. '
            'The first upper/lower failures are fixed for checkpoint comparisons; no report rows enter Replay.')
        (self.output / f'round_{round_index:02d}_train_failure_values.json').write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return result
