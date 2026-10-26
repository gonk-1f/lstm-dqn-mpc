"""Read-only V6 TD statistics, isolated from policy and replay sampling RNGs."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch


CATEGORIES = ('normal_nonterminal', 'normal_terminal', 'failure_terminal', 'failure_prefix')
RESERVOIR_SIZE = 8192


def replay_class_codes(replay, slots):
    done = replay.done[slots, 0] != 0
    return np.where(done, np.where(replay.reasons[slots] >= 2, 2, 1),
                    np.where(replay.outcomes[slots] == 1, 3, 0)).astype(np.uint8)


def replay_ids_for_slots(replay, slots):
    first_live = max(0, replay.insertions - replay.maxlen)
    return first_live + (slots - first_live) % replay.maxlen


class _TDGroup:
    def __init__(self):
        self.count = 0
        self.sums = np.zeros(3, dtype=np.float64)
        self.minimum = np.full(3, np.inf)
        self.maximum = np.full(3, -np.inf)
        self.absolute_sum = 0.
        self.reservoir = np.empty(RESERVOIR_SIZE, dtype=np.float32)

    def observe(self, values, rng):
        n = len(values)
        if not n:
            return
        self.sums += values.sum(axis=0, dtype=np.float64)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        absolute = np.abs(values[:, 2])
        self.absolute_sum += absolute.sum(dtype=np.float64)
        fill = min(n, max(0, RESERVOIR_SIZE - self.count))
        self.reservoir[self.count:self.count+fill] = absolute[:fill]
        if fill < n:
            positions = rng.integers(self.count+fill+np.arange(1, n-fill+1))
            for source in np.flatnonzero(positions < RESERVOIR_SIZE):
                self.reservoir[positions[source]] = absolute[fill+source]
        self.count += n

    def snapshot(self):
        n = self.count
        values = self.reservoir[:min(n, RESERVOIR_SIZE)]
        result = dict(sample_count=n, mae=float(self.absolute_sum/n) if n else None,
            absolute_residual_p95=float(np.percentile(values, 95)) if n else None,
            p95_sample_count=len(values),
            p95_method='exact_all_draws' if n <= RESERVOIR_SIZE else 'uniform_reservoir_estimate')
        for index, name in enumerate(('q', 'target', 'residual_target_minus_q')):
            result[name] = dict(mean=float(self.sums[index]/n) if n else None,
                min=float(self.minimum[index]) if n else None,
                max=float(self.maximum[index]) if n else None)
        return result

    def state_dict(self):
        return dict(count=self.count, sums=self.sums, minimum=self.minimum, maximum=self.maximum,
                    absolute_sum=self.absolute_sum, reservoir=self.reservoir[:min(self.count, RESERVOIR_SIZE)].copy())

    def load_state_dict(self, state):
        self.count = state['count']
        self.sums, self.minimum, self.maximum = state['sums'], state['minimum'], state['maximum']
        self.absolute_sum = state['absolute_sum']
        self.reservoir[:len(state['reservoir'])] = state['reservoir']


class OnlineTDDiagnostics:
    """All actual pre-Adam draws contribute exact moments; only P95 is sampled."""
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed + 40621)
        self.groups = [_TDGroup() for _ in CATEGORIES]
        self.failure_terminal_draws = {}
        self.failure_terminal_draws_window = {}

    def observe(self, replay, q, target, residual):
        slots = replay.last_sample_indices
        codes = replay_class_codes(replay, slots)
        values = torch.cat((q.detach(), target.detach(), residual.detach()), dim=1).cpu().numpy()
        for code, group in enumerate(self.groups):
            group.observe(values[codes == code], self.rng)
        ids = replay_ids_for_slots(replay, slots[codes == 2])
        for replay_id in ids.tolist():
            self.failure_terminal_draws[replay_id] = self.failure_terminal_draws.get(replay_id, 0) + 1
            self.failure_terminal_draws_window[replay_id] = self.failure_terminal_draws_window.get(replay_id, 0) + 1

    def failure_draws_for_id(self, replay_id):
        return self.failure_terminal_draws.get(int(replay_id), 0)

    def snapshot(self, reset=False):
        result = dict(scope='actual optimizer minibatch draws before Adam; count includes repeat draws',
            residual_definition='target minus Q(s, executed_action)',
            prefix_label_note='A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.',
            reservoir_capacity_per_class=RESERVOIR_SIZE,
            groups={name: group.snapshot() for name, group in zip(CATEGORIES, self.groups)},
            failure_terminal_draws_total=dict(self.failure_terminal_draws),
            failure_terminal_draws_window=dict(self.failure_terminal_draws_window))
        if reset:
            self.groups = [_TDGroup() for _ in CATEGORIES]
            self.failure_terminal_draws_window.clear()
        return result

    def state_dict(self):
        return dict(rng_state=self.rng.bit_generator.state,
            groups=[group.state_dict() for group in self.groups],
            failure_terminal_draws=dict(self.failure_terminal_draws),
            failure_terminal_draws_window=dict(self.failure_terminal_draws_window))

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng_state']
        for group, values in zip(self.groups, state['groups']):
            group.load_state_dict(values)
        self.failure_terminal_draws = dict(state['failure_terminal_draws'])
        self.failure_terminal_draws_window = dict(state['failure_terminal_draws_window'])


def checkpoint_replay_fit(agent, round_index, output):
    """Stratified reporting on the current Train replay, never an Adam batch."""
    from .dqn import masked_dqn_targets

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f'round_{round_index:02d}_replay_td_fit.csv.gz'
    replay = agent.replay
    slots = np.arange(len(replay))
    codes = replay_class_codes(replay, slots)
    rng = np.random.default_rng(92731 + round_index)
    result = dict(round=round_index, replay_count=len(replay), measured_count=0,
        scope='read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws',
        target_formula='reward + gamma*(1-done)*max_legal Q_target(next_state, action)',
        residual_definition='target minus Q_online(state, executed_action)',
        sampling='up to 8192 distinct entries per class; entire class when at most 8192',
        p95_note='P95 is exact for the saved diagnostic sample; it is not asserted to be the full-replay P95.',
        raw_samples=str(raw_path), groups={})
    fields = ['round', 'category', 'replay_id', 'action_index', 'action_delta_kw', 'reward', 'done',
        'q', 'target', 'residual_target_minus_q', 'absolute_residual', 'failure_terminal_actual_draws_total',
        *[f'state_{i}' for i in range(5)], *[f'next_state_{i}' for i in range(5)],
        *[f'next_legal_{i}' for i in range(13)], *[f'q_{i}' for i in range(13)],
        *[f'target_next_q_{i}' for i in range(13)]]
    with gzip.open(raw_path, 'wt', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for code, name in enumerate(CATEGORIES):
            population = slots[codes == code]
            selected = (population if len(population) <= RESERVOIR_SIZE else
                        rng.choice(population, RESERVOIR_SIZE, replace=False))
            group = _TDGroup()
            for start in range(0, len(selected), 1024):
                batch = selected[start:start+1024]
                states, actions, rewards, successors, done, masks = (
                    torch.as_tensor(values[batch], device=agent.device) for values in (
                        replay.states, replay.actions, replay.rewards, replay.next_states, replay.done, replay.masks))
                with torch.inference_mode():
                    q_values = agent.online(states)
                    q = q_values.gather(1, actions)
                    next_q = agent.target(successors)
                    target = masked_dqn_targets(rewards, next_q, done, next_action_masks=masks, gamma=agent.gamma)
                    residual = target-q
                if not (torch.isfinite(q_values).all() and torch.isfinite(next_q).all() and torch.isfinite(residual).all()):
                    raise FloatingPointError('non-finite Q or TD in read-only replay diagnostic')
                values = torch.cat((q, target, residual), dim=1).cpu().numpy()
                group.observe(values, rng)
                q_array, next_array = q_values.cpu().numpy(), next_q.cpu().numpy()
                ids = replay_ids_for_slots(replay, batch)
                for i, slot in enumerate(batch):
                    replay_id = int(ids[i])
                    action = int(replay.actions[slot, 0])
                    row = dict(round=round_index, category=name, replay_id=replay_id, action_index=action,
                        action_delta_kw=agent.action_values_kw[action], reward=float(replay.rewards[slot, 0]),
                        done=bool(replay.done[slot, 0]), q=float(values[i, 0]), target=float(values[i, 1]),
                        residual_target_minus_q=float(values[i, 2]), absolute_residual=float(abs(values[i, 2])),
                        failure_terminal_actual_draws_total=agent.td_diagnostics.failure_draws_for_id(replay_id)
                            if code == 2 and agent.td_diagnostics is not None else None)
                    row.update({f'state_{j}': float(value) for j, value in enumerate(replay.states[slot])})
                    row.update({f'next_state_{j}': float(value) for j, value in enumerate(replay.next_states[slot])})
                    row.update({f'next_legal_{j}': bool(value) for j, value in enumerate(replay.masks[slot])})
                    row.update({f'q_{j}': float(value) for j, value in enumerate(q_array[i])})
                    row.update({f'target_next_q_{j}': float(value) for j, value in enumerate(next_array[i])})
                    writer.writerow(row)
            summary = group.snapshot()
            summary.update(replay_count=len(population), measured_count=len(selected),
                           p95_method='exact_on_saved_diagnostic_sample')
            result['groups'][name] = summary
            result['measured_count'] += len(selected)
    (output / f'round_{round_index:02d}_replay_td_fit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result
