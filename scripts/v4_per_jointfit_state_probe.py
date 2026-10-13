"""Bounded independent joint-fit/state diagnostic, never formal training.

Fit frozen B100 Greedy Train reference labels with candidate replay samplers.
The clock arm only tests an additional causal observation in an independent
model. Neither labels nor probe weights are replacements for formal replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F

from dqn.networks.mlp_qnet import MLPQNetwork
from v2.data.formal_training_dataset import FormalTrainingDataset
from v3.control import EconomicMPC
from v4.control import ACTION_KW, ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN


ARMS = ('uniform_7d', 'per_7d', 'per_clock_8d')
SEED = 42
BATCH = 64
KEYS = (('zero_boundary_002', 160), ('zero_boundary_002', 161),
        ('zero_boundary_002', 397), ('zero_boundary_022', 155),
        ('zero_boundary_015', 97), ('zero_boundary_015', 120),
        ('zero_boundary_015', 133), ('zero_boundary_006', 170),
        ('zero_boundary_006', 350))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_model(cp, clock):
    model = MLPQNetwork(8 if clock else 7, 61, (128, 128))
    state = {k: v.clone() for k, v in cp['online_state'].items()}
    if clock:
        weight = state['layers.0.weight']
        state['layers.0.weight'] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
        state['input_norm_mean'] = torch.cat((state['input_norm_mean'], torch.zeros(1)))
        state['input_norm_std'] = torch.cat((state['input_norm_std'], torch.ones(1)))
    model.load_state_dict(state)
    return model


def group_split(data):
    samples = data['source_sample']
    groups = {str(s): ('completed' if np.any((samples == s) & (data['terminal_reason'] == 'completed'))
                      else 'physical_failure' if np.any((samples == s) & (data['category'] == 'failure_terminal'))
                      else 'data_truncation') for s in np.unique(samples)}
    rng = random.Random(SEED)
    held = set()
    by_kind = {}
    for kind in ('completed', 'physical_failure', 'data_truncation'):
        ordered = sorted(s for s, g in groups.items() if g == kind)
        rng.shuffle(ordered)
        count = max(1, math.ceil(len(ordered) * .2)) if len(ordered) > 1 else 0
        held.update(ordered[:count])
        by_kind[kind] = ordered[:count]
    hold_mask = np.isin(samples, sorted(held))
    assert not set(samples[hold_mask]).intersection(samples[~hold_mask])
    return ~hold_mask, hold_mask, by_kind


def inputs(data, clock, mean, std):
    x = data['state'].astype(np.float32)
    if clock:
        extra = ((data['elapsed_seconds'] - mean) / std).astype(np.float32)
        x = np.column_stack((x, extra))
    return torch.tensor(x)


def metrics(model, x, data, train, held, step):
    with torch.no_grad():
        q = model(x)
        chosen = q.gather(1, torch.tensor(data['action'], dtype=torch.long).reshape(-1, 1)).flatten().numpy()
    error = np.abs(chosen - data['fixed_target'])
    groups = {}
    for name, subset in (('train', train), ('heldout', held)):
        groups[name] = {}
        for kind in ('all', *data['category_names'].tolist()):
            mask = subset if kind == 'all' else subset & (data['category'] == kind)
            groups[name][kind] = dict(count=int(mask.sum()), mae=float(error[mask].mean()) if mask.any() else None,
                                     above_one_fraction=float((error[mask] > 1).mean()) if mask.any() else None)
    keys = []
    for sample, index in KEYS:
        hits = np.flatnonzero((data['source_sample'] == sample) & (data['source_index'] == index))
        if not len(hits):
            continue
        i = int(hits[0]); legal = np.flatnonzero(data['current_mask'][i])
        row = q[i].tolist()
        ordered = sorted(legal.tolist(), key=lambda a: (-row[a], a))
        original = int(data['action'][i]); previous = round(float(data['state'][i, 1]) * 600) // 10
        keys.append(dict(sample_id=sample, archive_index=index, fit_split='heldout' if held[i] else 'train',
            original_action_kw=original * 10, current_best_kw=ordered[0] * 10,
            original_action_legal_rank=ordered.index(original) + 1,
            q_original=row[original], fixed_target=float(data['fixed_target'][i]), abs_error=float(error[i]),
            q_maximum_physical_action=row[legal[-1]], q_previous_fc=row[previous],
            best_second_gap=row[ordered[0]] - row[ordered[1]] if len(ordered) > 1 else None,
            q_values=row, physical_actions_kw=(legal * 10).tolist()))
    return dict(step=step, groups=groups, key_states=keys)


def fit_arm(arm, cp, data, train, held, updates, mean, std):
    clock = arm == 'per_clock_8d'
    agent = DirectPowerDDQN(seed=SEED, learning_rate=.001, replay_capacity=300000,
        gamma=1, n_step=8, reward_scale=.001, failure_terminal_quota=2, target_tau=.001,
        replay_diagnostics=False, replay_sampling='uniform' if arm == 'uniform_7d' else 'prioritized',
        per_alpha=.6, per_beta=1, per_epsilon=1e-6)
    model = initial_model(cp, clock)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = inputs(data, clock, mean, std)
    with torch.no_grad():
        baseline = MLPQNetwork(7, 61, (128, 128)); baseline.load_state_dict(cp['online_state'])
        initial_difference = float((model(x) - baseline(torch.tensor(data['state'], dtype=torch.float32))).abs().max())
    assert initial_difference < 1e-5
    mapping = {}
    for i in np.flatnonzero(train).tolist():
        terminal = bool(data['done'][i])
        actions = tuple(int(a) * 10 for a in np.flatnonzero(data['next_mask'][i]))
        agent.remember(data['state'][i], int(data['action_kw'][i]), float(data['raw_reward'][i]),
            data['next_state'][i], done=terminal, next_feasible_actions=actions,
            bootstrap_steps=int(data['m'][i]), experience_outcome=str(data['experience_outcome'][i]),
            terminal_reason=str(data['terminal_reason'][i]) or None,
            failure_penalty_equivalent_cny=float(data['failure_penalty_raw'][i]),
            source_sample_id=str(data['source_sample'][i]), source_round=100,
            source_row_index=int(data['source_row_index'][i]))
        item = agent.replay[-1]
        assert abs(item.reward_cny - float(data['scaled_reward'][i])) < 1e-10
        mapping[item.replay_id] = i
    curves = [metrics(model, x, data, train, held, 0)]
    draws = Counter(); category_draws = Counter(); category_weights = Counter()
    normalizers, weight_means, grad_norms, losses, unweighted_losses = [], [], [], [], []
    clipped = 0
    for step in range(1, updates + 1):
        batch, weights, probabilities = agent._sample_replay_batch_with_weights(BATCH)
        ids = [mapping[item.replay_id] for item in batch]
        yi = torch.tensor(data['fixed_target'][ids], dtype=torch.float32).reshape(-1, 1)
        ai = torch.tensor(data['action'][ids], dtype=torch.long).reshape(-1, 1)
        pred = model(x[ids]).gather(1, ai)
        sample_loss = F.smooth_l1_loss(pred, yi, reduction='none')
        weight = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
        loss = sample_loss.mean() if arm == 'uniform_7d' else (sample_loss * weight).mean()
        assert torch.isfinite(loss)
        errors = (pred - yi).detach().abs().flatten().tolist()
        optimizer.zero_grad(); loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10))
        optimizer.step()
        priority_updates = {}
        for item, i, e, w in zip(batch, ids, errors, weights):
            draws[i] += 1; category_draws[str(data['category'][i])] += 1
            category_weights[str(data['category'][i])] += w
            if agent._priority_index is not None and agent._priority_index.contains(item.replay_id):
                priority_updates[item.replay_id] = max(priority_updates.get(item.replay_id, 0), e)
        for key, error in priority_updates.items():
            agent._priority_index.update(key, error)
        normalizers.append(1 if agent._priority_index is None else agent._priority_index.importance_normalizer(1))
        weight_means.append(float(np.mean(weights))); grad_norms.append(norm); clipped += int(norm > 10)
        losses.append(float(loss.detach())); unweighted_losses.append(float(sample_loss.detach().mean()))
        if step % 100 == 0 or step == updates:
            curves.append(metrics(model, x, data, train, held, step))
    key_draws = []
    for sample, index in KEYS:
        hits = np.flatnonzero((data['source_sample'] == sample) & (data['source_index'] == index))
        if len(hits):
            i = int(hits[0]); key_draws.append(dict(sample_id=sample, archive_index=index,
                fit_split='heldout' if held[i] else 'train', draws=int(draws[i])))
    return model.eval(), dict(arm=arm, input_dim=8 if clock else 7,
        initial_output_difference=initial_difference, updates=updates, sample_positions=updates * BATCH,
        unique_draws=len(draws), category_draws=dict(category_draws), weighted_category_mass=dict(category_weights),
        key_draws=key_draws, is_weight_mean=float(np.mean(weight_means)),
        priority_normalizer_after_update_mean=float(np.mean(normalizers)),
        priority_normalizer_after_update_max=float(max(normalizers)),
        priority_normalizer_timing='post-update snapshot, not the denominator used by the sampled batch',
        gradient_norm_mean=float(np.mean(grad_norms)), gradient_norm_max=float(max(grad_norms)),
        clipped_updates=clipped, mean_weighted_loss=float(np.mean(losses)),
        mean_unweighted_loss=float(np.mean(unweighted_losses)), target_updates=0, curves=curves)


def closed_loop(model, clock, episode, mean, std):
    rows = [i for i, mode in enumerate(episode.operating_mode) if mode == 'onboard']
    cursor = 0
    def policy(state, feasible):
        nonlocal cursor
        # Only the current observed timestamp is exposed, never length/deadline.
        elapsed = float(episode.time_s[rows[cursor]])
        cursor += 1
        value = (*state, (elapsed - mean) / std) if clock else state
        with torch.no_grad():
            q = model(torch.tensor([value], dtype=torch.float32))[0]
        return max(feasible, key=lambda a: (float(q[a // 10]), -a))
    try:
        result = replay_episode(episode, policy, accountant=EconomicMPC(nominal_cost_cny=1),
                                beta_soc=200, redistribute_battery_energy=False)
        transitions = result.transitions
        outcome = 'completed'; failure = None
        original = {k: float(getattr(result.total_ledger, k)) for k in
                    ('h2_cost_cny', 'fuel_cell_degradation_cost_cny', 'battery_degradation_cost_cny', 'shore_cost_cny')}
        terminal = {k: 0 if result.modeled_terminal_settlement is None else
                    float(getattr(result.modeled_terminal_settlement.ledger, k)) for k in original}
        total = result.comparable_cost_cny
    except ReplayExecutionError as error:
        if error.failure_kind == 'execution_error':
            raise
        transitions = error.executed_transitions
        outcome = 'data_truncation' if error.failure_kind == 'data_truncation' else 'physical_failure'
        failure = dict(row=error.row_index, kind=error.failure_kind, cause=error.failure_cause)
        original = {k: math.fsum(getattr(t.original_economic_ledger, k) for t in transitions) for k in
                    ('h2_cost_cny', 'fuel_cell_degradation_cost_cny', 'battery_degradation_cost_cny', 'shore_cost_cny')}
        terminal = None; total = None
    change = [t.action_kw - t.state[1] * 600 for t in transitions]
    soc = [t.actual_soc for t in transitions]
    return dict(sample_id=episode.sample_id, outcome=outcome, failure=failure, steps=len(transitions),
        economic_components=original, modeled_terminal_components=terminal,
        full_comparable_cost_cny=total,
        executed_prefix_cost_cny=sum(original.values()) if total is None else None,
        soc_min=min(soc) if soc else None, soc_max=max(soc) if soc else None,
        soc_terminal=soc[-1] if soc else None,
        high_soc_fraction=sum(s >= .79 for s in soc) / len(soc) if soc else None,
        low_soc_fraction=sum(s < .4 for s in soc) / len(soc) if soc else None,
        mean_abs_fc_delta=float(np.mean(np.abs(change))) if change else None,
        maximum_abs_fc_delta=max(map(abs, change), default=None),
        jumps_gt300=sum(abs(c) > 300 + 1e-8 for c in change),
        startup_count=sum(t.state[1] == 0 and t.action_kw > 0 for t in transitions),
        auxiliary_soc_total=math.fsum(t.soc_soft_penalty_cny for t in transitions))


def main(args):
    if not 1 <= args.updates <= 600:
        raise ValueError('bounded diagnostic permits at most 600 updates per arm')
    if args.output.exists():
        raise FileExistsError('refusing to overwrite a diagnostic result')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cp_hash = sha(args.checkpoint); data_hash = sha(args.dataset)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    data = dict(np.load(args.dataset, allow_pickle=False))
    assert cp['purpose'] == 'diagnostic_only' and cp['round'] == 100 and cp['test_payloads_opened'] == 0
    assert str(data['checkpoint_sha256']) == cp_hash and not bool(data['is_actual_replay'])
    assert np.all(data['data_split'] == 'train')
    train, held, by_kind = group_split(data)
    mean = float(data['elapsed_seconds'][train].mean())
    std = max(1., float(data['elapsed_seconds'][train].std()))
    summary = dict(purpose='independent frozen-reference joint-fit and small Train-only closed-loop diagnostic',
        seed=SEED, updates_per_arm=args.updates, batch_size=64, learning_rate=.001,
        reference_source='Greedy Train n8 references, not actual Replay tuples or true optimal values',
        checkpoint_sha256=cp_hash, dataset_sha256=data_hash,
        source_commit=cp['source_commit'], source_tree_sha256=cp['source_tree_sha256'],
        heldout_groups=by_kind, train_records=int(train.sum()), heldout_records=int(held.sum()),
        heldout_scope='Entire Train sample groups held out from this extra fit; original B network already trained on Train',
        clock_mean_seconds=mean, clock_std_seconds=std, clock_normalization='fit on probe training fold only, then frozen',
        clock_meaning='observed controller uptime since sample initialization, not remaining mission time',
        per=dict(alpha=.6, beta_is=1, epsilon=1e-6, failure_quota=2),
        formal_training_updates=0, validation_payload_reads=0, test_payload_reads=0,
        aborted_probe_attempt=dict(uniform_updates=600, per_updates=1,
            reason='probe priority refresh incorrectly included fixed-quota failure IDs; fixed to match candidate learn contains filter',
            retained_formal_state=False), results={})
    models = {'baseline': initial_model(cp, False).eval()}
    for arm in ARMS:
        print('diagnostic fit start', arm, flush=True)
        models[arm], summary['results'][arm] = fit_arm(arm, cp, data, train, held, args.updates, mean, std)
        print('diagnostic fit end', arm, summary['results'][arm]['curves'][-1]['groups'], flush=True)
    meta = json.loads(args.run_metadata.read_text(encoding='utf-8'))
    assert cp['source_tree_sha256'] == meta['source_tree_sha256'] and cp['manifest_sha256'] == meta['manifest_sha256']
    dataset = FormalTrainingDataset.open(*(Path(p) for p in meta['dataset_roots']))
    episodes = {e.sample_id: e for e in dataset.load_train()}
    selected = ['zero_boundary_002', 'zero_boundary_006', 'zero_boundary_015', 'zero_boundary_022', by_kind['completed'][0]]
    selected = list(dict.fromkeys(selected))
    assert len(selected) <= 5 and all(episodes[s].split == 'train' for s in selected)
    summary['closed_loop_samples'] = selected
    summary['closed_loop'] = {}
    for arm, model in models.items():
        rows = [closed_loop(model, arm == 'per_clock_8d', episodes[s], mean, std) for s in selected]
        summary['closed_loop'][arm] = rows
        print('diagnostic closed loop', arm, [(r['sample_id'], r['outcome'], r['steps']) for r in rows], flush=True)
    assert dataset.opened_test_payloads == 0
    assert sha(args.checkpoint) == cp_hash and sha(args.dataset) == data_hash
    summary['diagnostic_optimizer_updates'] = len(ARMS) * args.updates
    summary['diagnostic_physical_intervals'] = sum(r['steps'] for rows in summary['closed_loop'].values() for r in rows)
    summary['limits'] = [
        'Frozen labels retain the old 7d target network estimate, not corrected values or optimal returns.',
        'All samples are Train; this extra-fit heldout is not unseen Validation generalization.',
        'PER changes both stochastic sampling and IS-normalized effective gradients; it is not sampling alone.',
        'Clock can correlate with offline sample trimming; it is not certified physical mission progress.',
        'Five selected closed-loop tasks are a diagnostic, not formal Train/Validation qualification.',
        'No probe model is best_agent or suitable for resuming formal training.']
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    model_path = args.output.with_name('independent_probe_models.pt')
    assert not model_path.exists()
    torch.save(dict(purpose='diagnostic_only_not_formal_checkpoint', source_checkpoint_sha256=cp_hash,
                    clock_mean=mean, clock_std=std, models={k: m.state_dict() for k, m in models.items()}), model_path)
    print('diagnostic output', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'dataset', 'run-metadata', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600)
    main(parser.parse_args())
