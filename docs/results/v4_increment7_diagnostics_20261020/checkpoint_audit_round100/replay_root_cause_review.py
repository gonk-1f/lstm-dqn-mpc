"""Read-only Train replay analysis; no sampling, optimizer, or checkpoint writes."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from audit_support import OUT, RUN, ROOT, sha256, load_policy
from dqn.networks.mlp_qnet import MLPQNetwork
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


def dump(name, value):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(name, rows):
    path = OUT / name
    if path.exists():
        raise FileExistsError(path)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {'n': 0}
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()),
                p10=float(np.quantile(v, .1)), median=float(np.median(v)),
                p90=float(np.quantile(v, .9)), max=float(v.max()))


def main():
    protected = [RUN / n for n in ('training_state_latest.pt', 'run_metadata.json', 'report.json', 'round_metrics.csv')]
    protected += list((RUN / 'diagnostic_checkpoints').glob('round_*.pt'))
    protected += [ROOT / 'src/v4' / n for n in ('dqn.py', 'control.py', 'prioritized_replay.py', 'monitored_training.py', 'actions.py')]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    dump('root_review_readonly_baseline.json', before)
    print('Loading 77.6 MB round-boundary replay snapshot read-only', flush=True)
    state = torch.load(RUN / 'training_state_latest.pt', map_location='cpu', weights_only=False)
    policy, diag, metadata = load_policy()
    assert state['completed_round'] == 100
    assert state['metadata'] == metadata
    a = state['agent']
    for name, key in (('online', 'online_state'), ('target', 'target_state')):
        assert all(torch.equal(v, diag[key][k]) for k, v in a[name].items())
    replay = a['replay']
    n = len(replay)
    manifest = ROOT / 'data/processed/operating_dataset_independent_v2/splits_v2_exclude046_seg01'
    # Only metadata/Train provenance is checked; no split payload is loaded.
    assert all(t.source_sample_id.startswith('parent_') for t in replay)
    assert a['state_mode'] == 'minimal3' and a['action_mode'] == 'increment7'
    assert all(t.bootstrap_steps == 1 for t in replay)
    model = policy.model
    with torch.random.fork_rng(devices=[]):
        target = MLPQNetwork(3, 7, (128, 128))
    target.load_state_dict(a['target'], strict=True)
    target.eval().requires_grad_(False)
    x = np.array([t.state for t in replay], dtype=np.float32)
    act = np.array([t.action_index for t in replay])
    cats = np.array([DirectPowerDDQN._category(t) for t in replay])
    ids = np.array([t.replay_id for t in replay])
    rounds = np.array([t.source_round for t in replay])
    sources = np.array([t.source_sample_id for t in replay])
    raw = dict(a['priority_index']['entries'])
    failure = cats == 'failure_terminal'
    assert set(raw) == set(a['ordinary_pool_ids'])
    assert len(raw) + failure.sum() == n
    priority = np.array([raw.get(int(k), np.nan) for k in ids])
    alpha, beta, eps = (a['sampling_config'][key] for key in ('alpha', 'beta', 'epsilon'))
    mass = np.zeros(n)
    mass[~failure] = (priority[~failure] + eps) ** alpha
    conditional_prob = mass / mass.sum()
    minimum_prob = conditional_prob[~failure].min()
    normalizer = max(1., (len(raw) * minimum_prob) ** -beta)
    weights = np.full(n, 1 / normalizer)
    weights[~failure] = (len(raw) * conditional_prob[~failure]) ** -beta / normalizer
    # Exact expected batch-slot frequencies at this snapshot, not historical draws.
    slot_prob = conditional_prob * 62 / 64
    slot_prob[failure] = 2 / 64 / failure.sum()
    assert abs(slot_prob.sum() - 1.) < 1e-10
    q = np.empty(n)
    y = np.empty(n)
    with torch.inference_mode():
        for start in range(0, n, 2048):
            stop = min(n, start + 2048)
            batch = replay[start:stop]
            ns = torch.tensor([t.next_state for t in batch], dtype=torch.float32)
            mask = torch.zeros((len(batch), 7), dtype=torch.bool)
            for i, t in enumerate(batch):
                mask[i, list(t.next_feasible_indices)] = True
            q[start:stop] = model(torch.from_numpy(x[start:stop])).gather(
                1, torch.from_numpy(act[start:stop, None])).flatten().numpy()
            y[start:stop] = masked_double_dqn_targets(
                torch.tensor([[t.reward_cny] for t in batch], dtype=torch.float32),
                model(ns), target(ns), torch.tensor([[float(t.done)] for t in batch]),
                next_action_masks=mask, gamma=1.).flatten().numpy()
    residual = y - q
    huber_slope = np.minimum(np.abs(residual), 1.)
    effective = slot_prob * weights * huber_slope
    print('Replay forward pass complete', flush=True)
    coverage = []
    for category in ['all', *sorted(set(cats))]:
        cm = np.ones(n, dtype=bool) if category == 'all' else cats == category
        for index, delta in enumerate(a['action_values_kw']):
            m = cm & (act == index)
            if not m.any():
                continue
            pri = priority[m & ~failure]
            r = dict(category=category, action_index=index, delta_kw=delta, count=int(m.sum()),
                     fraction_within_category=float(m.sum() / cm.sum()),
                     priority_mean=float(pri.mean()) if len(pri) else '',
                     priority_median=float(np.median(pri)) if len(pri) else '',
                     priority_p90=float(np.quantile(pri, .9)) if len(pri) else '',
                     frozen_td_mean=float(residual[m].mean()), frozen_td_mae=float(abs(residual[m]).mean()),
                     mean_is_weight=float(weights[m].mean()), snapshot_draw_fraction=float(slot_prob[m].sum()),
                     expected_output_huber_abs_gradient=float(effective[m].sum()))
            coverage.append(r)
    write_csv('replay_action_coverage.csv', coverage)
    near_rows = []
    probes = list(csv.DictReader((OUT / 'q_action_audit_parent020.csv').open(encoding='utf-8-sig')))
    # Distances use current three normalized inputs, with no threshold-based sufficiency verdict.
    for probe in probes:
        p = np.array([float(probe[k]) for k in ('state_soc', 'state_load_over600', 'state_previous_fc_over600')])
        diff = x.astype(float) - p
        dist = np.sqrt((diff * diff).sum(axis=1))
        for index, delta in enumerate(a['action_values_kw']):
            eligible = np.flatnonzero(act == index)
            chosen = eligible[np.argsort(dist[eligible])[:128]]
            for k in (1, 8, 32, 128):
                ix = chosen[:k]
                near_rows.append(dict(probe_step=int(probe['step_1based']), delta_kw=delta, k=len(ix),
                    distance_min=float(dist[ix].min()), distance_max=float(dist[ix].max()),
                    distance_median=float(np.median(dist[ix])),
                    abs_soc_difference_mean=float(abs(diff[ix, 0]).mean()),
                    abs_load_kw_difference_mean=float(abs(diff[ix, 1]).mean() * 600),
                    abs_previous_fc_kw_difference_mean=float(abs(diff[ix, 2]).mean() * 600),
                    td_mean=float(residual[ix].mean()), td_mae=float(abs(residual[ix]).mean()),
                    priority_mean=float(np.nanmean(priority[ix])), mean_is_weight=float(weights[ix].mean()),
                    snapshot_expected_draws_per_update=float(slot_prob[ix].sum() * 64),
                    category_counts=json.dumps(dict(Counter(cats[ix])), sort_keys=True),
                    nearest_replay_id=int(ids[ix[0]]), nearest_source=str(sources[ix[0]]),
                    nearest_round=int(rounds[ix[0]])))
    write_csv('replay_local_action_neighbors.csv', near_rows)
    # Nearest same-action ordinary neighbor to every real successful terminal.
    pairs = []
    for i in np.flatnonzero(cats == 'normal_terminal'):
        js = np.flatnonzero((act == act[i]) & (cats == 'ordinary_nonterminal'))
        d = x[js].astype(float) - x[i]
        sq = (d * d).sum(axis=1)
        j = js[int(sq.argmin())]
        pairs.append(dict(terminal_replay_id=int(ids[i]), ordinary_replay_id=int(ids[j]),
            terminal_source=str(sources[i]), ordinary_source=str(sources[j]),
            delta_kw=a['action_values_kw'][int(act[i])], normalized_distance=float(np.sqrt(sq.min())),
            soc_difference=float(x[j, 0] - x[i, 0]), load_difference_kw=float((x[j, 1] - x[i, 1])*600),
            previous_fc_difference_kw=float((x[j, 2] - x[i, 2])*600),
            terminal_target=float(y[i]), ordinary_target=float(y[j]), target_gap=float(y[j]-y[i]),
            terminal_td=float(residual[i]), ordinary_td=float(residual[j])))
    write_csv('terminal_nearest_ordinary_pairs.csv', pairs)
    # Exact float32 input/action duplicates: label variance can be stochastic, not proof of an error.
    signature = np.column_stack((x, act.astype(np.float32)))
    _, inv, sizes = np.unique(signature, axis=0, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=y)
    squares = np.bincount(inv, weights=y*y)
    variance = np.maximum(0., squares/sizes - (sums/sizes)**2)
    duplicated = sizes > 1
    category_summary = {}
    for category in sorted(set(cats)):
        m = cats == category
        category_summary[category] = dict(count=int(m.sum()), td=stats(residual[m]),
            absolute_td=stats(abs(residual[m])), q=stats(q[m]), target=stats(y[m]),
            positive_td_fraction=float((residual[m] > 0).mean()),
            snapshot_draw_fraction=float(slot_prob[m].sum()),
            output_gradient_share=float(effective[m].sum()/effective.sum()),
            expected_mean_is_weight=float((slot_prob[m]*weights[m]).sum()/slot_prob[m].sum()))
    near_agg = {}
    for delta in a['action_values_kw']:
        z = [r for r in near_rows if r['delta_kw'] == delta and r['k'] == 32]
        near_agg[str(delta)] = {key: stats([r[key] for r in z]) for key in (
            'distance_median', 'abs_soc_difference_mean', 'abs_load_kw_difference_mean',
            'abs_previous_fc_kw_difference_mean', 'td_mean', 'td_mae', 'snapshot_expected_draws_per_update')}
    summary = dict(replay_count=n, completed_round=100, snapshot_identity_matches=True,
        source_rounds=stats(rounds), source_counts=dict(Counter(sources)), counts=a['counts'],
        category_summary=category_summary, sampling_config=a['sampling_config'],
        priority=stats(priority[~failure]), minimum_probability=float(minimum_prob),
        global_is_normalizer=float(normalizer), weights_live_pool=stats(weights),
        expected_batch_mean_is_weight=float((slot_prob*weights).sum()),
        stale_priority_current_absolute_td_correlation=float(np.corrcoef(priority[~failure], abs(residual[~failure]))[0,1]),
        near020_k32_summary=near_agg,
        terminal_nearest_pairs={key:stats([r[key] for r in pairs]) for key in (
            'normalized_distance', 'soc_difference', 'load_difference_kw', 'previous_fc_difference_kw', 'target_gap')},
        duplicated_state_action=dict(groups=int(duplicated.sum()), entries=int(sizes[duplicated].sum()),
            groups_with_target_range_variance_gt_1e_minus12=int((duplicated & (variance > 1e-12)).sum()),
            target_std=stats(np.sqrt(variance[duplicated]))),
        limitations=['Snapshot expected draw rates are not historical per-item counts.',
            'Huber output slopes are not full shared-parameter gradients or Adam steps.',
            'Near-state distance is normalized Euclidean; k sensitivity is reported, no sufficiency threshold.',
            'Float32 duplicate-label variation is not automatically non-Markov evidence.',
            'No train/fit/optimizer was called.'])
    dump('replay_root_cause_review.json', summary)
    after = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    assert before == after
    dump('root_review_readonly_verification.json', dict(all_hashes_unchanged=True, protected_files=len(before)))
    print(json.dumps({k:summary[k] for k in ('replay_count','source_rounds','category_summary','expected_batch_mean_is_weight','duplicated_state_action')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
