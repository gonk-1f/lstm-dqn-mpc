"""Bounded Train-only frozen-label diagnostics. Never writes formal state."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from collections import Counter

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
RUN = AUDIT.parent
ROOT = RUN.parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(AUDIT))
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial import cKDTree
from audit_support import sha256
from dqn.networks.mlp_qnet import MLPQNetwork
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.independent_dataset import IndependentTrainingDataset
from v2.data.supervisory_rules import normalize_onboard_load_kw

SEED = 42
UPDATES = 200
CATS = ('ordinary_nonterminal', 'normal_terminal', 'failure_prefix', 'failure_terminal')


def dump(name, obj):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)


def table(name, rows):
    with (HERE / name).open('x', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def model_from(state):
    with torch.random.fork_rng(devices=[]):
        model = MLPQNetwork(3, 7, (128, 128))
    model.load_state_dict(state)
    return model.eval()


def optimizer_from(model, state):
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    opt.load_state_dict(copy.deepcopy(state))
    assert opt.param_groups[0]['lr'] == .001
    return opt


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()


def forward(model, x):
    with torch.no_grad():
        return torch.cat([model(x[i:i+2048]) for i in range(0, len(x), 2048)])


def stats(v):
    v = np.asarray(v, dtype=float)
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                median=float(np.median(v)))


def main():
    start = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    dataset = IndependentTrainingDataset.open(ROOT / 'data/processed/operating_dataset_independent_v2')
    # Only Train payload paths are admitted. This process never loads Validation/Test.
    train_manifest = dataset._manifest[dataset._manifest['split'].eq('train')]
    source_parent = dict(zip(train_manifest.candidate_id, train_manifest.parent_sample_id))
    protected = [RUN / 'training_state_latest.pt', RUN / 'run_metadata.json', RUN / 'report.json',
                 RUN / 'round_metrics.csv', *list((RUN / 'diagnostic_checkpoints').glob('*.pt'))]
    protected += list((ROOT / 'src/v4').glob('*.py')) + list((ROOT / 'src/v2/models').glob('*.py'))
    protected += [ROOT / 'src/v3/control.py', ROOT / 'src/dqn/networks/mlp_qnet.py']
    protected += [dataset.root / p for p in dataset.identity_hashes()]
    protected += [dataset.root / p for p in train_manifest.relative_path]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    dump('protected_hashes_before.json', before)
    print('Loading one CPU checkpoint; freezing labels from its Online/Target pair', flush=True)
    cp = torch.load(RUN / 'training_state_latest.pt', map_location='cpu', weights_only=False)
    diag = torch.load(RUN / 'diagnostic_checkpoints/round_100.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
    assert cp['completed_round'] == 100 and cp['metadata'] == metadata
    a = cp['agent']
    for k in ('online', 'target'):
        assert all(torch.equal(v, diag[k+'_state'][n]) for n, v in a[k].items())
    assert a['state_mode'] == 'minimal3' and a['action_mode'] == 'increment7'
    replay = a['replay']
    assert len(replay) == 300000
    assert all(t.source_sample_id in source_parent and t.bootstrap_steps == 1 for t in replay)
    x = torch.tensor([t.state for t in replay], dtype=torch.float32)
    acts = torch.tensor([t.action_index for t in replay], dtype=torch.long)
    cats = np.array([DirectPowerDDQN._category(t) for t in replay])
    sources = np.array([t.source_sample_id for t in replay])
    parents = np.array([source_parent[s] for s in sources])
    online, target = model_from(a['online']), model_from(a['target'])
    labels = []
    with torch.no_grad():
        for b in range(0, len(replay), 2048):
            values = replay[b:b+2048]
            nx = torch.tensor([t.next_state for t in values], dtype=torch.float32)
            mask = torch.zeros(len(values), 7, dtype=torch.bool)
            for i, t in enumerate(values):
                mask[i, list(t.next_feasible_indices)] = True
            labels.append(masked_double_dqn_targets(
                torch.tensor([[t.reward_cny] for t in values], dtype=torch.float32),
                online(nx), target(nx), torch.tensor([[float(t.done)] for t in values]),
                next_action_masks=mask, gamma=1.).flatten())
    y = torch.cat(labels)
    # Parent-disjoint split avoids leakage between two segments of the same parent.
    failed_parents = set(parents[cats == 'failure_terminal'])
    rng = random.Random(SEED)
    held_parents = set()
    for pool in (sorted(failed_parents), sorted(set(parents)-failed_parents)):
        rng.shuffle(pool)
        held_parents.update(pool[:max(1, math.ceil(.2*len(pool)))])
    held = np.isin(parents, sorted(held_parents))
    train = ~held
    assert set(parents[train]).isdisjoint(set(parents[held]))
    idx_rng = np.random.default_rng(SEED)
    eval_ids = {}
    split_counts = []
    for split, sel in (('fit', train), ('heldout', held)):
        for cat in CATS:
            ids = np.flatnonzero(sel & (cats == cat))
            selected = idx_rng.choice(ids, min(2000, len(ids)), replace=False)
            eval_ids[(split, cat)] = selected
            split_counts.append(dict(split=split, category=cat, available=len(ids), evaluated=len(selected)))
    assert all(len(v) for v in eval_ids.values())
    table('split_counts.csv', split_counts)
    dump('design.json', dict(seed=SEED, updates_per_joint_arm=UPDATES, batch_size=64,
        heldout_parents=sorted(held_parents), fit_parents=sorted(set(parents)-held_parents),
        heldout_samples=sorted(set(sources[held])), fit_samples=sorted(set(sources[train])),
        target='Frozen formal masked 1-step DDQN target from the R100 Online/Target snapshot',
        optimizer='Independent copy of actual R100 Adam state, lr=.001 in both arms',
        arms=['uniform_with_failure_quota2', 'original_per_with_failure_quota2'],
        sampling='Formal sampler, filtered to fit parents. PER conditional tree and IS normalizer recomputed after excluding heldout; priorities refreshed against fixed labels only.',
        target_updates=0, formal_training=False, train_payload_only=True,
        checkpoint_sha256=before[str((RUN/'training_state_latest.pt').relative_to(ROOT))],
        frozen_label_sha256=hashlib.sha256(y.numpy().tobytes()).hexdigest()))
    # Train 015 probes are evaluation only; no Validation 020 states are fitted or read.
    probe_rows = list(csv.DictReader((AUDIT / 'q_action_audit_parent015.csv').open(encoding='utf-8-sig')))
    px = torch.tensor([[float(r[k]) for k in ('state_soc','state_load_over600','state_previous_fc_over600')]
                       for r in probe_rows], dtype=torch.float32)
    base_pq = online(px).detach()
    base_rank = []
    for r, q in zip(probe_rows, base_pq):
        legal = json.loads(r['legal_action_indices'])
        base_rank.append(max(legal, key=lambda j: float(q[j])))
    curves, draws, ranks, updates_rows = [], [], [], []
    all_eval = np.concatenate(list(eval_ids.values()))
    old_q_eval = forward(online, x[all_eval])
    eval_acts = acts[all_eval]
    unused_mask = torch.ones_like(old_q_eval, dtype=torch.bool)
    unused_mask[torch.arange(len(all_eval)), eval_acts] = False

    def metrics(model, arm, step):
        for (split, cat), ids in eval_ids.items():
            pred = forward(model, x[ids]).gather(1, acts[ids,None]).flatten()
            e = pred-y[ids]
            curves.append(dict(arm=arm, updates=step, split=split, category=cat, count=len(ids),
                mae=float(e.abs().mean()), signed_q_minus_y=float(e.mean()),
                huber=float(F.smooth_l1_loss(pred, y[ids])), rmse=float(e.square().mean().sqrt())))
        pq = forward(model, px)
        for i, (r, q) in enumerate(zip(probe_rows, pq)):
            legal = json.loads(r['legal_action_indices'])
            best = max(legal, key=lambda j: float(q[j]))
            actual = int(r['actual_action_index'])
            ranks.append(dict(arm=arm, updates=step, step_1based=int(r['step_1based']),
                source='parent_015_seg01', split='heldout' if source_parent['parent_015_seg01'] in held_parents else 'fit_parent_probe',
                original_action=actual, best_action=best, best_changed=best!=base_rank[i],
                actual_minus_hold=float(q[actual]-q[3]),
                **{f'q_{d}':float(q[j]) for j,d in enumerate(a['action_values_kw'])}))

    train_ids = np.flatnonzero(train).tolist()
    for arm, sampling in (('uniform', 'uniform'), ('per', 'prioritized')):
        agent = DirectPowerDDQN(seed=SEED, state_mode='minimal3', action_mode='increment7',
            replay_capacity=300000, gamma=1, n_step=1, learning_rate=.001,
            reward_scale=.001, failure_terminal_quota=2, target_tau=.001,
            replay_sampling=sampling, per_alpha=.6, per_beta=1., per_epsilon=1e-6,
            replay_diagnostics=False)
        agent.online.load_state_dict(a['online'])
        agent.optimizer.load_state_dict(copy.deepcopy(a['optimizer']))
        by_id = {t.replay_id: t for t in replay}
        agent.replay.extend(replay[i] for i in train_ids)
        fit_id_set = {replay[i].replay_id for i in train_ids}
        # Preserve checkpoint pool order, not merely pool membership.
        agent._ordinary_pool = [(j, by_id[j]) for j in a['ordinary_pool_ids'] if j in fit_id_set]
        agent._failure_terminal_pool = [(j, by_id[j]) for j in a['failure_pool_ids'] if j in fit_id_set]
        agent._ordinary_positions = {j:i for i,(j,_) in enumerate(agent._ordinary_pool)}
        agent._failure_terminal_positions = {j:i for i,(j,_) in enumerate(agent._failure_terminal_pool)}
        if sampling == 'prioritized':
            saved = copy.deepcopy(a['priority_index'])
            saved['entries'] = tuple((j,p) for j,p in saved['entries'] if j in fit_id_set)
            agent._priority_index.load_state_dict(saved)
        lookup = {t.replay_id:i for i,t in enumerate(replay)}
        counts = Counter()
        p0 = flat_params(agent.online)
        metrics(agent.online, arm, 0)
        for step in range(1, UPDATES+1):
            batch, weights, _ = agent._sample_replay_batch_with_weights(64)
            ids = np.array([lookup[t.replay_id] for t in batch])
            assert train[ids].all()
            pred = agent.online(x[ids]).gather(1, acts[ids,None]).flatten()
            raw_loss = F.smooth_l1_loss(pred, y[ids], reduction='none')
            loss = raw_loss.mean() if sampling == 'uniform' else (raw_loss*torch.tensor(weights)).mean()
            agent.optimizer.zero_grad()
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(agent.online.parameters(), 10))
            p_before = flat_params(agent.online)
            agent.optimizer.step()  # Copy-only supervised fit; never calls agent.learn or soft_update.
            displacement = float((flat_params(agent.online)-p_before).norm())
            residual = (pred.detach()-y[ids]).abs().tolist()
            if sampling == 'prioritized':
                priorities = {}
                for t, e in zip(batch, residual):
                    if agent._priority_index.contains(t.replay_id):
                        priorities[t.replay_id] = max(priorities.get(t.replay_id, 0), e)
                for j,e in priorities.items():
                    agent._priority_index.update(j,e)
            counts.update(cats[ids])
            updates_rows.append(dict(arm=arm, update=step, weighted_loss=float(loss.detach()),
                unweighted_huber=float(raw_loss.mean().detach()), mean_is_weight=float(np.mean(weights)),
                gradient_norm_before_clip=grad, actual_adam_parameter_l2=displacement))
            if step in (25,50,100,200):
                metrics(agent.online, arm, step)
        new_q_eval = forward(agent.online, x[all_eval])
        rank_changed = sum(r['best_changed'] for r in ranks if r['arm']==arm and r['updates']==200)
        draws.append(dict(arm=arm, sample_positions=64*UPDATES, categories=dict(counts),
            final_parameter_l2_from_start=float((flat_params(agent.online)-p0).norm()),
            final_relative_parameter_l2=float((flat_params(agent.online)-p0).norm()/p0.norm()),
            unexecuted_output_mean_absolute_drift=float((new_q_eval-old_q_eval)[unused_mask].abs().mean()),
            unexecuted_output_scope='Other six outputs, including potentially physically infeasible actions; not a correctness score',
            train015_masked_best_changed=rank_changed, train015_probe_count=len(probe_rows)))
        print(f'{arm}: {UPDATES} copy-only updates complete', flush=True)
        del agent
    table('jointfit_metrics.csv', curves)
    table('copy_update_statistics.csv', updates_rows)
    table('train015_q_ranking.csv', ranks)
    dump('jointfit_summary.json', draws)

    # One-step cross-category influence: identical start + Adam history for each trial.
    # A zero-current-gradient Adam control separates carried momentum from this batch's effect.
    pools = {c: np.flatnonzero(train & (cats==c)) for c in CATS}
    grad_vectors = {}
    for cat in CATS:
        ids = idx_rng.choice(pools[cat], min(256,len(pools[cat])), replace=False)
        m = model_from(a['online'])
        loss = F.smooth_l1_loss(m(x[ids]).gather(1,acts[ids,None]).flatten(), y[ids])
        gs = torch.autograd.grad(loss, tuple(m.parameters()))
        grad_vectors[cat] = torch.cat([g.flatten() for g in gs]).detach()
    cosine_rows = []
    for c in CATS:
        for d in CATS:
            g,h=grad_vectors[c],grad_vectors[d]
            cosine_rows.append(dict(source_category=c, compared_category=d,
                unweighted_huber_gradient_cosine=float(torch.dot(g,h)/(g.norm()*h.norm()).clamp_min(1e-20))))
    table('category_gradient_cosines.csv', cosine_rows)
    # Full fit-pool snapshot weights are reused as category diagnostics, no class balancing claim.
    pri = dict(a['priority_index']['entries'])
    ordinary_ids = [i for i in train_ids if cats[i]!='failure_terminal']
    mass = np.array([(pri[replay[i].replay_id]+1e-6)**.6 for i in ordinary_ids])
    normalizer = max(1., (len(mass)*mass.min()/mass.sum())**-1)
    snapshot_w = np.full(len(replay),1/normalizer)
    snapshot_w[ordinary_ids] = (len(mass)*mass/mass.sum())**-1/normalizer
    influence = []
    for cat in CATS:
        for trial in range(8):
            ids = idx_rng.choice(pools[cat], 64, replace=len(pools[cat])<64)
            m = model_from(a['online'])
            opt = optimizer_from(m, a['optimizer'])
            neutral = model_from(a['online'])
            no = optimizer_from(neutral, a['optimizer'])
            for p in neutral.parameters():
                p.grad = torch.zeros_like(p)
            no.step()
            loss = (F.smooth_l1_loss(m(x[ids]).gather(1,acts[ids,None]).flatten(),
                                    y[ids], reduction='none')*torch.tensor(snapshot_w[ids])).mean()
            opt.zero_grad(); loss.backward()
            gradnorm=float(torch.nn.utils.clip_grad_norm_(m.parameters(),10)); opt.step()
            for dest in CATS:
                j = eval_ids[('heldout',dest)]
                q0=forward(online,x[j]).gather(1,acts[j,None]).flatten()
                q1=forward(m,x[j]).gather(1,acts[j,None]).flatten()
                qn=forward(neutral,x[j]).gather(1,acts[j,None]).flatten()
                l0=float(F.smooth_l1_loss(q0,y[j])); l1=float(F.smooth_l1_loss(q1,y[j]))
                ln=float(F.smooth_l1_loss(qn,y[j]))
                influence.append(dict(source_category=cat, trial=trial, evaluated_category=dest,
                    heldout_huber_before=l0, heldout_huber_after=l1,
                    delta_huber=l1-l0, delta_huber_vs_momentum_control=l1-ln,
                    delta_mae=float((q1-y[j]).abs().mean()-(q0-y[j]).abs().mean()),
                    source_batch_gradient_norm=gradnorm))
    table('cross_category_one_step.csv', influence)
    print('Cross-category copy-only interventions complete; checking causal load history', flush=True)

    # Unique physical Train intervals, not repeated replay occurrences, for exogenous prediction.
    history_rows, history_metrics, history_episode_metrics = [], [], []
    episodes = dataset.load_train()
    for ep in episodes:
        loads = np.array([normalize_onboard_load_kw(float(p)) for p in ep.load_kw])
        d1=np.r_[0., np.diff(loads)]
        dprev=np.r_[0.,d1[:-1]]
        for horizon in (1,8):
            for t in range(len(loads)-horizon):
                target_value = loads[t+1] if horizon==1 else float(np.sum(
                    loads[t+1:t+1+horizon]*ep.dt_s[t+1:t+1+horizon])/3600)
                history_rows.append(dict(sample=ep.sample_id, parent=source_parent[ep.sample_id],
                    split='heldout' if source_parent[ep.sample_id] in held_parents else 'fit',
                    horizon=horizon, row=t, current=loads[t]/600, d1=d1[t]/600,
                    dprev=dprev[t]/600, target=target_value))
    assert dataset.opened_test_payloads==0
    for horizon in (1,8):
        hr=[r for r in history_rows if r['horizon']==horizon]
        features=np.array([[r[k] for k in ('current','d1','dprev')] for r in hr])
        hy=np.array([r['target'] for r in hr]); isfit=np.array([r['split']=='fit' for r in hr])
        names=np.array([r['sample'] for r in hr]); parent=np.array([r['parent'] for r in hr])
        predictions={}
        for mode,cols in (('current_only',[0]),('current_plus_two_causal_differences',[0,1,2])):
            f=features[:,cols]
            mu=f[isfit].mean(0); sd=np.maximum(f[isfit].std(0),1e-6)
            z=(f-mu)/sd
            xt=np.column_stack((np.ones(isfit.sum()),z[isfit]))
            xh=np.column_stack((np.ones((~isfit).sum()),z[~isfit]))
            reg=np.eye(xt.shape[1]); reg[0,0]=0
            coef=np.linalg.solve(xt.T@xt+reg,xt.T@hy[isfit])
            predictions[('ridge_alpha1',mode)]=xh@coef
            _, neighbors=cKDTree(z[isfit]).query(z[~isfit],k=32,workers=1)
            predictions[('knn_k32',mode)]=hy[isfit][neighbors].mean(1)
        eh=hy[~isfit]; nh=names[~isfit]; ph=parent[~isfit]
        for (method,mode), pred in predictions.items():
            per_parent=[]
            for group in sorted(set(ph)):
                m=ph==group
                per_parent.append(float(abs(pred[m]-eh[m]).mean()))
            history_metrics.append(dict(horizon_steps=horizon, target_unit='kW' if horizon==1 else 'kWh',
                method=method,features=mode, fit_rows=int(isfit.sum()), heldout_rows=int((~isfit).sum()),
                heldout_mae=float(abs(pred-eh).mean()), heldout_rmse=float(np.sqrt(np.mean((pred-eh)**2))),
                heldout_parent_macro_mae=float(np.mean(per_parent))))
            for sample in sorted(set(nh)):
                m=nh==sample
                history_episode_metrics.append(dict(horizon_steps=horizon, method=method, features=mode,
                    sample=sample, count=int(m.sum()),mae=float(abs(pred[m]-eh[m]).mean()),
                    rmse=float(np.sqrt(np.mean((pred[m]-eh[m])**2)))))
    table('history_prediction_metrics.csv',history_metrics)
    table('history_prediction_per_voyage.csv',history_episode_metrics)
    # No trained network copies or optimizer states are serialized.
    after={str(p.relative_to(ROOT)):sha256(p) for p in protected}
    assert before==after
    dump('verification.json',dict(protected_file_count=len(before),all_hashes_unchanged=before==after,
        validation_payloads_read=0,test_payloads_read=0, formal_updates=0, target_updates=0,
        copy_only_joint_gradient_steps=400, copy_only_category_gradient_steps=32,
        zero_current_gradient_momentum_control_steps=32, gradient_probe_batches=4,
        elapsed_seconds=time.time()-start, torch_threads=2, source_commit=metadata['source_commit'],
        heldout_never_sampled=True, no_model_weights_written=True))
    print('DONE; protected source/data/checkpoints unchanged',flush=True)


if __name__=='__main__':
    main()
