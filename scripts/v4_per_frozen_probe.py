"""Bounded frozen-target probe on previously exported Train diagnostic CSVs.

No dataset, formal replay, training checkpoint or live output is loaded. These
selected neighborhoods are NOT representative of the full replay distribution.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from v4.dqn import DirectPowerDDQN


def read_records(root):
    records = {}
    sources = []
    path = root / 'nearest_replay_examples.csv'
    sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # derived_* are Train replay provenance; query_* may be held-out queries.
            records[int(row['replay_id'])] = dict(
                state=json.loads(row['state_json']), action=int(row['action_kw']),
                target=float(row['current_stored_tuple_td_target']),
                sample=row['derived_sample'], source_round=int(row['derived_round']),
                kind=int(row['terminal_kind']),
                failed=row['experience_failed'] == 'True')
    path = root / 'normal_terminal_nearest_nonterminal.csv'
    sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            records[int(row['terminal_replay_id'])] = dict(
                state=json.loads(row['terminal_state_json']), action=int(row['action_kw']),
                target=float(row['terminal_target']), sample=row['terminal_sample'],
                source_round=int(row['terminal_round']), kind=3, failed=False)
    # The second export does not include its neighbor's action, so do not invent it.
    return records, sources


def run(records, sources, updates):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    groups = sorted({(r['sample'], r['source_round']) for r in records.values()})
    random.Random(42).shuffle(groups)
    held_groups = set(groups[:len(groups)//5])
    held_groups.add(('zero_boundary_015', 88))
    for group in (('zero_boundary_015', 76), ('zero_boundary_015', 79)):
        held_groups.discard(group)
    for row in records.values():
        if row['kind'] != 0 and row['failed']:
            held_groups.discard((row['sample'], row['source_round']))
    train = {i:r for i,r in records.items() if (r['sample'], r['source_round']) not in held_groups}
    held = {i:r for i,r in records.items() if i not in train}
    assert not {(r['sample'],r['source_round']) for r in train.values()}.intersection(
        (r['sample'],r['source_round']) for r in held.values())
    x = torch.tensor([r['state'] for r in records.values()], dtype=torch.float32)
    actions = torch.tensor([[r['action']//10] for r in records.values()])
    y = torch.tensor([[r['target']] for r in records.values()], dtype=torch.float32)
    all_keys = list(records)
    risk_keys = {493205, 516139, 584126}.intersection(records)
    results = {}
    for mode in ('uniform', 'prioritized'):
        agent = DirectPowerDDQN(seed=42, replay_capacity=len(train), replay_sampling=mode,
                                failure_terminal_quota=2, per_alpha=.6, per_beta=1.)
        original_ids = {}
        for key, row in train.items():
            # This is a supervised diagnostic label, not a replacement formal TD target.
            terminal = row['kind'] != 0
            failed = terminal and row['failed']
            agent.remember(row['state'], row['action'], row['target'], row['state'],
                done=terminal, next_feasible_actions=() if terminal else (0,),
                experience_outcome='failure' if row['failed'] else 'success',
                terminal_reason='failure_soc_limited' if failed else 'completed' if terminal else None)
            original_ids[agent.replay[-1].replay_id] = key
        curves, draws = [], Counter()
        def measure(step):
            with torch.no_grad():
                q = agent.online(x)
                errors = (q.gather(1, actions) - y).abs().flatten().tolist()
            values = {key:error for key,error in zip(all_keys,errors)}
            groups_mae = {}
            for name, subset in [('train',train),('heldout',held)]:
                for category in ('all','normal_terminal','failure_prefix','ordinary'):
                    selected = [key for key,row in subset.items() if category=='all' or
                        category=='normal_terminal' and row['kind']!=0 and not row['failed'] or
                        category=='failure_prefix' and row['failed'] and row['kind']==0 or
                        category=='ordinary' and not row['failed'] and row['kind']==0]
                    groups_mae[f'{name}_{category}_mae'] = sum(values[key] for key in selected)/len(selected) if selected else None
            risk = {str(key): {'abs_error':values[key], 'q_selected':float(q[all_keys.index(key),records[key]['action']//10]),
                'selected_action_rank_all61':1+int((q[all_keys.index(key)]>q[all_keys.index(key),records[key]['action']//10]).sum()),
                'top_three_actions_all61':[int(i)*10 for i in q[all_keys.index(key)].topk(3).indices],
                'split':'train' if key in train else 'heldout'} for key in risk_keys}
            curves.append(dict(step=step, **groups_mae, risk=risk))
        measure(0)
        for step in range(1,updates+1):
            batch, weights, _ = agent._sample_replay_batch_with_weights(64)
            states = torch.tensor([t.state for t in batch],dtype=torch.float32)
            selected = torch.tensor([[t.action_index] for t in batch])
            targets = torch.tensor([[t.reward_cny] for t in batch])
            q = agent.online(states).gather(1,selected)
            error = (targets-q).detach().abs().flatten().tolist()
            if mode=='uniform':
                loss=F.smooth_l1_loss(q,targets)
            else:
                loss=(F.smooth_l1_loss(q,targets,reduction='none')*torch.tensor(weights).reshape(-1,1)).mean()
            agent.optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online.parameters(),10.)
            agent.optimizer.step()
            updates_by_id={}
            for t,td in zip(batch,error):
                draws[original_ids[t.replay_id]]+=1
                if agent._priority_index is not None and agent._priority_index.contains(t.replay_id):
                    updates_by_id[t.replay_id]=max(updates_by_id.get(t.replay_id,0.),td)
            for key,td in updates_by_id.items():
                agent._priority_index.update(key,td)
            if step%50==0 or step==updates:
                measure(step)
        results[mode]={'curves':curves,'risk_draws':{str(i):draws[i] for i in risk_keys},
            'sample_count':sum(draws.values()),'unique_samples':len(draws),
            'normal_terminal_draws':sum(n for key,n in draws.items() if train[key]['kind']!=0 and not train[key]['failed']),
            'failure_prefix_draws':sum(n for key,n in draws.items() if train[key]['failed'] and train[key]['kind']==0),
            'failure_terminal_draws':sum(n for key,n in draws.items() if train[key]['kind']!=0 and train[key]['failed'])}
    return {'seed':42,'updates_per_group':updates,'batch':64,'network':'7-128-128-61',
        'learning_rate':.001,'alpha':.6,'beta_is':1.,'epsilon':1e-6,'failure_quota':2,
        'source_exports':sources,'total_records':len(records),'train_records':len(train),
        'heldout_records':len(held),'group_split':'(Train sample ID, source round), disjoint groups',
        'training_scope':'fixed exported final-A targets; fresh identical network initialization; no bootstrap recomputation',
        'target_updates':0,'environment_steps':0,'formal_replay_insertions':0,
        'limitations':'selected neighborhood exports, not full Replay; no original network checkpoint; all61 ranks are probe-only outputs, not historical decisions',
        'results':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=300)
    args=parser.parse_args()
    if not 1<=args.updates<=300:
        parser.error('this bounded probe permits at most 300 updates per group')
    if args.output.exists():
        parser.error('refusing to overwrite an existing diagnostic result')
    rows,sources=read_records(args.exports)
    result=run(rows,sources,args.updates)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({mode:{'first':r['curves'][0],'last':r['curves'][-1],
        'risk_draws':r['risk_draws']} for mode,r in result['results'].items()},ensure_ascii=False))
