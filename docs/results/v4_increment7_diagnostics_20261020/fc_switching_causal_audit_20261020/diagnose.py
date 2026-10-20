"""Output-only frozen-policy causal audit. No optimizer, Replay, Test, or training."""
from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import json
import math
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
OLD = OUT.parent
ROOT = OLD.parent.parent
NEW = ROOT / 'outputs/v4_independent35_m3_delta7_zhou_td1_per_interleaved_seed42_100r_20261020'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(OLD / 'checkpoint_audit_round100'))
import numpy as np
import pandas as pd
import torch
from audit_support import FrozenPolicy
from dqn.networks.mlp_qnet import MLPQNetwork
from v3.control import AccountState, EconomicMPC
from v3.episode_replay import _ledger_total
from v2.data.supervisory_rules import normalize_onboard_load_kw, OperatingMode
from v4.actions import FC_INCREMENT_KW, candidate_fc_actions
from v4.control import (build_state, feasible_fc_actions, ledger_training_reward,
    soc_deviation_squared, FC_SQUARED_RAMP_REWARD_PER_KW2, replay_episode,
    DirectTransition, ReplayExecutionError, NoFeasibleFCActionError, infeasibility_cause)
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue
from v4.independent_dataset import IndependentTrainingDataset
from v4.independent_economics import (ZhouEvaluationConfig, evaluate_zhou_economics,
    soc_equivalent_hydrogen_cost_cny)
from v4.trajectory_io import iter_trajectory_records, resolve_trajectory_path

torch.set_num_threads(1)  # Diagnostic process only; leave running training unchanged.
ACCOUNTANT = EconomicMPC(nominal_cost_cny=1.0)
ENERGY = BatteryEnergyValue.from_accountant(ACCOUNTANT)
PENALTY = FailurePenalty.from_reference()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def models(cp):
    online, target = [MLPQNetwork(3, 7, (128, 128)) for _ in range(2)]
    online.load_state_dict(cp['online_state'], strict=True)
    target.load_state_dict(cp['target_state'], strict=True)
    for m in (online, target):
        m.eval().requires_grad_(False)
    return FrozenPolicy(online), FrozenPolicy(target)


def snapshot():
    # Deserialize one atomically published latest checkpoint; keep ONLY weights/identity.
    # Training may publish a newer file later; this copy remains tied to its saved round.
    p = NEW / 'training_state_latest.pt'
    before = p.stat()
    raw = torch.load(p, map_location='cpu', weights_only=False)
    r = int(raw['completed_round'])
    cp = {**raw['metadata'], 'round': r, 'purpose': 'read_only_diagnostic_snapshot',
          'online_state': raw['agent']['online'], 'target_state': raw['agent']['target']}
    save = OUT / f'interleaved_round_{r:03d}_weights_readonly.pt'
    if save.exists():
        raise FileExistsError(save)
    torch.save(cp, save)
    info = {'loaded_completed_round': r, 'original_path': str(p),
            'original_stat_before': [before.st_size, before.st_mtime_ns],
            'snapshot_path': str(save), 'snapshot_sha256': sha(save),
            'note': 'Only network weights and immutable checkpoint metadata copied; no optimizer/Replay restore.'}
    del raw
    return cp, info


def physical_mask(p, load, duration):
    full = feasible_fc_actions(p, load, ACCOUNTANT, duration_s=duration)
    return full, candidate_fc_actions(full, p.previous_fc_kw, action_mode='increment7')


def decision_state(p, load):
    return build_state(p, (p.previous_fc_kw,) * 3, (load,) * 3, state_mode='minimal3')


def one_step(ep, j, p, fc, config, online, target):
    """Exact official interval, reward helpers, mask, failure finalizer, masked DDQN."""
    load = normalize_onboard_load_kw(float(ep.load_kw[j]))
    dt = float(ep.dt_s[j])
    full, legal = physical_mask(p, load, dt)
    assert fc in legal
    after, ledger, battery = ACCOUNTANT.interval(p, fc, load, duration_s=dt)
    state = decision_state(p, load)
    soc_cost = soc_equivalent_hydrogen_cost_cny(p.soc, after.soc, config)
    soc_penalty = 200.0 * soc_deviation_squared(after.soc)
    ramp = FC_SQUARED_RAMP_REWARD_PER_KW2 * (fc - p.previous_fc_kw) ** 2
    raw_reward = ledger_training_reward(ledger, 'zhou_joint') - soc_cost - soc_penalty - ramp
    normal_done = j + 1 == len(ep.dt_s)
    next_load = normalize_onboard_load_kw(float(ep.load_kw[j + 1])) if not normal_done else load
    ns = decision_state(after, next_load)
    nf, nl = ((), ()) if normal_done else physical_mask(after, next_load, float(ep.dt_s[j + 1]))
    delta = int(fc - p.previous_fc_kw)
    t = DirectTransition(state, int(fc), raw_reward, ns, ledger, battery, after.soc,
        normal_done, nl, soc_soft_penalty_cny=soc_penalty, original_reward=raw_reward,
        terminal_reason='completed' if normal_done else None,
        physical_feasible_actions=full, policy_candidate_actions=legal,
        fc_ramp_penalty_cny=ramp, duration_s=dt, raw_load_kw=float(ep.load_kw[j]),
        relative_time_s=float(ep.time_s[j]), action_mode='increment7',
        action_index=FC_INCREMENT_KW.index(delta), action_delta_kw=delta,
        soc_equivalent_hydrogen_cost_cny=soc_cost)
    if not normal_done and not nl:
        cause = 'increment_action_limited' if nf else infeasibility_cause(next_load, ACCOUNTANT)
        err = ReplayExecutionError(j + 1, OperatingMode.ONBOARD,
            NoFeasibleFCActionError('diagnostic exact next-observation failure'),
            executed_transitions=(t,), failure_kind='no_feasible_action', failure_cause=cause,
            failed_state=ns)
        t = prepare_failure_replay(err, penalty=PENALTY, energy_value=ENERGY,
                                  redistribute_battery_energy=False).transitions[-1]
    if t.done:
        boot, next_best = 0.0, None
    else:
        qn, tn = online.q(ns), target.q(ns)
        idx = [i for i, d in enumerate(FC_INCREMENT_KW) if fc + d in nl]
        next_best = max(idx, key=lambda i: qn[i])
        boot = tn[next_best]
    return t, after, {'r_normal': raw_reward * .001, 'r': t.reward_cny * .001,
        'bootstrap': boot, 'y': t.reward_cny * .001 + boot,
        'done': t.done, 'terminal_reason': t.terminal_reason,
        'next_best_index': next_best, 'soc_after': after.soc,
        'fc_loss_after': after.fc_loss_uv, 'battery_weighted_ah_after': after.battery_weighted_ah,
        'next_legal_kw': json.dumps(nl), 'h2': ledger.h2_cost_cny,
        'soc_equivalent': soc_cost, 'fc_degradation': ledger.fuel_cell_degradation_cost_cny,
        'battery_degradation': ledger.battery_degradation_cost_cny,
        'soc_aux': soc_penalty, 'ramp_aux': ramp,
        'failure_aux': t.failure_penalty_equivalent_cny}


def statistics(values):
    a = np.asarray(values, dtype=float)
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            **{k: float(np.quantile(a, q)) if len(a) else None
               for k, q in [('min', 0), ('p05', .05), ('p50', .5), ('p95', .95), ('max', 1)]}}


def aggregate(df):
    n = len(df)
    return {'n': n, 'immediate_change_worse': int((df.delta_r < -1e-10).sum()),
        'bootstrap_compensates': int(((df.delta_r < -1e-10) & (df.delta_y > 1e-10)).sum()),
        'q_positive_y_negative': int(((df.delta_q > 1e-10) & (df.delta_y < -1e-10)).sum()),
        'y_positive': int((df.delta_y > 1e-10).sum()),
        'online_target_best_disagree': int((df.online_best_index != df.target_best_index).sum()),
        'reversals': int(df.reversal.sum()),
        'gap_lt_1e_3': int((df.best_second_gap < .001).sum()),
        'gap_lt_1e_2': int((df.best_second_gap < .01).sum()),
        **{k: statistics(df[k]) for k in ['delta_r', 'delta_r_normal', 'delta_y', 'delta_q',
            'delta_bootstrap', 'best_second_gap', 'online_target_q_abs_mean']}}


def audit_run(label, run, cp, episodes):
    online, target = models(cp)
    cfg = ZhouEvaluationConfig(cp['hyperparameters']['economic_evaluation']['fc_average_efficiency'])
    path = resolve_trajectory_path(run / f'round_{cp["round"]:03d}_validation_trajectories.json')
    verified = {'archive_path': str(path), 'archive_sha256': sha(path), 'profiles': [],
                'action_mismatch': 0, 'soc_max_error': 0., 'reward_max_error': 0.,
                'mask_mismatch': 0, 'accountant_ledger_max_error': 0., 'clipping_count': 0}
    all_rows, pairs, legal_rows = [], [], []
    profiles = {}
    for section, profile in iter_trajectory_records(path):
        sid = profile['sample_id']
        ep = episodes[sid]
        profiles[sid] = profile
        archive = profile['transitions']
        p = AccountState()
        future = np.cumsum([x['scaled_training_reward'] for x in archive][::-1])[::-1]
        verified['profiles'].append({'sample_id': sid, 'completed': profile['completed'],
                                     'executed_steps': len(archive)})
        previous_delta = 0
        for j, old in enumerate(archive):
            load = normalize_onboard_load_kw(float(ep.load_kw[j]))
            full, legal = physical_mask(p, load, float(ep.dt_s[j]))
            state = decision_state(p, load)
            q, qt = online.q(state), target.q(state)
            idx = [i for i, d in enumerate(FC_INCREMENT_KW) if p.previous_fc_kw + d in legal]
            order = sorted(idx, key=lambda i: (-q[i], i))
            best = order[0]
            tb = max(idx, key=lambda i: qt[i])
            fc = int(old['fc_kw'])
            ai = FC_INCREMENT_KW.index(int(fc - p.previous_fc_kw))
            verified['action_mismatch'] += int(ai != best or ai != old['action_index'])
            verified['mask_mismatch'] += int(tuple(old['policy_candidate_actions_kw']) != legal)
            t, after, actual = one_step(ep, j, p, fc, cfg, online, target)
            verified['soc_max_error'] = max(verified['soc_max_error'], abs(after.soc - old['soc_after']),
                                           abs(p.soc - old['soc_before']))
            verified['reward_max_error'] = max(verified['reward_max_error'],
                abs(actual['r'] - old['scaled_training_reward']))
            for key, value in old['original_economic_ledger'].items():
                verified['accountant_ledger_max_error'] = max(verified['accountant_ledger_max_error'],
                                                            abs(getattr(t.executed_ledger, key) - value))
            delta = FC_INCREMENT_KW[ai]
            clipped = fc != p.previous_fc_kw + delta
            verified['clipping_count'] += clipped
            common = {'run': label, 'round': cp['round'], 'sample_id': sid, 'step_1based': j + 1,
                'time_s': float(ep.time_s[j]), 'soc': p.soc, 'load_kw': load,
                'previous_fc_kw': p.previous_fc_kw, 'fc_loss_before_uv': p.fc_loss_uv,
                'battery_weighted_ah_before': p.battery_weighted_ah,
                'load_change_kw': load - normalize_onboard_load_kw(float(ep.load_kw[j-1])) if j else 0.,
                'fc_kw': fc, 'actual_index': ai, 'actual_delta_kw': delta,
                'hold_legal': p.previous_fc_kw in legal, 'legal_count': len(legal),
                'online_best_index': best, 'target_best_index': tb,
                'q_actual': q[ai], 'q_hold': q[3], 'q_best': q[best],
                'q_second': q[order[1]] if len(order) > 1 else np.nan,
                'best_second_gap': q[best] - q[order[1]] if len(order) > 1 else np.nan,
                'online_target_q_abs_mean': float(np.mean(np.abs(np.array(q)[idx] - np.array(qt)[idx]))),
                'reversal': delta * previous_delta < 0, 'same_direction': delta * previous_delta > 0,
                'clipping': clipped, 'distance_to_observed_end': len(archive)-1-j,
                'completed': bool(profile['completed']), 'actual_return_G': float(future[j]),
                'q_minus_G': q[ai] - float(future[j]),
                **{f'actual_{key}': value for key, value in actual.items()},
                **{f'q_{d:+d}': q[i] for i, d in enumerate(FC_INCREMENT_KW)},
                **{f'target_q_{d:+d}': qt[i] for i, d in enumerate(FC_INCREMENT_KW)}}
            all_rows.append(common)
            if delta and p.previous_fc_kw in legal:
                _, _, hold = one_step(ep, j, p, int(p.previous_fc_kw), cfg, online, target)
                pair = {**common, **{f'hold_{key}': value for key, value in hold.items()},
                    'delta_r': actual['r'] - hold['r'],
                    'delta_r_normal': actual['r_normal'] - hold['r_normal'],
                    'delta_bootstrap': actual['bootstrap'] - hold['bootstrap'],
                    'delta_y': actual['y'] - hold['y'], 'delta_q': q[ai] - q[3]}
                pair['gap_residual'] = pair['delta_q'] - pair['delta_y']
                pairs.append(pair)
                for i in idx:
                    legal_rows.append({'run': label, 'sample_id': sid, 'step_1based': j+1,
                        'action_index': i, 'delta_fc_kw': FC_INCREMENT_KW[i],
                        'executed_fc_kw': p.previous_fc_kw + FC_INCREMENT_KW[i],
                        'online_q': q[i], 'target_q': qt[i], 'greedy_rank': order.index(i)+1,
                        'is_actual': i == ai, 'is_hold': i == 3})
            p = after
            previous_delta = delta
    assert not verified['action_mismatch'], verified
    assert not verified['mask_mismatch'] and not verified['clipping_count'], verified
    assert max(verified[k] for k in ('soc_max_error', 'reward_max_error',
                                    'accountant_ledger_max_error')) < 1e-8, verified
    return all_rows, pairs, legal_rows, verified, profiles, online, target, cfg


def rollout(ep, profile, online, cfg, start, variant):
    """Reconstruct exact full initial-to-branch prefix through FORMAL replay_episode.

    No alternate environment or approximate transfer model is used. Formal replay
    remains unchanged. One action at start differs; all following actions greedy.
    """
    count = 0
    before = AccountState()
    shadows = []
    def policy(state, feasible):
        nonlocal count, before
        load = normalize_onboard_load_kw(float(ep.load_kw[count]))
        assert max(abs(a-b) for a,b in zip(state, decision_state(before, load))) < 1e-10
        assert feasible == physical_mask(before, load, float(ep.dt_s[count]))[1]
        shadows.append(asdict(before))
        if count < start:
            fc = int(profile['transitions'][count]['fc_kw'])
        elif count == start:
            fc = int(before.previous_fc_kw) if variant == 'hold_once' else online(state, feasible)
        else:
            fc = online(state, feasible)
        assert fc in feasible
        before = ACCOUNTANT.interval(before, fc, load, duration_s=float(ep.dt_s[count]))[0]
        count += 1
        return fc
    error = None
    try:
        replay = replay_episode(ep, policy, accountant=ACCOUNTANT, beta_soc=200.,
            state_mode='minimal3', action_mode='increment7', reward_cost_mode='zhou_joint',
            economic_evaluation_config=cfg)
        transitions = replay.transitions
    except ReplayExecutionError as exc:
        error = exc
        transitions = prepare_failure_replay(exc, penalty=PENALTY, energy_value=ENERGY,
                                            redistribute_battery_energy=False).transitions
    assert len(shadows) == len(transitions)
    assert len(transitions) > start
    prefix = [{**shadows[j], 'executed_fc': t.action_kw, 'raw_reward': t.reward_cny}
              for j,t in enumerate(transitions[:start])]
    suffix = transitions[start:]
    evaluation = evaluate_zhou_economics(_ledger_total([t.executed_ledger for t in transitions]),
        soc_initial=.6, soc_terminal=before.soc, config=cfg)
    ds = [t.action_delta_kw for t in suffix]
    # Signed SOC conversion telescopes on this suffix; never call whole-voyage
    # evaluator with an arbitrary branch starting SOC or charge a terminal fee.
    suffix_h2 = sum(t.executed_ledger.h2_cost_cny for t in suffix)
    suffix_eq = soc_equivalent_hydrogen_cost_cny(suffix[0].state[0], suffix[-1].actual_soc, cfg)
    suffix_fc = sum(t.executed_ledger.fuel_cell_degradation_cost_cny for t in suffix)
    suffix_bat = sum(t.executed_ledger.battery_degradation_cost_cny for t in suffix)
    total_return = sum(t.reward_cny*.001 for t in suffix)
    summary = {'variant': variant, 'completed': error is None, 'executed_steps': len(transitions),
        'suffix_steps': len(suffix), 'failure_cause': error.failure_cause if error else None,
        'failed_row_index': error.row_index if error else None,
        'branch_full_state': shadows[start],
        'prefix_sha256': hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest(),
        'suffix_return': total_return, 'soc_min': min([suffix[0].state[0]]+[t.actual_soc for t in suffix]),
        'soc_max': max([suffix[0].state[0]]+[t.actual_soc for t in suffix]),
        'soc_end': before.soc, 'mean_abs_delta_kw': float(np.mean(np.abs(ds))),
        'direction_reversals': sum(a*b<0 for a,b in zip(ds,ds[1:])),
        'suffix_h2_equivalent_cost': suffix_h2+suffix_eq,
        'suffix_fc_degradation_cost': suffix_fc, 'suffix_bat_degradation_cost': suffix_bat,
        'suffix_zhou_three_cost': suffix_h2+suffix_eq+suffix_fc+suffix_bat,
        'full_zhou_evaluation': asdict(evaluation),
        'cost_scope': 'complete_voyage' if error is None else 'executed_prefix_not_complete_cost'}
    trace = [{'step_1based': start+j+1, 'time_s': t.relative_time_s, 'fc_kw': t.action_kw,
        'delta_kw': t.action_delta_kw, 'load_kw': t.raw_load_kw, 'soc_before': t.state[0],
        'soc_after': t.actual_soc, 'scaled_reward': t.reward_cny*.001, 'done': t.done,
        'terminal_reason': t.terminal_reason,
        'failure_penalty': t.failure_penalty_equivalent_cny,
        'h2': t.executed_ledger.h2_cost_cny, 'soc_equivalent': t.soc_equivalent_hydrogen_cost_cny,
        'fc_deg': t.executed_ledger.fuel_cell_degradation_cost_cny,
        'bat_deg': t.executed_ledger.battery_degradation_cost_cny} for j,t in enumerate(suffix)]
    if variant == 'greedy_once':
        old = profile['transitions']
        assert len(old) == len(transitions)
        assert max(abs(t.actual_soc-o['soc_after']) for t,o in zip(transitions, old)) < 1e-10
        assert all(t.action_kw == int(o['fc_kw']) for t,o in zip(transitions,old))
        assert abs(total_return - sum(x['scaled_training_reward'] for x in old[start:])) < 1e-9
    return summary, trace


def main():
    started = time.time()
    source_files = sorted((ROOT/'src/v4').glob('*.py')) + sorted((ROOT/'src/v3').glob('*.py'))
    protected = {str(p.relative_to(ROOT)): sha(p) for p in source_files}
    old_cp_path = OLD / 'diagnostic_checkpoints/round_100.pt'
    old_cp = torch.load(old_cp_path, map_location='cpu', weights_only=False)
    new_cp, snap_info = snapshot()
    identity = {'git_branch': git('branch','--show-current'), 'git_head': git('rev-parse','HEAD'),
        'git_status': git('status','--short'), 'source_sha256': protected,
        'old_checkpoint_sha256': sha(old_cp_path), 'new_snapshot': snap_info,
        'runs': {label: {k:cp.get(k) for k in ('round','source_commit','source_tree_sha256',
            'experiment_id','manifest_sha256','hyperparameters')} for label,cp in
            [('old_episode_end_r100',old_cp),('current_interleaved',new_cp)]},
        'test_payloads_opened': 0, 'training_or_backprop_executed': False}
    assert old_cp['manifest_sha256'] == new_cp['manifest_sha256']
    save_json('identity.json',identity)
    ds = IndependentTrainingDataset.open(ROOT/'data/processed/operating_dataset_independent_v2')
    episodes = {e.sample_id:e for e in ds.load_validation()}
    assert len(episodes) == 8
    all_rows, pairs, legal_rows, verification, contexts = [], [], [], {}, {}
    for label, run, cp in [('old_episode_end_r100',OLD,old_cp),('current_interleaved',NEW,new_cp)]:
        a,p,l,v,profiles,online,target,cfg = audit_run(label,run,cp,episodes)
        all_rows.extend(a); pairs.extend(p); legal_rows.extend(l); verification[label] = v
        contexts[label] = (profiles,online,cfg)
        print(label, 'archived states',len(a),'change-and-hold-legal',len(p),flush=True)
    states, pair = pd.DataFrame(all_rows), pd.DataFrame(pairs)
    states.to_csv(OUT/'all_validation_state_calibration.csv',index=False,encoding='utf-8-sig')
    pair.to_csv(OUT/'per_step_action_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(legal_rows).to_csv(OUT/'all_legal_action_q.csv',index=False,encoding='utf-8-sig')
    save_json('reproduction_verification.json',verification)
    summary = {}
    for label,g in pair.groupby('run'):
        summary[label] = aggregate(g)
        summary[label]['by_voyage'] = {sid:aggregate(s) for sid,s in g.groupby('sample_id')}
        summary[label]['all_visited_states'] = int((states.run==label).sum())
        summary[label]['hold_illegal_changes'] = int(((states.run==label)&~states.hold_legal&
                                                     (states.actual_delta_kw!=0)).sum())
    # Fixed original 64-state population explicitly separate from all Validation.
    zoom = pair[(pair.run=='old_episode_end_r100') & (pair.sample_id=='parent_020_seg01') &
                pair.step_1based.between(54,117)]
    summary['old_020_original_64'] = aggregate(zoom)
    save_json('summary.json',summary)
    pair['soc_bin'] = pd.cut(pair.soc,[.199999,.3,.4,.6,.79,.800001],include_lowest=True).astype(str)
    pair['load_change_bin'] = pd.cut(pair.load_change_kw,[-np.inf,-60,-10,10,60,np.inf]).astype(str)
    grouped=[]
    for grouping in ['sample_id','soc_bin','load_change_bin','actual_delta_kw','reversal']:
        for (label,value),g in pair.groupby(['run',grouping],observed=True):
            ag=aggregate(g)
            grouped.append({'run':label,'group_type':grouping,'group':str(value),
                **{k:v for k,v in ag.items() if not isinstance(v,dict)},
                **{k+'_'+s:ag[k][s] for k in ('delta_q','delta_r','delta_y','delta_bootstrap',
                                             'best_second_gap') for s in ('mean','p50','p95')}})
    pd.DataFrame(grouped).to_csv(OUT/'stratified_comparison.csv',index=False,encoding='utf-8-sig')
    states['distance_bin'] = pd.cut(states.distance_to_observed_end,
        [-1,0,7,31,99,299,np.inf],labels=['0','1-7','8-31','32-99','100-299','300+'])
    cal=[]
    for (label,complete,b),g in states.groupby(['run','completed','distance_bin'],observed=True):
        cal.append({'run':label,'completed':complete,'distance_bin':str(b),'n':len(g),
            'mean_Q':g.q_actual.mean(),'mean_G':g.actual_return_G.mean(),
            **{'Q_minus_G_'+k:v for k,v in statistics(g.q_minus_G).items()}})
    pd.DataFrame(cal).to_csv(OUT/'q_return_calibration_by_distance.csv',index=False,encoding='utf-8-sig')
    # Deterministic purposive cases; no population prevalence inferred from these.
    selections=[]
    oldpair=pair[pair.run=='old_episode_end_r100']
    def select(g,reason,sort,ascending=True):
        used={(x['run'],x['sample_id'],x['step_1based']) for x in selections}
        for _,row in g.sort_values(sort,ascending=ascending).iterrows():
            key=(row.run,row.sample_id,int(row.step_1based))
            if key not in used:
                selections.append({'run':row.run,'sample_id':row.sample_id,
                    'step_1based':int(row.step_1based),'reason':reason})
                return
    select(zoom[zoom.delta_y<0],'original_020_Q_vs_target_conflict_largest','delta_q',False)
    select(zoom[zoom.delta_y>0],'original_020_bootstrap_compensates','delta_y',False)
    select(zoom,'original_020_nearly_tied_best_second','best_second_gap')
    select(oldpair[oldpair.same_direction],'continuous_same_direction','delta_q',False)
    select(oldpair[~oldpair.completed & oldpair.distance_to_observed_end.between(8,100)],
           'failed_voyage_early_low_SOC','soc')
    select(oldpair[(oldpair.actual_delta_kw<0)&(oldpair.soc<.4)],'SOC_decline_negative_increment','soc')
    cur=pair[pair.run=='current_interleaved']
    select(cur,'current_snapshot_largest_Q_target_gap','gap_residual',False)
    select(cur[cur.reversal],'current_snapshot_reversal_small_margin','best_second_gap')
    save_json('branch_selection.json',selections)
    branches=[]
    for k,s in enumerate(selections):
        label,sid,j=s['run'],s['sample_id'],s['step_1based']-1
        profiles,online,cfg=contexts[label]
        result={**s}
        for variant in ['hold_once','greedy_once']:
            b,trace=rollout(episodes[sid],profiles[sid],online,cfg,j,variant)
            result[variant]=b
            pd.DataFrame(trace).to_csv(OUT/f'branch_{k+1:02d}_{sid}_{variant}.csv',
                                       index=False,encoding='utf-8-sig')
        h,g=result['hold_once'],result['greedy_once']
        assert h['prefix_sha256']==g['prefix_sha256'] and h['branch_full_state']==g['branch_full_state']
        match=pair[(pair.run==label)&(pair.sample_id==sid)&(pair.step_1based==j+1)].iloc[0]
        result.update(delta_G=g['suffix_return']-h['suffix_return'],
            delta_Q=float(match.delta_q),delta_Y=float(match.delta_y),delta_r=float(match.delta_r),
            greedy_Q=float(match.q_actual),hold_Q=float(match.q_hold),
            delta_full_zhou=g['full_zhou_evaluation']['total_cost_cny']-h['full_zhou_evaluation']['total_cost_cny'],
            both_completed=g['completed'] and h['completed'])
        branches.append(result)
        save_json('counterfactual_details.json',branches)
        print('branch',k+1,label,sid,j+1,'deltaG',result['delta_G'],flush=True)
    flat=[]
    for i,b in enumerate(branches):
        flat.append({'case':i+1,**{k:v for k,v in b.items() if not isinstance(v,dict)},
            **{variant+'_'+key:b[variant][key] for variant in ['hold_once','greedy_once']
               for key in ['completed','failure_cause','suffix_steps','suffix_return','soc_min',
                    'soc_max','soc_end','mean_abs_delta_kw','direction_reversals','suffix_zhou_three_cost']}})
    pd.DataFrame(flat).to_csv(OUT/'counterfactual_summary.csv',index=False,encoding='utf-8-sig')
    assert protected=={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    assert sha(old_cp_path)==identity['old_checkpoint_sha256']
    save_json('readonly_verification.json',{'formal_source_unchanged':True,
        'old_checkpoint_unchanged':True,'new_snapshot_unchanged':sha(snap_info['snapshot_path'])==snap_info['snapshot_sha256'],
        'new_training_latest_checkpoint_not_hashed_afterward':'Running trainer legitimately publishes newer rounds; audit never writes it.',
        'test_payloads_opened':0,'wall_seconds':time.time()-started})
    plot(states,pair,pd.DataFrame(flat))
    print('DONE',len(states),len(pair),'states/pairs',flush=True)


def plot(states,pair,branches):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for label,g in pair.groupby('run'):
        axs[0,0].hist(g.best_second_gap,bins=35,alpha=.45,label=label)
        axs[0,1].scatter(g.delta_y,g.delta_q,s=5,alpha=.25,label=label)
    axs[0,0].set(xlabel='Online best - second Q',ylabel='Comparable states')
    axs[0,0].legend(fontsize=7)
    axs[0,1].axvline(0,color='k',lw=.6);axs[0,1].axhline(0,color='k',lw=.6)
    axs[0,1].set(xlabel='Change - hold frozen DDQN target',ylabel='Change - hold Online Q')
    cal=pd.read_csv(OUT/'q_return_calibration_by_distance.csv')
    for (run,complete),g in cal.groupby(['run','completed']):
        g=g.set_index('distance_bin').reindex(['0','1-7','8-31','32-99','100-299','300+'])
        axs[1,0].plot(range(6),g.mean_Q-g.mean_G,marker='.',label=f'{run}, complete={complete}')
    axs[1,0].set_xticks(range(6),['0','1-7','8-31','32-99','100-299','300+'])
    axs[1,0].set(xlabel='Steps until observed terminal',ylabel='Mean pointwise Q - G')
    axs[1,0].legend(fontsize=6)
    axs[1,1].bar(branches['case'],branches.delta_G)
    axs[1,1].axhline(0,color='k',lw=.6)
    axs[1,1].set(xlabel='Purposively selected branch pair',ylabel='Actual return: change - hold')
    fig.tight_layout();fig.savefig(OUT/'q_target_return_diagnostics.png',dpi=160);plt.close(fig)
    # Full suffix profiles of selected exact state interventions.
    n=min(4,len(branches));fig,axs=plt.subplots(n,2,figsize=(13,3*n),squeeze=False)
    for i in range(n):
        b=branches.iloc[i]
        for variant in ['hold_once','greedy_once']:
            tr=pd.read_csv(OUT/f'branch_{i+1:02d}_{b.sample_id}_{variant}.csv')
            axs[i,0].plot(tr.time_s/3600,tr.fc_kw,label=variant,lw=.8)
            axs[i,1].plot(tr.time_s/3600,tr.soc_after,label=variant,lw=.8)
        axs[i,0].plot(tr.time_s/3600,tr.load_kw,color='k',alpha=.3,label='load')
        axs[i,0].set(title=f'Case {i+1}: {b.sample_id}, intervene step {b.step_1based}',ylabel='kW')
        axs[i,1].set(ylabel='SOC',xlabel='Voyage time (h)')
        axs[i,0].legend(fontsize=7);axs[i,1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'counterfactual_suffix_profiles.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    main()
