"""CPU/eval-only R100 Online/Target and frozen candidate Bellman-gap audit.

Creates new files exclusively. Does not load training_state, optimize, train, or
change any previously saved audit file. All trajectories are Train/Validation.
"""
import csv
import hashlib
import json
import math
from statistics import mean
import subprocess

import audit_support as a
from q_action_audit import distribution


OUTPUTS = ('online_target_disagreement.csv', 'action_pair_bellman_gap.csv',
           'online_target_review.json')


def read_csv(name):
    with (a.OUT / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_new(name, rows):
    with (a.OUT / name).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frozen(state_dict):
    with a.torch.random.fork_rng(devices=[]):
        model = a.MLPQNetwork(3, 7, (128, 128))
    model.load_state_dict(state_dict, strict=True)
    model.eval().requires_grad_(False)
    return a.FrozenPolicy(model)


def sign(x, eps=1e-12):
    return 1 if x > eps else -1 if x < -eps else 0


def packet(rows, fields):
    return {key: distribution([r[key] for r in rows]) for key in fields}


def state_summary(rows):
    return {
        'count': len(rows),
        'same_legal_best_count': sum(r['same_legal_best'] for r in rows),
        'different_legal_best_count': sum(not r['same_legal_best'] for r in rows),
        'online_best_delta_counts': {str(d): sum(r['online_best_delta_kw'] == d for r in rows)
                                    for d in a.FC_INCREMENT_KW},
        'target_best_delta_counts': {str(d): sum(r['target_best_delta_kw'] == d for r in rows)
                                    for d in a.FC_INCREMENT_KW},
        'statistics': packet(rows, ('online_minus_target_mean_all7', 'online_target_mae_all7',
            'online_minus_target_mean_legal', 'online_target_mae_legal',
            'centered_legal_advantage_mae', 'online_best_minus_second',
            'target_best_minus_second', 'target_regret_at_online_best',
            'online_regret_at_target_best', 'online_actual_minus_hold',
            'target_actual_minus_hold')),
        'all7_point_count': len(rows) * 7,
        'all7_online_above_target_count': sum(r[f'online_q_delta_{d:+d}_kw'] >
            r[f'target_q_delta_{d:+d}_kw'] for r in rows for d in a.FC_INCREMENT_KW),
    }


def pair_summary(rows):
    valid = [r for r in rows if r['delta_y_actual_minus_hold'] is not None]
    strict = [r for r in valid if sign(r['delta_q_actual_minus_hold']) != 0]
    opposite = [r for r in strict if sign(r['delta_q_actual_minus_hold']) *
                sign(r['delta_y_actual_minus_hold']) == -1]
    fields = ('delta_q_actual_minus_hold', 'delta_y_actual_minus_hold',
              'delta_q_minus_delta_y', 'immediate_actual_minus_hold',
              'target_bootstrap_actual_minus_hold',
              'current_target_actual_minus_hold')
    def basic(values):
        return {'count': len(values), 'statistics': packet(values, fields),
                'q_positive_count': sum(sign(r['delta_q_actual_minus_hold']) > 0 for r in values),
                'y_positive_count': sum(sign(r['delta_y_actual_minus_hold']) > 0 for r in values),
                'y_negative_count': sum(sign(r['delta_y_actual_minus_hold']) < 0 for r in values)}
    groups = {}
    for key in ('actual_delta_kw', 'soc_bin', 'load_bin_kw', 'previous_fc_bin_kw',
                'previous_fc_kw'):
        groups[key] = {str(label): basic([r for r in valid if r[key] == label])
                       for label in sorted(set(r[key] for r in valid))}
    opposite_groups = {}
    for key in ('actual_delta_kw', 'soc_bin', 'load_bin_kw', 'previous_fc_bin_kw',
                'previous_fc_kw'):
        opposite_groups[key] = {str(label): basic([r for r in opposite if r[key] == label])
                               for label in sorted(set(r[key] for r in opposite))}
    return {'total_states': len(rows), 'valid_comparable_targets': len(valid),
            'invalid_reasons': {reason: sum(r['comparison_status'] == reason for r in rows)
                                for reason in sorted(set(r['comparison_status'] for r in rows))
                                if reason != 'valid'},
            'nonzero_q_pairs': len(strict), 'opposite_q_y_sign_count': len(opposite),
            'overall': basic(valid), 'opposite_sign': basic(opposite),
            'by_group': groups, 'opposite_sign_by_group': opposite_groups}


def main():
    for name in OUTPUTS:
        if (a.OUT / name).exists():
            raise FileExistsError(a.OUT / name)
    initial_existing = {p.name: a.sha256(p) for p in a.OUT.iterdir() if p.is_file()}
    online, cp, metadata = a.load_policy()
    target = frozen(cp['target_state'])
    accountant, _ = a.accountant_config(cp)
    assert cp['hyperparameters']['gamma'] == 1.0
    assert cp['hyperparameters']['n_step'] == 1
    profiles = {sid: a.load_profile(sid, split) for sid, split in
                (('parent_020_seg01', 'validation'), ('parent_015_seg01', 'train'))}
    old_q = read_csv('q_action_audit_parent020.csv') + read_csv('q_action_audit_parent015.csv')
    rewards = {(r['sample_id'], int(r['step_1based']), role): r
               for r in read_csv('reward_decomposition.csv')
               for role in r['candidate_role'].split('|') if role in ('actual', 'hold')}
    old_boot = {int(r['step_1based']): r for r in read_csv('bootstrap_action_comparison.csv')}
    out_rows, pairs, states, legal_sets = [], [], [], []
    max_q_error, max_bootstrap_error, max_actual_target_error = 0., 0., 0.
    for qr in old_q:
        sid, step = qr['sample_id'], int(qr['step_1based'])
        index = step - 1
        records = profiles[sid]['transitions']
        item = records[index]
        state = (float(qr['state_soc']), float(qr['state_load_over600']),
                 float(qr['state_previous_fc_over600']))
        prev, load = float(qr['previous_fc_kw']), float(qr['load_kw'])
        legal = json.loads(qr['legal_action_indices'])
        full, candidate = a.physical_mask(a.AccountState(soc=state[0], previous_fc_kw=prev),
            load, accountant, item['duration_s'])
        assert list(full) == item['physical_feasible_actions_kw']
        assert list(candidate) == item['policy_candidate_actions_kw']
        assert legal == [i for i, d in enumerate(a.FC_INCREMENT_KW) if prev + d in candidate]
        oq, tq = online.q(state), target.q(state)
        for i, d in enumerate(a.FC_INCREMENT_KW):
            max_q_error = max(max_q_error, abs(oq[i] - float(qr[f'q_delta_{d:+d}_kw'])),
                              abs(tq[i] - float(qr[f'target_q_delta_{d:+d}_kw'])))
        assert max_q_error <= 1e-12
        orank = sorted(legal, key=lambda i: (-oq[i], i))
        trank = sorted(legal, key=lambda i: (-tq[i], i))
        oi, ti, actual = orank[0], trank[0], int(qr['actual_action_index'])
        assert oi == actual and 3 in legal
        om, tm = mean(oq[i] for i in legal), mean(tq[i] for i in legal)
        row = {'sample_id': sid, 'step_1based': step, 'soc': state[0],
            'load_kw': load, 'previous_fc_kw': prev, 'legal_indices': json.dumps(legal),
            'actual_delta_kw': a.FC_INCREMENT_KW[actual],
            'online_best_delta_kw': a.FC_INCREMENT_KW[oi],
            'target_best_delta_kw': a.FC_INCREMENT_KW[ti], 'same_legal_best': oi == ti,
            'online_best_minus_second': oq[oi] - oq[orank[1]] if len(legal) > 1 else None,
            'target_best_minus_second': tq[ti] - tq[trank[1]] if len(legal) > 1 else None,
            'online_minus_target_mean_all7': mean(x-y for x, y in zip(oq, tq)),
            'online_target_mae_all7': mean(abs(x-y) for x, y in zip(oq, tq)),
            'online_minus_target_mean_legal': om-tm,
            'online_target_mae_legal': mean(abs(oq[i]-tq[i]) for i in legal),
            'centered_legal_advantage_mae': mean(abs((oq[i]-om)-(tq[i]-tm)) for i in legal),
            'target_regret_at_online_best': tq[ti]-tq[oi],
            'online_regret_at_target_best': oq[oi]-oq[ti],
            'online_actual_minus_hold': oq[actual]-oq[3],
            'target_actual_minus_hold': tq[actual]-tq[3],
            'q_unit': 'scaled_reward_equivalent_cny'}
        for i, d in enumerate(a.FC_INCREMENT_KW):
            row.update({f'online_q_delta_{d:+d}_kw': oq[i], f'target_q_delta_{d:+d}_kw': tq[i],
                f'online_minus_target_delta_{d:+d}_kw': oq[i]-tq[i],
                f'online_legal_mean_advantage_delta_{d:+d}_kw': oq[i]-om if i in legal else None,
                f'target_legal_mean_advantage_delta_{d:+d}_kw': tq[i]-tm if i in legal else None,
                f'online_hold_advantage_delta_{d:+d}_kw': oq[i]-oq[3] if i in legal else None,
                f'target_hold_advantage_delta_{d:+d}_kw': tq[i]-tq[3] if i in legal else None})
        out_rows.append(row)
        states.append(state)
        legal_sets.append(legal)
        pair = {'sample_id': sid, 'step_1based': step, 'soc': state[0], 'load_kw': load,
            'previous_fc_kw': prev, 'actual_delta_kw': a.FC_INCREMENT_KW[actual],
            'soc_bin': f'[{math.floor(state[0]*10)/10:.1f},{(math.floor(state[0]*10)+1)/10:.1f})',
            'load_bin_kw': f'[{math.floor(load/200)*200},{(math.floor(load/200)+1)*200})',
            'previous_fc_bin_kw': f'[{math.floor(prev/100)*100},{(math.floor(prev/100)+1)*100})',
            'delta_q_actual_minus_hold': oq[actual]-oq[3],
            'current_target_actual_minus_hold': tq[actual]-tq[3],
            'archived_actual_done': item['done'], 'archived_actual_failure_penalty_scaled':
                item['failure_penalty_equivalent_cny'] * cp['hyperparameters']['reward_scale'],
            'archived_actual_fixed_ddqn_target': float(qr['fixed_ddqn_target']),
            'comparison_status': 'valid',
            'delta_y_actual_minus_hold': None, 'delta_q_minus_delta_y': None,
            'immediate_actual_minus_hold': None, 'target_bootstrap_actual_minus_hold': None,
            'opposite_q_y_sign': None}
        candidate_results = {}
        for role in ('actual', 'hold'):
            reward = rewards[sid, step, role]
            fc = float(reward['candidate_fc_kw'])
            soc = float(reward['candidate_soc_after'])
            immediate = float(reward['normal_training_reward_scaled_once'])
            bootstrap, y, next_delta, nlegal = None, None, None, []
            if item['done']:
                status = 'actual_terminal_no_matched_counterfactual_target'
            else:
                nxt = records[index+1]
                physical = a.AccountState(soc=soc, previous_fc_kw=fc,
                    fc_loss_uv=float(reward['candidate_fc_loss_after_uv']),
                    battery_weighted_ah=float(reward['candidate_battery_weighted_ah_after']))
                _, ncand = a.physical_mask(physical, nxt['load_kw'], accountant, nxt['duration_s'])
                nlegal = [i for i, d in enumerate(a.FC_INCREMENT_KW) if fc + d in ncand]
                if nlegal:
                    ns = (soc, nxt['load_kw']/600, fc/600)
                    nq, ntq = online.q(ns), target.q(ns)
                    ni = max(nlegal, key=lambda i: nq[i])
                    bootstrap, next_delta = ntq[ni], a.FC_INCREMENT_KW[ni]
                    y = immediate + cp['hyperparameters']['gamma']*bootstrap
                    status = 'valid'
                else:
                    status = f'{role}_no_legal_successor_no_invented_failure_penalty'
                if role == 'actual':
                    assert list(ncand) == item['next_feasible_actions_kw']
                    max_actual_target_error = max(max_actual_target_error,
                        abs(y - float(qr['fixed_ddqn_target'])))
            candidate_results[role] = (immediate, bootstrap, y, status)
            pair.update({f'{role}_candidate_soc_after': soc, f'{role}_candidate_fc_kw': fc,
                         f'{role}_normal_scaled_reward': immediate, f'{role}_next_target_q': bootstrap,
                         f'{role}_frozen_ddqn_target': y, f'{role}_next_best_delta_kw': next_delta,
                         f'{role}_next_legal_indices': json.dumps(nlegal)})
            if status != 'valid':
                pair['comparison_status'] = status
        pair['immediate_actual_minus_hold'] = candidate_results['actual'][0]-candidate_results['hold'][0]
        if pair['comparison_status'] == 'valid':
            dy = candidate_results['actual'][2]-candidate_results['hold'][2]
            pair['delta_y_actual_minus_hold'] = dy
            pair['delta_q_minus_delta_y'] = pair['delta_q_actual_minus_hold']-dy
            pair['target_bootstrap_actual_minus_hold'] = candidate_results['actual'][1]-candidate_results['hold'][1]
            pair['opposite_q_y_sign'] = sign(pair['delta_q_actual_minus_hold'])*sign(dy) == -1
        if sid == 'parent_020_seg01':
            old = old_boot[step]
            for ours, saved in (('delta_q_actual_minus_hold', 'current_q_actual_minus_hold'),
                    ('delta_y_actual_minus_hold', 'frozen_target_actual_minus_hold'),
                    ('delta_q_minus_delta_y', 'current_q_gap_minus_frozen_target_gap'),
                    ('immediate_actual_minus_hold', 'immediate_actual_minus_hold'),
                    ('target_bootstrap_actual_minus_hold', 'bootstrap_actual_minus_hold')):
                max_bootstrap_error = max(max_bootstrap_error, abs(pair[ours]-float(old[saved])))
        pairs.append(pair)
    assert len(out_rows) == 219 and len(old_boot) == 64
    assert max_bootstrap_error <= 1e-12 and max_actual_target_error <= 1e-12

    lag = []
    target_parameters = a.torch.cat([v.flatten().double() for v in cp['target_state'].values()])
    for path in sorted((a.RUN / 'diagnostic_checkpoints').glob('round_*.pt')):
        old = a.torch.load(path, map_location='cpu', weights_only=False)
        assert old['purpose'] == 'diagnostic_only'
        for key in ('source_commit', 'experiment_id', 'manifest_sha256', 'hyperparameters'):
            assert old[key] == metadata[key]
        old_online = frozen(old['online_state'])
        qold = [old_online.q(s) for s in states]
        params = a.torch.cat([v.flatten().double() for v in old['online_state'].values()])
        diffs = params-target_parameters
        all_diffs = [q[i]-r[f'target_q_delta_{d:+d}_kw'] for q, r in zip(qold, out_rows)
                     for i, d in enumerate(a.FC_INCREMENT_KW)]
        legal_diffs = [q[i]-r[f'target_q_delta_{a.FC_INCREMENT_KW[i]:+d}_kw']
                       for q, r, ls in zip(qold, out_rows, legal_sets) for i in ls]
        lag.append({'actual_saved_round': old['round'],
            'economic_optimizer_updates': old['economic_optimizer_updates'],
            'updates_before_r100': cp['economic_optimizer_updates']-old['economic_optimizer_updates'],
            'parameter_rmse_to_r100_target': float(diffs.square().mean().sqrt()),
            'parameter_l2_to_r100_target': float(diffs.square().sum().sqrt()),
            'all7_q_mae_to_r100_target_on_r100_states': mean(abs(x) for x in all_diffs),
            'legal_q_mae_to_r100_target_on_r100_states': mean(abs(x) for x in legal_diffs),
            'same_best_as_r100_target_on_r100_states': sum(
                a.FC_INCREMENT_KW[max(ls, key=lambda i: q[i])] == r['target_best_delta_kw']
                for q, r, ls in zip(qold, out_rows, legal_sets)),
            'checkpoint_sha256': a.sha256(path)})

    metrics = {}
    with (a.RUN / 'round_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if int(r['round']) in (90, 100):
                keys = ('economic_optimizer_updates', 'economic_optimizer_updates_cumulative',
                        'target_soft_update_calls_cumulative', 'economic_optimizer_updates_this_round')
                metrics[r['round']] = {k: int(r[k]) for k in keys}
    history = json.loads((a.RUN / 'round_history.json').read_text(encoding='utf-8'))
    assert metrics['100']['target_soft_update_calls_cumulative'] == cp['economic_optimizer_updates']
    assert history['bootstrap_optimizer_updates']+history['formal_training_optimizer_updates'] == cp['economic_optimizer_updates']
    tau = cp['hyperparameters']['target_tau']
    source_paths = ('src/v4/dqn.py', 'src/v4/decision_diagnostics.py')
    source_checks = {}
    for path in source_paths:
        source = subprocess.check_output(['git', 'show', cp['source_commit']+':'+path], cwd=a.ROOT)
        current = (a.ROOT/path).read_bytes()
        source_checks[path] = {'run_source_sha256': hashlib.sha256(source).hexdigest(),
                              'current_source_sha256': hashlib.sha256(current).hexdigest(),
                              'same_ignoring_crlf': source.replace(b'\r\n', b'\n') == current.replace(b'\r\n', b'\n')}
    current_existing = {name: a.sha256(a.OUT/name) for name in initial_existing}
    assert current_existing == initial_existing
    summary = {'checkpoint_path': str(a.RUN/'diagnostic_checkpoints/round_100.pt'),
        'checkpoint_sha256': a.sha256(a.RUN/'diagnostic_checkpoints/round_100.pt'),
        'run_source_commit': cp['source_commit'], 'current_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=a.ROOT, text=True).strip(),
        'purpose': 'diagnostic_only', 'device': 'cpu', 'eval': True,
        'inference_batch_size': 1, 'gradient_optimizer_training_calls': 0,
        'training_state_loaded': False, 'test_payloads_opened': 0,
        'formal_or_existing_audit_files_modified': False,
        'definitions': {
            'absolute_bias': 'Online Q minus Target Q, signed mean and mean absolute value; Target is not truth',
            'centered_advantage': 'Q(a) minus the arithmetic mean Q over the current legal action set',
            'hold_advantage': 'Q(a) minus Q(hold) at the same current state',
            'delta_q': 'Online(current actual increment) minus Online(hold)',
            'delta_y': 'frozen R100 DDQN one-step candidate target actual minus hold',
            'one_step_target': 'formal archived candidate normal scaled reward + gamma * Target(candidate successor, argmax legal Online)',
            'bins': 'SOC fixed width 0.1; current load fixed width 200 kW; previous FC fixed width 100 kW; exact previous FC also grouped',
            'sign_tolerance': 1e-12},
        'verified_max_abs_q_difference_against_existing219_states': max_q_error,
        'verified_max_abs_difference_against_existing020_candidate_comparison': max_bootstrap_error,
        'verified_max_abs_difference_against_archived_actual_frozen_targets': max_actual_target_error,
        'state_summaries': {sid: state_summary([r for r in out_rows if r['sample_id'] == sid]) for sid in profiles},
        'action_pair_summaries': {sid: pair_summary([r for r in pairs if r['sample_id'] == sid]) for sid in profiles},
        'target_update_audit': {'tau': tau, 'checkpoint_economic_optimizer_updates': cp['economic_optimizer_updates'],
            'checkpoint_target_sync_calls': cp['target_sync_calls'],
            'checkpoint_does_not_save_target_soft_update_calls': 'target_soft_update_calls' not in cp,
            'actual_round_metrics': metrics,
            'bootstrap_optimizer_updates': history['bootstrap_optimizer_updates'],
            'formal_training_optimizer_updates': history['formal_training_optimizer_updates'],
            'ema_mean_age_optimizer_updates_if_stationary': (1-tau)/tau,
            'ema_half_life_optimizer_updates': math.log(.5)/math.log(1-tau),
            'r90_target_retained_coefficient_at_r100': (1-tau)**(cp['economic_optimizer_updates']-
                next(r['economic_optimizer_updates'] for r in lag if r['actual_saved_round'] == 90)),
            'source_checks': source_checks,
            'source_evidence': ['src/v4/dqn.py:486 optimizer.step; :487 economic counter increment; :488 soft update',
                'src/v4/dqn.py:824-828 target.lerp_(online, tau) and soft counter increment',
                'src/v4/decision_diagnostics.py:128-129 checkpoint records economic counter and hard-sync counter']},
        'actual_saved_checkpoint_comparison': lag,
        'closest_saved_online_to_r100_target_by_parameter_rmse': min(lag, key=lambda r: r['parameter_rmse_to_r100_target']),
        'closest_saved_online_to_r100_target_by_all7_q_mae': min(lag, key=lambda r: r['all7_q_mae_to_r100_target_on_r100_states']),
        'limitations': ['All targets and Q are learned estimates, never optimal-value ground truth.',
            'Frozen R100 targets differ from historical optimizer-time targets.',
            'Candidate next load is the next actual recorded observation used only for offline diagnosis, never an online policy input.',
            'Candidate normal rewards and successor physical states reuse reward_decomposition.csv formal one-step results.',
            'The final parent015 transition is a real recorded failure terminal; no counterfactual failure penalty or terminal target is invented for hold.',
            'No legal successor comparisons are marked unavailable rather than inventing penalties.',
            '219 states are diagnostic trajectory states, not an independent estimate of full Replay or Train distributions.',
            'Old online checkpoint distances are computed only at actual saved rounds; no exact round38/62 weights are available.',
            'EMA lag algebra and sparse saved checkpoint similarity cannot identify which network is closer to optimal Q.']}
    write_csv_new(OUTPUTS[0], out_rows)
    write_csv_new(OUTPUTS[1], pairs)
    with (a.OUT/OUTPUTS[2]).open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({'new_outputs': list(OUTPUTS), 'state_counts': {k: v['count'] for k, v in summary['state_summaries'].items()},
        'same_best_counts': {k: v['same_legal_best_count'] for k, v in summary['state_summaries'].items()},
        'valid_pair_counts': {k: v['valid_comparable_targets'] for k, v in summary['action_pair_summaries'].items()},
        'opposite_sign_counts': {k: v['opposite_q_y_sign_count'] for k, v in summary['action_pair_summaries'].items()},
        'existing_outputs_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
