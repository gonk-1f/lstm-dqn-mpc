"""Read-only MPC integration; isolated fixed-physical-L2 Train reward audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import audit_ideal_reference_84_train as prior

ROOT = prior.ROOT
SOURCE = prior.OUT
OUT = ROOT / 'outputs/fixed_physical_l2_84_train_20260912'
TERMS = prior.TERMS
READS = set()


def read_csv(path):
    READS.add(str(path.relative_to(ROOT)))
    return pd.read_csv(path)


def arrays():
    states = read_csv(SOURCE / 'states.csv')
    path = SOURCE / 'state_physical_results.npz'
    READS.add(str(path.relative_to(ROOT)))
    with np.load(path) as d:
        f, exact, fc, valid = (d[k] for k in ('f', 'exact_f', 'fc_plans', 'valid'))
    assert states.segment_id.str.startswith('train_').all()
    assert np.array_equal(states.audit_id, np.arange(1440))
    assert valid[:, 4:].all() and f[:, 4:].shape == (1440, 84, 4)
    return states, f, exact, fc, valid


def reward(f):
    f = np.asarray(f, dtype=float)
    if f.shape[-1] != 4 or not np.isfinite(f).all():
        raise ValueError('four finite physical metrics are required')
    return 1. / (1. + np.linalg.norm(f / 6., axis=-1))


def shares(f):
    sq = (np.asarray(f) / 6.) ** 2
    denom = sq.sum(axis=-1, keepdims=True)
    # Undefined shares at a perfect zero vector are reported as zero, not rescaled.
    return np.divide(sq, denom, out=np.zeros_like(sq), where=denom != 0)


def metric_summary(f):
    sh = shares(f)
    return {'reward': prior.distribution(reward(f)),
            'average_terms': {t: prior.distribution(f[..., i]/6.) for i, t in enumerate(TERMS)},
            'squared_share': {t: prior.distribution(sh[..., i]) for i, t in enumerate(TERMS)},
            'fraction_share_above_90pct': {t: float(np.mean(sh[..., i] > .9)) for i, t in enumerate(TERMS)}}


def comparison(base_f, other_f, base_fc, other_fc):
    a, b = reward(base_f), reward(other_f)
    wa, wb = int(a.argmax()), int(b.argmax())
    return {'winner': wa, 'other_winner': wb, 'winner_changed': wa != wb,
            'max_reward_shift': float(abs(a-b).max()),
            'winner_regret_under_other': float(b.max()-b[wa]),
            'chosen_first_fc_change_kw': float(abs(base_fc[wa, 0]-other_fc[wb, 0])),
            'chosen_plan_rms_change_kw': float(np.sqrt(np.mean((base_fc[wa]-other_fc[wb])**2))),
            'max_same_action_plan_rms_change_kw': float(np.sqrt(np.mean((base_fc-other_fc)**2, axis=-1)).max())}


def rescore():
    states, all_f, exact, all_fc, valid = arrays()
    f, fc = all_f[:, 4:], all_fc[:, 4:]
    q = np.array([a.as_tuple() for a in prior.weight_grid()])
    r = reward(f); sh = shares(f); w = r.argmax(axis=1)
    sorted_r = np.sort(r, axis=1)
    out = states.copy()
    out['winner'] = w
    out['winner_reward'] = r[np.arange(len(w)), w]
    out['second_reward'] = sorted_r[:, -2]
    out['gap'] = sorted_r[:, -1]-sorted_r[:, -2]
    out['relative_gap'] = out.gap/out.winner_reward
    out['reward_range'] = r.max(axis=1)-r.min(axis=1)
    out['winner_fc'] = fc[np.arange(len(w)), w, 0]
    out['min_grid_fc'] = fc[:, :, 0].min(axis=1)
    out['max_grid_fc'] = fc[:, :, 0].max(axis=1)
    out['winner_fc_deficit_to_max'] = out.max_grid_fc-out.winner_fc
    for i, term in enumerate(TERMS):
        out['winner_q'+term] = q[w, i]
        out['winner_bar_'+term] = f[np.arange(len(w)), w, i]/6.
        out['winner_share_'+term] = sh[np.arange(len(w)), w, i]
        out['all_action_mean_share_'+term] = sh[:, :, i].mean(axis=1)
    out['load_bin'] = pd.cut(out.current_load_kw, [-np.inf, 300, 600, 900, np.inf],
                            labels=['<=300', '300-600', '600-900', '>900']).astype(str)
    pairs = np.array([(i, j) for i in range(84) for j in range(i+1, 84)
                      if np.abs(np.rint(q[i]*10)-np.rint(q[j]*10)).sum() == 2])
    equivalence = []
    for i in range(len(states)):
        rms = np.sqrt(np.mean((fc[i, pairs[:, 0]]-fc[i, pairs[:, 1]])**2, axis=1))
        dr = abs(r[i, pairs[:, 0]]-r[i, pairs[:, 1]])
        equivalence.append({'audit_id': i, 'neighbor_pairs': len(pairs),
                            'neighbor_plan_rms_lt_01kw': float(np.mean(rms < .1)),
                            'neighbor_plan_rms_lt_1kw': float(np.mean(rms < 1)),
                            'neighbor_reward_gap_lt_1e6': float(np.mean(dr < 1e-6)),
                            'reward_tie_count_1e10': int(np.sum(r[i].max()-r[i] < 1e-10))})
    eq = pd.DataFrame(equivalence)
    out = out.merge(eq, on='audit_id', validate='one_to_one')
    out.to_csv(OUT/'state_summary.csv', index=False)
    records = pd.DataFrame({'audit_id': np.repeat(states.audit_id, 84),
                            'action_id': np.tile(np.arange(84), len(states)), 'reward': r.ravel(),
                            'p_fc0': fc[:, :, 0].ravel()})
    for i, t in enumerate(TERMS):
        records['bar_'+t] = (f[:, :, i]/6.).ravel()
        records['share_'+t] = sh[:, :, i].ravel()
    records.to_csv(OUT/'action_scores.csv', index=False)
    group_rows = []
    for (cohort, soc, load_bin), g in out.groupby(['cohort', 'soc', 'load_bin']):
        group_rows.append({'cohort': cohort, 'soc': soc, 'load_bin': load_bin, 'count': len(g),
                           'winner_counts': g.winner.value_counts().to_dict(),
                           'gap': prior.distribution(g.gap), 'reward': prior.distribution(g.winner_reward),
                           **{'mean_winner_q'+t: float(g['winner_q'+t].mean()) for t in TERMS}})
    prior.write_json(OUT/'winner_groups.json', group_rows)
    matched = out[out.cohort == 'matched_soc']
    paired = []
    for pair_id, g in matched.groupby('pair_id'):
        g = g.set_index('soc').loc[[.55, .45, .35, .25, .22]]
        assert g.current_load_kw.nunique() == g.previous_fc_kw.nunique() == 1
        ids = g.audit_id.to_numpy(dtype=int)
        paired.append({'pair_id': pair_id, 'winner_reward_strictly_decreasing': bool((np.diff(g.winner_reward) < 0).all()),
                       'every_action_reward_strictly_decreasing': bool((np.diff(r[ids], axis=0) < 0).all()),
                       'qS_nondecreasing': bool((np.diff(g.winner_qS) >= 0).all()),
                       'first_fc_nondecreasing_01kw': bool((np.diff(g.winner_fc) >= -.1).all()),
                       'fc_022_minus_055': float(g.winner_fc.loc[.22]-g.winner_fc.loc[.55])})
    pd.DataFrame(paired).to_csv(OUT/'paired_soc_checks.csv', index=False)
    soc_stats = []
    for soc, g in matched.groupby('soc', sort=False):
        ids = g.audit_id.to_numpy(dtype=int)
        soc_stats.append({'soc': soc, 'count': len(g), 'reward': prior.distribution(g.winner_reward),
                          'qS': prior.distribution(g.winner_qS), 'gap': prior.distribution(g.gap),
                          'winner_counts': g.winner.value_counts().to_dict(),
                          'mean_all_action_share_S': float(sh[ids, :, 2].mean()),
                          'max_grid_fc_deficit': prior.distribution(g.winner_fc_deficit_to_max)})
    prior.write_json(OUT/'matched_soc_summary.json', soc_stats)
    exact_checks = pd.DataFrame([{'audit_id': i, **comparison(f[i], exact[i, 4:], fc[i], fc[i])}
                                for i in range(len(states))])
    exact_checks.to_csv(OUT/'exact_soc_sensitivity.csv', index=False)
    summary = {'states': len(states), 'successful_grid_solves_reused': int(valid[:, 4:].sum()),
               'all_action_metrics': metric_summary(f), 'winner_metrics': metric_summary(f[np.arange(len(w)), w]),
               'gap': prior.distribution(out.gap), 'relative_gap': prior.distribution(out.relative_gap),
               'reward_range': prior.distribution(out.reward_range),
               'winners': out.winner.value_counts().to_dict(),
               'neighbor_pairs': len(pairs),
               'equivalence': {c: prior.distribution(out[c]) for c in eq.columns if c != 'audit_id'},
               'paired_soc_checks': {c: int(pd.DataFrame(paired)[c].sum()) for c in paired[0] if c.endswith(('decreasing', '01kw'))},
               'raw_vs_exact_soc': {'max_reward_shift': prior.distribution(exact_checks.max_reward_shift),
                                    'winner_changes': int(exact_checks.winner_changed.sum()),
                                    'regret': prior.distribution(exact_checks.winner_regret_under_other),
                                    'chosen_first_fc_change_kw': prior.distribution(exact_checks.chosen_first_fc_change_kw)}}
    prior.write_json(OUT/'rescore_summary.json', summary)
    print('RESCORE_DONE', len(states), 'states;', len(records), 'action scores', flush=True)


def traces():
    old = read_csv(SOURCE/'closed_loop_summary.csv')
    summaries, frames = [], []
    for row in old.itertuples():
        rec = {'window': row.window, 'policy': row.policy, 'steps': row.steps,
               'completed': row.completed, 'final_soc': row.final_soc,
               'full_window_physical_feasibility': row.full_window_physical_feasibility,
               'new_policy_rollout': False}
        if row.steps:
            df = read_csv(SOURCE/f'trace_{row.window}_{row.policy}.csv')
            f = df[list(TERMS)].to_numpy()
            df['old_reward'] = df.reward
            df['new_reward'] = reward(f)
            for i, t in enumerate(TERMS):
                df['bar_'+t] = f[:, i]/6.
                df['share_'+t] = shares(f)[:, i]
            df['window'] = row.window; df['policy'] = row.policy
            frames.append(df)
            rec.update(metrics=metric_summary(f),
                       lag1_reward_correlation=float(df.new_reward.autocorr()) if len(df)>1 else None)
        else:
            rec['limitation'] = 'old anchor failure at step zero; no saved trajectory metrics'
        summaries.append(rec)
    all_traces = pd.concat(frames, ignore_index=True)
    all_traces.to_csv(OUT/'rescored_old_policy_traces.csv', index=False)
    prior.write_json(OUT/'rescored_old_policy_summary.json',
                     {'windows': summaries, 'pooled_saved_steps': len(all_traces),
                      'pooled_metrics': metric_summary(all_traces[list(TERMS)].to_numpy()),
                      'interpretation': 'Old policy trajectories only; neither a new reward greedy controller nor a DQN rollout.'})
    print('TRACE_RESCORE_DONE', len(all_traces), 'saved steps', flush=True)


def stability():
    states, all_f, all_exact, all_fc, valid = arrays()
    selected = pd.concat([g.iloc[np.linspace(0, len(g)-1, 10, dtype=int)]
                          for _, g in states.groupby(['cohort', 'soc'])]).sort_values('audit_id')
    assert len(selected) == 120
    selected.to_csv(OUT/'stability_states.csv', index=False)
    actions = prior.weight_grid()
    details, logs, failures, results = [], [], [], {}
    rep_path = SOURCE/'anchor_repetitions.npz'
    READS.add(str(rep_path.relative_to(ROOT)))
    with np.load(rep_path) as rep:
        anchor_reps = {k: rep[k] for k in rep.files}
    old_anchor_checks = []
    for phase, reverse, tight, cold in [('warm_forward', False, False, False), ('warm_reverse', True, False, False),
                                       ('tight_reverse', True, True, False), ('cold_default', False, False, True)]:
        bank = prior.make_bank(actions, tight=tight)
        plan = np.full((120, 84, 6), np.nan); fs = np.full((120, 84, 4), np.nan)
        good = np.zeros((120, 84), dtype=bool)
        for counter, row in enumerate((selected.iloc[::-1] if reverse else selected).itertuples()):
            i = row.audit_id; local = int(np.flatnonzero(selected.audit_id.to_numpy()==i)[0])
            rr = []
            for action in (actions[::-1] if reverse else actions):
                aid = action.action_id
                if cold:
                    entry = bank._entries[aid]
                    entry.solver.warm_start(x=np.zeros(25), y=np.zeros(entry.A.shape[0]))
                rec = prior.solve(bank, aid, np.full(6, row.current_load_kw), row.soc, row.previous_fc_kw)
                rr.append(rec)
                logs.append({'phase': phase, 'audit_id': i, **{k: v for k, v in rec.items() if k not in ('f', 'exact_f', 'fc')}})
                good[local, aid] = rec['valid']
                if rec['valid']:
                    fs[local, aid] = rec['f']; plan[local, aid] = rec['fc']
            if good[local].all():
                cmp = comparison(all_f[i, 4:], fs[local], all_fc[i, 4:], plan[local])
                cmp.update(phase=phase, audit_id=i, soc=row.soc, cohort=row.cohort)
                # Same grid perturbation with the ORIGINAL fixed anchors is an
                # offline sensitivity comparison, not the candidate reward.
                if valid[i, :4].all():
                    _, a = prior.reference_reward(all_f[i, 4:], all_f[i, :4])
                    _, b = prior.reference_reward(fs[local], all_f[i, :4])
                    cmp['old_reference_fixed_anchor_reward_shift'] = float(abs(a-b).max())
                    cmp['old_reference_fixed_anchor_winner_changed'] = bool(a.argmax()!=b.argmax())
                details.append(cmp)
            else:
                failures.append({'phase': phase, 'audit_id': i,
                                 'failed_actions': [x['action_id'] for x in rr if not x['valid']],
                                 'failure_class': prior.classify_failures(rr, np.full(6, row.current_load_kw), row.soc, row.previous_fc_kw)})
            if (counter+1)%40 == 0:
                print('STABILITY', phase, counter+1, '/120', flush=True)
        results[phase+'_f'] = fs; results[phase+'_fc'] = plan; results[phase+'_valid'] = good
    for row in selected.itertuples():
        i = row.audit_id
        for phase in ('reverse', 'tight'):
            if valid[i, :4].all() and anchor_reps[phase+'_valid'][i].all():
                _, a = prior.reference_reward(all_f[i, 4:], all_f[i, :4])
                _, b = prior.reference_reward(all_f[i, 4:], anchor_reps[phase][i])
                old_anchor_checks.append({'audit_id': i, 'phase': phase, 'max_reward_shift': float(abs(a-b).max()),
                                          'winner_changed': bool(a.argmax()!=b.argmax())})
    np.savez_compressed(OUT/'stability_results.npz', audit_ids=selected.audit_id.to_numpy(), **results)
    pd.DataFrame(logs).to_csv(OUT/'stability_solve_log.csv', index=False)
    df = pd.DataFrame(details); df.to_csv(OUT/'stability_comparisons.csv', index=False)
    pd.DataFrame(old_anchor_checks).to_csv(OUT/'old_anchor_sensitivity_comparison.csv', index=False)
    prior.write_json(OUT/'stability_failures.json', failures)
    summary = {}
    for phase, g in df.groupby('phase'):
        summary[phase] = {'compared_states': len(g), 'winner_changes': int(g.winner_changed.sum()),
                          **{c: prior.distribution(g[c]) for c in ['max_reward_shift', 'winner_regret_under_other',
                              'chosen_first_fc_change_kw', 'chosen_plan_rms_change_kw',
                              'max_same_action_plan_rms_change_kw', 'old_reference_fixed_anchor_reward_shift']}}
    prior.write_json(OUT/'stability_summary.json', summary)


def verify():
    states, f, exact, fc, valid = arrays()
    # Independent scalar hand calculation, and analytical physical examples.
    assert reward([6., 12., 18., 24.]) == 1./(1.+np.sqrt(30.))
    assert reward([0., 0., 0., 0.]) == 1.
    for soc, expected in [(0.55, 0.), (.45, 4.), (.35, 16.), (.25, 36.), (.22, 43.56)]:
        terms, _, _ = prior.direct_terms(np.full(6, 300.), np.full(6, 300.), soc, 300.)
        np.testing.assert_allclose(terms[2]/6., expected, atol=1e-12)
    # 1248 kW discharge reaches Bbar=4, not 4 per six-step sum.
    terms, _, _ = prior.direct_terms(np.full(6, 600.), np.full(6, 1848.), .55, 600.)
    np.testing.assert_allclose(terms[1]/6., 4.)
    score = read_csv(OUT/'action_scores.csv')
    np.testing.assert_allclose(score.reward.to_numpy().reshape(1440, 84), reward(f[:, 4:]), rtol=1e-14)
    np.testing.assert_allclose(shares(f[:, 4:]).sum(axis=-1), 1., atol=1e-15)
    np.testing.assert_array_equal(np.linalg.norm(f[:, 4:], axis=-1).argmin(axis=1), reward(f[:, 4:]).argmax(axis=1))
    log = read_csv(OUT/'stability_solve_log.csv')
    assert len(log) == 40320 and log.valid.all()
    supplement_log = read_csv(OUT/'window_supplement_solve_log.csv')
    assert len(supplement_log) == 3360 and supplement_log.valid.all()
    supplement = read_csv(OUT/'window_supplement_summary.csv')
    path = OUT/'window_supplement_results.npz'
    READS.add(str(path.relative_to(ROOT)))
    with np.load(path) as d:
        sr = reward(d['f'])
    np.testing.assert_allclose(supplement.winner_reward, sr.max(axis=1), rtol=1e-14)
    np.testing.assert_array_equal(supplement.winner, sr.argmax(axis=1))
    saved = read_csv(OUT/'rescored_old_policy_traces.csv')
    assert len(saved) == 6927
    np.testing.assert_allclose(saved.new_reward, reward(saved[list(TERMS)].to_numpy()), rtol=1e-14)
    previous = json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    changed = [p for p, h in previous.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    assert not changed, changed
    summary = {'protected_files_unchanged': len(previous), 'grid_results_verified': 1440*84,
               'new_stability_solves_verified': len(log), 'supplement_solves_verified': len(supplement_log),
               'old_policy_trace_scores_verified': len(saved),
               'hand_checks_passed': True, 'formal_reward_modified': False, 'dqn_trained': False,
               'validation_test_dataset_read': False, 'commit_push': False}
    prior.write_json(OUT/'verification_summary.json', summary)
    print('VERIFIED', summary, flush=True)


def window_states():
    """Supplement the original 150-688 kW snapshot range from saved Train traces."""
    meta = read_csv(SOURCE/'closed_loop_summary.csv')
    rows = []
    for row in meta[meta.policy == 'max_first_fc_grid_diagnostic'].itertuples():
        if row.steps:
            df = read_csv(SOURCE/f'trace_{row.window}_{row.policy}.csv')
        else:
            assert row.window == 'low_recharge'
            df = read_csv(SOURCE/'control_only_low_load_trace.csv')
            path = prior.TRAIN_ROOT / (row.segment+'.csv')
            READS.add(str(path.relative_to(ROOT)))
            first = float(pd.read_csv(path, usecols=['load_total_kw'], nrows=1).iloc[0, 0])
            df['load_current'] = df.load_actual.shift(1).fillna(first)
            df['previous_fc'] = df.fc.shift(1).fillna(row.initial_fc)
        for j in np.linspace(0, len(df)-1, 5, dtype=int):
            x = df.iloc[j]
            rows.append({'window': row.window, 'segment': row.segment, 'step': int(x.step),
                         'soc': x.soc_before, 'load': x.load_current, 'previous_fc': x.previous_fc})
    selected = pd.DataFrame(rows)
    selected.to_csv(OUT/'window_supplement_states.csv', index=False)
    actions = prior.weight_grid(); bank = prior.make_bank(actions)
    q = np.array([a.as_tuple() for a in actions])
    f = np.full((len(selected), 84, 4), np.nan)
    fc = np.full((len(selected), 84, 6), np.nan)
    logs, summary = [], []
    for i, row in enumerate(selected.itertuples()):
        rr = [prior.solve(bank, a.action_id, np.full(6, row.load), row.soc, row.previous_fc) for a in actions]
        logs.extend({'supplement_id': i, **{k:v for k,v in rec.items() if k not in ('f','exact_f','fc')}} for rec in rr)
        if not all(rec['valid'] for rec in rr):
            summary.append({'supplement_id': i, 'window': row.window,
                            'failure_class': prior.classify_failures(rr, np.full(6, row.load), row.soc, row.previous_fc)})
            continue
        f[i] = np.array([rec['f'] for rec in rr]); fc[i] = np.array([rec['fc'] for rec in rr])
        r = reward(f[i]); w = int(r.argmax()); ranked = np.sort(r)
        summary.append({'supplement_id': i, 'window': row.window, 'soc': row.soc, 'load': row.load,
                        'winner': w, 'winner_reward': float(r[w]), 'gap': float(ranked[-1]-ranked[-2]),
                        'reward_range': float(r.max()-r.min()), 'winner_fc': float(fc[i, w, 0]),
                        'max_grid_fc': float(fc[i, :, 0].max()),
                        'winner_share_S': float(shares(f[i])[w, 2]), 'reward_ties_1e10': int(np.sum(r.max()-r < 1e-10)),
                        **dict(zip(['q'+t for t in TERMS], q[w])), 'failure_class': 'none'})
    np.savez_compressed(OUT/'window_supplement_results.npz', f=f, fc=fc)
    pd.DataFrame(logs).to_csv(OUT/'window_supplement_solve_log.csv', index=False)
    df = pd.DataFrame(summary); df.to_csv(OUT/'window_supplement_summary.csv', index=False)
    good = np.isfinite(f).all(axis=(1,2))
    prior.write_json(OUT/'window_supplement_summary.json', {'states': len(df), 'valid_states': int(good.sum()),
                     'load': prior.distribution(selected.load), 'all_action_metrics': metric_summary(f[good]),
                     'gap': prior.distribution(df.gap), 'winner_counts': df.winner.value_counts().to_dict(),
                     'limitation': 'Five states per saved Train window; supplementary snapshot solves, not a new policy rollout.'})
    print('WINDOW_SUPPLEMENT_DONE', len(selected), 'states;', int(good.sum())*84, 'valid action solves', flush=True)


def precision():
    states, f, exact, fc, valid = arrays()
    path = OUT/'window_supplement_results.npz'
    READS.add(str(path.relative_to(ROOT)))
    with np.load(path) as d:
        wf, wp = d['f'], d['fc']
    summary = {}
    for name, values, plans in [('snapshots', f[:,4:], fc[:,4:]), ('window_supplement', wf, wp)]:
        r = reward(values); r32 = r.astype(np.float32)
        w = r.argmax(axis=1); w32 = r32.argmax(axis=1); ix = np.arange(len(w))
        d = pd.DataFrame({'row': ix, 'winner_changed': w != w32,
                          'regret': r.max(axis=1)-r[ix, w32],
                          'first_fc_change_kw': abs(plans[ix,w,0]-plans[ix,w32,0]),
                          'plan_rms_change_kw': np.sqrt(np.mean((plans[ix,w]-plans[ix,w32])**2,axis=1)),
                          'float32_best_tie_count': (r32==r32.max(axis=1,keepdims=True)).sum(axis=1)})
        d.to_csv(OUT/f'{name}_float32_diagnostic.csv', index=False)
        summary[name] = {'winner_changes': int(d.winner_changed.sum()),
                         **{c: prior.distribution(d[c]) for c in d.columns if c not in ('row', 'winner_changed')}}
    prior.write_json(OUT/'float32_summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['rescore', 'stability', 'window-states', 'precision', 'verify', 'all'], default='all')
    args = parser.parse_args()
    if args.phase in ('all', 'rescore'):
        rescore(); traces()
    if args.phase in ('all', 'stability'):
        stability()
    if args.phase in ('all', 'window-states'):
        window_states()
    if args.phase in ('all', 'precision'):
        precision()
    if args.phase in ('all', 'verify'):
        verify()
    prior.write_json(OUT/f'{args.phase}_provenance.json', {'reads': sorted(READS), 'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
