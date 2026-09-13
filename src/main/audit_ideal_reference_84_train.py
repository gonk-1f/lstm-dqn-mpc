"""Isolated Train-only audit; never imports a DQN agent or assigns failure rewards."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/main')]
from dqn.utils.action_mapper import MPCWeightAction
from mpc_solvers.formal_config import build_formal_mpc_config
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank
from mpc_solvers.mpc_qp_formulation import h2_quadratic_kg_step_coefficients

CONFIG = build_formal_mpc_config()
OUT = ROOT / 'outputs/ideal_reference_84_train_20260912'
SOURCE = ROOT / 'outputs/unified_objective_action_redesign_20260911/final_state_action_probe.csv'
TRAIN_ROOT = ROOT / 'data/processed/operating_dataset_final/train'
TERMS = ('H', 'B', 'S', 'F')
TERM_KEYS = ('h2_norm', 'battery_power_sq_norm', 'soc_reference_sq_norm', 'fc_variation_sq_norm')
ANCHORS = tuple(MPCWeightAction(84+i, *np.eye(4)[i], f'{TERMS[i]}_only') for i in range(4))
READS: set[str] = set()


def weight_grid():
    integers = [r for r in itertools.product(range(1, 8), repeat=4) if sum(r) == 10]
    return tuple(MPCWeightAction(i, *(v/10 for v in r), f'grid_{i:02d}') for i, r in enumerate(integers))


def reference_reward(values, matrix):
    values, matrix = np.asarray(values, dtype=float), np.asarray(matrix, dtype=float)
    ideal = matrix.min(axis=0)
    span = matrix.max(axis=0) - ideal
    if not np.isfinite(matrix).all() or np.any(span <= 0):
        raise ValueError('reference invalid or degenerate; no denominator epsilon applied')
    normalized = (values - ideal) / span
    return normalized, 1. / (1. + np.linalg.norm(normalized, axis=-1))


def pareto_mask(values, tolerance=0.):
    values = np.asarray(values, dtype=float)
    # [i,j] is true if i dominates j. Tolerance is explicit, never reward scaling.
    dominance = ((values[:, None, :] <= values[None, :, :] + tolerance).all(axis=2)
                 & (values[:, None, :] < values[None, :, :] - tolerance).any(axis=2))
    return ~dominance.any(axis=0)


def direct_terms(fc, load, soc0, prev):
    fc, load = np.asarray(fc, dtype=float), np.asarray(load, dtype=float)
    pb = load - fc
    soc = soc0 - np.cumsum(pb) / (3600. * CONFIG.battery_capacity_kwh)
    quad, linear, _, _ = h2_quadratic_kg_step_coefficients(CONFIG)
    h2ref = quad * 600.**2 + linear * 600.
    f = np.array([np.sum(quad*fc**2 + linear*fc)/h2ref,
                  np.sum((pb/624.)**2), np.sum(((soc-.55)/.05)**2),
                  np.sum((np.diff(np.r_[prev, fc])/48.)**2)])
    return f, soc, pb


def physical_feasibility(load, soc0, prev):
    """Independent zero-objective LP, variables [FC kW, stored battery energy kWh]."""
    load = np.asarray(load, dtype=float)
    n = len(load)
    eye = sparse.eye(n, format='csc')
    diff = eye - sparse.diags(np.ones(n-1), -1, shape=(n, n), format='csc')
    eq = sparse.hstack((-eye/3600., diff), format='csc')
    rhs = -load/3600.
    rhs[0] += soc0 * 624.
    ramp = sparse.hstack((diff, sparse.csc_matrix((n, n))), format='csc')
    limits_pos = np.full(n, 48.); limits_pos[0] += prev
    limits_neg = np.full(n, 48.); limits_neg[0] -= prev
    lb = np.maximum(0., load - 1248.)
    ub = np.minimum(600., load + 624.)
    if np.any(lb > ub):
        return {'classification': 'physical_infeasible', 'status': 2, 'message': 'instantaneous power bounds contradict'}
    bounds = list(zip(lb, ub)) + [(.20*624., .80*624.)] * n
    result = linprog(np.zeros(2*n), A_ub=sparse.vstack((ramp, -ramp)),
                     b_ub=np.r_[limits_pos, limits_neg], A_eq=eq, b_eq=rhs,
                     bounds=bounds, method='highs',
                     options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    label = 'feasible' if result.success else ('physical_infeasible' if result.status == 2 else 'unresolved')
    out = {'classification': label, 'status': int(result.status), 'message': str(result.message)}
    if result.success:
        out['witness_min_soc'] = float(result.x[n:].min()/624.)
    return out


def make_bank(actions, tight=False):
    bank = MpcWeightSolverBank(CONFIG, actions=actions)
    if tight:
        for entry in bank._entries.values():
            entry.solver.update_settings(eps_abs=1e-8, eps_rel=1e-8, max_iter=100000)
    return bank


def solve(bank, aid, load, soc, prev):
    start = time.perf_counter()
    record = {'action_id': aid, 'valid': False}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result, solve_ms = bank.solve(aid, load, soc, prev, .55)
        info = result.info
        record.update(status=str(info.status), solve_ms=float(solve_ms), iterations=int(info.iter),
                      primal_residual=float(info.prim_res), dual_residual=float(info.dual_res))
        if str(info.status).lower().startswith('solved') and result.x is not None:
            x = np.asarray(result.x)
            if x.shape == (25,) and np.isfinite(x).all():
                entry = bank._entries[aid]
                f = np.array([result.mpc_objective_terms[k] for k in TERM_KEYS])
                exact, sx, pb = direct_terms(x[:6], load, soc, prev)
                # Audit physical residuals separately from optimizer termination.
                balance = float(np.max(abs(x[:6]+x[6:12]-load)))
                soc_residual = float(np.max(abs(x[13:19]-sx)))
                fc_violation = max(0., float(-x[:6].min()), float(x[:6].max()-600.))
                batt_violation = max(0., float(-624.-x[6:12].min()), float(x[6:12].max()-1248.))
                soc_violation = max(0., float(.2-x[12:19].min()), float(x[12:19].max()-.8))
                ramp_violation = max(0., float(np.max(abs(np.diff(np.r_[prev, x[:6]])))-48.))
                valid = (balance <= .1 and soc_residual <= 1e-5 and fc_violation <= .1
                         and batt_violation <= .1 and soc_violation <= 1e-5 and ramp_violation <= .1)
                record.update(valid=bool(valid), f=f, exact_f=exact, fc=x[:6].copy(),
                              balance_kw=balance, soc_residual=soc_residual,
                              bound_violation_kw=max(fc_violation, batt_violation),
                              soc_violation=soc_violation, ramp_violation_kw=ramp_violation)
                if not valid:
                    record['status'] += '; physical residual check failed'
    except Exception as error:
        record['status'] = f'{type(error).__name__}: {error}'
    record['call_ms'] = (time.perf_counter()-start)*1000.
    return record


def classify_failures(results, load, soc, prev):
    failed = [r for r in results if not r['valid']]
    if not failed:
        return 'none'
    lp = physical_feasibility(load, soc, prev)
    if lp['classification'] == 'physical_infeasible':
        return 'physical_infeasible'
    return 'numerical_failure' if lp['classification'] == 'feasible' else 'unresolved'


def distribution(values):
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if not len(a):
        return {'count': 0}
    return {'count': int(len(a)), **dict(zip(('min', 'p01', 'p50', 'p99', 'max'),
            [float(x) for x in np.quantile(a, [0, .01, .5, .99, 1])])), 'mean': float(a.mean())}


def write_json(path, value):
    def default(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default, allow_nan=False)+'\n', encoding='utf-8')


def protected_hashes():
    paths = list((ROOT/'src').rglob('*.py')) + [ROOT/'README.md']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.resolve() != Path(__file__).resolve()}


def source_states():
    READS.add(str(SOURCE.relative_to(ROOT)))
    old = pd.read_csv(SOURCE).drop_duplicates('state_id')
    if len(old) != 840 or not old.segment_id.str.startswith('train_').all():
        raise RuntimeError('unexpected frozen Train state source')
    base_cols = ['segment_id', 'current_load_kw', 'previous_fc_kw', 'load_delta_kw']
    paired = old[base_cols].drop_duplicates().sort_values(base_cols).reset_index(drop=True)
    source = []
    for row in old.to_dict('records'):
        source.append({k: row[k] for k in ['state_id', 'soc', *base_cols]} | {'cohort': 'frozen840', 'pair_id': -1})
    for pair_id, row in enumerate(paired.to_dict('records')):
        for j, soc in enumerate((.55, .45, .35, .25, .22)):
            source.append(row | {'state_id': 10000+5*pair_id+j,
                                  'soc': soc, 'cohort': 'matched_soc', 'pair_id': pair_id})
    states = pd.DataFrame(source)
    states['audit_id'] = np.arange(len(states))
    return states


def load_train(segment):
    if not segment.startswith('train_') or '/' in segment or '\\' in segment:
        raise ValueError('only explicit Train filenames permitted')
    path = TRAIN_ROOT / (segment+'.csv')
    READS.add(str(path.relative_to(ROOT)))
    return pd.read_csv(path, usecols=['load_total_kw']).load_total_kw.to_numpy(dtype=float)


def state_audit(out):
    states = source_states()
    states.to_csv(out/'states.csv', index=False)
    actions = weight_grid()
    bank = make_bank(ANCHORS+actions)
    n = len(states)
    values = np.full((n, 88, 4), np.nan)
    exact = values.copy()
    plans = np.full((n, 88, 6), np.nan)
    valid = np.zeros((n, 88), dtype=bool)
    logs = []
    state_failure = []
    for row in states.itertuples():
        load = np.full(6, row.current_load_kw)
        results = []
        for j, action in enumerate(ANCHORS+actions):
            r = solve(bank, action.action_id, load, row.soc, row.previous_fc_kw)
            results.append(r)
            if 'f' in r:
                values[row.audit_id, j] = r['f']; exact[row.audit_id, j] = r['exact_f']
                plans[row.audit_id, j] = r['fc']
            valid[row.audit_id, j] = r['valid']
            logs.append({'audit_id': row.audit_id, 'phase': 'baseline',
                         **{k: v for k, v in r.items() if k not in ('f', 'exact_f', 'fc')}})
        state_failure.append(classify_failures(results, load, row.soc, row.previous_fc_kw))
        if (row.audit_id+1) % 100 == 0:
            print(f'STATE {row.audit_id+1}/{n}', flush=True)
    pd.DataFrame(logs).to_csv(out/'state_solve_log.csv', index=False)
    np.savez_compressed(out/'state_physical_results.npz', f=values, exact_f=exact,
                        fc_plans=plans, valid=valid)

    repetitions = {}
    repeat_valid = {}
    repeat_logs = []
    for phase, order, tight in [('reverse', states.iloc[::-1], False), ('tight', states, True)]:
        rb = make_bank(ANCHORS, tight=tight)
        vv = np.full((n, 4, 4), np.nan); ok = np.zeros((n, 4), dtype=bool)
        for count, row in enumerate(order.itertuples()):
            load = np.full(6, row.current_load_kw)
            for j, action in enumerate(ANCHORS):
                r = solve(rb, action.action_id, load, row.soc, row.previous_fc_kw)
                if 'f' in r: vv[row.audit_id, j] = r['f']
                ok[row.audit_id, j] = r['valid']
                repeat_logs.append({'audit_id': row.audit_id, 'phase': phase,
                                    **{k: v for k, v in r.items() if k not in ('f', 'exact_f', 'fc')}})
            if (count+1) % 200 == 0: print(f'ANCHORS {phase} {count+1}/{n}', flush=True)
        repetitions[phase] = vv; repeat_valid[phase] = ok
    pd.DataFrame(repeat_logs).to_csv(out/'anchor_repeat_solve_log.csv', index=False)
    np.savez_compressed(out/'anchor_repetitions.npz', **repetitions, **{k+'_valid': v for k, v in repeat_valid.items()})

    records, action_rows = [], []
    weights = np.array([a.as_tuple() for a in actions])
    all_reward = []
    for row in states.itertuples():
        i = row.audit_id
        rec = row._asdict() | {'failure_class': state_failure[i],
                               'anchors_valid': bool(valid[i, :4].all()),
                               'grid_valid_count': int(valid[i, 4:].sum())}
        m, v, vp = values[i, :4], values[i, 4:], plans[i, 4:]
        lo, hi = m.min(0), m.max(0); span = hi-lo
        rec.update({f'd_{t}': span[k] for k, t in enumerate(TERMS)})
        rec.update({f'ideal_{t}': lo[k] for k, t in enumerate(TERMS)})
        rec.update({f'reference_{t}': hi[k] for k, t in enumerate(TERMS)})
        rec['reference_valid'] = bool(valid[i, :4].all() and np.isfinite(span).all() and (span > 0).all())
        if rec['reference_valid'] and valid[i, 4:].all():
            hat, reward = reference_reward(v, m)
            order = np.argsort(-reward, kind='stable')
            best, second = order[:2]
            nd = pareto_mask(hat, 0.); ndtol = pareto_mask(hat, 1e-6)
            rms = np.sqrt(np.mean((vp[:, None, :]-vp[None, :, :])**2, axis=2))
            pairs = rms[np.triu_indices(84, 1)]
            rec.update(winner=int(best), winner_reward=float(reward[best]),
                       second_reward=float(reward[second]), reward_gap=float(reward[best]-reward[second]),
                       relative_gap=float((reward[best]-reward[second])/reward[best]),
                       winner_qS=float(weights[best, 2]), winner_fc=float(vp[best, 0]),
                       max_grid_fc=float(vp[:, 0].max()), min_grid_fc=float(vp[:, 0].min()),
                       S_anchor_fc=float(plans[i, 2, 0]), H_anchor_fc=float(plans[i, 0, 0]),
                       winner_batt=float(row.current_load_kw-vp[best, 0]),
                       pareto_count=int(nd.sum()), pareto_count_tol=int(ndtol.sum()),
                       pair_rms_lt_01kw=float((pairs < .1).mean()), pair_rms_lt_1kw=float((pairs < 1.).mean()),
                       negative_hat_count=int((hat < -1e-8).sum()), above_one_count=int((hat > 1+1e-8).sum()))
            extrema_error = np.zeros(4)
            reward_shift = 0.
            for phase in ('reverse', 'tight'):
                other = repetitions[phase][i]
                if repeat_valid[phase][i].all() and np.isfinite(other).all() and np.all(np.ptp(other, axis=0)>0):
                    _, rr = reference_reward(v, other)
                    shift = float(np.max(abs(rr-reward)))
                    reward_shift = max(reward_shift, shift)
                    extrema_error = np.maximum(extrema_error,
                        np.maximum(abs(other.min(0)-lo), abs(other.max(0)-hi)))
                    rec[phase+'_reward_shift'] = shift
                    rec[phase+'_winner_changed'] = bool(int(np.argmax(rr)) != best)
                    rec[phase+'_winner'] = int(np.argmax(rr))
            _, er = reference_reward(exact[i, 4:], m)
            model_error = np.max(abs(exact[i]-values[i]), axis=0)
            rec['model_consistent_reward_shift'] = float(np.max(abs(er-reward)))
            uncertainty = extrema_error + model_error
            # Empirical sensitivity indicator, NOT a rigorous mathematical error bound.
            rec.update({f'error_to_span_{t}': float(uncertainty[k]/span[k]) for k, t in enumerate(TERMS)})
            rec['span_within_10x_empirical_error'] = bool(np.any(span <= 10*uncertainty))
            rec['gap_within_2x_reward_sensitivity'] = bool(rec['reward_gap'] <= 2*max(reward_shift, rec['model_consistent_reward_shift']))
            rec['min_hat_S'] = float(hat[:, 2].min())
            rec['min_grid_S_regret'] = float(((v-lo)/span)[:, 2].min())
            for j, action in enumerate(actions):
                action_rows.append({'audit_id': i, 'cohort': row.cohort, 'pair_id': row.pair_id,
                     'soc': row.soc, 'action_id': j, **dict(zip(('qH','qB','qS','qF'), action.as_tuple())),
                     **dict(zip(TERMS, v[j])), **dict(zip(('hat_H','hat_B','hat_S','hat_F'), hat[j])),
                     'reward': reward[j], 'p_fc0': vp[j, 0],
                     'pareto': bool(nd[j]), 'pareto_tol': bool(ndtol[j])})
            all_reward.extend(reward.tolist())
        records.append(rec)
    details = pd.DataFrame(records)
    details.to_csv(out/'state_summary.csv', index=False)
    action_details = pd.DataFrame(action_rows)
    action_details.to_csv(out/'action_results.csv', index=False)
    groups = []
    for (cohort, soc), g in details.groupby(['cohort', 'soc']):
        successful = g.loc[g.reference_valid & g.grid_valid_count.eq(84)]
        groups.append({'cohort': cohort, 'soc': float(soc), 'states': len(g), 'valid': len(successful),
            'winner_reward_median': float(successful.winner_reward.median()),
            'winner_qS_median': float(successful.winner_qS.median()),
            'winner_qS_mean': float(successful.winner_qS.mean()),
            'winner_fc_mean': float(successful.winner_fc.mean()),
            'grid_max_fc_mean': float(successful.max_grid_fc.mean()),
            'winner_batt_mean': float(successful.winner_batt.mean())})
    pd.DataFrame(groups).to_csv(out/'soc_groups.csv', index=False)
    details.groupby(['cohort', 'soc', 'winner']).size().rename('count').reset_index().to_csv(out/'winner_by_soc.csv', index=False)
    action_details.groupby(['cohort', 'action_id']).agg(count=('reward','size'),
        reward_median=('reward','median'), pareto_share=('pareto','mean'), H_mean=('H','mean'),
        B_mean=('B','mean'), S_mean=('S','mean'), F_mean=('F','mean')).reset_index().to_csv(out/'action_summary.csv', index=False)
    summary = {'state_count': n, 'frozen_states': int((states.cohort=='frozen840').sum()),
        'matched_states': int((states.cohort=='matched_soc').sum()), 'matched_pairs': int(states.pair_id.max()+1),
        'primary_solves': n*88, 'repeat_anchor_solves': n*8,
        'failed_primary_solves': int((~valid).sum()),
        'failed_anchor_repeats': {k: int((~v).sum()) for k,v in repeat_valid.items()},
        'state_failure_classes': dict(Counter(state_failure)),
        'spans': {t: distribution(details['d_'+t]) for t in TERMS},
        'error_to_span': {t: distribution(details.get('error_to_span_'+t, [])) for t in TERMS},
        'successful_reward': distribution(all_reward),
        'reward_by_cohort': {k: distribution(g.reward) for k,g in action_details.groupby('cohort')},
        'reward_gaps': distribution(details.get('reward_gap', [])),
        'pareto_counts': distribution(details.get('pareto_count', [])),
        'pair_rms_lt_1kw': distribution(details.get('pair_rms_lt_1kw', [])),
        'near_numerical_gap_states': int(details.get('gap_within_2x_reward_sensitivity', pd.Series(dtype=bool)).sum()),
        'small_span_states': int(details.get('span_within_10x_empirical_error', pd.Series(dtype=bool)).sum()),
        'negative_hat_states': int((details.get('negative_hat_count', pd.Series(dtype=float)) > 0).sum()),
        'above_one_states': int((details.get('above_one_count', pd.Series(dtype=float)) > 0).sum()),
        'reverse_winner_changes': int(details.get('reverse_winner_changed', pd.Series(dtype=bool)).sum()),
        'tight_winner_changes': int(details.get('tight_winner_changed', pd.Series(dtype=bool)).sum()),
        'soc_groups': groups}
    write_json(out/'state_audit_summary.json', summary)
    return summary


# These match earlier Train-only windows, with predeclared SOC stress levels.
WINDOWS = (
    ('low_recharge', 'train_parent_037_02', 0, .22),
    ('ordinary_low_soc', 'train_parent_061_01', 1710, .22),
    ('fluctuating', 'train_parent_063_01', 5010, .25),
    ('high_feasible_reserve', 'train_parent_013_02', 2400, .25),
    ('high_thin_reserve', 'train_parent_013_02', 2400, .22),
    ('sustained_feasible_reserve', 'train_parent_021_01', 2820, .35),
    ('sustained_thin_reserve', 'train_parent_021_01', 2820, .22),
    ('rapid_rise', 'train_parent_021_01', 2310, .25),
)


def execution_error(fc, actual_load, soc, prev):
    batt = actual_load-fc
    next_soc = soc-batt/(3600.*624.)
    if not np.isfinite([fc, batt, next_soc]).all(): return 'nonfinite_execution'
    if fc < -.1 or fc > 600.1: return 'fc_bound_violation'
    if batt < -624.-1e-6 or batt > 1248.+1e-6: return 'battery_bound_violation'
    if next_soc < .2-1e-5 or next_soc > .8+1e-5: return 'soc_bound_violation'
    if abs(fc-prev) > 48.1: return 'ramp_bound_violation'
    return ''


def closed_loop_audit(out):
    actions = weight_grid()
    summaries, events = [], []
    for name, segment, offset, initial_soc in WINDOWS:
        loads = load_train(segment)[offset:offset+601]
        if len(loads) != 601: raise ValueError('Train window must provide 601 measured points')
        initial_fc = float(np.clip(loads[0], 0., 600.))
        # Uses actual future only to establish an ex-post physical feasibility baseline.
        full_lp = physical_feasibility(loads[1:], initial_soc, initial_fc)
        for policy in ('greedy_reference_reward', 'max_first_fc_grid_diagnostic'):
            bank = make_bank(ANCHORS+actions)
            soc, prev = initial_soc, initial_fc
            trace, failure, failure_class = [], '', 'none'
            failure_index = None
            solve_count, seconds = 0, time.perf_counter()
            for t in range(600):
                predicted = np.full(6, loads[t])
                all_results = [solve(bank, a.action_id, predicted, soc, prev) for a in ANCHORS+actions]
                solve_count += len(all_results)
                invalid = [r for r in all_results if not r['valid']]
                if invalid:
                    failure_class = classify_failures(all_results, predicted, soc, prev)
                    failure = 'anchor_failure' if any(r['action_id']>=84 for r in invalid) else 'grid_solve_failure'
                    events.append({'window': name, 'policy': policy, 'step': t, 'soc': soc,
                        'load': float(loads[t]), 'prev_fc': prev, 'failure_class': failure_class,
                        'solver_failures': [{k:v for k,v in r.items() if k not in ('f','fc','exact_f')} for r in invalid]})
                    failure_index = t
                    break
                m = np.array([r['f'] for r in all_results[:4]])
                f = np.array([r['f'] for r in all_results[4:]])
                try:
                    hat, rewards = reference_reward(f, m)
                except ValueError:
                    failure, failure_class, failure_index = 'degenerate_reference', 'evaluation_failure', t
                    break
                fc_all = np.array([r['fc'][0] for r in all_results[4:]])
                chosen = int(np.argmax(rewards if policy=='greedy_reference_reward' else fc_all))
                fc = float(fc_all[chosen]); batt = float(loads[t+1]-fc)
                err = execution_error(fc, loads[t+1], soc, prev)
                if err:
                    failure, failure_index = err, t
                    next_lp = physical_feasibility(np.array([loads[t+1]]), soc, prev)
                    failure_class = ('physical_infeasible' if next_lp['classification']=='physical_infeasible'
                                     else 'selected_execution_violation')
                    events.append({'window': name, 'policy': policy, 'step': t, 'soc': soc,
                                   'failure_class': failure_class, 'execution_error': err,
                                   'one_step_feasibility': next_lp})
                    break
                nxt = soc-batt/(3600.*624.)
                trace.append({'step': t, 'soc_before': soc, 'soc_after': nxt,
                    'load_current': loads[t], 'load_actual': loads[t+1], 'previous_fc': prev,
                    'action_id': chosen, 'qS': actions[chosen].q_soc, 'fc': fc, 'batt': batt,
                    'reward': rewards[chosen], 'winner_reward': float(rewards.max()),
                    'max_grid_fc': fc_all.max(), 'S_anchor_fc': all_results[2]['fc'][0],
                    'fc_delta': abs(fc-prev), 'd_S': float(np.ptp(m[:, 2])),
                    'five_call_ms': float(sum(r['call_ms'] for r in all_results[:4])+all_results[4+chosen]['call_ms']),
                    **dict(zip(TERMS, f[chosen]))})
                soc, prev = nxt, fc
                if (t+1)%100 == 0:
                    print(f'ROLLOUT {name} {policy} {t+1}/600 soc={soc:.6f}', flush=True)
            frame = pd.DataFrame(trace)
            frame.to_csv(out/f'trace_{name}_{policy}.csv', index=False)
            rec = {'window': name, 'segment': segment, 'offset': offset, 'policy': policy,
                'initial_soc': initial_soc, 'initial_fc': initial_fc, 'mean_load': float(loads[1:].mean()),
                'full_window_physical_feasibility': full_lp['classification'],
                'steps': len(trace), 'completed': len(trace)==600, 'final_soc': soc,
                'min_soc': min(initial_soc, float(frame.soc_after.min())) if len(frame) else initial_soc,
                'failure': failure, 'failure_class': failure_class, 'failure_step': failure_index,
                'solves': solve_count, 'elapsed_seconds': time.perf_counter()-seconds}
            if len(frame):
                quad, linear, _, _ = h2_quadratic_kg_step_coefficients(CONFIG)
                rec.update(mean_fc=float(frame.fc.mean()), mean_qS=float(frame.qS.mean()),
                    mean_reward=float(frame.reward.mean()), total_h2_kg=float((quad*frame.fc**2+linear*frame.fc).sum()),
                    battery_net_kwh=float(frame.batt.sum()/3600.), battery_throughput_kwh=float(frame.batt.abs().sum()/3600.),
                    fc_tv=float(frame.fc_delta.sum()), five_call_ms=distribution(frame.five_call_ms))
            if failure:
                rec['remaining_actual_load_feasibility'] = physical_feasibility(loads[len(trace)+1:], soc, prev)['classification']
            summaries.append(rec)
            write_json(out/'closed_loop_summary.json', summaries)
            write_json(out/'closed_loop_failures.json', events)
            pd.DataFrame([{k:v for k,v in r.items() if k!='five_call_ms'} for r in summaries]).to_csv(out/'closed_loop_summary.csv', index=False)
            print(f'ROLLOUT_DONE {name} {policy} steps={len(trace)} {failure_class}', flush=True)
    return summaries


def five_solve_benchmark(out):
    chosen = next(a for a in weight_grid() if a.as_tuple()==(.4,.2,.2,.2))
    states = pd.concat([g.iloc[np.linspace(0, len(g)-1, 12, dtype=int)]
                        for _, g in source_states().groupby(['cohort', 'soc'])])
    records = []
    for phase, tight in (('default', False), ('tight_anchors', True)):
        bank = make_bank(ANCHORS+(chosen,))
        if tight:
            for a in ANCHORS:
                bank._entries[a.action_id].solver.update_settings(eps_abs=1e-8, eps_rel=1e-8, max_iter=100000)
        for row in states.itertuples():
            start = time.perf_counter()
            load = np.full(6, row.current_load_kw)
            res = [solve(bank, a.action_id, load, row.soc, row.previous_fc_kw) for a in ANCHORS+(chosen,)]
            elapsed = (time.perf_counter()-start)*1000.
            records.append({'audit_id': row.audit_id, 'phase': phase, 'five_wall_ms': elapsed,
                'ordinary_call_ms': res[-1]['call_ms'], 'anchor_call_ms': sum(r['call_ms'] for r in res[:4]),
                'failure_class': classify_failures(res, load, row.soc, row.previous_fc_kw),
                'failed_anchors': sum(not r['valid'] for r in res[:4]),
                'failed_ordinary': not res[-1]['valid'],
                'total_iterations': sum(r.get('iterations',0) for r in res)})
    details = pd.DataFrame(records)
    details.to_csv(out/'five_solve_benchmark.csv', index=False)
    summary = {phase: {'five_wall_ms': distribution(g.five_wall_ms), 'ordinary_call_ms': distribution(g.ordinary_call_ms),
               'failed_anchor_solves': int(g.failed_anchors.sum()), 'failed_ordinary_solves': int(g.failed_ordinary.sum()),
               'failure_classes': g.failure_class.value_counts().to_dict()} for phase,g in details.groupby('phase')}
    write_json(out/'five_solve_benchmark.json', summary)
    return summary


def analytical_soc_check(out):
    """Known exact S-only solution at SOC=.55 when constant load is ramp-feasible.

    This substitutes an analytically certified row ONLY for diagnostic scoring.
    It does not modify anchors used in the primary audit or any formal reward.
    """
    states = pd.read_csv(out/'states.csv')
    data = np.load(out/'state_physical_results.npz')
    rows = []
    for row in states.itertuples():
        load, prev = row.current_load_kw, row.previous_fc_kw
        if not (row.soc==.55 and 0<=load<=600 and abs(load-prev)<=48): continue
        i = row.audit_id
        # Constant FC=load implies zero battery power and identically SOC=.55.
        exact, _, _ = direct_terms(np.full(6,load), np.full(6,load), row.soc, prev)
        assert exact[2] == 0.
        m = data['f'][i,:4].copy()
        rec = {'audit_id':i, 'soc':row.soc, 'load':load, 'previous_fc':prev,
               'S_anchor_valid':bool(data['valid'][i,2]),
               'exact_S':0., 'computed_S':float(m[2,2]),
               'exact_H':float(exact[0]), 'computed_H':float(m[2,0]),
               'exact_F':float(exact[3]), 'computed_F':float(m[2,3])}
        if data['valid'][i,:].all():
            v = data['f'][i,4:]
            _, r = reference_reward(v,m)
            m[2] = exact
            _, rr = reference_reward(v,m)
            rec.update(max_reward_shift=float(np.max(abs(rr-r))),
                       winner_changed=bool(np.argmax(r)!=np.argmax(rr)),
                       baseline_winner=int(np.argmax(r)), exact_S_winner=int(np.argmax(rr)))
        rows.append(rec)
    d = pd.DataFrame(rows)
    d.to_csv(out/'analytical_S_anchor_check.csv', index=False)
    summary = {'eligible_states':len(d), 'baseline_S_failures':int((~d.S_anchor_valid).sum()),
               'scoring_comparisons':int(d.max_reward_shift.notna().sum()),
               'max_reward_shift':distribution(d.max_reward_shift),
               'winner_changes':int(d.winner_changed.eq(True).sum()),
               'analytic_fact':'SOC=.55 and |load-prev_fc|<=48 allow constant FC=load, so S*=0 and B*=0.'}
    write_json(out/'analytical_S_anchor_summary.json',summary)
    return summary


def control_only_low_load_check(out):
    """Isolate grid control ability from the already recorded H-anchor failure."""
    actions = weight_grid(); bank = make_bank(actions)
    loads = load_train('train_parent_037_02')[:601]
    soc, prev = .22, float(loads[0])
    records=[]; failure=''; failures=[]
    for t in range(600):
        load=np.full(6,loads[t])
        rs=[solve(bank,a.action_id,load,soc,prev) for a in actions]
        bad=[r for r in rs if not r['valid']]
        if bad:
            failure=classify_failures(rs,load,soc,prev)
            failures=[{k:v for k,v in r.items() if k not in ('f','fc','exact_f')} for r in bad]
            break
        j=int(np.argmax([r['fc'][0] for r in rs]));fc=float(rs[j]['fc'][0])
        failure=execution_error(fc,loads[t+1],soc,prev)
        if failure: break
        nxt=soc-(loads[t+1]-fc)/(3600.*624.)
        records.append({'step':t,'soc_before':soc,'soc_after':nxt,'fc':fc,
                        'load_actual':loads[t+1],'action_id':j,'qS':actions[j].q_soc})
        soc,prev=nxt,fc
        if (t+1)%200==0: print('CONTROL_ONLY_LOW_LOAD',t+1,flush=True)
    pd.DataFrame(records).to_csv(out/'control_only_low_load_trace.csv',index=False)
    summary={'steps':len(records),'completed':len(records)==600,'initial_soc':.22,
        'final_soc':soc,'failure':failure,'failed_solves':failures,'reward_computed':False,
        'interpretation':'Grid-only maximum first FC diagnostic; isolates control capacity from anchor failure, not an RL controller.'}
    write_json(out/'control_only_low_load_summary.json',summary)
    return summary


def finalize_audit(out):
    s=pd.read_csv(out/'state_summary.csv')
    actions=pd.read_csv(out/'action_results.csv')
    log=pd.read_csv(out/'state_solve_log.csv')
    repeat=pd.read_csv(out/'anchor_repeat_solve_log.csv')
    closed=pd.read_csv(out/'closed_loop_summary.csv')
    assert len(s)==1440 and len(log)==126720 and len(repeat)==11520
    assert len(weight_grid())==84
    counts=actions.groupby('audit_id').size()
    expected_valid_states=int((s.reference_valid & s.grid_valid_count.eq(84)).sum())
    assert counts.eq(84).all() and len(counts)==expected_valid_states
    assert len(actions)==84*expected_valid_states and np.isfinite(actions[['H','B','S','F','reward']]).all().all()
    with np.load(out/'state_physical_results.npz') as ds:
        saved_f=ds['f']
    max_recompute_error=0.
    for aid,g in actions.groupby('audit_id'):
        _,r=reference_reward(saved_f[aid,4:],saved_f[aid,:4])
        max_recompute_error=max(max_recompute_error,float(np.max(abs(r-g.sort_values('action_id').reward.to_numpy()))))
    assert max_recompute_error<1e-12
    p=s[s.cohort.eq('matched_soc')].pivot(index='pair_id',columns='soc',values=['winner_reward','winner_qS','winner_fc','winner_batt'])
    summary={'paired_groups':len(p),'valid_reward_count':len(actions),
        'primary_grid_failure_count':int(((~log.valid)&log.action_id.lt(84)).sum()),
        'low_SOC_reward_increases_045_to022':int((p.winner_reward[.22]>p.winner_reward[.45]).sum()),
        'low_SOC_qS_increases_045_to022':int((p.winner_qS[.22]>p.winner_qS[.45]).sum()),
        'low_SOC_qS_decreases_045_to022':int((p.winner_qS[.22]<p.winner_qS[.45]).sum()),
        'median_fc_increase_045_to022':float((p.winner_fc[.22]-p.winner_fc[.45]).median()),
        'winner_count':int(s.winner.nunique()),
        'reverse_comparisons':int(s.reverse_reward_shift.notna().sum()),
        'tight_comparisons':int(s.tight_reward_shift.notna().sum()),
        'max_reverse_reward_shift':float(s.reverse_reward_shift.max()),
        'max_tight_reward_shift':float(s.tight_reward_shift.max()),
        'closed_loop_total_solves':int(closed.solves.sum()),
        'max_saved_reward_recompute_error':max_recompute_error,
        'formal_source_hashes_unchanged':protected_hashes()==json.loads((out/'protected_source_before.json').read_text(encoding='utf-8')),
        'no_failure_penalty_assigned':True}
    assert summary['formal_source_hashes_unchanged']
    write_json(out/'verification_summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all','states','closed-loop','benchmark','diagnostic','verify'], default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_path = OUT/'protected_source_before.json'
    if not baseline_path.exists(): write_json(baseline_path, protected_hashes())
    before = json.loads(baseline_path.read_text(encoding='utf-8'))
    if protected_hashes() != before: raise RuntimeError('pre-existing formal sources changed since audit began')
    write_json(OUT/'actions.json', [asdict(a) for a in weight_grid()+ANCHORS])
    if args.phase in ('all','states'): state_audit(OUT)
    if args.phase in ('all','closed-loop'): closed_loop_audit(OUT)
    if args.phase in ('all','benchmark'): five_solve_benchmark(OUT)
    if args.phase in ('all','diagnostic'):
        analytical_soc_check(OUT)
        control_only_low_load_check(OUT)
    if args.phase in ('all','verify'): finalize_audit(OUT)
    if protected_hashes() != before: raise RuntimeError('formal sources changed during audit')
    write_json(OUT/f'{args.phase}_provenance.json', {'data_reads': sorted(READS),
        'formal_source_hashes_unchanged': True, 'action_count': 84, 'failure_reward': None,
        'formal_config': asdict(CONFIG), 'phase': args.phase})
    print('AUDIT_PHASE_DONE', args.phase, flush=True)


if __name__ == '__main__':
    main()
