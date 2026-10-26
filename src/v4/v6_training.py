"""The single frozen V6 DQN run, using the existing physical executor and ledger."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from v3.control import AccountState, EconomicMPC
from v2.economics import HYDROGEN_PRICE_CNY_PER_KG
from .actions import candidate_fc_actions
from .control import ReplayExecutionError, feasible_fc_actions, replay_episode
from .dqn import DirectPowerDQN
from .failure_replay import FailurePenalty, prepare_failure_replay
from .independent_economics import ZhouEvaluationConfig
from .reward_feedback import BatteryEnergyValue
from .v6_dataset import V6TrainingDataset
from .v6_q_probes import build_probe_states, write_probe_snapshot
from .v6_td_diagnostics import checkpoint_replay_fit
from .v6_train_values import TrainFailureValues


CONFIG = dict(algorithm='DQN', network=[5, 256, 256, 13], seed=42,
    state_mode='history5', action_mode='increment13', sample_seconds=1,
    actions_kw=list(range(-60, 61, 10)), action_semantics='actual increment per 1-second decision',
    gamma=1., n_step=1, learning_rate=.0001, adam_betas=[.9, .999], adam_eps=1e-8,
    loss='SmoothL1', huber_beta=1., gradient_clip_norm=10., reward_scale=1.,
    soc_reference=.5, soc_denominator=.1, soc_weight=.2, fc_ramp_weight=.05,
    fc_ramp_denominator=600., replay_sampling='uniform', failure_terminal_quota=0,
    replay_capacity=1_000_000, batch_size=64, warmup_transitions=50_000,
    train_freq=1, gradient_steps=1, target_update='hard', target_interval=1000,
    rounds=8, epsilon_start=1., epsilon_end=.05, epsilon_decay_rounds=6,
    physical_ramp_status='simulation assumption; not vessel-calibrated',
    candidate_selection='Validation completed count, then duration coverage; full-set economic cost when all complete')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if rows:
        with Path(path).open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def epsilon_at(round_index, episode_index):
    progress = round_index - 1 + episode_index / 35.
    return 1. - .95 * min(progress / 6., 1.)


def sample_record(episode, values, status):
    durations = [t.duration_s for t in values]
    duration = math.fsum(durations)
    deltas = [t.action_delta_kw for t in values]
    changes = sum(d != 0 for d in deltas)
    reversals = sum(a * b < 0 for a, b in zip(deltas, deltas[1:]))
    soc = [.6, *(t.actual_soc for t in values)]
    parts = dict(
        fc_h2_cost_cny=math.fsum(t.executed_ledger.h2_cost_cny for t in values),
        battery_h2eq_cost_cny=math.fsum(t.soc_equivalent_hydrogen_cost_cny for t in values),
        fc_degradation_cost_cny=math.fsum(t.executed_ledger.fuel_cell_degradation_cost_cny for t in values),
        battery_degradation_cost_cny=math.fsum(t.executed_ledger.battery_degradation_cost_cny for t in values),
        shore_electricity_cost_cny=math.fsum(t.executed_ledger.shore_cost_cny for t in values))
    economic = math.fsum(parts.values())
    boundary = ('upper' if soc[-1] >= .79 else 'lower' if soc[-1] <= .21 else 'interior')
    return dict(sample_id=episode.sample_id, status=status, completed=status == 'completed',
        executed_steps=len(values), executed_duration_s=duration, voyage_duration_s=episode.duration_s,
        duration_coverage=duration / episode.duration_s, soc_min=min(soc), soc_max=max(soc),
        soc_terminal=soc[-1], fc_nonzero_changes=changes, fc_adjacent_reversals=reversals,
        previous_fc_terminal_kw=values[-1].action_kw if values else 0,
        failure_boundary_side=boundary if status != 'completed' else None,
        fc_changes_per_hour=changes * 3600. / duration if duration else 0.,
        fc_reversals_per_hour=reversals * 3600. / duration if duration else 0.,
        fc_mean_absolute_ramp_kw_per_s=math.fsum(abs(d) for d in deltas) / duration if duration else 0.,
        fc_starts=sum(t.state[2] == 0. and t.action_kw > 0 for t in values),
        **parts, fc_h2_kg=parts['fc_h2_cost_cny'] / HYDROGEN_PRICE_CNY_PER_KG,
        ledger_total_cost_cny=economic - parts['battery_h2eq_cost_cny'],
        executed_prefix_economic_cost_cny=economic,
        complete_economic_cost_cny=economic if status == 'completed' else None,
        soc_penalty_reward_units=math.fsum(t.soc_soft_penalty_cny for t in values),
        fc_ramp_penalty_reward_units=math.fsum(t.fc_ramp_penalty_cny for t in values),
        failure_penalty_reward_units=math.fsum(t.failure_penalty_equivalent_cny for t in values),
        training_return=math.fsum(t.reward_cny for t in values))


def split_summary(rows):
    reasons = {reason: sum(r['status'] == reason for r in rows)
               for reason in ('soc_limited', 'structural_power', 'increment_action_limited')}
    all_complete = all(r['completed'] for r in rows)
    duration = math.fsum(r['executed_duration_s'] for r in rows)
    return dict(episodes=len(rows), completed=sum(r['completed'] for r in rows),
        failure_reasons=reasons, environment_steps=sum(r['executed_steps'] for r in rows),
        failure_boundary_counts={side: sum(r['failure_boundary_side'] == side for r in rows)
                                 for side in ('upper', 'lower', 'interior')},
        executed_duration_s=duration, mean_duration_coverage=math.fsum(r['duration_coverage'] for r in rows)/len(rows),
        fixed_complete_set_cost_cny=math.fsum(r['complete_economic_cost_cny'] for r in rows) if all_complete else None,
        executed_prefix_cost_cny=math.fsum(r['executed_prefix_economic_cost_cny'] for r in rows),
        fc_changes_per_hour=sum(r['fc_nonzero_changes'] for r in rows)*3600./duration if duration else 0.,
        fc_reversals_per_hour=sum(r['fc_adjacent_reversals'] for r in rows)*3600./duration if duration else 0.,
        fc_mean_absolute_ramp_kw_per_s=sum(r['fc_mean_absolute_ramp_kw_per_s'] * r['executed_duration_s']
                                          for r in rows) / duration if duration else 0.,
        **{key: math.fsum(r[key] for r in rows) for key in (
            'fc_h2_kg', 'fc_h2_cost_cny', 'battery_h2eq_cost_cny', 'fc_degradation_cost_cny',
            'battery_degradation_cost_cny', 'shore_electricity_cost_cny', 'ledger_total_cost_cny')},
        soc_min=min(r['soc_min'] for r in rows), soc_max=max(r['soc_max'] for r in rows))


class WarmupComplete(BaseException):
    pass


def execute(episode, policy, accountant, economics, penalty, callback=None):
    try:
        result = replay_episode(episode, policy, accountant=accountant,
            state_mode='history5', action_mode='increment13', reward_cost_mode='v6_energy_soc_ramp',
            v6_soc_weight=CONFIG['soc_weight'], v6_fc_ramp_weight=CONFIG['fc_ramp_weight'],
            economic_evaluation_config=economics, on_transition=callback)
        return result.transitions, 'completed'
    except ReplayExecutionError as error:
        if error.failure_kind != 'no_feasible_action':
            raise
        view = prepare_failure_replay(error, penalty=penalty,
            energy_value=BatteryEnergyValue.from_accountant(accountant), redistribute_battery_energy=False)
        if callback is not None and view.transitions:
            callback(len(view.transitions)-1, view.transitions[-1])
        return view.transitions, error.failure_cause


def fixed_probes(source, validation, agent, accountant, round_index):
    episodes = {e.sample_id: e for e in validation}
    rows = []
    for probe in source:
        episode = episodes[probe['sample_id']]
        instant = (int(probe['step_1based'])-1)*30
        index = int(np.searchsorted(episode.time_s, instant))
        loads = episode.load_kw
        state = (float(probe['state'][0]), float(loads[index])/600., float(probe['state'][2]),
            float(loads[index]-loads[max(0,index-30)])/600.,
            float(np.mean(loads[max(0,index-300):index+1]))/600.)
        physical = AccountState(soc=state[0], previous_fc_kw=state[2]*600.)
        legal = candidate_fc_actions(feasible_fc_actions(physical, float(loads[index]), accountant, duration_s=1.),
                                     physical.previous_fc_kw, 'increment13')
        indices = [agent.action_index_for_power(power, state) for power in legal]
        with torch.inference_mode():
            q = agent.online(torch.tensor([state], dtype=torch.float32, device=agent.device))[0].cpu().tolist()
        chosen = max(indices, key=lambda i: q[i])
        rows.append(dict(round=round_index, sample_id=episode.sample_id, time_s=instant,
            state=list(state), hold_rank=1+sum(q[i] > q[6] for i in indices),
            greedy_delta_kw=agent.action_values_kw[chosen], q_hold=q[6],
            greedy_minus_hold=q[chosen]-q[6], original_action_minus_hold=q[agent.action_values_kw.index(probe['chosen_delta_kw'])]-q[6],
            q_values=q, legal_indices=indices, source='V5 physical SOC/FC/time with V6 causal history; Q units differ from V5'))
    return rows


def plot_profile(path, sample_id, values):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = [t.relative_time_s/3600. for t in values]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(times, [t.raw_load_kw for t in values], label='Load', lw=.7)
    axes[0].plot(times, [t.action_kw for t in values], label='FC', lw=.7)
    axes[0].plot(times, [t.actual_battery_kw for t in values], label='Battery', lw=.7)
    axes[0].set_ylabel('Power (kW)'); axes[0].legend(ncol=3)
    axes[1].plot(times, [t.actual_soc for t in values], lw=.8)
    axes[1].axhline(.2, color='r', ls='--', lw=.6); axes[1].axhline(.8, color='r', ls='--', lw=.6)
    axes[1].set_ylabel('SOC'); axes[1].set_ylim(.18,.82)
    axes[2].plot(times, [t.action_delta_kw for t in values], lw=.6)
    axes[2].set_ylabel('FC increment (kW)'); axes[2].set_xlabel('Elapsed time (h)')
    fig.suptitle(sample_id); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def evaluate(validation, agent, accountant, economics, penalty, output, round_index):
    rows = []
    folder = output / f'round_{round_index:02d}_validation'
    folder.mkdir()
    for episode in validation:
        values, status = execute(episode, lambda s,a: agent.select_power(s,a,epsilon=0.),
                                 accountant, economics, penalty)
        rows.append(sample_record(episode, values, status))
        profile = []
        cumulative = dict(fc_h2=0., battery_h2eq=0., soc_penalty=0., fc_ramp=0., failure_penalty=0., reward=0.)
        for start in range(0, len(values), 2048):
            batch = values[start:start+2048]
            with torch.inference_mode():
                q_batch = agent.online(torch.tensor([t.state for t in batch], dtype=torch.float32,
                                                    device=agent.device)).cpu().numpy()
            if not np.isfinite(q_batch).all():
                raise FloatingPointError('non-finite Validation Q')
            for t, q in zip(batch, q_batch):
                legal = [agent.action_index_for_power(p, t.state) for p in t.policy_candidate_actions]
                ranked_legal = sorted(legal, key=lambda i: float(q[i]), reverse=True)
                components = dict(fc_h2=t.executed_ledger.h2_cost_cny,
                    battery_h2eq=t.soc_equivalent_hydrogen_cost_cny, soc_penalty=t.soc_soft_penalty_cny,
                    fc_ramp=t.fc_ramp_penalty_cny, failure_penalty=t.failure_penalty_equivalent_cny, reward=t.reward_cny)
                for key, value in components.items(): cumulative[key] += value
                profile.append(dict(time_s=t.relative_time_s, load_kw=t.raw_load_kw, fc_kw=t.action_kw,
                    battery_kw=t.actual_battery_kw, soc_before=t.state[0], soc=t.actual_soc,
                    fc_delta_kw=t.action_delta_kw, done=t.done, terminal_reason=t.terminal_reason,
                    hold_q=float(q[6]), hold_legal=6 in legal,
                    hold_rank=1+sum(q[i] > q[6] for i in legal) if 6 in legal else None,
                    legal_indices=json.dumps(legal),
                    legal_action_ranking_delta_kw=json.dumps([agent.action_values_kw[i] for i in ranked_legal]),
                    best_minus_second_q=(float(q[ranked_legal[0]]-q[ranked_legal[1]])
                                         if len(ranked_legal)>1 else None),
                    selected_q=float(q[t.action_index]), selected_minus_hold=float(q[t.action_index]-q[6]),
                    fc_degradation_cost_cny=t.executed_ledger.fuel_cell_degradation_cost_cny,
                    battery_degradation_cost_cny=t.executed_ledger.battery_degradation_cost_cny,
                    **components, **{f'cumulative_{key}': value for key,value in cumulative.items()},
                    **{f'q_delta_{delta}':float(q[i]) for i,delta in enumerate(agent.action_values_kw)}))
        if profile:
            with gzip.open(folder/f'{episode.sample_id}.csv.gz', 'wt', encoding='utf-8', newline='') as handle:
                writer=csv.DictWriter(handle, fieldnames=list(profile[0])); writer.writeheader(); writer.writerows(profile)
            plot_profile(folder/f'{episode.sample_id}.png', episode.sample_id, values)
        print(f'validation round={round_index} sample={episode.sample_id} status={status} steps={len(values)}', flush=True)
    write_csv(folder/'sample_metrics.csv', rows)
    return split_summary(rows), rows


def model_snapshot(agent, round_index):
    return dict(configuration=CONFIG, completed_round=round_index, online=agent.online.state_dict(),
        target=agent.target.state_dict(), optimizer=agent.optimizer.state_dict(),
        adam_updates=agent.economic_optimizer_updates, target_copies=agent.target_sync_calls)


def compare_v5(old_path, rows, output):
    with old_path.open(encoding='utf-8-sig', newline='') as handle:
        old = {r['sample_id']:r for r in csv.DictReader(handle) if r['split']=='validation' and r['round']=='32'}
    comparison=[]
    for row in rows:
        prior=old[row['sample_id']]; duration=float(prior['executed_duration_s'])
        common=row['completed'] and prior['completed']=='True'
        comparison.append(dict(sample_id=row['sample_id'],v5_completed=prior['completed']=='True',
            v6_completed=row['completed'],v5_changes_per_hour=float(prior['fc_nonzero_changes'])*3600./duration,
            v6_changes_per_hour=row['fc_changes_per_hour'],
            v5_reversals_per_hour=float(prior['fc_adjacent_reversals'])*3600./duration,
            v6_reversals_per_hour=row['fc_reversals_per_hour'],
            v5_soc_terminal=float(prior['soc_terminal']),v6_soc_terminal=row['soc_terminal'],
            same_complete_sample=common,
            v5_complete_cost_cny=float(prior['complete_zhou_cost_cny']) if common else None,
            v6_complete_cost_cny=row['complete_economic_cost_cny'] if common else None))
    write_csv(output/'v5_v6_validation_comparison.csv',comparison)


def run(data_root, output, v5_root, *, resume=None, stop_after_round=None):
    saved=None
    if resume is not None:
        resume=Path(resume).resolve()
        if resume.parent.parent != output.resolve():
            raise ValueError('--resume must use its original --output-dir')
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        configuration=saved['report']['configuration']
        if {key:configuration[key] for key in CONFIG} != CONFIG:
            raise ValueError('checkpoint configuration differs from the current frozen V6 configuration')
        if any(group['lr'] != CONFIG['learning_rate']
               for group in saved['agent']['optimizer']['param_groups']):
            raise ValueError('checkpoint Adam learning rate differs from the current V6 configuration')
        if saved['report']['completed_rounds'] >= CONFIG['rounds']:
            return saved['report']
        if stop_after_round is not None and stop_after_round <= saved['report']['completed_rounds']:
            raise ValueError('--stop-after-round must be later than the completed checkpoint round')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'models').mkdir()
    torch.set_num_threads(1)
    dataset=V6TrainingDataset.open(data_root)
    train, validation=dataset.load_train(), dataset.load_validation()
    agent=DirectPowerDQN(seed=42, replay_capacity=1_000_000, device='cpu',
                         learning_rate=CONFIG['learning_rate'], td_diagnostics=True)
    accountant=EconomicMPC(nominal_cost_cny=1.)
    economics=ZhouEvaluationConfig.from_formal_curve()
    penalty=FailurePenalty.from_reference()
    source=json.loads((v5_root/'outputs/v5_qfit_interference_20261023/fixed_probe_states.json').read_text(encoding='utf-8'))['probes']
    train_values=TrainFailureValues(output)
    if saved is not None:
        report=saved['report']
        agent.load_training_state(saved['agent'])
        probes=json.loads((output/'q_probes/definitions.json').read_text(encoding='utf-8'))
        bank=output/'train_failure_fixed_states.json'
        if bank.exists():
            train_values.fixed_events=json.loads(bank.read_text(encoding='utf-8'))
        if 'elapsed_seconds' in report:
            elapsed_before=report['elapsed_seconds']
        else:
            elapsed_before=json.loads((output/'progress.json').read_text(encoding='utf-8'))['elapsed_seconds']
            report['elapsed_resume_basis']='last recorded progress; legacy checkpoint has no elapsed_seconds'
        report.update(status='continued',resumed_from_checkpoint=str(resume))
    else:
        probes=build_probe_states(source, validation,
            output.parent/'v6_dqn_1s_seed42_20261024_run01', accountant)
        report=dict(configuration={**CONFIG,'failure_penalty':penalty.amount},dataset=dataset.task_contract,
            status='warmup',completed_rounds=0,test_payloads_opened=0,rounds=[],fixed_probes=[],
            training_environment_steps=0,warmup_environment_steps=0,validation_environment_steps=0)
        report['q_reporting_contract']=dict(
            source='Only Train/warmup transitions enter Replay and Adam. Validation states are inference-only.',
            residual='one-step TD target minus Q_online(s, executed_action)',
            groups=['normal_nonterminal','normal_terminal','failure_terminal','failure_prefix'],
            boundary_classification='Failure side: upper SOC>=0.79, lower SOC<=0.21, otherwise interior; reporting only.',
            interpretation='TD targets move with the target network and are not Q*. '
                           'Observed failure-prefix returns are behavior-policy returns, not optimal values.',
            online_p95='8192 independent reservoir per group; an estimate after more than 8192 actual draws.',
            checkpoint_p95='Exact on saved stratified diagnostic samples, not necessarily full Replay.')
        write_json(output/'configuration.json',report['configuration'])
        report['fixed_probes'].extend(fixed_probes(source,validation,agent,accountant,0))
        write_probe_snapshot(agent,probes,0,output)
        write_json(output/'report.json',report)
        elapsed_before=0.
    started=time.monotonic()-elapsed_before

    def save_progress(round_index):
        progress=dict(status=report['status'],round=round_index,
            training_environment_steps=report['training_environment_steps'],
            warmup_environment_steps=report['warmup_environment_steps'],
            adam_updates=agent.economic_optimizer_updates,target_copies=agent.target_sync_calls,
            elapsed_seconds=time.monotonic()-started)
        write_json(output/'progress.json',progress)
        return progress

    def remember(t, learning):
        agent.remember(t.state,t.action_kw,t.reward_cny,t.next_state,done=t.done,
            next_feasible_actions=t.next_feasible_actions,terminal_reason=t.terminal_reason,
            failure_penalty_equivalent_cny=t.failure_penalty_equivalent_cny)
        phase='training_environment_steps' if learning else 'warmup_environment_steps'
        report[phase]+=1
        if learning: agent.learn(batch_size=64)
        total=report['training_environment_steps']+report['warmup_environment_steps']
        if total % 5000 == 0:
            print(json.dumps(save_progress(report['completed_rounds']+1)),flush=True)

    warmup_attempts=[]
    def warmup_observe(index,t):
        remember(t,False)
        if report['warmup_environment_steps']==50_000: raise WarmupComplete()
    if saved is None:
        try:
            rng=random.Random(42)
            while report['warmup_environment_steps']<50_000:
                order=list(train); rng.shuffle(order)
                for episode in order:
                    before=report['warmup_environment_steps']
                    replay_start=agent.replay.insertions
                    try:
                        values,status=execute(episode,lambda s,a:agent.select_power(s,a,epsilon=1.),
                            accountant,economics,penalty,warmup_observe)
                    except WarmupComplete:
                        warmup_attempts.append(dict(sample_id=episode.sample_id,status='observed_nonterminal_prefix',
                            inserted=report['warmup_environment_steps']-before))
                        raise
                    warmup_attempts.append(dict(sample_id=episode.sample_id,status=status,
                        inserted=report['warmup_environment_steps']-before))
                    if status != 'completed':
                        agent.finalize_streamed_failure_prefix(range(replay_start,agent.replay.insertions))
        except WarmupComplete:
            pass
        report['warmup_attempts']=warmup_attempts
        report['warmup_adam_updates']=agent.economic_optimizer_updates
        report['warmup_replay_fit']=checkpoint_replay_fit(agent,0,output)
        report['status']='training'
    write_json(output/'report.json',report)
    save_progress(report['completed_rounds']+1)
    best_score=max(((entry['validation']['completed'],entry['validation']['mean_duration_coverage'],
        -entry['validation']['fixed_complete_set_cost_cny']
        if entry['validation']['fixed_complete_set_cost_cny'] is not None else 0.)
        for entry in report['rounds']),default=None)
    try:
        for round_index in range(report['completed_rounds']+1,CONFIG['rounds']+1):
            before=agent.economic_optimizer_updates; before_target=agent.target_sync_calls
            order=list(train); random.Random(42+round_index).shuffle(order)
            training=[]
            for episode_index,episode in enumerate(order):
                eps=epsilon_at(round_index,episode_index)
                replay_start=agent.replay.insertions
                values,status=execute(episode,lambda s,a:agent.select_power(s,a,epsilon=eps),
                    accountant,economics,penalty,lambda index,t:remember(t,True))
                if status != 'completed':
                    agent.finalize_streamed_failure_prefix(range(replay_start,agent.replay.insertions))
                    train_values.capture(episode,values,status,round_index,replay_start)
                row=sample_record(episode,values,status); row['epsilon']=eps; training.append(row)
                print(f'train round={round_index}/8 sample={episode.sample_id} status={status} steps={len(values)} epsilon={eps:.6f}',flush=True)
            train_summary=split_summary(training)
            write_csv(output/f'round_{round_index:02d}_train_samples.csv',training)
            training_td_fit=agent.td_diagnostics.snapshot(reset=True)
            replay_fit=checkpoint_replay_fit(agent,round_index,output)
            probe_rows=write_probe_snapshot(agent,probes,round_index,output)
            failure_values=train_values.snapshot(agent,round_index,
                training_td_fit['failure_terminal_draws_total'])
            validation_summary,validation_rows=evaluate(validation,agent,accountant,economics,penalty,output,round_index)
            report['validation_environment_steps']+=validation_summary['environment_steps']
            entry=dict(round=round_index,epsilon_start=epsilon_at(round_index,0),
                epsilon_end=epsilon_at(round_index,35),train=train_summary,validation=validation_summary,
                adam_updates=agent.economic_optimizer_updates-before,
                target_copies=agent.target_sync_calls-before_target,
                cumulative_adam_updates=agent.economic_optimizer_updates,
                cumulative_target_copies=agent.target_sync_calls,
                q_diagnostics=dict(training_td_fit=training_td_fit,
                    checkpoint_replay_fit=replay_fit,
                    fixed_probes=dict(file=f'q_probes/round_{round_index:02d}.json',count=len(probe_rows)),
                    train_failure_value=failure_values))
            report['rounds'].append(entry); report['completed_rounds']=round_index
            pause_here=stop_after_round==round_index and round_index<CONFIG['rounds']
            if round_index == CONFIG['rounds']:
                report['status']='completed'
            elif pause_here:
                report['status']='paused_checkpoint'
            report['fixed_probes'].extend(fixed_probes(source,validation,agent,accountant,round_index))
            torch.save(model_snapshot(agent,round_index),output/'models'/f'round_{round_index:02d}.pt')
            full_cost=validation_summary['fixed_complete_set_cost_cny']
            score=(validation_summary['completed'],validation_summary['mean_duration_coverage'],
                   -full_cost if full_cost is not None else 0.)
            if best_score is None or score>best_score:
                best_score=score; report['best_research_candidate_round']=round_index
                torch.save(model_snapshot(agent,round_index),output/'models'/'best_research_candidate.pt')
            report.update(cumulative_adam_updates=agent.economic_optimizer_updates,
                cumulative_target_copies=agent.target_sync_calls,elapsed_seconds=time.monotonic()-started)
            torch.save(dict(agent=agent.training_state(),report=report),output/'models'/'training_state_latest.pt')
            torch.save(model_snapshot(agent,round_index),output/'models'/'latest.pt')
            write_json(output/'report.json',report)
            write_json(output/'fixed_probe_history.json',report['fixed_probes'])
            save_progress(round_index)
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[2]/'scripts/report_v6.py'),
                            str(output)],check=True)
            print('ROUND_COMPLETE '+json.dumps(entry),flush=True)
            if pause_here:
                return report
        report['status']='completed'
        compare_v5(v5_root/'docs/results/v5_economic4_paused_20261021/sample_metrics.csv',validation_rows,output)
    except Exception as error:
        report['status']='stopped_error'; report['error']=f'{type(error).__name__}: {error}'
        report['cumulative_adam_updates']=agent.economic_optimizer_updates
        report['cumulative_target_copies']=agent.target_sync_calls
        report['elapsed_seconds']=time.monotonic()-started
        write_json(output/'report.json',report)
        save_progress(report['completed_rounds']+1)
        raise
    report.update(cumulative_adam_updates=agent.economic_optimizer_updates,
        cumulative_target_copies=agent.target_sync_calls,elapsed_seconds=time.monotonic()-started)
    write_json(output/'report.json',report)
    save_progress(report['completed_rounds'])
    return report


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[2]
    parser.add_argument('--data-root',type=Path,default=root/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--v5-root',type=Path,default=root.parent/'v5-economic4-ddqn')
    parser.add_argument('--resume',type=Path,help='full round-boundary checkpoint in the existing output directory')
    parser.add_argument('--stop-after-round',type=int,choices=range(1,CONFIG['rounds']+1),
                        help='finish and save this absolute traversal number, then pause normally')
    args=parser.parse_args()
    run(args.data_root,args.output_dir,args.v5_root,resume=args.resume,stop_after_round=args.stop_after_round)


if __name__=='__main__': main()
