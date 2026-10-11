"""Per-round greedy monitoring with qualified economic checkpoint selection."""

from __future__ import annotations

import json
import math
import csv
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import torch

from v3.control import EconomicMPC

from .control import ACTION_KW, ReplayExecutionError, replay_episode
from .dqn import (
    DirectPowerDDQN, ECONOMIC_HIDDEN_DIMS, ECONOMIC_LEARNING_RATE, STATE_DIM, STATE_FEATURES,
)
from .experiment_schedule import (
    BestCheckpoint, EconomicUpdateSchedule, annotate_evaluation, fully_completed,
)
from .train import _run_episodes, _summarize
from .diagnostics import fixed_q_diagnostics, trajectory_record, executed_transition_statistics
from .learning_curves import FIGURES, write_learning_curves
from .reward_feedback import BatteryEnergyValue
from .experiment_paths import unarchived_output_path
from .failure_replay import FailurePenalty, prepare_failure_replay
from .decision_diagnostics import (
    DIAGNOSTIC_SAMPLE_IDS, MAX_DECISIONS_PER_SAMPLE,
    actual_greedy_q_records, save_diagnostic_snapshot,
)


def greedy_evaluate(episodes, agent, accountant, beta_soc: float, *,
                    redistribute_battery_energy: bool = False, trajectory_path: Path | None = None,
                    learn_no_feasible_failures: bool = False, failure_penalty: FailurePenalty | None = None,
                    abort_on_execution_error: bool = False,
                    decision_diagnostic_sink=None) -> dict:
    """Preserve all training RNG streams; never add replay or run optimizers."""
    random_state = agent.random.getstate()
    torch_state = torch.random.get_rng_state()
    failed_transitions = []
    failure_profiles = []
    failure_events = []
    reason_counts={'soc_limited':0,'structural_power':0,'data_truncation':0,'execution_error':0}
    event_only_count=0
    def on_failure(exc):
        nonlocal event_only_count
        failure_events.append(exc)
        if abort_on_execution_error and exc.failure_kind not in ('no_feasible_action','data_truncation'):
            raise exc
        cause=(exc.failure_cause if exc.failure_kind=='no_feasible_action' else
               'data_truncation' if exc.failure_kind=='data_truncation' else 'execution_error')
        reason_counts[cause]+=1
        transitions=exc.executed_transitions
        if learn_no_feasible_failures and exc.failure_kind=='no_feasible_action':
            preview=prepare_failure_replay(exc,penalty=failure_penalty or FailurePenalty.from_reference(),
                energy_value=BatteryEnergyValue.from_accountant(accountant),
                redistribute_battery_energy=redistribute_battery_energy)
            transitions=preview.transitions
            event_only_count+=int(preview.event_only)
        failed_transitions.extend(transitions)
        if decision_diagnostic_sink is not None and exc.sample_id in DIAGNOSTIC_SAMPLE_IDS:
            decision_diagnostic_sink(exc.sample_id, transitions, failed=True)
        profile=trajectory_record(
            exc.sample_id,
            transitions, completed=False, failure=str(exc), reward_scale=agent.reward_scale)
        profile.update(failure_cause=cause,failed_observed_state=exc.failed_state,
                       controllable_by_power_policy=False if cause=='structural_power' else None)
        failure_profiles.append(profile)
    try:
        results, failures = _run_episodes(
            episodes, lambda state, feasible: agent.select_power(state, feasible, epsilon=0.0),
            accountant, beta_soc=beta_soc,
            on_failure=on_failure, redistribute_battery_energy=redistribute_battery_energy,
        )
        if decision_diagnostic_sink is not None:
            for result in results:
                if result.sample_id in DIAGNOSTIC_SAMPLE_IDS:
                    decision_diagnostic_sink(result.sample_id, result.transitions, failed=False)
        summary = annotate_evaluation(
            _summarize(results, len(episodes), failures, failed_transitions),
            episodes, results, failure_events)
        summary.update(executed_transition_statistics(
            (*[t for result in results for t in result.transitions], *failed_transitions)))
        summary.update(failure_reason_counts=reason_counts,event_only_failures=event_only_count)
        if trajectory_path is not None:
            _write_json(trajectory_path, {
                'completed': [trajectory_record(r.sample_id,r.transitions,completed=True,
                    reward_scale=agent.reward_scale) for r in results],
                'failed': failure_profiles,
            })
            summary['trajectory_file'] = trajectory_path.name
        return summary
    finally:
        agent.random.setstate(random_state)
        torch.random.set_rng_state(torch_state)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _flatten_metrics(values: dict, prefix: str = '') -> dict:
    """Keep captured monetary, failure and numerical metrics in a portable CSV."""
    flattened = {}
    for key, value in values.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, f'{name}_'))
        elif isinstance(value, (list, tuple)):
            flattened[name] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[name] = value
    return flattened


def _write_round_metrics(report: dict, path: Path) -> None:
    records = []
    for row in report['rounds']:
        validation = row['greedy_validation']
        td = row['td_statistics']
        scale = report['hyperparameters']['reward_scale']
        record = {
            'round': row['round'], 'epsilon': row['epsilon'], 'reward_scale': scale,
            'exploratory_completed': row['exploratory_train']['completed'],
            'greedy_train_completed': row['greedy_train']['completed'],
            'greedy_validation_completed': None if validation is None else validation['completed'],
            'greedy_train_evaluable_episodes': row['greedy_train'].get('evaluable_episodes'),
            'greedy_validation_evaluable_episodes': (
                None if validation is None else validation.get('evaluable_episodes')),
            'validation_cost_cny': None if validation is None else validation['cost_cny'],
            'qualified': row['qualified_checkpoint'],
            'economic_optimizer_updates': row['economic_optimizer_updates_cumulative'],
            'economic_replay_insertions': row['economic_replay_insertions_cumulative'],
            'environment_transitions': row['training_environment_transitions_cumulative'],
            'td_error_mae_scaled_reward_units': td.get('mean_absolute_td_error'),
            'td_error_mae_original_reward_units': (
                None if td.get('mean_absolute_td_error') is None
                else td['mean_absolute_td_error']/scale),
        }
        for label, summary in (('exploratory', row['exploratory_train']),
                               ('train', row['greedy_train']), ('validation', validation)):
            if summary is not None:
                record.update(_flatten_metrics(summary, f'{label}_'))
        record.update(_flatten_metrics(td, 'td_'))
        record.update({key:value for key,value in row.items()
                       if not isinstance(value,(dict,list)) and key not in
                       ('economic_optimizer_updates','economic_replay_insertions')})
        record['economic_optimizer_updates_this_round'] = row['economic_optimizer_updates']
        record['economic_replay_insertions_this_round'] = row['economic_replay_insertions']
        records.append(record)
    fields = list(dict.fromkeys(key for record in records for key in record))
    if not fields:
        fields = ['round', 'epsilon', 'reward_scale', 'td_error_mae_original_reward_units']
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _persist_report(report: dict, output_dir: Path) -> None:
    _write_json(output_dir/'round_history.json', report)
    _write_json(output_dir/'report.json', report)
    _write_round_metrics(report, output_dir/'round_metrics.csv')


class _ConsoleLog:
    """Mirror output to the PyCharm terminal and its independent run log."""
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def run_monitored_training(
    dataset, *, output_dir: Path, rounds: int = 100, beta_soc: float = 0.0,
    cadence: str = "replay32", target_mode: str = "soft", target_interval: int = 500,
    seed: int = 42, batch_size: int = 64, updates_per_episode: int = 16,
    epsilon_start: float = 1.0, epsilon_end: float = 0.05, progress_every_steps: int = 0,
    redistribute_battery_energy: bool = False, n_step: int = 8,
    episode_credit_scope: str = 'sample', required_split_sizes: tuple[int,int] | None = None,
    capture_trajectories: bool = False,
    learn_no_feasible_failures: bool = True, failure_penalty_scale: float = 1.0,
    reward_scale: float = 0.001, failure_terminal_quota: int = 2,
    manifest_sha256: dict[str,str] | None = None,
    dataset_roots: tuple[str,...] | None = None,
    abort_on_execution_error: bool = False, capture_log: bool = False,
    diagnostic_rounds: tuple[int, ...] = (),
    resume_from: Path | None = None,
):
    """Run or resume only at an atomically saved, complete round boundary."""
    if not math.isfinite(reward_scale) or reward_scale <= 0:
        raise ValueError('reward_scale must be finite and positive')
    if type(failure_terminal_quota) is not int or failure_terminal_quota < 0:
        raise ValueError('failure_terminal_quota must be a nonnegative integer')
    if (type(diagnostic_rounds) is not tuple or
            any(type(value) is not int or not 1 <= value <= rounds for value in diagnostic_rounds) or
            len(set(diagnostic_rounds)) != len(diagnostic_rounds)):
        raise ValueError('diagnostic rounds must be distinct integers within the training budget')
    output_dir = unarchived_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        if any((output_dir/name).exists() for name in
               ('report.json','round_history.json','best_agent.pt','run_metadata.json',
                'diagnostic_checkpoints','training_state_latest.pt')):
            raise FileExistsError('training requires a fresh result directory')
        resume_state = None
    else:
        checkpoint_path = Path(resume_from).resolve()
        if checkpoint_path != (output_dir/'training_state_latest.pt').resolve():
            raise ValueError('resume checkpoint must be training_state_latest.pt in the output directory')
        resume_state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if resume_state.get('format_version') != 1:
            raise ValueError('unsupported complete-round checkpoint format')
    source_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=source_root,text=True).strip()
    dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=source_root,text=True).strip())
    failure_penalty = FailurePenalty.from_reference(scale=failure_penalty_scale)
    energy_value = BatteryEnergyValue.from_accountant(EconomicMPC(nominal_cost_cny=1.0))
    parameters = {
        'beta_soc':beta_soc,'rounds':rounds,'seed':seed,'batch_size':batch_size,
        'gamma':1.0,'learning_rate':ECONOMIC_LEARNING_RATE,'hidden_dims':list(ECONOMIC_HIDDEN_DIMS),
        'state_dim':STATE_DIM,'state_features':list(STATE_FEATURES),
        'action_kw':list(ACTION_KW),'replay_capacity':300000,
        'epsilon_start':epsilon_start,'epsilon_end':epsilon_end,
        'epsilon_linear_end_round':70,'epsilon_hold_from_round':71,
        'cadence':cadence,'updates_per_completed_episode':16,'target_mode':target_mode,
        'target_interval_optimizer_updates':target_interval if target_mode=='optimizer' else None,
        'target_tau':0.001 if target_mode=='soft' else None,
        'loss':'smooth_l1','huber_delta':1.0,'gradient_max_norm':10.0,
        'bootstrap_uses_same_cadence':True,'torch_threads':torch.get_num_threads(),
        'redistribute_battery_energy':redistribute_battery_energy,'n_step':n_step,
        'episode_credit_scope':episode_credit_scope,
        'required_split_sizes':list(required_split_sizes) if required_split_sizes is not None else None,
        'learn_no_feasible_failures':learn_no_feasible_failures,
        'failure_penalty':asdict(failure_penalty)|{'amount_equivalent_cny':failure_penalty.amount,
            'scaled_training_amount':failure_penalty.amount*reward_scale,
            'scaled_training_unit':'scaled_reward_equivalent_cny'},
        'battery_energy_value':{'coefficient_cny':energy_value.coefficient_cny,
                                'reference_soc':energy_value.reference_soc},
        'reward_scale':float(reward_scale),'failure_terminal_quota':failure_terminal_quota,
        'shore_settlement':'modeled_fixed_target_soc_0.6',
    }
    metadata = {
        'source_commit':source_commit,'source_worktree_dirty':dirty,
        'hyperparameters':parameters,'manifest_sha256':dict(manifest_sha256 or {}),
        'dataset_roots':list(dataset_roots or ()),
        'reward_units':{'economic_q':'scaled_reward_equivalent_cny',
            'stored_reward':'reward_scale * final_unscaled_reward',
            'original_rewards':'reward_equivalent_cny',
            'economic_ledger':'CNY; modeled fixed-target SHORE and terminal settlement separately tagged',
            'td_original_unit_conversion':'native_td_error / reward_scale'},
        'initialization':'new seeded weights; empty replay; new optimizer; no checkpoint loaded',
        'dataset_class':type(dataset).__name__,
        'diagnostic_configuration':{
            'rounds':list(diagnostic_rounds),
            'train_sample_ids':sorted(DIAGNOSTIC_SAMPLE_IDS),
            'max_decisions_per_sample':MAX_DECISIONS_PER_SAMPLE,
            'checkpoint_purpose':'diagnostic_only',
        },
    }
    if resume_state is None:
        _write_json(output_dir/'run_metadata.json',metadata)
        initial = metadata | {'completed_training':False,'run_status':'starting','rounds':[],
                              'best_checkpoint':None,'test_payloads_opened':dataset.opened_test_payloads}
        _persist_report(initial,output_dir)
    else:
        saved = resume_state['metadata']
        if (saved['source_commit'] != source_commit or saved['hyperparameters'] != parameters
                or saved['manifest_sha256'] != metadata['manifest_sha256']
                or saved['dataset_roots'] != metadata['dataset_roots']
                or saved['dataset_class'] != metadata['dataset_class']
                or saved['diagnostic_configuration'] != metadata['diagnostic_configuration']
                or json.loads((output_dir/'run_metadata.json').read_text(encoding='utf-8')) != saved):
            raise ValueError('resume source, architecture, hyperparameters or data manifest mismatch')
        metadata = saved
        best_path = output_dir/'best_agent.pt'
        committed_best = resume_state.get('best_checkpoint_bytes')
        if committed_best is not None and (not best_path.exists() or best_path.read_bytes() != committed_best):
            temporary = output_dir/'best_agent.pt.tmp'
            temporary.write_bytes(committed_best)
            temporary.replace(best_path)
        elif committed_best is None and best_path.exists():
            best_path.replace(output_dir/'best_agent_uncommitted.pt')
        existing = [output_dir/name for name in (*FIGURES, 'learning_curves_metadata.json')]
        existing.extend(output_dir/name for name in
                        json.loads((output_dir/'report.json').read_text(encoding='utf-8'))
                        .get('learning_curve_artifacts', {}).get('figures', ())
                        if Path(name).name == name)
        existing = list(dict.fromkeys(path for path in existing if path.exists()))
        if existing:
            archive = output_dir/f'resume_prior_curves_after_round_{resume_state["completed_round"]:03d}'
            suffix = 1
            while archive.exists():
                archive = output_dir/f'resume_prior_curves_after_round_{resume_state["completed_round"]:03d}_{suffix}'
                suffix += 1
            archive.mkdir()
            for path in existing:
                path.replace(archive/path.name)
    runtime = {'phase':'initialization','round':None,'episode':None}
    with ExitStack() as stack:
        if capture_log:
            log = stack.enter_context((output_dir/'train.log').open('a' if resume_state else 'x',encoding='utf-8'))
            stack.enter_context(redirect_stdout(_ConsoleLog(sys.stdout,log)))
            stack.enter_context(redirect_stderr(_ConsoleLog(sys.stderr,log)))
        try:
            agent, report = _run_monitored_training(dataset,output_dir=output_dir,rounds=rounds,
                beta_soc=beta_soc,cadence=cadence,target_mode=target_mode,target_interval=target_interval,
                seed=seed,batch_size=batch_size,updates_per_episode=updates_per_episode,
                epsilon_start=epsilon_start,epsilon_end=epsilon_end,progress_every_steps=progress_every_steps,
                redistribute_battery_energy=redistribute_battery_energy,n_step=n_step,
                episode_credit_scope=episode_credit_scope,required_split_sizes=required_split_sizes,
                capture_trajectories=capture_trajectories,learn_no_feasible_failures=learn_no_feasible_failures,
                failure_penalty_scale=failure_penalty_scale,reward_scale=reward_scale,
                failure_terminal_quota=failure_terminal_quota,
                run_metadata=metadata,runtime=runtime,abort_on_execution_error=abort_on_execution_error,
                diagnostic_rounds=diagnostic_rounds,resume_state=resume_state)
            report['learning_curve_artifacts'] = write_learning_curves(report,output_dir)
            _persist_report(report,output_dir)
            return agent, report
        except BaseException as exc:
            report = json.loads((output_dir/'report.json').read_text(encoding='utf-8'))
            report.update(completed_training=False,run_status='aborted')
            report['abort'] = {'exception_type':type(exc).__name__,'message':str(exc),
                'phase':runtime['phase'],'round':runtime['round'],'episode':runtime['episode'],
                'completed_rounds_retained':len(report['rounds']),
                'automatic_resume':False,'partial_round_is_not_a_completed_round':True}
            if 'agent' in runtime:
                agent = runtime['agent']
                report['abort']['actual_partial_execution_counts'] = {
                    'economic_replay_insertions':agent.economic_replay_insertions,
                    'economic_optimizer_updates':agent.economic_optimizer_updates,
                    'target_sync_calls_including_initial_copy':agent.target_sync_calls}
            _persist_report(report,output_dir)
            try:
                report['learning_curve_artifacts'] = write_learning_curves(report,output_dir)
            except Exception as plot_error:
                report['abort']['plot_preservation_error'] = str(plot_error)
            _persist_report(report,output_dir)
            if capture_log:
                traceback.print_exc()
            raise


def _run_monitored_training(
    dataset, *, output_dir: Path, rounds: int, beta_soc: float,
    cadence: str, target_mode: str, target_interval: int,
    seed: int, batch_size: int, updates_per_episode: int,
    epsilon_start: float, epsilon_end: float, progress_every_steps: int,
    redistribute_battery_energy: bool, n_step: int,
    episode_credit_scope: str, required_split_sizes: tuple[int,int] | None,
    capture_trajectories: bool,
    learn_no_feasible_failures: bool, failure_penalty_scale: float,
    reward_scale: float, failure_terminal_quota: int,
    run_metadata: dict, runtime: dict, abort_on_execution_error: bool,
    diagnostic_rounds: tuple[int, ...],
    resume_state: dict | None,
):
    if not 1 <= rounds <= 100 or updates_per_episode != 16 or batch_size < 1:
        raise ValueError("study permits at most 100 rounds and fixes legacy episode credit at 16")
    if (not 0 <= epsilon_end <= epsilon_start <= 1 or not math.isfinite(beta_soc) or beta_soc < 0
            or episode_credit_scope not in ('sample','voyage')):
        raise ValueError("invalid beta or epsilon schedule")
    if dataset.opened_test_payloads != 0:
        raise RuntimeError("Test must be closed before the study")
    started = time.monotonic()
    runtime['phase']='dataset_loading'
    train = dataset.load_train()
    validation = dataset.load_validation()
    if not train or not validation:
        raise ValueError("Train and Validation must be nonempty")
    if any(item.split != "train" for item in train) or any(item.split != "validation" for item in validation):
        raise ValueError("dataset split identity differs")
    if required_split_sizes is not None and (len(train),len(validation)) != required_split_sizes:
        raise ValueError('dataset sizes differ from the required checkpoint qualification set')
    agent = DirectPowerDDQN(seed=seed,n_step=n_step,reward_scale=reward_scale,
                            learning_rate=ECONOMIC_LEARNING_RATE, hidden_dims=ECONOMIC_HIDDEN_DIMS,
                            replay_capacity=300_000, failure_terminal_quota=failure_terminal_quota,
                            target_tau=0.001)
    runtime.update(agent=agent,phase='bootstrap')
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    schedule = EconomicUpdateSchedule(cadence, target_mode=target_mode, target_interval=target_interval)
    failure_penalty=FailurePenalty.from_reference(scale=failure_penalty_scale)
    selector = BestCheckpoint()
    bootstrap_failed = []
    bootstrap_prefix = 0
    bootstrap_completed_voyages = 0
    bootstrap_unknown_discarded = 0
    bootstrap_events = []
    failure_suffix_count = 0
    no_suffix_failure_events = 0
    initial_q_diagnostics = fixed_q_diagnostics(agent,accountant)
    bootstrap_reason_counts={'soc_limited':0,'structural_power':0,'data_truncation':0,'execution_error':0}

    def replay_counts():
        n=agent.economic_replay_insertions
        return {
            'success':agent.economic_success_replay_insertions,'failure':agent.economic_failure_replay_insertions,
            'failure_terminals':agent.economic_failure_terminal_insertions,
            'failure_terminals_in_pool':agent.failure_terminal_replay_count,
            'success_fraction':agent.economic_success_replay_insertions/n if n else None,
            'failure_fraction':agent.economic_failure_replay_insertions/n if n else None,
        }

    def failure_view(exc):
        if learn_no_feasible_failures and exc.failure_kind=='no_feasible_action':
            return prepare_failure_replay(exc,penalty=failure_penalty,
                energy_value=BatteryEnergyValue.from_accountant(accountant),
                redistribute_battery_energy=redistribute_battery_energy)
        return None

    def bootstrap_failure(exc):
        nonlocal bootstrap_prefix, bootstrap_completed_voyages, bootstrap_unknown_discarded
        nonlocal failure_suffix_count, no_suffix_failure_events
        bootstrap_events.append(exc)
        if abort_on_execution_error and exc.failure_kind not in ('no_feasible_action','data_truncation'):
            raise exc
        cause=(exc.failure_cause if exc.failure_kind=='no_feasible_action' else
               'data_truncation' if exc.failure_kind=='data_truncation' else 'execution_error')
        bootstrap_reason_counts[cause]+=1
        view=failure_view(exc)
        bootstrap_failed.extend(view.transitions if view else exc.executed_transitions)
        if exc.failure_kind == 'no_feasible_action' and (
                not exc.executed_transitions or exc.executed_transitions[-1].done):
            no_suffix_failure_events += 1
        if exc.failure_kind == 'data_truncation' and exc.executed_transitions:
            retained=agent.remember_completed_prefix(exc.executed_transitions)
            bootstrap_prefix += retained
            bootstrap_unknown_discarded += len(exc.executed_transitions) - retained
            bootstrap_completed_voyages += sum(
                t.shore_ledger is not None or t.is_successful_terminal
                for t in exc.executed_transitions[:retained])
        elif exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
            if view is not None:
                retained=agent.remember_trajectory(view.transitions)
                prefix=exc.executed_transitions[:view.successful_prefix_transitions]
            else:
                retained=agent.remember_completed_prefix(exc.executed_transitions)
                prefix=exc.executed_transitions[:retained]
            bootstrap_prefix += retained
            bootstrap_completed_voyages += sum(t.shore_ledger is not None or t.is_successful_terminal
                                               for t in prefix)
            failure_suffix_count += len(exc.executed_transitions)-retained

    if resume_state is None:
        print(f"bootstrap beta={beta_soc:g} cadence={cadence} target={target_mode} start", flush=True)
        bootstrap_policy_calls = 0
        def bootstrap_policy(state, feasible):
            nonlocal bootstrap_policy_calls
            power = min(feasible, key=lambda value: abs(value-state[4]*600))
            bootstrap_policy_calls += 1
            if progress_every_steps and bootstrap_policy_calls % progress_every_steps == 0:
                print(f'bootstrap progress step={bootstrap_policy_calls} soc={state[0]:.4f} fc={power}',flush=True)
            return power
        bootstrap, bootstrap_errors = _run_episodes(
            train, bootstrap_policy, accountant, beta_soc=beta_soc,
            on_failure=bootstrap_failure,
            redistribute_battery_energy=redistribute_battery_energy,
        )
        for result in bootstrap:
            agent.remember_trajectory(result.transitions)
            bootstrap_completed_voyages += sum(t.shore_ledger is not None or t.is_successful_terminal
                                               for t in result.transitions)
        schedule.grant(insertions=agent.economic_replay_insertions,
                       completed_episodes=(len(bootstrap) if episode_credit_scope == 'sample' else bootstrap_completed_voyages))
        bootstrap_losses = schedule.consume(agent, batch_size=batch_size)
        bootstrap_td_statistics = agent.td_statistics()
        if target_mode == "round":
            agent.sync_target()
        bootstrap_summary = annotate_evaluation(
            _summarize(bootstrap, len(train), bootstrap_errors, bootstrap_failed),
            train, bootstrap, bootstrap_events)
        bootstrap_summary.update(executed_transition_statistics(
            (*[t for result in bootstrap for t in result.transitions], *bootstrap_failed)))
        bootstrap_summary['failure_reason_counts']=bootstrap_reason_counts
        bootstrap_summary['unknown_discarded_economic_q_transitions'] = bootstrap_unknown_discarded
        bootstrap_insertions = agent.economic_replay_insertions
        print(f"bootstrap completed={len(bootstrap)}/{len(train)} economic_updates={agent.economic_optimizer_updates}", flush=True)
        history = []
        selected_steps = executed_training_steps = evaluation_steps = 0
        unknown_discarded_total = 0
        best_record = None
        diagnostic_artifacts: list[dict] = []
        first_qualification = None
        elapsed_before_resume = 0.0
    else:
        agent.load_training_state(resume_state['agent'])
        schedule.remaining_transition_credit = resume_state['schedule']['remaining_transition_credit']
        schedule.pending_episode_updates = resume_state['schedule']['pending_episode_updates']
        report = resume_state['report']
        history = report['rounds']
        if len(history) != resume_state['completed_round'] or len(history) >= rounds:
            raise ValueError('resume requires an incomplete run at a complete round boundary')
        bootstrap_summary = report['bootstrap']
        bootstrap_insertions = report['bootstrap_economic_replay_insertions']
        bootstrap_losses = [None] * report['bootstrap_optimizer_updates']
        bootstrap_td_statistics = report['bootstrap_td_statistics']
        initial_q_diagnostics = report['initial_fixed_state_q_diagnostics']
        counts = report['execution_counts']
        selected_steps = counts['training_policy_calls']
        executed_training_steps = counts['training_environment_transitions']
        evaluation_steps = counts['greedy_evaluation_environment_transitions']
        unknown_discarded_total = counts.get('unknown_discarded_economic_q_transitions', 0)
        failure_suffix_count = counts['failed_suffix_transitions_excluded_from_economic_replay']
        no_suffix_failure_events = counts['no_feasible_action_events_without_executed_suffix']
        best_record = report['best_checkpoint']
        diagnostic_artifacts = report['diagnostic_artifacts']
        first_qualification = report['first_qualification']
        selector.round = best_record['round'] if best_record else None
        selector.cost_cny = best_record['validation_comparable_cost_cny'] if best_record else None
        selector.first_eligible_round = report['first_qualification']['round'] if first_qualification else None
        selector.eligible_rounds = list(report['eligible_rounds'])
        elapsed_before_resume = report['elapsed_seconds']
        print(f"resume after round {len(history)}/{rounds}", flush=True)
    hyperparameters = run_metadata['hyperparameters']
    source_commit = run_metadata['source_commit']
    source_worktree_dirty = run_metadata['source_worktree_dirty']

    def snapshot_report(completed_training=False):
        return run_metadata | {
            "completed_training": completed_training, "source_commit": source_commit,
            'run_status':'completed' if completed_training else 'running',
            'source_worktree_dirty': source_worktree_dirty,
            "hyperparameters": hyperparameters, "bootstrap": bootstrap_summary,
            "bootstrap_optimizer_updates": len(bootstrap_losses),
            'formal_training_optimizer_updates': agent.economic_optimizer_updates-len(bootstrap_losses),
            'bootstrap_td_statistics': bootstrap_td_statistics,
            'initial_fixed_state_q_diagnostics': initial_q_diagnostics,
            'economic_replay_outcome_counts':replay_counts(),
            "bootstrap_economic_replay_insertions": bootstrap_insertions,
            "rounds": history, "best_checkpoint": best_record,
            "diagnostic_artifacts": diagnostic_artifacts,
            "first_qualification": first_qualification, "eligible_rounds": selector.eligible_rounds,
            "test_payloads_opened": dataset.opened_test_payloads,
            "execution_counts": {
                "training_environment_transitions": executed_training_steps,
                'training_policy_calls': selected_steps,
                "bootstrap_environment_transitions": bootstrap_summary["executed_onboard_transitions"],
                "greedy_evaluation_environment_transitions": evaluation_steps,
                "economic_replay_insertions": agent.economic_replay_insertions,
                "economic_optimizer_updates": agent.economic_optimizer_updates,
                "target_sync_calls_including_initial_copy": agent.target_sync_calls,
                "target_soft_update_calls": agent.target_soft_update_calls,
                "remaining_transition_credit": schedule.remaining_transition_credit,
                'bootstrap_economic_optimizer_updates': len(bootstrap_losses),
                'formal_training_economic_optimizer_updates': agent.economic_optimizer_updates-len(bootstrap_losses),
                'failed_suffix_transitions_excluded_from_economic_replay': failure_suffix_count,
                'unknown_discarded_economic_q_transitions': unknown_discarded_total,
                'no_feasible_action_events_without_executed_suffix': no_suffix_failure_events,
                'economic_success_replay_insertions':agent.economic_success_replay_insertions,
                'economic_failure_replay_insertions':agent.economic_failure_replay_insertions,
                'economic_failure_terminal_insertions':agent.economic_failure_terminal_insertions,
                'economic_failure_terminal_replay_size':agent.failure_terminal_replay_count,
            },
            "elapsed_seconds": elapsed_before_resume + time.monotonic()-started,
            "cost_rule": "ONBOARD actual plus modeled fixed-target SHORE plus modeled terminal if no final SHORE; incomplete split cost is null",
            "monitoring_rng_isolation": True,
            'failure_training_semantics': ('Genuine no-feasible-action suffixes enter economic Q with a tagged terminal, once-only training penalty and potential correction; ledgers unchanged'
                if learn_no_feasible_failures else 'Unfinished suffixes excluded from economic replay'),
            'dataset_class': type(dataset).__name__,
        }

    _persist_report(snapshot_report(),output_dir)
    for round_index in range(len(history)+1, rounds+1):
        runtime.update(phase='exploratory_training',round=round_index,episode=None)
        epsilon = epsilon_start + (epsilon_end-epsilon_start)*min(round_index-1, 69)/69
        print(f"round {round_index}/{rounds} start epsilon={epsilon:.4f}",flush=True)
        results, failures, failed_transitions, losses = [], [], [], []
        failure_profiles = []
        training_events = []
        reason_counts={'soc_limited':0,'structural_power':0,'data_truncation':0,'execution_error':0}
        completed_voyages = 0
        unknown_discarded_round = 0
        agent.reset_td_statistics()
        before_insertions = agent.economic_replay_insertions
        before_updates = agent.economic_optimizer_updates
        before_success=agent.economic_success_replay_insertions
        before_failure=agent.economic_failure_replay_insertions
        before_failure_terminals=agent.economic_failure_terminal_insertions
        round_train = list(train)
        random.Random(seed + round_index).shuffle(round_train)
        for episode_index, episode in enumerate(round_train, start=1):
            runtime['episode']=episode_index
            def training_policy(state, feasible):
                nonlocal selected_steps
                action = agent.select_power(state, feasible, epsilon=epsilon)
                selected_steps += 1
                if progress_every_steps and selected_steps % progress_every_steps == 0:
                    print(f"progress step={selected_steps} round={round_index}/{rounds} episode={episode_index}/{len(train)} sample={episode.sample_id} soc={state[0]:.4f} fc={action}",flush=True)
                return action

            before_episode_insertions = agent.economic_replay_insertions
            try:
                result = replay_episode(episode,training_policy,accountant=accountant,beta_soc=beta_soc,
                                         redistribute_battery_energy=redistribute_battery_energy)
            except ReplayExecutionError as exc:
                exc.sample_id = str(episode.sample_id)
                training_events.append(exc)
                if abort_on_execution_error and exc.failure_kind not in ('no_feasible_action','data_truncation'):
                    raise
                failures.append(f"{episode.sample_id}: {exc}")
                cause=(exc.failure_cause if exc.failure_kind=='no_feasible_action' else
                       'data_truncation' if exc.failure_kind=='data_truncation' else 'execution_error')
                reason_counts[cause]+=1
                view=failure_view(exc)
                transitions=view.transitions if view else exc.executed_transitions
                failed_transitions.extend(transitions)
                profile=trajectory_record(episode.sample_id,transitions,completed=False,failure=str(exc),
                    reward_scale=reward_scale)
                profile.update(failure_cause=cause,failed_observed_state=exc.failed_state,
                               controllable_by_power_policy=False if cause=='structural_power' else None)
                failure_profiles.append(profile)
                prefix_voyages = 0
                if exc.failure_kind == 'no_feasible_action' and (
                        not exc.executed_transitions or exc.executed_transitions[-1].done):
                    no_suffix_failure_events += 1
                if exc.failure_kind == 'data_truncation' and exc.executed_transitions:
                    retained=agent.remember_completed_prefix(exc.executed_transitions)
                    unknown_discarded_round += len(exc.executed_transitions) - retained
                    prefix_voyages = sum(
                        t.shore_ledger is not None or t.is_successful_terminal
                        for t in exc.executed_transitions[:retained])
                    completed_voyages += prefix_voyages
                elif exc.failure_kind == "no_feasible_action" and exc.executed_transitions:
                    if view is not None:
                        retained=agent.remember_trajectory(view.transitions)
                        prefix=exc.executed_transitions[:view.successful_prefix_transitions]
                    else:
                        retained=agent.remember_completed_prefix(exc.executed_transitions)
                        prefix=exc.executed_transitions[:retained]
                    prefix_voyages = sum(t.shore_ledger is not None or t.is_successful_terminal
                                         for t in prefix)
                    completed_voyages += prefix_voyages
                    failure_suffix_count += len(exc.executed_transitions)-retained
                schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,
                               completed_episodes=(prefix_voyages if episode_credit_scope == 'voyage' else 0))
                losses.extend(schedule.consume(agent,batch_size=batch_size))
                continue
            results.append(result)
            agent.remember_trajectory(result.transitions)
            voyages = sum(t.shore_ledger is not None or t.is_successful_terminal
                          for t in result.transitions)
            completed_voyages += voyages
            schedule.grant(insertions=agent.economic_replay_insertions-before_episode_insertions,
                           completed_episodes=(voyages if episode_credit_scope == 'voyage' else 1))
            losses.extend(schedule.consume(agent,batch_size=batch_size))
        if target_mode == "round":
            agent.sync_target()
        explore = annotate_evaluation(
            _summarize(results,len(train),failures,failed_transitions),
            train, results, training_events)
        explore.update(executed_transition_statistics(
            (*[t for result in results for t in result.transitions], *failed_transitions)))
        explore['failure_reason_counts']=reason_counts
        explore['unknown_discarded_economic_q_transitions'] = unknown_discarded_round
        unknown_discarded_total += unknown_discarded_round
        executed_training_steps += explore['executed_onboard_transitions']
        if capture_trajectories:
            path = output_dir/f'round_{round_index:03d}_exploratory_trajectories.json'
            _write_json(path,{'completed':[trajectory_record(r.sample_id,r.transitions,completed=True,
                                reward_scale=reward_scale) for r in results],
                              'failed':failure_profiles})
            explore['trajectory_file'] = path.name
        print(f"round {round_index}/{rounds} exploratory={len(results)}/{len(train)} greedy_train start",flush=True)
        runtime['phase']='greedy_train'
        decision_rows = [] if round_index in diagnostic_rounds else None
        def diagnostic_sink(sample_id, transitions, *, failed):
            if decision_rows is not None:
                decision_rows.extend(actual_greedy_q_records(
                    sample_id, transitions, agent, failed=failed))
        greedy_train = greedy_evaluate(train,agent,accountant,beta_soc,
            redistribute_battery_energy=redistribute_battery_energy,
            trajectory_path=output_dir/f'round_{round_index:03d}_train_trajectories.json' if capture_trajectories else None,
            learn_no_feasible_failures=learn_no_feasible_failures,failure_penalty=failure_penalty,
            abort_on_execution_error=abort_on_execution_error,
            decision_diagnostic_sink=diagnostic_sink if decision_rows is not None else None)
        evaluation_steps += greedy_train["executed_onboard_transitions"]
        runtime['phase']='greedy_validation'
        print(f"round {round_index}/{rounds} greedy_train={greedy_train['evaluable_completed']}/{greedy_train['evaluable_episodes']} validation start",flush=True)
        greedy_validation = greedy_evaluate(validation,agent,accountant,beta_soc,
            redistribute_battery_energy=redistribute_battery_energy,
            trajectory_path=output_dir/f'round_{round_index:03d}_validation_trajectories.json' if capture_trajectories else None,
            learn_no_feasible_failures=learn_no_feasible_failures,failure_penalty=failure_penalty,
            abort_on_execution_error=abort_on_execution_error)
        evaluation_steps += greedy_validation["executed_onboard_transitions"]
        row = {
            "round":round_index,"epsilon":epsilon,"exploratory_train":explore,
            "greedy_train":greedy_train,"greedy_validation":greedy_validation,
            "validation_skip_reason":None,
            "mean_loss":sum(losses)/len(losses) if losses else None,
            "economic_optimizer_updates":agent.economic_optimizer_updates-before_updates,
            "economic_optimizer_updates_cumulative":agent.economic_optimizer_updates,
            "economic_replay_insertions":agent.economic_replay_insertions-before_insertions,
            "economic_replay_insertions_cumulative":agent.economic_replay_insertions,
            "training_environment_transitions_cumulative":executed_training_steps,
            'training_policy_calls_cumulative': selected_steps,
            "target_sync_calls_cumulative":agent.target_sync_calls,
            "target_soft_update_calls_cumulative":agent.target_soft_update_calls,
            'reward_scale':reward_scale,
            "remaining_transition_credit":schedule.remaining_transition_credit,
            'td_statistics': agent.td_statistics(),
            'fixed_state_q_diagnostics': fixed_q_diagnostics(agent,accountant),
            'completed_voyages_entering_economic_replay': completed_voyages,
            'unknown_discarded_economic_q_transitions': unknown_discarded_round,
            'unknown_discarded_economic_q_transitions_cumulative': unknown_discarded_total,
            'failed_sample_count':explore['physical_failures'],
            'data_truncated_sample_count':explore['data_truncated_episodes'],
            'noncompleted_sample_count':explore['noncompleted_samples'],
            'failure_reason_counts':reason_counts,
            'economic_success_replay_insertions':agent.economic_success_replay_insertions-before_success,
            'economic_failure_replay_insertions':agent.economic_failure_replay_insertions-before_failure,
            'economic_failure_terminal_insertions':agent.economic_failure_terminal_insertions-before_failure_terminals,
            'economic_failure_terminal_replay_size':agent.failure_terminal_replay_count,
            'economic_replay_outcome_counts_cumulative':replay_counts(),
        }
        improved = selector.consider(round_index,greedy_train,greedy_validation)
        row["qualified_checkpoint"] = fully_completed(greedy_train) and fully_completed(greedy_validation)
        row["best_checkpoint_improved"] = improved
        if row["qualified_checkpoint"] and first_qualification is None:
            first_qualification = {
                "round":round_index,"economic_optimizer_updates":agent.economic_optimizer_updates,
                "environment_transitions_including_bootstrap":executed_training_steps+bootstrap_summary["executed_onboard_transitions"],
                "economic_replay_insertions":agent.economic_replay_insertions,
            }
        if improved:
            best_record = {
                "round":round_index,"validation_comparable_cost_cny":selector.cost_cny,
                "train":greedy_train,"validation":greedy_validation,"path":"best_agent.pt",
                "economic_optimizer_updates":agent.economic_optimizer_updates,
                "economic_replay_insertions":agent.economic_replay_insertions,
            }
            temporary = output_dir/"best_agent.pt.tmp"
            torch.save({
                "architecture":"v4_mlp_double_dqn_direct_power",
                "model_state":{key:value.detach().cpu().clone() for key,value in agent.online.state_dict().items()},
                "hyperparameters":hyperparameters,"selection":best_record,
                "source_commit":source_commit,"test_payloads_opened":0,
                'source_worktree_dirty':source_worktree_dirty,'reward_scale':reward_scale,
                'reward_units':run_metadata['reward_units'],
                'manifest_sha256':run_metadata['manifest_sha256'],
                'dataset_roots':run_metadata['dataset_roots'],
            },temporary)
            temporary.replace(output_dir/"best_agent.pt")
        if decision_rows is not None:
            artifact = save_diagnostic_snapshot(
                output_dir, round_index=round_index, agent=agent, decisions=decision_rows,
                run_metadata=run_metadata,
                training_environment_transitions=(
                    executed_training_steps + bootstrap_summary["executed_onboard_transitions"]),
                test_payloads_opened=dataset.opened_test_payloads,
            )
            row["diagnostic_artifacts"] = artifact
            diagnostic_artifacts.append({"round": round_index, **artifact})
        history.append(row)
        if dataset.opened_test_payloads != 0:
            raise RuntimeError("Test opened during monitored training")
        runtime['phase']='persisting_completed_round'
        complete_round_report = snapshot_report()
        _persist_report(complete_round_report,output_dir)
        checkpoint = {
            'format_version':1,'completed_round':round_index,
            'metadata':run_metadata,'report':complete_round_report,
            'agent':agent.training_state(),
            'best_checkpoint_bytes':((output_dir/'best_agent.pt').read_bytes()
                                     if (output_dir/'best_agent.pt').exists() else None),
            'schedule':{'remaining_transition_credit':schedule.remaining_transition_credit,
                        'pending_episode_updates':schedule.pending_episode_updates},
        }
        temporary = output_dir/'training_state_latest.pt.tmp'
        torch.save(checkpoint, temporary)
        temporary.replace(output_dir/'training_state_latest.pt')
        val = f"{greedy_validation['evaluable_completed']}/{greedy_validation['evaluable_episodes']}"
        cost = greedy_validation["cost_cny"]
        print(f"MONITOR round={round_index} exploratory={explore['evaluable_completed']}/{explore['evaluable_episodes']} greedy_train={greedy_train['evaluable_completed']}/{greedy_train['evaluable_episodes']} validation={val} cost={cost} updates={agent.economic_optimizer_updates} best_round={selector.round}",flush=True)
    report = snapshot_report(completed_training=True)
    _persist_report(report,output_dir)
    runtime['phase']='completed'
    return agent, report
