"""Quiet, eight-pass training through SB3's actual OffPolicyAlgorithm.learn.

Importing this module never starts training. Only the explicit CLI does so.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone, timedelta
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import signal
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from v4.v6_dataset import V6TrainingDataset
from v4.v6_training import CONFIG, epsilon_at, sample_record, split_summary, write_csv, write_json
from v4.v6_td_diagnostics import checkpoint_replay_fit
from v4.v6_q_probes import write_probe_snapshot
from v4.v6_train_values import TrainFailureValues
from .curriculum import TrainCurriculum
from .masked_dqn import MaskedDQN
from .reporting import describe_failure, quiet_evaluate, write_comparison


ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = 'fce060ca12fd842d8496aba8d6f922eae179075e'
SB3_COMMIT = '8908708f10c8ff29759c67f55c8acb56cab27463'


class _Finished(BaseException):
    pass


class _Pause(BaseException):
    pass


def local_time():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def runtime_snapshot(model):
    import gymnasium
    import pandas
    return dict(captured_at=local_time(), python_executable=sys.executable,
        python_version=platform.python_version(), pytorch_version=torch.__version__,
        stable_baselines3_version=stable_baselines3.__version__, gymnasium_version=gymnasium.__version__,
        numpy_version=np.__version__, pandas_version=pandas.__version__, device=str(model.device),
        torch_threads=torch.get_num_threads(), sb3_package_file=stable_baselines3.__file__,
        online_trainable_parameters=sum(p.numel() for p in model.online.parameters() if p.requires_grad),
        target_has_trainable_parameters=any(p.requires_grad for p in model.target.parameters()),
        optimizer=dict(learning_rate=model.optimizer.param_groups[0]['lr'],
            betas=list(model.optimizer.param_groups[0]['betas']), eps=model.optimizer.param_groups[0]['eps']),
        replay_array_bytes=sum(value.nbytes for value in vars(model.replay_buffer).values()
                               if isinstance(value, np.ndarray)),
        baseline_commit=BASE_COMMIT, official_sb3_commit=SB3_COMMIT,
        project_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        project_worktree_changes=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines())


def memory_snapshot():
    """One tiny Windows process sample at a pass boundary, never per step."""
    if os.name != 'nt':
        return {}
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
            *[(name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return dict(working_set_bytes=counters.WorkingSetSize, private_commit_bytes=counters.PrivateUsage)


def write_failure_files(output, round_index, failures):
    write_json(output/f'round_{round_index:02d}_failure_details.json',
        dict(round=round_index, failures=failures,
             note='F is the unexecuted failed decision; the last real action is F-1.'))
    rows = [{k:json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
             for k,v in row.items()} for row in failures]
    if rows:
        write_csv(output/f'round_{round_index:02d}_failure_details.csv', rows)
    else:
        (output/f'round_{round_index:02d}_failure_details.csv').write_text(
            'round,phase,sample_id,failure_reason\n', encoding='utf-8')


class _CollectionCallback(BaseCallback):
    def __init__(self, context):
        super().__init__(verbose=0)
        self.context = context

    def _on_rollout_start(self):
        self.model.exploration_rate = self.context.curriculum.epsilon

    def _on_step(self):
        # Returning False here would drop the final transition before storage.
        return True

    def _on_rollout_end(self):
        if self.context.curriculum.phase == 'warmup':
            self.context.finish_episodes()
            self.context.progress_if_due()
            if self.context.stop_requested:
                raise _Pause()


class TrainingContext:
    def __init__(self, output, curriculum, validation, model, *, report=None, probes=None):
        self.output = Path(output)
        self.curriculum = curriculum
        self.validation = validation
        self.model = model
        self.probes = probes or []
        self.stop_requested = False
        self.pass_rows = []
        self.pass_failures = []
        self.train_values = TrainFailureValues(self.output)
        self.report = report or dict(configuration={**CONFIG, 'framework':'Stable-Baselines3',
            'sb3_version':'2.9.0', 'budget_mode':'eight_passes', 'target_counter':'Adam updates'},
            status='warmup', completed_rounds=0, rounds=[], fixed_probes=[],
            training_environment_steps=0, warmup_environment_steps=0,
            validation_environment_steps=0, warmup_adam_updates=0,
            warmup_attempts=[], warmup_complete=False, test_payloads_opened=0,
            cumulative_adam_updates=0, cumulative_target_copies=1,
            budget_note='8 traversals of all35 Train voyages; total executed steps may differ from V6.',
            elapsed_seconds=0.)
        self.started = time.monotonic() - self.report.get('elapsed_seconds', 0.)
        self.callback = _CollectionCallback(self)
        model.after_update_observer = self.after_update

    def event(self, kind, **values):
        with (self.output/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(time=local_time(), event=kind, **values), ensure_ascii=False)+'\n')

    def update_counts(self):
        self.report.update(training_environment_steps=max(0, self.model.num_timesteps-self.model.learning_starts),
            warmup_environment_steps=min(self.model.num_timesteps, self.model.learning_starts),
            cumulative_adam_updates=self.model._n_updates,
            cumulative_target_copies=self.model.target_sync_calls,
            elapsed_seconds=time.monotonic()-self.started)

    def progress(self):
        self.update_counts()
        write_json(self.output/'progress.json', dict(status=self.report['status'],
            round=self.curriculum.pass_index, completed_rounds=self.report['completed_rounds'],
            sample_id=self.curriculum.episode.sample_id if self.curriculum.episode else None,
            episode_index=self.curriculum.episode_index,
            training_environment_steps=self.report['training_environment_steps'],
            warmup_environment_steps=self.report['warmup_environment_steps'],
            adam_updates=self.model._n_updates, target_copies=self.model.target_sync_calls,
            replay_entries=len(self.model.replay), replay_insertions=self.model.replay.insertions,
            epsilon=self.curriculum.epsilon, elapsed_seconds=self.report['elapsed_seconds']))

    def progress_if_due(self):
        if self.model.num_timesteps % 5000 == 0:
            self.progress()

    def finish_episodes(self):
        while self.curriculum.pending_completed:
            finished = self.curriculum.pending_completed.popleft()
            episode, values, status = (finished[k] for k in ('episode','values','status'))
            if status != 'completed':
                self.model.finalize_streamed_failure_prefix(
                    range(finished['replay_start'], self.model.replay.insertions))
            row = sample_record(episode, values, status)
            row['epsilon'] = finished['epsilon']
            self.event('voyage_complete', phase=finished['phase'],
                round=finished['pass_index'], **row)
            if finished['phase'] == 'warmup':
                self.report['warmup_attempts'].append(dict(sample_id=episode.sample_id,
                    status=status, inserted=len(values)))
                continue
            self.pass_rows.append(row)
            ri = finished['pass_index']
            if status != 'completed':
                self.train_values.capture(episode, values, status, ri, finished['replay_start'])
                self.pass_failures.append(describe_failure(episode, values, status, ri,
                    'train_exploration', finished['epsilon'], self.curriculum.accountant))
            if finished['episode_index'] == len(self.curriculum.train) - 1:
                self.finish_pass(ri)

    def finish_pass(self, ri):
        model, output = self.model, self.output
        training = split_summary(self.pass_rows)
        write_csv(output/f'round_{ri:02d}_train_samples.csv', self.pass_rows)
        td_fit = model.td_diagnostics.snapshot(reset=True)
        replay_fit = checkpoint_replay_fit(model, ri, output)
        probe_rows = write_probe_snapshot(model, self.probes, ri, output)
        failure_values = self.train_values.snapshot(model, ri, td_fit['failure_terminal_draws_total'])
        validation, validation_rows, failures = quiet_evaluate(self.validation, model,
            self.curriculum.accountant, self.curriculum.economics, self.curriculum.penalty, output, ri)
        write_failure_files(output, ri, self.pass_failures + failures)
        write_comparison(output, ri, validation_rows, ROOT)
        previous_updates = self.report['rounds'][-1]['cumulative_adam_updates'] if self.report['rounds'] else 0
        previous_targets = self.report['rounds'][-1]['cumulative_target_copies'] if self.report['rounds'] else 1
        loss_count = model.training_loss_count
        entry = dict(round=ri, epsilon_start=epsilon_at(ri, 0),
            epsilon_end=epsilon_at(ri, len(self.curriculum.train)), train=training, validation=validation,
            adam_updates=model._n_updates-previous_updates,
            target_copies=model.target_sync_calls-previous_targets,
            cumulative_adam_updates=model._n_updates, cumulative_target_copies=model.target_sync_calls,
            loss_mean=model.training_loss_sum/loss_count if loss_count else None,
            process_memory=memory_snapshot(), replay_entries=len(model.replay),
            replay_insertions=model.replay.insertions,
            q_diagnostics=dict(training_td_fit=td_fit, checkpoint_replay_fit=replay_fit,
                fixed_probes=dict(file=f'q_probes/round_{ri:02d}.json', count=len(probe_rows)),
                train_failure_value=failure_values))
        self.report['rounds'].append(entry)
        self.report['completed_rounds'] = ri
        self.report['validation_environment_steps'] += validation['environment_steps']
        self.report['status'] = 'completed' if ri == 8 else 'training'
        score = (validation['completed'], validation['mean_duration_coverage'],
                 -(validation['fixed_complete_set_cost_cny'] or 0.))
        previous_score = tuple(self.report.get('best_score', [-1, -1, -float('inf')]))
        if score > previous_score:
            self.report['best_score'] = list(score)
            self.report['best_research_candidate_round'] = ri
            model.save(output/'models/best_research_candidate.zip')
        model.training_loss_sum = 0.; model.training_loss_count = 0
        self.pass_rows.clear(); self.pass_failures.clear()
        self.progress()
        write_json(output/'report.json', self.report)
        self.write_reports()
        self.save_checkpoint(f'round_{ri:02d}')
        model.logger.dump(step=model.num_timesteps)
        self.event('pass_complete', **entry)
        print(f'Pass {ri}/8 | Train {training["completed"]}/35 | '
              f'Greedy Validation {validation["completed"]}/8 | '
              f'FC changes {validation["fc_changes_per_hour"]:.2f}/h | '
              f'reversals {validation["fc_reversals_per_hour"]:.2f}/h', flush=True)

    def write_reports(self):
        spec = importlib.util.spec_from_file_location('sb3_v6_report_formatter', ROOT/'scripts/report_v6.py')
        formatter = importlib.util.module_from_spec(spec); spec.loader.exec_module(formatter)
        entries = self.report['rounds']
        current = entries[-1]
        formatter.write_round_summary(self.output, current,
            entries[-2] if len(entries) > 1 else None, self.report['configuration'])
        path = self.output/f'round_{current["round"]:02d}_summary.md'
        path.write_text('# Stable-Baselines3 DQN 对照：固定8遍预算\n\n'+path.read_text(encoding='utf-8'), encoding='utf-8')
        records = [dict(round=e['round'], train_completed=e['train']['completed'],
            validation_completed=e['validation']['completed'],
            adam_updates=e['adam_updates'], cumulative_adam_updates=e['cumulative_adam_updates'],
            target_copies=e['target_copies'], replay_entries=e['replay_entries'], loss_mean=e['loss_mean'],
            epsilon_start=e['epsilon_start'], epsilon_end=e['epsilon_end'],
            fc_changes_per_hour=e['validation']['fc_changes_per_hour'],
            fc_reversals_per_hour=e['validation']['fc_reversals_per_hour']) for e in entries]
        write_csv(self.output/'training_records.csv', records)
        lines = ['# SB3 DQN 对照结果', '',
            f'状态：{self.report["status"]}；已完成 {self.report["completed_rounds"]}/8 遍。',
            '固定每遍35条Train；实际总步数随策略改变，不宣称与旧V6更新总量相同。',
            'Test payload读取0。TD目标不是Q*；Train包含随机探索，Greedy Validation不进入Replay。', '',
            '|遍数|Train完成|Greedy完成|FC变载/h|相邻反向/h|详情|', '|---|---:|---:|---:|---:|---|']
        for e in entries:
            v=e['validation']; i=e['round']
            lines.append(f'|{i}|{e["train"]["completed"]}/35|{v["completed"]}/8|'
                f'{v["fc_changes_per_hour"]:.3f}|{v["fc_reversals_per_hour"]:.3f}|[报告](round_{i:02d}_summary.md)|')
        lines += ['', '每轮reference_comparison记录V6第2/8轮与V5第32轮，只在共同完整航段上比较经济成本。',
            '完整逐秒Validation CSV.gz含全部13Q与合法排序；模型、Replay和随机状态在models/round_NN。',
            '若TD误差下降而失败或变载没有改善，不能把Loss变化解释成控制改善。', '']
        (self.output/'training_result.md').write_text('\n'.join(lines), encoding='utf-8')

    def after_update(self, model):
        self.finish_episodes()
        self.progress_if_due()
        if self.report['completed_rounds'] == 8:
            raise _Finished()
        if self.stop_requested:
            raise _Pause()

    def save_checkpoint(self, label):
        self.update_counts()
        folder = self.output/'models'/label
        folder.mkdir(parents=True, exist_ok=False)
        self.model.save(folder/'model.zip')
        self.model.save_replay_buffer(folder/'replay.pkl')
        state = dict(report=self.report, curriculum=self.curriculum.curriculum_state(),
            pass_rows=self.pass_rows, pass_failures=self.pass_failures,
            train_fixed_events=self.train_values.fixed_events,
            train_round_prefixes=self.train_values.round_prefixes,
            python_random=random.getstate(), numpy_random=np.random.get_state(),
            torch_random=torch.get_rng_state())
        torch.save(state, folder/'training_state.pt')
        # Marker is written last: a directory without it is not a full checkpoint.
        marker = dict(status='complete', checkpoint_name=label,
            completed_rounds=self.report['completed_rounds'],
            adam_updates=self.model._n_updates, target_copies=self.model.target_sync_calls,
            replay_insertions=self.model.replay.insertions, directory=str(folder.resolve()),
            files=['model.zip','replay.pkl','training_state.pt'])
        write_json(folder/'checkpoint.json', marker)
        write_json(self.output/'models/latest_checkpoint.json', marker)
        return folder


def resume_context(output, resume, curriculum, validation, probes):
    json.loads((resume/'checkpoint.json').read_text(encoding='utf-8'))
    saved = torch.load(resume/'training_state.pt', map_location='cpu', weights_only=False)
    if (output/'report.json').exists():
        existing = json.loads((output/'report.json').read_text(encoding='utf-8'))
        if existing['completed_rounds'] > saved['report']['completed_rounds']:
            raise ValueError('checkpoint is older than the saved completed passes; use the latest checkpoint')
    if any(saved['report']['configuration'][k] != v for k,v in CONFIG.items()):
        raise ValueError('saved checkpoint does not match the frozen V6 baseline configuration')
    env = DummyVecEnv([lambda:curriculum])
    model = MaskedDQN.load(resume/'model.zip', env=env, device='cpu', force_reset=False)
    model.load_replay_buffer(resume/'replay.pkl')
    observation, _ = curriculum.restore_curriculum(saved['curriculum'])
    model._last_obs = observation[None, :]
    context = TrainingContext(output, curriculum, validation, model, report=saved['report'], probes=probes)
    context.pass_rows = saved['pass_rows']; context.pass_failures = saved['pass_failures']
    context.train_values.fixed_events = saved['train_fixed_events']
    context.train_values.round_prefixes = saved['train_round_prefixes']
    random.setstate(saved['python_random']); np.random.set_state(saved['numpy_random'])
    torch.set_rng_state(saved['torch_random'])
    return context


def run(data_root, output, *, resume=None):
    if stable_baselines3.__version__ != '2.9.0':
        raise RuntimeError('This baseline requires the reviewed Stable-Baselines3 2.9.0')
    torch.set_num_threads(1)
    dataset = V6TrainingDataset.open(data_root)
    train, validation = dataset.load_train(), dataset.load_validation()
    if len(train) != 35 or len(validation) != 8:
        raise ValueError('The fixed baseline requires35 Train and8 Validation voyages')
    reference = ROOT/'data/reference/v6_run02'
    actual = json.loads((reference/'configuration.json').read_text(encoding='utf-8'))
    if any(actual[k] != value for k, value in CONFIG.items()):
        raise ValueError('Frozen V6 configuration differs from the saved run02 reference')
    probes = json.loads((reference/'fixed_probe_definitions.json').read_text(encoding='utf-8'))
    curriculum = TrainCurriculum(train)
    if curriculum.penalty.amount != actual['failure_penalty']:
        raise ValueError('failure penalty differs from the saved run02 value')
    if resume is None:
        output.mkdir(parents=True, exist_ok=False); (output/'models').mkdir()
        model = MaskedDQN(env=DummyVecEnv([lambda:curriculum]))
        context = TrainingContext(output, curriculum, validation, model, probes=probes)
        context.report['dataset'] = dataset.task_contract
        context.report['configuration']['failure_penalty'] = curriculum.penalty.amount
        write_json(output/'configuration.json', context.report['configuration'])
        (output/'q_probes').mkdir(); write_json(output/'q_probes/definitions.json', probes)
    else:
        context = resume_context(output, resume, curriculum, validation, probes)
        model = context.model
        if context.report['completed_rounds'] >= 8:
            curriculum.close(); return context.report
        context.report['status'] = 'warmup' if curriculum.phase == 'warmup' else 'training'
    session_name = ('start' if resume is None else 'resume_'+str(model.num_timesteps)+'_'+str(time.time_ns()))
    model.set_logger(configure(str(output/'sb3_logs'/session_name), format_strings=['csv']))
    runtime_name = 'runtime.json' if resume is None else 'runtime_'+session_name+'.json'
    write_json(output/runtime_name, runtime_snapshot(model))
    if resume is None:
        snapshot = output/'source_snapshot'
        snapshot.mkdir()
        for path in Path(__file__).parent.glob('*.py'):
            shutil.copy2(path, snapshot/path.name)
        shutil.copy2(ROOT/'requirements-sb3.txt', snapshot/'requirements-sb3.txt')
    write_json(output/'report.json', context.report)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda number, frame:setattr(context, 'stop_requested', True))
    print(f'SB3 2.9.0 | CPU | fixed 8 Train passes | output: {output.resolve()}\n'
          'Only pass summaries are printed. Ctrl+C requests a checkpoint after a complete step.', flush=True)
    try:
        remaining = model.learning_starts - context.report['warmup_environment_steps']
        if remaining > 0:
            model.learn(total_timesteps=remaining, callback=context.callback,
                reset_num_timesteps=False, log_interval=None, progress_bar=False)
        if not context.report['warmup_complete']:
            context.report['warmup_adam_updates'] = model._n_updates
            if curriculum.episode_actions:
                context.report['warmup_attempts'].append(dict(sample_id=curriculum.episode.sample_id,
                    status='observed_nonterminal_prefix', inserted=len(curriculum.episode_actions)))
            context.report['warmup_environment_steps'] = model.learning_starts
            context.report['warmup_replay_fit'] = checkpoint_replay_fit(model, 0, output)
            write_probe_snapshot(model, probes, 0, output)
            context.report['warmup_complete'] = True
        if curriculum.phase == 'warmup':
            curriculum.start_formal()
            model._last_obs = model.env.reset()
        context.report['status'] = 'training'
        context.progress(); write_json(output/'report.json', context.report)
        maximum_steps = sum(len(e.load_kw) for e in train) * 8
        model.learn(total_timesteps=maximum_steps, callback=context.callback,
            reset_num_timesteps=False, log_interval=None, progress_bar=False)
        if context.report['completed_rounds'] != 8:
            raise RuntimeError('Collection bound reached before all eight Train passes completed')
    except _Finished:
        pass
    except _Pause:
        context.report['status'] = 'paused_checkpoint'
        checkpoint = context.save_checkpoint('pause_step_'+str(model.num_timesteps))
        context.progress(); write_json(output/'report.json', context.report)
        print(f'Paused safely. Resume checkpoint: {checkpoint.resolve()}', flush=True)
    except Exception as error:
        context.report.update(status='stopped_error', error=f'{type(error).__name__}: {error}')
        context.progress(); write_json(output/'report.json', context.report)
        with (output/'error.log').open('a', encoding='utf-8') as handle:
            traceback.print_exc(file=handle)
        raise
    finally:
        curriculum.close()
        signal.signal(signal.SIGINT, previous_sigint)
        model.logger.close()
    context.progress(); write_json(output/'report.json', context.report)
    return context.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data/processed/operating_dataset_v6_1s')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='complete models/round_NN or pause_step_N directory')
    args = parser.parse_args()
    resume = args.resume.resolve() if args.resume else None
    if resume and resume.suffix == '.json':
        if resume.name == 'latest_checkpoint.json':
            marker = json.loads(resume.read_text(encoding='utf-8'))
            resume = resume.parent/marker['checkpoint_name']
        else:
            resume = resume.parent
    if resume:
        output = resume.parent.parent
        if args.output_dir and args.output_dir.resolve() != output:
            parser.error('--resume must use the checkpoint original output directory')
    else:
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        output = args.output_dir or ROOT/'outputs'/f'sb3_dqn_1s_seed42_{stamp}'
    run(args.data_root, output.resolve(), resume=resume)


if __name__ == '__main__':
    main()
