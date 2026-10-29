"""Direct, sequential Optuna trials around the V6 MaskedDQN/environment.

No work starts on import. Formal budgets must be supplied explicitly; --smoke
uses only a few synthetic seconds. Test has no loading route in this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import pickle
import random
import signal
import sys
from time import perf_counter, time_ns
from types import SimpleNamespace

from filelock import FileLock
import numpy as np
import optuna
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from v4.control import ReplayExecutionError, replay_episode
from v4.failure_replay import prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue
from v4.v6_dataset import V6TrainingDataset
from v4.v6_training import sample_record, split_summary, write_csv, write_json
from .curriculum import TrainCurriculum
from .environment import _Closed
from .masked_dqn import MaskedDQN
from .performance import PerformanceRecorder
from .reporting import describe_failure
from .search import sample_parameters, validation_objective
from .identity import execution_identity, dataset_identity


ROOT = Path(__file__).resolve().parents[2]


class PauseTrial(BaseException):
    pass


def execute_experiment(episode, policy, accountant, economics, penalty, reward_mode, callback=None):
    if reward_mode not in ('v6_energy_soc_ramp', 'zhou_economic_only'):
        raise ValueError('unsupported step-budget reward; use --episode-search for the one-plus mode')
    if reward_mode == 'v6_energy_soc_ramp':
        from v4.v6_training import execute
        return execute(episode, policy, accountant, economics, penalty, callback=callback)
    try:
        result = replay_episode(episode, policy, accountant=accountant,
            state_mode='history5', action_mode='increment13', reward_cost_mode='zhou_economic_only',
            beta_soc=0., redistribute_battery_energy=False,
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


class TrialCurriculum(TrainCurriculum):
    def __init__(self, train, *, reward_mode):
        super().__init__(train)
        self.reward_mode = reward_mode

    def step(self, action):
        result = super().step(action)
        # Optuna uses an explicit transition budget, not the baseline's eight
        # passes. Retain its voyage order/epsilon/physics while looping Train.
        self.schedule_finished = False
        return result

    def _run(self):
        if self.reward_mode == 'v6_energy_soc_ramp':
            return super()._run()
        try:
            callback = lambda index, transition:self._publish('transition', transition)
            values, status = execute_experiment(self.episode, self._policy, self.accountant,
                self.economics, self.penalty, self.reward_mode, callback=callback)
            self._publish('complete', (values, status))
        except _Closed:
            pass
        except BaseException as error:
            try:
                self._publish('error', error)
            except _Closed:
                pass


class TrialCallback(BaseCallback):
    def __init__(self, runner):
        super().__init__(verbose=0)
        self.runner = runner

    def _on_rollout_start(self):
        self.model.exploration_rate = self.runner.curriculum.epsilon

    def _on_step(self):
        return True  # False here would drop the collected transition.

    def _on_rollout_end(self):
        if self.runner.curriculum.phase == 'warmup':
            self.runner.consume_episodes()
            if self.runner.stop_requested(self.model):
                raise PauseTrial()


def evaluate_validation(episodes, model, curriculum, output, recorder):
    """Read-only Greedy metrics; no Replay insertion or Adam update."""
    rows, failures = [], []
    before = model._n_updates
    output.mkdir(parents=True, exist_ok=True)
    with recorder.instrument([(model, 'masked_predict', 'validation_prediction')]):
        for episode in episodes:
            with recorder.measure('validation_executor'):
                # Use the same reward/physics executor as the trial, without a
                # Gym worker. The evaluation ledger is independent of reward.
                values, status = execute_experiment(episode, lambda s,a:model.select_power(s,a,epsilon=0.),
                    curriculum.accountant, curriculum.economics, curriculum.penalty, curriculum.reward_mode)
            rows.append(sample_record(episode, values, status))
            if status != 'completed':
                failures.append(describe_failure(episode, values, status, 0,
                    'greedy_validation', 0., curriculum.accountant))
    summary = split_summary(rows)
    summary['evaluation_adam_updates'] = model._n_updates-before
    summary['mean_episode_return'] = math.fsum(row['training_return'] for row in rows)/len(rows)
    summary['score'] = validation_objective(summary)
    with recorder.measure('report_write'):
        write_csv(output/'sample_metrics.csv', rows)
        write_json(output/'summary.json', summary)
        write_json(output/'failures.json', dict(failures=failures))
    return summary


class TrialRunner:
    def __init__(self, trial, config, train, validation, directory, stop_requested, resume):
        self.trial, self.config = trial, config
        self.validation, self.directory = tuple(validation), Path(directory)
        self.resume_checkpoint = Path(resume) if resume else None
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_requested = stop_requested or (lambda model:False)
        self.profiler = PerformanceRecorder()
        self.curriculum = TrialCurriculum(train, reward_mode=config['reward_mode'])
        self.env = DummyVecEnv([lambda:self.curriculum])
        self.progress = dict(warmup_complete=False, formal_steps=0,
            evaluation_history=[], best_score=None, best_checkpoint=None,
            last_evaluated_step=0, last_pruning_checked_step=0,
            completed_train_episodes=0, failed_train_episodes=0, test_payloads_opened=0)
        if resume:
            resume = Path(resume)
            saved = torch.load(resume/'state.pt', map_location='cpu', weights_only=False)
            if saved['configuration'] != config:
                raise ValueError('Checkpoint identity/budget/reward differs from this trial.')
            self.model = MaskedDQN.load(resume/'model.zip', env=self.env, device='cpu', force_reset=False)
            self.model.load_replay_buffer(resume/'replay.pkl')
            obs, _ = self.curriculum.restore_curriculum(saved['curriculum'])
            self.model._last_obs = obs[None, :]
            self.progress = saved['progress']
            self.profiler.restore(json.loads((resume/'performance.json').read_text(encoding='utf-8')))
            random.setstate(saved['python_random']); np.random.set_state(saved['numpy_random'])
            torch.set_rng_state(saved['torch_random'])
        else:
            if config['reward_scale'] != 1.:
                raise ValueError('Reward scaling is not enabled: costs and failure penalties must be audited together.')
            if config['reward_mode'] not in ('v6_energy_soc_ramp', 'zhou_economic_only'):
                raise ValueError('Unknown audited reward mode.')
            self.model = MaskedDQN(env=self.env, seed=config['seed'], device='cpu',
                learning_starts=config['learning_starts'], **config['parameters'])
        self.model.set_logger(configure(None, []))
        self.model.after_update_observer = self.after_update
        self.callback = TrialCallback(self)
        write_json(self.directory/'configuration.json', config)

    def consume_episodes(self):
        while self.curriculum.pending_completed:
            item = self.curriculum.pending_completed.popleft()
            if item['status'] != 'completed':
                self.model.finalize_streamed_failure_prefix(range(item['replay_start'], self.model.replay.insertions))
                self.progress['failed_train_episodes'] += 1
            else:
                self.progress['completed_train_episodes'] += 1

    def after_update(self, model):
        self.consume_episodes()
        self.progress['formal_steps'] = int(model._n_updates)
        if self.stop_requested(model):
            raise PauseTrial()

    def checkpoint(self, *, as_best=False):
        folder = self.directory/'checkpoints'/f'step_{self.model.num_timesteps}_{time_ns()}'
        folder.mkdir(parents=True)
        if as_best:
            self.progress['best_checkpoint'] = str(folder.resolve())
        # Instrumentation contexts have exited before saving: no timed wrappers
        # or profiler closures are pickled with the SB3 model.
        with self.profiler.measure('checkpoint_write'):
            self.model.save(folder/'model.zip')
            self.model.save_replay_buffer(folder/'replay.pkl')
            state = dict(configuration=self.config, progress=self.progress,
                curriculum=self.curriculum.curriculum_state(), performance=self.profiler.snapshot(),
                python_random=random.getstate(), numpy_random=np.random.get_state(), torch_random=torch.get_rng_state())
            torch.save(state, folder/'state.pt')
        write_json(folder/'performance.json', self.profiler.snapshot())
        marker = dict(status='complete', directory=str(folder.resolve()),
                      formal_steps=self.progress['formal_steps'], reward_mode=self.config['reward_mode'])
        write_json(folder/'checkpoint.json', marker)
        write_json(self.directory/'latest_checkpoint.json', marker)
        return folder

    def collect(self, count):
        stage = 'warmup_collect_total' if self.curriculum.phase=='warmup' else 'formal_collect_total'
        with self.profiler.measure(stage):
            with self.profiler.instrument(self.profiler.model_bindings(self.model, self.curriculum)):
                self.model.learn(count, callback=self.callback, reset_num_timesteps=False,
                                 progress_bar=False, log_interval=None)

    def run(self):
        checkpoint = self.resume_checkpoint
        try:
            if not self.progress['warmup_complete']:
                remaining = self.config['learning_starts']-self.model.num_timesteps
                if remaining > 0:
                    self.collect(remaining)
                self.curriculum.start_formal()
                self.model._last_obs = self.env.reset()
                self.progress['warmup_complete'] = True
            while True:
                formal = self.progress['formal_steps']
                # A safe pause can occur immediately after the boundary Adam,
                # before Validation. Resume must evaluate that trained state
                # first, including when it is the final budget step.
                due = formal > self.progress['last_evaluated_step'] and (
                    formal % self.config['eval_every']==0 or formal==self.config['train_steps'])
                if due:
                    summary = evaluate_validation(self.validation, self.model, self.curriculum,
                        self.directory/f'validation_step_{formal}', self.profiler)
                    entry = dict(formal_steps=formal, score=summary['score'],
                        completed=summary['completed'], episodes=summary['episodes'],
                        coverage=summary['mean_duration_coverage'],
                        full_set_cost_cny=summary['fixed_complete_set_cost_cny'],
                        evaluation_adam_updates=summary['evaluation_adam_updates'])
                    self.progress['evaluation_history'].append(entry)
                    self.progress['last_evaluated_step'] = formal
                    improved = self.progress['best_score'] is None or summary['score'] > self.progress['best_score']
                    if improved:
                        self.progress['best_score'] = summary['score']
                    checkpoint = self.checkpoint(as_best=improved)
                # A checkpoint is committed before Optuna report/prune. Replay
                # that pending decision on resume before collecting any action.
                evaluated = self.progress['last_evaluated_step']
                if evaluated > self.progress.get('last_pruning_checked_step', 0):
                    entry = self.progress['evaluation_history'][-1]
                    reported = self.trial.storage.get_trial(self.trial._trial_id).intermediate_values
                    if evaluated not in reported:
                        self.trial.report(entry['score'], evaluated)
                    if self.trial.should_prune():
                        write_json(self.directory/'result.json', self.result('pruned', checkpoint))
                        raise optuna.TrialPruned(f'Validation pruned at {evaluated} formal updates')
                    self.progress['last_pruning_checked_step'] = evaluated
                    if self.stop_requested(self.model):
                        raise PauseTrial()
                if formal >= self.config['train_steps']:
                    break
                next_step = min(((formal//self.config['eval_every'])+1)*self.config['eval_every'], self.config['train_steps'])
                self.collect(next_step-formal)
            if checkpoint is None:
                checkpoint = self.checkpoint()
            result = self.result('completed', checkpoint)
            write_json(self.directory/'result.json', result)
            return result
        except PauseTrial:
            checkpoint = self.checkpoint()
            result = self.result('paused', checkpoint)
            write_json(self.directory/'result.json', result)
            return result
        finally:
            write_json(self.directory/'performance.json', self.profiler.snapshot())
            self.curriculum.close()

    def result(self, status, checkpoint):
        return dict(status=status, checkpoint=str(checkpoint.resolve()),
            best_checkpoint=self.progress['best_checkpoint'], best_score=self.progress['best_score'],
            adam_updates=int(self.model._n_updates), replay_insertions=self.model.replay.insertions,
            test_payloads_opened=0, evaluation_history=list(self.progress['evaluation_history']))


def train_trial(trial, config, train, validation, directory, *, stop_requested=None, resume=None):
    torch.set_num_threads(1)
    return TrialRunner(trial, config, train, validation, directory, stop_requested, resume).run()


def make_smoke_episodes():
    def make(name, split, loads):
        count = len(loads)
        return SimpleNamespace(sample_id=name, split=split, task_mode='independent_voyage',
            operating_mode=('onboard',)*count, load_kw=tuple(loads),
            time_s=tuple(float(i) for i in range(count)), dt_s=(1.,)*count, duration_s=float(count))
    return ((make('synthetic_train_a','train',(80.,100.,60.,80.)),
             make('synthetic_train_b','train',(0.,20.,10.,0.))),
            (make('synthetic_validation_a','validation',(80.,80.,80.)),
             make('synthetic_validation_b','validation',(100.,90.,100.))))


def validate_allocation(study, active, trial_budget):
    allocated = study.get_trials(deepcopy=False)
    running = [trial for trial in allocated if not trial.state.is_finished()]
    if len(allocated)>trial_budget:
        raise ValueError('Allocated trial count exceeds this fixed study budget.')
    if running and (active is None or len(running)!=1 or running[0]._trial_id!=active['trial_id']):
        raise ValueError('Orphan/foreign RUNNING trial: refusing to allocate another trial.')
    if active is not None and not running:
        raise ValueError('Active marker has no matching RUNNING trial.')


def atomic_pickle(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(pickle.dumps(value))
    temporary.replace(path)


def trial_configuration(trial, identity, data_identity):
    """Rebuild the immutable trial contract from study identity and SQLite."""
    params = trial.params
    expected = {'network_width', 'learning_rate', 'batch_size', 'buffer_size', 'target_update_interval'}
    if set(params) != expected:
        raise ValueError('Trial configuration has incomplete or foreign sampled parameters.')
    parameters = {name:params[name] for name in expected if name!='network_width'}
    parameters['policy_kwargs'] = dict(net_arch=[params['network_width']]*2)
    return dict(experiment_id=('synthetic_' if identity['smoke'] else '')+'v6_optuna_'+identity['reward_mode'],
        reward_mode=identity['reward_mode'], reward_scale=identity['reward_scale'], seed=identity['seed'],
        train_steps=identity['train_steps'], learning_starts=identity['learning_starts'],
        eval_every=identity['eval_every'], parameters=parameters,
        dataset_identity=data_identity, baseline_commit=identity['baseline_commit'],
        execution_identity=identity['execution_identity'])


def run_study(directory, **kwargs):
    if kwargs.get('reward_mode', 'v6_energy_soc_ramp') not in ('v6_energy_soc_ramp', 'zhou_economic_only'):
        raise ValueError('unsupported step-budget reward; use --episode-search for the one-plus mode')
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    # SQLite search is intentionally single-writer, including separate CLI
    # processes. An OS file lock is released on exit/crash.
    with FileLock(str(directory)+'.lock', timeout=0):
        return _run_study(directory, **kwargs)


def _run_study(directory, *, trials, train_steps, learning_starts, eval_every, seed=42,
              sampler_seed=42, smoke=False, reward_mode='v6_energy_soc_ramp', resume=False, pruner='median'):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=resume)
    identity = dict(baseline_commit='544908d', reward_mode=reward_mode, reward_scale=1.,
        seed=seed, sampler_seed=sampler_seed, train_steps=train_steps,
        learning_starts=learning_starts, eval_every=eval_every, smoke=smoke,
        action_mode='increment13', model_step_s=1, gamma=1., n_step=1,
        train_freq=1, gradient_steps=1, data_order_seed=42, pruner=pruner, trial_budget=trials)
    from importlib.metadata import version
    write_json(directory/('runtime_resume_'+str(time_ns())+'.json' if resume else 'runtime.json'),
        dict(python_executable=sys.executable, python_version=sys.version,
             pytorch_version=torch.__version__, stable_baselines3_version=version('stable-baselines3'),
             optuna_version=optuna.__version__, device='cpu', n_jobs=1,
             environment='isolated .venv-optuna; shared frozen scientific packages',
             sampler='TPESampler', sampler_seed=sampler_seed, model_seed=seed))
    if smoke:
        train, validation = make_smoke_episodes()
        data_identity = 'synthetic_smoke_only'
    else:
        dataset = V6TrainingDataset.open(ROOT/'data/processed/operating_dataset_v6_1s')
        train, validation = dataset.load_train(), dataset.load_validation()
        if (len(train),len(validation)) != (35,8):
            raise ValueError('Formal trials require the frozen 35/8 Train/Validation split.')
        data_identity = dataset_identity(dataset)
    identity['execution_identity'] = execution_identity(ROOT, data_identity)
    identity_path = directory/'study_identity.json'
    if resume:
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Resume must keep data, code, dependencies, economics, reward, budget and seeds unchanged.')
    else:
        write_json(identity_path, identity)
    sampler_path = directory/'sampler.pkl'
    sampler = pickle.loads(sampler_path.read_bytes()) if resume else optuna.samplers.TPESampler(seed=sampler_seed)
    pruning = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=eval_every) if pruner=='median' else optuna.pruners.NopPruner()
    study = optuna.create_study(study_name='v6_masked_dqn',
        storage='sqlite:///'+str(directory/'study.sqlite3'), direction='maximize',
        sampler=sampler, pruner=pruning, load_if_exists=resume)
    stop = [False]
    previous_signal = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda number,frame:stop.__setitem__(0,True))
    active_path = directory/'active_trial.json'
    try:
        while True:
            active = json.loads(active_path.read_text(encoding='utf-8')) if resume and active_path.exists() else None
            validate_allocation(study, active, trials)
            if active is None and len(study.trials)>=trials:
                break
            if active:
                # Trial's constructor is the pinned Optuna API for recovering
                # a persisted RUNNING trial; its integer ID is stored on ask.
                trial = optuna.trial.Trial(study, active['trial_id'])
                config = json.loads((directory/active['trial_directory']/'configuration.json').read_text(encoding='utf-8'))
                if config != trial_configuration(trial, identity, data_identity):
                    raise ValueError('Active trial configuration differs from the fixed study identity or sampled parameters.')
                checkpoint = json.loads((directory/active['trial_directory']/'latest_checkpoint.json').read_text(encoding='utf-8'))['directory']
            else:
                trial = study.ask()
                if smoke:
                    width = trial.suggest_categorical('network_width', [32])
                    parameters = dict(learning_rate=trial.suggest_categorical('learning_rate',[.0001]),
                        batch_size=trial.suggest_categorical('batch_size',[8]),
                        buffer_size=trial.suggest_categorical('buffer_size',[64]),
                        target_update_interval=trial.suggest_categorical('target_update_interval',[4]),
                        policy_kwargs=dict(net_arch=[width,width]))
                else:
                    parameters = sample_parameters(trial)
                config = trial_configuration(trial, identity, data_identity)
                active = dict(trial_id=trial._trial_id, trial_number=trial.number,
                              trial_directory=f'trial_{trial.number:04d}')
                write_json(active_path, active)
                atomic_pickle(sampler_path, study.sampler)
                checkpoint = None
            trial.set_user_attr('seed', seed)
            trial.set_user_attr('experiment_id', config['experiment_id'])
            trial.set_user_attr('trial_directory', active['trial_directory'])
            try:
                result = train_trial(trial, config, train, validation, directory/active['trial_directory'],
                    resume=checkpoint, stop_requested=lambda model:stop[0])
                if result['status']=='paused':
                    print(f'Trial {trial.number} paused safely; resume the same study.', flush=True)
                    break
                trial.set_user_attr('best_checkpoint', result['best_checkpoint'])
                trial.set_user_attr('validation_history', result['evaluation_history'])
                study.tell(trial, result['best_score'])
                print(f'Trial {trial.number} completed | score {result["best_score"]:.6f}', flush=True)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f'Trial {trial.number} pruned.', flush=True)
            except Exception:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                active_path.unlink()
                atomic_pickle(sampler_path, study.sampler)
                study.trials_dataframe().to_csv(directory/'trials.csv', index=False, encoding='utf-8-sig')
                raise
            active_path.unlink()
            atomic_pickle(sampler_path, study.sampler)
            study.trials_dataframe().to_csv(directory/'trials.csv', index=False, encoding='utf-8-sig')
        complete = [trial for trial in study.trials if trial.state==optuna.trial.TrialState.COMPLETE]
        if complete:
            best = study.best_trial
            write_json(directory/'selected_trial.json', dict(trial_number=best.number,
                score=best.value, parameters=best.params, user_attrs=best.user_attrs,
                selection_data='Train updates and Validation only; Test payloads opened: 0'))
    finally:
        signal.signal(signal.SIGINT, previous_signal)
    return study


def main():
    if '--episode-search' in sys.argv[1:]:
        from .episode_tuning import main as episode_main
        return episode_main([arg for arg in sys.argv[1:] if arg!='--episode-search'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-search', action='store_true',
                        help='use the shared economic episode runner; combine with --help for its options')
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='one tiny synthetic trial; no formal data')
    parser.add_argument('--trials', type=int)
    parser.add_argument('--train-steps', type=int, help='formal transition/Adam budget per trial, excluding warmup')
    parser.add_argument('--learning-starts', type=int)
    parser.add_argument('--eval-every', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampler-seed', type=int, default=42)
    parser.add_argument('--reward-mode', choices=('v6_energy_soc_ramp','zhou_economic_only'), default='v6_energy_soc_ramp')
    parser.add_argument('--pruner', choices=('median','none'), default='median')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.smoke and (args.trials is None or args.train_steps is None):
        parser.error('Formal search requires explicit --trials and --train-steps; no automatic long-training budget.')
    trials = args.trials if args.trials is not None else 1
    steps = args.train_steps if args.train_steps is not None else 16
    warmup = args.learning_starts if args.learning_starts is not None else (4 if args.smoke else 50_000)
    interval = args.eval_every if args.eval_every is not None else (8 if args.smoke else 100_000)
    if min(trials,steps,interval)<1 or warmup<1:
        parser.error('Budgets/trials/evaluation interval must be positive.')
    if args.smoke and (trials>2 or steps>64 or warmup>32):
        parser.error('Smoke is capped at two trials, 64 formal steps and 32 warmup steps.')
    run_study(args.study_dir, trials=trials, train_steps=steps, learning_starts=warmup,
        eval_every=interval, seed=args.seed, sampler_seed=args.sampler_seed,
        smoke=args.smoke, reward_mode=args.reward_mode, resume=args.resume, pruner=args.pruner)


if __name__=='__main__':
    main()
