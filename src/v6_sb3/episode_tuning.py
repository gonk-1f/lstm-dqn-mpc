"""Episode-budget Optuna adapter using the shared direct-environment runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import signal
from types import SimpleNamespace

from filelock import FileLock
import optuna

from .identity import dataset_identity, execution_identity

ROOT = Path(__file__).resolve().parents[2]
STUDY_NAME = 'v6_economic_episode_dqn'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_sampler(path, sampler):
    temporary = path.with_suffix('.pkl.tmp')
    temporary.write_bytes(pickle.dumps(sampler))
    temporary.replace(path)


def run_episode_training(*args, **kwargs):
    from .episode_training import run_episode_training as run
    return run(*args, **kwargs)


def synthetic_episodes():
    def make(name, split, loads):
        n = len(loads)
        return SimpleNamespace(sample_id=name, split=split, task_mode='independent_voyage',
            operating_mode=('onboard',)*n, load_kw=tuple(loads), time_s=tuple(map(float, range(n))),
            dt_s=(1.,)*n, duration_s=float(n))
    return ((make('synthetic_episode_train_a','train',(80.,100.,60.,80.)*16),
             make('synthetic_episode_train_b','train',(0.,20.,10.,0.)*16)),
            (make('synthetic_episode_val_a','validation',(80.,)*8),
             make('synthetic_episode_val_b','validation',(100.,90.)*4)))


def inputs(smoke):
    if smoke:
        train, validation = synthetic_episodes()
        return train, validation, 'synthetic_episode_smoke_64s_v1'
    from v4.v6_dataset import V6TrainingDataset
    data = V6TrainingDataset.open(ROOT/'data/processed/operating_dataset_v6_1s')
    train, validation = data.load_train(), data.load_validation()
    if (len(train), len(validation)) != (35, 8):
        raise ValueError('Episode trials require the frozen 35/8 Train/Validation split.')
    return train, validation, dataset_identity(data)


def sample_parameters(trial, smoke=False):
    if smoke:
        return dict(learning_rate=trial.suggest_categorical('learning_rate', [1e-4]),
            batch_size=trial.suggest_categorical('batch_size', [8]),
            buffer_size=trial.suggest_categorical('buffer_size', [256]),
            network_width=trial.suggest_categorical('network_width', [16]),
            target_update_interval=trial.suggest_categorical('target_update_interval', [32]))
    return dict(learning_rate=trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
        batch_size=trial.suggest_categorical('batch_size', [32,64,128]),
        buffer_size=trial.suggest_categorical('buffer_size', [100000,300000,1000000]),
        network_width=trial.suggest_categorical('network_width', [128,256]),
        target_update_interval=trial.suggest_categorical('target_update_interval', [5000,10000,20000]))


def trial_config(identity, parameters):
    return dict(episodes=identity['episodes_per_trial'], learning_starts=identity['learning_starts'],
        train_freq=32, gradient_steps=1, validation_every=identity['validation_every'],
        reward_mode='zhou_economic_only', seed=identity['seed'], order_seed=identity['order_seed'],
        smoke=identity['smoke'], acknowledge_failure_penalty=identity['acknowledge_failure_penalty'],
        **parameters)


def _export(study, directory):
    study.trials_dataframe().to_csv(directory/'trials.csv', index=False, encoding='utf-8-sig')
    if any(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials):
        best = study.best_trial
        write_json(directory/'selected_trial.json', dict(number=best.number, score=best.value,
            parameters=best.params, user_attrs=best.user_attrs, selection_split='validation'))


def run_study(directory, *, trials, episodes_per_trial, learning_starts=50000,
              validation_every=50, seed=42, order_seed=42, sampler_seed=42,
              smoke=False, acknowledge_failure_penalty=False, resume=False, pruner='median',
              stop_after_steps=None):
    if not smoke and not acknowledge_failure_penalty:
        raise ValueError('Review the Train-only failure penalty audit and explicitly acknowledge the legacy failure penalty before formal search.')
    if min(trials, episodes_per_trial, validation_every) < 1 or learning_starts < 1:
        raise ValueError('Episode, Trial, warmup and Validation budgets must be positive.')
    if smoke and (trials>2 or episodes_per_trial>3 or learning_starts>64):
        raise ValueError('Episode smoke is limited to 2 Trials, 3 episodes and 64 warmup steps.')
    if stop_after_steps is not None and stop_after_steps < 0:
        raise ValueError('The per-invocation pause budget cannot be negative.')
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory)+'.lock', timeout=0):
        directory.mkdir(parents=True, exist_ok=resume)
        train, validation, data = inputs(smoke)
        identity = dict(schema=1, base_commit='b212483', episodes_per_trial=episodes_per_trial,
            trial_budget=trials, learning_starts=learning_starts, validation_every=validation_every,
            seed=seed, order_seed=order_seed, sampler_seed=sampler_seed, smoke=smoke, pruner=pruner,
            acknowledge_failure_penalty=bool(acknowledge_failure_penalty or smoke),
            reward_mode='zhou_economic_only', gamma=1., train_freq=32, gradient_steps=1,
            target_sync_unit='formal_env_steps', execution=execution_identity(ROOT, data))
        identity_path = directory/'study_identity.json'
        sampler_path = directory/'sampler.pkl'
        if resume:
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('Study identity changed; keep budgets, seeds, data, code and economics fixed.')
            sampler = pickle.loads(sampler_path.read_bytes())
        else:
            write_json(identity_path, identity)
            sampler = optuna.samplers.TPESampler(seed=sampler_seed)
            save_sampler(sampler_path, sampler)
        pruning = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=validation_every)
                   if pruner=='median' else optuna.pruners.NopPruner())
        study = optuna.create_study(study_name=STUDY_NAME, storage='sqlite:///'+str(directory/'study.sqlite3'),
            direction='maximize', sampler=sampler, pruner=pruning, load_if_exists=resume)
        stopped = [False]
        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, lambda *args:stopped.__setitem__(0, True))
        active_path = directory/'active_trial.json'
        try:
            while True:
                active = json.loads(active_path.read_text(encoding='utf-8')) if active_path.exists() else None
                allocated = study.get_trials(deepcopy=False)
                running = [t for t in allocated if not t.state.is_finished()]
                if len(allocated)>trials:
                    raise ValueError('Allocated Trials exceed the fixed study budget.')
                if running and (not active or len(running)!=1 or running[0]._trial_id!=active['trial_id']):
                    raise ValueError('Orphan RUNNING Trial; refusing a replacement allocation.')
                if active and not running:
                    raise ValueError('Active marker does not match a RUNNING Trial.')
                if not active and len(allocated)>=trials:
                    break
                continuing = active is not None
                if continuing:
                    trial = optuna.trial.Trial(study, active['trial_id'])
                    config = trial_config(identity, trial.params)
                    if json.loads((directory/(active['directory']+'_configuration.json')).read_text(encoding='utf-8'))!=config:
                        raise ValueError('Active Trial configuration differs from the study identity.')
                else:
                    trial = study.ask()
                    config = trial_config(identity, sample_parameters(trial, smoke))
                    active = dict(trial_id=trial._trial_id, number=trial.number, directory=f'trial_{trial.number:04d}')
                    folder = directory/active['directory'];folder.mkdir()
                    write_json(directory/(active['directory']+'_configuration.json'), config)
                    write_json(active_path, active)
                    save_sampler(sampler_path, study.sampler)
                trial.set_user_attr('model_seed', seed)
                trial.set_user_attr('order_seed', order_seed)
                trial.set_user_attr('budget_episodes', episodes_per_trial)
                trial.set_user_attr('target_sync_unit', 'formal_env_steps')

                def evaluate(summary, progress):
                    step = progress['completed_episodes']
                    if step not in trial.storage.get_trial(trial._trial_id).intermediate_values:
                        trial.report(summary['score'], step)
                    trial.set_user_attr('latest_validation', summary)
                    trial.set_user_attr('latest_progress', progress)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f'Episode {step}: Validation pruning')

                try:
                    result = run_episode_training(config, train, validation, directory/active['directory'],
                        resume=continuing, stop_requested=lambda *args:stopped[0], on_validation=evaluate,
                        stop_after_steps=stop_after_steps)
                    if result['status']=='paused':
                        save_sampler(sampler_path, study.sampler)
                        _export(study, directory)
                        print(f'Trial {trial.number} paused safely; resume the same episode study.', flush=True)
                        break
                    trial.set_user_attr('best_checkpoint', result['best_checkpoint'])
                    trial.set_user_attr('validation_history', result['validation_history'])
                    study.tell(trial, result['best_score'])
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                except Exception:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    active_path.unlink()
                    save_sampler(sampler_path, study.sampler)
                    _export(study, directory)
                    raise
                active_path.unlink()
                save_sampler(sampler_path, study.sampler)
                _export(study, directory)
            _export(study, directory)
        finally:
            signal.signal(signal.SIGINT, previous)
        return study


def export_best(directory, output):
    directory = Path(directory).resolve()
    identity = json.loads((directory/'study_identity.json').read_text(encoding='utf-8'))
    study = optuna.load_study(study_name=STUDY_NAME, storage='sqlite:///'+str(directory/'study.sqlite3'))
    best = study.best_trial
    value = dict(schema='v6_episode_parameters_v1', parameters=best.params,
        target_sync_unit='formal_env_steps', reward_mode='zhou_economic_only',
        source_study=str(directory), source_trial=best.number, source_score=best.value,
        source_budget_episodes=identity['episodes_per_trial'],
        retraining='Initialize a new model and Replay; do not load the Trial checkpoint.')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_json(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--trials', type=int)
    parser.add_argument('--episodes-per-trial', type=int)
    parser.add_argument('--learning-starts', type=int)
    parser.add_argument('--validation-every', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--order-seed', type=int, default=42)
    parser.add_argument('--sampler-seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-steps', type=int,
                        help='request a safe resumable pause after N transitions in this invocation')
    parser.add_argument('--acknowledge-failure-penalty', action='store_true')
    parser.add_argument('--pruner', choices=('median','none'), default='median')
    parser.add_argument('--export-best-parameters', type=Path)
    args = parser.parse_args(argv)
    if args.export_best_parameters:
        export_best(args.study_dir, args.export_best_parameters)
        print(f'Fresh-retraining parameters written to {args.export_best_parameters}')
        return
    if not args.smoke and (args.trials is None or args.episodes_per_trial is None):
        parser.error('Formal episode search requires explicit --trials and --episodes-per-trial.')
    run_study(args.study_dir, trials=args.trials if args.trials is not None else 1,
        episodes_per_trial=args.episodes_per_trial if args.episodes_per_trial is not None else 2,
        learning_starts=args.learning_starts if args.learning_starts is not None else (4 if args.smoke else 50000),
        validation_every=args.validation_every if args.validation_every is not None else (1 if args.smoke else 50),
        seed=args.seed, order_seed=args.order_seed, sampler_seed=args.sampler_seed, smoke=args.smoke,
        acknowledge_failure_penalty=args.acknowledge_failure_penalty, resume=args.resume, pruner=args.pruner,
        stop_after_steps=args.stop_after_steps)


if __name__=='__main__':
    main()
