"""Explicit timestep or complete formal-Episode Zoo training and optimization."""
import argparse
import json
from pathlib import Path

import torch

from .manager import (ALGO_KEY, CONFIG_PATH, ROOT, VesselExperimentManager,
    build_identity, guard_study, prepare_output, training_summary, write_json, zoo_bindings)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'optimize', 'evaluate'):
        sub = commands.add_parser(command)
        sub.add_argument('--data-root', type=Path, required=True)
        sub.add_argument('--output-dir', type=Path, required=True)
        sub.add_argument('--seed', type=int, default=42)
        if command == 'evaluate':
            sub.add_argument('--model', type=Path, required=True)
        else:
            budget = sub.add_mutually_exclusive_group(required=True)
            budget.add_argument('--n-timesteps', type=int,
                help='SB3 total Train environment steps including warmup; not an episode budget')
            budget.add_argument('--n-episodes', type=int, nargs='?', const=500,
                help='complete formal Train Episodes, excluding warmup voyages; default selected value: 500')
            sub.add_argument('--exploration-schedule', choices=('timestep', 'episode'),
                help='default follows budget mode; an explicit selection is never replaced')
            sub.add_argument('--exploration-episode-target', type=int,
                help='Episode epsilon horizon for a timestep budget (default 500)')
            sub.add_argument('--max-env-steps', type=int,
                help='Episode mode only: hard safety ceiling, never an equivalent timestep budget')
            sub.add_argument('--bounded', action='store_true',
                help='small Train/model budget; all eight real Validation voyages are still evaluated fully')
            sub.add_argument('--eval-freq', type=int)
            sub.add_argument('--eval-every-episodes', type=int,
                help='Episode mode only: full eight-voyage Validation every N formal Episodes (default 50)')
            sub.add_argument('--save-freq', type=int, default=-1)
            if command == 'optimize':
                sub.add_argument('--n-trials', type=int, default=1)
                sub.add_argument('--n-evaluations', type=int, default=1)
                sub.add_argument('--study-name', default='vessel_zoo_dqn')
                sub.add_argument('--resume-study', action='store_true')
    return parser


def resolve_budget(args, parser):
    """Validate and expose the two independent choices before any output/data IO."""
    if args.command == 'evaluate':
        return
    episode_mode = args.n_episodes is not None
    args.budget_mode = 'episode' if episode_mode else 'timestep'
    if (episode_mode and args.n_episodes < 1) or (not episode_mode and args.n_timesteps < 1):
        parser.error('positive explicit Episode or timestep budget required')
    if args.exploration_schedule is None:
        args.exploration_schedule = args.budget_mode
    if args.exploration_episode_target is not None and args.exploration_episode_target < 1:
        parser.error('--exploration-episode-target must be positive')
    if episode_mode and args.exploration_episode_target is not None:
        parser.error('Episode budget already specifies its exploration Episode target')
    args.exploration_episode_target = args.n_episodes if episode_mode else (args.exploration_episode_target or 500)
    if args.max_env_steps is not None and (not episode_mode or args.max_env_steps < 1):
        parser.error('--max-env-steps is a positive safety limit for Episode budgets only')
    if args.eval_freq is not None and (episode_mode or args.eval_freq < 1):
        parser.error('--eval-freq is a positive timestep-mode setting; use --eval-every-episodes for Episodes')
    if args.eval_every_episodes is not None and (not episode_mode or args.eval_every_episodes < 1):
        parser.error('--eval-every-episodes is a positive Episode-mode setting')
    args.eval_every_episodes = args.eval_every_episodes or 50
    if args.command == 'optimize' and (args.n_trials < 1 or args.n_evaluations < 1):
        parser.error('positive Trial/evaluation budgets are required')
    if args.bounded and ((episode_mode and args.n_episodes > 3) or
                         (not episode_mode and args.n_timesteps > 8192) or
                         (args.command == 'optimize' and args.n_trials != 1)):
        parser.error('--bounded permits at most three complete formal Episodes OR 8192 timesteps, and one Trial')


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    resolve_budget(args, parser)
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT/'outputs/zoo_vessel_dev').resolve()):
        parser.error('output-dir must stay inside this worktree outputs/zoo_vessel_dev')
    science = build_identity(args.data_root)
    run = {key:str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
           if key not in ('data_root', 'output_dir', 'resume_study')}
    identity = dict(schema=1, science=science, run=run)
    resume = getattr(args, 'resume_study', False)
    prepare_output(output, identity, resume=resume)
    torch.set_num_threads(1)
    from .algorithm import ZooMaskedDQN
    from .env import ENV_ID, register_envs
    register_envs()
    if args.command == 'evaluate':
        from stable_baselines3.common.vec_env import DummyVecEnv
        from .callbacks import evaluate_all_validation
        from .env import VesselEnergyEnv
        model_identity_path = args.model.resolve().parent/'vessel_identity.json'
        if not model_identity_path.exists():
            raise ValueError('model has no adjacent Vessel identity; refusing an unknown/legacy model')
        model_identity = json.loads(model_identity_path.read_text(encoding='utf-8'))
        if model_identity.get('science') != science:
            raise ValueError('model source/data/reward/economic identity differs')
        env = DummyVecEnv([lambda:VesselEnergyEnv(data_root=args.data_root, split='validation')])
        try:
            model = ZooMaskedDQN.load(args.model, env=env, device='cpu')
            summary = evaluate_all_validation(model, env)
            write_json(output/'validation.json', summary)
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
            return summary
        finally:
            env.close()
    hyperparams = None
    if args.bounded:
        hyperparams = dict(learning_starts=64, batch_size=32, buffer_size=8192,
            policy_kwargs='dict(net_arch=[32, 32])', target_update_interval=128)
    optimizing = args.command == 'optimize'
    storage = 'sqlite:///'+str(output/'study.sqlite3') if optimizing else None
    if optimizing:
        guard_study(storage, args.study_name, identity, resume=resume)
    with zoo_bindings(ZooMaskedDQN, bounded=args.bounded):
        # The positive placeholder is replaced from the actual Train horizon
        # during official setup_experiment.create_envs, before model.learn.
        learn_limit = args.n_timesteps if args.n_timesteps is not None else (args.max_env_steps or 1)
        manager = VesselExperimentManager(args, algo=ALGO_KEY, env_id=ENV_ID,
            data_root=args.data_root, bounded=args.bounded, log_folder=str(output),
            tensorboard_log=str(output/'tensorboard'), n_timesteps=learn_limit,
            n_episodes=args.n_episodes, exploration_schedule_mode=args.exploration_schedule,
            exploration_episode_target=args.exploration_episode_target,
            eval_every_episodes=args.eval_every_episodes, max_env_steps=args.max_env_steps,
            eval_freq=args.eval_freq or learn_limit, n_eval_episodes=8, save_freq=args.save_freq,
            hyperparams=hyperparams, optimize_hyperparameters=optimizing,
            optimization_log_path=str(output/'trials') if optimizing else None,
            storage=storage, study_name=args.study_name if optimizing else None,
            n_trials=args.n_trials if optimizing else 1,
            max_total_trials=args.n_trials if optimizing else None,
            n_evaluations=args.n_evaluations if optimizing else 1,
            n_jobs=1, sampler='tpe', pruner='none', no_optim_plots=True,
            seed=args.seed, log_interval=-2, save_replay_buffer=True,
            device='cpu', config=str(CONFIG_PATH), verbose=1)
        result = manager.setup_experiment()
        write_json(Path(manager.save_path)/'vessel_identity.json', identity)
        if result is None:
            manager.hyperparameters_optimization()
            return None
        model, _ = result
        model.final_model_path = str(Path(manager.save_path)/f'{ENV_ID}.zip')
        model.best_model_path = str(Path(manager.save_path)/'best_model.zip')
        try:
            manager.learn(model)
            if args.n_episodes is not None and model.completed_formal_episodes != args.n_episodes:
                raise RuntimeError('Episode training ended before its complete formal Episode budget')
        except BaseException:
            value = training_summary(model, requested_n_timesteps=args.n_timesteps,
                internal_safety_limit=manager.n_timesteps if args.n_episodes is not None else None)
            value['outcome'] = 'FAIL'
            write_json(output/'run_summary.json', value)
            raise
        manager.save_trained_model(model)
        best_path = Path(manager.save_path)/'best_model.zip'
        summary = training_summary(model, requested_n_timesteps=args.n_timesteps,
            internal_safety_limit=manager.n_timesteps if args.n_episodes is not None else None,
            model_path=Path(manager.save_path)/f'{ENV_ID}.zip',
            best_model_path=best_path if best_path.exists() else None)
        summary.update(outcome='COMPLETE', eval_every_episodes=args.eval_every_episodes)
        write_json(output/'run_summary.json', summary)
        return model


if __name__ == '__main__':
    main()
