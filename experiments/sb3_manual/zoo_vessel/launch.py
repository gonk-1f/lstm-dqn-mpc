"""Explicit timestep-budget Zoo training, one-study optimization and Validation."""
import argparse
import json
from pathlib import Path

import torch

from .manager import (ALGO_KEY, CONFIG_PATH, ROOT, VesselExperimentManager,
    build_identity, guard_study, prepare_output, write_json, zoo_bindings)


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
            sub.add_argument('--n-timesteps', type=int, required=True,
                help='SB3 total Train environment steps including warmup; not an episode budget')
            sub.add_argument('--bounded', action='store_true',
                help='small Train/model budget; all eight real Validation voyages are still evaluated fully')
            sub.add_argument('--eval-freq', type=int)
            sub.add_argument('--save-freq', type=int, default=-1)
            if command == 'optimize':
                sub.add_argument('--n-trials', type=int, default=1)
                sub.add_argument('--n-evaluations', type=int, default=1)
                sub.add_argument('--study-name', default='vessel_zoo_dqn')
                sub.add_argument('--resume-study', action='store_true')
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command != 'evaluate':
        if args.n_timesteps < 1 or (args.eval_freq is not None and args.eval_freq < 1):
            parser.error('positive explicit timestep and evaluation budgets are required')
        if args.command == 'optimize' and (args.n_trials < 1 or args.n_evaluations < 1):
            parser.error('positive Trial/evaluation budgets are required')
        if args.bounded and (args.n_timesteps > 8192 or
                            (args.command == 'optimize' and args.n_trials != 1)):
            parser.error('--bounded permits at most 8192 Train steps and exactly one Trial')
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
        manager = VesselExperimentManager(args, algo=ALGO_KEY, env_id=ENV_ID,
            data_root=args.data_root, bounded=args.bounded, log_folder=str(output),
            tensorboard_log=str(output/'tensorboard'), n_timesteps=args.n_timesteps,
            eval_freq=args.eval_freq or args.n_timesteps, n_eval_episodes=8, save_freq=args.save_freq,
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
        manager.learn(model)
        manager.save_trained_model(model)
        write_json(output/'run_summary.json', dict(n_timesteps=model.num_timesteps,
            requested_n_timesteps=args.n_timesteps, adam_updates=model._n_updates,
            formal_env_steps=model.formal_env_steps, replay_insertions=model.replay_buffer.insertions,
            target_sync_history=model.target_sync_history,
            model=str(Path(manager.save_path)/f'{ENV_ID}.zip'),
            budget_unit='SB3_total_env_steps_including_warmup', test_payloads_opened=0))
        return model


if __name__ == '__main__':
    main()
