"""Zoo adapter wiring and isolation; these unit tests do not train a vessel."""
from importlib import import_module

import optuna
import pytest
import yaml


def adapter():
    return import_module('experiments.sb3_manual.zoo_vessel.manager')


def test_local_yaml_freezes_mechanism_and_has_explicit_env_key():
    module = adapter()
    data = yaml.safe_load(module.CONFIG_PATH.read_text(encoding='utf-8'))['VesselEnergy-v0']
    assert data['n_envs'] == 1 and data['gamma'] == 1.0
    assert data['train_freq'] == 32 and data['gradient_steps'] == 1
    assert data['learning_starts'] == 50000 and data['batch_size'] == 64
    assert data['policy_kwargs'] == 'dict(net_arch=[256, 256])'


def test_bounded_sampler_keeps_frozen_mechanism_and_small_resources():
    module = adapter()
    trial = optuna.trial.FixedTrial(dict(learning_rate=1e-4, batch_size=32,
        buffer_size=8192, network_width=32, target_update_interval=128))
    sampled = module.sample_vessel_params(trial, 0, 1, {}, bounded=True)
    assert sampled['gamma'] == 1.0
    assert sampled['train_freq'] == 32 and sampled['gradient_steps'] == 1
    assert sampled['buffer_size'] == 8192 and sampled['policy_kwargs']['net_arch'] == [32, 32]
    assert 'one_minus_gamma' not in trial.params


def test_registration_is_scoped_and_does_not_replace_standard_dqn():
    module = adapter()
    from rl_zoo3 import exp_manager as official
    before = official.ALGOS['dqn']
    previous = official.ALGOS.get(module.ALGO_KEY)
    class FakeAlgorithm:
        pass
    with pytest.raises(RuntimeError, match='fixture'):
        with module.zoo_bindings(FakeAlgorithm, bounded=True):
            assert official.ALGOS[module.ALGO_KEY] is FakeAlgorithm
            assert official.ALGOS['dqn'] is before
            assert module.ALGO_KEY in official.HYPERPARAMS_SAMPLER
            raise RuntimeError('fixture')
    assert official.ALGOS['dqn'] is before
    assert official.ALGOS.get(module.ALGO_KEY) is previous


def test_manager_inherits_official_training_and_optimization():
    module = adapter()
    from rl_zoo3.exp_manager import ExperimentManager
    assert module.VesselExperimentManager.learn is ExperimentManager.learn
    assert module.VesselExperimentManager.hyperparameters_optimization is ExperimentManager.hyperparameters_optimization
    assert module.VesselExperimentManager.setup_experiment is ExperimentManager.setup_experiment


def test_objective_calls_official_body_with_recorded_seed_and_restores_globals(monkeypatch):
    module = adapter()
    from rl_zoo3 import exp_manager as official
    constructed = []
    class FakeAlgorithm:
        def __init__(self, *, seed):
            constructed.append(seed)
    manager = object.__new__(module.VesselExperimentManager)
    manager.seed, manager.n_jobs = 42, 1
    manager.n_timesteps, manager.bounded = 1024, True
    manager._trial_envs = []
    old_callback = official.TrialEvalCallback
    def objective(self, trial):
        assert official.TrialEvalCallback is not old_callback
        official.ALGOS[module.ALGO_KEY](seed=None)
        return 8.5
    monkeypatch.setattr(official.ExperimentManager, 'objective', objective)
    study = optuna.create_study()
    trial = study.ask()
    with module.zoo_bindings(FakeAlgorithm, bounded=True):
        assert manager.objective(trial) == 8.5
        assert official.ALGOS[module.ALGO_KEY] is FakeAlgorithm
    assert official.TrialEvalCallback is old_callback
    assert constructed == [42 + trial.number]
    assert trial.user_attrs['model_seed'] == 42 + trial.number


def test_objective_does_not_mislabel_official_caught_program_errors_as_pruning(monkeypatch):
    module = adapter()
    from rl_zoo3 import exp_manager as official
    manager = object.__new__(module.VesselExperimentManager)
    manager.seed, manager.n_jobs = 42, 1
    manager.n_timesteps, manager.bounded = 1024, True
    manager._trial_envs = []
    old_callback = official.TrialEvalCallback
    def objective(self, trial):
        raise optuna.TrialPruned() from ValueError('fixture data error')
    monkeypatch.setattr(official.ExperimentManager, 'objective', objective)
    with module.zoo_bindings(type('FakeAlgorithm', (), {})):
        with pytest.raises(RuntimeError, match='not pruning'):
            manager.objective(optuna.create_study().ask())
    assert official.TrialEvalCallback is old_callback


def test_cli_requires_step_budget_and_rejects_episode_equivalence(tmp_path):
    launch = import_module('experiments.sb3_manual.zoo_vessel.launch')
    for args in (['train'], ['optimize'], ['train', '--episodes', '500']):
        with pytest.raises(SystemExit) as error:
            launch.main([*args, '--data-root', str(tmp_path), '--output-dir', str(tmp_path/'out')])
        assert error.value.code == 2
    assert not (tmp_path/'out').exists()


def test_bounded_budget_rejects_long_training_or_multiple_trials(tmp_path):
    launch = import_module('experiments.sb3_manual.zoo_vessel.launch')
    for extra in (['--n-timesteps','8193'], ['--n-timesteps','1024','--n-trials','2']):
        with pytest.raises(SystemExit) as error:
            launch.main(['optimize','--data-root',str(tmp_path),'--output-dir',str(tmp_path/'out'),
                         '--bounded',*extra])
        assert error.value.code == 2
    assert not (tmp_path/'out').exists()


def test_output_guard_never_reuses_old_results_and_checks_resume_identity(tmp_path):
    module = adapter()
    directory = tmp_path/'new'
    identity = {'schema':1,'code':'fixture'}
    module.prepare_output(directory, identity)
    with pytest.raises(ValueError, match='new|empty|exist'):
        module.prepare_output(directory, identity)
    module.prepare_output(directory, identity, resume=True)
    with pytest.raises(ValueError, match='identity'):
        module.prepare_output(directory, {'schema':1,'code':'changed'}, resume=True)


def test_data_value_error_is_visible_and_not_an_optuna_pruning_signal():
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    module = adapter()
    class BadData(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)
        def reset(self, **kwargs):
            raise ValueError('fixture invalid economic denominator')
        def step(self, action):
            raise AssertionError('not reached')
    env = module._FatalDataErrors(DummyVecEnv([BadData]))
    try:
        with pytest.raises(RuntimeError, match='data error') as error:
            env.reset()
        assert isinstance(error.value.__cause__, ValueError)
    finally:
        env.close()


def test_study_guard_rejects_changed_identity_and_unfinished_trial(tmp_path):
    module = adapter()
    storage = 'sqlite:///'+str(tmp_path/'study.sqlite3')
    identity = {'science':'fixture'}
    study = module.guard_study(storage, 'vessel_unit', identity)
    module.guard_study(storage, 'vessel_unit', identity, resume=True)
    with pytest.raises(ValueError, match='identity'):
        module.guard_study(storage, 'vessel_unit', {'science':'changed'}, resume=True)
    study.ask()
    with pytest.raises(ValueError, match='interrupted'):
        module.guard_study(storage, 'vessel_unit', identity, resume=True)
