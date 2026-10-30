"""Budget/official-Zoo wiring tests; no vessel payloads or long training."""
from importlib import import_module
from types import SimpleNamespace

import optuna
import pytest


def modules():
    return (import_module('experiments.sb3_manual.zoo_vessel.launch'),
            import_module('experiments.sb3_manual.zoo_vessel.manager'))


def parsed(*options):
    launch, _ = modules()
    parser = launch._parser()
    args = parser.parse_args(['train', '--data-root', 'data', '--output-dir', 'out', *options])
    assert hasattr(launch, 'resolve_budget'), 'budget resolution must be explicit and testable'
    launch.resolve_budget(args, parser)
    return args


def test_episode_budget_defaults_to_500_only_when_explicitly_selected():
    args = parsed('--n-episodes')
    assert args.n_episodes == 500 and args.n_timesteps is None
    assert args.budget_mode == args.exploration_schedule == 'episode'
    assert args.eval_every_episodes == 50


@pytest.mark.parametrize('options', [(), ('--n-episodes', '3', '--n-timesteps', '64')])
def test_budget_choice_is_required_and_mutually_exclusive(options):
    launch, _ = modules()
    with pytest.raises(SystemExit) as error:
        launch._parser().parse_args(['train', '--data-root', 'data', '--output-dir', 'out', *options])
    assert error.value.code == 2


def test_original_timestep_default_and_explicit_cross_schedule_are_retained():
    args = parsed('--n-timesteps', '4096')
    assert args.budget_mode == args.exploration_schedule == 'timestep'
    assert args.n_episodes is None
    explicit = parsed('--n-timesteps', '4096', '--exploration-schedule', 'episode')
    assert explicit.exploration_schedule == 'episode' and explicit.exploration_episode_target == 500
    explicit = parsed('--n-episodes', '3', '--exploration-schedule', 'timestep')
    assert explicit.exploration_schedule == 'timestep'


@pytest.mark.parametrize('options', [('--n-episodes', '0'), ('--n-episodes', '-1'),
    ('--n-episodes', '4', '--bounded'), ('--n-episodes', '3', '--max-env-steps', '0'),
    ('--n-timesteps', '32', '--max-env-steps', '33'),
    ('--n-timesteps', '32', '--eval-every-episodes', '10'),
    ('--n-episodes', '3', '--eval-freq', '1000')])
def test_invalid_or_ambiguous_budget_options_fail_before_loading_data(options):
    with pytest.raises(SystemExit) as error:
        parsed(*options)
    assert error.value.code == 2


def test_episode_safety_limit_is_a_mathematical_bound_not_a_timestep_estimate():
    _, manager = modules()
    assert hasattr(manager, 'episode_safety_limit'), 'explicit safety bound function required'
    assert manager.episode_safety_limit(50000, 500, 30000) == 15080001
    assert manager.episode_safety_limit(64, 3, 30) == 185
    for values in ((-1, 3, 30), (64, 0, 30), (64, 3, 0)):
        with pytest.raises(ValueError):
            manager.episode_safety_limit(*values)


def test_summary_distinguishes_budget_and_safety_limit_and_reports_actual_clocks(tmp_path):
    _, manager = modules()
    assert hasattr(manager, 'training_summary'), 'one truthful summary schema required'
    model = SimpleNamespace(num_timesteps=102, learning_starts=5, formal_env_steps=97,
        completed_formal_episodes=3, formal_successes=1, formal_failures=2,
        warmup_env_steps=5, warmup_bridge_env_steps=3, n_episodes=3,
        budget_mode='episode', exploration_schedule_mode='episode',
        exploration_episode_target=3, exploration_rate=.05, _n_updates=3,
        replay_buffer=SimpleNamespace(insertions=102), target_sync_calls=7,
        target_sync_history=[1, 2, 3, 4, 5, 6],
        last_validation_summary=dict(completion_rate=.125, completed_formal_episodes=3))
    value = manager.training_summary(model, requested_n_timesteps=None,
        internal_safety_limit=1000, model_path=tmp_path/'final.zip', best_model_path=tmp_path/'best.zip')
    assert value['budget_mode'] == 'episode' and value['requested_n_episodes'] == 3
    assert value['requested_n_timesteps'] is None and value['internal_safety_limit'] == 1000
    assert value['n_timesteps'] == value['replay_insertions'] == 102
    assert value['formal_env_steps'] == 97 and value['adam_updates'] == 3
    assert value['train_successes'] == 1 and value['train_failures'] == 2
    assert value['target_sync_count'] == 6 and value['target_sync_calls_including_initial'] == 7
    assert value['final_epsilon'] == .05 and value['final_validation_completion_rate'] == .125


def test_manager_does_not_replace_official_learning_or_optuna_dispatch():
    _, manager = modules()
    from rl_zoo3.exp_manager import ExperimentManager
    assert manager.VesselExperimentManager.learn is ExperimentManager.learn
    assert manager.VesselExperimentManager.setup_experiment is ExperimentManager.setup_experiment
    assert manager.VesselExperimentManager.hyperparameters_optimization is ExperimentManager.hyperparameters_optimization


def test_episode_trial_passes_controls_through_official_objective_and_records_budget(monkeypatch):
    _, manager = modules()
    from rl_zoo3 import exp_manager as official
    owner = object.__new__(manager.VesselExperimentManager)
    owner.seed, owner.n_jobs = 42, 1
    owner.n_timesteps, owner.bounded = 1000, True
    owner.n_episodes, owner.budget_mode = 3, 'episode'
    owner.exploration_schedule_mode, owner.exploration_episode_target = 'episode', 3
    owner.eval_every_episodes, owner._trial_envs = 50, []
    seen = []
    class FakeAlgorithm:
        def __init__(self, **kwargs):
            seen.append(kwargs)
    def objective(self, trial):
        official.ALGOS[manager.ALGO_KEY](seed=None)
        return 2.5
    monkeypatch.setattr(official.ExperimentManager, 'objective', objective)
    trial = optuna.create_study(direction='maximize').ask()
    with manager.zoo_bindings(FakeAlgorithm):
        assert owner.objective(trial) == 2.5
    assert seen[0]['n_episodes'] == 3 and seen[0]['exploration_schedule_mode'] == 'episode'
    assert trial.user_attrs['budget_unit'] == 'complete_formal_train_episodes'
    assert trial.user_attrs['n_episodes'] == 3
    assert trial.user_attrs['internal_safety_limit'] == 1000


@pytest.mark.parametrize('outcome', ['COMPLETE', 'PRUNED', 'FAIL', 'PRUNED_RETURN'])
def test_trial_outcome_and_artifacts_cannot_claim_incomplete_episode_training_complete(
        monkeypatch, tmp_path, outcome):
    """The official body still dispatches learn; test only the adapter's guards."""
    _, manager = modules()
    from rl_zoo3 import exp_manager as official
    owner = object.__new__(manager.VesselExperimentManager)
    owner.seed, owner.n_jobs, owner.n_timesteps, owner.bounded = 42, 1, 1000, True
    owner.n_episodes, owner.budget_mode = 3, 'episode'
    if outcome == 'PRUNED_RETURN':
        owner.n_episodes, owner.budget_mode = None, 'timestep'
    owner.exploration_schedule_mode, owner.exploration_episode_target = 'episode', 3
    owner.eval_every_episodes, owner._trial_envs = 50, []
    owner.log_folder, owner.optimization_log_path = str(tmp_path), str(tmp_path/'trials')
    manager.write_json(tmp_path/'vessel_identity.json', {'science':'fixture'})
    class FakeAlgorithm:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.num_timesteps, self.formal_env_steps, self.learning_starts = 5, 0, 5
            self.completed_formal_episodes, self.formal_successes, self.formal_failures = 0, 0, 0
            self.warmup_env_steps, self.warmup_bridge_env_steps = 5, 0
            self.budget_mode, self.exploration_rate, self._n_updates = owner.budget_mode, 1., 0
            self.replay_buffer = SimpleNamespace(insertions=5)
            self.target_sync_calls, self.target_sync_history = 1, []
            self.last_validation_summary = None
        def learn(self, *args, **kwargs):
            if outcome == 'PRUNED':
                raise optuna.TrialPruned('fixture physical evaluation prune')
            if outcome == 'COMPLETE':
                self.completed_formal_episodes = 3
            return self
        def save(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture model marker', encoding='utf-8')
    def objective(self, trial):
        model = official.ALGOS[manager.ALGO_KEY](seed=None)
        callback = SimpleNamespace(is_pruned=outcome == 'PRUNED_RETURN')
        model.learn(self.n_timesteps, callback=[callback])
        if callback.is_pruned:
            # This is the installed Zoo objective's post-learn pruning order.
            raise optuna.TrialPruned()
        return .5
    monkeypatch.setattr(official.ExperimentManager, 'objective', objective)
    study = optuna.create_study(direction='maximize')
    with manager.zoo_bindings(FakeAlgorithm):
        study.optimize(owner.objective, n_trials=1, catch=(RuntimeError,))
    trial = study.trials[0]
    expected_outcome = 'PRUNED' if outcome == 'PRUNED_RETURN' else outcome
    assert trial.state.name == expected_outcome
    assert trial.user_attrs['training_summary']['outcome'] == expected_outcome
    assert trial.user_attrs['training_summary']['n_timesteps'] == 5
    assert trial.user_attrs['training_summary']['final_validation_completion_rate'] is None
    folder = tmp_path/'trials/trial_0'
    assert (folder/'run_summary.json').exists()
    assert (folder/'final_model.zip').exists() == (outcome == 'COMPLETE')


@pytest.mark.parametrize('explicit_ceiling,expected', [(None, 185), (10, 10)])
def test_manager_uses_train_natural_horizon_for_guard_without_converting_episode_budget(
        monkeypatch, explicit_ceiling, expected):
    _, manager = modules()
    import gymnasium as gym
    import numpy as np
    from stable_baselines3.common.vec_env import DummyVecEnv
    from rl_zoo3 import exp_manager as official
    class HorizonEnv(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        action_space = gym.spaces.Discrete(13)
        max_episode_steps = 30
        def reset(self, **kwargs):
            return np.zeros(5, np.float32), {}
        def step(self, action):
            raise AssertionError('a safety bound does not step or estimate a voyage')
    monkeypatch.setattr(official.ExperimentManager, 'create_envs',
                        lambda *args, **kwargs:DummyVecEnv([HorizonEnv]))
    owner = object.__new__(manager.VesselExperimentManager)
    owner.n_envs, owner.n_episodes, owner.max_env_steps = 1, 3, explicit_ceiling
    owner._episode_learning_starts, owner.n_timesteps = 64, 1
    owner.env_kwargs, owner.eval_env_kwargs = {'split':'train'}, {'split':'validation'}
    owner.optimize_hyperparameters = False
    env = owner.create_envs(1)
    try:
        assert owner.n_timesteps == expected and owner.n_episodes == 3
    finally:
        env.close()


def test_real_official_objective_and_native_learn_preserve_timestep_pruned_outcome(monkeypatch, tmp_path):
    """Exercise the installed Zoo objective, native collection, and Optuna state."""
    _, manager = modules()
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    callbacks = import_module('experiments.sb3_manual.zoo_vessel.callbacks')
    torch.set_num_threads(1)
    class LegalEnv(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        action_space = gym.spaces.Discrete(13)
        def reset(self, **kwargs):
            super().reset(seed=kwargs.get('seed'))
            self.index = 0
            return np.zeros(5, np.float32), {}
        def action_masks(self):
            return np.arange(13) == 2
        def step(self, action):
            assert int(action) == 2
            self.index += 1
            done = self.index == 4
            info = dict(next_action_mask=np.zeros(13, bool) if done else self.action_masks())
            return np.zeros(5, np.float32), 1., done, False, info
    class RealPruningCallback(BaseCallback):
        def __init__(self, eval_env, trial, **kwargs):
            super().__init__()
            self.trial, self.is_pruned, self.last_mean_reward = trial, False, 0.
        def _on_step(self):
            if self.n_calls == 2:
                self.trial.report(0., step=2)
                self.is_pruned = self.trial.should_prune()
            return not self.is_pruned
    monkeypatch.setattr(callbacks, 'VesselTrialEvalCallback', RealPruningCallback)
    owner = object.__new__(manager.VesselExperimentManager)
    owner.seed, owner.n_jobs, owner.n_timesteps, owner.bounded = 42, 1, 33, True
    owner.n_episodes, owner.budget_mode = None, 'timestep'
    owner.exploration_schedule_mode, owner.exploration_episode_target = 'timestep', 500
    owner.eval_every_episodes, owner._trial_envs = 50, []
    owner.log_folder, owner.optimization_log_path = str(tmp_path), str(tmp_path/'trials')
    owner._hyperparams = dict(buffer_size=128, learning_starts=5, batch_size=4,
        policy_kwargs=dict(net_arch=[8, 8]), td_diagnostics=False)
    owner.n_envs = owner.n_eval_envs = owner.n_evaluations = 1
    owner.n_eval_episodes = 8
    owner.n_actions, owner.algo, owner.device, owner.verbose = 13, manager.ALGO_KEY, 'cpu', 0
    owner.specified_callbacks, owner.deterministic_eval = [], True
    owner.create_envs = lambda *args, **kwargs:DummyVecEnv([LegalEnv])
    manager.write_json(tmp_path/'vessel_identity.json', {'science':'fixture'})
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.ThresholdPruner(upper=-1.))
    with manager.zoo_bindings(ZooMaskedDQN, bounded=True):
        study.optimize(owner.objective, n_trials=1)
    trial = study.trials[0]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert trial.user_attrs['training_summary']['outcome'] == 'PRUNED'
    assert trial.user_attrs['training_summary']['n_timesteps'] == 2
    assert not (tmp_path/'trials/trial_0/final_model.zip').exists()
