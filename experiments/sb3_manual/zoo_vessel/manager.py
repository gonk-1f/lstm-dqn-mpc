"""Scoped Vessel extensions around the installed RL Zoo ExperimentManager.

Zoo owns setup, optimization and model.learn; no legacy episode runner is used.
The registry and TrialEval bridge exist only inside this launcher's process.
"""
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
import sys

import optuna
from rl_zoo3 import exp_manager as zoo
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT/'src') not in sys.path:
    sys.path.insert(0, str(ROOT/'src'))
CONFIG_PATH = Path(__file__).with_name('dqn_vessel.yml')
ALGO_KEY = 'dqn_vessel'


def check_versions():
    required = {'rl_zoo3':'2.9.0', 'stable-baselines3':'2.9.2', 'sb3-contrib':'2.9.2'}
    actual = {name:version(name) for name in required}
    if actual != required:
        raise RuntimeError(f'unsupported Zoo runtime: expected {required}, found {actual}')
    return actual


def episode_safety_limit(learning_starts, n_episodes, max_episode_steps):
    """A conservative natural-horizon bound, never an equivalent step budget.

    At most one voyage straddles warmup. Each counted voyage lasts no longer
    than the largest real Train horizon. One extra step keeps native learn's
    while condition above an exactly-on-boundary completion.
    """
    if learning_starts < 0 or n_episodes < 1 or max_episode_steps < 1:
        raise ValueError('nonnegative warmup and positive episode/horizon bounds required')
    return int(learning_starts + (n_episodes + 1) * max_episode_steps + 1)


def training_summary(model, *, requested_n_timesteps=None, internal_safety_limit=None,
                     model_path=None, best_model_path=None):
    """Actual training state shared by normal runs and official Zoo Trials."""
    validation = getattr(model, 'last_validation_summary', None)
    mode = getattr(model, 'budget_mode', 'timestep')
    return dict(n_timesteps=int(model.num_timesteps), requested_n_timesteps=requested_n_timesteps,
        budget_mode=mode, requested_n_episodes=getattr(model, 'n_episodes', None),
        exploration_schedule_mode=getattr(model, 'exploration_schedule_mode', 'timestep'),
        exploration_episode_target=getattr(model, 'exploration_episode_target', 500),
        internal_safety_limit=internal_safety_limit,
        completed_formal_episodes=int(getattr(model, 'completed_formal_episodes', 0)),
        train_successes=int(getattr(model, 'formal_successes', 0)),
        train_failures=int(getattr(model, 'formal_failures', 0)),
        warmup_env_steps=int(getattr(model, 'warmup_env_steps', min(model.num_timesteps, model.learning_starts))),
        warmup_bridge_env_steps=int(getattr(model, 'warmup_bridge_env_steps', 0)),
        formal_env_steps=int(model.formal_env_steps), adam_updates=int(model._n_updates),
        replay_insertions=int(model.replay_buffer.insertions),
        target_sync_count=len(model.target_sync_history),
        target_sync_calls_including_initial=int(model.target_sync_calls),
        target_sync_history=model.target_sync_history, final_epsilon=float(model.exploration_rate),
        final_validation_completion_rate=validation['completion_rate'] if validation else None,
        final_validation=validation,
        model=str(model_path) if model_path is not None else None,
        final_model=str(model_path) if model_path is not None else None,
        best_model=str(best_model_path) if best_model_path is not None else None,
        budget_unit=('complete_formal_train_episodes' if mode == 'episode'
                     else 'SB3_total_env_steps_including_warmup'), test_payloads_opened=0)


def _callback_is_pruned(callback):
    if isinstance(callback, (list, tuple)):
        return any(_callback_is_pruned(item) for item in callback)
    return bool(getattr(callback, 'is_pruned', False)) or any(
        _callback_is_pruned(item) for item in getattr(callback, 'callbacks', ()))


def convert_vessel_params(params):
    values = dict(params)
    width = values.pop('network_width')
    values.update(policy_kwargs=dict(net_arch=[width, width]), gamma=1., train_freq=32, gradient_steps=1)
    return values


def sample_vessel_params(trial, n_actions, n_envs, additional_args, *, bounded=False):
    if n_envs != 1:
        raise ValueError('Vessel search requires n_envs=1')
    if bounded:
        params = dict(learning_rate=trial.suggest_categorical('learning_rate', [1e-4]),
            batch_size=trial.suggest_categorical('batch_size', [32]),
            buffer_size=trial.suggest_categorical('buffer_size', [8192]),
            network_width=trial.suggest_categorical('network_width', [32]),
            target_update_interval=trial.suggest_categorical('target_update_interval', [128]))
    else:
        params = dict(learning_rate=trial.suggest_float('learning_rate', 1e-5, 3e-4, log=True),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
            buffer_size=trial.suggest_categorical('buffer_size', [100000, 300000, 1000000]),
            network_width=trial.suggest_categorical('network_width', [128, 256]),
            target_update_interval=trial.suggest_categorical('target_update_interval', [5000, 10000, 20000]))
    return convert_vessel_params(params)


@contextmanager
def zoo_bindings(algorithm, *, bounded=False):
    """Register only the Vessel key; restore every registry even after failure."""
    entries = ((zoo.ALGOS, algorithm),
        (zoo.HYPERPARAMS_SAMPLER, lambda *args:sample_vessel_params(*args, bounded=bounded)),
        (zoo.HYPERPARAMS_CONVERTER, convert_vessel_params))
    missing = object()
    previous = [(mapping, mapping.get(ALGO_KEY, missing)) for mapping, _ in entries]
    for mapping, value in entries:
        mapping[ALGO_KEY] = value
    try:
        yield
    finally:
        for mapping, value in previous:
            if value is missing:
                mapping.pop(ALGO_KEY, None)
            else:
                mapping[ALGO_KEY] = value


class _FatalDataErrors(VecEnvWrapper):
    """Keep vessel/data errors out of Zoo's ValueError -> TrialPruned handler."""
    def reset(self):
        try:
            return self.venv.reset()
        except Exception as error:
            raise RuntimeError('Vessel reset/data error; this is not an ordinary physical failure') from error

    def step_wait(self):
        try:
            return self.venv.step_wait()
        except Exception as error:
            raise RuntimeError('Vessel execution/data error; this is not an ordinary physical failure') from error


class VesselExperimentManager(zoo.ExperimentManager):
    def __init__(self, *args, data_root, bounded=False, n_episodes=None,
                 exploration_schedule_mode='timestep', exploration_episode_target=500,
                 eval_every_episodes=50, max_env_steps=None, **kwargs):
        from .env import register_envs
        check_versions()
        register_envs()
        if kwargs.get('n_jobs', 1) != 1 or kwargs.get('vec_env_type', 'dummy') != 'dummy':
            raise ValueError('Vessel Zoo requires n_jobs=1 and DummyVecEnv')
        if kwargs.get('n_eval_envs', 1) != 1:
            raise ValueError('Vessel Validation requires one environment')
        self.bounded = bounded
        self._trial_envs = []
        self.n_episodes = n_episodes
        self.budget_mode = 'episode' if n_episodes is not None else 'timestep'
        self.exploration_schedule_mode = exploration_schedule_mode
        self.exploration_episode_target = n_episodes or exploration_episode_target
        self.eval_every_episodes = eval_every_episodes
        self.max_env_steps = max_env_steps
        hyperparams = dict(kwargs.get('hyperparams') or {})
        self._episode_learning_starts = hyperparams.get('learning_starts', 50000)
        hyperparams.update(self._model_budget_parameters())
        kwargs['hyperparams'] = hyperparams
        kwargs.update(env_kwargs=dict(data_root=str(Path(data_root).resolve()), split='train'),
            eval_env_kwargs=dict(data_root=str(Path(data_root).resolve()), split='validation'),
            vec_env_type='dummy', n_eval_envs=1, n_jobs=1, device='cpu')
        super().__init__(*args, **kwargs)

    def _model_budget_parameters(self):
        # __new__-based wiring tests for the earlier timestep API have no new
        # attributes; their fake algorithms must keep their original signature.
        if not hasattr(self, 'budget_mode'):
            return {}
        return dict(n_episodes=self.n_episodes,
            exploration_schedule_mode=self.exploration_schedule_mode,
            exploration_episode_target=self.exploration_episode_target)

    def create_envs(self, n_envs, eval_env=False, no_log=False):
        if n_envs != 1 or self.n_envs != 1:
            raise ValueError('Vessel Zoo requires n_envs=1')
        expected = 'validation' if eval_env else 'train'
        kwargs = self.eval_env_kwargs if eval_env else self.env_kwargs
        if kwargs.get('split') != expected:
            raise ValueError('Train/Validation environment split mismatch')
        env = _FatalDataErrors(super().create_envs(n_envs, eval_env=eval_env, no_log=no_log))
        if not eval_env and self.n_episodes is not None:
            horizon = int(env.get_attr('max_episode_steps')[0])
            self.n_timesteps = self.max_env_steps or episode_safety_limit(
                self._episode_learning_starts, self.n_episodes, horizon)
        if self.optimize_hyperparameters:
            self._trial_envs.append(env)
        return env

    def create_callbacks(self):
        from .callbacks import VesselEvalCallback, VesselProgressCallback
        if self.optimize_hyperparameters or self.eval_freq <= 0:
            self.callbacks.append(VesselProgressCallback(log_every=1000))
        if self.save_freq > 0:
            self.callbacks.append(CheckpointCallback(save_freq=self.save_freq,
                save_path=self.save_path, name_prefix='vessel', save_replay_buffer=False))
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            self.callbacks.append(VesselEvalCallback(self.create_envs(1, eval_env=True),
                n_eval_episodes=8, eval_freq=self.eval_freq, log_path=self.save_path,
                best_model_save_path=self.save_path, deterministic=True, verbose=self.verbose,
                eval_every_episodes=self.eval_every_episodes))

    def objective(self, trial):
        from .callbacks import VesselTrialEvalCallback
        if self.n_jobs != 1:
            raise ValueError('scoped Vessel objective requires n_jobs=1')
        algorithm = zoo.ALGOS[ALGO_KEY]
        previous_callback = zoo.TrialEvalCallback
        trial_seed = self.seed + trial.number
        owner = self
        episode_mode = getattr(self, 'budget_mode', 'timestep') == 'episode'
        trial_folder = (Path(self.optimization_log_path)/f'trial_{trial.number}'
                        if getattr(self, 'optimization_log_path', None) else None)
        trial.set_user_attr('model_seed', trial_seed)
        trial.set_user_attr('budget_unit', 'complete_formal_train_episodes' if episode_mode
                            else 'SB3_total_env_steps_including_warmup')
        trial.set_user_attr('n_timesteps', None if episode_mode else self.n_timesteps)
        trial.set_user_attr('n_episodes', getattr(self, 'n_episodes', None))
        trial.set_user_attr('internal_safety_limit', self.n_timesteps if episode_mode else None)
        trial.set_user_attr('exploration_schedule_mode', getattr(self, 'exploration_schedule_mode', 'timestep'))
        trial.set_user_attr('eval_every_episodes', getattr(self, 'eval_every_episodes', 50))
        if getattr(self, 'optimization_log_path', None):
            identity_path = Path(self.log_folder)/'vessel_identity.json'
            write_json(Path(self.optimization_log_path)/f'trial_{trial.number}'/'vessel_identity.json',
                       json.loads(identity_path.read_text(encoding='utf-8')))
        class SeededTrialAlgorithm(algorithm):
            def __init__(self, *args, **kwargs):
                kwargs['seed'] = trial_seed
                kwargs.update(owner._model_budget_parameters())
                super().__init__(*args, **kwargs)

            def learn(self, *args, **kwargs):
                # Zoo objective omits log_interval, whose DQN default would
                # otherwise create extra episode-based native dumps.
                kwargs.setdefault('log_interval', None)
                if trial_folder is not None:
                    self.final_model_path = str(trial_folder/'final_model.zip')
                    self.best_model_path = str(trial_folder/'best_model.zip')
                outcome = 'FAIL'
                try:
                    result = super().learn(*args, **kwargs)
                    # Timestep callbacks may stop native collection by returning
                    # False. Official Zoo checks is_pruned only AFTER learn has
                    # returned; do not write a fake COMPLETE model meanwhile.
                    callback = kwargs.get('callback', args[1] if len(args) > 1 else None)
                    if _callback_is_pruned(callback):
                        outcome = 'PRUNED'
                        return result
                    if episode_mode and self.completed_formal_episodes != owner.n_episodes:
                        raise RuntimeError('Episode Trial stopped before its complete formal Episode budget')
                    outcome = 'COMPLETE'
                    return result
                except optuna.TrialPruned:
                    outcome = 'PRUNED'
                    raise
                finally:
                    final_path = trial_folder/'final_model.zip' if trial_folder is not None else None
                    best_path = trial_folder/'best_model.zip' if trial_folder is not None else None
                    summary = training_summary(self,
                        requested_n_timesteps=None if episode_mode else owner.n_timesteps,
                        internal_safety_limit=owner.n_timesteps if episode_mode else None,
                        model_path=final_path if outcome == 'COMPLETE' else None,
                        best_model_path=best_path if best_path is not None and best_path.exists() else None)
                    summary.update(outcome=outcome, hyperparameters=dict(trial.params),
                                   model_seed=trial_seed)
                    trial.set_user_attr('training_summary', summary)
                    if trial_folder is not None:
                        if outcome == 'COMPLETE':
                            self.save(final_path)
                        write_json(trial_folder/'run_summary.json', summary)
        class ScopedTrialCallback(VesselTrialEvalCallback):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault('eval_every_episodes', getattr(owner, 'eval_every_episodes', 50))
                super().__init__(*args, **kwargs)
        self._trial_envs = []
        zoo.ALGOS[ALGO_KEY] = SeededTrialAlgorithm
        zoo.TrialEvalCallback = ScopedTrialCallback
        try:
            return super().objective(trial)
        except optuna.TrialPruned as error:
            if isinstance(error.__cause__, (ValueError, AssertionError)):
                raise RuntimeError('Vessel Zoo program/data error is fatal, not pruning') from error.__cause__
            raise
        finally:
            zoo.ALGOS[ALGO_KEY] = algorithm
            zoo.TrialEvalCallback = previous_callback
            for env in self._trial_envs:
                env.close()
            self._trial_envs = []


def build_identity(data_root):
    from v4.v6_dataset import V6TrainingDataset
    from v6_sb3.identity import dataset_identity, execution_identity, file_digest
    from v6_sb3.rewards import INVERSE_ONE_PLUS_ECONOMIC
    data = V6TrainingDataset.open(Path(data_root))
    if data.split_sizes != (35, 8):
        raise ValueError('Vessel Zoo requires the frozen 35 Train / 8 Validation split')
    identity = execution_identity(ROOT, dataset_identity(data), reward_mode=INVERSE_ONE_PLUS_ECONOMIC)
    package = Path(__file__).parent
    identity.update(adapter='vessel_rl_zoo_v2', zoo_versions=check_versions(),
        adapter_files={path.relative_to(ROOT).as_posix():file_digest(path)
                       for path in sorted((*package.glob('*.py'), *package.glob('*.yml')))},
        budget_semantics='explicit timestep OR complete formal Episode budget; no estimated equivalence',
        validation_scope='all eight fixed Validation voyages; natural completion or physical failure',
        test_payloads_opened=0)
    return identity


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def prepare_output(directory, identity, *, resume=False):
    directory = Path(directory).resolve()
    path = directory/'vessel_identity.json'
    if resume:
        if not path.exists() or json.loads(path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Vessel output identity changed; refusing result/Study reuse')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError('a new empty output directory is required')
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, identity)


def guard_study(storage, name, identity, *, resume=False):
    if resume:
        study = optuna.load_study(storage=storage, study_name=name)
        if study.user_attrs.get('vessel_identity') != identity:
            raise ValueError('Vessel Study identity differs')
        if any(trial.state == optuna.trial.TrialState.RUNNING for trial in study.trials):
            raise ValueError('an interrupted Zoo Trial cannot be resumed exactly; keep it and use a new Study')
    else:
        study = optuna.create_study(storage=storage, study_name=name, direction='maximize', load_if_exists=False)
        study.set_user_attr('vessel_identity', identity)
    return study
