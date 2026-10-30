"""Native SB3 logging and completion-first, fully independent Validation."""
from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
import random
import sys

import numpy as np
import optuna
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import (
    CSVOutputFormat, HumanOutputFormat, TensorBoardOutputFormat, configure,
)

from v6_sb3.search import validation_objective


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def evaluate_all_validation(model, eval_env):
    """Evaluate exactly the eight Validation IDs using that environment's masks.

    Fixed voyage horizons bound execution; physical failure is a real terminal,
    not a timeout. No training environment, optimizer, Replay or RNG is advanced.
    """
    if eval_env.num_envs != 1:
        raise RuntimeError('vessel Validation requires one independent environment')
    expected = tuple(eval_env.get_attr('episode_ids')[0])
    if len(expected) != 8 or len(set(expected)) != 8:
        raise RuntimeError('Validation must contain eight distinct frozen voyages')
    bound = int(eval_env.get_attr('evaluation_step_budget')[0])
    def training_signature():
        return (model.num_timesteps, model._n_updates, model.replay_buffer.insertions,
                model.formal_env_steps, model.target_sync_calls, model.exploration_rate,
                *(getattr(model, name, None) for name in ('completed_formal_episodes',
                    'formal_successes', 'formal_failures', 'warmup_env_steps',
                    'warmup_bridge_env_steps', 'current_episode_is_formal')))
    before = training_signature()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    was_training = model.policy.training
    rows, steps = [], 0
    try:
        model.policy.set_training_mode(False)
        eval_env.env_method('begin_evaluation')
        observation = eval_env.reset()
        with torch.inference_mode():
            while len(rows) < 8:
                masks = np.asarray(eval_env.env_method('action_masks'), dtype=bool)
                actions, _ = model.masked_predict(observation, masks, deterministic=True)
                observation, rewards, dones, infos = eval_env.step(np.asarray(actions).reshape(1))
                steps += 1
                if steps > bound:
                    raise RuntimeError('Validation exceeded the sum of its real voyage horizons')
                if not np.isfinite(rewards).all():
                    raise RuntimeError('nonfinite Validation reward')
                if dones[0]:
                    row = dict(infos[0]['vessel_episode'])
                    if row['split'] != 'validation' or row['sample_id'] not in expected:
                        raise RuntimeError('Validation split isolation violation')
                    if row['sample_id'] in {item['sample_id'] for item in rows}:
                        raise RuntimeError('Validation voyage evaluated twice')
                    rows.append(row)
    except Exception as error:
        # Zoo's stock objective prunes ValueError/AssertionError. Data errors
        # must stay visible and cannot be classified as ordinary physical failure.
        raise RuntimeError('eight-voyage Validation failed') from error
    finally:
        model.policy.set_training_mode(was_training)
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
    after = training_signature()
    if before != after:
        raise RuntimeError('Validation changed training state')
    completed = sum(bool(row['completed']) for row in rows)
    summary = dict(episodes=8, completed=completed, completion_rate=completed/8,
        mean_duration_coverage=sum(row['duration_coverage'] for row in rows)/8,
        fixed_complete_set_cost_cny=(sum(row['complete_economic_cost_cny'] for row in rows)
                                     if completed == 8 else None),
        environment_steps=steps, samples=rows, selection_split='validation',
        test_payloads_opened=0, evaluation_adam_updates=0, evaluation_replay_insertions=0)
    summary['score'] = validation_objective(summary)
    return summary


class VesselProgressCallback(BaseCallback):
    """Add scalars to SB3's Logger; its own dump_logs renders every table."""
    def __init__(self, log_every=1000, log_dir=None, verbose=0):
        super().__init__(verbose)
        if log_every < 1:
            raise ValueError('log_every must be positive')
        self.log_every, self.log_dir = log_every, log_dir
        self._last_dump = None
        self._last_vessel = None

    def _on_training_start(self):
        if self.log_dir is not None:
            # Still the official Logger and its official output formats. Needed
            # for Zoo objective, which deliberately disables its default logging.
            self.model.logger.close()
            self.model.set_logger(configure(str(self.log_dir), ['stdout', 'csv', 'tensorboard']))
        logger = self.model.logger
        directory = logger.get_dir()
        if directory is None:
            raise RuntimeError('a native logging directory is required')
        for kind, make in (
            (CSVOutputFormat, lambda:CSVOutputFormat(str(Path(directory)/'progress.csv'))),
            (TensorBoardOutputFormat, lambda:TensorBoardOutputFormat(str(directory))),
            (HumanOutputFormat, lambda:HumanOutputFormat(sys.stdout)),
        ):
            if not any(isinstance(writer, kind) for writer in logger.output_formats):
                logger.output_formats.append(make())

    def _record_progress(self):
        logger, model = self.logger, self.model
        if self._last_vessel is not None:
            row = self._last_vessel
            for name in ('soc', 'fc_power_kw', 'fc_start_count', 'economic_cost_cny', 'sample_id'):
                logger.record('vessel/'+name, row[name])
        logger.record('rollout/exploration_rate', model.exploration_rate)
        logger.record('replay/buffer_size', model.replay_buffer.size())
        logger.record('replay/insertions', model.replay_buffer.insertions)
        logger.record('replay/pending_transition', int(model.num_timesteps > model.replay_buffer.insertions))
        logger.record('train/n_updates', model._n_updates)
        logger.record('train/adam_updates', model._n_updates)
        logger.record('vessel/formal_env_steps', model.formal_env_steps)
        logger.record('train/formal_env_steps', model.formal_env_steps)
        logger.record('train/warmup_env_steps', getattr(model, 'warmup_env_steps',
            min(model.num_timesteps, getattr(model, 'learning_starts', 0))))
        logger.record('train/warmup_bridge_env_steps', getattr(model, 'warmup_bridge_env_steps', 0))
        completed = getattr(model, 'completed_formal_episodes', 0)
        successes = getattr(model, 'formal_successes', 0)
        failures = getattr(model, 'formal_failures', 0)
        logger.record('train/completed_formal_episodes', completed)
        logger.record('train/formal_successes', successes)
        logger.record('train/formal_failures', failures)
        logger.record('train/success_rate', successes/completed if completed else None)
        logger.record('train/failure_rate', failures/completed if completed else None)
        logger.record('train/target_episodes', getattr(model, 'n_episodes', None))
        logger.record('train/budget_mode', getattr(model, 'budget_mode', 'timestep'))
        logger.record('train/exploration_schedule_mode', getattr(model, 'exploration_schedule_mode', 'timestep'))
        if model.last_training_loss is not None:
            logger.record('train/loss', model.last_training_loss)
        if model.last_td_mae is not None:
            logger.record('train/td_mae', model.last_td_mae)

    def _dump(self):
        signature = (self.model.num_timesteps, self.model._n_updates,
                     self.model.replay_buffer.insertions)
        if signature == self._last_dump:
            return
        self._record_progress()
        self.model.dump_logs()
        self._last_dump = signature

    def _on_step(self):
        infos = self.locals.get('infos', [])
        if infos:
            self._last_vessel = infos[0].get('vessel', self._last_vessel)
        if self.num_timesteps % self.log_every == 0:
            self._dump()
        return True

    def _on_training_end(self):
        self._record_training_summary()
        self._dump()

    def _record_training_summary(self):
        """Append final metadata to the same native terminal table, not a new one."""
        model = self.model
        validation = getattr(model, 'last_validation_summary', None)
        for key, value in (
            ('target_sync_count', len(getattr(model, 'target_sync_history', ()))),
            ('target_sync_calls_including_initial', getattr(model, 'target_sync_calls', 0)),
            ('final_epsilon', model.exploration_rate),
            ('final_validation_completion_rate', validation['completion_rate'] if validation else None),
            ('final_model_path', getattr(model, 'final_model_path', None)),
            ('best_model_path', getattr(model, 'best_model_path', None)),
        ):
            self.logger.record('summary/'+key, value)


class VesselEvalCallback(VesselProgressCallback):
    def __init__(self, eval_env, n_eval_episodes=8, eval_freq=10000,
                 best_model_save_path=None, log_path=None, deterministic=True,
                 verbose=0, log_every=1000, log_dir=None, eval_every_episodes=50, **kwargs):
        if n_eval_episodes != 8 or not deterministic:
            raise ValueError('vessel evaluation is greedy over all eight Validation voyages')
        super().__init__(log_every=log_every, log_dir=log_dir, verbose=verbose)
        if eval_every_episodes < 1:
            raise ValueError('eval_every_episodes must be positive')
        self.eval_every_episodes = eval_every_episodes
        self.eval_env, self.eval_freq = eval_env, eval_freq
        self.best_model_save_path, self.log_path = best_model_save_path, log_path
        self.best_score = -float('inf')
        self.last_summary = None
        self.last_mean_reward = -float('inf')  # Official Zoo objective API alias; it is a score.
        self.evaluations = []
        self._previous_episode_observer = None
        self._bound_episode_observer = None
        self._last_evaluated_episode = None

    def _episode_budget(self):
        return getattr(self.model, 'budget_mode', 'timestep') == 'episode'

    def _on_training_start(self):
        super()._on_training_start()
        if self._episode_budget():
            self._previous_episode_observer = self.model.episode_boundary_observer
            def observe(model):
                if self._previous_episode_observer is not None:
                    self._previous_episode_observer(model)
                self._after_episode_boundary(model)
            self._bound_episode_observer = observe
            self.model.episode_boundary_observer = observe

    def _restore_episode_observer(self):
        if (self._bound_episode_observer is not None and
                self.model.episode_boundary_observer is self._bound_episode_observer):
            self.model.episode_boundary_observer = self._previous_episode_observer

    def _after_episode_boundary(self, model):
        """Only the algorithm's post-storage, post-optimization hook calls this."""
        if model is not self.model or not self._episode_budget():
            raise RuntimeError('Episode Validation received a different training model/budget')
        completed = model.completed_formal_episodes
        due = (completed > 0 and completed != self._last_evaluated_episode and
               (completed % self.eval_every_episodes == 0 or completed == model.n_episodes))
        if due:
            if model.num_timesteps != model.replay_buffer.insertions:
                raise RuntimeError('Episode Validation must follow terminal Replay insertion')
            if model._n_updates != model.formal_env_steps // 32:
                raise RuntimeError('Episode Validation must follow every due Adam update')
            self._last_evaluated_episode = completed
            self._evaluate()
        # Native on_training_end appends the final summary to the sole terminal
        # table. Do not emit that table here before its metadata is available.
        final = completed == model.n_episodes
        if (not final and not getattr(self, 'is_pruned', False) and
                (due or model.num_timesteps % self.log_every == 0)):
            self._dump()

    def _on_training_end(self):
        try:
            super()._on_training_end()
        finally:
            self._restore_episode_observer()

    def _evaluate(self):
        value = evaluate_all_validation(self.model, self.eval_env)
        value.update(training_env_steps=self.model.num_timesteps,
            formal_env_steps=self.model.formal_env_steps, adam_updates=self.model._n_updates,
            replay_insertions=self.model.replay_buffer.insertions,
            completed_formal_episodes=getattr(self.model, 'completed_formal_episodes', 0),
            warmup_env_steps=getattr(self.model, 'warmup_env_steps',
                min(self.model.num_timesteps, getattr(self.model, 'learning_starts', 0))),
            warmup_bridge_env_steps=getattr(self.model, 'warmup_bridge_env_steps', 0),
            formal_successes=getattr(self.model, 'formal_successes', 0),
            formal_failures=getattr(self.model, 'formal_failures', 0),
            budget_mode=getattr(self.model, 'budget_mode', 'timestep'),
            exploration_schedule_mode=getattr(self.model, 'exploration_schedule_mode', 'timestep'),
            evaluation_position=('after terminal Replay insertion and due Adam update' if self._episode_budget()
                                 else 'native callback before current transition storage and next Adam'))
        self.last_summary, self.last_mean_reward = value, value['score']
        self.model.last_validation_summary = value
        self.evaluations.append(value)
        for key, number in (
            ('completion_rate', value['completion_rate']),
            ('mean_duration_coverage', value['mean_duration_coverage']),
            ('economic_cost_cny', value['fixed_complete_set_cost_cny']),
            ('economic_cost_available', int(value['fixed_complete_set_cost_cny'] is not None)),
            ('objective_score', value['score']),
            ('evaluated_adam_updates', value['adam_updates']),
            ('evaluated_env_steps', value['training_env_steps']),
            ('evaluated_formal_episodes', value['completed_formal_episodes']),
        ):
            self.logger.record('validation/'+key, number)
        if self.log_path:
            _json(Path(self.log_path)/'validation.json', self.evaluations)
        if value['score'] > self.best_score:
            self.best_score = value['score']
            if self.best_model_save_path:
                folder = Path(self.best_model_save_path)
                folder.mkdir(parents=True, exist_ok=True)
                self.model.save(folder/'best_model.zip')
                _json(folder/'best_validation.json', value)

    def _on_step(self):
        infos = self.locals.get('infos', [])
        if infos:
            self._last_vessel = infos[0].get('vessel', self._last_vessel)
        if self._episode_budget():
            # Never stop from SB3's pre-storage on_step. A terminal periodic
            # dump is deferred to the post-update hook to avoid two tables at
            # the same timestep with different Replay/Adam counts.
            formal_terminal = (bool(infos) and 'vessel_episode' in infos[0] and
                               getattr(self.model, 'current_episode_is_formal', True))
            if self.num_timesteps % self.log_every == 0 and not formal_terminal:
                self._dump()
            return True
        evaluate = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluate:
            self._evaluate()
        # At the exact final budget, defer the dump to training_end, after the
        # native final Replay insertion/Adam. The evaluation's own clocks above
        # identify the weights it actually assessed. No duplicate native dump.
        final = self.num_timesteps == self.model._total_timesteps
        if (evaluate or self.num_timesteps % self.log_every == 0) and not final:
            self._dump()
        return True


class VesselTrialEvalCallback(VesselEvalCallback):
    def __init__(self, eval_env, trial, **kwargs):
        if kwargs.get('log_path'):
            kwargs.setdefault('log_dir', str(Path(kwargs['log_path'])/'native_logs'))
        super().__init__(eval_env, **kwargs)
        self.trial, self.eval_idx, self.is_pruned = trial, 0, False

    def _evaluate(self):
        super()._evaluate()
        self.eval_idx += 1
        self.trial.report(self.last_summary['score'], self.eval_idx)
        self.trial.set_user_attr('validation', self.last_summary)
        self.is_pruned = self.trial.should_prune()

    def _on_step(self):
        super()._on_step()
        if self._episode_budget():
            return True
        return not self.is_pruned

    def _after_episode_boundary(self, model):
        super()._after_episode_boundary(model)
        if self.is_pruned:
            # The terminal is already stored and any due optimizer update has
            # finished. Raising here preserves official Zoo/Optuna pruning
            # without losing the transition through pre-storage on_step=False.
            # A pruning exception bypasses native on_training_end; the parent
            # deferred its table so the committed actor and summary dump once.
            self._record_training_summary()
            self._dump()
            self._restore_episode_observer()
            raise optuna.TrialPruned('Episode Validation requested pruning after a committed boundary')
