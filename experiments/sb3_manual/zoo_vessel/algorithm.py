"""Vessel masks and formal Target timing inside the native SB3 learn loop.

SB3 still owns collection, replay insertion, optimization dispatch and Logger.
Warmup belongs to its total-step clock, while V6 Target/Adam cadence counts
stored formal transitions. This is a separate Zoo algorithm, not an unmasked
DQN replacement. Episode budgets add terminal-boundary control around the same
native collector; they never turn a timestep estimate into an episode budget.
"""
from contextlib import contextmanager
from math import fsum
from numbers import Integral

import numpy as np

from stable_baselines3.common.type_aliases import RolloutReturn, TrainFreq, TrainFrequencyUnit

from v6_sb3.masked_dqn import MaskedDQN


class _EpisodeRolloutBoundary(Exception):
    """Leave native collection before another action, after terminal bookkeeping."""


class ZooMaskedDQN(MaskedDQN):
    """Keep V6 masked single-step TD while using SB3 model.learn and logging."""

    zoo_adapter_version = 'zoo_vessel_masked_dqn_v2'

    def __init__(self, policy='MlpPolicy', env=None, *,
                 target_sync_unit='formal_env_steps', n_episodes=None,
                 exploration_schedule_mode='timestep', exploration_episode_target=500,
                 **kwargs):
        if target_sync_unit != 'formal_env_steps':
            raise ValueError('ZooMaskedDQN requires the formal_env_steps Target clock')
        if n_episodes is not None and (isinstance(n_episodes, bool)
                or not isinstance(n_episodes, Integral) or n_episodes < 1):
            raise ValueError('n_episodes must be a positive integer or None')
        if exploration_schedule_mode not in ('timestep', 'episode'):
            raise ValueError('exploration_schedule_mode must be timestep or episode')
        if (isinstance(exploration_episode_target, bool)
                or not isinstance(exploration_episode_target, Integral) or exploration_episode_target < 1):
            raise ValueError('exploration_episode_target must be a positive integer')
        if n_episodes is not None and (kwargs.get('train_freq', 32) != 32
                or kwargs.get('gradient_steps', 1) != 1):
            raise ValueError('Episode budgets require the unchanged train_freq=32, gradient_steps=1')
        self.n_episodes = n_episodes
        self.budget_mode = 'episode' if n_episodes is not None else 'timestep'
        self.exploration_schedule_mode = exploration_schedule_mode
        self.exploration_episode_target = n_episodes or exploration_episode_target
        self.completed_formal_episodes = 0
        self.formal_successes = self.formal_failures = 0
        self.formal_episode_history = []
        self.warmup_bridge_env_steps = 0
        self.episode_boundary_observer = None
        self._episode_rollout_boundary = False
        self._pending_episode_notification = False
        self._evaluation_mask_env = None
        self.last_td_mae = None
        kwargs.setdefault('train_freq', 32)
        kwargs.setdefault('gradient_steps', 1)
        kwargs.setdefault('target_update_interval', 10000)
        super().__init__(policy, env, target_sync_unit=target_sync_unit, **kwargs)
        self._current_episode_is_formal = self.learning_starts == 0
        self._current_voyage_env_steps = 0
        self._current_episode_epsilon = (self.exploration_initial_eps
                                        if self._current_episode_is_formal else 1.)
        if self.exploration_schedule_mode == 'episode':
            if not (0. < self.exploration_fraction <= 1.
                    and 0. <= self.exploration_final_eps <= self.exploration_initial_eps <= 1.):
                raise ValueError('Episode exploration requires valid probabilities and 0 < fraction <= 1')
            self.exploration_rate = self._current_episode_epsilon

    @property
    def warmup_env_steps(self):
        return min(self.num_timesteps, self.learning_starts)

    @property
    def episode_budget_complete(self):
        return self.n_episodes is not None and self.completed_formal_episodes == self.n_episodes

    @property
    def current_episode_is_formal(self):
        return self._current_episode_is_formal

    @property
    def last_formal_episode_summary(self):
        return self.formal_episode_history[-1] if self.formal_episode_history else None

    def episode_epsilon(self):
        """Reuse SB3's schedule with completed full formal episodes as progress."""
        if self.completed_formal_episodes >= self.exploration_fraction * self.exploration_episode_target:
            return float(self.exploration_final_eps)
        progress = 1. - self.completed_formal_episodes / self.exploration_episode_target
        return float(self.exploration_schedule(progress))

    def learn(self, *args, **kwargs):
        """SB3 owns the loop; an internal timestep ceiling only guards bad runs."""
        if self.n_episodes is not None and self.num_timesteps:
            raise RuntimeError('Episode-budget learn requires a fresh model: live physical/RNG '
                               'training resume is not supported by this Zoo route')
        observer = self.episode_boundary_observer
        try:
            result = super().learn(*args, **kwargs)
        finally:
            # Also release callbacks attached during _on_training_start when
            # Optuna pruning or another error bypasses native on_training_end.
            self.episode_boundary_observer = observer
        if self.n_episodes is not None and not self.episode_budget_complete:
            raise RuntimeError('Episode budget not completed: callback stop or timestep safety ceiling '
                               f'({self.completed_formal_episodes}/{self.n_episodes})')
        return result

    def collect_rollouts(self, env, callback, train_freq, replay_buffer,
                         action_noise=None, learning_starts=0, log_interval=None):
        """End the final warmup block exactly at learning_starts.

        For example LS=50000 and train_freq=32 first collect the remaining 16
        warmup steps, with no Adam update at LS. The next complete 32-step
        rollout then contains 32 formal transitions. Other rollout behavior,
        including the native total_timesteps lower-bound semantics, is SB3's.
        """
        if env.num_envs != 1 or train_freq.unit != TrainFrequencyUnit.STEP:
            raise ValueError('Zoo vessel collection requires one environment and step train_freq')
        if self.episode_budget_complete:
            return RolloutReturn(0, 0, continue_training=False)
        self._episode_rollout_boundary = False
        remaining_warmup = learning_starts - self.num_timesteps
        if 0 < remaining_warmup < train_freq.frequency:
            train_freq = TrainFreq(int(remaining_warmup), TrainFrequencyUnit.STEP)
        elif self.n_episodes is not None and remaining_warmup <= 0:
            # A prior voyage boundary may have ended a partial native block.
            train_freq = TrainFreq(32 - self.formal_env_steps % 32, TrainFrequencyUnit.STEP)
        steps_before, episodes_before = self.num_timesteps, self._episode_num
        try:
            return super().collect_rollouts(env, callback, train_freq, replay_buffer,
                action_noise=action_noise, learning_starts=learning_starts, log_interval=log_interval)
        except _EpisodeRolloutBoundary:
            # The next _sample_action runs after SB3 has stored the terminal,
            # updated _episode_num/Monitor and handled native episode logging.
            # Only the normal rollout-end notification remains outstanding.
            callback.on_rollout_end()
            return RolloutReturn(self.num_timesteps - steps_before,
                self._episode_num - episodes_before, continue_training=True)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.n_episodes is not None:
            if self._episode_rollout_boundary:
                raise _EpisodeRolloutBoundary
            if self.num_timesteps >= self._total_timesteps:
                raise RuntimeError('Episode timestep safety ceiling reached before the episode budget')
        return super()._sample_action(learning_starts, action_noise=action_noise, n_envs=n_envs)

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, infos)
        self._current_voyage_env_steps += 1
        if self.num_timesteps > self.learning_starts and not self._current_episode_is_formal:
            self.warmup_bridge_env_steps += 1
        if bool(dones[0]):
            if self._current_episode_is_formal:
                row = infos[0].get('vessel_episode', {})
                completed = row.get('completed', infos[0].get('terminal_reason') == 'completed')
                self.completed_formal_episodes += 1
                self.formal_successes += int(completed)
                self.formal_failures += int(not completed)
                summary = dict(row, episode=self.completed_formal_episodes,
                    completed=bool(completed),
                    terminal_reason=row.get('terminal_reason', infos[0].get('terminal_reason')),
                    total_env_steps=self.num_timesteps,
                    formal_env_steps=max(0, self.num_timesteps - self.learning_starts),
                    episode_env_steps=row.get('episode_env_steps', self._current_voyage_env_steps),
                    epsilon_used=self.exploration_rate,
                    replay_insertions=replay_buffer.insertions,
                    adam_updates=self._n_updates,
                    storage_adam_updates=self._n_updates)
                self.formal_episode_history.append(summary)
                if len(self.formal_episode_history) > self.exploration_episode_target:
                    self.formal_episode_history.pop(0)
                if self.n_episodes is not None:
                    self._episode_rollout_boundary = True
                    self._pending_episode_notification = True
            # DummyVecEnv has naturally reset after this terminal. A reset at
            # exactly LS is eligible, while the voyage spanning LS is excluded.
            self._current_episode_is_formal = self.num_timesteps >= self.learning_starts
            self._current_episode_epsilon = (self.episode_epsilon()
                if self._current_episode_is_formal else 1.)
            self._current_voyage_env_steps = 0

    def _on_step(self):
        # Native SB3 calls this after replay insertion and progress update.
        # Calling DQN._on_step would also copy Target using the total-step clock;
        # use only the masked parent's counter/Logger and the V6 formal clock.
        self.exploration_rate = (self._current_episode_epsilon
            if self.exploration_schedule_mode == 'episode'
            else self.exploration_schedule(self._current_progress_remaining))
        super()._on_step()
        if self.num_timesteps > self.learning_starts:
            self.record_formal_transition()

    def train(self, gradient_steps, batch_size=64):
        """Observe actual sampled TD residuals without changing parent training."""
        if self.n_episodes is not None and self._n_updates >= self.formal_env_steps // 32:
            self._notify_episode_boundary()
            return
        observer = self.train_step_observer
        batch_td_maes = []

        def observe(model, batch, q, target):
            batch_td_maes.append(float((target - q).abs().mean().item()))
            self.last_td_mae = batch_td_maes[-1]
            if observer is not None:
                observer(model, batch, q, target)

        self.train_step_observer = observe
        try:
            super().train(gradient_steps, batch_size=batch_size)
        finally:
            self.train_step_observer = observer
        if batch_td_maes:
            self.last_td_mae = fsum(batch_td_maes) / len(batch_td_maes)
            self.logger.record('train/td_mae', self.last_td_mae)
        # The parent excludes this from TensorBoard for the historical runner.
        # Zoo keeps the native Logger and exposes the actual count to all sinks.
        self.logger.record('train/n_updates', self._n_updates)
        self._notify_episode_boundary()

    def _notify_episode_boundary(self):
        if self._pending_episode_notification:
            self._pending_episode_notification = False
            self.last_formal_episode_summary['adam_updates'] = self._n_updates
            if self.episode_boundary_observer is not None:
                self.episode_boundary_observer(self)

    @contextmanager
    def evaluation_mask_source(self, eval_vec_env):
        """Bind native evaluate_policy.predict calls to that evaluation env."""
        if not callable(getattr(eval_vec_env, 'env_method', None)):
            raise TypeError('evaluation mask source must be a VecEnv with env_method')
        previous = self._evaluation_mask_env
        self._evaluation_mask_env = eval_vec_env
        try:
            yield self
        finally:
            self._evaluation_mask_env = previous

    def predict(self, observation, state=None, episode_start=None,
                deterministic=False, *, action_masks=None):
        """Use explicit masks, bound evaluation masks, or the training env's.

        Native collection uses the inherited _sample_action and its training
        mask directly. A loaded model without an env must receive explicit
        masks or an evaluation context rather than infer an unmasked action.
        """
        if action_masks is None:
            source = self._evaluation_mask_env if self._evaluation_mask_env is not None else self.env
            if source is None:
                raise ValueError('predict requires explicit action masks or an environment mask source')
            action_masks = np.asarray(source.env_method('action_masks'), dtype=np.bool_)
        actions, _ = self.masked_predict(observation, action_masks, deterministic=deterministic)
        return actions, state

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), '_evaluation_mask_env', 'episode_boundary_observer']
