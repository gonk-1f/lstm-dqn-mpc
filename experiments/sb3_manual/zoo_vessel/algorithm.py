"""Vessel masks and formal Target timing inside the native SB3 learn loop.

SB3 still owns collection, replay insertion, optimization dispatch and Logger.
Warmup belongs to its total-step clock, while V6 Target/Adam cadence counts
stored formal transitions. This is a separate Zoo algorithm, not an unmasked
DQN replacement and not the old episode-based epsilon schedule.
"""
from contextlib import contextmanager
from math import fsum

import numpy as np

from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from v6_sb3.masked_dqn import MaskedDQN


class ZooMaskedDQN(MaskedDQN):
    """Keep V6 masked single-step TD while using SB3 model.learn and logging."""

    zoo_adapter_version = 'zoo_vessel_masked_dqn_v1'

    def __init__(self, policy='MlpPolicy', env=None, *,
                 target_sync_unit='formal_env_steps', **kwargs):
        if target_sync_unit != 'formal_env_steps':
            raise ValueError('ZooMaskedDQN requires the formal_env_steps Target clock')
        self._evaluation_mask_env = None
        self.last_td_mae = None
        kwargs.setdefault('train_freq', 32)
        kwargs.setdefault('gradient_steps', 1)
        kwargs.setdefault('target_update_interval', 10000)
        super().__init__(policy, env, target_sync_unit=target_sync_unit, **kwargs)

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
        remaining_warmup = learning_starts - self.num_timesteps
        if 0 < remaining_warmup < train_freq.frequency:
            train_freq = TrainFreq(int(remaining_warmup), TrainFrequencyUnit.STEP)
        return super().collect_rollouts(env, callback, train_freq, replay_buffer,
            action_noise=action_noise, learning_starts=learning_starts, log_interval=log_interval)

    def _on_step(self):
        # Native SB3 calls this after replay insertion and progress update.
        # Calling DQN._on_step would also copy Target using the total-step clock;
        # use only the masked parent's counter/Logger and the V6 formal clock.
        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        super()._on_step()
        if self.num_timesteps > self.learning_starts:
            self.record_formal_transition()

    def train(self, gradient_steps, batch_size=64):
        """Observe actual sampled TD residuals without changing parent training."""
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
        return [*super()._excluded_save_params(), '_evaluation_mask_env']
