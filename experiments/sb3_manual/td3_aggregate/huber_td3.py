"""SB3 TD3 with a Huber regression loss for both critics."""

import random

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import TD3
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.utils import polyak_update, update_learning_rate


CRITIC_LOSS_ID = "huber_delta_1_v1"


class HuberTD3(TD3):
    """Use Huber with delta 1 while retaining TD3's update schedule."""

    def __init__(
        self,
        *args,
        log_td_errors: bool = False,
        require_complete_n_step_batch: bool = False,
        actor_learning_rate: float | None = None,
        critic_learning_rate: float | None = None,
        actor_output_init_gain: float | None = None,
        log_actor_gradients: bool = False,
        **kwargs,
    ) -> None:
        if (actor_learning_rate is None) != (critic_learning_rate is None):
            raise ValueError("Set both actor_learning_rate and critic_learning_rate for independent rates")
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.log_td_errors = log_td_errors
        self.require_complete_n_step_batch = require_complete_n_step_batch
        self.actor_output_init_gain = actor_output_init_gain
        self.log_actor_gradients = log_actor_gradients
        super().__init__(*args, **kwargs)
        # SB3 load constructs with _init_setup_model=False, then restores the networks.
        if actor_output_init_gain is not None and hasattr(self, "actor"):
            output = next(layer for layer in reversed(self.actor.mu) if isinstance(layer, th.nn.Linear))
            th.nn.init.orthogonal_(output.weight, gain=actor_output_init_gain)
            th.nn.init.zeros_(output.bias)
            self.actor_target.load_state_dict(self.actor.state_dict())

    def _setup_model(self) -> None:
        super()._setup_model()
        self._apply_independent_learning_rates()

    def _apply_independent_learning_rates(self) -> None:
        if self.actor_learning_rate is not None and self.critic_learning_rate is not None:
            update_learning_rate(self.actor.optimizer, self.actor_learning_rate)
            update_learning_rate(self.critic.optimizer, self.critic_learning_rate)

    def _update_learning_rate(self, optimizers: list[th.optim.Optimizer] | th.optim.Optimizer) -> None:
        if self.actor_learning_rate is None:
            super()._update_learning_rate(optimizers)
            return
        self._apply_independent_learning_rates()
        self.logger.record("train/actor_learning_rate", self.actor.optimizer.param_groups[0]["lr"])
        self.logger.record("train/critic_learning_rate", self.critic.optimizer.param_groups[0]["lr"])

    def eligible_n_step_starts(self) -> int:
        """Count distinct starts with a full horizon or an observed episode boundary."""
        replay = self.replay_buffer
        if not isinstance(replay, NStepReplayBuffer) or replay.n_envs != 1:
            raise ValueError("Complete n-step batches require the native SB3 buffer and one environment")
        stored = replay.size()
        tail_size = min(stored, replay.n_steps - 1)
        eligible = stored - tail_size
        # Only the newest incomplete horizons need checking. A real boundary
        # makes all earlier starts valid, including those truncated by a timeout.
        tail = (replay.pos + np.arange(-tail_size, 0)) % replay.buffer_size
        boundaries = np.flatnonzero(np.logical_or(replay.dones[tail, 0], replay.timeouts[tail, 0]))
        if boundaries.size:
            eligible += int(boundaries[-1]) + 1
        return eligible

    @th.no_grad()
    def _record_actor_gradients(self, actions: th.Tensor) -> None:
        parameter_norms = th.stack([
            th.linalg.vector_norm(parameter.grad)
            for parameter in self.actor.parameters() if parameter.grad is not None
        ])
        grad_l2, derivative, saturation = th.stack((
            th.linalg.vector_norm(parameter_norms),
            (1.0 - actions.square()).mean(),
            (actions.abs() >= 0.99).float().mean(),
        )).tolist()
        self.logger.record("train/actor_grad_l2", grad_l2)
        self.logger.record("train/actor_tanh_derivative_mean", derivative)
        self.logger.record("train/actor_action_saturation_fraction", saturation)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Adapted from SB3 TD3.train with Huber critic loss and optional diagnostics.
        replay = self.replay_buffer
        eligible = None
        if self.require_complete_n_step_batch:
            eligible = self.eligible_n_step_starts()
            if eligible < max(batch_size, self.batch_size):
                return
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        absolute_td_errors = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            if eligible is None:
                replay_data = replay.sample(batch_size, env=self._vec_normalize_env)
            else:
                # Draw distinct chronological starts without scanning the buffer.
                # SB3 still computes returns, discounts and next observations.
                oldest = replay.pos if replay.full else 0
                starts = np.asarray(random.sample(range(eligible), batch_size), dtype=np.intp)
                indices = (oldest + starts) % replay.buffer_size
                replay_data = replay._get_samples(indices, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Smooth the target actions and use the smaller target critic value.
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            if self.log_td_errors:
                with th.no_grad():
                    errors = th.cat(current_q_values, dim=1) - target_q_values
                    absolute_td_errors.append(errors.abs().flatten())
            critic_loss = sum(F.huber_loss(current_q, target_q_values, delta=1.0) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actor_actions).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if self.log_actor_gradients:
                    self._record_actor_gradients(actor_actions)
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if absolute_td_errors:
            errors = th.cat(absolute_td_errors).cpu().numpy()
            median, p90, p99 = np.percentile(errors, [50, 90, 99])
            self.logger.record("train/td_error_abs_mean", errors.mean())
            self.logger.record("train/td_error_abs_median", median)
            self.logger.record("train/td_error_abs_p90", p90)
            self.logger.record("train/td_error_abs_p99", p99)
            self.logger.record("train/td_error_fraction_above_1", np.mean(errors > 1.0))
