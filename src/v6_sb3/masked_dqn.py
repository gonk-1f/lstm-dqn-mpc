"""V6 masks on the actual Stable-Baselines3 DQN implementation.

The training method follows SB3 2.9.0 dqn/dqn.py, commit
8908708f10c8ff29759c67f55c8acb56cab27463 (MIT, Antonin Raffin).
Only legal-action selection/targets, V6 optimizer-count Target timing and
bounded diagnostics are added. Policy, collection, storage and sampling
remain SB3 implementations.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer

from v4.actions import FC_INCREMENT_13_KW, previous_grid_power_kw
from v4.failure_replay import FAILURE_TERMINALS
from v4.v6_td_diagnostics import OnlineTDDiagnostics


class MaskReplaySamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    next_action_masks: torch.Tensor
    discounts: torch.Tensor | None = None


def masked_td_targets(rewards, next_q, dones, next_action_masks, gamma=1.0):
    """Ordinary DQN; terminal empty masks contribute exactly zero bootstrap."""
    continuing = dones.flatten() == 0
    if (continuing & ~next_action_masks.any(dim=1)).any():
        raise ValueError('a nonterminal next state needs a feasible action')
    maximum = next_q.masked_fill(~next_action_masks, -torch.inf).amax(dim=1, keepdim=True)
    continuation = torch.where(continuing[:, None], gamma * maximum, torch.zeros_like(rewards))
    return rewards + continuation


class MaskedReplayBuffer(ReplayBuffer):
    """SB3 FIFO and uniform sampling plus the real successor mask/labels."""

    reason_names = (None, 'completed', *sorted(FAILURE_TERMINALS))
    outcome_names = ('success', 'failure', 'unresolved')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise ValueError('the V6 baseline uses one environment and explicit successor observations')
        self.next_action_masks = np.zeros((self.buffer_size, 1, 13), dtype=np.bool_)
        self.outcomes = np.zeros(self.buffer_size, dtype=np.uint8)
        self.reasons = np.zeros(self.buffer_size, dtype=np.uint8)
        self.failure_penalties = np.zeros(self.buffer_size, dtype=np.float32)
        self.insertions = 0
        self.last_sample_indices = np.empty(0, dtype=np.int64)

    def add(self, obs, next_obs, action, reward, done, infos):
        masks = np.asarray([info['next_action_mask'] for info in infos], dtype=np.bool_)
        effective_done = np.asarray(done, dtype=bool) & ~np.asarray(
            [self.handle_timeout_termination and info.get('TimeLimit.truncated', False) for info in infos])
        if masks.shape != (1, 13):
            raise ValueError('each successor mask must contain 13 action flags')
        if (~effective_done & ~masks.any(axis=1)).any():
            raise ValueError('a nonterminal replay item needs a feasible next action')
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.next_action_masks[slot] = masks
        reason = infos[0].get('terminal_reason')
        if reason is None and effective_done[0]:
            reason = 'completed'
        self.reasons[slot] = self.reason_names.index(reason)
        self.outcomes[slot] = (1 if reason in FAILURE_TERMINALS else 0 if effective_done[0] else 2)
        self.failure_penalties[slot] = infos[0].get('failure_penalty_equivalent_cny', 0.0)
        self.insertions += 1

    def _get_samples(self, batch_inds, env=None):
        # Keep SB3's actual normalization, NumPy sampling and tensor conversion.
        data = super()._get_samples(batch_inds, env=env)
        self.last_sample_indices = np.asarray(batch_inds).copy()
        return MaskReplaySamples(data.observations, data.actions, data.next_observations,
            data.dones, data.rewards, self.to_torch(self.next_action_masks[batch_inds, 0]), data.discounts)


class ReplayDiagnosticsView:
    """Zero-copy single-env views for the existing read-only V6 reports."""

    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return self.buffer.size()

    @property
    def maxlen(self): return self.buffer.buffer_size

    @property
    def insertions(self): return self.buffer.insertions

    @property
    def states(self): return self.buffer.observations[:, 0]

    @property
    def actions(self): return self.buffer.actions[:, 0]

    @property
    def rewards(self): return self.buffer.rewards

    @property
    def next_states(self): return self.buffer.next_observations[:, 0]

    @property
    def done(self): return self.buffer.dones

    @property
    def masks(self): return self.buffer.next_action_masks[:, 0]

    @property
    def outcomes(self): return self.buffer.outcomes

    @property
    def reasons(self): return self.buffer.reasons

    @property
    def failure_penalties(self): return self.buffer.failure_penalties

    @property
    def last_sample_indices(self): return self.buffer.last_sample_indices

    @property
    def _reason_names(self): return self.buffer.reason_names

    @property
    def reason_names(self): return self.buffer.reason_names


class MaskedDQN(DQN):
    """SB3 DQN with current/successor masks and V6's optimizer-step schedule."""

    algorithm = 'dqn'
    state_mode = 'history5'
    action_mode = 'increment13'
    state_dim, action_dim = 5, 13
    action_values_kw = FC_INCREMENT_13_KW
    reward_scale = 1.0
    n_step = 1
    replay_sampling = 'uniform'
    failure_terminal_quota = 0
    target_soft_update_calls = 0

    def __init__(self, policy='MlpPolicy', env=None, *, td_diagnostics=True,
                 train_step_observer=None, after_update_observer=None,
                 target_sync_unit='adam_updates', **kwargs):
        if target_sync_unit not in ('adam_updates', 'formal_env_steps'):
            raise ValueError('Target synchronization unit must be adam_updates or formal_env_steps')
        self.target_sync_unit = target_sync_unit
        self.formal_env_steps = 0
        self.target_sync_history = []
        defaults = dict(learning_rate=.0001, buffer_size=1_000_000, learning_starts=50_000,
            batch_size=64, tau=1., gamma=1., train_freq=1, gradient_steps=1,
            replay_buffer_class=MaskedReplayBuffer, optimize_memory_usage=False, n_steps=1,
            target_update_interval=1000, exploration_initial_eps=1., exploration_final_eps=.05,
            max_grad_norm=10., verbose=0, seed=42, device='cpu')
        defaults.update(kwargs)
        policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.ReLU,
            optimizer_class=torch.optim.Adam, optimizer_kwargs=dict(betas=(.9, .999), eps=1e-8))
        policy_kwargs.update(defaults.pop('policy_kwargs', {}))
        self.target_sync_calls = 1  # SB3 DQNPolicy._build initializes Target by copying Online.
        self.td_diagnostics = OnlineTDDiagnostics(defaults['seed'] or 42) if td_diagnostics else None
        self.train_step_observer = train_step_observer
        self.after_update_observer = after_update_observer
        self.last_training_loss = None
        self.training_loss_sum = 0.
        self.training_loss_count = 0
        super().__init__(policy, env, policy_kwargs=policy_kwargs, **defaults)

    def _setup_model(self):
        super()._setup_model()
        self.q_net_target.requires_grad_(False)
        self.q_net_target.eval()

    @property
    def online(self): return self.q_net

    @property
    def target(self): return self.q_net_target

    @property
    def optimizer(self): return self.policy.optimizer

    @property
    def replay(self): return ReplayDiagnosticsView(self.replay_buffer)

    @property
    def economic_optimizer_updates(self): return self._n_updates

    @property
    def economic_replay_insertions(self): return self.replay_buffer.insertions

    def _on_step(self):
        # SB3 calls this after collection. V6 epsilon is set by voyage, and Target
        # is copied only after real optimizer steps, never during random warmup.
        self._n_calls += 1
        self.logger.record('rollout/exploration_rate', self.exploration_rate)

    def record_formal_transition(self):
        """Advance only after storing one actual formal transition.

        The episode runner calls this before an Adam update due at this step.
        Warmup and Greedy evaluation do not advance this clock.
        """
        if self.target_sync_unit != 'formal_env_steps':
            raise ValueError('formal transition clock requires formal_env_steps Target mode')
        self.formal_env_steps += 1
        if self.formal_env_steps % self.target_update_interval == 0:
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.target_sync_calls += 1
            self.target_sync_history.append(dict(formal_env_steps=self.formal_env_steps,
                                                 adam_updates=int(self._n_updates)))

    def masked_predict(self, observation, action_masks, *, deterministic=False, epsilon=None):
        observation = np.asarray(observation, dtype=np.float32)
        obs, vectorized = self.policy.obs_to_tensor(observation)
        masks = np.asarray(action_masks, dtype=np.bool_)
        if masks.ndim == 1:
            masks = masks[None, :]
        if masks.shape != (obs.shape[0], 13) or not masks.any(axis=1).all():
            raise ValueError('action selection requires a nonempty 13-action mask for each state')
        rate = self.exploration_rate if epsilon is None else epsilon
        if not deterministic and np.random.rand() < rate:
            actions = np.asarray([np.random.choice(np.flatnonzero(mask)) for mask in masks])
        else:
            with torch.no_grad():
                q = self.q_net(obs)
                if not torch.isfinite(q).all():
                    raise FloatingPointError('non-finite online action values')
                actions = q.masked_fill(~torch.as_tensor(masks, device=self.device), -torch.inf).argmax(1).cpu().numpy()
        return (actions if vectorized else actions.squeeze(axis=0)), None

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        masks = np.asarray(self.env.env_method('action_masks'), dtype=np.bool_)
        actions, _ = self.masked_predict(observation, masks, deterministic=deterministic)
        return actions, state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        masks = np.asarray(self.env.env_method('action_masks'), dtype=np.bool_)
        if not masks.any(axis=1).all():
            raise ValueError('the environment has no legal action; it must terminate the preceding transition')
        if self.num_timesteps < learning_starts:
            actions = np.asarray([np.random.choice(np.flatnonzero(mask)) for mask in masks], dtype=np.int64)
        else:
            actions, _ = self.masked_predict(self._last_obs, masks, deterministic=False)
        return actions, actions.copy()

    def record_train_step(self, q, target, residual):
        if self.td_diagnostics is not None:
            self.td_diagnostics.observe(self.replay, q, target, residual)

    def train(self, gradient_steps, batch_size=64):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with torch.no_grad():
                next_q = self.q_net_target(batch.next_observations)
                target = masked_td_targets(batch.rewards, next_q, batch.dones,
                    batch.next_action_masks, gamma=self.gamma)
            all_q = self.q_net(batch.observations)
            q = all_q.gather(1, batch.actions.long())
            loss = F.smooth_l1_loss(q, target, beta=1.)
            if not (torch.isfinite(all_q).all() and torch.isfinite(next_q).all()
                    and torch.isfinite(target).all() and torch.isfinite(loss)):
                raise FloatingPointError('non-finite Q, TD target or loss; Adam was not updated')
            self.record_train_step(q, target, target - q)
            if self.train_step_observer is not None:
                self.train_step_observer(self, batch, q.detach(), target.detach())
            self.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.policy.optimizer.step()
            self._n_updates += 1
            self.last_training_loss = float(loss.item())
            self.training_loss_sum += self.last_training_loss
            self.training_loss_count += 1
            if self.target_sync_unit == 'adam_updates' and self._n_updates % self.target_update_interval == 0:
                self.q_net_target.load_state_dict(self.q_net.state_dict())
                self.target_sync_calls += 1
            self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
            self.logger.record('train/loss', self.last_training_loss)
            if self.after_update_observer is not None:
                self.after_update_observer(self)

    def finalize_streamed_failure_prefix(self, replay_ids):
        buffer = self.replay_buffer
        first_live = max(0, buffer.insertions - buffer.buffer_size)
        for replay_id in replay_ids:
            if first_live <= replay_id < buffer.insertions:
                buffer.outcomes[replay_id % buffer.buffer_size] = 1

    def action_index_for_power(self, power_kw, state):
        previous = previous_grid_power_kw(float(state[2]) * 600.)
        return self.action_values_kw.index(power_kw - previous)

    def action_power_kw(self, index, state):
        previous = previous_grid_power_kw(float(state[2]) * 600.)
        return previous + self.action_values_kw[index]

    def select_power(self, state, feasible_actions, *, epsilon=0.):
        mask = np.zeros(13, dtype=np.bool_)
        for power in feasible_actions:
            mask[self.action_index_for_power(power, state)] = True
        action, _ = self.masked_predict(np.asarray(state, dtype=np.float32), mask,
            deterministic=epsilon == 0., epsilon=epsilon)
        return self.action_power_kw(int(action), state)

    def _excluded_save_params(self):
        return [*super()._excluded_save_params(), 'train_step_observer', 'after_update_observer']
