"""Masked direct-power Double-DQN; scale fully composed rewards at insertion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Sequence

import torch
from torch import nn

from dqn.networks.mlp_qnet import MLPQNetwork

from .control import ACTION_KW, DirectTransition
from .failure_replay import FAILURE_TERMINALS
from .prioritized_replay import PriorityIndex


STATE_DIM = 7
STATE_FEATURES = (
    'soc', 'previous_fc_kw_over_600', 'previous_fc_delta_kw_over_600',
    'second_previous_fc_delta_kw_over_600', 'current_load_kw_over_600',
    'current_load_delta_kw_over_600', 'previous_load_delta_kw_over_600',
)
ECONOMIC_LEARNING_RATE = 1e-3
ECONOMIC_HIDDEN_DIMS = (128, 128)
ECONOMIC_REPLAY_FORMAT_VERSION = 2
PRIORITIZED_REPLAY_FORMAT_VERSION = 3
WATCHED_REPLAY_SOURCES = frozenset((('zero_boundary_015', 97),
                                  ('zero_boundary_005', 208), ('zero_boundary_006', 350)))


def masked_double_dqn_targets(
    rewards: torch.Tensor, online_next_q: torch.Tensor,
    target_next_q: torch.Tensor, done: torch.Tensor, *,
    next_action_masks: torch.Tensor, gamma: float,
    bootstrap_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Choose the next feasible action online; evaluate it with the target net."""
    if online_next_q.shape != target_next_q.shape or online_next_q.shape != next_action_masks.shape:
        raise ValueError("next-Q arrays and action mask must share a shape")
    if rewards.shape != done.shape or rewards.shape != (online_next_q.shape[0], 1):
        raise ValueError("rewards and done must be column vectors")
    if ((done == 0).flatten() & ~next_action_masks.any(dim=1)).any():
        raise ValueError("a nonterminal next state needs a feasible action")
    selected = online_next_q.masked_fill(~next_action_masks, -torch.inf).argmax(dim=1, keepdim=True)
    if bootstrap_steps is not None:
        if bootstrap_steps.shape != rewards.shape or (bootstrap_steps < 1).any():
            raise ValueError('bootstrap_steps must be positive column vectors')
    discount = gamma if bootstrap_steps is None else gamma ** bootstrap_steps
    # A terminal never uses next-Q, even when that unused value is NaN/Inf.
    continuation=torch.where(done==0,discount*target_next_q.gather(1,selected),torch.zeros_like(rewards))
    return rewards+continuation


@dataclass(frozen=True)
class ReplayItem:
    state: tuple[float, ...]
    action_index: int
    reward_cny: float  # Legacy name: economic training reward in reward_scale units.
    next_state: tuple[float, ...]
    done: bool
    next_feasible_indices: tuple[int, ...]
    bootstrap_steps: int = 1
    experience_outcome: str = 'success'
    terminal_reason: str | None = None
    failure_penalty_equivalent_cny: float = 0.0
    # Diagnostic provenance only; never part of policy observations or TD inputs.
    replay_id: int | None = None
    source_sample_id: str | None = None
    source_round: int | None = None
    source_row_index: int | None = None


class DirectPowerDDQN:
    def __init__(
        self, *, seed: int = 42, gamma: float = 1.0,
        learning_rate: float = ECONOMIC_LEARNING_RATE, replay_capacity: int = 100_000,
        hidden_dims: tuple[int, ...] = ECONOMIC_HIDDEN_DIMS,
        n_step: int = 1,
        reward_scale: float = 1.0,
        failure_terminal_quota: int = 0,
        target_tau: float = 0.001,
        replay_diagnostics: bool = True,
        replay_sampling: str = 'uniform',
        per_alpha: float = 0.6,
        per_beta: float = 1.0,
        per_epsilon: float = 1e-6,
    ) -> None:
        if not 0 <= gamma <= 1 or learning_rate <= 0 or replay_capacity < 1:
            raise ValueError("invalid Double-DQN hyperparameters")
        if type(n_step) is not int or n_step not in (1, 8):
            raise ValueError('n_step must be 1 or 8')
        if not math.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('reward_scale must be finite and positive')
        if type(failure_terminal_quota) is not int or failure_terminal_quota < 0:
            raise ValueError('failure_terminal_quota must be a nonnegative integer')
        if not math.isfinite(target_tau) or not 0 < target_tau <= 1:
            raise ValueError('target_tau must be in (0, 1]')
        if replay_sampling not in ('uniform', 'prioritized'):
            raise ValueError('replay_sampling must be uniform or prioritized')
        if (not math.isfinite(per_alpha) or not 0 <= per_alpha <= 1
                or not math.isfinite(per_beta) or not 0 <= per_beta <= 1
                or not math.isfinite(per_epsilon) or per_epsilon <= 0):
            raise ValueError('invalid prioritized replay parameters')
        self.replay_sampling = replay_sampling
        self.per_alpha, self.per_beta, self.per_epsilon = float(per_alpha), float(per_beta), float(per_epsilon)
        # With a quota, PER operates only on the ordinary pool. Quota=0 uses all live entries.
        self._priority_index = (PriorityIndex(replay_capacity, alpha=self.per_alpha, epsilon=self.per_epsilon)
                                if replay_sampling == 'prioritized' else None)
        self.reward_scale = float(reward_scale)
        self.target_tau = float(target_tau)
        self.failure_terminal_quota = failure_terminal_quota
        self.n_step = n_step
        self.replay_diagnostics = replay_diagnostics
        self.gamma = float(gamma)
        self.random = random.Random(seed)
        torch.manual_seed(seed)
        self.online = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.target = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.target_sync_calls = 0
        self.target_soft_update_calls = 0
        self.economic_optimizer_updates = 0
        self.economic_replay_insertions = 0
        self.economic_success_replay_insertions = 0
        self.economic_failure_replay_insertions = 0
        self.economic_failure_terminal_insertions = 0
        self.economic_unresolved_replay_insertions = 0
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[ReplayItem] = deque(maxlen=replay_capacity)
        # Track live FIFO entries in disjoint pools without rescanning replay at each update.
        self._replay_ids: deque[int] = deque()
        self._next_replay_id = 0
        self._failure_terminal_pool: list[tuple[int, ReplayItem]] = []
        self._ordinary_pool: list[tuple[int, ReplayItem]] = []
        self._failure_terminal_positions: dict[int, int] = {}
        self._ordinary_positions: dict[int, int] = {}
        self._pool_categories: dict[str, int] = {}
        self._pool_coverage: dict[str, dict[str, int]] = {}
        self._watched_ids: dict[tuple[str, int], deque[int]] = {}
        self._watched_draws: dict[int, int] = {}
        self._td_updates = self._td_samples = 0
        self._td_loss_sum = self._td_sum = self._td_abs_sum = self._td_square_sum = self._td_abs_max = 0.0
        self._failure_td_samples = 0
        self._failure_td_sum = self._failure_td_abs_sum = self._failure_td_square_sum = self._failure_td_abs_max = 0.0
        self._reset_extended_statistics()

    @staticmethod
    def _state(value: Sequence[float]) -> tuple[float, ...]:
        state = tuple(float(item) for item in value)
        if len(state) != STATE_DIM or not all(math.isfinite(item) for item in state):
            raise ValueError("state must have seven finite features")
        return state

    @staticmethod
    def _indices(actions: Sequence[int]) -> tuple[int, ...]:
        values = tuple(actions)
        if any(type(value) is not int or value not in ACTION_KW for value in values):
            raise ValueError("feasible actions must use the 10 kW grid")
        if len(set(values)) != len(values):
            raise ValueError("feasible actions contain duplicates")
        return tuple(ACTION_KW.index(value) for value in values)

    def select_power(
        self, state: Sequence[float], feasible_actions: Sequence[int], *, epsilon: float = 0.0,
    ) -> int:
        values = self._state(state)
        indices = self._indices(feasible_actions)
        if not indices:
            raise ValueError("no feasible FC action")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.random.random() < epsilon:
            return ACTION_KW[self.random.choice(indices)]
        with torch.no_grad():
            q = self.online(torch.tensor([values], dtype=torch.float32))[0]
        if not torch.isfinite(q).all():
            raise ValueError('online Q model produced non-finite values')
        mask = torch.full_like(q, -torch.inf)
        mask[list(indices)] = q[list(indices)]
        return ACTION_KW[int(mask.argmax().item())]

    def remember(
        self, state: Sequence[float], action_kw: int, reward_cny: float,
        next_state: Sequence[float], *, done: bool,
        next_feasible_actions: Sequence[int],
        bootstrap_steps: int = 1,
        experience_outcome: str = 'success', terminal_reason: str | None = None,
        failure_penalty_equivalent_cny: float = 0.0,
        source_sample_id: str | None = None, source_round: int | None = None,
        source_row_index: int | None = None,
    ) -> None:
        """Insert an original-unit, fully composed reward; scale exactly once.

        Callers assemble economic/shaping/redistribution/terminal/failure and
        n-step components before this boundary. Failure penalty metadata
        remains in original units, independently of this training scale.
        """
        if type(action_kw) is not int or action_kw not in ACTION_KW:
            raise ValueError("action must use the FC grid")
        if not math.isfinite(reward_cny) or type(done) is not bool:
            raise ValueError("reward must be finite and done must be bool")
        scaled_reward = float(reward_cny) * self.reward_scale
        if not math.isfinite(scaled_reward):
            raise ValueError('scaled reward must be finite')
        if type(bootstrap_steps) is not int or not 1 <= bootstrap_steps <= self.n_step:
            raise ValueError('bootstrap_steps outside configured n-step horizon')
        next_indices = self._indices(next_feasible_actions)
        if (done and next_indices) or (not done and not next_indices):
            raise ValueError("next feasible actions do not match terminal flag")
        reason=('completed' if done else None) if terminal_reason is None else terminal_reason
        if (experience_outcome not in ('success','failure','unresolved') or
                (reason not in (None,'completed') and reason not in FAILURE_TERMINALS) or
                bool(reason) != done or (reason in FAILURE_TERMINALS and experience_outcome!='failure') or
                (reason=='completed' and experience_outcome!='success') or
                (experience_outcome=='unresolved' and (done or reason is not None or failure_penalty_equivalent_cny)) or
                not math.isfinite(failure_penalty_equivalent_cny) or failure_penalty_equivalent_cny<0 or
                (failure_penalty_equivalent_cny and reason not in FAILURE_TERMINALS)):
            raise ValueError('invalid economic experience outcome or terminal reason')
        if (source_sample_id is not None and (not isinstance(source_sample_id, str) or not source_sample_id)
                or any(value is not None and (type(value) is not int or value < 0)
                       for value in (source_round, source_row_index))):
            raise ValueError('invalid replay source metadata')
        item = ReplayItem(
            self._state(state), ACTION_KW.index(action_kw), scaled_reward,
            self._state(next_state), done, next_indices, bootstrap_steps,
            experience_outcome,reason,failure_penalty_equivalent_cny,
            self._next_replay_id, source_sample_id, source_round, source_row_index,
        )
        if len(self.replay) == self.replay.maxlen:
            expired_id = self._replay_ids.popleft()
            if self._priority_index is not None:
                self._priority_index.remove(expired_id)
            if expired_id in self._failure_terminal_positions:
                pool, positions = self._failure_terminal_pool, self._failure_terminal_positions
            else:
                pool, positions = self._ordinary_pool, self._ordinary_positions
            expired_position = positions.pop(expired_id)
            self._record_pool_entry(pool[expired_position][1], -1)
            self._watched_draws.pop(expired_id, None)
            for ids in self._watched_ids.values():
                if expired_id in ids:
                    ids.remove(expired_id)
            last_entry = pool.pop()
            if expired_position < len(pool):
                pool[expired_position] = last_entry
                positions[last_entry[0]] = expired_position
        entry_id = self._next_replay_id
        self._next_replay_id += 1
        self.replay.append(item)
        self._replay_ids.append(entry_id)
        if reason in FAILURE_TERMINALS:
            pool, positions = self._failure_terminal_pool, self._failure_terminal_positions
        else:
            pool, positions = self._ordinary_pool, self._ordinary_positions
        positions[entry_id] = len(pool)
        pool.append((entry_id, item))
        if self._priority_index is not None and (self.failure_terminal_quota == 0 or reason not in FAILURE_TERMINALS):
            self._priority_index.add(entry_id)
        self._record_pool_entry(item, 1)
        watch = (source_sample_id, source_row_index)
        if self.replay_diagnostics and watch in WATCHED_REPLAY_SOURCES:
            ids = self._watched_ids.setdefault(watch, deque())
            if len(ids) == 3:
                self._watched_draws.pop(ids.popleft(), None)
            ids.append(entry_id)
            self._watched_draws[entry_id] = 0
        self.economic_replay_insertions += 1
        if experience_outcome=='failure':
            self.economic_failure_replay_insertions += 1
        elif experience_outcome=='success':
            self.economic_success_replay_insertions += 1
        else:
            self.economic_unresolved_replay_insertions += 1
        self.economic_failure_terminal_insertions += int(reason in FAILURE_TERMINALS)

    def remember_transition(self, transition: DirectTransition) -> None:
        if self.n_step != 1:
            raise ValueError('n-step insertion requires a completed trajectory')
        self.remember(
            transition.state, transition.action_kw, transition.reward_cny,
            transition.next_state, done=transition.done,
            next_feasible_actions=transition.next_feasible_actions,
            experience_outcome='failure' if transition.terminal_reason in FAILURE_TERMINALS else 'success',
            terminal_reason=transition.terminal_reason,
            failure_penalty_equivalent_cny=transition.failure_penalty_equivalent_cny,
        )

    @staticmethod
    def _source_rows(values, rows):
        if rows is None:
            return (None,) * len(values)
        rows = tuple(rows)
        if len(rows) != len(values) or any(type(row) is not int or row < 0 for row in rows):
            raise ValueError('source rows must match executed decisions')
        return rows

    def remember_trajectory(self, transitions: Sequence[DirectTransition], *,
                            source_sample_id=None, source_round=None, source_rows=None) -> int:
        """One entry per real decision; windows may cross shore within a sample."""
        values = tuple(transitions)
        rows = self._source_rows(values, source_rows)
        if not values:
            return 0
        if not values[-1].done:
            raise ValueError('economic replay requires completed or explicitly failure-terminated voyages')
        # Validate the whole input before modifying the replay buffer.
        for transition in values:
            self._state(transition.state)
            self._state(transition.next_state)
            self._indices(transition.next_feasible_actions)
            if (not math.isfinite(transition.reward_cny)
                    or type(transition.action_kw) is not int or type(transition.done) is not bool
                    or transition.action_kw not in ACTION_KW
                    or bool(transition.next_feasible_actions) == transition.done):
                raise ValueError('invalid completed trajectory transition')
        if any(not left.done and left.next_state != right.state
               for left, right in zip(values, values[1:])):
            raise ValueError('economic trajectories must be contiguous within each voyage')
        segment_start = 0
        for segment_end, transition in enumerate(values, start=1):
            if not transition.done:
                continue
            for index in range(segment_start, segment_end):
                end = min(index + self.n_step, segment_end)
                tail = values[end - 1]
                reward = (values[index].reward_cny if self.n_step == 1 else math.fsum(
                    self.gamma ** offset * item.reward_cny
                    for offset, item in enumerate(values[index:end])))
                self.remember(values[index].state, values[index].action_kw, reward,
                              tail.next_state, done=tail.done,
                              next_feasible_actions=tail.next_feasible_actions,
                              bootstrap_steps=end-index,
                              experience_outcome='failure' if transition.terminal_reason in FAILURE_TERMINALS else 'success',
                              terminal_reason=tail.terminal_reason,
                              failure_penalty_equivalent_cny=tail.failure_penalty_equivalent_cny,
                              source_sample_id=source_sample_id, source_round=source_round,
                              source_row_index=rows[index])
            segment_start = segment_end
        return len(values)

    def remember_completed_prefix(self, transitions: Sequence[DirectTransition], *,
                                  source_sample_id=None, source_round=None, source_rows=None) -> int:
        """Retain only fully ended voyages before a later failed voyage."""
        values = tuple(transitions)
        rows = self._source_rows(values, source_rows)
        last_completed_index = max(
            (index for index, transition in enumerate(values) if transition.is_successful_terminal),
            default=-1,
        )
        self.remember_trajectory(values[:last_completed_index + 1],
            source_sample_id=source_sample_id, source_round=source_round,
            source_rows=None if source_rows is None else rows[:last_completed_index+1])
        return last_completed_index + 1

    def remember_observed_prefix(self, transitions: Sequence[DirectTransition], *,
                                 source_sample_id=None, source_round=None, source_rows=None) -> int:
        """For data truncation only: full n-step windows with an observed successor.

        Unlike a real terminal, the unclosed tail is never shortened or assigned
        zero continuation. Keep the existing masks, rewards and shore costs.
        """
        values = tuple(transitions)
        rows = self._source_rows(values, source_rows)
        if not values:
            return 0
        if (values[-1].done or any(t.terminal_reason in FAILURE_TERMINALS for t in values)
                or values[-1].terminal_reason not in (None, 'data_truncation')):
            raise ValueError('observed prefix is for data truncation, not physical termination')
        # Validate before inserting anything. Only the last unobserved successor
        # may have an empty mask or a data-truncation reason.
        for i, t in enumerate(values):
            self._state(t.state)
            self._state(t.next_state)
            self._indices(t.next_feasible_actions)
            if (not math.isfinite(t.reward_cny) or type(t.action_kw) is not int
                    or t.action_kw not in ACTION_KW or type(t.done) is not bool
                    or t.failure_penalty_equivalent_cny
                    or (t.done and (t.next_feasible_actions or not t.is_successful_terminal))
                    or (not t.done and i < len(values)-1 and
                        (not t.next_feasible_actions or t.terminal_reason is not None))):
                raise ValueError('invalid observed data-truncation prefix')
        if any(not left.done and left.next_state != right.state for left, right in zip(values, values[1:])):
            raise ValueError('observed prefix must be contiguous before data truncation')
        completed = self.remember_completed_prefix(values, source_sample_id=source_sample_id,
            source_round=source_round, source_rows=None if source_rows is None else rows)
        inserted = completed
        for index in range(completed, len(values)-self.n_step):
            end = index+self.n_step
            tail = values[end-1]
            if tail.done or not tail.next_feasible_actions or tail.next_state != values[end].state:
                raise ValueError('observed n-step window needs a real feasible successor')
            reward = math.fsum(self.gamma**j*t.reward_cny for j, t in enumerate(values[index:end]))
            self.remember(values[index].state, values[index].action_kw, reward,
                tail.next_state, done=False, next_feasible_actions=tail.next_feasible_actions,
                bootstrap_steps=self.n_step, experience_outcome='unresolved',
                source_sample_id=source_sample_id, source_round=source_round, source_row_index=rows[index])
            inserted += 1
        return inserted

    def learn(self, *, batch_size: int = 64) -> float | None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.replay) < batch_size:
            return None
        batch, weights, probabilities = self._sample_replay_batch_with_weights(batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item.next_state for item in batch], dtype=torch.float32)
        actions = torch.tensor([[item.action_index] for item in batch], dtype=torch.long)
        rewards = torch.tensor([[item.reward_cny] for item in batch], dtype=torch.float32)
        done = torch.tensor([[float(item.done)] for item in batch], dtype=torch.float32)
        masks = torch.zeros((batch_size, len(ACTION_KW)), dtype=torch.bool)
        for row, item in enumerate(batch):
            masks[row, list(item.next_feasible_indices)] = True
        q = self.online(states).gather(1, actions)
        with torch.no_grad():
            target = masked_double_dqn_targets(
                rewards, self.online(next_states), self.target(next_states), done,
                next_action_masks=masks, gamma=self.gamma,
                bootstrap_steps=(torch.tensor([[item.bootstrap_steps] for item in batch])
                                 if self.n_step != 1 else None),
            )
        if self.replay_sampling == 'uniform':
            loss = nn.functional.smooth_l1_loss(q, target)
        else:
            # Apply IS before reducing; weighting an already averaged Huber loss is wrong.
            sample_loss = nn.functional.smooth_l1_loss(q, target, reduction='none')
            loss = (sample_loss * torch.tensor(weights, dtype=q.dtype).reshape(-1, 1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite economic TD loss; optimizer was not updated')
        self.optimizer.zero_grad()
        loss.backward()
        preclip_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.economic_optimizer_updates += 1
        self.soft_update_target()
        with torch.no_grad():
            error = target - q
            if self._priority_index is not None:
                # Repeated PER draws are real entries; update each ID once with its unweighted TD.
                priority_errors = {}
                for item, absolute_error in zip(batch, error.abs().flatten().tolist()):
                    if self._priority_index.contains(item.replay_id):
                        priority_errors[item.replay_id] = max(priority_errors.get(item.replay_id, 0.), absolute_error)
                for key, absolute_error in priority_errors.items():
                    self._priority_index.update(key, absolute_error)
            self._sampling_weight_sum += sum(weights)
            self._sampling_weight_min = min(self._sampling_weight_min, min(weights))
            self._sampling_weight_max = max(self._sampling_weight_max, max(weights))
            self._sampling_repeated_draws += len(batch) - len({t.replay_id for t in batch})
            self._td_updates += 1
            self._td_samples += batch_size
            self._td_loss_sum += float(loss.item())
            self._unweighted_loss_sum += (float(loss.item()) if self.replay_sampling == 'uniform'
                                         else float(sample_loss.mean().item()))
            self._td_sum += float(error.sum().item())
            self._td_abs_sum += float(error.abs().sum().item())
            self._td_square_sum += float(error.square().sum().item())
            self._td_abs_max = max(self._td_abs_max, float(error.abs().max().item()))
            norm = float(preclip_norm.item())
            self._gradient_norm_sum += norm
            self._gradient_norm_max = max(self._gradient_norm_max, norm)
            self._gradient_clipped_updates += int(norm > 10.0)
            self._sample_success += sum(item.experience_outcome == 'success' for item in batch)
            self._sample_failure += sum(item.experience_outcome == 'failure' for item in batch)
            self._sample_unresolved += sum(item.experience_outcome == 'unresolved' for item in batch)
            if self.replay_diagnostics:
                self._sampled_replay_ids_since_reset.update(item.replay_id for item in batch)
                for item, q_value, y_value, td_value in zip(batch, q.flatten().tolist(),
                                                         target.flatten().tolist(), error.flatten().tolist()):
                    category = self._category(item)
                    group = self._category_td.setdefault(category, [0, 0., 0., 0., 0., 0., 0., 0])
                    group[0] += 1
                    group[1] += td_value
                    group[2] += abs(td_value)
                    group[3] += td_value**2
                    group[4] = max(group[4], abs(td_value))
                    group[5] += q_value
                    group[6] += y_value
                    group[7] += int(abs(td_value) > 1.)
                    self._record_coverage(self._draw_coverage, item, 1)
                    if item.replay_id in self._watched_draws:
                        self._watched_draws[item.replay_id] += 1
            for name, values in (('q', q), ('target', target)):
                values = values.double()
                self._distribution_sum[name] += float(values.sum().item())
                self._distribution_square_sum[name] += float(values.square().sum().item())
                self._distribution_min[name] = min(self._distribution_min[name], float(values.min().item()))
                self._distribution_max[name] = max(self._distribution_max[name], float(values.max().item()))
            failure_mask=torch.tensor([item.terminal_reason in FAILURE_TERMINALS for item in batch],dtype=torch.bool)
            failure_error=error.flatten()[failure_mask]
            if failure_error.numel():
                self._failure_td_samples += failure_error.numel()
                self._failure_td_sum += float(failure_error.sum().item())
                self._failure_td_abs_sum += float(failure_error.abs().sum().item())
                self._failure_td_square_sum += float(failure_error.square().sum().item())
                self._failure_td_abs_max=max(self._failure_td_abs_max,float(failure_error.abs().max().item()))
        return float(loss.item())

    @property
    def failure_terminal_replay_count(self) -> int:
        return len(self._failure_terminal_pool)

    @property
    def normal_terminal_replay_count(self) -> int:
        return self._pool_categories.get('normal_terminal', 0)

    @staticmethod
    def _category(item: ReplayItem) -> str:
        if item.terminal_reason in FAILURE_TERMINALS:
            return 'failure_terminal'
        if item.done:
            return 'normal_terminal'
        if item.experience_outcome == 'failure':
            return 'failure_prefix'
        if item.experience_outcome == 'unresolved':
            return 'unresolved_prefix'
        return 'ordinary_nonterminal'

    @staticmethod
    def _record_coverage(coverage, item, increment):
        soc, load = item.state[0], item.state[4]*600.
        labels = {
            'soc': '<0.4' if soc < .4 else '0.4-0.6' if soc < .6 else '0.6-0.79' if soc < .79 else '>=0.79',
            'load_kw': '<200' if load < 200 else '200-600' if load < 600 else '>=600',
            'action_kw': str(ACTION_KW[item.action_index]),
            'source_sample': item.source_sample_id or 'unrecorded',
            'source_round': 'unrecorded' if item.source_round is None else str(item.source_round),
        }
        for dimension, label in labels.items():
            counts = coverage.setdefault(dimension, {})
            counts[label] = counts.get(label, 0) + increment
            if counts[label] == 0:
                del counts[label]

    def _record_pool_entry(self, item, increment):
        category = self._category(item)
        self._pool_categories[category] = self._pool_categories.get(category, 0) + increment
        if self.replay_diagnostics:
            self._record_coverage(self._pool_coverage, item, increment)

    def _category_statistics(self, category):
        n, total, absolute, square, maximum, q_sum, y_sum, above = self._category_td.get(category, [0]*8)
        errors = {'mean_td_error': total/n if n else None,
                  'mean_absolute_td_error': absolute/n if n else None,
                  'root_mean_square_td_error': math.sqrt(square/n) if n else None,
                  'max_absolute_td_error': maximum if n else None}
        return {'sample_count': n, **errors, 'original_units': self._original_units(errors),
                'mean_q_scaled': q_sum/n if n else None, 'mean_target_scaled': y_sum/n if n else None,
                'absolute_td_above_one_count': above, 'absolute_td_above_one_fraction': above/n if n else None}

    def _watched_target_statistics(self):
        """Recompute bounded live probes at ONE current snapshot; no stored TD y."""
        items = []
        for ids in self._watched_ids.values():
            for key in ids:
                if key in self._ordinary_positions:
                    items.append(self._ordinary_pool[self._ordinary_positions[key]][1])
                elif key in self._failure_terminal_positions:
                    items.append(self._failure_terminal_pool[self._failure_terminal_positions[key]][1])
        if not items:
            return []
        with torch.no_grad():
            states = torch.tensor([t.state for t in items], dtype=torch.float32)
            successors = torch.tensor([t.next_state for t in items], dtype=torch.float32)
            masks = torch.zeros((len(items), len(ACTION_KW)), dtype=torch.bool)
            for i, item in enumerate(items):
                masks[i, list(item.next_feasible_indices)] = True
            q = self.online(states).gather(1, torch.tensor([[t.action_index] for t in items]))
            y = masked_double_dqn_targets(torch.tensor([[t.reward_cny] for t in items]),
                self.online(successors), self.target(successors), torch.tensor([[float(t.done)] for t in items]),
                next_action_masks=masks, gamma=self.gamma,
                bootstrap_steps=torch.tensor([[t.bootstrap_steps] for t in items]))
        return [{'replay_id': t.replay_id, 'source_sample_id': t.source_sample_id,
                 'source_round': t.source_round, 'source_row_index': t.source_row_index,
                 'action_kw': ACTION_KW[t.action_index], 'done': t.done,
                 'bootstrap_steps': t.bootstrap_steps, 'experience_outcome': t.experience_outcome,
                 'stored_reward_scaled': t.reward_cny, 'q_scaled': float(q[i].item()),
                 'target_scaled': float(y[i].item()), 'td_error_scaled': float((y[i]-q[i]).item()),
                 'draw_count_cumulative': self._watched_draws[t.replay_id],
                 'snapshot_optimizer_update': self.economic_optimizer_updates,
                 'scope': 'current Online/Target snapshot; target is not a historically stored label'}
                for i, t in enumerate(items)]

    def _sample_replay_batch(self, batch_size: int) -> list[ReplayItem]:
        """Draw a fixed terminal quota, falling back when either pool is short."""
        if self.replay_sampling == 'prioritized':
            return self._sample_replay_batch_with_weights(batch_size)[0]
        if self.failure_terminal_quota == 0:
            return self.random.sample(tuple(self.replay), batch_size)
        failure_count = min(self.failure_terminal_quota, len(self._failure_terminal_pool), batch_size)
        failure_count = max(failure_count, batch_size - len(self._ordinary_pool))
        ordinary_count = batch_size - failure_count
        batch = [item for _, item in self.random.sample(self._failure_terminal_pool, failure_count)]
        batch.extend(item for _, item in self.random.sample(self._ordinary_pool, ordinary_count))
        self.random.shuffle(batch)
        return batch

    def _item_for_replay_id(self, key: int) -> ReplayItem:
        if key in self._ordinary_positions:
            return self._ordinary_pool[self._ordinary_positions[key]][1]
        return self._failure_terminal_pool[self._failure_terminal_positions[key]][1]

    def _sample_replay_batch_with_weights(self, batch_size: int):
        """PER/IS over the ordinary pool, retaining the original fixed failure quota.

        Conditional IS restores the old quota objective, not uniform replay over
        both classes. A SINGLE pool-level normalizer scales both classes equally.
        Ordinary PER draws are with replacement; quota draws remain without it.
        """
        if self.replay_sampling == 'uniform':
            batch = self._sample_replay_batch(batch_size)
            return batch, [1.] * batch_size, [None] * batch_size
        if batch_size < 1 or batch_size > len(self.replay):
            raise ValueError('batch size must fit live replay')
        if self.failure_terminal_quota == 0:
            ids, probabilities = self._priority_index.sample(self.random, batch_size)
            weights = self._priority_index.importance_weights(probabilities, self.per_beta)
            return [self._item_for_replay_id(key) for key in ids], list(weights), list(probabilities)
        failure_count = min(self.failure_terminal_quota, len(self._failure_terminal_pool), batch_size)
        failure_count = max(failure_count, batch_size - len(self._ordinary_pool))
        ordinary_count = batch_size - failure_count
        normalizer = self._priority_index.importance_normalizer(self.per_beta) if ordinary_count else 1.
        entries = [(item, 1. / normalizer, 1. / len(self._failure_terminal_pool))
                   for _, item in self.random.sample(self._failure_terminal_pool, failure_count)]
        if ordinary_count:
            ids, probabilities = self._priority_index.sample(self.random, ordinary_count)
            weights = self._priority_index.importance_weights(probabilities, self.per_beta)
            entries.extend((self._item_for_replay_id(key), weight, probability)
                           for key, weight, probability in zip(ids, weights, probabilities))
        self.random.shuffle(entries)
        return ([t for t, _, _ in entries], [w for _, w, _ in entries], [p for _, _, p in entries])

    def reset_td_statistics(self) -> None:
        """Reset diagnostic accumulators, without changing training state."""
        self._td_updates = self._td_samples = 0
        self._td_loss_sum = self._td_sum = self._td_abs_sum = self._td_square_sum = self._td_abs_max = 0.0
        self._failure_td_samples = 0
        self._failure_td_sum = self._failure_td_abs_sum = self._failure_td_square_sum = self._failure_td_abs_max = 0.0
        self._reset_extended_statistics()

    def _reset_extended_statistics(self) -> None:
        self._sampling_weight_sum = self._sampling_weight_max = 0.
        self._unweighted_loss_sum = 0.
        self._sampling_weight_min = math.inf
        self._sampling_repeated_draws = 0
        self._gradient_norm_sum = self._gradient_norm_max = 0.0
        self._gradient_clipped_updates = 0
        self._sample_success = self._sample_failure = self._sample_unresolved = 0
        self._category_td = {}
        self._draw_coverage = {}
        self._sampled_replay_ids_since_reset = set()
        self._coverage_initial_pool_size = len(self.replay)
        self._coverage_insertions_at_reset = self.economic_replay_insertions
        self._distribution_sum = dict.fromkeys(('q', 'target'), 0.0)
        self._distribution_square_sum = dict.fromkeys(('q', 'target'), 0.0)
        self._distribution_min = dict.fromkeys(('q', 'target'), math.inf)
        self._distribution_max = dict.fromkeys(('q', 'target'), -math.inf)

    def _distribution_statistics(self, name: str) -> dict:
        n = self._td_samples
        mean = self._distribution_sum[name] / n if n else None
        values = {
            'mean': mean,
            'std': math.sqrt(max(0.0, self._distribution_square_sum[name] / n - mean**2)) if n else None,
            'min': self._distribution_min[name] if n else None,
            'max': self._distribution_max[name] if n else None,
        }
        return {'sample_count': n, **values, 'original_units': self._original_units(values),
                'scope': 'sampled state-action Q before update' if name == 'q' else 'sampled masked Double-DQN targets'}

    def _original_units(self, values: dict) -> dict:
        return {key: None if value is None else value / self.reward_scale for key, value in values.items()}

    def td_statistics(self) -> dict:
        n = self._td_samples
        coverage_insertions = self.economic_replay_insertions - self._coverage_insertions_at_reset
        coverage_denominator = self._coverage_initial_pool_size + coverage_insertions
        result = {
            'optimizer_updates': self._td_updates, 'sample_count': n,
            'mean_smooth_l1_loss': self._td_loss_sum / self._td_updates if self._td_updates else None,
            'mean_td_error': self._td_sum / n if n else None,
            'mean_absolute_td_error': self._td_abs_sum / n if n else None,
            'root_mean_square_td_error': math.sqrt(self._td_square_sum / n) if n else None,
            'max_absolute_td_error': self._td_abs_max if n else None,
            'failure_terminal': {
                'sample_count':self._failure_td_samples,
                'mean_td_error':self._failure_td_sum/self._failure_td_samples if self._failure_td_samples else None,
                'mean_absolute_td_error':self._failure_td_abs_sum/self._failure_td_samples if self._failure_td_samples else None,
                'root_mean_square_td_error':math.sqrt(self._failure_td_square_sum/self._failure_td_samples) if self._failure_td_samples else None,
                'max_absolute_td_error':self._failure_td_abs_max if self._failure_td_samples else None,
            },
        }
        error_names = ('mean_td_error', 'mean_absolute_td_error',
                       'root_mean_square_td_error', 'max_absolute_td_error')
        result.update({
            'reward_scale': self.reward_scale,
            'replay_sampling': {
                'mode': self.replay_sampling, 'alpha': self.per_alpha, 'beta': self.per_beta,
                'epsilon': self.per_epsilon, 'draw_count': n,
                'priority_pool_size': len(self._priority_index) if self._priority_index is not None else None,
                'priority_pool': 'all replay' if self.failure_terminal_quota == 0 else 'non-failure-terminal replay',
                'importance_reference': ('all-pool uniform objective' if self.failure_terminal_quota == 0
                    else 'existing fixed failure-quota objective; not all-pool uniform'),
                'mean_importance_weight': self._sampling_weight_sum / n if n else None,
                'mean_unweighted_smooth_l1_loss': self._unweighted_loss_sum / self._td_updates
                    if self._td_updates else None,
                'min_importance_weight': self._sampling_weight_min if n else None,
                'max_importance_weight': self._sampling_weight_max if n else None,
                'within_batch_repeat_draws': self._sampling_repeated_draws,
                'ordinary_with_replacement': self.replay_sampling == 'prioritized',
                'td_error_statistics_are_unweighted': True,
            },
            'reward_units': 'reward_scale * original CNY-equivalent training reward; not actual expenditure',
            'original_reward_units': 'original CNY-equivalent training reward; shaping/failure are not expenditure',
            'loss_units': ('per-sample IS-weighted Smooth L1 beta=1 in scaled training units'
                if self.replay_sampling == 'prioritized' else
                'native Smooth L1 beta=1 in scaled training units; not an unscaled loss divided by alpha'),
            'original_units': self._original_units({key: result[key] for key in error_names}),
            'q_distribution': self._distribution_statistics('q'),
            'target_distribution': self._distribution_statistics('target'),
            'gradient_statistics': {
                'update_count': self._td_updates,
                'mean_preclip_norm': self._gradient_norm_sum / self._td_updates if self._td_updates else None,
                'max_preclip_norm': self._gradient_norm_max if self._td_updates else None,
                'clip_threshold': 10.0, 'clipped_updates': self._gradient_clipped_updates,
                'clipped_fraction': self._gradient_clipped_updates / self._td_updates if self._td_updates else None,
            },
            'sample_outcomes': {
                'success': self._sample_success, 'failure': self._sample_failure,
                'unresolved': self._sample_unresolved,
                'failure_terminal': self._failure_td_samples,
                'success_fraction': self._sample_success / n if n else None,
                'failure_fraction': self._sample_failure / n if n else None,
                'unresolved_fraction': self._sample_unresolved / n if n else None,
                'failure_terminal_fraction': self._failure_td_samples / n if n else None,
            },
            'normal_terminal': self._category_statistics('normal_terminal'),
            'nonterminal_categories': {name: self._category_statistics(name) for name in
                ('ordinary_nonterminal', 'failure_prefix', 'unresolved_prefix')},
            'replay_coverage': {'pool_categories': dict(self._pool_categories),
                'pool': {key: dict(counts) for key, counts in self._pool_coverage.items()},
                'actual_draws_since_reset': {key: dict(counts) for key, counts in self._draw_coverage.items()},
                'unique_replay_ids_since_reset': {
                    'sampled_count': len(self._sampled_replay_ids_since_reset),
                    'initial_pool_size': self._coverage_initial_pool_size,
                    'insertions_since_reset': coverage_insertions,
                    'denominator': coverage_denominator,
                    'coverage_rate': len(self._sampled_replay_ids_since_reset)/coverage_denominator
                        if coverage_denominator else None,
                    'denominator_definition': 'Pool size at reset plus all replay insertions since reset; '
                        'includes entries evicted before being sampled',
                    'scope': 'Unique replay IDs in actual successful optimizer batches since reset; '
                        'repeated draws count once; bounded to this diagnostic round',
                } if self.replay_diagnostics else None,
                'diagnostics_enabled': self.replay_diagnostics,
                'terminal_counts_are_n_step_windows_not_failure_or_completion_events': True},
            'watched_replay_targets': self._watched_target_statistics() if self.replay_diagnostics else [],
        })
        result['failure_terminal']['original_units'] = self._original_units(
            {key: result['failure_terminal'][key] for key in error_names})
        return result

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target_sync_calls += 1

    def soft_update_target(self) -> None:
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.lerp_(online, self.target_tau)
        self.target_soft_update_calls += 1

    def training_state(self) -> dict:
        """All optimizer, replay, network and RNG state needed at a round boundary."""
        return {
            # Old loaders must reject a PER checkpoint instead of silently dropping priorities.
            'format_version': (PRIORITIZED_REPLAY_FORMAT_VERSION if self.replay_sampling == 'prioritized'
                               else ECONOMIC_REPLAY_FORMAT_VERSION),
            'online': self.online.state_dict(), 'target': self.target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'replay': tuple(self.replay), 'replay_ids': tuple(self._replay_ids),
            'next_replay_id': self._next_replay_id,
            'sampling_config': self._sampling_config(),
            'priority_index': self._priority_index.state_dict() if self._priority_index is not None else None,
            'failure_pool_ids': tuple(key for key, _ in self._failure_terminal_pool),
            'ordinary_pool_ids': tuple(key for key, _ in self._ordinary_pool),
            'random_state': self.random.getstate(),
            'torch_rng_state': torch.random.get_rng_state(),
            'replay_diagnostics': self.replay_diagnostics,
            'pool_categories': dict(self._pool_categories), 'pool_coverage': self._pool_coverage,
            'watched_ids': {key: tuple(ids) for key, ids in self._watched_ids.items()},
            'watched_draws': dict(self._watched_draws),
            'counts': {name: getattr(self, name) for name in (
                'target_sync_calls', 'target_soft_update_calls',
                'economic_optimizer_updates', 'economic_replay_insertions',
                'economic_success_replay_insertions', 'economic_failure_replay_insertions',
                'economic_failure_terminal_insertions', 'economic_unresolved_replay_insertions')},
        }

    def load_training_state(self, state: dict) -> None:
        expected_version = (PRIORITIZED_REPLAY_FORMAT_VERSION if self.replay_sampling == 'prioritized'
                            else ECONOMIC_REPLAY_FORMAT_VERSION)
        if state.get('format_version') != expected_version:
            raise ValueError('economic replay checkpoint format/sampling is incompatible; start a new run')
        if len(state['replay']) > self.replay.maxlen:
            raise ValueError('checkpoint replay exceeds configured capacity')
        # Fail before overwriting parameters if the sampler or priority state is incompatible.
        saved_sampling = state.get('sampling_config', {
            'mode': 'uniform', 'alpha': .6, 'beta': 1., 'epsilon': 1e-6,
            'failure_terminal_quota': self.failure_terminal_quota})
        if saved_sampling != self._sampling_config():
            raise ValueError('checkpoint replay sampling configuration mismatch; start a new run')
        if self._priority_index is not None:
            if not isinstance(state.get('priority_index'), dict):
                raise ValueError('checkpoint priority state is missing')
            restored = PriorityIndex(self.replay.maxlen, alpha=self.per_alpha, epsilon=self.per_epsilon)
            restored.load_state_dict(state['priority_index'])
            eligible = state['replay_ids'] if self.failure_terminal_quota == 0 else state['ordinary_pool_ids']
            if set(key for key, _ in restored.state_dict()['entries']) != set(eligible):
                raise ValueError('checkpoint priority entries do not match live replay')
            self._priority_index = restored
        self.online.load_state_dict(state['online'], strict=True)
        self.target.load_state_dict(state['target'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.replay = deque(state['replay'], maxlen=self.replay.maxlen)
        self._replay_ids = deque(state['replay_ids'])
        if len(self.replay) != len(self._replay_ids):
            raise ValueError('checkpoint replay IDs do not match replay items')
        self._next_replay_id = state['next_replay_id']
        by_id = dict(zip(self._replay_ids, self.replay))
        self._failure_terminal_pool = [(key, by_id[key]) for key in state['failure_pool_ids']]
        self._ordinary_pool = [(key, by_id[key]) for key in state['ordinary_pool_ids']]
        if (len(self._failure_terminal_pool) + len(self._ordinary_pool) != len(self.replay)
                or set(state['failure_pool_ids']).intersection(state['ordinary_pool_ids'])):
            raise ValueError('checkpoint sampling pools do not partition the replay')
        self._failure_terminal_positions = {key: index for index, (key, _) in enumerate(self._failure_terminal_pool)}
        self._ordinary_positions = {key: index for index, (key, _) in enumerate(self._ordinary_pool)}
        self.replay_diagnostics = state['replay_diagnostics']
        self._pool_categories = dict(state['pool_categories'])
        self._pool_coverage = {key: dict(counts) for key, counts in state['pool_coverage'].items()}
        self._watched_ids = {key: deque(ids) for key, ids in state['watched_ids'].items()}
        self._watched_draws = dict(state['watched_draws'])
        if any(key not in by_id for ids in self._watched_ids.values() for key in ids):
            raise ValueError('checkpoint has expired diagnostic replay references')
        for name, value in state['counts'].items():
            setattr(self, name, value)
        self.random.setstate(state['random_state'])
        torch.random.set_rng_state(state['torch_rng_state'])

    def _sampling_config(self) -> dict:
        return {'mode': self.replay_sampling, 'alpha': self.per_alpha, 'beta': self.per_beta,
                'epsilon': self.per_epsilon, 'failure_terminal_quota': self.failure_terminal_quota}
