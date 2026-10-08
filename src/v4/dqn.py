"""Masked MLP Double-DQN for direct FC power and raw-CNY rewards."""

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


STATE_DIM = 8


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
    reward_cny: float
    next_state: tuple[float, ...]
    done: bool
    next_feasible_indices: tuple[int, ...]
    bootstrap_steps: int = 1
    experience_outcome: str = 'success'
    terminal_reason: str | None = None
    failure_penalty_equivalent_cny: float = 0.0


@dataclass(frozen=True)
class OutcomeItem:
    """Observed rollout result; the label is not an economic cost."""

    state: tuple[float, ...]
    action_index: int
    actual_reward_cny: float
    failed: bool


class DirectPowerDDQN:
    def __init__(
        self, *, seed: int = 42, gamma: float = 1.0,
        learning_rate: float = 1e-4, replay_capacity: int = 100_000,
        hidden_dims: tuple[int, ...] = (128, 64),
        n_step: int = 1,
    ) -> None:
        if not 0 <= gamma <= 1 or learning_rate <= 0 or replay_capacity < 1:
            raise ValueError("invalid Double-DQN hyperparameters")
        if type(n_step) is not int or n_step not in (1, 8):
            raise ValueError('n_step must be 1 or 8')
        self.n_step = n_step
        self.gamma = float(gamma)
        self.random = random.Random(seed)
        torch.manual_seed(seed)
        self.online = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.target = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.target_sync_calls = 0
        self.economic_optimizer_updates = 0
        self.economic_replay_insertions = 0
        self.economic_success_replay_insertions = 0
        self.economic_failure_replay_insertions = 0
        self.economic_failure_terminal_insertions = 0
        self.outcome_optimizer_updates = 0
        self.sync_target()
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay: deque[ReplayItem] = deque(maxlen=replay_capacity)
        self.outcome_model = MLPQNetwork(STATE_DIM, len(ACTION_KW), hidden_dims)
        self.outcome_optimizer = torch.optim.Adam(self.outcome_model.parameters(), lr=learning_rate)
        self.outcome_random = random.Random(seed + 1)
        self.outcome_replay: deque[OutcomeItem] = deque(maxlen=replay_capacity)
        self._td_updates = self._td_samples = 0
        self._td_loss_sum = self._td_sum = self._td_abs_sum = self._td_square_sum = self._td_abs_max = 0.0
        self._failure_td_samples = 0
        self._failure_td_sum = self._failure_td_abs_sum = self._failure_td_square_sum = self._failure_td_abs_max = 0.0

    @staticmethod
    def _state(value: Sequence[float]) -> tuple[float, ...]:
        state = tuple(float(item) for item in value)
        if len(state) != STATE_DIM or not all(math.isfinite(item) for item in state):
            raise ValueError("state must have eight finite features")
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
    ) -> None:
        if type(action_kw) is not int or action_kw not in ACTION_KW:
            raise ValueError("action must use the FC grid")
        if not math.isfinite(reward_cny) or type(done) is not bool:
            raise ValueError("reward must be finite and done must be bool")
        if type(bootstrap_steps) is not int or not 1 <= bootstrap_steps <= self.n_step:
            raise ValueError('bootstrap_steps outside configured n-step horizon')
        next_indices = self._indices(next_feasible_actions)
        if (done and next_indices) or (not done and not next_indices):
            raise ValueError("next feasible actions do not match terminal flag")
        reason=('completed' if done else None) if terminal_reason is None else terminal_reason
        if (experience_outcome not in ('success','failure') or
                (reason not in (None,'completed') and reason not in FAILURE_TERMINALS) or
                bool(reason) != done or (reason in FAILURE_TERMINALS and experience_outcome!='failure') or
                (reason=='completed' and experience_outcome!='success') or
                not math.isfinite(failure_penalty_equivalent_cny) or failure_penalty_equivalent_cny<0 or
                (failure_penalty_equivalent_cny and reason not in FAILURE_TERMINALS)):
            raise ValueError('invalid economic experience outcome or terminal reason')
        self.replay.append(ReplayItem(
            self._state(state), ACTION_KW.index(action_kw), float(reward_cny),
            self._state(next_state), done, next_indices, bootstrap_steps,
            experience_outcome,reason,failure_penalty_equivalent_cny,
        ))
        self.economic_replay_insertions += 1
        if experience_outcome=='failure':
            self.economic_failure_replay_insertions += 1
        else:
            self.economic_success_replay_insertions += 1
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

    def remember_trajectory(self, transitions: Sequence[DirectTransition]) -> int:
        """One replay entry per executed decision, with no cross-voyage return."""
        values = tuple(transitions)
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
                              failure_penalty_equivalent_cny=tail.failure_penalty_equivalent_cny)
            segment_start = segment_end
        return len(values)

    def remember_completed_prefix(self, transitions: Sequence[DirectTransition]) -> int:
        """Retain only fully ended voyages before a later failed voyage."""
        values = tuple(transitions)
        last_completed_index = max(
            (index for index, transition in enumerate(values) if transition.is_successful_terminal),
            default=-1,
        )
        self.remember_trajectory(values[:last_completed_index + 1])
        return last_completed_index + 1

    def remember_outcome_trajectory(
        self, transitions: Sequence[DirectTransition], *, failed: bool,
    ) -> None:
        """Retain all executed actions with their eventual observed outcome.

        A failed suffix is never converted to a zero-cost economic terminal.
        Labels describe the sampled behavior trajectory, not causal blame for
        every earlier action or a guaranteed future safety certificate.
        """
        values = tuple(transitions)
        if not values or type(failed) is not bool:
            raise ValueError("outcome trajectory needs executed transitions and a bool label")
        if failed and (values[-1].done or values[-1].next_feasible_actions):
            raise ValueError("failed outcome requires a nonterminal infeasible next state")
        if not failed and not values[-1].is_successful_terminal:
            raise ValueError("completed outcome must end at a completed transition")
        last_completed_index = max(
            (index for index, transition in enumerate(values) if transition.is_successful_terminal),
            default=-1,
        )
        for index, transition in enumerate(values):
            observed_cost = transition.executed_ledger.total_cost_cny
            if transition.shore_ledger is not None:
                observed_cost += transition.shore_ledger.total_cost_cny
            self.outcome_replay.append(OutcomeItem(
                self._state(transition.state), ACTION_KW.index(transition.action_kw),
                -observed_cost, failed and index > last_completed_index,
            ))

    def learn_outcome(self, *, batch_size: int = 64) -> float | None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.outcome_replay) < batch_size:
            return None
        batch = self.outcome_random.sample(tuple(self.outcome_replay), batch_size)
        states = torch.tensor([item.state for item in batch], dtype=torch.float32)
        actions = torch.tensor([[item.action_index] for item in batch], dtype=torch.long)
        failed = torch.tensor([[float(item.failed)] for item in batch], dtype=torch.float32)
        logits = self.outcome_model(states).gather(1, actions)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, failed)
        self.outcome_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.outcome_model.parameters(), 10.0)
        self.outcome_optimizer.step()
        self.outcome_optimizer_updates += 1
        return float(loss.item())

    def learn(self, *, batch_size: int = 64) -> float | None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(self.replay) < batch_size:
            return None
        batch = self.random.sample(tuple(self.replay), batch_size)
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
        loss = nn.functional.smooth_l1_loss(q, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite economic TD loss; optimizer was not updated')
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.economic_optimizer_updates += 1
        with torch.no_grad():
            error = target - q
            self._td_updates += 1
            self._td_samples += batch_size
            self._td_loss_sum += float(loss.item())
            self._td_sum += float(error.sum().item())
            self._td_abs_sum += float(error.abs().sum().item())
            self._td_square_sum += float(error.square().sum().item())
            self._td_abs_max = max(self._td_abs_max, float(error.abs().max().item()))
            failure_mask=torch.tensor([item.terminal_reason in FAILURE_TERMINALS for item in batch],dtype=torch.bool)
            failure_error=error.flatten()[failure_mask]
            if failure_error.numel():
                self._failure_td_samples += failure_error.numel()
                self._failure_td_sum += float(failure_error.sum().item())
                self._failure_td_abs_sum += float(failure_error.abs().sum().item())
                self._failure_td_square_sum += float(failure_error.square().sum().item())
                self._failure_td_abs_max=max(self._failure_td_abs_max,float(failure_error.abs().max().item()))
        return float(loss.item())

    def reset_td_statistics(self) -> None:
        """Reset diagnostic accumulators, without changing training state."""
        self._td_updates = self._td_samples = 0
        self._td_loss_sum = self._td_sum = self._td_abs_sum = self._td_square_sum = self._td_abs_max = 0.0
        self._failure_td_samples = 0
        self._failure_td_sum = self._failure_td_abs_sum = self._failure_td_square_sum = self._failure_td_abs_max = 0.0

    def td_statistics(self) -> dict:
        n = self._td_samples
        return {
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

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())
        self.target_sync_calls += 1
