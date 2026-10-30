"""Single-thread, one-second V6 Gym environment over the formal shared physics."""
from __future__ import annotations

from dataclasses import asdict, replace
from math import fsum, isfinite

import gymnasium as gym
import numpy as np

from v2.data.supervisory_rules import OperatingMode, normalize_onboard_load_kw
from v3.control import AccountState, EconomicMPC, SOC_MAX, SOC_MIN
from v4.actions import candidate_fc_actions
from v4.control import (
    DirectTransition, NoFeasibleFCActionError, ReplayExecutionError,
    _causal_load_history, _independent_timing, _v6_feasible_fc_actions,
    _validated_rows_v4, build_state, infeasibility_cause, onboard_reward_components,
)
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.independent_economics import ZhouEvaluationConfig
from v4.reward_feedback import BatteryEnergyValue

from .environment import ACTION_DELTAS, InfeasibleReset, _action_mask
from .rewards import (
    INVERSE_ONE_PLUS_ECONOMIC, REWARD_MODES, economic_cost_cny,
    inverse_one_plus_economic_reward, reward_contract,
)


class DirectVesselEnv(gym.Env):
    """Execute one actual interval per call, retaining only the latest transition.

    Each reset starts a fresh independent voyage (SOC .6, FC off, fresh ageing).
    Real physical failure is attached to the last executed action before step
    returns. An infeasible initial state raises InfeasibleReset with no invented
    action. Checkpoints store the small physical state and causal histories;
    restore never executes an action or reconstructs an action prefix.
    """

    metadata = {'render_modes': []}

    def __init__(self, episode=None, *, accountant=None, economics=None, penalty=None,
                 reward_mode='zhou_economic_only'):
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f'reward_mode must be one of {REWARD_MODES}')
        if reward_mode == INVERSE_ONE_PLUS_ECONOMIC and penalty is not None:
            raise ValueError('inverse_one_plus_economic does not accept a legacy penalty argument')
        self.action_space = gym.spaces.Discrete(13)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.episode = episode
        self.accountant = accountant or EconomicMPC(nominal_cost_cny=1.)
        if not isinstance(self.accountant, EconomicMPC):
            raise TypeError('accountant must supply the formal v3 interval ledger')
        self.economics = economics or ZhouEvaluationConfig.from_formal_curve()
        if type(self.economics) is not ZhouEvaluationConfig:
            raise TypeError('economics must be an explicit ZhouEvaluationConfig')
        self.penalty = (None if reward_mode == INVERSE_ONE_PLUS_ECONOMIC
                        else penalty or FailurePenalty.from_reference())
        self.reward_mode = reward_mode
        self._energy_value = (None if reward_mode == INVERSE_ONE_PLUS_ECONOMIC
                              else BatteryEnergyValue.from_accountant(self.accountant))
        self._physical = AccountState()
        self._index = 0
        self._history = None
        self._fc_history = (0., 0., 0.)
        self._physical_actions = ()
        self._candidates = ()
        self._mask = np.zeros(13, dtype=np.bool_)
        self._ready = False
        self._terminated = False
        self.current_decision_state = None
        self.last_transition = None
        self.episode_status = None

    def _load_episode(self, episode):
        if episode is None:
            raise ValueError('reset requires a canonical V6 episode')
        if getattr(episode, 'task_mode', None) != 'independent_voyage':
            raise ValueError('DirectVesselEnv requires an independent voyage')
        modes, loads, _ = _validated_rows_v4(episode)
        times, durations = _independent_timing(episode, len(modes))
        if any(duration != 1. for duration in durations):
            raise ValueError('DirectVesselEnv requires canonical one-second intervals')
        self.episode = episode
        self._loads, self._times, self._durations = loads, times, durations
        self._lagged_load, self._mean_load = _causal_load_history(loads, times)

    def _state(self, index, history, fc_history):
        return build_state(self._physical, fc_history, history, state_mode='history5',
            load_30s_ago_kw=self._lagged_load[index], mean_load_300s_kw=self._mean_load[index])

    def _legal_actions(self, index):
        physical = _v6_feasible_fc_actions(self._physical, self._loads[index], self.accountant,
                                         duration_s=self._durations[index])
        candidates = candidate_fc_actions(physical, self._physical.previous_fc_kw, 'increment13')
        return physical, candidates

    def _failure_cause(self, index, physical_actions):
        return ('increment_action_limited' if physical_actions else
                infeasibility_cause(self._loads[index], self.accountant))

    def _info(self):
        return {'sample_id': self.episode.sample_id, 'action_mask': self.action_masks()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        options = options or {}
        self._load_episode(options.get('episode', self.episode))
        self._physical = AccountState()
        self._index = 0
        self._history = None
        self._fc_history = (0., 0., 0.)
        self._terminated = False
        self.episode_status = None
        self.last_transition = None
        self.current_decision_state = self._state(0, (self._loads[0],) * 3, self._fc_history)
        self._physical_actions, self._candidates = self._legal_actions(0)
        self._mask = _action_mask(self.current_decision_state, self._candidates)
        if not self._candidates:
            self._terminated = True
            self.episode_status = self._failure_cause(0, self._physical_actions)
            raise InfeasibleReset(self.episode.sample_id, self.episode_status)
        self._ready = True
        return np.asarray(self.current_decision_state, dtype=np.float32), self._info()

    def action_masks(self):
        return self._mask.copy()

    def step(self, action):
        if not self._ready or self._terminated:
            raise RuntimeError('step requires reset and a nonterminal decision')
        if not self.action_space.contains(action) or not self._mask[int(action)]:
            raise ValueError(f'action {action} is not legal at the current V6 decision')
        index = self._index
        state = self.current_decision_state
        current_history = ((self._loads[index],) * 3 if self._history is None
                           else (*self._history[-2:], self._loads[index]))
        previous_fc = self._physical.previous_fc_kw
        fc_kw = int(round(previous_fc)) + ACTION_DELTAS[int(action)]
        next_physical, ledger, battery_kw = self.accountant.interval(self._physical, fc_kw,
            normalize_onboard_load_kw(self._loads[index]), duration_s=self._durations[index])
        if not self.accountant.plant.battery_charge_min_kw <= battery_kw <= self.accountant.plant.battery_discharge_max_kw:
            raise RuntimeError('actual battery power exceeded physical bounds')
        if not SOC_MIN <= next_physical.soc <= SOC_MAX:
            raise RuntimeError('actual SOC exceeded physical bounds')
        reward, soc_penalty, ramp_penalty, energy_cost = onboard_reward_components(ledger,
            reward_cost_mode=('zhou_economic_only' if self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC
                              else self.reward_mode),
            soc_before=state[0], soc_after=next_physical.soc,
            fc_kw=fc_kw, previous_fc_kw=previous_fc, economic_evaluation_config=self.economics)
        if self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
            total_cost = economic_cost_cny(ledger, energy_cost)
            reward = inverse_one_plus_economic_reward(total_cost)
        self._physical = next_physical
        self._fc_history = (*self._fc_history[-2:], float(fc_kw))
        self._history = current_history
        self._index += 1
        done = self._index == len(self._loads)
        successor_index = index if done else self._index
        successor_history = (current_history if done else
                             (*current_history[-2:], self._loads[successor_index]))
        next_state = self._state(successor_index, successor_history, self._fc_history)
        next_physical_actions, next_candidates = ((), ()) if done else self._legal_actions(self._index)
        excluded = (fsum((ledger.fuel_cell_degradation_cost_cny, ledger.battery_degradation_cost_cny))
                    if self.reward_mode == 'v6_energy_soc_ramp' else 0.)
        transition = DirectTransition(state, fc_kw, reward, next_state, ledger, battery_kw,
            next_physical.soc, done, next_candidates, soc_soft_penalty_cny=soc_penalty,
            original_reward=reward, terminal_reason='completed' if done else None,
            physical_feasible_actions=self._physical_actions, policy_candidate_actions=self._candidates,
            excluded_degradation_cost_cny=excluded, fc_ramp_penalty_cny=ramp_penalty,
            duration_s=self._durations[index], raw_load_kw=self._loads[index],
            relative_time_s=self._times[index], action_mode='increment13', action_index=int(action),
            action_delta_kw=int(fc_kw - previous_fc), soc_equivalent_hydrogen_cost_cny=energy_cost)
        if not done and not next_candidates:
            cause = self._failure_cause(self._index, next_physical_actions)
            if self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
                transition = replace(transition,
                    reward_cny=inverse_one_plus_economic_reward(total_cost, physical_failure=True),
                    done=True, next_feasible_actions=(), terminal_reason='failure_' + cause)
            else:
                error = ReplayExecutionError(self._index, OperatingMode.ONBOARD,
                    NoFeasibleFCActionError('no feasible FC action at successor decision'),
                    executed_transitions=(transition,), failure_kind='no_feasible_action',
                    failure_cause=cause, failed_state=next_state)
                transition = prepare_failure_replay(error, penalty=self.penalty,
                    energy_value=self._energy_value, redistribute_battery_energy=False).transitions[-1]
            self.episode_status = cause
        elif done:
            self.episode_status = 'completed'
        self.last_transition = transition
        self.current_decision_state = next_state
        self._physical_actions, self._candidates = next_physical_actions, next_candidates
        self._terminated = transition.done
        self._mask = _action_mask(next_state, transition.next_feasible_actions)
        info = {'sample_id': self.episode.sample_id, 'transition': transition,
            'next_action_mask': self.action_masks(), 'terminal_reason': transition.terminal_reason,
            'failure_penalty_equivalent_cny': transition.failure_penalty_equivalent_cny}
        if self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
            info['total_cost_cny'] = total_cost
        if transition.done:
            info.update(status=self.episode_status, executed_steps=self._index)
        return (np.asarray(next_state, dtype=np.float32), float(transition.reward_cny),
                bool(transition.done), False, info)

    def snapshot(self):
        """Small JSON-compatible state: no trajectory or action-prefix replay."""
        snapshot = {'version': 1, 'sample_id': None if self.episode is None else self.episode.sample_id,
            'reward_mode': self.reward_mode,
            'failure_penalty': None if self.penalty is None else self.penalty.amount,
            'fc_average_efficiency': self.economics.fc_average_efficiency,
            'index': self._index, 'physical': asdict(self._physical),
            'history': None if self._history is None else list(self._history),
            'fc_history': list(self._fc_history),
            'current_state': None if self.current_decision_state is None else list(self.current_decision_state),
            'terminated': self._terminated, 'episode_status': self.episode_status}
        if self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
            snapshot['reward_contract'] = reward_contract(self.reward_mode)
        return snapshot

    def restore(self, snapshot, episode):
        """Restore the physical state directly; return the current Gym decision."""
        if (snapshot.get('version') != 1 or snapshot.get('sample_id') != episode.sample_id
                or snapshot.get('reward_mode') != self.reward_mode
                or snapshot.get('failure_penalty') != (None if self.penalty is None else self.penalty.amount)
                or snapshot.get('fc_average_efficiency') != self.economics.fc_average_efficiency
                or (self.reward_mode == INVERSE_ONE_PLUS_ECONOMIC
                    and snapshot.get('reward_contract') != reward_contract(self.reward_mode))):
            raise ValueError('snapshot episode or reward configuration differs from this environment')
        self.close()
        self._load_episode(episode)
        index = snapshot['index']
        if type(index) is not int or not 0 <= index <= len(self._loads):
            raise ValueError('snapshot index is outside the voyage')
        physical = AccountState(**snapshot['physical'])
        if any(not isfinite(value) for value in asdict(physical).values()) or not SOC_MIN <= physical.soc <= SOC_MAX:
            raise ValueError('snapshot physical state must be finite and feasible')
        self._physical = physical
        self._index = index
        self._history = None if snapshot['history'] is None else tuple(snapshot['history'])
        self._fc_history = tuple(snapshot['fc_history'])
        self.current_decision_state = tuple(snapshot['current_state'])
        self._terminated = bool(snapshot['terminated'])
        self.episode_status = snapshot['episode_status']
        if not self._terminated and index == len(self._loads):
            raise ValueError('a finished snapshot must be terminated')
        self._physical_actions, self._candidates = ((), ()) if self._terminated else self._legal_actions(index)
        self._mask = _action_mask(self.current_decision_state, self._candidates)
        self.last_transition = None
        self._ready = True
        info = self._info()
        info['restored_steps'] = index
        return np.asarray(self.current_decision_state, dtype=np.float32), info

    def close(self):
        self._ready = False
        self._mask = np.zeros(13, dtype=np.bool_)
