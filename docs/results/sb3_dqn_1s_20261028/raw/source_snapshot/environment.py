"""Gymnasium steps around the unchanged V6 voyage executor.

The worker is a synchronous coroutine implemented with two one-item queues.
It never selects an action: it waits for the caller at each original policy
callback. Physical execution, causal state, reward and failure finalization
all remain in ``v4.v6_training.execute`` / ``v4.control.replay_episode``.
"""
from __future__ import annotations

from queue import Empty, Full, Queue
from threading import Event, Thread

import gymnasium as gym
import numpy as np

from v3.control import EconomicMPC
from v4.actions import action_values_kw
from v4.failure_replay import FailurePenalty
from v4.independent_economics import ZhouEvaluationConfig
from v4.v6_training import execute


ACTION_DELTAS = action_values_kw('increment13')


class InfeasibleReset(RuntimeError):
    """The original executor failed before any real action was executed."""

    def __init__(self, sample_id: str, status: str):
        self.sample_id = sample_id
        self.status = status
        super().__init__(f'{sample_id}: {status} before the first executed action')


class _Closed(BaseException):
    pass


def _action_mask(state, powers):
    previous_fc = int(round(float(state[2]) * 600.0))
    mask = np.zeros(13, dtype=np.bool_)
    for power in powers:
        mask[ACTION_DELTAS.index(int(power) - previous_fc)] = True
    return mask


class VesselEnergyEnv(gym.Env):
    """One finite canonical 1-second voyage, with V6's original semantics.

    ``reset(options={'episode': episode})`` chooses a voyage. Reset with no
    options restarts the current voyage, as required by DummyVecEnv. A runner
    may subclass reset to select its next scheduled voyage.

    Normal finite-horizon completion and genuine control failure are both
    ``terminated``. ``truncated`` remains false: a training-budget stop closes
    an unfinished prefix without fabricating another transition; corrupt or
    UNKNOWN input propagates the original executor error rather than inventing
    a bootstrap state. The frozen canonical dataset contains ONBOARD rows only.

    Complete transitions/status are available before terminal ``step`` returns.
    A scheduling wrapper must consume them before DummyVecEnv's automatic reset.
    """

    metadata = {'render_modes': []}

    def __init__(self, episode=None, *, accountant=None, economics=None, penalty=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(13)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.episode = episode
        self.accountant = accountant or EconomicMPC(nominal_cost_cny=1.0)
        self.economics = economics or ZhouEvaluationConfig.from_formal_curve()
        self.penalty = penalty or FailurePenalty.from_reference()
        self._worker = None
        self._stop = Event()
        self._commands = Queue(maxsize=1)
        self._events = Queue(maxsize=1)
        self._mask = np.zeros(13, dtype=np.bool_)
        self.current_decision_state = None
        self.last_transition = None
        self.episode_transitions = ()
        self.episode_status = None
        self.episode_actions = []
        self._terminated = False

    def _publish(self, kind, value):
        while not self._stop.is_set():
            try:
                self._events.put((kind, value), timeout=0.1)
                return
            except Full:
                continue
        raise _Closed()

    def _policy(self, state, powers):
        self._publish('decision', (state, powers))
        action = self._commands.get()
        if self._stop.is_set() or action is None:
            raise _Closed()
        return int(round(float(state[2]) * 600.0)) + ACTION_DELTAS[action]

    def _run(self):
        try:
            values, status = execute(
                self.episode, self._policy, self.accountant,
                self.economics, self.penalty,
                callback=lambda index, t: self._publish('transition', t))
            self._publish('complete', (values, status))
        except _Closed:
            pass
        except BaseException as error:
            try:
                self._publish('error', error)
            except _Closed:
                pass

    def _receive(self):
        kind, value = self._events.get()
        if kind == 'error':
            raise value
        return kind, value

    def _decision(self, value):
        state, powers = value
        self.current_decision_state = tuple(state)
        self._mask = _action_mask(state, powers)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        options = options or {}
        if 'episode' in options:
            self.episode = options['episode']
        if self.episode is None:
            raise ValueError('reset requires a canonical V6 episode')
        self._stop = Event()
        self._commands = Queue(maxsize=1)
        self._events = Queue(maxsize=1)
        self._mask = np.zeros(13, dtype=np.bool_)
        self.current_decision_state = None
        self.last_transition = None
        self.episode_transitions = ()
        self.episode_status = None
        self.episode_actions = []
        self._terminated = False
        self._worker = Thread(target=self._run, name='v6-voyage-executor', daemon=True)
        self._worker.start()
        kind, value = self._receive()
        if kind == 'complete':
            self.episode_transitions, self.episode_status = value
            self._terminated = True
            raise InfeasibleReset(self.episode.sample_id, self.episode_status)
        if kind != 'decision':
            raise RuntimeError(f'expected initial decision, received {kind}')
        self._decision(value)
        for action in options.get('action_prefix', ()):
            _, _, terminated, truncated, _ = self.step(action)
            if terminated or truncated:
                raise ValueError('a restored unfinished prefix must not end in a terminal action')
        return np.asarray(self.current_decision_state, dtype=np.float32), {
            'sample_id': self.episode.sample_id,
            'action_mask': self.action_masks(),
            'restored_prefix_steps': len(self.episode_actions),
        }

    def action_masks(self):
        return self._mask.copy()

    def step(self, action):
        if self._worker is None or self._terminated:
            raise RuntimeError('step requires reset and a nonterminal decision')
        if not self.action_space.contains(action) or not self._mask[int(action)]:
            raise ValueError(f'action {action} is not legal at the current V6 decision')
        action = int(action)
        self._commands.put(action)
        kind, transition = self._receive()
        if kind != 'transition':
            raise RuntimeError(f'expected executed transition, received {kind}')
        self.last_transition = transition
        self.episode_actions.append(action)
        kind, value = self._receive()
        if transition.done:
            if kind != 'complete':
                raise RuntimeError(f'expected voyage completion, received {kind}')
            self.episode_transitions, self.episode_status = value
            self.current_decision_state = tuple(transition.next_state)
            self._mask = np.zeros(13, dtype=np.bool_)
            self._terminated = True
        else:
            if kind != 'decision':
                raise RuntimeError(f'expected successor decision, received {kind}')
            self._decision(value)
        info = {
            'sample_id': self.episode.sample_id,
            'transition': transition,
            'next_action_mask': self.action_masks(),
            'terminal_reason': transition.terminal_reason,
            'failure_penalty_equivalent_cny': transition.failure_penalty_equivalent_cny,
        }
        if transition.done:
            info.update(status=self.episode_status, executed_steps=len(self.episode_transitions))
        return (np.asarray(transition.next_state, dtype=np.float32),
                float(transition.reward_cny), bool(transition.done), False, info)

    def snapshot(self):
        """State reconstruction recipe; contains no model or random draw."""
        return {
            'sample_id': None if self.episode is None else self.episode.sample_id,
            'episode_actions': list(self.episode_actions),
            'current_state': (None if self.current_decision_state is None
                              else list(self.current_decision_state)),
            'terminated': self._terminated,
            'episode_status': self.episode_status,
        }

    def close(self):
        self._stop.set()
        if self._worker is not None:
            try:
                self._commands.put_nowait(None)
            except Full:
                pass
            self._worker.join()
            self._worker = None
        while True:
            try:
                self._events.get_nowait()
            except Empty:
                break
        self.last_transition = None
        self.episode_transitions = ()

