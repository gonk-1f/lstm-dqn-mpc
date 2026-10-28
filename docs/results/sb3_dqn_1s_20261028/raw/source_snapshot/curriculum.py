"""The original warmup order and eight Train traversals around the Gym adapter."""
from __future__ import annotations

from collections import deque
import random

from v4.v6_training import epsilon_at
from .environment import VesselEnergyEnv


class TrainCurriculum(VesselEnergyEnv):
    def __init__(self, train, **kwargs):
        super().__init__(**kwargs)
        self.train = tuple(train)
        self.by_id = {e.sample_id: e for e in train}
        self.phase = 'warmup'
        self.pass_index = 0
        self.episode_index = -1
        self.order = []
        self.warmup_random = random.Random(42)
        self.pending_completed = deque()
        self.emitted_steps = 0
        self.replay_start = 0
        self.schedule_finished = False
        self.reconstructing = False

    @property
    def epsilon(self):
        return 1.0 if self.phase == 'warmup' else epsilon_at(self.pass_index, self.episode_index)

    def start_formal(self):
        # Discard only the physical warmup prefix. Its last stored experience
        # remains a real nonterminal transition, as in the original V6 run.
        self.close()
        self.phase = 'training'
        self.pass_index = 0
        self.episode_index = -1
        self.order = []
        self.pending_completed.clear()

    def reset(self, *, seed=None, options=None):
        if options and 'episode' in options:
            return super().reset(seed=seed, options=options)
        if self.schedule_finished:
            # DummyVecEnv must reset after the last terminal step. This reset
            # executes no action; after_update stops before another rollout.
            return super().reset(seed=seed, options={'episode': self.episode})
        self.episode_index += 1
        if self.episode_index >= len(self.order):
            self.episode_index = 0
            self.order = list(self.train)
            if self.phase == 'warmup':
                self.warmup_random.shuffle(self.order)
            else:
                self.pass_index += 1
                random.Random(42 + self.pass_index).shuffle(self.order)
        self.replay_start = self.emitted_steps
        return super().reset(seed=seed, options={'episode': self.order[self.episode_index]})

    def step(self, action):
        result = super().step(action)
        if self.reconstructing:
            return result
        self.emitted_steps += 1
        if result[2] or result[3]:
            self.pending_completed.append(dict(episode=self.episode,
                values=self.episode_transitions, status=self.episode_status,
                phase=self.phase, pass_index=self.pass_index,
                episode_index=self.episode_index, epsilon=self.epsilon,
                replay_start=self.replay_start))
            if self.phase == 'training' and self.episode_index == len(self.train) - 1 and self.pass_index == 8:
                self.schedule_finished = True
        return result

    def curriculum_state(self):
        return dict(phase=self.phase, pass_index=self.pass_index,
            episode_index=self.episode_index, order=[e.sample_id for e in self.order],
            warmup_random=self.warmup_random.getstate(), emitted_steps=self.emitted_steps,
            replay_start=self.replay_start, schedule_finished=self.schedule_finished,
            physical=self.snapshot(), gym_random=self.np_random.bit_generator.state,
            action_space_random=self.action_space.np_random.bit_generator.state)

    def restore_curriculum(self, saved):
        self.phase = saved['phase']
        self.pass_index = saved['pass_index']
        self.episode_index = saved['episode_index']
        self.order = [self.by_id[name] for name in saved['order']]
        self.warmup_random.setstate(saved['warmup_random'])
        self.emitted_steps = saved['emitted_steps']
        self.replay_start = saved['replay_start']
        self.schedule_finished = saved['schedule_finished']
        physical = saved['physical']
        self.reconstructing = True
        try:
            observation, info = super().reset(options={
                'episode': self.by_id[physical['sample_id']],
                'action_prefix': physical['episode_actions']})
        finally:
            self.reconstructing = False
        if tuple(physical['current_state']) != self.current_decision_state:
            raise ValueError('restored saved action prefix produced a different physical state')
        self.np_random.bit_generator.state = saved['gym_random']
        self.action_space.np_random.bit_generator.state = saved['action_space_random']
        self.pending_completed.clear()
        return observation, info
