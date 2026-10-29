"""Zoo-facing finite voyages over the unchanged V6 physical executor.

Only masked legal actions can be stepped. Active unmasked Gym/SB3 checkers
sample illegal actions and are not suitable for this environment contract.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.registration import register, registry

from v4.v6_dataset import V6TrainingDataset
from v6_sb3.direct_environment import DirectVesselEnv
from v6_sb3.environment import InfeasibleReset
from v6_sb3.rewards import INVERSE_ONE_PLUS_ECONOMIC


ENV_ID = 'VesselEnergy-v0'
ENTRY_POINT = 'experiments.sb3_manual.zoo_vessel.env:VesselEnergyEnv'


class VesselEnvironmentError(RuntimeError):
    """Invalid program/data input; never an ordinary physical terminal reward."""


class IllegalVesselAction(VesselEnvironmentError):
    """An action outside the current physical mask was not executed."""


class VesselEnergyEnv(gym.Env):
    """One immutable split, one complete voyage per reset, no time-limit wrapper.

    Train draws uniformly with replacement from Gym's per-instance seeded RNG.
    Validation cycles through its eight sorted IDs. Call ``begin_evaluation``
    before every evaluation, followed by the usual VecEnv/evaluator reset:
    DummyVecEnv auto-resets after the eighth terminal and preselects a ninth
    voyage, which must not advance the next evaluation's starting position.
    """

    metadata = {'render_modes': []}

    def __init__(self, *, data_root, split='train'):
        super().__init__()
        if split not in ('train','validation'):
            raise ValueError('split must be train or validation; Test payloads are never loaded')
        self._split = split
        self._data_root = Path(data_root).resolve()
        try:
            dataset = V6TrainingDataset.open(self._data_root)
        except (ValueError,TypeError,FloatingPointError) as error:
            raise VesselEnvironmentError(f'canonical metadata data/program error: {error}') from error
        if dataset.split_sizes != (35,8):
            raise VesselEnvironmentError('requires the frozen 35 Train / 8 Validation metadata')
        train_ids = tuple(sorted(dataset.split_episode_ids('train')))
        validation_ids = tuple(sorted(dataset.split_episode_ids('validation')))
        if (len(set(train_ids)) != 35 or len(set(validation_ids)) != 8
                or set(train_ids)&set(validation_ids)):
            raise VesselEnvironmentError('Train/Validation identities must be distinct and disjoint')
        try:
            episodes = dataset.load_train() if split=='train' else dataset.load_validation()
        except (ValueError,TypeError,FloatingPointError) as error:
            raise VesselEnvironmentError(f'{split} canonical payload data/program error: {error}') from error
        self._episodes = tuple(sorted(episodes,key=lambda episode:episode.sample_id))
        expected_ids = train_ids if split=='train' else validation_ids
        if (tuple(episode.sample_id for episode in self._episodes) != expected_ids
                or any(episode.split != split for episode in self._episodes)):
            raise VesselEnvironmentError('loaded episodes do not match the immutable split identity')
        self._episode_ids = expected_ids
        self._evaluation_step_budget = sum(len(episode.load_kw) for episode in self._episodes)
        self._validation_cursor = 0
        self._direct = DirectVesselEnv(reward_mode=INVERSE_ONE_PLUS_ECONOMIC)
        self.action_space = self._direct.action_space
        self.observation_space = self._direct.observation_space
        self._last_reset_failure = None
        self._reset_totals()

    @property
    def split(self):
        return self._split

    @property
    def data_root(self):
        return self._data_root

    @property
    def reward_mode(self):
        return INVERSE_ONE_PLUS_ECONOMIC

    @property
    def episode_ids(self):
        """Sorted immutable identities; readable through VecEnv.get_attr."""
        return self._episode_ids

    @property
    def sample_ids(self):
        return self.episode_ids

    @property
    def evaluation_step_budget(self):
        """Natural total length of this pool; Validation's eight-voyage bound."""
        return self._evaluation_step_budget

    @property
    def last_transition(self):
        return self._direct.last_transition

    @property
    def last_reset_failure(self):
        return None if self._last_reset_failure is None else dict(self._last_reset_failure)

    def _reset_totals(self):
        self._executed_steps = 0
        self._economic_cost = 0.
        self._return_sum = 0.
        self._fc_start_count = 0
        self._previous_fc_kw = 0.

    def begin_evaluation(self):
        if self.split != 'validation':
            raise VesselEnvironmentError('begin_evaluation requires an immutable validation environment')
        self._validation_cursor = 0
        return self.episode_ids

    def reset(self, *, seed=None, options=None):
        if options:
            raise VesselEnvironmentError('reset options cannot override the immutable split or selected voyage')
        super().reset(seed=seed)
        if self.split=='train':
            episode = self._episodes[int(self.np_random.integers(len(self._episodes)))]
        else:
            if seed is not None:
                self._validation_cursor = 0
            episode = self._episodes[self._validation_cursor]
            self._validation_cursor = (self._validation_cursor+1)%len(self._episodes)
        self._reset_totals()
        self._last_reset_failure = None
        try:
            observation,info = self._direct.reset(seed=seed,options={'episode':episode})
        except InfeasibleReset as error:
            self._last_reset_failure = dict(sample_id=episode.sample_id,split=self.split,
                terminal_reason=error.status,executed_steps=0,episode_env_steps=0,
                transition_created=False)
            raise
        except (ValueError,TypeError,FloatingPointError) as error:
            raise VesselEnvironmentError(f'{episode.sample_id}: reset data/program error: {error}') from error
        return observation,{**info,'split':self.split}

    def action_masks(self):
        return self._direct.action_masks()

    def step(self, action):
        # Do not clip, remap, sample a replacement or invent a failure step.
        if not self.action_space.contains(action) or not self.action_masks()[int(action)]:
            raise IllegalVesselAction(f'action {action!r} is not legal at the current vessel decision')
        try:
            observation,reward,terminated,truncated,info = self._direct.step(action)
        except (ValueError,TypeError,FloatingPointError) as error:
            raise VesselEnvironmentError(f'vessel step data/program error: {error}') from error
        transition = self.last_transition
        step_cost = float(info['total_cost_cny'])
        self._economic_cost += step_cost
        self._return_sum += float(reward)
        self._executed_steps += 1
        fc_kw = float(transition.action_kw)
        self._fc_start_count += int(self._previous_fc_kw<=0. and fc_kw>0.)
        self._previous_fc_kw = fc_kw
        if not all(isfinite(value) for value in (step_cost,self._economic_cost,self._return_sum)):
            raise VesselEnvironmentError('nonfinite vessel reward/economic summary')
        info = {**info,'split':self.split,'vessel':dict(
            sample_id=info['sample_id'],split=self.split,soc=float(transition.actual_soc),
            fc_power_kw=fc_kw,fc_start_count=self._fc_start_count,
            economic_cost_cny=self._economic_cost,step_cost_cny=step_cost,
            executed_steps=self._executed_steps)}
        if terminated:
            status = self._direct.episode_status
            completed = status=='completed'
            info['vessel_episode'] = dict(sample_id=info['sample_id'],split=self.split,
                completed=completed,terminal_reason=status,
                executed_steps=self._executed_steps,episode_env_steps=self._executed_steps,
                duration_coverage=self._executed_steps/self._direct.episode.duration_s,
                soc_final=float(transition.actual_soc),return_sum=self._return_sum,
                economic_cost_cny=self._economic_cost,
                executed_prefix_economic_cost_cny=self._economic_cost,
                complete_economic_cost_cny=self._economic_cost if completed else None)
        return observation,float(reward),bool(terminated),bool(truncated),info

    def close(self):
        self._direct.close()


def register_envs():
    """Register idempotently; retain Gym's passive checks without a TimeLimit."""
    if ENV_ID in registry:
        if registry[ENV_ID].entry_point != ENTRY_POINT:
            raise VesselEnvironmentError(f'{ENV_ID} is already registered to a different environment')
        return
    register(id=ENV_ID,entry_point=ENTRY_POINT)
