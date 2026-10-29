"""Zoo adapter contracts using synthetic fixed split identities only."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest


def module():
    path = Path(__file__).parents[1]/'experiments/sb3_manual/zoo_vessel/env.py'
    assert path.is_file(), 'the isolated Zoo vessel environment must exist'
    spec = importlib.util.spec_from_file_location('zoo_vessel_env_under_test',path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def fake_dataset(monkeypatch):
    loaded = module()
    def episode(sample_id,split,loads):
        return SimpleNamespace(sample_id=sample_id,split=split,task_mode='independent_voyage',
            operating_mode=('onboard',)*len(loads),load_kw=tuple(loads),
            time_s=tuple(float(i) for i in range(len(loads))),dt_s=(1.,)*len(loads),
            duration_s=float(len(loads)))
    class Dataset:
        calls = []
        loads = (0.,100.,70.)
        split_sizes = (35,8)
        split_counts = {'train':35,'validation':8,'test':4}
        @classmethod
        def open(cls,path):
            cls.calls.append(('open',str(path)))
            return cls()
        def split_episode_ids(self,split):
            assert split in ('train','validation'), 'Test identities are not needed'
            n = 35 if split=='train' else 8
            return tuple(f'{split}_{i:03d}' for i in range(n))
        def load_train(self):
            self.calls.append(('payload','train'))
            return tuple(episode(sid,'train',self.loads) for sid in self.split_episode_ids('train'))
        def load_validation(self):
            self.calls.append(('payload','validation'))
            return tuple(episode(sid,'validation',self.loads) for sid in self.split_episode_ids('validation'))
    monkeypatch.setattr(loaded.V6TrainingDataset,'open',Dataset.open)
    return loaded,Dataset


def test_train_constructor_reads_only_train_and_split_is_fixed(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path,split='train')
    assert [item for item in data.calls if item[0]=='payload']==[('payload','train')]
    assert env.split=='train' and len(env.sample_ids)==35
    assert isinstance(env.sample_ids,tuple)
    assert env.episode_ids==env.sample_ids
    assert env.evaluation_step_budget==35*3
    with pytest.raises(AttributeError):
        env.split='validation'
    assert env.reward_mode=='inverse_one_plus_economic'
    with pytest.raises(AttributeError):
        env.reward_mode='zhou_economic_only'
    env.close()


def test_test_split_rejected_before_opening_any_dataset(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    with pytest.raises((ValueError,RuntimeError),match='train|validation|split'):
        loaded.VesselEnergyEnv(data_root=tmp_path,split='test')
    assert data.calls==[]


def test_train_draws_match_seeded_gym_uniform_rng_with_replacement(fake_dataset,tmp_path):
    loaded,_ = fake_dataset
    a = loaded.VesselEnergyEnv(data_root=tmp_path)
    b = loaded.VesselEnergyEnv(data_root=tmp_path)
    generator = np.random.default_rng(91)
    expected = [a.sample_ids[int(generator.integers(35))] for _ in range(80)]
    for index,sid in enumerate(expected):
        for env in (a,b):
            _,info = env.reset(seed=91 if index==0 else None)
            assert info['sample_id']==sid
    assert len(set(expected))<len(expected)
    a.close();b.close()


def test_validation_cycle_is_sorted_and_explicitly_restarts_after_autoreset(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path,split='validation')
    assert [item for item in data.calls if item[0]=='payload']==[('payload','validation')]
    assert env.evaluation_step_budget==8*3
    with pytest.raises(AttributeError):
        env.episode_ids=('outside_split',)
    env.begin_evaluation()
    observed = [env.reset()[1]['sample_id'] for _ in range(8)]
    assert observed==list(env.sample_ids)
    assert env.reset()[1]['sample_id']==env.sample_ids[0]  # DummyVecEnv's ninth prefetch.
    env.begin_evaluation()
    assert [env.reset()[1]['sample_id'] for _ in range(8)]==list(env.sample_ids)
    env.close()


def test_training_cannot_be_switched_into_an_evaluation_cycle(fake_dataset,tmp_path):
    loaded,_ = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    with pytest.raises(RuntimeError,match='validation'):
        env.begin_evaluation()


def test_legally_masked_steps_keep_gym_shape_and_terminal_economics(fake_dataset,tmp_path):
    loaded,_ = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    observation,info = env.reset(seed=42)
    assert observation.shape==(5,) and observation.dtype==np.float32
    assert env.observation_space.contains(observation) and env.action_space.n==13
    assert env.action_masks().shape==(13,) and env.action_masks().dtype==np.bool_
    assert info['split']=='train'
    total_cost,total_return = 0.,0.
    for index in range(3):
        assert env.action_masks()[6]
        observation,reward,terminated,truncated,info = env.step(6)
        assert observation.shape==(5,) and observation.dtype==np.float32
        assert not truncated and terminated==(index==2)
        assert reward==pytest.approx(1/(1+info['total_cost_cny']))
        assert info['failure_penalty_equivalent_cny']==0
        assert info['vessel']['executed_steps']==index+1
        assert info['vessel']['fc_start_count']==0
        total_cost+=info['total_cost_cny'];total_return+=reward
        assert info['vessel']['economic_cost_cny']==pytest.approx(total_cost)
        if not terminated:
            assert 'vessel_episode' not in info
    summary = info['vessel_episode']
    assert summary['completed'] and summary['terminal_reason']=='completed'
    assert summary['executed_steps']==summary['episode_env_steps']==3
    assert summary['duration_coverage']==1
    assert summary['return_sum']==pytest.approx(total_return)
    assert summary['economic_cost_cny']==summary['executed_prefix_economic_cost_cny']==pytest.approx(total_cost)
    assert summary['complete_economic_cost_cny']==pytest.approx(total_cost)
    assert not info['next_action_mask'].any()
    env.close()


def test_start_count_counts_off_to_positive_events_once(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    data.loads=(0.,0.,0.,0.)
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    env.reset(seed=12)
    for action,starts in zip((7,6,5,7),(1,1,1,2)):
        assert env.step(action)[4]['vessel']['fc_start_count']==starts
    env.close()


def test_illegal_actions_raise_without_advancing_or_fabricating_steps(fake_dataset,tmp_path):
    loaded,_ = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    observation,_ = env.reset(seed=0)
    with pytest.raises(RuntimeError,match='legal|action'):
        env.step(0)
    assert env.last_transition is None
    successor,reward,done,truncated,info = env.step(6)
    assert info['vessel']['executed_steps']==1
    assert observation[0]==successor[0] and reward==1. and not done and not truncated


def test_real_failure_replaces_last_reward_without_a_fake_transition(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    data.loads=(100.,1900.)
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    env.reset(seed=0)
    _,reward,terminated,truncated,info = env.step(6)
    assert reward==-1. and terminated and not truncated
    assert info['terminal_reason']=='failure_structural_power'
    assert info['vessel_episode']['terminal_reason']=='structural_power'
    assert info['vessel_episode']['executed_steps']==1
    assert not info['vessel_episode']['completed']
    assert info['vessel_episode']['complete_economic_cost_cny'] is None
    assert info['vessel']['step_cost_cny']>0
    assert info['failure_penalty_equivalent_cny']==0 and not info['next_action_mask'].any()


def test_infeasible_reset_is_an_event_with_no_fake_step(fake_dataset,tmp_path):
    loaded,data = fake_dataset
    data.loads=(1900.,)
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    with pytest.raises(loaded.InfeasibleReset):
        env.reset(seed=0)
    assert env.last_transition is None
    assert env.last_reset_failure['executed_steps']==0
    assert env.last_reset_failure['terminal_reason']=='structural_power'
    assert env.last_reset_failure['split']=='train'
    with pytest.raises(RuntimeError):
        env.step(6)


def test_reset_cannot_override_the_frozen_episode_or_split(fake_dataset,tmp_path):
    loaded,_ = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    with pytest.raises(RuntimeError,match='options|override'):
        env.reset(options={'split':'test'})
    with pytest.raises(RuntimeError,match='options|override'):
        env.reset(options={'episode':object()})


def test_nonphysical_errors_propagate_as_errors_not_failure_rewards(fake_dataset,tmp_path,monkeypatch):
    loaded,_ = fake_dataset
    env = loaded.VesselEnergyEnv(data_root=tmp_path)
    env.reset(seed=0)
    def broken_step(action):
        raise ValueError('broken economic data')
    monkeypatch.setattr(env._direct,'step',broken_step)
    with pytest.raises(RuntimeError,match='broken economic data') as error:
        env.step(6)
    assert isinstance(error.value.__cause__,ValueError)
    assert env.last_transition is None


@pytest.mark.parametrize('phase',['metadata','payload'])
def test_loader_data_value_errors_are_fatal_runtime_errors(fake_dataset,tmp_path,monkeypatch,phase):
    loaded,data = fake_dataset
    def broken(*args,**kwargs):
        raise ValueError('corrupt canonical data')
    if phase=='metadata':
        monkeypatch.setattr(loaded.V6TrainingDataset,'open',broken)
    else:
        monkeypatch.setattr(data,'load_train',broken)
    with pytest.raises(RuntimeError,match='corrupt canonical data') as error:
        loaded.VesselEnergyEnv(data_root=tmp_path)
    assert isinstance(error.value.__cause__,ValueError)


def test_registration_is_idempotent_and_has_no_time_limit():
    loaded = module()
    loaded.register_envs();loaded.register_envs()
    spec = gym.spec('VesselEnergy-v0')
    assert spec.entry_point=='experiments.sb3_manual.zoo_vessel.env:VesselEnergyEnv'
    assert spec.max_episode_steps is None
