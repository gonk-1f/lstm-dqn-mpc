"""Native SB3 logging and an independent full Validation cycle."""
from pathlib import Path
import json
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class TinyVoyages(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = gym.spaces.Discrete(13)

    def __init__(self, split='train', fail=False):
        self.split, self.fail = split, fail
        self.episode_ids = tuple(f'{split}_{i}' for i in range(8 if split=='validation' else 35))
        self.evaluation_step_budget = len(self.episode_ids)*2
        self.cursor = 0

    def begin_evaluation(self):
        assert self.split == 'validation'
        self.cursor = 0

    def action_masks(self):
        mask = np.zeros(13, dtype=bool)
        mask[5 if self.split=='validation' else 6] = True
        return mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.name = self.episode_ids[self.cursor % len(self.episode_ids)]
        self.cursor += 1
        self.steps = 0
        return np.array([.6, 0., 0., 0., 0.], np.float32), {}

    def step(self, action):
        assert self.action_masks()[int(action)], 'evaluation used the training mask'
        self.steps += 1
        done = self.steps == 2
        reward = -1. if done and self.fail else 1.
        info = dict(next_action_mask=np.zeros(13,dtype=bool) if done else self.action_masks(),
            failure_penalty_equivalent_cny=0.,
            terminal_reason=('failure_increment_action_limited' if self.fail else 'completed') if done else None,
            vessel=dict(sample_id=self.name, split=self.split, soc=.6, fc_power_kw=0.,
                        fc_start_count=0, economic_cost_cny=0., step_cost_cny=0., executed_steps=self.steps))
        if done:
            info['vessel_episode'] = dict(sample_id=self.name, split=self.split, completed=not self.fail,
                terminal_reason='increment_action_limited' if self.fail else 'completed',
                executed_steps=2, episode_env_steps=2, duration_coverage=1. if not self.fail else .5,
                soc_final=.6, return_sum=0. if self.fail else 2., economic_cost_cny=0.,
                executed_prefix_economic_cost_cny=0., complete_economic_cost_cny=None if self.fail else 0.)
        return np.array([.6,self.steps/10,0.,0.,0.],np.float32),reward,done,False,info


def make_model(tmp_path):
    from experiments.sb3_manual.zoo_vessel.algorithm import ZooMaskedDQN
    env=DummyVecEnv([lambda:Monitor(TinyVoyages())])
    model=ZooMaskedDQN(env=env,learning_starts=32,buffer_size=4096,batch_size=8,
        train_freq=32,gradient_steps=1,target_update_interval=64,seed=7,
        policy_kwargs={'net_arch':[8,8]},tensorboard_log=str(tmp_path/'tb'),verbose=1)
    return model,env


def test_full_validation_uses_its_own_mask_and_preserves_training_state(tmp_path):
    from experiments.sb3_manual.zoo_vessel.callbacks import evaluate_all_validation
    model,env=make_model(tmp_path)
    evaluation=DummyVecEnv([lambda:Monitor(TinyVoyages('validation'))])
    before=(model.num_timesteps,model._n_updates,model.replay_buffer.insertions,
            pickle.dumps(np.random.get_state()),torch.get_rng_state().clone())
    try:
        first=evaluate_all_validation(model,evaluation)
        second=evaluate_all_validation(model,evaluation)
        assert first == second
        assert first['episodes']==first['completed']==8
        assert first['fixed_complete_set_cost_cny']==0.
        assert len({row['sample_id'] for row in first['samples']})==8
        assert before[:3] == (model.num_timesteps,model._n_updates,model.replay_buffer.insertions)
        assert before[3] == pickle.dumps(np.random.get_state())
        assert torch.equal(before[4],torch.get_rng_state())
    finally:
        env.close();evaluation.close()


def test_native_logger_stdout_csv_and_tensorboard_without_fake_incomplete_cost(tmp_path,capsys):
    from experiments.sb3_manual.zoo_vessel.callbacks import VesselEvalCallback
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import pandas as pd
    model,env=make_model(tmp_path)
    evaluation=DummyVecEnv([lambda:Monitor(TinyVoyages('validation',fail=True))])
    callback=VesselEvalCallback(evaluation,eval_freq=1024,n_eval_episodes=8,
        log_path=str(tmp_path/'validation'),best_model_save_path=str(tmp_path/'best'),log_every=1000)
    try:
        model.learn(total_timesteps=1024,callback=callback,log_interval=None)
        output=capsys.readouterr().out
        assert '| time/' in output and '| vessel/' in output and '| train/' in output
        assert 'td_mae' in output and 'completion_rate' in output
        csvs=list((tmp_path/'tb').rglob('progress.csv'))
        assert len(csvs)==1
        frame=pd.read_csv(csvs[0])
        for name in ('time/total_timesteps','rollout/exploration_rate','train/n_updates',
                     'train/loss','train/td_mae','replay/buffer_size','vessel/soc',
                     'validation/completion_rate','validation/mean_duration_coverage'):
            assert name in frame.columns,name
        assert frame['validation/completion_rate'].dropna().iloc[-1]==0.
        assert frame['validation/economic_cost_cny'].isna().all()
        assert model._n_updates==31 and model.replay_buffer.insertions==1024
        events=list((tmp_path/'tb').rglob('events.out.tfevents.*'))
        assert events
        accumulator=EventAccumulator(str(events[0].parent));accumulator.Reload()
        for name in ('train/loss','train/td_mae','train/n_updates','vessel/soc','validation/completion_rate'):
            assert name in accumulator.Tags()['scalars'],name
        assert 'validation/economic_cost_cny' not in accumulator.Tags()['scalars']
    finally:
        model.logger.close();env.close();evaluation.close()


def test_logging_and_validation_do_not_change_actions_sampling_adam_or_target(tmp_path):
    from experiments.sb3_manual.zoo_vessel.callbacks import VesselEvalCallback
    baseline,first_env=make_model(tmp_path/'plain')
    baseline.learn(total_timesteps=128,log_interval=None)
    first_rng=pickle.dumps(np.random.get_state())
    observed,second_env=make_model(tmp_path/'observed')
    evaluation=DummyVecEnv([lambda:Monitor(TinyVoyages('validation'))])
    try:
        observed.learn(total_timesteps=128,log_interval=None,
            callback=VesselEvalCallback(evaluation,eval_freq=64,log_every=32))
        assert first_rng == pickle.dumps(np.random.get_state())
        assert baseline.num_timesteps==observed.num_timesteps==128
        assert baseline._n_updates==observed._n_updates==3
        assert baseline.target_sync_history==observed.target_sync_history
        assert baseline.exploration_rate==observed.exploration_rate
        for key,value in baseline.policy.state_dict().items():
            assert torch.equal(value,observed.policy.state_dict()[key]),key
        for name in ('actions','rewards','next_action_masks'):
            np.testing.assert_array_equal(getattr(baseline.replay_buffer,name)[:128],
                                          getattr(observed.replay_buffer,name)[:128])
    finally:
        baseline.logger.close();observed.logger.close()
        first_env.close();second_env.close();evaluation.close()
