"""Small synthetic contracts for the frozen formal training path."""
import io
import json

import pytest
import torch

from test_v4_training import FakeDataset
from v4 import monitored_training
from v4.dqn import DirectPowerDDQN


def test_economic_soft_target_updates_once_per_adam_step_only():
    agent = DirectPowerDDQN(seed=42, n_step=8, target_tau=.001)
    state = (.6, .2, 0., 0., 0., 0., 0., 1.)
    agent.remember(state, 100, -5., state, done=True, next_feasible_actions=())
    old_target = {key: value.clone() for key, value in agent.target.state_dict().items()}
    assert agent.learn(batch_size=1) is not None
    assert agent.target_sync_calls == 1
    assert agent.target_soft_update_calls == agent.economic_optimizer_updates == 1
    for key, value in agent.target.state_dict().items():
        assert torch.allclose(value, torch.lerp(old_target[key], agent.online.state_dict()[key], .001))
    agent.learn_outcome(batch_size=1)
    assert agent.target_soft_update_calls == 1


def test_training_state_restores_live_fifo_quota_pools_and_rng():
    settings = dict(seed=7, n_step=8, replay_capacity=4, failure_terminal_quota=2)
    agent = DirectPowerDDQN(**settings)
    for index in range(6):
        state = (.6, index/10., 0., 0., 0., 0., 0., 1.)
        terminal = index in (1,4)
        agent.remember(state, 100, -float(index), state, done=True,
                       next_feasible_actions=(),
                       experience_outcome='failure' if terminal else 'success',
                       terminal_reason='failure_soc_limited' if terminal else 'completed',
                       failure_penalty_equivalent_cny=10. if terminal else 0.)
    assert agent.learn(batch_size=3) is not None
    serialized = io.BytesIO()
    torch.save(agent.training_state(), serialized)
    serialized.seek(0)
    restored = DirectPowerDDQN(**settings)
    restored.load_training_state(torch.load(serialized, weights_only=False))
    assert list(restored.replay) == list(agent.replay)
    assert restored.failure_terminal_replay_count == agent.failure_terminal_replay_count == 1
    assert restored._sample_replay_batch(3) == agent._sample_replay_batch(3)
    assert restored.learn(batch_size=3) == pytest.approx(agent.learn(batch_size=3))
    assert all(torch.equal(value, agent.online.state_dict()[key])
               for key, value in restored.online.state_dict().items())
    assert restored.target_soft_update_calls == agent.target_soft_update_calls == 2


def test_complete_round_resume_restores_replay_rng_and_epsilon(tmp_path, monkeypatch):
    settings = dict(rounds=2, beta_soc=0., cadence='replay32', target_mode='soft',
                    n_step=8, reward_scale=.001, failure_terminal_quota=2,
                    batch_size=1, capture_trajectories=False)
    uninterrupted, reference = monitored_training.run_monitored_training(
        FakeDataset(), output_dir=tmp_path/'reference', **settings)
    original = monitored_training.replay_episode
    calls = 0
    def interrupt_second_round(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(monitored_training, 'replay_episode', interrupt_second_round)
    output = tmp_path/'interrupted'
    with pytest.raises(RuntimeError, match='synthetic interruption'):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=output, **settings)
    checkpoint = output/'training_state_latest.pt'
    assert checkpoint.exists()
    assert len(json.loads((output/'report.json').read_text(encoding='utf-8'))['rounds']) == 1
    monkeypatch.setattr(monitored_training, 'replay_episode', original)
    resumed, report = monitored_training.run_monitored_training(
        FakeDataset(), output_dir=output, resume_from=checkpoint, **settings)
    assert report['rounds'] == reference['rounds']
    assert [row['epsilon'] for row in report['rounds']] == pytest.approx((1., 1.-.95/69))
    assert list(resumed.replay) == list(uninterrupted.replay)
    assert all(torch.equal(value, uninterrupted.online.state_dict()[key])
               for key, value in resumed.online.state_dict().items())
    assert report['test_payloads_opened'] == 0
