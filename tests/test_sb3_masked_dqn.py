"""Fixed-sample adapter checks; these do not run a vessel training experiment."""
import threading

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import configure
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from dqn.networks.mlp_qnet import MLPQNetwork
from v6_sb3.masked_dqn import MaskedDQN, MaskedReplayBuffer, masked_td_targets


class MaskFixture(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
    action_space = spaces.Discrete(13)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(5, dtype=np.float32), {}

    def action_masks(self):
        mask = np.zeros(13, dtype=bool)
        mask[[2, 8]] = True
        return mask


def model_fixture(**kwargs):
    torch.set_num_threads(1)
    model = MaskedDQN(env=MaskFixture(), buffer_size=16, **kwargs)
    model.set_logger(configure(None, []))
    return model


def add_fixture(buffer, value=0., *, done=False, reason=None, legal_index=8):
    mask = np.zeros(13, dtype=bool)
    if not done:
        mask[legal_index] = True
    buffer.add(np.full((1, 5), value, np.float32), np.full((1, 5), value+.1, np.float32),
        np.array([8]), np.array([-10000. if done else -.2]), np.array([done]),
        [dict(next_action_mask=mask, terminal_reason=reason,
              failure_penalty_equivalent_cny=10000. if reason and reason.startswith('failure_') else 0.)])


def test_fixed_masked_target_terminal_and_no_bootstrap():
    q = torch.arange(39, dtype=torch.float32).reshape(3, 13)
    masks = torch.zeros((3, 13), dtype=torch.bool)
    masks[0, [1, 4]] = True
    rewards = torch.tensor([[-1.], [-10000.], [-.3]])
    dones = torch.tensor([[0.], [1.], [1.]])
    targets = masked_td_targets(rewards, q, dones, masks)
    torch.testing.assert_close(targets, torch.tensor([[3.], [-10000.], [-.3]]))
    assert torch.isfinite(targets).all()
    with pytest.raises(ValueError, match='nonterminal'):
        masked_td_targets(rewards, q, torch.zeros_like(dones), masks)


def test_official_replay_storage_uniform_sampling_and_fifo_labels():
    buffer = MaskedReplayBuffer(3, MaskFixture.observation_space, MaskFixture.action_space, device='cpu')
    for value in range(4):
        add_fixture(buffer, value, done=value == 3, reason='failure_soc_limited' if value == 3 else None)
    assert buffer.size() == 3 and buffer.insertions == 4 and buffer.pos == 1
    np.testing.assert_equal(buffer.observations[0, 0], np.full(5, 3.))
    assert buffer.reason_names[buffer.reasons[0]] == 'failure_soc_limited'
    assert buffer.outcomes[0] == 1 and buffer.failure_penalties[0] == 10000.
    sample = buffer.sample(20)
    assert sample.observations.shape == (20, 5)
    assert sample.next_action_masks.shape == (20, 13)
    assert len(set(buffer.last_sample_indices.tolist())) < 20
    assert MaskedReplayBuffer.sample is ReplayBuffer.sample
    np.testing.assert_equal(sample.observations.numpy(), buffer.observations[buffer.last_sample_indices, 0])
    assert (~sample.dones.bool().flatten() | ~sample.next_action_masks.any(1)).all()


def test_network_initialization_matches_v6_and_target_is_frozen():
    torch.manual_seed(42)
    v6 = MLPQNetwork(5, 13, (256, 256))
    model = model_fixture()
    for old, online, target in zip(v6.parameters(), model.q_net.parameters(), model.q_net_target.parameters()):
        assert torch.equal(old, online) and torch.equal(online, target)
        assert online.data_ptr() != target.data_ptr()
    assert sum(p.numel() for p in model.q_net.parameters()) == 70669
    assert all(not p.requires_grad for p in model.q_net_target.parameters())
    assert set(map(id, model.policy.optimizer.param_groups[0]['params'])) == set(map(id, model.q_net.parameters()))
    assert MaskedDQN.learn is DQN.learn
    assert MaskedDQN.collect_rollouts is OffPolicyAlgorithm.collect_rollouts


def test_greedy_exploration_and_warmup_all_use_the_real_mask():
    model = model_fixture()
    with torch.no_grad():
        for p in model.q_net.parameters(): p.zero_()
        model.q_net.q_net[-1].bias.copy_(torch.arange(13.))
    obs = np.zeros(5, np.float32)
    mask = MaskFixture().action_masks()
    action, _ = model.masked_predict(obs, mask, deterministic=True)
    assert int(action) == 8  # raw argmax12 is illegal
    model.exploration_rate = 1.
    assert {int(model.masked_predict(obs, mask)[0]) for _ in range(30)} <= {2, 8}
    model._last_obs = obs[None]
    assert {int(model._sample_action(50000)[0][0]) for _ in range(30)} <= {2, 8}
    with pytest.raises(ValueError, match='nonempty'):
        model.masked_predict(obs, np.zeros(13, bool), deterministic=True)


def test_actual_adam_updates_gather_and_only_optimizer_clock_copies_target():
    model = model_fixture()
    add_fixture(model.replay_buffer, .2)
    add_fixture(model.replay_buffer, .5, done=True, reason='failure_soc_limited')
    original = [p.detach().clone() for p in model.q_net.parameters()]
    target_before = [p.detach().clone() for p in model.q_net_target.parameters()]
    observed = []

    def inspect_batch(agent, batch, q, target):
        with torch.no_grad():
            expected_q = agent.q_net(batch.observations).gather(1, batch.actions.long())
        torch.testing.assert_close(q, expected_q)
        assert torch.equal(target[batch.dones.bool()], batch.rewards[batch.dones.bool()])

    model.train_step_observer = inspect_batch
    model.after_update_observer = lambda m: observed.append((m._n_updates, m.target_sync_calls))
    model._n_updates = 998
    for _ in range(3): model._on_step()
    assert model.target_sync_calls == 1
    model.train(1, batch_size=2)
    assert observed == [(999, 1)]
    assert any(not torch.equal(a, b) for a, b in zip(original, model.q_net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(target_before, model.q_net_target.parameters()))
    model.train(1, batch_size=2)
    assert observed[-1] == (1000, 2)
    assert all(torch.equal(a, b) for a, b in zip(model.q_net.parameters(), model.q_net_target.parameters()))
    assert all(p.grad is None for p in model.q_net_target.parameters())
    assert len(model.policy.optimizer.state) == 6
    assert model.training_loss_count == 2 and model.target_soft_update_calls == 0
    assert sum(g.count for g in model.td_diagnostics.groups) == 4


def test_model_and_replay_roundtrip_exclude_live_callbacks(tmp_path):
    model = model_fixture()
    add_fixture(model.replay_buffer, .4, done=True, reason='failure_soc_limited')
    model.finalize_streamed_failure_prefix(range(1))
    model.train(1, batch_size=1)
    lock = threading.Lock()
    model.after_update_observer = lambda _: lock.acquire(False)
    model.train_step_observer = lambda *args: lock.acquire(False)
    model.target_sync_calls = 7
    model.save(tmp_path/'model')
    model.save_replay_buffer(tmp_path/'replay.pkl')
    restored = MaskedDQN.load(tmp_path/'model', env=MaskFixture(), device='cpu')
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    assert restored.after_update_observer is None and restored.train_step_observer is None
    assert restored.target_sync_calls == 7 and restored.replay.insertions == 1
    assert restored._n_updates == 1
    assert len(restored.optimizer.state) == 6
    assert restored.replay.outcomes[0] == 1 and restored.replay.done[0, 0] == 1
    for left, right in zip(model.q_net.parameters(), restored.q_net.parameters()):
        assert torch.equal(left, right)
    for left, right in zip(model.optimizer.state.values(), restored.optimizer.state.values()):
        assert torch.equal(left['step'], right['step'])
        assert torch.equal(left['exp_avg'], right['exp_avg'])
        assert torch.equal(left['exp_avg_sq'], right['exp_avg_sq'])
