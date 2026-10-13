"""Small synthetic contracts; no dataset, rollout or live checkpoint is read."""
import copy

import pytest
import torch
from torch.nn import functional as F

from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.failure_replay import FAILURE_TERMINALS


def insert(agent, count=70, failures=4):
    for i in range(count + failures):
        state = (.6, i / 1000., 0., 0., .5, 0., 0.)
        failed = i >= count
        agent.remember(state, 100, -10000. if failed else -3., state,
            done=failed, next_feasible_actions=() if failed else (100, 200),
            experience_outcome='failure' if failed else 'success',
            terminal_reason='failure_soc_limited' if failed else None,
            failure_penalty_equivalent_cny=10000. if failed else 0.)


def candidate(**kwargs):
    return DirectPowerDDQN(seed=42, replay_capacity=80, reward_scale=.001,
        failure_terminal_quota=2, replay_sampling='prioritized', **kwargs)


def test_uniform_option_preserves_batch_rng_and_unweighted_loss():
    old = DirectPowerDDQN(seed=42, replay_capacity=80, failure_terminal_quota=2)
    explicit = DirectPowerDDQN(seed=42, replay_capacity=80, failure_terminal_quota=2,
        replay_sampling='uniform')
    insert(old); insert(explicit)
    assert old._sample_replay_batch(64) == explicit._sample_replay_batch(64)
    assert old.random.getstate() == explicit.random.getstate()
    assert old.learn() == explicit.learn()
    assert all(torch.equal(v, explicit.online.state_dict()[k]) for k, v in old.online.state_dict().items())
    assert old.economic_optimizer_updates == old.target_soft_update_calls == 1
    assert old.training_state()['format_version'] == 2


def test_prioritized_quota_and_paired_weights_are_correct():
    agent = candidate()
    insert(agent)
    for replay_id in range(70):
        agent._priority_index.update(replay_id, 100. if replay_id == 0 else .1)
    batch, weights, probabilities = agent._sample_replay_batch_with_weights(64)
    assert len(batch) == len(weights) == len(probabilities) == 64
    assert sum(t.terminal_reason in FAILURE_TERMINALS for t in batch) == 2
    failures = [t.replay_id for t in batch if t.terminal_reason in FAILURE_TERMINALS]
    assert len(set(failures)) == 2
    normalizer = agent._priority_index.importance_normalizer(1.)
    for t, weight, probability in zip(batch, weights, probabilities):
        expected = 1. if t.terminal_reason in FAILURE_TERMINALS else (70 * probability) ** -1
        assert weight == pytest.approx(expected / normalizer)
    assert sum(t.replay_id == 0 for t in batch) > 2


def test_weighted_huber_priority_and_target_sync_use_same_unweighted_td():
    agent = candidate()
    insert(agent)
    batch, weights, probabilities = agent._sample_replay_batch_with_weights(64)
    agent._sample_replay_batch_with_weights = lambda _: (batch, weights, probabilities)
    states = torch.tensor([t.state for t in batch])
    nxt = torch.tensor([t.next_state for t in batch])
    masks = torch.zeros(64, 61, dtype=torch.bool)
    for i, t in enumerate(batch):
        masks[i, list(t.next_feasible_indices)] = True
    with torch.no_grad():
        q = agent.online(states).gather(1, torch.tensor([[t.action_index] for t in batch]))
        y = masked_double_dqn_targets(torch.tensor([[t.reward_cny] for t in batch]),
            agent.online(nxt), agent.target(nxt), torch.tensor([[float(t.done)] for t in batch]),
            next_action_masks=masks, gamma=1.)
        expected_loss = (F.smooth_l1_loss(q, y, reduction='none').flatten() * torch.tensor(weights)).mean()
        errors = (y - q).abs().flatten().tolist()
    assert agent.learn() == pytest.approx(expected_loss.item())
    priorities = dict(agent._priority_index.state_dict()['entries'])
    for t, error in zip(batch, errors):
        if t.terminal_reason not in FAILURE_TERMINALS:
            assert priorities[t.replay_id] == pytest.approx(error)
    assert agent.economic_optimizer_updates == agent.target_soft_update_calls == 1
    stats = agent.td_statistics()
    assert stats['replay_sampling']['mode'] == 'prioritized'
    assert stats['replay_sampling']['draw_count'] == 64
    assert stats['failure_terminal']['sample_count'] == 2
    assert stats['mean_absolute_td_error'] == pytest.approx(sum(errors) / 64, rel=1e-6)
    assert stats['replay_sampling']['mean_unweighted_smooth_l1_loss'] == pytest.approx(
        F.smooth_l1_loss(q, y).item())


def test_prioritized_fifo_and_resume_reproduce_draws_and_updates():
    agent = candidate()
    insert(agent)
    agent.learn()
    insert(agent, count=10, failures=0)
    assert len(agent.replay) == 80
    assert set(dict(agent._priority_index.state_dict()['entries'])) == set(agent._ordinary_positions)
    assert not agent._priority_index.contains(0)
    saved = copy.deepcopy(agent.training_state())
    assert saved['format_version'] == 3  # Old uniform-only loaders cannot silently resume this.
    restored = candidate()
    restored.load_training_state(saved)
    assert restored._sample_replay_batch_with_weights(64) == agent._sample_replay_batch_with_weights(64)
    assert restored.learn() == pytest.approx(agent.learn())
    assert all(torch.equal(v, restored.online.state_dict()[k]) for k, v in agent.online.state_dict().items())
    with pytest.raises(ValueError, match='sampling'):
        DirectPowerDDQN(replay_capacity=80).load_training_state(saved)
    with pytest.raises(ValueError, match='sampling'):
        candidate(per_alpha=.5).load_training_state(saved)
    broken = copy.deepcopy(saved)
    broken['priority_index'] = None
    with pytest.raises(ValueError, match='priority'):
        candidate().load_training_state(broken)


@pytest.mark.parametrize('failures', [0, 1, 70])
def test_quota_shortage_fallback_matches_original_counts(failures):
    agent = candidate()
    insert(agent, count=74 - failures, failures=failures)
    batch = agent._sample_replay_batch(64)
    expected = max(min(2, failures, 64), 64 - (74 - failures))
    assert sum(t.terminal_reason in FAILURE_TERMINALS for t in batch) == expected


def test_quota_zero_indexes_both_categories_and_policy_is_unchanged():
    prioritized = DirectPowerDDQN(seed=42, replay_capacity=80, replay_sampling='prioritized')
    uniform = DirectPowerDDQN(seed=42, replay_capacity=80)
    insert(prioritized); insert(uniform)
    assert len(prioritized._priority_index) == 74
    assert prioritized._priority_index.contains(73)
    state = (.2, .1, 0., 0., 1.6, 0., 0.)
    for epsilon in (0., 1., .5):
        assert prioritized.select_power(state, (400, 600), epsilon=epsilon) == uniform.select_power(state, (400, 600), epsilon=epsilon)
    assert prioritized.random.getstate() == uniform.random.getstate()


@pytest.mark.parametrize('kwargs', [dict(replay_sampling='bad'), dict(per_alpha=-1),
    dict(per_beta=2), dict(per_epsilon=0), dict(per_epsilon=float('nan'))])
def test_invalid_sampling_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DirectPowerDDQN(**kwargs)
