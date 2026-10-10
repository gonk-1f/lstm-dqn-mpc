from dataclasses import replace
from math import fsum

import pytest
import torch

from test_v4_control import NoSolveAccountant, episode
from v3.control import AccountState
from v4.control import ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue


@pytest.mark.parametrize('n', (1,8))
def test_n_step_preserves_terminal_settlement_correction_and_endpoint_mask(n):
    result = replay_episode(episode(('onboard',)*11, (100,)*11, (0,)*11),
                            lambda _s,_a:0, accountant=NoSolveAccountant(), beta_soc=500,
                            redistribute_battery_energy=True)
    agent = DirectPowerDDQN(seed=42, n_step=n)
    assert agent.remember_trajectory(result.transitions) == 11
    assert agent.economic_replay_insertions == 11
    for index,item in enumerate(agent.replay):
        end = min(index+n, 11)
        tail = result.transitions[end-1]
        assert item.reward_cny == pytest.approx(fsum(t.reward_cny for t in result.transitions[index:end]))
        assert item.bootstrap_steps == end-index
        assert item.done == tail.done and item.next_state == tail.next_state
        assert tuple(i*10 for i in item.next_feasible_indices) == tail.next_feasible_actions
    assert result.transitions[-1].modeled_terminal_ledger is not None
    assert agent.learn(batch_size=4) is not None
    assert agent.td_statistics()['optimizer_updates'] == 1
    assert agent.td_statistics()['sample_count'] == 4


def test_n_step_crosses_shore_within_sample_but_not_sample_terminal():
    result = replay_episode(episode(('onboard',)*3+('shore_charging',)+('onboard',)*9,
                                    (100,)*3+(0,)+(200,)*9, (0,)*3+(-624,)+(0,)*9),
                            lambda _s,_a:100, accountant=NoSolveAccountant(), beta_soc=500,
                            redistribute_battery_energy=True)
    agent = DirectPowerDDQN(seed=42, n_step=8, replay_capacity=4)
    assert agent.remember_trajectory(result.transitions) == 12
    assert agent.economic_replay_insertions == 12 and len(agent.replay) == 4
    full = DirectPowerDDQN(seed=42, n_step=8)
    full.remember_trajectory(result.transitions)
    assert full.replay[0].bootstrap_steps == 8 and not full.replay[0].done
    assert full.replay[0].reward_cny == pytest.approx(fsum(t.reward_cny for t in result.transitions[:8]))
    assert result.transitions[2].shore_ledger is not None and not result.transitions[2].done
    assert result.transitions[2].next_state == result.transitions[3].state
    assert full.replay[3].bootstrap_steps == 8 and not full.replay[3].done
    assert full.replay[3].reward_cny == pytest.approx(fsum(t.reward_cny for t in result.transitions[3:11]))
    with pytest.raises(ValueError, match='completed'):
        full.remember_trajectory((replace(result.transitions[-1],done=False),))


def test_n_step_target_uses_discount_power_and_online_feasible_selection():
    result = masked_double_dqn_targets(torch.tensor([[-5.],[-7.]]),
        torch.tensor([[99.,4.,3.],[99.,4.,3.]]),
        torch.tensor([[1000.,11.,22.],[1000.,11.,22.]]), torch.tensor([[0.],[1.]]),
        next_action_masks=torch.tensor([[False,True,True],[False,False,False]]),
        gamma=.9, bootstrap_steps=torch.tensor([[8],[3]]))
    assert result[:,0].tolist() == pytest.approx([-5+.9**8*11,-7],abs=1e-6)


def test_n_step_refuses_spliced_noncontiguous_voyages_without_boundary():
    result = replay_episode(episode(('onboard',)*3,(100,200,0),(0,)*3),
                            lambda _s,_a:100,accountant=NoSolveAccountant())
    agent = DirectPowerDDQN(seed=42,n_step=8)
    mixed = (result.transitions[0],replace(result.transitions[-1],state=(.3,)*8))
    with pytest.raises(ValueError,match='contiguous'):
        agent.remember_trajectory(mixed)
    assert not agent.replay and agent.economic_replay_insertions == 0


def _soc_limited_failure(executed_steps, *, previous_voyage=False):
    """Only the next, unexecuted 1000 kW row is infeasible after 600 kW discharge."""
    accountant = NoSolveAccountant()
    loads = (0.,) * (executed_steps - 1) + (600., 1000.)
    modes = ('onboard',) * (executed_steps + 1)
    shore = (0.,) * len(modes)
    initial_soc = .2135
    if previous_voyage:
        modes = ('onboard',) * 3 + ('shore_charging',) + modes
        loads = (0.,) * 4 + loads
        shore = (0., 0., 0., -100.) + shore
        initial_soc = .212
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(modes, loads, shore), lambda _s, _mask: 0,
                       accountant=accountant, initial_state=AccountState(soc=initial_soc),
                       beta_soc=500, redistribute_battery_energy=True)
    error = caught.value
    assert error.failure_kind == 'no_feasible_action'
    assert error.failure_cause == 'soc_limited'
    penalty = FailurePenalty.from_reference()
    view = prepare_failure_replay(
        error, penalty=penalty, energy_value=BatteryEnergyValue.from_accountant(accountant),
        redistribute_battery_energy=True,
    )
    return error, view, penalty


@pytest.mark.parametrize('executed_steps', (2, 7, 8, 9, 12))
def test_n_step_eight_soc_failure_windows_stop_at_real_terminal(executed_steps):
    error, view, penalty = _soc_limited_failure(executed_steps)
    assert len(error.executed_transitions) == executed_steps
    assert len(view.transitions) == executed_steps
    assert view.failed_suffix_transitions == executed_steps
    assert sum(t.terminal_reason == 'failure_soc_limited' for t in view.transitions) == 1
    assert sum(t.failure_penalty_equivalent_cny for t in view.transitions) == pytest.approx(penalty.amount)
    assert fsum(t.reward_cny for t in view.transitions) == pytest.approx(
        fsum(t.original_reward for t in error.executed_transitions) - penalty.amount)
    assert all(left.executed_ledger == right.executed_ledger
               for left, right in zip(error.executed_transitions, view.transitions))
    assert all(t.shore_ledger is None and t.modeled_terminal_ledger is None
               for t in view.transitions)

    agent = DirectPowerDDQN(seed=42, n_step=8, reward_scale=.001,
                            failure_terminal_quota=2)
    assert agent.remember_trajectory(view.transitions) == executed_steps
    terminal_entries = min(8, executed_steps)
    assert len(agent.replay) == agent.economic_failure_replay_insertions == executed_steps
    assert agent.failure_terminal_replay_count == terminal_entries
    assert agent.economic_failure_terminal_insertions == terminal_entries
    for index, item in enumerate(agent.replay):
        end = min(index + 8, executed_steps)
        tail = view.transitions[end - 1]
        assert item.reward_cny == pytest.approx(.001 * fsum(
            t.reward_cny for t in view.transitions[index:end]))
        unpenalized = fsum(
            t.original_reward + t.immediate_battery_energy_adjustment + t.terminal_correction
            for t in view.transitions[index:end])
        assert item.reward_cny == pytest.approx(
            .001 * (unpenalized - (penalty.amount if end == executed_steps else 0.)))
        assert item.bootstrap_steps == end - index
        assert item.next_state == tail.next_state
        assert tuple(action * 10 for action in item.next_feasible_indices) == tail.next_feasible_actions
        assert item.experience_outcome == 'failure'
        assert item.done == (end == executed_steps)
        assert item.terminal_reason == ('failure_soc_limited' if item.done else None)
        assert item.failure_penalty_equivalent_cny == (penalty.amount if item.done else 0.)
        if item.done:
            assert not item.next_feasible_indices
            target = masked_double_dqn_targets(
                torch.tensor([[item.reward_cny]]), torch.zeros((1, 61)),
                torch.full((1, 61), 999.), torch.ones((1, 1)),
                next_action_masks=torch.zeros((1, 61), dtype=torch.bool), gamma=1.,
                bootstrap_steps=torch.tensor([[item.bootstrap_steps]]))
            assert target.item() == pytest.approx(item.reward_cny, abs=1e-6)


def test_n_step_eight_failure_after_shore_propagates_across_same_sample():
    accountant = NoSolveAccountant()
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',)*3 + ('shore_charging','onboard'),
                               (0.,0.,0.,0.,2000.), (0.,)*5),
                       lambda _s,_mask:0, accountant=accountant)
    error = caught.value
    penalty = FailurePenalty.from_reference()
    view = prepare_failure_replay(error, penalty=penalty,
        energy_value=BatteryEnergyValue.from_accountant(accountant),
        redistribute_battery_energy=False)
    assert len(error.executed_transitions) == len(view.transitions) == 3
    assert view.successful_prefix_transitions == 0
    assert view.failed_suffix_transitions == 3
    assert not view.transitions[2].is_successful_terminal
    assert view.transitions[2].shore_ledger is not None
    assert all(t.failure_penalty_equivalent_cny == 0 for t in view.transitions[:2])

    agent = DirectPowerDDQN(seed=42, n_step=8, failure_terminal_quota=2)
    agent.remember_trajectory(view.transitions)
    assert all(item.done and item.terminal_reason == 'failure_structural_power'
               for item in agent.replay)
    assert agent.replay[0].reward_cny == pytest.approx(fsum(t.reward_cny for t in view.transitions))
    assert agent.failure_terminal_replay_count == 3
    assert sum(item.failure_penalty_equivalent_cny for item in agent.replay) == pytest.approx(
        3 * penalty.amount)  # Overlapping returns, one physical penalty.
    assert sum(t.failure_penalty_equivalent_cny for t in view.transitions) == pytest.approx(
        penalty.amount)


def test_n_step_eight_quota_samples_windows_not_independent_failure_events():
    _error, view, penalty = _soc_limited_failure(9)
    agent = DirectPowerDDQN(seed=17, n_step=8, reward_scale=.001,
                            failure_terminal_quota=2, replay_capacity=80)
    agent.remember_trajectory(view.transitions)
    for index in range(62):
        state = (.4, index / 100., 0., 0., 0., 0., 0., 0.)
        agent.remember(state, 0, 0., state, done=True, next_feasible_actions=())
    assert agent.economic_failure_terminal_insertions == 8
    assert agent.failure_terminal_replay_count == 8
    assert len(agent.replay) == 71
    batch = agent._sample_replay_batch(64)
    terminal = [item for item in batch if item.terminal_reason == 'failure_soc_limited']
    assert len(batch) == len({id(item) for item in batch}) == 64
    assert len(terminal) == 2
    assert all(item.failure_penalty_equivalent_cny == penalty.amount for item in terminal)
    assert agent.learn(batch_size=64) is not None
    assert agent.td_statistics()['sample_outcomes']['failure_terminal'] == 2
    for index in range(80):
        state = (.5, index / 100., 0., 0., 0., 0., 0., 0.)
        agent.remember(state, 0, 0., state, done=True, next_feasible_actions=())
    assert agent.failure_terminal_replay_count == 0
    assert not any(item.terminal_reason == 'failure_soc_limited' for item in agent.replay)


@pytest.mark.parametrize('interval', (250,500,1000))
@pytest.mark.parametrize('stride', (16,32))
def test_credit_schedule_and_hard_target_sync_are_exact(stride, interval):
    from test_v4_experiment_schedule import FakeAgent
    from v4.experiment_schedule import EconomicUpdateSchedule
    a = FakeAgent()
    schedule = EconomicUpdateSchedule(f'replay{stride}',target_mode='optimizer',target_interval=interval)
    for count in (7,9,stride*1001-13):
        schedule.grant(insertions=count,completed_episodes=5)
        schedule.consume(a,batch_size=64)
    assert a.economic_optimizer_updates == 1001
    assert schedule.remaining_transition_credit == 3
    assert a.sync_at == list(range(interval,1002,interval))
