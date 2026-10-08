from dataclasses import replace
from math import fsum

import pytest
import torch

from test_v4_control import NoSolveAccountant, episode
from v4.control import replay_episode
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets


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


def test_n_step_does_not_cross_shore_or_voyage_boundary():
    result = replay_episode(episode(('onboard',)*3+('shore_charging',)+('onboard',)*9,
                                    (100,)*3+(0,)+(200,)*9, (0,)*3+(-624,)+(0,)*9),
                            lambda _s,_a:100, accountant=NoSolveAccountant(), beta_soc=500,
                            redistribute_battery_energy=True)
    agent = DirectPowerDDQN(seed=42, n_step=8, replay_capacity=4)
    assert agent.remember_trajectory(result.transitions) == 12
    assert agent.economic_replay_insertions == 12 and len(agent.replay) == 4
    full = DirectPowerDDQN(seed=42, n_step=8)
    full.remember_trajectory(result.transitions)
    assert full.replay[0].bootstrap_steps == 3 and full.replay[0].done
    assert full.replay[0].reward_cny == pytest.approx(fsum(t.reward_cny for t in result.transitions[:3]))
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
