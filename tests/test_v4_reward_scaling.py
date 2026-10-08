"""Economic replay scaling contracts; all physical examples are synthetic."""
from dataclasses import replace
from math import fsum
from types import SimpleNamespace

import pytest
import torch

from v3.control import AccountState, EconomicMPC
from v4.control import ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue


def episode(modes, loads, shore=None):
    return SimpleNamespace(sample_id='synthetic_scale', split='train',
        operating_mode=tuple(modes), load_kw=tuple(loads),
        battery_bus_kw=tuple(shore or (0.,)*len(loads)))


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf')])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError, match='reward_scale'):
        DirectPowerDDQN(reward_scale=scale)


@pytest.mark.parametrize('modes,loads,shore,powers', [
    (('onboard',)*3, (0.,100.,0.), None, (0,0,0)),
    (('onboard',)*3, (0.,100.,0.), None, (600,600,600)),
    (('onboard','onboard','shore_charging'), (100.,0.,0.), (0.,0.,-100.), (0,0)),
    (('onboard','shore_charging','onboard'), (100.,0.,100.), (0.,-100.,0.), (0,600)),
])
@pytest.mark.parametrize('n', [1,8])
def test_complete_shore_settlement_feedback_and_n_step_scale_once(modes,loads,shore,powers,n):
    accountant=EconomicMPC(nominal_cost_cny=1)
    data=episode(modes,loads,shore)
    original=[]
    for feedback in (False,True):
        actions=iter(powers)
        result=replay_episode(data,lambda *args:next(actions),accountant=accountant,
            beta_soc=500,redistribute_battery_energy=feedback)
        normal=DirectPowerDDQN(seed=42,n_step=n)
        scaled=DirectPowerDDQN(seed=42,n_step=n,reward_scale=.001)
        normal.remember_trajectory(result.transitions)
        scaled.remember_trajectory(result.transitions)
        assert len(normal.replay)==len(scaled.replay)==len(result.transitions)
        for left,right in zip(normal.replay,scaled.replay):
            assert right.reward_cny==pytest.approx(.001*left.reward_cny)
            assert replace(right,reward_cny=left.reward_cny)==left
        # Outcomes continue to use the observed ledger, not the economic-Q scale.
        normal.remember_outcome_trajectory(result.transitions,failed=False)
        scaled.remember_outcome_trajectory(result.transitions,failed=False)
        assert tuple(normal.outcome_replay)==tuple(scaled.outcome_replay)
        original.append(result)
    a,b=original
    assert a.total_ledger==b.total_ledger and a.modeled_terminal_settlement==b.modeled_terminal_settlement
    assert a.soc_by_row==b.soc_by_row and a.fc_power_kw_by_row==b.fc_power_kw_by_row
    assert fsum(t.reward_cny for t in a.transitions)==pytest.approx(fsum(t.reward_cny for t in b.transitions),abs=1e-8)


@pytest.mark.parametrize('n', [1,8])
def test_failure_once_only_penalty_identity_no_bootstrap(n):
    accountant=EconomicMPC(nominal_cost_cny=1)
    data=episode(('onboard',)*3,(600.,1000.,0.))
    totals=[]
    for feedback in (False,True):
        with pytest.raises(ReplayExecutionError) as caught:
            replay_episode(data,lambda *args:0,accountant=accountant,
                initial_state=AccountState(soc=.2135),beta_soc=500,
                redistribute_battery_energy=feedback)
        exc=caught.value
        view=prepare_failure_replay(exc,penalty=FailurePenalty.from_reference(),
            energy_value=BatteryEnergyValue.from_accountant(accountant),
            redistribute_battery_energy=feedback)
        plain=DirectPowerDDQN(seed=42,n_step=n)
        scaled=DirectPowerDDQN(seed=42,n_step=n,reward_scale=.001)
        plain.remember_trajectory(view.transitions);scaled.remember_trajectory(view.transitions)
        assert len(view.transitions)==1
        t=view.transitions[-1];item=scaled.replay[-1]
        assert t.done and not t.is_successful_terminal and not item.next_feasible_indices
        assert t.shore_ledger is None and t.modeled_terminal_ledger is None
        assert item.failure_penalty_equivalent_cny==t.failure_penalty_equivalent_cny
        assert sum(t.failure_penalty_equivalent_cny>0 for t in view.transitions)==1
        assert item.reward_cny==pytest.approx(plain.replay[-1].reward_cny*.001)
        target=masked_double_dqn_targets(torch.tensor([[item.reward_cny]]),
            torch.zeros((1,61)),torch.full((1,61),float('nan')),torch.ones((1,1)),
            next_action_masks=torch.zeros((1,61),dtype=torch.bool),gamma=1)
        assert target.item()==pytest.approx(item.reward_cny,rel=1e-6)
        assert tuple(t.original_economic_ledger for t in view.transitions)==tuple(
            t.original_economic_ledger for t in exc.executed_transitions)
        totals.append(fsum(t.reward_cny for t in view.transitions))
    assert totals[0]==pytest.approx(totals[1],abs=1e-8)


def test_capacity_insertion_and_direct_transition_share_only_one_scale():
    data=episode(('onboard',),(100.,))
    result=replay_episode(data,lambda *args:0,accountant=EconomicMPC(nominal_cost_cny=1),beta_soc=500)
    a=DirectPowerDDQN(reward_scale=.001,replay_capacity=1)
    a.remember_transition(result.transitions[0])
    a.remember_trajectory(result.transitions)
    assert len(a.replay)==1 and a.economic_replay_insertions==2
    assert a.replay[-1].reward_cny==pytest.approx(result.transitions[0].reward_cny*.001)


def test_real_optimizer_statistics_units_sampling_and_clip_observation():
    a=DirectPowerDDQN(reward_scale=.001)
    with torch.no_grad():
        for p in a.online.parameters():p.zero_()
        for p in a.target.parameters():p.zero_()
    state=(.6,)*8
    a.remember(state,0,-1000.,state,done=True,next_feasible_actions=())
    a.remember(state,10,-11000.,state,done=True,next_feasible_actions=(),
        experience_outcome='failure',terminal_reason='failure_soc_limited',
        failure_penalty_equivalent_cny=10000.)
    a.learn(batch_size=2)
    stats=a.td_statistics()
    assert stats['reward_scale']==.001 and stats['sample_count']==2
    assert stats['mean_absolute_td_error']==pytest.approx(6.)
    assert stats['original_units']['mean_absolute_td_error']==pytest.approx(6000.)
    assert stats['failure_terminal']['mean_absolute_td_error']==pytest.approx(11.)
    assert stats['failure_terminal']['original_units']['mean_absolute_td_error']==pytest.approx(11000.)
    assert stats['q_distribution']['mean']==0
    assert stats['target_distribution']['original_units']['mean']==pytest.approx(-6000.)
    assert stats['sample_outcomes']['failure_fraction']==.5
    assert stats['gradient_statistics']['update_count']==1
    assert stats['gradient_statistics']['mean_preclip_norm']>0
    assert stats['gradient_statistics']['clipped_updates']==0
    assert a.economic_optimizer_updates==1 and a.target_sync_calls==1
    a.reset_td_statistics()
    assert a.td_statistics()['sample_count']==0 and a.economic_optimizer_updates==1


def test_default_one_is_exact_explicit_one_for_optimizer_and_rng():
    a,b=DirectPowerDDQN(seed=42),DirectPowerDDQN(seed=42,reward_scale=1)
    state=(.6,)*8
    for agent in (a,b):
        for power in (0,600):agent.remember(state,power,-123.,state,done=True,next_feasible_actions=())
        agent.learn(batch_size=2)
    assert tuple(a.replay)==tuple(b.replay) and a.random.getstate()==b.random.getstate()
    assert a.td_statistics()==b.td_statistics()
    for name in ('online','target','outcome_model'):
        assert all(torch.equal(v,getattr(b,name).state_dict()[k]) for k,v in getattr(a,name).state_dict().items())
