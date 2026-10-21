from types import SimpleNamespace

import pytest

from v3.control import EconomicMPC
from v4.control import replay_episode
from v4.dqn import DirectPowerDDQN
from v4.failure_replay import FailurePenalty
from v4.independent_economics import ZhouEvaluationConfig
from v4.v5_metrics import economic_action_pair_statistics


def test_exact_pair_metrics_preserve_agent_and_reconstruct_causal_successors():
    ep = SimpleNamespace(sample_id='synthetic', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',)*4, load_kw=(100.,120.,90.,80.),
        battery_bus_kw=(0.,)*4, time_s=(0.,30.,60.,90.), dt_s=(30.,30.,30.,7.))
    commands = iter((60,120,90,90))
    accountant = EconomicMPC(nominal_cost_cny=1.)
    config = ZhouEvaluationConfig.from_formal_curve()
    result = replay_episode(ep,lambda s,f:next(commands),accountant=accountant,
        beta_soc=0.,state_mode='load_delta4',action_mode='increment7',
        reward_cost_mode='zhou_economic_only',economic_evaluation_config=config)
    agent=DirectPowerDDQN(state_mode='load_delta4',action_mode='increment7',n_step=1,
        gamma=1.,reward_scale=.001,replay_sampling='prioritized',failure_terminal_quota=2)
    counts=(agent.economic_optimizer_updates,agent.economic_replay_insertions,agent.target_soft_update_calls)
    rng=agent.random.getstate()
    metrics=economic_action_pair_statistics(result.transitions,agent,accountant,config,
        FailurePenalty.from_reference(),episode=ep)
    assert metrics['visited_steps']==4
    assert metrics['compared_change_hold_states']==3
    assert metrics['max_physical_reconstruction_error']<1e-9
    assert metrics['max_actual_reward_reconstruction_error']<1e-9
    assert metrics['economic_reward_components_cny']['signed_soc_equivalent_h2']!=0
    assert counts==(agent.economic_optimizer_updates,agent.economic_replay_insertions,agent.target_soft_update_calls)
    assert rng==agent.random.getstate()
    assert all('delta_y' in r and 'delta_q' in r for r in metrics['action_pairs'])
    assert len(metrics['action_pairs'])<=16
    # Actual evaluation path runs both exact diagnostics and summary, without learning.
    from v4.monitored_training import greedy_evaluate
    summary=greedy_evaluate((ep,),agent,accountant,0.,reward_cost_mode='zhou_economic_only',
        state_mode='load_delta4',action_mode='increment7',economic_evaluation_config=config,
        learn_no_feasible_failures=True,economic_action_pair_diagnostics=True)
    assert summary['completed']==1
    assert len(summary['economic_action_pair_diagnostics']['samples'])==1
    assert counts==(agent.economic_optimizer_updates,agent.economic_replay_insertions,agent.target_soft_update_calls)
    assert rng==agent.random.getstate()


def test_pair_metrics_reject_missing_exact_episode_timing():
    with pytest.raises(ValueError,match='timing'):
        economic_action_pair_statistics((),None,None,None,None,episode=None)


def test_four_dimensional_interleaving_and_failure_finalize_once():
    from v4.control import ReplayExecutionError
    from v4.experiment_schedule import EconomicUpdateSchedule
    from v4.interleaved_replay import InterleavedReplayWriter
    from v4.failure_replay import prepare_failure_replay
    from v4.reward_feedback import BatteryEnergyValue
    ep=SimpleNamespace(sample_id='stream4',split='train',task_mode='independent_voyage',
        operating_mode=('onboard',)*67,load_kw=(0.,)*66+(2000.,),
        time_s=tuple(30.*i for i in range(67)),dt_s=(30.,)*67,
        source_1s_row=tuple(30*i for i in range(67)))
    accountant=EconomicMPC(nominal_cost_cny=1.)
    config=ZhouEvaluationConfig.from_formal_curve()
    agent=DirectPowerDDQN(state_mode='load_delta4',action_mode='increment7',n_step=1,
        gamma=1.,reward_scale=.001,replay_sampling='prioritized',failure_terminal_quota=2)
    for _ in range(64):
        agent.remember((.6,0.,0.,0.),0,-1.,(.6,0.,0.,0.),done=True,next_feasible_actions=())
    writer=InterleavedReplayWriter(agent,EconomicUpdateSchedule('replay32',target_mode='soft'),
        ep,source_round=1,batch_size=64,losses=[])
    observed_updates=[]
    def policy(state,feasible):
        observed_updates.append(agent.economic_optimizer_updates)
        return agent.select_power(state,feasible,epsilon=0.)
    with pytest.raises(ReplayExecutionError) as raised:
        replay_episode(ep,policy,accountant=accountant,beta_soc=0.,state_mode='load_delta4',
            action_mode='increment7',reward_cost_mode='zhou_economic_only',
            economic_evaluation_config=config,on_transition=writer.observe)
    penalty=FailurePenalty.from_reference()
    failure=prepare_failure_replay(raised.value,penalty=penalty,
        energy_value=BatteryEnergyValue.from_accountant(accountant),redistribute_battery_energy=False)
    writer.finish_failure(failure.transitions)
    assert observed_updates[31]==0 and observed_updates[32]==1
    assert agent.economic_optimizer_updates==agent.target_soft_update_calls==2
    items=[t for t in agent.replay if t.source_sample_id=='stream4']
    assert len(items)==66 and sum(t.done for t in items)==1
    assert sum(t.failure_penalty_equivalent_cny>0 for t in items)==1
    assert items[-1].reward_cny==pytest.approx(failure.transitions[-1].reward_cny*.001)
    assert items[-1].terminal_reason=='failure_structural_power'
    assert items[-1].next_feasible_indices==()
    metrics=economic_action_pair_statistics(failure.transitions,agent,accountant,config,
        penalty,episode=ep)
    assert metrics['failure_penalty_reward_units']==penalty.amount
    assert metrics['max_actual_reward_reconstruction_error']<1e-8
