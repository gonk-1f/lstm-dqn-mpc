"""Small synthetic checks of update timing, not a formal training experiment."""
from types import SimpleNamespace

import pytest
import torch

from v3.control import EconomicMPC
from v4.control import ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN
from v4.experiment_schedule import EconomicUpdateSchedule
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.independent_economics import ZhouEvaluationConfig
from v4.reward_feedback import BatteryEnergyValue


def episode(*, fail=False, split='train'):
    loads = (0.,)*66 + ((2000.,) if fail else ())
    return SimpleNamespace(sample_id='synthetic_'+split, split=split,
        task_mode='independent_voyage', operating_mode=('onboard',)*len(loads),
        load_kw=loads, time_s=tuple(30.*i for i in range(len(loads))),
        dt_s=(30.,)*len(loads), duration_s=30.*len(loads),
        source_1s_row=tuple(1000+30*i for i in range(len(loads))))


def replay_kwargs():
    return dict(accountant=EconomicMPC(nominal_cost_cny=1.), state_mode='minimal3',
        action_mode='increment7', reward_cost_mode='zhou_joint', beta_soc=200.,
        economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())


def agent_with_ready_replay():
    agent = DirectPowerDDQN(state_mode='minimal3', action_mode='increment7', n_step=1,
        replay_capacity=300000, learning_rate=.001, reward_scale=.001,
        replay_sampling='prioritized', failure_terminal_quota=2)
    for _ in range(64):
        agent.remember((.6,0.,0.),0,-1.,(.6,0.,0.),done=True,next_feasible_actions=())
    return agent


@pytest.mark.parametrize('fail', [False, True])
def test_closed_callback_never_exposes_unpenalized_failure_tail(fail):
    data=episode(fail=fail)
    callbacks=[]
    kwargs=replay_kwargs()
    if fail:
        with pytest.raises(ReplayExecutionError) as raised:
            replay_episode(data,lambda s,a:0, on_transition=lambda i,t:callbacks.append((i,t)),**kwargs)
        transitions=raised.value.executed_transitions
        assert len(callbacks)==65 and len(transitions)==66
        assert not transitions[-1].done and not transitions[-1].next_feasible_actions
    else:
        result=replay_episode(data,lambda s,a:0,
            on_transition=lambda i,t:callbacks.append((i,t)),**kwargs)
        reference=replay_episode(data,lambda s,a:0,**kwargs)
        assert result==reference
        assert len(callbacks)==66 and callbacks[-1][1].done
    assert all(t.done or t.next_feasible_actions for _,t in callbacks)
    assert [i for i,_ in callbacks]==list(range(len(callbacks)))


@pytest.mark.parametrize('fail', [False, True])
def test_interleaved_updates_before_next_decision_and_preserves_failure_replay(fail):
    from v4.interleaved_replay import InterleavedReplayWriter
    data=episode(fail=fail)
    agent=agent_with_ready_replay()
    schedule=EconomicUpdateSchedule('replay32',target_mode='soft')
    losses=[]
    writer=InterleavedReplayWriter(agent,schedule,data,source_round=1,batch_size=64,losses=losses)
    observations=[]
    def policy(s,a):
        observations.append(agent.economic_optimizer_updates)
        return 0
    if fail:
        with pytest.raises(ReplayExecutionError) as raised:
            replay_episode(data,policy,on_transition=writer.observe,**replay_kwargs())
        error=raised.value
        kw=replay_kwargs()
        view=prepare_failure_replay(error,penalty=FailurePenalty.from_reference(),
            energy_value=BatteryEnergyValue.from_accountant(kw['accountant']),
            redistribute_battery_energy=False)
        priorities=agent._priority_index.state_dict()
        writer.finish_failure(view.transitions)
        assert priorities==agent._priority_index.state_dict()
        expected=DirectPowerDDQN(state_mode='minimal3',action_mode='increment7',n_step=1,
            reward_scale=.001)
        expected.remember_trajectory(view.transitions,source_sample_id=data.sample_id,
            source_round=1,source_rows=data.source_1s_row[:66])
        actual=[t for t in agent.replay if t.source_sample_id==data.sample_id]
        assert [t.__dict__ | {'replay_id':None} for t in actual]==[
            t.__dict__ | {'replay_id':None} for t in expected.replay]
        assert sum(t.failure_penalty_equivalent_cny>0 for t in actual)==1
        assert sum(t.done for t in actual)==1 and actual[-1].terminal_reason=='failure_structural_power'
        assert agent.economic_success_replay_insertions==64
        assert agent.economic_failure_replay_insertions==66
        assert agent._pool_categories.get('failure_prefix')==65
        assert agent.failure_terminal_replay_count==1
    else:
        result=replay_episode(data,policy,on_transition=writer.observe,**replay_kwargs())
        writer.finish_success(result.transitions)
    assert observations[31]==0 and observations[32]==1
    assert observations[63]==1 and observations[64]==2
    assert agent.economic_optimizer_updates==agent.target_soft_update_calls==2
    assert agent.target_sync_calls==1
    assert schedule.remaining_transition_credit==2 and len(losses)==2
    assert writer.inserted_count==66


@pytest.mark.parametrize('fail', [False, True])
def test_interleaved_metadata_and_resume_are_bound_to_timing(tmp_path,monkeypatch,fail):
    from v4 import monitored_training
    monkeypatch.setattr(monitored_training,'write_learning_curves',lambda *args:{})
    class Data:
        task_mode='independent_voyage'
        task_contract={'version':'synthetic'}
        opened_test_payloads=0
        def load_train(self): return (episode(fail=fail),)
        def load_validation(self): return (episode(split='validation'),)
    config=dict(rounds=2,state_mode='minimal3',action_mode='increment7',n_step=1,
        reward_cost_mode='zhou_joint',replay_sampling='prioritized',batch_size=64,
        update_timing='interleaved',experiment_id='SYNTHETIC-INTERLEAVED',
        economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
    reference,reference_report=monitored_training.run_monitored_training(Data(),
        output_dir=tmp_path/'reference',**config)
    replay=monitored_training.replay_episode
    explorations=0
    def interrupt_second_exploration(*args,**kwargs):
        nonlocal explorations
        if kwargs.get('on_transition') is not None:
            explorations+=1
            if explorations==2:
                raise KeyboardInterrupt('synthetic interruption after one complete round')
        return replay(*args,**kwargs)
    monkeypatch.setattr(monitored_training,'replay_episode',interrupt_second_exploration)
    output=tmp_path/'resume'
    with pytest.raises(KeyboardInterrupt):
        monitored_training.run_monitored_training(Data(),output_dir=output,**config)
    checkpoint=torch.load(output/'training_state_latest.pt',weights_only=False)
    report=checkpoint['report']
    saved=checkpoint['agent']
    assert report['hyperparameters']['update_timing']=='interleaved'
    assert report['bootstrap']['executed_onboard_transitions']==66
    assert saved['counts']['economic_replay_insertions']==132
    assert saved['counts']['economic_optimizer_updates']==saved['counts']['target_soft_update_calls']==132//32
    row=report['rounds'][0]
    assert row['onboard_callback_optimizer_updates']==2
    assert row['episode_end_optimizer_updates']==0
    assert row['validation_skip_reason'] is None
    assert row['greedy_validation']['evaluable_episodes']==1
    if fail:
        assert saved['counts']['economic_failure_terminal_insertions']==2
        assert saved['counts']['economic_failure_replay_insertions']==132
        assert sum(t.failure_penalty_equivalent_cny>0 for t in saved['replay'])==2
    monkeypatch.setattr(monitored_training,'replay_episode',replay)
    restored,again=monitored_training.run_monitored_training(Data(),output_dir=output,
        resume_from=output/'training_state_latest.pt',**config)
    assert restored.economic_optimizer_updates==reference.economic_optimizer_updates
    assert restored.target_soft_update_calls==reference.target_soft_update_calls
    assert tuple(restored.replay)==tuple(reference.replay)
    assert restored.random.getstate()==reference.random.getstate()
    for kind in ('online','target'):
        assert all(torch.equal(v,getattr(restored,kind).state_dict()[k])
                   for k,v in getattr(reference,kind).state_dict().items())
    before=reference.training_state()
    assert restored.optimizer.state_dict()['param_groups']==before['optimizer']['param_groups']
    for key, state in before['optimizer']['state'].items():
        for field,value in state.items():
            loaded=restored.optimizer.state_dict()['state'][key][field]
            assert torch.equal(value,loaded) if isinstance(value,torch.Tensor) else value==loaded
    assert again['execution_counts']['remaining_transition_credit']==reference_report['execution_counts']['remaining_transition_credit']
    class NotLoaded(Data):
        def load_train(self): pytest.fail('timing mismatch must reject before loading data')
    with pytest.raises(ValueError,match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(NotLoaded(),output_dir=output,
            resume_from=output/'training_state_latest.pt',**(config|{'update_timing':'episode_end'}))


@pytest.mark.parametrize('override', [{'cadence':'replay16'},{'n_step':8},{'target_mode':'round'},
                                      {'learn_no_feasible_failures':False}])
def test_interleaved_invalid_scope_rejected_before_data(tmp_path,override):
    from v4.monitored_training import run_monitored_training
    class Unopened:
        task_mode='independent_voyage'
        def load_train(self): pytest.fail('invalid mode must reject before payload access')
    config=dict(state_mode='minimal3',action_mode='increment7',reward_cost_mode='zhou_joint',
        n_step=1,update_timing='interleaved',experiment_id='SYNTHETIC-INTERLEAVED',
        economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())|override
    with pytest.raises(ValueError):
        run_monitored_training(Unopened(),output_dir=tmp_path,**config)


def test_interleaved_cli_routes_mode_without_reading_payloads(tmp_path,monkeypatch):
    from v4 import feedback_study
    dataset=SimpleNamespace(opened_test_payloads=0,split_sizes=(35,8),identity_hashes=lambda:{'split':'frozen'})
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset,'open',lambda root:dataset)
    captured={}
    def capture(selected,**kwargs):
        captured.update(kwargs); kwargs['output_dir'].mkdir()
        return None,{'rounds':[],'best_checkpoint':None}
    monkeypatch.setattr(feedback_study,'run_monitored_training',capture)
    assert feedback_study.main(['--output-dir',str(tmp_path/'run'),'--dataset-mode','independent',
        '--state-mode','minimal3','--action-mode','increment7','--reward-cost-mode','zhou_joint',
        '--n-step','1','--replay-sampling','prioritized','--update-timing','interleaved',
        '--experiment-id','SYNTHETIC-INTERLEAVED'])==0
    assert captured['update_timing']=='interleaved' and captured['cadence']=='replay32'
    assert captured['batch_size']==64 and captured['n_step']==1


def test_first_step_infeasible_never_creates_a_replay_action():
    from v4.interleaved_replay import InterleavedReplayWriter
    agent=agent_with_ready_replay()
    schedule=EconomicUpdateSchedule('replay32',target_mode='soft')
    data=episode()
    data.load_kw=(2000.,)
    data.operating_mode=('onboard',)
    data.time_s=(0.,); data.dt_s=(30.,); data.duration_s=30.
    writer=InterleavedReplayWriter(agent,schedule,data,source_round=1,batch_size=64,losses=[])
    with pytest.raises(ReplayExecutionError) as raised:
        replay_episode(data,lambda *args:pytest.fail('no action may be selected'),
                       on_transition=writer.observe,**replay_kwargs())
    assert raised.value.executed_transitions==()
    assert writer.finish_failure(())==0
    assert agent.economic_replay_insertions==64 and agent.failure_terminal_replay_count==0
    assert agent.economic_optimizer_updates==schedule.remaining_transition_credit==0


@pytest.mark.parametrize('extra', [[], ['--cadence','replay16','--experiment-id','SYNTHETIC-MIXED']])
def test_interleaved_cli_cannot_mix_update_density_or_omit_identity(tmp_path,monkeypatch,extra):
    from v4 import feedback_study
    monkeypatch.setattr(feedback_study.IndependentTrainingDataset,'open',
        lambda *args:pytest.fail('invalid study must reject before opening a dataset'))
    with pytest.raises(SystemExit) as raised:
        feedback_study.main(['--output-dir',str(tmp_path/'run'),'--dataset-mode','independent',
            '--state-mode','minimal3','--action-mode','increment7','--reward-cost-mode','zhou_joint',
            '--n-step','1','--replay-sampling','prioritized','--update-timing','interleaved',*extra])
    assert raised.value.code==2 and not (tmp_path/'run').exists()
