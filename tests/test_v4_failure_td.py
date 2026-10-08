from dataclasses import replace
from math import fsum

import pytest
import torch

from test_v4_control import NoSolveAccountant, episode
from v3.control import AccountState
from v2.data.supervisory_rules import OperatingMode
from v4.control import ACTION_KW, ReplayExecutionError, feasible_fc_actions, replay_episode
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.reward_feedback import BatteryEnergyValue


def branch(first_power=0, *, feedback=False, modes=None, loads=None, initial_soc=.2135):
    accountant = NoSolveAccountant()
    data = episode(modes or ('onboard',)*3, loads or (600,1000,0), (0,)*len(modes or ('onboard',)*3))
    powers = iter((first_power,600,0))
    try:
        result = replay_episode(data,lambda _s,_a:next(powers),accountant=accountant,
            initial_state=AccountState(soc=initial_soc),beta_soc=500,
            redistribute_battery_energy=feedback)
        return result, accountant
    except ReplayExecutionError as error:
        return error, accountant


def prepare(error, accountant, *, feedback=False, scale=1):
    from v4.failure_replay import FailurePenalty, prepare_failure_replay
    return prepare_failure_replay(error,penalty=FailurePenalty.from_reference(scale=scale),
        energy_value=BatteryEnergyValue.from_accountant(accountant),redistribute_battery_energy=feedback)


def test_same_state_actions_have_controllable_soc_failure_and_safe_completion():
    bad, accountant = branch(0)
    good, _ = branch(600)
    assert isinstance(bad,ReplayExecutionError)
    assert bad.failure_cause == 'soc_limited'
    assert good.transitions[0].state == bad.executed_transitions[0].state
    assert 0 in feasible_fc_actions(AccountState(soc=.2135),600,accountant)
    assert 600 in feasible_fc_actions(AccountState(soc=.2135),600,accountant)
    prepared = prepare(bad,accountant)
    assert prepared.failed_suffix_transitions == 1
    assert not prepared.event_only
    terminal = prepared.transitions[-1]
    assert terminal.done and terminal.terminal_reason == 'failure_soc_limited'
    assert not terminal.is_successful_terminal and terminal.next_feasible_actions == ()
    assert terminal.shore_ledger is terminal.modeled_terminal_ledger is None
    assert terminal.state == bad.executed_transitions[-1].state
    assert terminal.next_state == bad.executed_transitions[-1].next_state
    assert terminal.executed_ledger == bad.executed_transitions[-1].executed_ledger


@pytest.mark.parametrize('feedback',(False,True))
def test_failed_suffix_enters_economic_replay_without_bootstrap_or_fake_cost(feedback):
    error,accountant = branch(feedback=feedback)
    prepared = prepare(error,accountant,feedback=feedback)
    agent = DirectPowerDDQN(seed=42)
    assert agent.remember_trajectory(prepared.transitions) == 1
    item = agent.replay[-1]
    assert item.done and item.experience_outcome == 'failure'
    assert item.terminal_reason == 'failure_soc_limited'
    assert not item.next_feasible_indices
    assert agent.economic_failure_replay_insertions == 1
    assert agent.economic_success_replay_insertions == 0
    assert agent.economic_failure_terminal_insertions == 1
    target = masked_double_dqn_targets(torch.tensor([[item.reward_cny]]),
        torch.full((1,61),float('nan')),torch.full((1,61),float('nan')),torch.ones((1,1)),
        next_action_masks=torch.zeros((1,61),dtype=torch.bool),gamma=1.)
    assert target.item() == pytest.approx(item.reward_cny,rel=1e-6)
    assert prepared.transitions[-1].original_economic_ledger == error.executed_transitions[-1].original_economic_ledger
    assert fsum(t.reward_cny for t in prepared.transitions) == pytest.approx(
        fsum(t.original_reward for t in error.executed_transitions)-item.failure_penalty_equivalent_cny,
        rel=1e-12,abs=1e-8)


def test_original_and_redistributed_failed_trajectories_have_identical_penalized_returns():
    old,a = branch(feedback=False)
    new,b = branch(feedback=True)
    original = prepare(old,a,feedback=False)
    shifted = prepare(new,b,feedback=True)
    assert shifted.transitions[-1].terminal_correction != 0
    assert fsum(t.reward_cny for t in original.transitions) == pytest.approx(
        fsum(t.reward_cny for t in shifted.transitions),rel=1e-12,abs=1e-8)
    assert sum(t.failure_penalty_equivalent_cny>0 for t in shifted.transitions) == 1
    assert old.executed_transitions[-1].done is False
    assert new.executed_transitions[-1].terminal_correction == 0
    repeated = prepare(new,b,feedback=True)
    assert repeated == shifted
    double = ReplayExecutionError(new.row_index,new.mode,new.cause,
        executed_transitions=shifted.transitions,failure_kind=new.failure_kind,
        failure_cause=new.failure_cause)
    with pytest.raises(ValueError,match='already'):
        prepare(double,b,feedback=True)


def test_structural_capacity_failure_is_labeled_separately_from_soc():
    error,a = branch(modes=('onboard',)*2,loads=(600,2000),initial_soc=.6)
    assert error.failure_cause == 'structural_power'
    assert error.failed_state == error.executed_transitions[-1].next_state
    prepared = prepare(error,a)
    assert prepared.failure_cause == 'structural_power'
    assert prepared.transitions[-1].terminal_reason == 'failure_structural_power'
    assert not prepared.controllable_by_power_policy


def test_shore_departure_failure_with_no_executed_suffix_does_not_penalize_previous_voyage():
    error,a = branch(100,modes=('onboard','shore_charging','onboard'),loads=(100,0,2000),initial_soc=.6)
    prepared = prepare(error,a)
    assert prepared.event_only and prepared.failed_suffix_transitions == 0
    assert prepared.transitions == error.executed_transitions
    assert prepared.transitions[-1].is_successful_terminal
    assert prepared.transitions[-1].failure_penalty_equivalent_cny == 0
    agent = DirectPowerDDQN(seed=42)
    agent.remember_trajectory(prepared.transitions)
    assert agent.economic_success_replay_insertions == 1 and agent.economic_failure_replay_insertions == 0


def test_completed_voyage_then_failed_voyage_are_separate_td_segments():
    data = episode(('onboard','shore_charging','onboard','onboard'),(0,0,600,1000),(0,0,0,0))
    a=NoSolveAccountant()
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(data,lambda _s,_a:0,accountant=a,initial_state=AccountState(soc=.2135),
                       beta_soc=500,redistribute_battery_energy=True)
    prepared=prepare(caught.value,a,feedback=True)
    first,last=prepared.transitions
    assert first.is_successful_terminal and not last.is_successful_terminal
    assert first == caught.value.executed_transitions[0]
    agent=DirectPowerDDQN(seed=42)
    agent.remember_trajectory(prepared.transitions)
    assert [t.experience_outcome for t in agent.replay] == ['success','failure']
    assert all(t.done and not t.next_feasible_indices for t in agent.replay)


def test_program_exception_cannot_be_converted_to_power_failure():
    cause=RuntimeError('model calculation failed')
    error=ReplayExecutionError(0,OperatingMode.ONBOARD,cause,failure_kind='execution_error')
    with pytest.raises(ValueError,match='no_feasible'):
        prepare(error,NoSolveAccountant())


def test_penalty_reference_is_frozen_train_only_and_scaled_not_a_ledger_charge():
    from v4.failure_replay import FailurePenalty
    base=FailurePenalty.from_reference()
    assert base.reference_split == 'train' and base.reference_count == 30
    assert base.reference_statistic == 'nearest_rank_p95_comparable_economic_cost'
    assert base.unit == 'CNY-equivalent reward units; not actual expenditure'
    assert FailurePenalty.from_reference(scale=.5).amount == base.amount/2
    assert FailurePenalty.from_reference(scale=2).amount == base.amount*2
    with pytest.raises(ValueError):
        FailurePenalty.from_reference(scale=float('nan'))


def test_failure_conversion_rejects_mismatched_redistribution_configuration():
    original,a=branch(feedback=False)
    shifted,b=branch(feedback=True)
    with pytest.raises(ValueError,match='redistribution'):
        prepare(original,a,feedback=True)
    with pytest.raises(ValueError,match='redistribution'):
        prepare(shifted,b,feedback=False)


def test_failure_reference_matches_immutable_train_archive():
    import hashlib,json,math
    from pathlib import Path
    from v4.failure_replay import FailurePenalty
    p=FailurePenalty.from_reference()
    source=Path(p.source_file)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==p.source_sha256
    train=json.loads(source.read_text(encoding='utf-8'))['train']
    values=sorted(r['comparable_cost_cny'] for r in train.values())
    assert p.reference_amount==values[math.ceil(.95*len(values))-1]


def test_normal_complete_contracts_match_previous_commit_archive_exactly():
    import json
    from pathlib import Path
    from test_v4_reward_feedback import CASES,fixed_replay
    from v4.diagnostics import ledger_components
    archived=json.loads(Path('docs/results/v4_reward_feedback_implementation_20261008/reward_contract_verification.json').read_text())
    previous={r['case']:r for r in archived['cases']}
    for case in CASES:
        for feedback in (False,True):
            current,_=fixed_replay(case,feedback=feedback)
            expected=previous[case[0]]
            assert ledger_components(current.total_ledger)==expected['original_economic_ledger']
            assert ledger_components(current.modeled_terminal_settlement.ledger if current.modeled_terminal_settlement else None)==expected['modeled_terminal_ledger']
            assert fsum(t.reward_cny for t in current.transitions)==expected['new_reward_sum' if feedback else 'old_reward_sum']
            assert all(t.failure_penalty_equivalent_cny==0 for t in current.transitions)


def test_model_exception_named_no_feasible_does_not_fake_physical_failure():
    from v4.control import NoFeasibleFCActionError
    class BrokenAccountant(NoSolveAccountant):
        def interval(self,*args):
            raise NoFeasibleFCActionError('model exception, physical mask is nonempty')
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',),(100,),(0,)),lambda _s,_a:100,accountant=BrokenAccountant())
    assert caught.value.failure_kind=='execution_error'
    assert caught.value.failure_cause is None
    with pytest.raises(ValueError,match='no_feasible'):
        prepare(caught.value,NoSolveAccountant())


def test_nonfinite_online_model_is_program_error_not_supply_failure():
    agent=DirectPowerDDQN(seed=42)
    with torch.no_grad():
        next(agent.online.parameters()).fill_(float('nan'))
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',),(100,),(0,)),
            lambda state,mask:agent.select_power(state,mask),accountant=NoSolveAccountant())
    assert caught.value.failure_kind=='execution_error'
    assert not caught.value.executed_transitions


def test_greedy_failure_preview_is_readonly_and_not_successful_terminal(tmp_path):
    import json
    from types import SimpleNamespace
    from v4.failure_replay import FailurePenalty
    from v4.monitored_training import greedy_evaluate
    data=SimpleNamespace(sample_id='readonly_failure',split='train',operating_mode=('onboard',)*50,
                         load_kw=(1000.,)*50,battery_bus_kw=(0.,)*50)
    agent=DirectPowerDDQN(seed=42)
    saved=(agent.random.getstate(),agent.outcome_random.getstate(),torch.random.get_rng_state().clone())
    weights={key:t.clone() for key,t in agent.online.state_dict().items()}
    # Preserve select_power's RNG side effect while fixing only the test actor's power.
    original=agent.select_power
    def low_power(state,mask,**kwargs):
        original(state,mask,**kwargs)
        return min(mask)
    agent.select_power=low_power
    summary=greedy_evaluate((data,),agent,NoSolveAccountant(),500,redistribute_battery_energy=True,
        learn_no_feasible_failures=True,failure_penalty=FailurePenalty.from_reference(),
        trajectory_path=tmp_path/'readonly.json')
    profile=json.loads((tmp_path/'readonly.json').read_text())['failed'][0]
    terminal=profile['transitions'][-1]
    assert terminal['done'] and not terminal['successful_terminal']
    assert summary['completed']==summary['completed_voyages']==0
    assert summary['failure_terminal_count']==1
    assert summary['failure_reason_counts']['soc_limited']==1
    assert agent.random.getstate()==saved[0] and agent.outcome_random.getstate()==saved[1]
    assert torch.equal(torch.random.get_rng_state(),saved[2])
    assert all(torch.equal(t,agent.online.state_dict()[key]) for key,t in weights.items())
    assert not agent.replay and not agent.outcome_replay
    assert not agent.optimizer.state and agent.economic_optimizer_updates==0


def test_controlled_repeated_failure_td_changes_online_q_and_full_mask_greedy(tmp_path):
    import json
    from v4.experiment_schedule import EconomicUpdateSchedule
    corpus=[]
    safe_actions=[]
    for power in ACTION_KW:
        result,accountant=branch(power)
        if isinstance(result,ReplayExecutionError):
            values=prepare(result,accountant).transitions
            corpus.extend([values]*16)  # balanced stress fixture: repeat the4 failing alternatives
        else:
            values=result.transitions
            safe_actions.append(power)
            corpus.append(values)
    state=corpus[0][0].state
    mask=feasible_fc_actions(AccountState(soc=.2135),600,accountant)
    threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent=DirectPowerDDQN(seed=42)
        final_linear=[m for m in agent.online.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():
            final_linear.bias[0]+=1.  # intentional synthetic FC=0 starting preference, no architecture change
        agent.sync_target()
        initial_target_copies=agent.target_sync_calls
        outcome={k:t.clone() for k,t in agent.outcome_model.state_dict().items()}
        before=agent.online(torch.tensor([state],dtype=torch.float32)).detach()[0]
        assert agent.select_power(state,mask)==0
        schedule=EconomicUpdateSchedule('replay32',target_mode='optimizer',target_interval=500)
        for _ in range(82):
            count=0
            for values in corpus:
                count+=agent.remember_trajectory(values)
            schedule.grant(insertions=count,completed_episodes=0)
            schedule.consume(agent,batch_size=64)
        after=agent.online(torch.tensor([state],dtype=torch.float32)).detach()[0]
        greedy=agent.select_power(state,mask)
        assert greedy in safe_actions and greedy!=0
        actual_greedy=replay_episode(episode(('onboard',)*3,(600,1000,0),(0,)*3),
            lambda s,m:agent.select_power(s,m),accountant=accountant,
            initial_state=AccountState(soc=.2135),beta_soc=500)
        assert len(actual_greedy.transitions)==3
        assert (after[0]-after[60]).item() < (before[0]-before[60]).item()
        assert agent.economic_optimizer_updates==agent.economic_replay_insertions//32
        assert agent.target_sync_calls==initial_target_copies+agent.economic_optimizer_updates//500
        assert agent.outcome_optimizer_updates==0
        assert all(torch.equal(value,agent.outcome_model.state_dict()[name]) for name,value in outcome.items())
        assert agent.td_statistics()['failure_terminal']['sample_count']>0
        (tmp_path/'learning_order_evidence.json').write_text(json.dumps({
            'scope':'synthetic repeated economic replay only; no formal dataset, outcome updates or evaluation',
            'safe_actions_kw':safe_actions,'initial_greedy_kw':0,'final_greedy_kw':greedy,
            'before_q_values':before.tolist(),'after_q_values':after.tolist(),
            'actual_greedy_power_kw':[t.action_kw for t in actual_greedy.transitions],
            'economic_replay_insertions':agent.economic_replay_insertions,
            'successful_insertions':agent.economic_success_replay_insertions,
            'failed_insertions':agent.economic_failure_replay_insertions,
            'failure_terminal_insertions':agent.economic_failure_terminal_insertions,
            'optimizer_updates':agent.economic_optimizer_updates,'target_sync_calls':agent.target_sync_calls,
            'initial_target_copies':initial_target_copies,'remaining_credit':schedule.remaining_transition_credit,
            'td_statistics':agent.td_statistics(),
        },indent=2))
    finally:
        torch.set_num_threads(threads)


def test_failure_insertion_drives_replay32_budget_and_readonly_greedy(tmp_path):
    from types import SimpleNamespace
    from test_v4_training import FakeDataset
    from v4.monitored_training import run_monitored_training
    class FeasibleHighLoad(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id='synthetic_feasible_route',split='train',
                operating_mode=('onboard',)*69,load_kw=(1000.,)*67+(1800.,0.),battery_bus_kw=(0.,)*69),)
    agent,report=run_monitored_training(FeasibleHighLoad(),output_dir=tmp_path,rounds=1,
        beta_soc=500,cadence='replay32',target_mode='optimizer',target_interval=500,batch_size=64,
        n_step=1,redistribute_battery_energy=True,capture_trajectories=True)
    assert report['bootstrap']['completed']==1
    assert report['rounds'][0]['exploratory_train']['failed']
    counts=report['execution_counts']
    assert counts['economic_failure_replay_insertions']>0
    assert counts['economic_success_replay_insertions']==69
    assert counts['economic_failure_terminal_insertions']==1
    assert counts['failed_suffix_transitions_excluded_from_economic_replay']==0
    assert counts['economic_optimizer_updates']==counts['economic_replay_insertions']//32
    assert counts['target_sync_calls_including_initial_copy']==1+counts['economic_optimizer_updates']//500
    assert report['rounds'][0]['failure_reason_counts']['soc_limited']==1
    assert report['test_payloads_opened']==0
    import json
    profile=json.loads((tmp_path/'round_001_exploratory_trajectories.json').read_text())['failed'][0]
    assert profile['transitions'][-1]['terminal_reason']=='failure_soc_limited'
    assert profile['transitions'][-1]['failure_penalty_equivalent_cny']>0
