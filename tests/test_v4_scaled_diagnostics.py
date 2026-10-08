"""Synthetic monitoring reports distinguish training units from economic CNY."""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def transition(*, soc=.5, before=.55, fc=200, prior_fc=100, reward=-20,
               done=False, reason=None):
    ledger = SimpleNamespace(h2_cost_cny=10., fuel_cell_degradation_cost_cny=2.,
                             battery_degradation_cost_cny=3., shore_cost_cny=0.)
    return SimpleNamespace(state=(before, .5, 0., 0., prior_fc/600, 0., 0., 0.),
        actual_soc=soc, action_kw=fc, actual_battery_kw=300-fc,
        reward_cny=float(reward), original_reward=float(reward)-1.,
        immediate_battery_energy_adjustment=-2., terminal_correction=3.,
        soc_soft_penalty_cny=5., original_economic_ledger=ledger, shore_ledger=None,
        modeled_terminal_ledger=None, done=done, next_feasible_actions=() if done else (0,600),
        terminal_reason=reason, is_successful_terminal=done and reason in (None,'completed'),
        failure_penalty_equivalent_cny=10132. if reason else 0.)


def test_trajectory_scale_labels_leave_original_rewards_and_ledgers_untouched():
    from v4.diagnostics import trajectory_record
    values = (transition(), transition(reward=-10000,done=True,reason='failure_soc_limited'))
    raw = trajectory_record('synthetic',values,completed=False)
    scaled = trajectory_record('synthetic',iter(values),completed=False,reward_scale=.001)
    assert scaled['reward_scale'] == .001
    assert scaled['training_reward_unit'] == 'scaled reward-equivalent CNY'
    assert scaled['economic_ledger_unit'] == 'CNY'
    assert scaled['reward_feedback_totals'] == raw['reward_feedback_totals']
    for actual, expected in zip(scaled['transitions'], raw['transitions']):
        assert actual['scaled_training_reward'] == pytest.approx(.001*expected['new_reward'])
        assert actual['new_reward'] == expected['new_reward']
        assert actual['original_economic_ledger'] == expected['original_economic_ledger']
        assert actual['failure_penalty_equivalent_cny'] == expected['failure_penalty_equivalent_cny']
    assert scaled['scaled_training_reward_total'] == pytest.approx(-10.02)


@pytest.mark.parametrize('scale', [0., -1., float('nan'),float('inf')])
def test_trajectory_diagnostics_reject_invalid_scale(scale):
    from v4.diagnostics import trajectory_record
    with pytest.raises(ValueError):
        trajectory_record('invalid',(),completed=False,reward_scale=scale)


def test_executed_power_statistics_include_failed_prefix_and_use_observed_previous_fc():
    from v4.diagnostics import executed_transition_statistics
    # Two voyages reset priorFC to zero: cannot subtract the previous voyage's600.
    values = (transition(soc=.3,fc=600,prior_fc=0,done=True),
              transition(soc=.79,fc=200,prior_fc=0),
              transition(soc=.5,fc=0,prior_fc=200,done=True,reason='failure_soc_limited'))
    stats = executed_transition_statistics(iter(values))
    assert stats['onboard_soc_max_including_failed_prefix'] == .79
    power = stats['fc_power_statistics_kw']
    assert power['sample_count'] == 3
    assert power['mean'] == pytest.approx(800/3)
    assert power['minimum'] == 0 and power['maximum'] == 600
    assert len(power['distribution_step_counts']) == 61
    assert power['distribution_step_counts']['0'] == 1
    assert sum(power['distribution_step_counts'].values()) == 3
    change = stats['fc_change_statistics_kw']
    assert change['mean_signed'] == pytest.approx(600/3)
    assert change['mean_absolute'] == pytest.approx(1000/3)
    assert change['max_absolute'] == 600
    empty = executed_transition_statistics(())
    assert empty['onboard_soc_max_including_failed_prefix'] is None
    assert empty['fc_power_statistics_kw']['mean'] is None


@pytest.mark.parametrize('steady_kw',[110,220,230,310,440,460])
def test_fc_change_statistics_do_not_count_state_normalization_roundoff_as_change(steady_kw):
    from v4.diagnostics import executed_transition_statistics
    result=executed_transition_statistics((transition(fc=steady_kw,prior_fc=steady_kw),))
    change=result['fc_change_statistics_kw']
    assert change['nonzero_fraction']==0
    assert change['mean_absolute']==0
    assert change['max_absolute']==0


def test_fixed_q_probe_converts_units_without_sampling_or_mutating_training_state():
    import torch
    from v3.control import EconomicMPC
    from v4.dqn import DirectPowerDDQN
    from v4.diagnostics import fixed_q_diagnostics
    agent = DirectPowerDDQN(seed=42,reward_scale=.001)
    weights = {name:value.clone() for name,value in agent.online.state_dict().items()}
    rng = (agent.random.getstate(),agent.outcome_random.getstate(),torch.random.get_rng_state().clone())
    rows = fixed_q_diagnostics(agent,EconomicMPC(nominal_cost_cny=1.))
    assert len(rows) == 3
    for row in rows:
        assert row['reward_scale'] == .001
        assert row['q_unit'] == 'scaled reward-equivalent CNY'
        assert len(row['q_values_original_reward_units']) == 61
        assert row['q_values_original_reward_units'] == pytest.approx([x/.001 for x in row['q_values']])
        assert row['fc_zero_q_original_reward_units'] == pytest.approx(row['fc_zero_q']/.001)
    assert agent.random.getstate() == rng[0] and agent.outcome_random.getstate() == rng[1]
    assert torch.equal(torch.random.get_rng_state(),rng[2])
    assert all(torch.equal(value,weights[name]) for name,value in agent.online.state_dict().items())
    assert not agent.replay and not agent.economic_optimizer_updates


def summary(*, completed=2, cost=100.):
    return {'episodes':2,'completed':completed,'cost_cny':cost if completed==2 else None,
        'completed_observed_cost_cny':80.,'completed_modeled_terminal_cost_cny':20.,
        'onboard_soc_mean':.5,'onboard_soc_min_including_failed_prefix':.3,
        'onboard_soc_max_including_failed_prefix':.79,'terminal_onboard_soc_mean':.55,
        'fc_zero_fraction':.2,'soc_time_occupancy':{'fractions':{
            'below_0p4':.1,'0p4_to_0p6':.7,'above_0p6_below_0p79':.1,'at_least_0p79':.1}}}


def synthetic_report():
    return {'hyperparameters':{'reward_scale':.001,'seed':42},'source_commit':'synthetic',
        'test_payloads_opened':0,'execution_counts':{'economic_optimizer_updates':3},
        'rounds':[{'round':i,'epsilon':1. if i==1 else .05,
            'exploratory_train':summary(completed=1), 'greedy_train':summary(completed=1 if i==1 else 2),
            'greedy_validation':None if i==1 else summary(completed=1),
            'qualified_checkpoint':False,'economic_optimizer_updates_cumulative':i,
            'economic_replay_insertions_cumulative':32*i,'training_environment_transitions_cumulative':32*i,
            'target_sync_calls_cumulative':1,'mean_loss':.4,'td_statistics':{
                'reward_scale':.001,'mean_absolute_td_error':.2,
                'original_units':{'mean_absolute_td_error':200.},
                'failure_terminal':{'mean_absolute_td_error':.3,
                                    'original_units':{'mean_absolute_td_error':300.}},
                'gradient_statistics':{'clipped_fraction':.25}}} for i in (1,2)]}


def test_learning_curves_mask_skipped_and_incomplete_validation_and_label_units(tmp_path):
    from v4.learning_curves import write_learning_curves
    report = synthetic_report()
    before = deepcopy(report)
    result = write_learning_curves(report,tmp_path/'synthetic-curves')
    assert report == before
    assert result['reward_scale'] == .001
    assert result['round_count'] == 2
    assert result['series']['greedy_validation_completed'] == [None,1]
    assert result['series']['validation_comparable_cost_cny'] == [None,None]
    assert result['series']['validation_observed_cost_cny'] == [None,None]
    assert result['series']['td_mae_scaled_reward_units'] == [.2,.2]
    assert result['series']['td_mae_original_reward_units'] == pytest.approx([200,200])
    assert result['series']['failure_td_mae_original_reward_units'] == pytest.approx([300,300])
    assert result['series']['gradient_clipping_fraction'] == [.25,.25]
    assert result['units']['economic_cost'] == 'CNY'
    assert len(result['figures']) >= 5
    for name in result['figures']:
        path = tmp_path/'synthetic-curves'/name
        assert path.is_file() and path.stat().st_size > 1000
    persisted = json.loads((tmp_path/'synthetic-curves'/'learning_curves_metadata.json').read_text(encoding='utf-8'))
    assert persisted == result


def test_learning_curves_preserve_original_economic_cost_when_validation_completes(tmp_path):
    from v4.learning_curves import write_learning_curves
    report = synthetic_report()
    report['rounds'][1]['greedy_validation'] = summary(cost=321.)
    result = write_learning_curves(report,tmp_path/'completed')
    assert result['series']['validation_comparable_cost_cny'] == [None,321.]
    assert result['series']['validation_observed_cost_cny'] == [None,80.]
    assert result['series']['validation_modeled_cost_cny'] == [None,20.]


def test_learning_curves_refuse_archives_and_existing_artifacts(tmp_path):
    from v4.learning_curves import write_learning_curves
    archive = tmp_path/'docs'/'results'/'prior-experiment'
    with pytest.raises(ValueError,match='archive'):
        write_learning_curves(synthetic_report(),archive)
    assert not archive.exists()
    destination=tmp_path/'existing'
    destination.mkdir()
    existing=destination/'learning_curves_metadata.json'
    existing.write_text('preserve old artifact',encoding='utf-8')
    with pytest.raises(FileExistsError):
        write_learning_curves(synthetic_report(),destination)
    assert existing.read_text(encoding='utf-8') == 'preserve old artifact'


def test_learning_curves_handle_abort_before_any_complete_round(tmp_path):
    from v4.learning_curves import write_learning_curves
    report=synthetic_report()
    report['rounds']=[]
    result=write_learning_curves(report,tmp_path/'zero-round-abort')
    assert result['round_count']==0
    assert result['series']['round']==[]


def test_representative_curves_read_only_captured_onboard_steps_with_no_modeled_extension(tmp_path):
    from v4.learning_curves import write_learning_curves
    report=synthetic_report()
    source=tmp_path/'round_002_train_trajectories.json'
    profile={'sample_id':'captured-synthetic','transitions':[
        {'voyage_index':0,'soc_before':.55,'soc_after':.54,'fc_kw':100,
         'battery_bus_kw':200,'load_kw':300},
        {'voyage_index':1,'soc_before':.6,'soc_after':.6,'fc_kw':300,
         'battery_bus_kw':0,'load_kw':300}]}
    source.write_text(json.dumps({'completed':[profile],'failed':[]}),encoding='utf-8')
    original=source.read_bytes()
    report['rounds'][1]['greedy_train']['trajectory_file']=source.name
    result=write_learning_curves(report,tmp_path)
    assert source.read_bytes()==original
    selected=result['representative_trajectories']
    assert len(selected)==1 and selected[0]['outcome']=='completed'
    assert selected[0]['source']==source.name
    assert (tmp_path/selected[0]['figure']).is_file()
    assert 'no appended modeled charging' in result['representative_scope']


def test_representative_curves_reject_trajectory_path_outside_current_run(tmp_path):
    from v4.learning_curves import write_learning_curves
    report=synthetic_report()
    report['rounds'][1]['greedy_train']['trajectory_file']='../outside.json'
    with pytest.raises(ValueError,match='this output directory'):
        write_learning_curves(report,tmp_path/'new-output')
    assert not (tmp_path/'new-output').exists()


@pytest.mark.skipif(os.name!='nt',reason='Windows MAX_PATH regression')
def test_representative_curve_saves_long_windows_image_path(tmp_path):
    from v4.learning_curves import write_learning_curves
    # Keep the directory/JSON names under260; the representative PNG exceeds it.
    suffix_length=max(1,210-len(str(tmp_path.resolve()))-1)
    output=tmp_path/('long-output-'+ 'x'*suffix_length)
    output.mkdir()
    assert len(str(output.resolve()))<233
    profile={'sample_id':'long-path','transitions':[
        {'voyage_index':0,'soc_before':.55,'soc_after':.54,'fc_kw':100,
         'battery_bus_kw':200,'load_kw':300}]}
    source=output/'trajectories.json'
    source.write_text(json.dumps({'completed':[profile],'failed':[]}),encoding='utf-8')
    report=synthetic_report()
    report['rounds'][1]['greedy_validation']=summary()
    report['rounds'][1]['greedy_validation']['trajectory_file']=source.name
    expected=output/'final_round_002_greedy_validation_completed_trajectory.png'
    assert len(str(expected.resolve()))>=260
    result=write_learning_curves(report,output)
    assert expected.name in result['figures']
    # Windows stat also needs the prefix when system long-path support is off.
    extended=Path('\\\\?\\'+str(expected.resolve()))
    assert extended.is_file() and extended.stat().st_size>1000
    assert extended.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
