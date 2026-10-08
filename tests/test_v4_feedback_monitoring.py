import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_v4_training import FakeDataset
from v3.control import EconomicMPC
from v4.dqn import DirectPowerDDQN
from v4.monitored_training import greedy_evaluate, run_monitored_training


class SizedSyntheticDataset(FakeDataset):
    def load_train(self):
        return tuple(SimpleNamespace(sample_id=f't_{i}',split='train',
            operating_mode=('onboard',)*4, load_kw=(0.,100.,100.,0.),
            battery_bus_kw=(0.,)*4) for i in range(30))

    def load_validation(self):
        return tuple(SimpleNamespace(sample_id=f'v_{i}',split='validation',
            operating_mode=('onboard',)*3, load_kw=(0.,80.,0.),
            battery_bus_kw=(0.,)*3) for i in range(8))


@pytest.mark.parametrize('cadence,stride,rounds', [('episode16',None,1),('replay32',32,2),('replay16',16,2)])
def test_synthetic_schedules_feedback_diagnostics_and_checkpoint_gate(tmp_path,cadence,stride,rounds):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent, report = run_monitored_training(SizedSyntheticDataset(), output_dir=tmp_path,
            rounds=rounds, beta_soc=500, cadence=cadence,target_mode='optimizer',target_interval=250,
            redistribute_battery_energy=True,n_step=1,episode_credit_scope='voyage',
            required_split_sizes=(30,8),capture_trajectories=True)
    finally:
        torch.set_num_threads(threads)
    insertions = 120*(1+rounds)
    updates = 16*30*(1+rounds) if stride is None else insertions//stride
    assert agent.economic_replay_insertions == insertions
    assert agent.economic_optimizer_updates == updates
    assert report['bootstrap_optimizer_updates'] == (480 if stride is None else 120//stride)
    assert report['formal_training_optimizer_updates'] == updates-report['bootstrap_optimizer_updates']
    assert agent.target_sync_calls == 1+updates//250
    assert report['execution_counts']['remaining_transition_credit'] == (0 if stride is None else insertions%stride)
    assert report['test_payloads_opened'] == 0
    for row in report['rounds']:
        assert row['greedy_train']['completed'] == 30 and row['greedy_validation']['completed'] == 8
        assert len(row['fixed_state_q_diagnostics']) == 3
        assert all(len(d['q_values']) == 61 and len(d['action_order_kw']) == 61
                   for d in row['fixed_state_q_diagnostics'])
        assert row['td_statistics']['optimizer_updates'] == row['economic_optimizer_updates']
        totals = row['greedy_train']['completed_reward_feedback']
        assert totals['new_reward'] == pytest.approx(totals['original_reward'],rel=1e-12,abs=1e-8)
        assert sum(row['greedy_validation']['completed_observed_components_cny'].values()) == pytest.approx(
            row['greedy_validation']['completed_observed_cost_cny'])
        trajectory = json.loads((tmp_path/row['greedy_train']['trajectory_file']).read_text())
        assert len(trajectory['completed']) == 30
        for frame in trajectory['completed'][0]['transitions']:
            assert all(key in frame for key in ('original_reward','immediate_battery_energy_adjustment',
                'terminal_correction','new_reward','original_economic_ledger','fc_kw','battery_bus_kw','soc_after'))
    selection = torch.load(tmp_path/'best_agent.pt',weights_only=True)['selection']
    assert selection['train']['completed'] == 30 and selection['validation']['completed'] == 8


def test_feedback_greedy_preserves_every_training_state_and_stream(tmp_path):
    agent = DirectPowerDDQN(seed=42,n_step=8)
    snapshots = {name:{k:t.clone() for k,t in getattr(agent,name).state_dict().items()}
                 for name in ('online','target','outcome_model')}
    rng = (agent.random.getstate(),agent.outcome_random.getstate(),torch.random.get_rng_state().clone())
    result = greedy_evaluate(SizedSyntheticDataset().load_train(),agent,
        EconomicMPC(nominal_cost_cny=1),500,redistribute_battery_energy=True,
        trajectory_path=tmp_path/'greedy.json')
    assert result['completed'] == 30
    assert agent.random.getstate() == rng[0] and agent.outcome_random.getstate() == rng[1]
    assert torch.equal(torch.random.get_rng_state(),rng[2])
    assert agent.economic_optimizer_updates == agent.outcome_optimizer_updates == 0
    assert not agent.replay and not agent.outcome_replay
    assert not agent.optimizer.state and not agent.outcome_optimizer.state
    for name, state in snapshots.items():
        assert all(torch.equal(t,getattr(agent,name).state_dict()[k]) for k,t in state.items())


def test_new_episode_credit_counts_completed_voyages_not_sample_containers(tmp_path):
    class TwoVoyages(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id='two',split='train',
                operating_mode=('onboard','shore_charging','onboard'),
                load_kw=(0.,0.,0.),battery_bus_kw=(0.,-10.,0.)),)
    agent, report = run_monitored_training(TwoVoyages(),output_dir=tmp_path,rounds=1,beta_soc=500,
        batch_size=1,target_mode='optimizer',redistribute_battery_energy=True,episode_credit_scope='voyage',n_step=8)
    assert report['bootstrap_optimizer_updates'] == 32
    assert agent.economic_optimizer_updates == 64
    assert report['rounds'][0]['completed_voyages_entering_economic_replay'] == 2


def test_feedback_entry_requires_explicit_small_run_and_rejects_archive_paths(tmp_path,monkeypatch):
    from v4 import feedback_study
    with pytest.raises(SystemExit):
        feedback_study.main([])
    monkeypatch.setattr(feedback_study.FormalTrainingDataset,'open',
                        lambda *a: (_ for _ in ()).throw(AssertionError('archive check must precede data loading')))
    with pytest.raises(ValueError,match='archive'):
        feedback_study.main(['--output-dir',str(Path('docs/results/v4_staged_beta_cadence_seed42')),
                             '--rounds','1'])


def test_diagnostic_probe_neither_samples_actions_nor_changes_model():
    from v4.diagnostics import fixed_q_diagnostics
    agent = DirectPowerDDQN(seed=42)
    saved = (agent.random.getstate(),torch.random.get_rng_state().clone())
    probes = fixed_q_diagnostics(agent,EconomicMPC(nominal_cost_cny=1))
    assert len(probes) == 3 and all(len(p['q_values'])==61 for p in probes)
    assert agent.random.getstate()==saved[0] and torch.equal(torch.random.get_rng_state(),saved[1])
    assert all(p['greedy_feasible_action_kw'] in p['feasible_actions_kw'] for p in probes)


def test_single_run_cli_end_to_end_with_synthetic_data_and_n_step8(tmp_path,monkeypatch):
    from v4 import feedback_study
    dataset = SizedSyntheticDataset()
    monkeypatch.setattr(feedback_study.FormalTrainingDataset,'open',lambda *a:dataset)
    monkeypatch.setattr(feedback_study,'_default_data_root',lambda name:tmp_path/name)
    assert feedback_study.main(['--output-dir',str(tmp_path/'new_run'),'--rounds','1',
                                '--n-step','8','--cadence','replay32']) == 0
    report = json.loads((tmp_path/'new_run/report.json').read_text())
    assert report['hyperparameters']['redistribute_battery_energy']
    assert report['hyperparameters']['target_interval_optimizer_updates'] == 500
    assert report['hyperparameters']['n_step'] == 8
    assert report['test_payloads_opened'] == 0 and report['dataset_manifests_unchanged']
    assert report['execution_counts']['economic_optimizer_updates'] == 7
    assert (tmp_path/'new_run/round_metrics.csv').exists()


def test_failed_greedy_trajectory_retains_sample_identity_and_uncorrected_suffix(tmp_path,monkeypatch):
    data = SimpleNamespace(sample_id='failed_actual_prefix',split='train',
        operating_mode=('onboard',)*50,load_kw=(1000.,)*50,battery_bus_kw=(0.,)*50)
    agent = DirectPowerDDQN(seed=42)
    monkeypatch.setattr(agent,'select_power',lambda _s,feasible,**k:min(feasible))
    summary = greedy_evaluate((data,),agent,EconomicMPC(nominal_cost_cny=1),500,
        redistribute_battery_energy=True,trajectory_path=tmp_path/'failed.json')
    profile = json.loads((tmp_path/'failed.json').read_text())['failed'][0]
    assert summary['completed'] == 0 and summary['cost_cny'] is None
    assert profile['sample_id'] == data.sample_id
    assert profile['transitions'] and not profile['transitions'][-1]['done']
    assert profile['transitions'][-1]['next_feasible_actions_kw'] == []
    assert all(frame['terminal_correction']==0 for frame in profile['transitions'])
    assert not agent.replay and not agent.outcome_replay


def test_historical_resume_cannot_rewrite_archived_summaries(tmp_path,monkeypatch):
    from v4 import staged_study
    archived = tmp_path/'v4_staged_beta_cadence_seed42'
    archived.mkdir()
    summary = archived/'stage1_summary.json'
    summary.write_text('{"preserve":true}')
    monkeypatch.setattr(staged_study,'_parallel_jobs',
                        lambda *a: (_ for _ in ()).throw(AssertionError('archive guard must run first')))
    with pytest.raises(ValueError,match='archive'):
        staged_study.main(['--output-dir',str(archived),'--resume'])
    assert summary.read_text() == '{"preserve":true}'


def test_other_checkout_archive_paths_are_protected(tmp_path):
    from v4.feedback_study import fresh_output_path
    with pytest.raises(ValueError,match='archive'):
        fresh_output_path(tmp_path/'sibling_checkout/docs/results/old_run/new_run')


def test_first_step_infeasible_after_shore_does_not_mislabel_completed_prefix(tmp_path):
    class InfeasibleDeparture(FakeDataset):
        def load_train(self):
            return (SimpleNamespace(sample_id='failed_departure',split='train',
                operating_mode=('onboard','shore_charging','onboard'),
                load_kw=(100.,0.,2000.),battery_bus_kw=(0.,0.,0.)),)
    agent, report = run_monitored_training(InfeasibleDeparture(),output_dir=tmp_path,rounds=1,
        beta_soc=500,batch_size=1,target_mode='optimizer',redistribute_battery_energy=True,
        episode_credit_scope='voyage',n_step=8)
    assert report['bootstrap']['failed'] and report['rounds'][0]['exploratory_train']['failed']
    assert len(agent.replay) == 2 and all(item.done for item in agent.replay)
    assert agent.economic_optimizer_updates == 32
    assert len(agent.outcome_replay) == 2 and not any(item.failed for item in agent.outcome_replay)
    assert report['execution_counts']['no_feasible_action_events_without_executed_suffix'] == 2
    assert report['execution_counts']['failed_suffix_transitions_excluded_from_economic_replay'] == 0
