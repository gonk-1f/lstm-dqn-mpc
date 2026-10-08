"""Entry contracts use synthetic rows only; no formal payload is opened."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from test_v4_training import FakeDataset


@pytest.mark.parametrize('scale', ['0', '-0.001', 'nan', 'inf'])
def test_invalid_reward_scale_is_rejected_before_dataset_open(tmp_path, monkeypatch, capsys, scale):
    from v4 import feedback_study
    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open',
        lambda *args: pytest.fail('invalid scale must precede dataset opening'))
    with pytest.raises(SystemExit):
        feedback_study.main(['--output-dir', str(tmp_path/'invalid'), '--rounds', '1',
                             '--reward-scale', scale])
    assert 'reward-scale must be finite and positive' in capsys.readouterr().err
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('n_step', [1, 8])
def test_monitored_scale_only_changes_replay_rewards_and_metadata(tmp_path, n_step):
    from v4.monitored_training import run_monitored_training
    agents, reports = [], []
    for scale in (1., .001):
        output = tmp_path/str(scale)
        class ProvenanceDataset(FakeDataset):
            def load_train(self):
                metadata = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
                assert metadata['hyperparameters']['reward_scale'] == scale
                assert metadata['manifest_sha256'] == {'synthetic_manifest': 'abc'}
                assert (output/'report.json').exists()
                return super().load_train()
        agent, report = run_monitored_training(ProvenanceDataset(), output_dir=output,
            rounds=1, beta_soc=500, cadence='replay32', target_mode='optimizer',
            target_interval=500, seed=42, batch_size=64, n_step=n_step,
            redistribute_battery_energy=True, reward_scale=scale,
            manifest_sha256={'synthetic_manifest': 'abc'})
        agents.append(agent); reports.append(report)
        assert report['hyperparameters']['reward_scale'] == scale
        assert report['reward_units']['economic_q'] == 'scaled_reward_equivalent_cny'
        assert report['manifest_sha256'] == {'synthetic_manifest': 'abc'}
        assert report['execution_counts']['economic_optimizer_updates'] == 0
        assert report['completed_training'] and report['run_status'] == 'completed'
        checkpoint = torch.load(output/'best_agent.pt', weights_only=True)
        assert checkpoint['reward_scale'] == scale
        assert checkpoint['manifest_sha256'] == report['manifest_sha256']
        assert checkpoint['source_commit'] == report['source_commit']
        assert (output/'round_metrics.csv').exists()
    assert [t.reward_cny for t in agents[1].replay] == pytest.approx(
        [.001*t.reward_cny for t in agents[0].replay])
    for left, right in zip(agents[0].replay, agents[1].replay):
        assert (left.state, left.next_state, left.done) == (right.state, right.next_state, right.done)
    assert reports[0]['rounds'][0]['greedy_train'] == reports[1]['rounds'][0]['greedy_train']
    for name in ('online', 'target', 'outcome_model'):
        assert all(torch.equal(value, getattr(agents[1], name).state_dict()[key])
            for key, value in getattr(agents[0], name).state_dict().items())


def test_aborted_training_preserves_completed_round_without_resuming(tmp_path, monkeypatch):
    from v4 import monitored_training
    original = monitored_training.replay_episode
    calls = 0
    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic computation exception')
        return original(*args, **kwargs)
    monkeypatch.setattr(monitored_training, 'replay_episode', interrupted)
    with pytest.raises(RuntimeError, match='synthetic computation exception'):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path,
            rounds=2, beta_soc=500, cadence='replay32', target_mode='optimizer',
            target_interval=500, reward_scale=.001)
    report = json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    history = json.loads((tmp_path/'round_history.json').read_text(encoding='utf-8'))
    assert report['run_status'] == 'aborted' and not report['completed_training']
    assert report['abort']['exception_type'] == 'RuntimeError'
    assert report['abort']['message'] == 'synthetic computation exception'
    assert len(report['rounds']) == len(history['rounds']) == 1
    assert report['best_checkpoint']['round'] == 1
    assert report['hyperparameters']['reward_scale'] == .001
    assert len(list(csv.DictReader((tmp_path/'round_metrics.csv').open(encoding='utf-8')))) == 1
    assert not (tmp_path/'best_agent.pt.tmp').exists()
    with pytest.raises(FileExistsError):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path,
            rounds=1, beta_soc=500, reward_scale=.001)


def test_formal_cli_scale_metadata_logging_and_fixed_configuration_with_synthetic_rows(tmp_path, monkeypatch):
    from v4 import feedback_study
    class FixedSynthetic(FakeDataset):
        def load_train(self):
            return tuple(SimpleNamespace(sample_id=f't_{i}', split='train',
                operating_mode=('onboard',)*4, load_kw=(0., 100., 100., 0.),
                battery_bus_kw=(0.,)*4) for i in range(30))
        def load_validation(self):
            return tuple(SimpleNamespace(sample_id=f'v_{i}', split='validation',
                operating_mode=('onboard',)*3, load_kw=(0., 80., 0.),
                battery_bus_kw=(0.,)*3) for i in range(8))
    output = tmp_path/'scaled_single_run'
    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open', lambda *args: FixedSynthetic())
    monkeypatch.setattr(feedback_study, '_default_data_root', lambda name: tmp_path/name)
    assert feedback_study.main(['--output-dir', str(output), '--rounds', '1',
        '--reward-scale', '0.001', '--reward-feedback', 'redistributed', '--beta-soc', '500',
        '--failure-penalty-scale', '1', '--cadence', 'replay32', '--target-interval', '500',
        '--n-step', '1']) == 0
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    hp = report['hyperparameters']
    assert (hp['reward_scale'], hp['learning_rate'], hp['gamma'], hp['batch_size']) == (.001, .0001, 1., 64)
    assert hp['hidden_dims'] == [128, 64] and hp['replay_capacity'] == 100000
    assert hp['epsilon_start'] == 1 and hp['epsilon_end'] == .05
    assert report['execution_counts']['economic_optimizer_updates'] == 7
    assert report['test_payloads_opened'] == 0 and report['dataset_manifests_unchanged']
    assert report['manifest_sha256'] == {}
    assert 'MONITOR round=1' in (output/'train.log').read_text(encoding='utf-8')
    assert 'progress step=50' in (output/'train.log').read_text(encoding='utf-8')
    checkpoint = torch.load(output/'best_agent.pt', weights_only=True)
    assert checkpoint['reward_scale'] == .001 and checkpoint['manifest_sha256'] == {}
    row = next(csv.DictReader((output/'round_metrics.csv').open(encoding='utf-8')))
    assert row['reward_scale'] == '0.001'
    assert float(row['economic_optimizer_updates']) == 7
    assert 'td_error_mae_original_reward_units' in row


def test_strict_entry_aborts_model_error_instead_of_creating_physical_failure(tmp_path, monkeypatch):
    from v4 import monitored_training
    from v4.control import ReplayExecutionError
    from v2.data.supervisory_rules import OperatingMode
    def model_error(*args, **kwargs):
        raise ReplayExecutionError(0, OperatingMode.ONBOARD, ValueError('synthetic model error'))
    monkeypatch.setattr(monitored_training, 'replay_episode', model_error)
    with pytest.raises(ReplayExecutionError, match='synthetic model error'):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path,
            rounds=1, beta_soc=500, reward_scale=.001, cadence='replay32',
            target_mode='optimizer', target_interval=500, abort_on_execution_error=True)
    report = json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    assert report['run_status'] == 'aborted' and report['rounds'] == []
    assert report['abort']['phase'] == 'exploratory_training'
    assert report['abort']['actual_partial_execution_counts']['economic_replay_insertions'] == 1
    assert report['best_checkpoint'] is None and not (tmp_path/'best_agent.pt').exists()


def test_bootstrap_exception_keeps_initial_provenance_and_log(tmp_path, monkeypatch):
    from v4 import monitored_training
    def fail_bootstrap(*args, **kwargs):
        raise RuntimeError('synthetic bootstrap failure')
    monkeypatch.setattr(monitored_training, '_run_episodes', fail_bootstrap)
    with pytest.raises(RuntimeError, match='synthetic bootstrap failure'):
        monitored_training.run_monitored_training(FakeDataset(), output_dir=tmp_path,
            rounds=1, beta_soc=500, reward_scale=.001, capture_log=True,
            manifest_sha256={'fake_data_manifest':'frozen'})
    report = json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    assert report['run_status'] == 'aborted' and report['rounds'] == []
    assert report['abort']['phase'] == 'bootstrap'
    assert report['manifest_sha256'] == {'fake_data_manifest':'frozen'}
    assert report['hyperparameters']['reward_scale'] == .001
    assert 'synthetic bootstrap failure' in (tmp_path/'train.log').read_text(encoding='utf-8')
    assert (tmp_path/'round_metrics.csv').exists()


def test_post_training_manifest_guard_marks_report_aborted_and_preserves_evidence(tmp_path, monkeypatch):
    from v4 import feedback_study
    class OneStepFormalFake(FakeDataset):
        def load_train(self):
            return tuple(SimpleNamespace(sample_id=f't_{i}', split='train',
                operating_mode=('onboard',), load_kw=(100.,), battery_bus_kw=(0.,))
                for i in range(30))
        def load_validation(self):
            return tuple(SimpleNamespace(sample_id=f'v_{i}', split='validation',
                operating_mode=('onboard',), load_kw=(80.,), battery_bus_kw=(0.,))
                for i in range(8))
    hashes = iter(({'fake_manifest':'before'}, {'fake_manifest':'after'}))
    output = tmp_path/'guarded'
    monkeypatch.setattr(feedback_study, '_manifest_hashes', lambda roots: next(hashes))
    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open', lambda *args: OneStepFormalFake())
    monkeypatch.setattr(feedback_study, '_default_data_root', lambda name: tmp_path/name)
    with pytest.raises(RuntimeError, match='dataset manifests changed'):
        feedback_study.main(['--output-dir', str(output), '--rounds', '1', '--reward-scale', '.001'])
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    history = json.loads((output/'round_history.json').read_text(encoding='utf-8'))
    assert report['run_status'] == history['run_status'] == 'aborted'
    assert not report['completed_training'] and not report['dataset_manifests_unchanged']
    assert report['checkpoint_selection_valid'] is False
    assert report['manifest_sha256'] == {'fake_manifest':'before'}
    assert report['final_manifest_sha256'] == {'fake_manifest':'after'}
    assert len(report['rounds']) == 1 and (output/'best_agent.pt').exists()
    assert report['best_checkpoint']['selection_valid'] is False
    assert 'dataset manifests changed' in (output/'train.log').read_text(encoding='utf-8')
