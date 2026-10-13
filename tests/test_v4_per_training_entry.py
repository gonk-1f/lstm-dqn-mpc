"""Replay entry contracts without dataset payloads, rollouts or optimizers."""
import json
from types import SimpleNamespace

import pytest


class _StopAtAgent(RuntimeError):
    pass


class _MetadataOnlyDataset:
    opened_test_payloads = 0

    def load_train(self):
        return (SimpleNamespace(sample_id='synthetic_train', split='train'),)

    def load_validation(self):
        return (SimpleNamespace(sample_id='synthetic_validation', split='validation'),)


def _capture_cli(tmp_path, monkeypatch, arguments):
    from v4 import feedback_study

    dataset = _MetadataOnlyDataset()
    captured = {}
    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open', lambda *roots: dataset)
    monkeypatch.setattr(feedback_study, '_default_data_root', lambda name: tmp_path/name)
    monkeypatch.setattr(feedback_study, '_manifest_hashes', lambda roots: {})

    def capture(_dataset, **kwargs):
        assert _dataset is dataset
        captured.update(kwargs)
        kwargs['output_dir'].mkdir()
        return None, {'rounds': [], 'best_checkpoint': None}

    monkeypatch.setattr(feedback_study, 'run_monitored_training', capture)
    assert feedback_study.main(['--output-dir', str(tmp_path/'cli'), *arguments]) == 0
    assert dataset.opened_test_payloads == 0
    return captured


def test_cli_defaults_to_uniform_and_preserves_frozen_training_configuration(tmp_path, monkeypatch):
    captured = _capture_cli(tmp_path, monkeypatch, [])
    assert captured['replay_sampling'] == 'uniform'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.6, 1., 1e-6)
    assert captured['rounds'] == 100 and captured['batch_size'] == 64
    assert captured['cadence'] == 'replay32' and captured['failure_terminal_quota'] == 2
    assert captured['seed'] == 42 and captured['n_step'] == 8
    assert captured['reward_scale'] == .001 and captured['beta_soc'] == 200
    assert captured['epsilon_start'] == 1. and captured['epsilon_end'] == .05
    assert captured['target_mode'] == 'soft' and captured['target_interval'] == 500
    assert captured['unknown_replay_mode'] == 'observed_nstep'
    assert captured['required_split_sizes'] == (30, 8)


def test_cli_routes_explicit_prioritized_replay_configuration(tmp_path, monkeypatch):
    captured = _capture_cli(tmp_path, monkeypatch, [
        '--replay-sampling', 'prioritized', '--per-alpha', '.7',
        '--per-beta', '.4', '--per-epsilon', '0.00001'])
    assert captured['replay_sampling'] == 'prioritized'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.7, .4, 1e-5)
    assert captured['rounds'] == 100 and captured['failure_terminal_quota'] == 2


def test_cli_reaches_agent_through_both_real_training_wrappers(tmp_path, monkeypatch):
    from v4 import feedback_study, monitored_training

    class FormalSizedMetadataDataset(_MetadataOnlyDataset):
        def load_train(self):
            return tuple(SimpleNamespace(sample_id=f'train_{index}', split='train') for index in range(30))

        def load_validation(self):
            return tuple(SimpleNamespace(sample_id=f'validation_{index}', split='validation') for index in range(8))

    captured = {}

    def stop_at_agent(**kwargs):
        captured.update(kwargs)
        raise _StopAtAgent('actual CLI route reached agent')

    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open',
                        lambda *roots: FormalSizedMetadataDataset())
    monkeypatch.setattr(feedback_study, '_default_data_root', lambda name: tmp_path/name)
    monkeypatch.setattr(feedback_study, '_manifest_hashes', lambda roots: {})
    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop_at_agent)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    with pytest.raises(_StopAtAgent, match='actual CLI route'):
        feedback_study.main(['--output-dir', str(tmp_path/'actual_cli'),
                             '--replay-sampling', 'prioritized'])
    assert captured['replay_sampling'] == 'prioritized'
    assert (captured['per_alpha'], captured['per_beta'], captured['per_epsilon']) == (.6, 1., 1e-6)
    metadata = json.loads((tmp_path/'actual_cli'/'run_metadata.json').read_text(encoding='utf-8'))
    assert metadata['hyperparameters']['replay_sampling'] == captured['replay_sampling']


_INVALID_REPLAY_ARGUMENTS = [
    ('--replay-sampling', 'invalid'),
    *[(flag, value) for flag in ('--per-alpha', '--per-beta')
      for value in ('-.1', '1.1', 'nan', 'inf')],
    *[('--per-epsilon', value) for value in ('0', '-.1', 'nan', 'inf')],
]


@pytest.mark.parametrize('arguments', _INVALID_REPLAY_ARGUMENTS)
def test_cli_rejects_invalid_replay_configuration_before_dataset_open(tmp_path, monkeypatch, arguments):
    from v4 import feedback_study

    monkeypatch.setattr(feedback_study.FormalTrainingDataset, 'open',
                        lambda *roots: pytest.fail('invalid replay configuration must precede dataset opening'))
    monkeypatch.setattr(feedback_study, '_default_data_root',
                        lambda name: pytest.fail('invalid replay configuration must precede data resolution'))
    output = tmp_path/'invalid'
    with pytest.raises(SystemExit) as error:
        feedback_study.main(['--output-dir', str(output), *arguments])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('configuration', [
    {'replay_sampling': 'invalid'},
    *[{name: value} for name in ('per_alpha', 'per_beta')
      for value in (-.1, 1.1, float('nan'), float('inf'))],
    *[{'per_epsilon': value} for value in (0., -.1, float('nan'), float('inf'))],
])
def test_public_wrapper_rejects_invalid_replay_configuration_before_data_or_output(tmp_path, configuration):
    from v4.monitored_training import run_monitored_training

    with pytest.raises(ValueError, match='replay_sampling|per_alpha|per_beta|per_epsilon'):
        run_monitored_training(object(), output_dir=tmp_path/'invalid', **configuration)
    assert not (tmp_path/'invalid').exists()


def _capture_agent_configuration(tmp_path, monkeypatch, configuration):
    from v4 import monitored_training

    captured = {}

    def stop_at_agent(**kwargs):
        captured.update(kwargs)
        raise _StopAtAgent('constructor reached through public and private wrappers')

    monkeypatch.setattr(monitored_training, 'DirectPowerDDQN', stop_at_agent)
    monkeypatch.setattr(monitored_training, 'write_learning_curves', lambda *args: {})
    with pytest.raises(_StopAtAgent, match='constructor reached'):
        monitored_training.run_monitored_training(
            _MetadataOnlyDataset(), output_dir=tmp_path, **configuration)
    metadata = json.loads((tmp_path/'run_metadata.json').read_text(encoding='utf-8'))
    report = json.loads((tmp_path/'report.json').read_text(encoding='utf-8'))
    assert report['hyperparameters'] == metadata['hyperparameters']
    assert report['test_payloads_opened'] == 0 and report['rounds'] == []
    return captured, metadata


@pytest.mark.parametrize('configuration, expected', [
    ({}, ('uniform', .6, 1., 1e-6)),
    ({'replay_sampling': 'prioritized'}, ('prioritized', .6, 1., 1e-6)),
    ({'replay_sampling': 'prioritized', 'per_alpha': .7, 'per_beta': .4, 'per_epsilon': 1e-5},
     ('prioritized', .7, .4, 1e-5)),
])
def test_public_wrapper_routes_actual_replay_configuration_to_agent_and_metadata(
        tmp_path, monkeypatch, configuration, expected):
    captured, metadata = _capture_agent_configuration(tmp_path, monkeypatch, configuration)
    names = ('replay_sampling', 'per_alpha', 'per_beta', 'per_epsilon')
    assert tuple(captured[name] for name in names) == expected
    assert tuple(metadata['hyperparameters'][name] for name in names) == expected
    assert metadata['hyperparameters']['batch_size'] == 64
    assert metadata['hyperparameters']['cadence'] == 'replay32'
    assert captured['failure_terminal_quota'] == 2 and captured['replay_capacity'] == 300_000
    assert captured['seed'] == 42 and captured['n_step'] == 8
    assert captured['reward_scale'] == .001 and captured['target_tau'] == .001


@pytest.mark.parametrize('mismatch', [
    'replay_sampling', 'per_alpha', 'per_beta', 'per_epsilon', 'old_configuration', 'source_tree_sha256'])
def test_resume_rejects_replay_or_source_mismatch_before_dataset_loading(tmp_path, monkeypatch, mismatch):
    from v4 import monitored_training

    configuration = {'replay_sampling': 'prioritized'}
    _, saved = _capture_agent_configuration(tmp_path, monkeypatch, configuration)
    if mismatch == 'old_configuration':
        for name in ('replay_sampling', 'per_alpha', 'per_beta', 'per_epsilon'):
            saved['hyperparameters'].pop(name)
    elif mismatch == 'source_tree_sha256':
        saved[mismatch] = 'previous_source_tree'
    else:
        saved['hyperparameters'][mismatch] = {
            'replay_sampling': 'uniform', 'per_alpha': .5,
            'per_beta': .5, 'per_epsilon': 1e-5}[mismatch]
    (tmp_path/'run_metadata.json').write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(monitored_training.torch, 'load',
                        lambda *args, **kwargs: {'format_version': 1, 'metadata': saved})

    class UnreadableDataset(_MetadataOnlyDataset):
        def load_train(self):
            pytest.fail('incompatible resume must precede dataset loading')

    # Preserve dataset identity so this assertion exercises the replay/source guard.
    dataset = _MetadataOnlyDataset()
    monkeypatch.setattr(dataset, 'load_train', UnreadableDataset().load_train)
    with pytest.raises(ValueError, match='resume source, architecture, hyperparameters'):
        monitored_training.run_monitored_training(
            dataset, output_dir=tmp_path, resume_from=tmp_path/'training_state_latest.pt', **configuration)
