"""Real SQLite/sampler lifecycle with mocked training and synthetic inputs only."""
import csv
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import optuna
import pandas as pd
import pytest

from v6_sb3 import tuning
from v6_sb3.identity import dataset_identity


@pytest.fixture
def synthetic_study(tmp_path, monkeypatch):
    root = tmp_path/'synthetic_repository'
    (root/'src').mkdir(parents=True)
    (root/'src/model.py').write_text('SYNTHETIC_SOURCE = 1\n', encoding='utf-8')
    monkeypatch.setattr(tuning, 'ROOT', root)

    def reject_real_dataset(*args, **kwargs):
        pytest.fail('Smoke must never open a real dataset')

    monkeypatch.setattr(tuning.V6TrainingDataset, 'open', reject_real_dataset)
    return dict(trials=1, train_steps=2, learning_starts=2, eval_every=2, smoke=True, pruner='none')


def fake_training(monkeypatch, *, pause_first=False, fail=False):
    calls = []

    def run(trial, config, train, validation, directory, *, resume, **kwargs):
        assert all(item.sample_id.startswith('synthetic_') for item in (*train, *validation))
        calls.append(dict(number=trial.number, parameters=config['parameters'], resume=resume))
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        tuning.write_json(directory/'configuration.json', config)
        marker = dict(directory=str((directory/'mock_checkpoint').resolve()))
        tuning.write_json(directory/'latest_checkpoint.json', marker)
        if fail:
            raise RuntimeError('synthetic training failure')
        status = 'paused' if pause_first and len(calls)==1 else 'completed'
        return dict(status=status, best_checkpoint=marker['directory'], best_score=1., evaluation_history=[])

    monkeypatch.setattr(tuning, 'train_trial', run)
    return calls


def read_study(directory):
    return optuna.load_study(study_name='v6_masked_dqn', storage='sqlite:///'+str(directory/'study.sqlite3'))


def test_completed_resume_never_allocates_extra_trial(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch)
    directory = tmp_path/'study'
    tuning.run_study(directory, **synthetic_study)
    resumed = tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(calls) == len(resumed.trials) == 1
    assert resumed.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert not (directory/'active_trial.json').exists()
    assert (directory/'selected_trial.json').is_file()


def test_paused_trial_reuses_same_allocation_then_stays_within_budget(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    paused = tuning.run_study(directory, **synthetic_study)
    assert paused.trials[0].state == optuna.trial.TrialState.RUNNING
    resumed = tuning.run_study(directory, resume=True, **synthetic_study)
    tuning.run_study(directory, resume=True, **synthetic_study)
    assert [call['number'] for call in calls] == [0, 0]
    assert calls[0]['resume'] is None and calls[1]['resume'] is not None
    assert len(resumed.trials) == 1


def test_resume_refuses_missing_sampler_without_allocating(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    tuning.run_study(directory, **synthetic_study)
    (directory/'sampler.pkl').unlink()
    with pytest.raises(FileNotFoundError, match='sampler.pkl'):
        tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(calls) == len(read_study(directory).trials) == 1


def test_orphan_running_trial_refuses_replacement_even_with_budget_remaining(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    settings = dict(synthetic_study, trials=2)
    tuning.run_study(directory, **settings)
    (directory/'active_trial.json').unlink()
    with pytest.raises(ValueError, match='Orphan/foreign RUNNING'):
        tuning.run_study(directory, resume=True, **settings)
    assert len(calls) == len(read_study(directory).trials) == 1


def test_resume_refuses_current_source_changes(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    tuning.run_study(directory, **synthetic_study)
    (tuning.ROOT/'src/model.py').write_text('SYNTHETIC_SOURCE = 2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Resume must keep data, code'):
        tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(calls) == 1


def make_synthetic_dataset(root):
    (root/'metadata').mkdir(parents=True)
    (root/'metadata/active_split.json').write_text('{"synthetic":true}', encoding='utf-8')
    (root/'manifest.csv').write_text('synthetic manifest', encoding='utf-8')
    for name in ('train.csv', 'validation.csv'):
        (root/name).write_text('synthetic payload', encoding='utf-8')
    # Deliberately absent: any attempt to hash or open Test fails.
    rows = pd.DataFrame(dict(split=['train', 'validation', 'test'],
                             relative_path=['train.csv', 'validation.csv', 'never_open_test.csv']))
    train, validation = tuning.make_smoke_episodes()
    return SimpleNamespace(root=root, _metadata=dict(manifest_relative_path='manifest.csv'),
        _manifest=rows, task_contract=dict(synthetic_only=True),
        load_train=lambda:(train*18)[:35], load_validation=lambda:validation*4)


@pytest.mark.parametrize('split', ['train', 'validation'])
def test_resume_refuses_changed_train_validation_payloads_without_reading_test(
        tmp_path, monkeypatch, synthetic_study, split):
    dataset = make_synthetic_dataset(tmp_path/'synthetic_data')
    monkeypatch.setattr(tuning.V6TrainingDataset, 'open', lambda root:dataset)
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    settings = dict(synthetic_study, smoke=False)
    tuning.run_study(directory, **settings)
    identity = dataset_identity(dataset)
    assert 'never_open_test.csv' not in identity['train_validation_files']
    (dataset.root/f'{split}.csv').write_text('changed synthetic payload', encoding='utf-8')
    with pytest.raises(ValueError, match='Resume must keep data, code'):
        tuning.run_study(directory, resume=True, **settings)
    assert len(calls) == 1


def test_resume_rejects_active_config_that_disagrees_with_study(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, pause_first=True)
    directory = tmp_path/'study'
    tuning.run_study(directory, **synthetic_study)
    path = directory/'trial_0000/configuration.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    config['seed'] += 1
    tuning.write_json(path, config)
    with pytest.raises(ValueError, match='configuration'):
        tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(calls) == 1
    assert read_study(directory).trials[0].state == optuna.trial.TrialState.RUNNING


def test_failed_trial_is_persisted_and_exported_without_replacement(tmp_path, monkeypatch, synthetic_study):
    calls = fake_training(monkeypatch, fail=True)
    directory = tmp_path/'study'
    with pytest.raises(RuntimeError, match='synthetic training failure'):
        tuning.run_study(directory, **synthetic_study)
    assert read_study(directory).trials[0].state == optuna.trial.TrialState.FAIL
    assert not (directory/'active_trial.json').exists()
    assert isinstance(pickle.loads((directory/'sampler.pkl').read_bytes()), optuna.samplers.TPESampler)
    with (directory/'trials.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1 and rows[0]['state'] == 'FAIL'
    resumed = tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(resumed.trials) == len(calls) == 1


def test_resumed_sampler_matches_uninterrupted_parameter_sequence(tmp_path, monkeypatch, synthetic_study):
    dataset = make_synthetic_dataset(tmp_path/'synthetic_data')
    monkeypatch.setattr(tuning.V6TrainingDataset, 'open', lambda root:dataset)
    settings = dict(synthetic_study, smoke=False, trials=2)
    full_calls = fake_training(monkeypatch)
    full = tuning.run_study(tmp_path/'full', **settings)
    resumed_calls = fake_training(monkeypatch, pause_first=True)
    tuning.run_study(tmp_path/'resume', **settings)
    resumed = tuning.run_study(tmp_path/'resume', resume=True, **settings)
    assert [trial.params for trial in resumed.trials] == [trial.params for trial in full.trials]
    assert [call['number'] for call in full_calls] == [0, 1]
    assert [call['number'] for call in resumed_calls] == [0, 0, 1]


def test_pruned_trial_uses_budget_and_is_exported(tmp_path, monkeypatch, synthetic_study):
    calls = []

    def prune(trial, config, train, validation, directory, **kwargs):
        assert all(item.sample_id.startswith('synthetic_') for item in (*train, *validation))
        calls.append(trial.number)
        trial.report(.5, 2)
        raise optuna.TrialPruned('synthetic prune')

    monkeypatch.setattr(tuning, 'train_trial', prune)
    directory = tmp_path/'study'
    tuning.run_study(directory, **synthetic_study)
    resumed = tuning.run_study(directory, resume=True, **synthetic_study)
    assert len(resumed.trials) == len(calls) == 1
    assert resumed.trials[0].state == optuna.trial.TrialState.PRUNED
    assert resumed.trials[0].intermediate_values == {2:.5}
    assert not (directory/'active_trial.json').exists()
    with (directory/'trials.csv').open(encoding='utf-8-sig', newline='') as handle:
        assert list(csv.DictReader(handle))[0]['state'] == 'PRUNED'
