"""New reward wiring through mock episode runners; no vessel training or Test.

SQLite/Trial lifecycle is real. The runner only captures configuration and
returns fixed pause/Validation results, so these checks do not optimize a model.
"""
import json
from pathlib import Path

import optuna
import pytest

from v6_sb3 import episode_tuning as tuning


MODE = 'inverse_one_plus_economic'
NEW_STUDY_NAME = 'v6_one_plus_episode_dqn'
LEGACY_MODE = 'zhou_economic_only'


@pytest.fixture
def configured(tmp_path, monkeypatch):
    calls = []
    identity_calls = []

    def identity(root, data, **kwargs):
        identity_calls.append(dict(kwargs))
        result = dict(schema=1, data=data,
            economics=dict(evaluation_version='fixture_frozen_economics', fc_average_efficiency=.57),
            source_files={'v6_sb3/episode_tuning.py':'fixture_source_digest'},
            failure_penalty={'reference_amount':10132.660087898294})
        if kwargs.get('reward_mode') == MODE:
            from v6_sb3.rewards import reward_contract
            result.update(failure_penalty=None, reward_contract=reward_contract(MODE))
        return result

    def run(config, train, validation, directory, *, resume=False, on_validation=None, **kwargs):
        assert train == ('synthetic_train',) and validation == ('synthetic_validation',)
        calls.append(dict(config=dict(config), resume=resume, directory=Path(directory)))
        if not resume:
            return dict(status='paused', best_score=None, best_checkpoint=None, validation_history=[])
        summary = dict(score=3.5, completed=2, episodes=2, mean_duration_coverage=1.,
            fixed_complete_set_cost_cny=100.)
        on_validation(summary, dict(completed_episodes=config['episodes'], formal_env_steps=128,
                                   adam_updates=4))
        return dict(status='completed', best_score=3.5, best_checkpoint=str(Path(directory)/'best'),
                    validation_history=[summary])

    monkeypatch.setattr(tuning, 'execution_identity', identity)
    monkeypatch.setattr(tuning, 'inputs', lambda smoke:(('synthetic_train',), ('synthetic_validation',),
                                                      'fixture_train_validation_only'))
    monkeypatch.setattr(tuning, 'run_episode_training', run)
    options = dict(trials=1, episodes_per_trial=2, learning_starts=4, validation_every=1,
                   smoke=True, pruner='none')
    return calls, identity_calls, options


def _contract(identity):
    return identity.get('reward_contract', identity['execution'].get('reward_contract'))


def test_new_study_pause_resume_preserves_mode_contract_and_episode_mechanism(tmp_path, configured):
    calls, identity_calls, options = configured
    directory = tmp_path/'new_reward_study'
    study = tuning.run_study(directory, reward_mode=MODE, **options)
    assert study.study_name == NEW_STUDY_NAME
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    saved = json.loads((directory/'study_identity.json').read_text(encoding='utf-8'))
    assert saved['reward_mode'] == MODE
    assert _contract(saved)
    assert saved['execution']['failure_penalty'] is None
    assert identity_calls == [{'reward_mode':MODE}]
    study = tuning.run_study(directory, resume=True, reward_mode=MODE, **options)
    assert len(study.trials) == 1 and study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert [row['resume'] for row in calls] == [False, True]
    for row in calls:
        assert row['config']['reward_mode'] == MODE
        assert row['config']['train_freq'] == 32
        assert row['config']['gradient_steps'] == 1
        assert row['config']['episodes'] == 2
    assert study.best_trial.user_attrs['latest_validation']['fixed_complete_set_cost_cny'] == 100.


@pytest.mark.parametrize('initial,resumed', [(MODE, LEGACY_MODE), (LEGACY_MODE, MODE)])
def test_resume_rejects_reward_mode_change_before_any_new_runner_call(tmp_path, configured,
                                                                    initial, resumed):
    calls, _, options = configured
    directory = tmp_path/'study'
    tuning.run_study(directory, reward_mode=initial, **options)
    with pytest.raises(ValueError, match='identity|reward'):
        tuning.run_study(directory, resume=True, reward_mode=resumed, **options)
    assert len(calls) == 1


def test_resume_rejects_tampered_reward_contract_before_any_new_runner_call(tmp_path, configured):
    calls, _, options = configured
    directory = tmp_path/'study'
    tuning.run_study(directory, reward_mode=MODE, **options)
    path = directory/'study_identity.json'
    identity = json.loads(path.read_text(encoding='utf-8'))
    if 'reward_contract' in identity:
        identity['reward_contract'] = {'formula':'incorrect_formula'}
    else:
        identity['execution']['reward_contract'] = {'formula':'incorrect_formula'}
    path.write_text(json.dumps(identity), encoding='utf-8')
    with pytest.raises(ValueError, match='identity|reward'):
        tuning.run_study(directory, resume=True, reward_mode=MODE, **options)
    assert len(calls) == 1


def test_new_export_binds_actual_study_reward_economics_sources_and_target_units(tmp_path, configured):
    calls, _, options = configured
    directory = tmp_path/'study'
    tuning.run_study(directory, reward_mode=MODE, **options)
    tuning.run_study(directory, resume=True, reward_mode=MODE, **options)
    identity = json.loads((directory/'study_identity.json').read_text(encoding='utf-8'))
    output = tmp_path/'best_parameters.json'
    exported = tuning.export_best(directory, output)
    assert exported['schema'] == 'v6_episode_parameters_v2'
    assert exported['reward_mode'] == MODE
    assert exported['reward_contract'] == _contract(identity)
    assert exported['economics'] == identity['execution']['economics']
    assert exported['source_files'] == identity['execution']['source_files']
    assert exported['selection_split'] == 'validation'
    assert exported['target_sync_unit'] == 'formal_env_steps'
    assert 'checkpoint' not in exported['parameters']
    assert len(calls) == 2
    assert json.loads(output.read_text(encoding='utf-8')) == exported


def test_legacy_default_study_and_v1_export_keep_their_existing_identity_shape(tmp_path, configured):
    _, identity_calls, options = configured
    directory = tmp_path/'legacy_study'
    tuning.run_study(directory, **options)
    study = tuning.run_study(directory, resume=True, **options)
    assert study.study_name == 'v6_economic_episode_dqn'
    assert identity_calls == [{}, {}]
    exported = tuning.export_best(directory, tmp_path/'legacy_parameters.json')
    assert exported['schema'] == 'v6_episode_parameters_v1'
    assert exported['reward_mode'] == LEGACY_MODE
    assert exported['target_sync_unit'] == 'formal_env_steps'


def test_new_mode_does_not_require_acknowledging_unused_legacy_failure_penalty(tmp_path, configured):
    calls, _, options = configured
    options = {**options, 'smoke':False}
    study = tuning.run_study(tmp_path/'new_formal_contract_only', reward_mode=MODE,
        acknowledge_failure_penalty=False, **options)
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    assert len(calls) == 1 and calls[0]['config']['reward_mode'] == MODE
    assert calls[0]['config']['acknowledge_failure_penalty'] is False


def test_unknown_reward_mode_is_rejected_before_creating_study_or_calling_runner(tmp_path, configured):
    calls, identity_calls, options = configured
    directory = tmp_path/'must_not_exist'
    with pytest.raises(ValueError, match='reward'):
        tuning.run_study(directory, reward_mode='unknown_reward', **options)
    assert calls == [] and identity_calls == []
    assert not directory.exists()


def test_cli_forwards_explicit_new_reward_mode_to_episode_study(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tuning, 'run_study', lambda directory, **kwargs:calls.append(kwargs))
    tuning.main(['--study-dir', str(tmp_path/'study'), '--smoke', '--reward-mode', MODE])
    assert len(calls) == 1 and calls[0]['reward_mode'] == MODE
