"""SQLite lifecycle of the episode adapter; real training is covered by smoke."""
import json
from pathlib import Path
import optuna
import pytest
from v6_sb3 import episode_tuning as tuning


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, 'execution_identity', lambda root, data:dict(data=data, code='fixture'))
    monkeypatch.setattr(tuning, 'inputs', lambda smoke:(('synthetic_train',), ('synthetic_val',), 'synthetic'))
    calls = []

    def run(config, train, validation, directory, *, resume=False, on_validation=None, **kwargs):
        assert train == ('synthetic_train',) and validation == ('synthetic_val',)
        calls.append(dict(config=dict(config), resume=resume, stop_after_steps=kwargs.get('stop_after_steps')))
        if len(calls)==1:
            return dict(status='paused', best_score=None, best_checkpoint=None, validation_history=[])
        summary = dict(score=3.5, completed=2, episodes=2, mean_duration_coverage=1.,
                       fixed_complete_set_cost_cny=100.)
        on_validation(summary, dict(completed_episodes=config['episodes'], formal_env_steps=128, adam_updates=4))
        return dict(status='completed', best_score=3.5, best_checkpoint=str(Path(directory)/'best'),
                    validation_history=[summary])

    monkeypatch.setattr(tuning, 'run_episode_training', run)
    return calls, dict(trials=1, episodes_per_trial=2, learning_starts=4,
                       validation_every=1, smoke=True, pruner='none')


def test_episode_trial_pause_resume_budget_and_fresh_parameter_export(tmp_path, configured):
    calls, options = configured
    directory = tmp_path/'study'
    study = tuning.run_study(directory, **options)
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    study = tuning.run_study(directory, resume=True, **options)
    tuning.run_study(directory, resume=True, **options)
    assert len(study.trials)==1 and len(calls)==2
    assert [c['resume'] for c in calls] == [False, True]
    for item in calls:
        assert item['config']['train_freq']==32 and item['config']['gradient_steps']==1
        assert item['config']['reward_mode']=='zhou_economic_only'
        assert item['config']['episodes']==2
    output = tmp_path/'best_parameters.json'
    exported = tuning.export_best(directory, output)
    assert output.is_file() and exported['schema']=='v6_episode_parameters_v1'
    assert 'checkpoint' not in exported['parameters']
    assert exported['parameters']['target_update_interval'] > 0
    assert exported['target_sync_unit']=='formal_env_steps'
    assert json.loads(output.read_text(encoding='utf-8'))==exported


def test_resume_rejects_changed_budget_before_any_new_training(tmp_path, configured):
    calls, options = configured
    tuning.run_study(tmp_path/'study', **options)
    with pytest.raises(ValueError, match='identity'):
        tuning.run_study(tmp_path/'study', resume=True, **dict(options, episodes_per_trial=3))
    assert len(calls)==1


def test_missing_sampler_does_not_replace_paused_trial(tmp_path, configured):
    calls, options = configured
    directory = tmp_path/'study'
    tuning.run_study(directory, **options)
    (directory/'sampler.pkl').unlink()
    with pytest.raises(FileNotFoundError):
        tuning.run_study(directory, resume=True, **options)
    assert len(calls)==1


def test_formal_search_requires_explicit_failure_penalty_acknowledgement(tmp_path):
    with pytest.raises(ValueError, match='failure penalty'):
        tuning.run_study(tmp_path/'study', trials=1, episodes_per_trial=100, smoke=False)


def test_bounded_pause_is_per_invocation_and_not_a_changed_study_budget(tmp_path, configured):
    calls, options = configured
    tuning.run_study(tmp_path/'study', stop_after_steps=37, **options)
    tuning.run_study(tmp_path/'study', resume=True, **options)
    assert [call['stop_after_steps'] for call in calls]==[37, None]


def test_orphan_trial_and_changed_active_parameters_are_not_reallocated(tmp_path, configured):
    calls, options = configured
    directory = tmp_path/'study'
    tuning.run_study(directory, **options)
    configuration = directory/'trial_0000_configuration.json'
    old = json.loads(configuration.read_text(encoding='utf-8'))
    configuration.write_text(json.dumps(dict(old, seed=999)), encoding='utf-8')
    with pytest.raises(ValueError, match='configuration'):
        tuning.run_study(directory, resume=True, **options)
    configuration.write_text(json.dumps(old), encoding='utf-8')
    (directory/'active_trial.json').unlink()
    with pytest.raises(ValueError, match='Orphan'):
        tuning.run_study(directory, resume=True, **options)
    assert len(calls)==1
