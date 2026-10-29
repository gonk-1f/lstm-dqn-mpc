"""Budget arithmetic and timing contracts; no model or dataset is executed."""
import json

import pytest

from v6_sb3.budget import estimate
from v6_sb3.performance import PerformanceRecorder


def profile(tmp_path, stages, evaluations=0, status='completed'):
    (tmp_path/'configuration.json').write_text(json.dumps({
        'experiment_id': 'synthetic_budget_fixture',
    }), encoding='utf-8')
    (tmp_path/'performance.json').write_text(json.dumps({'stages': stages}), encoding='utf-8')
    (tmp_path/'result.json').write_text(json.dumps({
        'status': status,
        'adam_updates': stages.get('training_update_total', {}).get('calls', 0),
    }), encoding='utf-8')
    for index in range(evaluations):
        folder = tmp_path/f'validation_step_{index+1}'
        folder.mkdir()
        (folder/'summary.json').write_text('{}', encoding='utf-8')
    return tmp_path


def stage(calls, seconds):
    return dict(calls=calls, inclusive_seconds=seconds, exclusive_seconds=seconds)


def measured_stages():
    return {
        'warmup_collect_total': stage(1, 12.),
        'formal_collect_total': stage(2, 40.),
        'training_update_total': stage(20, 10.),
        'validation_executor': stage(16, 30.),
        'report_write': stage(2, 4.),
        'checkpoint_write': stage(3, 9.),
    }


@pytest.mark.parametrize('stages', [
    {},
    {'warmup_collect_total': stage(1, 5.)},
    {'formal_collect_total': stage(1, 5.), 'training_update_total': stage(2, 1.)},
])
def test_prefix_before_first_evaluation_cannot_estimate(tmp_path, stages):
    result = estimate(profile(tmp_path, stages), formal_steps=20, eval_every=10)
    assert result['can_estimate'] is False
    assert result['completed_evaluations'] == 0
    assert 'estimated_seconds_per_trial' not in result


@pytest.mark.parametrize('formal_steps, eval_every', [(0, 10), (-1, 10), (10, 0), (10, -1)])
def test_budget_requires_positive_steps_and_interval(tmp_path, formal_steps, eval_every):
    directory = profile(tmp_path, measured_stages(), evaluations=2)
    with pytest.raises(ValueError, match='positive'):
        estimate(directory, formal_steps=formal_steps, eval_every=eval_every)


@pytest.mark.parametrize('missing', ['validation_executor', 'report_write', 'checkpoint_write'])
def test_summary_without_required_measurements_cannot_estimate(tmp_path, missing):
    stages = measured_stages()
    del stages[missing]
    result = estimate(profile(tmp_path, stages, evaluations=2), formal_steps=25, eval_every=10)
    assert result['can_estimate'] is False
    assert missing in result['reason']


def test_restored_profile_with_newer_summary_files_cannot_estimate(tmp_path):
    # A checkpoint contains two evaluations; a later, unsaved evaluation left
    # its report directory behind. Do not divide the two timings by three.
    result = estimate(profile(tmp_path, measured_stages(), evaluations=3), formal_steps=25, eval_every=10)
    assert result['can_estimate'] is False
    assert 'do not match' in result['reason']


@pytest.mark.parametrize('status', ['completed', 'paused', 'pruned'])
def test_estimate_uses_completed_evaluations_and_nonoverlapping_totals(tmp_path, status):
    stages = measured_stages()
    # These are nested inside collection or executor and must not be added.
    stages.update(environment_step=stage(20, 25.), online_forward=stage(25, 3.),
                  replay_sample=stage(20, 1.), adam_step=stage(20, 2.))
    result = estimate(profile(tmp_path, stages, evaluations=2, status=status), formal_steps=25, eval_every=10)
    assert result['can_estimate'] is True
    assert result['requested_evaluations'] == 3
    assert result['measured_validation_seconds_per_evaluation'] == 15.
    assert result['estimated_seconds_per_trial'] == pytest.approx(12. + 25.*2. + 3.*(15.+2.+3.))
    assert 'NOT a real-voyage runtime prediction' in result['scope']


def test_failed_train_call_is_not_treated_as_a_completed_adam_update(tmp_path):
    stages = measured_stages()
    stages['training_update_total']['calls'] = 21
    directory = profile(tmp_path, stages, evaluations=2, status='paused')
    # The resumed run failed before its next Adam update; result.json still
    # describes the previously safe pause at 20 updates, while timing saw 21 calls.
    (directory/'result.json').write_text(json.dumps({
        'status': 'paused', 'adam_updates': 20,
    }), encoding='utf-8')
    result = estimate(directory, formal_steps=25, eval_every=10)
    assert result['can_estimate'] is False
    assert result['measured_formal_updates'] == 20
    assert 'do not match' in result['reason']
    assert 'estimated_seconds_per_trial' not in result


@pytest.mark.parametrize('saved_result', [
    None,
    '{}',
    'null',
    '{unfinished',
    '{"status": "failed", "adam_updates": 20}',
    '{"status": "running", "adam_updates": 20}',
    '{"status": "completed"}',
    '{"status": "completed", "adam_updates": 20.0}',
])
def test_budget_without_trustworthy_trial_result_cannot_estimate(tmp_path, saved_result):
    directory = profile(tmp_path, measured_stages(), evaluations=2)
    if saved_result is None:
        (directory/'result.json').unlink()
    else:
        (directory/'result.json').write_text(saved_result, encoding='utf-8')
    result = estimate(directory, formal_steps=25, eval_every=10)
    assert result['can_estimate'] is False
    assert 'result.json' in result['reason']
    assert 'estimated_seconds_per_trial' not in result


def test_nested_measurements_and_resumed_totals_are_exact(monkeypatch):
    clock = iter([0., 10., 14., 20., 30., 32.])
    monkeypatch.setattr('v6_sb3.performance.perf_counter', lambda:next(clock))
    recorder = PerformanceRecorder()
    with recorder.measure('outer'):
        with recorder.measure('inner'):
            pass
    saved = recorder.snapshot()
    assert saved['stages']['outer'] == dict(calls=1, inclusive_seconds=20., exclusive_seconds=16.)
    assert saved['stages']['inner'] == dict(calls=1, inclusive_seconds=4., exclusive_seconds=4.)
    restored = PerformanceRecorder()
    restored.restore(saved)
    with restored.measure('outer'):
        pass
    assert restored.snapshot()['stages']['outer'] == dict(calls=2, inclusive_seconds=22., exclusive_seconds=18.)
    assert saved['stages']['outer']['calls'] == 1  # snapshots are detached


def test_exception_records_elapsed_time_and_restores_wrapped_method(monkeypatch):
    clock = iter([1., 4.])
    monkeypatch.setattr('v6_sb3.performance.perf_counter', lambda:next(clock))
    recorder = PerformanceRecorder()

    class Owner:
        def call(self):
            raise RuntimeError('synthetic failure')

    owner = Owner()
    original = owner.call
    with pytest.raises(RuntimeError, match='synthetic failure'):
        with recorder.instrument([(owner, 'call', 'operation')]):
            owner.call()
    assert owner.call == original
    assert 'call' not in vars(owner)
    assert recorder.snapshot()['stages']['operation'] == stage(1, 3.)
