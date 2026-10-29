"""No-training checks for the independent tuning/measurement contract."""
import time

import pytest

from v6_sb3.search import sample_parameters, validation_objective
from v6_sb3.performance import PerformanceRecorder


def result(completed=7, coverage=.9, cost=None):
    return dict(episodes=8, completed=completed, mean_duration_coverage=coverage,
                fixed_complete_set_cost_cny=cost)


def test_selection_prioritizes_completion_and_complete_cost_not_reward():
    assert validation_objective(result(8, 1., 1e15)) > validation_objective(result(7, 1.))
    assert validation_objective(result(7, .9)) > validation_objective(result(7, .8))
    assert validation_objective(result(8, 1., 100.)) > validation_objective(result(8, 1., 200.))
    assert validation_objective(result(8, 1., -100.)) > validation_objective(result(8, 1., 100.))


class FixedTrial:
    def suggest_float(self, name, low, high, **kwargs):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def test_search_does_not_mutate_reward_physics_gamma_or_masks():
    parameters = sample_parameters(FixedTrial())
    assert parameters['learning_rate'] > 0
    assert parameters['batch_size'] in (32, 64, 128)
    assert parameters['target_update_interval'] in (500, 1000, 2000)
    assert not {'gamma', 'reward_scale', 'reward_mode', 'action_mode', 'train_freq',
                'gradient_steps', 'failure_penalty'}.intersection(parameters)


def test_nested_performance_scopes_have_exclusive_and_inclusive_times():
    recorder = PerformanceRecorder()
    with recorder.measure('outer'):
        time.sleep(.002)
        with recorder.measure('inner'):
            time.sleep(.002)
    data = recorder.snapshot()
    assert data['stages']['inner']['calls'] == 1
    assert data['stages']['outer']['inclusive_seconds'] >= data['stages']['inner']['inclusive_seconds']
    assert data['stages']['outer']['exclusive_seconds'] < data['stages']['outer']['inclusive_seconds']


def test_instrumentation_restores_methods_before_model_checkpointing():
    recorder = PerformanceRecorder()

    class MethodOwner:
        def call(self):
            return 42

    owner = MethodOwner()
    original = owner.call
    with recorder.instrument([(owner, 'call', 'call')]):
        assert owner.call() == 42
    assert owner.call == original
    assert 'call' not in owner.__dict__
    assert recorder.snapshot()['stages']['call']['calls'] == 1
