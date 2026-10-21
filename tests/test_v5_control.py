"""Narrow checks for the economic-only four-feature experiment."""

from math import fsum
from types import SimpleNamespace

import pytest

from v3.control import AccountState, EconomicMPC
from v4 import control
from v4.independent_economics import ZhouEvaluationConfig, soc_equivalent_hydrogen_cost_cny


def _episode(loads, *, durations=None):
    durations = tuple(durations or (30.0,) * len(loads))
    elapsed, times = 0.0, []
    for duration in durations:
        times.append(elapsed)
        elapsed += duration
    return SimpleNamespace(
        sample_id='v5_synthetic', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(times), dt_s=durations,
    )


def _execute(loads, powers, **options):
    commands = iter(powers)
    config = dict(state_mode='load_delta4', action_mode='increment7',
                  reward_cost_mode='zhou_economic_only', beta_soc=0.0,
                  economic_evaluation_config=ZhouEvaluationConfig(0.5))
    config.update(options)
    return control.replay_episode(
        _episode(loads), lambda _s, _a: next(commands),
        accountant=EconomicMPC(nominal_cost_cny=1.0), **config,
    )


def test_load_delta4_features_are_named_causal_and_unclipped():
    state = control.build_state(
        AccountState(soc=0.58, previous_fc_kw=180.0), (60.0, 120.0, 180.0),
        (100.0, 200.0, 900.0), state_mode='load_delta4')
    assert state == pytest.approx((0.58, 1.5, 0.3, 700.0 / 600.0))
    assert control.state_feature_value(state, 'previous_fc_kw_over_600') == pytest.approx(0.3)
    assert control.state_feature_value(state, 'current_load_delta_kw_over_600',
                                       state_mode='load_delta4') == pytest.approx(700.0 / 600.0)
    with pytest.raises(ValueError, match='dimension'):
        control.state_feature_value(state, 'soc', state_mode='minimal3')


def test_v5_replay_histories_use_measured_load_and_reset_between_voyages():
    result = _execute((-0.5, 100.0, 40.0), (60, 120, 90))
    transitions = result.transitions
    assert transitions[0].state == pytest.approx((0.6, -0.5 / 600.0, 0.0, 0.0))
    assert transitions[1].state[3] == pytest.approx(100.5 / 600.0)
    assert transitions[2].state[3] == pytest.approx(-60.0 / 600.0)
    assert [t.action_index for t in transitions] == [6, 6, 1]
    assert [t.action_delta_kw for t in transitions] == [60, 60, -30]
    for left, right in zip(transitions, transitions[1:]):
        assert left.next_state == right.state
        assert left.next_feasible_actions == right.policy_candidate_actions
        assert control.state_feature_value(right.state, 'previous_fc_kw_over_600') * 600 == pytest.approx(left.action_kw)
    assert transitions[-1].done and transitions[-1].next_feasible_actions == ()
    restarted = _execute((500.0,), (60,))
    assert restarted.transitions[0].state == pytest.approx((0.6, 500.0 / 600.0, 0.0, 0.0))


def test_pure_economics_telescopes_without_auxiliary_terms_or_physics_changes():
    loads, powers = (100.0, 100.0, 40.0, 100.0, 100.0), (60, 120, 90, 30, 0)
    result = _execute(loads, powers)
    previous = _execute(loads, powers, state_mode='minimal3',
                        reward_cost_mode='zhou_joint', beta_soc=200.0)
    assert result.final_state == previous.final_state
    assert result.total_ledger == previous.total_ledger
    assert result.economic_evaluation == previous.economic_evaluation
    corrections = []
    for transition in result.transitions:
        ledger = transition.executed_ledger
        correction = soc_equivalent_hydrogen_cost_cny(transition.state[0], transition.actual_soc,
                                                     ZhouEvaluationConfig(0.5))
        corrections.append(correction)
        assert transition.reward_cny == pytest.approx(-fsum((
            ledger.h2_cost_cny, correction, ledger.fuel_cell_degradation_cost_cny,
            ledger.battery_degradation_cost_cny)))
        assert transition.soc_soft_penalty_cny == 0.0
        assert transition.fc_ramp_penalty_cny == 0.0
        assert transition.excluded_degradation_cost_cny == 0.0
        assert transition.shore_ledger is None and transition.modeled_terminal_ledger is None
        assert transition.immediate_battery_energy_adjustment == 0.0
        assert transition.terminal_correction == 0.0
    assert min(corrections) < 0 < max(corrections)
    assert fsum(corrections) == pytest.approx(soc_equivalent_hydrogen_cost_cny(
        0.6, result.final_state.soc, ZhouEvaluationConfig(0.5)))
    assert -fsum(t.reward_cny for t in result.transitions) == pytest.approx(result.comparable_cost_cny)


def test_pure_mode_rejects_soc_auxiliary_redistribution_and_nonindependent_task():
    with pytest.raises(ValueError, match='beta_soc=0'):
        _execute((100.0,), (60,), beta_soc=200.0)
    with pytest.raises(ValueError, match='redistribution'):
        _execute((100.0,), (60,), redistribute_battery_energy=True)
    data = SimpleNamespace(sample_id='legacy', split='train', operating_mode=('onboard',),
                           load_kw=(100.0,), battery_bus_kw=(0.0,))
    with pytest.raises(ValueError, match='independent'):
        control.replay_episode(data, lambda _s, _a: 0,
            accountant=EconomicMPC(nominal_cost_cny=1.0), reward_cost_mode='zhou_economic_only')


def test_v5_increment_failure_preserves_unpenalized_real_prefix_for_existing_handler():
    with pytest.raises(control.ReplayExecutionError) as caught:
        _execute((100.0, 1400.0), (60,))
    error = caught.value
    assert error.failure_cause == 'increment_action_limited'
    assert len(error.executed_transitions) == 1
    last = error.executed_transitions[-1]
    assert len(last.state) == len(last.next_state) == 4
    assert not last.done and not last.next_feasible_actions
    assert last.failure_penalty_equivalent_cny == 0.0
    assert last.action_index == 6 and last.action_kw == 60
