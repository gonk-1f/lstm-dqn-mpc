"""Focused checks for V6 one-second controls and causal history."""

from math import fsum
from types import SimpleNamespace

import pytest

from v3.control import AccountState, EconomicMPC
from v4 import control
from v4.actions import action_values_kw, candidate_fc_actions
from v4.independent_economics import ZhouEvaluationConfig


def _execute(loads, powers, **options):
    data = SimpleNamespace(
        sample_id='v6_synthetic', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(float(t) for t in range(len(loads))), dt_s=(1.0,) * len(loads),
    )
    commands = iter(powers)
    config = dict(state_mode='history5', action_mode='increment13',
                  reward_cost_mode='v6_energy_soc_ramp', beta_soc=0.0,
                  economic_evaluation_config=ZhouEvaluationConfig(0.5))
    config.update(options)
    return control.replay_episode(
        data, lambda _s, _a: next(commands),
        accountant=EconomicMPC(nominal_cost_cny=1.0), **config,
    )


def test_increment13_executes_unscaled_kw_and_preserves_mask_indices():
    assert action_values_kw('increment13') == tuple(range(-60, 61, 10))
    assert candidate_fc_actions(control.ACTION_KW, 0.0, 'increment13') == tuple(range(0, 61, 10))
    assert candidate_fc_actions(control.ACTION_KW, 600.0, 'increment13') == tuple(range(540, 601, 10))
    replay = _execute((100.0, 50.0, 40.0, 100.0), (30, 90, 40, 0))
    assert [t.action_kw for t in replay.transitions] == [30, 90, 40, 0]
    assert [t.action_delta_kw for t in replay.transitions] == [30, 60, -50, -40]
    assert [t.action_index for t in replay.transitions] == [9, 12, 1, 2]
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    for transition in replay.transitions:
        physical = AccountState(soc=transition.state[0], previous_fc_kw=transition.state[2] * 600.0)
        assert transition.physical_feasible_actions == control.feasible_fc_actions(
            physical, transition.raw_load_kw, accountant, duration_s=1.0)
    for left, right in zip(replay.transitions, replay.transitions[1:]):
        assert left.next_state == right.state
        assert left.next_feasible_actions == right.policy_candidate_actions
    assert replay.transitions[-1].done and replay.transitions[-1].next_feasible_actions == ()


def test_v6_one_second_soc_reward_uses_energy_and_fixed_auxiliary_weights():
    replay = _execute((100.0, 20.0), (30, 90))
    baseline = _execute((100.0, 20.0), (30, 90), reward_cost_mode='energy_only')
    assert replay.total_ledger == baseline.total_ledger
    assert replay.final_state == baseline.final_state
    corrections = []
    for t in replay.transitions:
        battery = t.raw_load_kw - t.action_kw
        cell_power = battery / 0.95 if battery > 0 else battery * 0.95
        assert t.actual_soc == pytest.approx(t.state[0] - cell_power / 3600.0 / 624.0)
        correction = (t.state[0] - t.actual_soc) * 624.0 / (0.5 * (120.0 / 3.6)) * 21.9
        soc_penalty = 0.2 * ((t.actual_soc - 0.5) / 0.1) ** 2
        ramp_penalty = 0.05 * t.action_delta_kw ** 2 / 600.0
        corrections.append(correction)
        assert t.duration_s == 1.0
        assert t.soc_equivalent_hydrogen_cost_cny == pytest.approx(correction)
        assert t.soc_soft_penalty_cny == pytest.approx(soc_penalty)
        assert t.fc_ramp_penalty_cny == pytest.approx(ramp_penalty)
        assert t.reward_cny == pytest.approx(-fsum((
            t.executed_ledger.h2_cost_cny, correction, soc_penalty, ramp_penalty)))
        assert t.excluded_degradation_cost_cny == pytest.approx(
            t.executed_ledger.fuel_cell_degradation_cost_cny
            + t.executed_ledger.battery_degradation_cost_cny)
    assert min(corrections) < 0 < max(corrections)


def test_history5_uses_only_causal_seconds_inclusive_window_and_resets():
    loads = tuple(float(t) for t in range(330))
    replay = _execute(loads, (0,) * len(loads))
    for t in (0, 1, 29, 30, 31, 299, 300, 301, 329):
        state = replay.transitions[t].state
        start = max(0, t - 300)
        assert state[1:] == pytest.approx((
            loads[t] / 600.0, 0.0,
            (loads[t] - loads[max(0, t - 30)]) / 600.0,
            fsum(loads[start:t + 1]) / (t - start + 1) / 600.0,
        ))
    restarted = _execute((900.0,), (60,)).transitions[0]
    assert restarted.state == pytest.approx((0.6, 1.5, 0.0, 0.0, 1.5))
    assert restarted.next_state[3:] == restarted.state[3:]


def test_v6_empty_successor_withholds_unpenalized_failure_tail():
    seen = []
    with pytest.raises(control.ReplayExecutionError) as caught:
        _execute((100.0, 1400.0), (60,), on_transition=lambda i, t: seen.append(t))
    error = caught.value
    assert error.failure_cause == 'increment_action_limited'
    assert seen == []
    assert len(error.executed_transitions) == 1
    last = error.executed_transitions[0]
    assert not last.done and last.next_feasible_actions == ()
    assert last.action_kw == 60 and last.action_index == 12
    assert last.failure_penalty_equivalent_cny == 0.0
