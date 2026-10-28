"""Fixed short trajectories and offline reward arithmetic; no training or Test data."""

import importlib.util
from math import fsum
from pathlib import Path
from types import SimpleNamespace

import pytest

from v2.models.fuel_cell_degradation import FC_EOL_VOLTAGE_LOSS_UV
from v3.control import AccountState, EconomicMPC
from v4.control import replay_episode
from v4.independent_economics import ZhouEvaluationConfig


def _audit():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_v6_rewards.py'
    assert path.is_file(), 'the independent offline reward comparison tool must exist'
    spec = importlib.util.spec_from_file_location('audit_v6_rewards', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay(mode, *, durations=(1., 1., 1., 1.), beta_soc=0.):
    loads = (0., 100., 20., 100.)
    powers = iter((20, 20, 0, 20))
    times = tuple(fsum(durations[:i]) for i in range(4))
    episode = SimpleNamespace(sample_id='fixed_audit', split='train',
        task_mode='independent_voyage', operating_mode=('onboard',) * 4,
        load_kw=loads, time_s=times, dt_s=durations)
    return replay_episode(episode, lambda _s, _a: next(powers),
        accountant=EconomicMPC(nominal_cost_cny=1.),
        economic_evaluation_config=ZhouEvaluationConfig(.5),
        state_mode='history5', action_mode='increment13',
        reward_cost_mode=mode, beta_soc=beta_soc)


@pytest.mark.parametrize('durations', ((1., 1., 1., 1.), (1., 2., 3., 4.)))
def test_zhou_economic_only_gamma_one_equals_negative_complete_evaluation(durations):
    replay = _replay('zhou_economic_only', durations=durations)
    assert fsum(t.reward_cny for t in replay.transitions) == pytest.approx(
        -replay.economic_evaluation.total_cost_cny, abs=1e-9)
    assert all(t.soc_soft_penalty_cny == t.fc_ramp_penalty_cny == 0.
               for t in replay.transitions)
    assert replay.modeled_terminal_settlement is None
    assert replay.final_state.previous_fc_kw == 20.


def test_four_reward_modes_preserve_physics_and_have_different_objectives():
    modes = ('full_economic', 'zhou_economic_only', 'zhou_joint', 'v6_energy_soc_ramp')
    results = {mode: _replay(mode, beta_soc=7. if mode == 'zhou_joint' else 0.) for mode in modes}
    economic = results['zhou_economic_only']
    for result in results.values():
        assert result.total_ledger == economic.total_ledger
        assert result.final_state == economic.final_state
    for mode, result in results.items():
        for t in result.transitions:
            expected = _audit().step_reward_comparison(
                h2_cost=t.executed_ledger.h2_cost_cny,
                fc_degradation=t.executed_ledger.fuel_cell_degradation_cost_cny,
                battery_degradation=t.executed_ledger.battery_degradation_cost_cny,
                soc_before=t.state[0], soc_after=t.actual_soc,
                delta_kw=t.action_delta_kw, config=ZhouEvaluationConfig(.5),
                zhou_beta_soc=7.)
            assert t.reward_cny == pytest.approx(expected[mode], abs=1e-10)


def test_failure_penalty_is_separate_and_added_exactly_once_to_each_mode():
    audit = _audit()
    options = dict(h2_cost=.4, fc_degradation=360., battery_degradation=.02,
        soc_before=.6, soc_after=.599, delta_kw=20., config=ZhouEvaluationConfig(.5))
    unpenalized = audit.step_reward_comparison(**options)
    penalized = audit.step_reward_comparison(**options, failure_penalty=123.)
    for mode in ('full_economic', 'zhou_economic_only', 'zhou_joint', 'v6_energy_soc_ramp'):
        assert penalized[mode] == pytest.approx(unpenalized[mode] - 123.)
    assert penalized['failure_penalty'] == 123.


def test_fc_start_runtime_transient_and_no_duplicate_shutdown_cycle():
    audit = _audit()
    accountant = EconomicMPC(nominal_cost_cny=1.)
    state = AccountState()
    for power in (20., 20., 0., 20.):
        parts = audit.fc_cost_components(state.previous_fc_kw, power, 1., state.fc_loss_uv)
        next_state, ledger, _ = accountant.interval(state, power, 20., duration_s=1.)
        assert parts['charged_total_cny'] == pytest.approx(ledger.fuel_cell_degradation_cost_cny)
        assert parts['start_raw_cny'] == pytest.approx(358.65 if state.previous_fc_kw == 0. and power else 0.)
        assert parts['transient_raw_cny'] == pytest.approx(.11025 * abs(power - state.previous_fc_kw))
        assert parts['runtime_raw_cny'] == pytest.approx(.042375 if power else 0.)
        state = next_state


def test_fc_eol_crossing_charges_remaining_life_once_then_zero():
    audit = _audit()
    state = AccountState(fc_loss_uv=FC_EOL_VOLTAGE_LOSS_UV - 1.)
    accountant = EconomicMPC(nominal_cost_cny=1.)
    next_state, ledger, _ = accountant.interval(state, 20., 20., duration_s=1.)
    first = audit.fc_cost_components(0., 20., 1., state.fc_loss_uv)
    assert first['charged_total_cny'] == pytest.approx(15.)
    assert first['charged_total_cny'] == pytest.approx(ledger.fuel_cell_degradation_cost_cny)
    assert first['raw_total_cny'] > first['charged_total_cny']
    after = audit.fc_cost_components(20., 0., 1., next_state.fc_loss_uv)
    assert after['raw_total_cny'] > 0.
    assert after['charged_total_cny'] == 0.


def test_one_second_soc_penalty_accumulates_with_decisions_not_elapsed_duration():
    # Frozen V6 applies a state penalty once per decision, without a dt factor.
    # Equal 30 seconds at SOC .6 therefore has 30 times the penalty at 1 Hz.
    one_second = _audit().step_reward_comparison(h2_cost=0., fc_degradation=0.,
        battery_degradation=0., soc_before=.6, soc_after=.6, delta_kw=0.,
        config=ZhouEvaluationConfig(.5))
    assert one_second['v6_soc_penalty'] == pytest.approx(.2)
    assert 30 * one_second['v6_soc_penalty'] == pytest.approx(6.)


def test_squared_ramp_prefers_two_small_moves_while_formal_transient_is_linear():
    audit = _audit()
    def step(delta):
        return audit.step_reward_comparison(h2_cost=0., fc_degradation=0.,
            battery_degradation=0., soc_before=.5, soc_after=.5, delta_kw=delta,
            config=ZhouEvaluationConfig(.5))
    assert 2 * step(30.)['v6_ramp_penalty'] == pytest.approx(step(60.)['v6_ramp_penalty'] / 2.)
    assert 2 * audit.fc_cost_components(100., 130., 1.)['transient_raw_cny'] == pytest.approx(
        audit.fc_cost_components(100., 160., 1.)['transient_raw_cny'])


def test_sparse_train_prefix_does_not_claim_complete_or_eol_capped_cost():
    audit = _audit()
    rows = [dict(time_s='0', soc_before='.6', soc_after='.599', fc_kw='20', delta_kw='20',
                 fc_h2='.1', battery_h2eq='.819936', soc_penalty='.19602',
                 fc_ramp_penalty='.033333333333', failure_penalty='0', immediate_reward='-1.149289333333'),
            dict(time_s='20', soc_before='.59', soc_after='.589', fc_kw='20', delta_kw='0',
                 fc_h2='.1', battery_h2eq='.819936', soc_penalty='.15842',
                 fc_ramp_penalty='0', failure_penalty='100', immediate_reward='-101.078356')]
    result = audit.summarize_logged_rows(rows, config=ZhouEvaluationConfig(.5), source_kind='train_sparse_prefix')
    assert result['contiguous_from_reset'] is False
    assert result['complete_economic_cost_cny'] is None
    assert result['zhou_economic_only_return'] is None
    assert result['fc_degradation_logged_cny'] is None


def test_complete_logged_validation_reconstructs_physical_ledger_and_rewards():
    replay = _replay('v6_energy_soc_ramp')
    rows = [dict(time_s=t.relative_time_s, soc_before=t.state[0], soc=t.actual_soc,
        fc_kw=t.action_kw, fc_delta_kw=t.action_delta_kw,
        fc_h2=t.executed_ledger.h2_cost_cny,
        fc_degradation_cost_cny=t.executed_ledger.fuel_cell_degradation_cost_cny,
        battery_degradation_cost_cny=t.executed_ledger.battery_degradation_cost_cny,
        battery_h2eq=t.soc_equivalent_hydrogen_cost_cny,
        soc_penalty=t.soc_soft_penalty_cny, fc_ramp=t.fc_ramp_penalty_cny,
        failure_penalty=t.failure_penalty_equivalent_cny, reward=t.reward_cny,
        done=t.done, terminal_reason=t.terminal_reason) for t in replay.transitions]
    summary = _audit().summarize_logged_rows(rows, config=ZhouEvaluationConfig(.5),
                                           source_kind='validation_trajectory')
    assert summary['completed'] and summary['contiguous_from_reset']
    assert summary['complete_economic_cost_cny'] == pytest.approx(replay.economic_evaluation.total_cost_cny)
    assert summary['fc_logged_max_step_residual_cny'] < 1e-10
    assert summary['v6_reward_max_step_residual'] < 1e-10
    assert summary['zhou_return_plus_economic_plus_failure'] == pytest.approx(0., abs=1e-9)


def test_gamma_less_than_one_is_a_different_discounted_objective():
    replay = _replay('zhou_economic_only')
    discounted = fsum(.99 ** i * t.reward_cny for i,t in enumerate(replay.transitions))
    assert abs(discounted + replay.economic_evaluation.total_cost_cny) > 1.


def test_offline_cli_refuses_test_payload_paths(tmp_path):
    test_path = tmp_path / 'test_round_05' / 'payload.csv.gz'
    with pytest.raises(ValueError, match='Test'):
        _audit().validate_source_path(test_path)
