from types import SimpleNamespace

import pytest

from v2.economics import HYDROGEN_PRICE_CNY_PER_KG, RawCnyIntervalLedger
from v3.control import AccountState, EconomicMPC
from v4.control import ReplayExecutionError, feasible_fc_actions, replay_episode


def independent_episode(loads, durations=None):
    durations = tuple(durations or (30.0,) * len(loads))
    elapsed, times = 0.0, []
    for duration in durations:
        times.append(elapsed)
        elapsed += duration
    return SimpleNamespace(
        sample_id='synthetic_independent', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(times), dt_s=durations,
    )


def configuration(eta=0.5):
    from v4.independent_economics import ZhouEvaluationConfig
    return ZhouEvaluationConfig(fc_average_efficiency=eta)


@pytest.mark.parametrize('terminal_soc', (0.4, 0.6, 0.8))
def test_signed_soc_equivalent_hydrogen_uses_no_extra_battery_efficiency(terminal_soc):
    from v4.independent_economics import evaluate_zhou_economics
    ledger = RawCnyIntervalLedger(21.9, 2.0, 3.0, 0.0)
    evaluation = evaluate_zhou_economics(
        ledger, soc_initial=0.6, soc_terminal=terminal_soc, config=configuration())
    correction = (0.6 - terminal_soc) * 624.0 / (0.5 * (120.0 / 3.6))
    assert evaluation.actual_hydrogen_mass_kg == pytest.approx(1.0)
    assert evaluation.soc_equivalent_hydrogen_mass_kg == pytest.approx(correction)
    assert evaluation.equivalent_hydrogen_mass_kg == pytest.approx(1.0 + correction)
    assert evaluation.equivalent_hydrogen_cost_cny == pytest.approx((1.0 + correction) * 21.9)
    assert evaluation.fuel_cell_degradation_cost_cny == 2.0
    assert evaluation.battery_degradation_cost_cny == 3.0
    assert evaluation.total_cost_cny == pytest.approx((1.0 + correction) * 21.9 + 5.0)


def test_independent_replay_ends_without_shore_or_terminal_charge():
    class NoShore(EconomicMPC):
        def shore_interval(self, *args, **kwargs):
            raise AssertionError('independent voyages cannot execute shore intervals')

    accountant = NoShore(nominal_cost_cny=1.0)
    result = replay_episode(
        independent_episode((100.0, 100.0)), lambda _s, _a: 0,
        accountant=accountant, economic_evaluation_config=configuration(),
        state_mode='minimal3', reward_cost_mode='control_only', beta_soc=200.0)
    assert result.task_mode == 'independent_voyage'
    assert result.shore_blocks == ()
    assert result.modeled_terminal_settlement is None
    assert result.final_state.soc < 0.6
    assert result.transitions[-1].done
    assert result.transitions[-1].terminal_reason == 'completed'
    assert result.transitions[-1].next_feasible_actions == ()
    assert all(t.shore_ledger is None and t.modeled_terminal_ledger is None for t in result.transitions)
    assert result.economic_evaluation.soc_terminal == result.final_state.soc
    assert result.comparable_cost_cny == result.economic_evaluation.total_cost_cny
    assert result.transitions[-1].reward_cny == pytest.approx(-200.0 * (result.final_state.soc - 0.6) ** 2)


def test_short_tail_uses_same_duration_for_next_mask_and_execution():
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    result = replay_episode(
        independent_episode((624.0,) * 46, (30.0,) * 45 + (10.0,)),
        lambda _s, _a: 0, accountant=accountant, economic_evaluation_config=configuration())
    before_tail = AccountState(soc=result.transitions[-2].actual_soc)
    assert 0 not in feasible_fc_actions(before_tail, 624.0, accountant)
    assert 0 in feasible_fc_actions(before_tail, 624.0, accountant, duration_s=10.0)
    assert 0 in result.transitions[-2].next_feasible_actions
    assert result.transitions[-1].duration_s == 10.0
    assert result.transitions[-1].relative_time_s == 1350.0
    expected = 0.6 - 624.0 * (45 * 30.0 + 10.0) / (3600.0 * 0.95 * 624.0)
    assert result.final_state.soc == pytest.approx(expected)


def test_independent_physical_failure_keeps_only_real_prefix():
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(
            independent_episode((100.0, 2000.0), (30.0, 7.0)), lambda _s, _a: 0,
            accountant=EconomicMPC(nominal_cost_cny=1.0), economic_evaluation_config=configuration())
    error = caught.value
    assert error.failure_kind == 'no_feasible_action'
    assert error.failure_cause == 'structural_power'
    assert error.row_index == 1
    assert len(error.executed_transitions) == 1
    assert not error.executed_transitions[-1].done
    assert error.executed_transitions[-1].next_feasible_actions == ()
    assert error.executed_transitions[-1].modeled_terminal_ledger is None


def test_accountant_duration_scales_energy_and_preserves_formal_degradation_costs():
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    initial = AccountState()
    short, short_ledger, _ = accountant.interval(initial, 100.0, 200.0, duration_s=10.0)
    full, full_ledger, _ = accountant.interval(initial, 100.0, 200.0)
    assert 0.6 - short.soc == pytest.approx((0.6 - full.soc) / 3.0)
    assert short_ledger.h2_cost_cny == pytest.approx(full_ledger.h2_cost_cny / 3.0)
    assert short_ledger.battery_degradation_cost_cny == pytest.approx(full_ledger.battery_degradation_cost_cny / 3.0)
    assert short.fc_loss_uv < full.fc_loss_uv


def test_legacy_replay_keeps_existing_terminal_settlement():
    episode = SimpleNamespace(sample_id='legacy', split='train', operating_mode=('onboard',),
                              load_kw=(100.0,), battery_bus_kw=(0.0,))
    result = replay_episode(episode, lambda _s, _a: 0, accountant=EconomicMPC(nominal_cost_cny=1.0))
    assert result.task_mode == 'legacy'
    assert result.economic_evaluation is None
    assert result.modeled_terminal_settlement is not None
    assert result.comparable_cost_cny == pytest.approx(
        result.total_cost_cny + result.modeled_terminal_settlement.ledger.total_cost_cny)


def test_formal_curve_reference_is_the_frozen_sixty_point_arithmetic_mean():
    from math import fsum
    from v2.models.fuel_cell_efficiency import formal_fuel_cell_efficiency_map
    from v4.independent_economics import ZhouEvaluationConfig
    curve = formal_fuel_cell_efficiency_map()
    powers = tuple(range(10, 601, 10))
    efficiencies = tuple(float(curve.eta(power)) for power in powers)
    config = ZhouEvaluationConfig.from_formal_curve()
    assert config.fc_average_efficiency == fsum(efficiencies) / 60
    identity = config.identity()
    assert identity['fc_reference_power_kw'] == list(powers)
    assert identity['fc_reference_efficiencies'] == list(efficiencies)
    assert identity['fc_reference_count'] == 60
    assert identity['fc_reference_method'] == 'math.fsum(eta(P) for P in range(10, 601, 10)) / 60'
    assert identity['fc_curve_source']['workbook_sha256']
    assert identity['actual_hydrogen_method'] == 'original instantaneous formal fuel-cell efficiency curve'
    assert identity['fc_reference_is_vessel_calibration'] is False


@pytest.mark.parametrize('eta', (0.0, -0.1, 1.1, float('nan'), float('inf'), True, '0.5'))
def test_fc_average_efficiency_must_be_explicit_and_valid(eta):
    from v4.independent_economics import ZhouEvaluationConfig
    with pytest.raises(ValueError):
        ZhouEvaluationConfig(fc_average_efficiency=eta)
    with pytest.raises(TypeError):
        ZhouEvaluationConfig()


def test_independent_evaluation_preserves_original_eol_caps():
    from v2.models.battery_degradation import BATTERY_LIFETIME_Q_AH
    from v2.models.fuel_cell_degradation import FC_EOL_VOLTAGE_LOSS_UV
    from v4.independent_economics import evaluate_zhou_economics
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    initial = AccountState(fc_loss_uv=FC_EOL_VOLTAGE_LOSS_UV,
                           battery_weighted_ah=BATTERY_LIFETIME_Q_AH)
    final, ledger, _ = accountant.interval(initial, 100.0, 200.0, duration_s=7.0)
    assert final.fc_loss_uv > initial.fc_loss_uv
    assert final.battery_weighted_ah > initial.battery_weighted_ah
    evaluation = evaluate_zhou_economics(
        ledger, soc_initial=0.6, soc_terminal=final.soc, config=configuration())
    assert evaluation.fuel_cell_degradation_cost_cny == ledger.fuel_cell_degradation_cost_cny == 0.0
    assert evaluation.battery_degradation_cost_cny == ledger.battery_degradation_cost_cny == 0.0


def test_independent_terminal_keeps_last_fc_and_instantaneous_hydrogen_model():
    from v2.models.fuel_cell_efficiency import hydrogen_mass_from_map_kg
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    result = replay_episode(
        independent_episode((100.0,), (7.0,)), lambda _s, _a: 600,
        accountant=accountant, economic_evaluation_config=configuration())
    final, original_ledger, _ = accountant.interval(AccountState(), 600.0, 100.0, duration_s=7.0)
    assert result.final_state == final
    assert result.final_state.previous_fc_kw == 600.0
    assert result.total_ledger == original_ledger
    assert result.economic_evaluation.actual_hydrogen_mass_kg == pytest.approx(
        hydrogen_mass_from_map_kg(600.0, 7.0, accountant.fc_efficiency))
    assert result.economic_evaluation.fuel_cell_degradation_cost_cny == original_ledger.fuel_cell_degradation_cost_cny
    assert result.economic_evaluation.battery_degradation_cost_cny == original_ledger.battery_degradation_cost_cny
    assert result.economic_evaluation.actual_hydrogen_mass_kg == pytest.approx(
        original_ledger.h2_cost_cny / HYDROGEN_PRICE_CNY_PER_KG)


def test_independent_raw_negative_load_is_retained_while_existing_deadband_applies():
    result = replay_episode(
        independent_episode((-0.5,), (7.0,)), lambda _s, _a: 0,
        accountant=EconomicMPC(nominal_cost_cny=1.0), economic_evaluation_config=configuration())
    assert result.transitions[0].raw_load_kw == -0.5
    assert result.transitions[0].actual_battery_kw == 0.0
    assert result.transitions[0].duration_s == 7.0
    assert result.transitions[0].relative_time_s == 0.0
    assert result.final_state.soc == 0.6


def test_independent_task_rejects_nondefault_initial_state_or_missing_configuration():
    data = independent_episode((100.0,))
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    with pytest.raises(ValueError, match='explicit ZhouEvaluationConfig'):
        replay_episode(data, lambda _s, _a: 0, accountant=accountant)
    for initial in (AccountState(soc=0.5), AccountState(previous_fc_kw=100.0),
                    AccountState(fc_loss_uv=1.0), AccountState(battery_weighted_ah=1.0)):
        with pytest.raises(ValueError, match='default SOC'):
            replay_episode(data, lambda _s, _a: 0, accountant=accountant,
                           initial_state=initial, economic_evaluation_config=configuration())


@pytest.mark.parametrize('mode', ('shore_pending', 'shore_charging', 'unknown'))
def test_independent_task_rejects_nononboard_rows_before_policy(mode):
    data = independent_episode((100.0,))
    data.operating_mode = (mode,)
    def forbidden_policy(_state, _actions):
        raise AssertionError('invalid independent rows cannot invoke the policy')
    with pytest.raises(ValueError, match='all rows to be ONBOARD'):
        replay_episode(data, forbidden_policy, accountant=EconomicMPC(nominal_cost_cny=1.0),
                       economic_evaluation_config=configuration())


@pytest.mark.parametrize('duration', (0.0, -1.0, float('nan'), float('inf'), True))
def test_interval_and_mask_reject_invalid_duration(duration):
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    with pytest.raises(ValueError, match='duration'):
        accountant.interval(AccountState(), 0.0, 100.0, duration_s=duration)
    with pytest.raises(ValueError, match='duration'):
        feasible_fc_actions(AccountState(), 100.0, accountant, duration_s=duration)
