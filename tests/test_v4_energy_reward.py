"""Synthetic contracts for the energy-only reward; no formal data are read."""
from math import fsum

import pytest

from test_v4_control import NoSolveAccountant, episode
from v2.economics import FORMAL_PRICE_CATALOG
from v3.control import AccountState
from v4.control import ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue


def test_energy_only_changes_reward_but_not_fixed_action_physics_or_full_ledger():
    data = episode(('onboard',) * 3, (100., 300., 100.), (0.,) * 3)
    results = []
    for mode in ('full_economic', 'energy_only'):
        actions = iter((100, 200, 100))
        results.append(replay_episode(data, lambda _s, _a: next(actions),
            accountant=NoSolveAccountant(), initial_state=AccountState(soc=.55),
            beta_soc=200., reward_cost_mode=mode))
    full, energy = results
    assert energy.final_state == full.final_state
    assert energy.total_ledger == full.total_ledger
    assert energy.modeled_terminal_settlement == full.modeled_terminal_settlement
    assert energy.soc_by_row == full.soc_by_row
    assert energy.fc_power_kw_by_row == full.fc_power_kw_by_row
    assert energy.battery_bus_kw_by_row == full.battery_bus_kw_by_row
    for old, new in zip(full.transitions, energy.transitions):
        ledger = new.executed_ledger
        excluded = ledger.fuel_cell_degradation_cost_cny + ledger.battery_degradation_cost_cny
        if new.shore_ledger is not None:
            excluded += new.shore_ledger.fuel_cell_degradation_cost_cny + new.shore_ledger.battery_degradation_cost_cny
        if new.modeled_terminal_ledger is not None:
            excluded += new.modeled_terminal_ledger.fuel_cell_degradation_cost_cny + new.modeled_terminal_ledger.battery_degradation_cost_cny
        assert new.reward_cny - old.reward_cny == pytest.approx(excluded)
        assert new.excluded_degradation_cost_cny == pytest.approx(excluded)
        assert old.excluded_degradation_cost_cny == 0.
        assert new.soc_soft_penalty_cny == pytest.approx(200. * (new.actual_soc - .6) ** 2)
        assert new.original_reward == new.reward_cny
        assert new.immediate_battery_energy_adjustment == new.terminal_correction == 0.
    terminal = energy.modeled_terminal_settlement
    assert terminal is not None
    assert fsum(t.reward_cny for t in energy.transitions) == pytest.approx(-(
        energy.total_ledger.h2_cost_cny + energy.total_ledger.shore_cost_cny
        + terminal.ledger.shore_cost_cny + energy.soc_soft_penalty_cny))


def _energy_replay(modes, loads, actions, *, soc=.55, beta=200.):
    powers = iter(actions)
    return replay_episode(episode(modes, loads, (0.,) * len(modes)),
        lambda _s, _a: next(powers), accountant=NoSolveAccountant(),
        initial_state=AccountState(soc=soc), beta_soc=beta, reward_cost_mode='energy_only')


def _energy_cost(result):
    terminal = result.modeled_terminal_settlement
    return (result.total_ledger.h2_cost_cny + result.total_ledger.shore_cost_cny
        + (0. if terminal is None else terminal.ledger.shore_cost_cny))


def test_pure_onboard_discharge_has_one_nonfree_modeled_terminal_electricity_charge():
    result = _energy_replay(('onboard',) * 3, (300.,) * 3, (0,) * 3, beta=0.)
    settlement = result.modeled_terminal_settlement
    assert settlement is not None
    assert settlement.grid_energy_kwh > 0. and settlement.ledger.shore_cost_cny > 0.
    assert settlement.ledger.battery_degradation_cost_cny > 0.
    assert result.total_ledger.shore_cost_cny == 0.
    assert sum(t.modeled_terminal_ledger is not None for t in result.transitions) == 1
    assert result.final_state.soc < .55  # Accounting-only terminal charge does not alter sailing state.
    assert result.transitions[-1].done
    assert fsum(t.reward_cny for t in result.transitions) == pytest.approx(-_energy_cost(result))


def test_more_battery_discharge_is_observed_as_more_terminal_electricity_cost():
    # Compare two legal fixed controls at identical initial SOC and load.
    battery_heavy = _energy_replay(('onboard',) * 3, (300.,) * 3, (0,) * 3, beta=0.)
    fc_heavy = _energy_replay(('onboard',) * 3, (300.,) * 3, (300,) * 3, beta=0.)
    low = battery_heavy.modeled_terminal_settlement
    high = fc_heavy.modeled_terminal_settlement
    assert low is not None and high is not None
    assert battery_heavy.final_state.soc < fc_heavy.final_state.soc
    assert low.ledger.shore_cost_cny > high.ledger.shore_cost_cny
    accountant = NoSolveAccountant()
    eta_chg, _ = accountant.efficiency.require_calibrated()
    delta_soc = fc_heavy.final_state.soc - battery_heavy.final_state.soc
    expected_extra_bill = (delta_soc * accountant.plant.battery_nominal_energy_kwh
        / eta_chg * FORMAL_PRICE_CATALOG.shore_cny_per_kwh)
    assert low.ledger.shore_cost_cny - high.ledger.shore_cost_cny == pytest.approx(
        expected_extra_bill, rel=1e-10, abs=1e-9)
    assert low.grid_energy_kwh - high.grid_energy_kwh == pytest.approx(
        delta_soc * accountant.plant.battery_nominal_energy_kwh / eta_chg)
    # The cost is deferred to the real final transition rather than charged every step.
    assert battery_heavy.transitions[0].reward_cny == battery_heavy.transitions[1].reward_cny == 0.
    assert battery_heavy.transitions[-1].reward_cny == pytest.approx(-low.ledger.shore_cost_cny)


def test_tail_shore_restores_soc_and_is_not_billed_again_as_terminal_settlement():
    result = _energy_replay(('onboard', 'shore_charging', 'shore_charging'),
        (300., 0., 0.), (100,))
    assert len(result.transitions) == len(result.shore_blocks) == 1
    assert result.final_state.soc == .6
    terminal = result.transitions[-1]
    assert terminal.done and terminal.terminal_reason == 'completed'
    assert terminal.shore_ledger == result.shore_blocks[0].ledger
    assert terminal.shore_ledger.shore_cost_cny > 0.
    assert terminal.shore_ledger.battery_degradation_cost_cny > 0.
    assert terminal.modeled_terminal_ledger is result.modeled_terminal_settlement is None
    assert result.shore_blocks[0].settlement_basis == 'modeled_fixed_target_soc_0.6'
    assert fsum(t.reward_cny for t in result.transitions) == pytest.approx(-(
        _energy_cost(result) + result.soc_soft_penalty_cny))


def test_cross_shore_uses_real_post_charge_successor_and_n8_reuses_one_source_fee():
    # Ten real decisions and one port call: the port has no virtual Q decision.
    modes = ('onboard',) * 2 + ('shore_charging',) + ('onboard',) * 8
    loads = (200.,) * 2 + (0.,) + (200.,) * 8
    result = _energy_replay(modes, loads, (100,) * 10)
    values = result.transitions
    assert len(values) == 10 and len(result.shore_blocks) == 1
    assert sum(t.shore_ledger is not None for t in values) == 1
    first_leg_last = values[1]
    assert first_leg_last.shore_ledger == result.shore_blocks[0].ledger
    assert not first_leg_last.done
    assert first_leg_last.next_state == values[2].state
    assert values[2].state[0] == .6
    assert values[2].state[1:4] == (0.,) * 3  # FC shut down at the modeled port call.
    assert values[2].state[5:] == (0., 0.)
    assert first_leg_last.next_feasible_actions == values[2].physical_feasible_actions
    original_ledger = result.total_ledger
    source_charge = first_leg_last.shore_ledger.shore_cost_cny
    assert source_charge == result.total_ledger.shore_cost_cny
    assert fsum(t.reward_cny for t in values) == pytest.approx(-(
        _energy_cost(result) + result.soc_soft_penalty_cny))
    agent = DirectPowerDDQN(n_step=8, gamma=1., reward_scale=.001,
        failure_terminal_quota=2, replay_sampling='prioritized')
    assert agent.remember_trajectory(values) == 10
    for i, item in enumerate(agent.replay):
        end = min(i + 8, len(values))
        assert item.reward_cny == pytest.approx(.001 * fsum(t.reward_cny for t in values[i:end]))
        assert item.bootstrap_steps == end - i
        assert item.done == values[end - 1].done
        assert item.next_state == values[end - 1].next_state
        assert item.next_feasible_indices == tuple(a // 10 for a in values[end - 1].next_feasible_actions)
    assert not agent.replay[0].done  # This eight-step target crosses SHORE and bootstraps later.
    assert agent.replay[0].next_state == values[8].state
    # Both first windows contain the same source observation, without another physical bill.
    assert sum(i <= 1 < min(i + 8, len(values)) for i in range(len(values))) == 2
    assert result.total_ledger == original_ledger
    assert result.total_ledger.shore_cost_cny == source_charge


def test_failure_has_one_separate_penalty_and_no_fictional_shore_or_terminal_recharge():
    accountant = NoSolveAccountant()
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',) * 2, (600., 1000.), (0.,) * 2),
            lambda _s, _a: 0, accountant=accountant,
            initial_state=AccountState(soc=.2135), beta_soc=200., reward_cost_mode='energy_only')
    error = caught.value
    assert error.failure_kind == 'no_feasible_action' and error.failure_cause == 'soc_limited'
    assert len(error.executed_transitions) == 1
    executed = error.executed_transitions[0]
    assert not executed.done
    penalty = FailurePenalty.from_reference()
    view = prepare_failure_replay(error, penalty=penalty,
        energy_value=BatteryEnergyValue.from_accountant(accountant), redistribute_battery_energy=False)
    terminal = view.transitions[-1]
    assert terminal.done and terminal.terminal_reason == 'failure_soc_limited'
    assert not terminal.next_feasible_actions
    assert terminal.shore_ledger is terminal.modeled_terminal_ledger is None
    assert terminal.executed_ledger == executed.executed_ledger
    assert terminal.terminal_correction == terminal.immediate_battery_energy_adjustment == 0.
    assert fsum(t.reward_cny for t in view.transitions) == pytest.approx(
        fsum(t.reward_cny for t in error.executed_transitions) - penalty.amount)
    assert sum(t.failure_penalty_equivalent_cny for t in view.transitions) == penalty.amount
    assert terminal.original_reward == executed.original_reward
    agent = DirectPowerDDQN(n_step=8, reward_scale=.001, failure_terminal_quota=2,
        replay_sampling='prioritized')
    assert agent.remember_trajectory(view.transitions) == 1
    assert agent.replay[0].done and not agent.replay[0].next_feasible_indices
    assert agent.replay[0].reward_cny == pytest.approx(.001 * terminal.reward_cny)
    assert agent.failure_terminal_replay_count == 1


def test_unknown_keeps_only_observed_n8_windows_and_has_no_fake_terminal_cost():
    with pytest.raises(ReplayExecutionError) as caught:
        _energy_replay(('onboard',) * 12 + ('unknown',), (200.,) * 12 + (0.,), (100,) * 12)
    error = caught.value
    values = error.executed_transitions
    assert error.failure_kind == 'data_truncation'
    assert len(values) == 12 and all(not t.done for t in values)
    assert all(t.shore_ledger is t.modeled_terminal_ledger is None for t in values)
    assert all(t.failure_penalty_equivalent_cny == t.terminal_correction == 0. for t in values)
    assert values[-1].terminal_reason == 'data_truncation'
    agent = DirectPowerDDQN(n_step=8, reward_scale=.001, failure_terminal_quota=2,
        replay_sampling='prioritized')
    with pytest.raises(ValueError, match='completed or explicitly failure'):
        agent.remember_trajectory(values)
    assert not agent.replay
    assert agent.remember_observed_prefix(values) == 4
    assert all(not item.done and item.terminal_reason is None for item in agent.replay)
    assert all(item.bootstrap_steps == 8 and item.experience_outcome == 'unresolved' for item in agent.replay)
    assert agent.economic_failure_replay_insertions == 0
    for i, item in enumerate(agent.replay):
        assert item.next_state == values[i + 8].state
        assert item.reward_cny == pytest.approx(.001 * fsum(t.reward_cny for t in values[i:i + 8]))


def test_invalid_cost_mode_is_rejected_before_executing_policy():
    calls = []
    with pytest.raises(ValueError, match='reward_cost_mode'):
        replay_episode(episode(('onboard',), (100.,), (0.,)),
            lambda _s, _a: calls.append(1) or 100, accountant=NoSolveAccountant(),
            reward_cost_mode='unapproved')
    assert not calls


@pytest.mark.parametrize('modes,loads,actions,previous', [
    (('onboard',) * 3, (100., 300., 100.), (100, 200, 100), 40.),
    (('onboard', 'shore_charging'), (200., 0.), (100,), 40.),
    (('onboard', 'shore_charging', 'onboard', 'onboard'),
     (200., 0., 200., 200.), (100, 200, 200), 40.),
    (('onboard',) * 3, (100.,) * 3, (100,) * 3, 100.),
])
def test_fc_ramp_is_one_onboard_reward_term_not_an_economic_charge(modes, loads, actions, previous):
    from v4.control import FC_RAMP_REWARD_CNY_PER_KW
    from v4.diagnostics import reward_totals, trajectory_record

    assert FC_RAMP_REWARD_CNY_PER_KW == pytest.approx(.11025)
    results = []
    for mode in ('energy_only', 'energy_with_fc_ramp'):
        powers = iter(actions)
        results.append(replay_episode(episode(modes, loads, (0.,) * len(modes)),
            lambda _s, _a: next(powers), accountant=NoSolveAccountant(),
            initial_state=AccountState(soc=.55, previous_fc_kw=previous),
            beta_soc=200., reward_cost_mode=mode))
    old, new = results
    assert (new.final_state, new.total_ledger, new.shore_blocks, new.modeled_terminal_settlement) == (
        old.final_state, old.total_ledger, old.shore_blocks, old.modeled_terminal_settlement)
    assert (new.soc_by_row, new.fc_power_kw_by_row, new.battery_bus_kw_by_row) == (
        old.soc_by_row, old.fc_power_kw_by_row, old.battery_bus_kw_by_row)
    for before, after in zip(old.transitions, new.transitions):
        expected = .11025 * abs(after.action_kw - before.state[1] * 600.)
        assert after.fc_ramp_penalty_cny == pytest.approx(expected)
        assert before.fc_ramp_penalty_cny == 0.
        assert after.reward_cny - before.reward_cny == pytest.approx(-expected)
        assert after.original_reward == after.reward_cny
        assert after.executed_ledger == before.executed_ledger
        assert after.excluded_degradation_cost_cny == before.excluded_degradation_cost_cny
        assert after.next_state == before.next_state
        assert after.next_feasible_actions == before.next_feasible_actions
    expected_total = fsum(t.fc_ramp_penalty_cny for t in new.transitions)
    assert reward_totals(new.transitions)['fc_ramp_penalty_cny'] == pytest.approx(expected_total)
    archived = trajectory_record('synthetic', new.transitions, completed=True, reward_scale=.001)
    assert fsum(t['fc_ramp_penalty_cny'] for t in archived['transitions']) == pytest.approx(expected_total)
    agent = DirectPowerDDQN(n_step=8, reward_scale=.001, failure_terminal_quota=2,
                            replay_sampling='prioritized')
    agent.remember_trajectory(new.transitions)
    for i, item in enumerate(agent.replay):
        assert item.reward_cny == pytest.approx(.001 * fsum(
            t.reward_cny for t in new.transitions[i:i+8]))


@pytest.mark.parametrize('mode', ['energy_only', 'energy_with_fc_ramp'])
def test_fc_ramp_failure_and_unknown_do_not_create_boundary_penalties(mode):
    accountant = NoSolveAccountant()
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',) * 2, (600., 1000.), (0.,) * 2),
            lambda _s, _a: 0, accountant=accountant,
            initial_state=AccountState(soc=.2135, previous_fc_kw=100.),
            beta_soc=200., reward_cost_mode=mode)
    error = caught.value
    assert error.failure_kind == 'no_feasible_action'
    executed = error.executed_transitions
    assert len(executed) == 1
    expected_ramp = 11.025 if mode == 'energy_with_fc_ramp' else 0.
    assert executed[0].fc_ramp_penalty_cny == pytest.approx(expected_ramp)
    penalty = FailurePenalty.from_reference()
    view = prepare_failure_replay(error, penalty=penalty,
        energy_value=BatteryEnergyValue.from_accountant(accountant), redistribute_battery_energy=False)
    terminal = view.transitions[-1]
    assert terminal.reward_cny == pytest.approx(executed[-1].reward_cny - penalty.amount)
    assert terminal.fc_ramp_penalty_cny == pytest.approx(expected_ramp)
    assert terminal.shore_ledger is terminal.modeled_terminal_ledger is None
    assert terminal.executed_ledger == executed[-1].executed_ledger
    agent = DirectPowerDDQN(n_step=8, reward_scale=.001, failure_terminal_quota=2,
                            replay_sampling='prioritized')
    agent.remember_trajectory(view.transitions)
    assert agent.replay[-1].reward_cny == pytest.approx(.001 * terminal.reward_cny)
    with pytest.raises(ReplayExecutionError) as truncated:
        replay_episode(episode(('onboard',) * 10 + ('unknown',), (200.,) * 10 + (0.,), (0.,) * 11),
            lambda _s, _a: 100, accountant=NoSolveAccountant(),
            initial_state=AccountState(previous_fc_kw=100.), beta_soc=200., reward_cost_mode=mode)
    prefix = truncated.value.executed_transitions
    assert truncated.value.failure_kind == 'data_truncation'
    assert all(t.fc_ramp_penalty_cny == t.failure_penalty_equivalent_cny == 0. for t in prefix)
    assert all(not t.done and t.shore_ledger is t.modeled_terminal_ledger is None for t in prefix)
    unknown_agent = DirectPowerDDQN(n_step=8, reward_scale=.001, replay_sampling='prioritized')
    assert unknown_agent.remember_observed_prefix(prefix) == 2
    assert all(not t.done and t.experience_outcome == 'unresolved' for t in unknown_agent.replay)
