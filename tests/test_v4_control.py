from types import SimpleNamespace

import pytest

from v2.economics import ShoreEnergyClassification
from v3.control import AccountState, EconomicMPC
from v4.control import ACTION_KW, ReplayExecutionError, build_state, replay_episode, soc_deviation_squared


def episode(modes, loads, shore):
    return SimpleNamespace(
        sample_id="synthetic_train", split="train",
        operating_mode=tuple(modes), load_kw=tuple(loads),
        battery_bus_kw=tuple(shore),
    )


class NoSolveAccountant(EconomicMPC):
    def __init__(self):
        super().__init__(nominal_cost_cny=1.0)

    def solve(self, *args, **kwargs):
        raise AssertionError("direct power control must not solve MPC")


def test_action_grid_and_causal_eight_feature_state():
    accountant = NoSolveAccountant()
    assert ACTION_KW == tuple(range(0, 601, 10))
    first = build_state(AccountState(soc=0.6), (120.0,), accountant, departure=True)
    assert len(first) == 8
    assert first == (0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0)
    later = build_state(AccountState(soc=0.5, previous_fc_kw=100.0), (0.0, 120.0), accountant, departure=False)
    assert later[:5] == pytest.approx((0.5, 0.2, 0.2, 0.0, 1.0 / 6.0))
    assert later[7] == 0.0


def test_mode_gate_shore_settlement_and_next_voyage_soc_carry():
    data = episode(
        ("onboard", "onboard", "shore_pending", "shore_charging", "onboard"),
        (100.0, 120.0, 0.0, 0.0, 80.0),
        (0.0, 0.0, -100.0, -100.0, 0.0),
    )
    seen = []

    def policy(state, feasible_actions):
        seen.append(state)
        assert 100 in feasible_actions
        return 100

    result = replay_episode(data, policy, accountant=NoSolveAccountant())
    assert len(seen) == len(result.transitions) == 3
    assert [item.done for item in result.transitions] == [False, False, True]
    assert result.fc_power_kw_by_row == (100.0, 100.0, 0.0, 0.0, 100.0)
    assert result.transitions[0].actual_battery_kw == 0.0
    assert result.transitions[1].actual_battery_kw == 20.0
    assert result.transitions[1].shore_ledger is not None
    assert result.transitions[1].shore_ledger.fuel_cell_degradation_cost_cny > 0
    assert result.transitions[1].reward_cny == pytest.approx(-(
        result.transitions[1].executed_ledger.total_cost_cny
        + result.transitions[1].shore_ledger.total_cost_cny
    ))
    assert result.transitions[2].state[0] == pytest.approx(result.shore_blocks[0].end_soc)
    assert result.transitions[2].state[7] == 1.0
    assert result.shore_blocks[0].end_soc == 0.6
    assert result.transitions[0].next_state == result.transitions[1].state
    assert result.transitions[1].next_state == result.transitions[2].state
    assert result.transitions[1].next_feasible_actions == result.transitions[2].policy_candidate_actions
    assert result.shore_blocks[0].settlement_basis == 'modeled_fixed_target_soc_0.6'
    assert result.shore_blocks[0].accepted_battery_bus_kw == ()
    assert result.total_cost_cny == pytest.approx(-sum(item.reward_cny for item in result.transitions))


def test_invalid_action_or_actual_battery_overload_fails_with_row_index():
    short = episode(("onboard",), (100.0,), (0.0,))
    with pytest.raises(ReplayExecutionError, match="row 0"):
        replay_episode(short, lambda _state, _actions: 15, accountant=NoSolveAccountant())
    overloaded = episode(("onboard",), (2000.0,), (0.0,))
    with pytest.raises(ReplayExecutionError, match="row 0") as captured:
        replay_episode(overloaded, lambda _state, _actions: 0, accountant=NoSolveAccountant())
    assert "battery power" in str(captured.value.cause)


def test_unresolved_mode_is_rejected_before_policy_call():
    called = []
    data = episode(("onboard", "unresolved"), (100.0, 0.0), (0.0, 0.0))
    with pytest.raises(ValueError, match="unresolved"):
        replay_episode(data, lambda state, actions: called.append(state) or 0, accountant=NoSolveAccountant())
    assert not called


def test_first_decision_sees_current_measured_load_and_feasible_grid():
    data = episode(("onboard",), (100.0,), (0.0,))
    seen = []

    def policy(state, feasible_actions):
        seen.append((state, feasible_actions))
        return 100

    replay_episode(data, policy, accountant=NoSolveAccountant())
    state, feasible = seen[0]
    assert state[1] == pytest.approx(100.0 / 600.0)
    assert state[7] == 1.0
    assert 100 in feasible
    assert 0 in feasible


def test_modeled_shore_reaches_target_independent_of_recorded_request():
    data = episode(("shore_charging", "onboard"), (0.0, 80.0), (-10.0, 0.0))
    result = replay_episode(
        data, lambda _state, feasible: min(feasible, key=lambda value: abs(value - 80)),
        accountant=NoSolveAccountant(), initial_state=AccountState(soc=0.5),
    )
    assert result.shore_blocks[0].end_soc == pytest.approx(0.6)
    assert result.transitions[0].state[0] == pytest.approx(result.shore_blocks[0].end_soc)
    assert result.fc_power_kw_by_row[0] == 0.0
    assert result.shore_blocks[0].ledger.shore_cost_cny > 0
    assert result.shore_blocks[0].ledger.battery_degradation_cost_cny > 0
    other = replay_episode(
        episode(('shore_charging','onboard'), (0.,80.), (-500.,0.)),
        lambda _state, feasible: min(feasible, key=lambda value: abs(value-80)),
        accountant=NoSolveAccountant(), initial_state=AccountState(soc=.5),
    )
    assert other.shore_blocks[0].ledger == result.shore_blocks[0].ledger
    assert other.transitions[0].state == result.transitions[0].state


def test_modeled_shore_keeps_soc_above_target_without_grid_purchase():
    result = replay_episode(
        episode(('shore_charging','onboard'), (0.,0.), (-624.,0.)),
        lambda _state, feasible: 0, accountant=NoSolveAccountant(),
        initial_state=AccountState(soc=.7),
    )
    assert result.shore_blocks[0].end_soc == pytest.approx(.7)
    assert result.shore_blocks[0].ledger.shore_cost_cny == 0
    assert result.shore_blocks[0].ledger.battery_degradation_cost_cny == 0


def test_terminal_deficit_has_separate_modeled_grid_and_battery_degradation():
    data = episode(("onboard",), (100.0,), (0.0,))
    result = replay_episode(data, lambda _state, _actions: 0, accountant=NoSolveAccountant())

    settlement = result.modeled_terminal_settlement
    assert settlement is not None
    assert settlement.classification is ShoreEnergyClassification.MODELED
    assert result.final_state.soc == pytest.approx(result.transitions[-1].actual_soc)
    assert settlement.grid_energy_kwh == pytest.approx(
        (0.6 - result.final_state.soc) * 624.0 / 0.95
    )
    assert settlement.ledger.shore_cost_cny == pytest.approx(settlement.grid_energy_kwh * 1.10)
    assert settlement.ledger.battery_degradation_cost_cny > 0
    assert settlement.ledger.fuel_cell_degradation_cost_cny == 0
    assert result.total_ledger.shore_cost_cny == 0
    assert result.comparable_cost_cny == pytest.approx(
        result.total_cost_cny + settlement.ledger.total_cost_cny
    )
    assert result.transitions[-1].reward_cny == pytest.approx(-result.comparable_cost_cny)


def test_modeled_shore_and_terminal_surplus_never_get_duplicate_recharge():
    actual = episode(("onboard", "shore_charging"), (100.0, 0.0), (0.0, -100.0))
    actual_result = replay_episode(actual, lambda _state, _actions: 0, accountant=NoSolveAccountant())
    assert actual_result.shore_blocks
    assert actual_result.modeled_terminal_settlement is None
    assert actual_result.final_state.soc == pytest.approx(0.6)
    assert actual_result.total_ledger.shore_cost_cny > 0
    assert actual_result.comparable_cost_cny == pytest.approx(actual_result.total_cost_cny)

    surplus = episode(("onboard",), (100.0,), (0.0,))
    surplus_result = replay_episode(surplus, lambda _state, _actions: 600, accountant=NoSolveAccountant())
    assert surplus_result.final_state.soc > 0.6
    assert surplus_result.modeled_terminal_settlement is None


def test_supply_failure_exposes_only_executed_prefix_and_failure_outcome():
    data = episode(("onboard", "onboard"), (1000.0, 1000.0), (0.0, 0.0))
    with pytest.raises(ReplayExecutionError, match="row 1") as captured:
        replay_episode(
            data, lambda _state, _actions: 0,
            accountant=NoSolveAccountant(), initial_state=AccountState(soc=0.215),
        )
    failure = captured.value
    assert failure.failure_kind == "no_feasible_action"
    assert len(failure.executed_transitions) == 1
    executed = failure.executed_transitions[0]
    assert executed.action_kw == 0
    assert executed.reward_cny == pytest.approx(-executed.executed_ledger.total_cost_cny)
    assert executed.actual_soc >= 0.2
    assert executed.done is False
    assert executed.next_feasible_actions == ()


@pytest.mark.parametrize(("soc", "expected"), (
    (0.2, 0.04), (0.3, 0.01), (0.4, 0.0), (0.5, 0.0),
    (0.6, 0.0), (0.7, 0.01), (0.8, 0.04),
))
def test_soc_soft_penalty_has_exact_piecewise_boundaries(soc, expected):
    assert soc_deviation_squared(soc) == pytest.approx(expected)


def test_soc_soft_penalty_is_onboard_only_and_preserves_actual_shore_cost():
    data = episode(("onboard", "shore_charging"), (100.0, 0.0), (0.0, -100.0))
    kwargs = dict(accountant=NoSolveAccountant(), initial_state=AccountState(soc=0.35))
    baseline = replay_episode(data, lambda _state, _actions: 0, beta_soc=0.0, **kwargs)
    shaped = replay_episode(data, lambda _state, _actions: 0, beta_soc=1000.0, **kwargs)
    onboard_soc = shaped.transitions[0].actual_soc
    expected_penalty = 1000.0 * (0.4 - onboard_soc) ** 2
    assert shaped.transitions[0].soc_soft_penalty_cny == pytest.approx(expected_penalty)
    assert shaped.transitions[0].reward_cny == pytest.approx(
        baseline.transitions[0].reward_cny - expected_penalty
    )
    assert shaped.total_cost_cny == pytest.approx(baseline.total_cost_cny)
    assert shaped.shore_blocks[0].end_soc == pytest.approx(baseline.shore_blocks[0].end_soc)
    assert shaped.soc_soft_penalty_cny == pytest.approx(expected_penalty)


def test_soc_soft_penalty_does_not_replace_modeled_terminal_settlement():
    data = episode(("onboard",), (100.0,), (0.0,))
    result = replay_episode(
        data, lambda _state, _actions: 0, accountant=NoSolveAccountant(),
        initial_state=AccountState(soc=0.35), beta_soc=1000.0,
    )
    assert result.modeled_terminal_settlement is not None
    assert result.transitions[-1].reward_cny == pytest.approx(
        -result.comparable_cost_cny - result.soc_soft_penalty_cny
    )
