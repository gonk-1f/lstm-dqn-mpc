from types import SimpleNamespace

import pytest

from v2.economics import ShoreEnergyClassification
from v3.control import AccountState, EconomicMPC
from v4.control import ACTION_KW, ReplayExecutionError, build_state, replay_episode


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
    assert [item.done for item in result.transitions] == [False, True, True]
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


def test_partial_shore_charge_carries_actual_soc_without_forcing_target():
    data = episode(("shore_charging", "onboard"), (0.0, 80.0), (-10.0, 0.0))
    result = replay_episode(
        data, lambda _state, feasible: min(feasible, key=lambda value: abs(value - 80)),
        accountant=NoSolveAccountant(), initial_state=AccountState(soc=0.5),
    )
    assert 0.5 < result.shore_blocks[0].end_soc < 0.6
    assert result.transitions[0].state[0] == pytest.approx(result.shore_blocks[0].end_soc)
    assert result.fc_power_kw_by_row[0] == 0.0


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


def test_actual_shore_and_terminal_surplus_never_get_modeled_recharge():
    actual = episode(("onboard", "shore_charging"), (100.0, 0.0), (0.0, -100.0))
    actual_result = replay_episode(actual, lambda _state, _actions: 0, accountant=NoSolveAccountant())
    assert actual_result.shore_blocks
    assert actual_result.modeled_terminal_settlement is None
    assert actual_result.final_state.soc < 0.6
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
