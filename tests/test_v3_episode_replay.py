from types import SimpleNamespace

import pytest

from v3.control import AccountState, EconomicMPC, MPCWeights
from v3.episode_replay import ReplayExecutionError, replay_episode


class CountingMPC(EconomicMPC):
    def __init__(self):
        super().__init__(nominal_cost_cny=8.927633172298517)
        self.solve_count = 0

    def solve(self, forecast_kw, reference_kw, state, weights):
        self.solve_count += 1
        return super().solve(forecast_kw, reference_kw, state, weights)


def episode(modes, loads, shore_powers):
    return SimpleNamespace(
        sample_id="synthetic_train",
        split="train",
        operating_mode=tuple(modes),
        load_kw=tuple(loads),
        battery_bus_kw=tuple(shore_powers),
    )


def test_explicit_mode_gate_pauses_policy_and_mpc_through_both_shore_modes():
    data = episode(
        ("onboard", "onboard", "shore_pending", "shore_charging", "onboard"),
        (100.0, 120.0, 0.0, 0.0, 80.0),
        (0.0, 0.0, -100.0, -100.0, 0.0),
    )
    mpc = CountingMPC()
    observed_states = []

    def policy(state):
        observed_states.append(state)
        return MPCWeights(4.13616998080572, 0.8)

    result = replay_episode(data, policy, mpc=mpc)

    assert result.onboard_steps == len(result.transitions) == len(observed_states) == mpc.solve_count == 3
    assert result.shore_steps == 2
    assert [item.done for item in result.transitions] == [False, True, True]
    assert result.fc_power_kw_by_row[2:4] == (0.0, 0.0)
    assert result.battery_bus_kw_by_row[2:4] == result.transitions[1].shore_accepted_battery_bus_kw
    assert result.transitions[1].shore_requested_battery_bus_kw == (-100.0, -100.0)
    assert result.transitions[2].state[1] == 0.0  # new voyage virtual zero-load boundary
    assert result.transitions[2].state[0] == pytest.approx(result.shore_blocks[0].end_soc)
    assert result.shore_blocks[0].end_soc < 0.6
    assert result.total_cost_cny == pytest.approx(-sum(t.reward_cny for t in result.transitions))
    assert result.soc_by_row[-1] == pytest.approx(result.final_state.soc)


def test_leading_shore_is_settled_without_dqn_action_and_soc_is_carried():
    data = episode(
        ("shore_pending", "shore_charging", "onboard"),
        (0.0, 0.0, 50.0),
        (-100.0, -100.0, 0.0),
    )
    calls = []

    def policy(state):
        calls.append(state)
        return MPCWeights(4.13616998080572, 0.8)

    result = replay_episode(data, policy, mpc=CountingMPC(), initial_state=AccountState(soc=0.5))

    assert len(calls) == 1
    assert result.shore_blocks[0].start_soc == 0.5
    assert 0.5 < result.shore_blocks[0].end_soc < 0.6
    assert calls[0][0] == pytest.approx(result.shore_blocks[0].end_soc)
    assert result.shore_blocks[0].ledger.shore_cost_cny > 0.0
    assert result.total_cost_cny > -result.transitions[0].reward_cny


def test_unresolved_mode_and_positive_shore_power_fail_before_any_action():
    calls = []

    def policy(state):
        calls.append(state)
        return MPCWeights(0.0, 0.0)

    with pytest.raises(ValueError, match="unresolved"):
        replay_episode(
            episode(("onboard", "unresolved"), (100.0, 0.0), (0.0, 0.0)),
            policy, mpc=CountingMPC(),
        )
    with pytest.raises(ValueError, match="shore"):
        replay_episode(
            episode(("onboard", "shore_charging"), (100.0, 0.0), (0.0, 10.0)),
            policy, mpc=CountingMPC(),
        )
    assert calls == []


def test_real_load_exceeding_battery_limit_is_reported_at_its_physical_row():
    data = episode(("onboard",), (2000.0,), (0.0,))
    with pytest.raises(ReplayExecutionError, match="row 0") as captured:
        replay_episode(data, lambda _state: MPCWeights(0.0, 0.0), mpc=CountingMPC())
    assert captured.value.row_index == 0
    assert captured.value.mode.value == "onboard"
    assert "battery power bounds" in str(captured.value.cause)
