"""Conservative FC continuation gate; physical mask and ledger stay unchanged."""

import pytest
import torch

from test_v4_control import NoSolveAccountant, episode
from v3.control import AccountState
from v4.control import ReplayExecutionError, feasible_fc_actions, replay_episode
from v4.dqn import DirectPowerDDQN, masked_double_dqn_targets
from v4.diagnostics import trajectory_record
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.reward_feedback import BatteryEnergyValue


def test_active_fc_cannot_voluntarily_stop_and_next_td_mask_matches():
    data = episode(("onboard", "onboard", "onboard"), (100., 100., 100.), (0.,) * 3)
    seen = []

    def policy(_state, actions):
        seen.append(actions)
        return 100 if len(seen) == 1 else min(actions)

    replay = replay_episode(data, policy, accountant=NoSolveAccountant())
    assert 0 in feasible_fc_actions(AccountState(soc=.6, previous_fc_kw=100.), 100., NoSolveAccountant())
    assert 0 in seen[0]
    assert all(0 not in mask for mask in seen[1:])
    assert replay.transitions[0].next_feasible_actions == seen[1]
    assert replay.transitions[1].next_feasible_actions == seen[2]
    assert replay.transitions[-1].next_feasible_actions == ()
    agent = DirectPowerDDQN(seed=42, n_step=1)
    agent.remember_trajectory(replay.transitions)
    assert tuple(index * 10 for index in agent.replay[0].next_feasible_indices) == seen[1]

    long = replay_episode(episode(("onboard",) * 10, (100.,) * 10, (0.,) * 10),
                          lambda _state, actions: 100, accountant=NoSolveAccountant())
    multi = DirectPowerDDQN(seed=42, n_step=8)
    multi.remember_trajectory(long.transitions)
    assert multi.replay[0].bootstrap_steps == 8
    assert not multi.replay[0].done
    assert tuple(index * 10 for index in multi.replay[0].next_feasible_indices) == (
        long.transitions[7].next_feasible_actions)
    assert 0 not in long.transitions[7].next_feasible_actions


def test_forced_stop_and_off_state_keep_zero_candidate():
    accountant = NoSolveAccountant()
    physical = AccountState(soc=.79995, previous_fc_kw=100.)
    assert feasible_fc_actions(physical, 0., accountant) == (0,)
    seen = []
    forced = replay_episode(episode(("onboard",), (0.,), (0.,)),
                            lambda _state, actions: seen.append(actions) or 0,
                            accountant=accountant, initial_state=physical)
    assert seen == [(0,)]
    record = trajectory_record("synthetic_train", forced.transitions, completed=True)
    assert record["transitions"][0]["fc_stop_reason"] == "forced_no_positive_action"
    assert record["transitions"][0]["physical_feasible_actions_kw"] == [0]

    seen.clear()
    replay_episode(episode(("onboard", "onboard"), (100., 100.), (0., 0.)),
                   lambda _state, actions: seen.append(actions) or 0,
                   accountant=accountant)
    assert len(seen) == 2 and 0 in seen[0] and 0 in seen[1]


@pytest.mark.parametrize("epsilon", (0., 1.))
def test_greedy_and_exploration_use_same_strategy_candidates(epsilon):
    agent = DirectPowerDDQN(seed=23)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
    actions = []

    def policy(state, mask):
        actions.append(mask)
        return agent.select_power(state, mask, epsilon=epsilon)

    replay_episode(episode(("onboard", "onboard"), (100., 100.), (0., 0.)),
                   policy, accountant=NoSolveAccountant(),
                   initial_state=AccountState(previous_fc_kw=100.))
    assert all(0 not in mask for mask in actions)


def test_shore_resets_candidate_gate_and_leaves_old_fixed_ledger_unchanged():
    data = episode(("onboard", "onboard", "shore_charging", "onboard", "onboard"),
                   (100., 120., 0., 120., 140.), (0., 0., -100., 0., 0.))
    sequence = iter((100, 100, 120, 120))
    masks = []

    def policy(_state, actions):
        masks.append(actions)
        value = next(sequence)
        assert value in actions
        return value

    replay = replay_episode(data, policy, accountant=NoSolveAccountant(),
                            beta_soc=500., redistribute_battery_energy=True)
    assert 0 in masks[0] and 0 not in masks[1]
    assert 0 in masks[2] and 0 not in masks[3]
    assert [item.actual_soc for item in replay.transitions] == pytest.approx(
        [.6, .5997188484030589, .6, .5997188484030589])
    ledger = replay.total_ledger
    assert (ledger.h2_cost_cny, ledger.fuel_cell_degradation_cost_cny,
            ledger.battery_degradation_cost_cny, ledger.shore_cost_cny) == pytest.approx(
        (3.821341624345254, 757.6649999999998, .10476266185987847, .20313942751617078))
    assert replay.transitions[1].next_feasible_actions == ()
    assert replay.transitions[3].next_feasible_actions == ()


def test_shore_pending_and_charging_do_not_call_policy_and_reset_departure():
    data = episode(("onboard", "shore_pending", "shore_charging", "onboard"),
                   (100., 0., 0., 100.), (0., 0., -100., 0.))
    seen = []

    def policy(state, actions):
        seen.append((state, actions))
        return 100

    replay = replay_episode(data, policy, accountant=NoSolveAccountant())
    assert len(seen) == 2
    assert all(0 in actions for _state, actions in seen)
    assert all(state[7] == 1. for state, _actions in seen)
    assert len(replay.shore_blocks) == 1
    assert (replay.shore_blocks[0].start_index,
            replay.shore_blocks[0].end_index_exclusive) == (1, 3)
    assert replay.transitions[0].next_feasible_actions == ()


def test_empty_physical_mask_remains_failure_terminal_without_bootstrap():
    accountant = NoSolveAccountant()
    data = episode(("onboard", "onboard"), (1000., 1000.), (0., 0.))
    with pytest.raises(ReplayExecutionError) as captured:
        replay_episode(data, lambda _state, mask: 0, accountant=accountant,
                       initial_state=AccountState(soc=.215))
    error = captured.value
    assert error.failure_kind == "no_feasible_action"
    assert error.executed_transitions
    prepared = prepare_failure_replay(error, penalty=FailurePenalty.from_reference(scale=1.),
        energy_value=BatteryEnergyValue.from_accountant(accountant),
        redistribute_battery_energy=False)
    assert prepared.transitions[-1].done
    assert prepared.transitions[-1].next_feasible_actions == ()
    agent = DirectPowerDDQN(seed=42, n_step=8)
    agent.remember_trajectory(prepared.transitions)
    assert all(item.done and not item.next_feasible_indices for item in agent.replay)
    terminal = agent.replay[-1]
    target = masked_double_dqn_targets(
        torch.tensor([[terminal.reward_cny]]), torch.zeros(1, 61), torch.zeros(1, 61),
        torch.ones(1, 1), next_action_masks=torch.zeros(1, 61, dtype=torch.bool), gamma=1.)
    assert target.item() == pytest.approx(terminal.reward_cny, rel=1e-6)
