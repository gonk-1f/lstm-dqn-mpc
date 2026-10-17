from math import fsum
from types import SimpleNamespace

import pytest

from v3.control import AccountState, EconomicMPC
from v4.control import ReplayExecutionError, feasible_fc_actions, replay_episode
from v4.independent_economics import ZhouEvaluationConfig


def episode(loads, durations=None):
    durations = tuple(durations or (30.0,) * len(loads))
    times, elapsed = [], 0.0
    for duration in durations:
        times.append(elapsed)
        elapsed += duration
    return SimpleNamespace(
        sample_id='synthetic_increment', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(times), dt_s=durations,
    )


def execute(data, powers, **options):
    commands = iter(powers)
    return replay_episode(
        data, lambda _state, _actions: next(commands),
        accountant=EconomicMPC(nominal_cost_cny=1.0),
        state_mode='minimal3', economic_evaluation_config=ZhouEvaluationConfig(0.5),
        beta_soc=200.0, **options,
    )


def test_increment_actions_keep_exact_indices_and_successor_masks():
    result = execute(episode((100.0,) * 5), (60, 120, 90, 30, 0),
                     action_mode='increment7', reward_cost_mode='zhou_joint')
    assert [t.action_delta_kw for t in result.transitions] == [60, 60, -30, -60, -30]
    assert [t.action_index for t in result.transitions] == [6, 6, 1, 0, 1]
    assert result.transitions[0].policy_candidate_actions == (0, 10, 30, 60)
    for left, right in zip(result.transitions, result.transitions[1:]):
        assert left.next_state == right.state
        assert left.next_feasible_actions == right.policy_candidate_actions
        assert right.state[2] * 600.0 == pytest.approx(left.action_kw)
    assert result.transitions[-1].done
    assert result.transitions[-1].next_feasible_actions == ()
    assert result.shore_blocks == ()
    assert result.modeled_terminal_settlement is None


def test_joint_reward_telescopes_to_zhou_cost_with_short_tail_and_same_physics():
    data = episode((100.0, 100.0, 40.0, 100.0, 100.0), (30.0,) * 4 + (7.0,))
    powers = (60, 120, 90, 30, 0)
    result = execute(data, powers, action_mode='increment7', reward_cost_mode='zhou_joint')
    baseline = execute(data, powers, reward_cost_mode='control_only')
    assert result.total_ledger == baseline.total_ledger
    assert result.final_state == baseline.final_state
    assert result.economic_evaluation == baseline.economic_evaluation
    corrections = []
    for transition in result.transitions:
        expected_correction = ((transition.state[0] - transition.actual_soc) * 624.0
                               / (0.5 * (120.0 / 3.6)) * 21.9)
        corrections.append(expected_correction)
        ramp = (0.11025 / 60.0) * transition.action_delta_kw ** 2
        soc = 200.0 * (transition.actual_soc - 0.6) ** 2
        assert transition.soc_equivalent_hydrogen_cost_cny == pytest.approx(expected_correction)
        assert transition.fc_ramp_penalty_cny == pytest.approx(ramp)
        assert transition.reward_cny == pytest.approx(-(
            transition.executed_ledger.total_cost_cny + expected_correction + ramp + soc))
        assert transition.excluded_degradation_cost_cny == 0.0
        assert transition.shore_ledger is None and transition.modeled_terminal_ledger is None
    assert min(corrections) < 0.0 < max(corrections)
    assert fsum(corrections) == pytest.approx(
        result.economic_evaluation.soc_equivalent_hydrogen_mass_kg * 21.9)
    assert -fsum(t.reward_cny for t in result.transitions) == pytest.approx(
        result.comparable_cost_cny + fsum(t.fc_ramp_penalty_cny + t.soc_soft_penalty_cny
                                        for t in result.transitions))


def test_no_increment_candidate_does_not_fallback_to_absolute_action():
    accountant = EconomicMPC(nominal_cost_cny=1.0)
    assert 200 in feasible_fc_actions(AccountState(), 1400.0, accountant)
    called = []
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode((1400.0,)), lambda _s, _a: called.append(True) or 200,
                       accountant=accountant, state_mode='minimal3',
                       economic_evaluation_config=ZhouEvaluationConfig(0.5),
                       action_mode='increment7', reward_cost_mode='zhou_joint')
    assert not called
    assert caught.value.failure_kind == 'no_feasible_action'
    assert caught.value.failure_cause == 'increment_action_limited'
    assert caught.value.executed_transitions == ()


def test_increment_failure_preserves_real_prefix_and_empty_successor_mask():
    with pytest.raises(ReplayExecutionError) as caught:
        execute(episode((100.0, 1400.0)), (60,), action_mode='increment7',
                reward_cost_mode='zhou_joint')
    assert caught.value.row_index == 1
    assert caught.value.failure_cause == 'increment_action_limited'
    prefix = caught.value.executed_transitions
    assert len(prefix) == 1 and prefix[0].action_kw == 60
    assert not prefix[0].done and prefix[0].next_feasible_actions == ()
    assert prefix[0].failure_penalty_equivalent_cny == 0.0


def test_increment_rejects_non_increment_commands_without_clipping():
    with pytest.raises(ReplayExecutionError, match='candidate mask'):
        execute(episode((100.0,)), (100,), action_mode='increment7',
                reward_cost_mode='zhou_joint')


def test_joint_reward_rejects_old_redistribution():
    with pytest.raises(ValueError, match='redistribution'):
        execute(episode((100.0,)), (60,), action_mode='increment7',
                reward_cost_mode='zhou_joint', redistribute_battery_energy=True)


def test_increment_failure_attaches_once_to_real_action_and_one_step_replay():
    from v4.dqn import DirectPowerDDQN
    from v4.failure_replay import FailurePenalty, prepare_failure_replay
    from v4.reward_feedback import BatteryEnergyValue

    with pytest.raises(ReplayExecutionError) as caught:
        execute(episode((100.0, 1400.0)), (60,), action_mode='increment7',
                reward_cost_mode='zhou_joint')
    error = caught.value
    original = error.executed_transitions[0]
    penalty = FailurePenalty.from_reference(scale=1.0)
    prepared = prepare_failure_replay(
        error, penalty=penalty, redistribute_battery_energy=False,
        energy_value=BatteryEnergyValue.from_accountant(EconomicMPC(nominal_cost_cny=1.0)))
    assert len(prepared.transitions) == 1
    last = prepared.transitions[0]
    assert last.done and last.next_feasible_actions == ()
    assert last.terminal_reason == 'failure_increment_action_limited'
    assert last.reward_cny == pytest.approx(original.reward_cny - penalty.amount)
    assert last.failure_penalty_equivalent_cny == penalty.amount
    assert last.executed_ledger == original.executed_ledger
    assert last.state == original.state and last.next_state == original.next_state
    assert last.action_kw == 60 and last.action_index == 6 and last.action_delta_kw == 60
    assert last.shore_ledger is None and last.modeled_terminal_ledger is None
    assert last.terminal_correction == 0.0
    agent = DirectPowerDDQN(seed=42, state_mode='minimal3', action_mode='increment7',
                            n_step=1, gamma=1.0, reward_scale=0.001,
                            failure_terminal_quota=2, replay_sampling='prioritized')
    assert agent.remember_trajectory(prepared.transitions) == 1
    item = agent.replay[0]
    assert item.done and item.bootstrap_steps == 1 and item.next_feasible_indices == ()
    assert item.action_index == 6
    assert item.reward_cny == pytest.approx(0.001 * (original.reward_cny - penalty.amount))
    assert item.failure_penalty_equivalent_cny == penalty.amount
    assert agent.failure_terminal_replay_count == 1
