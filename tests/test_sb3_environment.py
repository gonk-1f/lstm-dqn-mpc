"""Small fixed samples compare the adapter with the original V6 executor."""
from types import SimpleNamespace

import numpy as np
import pytest

from v3.control import EconomicMPC
from v4.failure_replay import FailurePenalty
from v4.independent_economics import ZhouEvaluationConfig
from v4.v6_training import execute
from v6_sb3.environment import VesselEnergyEnv


def episode(loads):
    return SimpleNamespace(
        sample_id='sb3_fixed_sample', split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(float(i) for i in range(len(loads))), dt_s=(1.0,) * len(loads),
        duration_s=float(len(loads)))


def original(data, powers):
    commands = iter(powers)
    return execute(data, lambda state, candidates: next(commands),
                   EconomicMPC(nominal_cost_cny=1.0),
                   ZhouEvaluationConfig.from_formal_curve(), FailurePenalty.from_reference())


def test_gym_fixed_actions_match_original_v6_transition_fields():
    data = episode((100.0, 50.0, 40.0, 100.0))
    expected, status = original(data, (30, 90, 40, 0))
    env = VesselEnergyEnv(data)
    try:
        observation, _ = env.reset(seed=42)
        assert observation.dtype == np.float32
        with pytest.raises(ValueError, match='not legal'):
            env.step(0)  # -60 kW from FC off is not an executable command.
        for i, (action, reference) in enumerate(zip((9, 12, 1, 2), expected)):
            assert env.current_decision_state == reference.state
            observation, reward, terminated, truncated, info = env.step(action)
            actual = info['transition']
            assert actual == reference
            assert reward == reference.reward_cny
            assert actual.action_kw == reference.action_kw
            assert actual.actual_battery_kw == reference.actual_battery_kw
            assert actual.actual_soc == reference.actual_soc
            np.testing.assert_allclose(observation, reference.next_state, rtol=1e-7, atol=1e-7)
            assert terminated == reference.done
            assert not truncated
            previous_fc = reference.action_kw
            expected_indices = [(p - previous_fc + 60) // 10 for p in reference.next_feasible_actions]
            assert np.flatnonzero(info['next_action_mask']).tolist() == expected_indices
            if i < len(expected) - 1:
                assert env.current_decision_state == reference.next_state
        assert env.episode_status == status == 'completed'
        assert env.episode_transitions == expected
    finally:
        env.close()
    assert env._worker is None


def test_empty_successor_penalizes_last_actual_action_once():
    data = episode((100.0, 1400.0))
    expected, status = original(data, (60,))
    env = VesselEnergyEnv(data)
    try:
        env.reset()
        observation, reward, terminated, truncated, info = env.step(12)
        assert terminated and not truncated
        assert env.episode_status == status == 'increment_action_limited'
        assert len(env.episode_transitions) == 1
        assert env.last_transition == expected[0]
        assert reward == expected[0].reward_cny
        assert info['failure_penalty_equivalent_cny'] == env.penalty.amount
        assert not info['next_action_mask'].any()
        assert info['terminal_reason'] == 'failure_increment_action_limited'
        assert observation.tolist() == pytest.approx(expected[0].next_state, abs=1e-7)
        with pytest.raises(RuntimeError, match='nonterminal'):
            env.step(6)
    finally:
        env.close()


def test_budget_close_does_not_synthesize_terminal_and_prefix_restores_exactly():
    data = episode((100.0, 80.0, 90.0, 110.0))
    env = VesselEnergyEnv(data)
    restored = VesselEnergyEnv(data)
    try:
        env.reset()
        env.step(9)
        _, _, terminated, truncated, _ = env.step(7)
        snapshot = env.snapshot()
        transition = env.last_transition
        assert not terminated and not truncated and not transition.done
        assert transition.failure_penalty_equivalent_cny == 0
        env.close()
        restored.reset(options={'episode': data, 'action_prefix': snapshot['episode_actions']})
        assert restored.current_decision_state == tuple(snapshot['current_state'])
        assert restored.last_transition == transition
        assert restored.snapshot()['episode_actions'] == [9, 7]
    finally:
        env.close()
        restored.close()

