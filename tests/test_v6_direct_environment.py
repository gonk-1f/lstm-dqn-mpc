"""Stepwise contracts for the direct V6 environment and frozen replay oracle."""
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from v3.control import AccountState, EconomicMPC
from v4.control import ReplayExecutionError, replay_episode
from v4.failure_replay import FailurePenalty, prepare_failure_replay
from v4.independent_economics import ZhouEvaluationConfig
from v4.reward_feedback import BatteryEnergyValue
from v6_sb3.environment import InfeasibleReset, VesselEnergyEnv


def episode(loads, sample_id='direct_fixed'):
    return SimpleNamespace(sample_id=sample_id, split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(float(i) for i in range(len(loads))), dt_s=(1.,) * len(loads),
        duration_s=float(len(loads)))


def direct_env(*args, **kwargs):
    assert importlib.util.find_spec('v6_sb3.direct_environment') is not None, (
        'the direct single-thread V6 environment must exist')
    from v6_sb3.direct_environment import DirectVesselEnv
    return DirectVesselEnv(*args, **kwargs)


def oracle(data, powers, mode):
    accountant = EconomicMPC(nominal_cost_cny=1.)
    commands = iter(powers)
    try:
        result = replay_episode(data, lambda state, candidates: next(commands), accountant=accountant,
            state_mode='history5', action_mode='increment13', reward_cost_mode=mode,
            economic_evaluation_config=ZhouEvaluationConfig.from_formal_curve())
        return result.transitions, 'completed'
    except ReplayExecutionError as error:
        view = prepare_failure_replay(error, penalty=FailurePenalty.from_reference(),
            energy_value=BatteryEnergyValue.from_accountant(accountant),
            redistribute_battery_energy=False)
        return view.transitions, error.failure_cause


@pytest.mark.parametrize('mode', ['zhou_economic_only', 'v6_energy_soc_ramp'])
@pytest.mark.parametrize('loads,powers', [
    ((100., 50., 40., 100.), (30, 90, 40, 0)),
    ((-.01, 0., 100., -.01), (0, 60, 120, 60)),
    ((0.,), (0,)),
    ((100., 1400.), (60,)),
    ((100., 1900.), (60,)),
])
def test_direct_matches_every_replay_transition_and_mask(mode, loads, powers):
    data = episode(loads)
    expected, status = oracle(data, powers, mode)
    env = direct_env(data, reward_mode=mode)
    try:
        observation, info = env.reset(seed=42)
        assert observation.dtype == np.float32
        assert info['sample_id'] == data.sample_id
        for transition in expected:
            assert env.current_decision_state == transition.state
            assert np.flatnonzero(env.action_masks()).tolist() == [
                (p - round(transition.state[2] * 600) + 60) // 10
                for p in transition.policy_candidate_actions]
            observation, reward, terminated, truncated, info = env.step(transition.action_index)
            assert env.last_transition == transition
            assert info['transition'] == transition
            assert reward == transition.reward_cny
            assert terminated is transition.done
            assert truncated is False
            np.testing.assert_allclose(observation, transition.next_state, rtol=1e-7, atol=1e-7)
            assert np.flatnonzero(env.action_masks()).tolist() == [
                (p - transition.action_kw + 60) // 10 for p in transition.next_feasible_actions]
        assert env.episode_status == status
        assert info['executed_steps'] == len(expected)
        with pytest.raises(RuntimeError, match='nonterminal'):
            env.step(6)
    finally:
        env.close()


def test_direct_history_mode_is_identical_to_queue_environment():
    data = episode((100., 50., 40., 100.))
    env = direct_env(data, reward_mode='v6_energy_soc_ramp')
    old = VesselEnergyEnv(data)
    try:
        np.testing.assert_array_equal(env.reset()[0], old.reset()[0])
        for action in (9, 12, 1, 2):
            actual, expected = env.step(action), old.step(action)
            assert actual[1:4] == expected[1:4]
            assert env.last_transition == old.last_transition
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(env.action_masks(), old.action_masks())
        assert env.episode_status == old.episode_status
    finally:
        env.close()
        old.close()


@pytest.mark.parametrize('loads,status', [((1400.,), 'increment_action_limited'),
                                         ((1900.,), 'structural_power')])
def test_infeasible_reset_does_not_fabricate_transition(loads, status):
    env = direct_env(episode(loads))
    with pytest.raises(InfeasibleReset) as error:
        env.reset()
    assert error.value.status == env.episode_status == status
    assert env.last_transition is None
    assert not env.action_masks().any()
    assert env.snapshot()['index'] == 0
    env.close()


def test_reset_is_fresh_and_invalid_actions_cannot_advance_physics():
    env = direct_env(episode((100., 80., 90., 110.)))
    initial = env.reset()[0]
    snapshot = env.snapshot()
    with pytest.raises(ValueError, match='not legal'):
        env.step(0)
    assert env.snapshot() == snapshot
    env.step(9)
    assert env.snapshot()['physical'] != snapshot['physical']
    np.testing.assert_array_equal(env.reset()[0], initial)
    assert env.snapshot()['physical'] == dict(soc=.6, previous_fc_kw=0., fc_loss_uv=0., battery_weighted_ah=0.)
    assert env.last_transition is None
    assert not hasattr(env, '_worker')
    assert not getattr(env, 'episode_transitions', ())
    env.close()


def test_snapshot_restores_account_state_without_replaying_prefix():
    class CountingAccountant(EconomicMPC):
        calls = 0

        def interval(self, *args, **kwargs):
            self.calls += 1
            return super().interval(*args, **kwargs)

    data = episode((100., 80., 90., 110., 0.))
    accountant = CountingAccountant(nominal_cost_cny=1.)
    env = direct_env(data)
    restored = direct_env(accountant=accountant)
    try:
        env.reset()
        for action in (9, 7):
            env.step(action)
        snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        observation, info = restored.restore(snapshot, data)
        assert accountant.calls == 0
        assert restored.current_decision_state == env.current_decision_state
        assert restored.snapshot() == snapshot
        assert 'episode_actions' not in snapshot
        assert info['restored_steps'] == 2
        np.testing.assert_array_equal(observation, np.asarray(env.current_decision_state, dtype=np.float32))
        for action in (6, 5, 3):
            assert restored.step(action)[1:4] == env.step(action)[1:4]
            assert restored.last_transition == env.last_transition
        terminal = env.snapshot()
        restored.restore(terminal, data)
        assert restored.episode_status == 'completed'
        assert not restored.action_masks().any()
    finally:
        env.close()
        restored.close()


@pytest.mark.parametrize('mode', ['zhou_economic_only', 'v6_energy_soc_ramp'])
def test_long_causal_windows_and_soc_failure_match_replay(mode):
    # Reach the real SOC boundary at maximum FC power, ensuring the next mask
    # is physically empty rather than simply limited by the incremental grid.
    data = episode((1248.,) * 1600)
    powers = (*range(60, 601, 60), *((600,) * 1590))
    expected, status = oracle(data, powers, mode)
    assert status == 'soc_limited'
    assert len(expected) > 300
    env = direct_env(data, reward_mode=mode)
    try:
        env.reset()
        for transition in expected:
            env.step(transition.action_index)
            assert env.last_transition == transition
        assert env.episode_status == 'soc_limited'
        assert env.last_transition.terminal_reason == 'failure_soc_limited'
        assert env.last_transition.failure_penalty_equivalent_cny == env.penalty.amount
    finally:
        env.close()


def test_snapshot_size_does_not_grow_with_episode_steps_and_validates_identity():
    data = episode(tuple(float(i % 151) for i in range(350)))
    env = direct_env(data)
    other = direct_env(reward_mode='v6_energy_soc_ramp')
    try:
        env.reset()
        early_size = len(json.dumps(env.snapshot()))
        for _ in range(320):
            env.step(6)
        snapshot = env.snapshot()
        assert len(json.dumps(snapshot)) < early_size + 250
        assert len(snapshot['history']) == len(snapshot['fc_history']) == 3
        assert not hasattr(env, 'episode_actions')
        assert not hasattr(env, 'episode_transitions')
        with pytest.raises(ValueError, match='configuration'):
            other.restore(snapshot, data)
    finally:
        env.close()
        other.close()


@pytest.mark.parametrize('mode', ['zhou_economic_only', 'v6_energy_soc_ramp'])
def test_varying_load_history_across_full_windows_matches_replay(mode):
    data = episode(tuple(-.01 if i % 19 == 0 else float(i % 211) for i in range(365)))
    expected, status = oracle(data, (60,) * 365, mode)
    env = direct_env(data, reward_mode=mode)
    try:
        env.reset()
        for transition in expected:
            env.step(transition.action_index)
            assert env.last_transition == transition
        assert env.episode_status == status == 'completed'
    finally:
        env.close()


def test_failure_boundary_snapshot_restores_terminal_status_without_another_transition():
    data = episode((100., 1900.))
    env = direct_env(data)
    restored = direct_env()
    try:
        env.reset()
        env.step(12)
        snapshot = env.snapshot()
        observation, _ = restored.restore(snapshot, data)
        assert restored.snapshot() == snapshot
        assert restored.episode_status == 'structural_power'
        assert restored.last_transition is None
        assert not restored.action_masks().any()
        with pytest.raises(RuntimeError, match='nonterminal'):
            restored.step(6)
    finally:
        env.close()
        restored.close()


def test_completed_economic_reward_equals_signed_total_and_failure_penalty_is_separate():
    env = direct_env(episode((0., 0., 0.)))
    try:
        env.reset()
        transitions = []
        for action in (12, 6, 6):
            env.step(action)
            transitions.append(env.last_transition)
        economic = sum(t.executed_ledger.total_cost_cny + t.soc_equivalent_hydrogen_cost_cny
                       for t in transitions)
        assert sum(t.reward_cny for t in transitions) == pytest.approx(-economic)
        assert all(t.soc_soft_penalty_cny == t.fc_ramp_penalty_cny == t.terminal_correction == 0.
                   for t in transitions)
        env.reset(options={'episode': episode((100., 1900.))})
        _, reward, done, _, info = env.step(12)
        t = env.last_transition
        assert done
        assert reward == pytest.approx(-t.executed_ledger.total_cost_cny
            - t.soc_equivalent_hydrogen_cost_cny - env.penalty.amount)
        assert info['failure_penalty_equivalent_cny'] == env.penalty.amount
    finally:
        env.close()
