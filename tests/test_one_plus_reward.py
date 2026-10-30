"""Explicit one-plus reward contracts over synthetic one-second voyages."""

import importlib
import importlib.util
import json
from math import fsum, nextafter
from types import SimpleNamespace

import numpy as np
import pytest

from v2.economics import RawCnyIntervalLedger
from v3.control import EconomicMPC
from v4.failure_replay import FailurePenalty
from v6_sb3.direct_environment import DirectVesselEnv
from v6_sb3.environment import ACTION_DELTAS, InfeasibleReset


MODE = 'inverse_one_plus_economic'


def rewards_module():
    assert importlib.util.find_spec('v6_sb3.rewards') is not None, (
        'the explicit inverse_one_plus_economic reward implementation is missing')
    return importlib.import_module('v6_sb3.rewards')


def episode(loads, sample_id='one_plus_synthetic'):
    return SimpleNamespace(sample_id=sample_id, split='train', task_mode='independent_voyage',
        operating_mode=('onboard',) * len(loads), load_kw=tuple(loads),
        time_s=tuple(float(i) for i in range(len(loads))), dt_s=(1.,) * len(loads),
        duration_s=float(len(loads)))


@pytest.mark.parametrize('cost,expected', [(0., 1.), (1., .5), (3., .25), (-.5, 2.),
                                         (-0.00000068368235145, 1. / (1. - 0.00000068368235145))])
def test_exact_one_plus_map_includes_zero_and_negative_cost(cost, expected):
    assert rewards_module().inverse_one_plus_economic_reward(cost) == expected


def test_one_plus_uses_no_cost_floor_and_remains_unbounded_near_minus_one():
    reward = rewards_module().inverse_one_plus_economic_reward
    cost = nextafter(-1., 0.)
    assert reward(cost) == 1. / (1. + cost)
    assert reward(cost) > 1e15
    assert reward(.01) > reward(10.) > 0.


@pytest.mark.parametrize('cost', [-1., -2., float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('physical_failure', [False, True])
def test_invalid_cost_rejected_even_for_physical_failure(cost, physical_failure):
    reward = rewards_module().inverse_one_plus_economic_reward
    with pytest.raises((ValueError, FloatingPointError)):
        reward(cost, physical_failure=physical_failure)


@pytest.mark.parametrize('cost', [0., 1., -.5])
def test_real_failure_replaces_normal_reward_after_validating_cost(cost):
    assert rewards_module().inverse_one_plus_economic_reward(cost, physical_failure=True) == -1.


def test_cost_helper_preserves_raw_components_and_signed_soc_value():
    module = rewards_module()
    ledger = RawCnyIntervalLedger(.01, .02, .03, 0.)
    assert module.economic_cost_cny(ledger, -.04) == fsum((.01, .02, .03, 0., -.04))
    assert ledger == RawCnyIntervalLedger(.01, .02, .03, 0.)


def test_reward_contract_is_explicit_json_safe_and_mode_specific():
    module = rewards_module()
    contract = module.reward_contract(MODE)
    assert json.loads(json.dumps(contract, allow_nan=False)) == contract
    assert contract['mode'] == MODE
    assert contract['reward_unit'] == 'dimensionless'
    assert contract['cost_unit'] == 'CNY'
    assert contract['legacy_failure_penalty_enabled'] is False
    assert contract != module.reward_contract('zhou_economic_only')
    with pytest.raises(ValueError):
        module.reward_contract('unsupported_reward')


def test_idle_zero_cost_completes_normally_with_reward_one():
    env = DirectVesselEnv(episode((0.,)), reward_mode=MODE)
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(6)
        assert reward == 1.
        assert terminated and not truncated
        assert env.episode_status == 'completed'
        assert info['total_cost_cny'] == 0.
        transition = env.last_transition
        assert transition.executed_ledger.total_cost_cny == 0.
        assert transition.soc_equivalent_hydrogen_cost_cny == 0.
        assert transition.failure_penalty_equivalent_cny == 0.
        assert transition.terminal_reason == 'completed'
    finally:
        env.close()


def test_new_mode_preserves_executed_physics_cost_and_action_masks():
    data = episode((0., 100., 20., 100.))
    env = DirectVesselEnv(data, reward_mode=MODE)
    legacy = DirectVesselEnv(data, reward_mode='zhou_economic_only')
    module = rewards_module()
    try:
        np.testing.assert_array_equal(env.reset()[0], legacy.reset()[0])
        for power in (20, 20, 0, 20):
            action = ACTION_DELTAS.index(power - round(env._physical.previous_fc_kw))
            np.testing.assert_array_equal(env.action_masks(), legacy.action_masks())
            actual = env.step(action)
            expected = legacy.step(action)
            t = env.last_transition
            old = legacy.last_transition
            assert t.executed_ledger == old.executed_ledger
            assert env._physical == legacy._physical
            assert t.state == old.state and t.next_state == old.next_state
            assert t.actual_soc == old.actual_soc
            assert t.actual_battery_kw == old.actual_battery_kw
            assert t.soc_equivalent_hydrogen_cost_cny == old.soc_equivalent_hydrogen_cost_cny
            assert t.soc_soft_penalty_cny == t.fc_ramp_penalty_cny == 0.
            cost = module.economic_cost_cny(t.executed_ledger, t.soc_equivalent_hydrogen_cost_cny)
            assert actual[1] == 1. / (1. + cost)
            assert actual[2:4] == expected[2:4]
            assert actual[4]['total_cost_cny'] == cost
        assert env.episode_status == legacy.episode_status == 'completed'
    finally:
        env.close()
        legacy.close()


class CountingAccountant(EconomicMPC):
    calls = 0

    def interval(self, *args, **kwargs):
        self.calls += 1
        return super().interval(*args, **kwargs)


@pytest.mark.parametrize('next_load,cause', [(1400., 'increment_action_limited'), (1900., 'structural_power')])
def test_failure_replaces_only_last_real_step_without_ledger_penalty(next_load, cause):
    accountant = CountingAccountant(nominal_cost_cny=1.)
    env = DirectVesselEnv(episode((100., next_load)), accountant=accountant, reward_mode=MODE)
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(12)
        t = env.last_transition
        assert reward == -1.
        assert terminated and not truncated
        assert accountant.calls == 1
        assert info['executed_steps'] == env.snapshot()['index'] == 1
        assert env.episode_status == cause
        assert t.terminal_reason == 'failure_' + cause
        assert not t.next_feasible_actions and not env.action_masks().any()
        assert t.failure_penalty_equivalent_cny == t.terminal_correction == 0.
        assert info['failure_penalty_equivalent_cny'] == 0.
        assert t.original_reward == 1. / (1. + info['total_cost_cny'])
        assert t.executed_ledger.total_cost_cny > 0.
        assert t.modeled_terminal_ledger is None
        with pytest.raises(RuntimeError):
            env.step(6)
        assert accountant.calls == 1
    finally:
        env.close()


def test_short_soc_failure_uses_same_terminal_replacement():
    data = episode((100., 1248.))
    env = DirectVesselEnv(data, reward_mode=MODE)
    try:
        env.reset()
        snapshot = env.snapshot()
        snapshot['physical'].update(soc=.20004, previous_fc_kw=600.)
        snapshot['fc_history'] = [600., 600., 600.]
        snapshot['current_state'] = [.20004, 100. / 600., 1., 0., 100. / 600.]
        env.restore(snapshot, data)
        _, reward, terminated, truncated, _ = env.step(6)
        assert reward == -1. and terminated and not truncated
        assert env.episode_status == 'soc_limited'
        assert env.last_transition.terminal_reason == 'failure_soc_limited'
        assert env.last_transition.failure_penalty_equivalent_cny == 0.
    finally:
        env.close()


@pytest.mark.parametrize('load', [1400., 1900.])
def test_infeasible_reset_has_no_fabricated_transition_or_reward(load):
    env = DirectVesselEnv(episode((load,)), reward_mode=MODE)
    try:
        with pytest.raises(InfeasibleReset):
            env.reset()
        assert env.last_transition is None
        assert env.snapshot()['index'] == 0
        assert not env.action_masks().any()
    finally:
        env.close()


def test_new_mode_never_reads_or_uses_legacy_failure_reference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy F path must not be used by the new mode')
    monkeypatch.setattr(FailurePenalty, 'from_reference', forbidden)
    monkeypatch.setattr('v6_sb3.direct_environment.prepare_failure_replay', forbidden)
    data = episode((100., 1900.))
    env = DirectVesselEnv(data, reward_mode=MODE)
    restored = DirectVesselEnv(data, reward_mode=MODE)
    try:
        assert env.penalty is None
        env.reset()
        before = env.snapshot()
        assert before['failure_penalty'] is None
        assert before['reward_contract'] == rewards_module().reward_contract(MODE)
        restored.restore(before, data)
        assert env.step(12)[1] == restored.step(12)[1] == -1.
        terminal = env.snapshot()
        restored.restore(terminal, data)
        assert restored.snapshot() == terminal
    finally:
        env.close()
        restored.close()


def test_new_mode_rejects_explicit_legacy_penalty_argument():
    penalty = FailurePenalty.from_reference(scale=0.)
    with pytest.raises(ValueError, match='penalty'):
        DirectVesselEnv(episode((0.,)), reward_mode=MODE, penalty=penalty)


def test_new_mode_snapshot_restores_without_replaying_and_rejects_other_contract():
    data = episode((100., 50., 40.))
    env = DirectVesselEnv(data, reward_mode=MODE)
    accountant = CountingAccountant(nominal_cost_cny=1.)
    restored = DirectVesselEnv(data, accountant=accountant, reward_mode=MODE)
    legacy = DirectVesselEnv(data)
    try:
        env.reset()
        env.step(9)
        snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        restored.restore(snapshot, data)
        assert accountant.calls == 0
        assert restored.snapshot() == snapshot
        assert restored.step(6)[1:4] == env.step(6)[1:4]
        assert restored.last_transition == env.last_transition
        with pytest.raises(ValueError):
            legacy.restore(snapshot, data)
        broken = json.loads(json.dumps(snapshot))
        broken['reward_contract']['version'] = 'unrecognized-version'
        with pytest.raises(ValueError):
            restored.restore(broken, data)
    finally:
        env.close()
        restored.close()
        legacy.close()


def test_program_fault_is_not_transformed_into_terminal_failure(monkeypatch):
    env = DirectVesselEnv(episode((100., 1900.)), reward_mode=MODE)
    def broken(*args, **kwargs):
        raise RuntimeError('deliberate accounting fault')
    try:
        env.reset()
        monkeypatch.setattr(env.accountant, 'interval', broken)
        with pytest.raises(RuntimeError, match='deliberate accounting fault'):
            env.step(12)
        assert env.last_transition is None
        assert env.snapshot()['index'] == 0
    finally:
        env.close()
