"""Fixed trajectories test reward timing independently of a learned policy."""
from dataclasses import replace
from math import fsum
from types import SimpleNamespace

import pytest

from test_v4_control import NoSolveAccountant, episode
from v2.economics import FORMAL_PRICE_CATALOG, ShoreEnergyClassification
from v3.control import AccountState, SHORE_TARGET_SOC
from v4.control import ReplayExecutionError, replay_episode
from v4.dqn import DirectPowerDDQN


CASES = [
    ('A_modeled', .55, ('onboard',)*4, (0,250,300,0), (0,100,100,0), (0,)*4),
    ('B_high_terminal', .7, ('onboard',)*3, (0,100,0), (0,0,0), (0,)*3),
    ('C_actual_shore', .52, ('onboard',)*3+('shore_charging',)*2,
     (0,300,0,0,0), (0,100,0), (0,0,0,-624,-624)),
    ('D_two_voyages', .52, ('onboard',)*2+('shore_charging',)*2+('onboard',)*3,
     (100,0,0,0,300,100,0), (100,0,0,200,0), (0,0,-624,-624,0,0,0)),
    ('E_fc_charging', .5, ('onboard',)*4, (0,100,100,0), (0,600,600,0), (0,)*4),
    ('F_cross_working_band', .4014, ('onboard',)*33,
     (0,1000)+ (0,)*31, (0,0)+(600,)*30+(0,), (0,)*33),
    ('G_unchanged', .55, ('onboard',)*3, (0,200,0), (0,200,0), (0,)*3),
]
CASES += [(f'H_length{n}_soc{s}', s, ('onboard',)*n, (100,)*n,
           (100,)*n, (0,)*n) for n,s in ((1,.3),(9,.5),(37,.7))]


def fixed_replay(case, *, feedback):
    name, soc, modes, loads, actions, requests = case
    accountant = NoSolveAccountant()
    powers = iter(actions)
    def policy(_state, feasible):
        power = next(powers)
        assert power in feasible
        return power
    result = replay_episode(episode(modes, loads, requests), policy,
                            accountant=accountant, initial_state=AccountState(soc=soc),
                            beta_soc=500, redistribute_battery_energy=feedback)
    return result, accountant


@pytest.mark.parametrize('case', CASES, ids=[case[0] for case in CASES])
def test_fixed_trajectories_preserve_ledger_and_each_voyage_reward_sum(case):
    old, _ = fixed_replay(case, feedback=False)
    new, accountant = fixed_replay(case, feedback=True)
    assert new.total_ledger == old.total_ledger
    assert new.modeled_terminal_settlement == old.modeled_terminal_settlement
    assert new.shore_blocks == old.shore_blocks
    assert new.soc_by_row == old.soc_by_row
    assert new.final_state == old.final_state
    assert new.comparable_cost_cny == old.comparable_cost_cny
    assert new.soc_soft_penalty_cny == old.soc_soft_penalty_cny
    eta, _ = accountant.efficiency.require_calibrated()
    coefficient = (FORMAL_PRICE_CATALOG.shore_cny_per_kwh
                   * accountant.plant.battery_nominal_energy_kwh / eta)
    segment_start = new.transitions[0].state[0]
    originals, adjusted = [], []
    for before, after in zip(old.transitions, new.transitions):
        assert before.executed_ledger == after.executed_ledger
        assert before.shore_ledger == after.shore_ledger
        assert before.modeled_terminal_ledger == after.modeled_terminal_ledger
        assert after.original_reward == before.reward_cny
        delta_b = coefficient * (after.state[0] - after.actual_soc)
        assert after.immediate_battery_energy_adjustment == pytest.approx(-delta_b, abs=1e-10)
        assert after.new_reward == after.reward_cny
        assert after.reward_cny == pytest.approx(after.original_reward
               + after.immediate_battery_energy_adjustment + after.terminal_correction, abs=1e-10)
        if after.actual_battery_kw > 0:
            assert after.immediate_battery_energy_adjustment < 0
        elif after.actual_battery_kw < 0:
            assert after.immediate_battery_energy_adjustment > 0
        else:
            assert after.immediate_battery_energy_adjustment == 0
        originals.append(after.original_reward)
        adjusted.append(after.new_reward)
        if after.done:
            assert after.terminal_correction == pytest.approx(
                coefficient*(segment_start-after.actual_soc), abs=1e-10)
            assert fsum(originals) == pytest.approx(fsum(adjusted), rel=1e-12, abs=1e-8)
            originals, adjusted = [], []
            segment_start = None
        else:
            assert after.terminal_correction == 0
        if segment_start is None and after is not new.transitions[-1]:
            next_index = new.transitions.index(after)+1
            segment_start = new.transitions[next_index].state[0]
        assert after.original_economic_ledger.shore_cost_cny == (
            0 if after.shore_ledger is None else after.shore_ledger.shore_cost_cny)
    assert not originals
    assert fsum(t.reward_cny for t in old.transitions) == pytest.approx(
        fsum(t.reward_cny for t in new.transitions), rel=1e-12, abs=1e-8)
    if case[0] == 'A_modeled':
        assert new.modeled_terminal_settlement.classification is ShoreEnergyClassification.MODELED
        assert new.total_ledger.shore_cost_cny == 0
    if case[0] == 'B_high_terminal':
        assert new.transitions[-1].actual_soc > SHORE_TARGET_SOC
        assert new.modeled_terminal_settlement is None
    if case[0] == 'C_actual_shore':
        terminal = new.transitions[-1]
        assert terminal.actual_soc != new.final_state.soc
        assert terminal.terminal_correction != pytest.approx(
            coefficient*(new.transitions[0].state[0]-new.final_state.soc))
        assert new.modeled_terminal_settlement is None
        assert new.total_ledger.shore_cost_cny > 0
    if case[0] == 'F_cross_working_band':
        assert min(new.soc_by_row) < .4 and max(new.soc_by_row) > .6


def test_value_uses_existing_formal_parameters_and_is_linear_above_reference():
    from v4.reward_feedback import BatteryEnergyValue
    accountant = NoSolveAccountant()
    value = BatteryEnergyValue.from_accountant(accountant)
    eta, _ = accountant.efficiency.require_calibrated()
    assert value.coefficient_cny == pytest.approx(
        FORMAL_PRICE_CATALOG.shore_cny_per_kwh*accountant.plant.battery_nominal_energy_kwh/eta)
    assert value.reference_soc == SHORE_TARGET_SOC
    assert value(.6) == 0 and value(.7) < 0
    accountant.plant = replace(accountant.plant, battery_nominal_energy_kwh=312)
    assert BatteryEnergyValue.from_accountant(accountant).coefficient_cny == value.coefficient_cny/2
    with pytest.raises(ValueError):
        value(float('nan'))


def test_failed_unfinished_suffix_has_no_correction_and_no_economic_terminal():
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(episode(('onboard',)*2, (1000,1800), (0,0)),
                       lambda _s,_a:0, accountant=NoSolveAccountant(),
                       initial_state=AccountState(soc=.215), beta_soc=500,
                       redistribute_battery_energy=True)
    prefix = caught.value.executed_transitions
    assert len(prefix) == 1 and not prefix[-1].done
    assert prefix[-1].terminal_correction == 0
    assert prefix[-1].next_feasible_actions == ()
    assert prefix[-1].modeled_terminal_ledger is None
    assert prefix[-1].new_reward != prefix[-1].original_reward
    agent = DirectPowerDDQN(seed=42)
    assert agent.remember_completed_prefix(prefix) == 0
    agent.remember_outcome_trajectory(prefix, failed=True)
    assert not agent.replay and agent.outcome_replay[-1].failed


def test_completed_segment_before_failed_suffix_is_corrected_separately():
    data = episode(('onboard','shore_charging','onboard','onboard'),
                   (100,0,1000,1800), (0,0,0,0))
    powers = iter((100,0))
    with pytest.raises(ReplayExecutionError) as caught:
        replay_episode(data, lambda _s,_a:next(powers), accountant=NoSolveAccountant(),
                       initial_state=AccountState(soc=.215), beta_soc=500,
                       redistribute_battery_energy=True)
    completed, failed = caught.value.executed_transitions
    assert completed.done and not failed.done
    assert completed.original_reward == pytest.approx(completed.new_reward)
    assert failed.terminal_correction == 0
    agent = DirectPowerDDQN(seed=42)
    assert agent.remember_completed_prefix((completed,failed)) == 1
    assert len(agent.replay) == 1 and agent.replay[0].done
