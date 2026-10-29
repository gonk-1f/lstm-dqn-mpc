"""Executable mathematical audit; this does not implement a production mode.

The local reference functions describe the requested reward on its domain.
Toy costs/MDP paths are mathematical counterexamples, not vessel trajectories.
Only the terminal-target test calls the existing production TD operator.
"""
from math import fsum, isfinite

import pytest
import torch

from v6_sb3.masked_dqn import masked_td_targets


def _inverse_reference(cost_cny):
    """Audit-only partial function: no epsilon, abs, clipping or offsets."""
    if not isfinite(cost_cny) or cost_cny <= 0:
        raise ValueError('the inverse reward requires finite, strictly positive cost')
    reward = 1.0 / cost_cny
    if not isfinite(reward):
        raise FloatingPointError('a valid cost can still overflow its reciprocal')
    return reward


def _path_rewards(costs_cny, *, physical_failure=False):
    """Score real toy transitions; an empty event has no invented TD sample."""
    costs_cny = tuple(costs_cny)
    if any(not isfinite(cost) for cost in costs_cny):
        raise ValueError('nonfinite economic ledger is an error, even at a physical terminal')
    if physical_failure and costs_cny:
        return (*(_inverse_reference(cost) for cost in costs_cny[:-1]), -1.0)
    return tuple(_inverse_reference(cost) for cost in costs_cny)


@pytest.mark.parametrize('cost', [0.0, -0.0, -1.0, float('nan'), float('inf'), -float('inf')])
def test_inverse_domain_rejects_nonpositive_or_nonfinite_without_failure_conversion(cost):
    with pytest.raises(ValueError, match='strictly positive'):
        _inverse_reference(cost)


def test_finite_positive_cost_is_necessary_but_not_sufficient_for_finite_reward():
    cost = 1e-320
    assert isfinite(cost) and cost > 0
    with pytest.raises(FloatingPointError, match='overflow'):
        _inverse_reference(cost)


def test_positive_cost_below_one_cny_gives_reward_above_one():
    assert _inverse_reference(0.01) == pytest.approx(100.0)
    assert _inverse_reference(0.07) == pytest.approx(14.285714285714286)
    assert _inverse_reference(1000.0) == pytest.approx(0.001)


def test_one_step_order_is_monotonic_but_complete_cost_order_can_reverse():
    assert _inverse_reference(1.0) > _inverse_reference(2.0)
    cheap = (1.0, 1.0)
    expensive = (0.1, 10.0)
    assert fsum(cheap) == 2.0 < fsum(expensive)
    assert fsum(_path_rewards(cheap)) == 2.0 < fsum(_path_rewards(expensive))


def test_same_total_cost_unequal_distribution_receives_higher_inverse_return():
    constant = (1.0, 1.0)
    uneven = (0.1, 1.9)
    assert fsum(constant) == pytest.approx(fsum(uneven))
    assert fsum(_path_rewards(uneven)) > fsum(_path_rewards(constant))


def test_risk_preference_can_change_even_with_equal_expected_single_step_cost():
    # Two equiprobable outcomes of a synthetic one-step action.
    safe_cost = 1.0
    random_costs = (0.1, 1.9)
    assert fsum(random_costs) / 2 == pytest.approx(safe_cost)
    assert fsum(_path_rewards(random_costs)) / 2 > _inverse_reference(safe_cost)


def test_failure_replaces_only_last_real_reward_and_does_not_add_legacy_penalty():
    costs = (0.5, 0.1)
    rewards = _path_rewards(costs, physical_failure=True)
    assert rewards == (2.0, -1.0)
    assert len(rewards) == len(costs)
    assert fsum(rewards) == 1.0
    assert fsum(rewards) != fsum(_path_rewards(costs)) - 1.0
    assert fsum(rewards) != fsum(_path_rewards(costs)) - 10132.660087898294


@pytest.mark.parametrize('terminal_cost', [0.0, -0.1, 1e-320])
def test_physical_terminal_uses_minus_one_without_reciprocating_its_finite_cost(terminal_cost):
    assert _path_rewards((0.5, terminal_cost), physical_failure=True) == (2.0, -1.0)


@pytest.mark.parametrize('terminal_cost', [float('nan'), float('inf'), -float('inf')])
def test_physical_terminal_does_not_hide_nonfinite_economic_ledger(terminal_cost):
    with pytest.raises(ValueError, match='finite'):
        _path_rewards((0.5, terminal_cost), physical_failure=True)


def test_event_only_failure_creates_no_transition_or_minus_one_sample():
    assert _path_rewards((), physical_failure=True) == ()


def test_normal_completion_keeps_its_positive_last_step_reward():
    assert _path_rewards((0.5, 0.1)) == (2.0, 10.0)


def test_completion_wins_when_rewarded_prefix_is_identical():
    # A sufficient algebraic condition, not a comparison of arbitrary policies.
    failed = _path_rewards((0.5, 10.0), physical_failure=True)
    complete = _path_rewards((0.5, 2.0, 3.0))
    assert failed[:-1] == complete[:1]
    assert fsum(complete) > fsum(failed)


def test_a_different_failed_path_can_beat_completion_despite_positive_normal_rewards():
    # In a three-decision toy MDP, two actions at the initial state lead to
    # different paths. One completes; one has no feasible third decision.
    complete = _path_rewards((1.0, 1.0, 1.0))
    failed = _path_rewards((0.01, 10.0), physical_failure=True)
    assert fsum(complete) == 3.0
    assert fsum(failed) == 99.0 > fsum(complete)


def test_fixed_terminal_value_has_cost_dependent_decrement_from_normal_reward():
    small_cost_drop = _inverse_reference(0.01) - (-1.0)
    high_cost_drop = _inverse_reference(1000.0) - (-1.0)
    assert small_cost_drop == 101.0
    assert high_cost_drop == pytest.approx(1.001)


def test_reward_units_matter_because_failure_constant_is_not_currency_scaled():
    # This is why metadata must freeze numeric costs in CNY. No unit change
    # or reward scaling is applied to any production environment here.
    complete = (1.0, 1.0)
    failed = (0.1, 10.0)
    assert fsum(_path_rewards(failed, physical_failure=True)) > fsum(_path_rewards(complete))
    complete_fen = tuple(100 * cost for cost in complete)
    failed_fen = tuple(100 * cost for cost in failed)
    assert fsum(_path_rewards(failed_fen, physical_failure=True)) < fsum(_path_rewards(complete_fen))


def test_equal_per_step_cost_gives_return_linear_in_voyage_length_at_gamma_one():
    short = fsum(_path_rewards((0.07,) * 1000))
    long = fsum(_path_rewards((0.07,) * 10000))
    assert short == pytest.approx(14285.714285714286)
    assert long == pytest.approx(10 * short)


def test_reciprocal_of_interval_cost_depends_quadratically_on_time_resolution():
    # Constant physical cost rate, same elapsed time and total CNY cost.
    rate_cny_per_second = 0.07
    horizon_seconds = 30
    one_second_costs = (rate_cny_per_second,) * horizon_seconds
    thirty_second_costs = (rate_cny_per_second * horizon_seconds,)
    assert fsum(one_second_costs) == pytest.approx(fsum(thirty_second_costs))
    assert fsum(_path_rewards(one_second_costs)) == pytest.approx(
        900 * fsum(_path_rewards(thirty_second_costs)))


def test_small_step_cost_can_exceed_replay_float32_range_even_when_double_is_finite():
    reward = _inverse_reference(1e-40)
    assert isfinite(reward)
    assert not torch.isfinite(torch.tensor(reward, dtype=torch.float32))


def test_thirteen_action_values_can_lose_small_advantages_at_large_float32_scale():
    q = torch.full((13,), 1e8, dtype=torch.float32)
    q[7] = 1e8 + 1.0
    # A numeric precision example, not a measured network output or action bias.
    assert q[7] == q[0]
    assert torch.finfo(torch.float32).eps * 1e8 > 1.0


def test_existing_td_operator_has_no_terminal_bootstrap_for_minus_one():
    reward = torch.tensor([[-1.0], [2.0]])
    next_q = torch.full((2, 13), 1e20)
    done = torch.tensor([[1.0], [0.0]])
    masks = torch.zeros((2, 13), dtype=torch.bool)
    masks[1, 6] = True
    target = masked_td_targets(reward, next_q, done, masks, gamma=1.0)
    assert target[0].item() == -1.0
    assert target[1].item() > 1e19
    assert torch.isfinite(target).all()
