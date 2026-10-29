"""Explicit V6 learning rewards over unchanged executed economic costs."""
from math import fsum, isfinite
from numbers import Real

from v2.economics import RawCnyIntervalLedger
from v4.independent_economics import EVALUATION_VERSION


INVERSE_ONE_PLUS_ECONOMIC = 'inverse_one_plus_economic'
REWARD_MODES = ('zhou_economic_only', 'v6_energy_soc_ramp', INVERSE_ONE_PLUS_ECONOMIC)


def _finite_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a finite real number')
    try:
        result = float(value)
    except OverflowError as error:
        raise FloatingPointError(f'{name} must be finite') from error
    if not isfinite(result):
        raise FloatingPointError(f'{name} must be finite')
    return result


def economic_cost_cny(ledger, soc_equivalent_hydrogen_cost_cny):
    """Sum actual ledger costs and exactly one signed reference SOC value."""
    if type(ledger) is not RawCnyIntervalLedger:
        raise TypeError('economic cost requires the original RawCnyIntervalLedger')
    soc_cost = _finite_real(soc_equivalent_hydrogen_cost_cny, 'SOC-equivalent cost')
    try:
        cost = fsum((*ledger.components_cny, soc_cost))
    except OverflowError as error:
        raise FloatingPointError('total economic cost must be finite') from error
    return _finite_real(cost, 'total economic cost')


def inverse_one_plus_economic_reward(cost_cny, *, physical_failure=False):
    """Use exact 1/(1+C_CNY), or replace a genuine failure's last reward by -1.

    Even an actual physical failure must not hide an invalid economic cost.
    The caller classifies physical failure; program/data errors must propagate.
    """
    cost = _finite_real(cost_cny, 'economic cost C_CNY')
    denominator = 1.0 + cost
    if not isfinite(denominator) or denominator <= 0.0:
        raise ValueError('inverse_one_plus_economic requires finite C_CNY > -1 '
                         'and a finite positive denominator 1+C_CNY')
    if type(physical_failure) is not bool:
        raise TypeError('physical_failure must be a bool')
    reward = -1.0 if physical_failure else 1.0 / denominator
    if not isfinite(reward):
        raise FloatingPointError('inverse_one_plus_economic reward must be finite')
    return reward


def reward_contract(reward_mode):
    """JSON-safe reward identity; callers bind the entire contract to a run."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(f'reward_mode must be one of {REWARD_MODES}')
    if reward_mode == INVERSE_ONE_PLUS_ECONOMIC:
        return dict(mode=reward_mode, version='v6_inverse_one_plus_economic_v1',
            normal_reward_formula='1/(1+C_CNY)',
            cost_formula='fsum(C_H2,C_FCdeg,C_Batdeg,C_shore,C_SOCeq)',
            cost_unit='CNY', reward_unit='dimensionless',
            physical_failure_reward=-1.0,
            failure_application='replace_last_real_transition_after_cost_domain_validation',
            cost_domain='finite C_CNY > -1; finite positive 1+C_CNY',
            soc_equivalence_model=EVALUATION_VERSION,
            soc_efficiency='existing explicit fixed reference efficiency',
            legacy_failure_penalty_enabled=False, cost_floor=None, clipping=None)
    return dict(mode=reward_mode, version='v6_legacy_reward_contract_v1',
        normal_reward_formula=('-full_ledger_cost_CNY-C_SOCeq' if reward_mode == 'zhou_economic_only'
            else '-C_H2-C_SOCeq-0.2*((SOC_after-0.5)/0.1)**2-0.05*(delta_FC_kW)**2/600'),
        cost_unit='CNY', reward_unit='legacy_CNY_score',
        physical_failure_reward='normal_reward-legacy_failure_penalty_CNY',
        failure_application='existing_prepare_failure_replay',
        soc_equivalence_model=EVALUATION_VERSION, legacy_failure_penalty_enabled=True)
