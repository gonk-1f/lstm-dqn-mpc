import math

import pytest

from v3.control import AccountState, EconomicMPC, MPCWeights


def test_objective_terms_match_the_solved_plan():
    mpc = EconomicMPC(nominal_cost_cny=10.0)
    state = AccountState(soc=0.5, previous_fc_kw=100.0)
    weights = MPCWeights(lambda_ref=0.4, lambda_soc=0.2)
    plan = mpc.solve((200.0,) * 5, (180.0,) * 5, state, weights)

    terms = mpc.objective_terms(plan.fc_power_kw, plan.forecast_kw, plan.reference_kw, state)

    assert math.isclose(terms.objective(weights), plan.objective_value, rel_tol=1e-10)
    assert terms.economic_cost_cny > 0.0
    assert math.isclose(terms.economic_normalized, terms.economic_cost_cny / 10.0)
    assert terms.reference_penalty >= 0.0
    assert terms.soc_penalty >= 0.0


def test_calibrated_axes_form_sixteen_independent_pairs():
    from v3.action_calibration import four_level_axis, cartesian_actions

    ref_axis = four_level_axis(20.0)
    soc_axis = four_level_axis(0.2)
    actions = cartesian_actions(ref_axis, soc_axis)

    assert ref_axis == (0.0, 5.0, 20.0, 80.0)
    assert soc_axis == (0.0, 0.05, 0.2, 0.8)
    assert len(actions) == 16
    assert len(set(actions)) == 16
    assert actions[0] == MPCWeights(0.0, 0.0)
    assert actions[-1] == MPCWeights(80.0, 0.8)
    with pytest.raises(ValueError):
        four_level_axis(0.0)
    assert four_level_axis(0.2, positive_multiplier=16.0) == (0.0, 0.8, 3.2, 12.8)


def test_weight_bases_balance_train_economics_against_material_penalties():
    from v3.action_calibration import calibrate_weight_bases

    ref_base, soc_base = calibrate_weight_bases(
        economic_costs_cny=(5.0, 10.0, 15.0),
        nominal_cost_cny=10.0,
        material_reference_penalty=0.05,
        material_soc_penalty=5.0,
    )
    assert ref_base == pytest.approx(20.0)
    assert soc_base == pytest.approx(0.2)
    with pytest.raises(ValueError):
        calibrate_weight_bases((0.0,), 10.0, 0.05, 5.0)


def test_reference_snapshot_clips_previous_lpf_to_physical_fc_rating():
    from v3.action_calibration import CalibrationOrigin, _baseline_terms

    origin = CalibrationOrigin("high_load", 1, 900.0, 800.0, 750.0, 100.0)
    terms = _baseline_terms(EconomicMPC(nominal_cost_cny=1.0), origin, 0.5)
    assert terms.economic_cost_cny > 0.0
