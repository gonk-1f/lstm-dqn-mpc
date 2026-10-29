"""Independent mathematical audit toys, never a production economics model.

Only the existing read-only FC map is reused. These checks do not run a voyage,
train an agent, read any held-out split, or alter physics/rewards.
Run with pytest, or execute this file to print the reproducible numerical cases.
"""

from dataclasses import dataclass
from math import fsum, isfinite
import json

import pytest

from v2.models.fuel_cell_efficiency import formal_fuel_cell_efficiency_map


LHV = 120.0 / 3.6
ETA_CHG = ETA_DIS = 0.95
PRICE = 21.9
CAPACITY = 624.0
SOC_MIN = 0.2


def hydrogen_kg(power_kw, duration_s=1.0):
    """Independent scalar arithmetic over the existing source-backed curve."""
    if power_kw == 0.0:
        return 0.0
    eta = formal_fuel_cell_efficiency_map().eta(power_kw)
    return power_kw * duration_s / 3600.0 / (eta * LHV)


def acquired_charge_value(fc_kw, load_kw, duration_s=1.0):
    """Capitalize exactly the charging share of already consumed hydrogen."""
    assert 0.0 <= load_kw <= fc_kw
    if fc_kw == 0.0:
        return 0.0, 0.0
    stored = ETA_CHG * (fc_kw - load_kw) * duration_s / 3600.0
    eta = formal_fuel_cell_efficiency_map().eta(fc_kw)
    value = stored / (ETA_CHG * eta * LHV)
    return stored, value


@dataclass(frozen=True)
class InventoryToy:
    """Usable energy E [kWh] and historical hydrogen cost basis H [kg]."""

    energy: float
    value: float

    def charge(self, energy, acquired_value):
        assert energy >= 0.0 and acquired_value >= 0.0
        return InventoryToy(self.energy + energy, self.value + acquired_value), -acquired_value

    def discharge(self, energy):
        assert self.energy > 0.0 and 0.0 <= energy <= self.energy
        withdrawn_value = self.value * energy / self.energy
        return InventoryToy(self.energy - energy, self.value - withdrawn_value), withdrawn_value


def numeric_cases():
    curve = formal_fuel_cell_efficiency_map()
    average = fsum(curve.eta(p) for p in range(10, 601, 10)) / 60.0
    energy, allocation = acquired_charge_value(60.0, 20.0)
    actual = hydrogen_kg(60.0)
    delivered_kw = energy * ETA_DIS * 3600.0
    initial = InventoryToy(1.0, 0.06)
    charged, credit = initial.charge(1.0, 0.05)
    final, debit = charged.discharge(1.0)
    # Same SOC, FC off, and identical causal load features, distinct cost bases.
    observed = (SOC_MIN + 1.0 / CAPACITY, 0.0, 0.0, 0.0, 0.0)
    basis_a = initial
    basis_b = final
    next_a, debit_a = basis_a.discharge(0.01)
    next_b, debit_b = basis_b.discharge(0.01)
    dynamic_start = 1.0 * 0.06
    dynamic_end = 1.0 * 0.05
    common_energy = CAPACITY * (.6 - SOC_MIN)
    common_value = common_energy / (ETA_CHG * curve.eta(60.) * LHV)
    common = InventoryToy(common_energy, common_value)
    energy_a, acquisition_a = acquired_charge_value(40., 0., 3.)
    energy_b, acquisition_b = acquired_charge_value(60., 0., 2.)
    causal_a = common.charge(energy_a, acquisition_a)[0]
    causal_b = common.charge(energy_b, acquisition_b)[0]
    next_withdrawal = 36.1 / 3600. / ETA_DIS
    causal_debit_a = causal_a.discharge(next_withdrawal)[1]
    causal_debit_b = causal_b.discharge(next_withdrawal)[1]
    return {
        "eta_formal_average": average,
        "efficiencies": {str(p): curve.eta(p) for p in (0, 10, 60, 70, 100, 300, 600)},
        "charge_60kw_load_20kw_1s": {
            "stored_kwh": energy,
            "actual_hydrogen_kg": actual,
            "allocated_hydrogen_kg": allocation,
            "direct_load_hydrogen_kg": actual - allocation,
            "allocated_cost_cny": allocation * PRICE,
            "fc_off_discharge_kw_to_close_energy_cycle": delivered_kw,
            "delivered_kwh": delivered_kw / 3600.0,
            "round_trip_efficiency": ETA_CHG * ETA_DIS,
        },
        "equal_energy_inventory_cycle": {
            "initial_energy_kwh": initial.energy,
            "initial_value_kg": initial.value,
            "charged_energy_kwh": 1.0,
            "charged_value_kg": 0.05,
            "final_energy_kwh": final.energy,
            "final_value_kg": final.value,
            "soc_correction_kg": credit + debit,
            "total_actual_plus_correction_kg": 0.05 + credit + debit,
        },
        "history5_alias": {
            "identical_current_observation": observed,
            "next_discharge_energy_kwh": 0.01,
            "state_a_inventory_kg": basis_a.value,
            "state_b_inventory_kg": basis_b.value,
            "state_a_soc_cost_cny": debit_a * PRICE,
            "state_b_soc_cost_cny": debit_b * PRICE,
            "same_next_energy": next_a.energy == next_b.energy,
        },
        "same_zero_load_legal_prefix_alias": {
            "fc_sequence_a_kw": [40., 40., 40., 0.],
            "fc_sequence_b_kw": [60., 60., 0., 0.],
            "all_four_loads_kw": 0.,
            "same_energy_added_kwh": energy_a,
            "same_current_soc": SOC_MIN + causal_a.energy / CAPACITY,
            "inventory_hydrogen_a_kg": causal_a.value,
            "inventory_hydrogen_b_kg": causal_b.value,
            "initial_value_convention": "same illustrative replenishment-at-60-kW basis in both prefixes",
            "next_fc_off_load_kw": 36.1,
            "next_soc_cost_a_cny": causal_debit_a * PRICE,
            "next_soc_cost_b_cny": causal_debit_b * PRICE,
            "next_soc_cost_difference_cny": (causal_debit_a - causal_debit_b) * PRICE,
        },
        "dynamic_potential_revaluation_without_energy_change": {
            "start_value_kg": dynamic_start,
            "end_value_kg": dynamic_end,
            "full_potential_soc_cost_kg": dynamic_start - dynamic_end,
            "energy_only_term_kg": 0.0,
        },
        "nonlinear_inverse_breaks_potential_invariance": {
            "same_total_cost_cny": 2.0,
            "base_costs_cny": [1.0, 1.0],
            "redistributed_costs_cny": [1.5, 0.5],
            "base_inverse_return": 2.0,
            "redistributed_inverse_return": 1.0 / 1.5 + 1.0 / 0.5,
        },
        "fixed_average_credit_ratio_at_70kw_zero_load": ETA_CHG * curve.eta(70) / average,
    }


@pytest.mark.parametrize("fc_kw,load_kw", [(10., 0.), (60., 20.), (70., 0.), (100., 100.), (600., 500.), (0., 0.)])
def test_charge_credit_never_exceeds_actual_hydrogen(fc_kw, load_kw):
    stored, credit = acquired_charge_value(fc_kw, load_kw)
    actual = hydrogen_kg(fc_kw)
    assert 0.0 <= credit <= actual + 1e-15
    if fc_kw:
        assert credit == pytest.approx(actual * (fc_kw - load_kw) / fc_kw, abs=1e-15)
        assert stored == pytest.approx(credit * ETA_CHG * formal_fuel_cell_efficiency_map().eta(fc_kw) * LHV)


def test_zero_fc_discharge_uses_stored_value_and_remains_finite():
    inventory, debit = InventoryToy(1., .06).discharge(.01)
    assert formal_fuel_cell_efficiency_map().eta(0.) == 0.
    assert inventory.energy == pytest.approx(.99)
    assert debit == pytest.approx(.0006)
    assert isfinite(debit)


def test_retained_energy_value_and_acquisition_cost_have_different_loss_timing():
    stored, acquisition = acquired_charge_value(60., 20.)
    eta = formal_fuel_cell_efficiency_map().eta(60.)
    retained = stored / (eta * LHV)
    assert retained == pytest.approx(ETA_CHG * acquisition)
    assert acquisition - retained == pytest.approx((1. - ETA_CHG) * acquisition)
    assert retained < acquisition < hydrogen_kg(60.)


def test_complete_charge_discharge_cycle_from_empty_basis_counts_hydrogen_once():
    energy, allocation = acquired_charge_value(60., 20.)
    initial = InventoryToy(0., 0.)
    charged, charge_correction = initial.charge(energy, allocation)
    final, discharge_correction = charged.discharge(energy)
    assert final == initial
    assert charge_correction + discharge_correction == pytest.approx(0., abs=1e-15)
    assert hydrogen_kg(60.) + charge_correction + discharge_correction == pytest.approx(hydrogen_kg(60.))
    assert ETA_DIS * energy == pytest.approx((40. / 3600.) * ETA_CHG * ETA_DIS)


def test_initial_usable_energy_value_must_be_explicit():
    usable = CAPACITY * (.6 - SOC_MIN)
    assert usable == pytest.approx(249.6)
    historical_rate = 1. / (ETA_CHG * formal_fuel_cell_efficiency_map().eta(60.) * LHV)
    initial = InventoryToy(usable, usable * historical_rate)
    final, correction = initial.discharge(1.)
    assert correction == pytest.approx(historical_rate)
    assert initial.value - final.value == pytest.approx(correction)
    # Choosing H0=0 is a free-initial-energy convention, not an efficiency fix.
    assert InventoryToy(usable, 0.).discharge(1.)[1] == 0.


def test_mixed_inventory_corrections_telescope_to_book_value_endpoints():
    initial = InventoryToy(2., .12)
    state = initial
    corrections = []
    for operation, energy, acquired in [("charge", .4, .021), ("discharge", .7, 0.), ("charge", .2, .011), ("discharge", .1, 0.)]:
        state, correction = state.charge(energy, acquired) if operation == "charge" else state.discharge(energy)
        corrections.append(correction)
    assert fsum(corrections) == pytest.approx(initial.value - state.value, abs=1e-15)
    actual_hydrogen = .021 + .011
    assert actual_hydrogen + fsum(corrections) == pytest.approx(actual_hydrogen + initial.value - state.value)


def test_equal_soc_is_insufficient_to_cancel_historical_inventory_correction():
    initial = InventoryToy(1., .06)
    charged, credit = initial.charge(1., .05)
    final, debit = charged.discharge(1.)
    assert initial.energy == final.energy
    assert final.value == pytest.approx(.055)
    assert credit + debit == pytest.approx(.005)
    assert .05 + credit + debit == pytest.approx(.055)


def test_empty_usable_inventory_has_no_division_on_idle_or_charge():
    empty = InventoryToy(0., 0.)
    assert empty.charge(0., 0.) == (empty, 0.)
    charged, _ = empty.charge(.01, .0005)
    assert charged.discharge(.01)[0] == empty
    with pytest.raises(AssertionError):
        empty.discharge(.01)


def test_history5_does_not_identify_historical_inventory_value():
    a = InventoryToy(1., .06)
    b = a.charge(1., .05)[0].discharge(1.)[0]
    def history5(inventory):
        return (SOC_MIN + inventory.energy / CAPACITY, 0., 0., 0., 0.)
    assert history5(a) == history5(b)
    after_a, debit_a = a.discharge(.01)
    after_b, debit_b = b.discharge(.01)
    assert history5(after_a) == history5(after_b)
    assert debit_a != debit_b
    assert debit_a * PRICE == pytest.approx(.01314)
    assert debit_b * PRICE == pytest.approx(.012045)


def test_same_load_and_legal_increment_sequences_create_hidden_inventory_alias():
    # A: FC [40,40,40,0]; B: [60,60,0,0]; all loads zero, all dt=1s.
    # Both sequences obey the unchanged +/-60 kW incremental action domain,
    # and their maximum charging magnitude is 60 kW with SOC near 0.6.
    common = InventoryToy(CAPACITY * (.6 - SOC_MIN), 10.)
    energy_a, cost_a = acquired_charge_value(40., 0., 3.)
    energy_b, cost_b = acquired_charge_value(60., 0., 2.)
    a = common.charge(energy_a, cost_a)[0]
    b = common.charge(energy_b, cost_b)[0]
    assert a.energy == b.energy
    assert (SOC_MIN + a.energy / CAPACITY, 0., 0., 0., 0.) == (SOC_MIN + b.energy / CAPACITY, 0., 0., 0., 0.)
    assert a.value != b.value
    # A matching FC-off 36.1 kW discharge gives equal physical successors,
    # but unequal hydrogen-inventory debits for the same observation/action.
    next_a, debit_a = a.discharge(36.1 / 3600. / ETA_DIS)
    next_b, debit_b = b.discharge(36.1 / 3600. / ETA_DIS)
    assert next_a.energy == next_b.energy
    assert debit_a != debit_b


def test_dynamic_energy_only_weight_omits_revaluation_term():
    energies = (1., 1.1, 1.)
    rates = (.06, .05, .07)
    energy_only = fsum(rates[t] * (energies[t] - energies[t + 1]) for t in range(2))
    revaluation = fsum(energies[t + 1] * (rates[t] - rates[t + 1]) for t in range(2))
    endpoint = rates[0] * energies[0] - rates[-1] * energies[-1]
    assert energy_only == pytest.approx(-.001)
    assert energy_only + revaluation == pytest.approx(endpoint)
    assert energy_only != pytest.approx(endpoint)


def test_full_dynamic_potential_telescopes_only_when_full_endpoints_match():
    energies = (1., 1.1, 1.)
    rates = (.06, .05, .06)
    values = tuple(e * k for e, k in zip(energies, rates))
    assert fsum(values[i] - values[i + 1] for i in range(2)) == pytest.approx(0.)
    assert energies[0] * .06 - energies[-1] * .05 == pytest.approx(.01)


def test_a_soc_only_potential_cannot_match_two_actual_charge_efficiencies():
    # Same SOC change (same 10 kW charging surplus), different actual FC eta.
    energy_a, charge_a = acquired_charge_value(60., 50.)
    energy_b, charge_b = acquired_charge_value(100., 90.)
    assert energy_a == energy_b
    assert charge_a != pytest.approx(charge_b)


def test_dynamic_marginal_recharge_can_be_defined_at_fc_off_by_counterfactual():
    # An explicitly selected 60 kW replenishment opportunity, not eta(0).
    energy, acquired = acquired_charge_value(60., 20.)
    counterfactual = hydrogen_kg(60.) - hydrogen_kg(0.)
    assert counterfactual > 0.
    assert acquired / energy == pytest.approx(1. / (ETA_CHG * formal_fuel_cell_efficiency_map().eta(60.) * LHV))
    # Whole counterfactual H2 includes serving its 20 kW load; a declared
    # counterfactual allocation rule is still necessary for an energy price.
    assert counterfactual > acquired


def test_inverse_of_redistributed_cost_does_not_preserve_telescoping_return():
    base = (1., 1.)
    adjusted = (1.5, .5)
    assert fsum(base) == fsum(adjusted)
    assert fsum(1. / c for c in base) == 2.
    assert fsum(1. / c for c in adjusted) == pytest.approx(8. / 3.)


def test_idle_remains_zero_under_flow_inventory_or_unchanged_state_potential():
    inventory = InventoryToy(1., .06)
    updated, correction = inventory.charge(0., 0.)
    assert updated == inventory
    assert correction == 0.
    assert inventory.value - updated.value == 0.
    with pytest.raises(ZeroDivisionError):
        _ = 1. / (hydrogen_kg(0.) + correction)


if __name__ == "__main__":
    print(json.dumps(numeric_cases(), indent=2, ensure_ascii=False))
