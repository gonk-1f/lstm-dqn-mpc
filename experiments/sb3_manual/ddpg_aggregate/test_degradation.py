"""Aggregate degradation checks without data loading or policy training."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from v2.models import battery_degradation as battery
from v2.models import battery_energy
from v2.models import fuel_cell_degradation as fuel_cell
from v3.control import AccountState, EconomicMPC


class FuelCellDegradationTests(unittest.TestCase):
    def test_mylonopoulos_single_cell_end_of_life_and_replacement_cost(self):
        self.assertEqual(fuel_cell.FC_INITIAL_CELL_VOLTAGE_V, 0.95)
        self.assertEqual(fuel_cell.FC_EOL_VOLTAGE_DROP_FRACTION, 0.15)
        self.assertEqual(fuel_cell.FC_EOL_VOLTAGE_LOSS_UV, 142500.0)
        self.assertFalse(fuel_cell.fuel_cell_life_state(142499.0).eol_reached)
        end = fuel_cell.fuel_cell_life_state(142500.0)
        self.assertTrue(end.eol_reached)
        self.assertEqual(end.raw_life_fraction, 1.0)
        self.assertEqual(end.economic_life_fraction, 1.0)
        self.assertEqual(fuel_cell.FC_REPLACEMENT_FRACTION, 0.5)
        self.assertEqual(fuel_cell.FC_AGGREGATE_REPLACEMENT_COST_CNY, 3500 * 600 * 0.5)
        self.assertEqual(fuel_cell.FC_AGGREGATE_TO_REFERENCE_POWER_RATIO, 1.0 / 6.0)

    def test_four_condition_coefficients_and_project_high_load_boundary(self):
        self.assertEqual(fuel_cell.FC_LOW_RUNTIME_LOSS_UV_PER_HOUR, 10.17)
        self.assertEqual(fuel_cell.FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR, 11.74)
        self.assertEqual(fuel_cell.FC_TRANSIENT_LOSS_UV_PER_DELTA_KW, 0.0441)
        self.assertEqual(fuel_cell.FC_START_STOP_LOSS_UV_PER_CYCLE, 23.91)
        self.assertEqual(fuel_cell.FC_HIGH_LOAD_FRACTION, 0.8)
        self.assertEqual(fuel_cell.FC_HIGH_LOAD_BOUNDARY_STATUS, "PROJECT_MODELING_ASSUMPTION")
        low = fuel_cell.reference_unit_voltage_loss_step_uv(79.0, 79.0, 3600.0, 100.0, is_on=True)
        high = fuel_cell.reference_unit_voltage_loss_step_uv(80.0, 80.0, 3600.0, 100.0, is_on=True)
        self.assertEqual((low.low_runtime_uv, low.high_runtime_uv), (10.17, 0.0))
        self.assertEqual((high.low_runtime_uv, high.high_runtime_uv), (0.0, 11.74))

    def test_aggregate_start_ramp_and_new_lifetime_cost(self):
        physics = EconomicMPC(nominal_cost_cny=1.0)
        after, ledger, _ = physics.interval(AccountState(), 600.0, 600.0, duration_s=1.0)
        expected_uv = 11.74 / 3600.0 + 0.0441 * (600.0 / 6.0) + 23.91
        self.assertAlmostEqual(after.fc_loss_uv, expected_uv)
        self.assertAlmostEqual(ledger.fuel_cell_degradation_cost_cny, expected_uv / 142500.0 * 1050000.0)
        stopped = fuel_cell.formal_aggregate_fc_voltage_loss_step_uv(
            600.0, 0.0, 1.0, 600.0, is_on=False, mapping=physics.fc_mapping,
            aggregate_start_stop_cycles=fuel_cell.aggregate_start_cycle_count(600.0, 0.0),
        )
        self.assertAlmostEqual(stopped.transient_uv, 4.41)
        self.assertEqual(stopped.runtime_uv, 0.0)
        self.assertEqual(stopped.start_stop_uv, 0.0)


class BatteryDegradationTests(unittest.TestCase):
    def test_soc_current_stress_and_frozen_lifetime_normalization(self):
        nominal = battery.BATTERY_NOMINAL_CHARGE_CAPACITY_AH
        self.assertAlmostEqual(battery.soc_stress(0.6), 1.52)
        self.assertAlmostEqual(battery.current_stress(nominal, nominal), 1.45)
        self.assertAlmostEqual(battery.current_stress(-nominal, nominal), 1.55)
        step = battery.battery_degradation_step(0.6, -nominal, 1.0, nominal)
        self.assertAlmostEqual(step.raw_ah, nominal / 3600.0)
        self.assertAlmostEqual(step.weighted_ah, nominal / 3600.0 * 1.52 * 1.55)
        self.assertEqual(battery.BATTERY_LIFETIME_THROUGHPUT_FACTOR, 15000.0)
        self.assertEqual(battery.BATTERY_LIFETIME_Q_AH, 15000.0 * nominal)
        self.assertEqual(battery.BATTERY_REPLACEMENT_COST_CNY, 2000.0 * 624.0)

    def test_signed_model_current_is_derived_from_executed_soc_change(self):
        current = getattr(battery_energy, "battery_current_from_soc_change", None)
        self.assertTrue(callable(current), "Missing current derived from physical SOC update")
        nominal = battery.BATTERY_NOMINAL_CHARGE_CAPACITY_AH
        self.assertAlmostEqual(current(0.6, 0.59, 30.0, nominal), 0.01 * nominal * 120.0)
        self.assertAlmostEqual(current(0.6, 0.61, 30.0, nominal), -0.01 * nominal * 120.0)
        self.assertEqual(current(0.6, 0.6, 1.0, nominal), 0.0)

    def test_interval_uses_actual_cell_current_with_efficiency_once(self):
        physics = EconomicMPC(nominal_cost_cny=1.0)
        nominal = physics.battery_normalization.nominal_charge_capacity_ah
        for load, cell_kw in ((400.0, 100.0 / 0.95), (100.0, -200.0 * 0.95), (300.0, 0.0)):
            with self.subTest(load_kw=load):
                before = AccountState(soc=0.6, previous_fc_kw=300.0)
                with patch("v3.control.battery_degradation_step", wraps=battery.battery_degradation_step) as degrade:
                    after, ledger, bus_kw = physics.interval(before, 300.0, load, duration_s=1.0)
                expected_soc = 0.6 - cell_kw / 3600.0 / 624.0
                self.assertEqual(after.soc, expected_soc)
                self.assertEqual(bus_kw, load - 300.0)
                current = degrade.call_args.args[1]
                expected_current = cell_kw * 1000.0 / battery.BATTERY_NOMINAL_VOLTAGE_V
                self.assertAlmostEqual(current, expected_current, places=8)
                expected_step = battery.battery_degradation_step(0.6, expected_current, 1.0, nominal)
                self.assertAlmostEqual(after.battery_weighted_ah, expected_step.weighted_ah, places=10)
                self.assertAlmostEqual(ledger.battery_degradation_cost_cny,
                                       expected_step.weighted_ah / (15000.0 * nominal) * 1248000.0,
                                       places=10)

    def test_shore_current_matches_existing_shore_soc_without_extra_efficiency(self):
        physics = EconomicMPC(nominal_cost_cny=1.0)
        before = AccountState(soc=0.55)
        with patch("v3.control.battery_degradation_step", wraps=battery.battery_degradation_step) as degrade:
            after, _, accepted_kw = physics.shore_interval(before, -300.0)
        self.assertEqual(accepted_kw, -300.0)
        self.assertEqual(after.soc, 0.55 + 300.0 * 30.0 / 3600.0 / 624.0)
        expected_current = (before.soc - after.soc) * physics.battery_normalization.nominal_charge_capacity_ah * 120.0
        self.assertAlmostEqual(degrade.call_args.args[1], expected_current, places=10)


if __name__ == "__main__":
    unittest.main()
