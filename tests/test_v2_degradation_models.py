from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FuelCellDegradationTests(unittest.TestCase):
    def test_version_coefficients_and_sources_are_exact(self) -> None:
        from v2.models.fuel_cell_degradation import (
            FC_ACCOUNTING_STRUCTURE_SOURCE_DOI,
            FC_DEGRADATION_MODEL_VERSION,
            FC_DEGRADATION_SOURCE_DOI,
            FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR,
            FC_LOW_RUNTIME_LOSS_UV_PER_HOUR,
            FC_START_STOP_LOSS_UV_PER_CYCLE,
            FC_TRANSIENT_LOSS_UV_PER_DELTA_KW,
        )

        self.assertEqual(FC_DEGRADATION_MODEL_VERSION, "aggregate_four_condition_voltage_loss_v1")
        self.assertEqual(FC_LOW_RUNTIME_LOSS_UV_PER_HOUR, 10.17)
        self.assertEqual(FC_HIGH_RUNTIME_LOSS_UV_PER_HOUR, 11.74)
        self.assertEqual(FC_TRANSIENT_LOSS_UV_PER_DELTA_KW, 0.0441)
        self.assertEqual(FC_START_STOP_LOSS_UV_PER_CYCLE, 23.91)
        self.assertEqual(FC_DEGRADATION_SOURCE_DOI, "10.1016/j.ijhydene.2024.02.349")
        self.assertEqual(FC_ACCOUNTING_STRUCTURE_SOURCE_DOI, "10.3390/jmse13010034")

    def test_runtime_boundary_is_high_at_exactly_eighty_percent_rated(self) -> None:
        from v2.models.fuel_cell_degradation import fc_voltage_loss_step_uv

        below = fc_voltage_loss_step_uv(479.999, 479.999, 3600.0, 600.0, is_on=True)
        boundary = fc_voltage_loss_step_uv(480.0, 480.0, 3600.0, 600.0, is_on=True)
        above = fc_voltage_loss_step_uv(500.0, 500.0, 3600.0, 600.0, is_on=True)

        self.assertAlmostEqual(below.low_runtime_uv, 10.17)
        self.assertEqual(below.high_runtime_uv, 0.0)
        self.assertEqual(boundary.low_runtime_uv, 0.0)
        self.assertAlmostEqual(boundary.high_runtime_uv, 11.74)
        self.assertAlmostEqual(above.high_runtime_uv, 11.74)

    def test_off_excludes_runtime_but_not_executed_power_transient(self) -> None:
        from v2.models.fuel_cell_degradation import fc_voltage_loss_step_uv

        loss = fc_voltage_loss_step_uv(100.0, 0.0, 3600.0, 600.0, is_on=False)

        self.assertEqual(loss.low_runtime_uv, 0.0)
        self.assertEqual(loss.high_runtime_uv, 0.0)
        self.assertEqual(loss.runtime_uv, 0.0)
        self.assertAlmostEqual(loss.transient_uv, 0.0441 * 100.0)
        self.assertAlmostEqual(loss.total_uv, loss.transient_uv)

    def test_step_exposes_all_raw_components_and_start_stop_proxy(self) -> None:
        from v2.models.fuel_cell_degradation import fc_voltage_loss_step_uv

        loss = fc_voltage_loss_step_uv(
            100.0, 150.0, 1800.0, 600.0, is_on=True, aggregate_start_stop_cycles=2
        )

        self.assertAlmostEqual(loss.low_runtime_uv, 10.17 * 0.5)
        self.assertEqual(loss.high_runtime_uv, 0.0)
        self.assertAlmostEqual(loss.runtime_uv, 10.17 * 0.5)
        self.assertAlmostEqual(loss.transient_uv, 0.0441 * 50.0)
        self.assertAlmostEqual(loss.start_stop_uv, 23.91 * 2)
        self.assertAlmostEqual(loss.total_uv, loss.runtime_uv + loss.transient_uv + loss.start_stop_uv)

    def test_fc_cumulative_accounting_sums_components_without_normalizing(self) -> None:
        from v2.models.fuel_cell_degradation import FuelCellVoltageLossAccount, fc_voltage_loss_step_uv

        account = FuelCellVoltageLossAccount()
        first = fc_voltage_loss_step_uv(0.0, 100.0, 3600.0, 600.0, is_on=True)
        second = fc_voltage_loss_step_uv(
            100.0, 500.0, 1800.0, 600.0, is_on=True, aggregate_start_stop_cycles=1
        )
        account.add(first)
        account.add(second)

        self.assertAlmostEqual(account.low_runtime_uv, first.low_runtime_uv)
        self.assertAlmostEqual(account.high_runtime_uv, second.high_runtime_uv)
        self.assertAlmostEqual(account.transient_uv, first.transient_uv + second.transient_uv)
        self.assertAlmostEqual(account.start_stop_uv, 23.91)
        self.assertAlmostEqual(account.total_uv, first.total_uv + second.total_uv)
        self.assertFalse(hasattr(account, "relative_life_loss"))

    def test_fc_step_inputs_are_strict_and_domain_checked(self) -> None:
        from v2.models.fuel_cell_degradation import fc_voltage_loss_step_uv

        base = dict(previous_power_kw=100.0, power_kw=100.0, dt_seconds=1.0, rated_power_kw=600.0, is_on=True)
        for field, bad in (
            ("previous_power_kw", -1.0), ("previous_power_kw", 601.0),
            ("power_kw", -1.0), ("power_kw", 601.0),
            ("dt_seconds", 0.0), ("dt_seconds", -1.0),
            ("rated_power_kw", 0.0), ("rated_power_kw", -1.0),
        ):
            arguments = dict(base)
            arguments[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                fc_voltage_loss_step_uv(**arguments)

        for field in ("previous_power_kw", "power_kw", "dt_seconds", "rated_power_kw"):
            for bad in (math.nan, math.inf, -math.inf, True, "1"):
                arguments = dict(base)
                arguments[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises((TypeError, ValueError)):
                    fc_voltage_loss_step_uv(**arguments)

        for bad in (1, np.bool_(True), "yes"):
            arguments = dict(base, is_on=bad)
            with self.subTest(is_on=bad), self.assertRaises(TypeError):
                fc_voltage_loss_step_uv(**arguments)

        for bad in (-1, 1.5, True, "1"):
            arguments = dict(base, aggregate_start_stop_cycles=bad)
            with self.subTest(cycles=bad), self.assertRaises((TypeError, ValueError)):
                fc_voltage_loss_step_uv(**arguments)

    def test_hysteresis_requires_continuous_dwell_and_counts_one_start(self) -> None:
        from v2.models.fuel_cell_degradation import AggregateFcOnOffTracker

        tracker = AggregateFcOnOffTracker(5.0, 10.0, 60.0, 600.0, initially_on=False)
        self.assertEqual(tracker.update(12.0, 30.0).starts, 0)
        self.assertEqual(tracker.update(7.0, 120.0).starts, 0)
        self.assertFalse(tracker.is_on)
        self.assertEqual(tracker.update(12.0, 30.0).starts, 0)
        transition = tracker.update(10.0, 30.0)
        self.assertEqual(transition.starts, 1)
        self.assertEqual(transition.stops, 0)
        self.assertEqual(transition.cumulative_starts, 1)
        self.assertTrue(transition.is_on)
        self.assertTrue(tracker.is_on)

    def test_on_to_off_dwell_resets_in_band_and_boundary_is_inclusive(self) -> None:
        from v2.models.fuel_cell_degradation import AggregateFcOnOffTracker

        tracker = AggregateFcOnOffTracker(5.0, 10.0, 60.0, 600.0, initially_on=True)
        self.assertEqual(tracker.update(5.0, 30.0).stops, 0)
        tracker.update(7.0, 60.0)
        self.assertTrue(tracker.is_on)
        self.assertEqual(tracker.update(5.0, 30.0).stops, 0)
        transition = tracker.update(0.0, 30.0)
        self.assertEqual(transition.starts, 0)
        self.assertEqual(transition.stops, 1)
        self.assertFalse(transition.is_on)

    def test_tracker_constructor_and_updates_are_strict(self) -> None:
        from v2.models.fuel_cell_degradation import AggregateFcOnOffTracker

        valid = dict(p_off_threshold_kw=5.0, p_on_threshold_kw=10.0, minimum_dwell_seconds=60.0, rated_power_kw=600.0, initially_on=False)
        for field, bad in (
            ("p_off_threshold_kw", -1.0), ("p_on_threshold_kw", 0.0),
            ("p_on_threshold_kw", 601.0), ("minimum_dwell_seconds", 0.0),
            ("rated_power_kw", 0.0),
        ):
            arguments = dict(valid)
            arguments[field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                AggregateFcOnOffTracker(**arguments)

        with self.assertRaises(ValueError):
            AggregateFcOnOffTracker(**dict(valid, p_on_threshold_kw=5.0))

        for field in ("p_off_threshold_kw", "p_on_threshold_kw", "minimum_dwell_seconds", "rated_power_kw"):
            for bad in (math.nan, math.inf, True, "1"):
                arguments = dict(valid)
                arguments[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises((TypeError, ValueError)):
                    AggregateFcOnOffTracker(**arguments)

        for bad in (0, np.bool_(False), "no"):
            with self.subTest(initially_on=bad), self.assertRaises(TypeError):
                AggregateFcOnOffTracker(**dict(valid, initially_on=bad))

        tracker = AggregateFcOnOffTracker(**valid)
        for power, duration in ((-1.0, 1.0), (601.0, 1.0), (1.0, 0.0)):
            with self.subTest(power=power, duration=duration), self.assertRaises(ValueError):
                tracker.update(power, duration)
        for power, duration in ((True, 1.0), ("1", 1.0), (1.0, math.nan), (1.0, True)):
            with self.subTest(power=power, duration=duration), self.assertRaises((TypeError, ValueError)):
                tracker.update(power, duration)

    def test_fc_formal_normalization_has_no_default_and_fails_closed(self) -> None:
        from v2.models.fuel_cell_degradation import (
            FC_LIFETIME_NORMALIZATION_STATUS,
            FuelCellLifetimeNormalization,
            formal_fuel_cell_relative_life_loss,
        )

        self.assertEqual(FC_LIFETIME_NORMALIZATION_STATUS, "NO-GO")
        with self.assertRaises(TypeError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=500.0)

        unresolved = FuelCellLifetimeNormalization(500.0, "aggregate system terminal voltage", "10.0000/unverified", "unverified 600 kW aggregate")
        with self.assertRaises(ValueError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=unresolved)

        with self.assertRaises(TypeError):
            FuelCellLifetimeNormalization()

    def test_fc_formal_normalization_rejects_subclasses_and_forged_fields(self) -> None:
        from v2.models.fuel_cell_degradation import FuelCellLifetimeNormalization, formal_fuel_cell_relative_life_loss

        class ForgedNormalization(FuelCellLifetimeNormalization):
            def require_verified(self) -> FuelCellLifetimeNormalization:
                return self

        with self.assertRaises(TypeError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=ForgedNormalization(500.0, "basis", "fake-doi", "fake-system"))

        forged_string = type("ForgedString", (str,), {})("basis")
        for arguments in (
            (True, "basis", "doi", "system"), (500.0, "", "doi", "system"),
            (500.0, "basis", "", "system"), (500.0, "basis", "doi", ""),
            (500.0, forged_string, "doi", "system"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                FuelCellLifetimeNormalization(*arguments)

    def test_fc_unverified_formula_is_explicit_and_raw_uv_cannot_be_priced(self) -> None:
        from v2.models.fuel_cell_degradation import formal_fuel_cell_degradation_cost_cny, fuel_cell_relative_life_loss_unverified

        self.assertAlmostEqual(fuel_cell_relative_life_loss_unverified(50_000_000.0, v_init_v=500.0, voltage_basis="explicit synthetic system basis"), 1.0)
        with self.assertRaises(TypeError):
            formal_fuel_cell_degradation_cost_cny(100.0, replacement_cost_cny=1_000_000.0, normalization=500.0)


class BatteryDegradationTests(unittest.TestCase):
    def test_version_source_and_stress_functions_are_exact(self) -> None:
        from v2.models.battery_degradation import BATTERY_DEGRADATION_MODEL_VERSION, BATTERY_DEGRADATION_SOURCE_DOI, current_stress, soc_stress

        self.assertEqual(BATTERY_DEGRADATION_MODEL_VERSION, "soc_current_weighted_throughput_v1")
        self.assertEqual(BATTERY_DEGRADATION_SOURCE_DOI, "10.3390/en14133810")
        self.assertAlmostEqual(soc_stress(0.5), 1.8125)
        self.assertAlmostEqual(current_stress(100.0, 100.0), 1.45)
        self.assertAlmostEqual(current_stress(-100.0, 100.0), 1.55)
        self.assertEqual(current_stress(0.0, 100.0), 1.0)

    def test_weighted_ah_uses_raw_current_and_not_energy_efficiency(self) -> None:
        from v2.models.battery_degradation import battery_degradation_step

        result = battery_degradation_step(0.5, 100.0, 3600.0, 100.0)
        self.assertAlmostEqual(result.raw_ah, 100.0)
        self.assertAlmostEqual(result.soc_stress, 1.8125)
        self.assertAlmostEqual(result.current_stress, 1.45)
        self.assertAlmostEqual(result.weighted_ah, 100.0 * 1.8125 * 1.45)
        self.assertFalse(hasattr(result, "eta_chg"))
        self.assertFalse(hasattr(result, "eta_dis"))
        self.assertFalse(hasattr(result, "relative_life_loss"))

    def test_charge_and_discharge_use_absolute_raw_ah_but_distinct_stress(self) -> None:
        from v2.models.battery_degradation import battery_degradation_step

        discharge = battery_degradation_step(0.8, 50.0, 1800.0, 100.0)
        charge = battery_degradation_step(0.8, -50.0, 1800.0, 100.0)
        self.assertEqual(discharge.raw_ah, 25.0)
        self.assertEqual(charge.raw_ah, 25.0)
        self.assertAlmostEqual(discharge.current_stress, 1.225)
        self.assertAlmostEqual(charge.current_stress, 1.275)
        self.assertGreater(charge.weighted_ah, discharge.weighted_ah)

    def test_battery_cumulative_accounting_sums_raw_and_weighted_ah(self) -> None:
        from v2.models.battery_degradation import BatteryThroughputAccount, battery_degradation_step

        account = BatteryThroughputAccount()
        first = battery_degradation_step(0.5, 100.0, 3600.0, 100.0)
        second = battery_degradation_step(0.8, -50.0, 1800.0, 100.0)
        account.add(first)
        account.add(second)
        self.assertAlmostEqual(account.raw_ah, first.raw_ah + second.raw_ah)
        self.assertAlmostEqual(account.weighted_ah, first.weighted_ah + second.weighted_ah)
        self.assertFalse(hasattr(account, "relative_life_loss"))

    def test_battery_inputs_are_strict_and_domain_checked(self) -> None:
        from v2.models.battery_degradation import battery_degradation_step

        valid = dict(soc=0.5, current_a=1.0, dt_seconds=1.0, nominal_current_a=100.0)
        for field, bad in (("soc", -0.001), ("soc", 1.001), ("dt_seconds", 0.0), ("dt_seconds", -1.0), ("nominal_current_a", 0.0), ("nominal_current_a", -1.0)):
            arguments = dict(valid)
            arguments[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                battery_degradation_step(**arguments)
        for field in ("soc", "current_a", "dt_seconds", "nominal_current_a"):
            for bad in (math.nan, math.inf, -math.inf, True, "1"):
                arguments = dict(valid)
                arguments[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises((TypeError, ValueError)):
                    battery_degradation_step(**arguments)

    def test_stress_helpers_apply_the_same_strict_validation(self) -> None:
        from v2.models.battery_degradation import current_stress, soc_stress

        for bad in (-0.001, 1.001, math.nan, math.inf, True, "0.5"):
            with self.subTest(soc=bad), self.assertRaises((TypeError, ValueError)):
                soc_stress(bad)
        for current, nominal in ((math.nan, 100.0), (True, 100.0), ("1", 100.0), (1.0, 0.0), (1.0, math.inf), (1.0, True)):
            with self.subTest(current=current, nominal=nominal), self.assertRaises((TypeError, ValueError)):
                current_stress(current, nominal)

    def test_q_nominal_and_q_lifetime_are_distinct_and_formal_gate_is_no_go(self) -> None:
        from v2.models.battery_degradation import BATTERY_LIFETIME_NORMALIZATION_STATUS, BatteryLifetimeNormalization, formal_battery_relative_life_loss

        self.assertEqual(BATTERY_LIFETIME_NORMALIZATION_STATUS, "NO-GO")
        with self.assertRaises(TypeError):
            formal_battery_relative_life_loss(10.0, normalization=10_000.0)
        unresolved = BatteryLifetimeNormalization(10_000.0, "10.0000/unverified", "weighted Ah", "unverified battery system")
        with self.assertRaises(ValueError):
            formal_battery_relative_life_loss(10.0, normalization=unresolved)
        self.assertNotIn("nominal", " ".join(BatteryLifetimeNormalization.__annotations__))

    def test_battery_formal_normalization_rejects_subclasses_and_forged_fields(self) -> None:
        from v2.models.battery_degradation import BatteryLifetimeNormalization, formal_battery_relative_life_loss

        class ForgedNormalization(BatteryLifetimeNormalization):
            def require_verified(self) -> BatteryLifetimeNormalization:
                return self

        with self.assertRaises(TypeError):
            formal_battery_relative_life_loss(10.0, normalization=ForgedNormalization(10_000.0, "doi", "weighted Ah", "fake-system"))

        forged_string = type("ForgedString", (str,), {})("weighted Ah")
        for arguments in (
            (True, "doi", "weighted Ah", "system"), (0.0, "doi", "weighted Ah", "system"),
            (10_000.0, "", "weighted Ah", "system"), (10_000.0, "doi", "", "system"),
            (10_000.0, "doi", "weighted Ah", ""), (10_000.0, "doi", forged_string, "system"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                BatteryLifetimeNormalization(*arguments)

    def test_unverified_raw_normalization_is_explicit_and_not_formal(self) -> None:
        from v2.models.battery_degradation import battery_relative_life_loss_unverified

        self.assertAlmostEqual(battery_relative_life_loss_unverified(250.0, q_lifetime_ah=10_000.0), 0.025)
        for weighted, lifetime in ((-1.0, 10.0), (1.0, 0.0), (True, 10.0), (1.0, "10")):
            with self.subTest(weighted=weighted, lifetime=lifetime), self.assertRaises((TypeError, ValueError)):
                battery_relative_life_loss_unverified(weighted, q_lifetime_ah=lifetime)

    def test_raw_ah_cannot_enter_formal_cost_without_verified_normalization(self) -> None:
        from v2.models.battery_degradation import formal_battery_degradation_cost_cny

        with self.assertRaises(TypeError):
            formal_battery_degradation_cost_cny(250.0, replacement_cost_cny=1_000_000.0, normalization=10_000.0)


class DegradationExportTests(unittest.TestCase):
    def test_models_package_exports_raw_models_but_no_formal_default_normalization(self) -> None:
        import v2.models as models

        for name in ("fc_voltage_loss_step_uv", "AggregateFcOnOffTracker", "battery_degradation_step", "soc_stress", "current_stress"):
            self.assertTrue(hasattr(models, name), name)
        self.assertFalse(hasattr(models, "formal_fc_lifetime_normalization"))
        self.assertFalse(hasattr(models, "formal_battery_lifetime_normalization"))


if __name__ == "__main__":
    unittest.main()
