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
        from v2.models.fuel_cell_degradation import reference_unit_voltage_loss_step_uv

        below = reference_unit_voltage_loss_step_uv(79.999, 79.999, 3600.0, 100.0, is_on=True)
        boundary = reference_unit_voltage_loss_step_uv(80.0, 80.0, 3600.0, 100.0, is_on=True)
        above = reference_unit_voltage_loss_step_uv(90.0, 90.0, 3600.0, 100.0, is_on=True)

        self.assertAlmostEqual(below.low_runtime_uv, 10.17)
        self.assertEqual(below.high_runtime_uv, 0.0)
        self.assertEqual(boundary.low_runtime_uv, 0.0)
        self.assertAlmostEqual(boundary.high_runtime_uv, 11.74)
        self.assertAlmostEqual(above.high_runtime_uv, 11.74)

    def test_off_excludes_runtime_but_not_executed_power_transient(self) -> None:
        from v2.models.fuel_cell_degradation import reference_unit_voltage_loss_step_uv

        loss = reference_unit_voltage_loss_step_uv(100.0, 0.0, 3600.0, 100.0, is_on=False)

        self.assertEqual(loss.low_runtime_uv, 0.0)
        self.assertEqual(loss.high_runtime_uv, 0.0)
        self.assertEqual(loss.runtime_uv, 0.0)
        self.assertAlmostEqual(loss.transient_uv, 0.0441 * 100.0)
        self.assertAlmostEqual(loss.total_uv, loss.transient_uv)

    def test_step_exposes_all_raw_components_and_start_stop_proxy(self) -> None:
        from v2.models.fuel_cell_degradation import reference_unit_voltage_loss_step_uv

        loss = reference_unit_voltage_loss_step_uv(
            10.0, 15.0, 1800.0, 100.0, is_on=True, aggregate_start_stop_cycles=2
        )

        self.assertAlmostEqual(loss.low_runtime_uv, 10.17 * 0.5)
        self.assertEqual(loss.high_runtime_uv, 0.0)
        self.assertAlmostEqual(loss.runtime_uv, 10.17 * 0.5)
        self.assertAlmostEqual(loss.transient_uv, 0.0441 * 5.0)
        self.assertAlmostEqual(loss.start_stop_uv, 23.91 * 2)
        self.assertAlmostEqual(loss.total_uv, loss.runtime_uv + loss.transient_uv + loss.start_stop_uv)

    def test_fc_cumulative_accounting_sums_components_without_normalizing(self) -> None:
        from v2.models.fuel_cell_degradation import FuelCellVoltageLossAccount, reference_unit_voltage_loss_step_uv

        account = FuelCellVoltageLossAccount()
        first = reference_unit_voltage_loss_step_uv(0.0, 10.0, 3600.0, 100.0, is_on=True)
        second = reference_unit_voltage_loss_step_uv(
            10.0, 90.0, 1800.0, 100.0, is_on=True, aggregate_start_stop_cycles=1
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
        from v2.models.fuel_cell_degradation import reference_unit_voltage_loss_step_uv

        base = dict(previous_reference_power_kw=10.0, reference_power_kw=10.0, dt_seconds=1.0, reference_rated_power_kw=100.0, is_on=True)
        for field, bad in (
            ("previous_reference_power_kw", -1.0), ("previous_reference_power_kw", 101.0),
            ("reference_power_kw", -1.0), ("reference_power_kw", 101.0),
            ("dt_seconds", 0.0), ("dt_seconds", -1.0),
            ("reference_rated_power_kw", 0.0), ("reference_rated_power_kw", -1.0),
        ):
            arguments = dict(base)
            arguments[field] = bad
            with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                reference_unit_voltage_loss_step_uv(**arguments)

        for field in ("previous_reference_power_kw", "reference_power_kw", "dt_seconds", "reference_rated_power_kw"):
            for bad in (math.nan, math.inf, -math.inf, True, "1"):
                arguments = dict(base)
                arguments[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises((TypeError, ValueError)):
                    reference_unit_voltage_loss_step_uv(**arguments)

        for bad in (1, np.bool_(True), "yes"):
            arguments = dict(base, is_on=bad)
            with self.subTest(is_on=bad), self.assertRaises(TypeError):
                reference_unit_voltage_loss_step_uv(**arguments)

        for bad in (-1, 1.5, True, "1"):
            arguments = dict(base, aggregate_start_stop_cycles=bad)
            with self.subTest(cycles=bad), self.assertRaises((TypeError, ValueError)):
                reference_unit_voltage_loss_step_uv(**arguments)

    def test_aggregate_power_cannot_masquerade_as_source_compatible_power(self) -> None:
        import v2.models as models
        from v2.models.fuel_cell_degradation import (
            AggregateFcPowerMapping,
            formal_aggregate_fc_voltage_loss_step_uv,
        )

        self.assertFalse(hasattr(models, "fc_voltage_loss_step_uv"))
        with self.assertRaises(TypeError):
            formal_aggregate_fc_voltage_loss_step_uv(
                0.0, 600.0, 1.0, 600.0, is_on=True, mapping=6.0
            )

        unresolved = AggregateFcPowerMapping(
            6.0, "10.0000/unverified", "unverified aggregate-to-reference mapping"
        )
        with self.assertRaises(ValueError):
            formal_aggregate_fc_voltage_loss_step_uv(
                0.0, 600.0, 1.0, 600.0, is_on=True, mapping=unresolved
            )

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

    def test_fc_aggregate_normalization_is_verified_while_legacy_single_cell_api_stays_closed(self) -> None:
        from v2.models.fuel_cell_degradation import (
            FC_SINGLE_CELL_VOLTAGE_BASIS,
            FC_LIFETIME_NORMALIZATION_STATUS,
            FuelCellLifetimeNormalization,
            formal_fuel_cell_relative_life_loss,
        )

        self.assertEqual(FC_LIFETIME_NORMALIZATION_STATUS, "VERIFIED")
        with self.assertRaises(TypeError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=500.0)

        unresolved = FuelCellLifetimeNormalization(0.7, FC_SINGLE_CELL_VOLTAGE_BASIS, "10.0000/unverified", "unverified cell")
        with self.assertRaises(ValueError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=unresolved)

        with self.assertRaises(TypeError):
            FuelCellLifetimeNormalization()

    def test_fc_formal_normalization_rejects_subclasses_and_forged_fields(self) -> None:
        from v2.models.fuel_cell_degradation import FC_SINGLE_CELL_VOLTAGE_BASIS, FuelCellLifetimeNormalization, formal_fuel_cell_relative_life_loss

        class ForgedNormalization(FuelCellLifetimeNormalization):
            def require_verified(self) -> FuelCellLifetimeNormalization:
                return self

        with self.assertRaises(TypeError):
            formal_fuel_cell_relative_life_loss(10.0, normalization=ForgedNormalization(0.7, FC_SINGLE_CELL_VOLTAGE_BASIS, "fake-doi", "fake-cell"))

        forged_string = type("ForgedString", (str,), {})("basis")
        for arguments in (
            (True, "basis", "doi", "system"), (500.0, "", "doi", "system"),
            (500.0, "basis", "", "system"), (500.0, "basis", "doi", ""),
            (500.0, forged_string, "doi", "system"),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                FuelCellLifetimeNormalization(*arguments)

    def test_fc_legacy_formula_is_explicit_and_old_cost_signature_is_rejected(self) -> None:
        from v2.models.fuel_cell_degradation import (
            FC_SINGLE_CELL_VOLTAGE_BASIS,
            FuelCellLifetimeNormalization,
            formal_fuel_cell_degradation_cost_cny,
            fuel_cell_relative_life_loss_unverified,
        )

        self.assertAlmostEqual(fuel_cell_relative_life_loss_unverified(70_000.0, v_init_v=0.7, voltage_basis=FC_SINGLE_CELL_VOLTAGE_BASIS), 1.0)
        with self.assertRaises(ValueError):
            fuel_cell_relative_life_loss_unverified(
                70_000.0, v_init_v=0.7, voltage_basis="aggregate system voltage"
            )
        with self.assertRaises(ValueError):
            FuelCellLifetimeNormalization(
                0.7, "stack voltage", "10.0000/unverified", "unverified stack"
            )
        with self.assertRaises(TypeError):
            formal_fuel_cell_degradation_cost_cny(100.0, replacement_cost_cny=1_000_000.0, normalization=500.0)

    def test_fc_records_accounts_reject_forged_fields_and_overflow_atomically(self) -> None:
        from v2.models.fuel_cell_degradation import FuelCellVoltageLoss, FuelCellVoltageLossAccount

        for values in (
            (True, 0.0, 0.0, 0.0),
            ("1", 0.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0, 0.0),
            (math.inf, 0.0, 0.0, 0.0),
        ):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                FuelCellVoltageLoss(*values)
            with self.subTest(account_values=values), self.assertRaises((TypeError, ValueError)):
                FuelCellVoltageLossAccount(*values)

        account = FuelCellVoltageLossAccount(1.0e308, 0.0, 0.0, 0.0)
        before = (account.low_runtime_uv, account.high_runtime_uv, account.transient_uv, account.start_stop_uv)
        with self.assertRaises(ValueError):
            account.add(FuelCellVoltageLoss(1.0e308, 0.0, 0.0, 0.0))
        self.assertEqual(
            (account.low_runtime_uv, account.high_runtime_uv, account.transient_uv, account.start_stop_uv),
            before,
        )

    def test_fc_records_and_accounts_reject_cross_component_overflow(self) -> None:
        from v2.models.fuel_cell_degradation import FuelCellVoltageLoss, FuelCellVoltageLossAccount

        for values in (
            (1.0e308, 1.0e308, 0.0, 0.0),
            (1.0e308, 0.0, 1.0e308, 0.0),
        ):
            with self.subTest(record=values), self.assertRaises(ValueError):
                FuelCellVoltageLoss(*values)
            with self.subTest(account=values), self.assertRaises(ValueError):
                FuelCellVoltageLossAccount(*values)

        for account, step in (
            (
                FuelCellVoltageLossAccount(9.0e307, 0.0, 0.0, 0.0),
                FuelCellVoltageLoss(0.0, 9.0e307, 0.0, 0.0),
            ),
            (
                FuelCellVoltageLossAccount(9.0e307, 0.0, 0.0, 0.0),
                FuelCellVoltageLoss(0.0, 0.0, 9.0e307, 0.0),
            ),
        ):
            before = (
                account.low_runtime_uv,
                account.high_runtime_uv,
                account.transient_uv,
                account.start_stop_uv,
            )
            with self.subTest(add=(account, step)), self.assertRaises(ValueError):
                account.add(step)
            self.assertEqual(
                (
                    account.low_runtime_uv,
                    account.high_runtime_uv,
                    account.transient_uv,
                    account.start_stop_uv,
                ),
                before,
            )


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

    def test_q_nominal_and_q_lifetime_are_distinct_and_literature_calibrated(self) -> None:
        from v2.models.battery_degradation import (
            BATTERY_LIFETIME_CONFIGURATION_STATUS,
            BATTERY_LIFETIME_EVIDENCE_STATUS,
            formal_battery_lifetime_normalization,
            formal_battery_relative_life_loss,
        )

        self.assertEqual(
            BATTERY_LIFETIME_CONFIGURATION_STATUS,
            "FROZEN",
        )
        self.assertEqual(
            BATTERY_LIFETIME_EVIDENCE_STATUS,
            "SECONDARY_LITERATURE / LITERATURE-CALIBRATED",
        )
        with self.assertRaises(TypeError):
            formal_battery_relative_life_loss(10.0, normalization=10_000.0)
        normalization = formal_battery_lifetime_normalization()
        self.assertAlmostEqual(
            formal_battery_relative_life_loss(
                10.0,
                normalization=normalization,
            ),
            10.0 / normalization.q_lifetime_ah,
        )
        self.assertNotEqual(
            normalization.nominal_charge_capacity_ah,
            normalization.q_lifetime_ah,
        )

    def test_battery_formal_normalization_rejects_subclasses_and_forged_fields(self) -> None:
        from v2.models.battery_degradation import (
            BatteryLifetimeNormalization,
            formal_battery_lifetime_normalization,
            formal_battery_relative_life_loss,
        )

        class ForgedNormalization(BatteryLifetimeNormalization):
            def require_literature_calibrated(self) -> BatteryLifetimeNormalization:
                return self

        approved = formal_battery_lifetime_normalization()
        with self.assertRaises(TypeError):
            formal_battery_relative_life_loss(
                10.0,
                normalization=ForgedNormalization(
                    approved.q_lifetime_ah,
                    approved.throughput_factor,
                    approved.nominal_charge_capacity_ah,
                    approved.provenance_classification,
                    approved.evidence_basis,
                    approved.applicability,
                ),
            )

        for arguments in (
            (
                True,
                approved.throughput_factor,
                approved.nominal_charge_capacity_ah,
                approved.provenance_classification,
                approved.evidence_basis,
                approved.applicability,
            ),
            (
                approved.q_lifetime_ah,
                10_000.0,
                approved.nominal_charge_capacity_ah,
                approved.provenance_classification,
                approved.evidence_basis,
                approved.applicability,
            ),
            (
                approved.q_lifetime_ah,
                approved.throughput_factor,
                approved.nominal_charge_capacity_ah,
                "manufacturer specification",
                approved.evidence_basis,
                approved.applicability,
            ),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                BatteryLifetimeNormalization(*arguments)

    def test_unverified_raw_normalization_is_explicit_and_not_formal(self) -> None:
        from v2.models.battery_degradation import battery_relative_life_loss_unverified

        self.assertAlmostEqual(battery_relative_life_loss_unverified(250.0, q_lifetime_ah=10_000.0), 0.025)
        for weighted, lifetime in ((-1.0, 10.0), (1.0, 0.0), (True, 10.0), (1.0, "10")):
            with self.subTest(weighted=weighted, lifetime=lifetime), self.assertRaises((TypeError, ValueError)):
                battery_relative_life_loss_unverified(weighted, q_lifetime_ah=lifetime)

    def test_bare_number_cannot_enter_formal_interval_cost(self) -> None:
        from v2.models.battery_degradation import formal_battery_degradation_cost_cny

        with self.assertRaises(TypeError):
            formal_battery_degradation_cost_cny(
                0.0,
                250.0,
                replacement_cost_cny=1_000_000.0,
                normalization=10_000.0,
            )

    def test_battery_records_accounts_reject_forged_fields_and_overflow_atomically(self) -> None:
        from v2.models.battery_degradation import BatteryDegradationStep, BatteryThroughputAccount

        for values in (
            (True, 0.0, 1.0, 1.0),
            ("1", 0.0, 1.0, 1.0),
            (-1.0, 0.0, 1.0, 1.0),
            (math.inf, 0.0, 1.0, 1.0),
        ):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                BatteryDegradationStep(*values)

        for values in ((True, 0.0), ("1", 0.0), (-1.0, 0.0), (math.inf, 0.0)):
            with self.subTest(account_values=values), self.assertRaises((TypeError, ValueError)):
                BatteryThroughputAccount(*values)

        account = BatteryThroughputAccount(1.0e308, 1.0e308)
        before = (account.raw_ah, account.weighted_ah)
        with self.assertRaises(ValueError):
            account.add(BatteryDegradationStep(1.0e308, 1.0e308, 1.0, 1.0))
        self.assertEqual((account.raw_ah, account.weighted_ah), before)


class DegradationExportTests(unittest.TestCase):
    def test_models_package_exports_raw_models_but_no_formal_default_normalization(self) -> None:
        import v2.models as models

        for name in ("reference_unit_voltage_loss_step_uv", "AggregateFcOnOffTracker", "battery_degradation_step", "soc_stress", "current_stress"):
            self.assertTrue(hasattr(models, name), name)
        self.assertFalse(hasattr(models, "formal_fc_lifetime_normalization"))
        self.assertFalse(hasattr(models, "formal_battery_lifetime_normalization"))


if __name__ == "__main__":
    unittest.main()
