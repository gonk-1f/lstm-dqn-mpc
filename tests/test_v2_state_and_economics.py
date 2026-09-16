from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CandidateOperatingStateTests(unittest.TestCase):
    def setUp(self) -> None:
        from v2.dqn.state import OperatingHistorySample, StateNormalization

        self.Sample = OperatingHistorySample
        self.scales = StateNormalization(
            fuel_cell_rated_kw=100.0,
            battery_power_scale_kw=50.0,
            load_power_scale_kw=200.0,
            base_load_power_scale_kw=100.0,
        )

    def test_nine_semantic_groups_flatten_to_ten_deterministic_scalars(self) -> None:
        from v2.dqn.state import (
            CANDIDATE_STATE_FEATURE_NAMES,
            CANDIDATE_STATE_GROUP_NAMES,
            build_candidate_operating_state,
        )

        history = (
            self.Sample(0.0, 0.50, 20.0, -10.0, 100.0, 80.0),
            self.Sample(10.0, 0.51, 30.0, 5.0, 120.0, 90.0),
            self.Sample(30.0, 0.53, 40.0, 10.0, 160.0, 100.0),
        )
        state = build_candidate_operating_state(
            history,
            current_time_seconds=30.0,
            window_seconds=30.0,
            normalization=self.scales,
        )

        self.assertEqual(len(CANDIDATE_STATE_GROUP_NAMES), 9)
        self.assertEqual(len(CANDIDATE_STATE_FEATURE_NAMES), 10)
        self.assertEqual(
            CANDIDATE_STATE_FEATURE_NAMES,
            (
                "soc",
                "fuel_cell_power_fraction",
                "previous_fuel_cell_power_fraction",
                "battery_power_fraction",
                "load_power_fraction",
                "recent_load_mean_fraction",
                "recent_load_population_std_fraction",
                "recent_load_window_trend_fraction",
                "causal_base_load_fraction",
                "recent_delta_soc",
            ),
        )
        self.assertIs(type(state), tuple)
        self.assertEqual(len(state), 10)
        expected_slope_kw_per_s = 2.0
        expected = (
            0.53,
            0.4,
            0.3,
            0.2,
            0.8,
            np.mean([100.0, 120.0, 160.0]) / 200.0,
            np.std([100.0, 120.0, 160.0], ddof=0) / 200.0,
            expected_slope_kw_per_s * 30.0 / 200.0,
            1.0,
            0.03,
        )
        np.testing.assert_allclose(state, expected, rtol=0.0, atol=1e-12)

    def test_irregular_window_is_seconds_based_inclusive_and_never_reads_future(self) -> None:
        from v2.dqn.state import build_candidate_operating_state

        causal = (
            self.Sample(0.0, 0.10, 1.0, 1.0, 10.0, 5.0),
            self.Sample(9.0, 0.20, 2.0, 2.0, 20.0, 6.0),
            self.Sample(10.0, 0.30, 3.0, 3.0, 30.0, 7.0),
            self.Sample(19.0, 0.40, 4.0, 4.0, 40.0, 8.0),
            self.Sample(20.0, 0.50, 5.0, 5.0, 50.0, 9.0),
        )
        future_a = self.Sample(21.0, 0.99, 99.0, 99.0, 9_999.0, 99.0)
        future_b = self.Sample(21.0, 0.01, 0.0, -99.0, -9_999.0, -99.0)

        first = build_candidate_operating_state(
            causal + (future_a,),
            current_time_seconds=20.0,
            window_seconds=10.0,
            normalization=self.scales,
        )
        second = build_candidate_operating_state(
            causal + (future_b,),
            current_time_seconds=20.0,
            window_seconds=10.0,
            normalization=self.scales,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first[2], 4.0 / 100.0)
        self.assertAlmostEqual(first[5], np.mean([30.0, 40.0, 50.0]) / 200.0)
        self.assertAlmostEqual(first[6], np.std([30.0, 40.0, 50.0]) / 200.0)
        expected_slope = np.polyfit([10.0, 19.0, 20.0], [30.0, 40.0, 50.0], 1)[0]
        self.assertAlmostEqual(first[7], expected_slope * 10.0 / 200.0)
        self.assertAlmostEqual(first[9], 0.20)

    def test_left_boundary_uses_sample_age_without_subtraction_drift(self) -> None:
        from v2.dqn.state import build_candidate_operating_state

        history = (
            self.Sample(0.1, 0.4, 10.0, 0.0, 20.0, 5.0),
            self.Sample(1.1, 0.5, 20.0, 0.0, 30.0, 5.0),
        )
        state = build_candidate_operating_state(
            history,
            current_time_seconds=1.1,
            window_seconds=1.0,
            normalization=self.scales,
        )

        self.assertAlmostEqual(state[2], 0.1)
        self.assertAlmostEqual(state[5], 25.0 / 200.0)
        self.assertAlmostEqual(state[9], 0.1)

    def test_nonfinite_extreme_ages_are_causally_outside_window(self) -> None:
        from v2.dqn.state import StateNormalization, build_candidate_operating_state

        scales = StateNormalization(1.0, 1.0, 1.0, 1.0)
        with_far_old = (
            self.Sample(-1.0e308, 0.0, 0.0, 0.0, 1.0e308, 0.0),
            self.Sample(0.0, 0.4, 0.0, 0.0, 0.0, 0.0),
            self.Sample(1.0e308, 0.5, 0.0, 0.0, 1.0, 0.0),
        )
        old_filtered = build_candidate_operating_state(
            with_far_old,
            current_time_seconds=1.0e308,
            window_seconds=1.0e308,
            normalization=scales,
        )
        self.assertAlmostEqual(old_filtered[5], 0.5)
        self.assertAlmostEqual(old_filtered[9], 0.1)

        with_far_future = (
            self.Sample(-1.5e308, 0.4, 0.0, 0.0, 0.0, 0.0),
            self.Sample(-1.0e308, 0.5, 0.0, 0.0, 1.0, 0.0),
            self.Sample(1.0e308, 0.9, 0.0, 0.0, -1.0e308, 0.0),
        )
        future_filtered = build_candidate_operating_state(
            with_far_future,
            current_time_seconds=-1.0e308,
            window_seconds=1.0e308,
            normalization=scales,
        )
        self.assertAlmostEqual(future_filtered[5], 0.5)
        self.assertAlmostEqual(future_filtered[9], 0.1)

    def test_history_and_scalars_are_strict_immutable_and_domain_checked(self) -> None:
        from v2.dqn.state import (
            OperatingHistorySample,
            StateNormalization,
            build_candidate_operating_state,
        )

        sample = self.Sample(0.0, 0.5, 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(FrozenInstanceError):
            sample.soc = 0.6  # type: ignore[misc]

        base = (sample, self.Sample(1.0, 0.6, 2.0, 2.0, 2.0, 2.0))
        with self.assertRaises(TypeError):
            build_candidate_operating_state(
                list(base),  # type: ignore[arg-type]
                current_time_seconds=1.0,
                window_seconds=1.0,
                normalization=self.scales,
            )
        with self.assertRaises(TypeError):
            build_candidate_operating_state(
                (sample, object()),  # type: ignore[arg-type]
                current_time_seconds=1.0,
                window_seconds=1.0,
                normalization=self.scales,
            )
        with self.assertRaises(ValueError):
            build_candidate_operating_state(
                (sample, self.Sample(0.0, 0.6, 2.0, 2.0, 2.0, 2.0)),
                current_time_seconds=0.0,
                window_seconds=1.0,
                normalization=self.scales,
            )
        with self.assertRaises(ValueError):
            build_candidate_operating_state(
                base,
                current_time_seconds=1.0,
                window_seconds=0.0,
                normalization=self.scales,
            )
        with self.assertRaises(ValueError):
            build_candidate_operating_state(
                base,
                current_time_seconds=0.5,
                window_seconds=1.0,
                normalization=self.scales,
            )

        for field in (
            "timestamp_seconds",
            "soc",
            "fuel_cell_power_kw",
            "battery_power_kw",
            "load_power_kw",
            "causal_base_load_kw",
        ):
            for bad in (True, np.bool_(True), "1", math.nan, math.inf, -math.inf):
                values = dict(
                    timestamp_seconds=0.0,
                    soc=0.5,
                    fuel_cell_power_kw=1.0,
                    battery_power_kw=1.0,
                    load_power_kw=1.0,
                    causal_base_load_kw=1.0,
                )
                values[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(
                    (TypeError, ValueError)
                ):
                    OperatingHistorySample(**values)
        for bad_soc in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                self.Sample(0.0, bad_soc, 1.0, 1.0, 1.0, 1.0)

        for bad in (0.0, -1.0, math.inf, True, "1"):
            with self.subTest(scale=bad), self.assertRaises((TypeError, ValueError)):
                StateNormalization(bad, 1.0, 1.0, 1.0)

    def test_previous_and_trend_require_two_samples_inside_window(self) -> None:
        from v2.dqn.state import build_candidate_operating_state

        history = (
            self.Sample(0.0, 0.5, 1.0, 1.0, 1.0, 1.0),
            self.Sample(100.0, 0.6, 2.0, 2.0, 2.0, 2.0),
        )
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_candidate_operating_state(
                history,
                current_time_seconds=100.0,
                window_seconds=10.0,
                normalization=self.scales,
            )

    def test_extreme_finite_time_and_load_keep_representable_trend_finite(self) -> None:
        from v2.dqn.state import StateNormalization, build_candidate_operating_state

        history = (
            self.Sample(0.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            self.Sample(1.0e308, 0.5, 0.0, 0.0, 1.0e308, 0.0),
        )
        state = build_candidate_operating_state(
            history,
            current_time_seconds=1.0e308,
            window_seconds=1.0e308,
            normalization=StateNormalization(1.0, 1.0, 1.0e308, 1.0),
        )

        self.assertAlmostEqual(state[6], 0.5)
        self.assertAlmostEqual(state[7], 1.0)
        self.assertTrue(all(math.isfinite(value) for value in state))

    def test_extreme_symmetric_load_population_std_is_stable(self) -> None:
        from v2.dqn.state import StateNormalization, build_candidate_operating_state

        history = (
            self.Sample(0.0, 0.5, 0.0, 0.0, -1.0e308, 0.0),
            self.Sample(2.0, 0.5, 0.0, 0.0, 1.0e308, 0.0),
        )
        state = build_candidate_operating_state(
            history,
            current_time_seconds=2.0,
            window_seconds=2.0,
            normalization=StateNormalization(1.0, 1.0, 1.0e308, 1.0),
        )

        self.assertAlmostEqual(state[5], 0.0)
        self.assertAlmostEqual(state[6], 1.0)
        self.assertAlmostEqual(state[7], 2.0)
        self.assertTrue(all(math.isfinite(value) for value in state))


class EconomicCostTests(unittest.TestCase):
    def test_price_constants_and_exact_provenance(self) -> None:
        from v2.economics import (
            BATTERY_PRICE_CNY_PER_KWH,
            EQUIPMENT_PRICE_SOURCE,
            FUEL_CELL_PRICE_CNY_PER_KW,
            HYDROGEN_PRICE_CNY_PER_KG,
            SHORE_TARIFF_CNY_PER_KWH,
            SHORE_TARIFF_SOURCE,
            PriceSource,
        )

        self.assertEqual(HYDROGEN_PRICE_CNY_PER_KG, 35.0)
        self.assertEqual(FUEL_CELL_PRICE_CNY_PER_KW, 3500.0)
        self.assertEqual(BATTERY_PRICE_CNY_PER_KWH, 2000.0)
        self.assertEqual(EQUIPMENT_PRICE_SOURCE.source_doi, "10.3390/jmse13010034")
        self.assertEqual(
            EQUIPMENT_PRICE_SOURCE.role,
            "hydrogen and equipment unit-price source only",
        )
        self.assertNotIn("600", EQUIPMENT_PRICE_SOURCE.role)
        self.assertNotIn("624", EQUIPMENT_PRICE_SOURCE.role)
        self.assertEqual(SHORE_TARIFF_CNY_PER_KWH, 1.10)
        self.assertEqual(
            SHORE_TARIFF_SOURCE.source_doi,
            "10.11930/j.issn.1004-9649.202507065",
        )
        self.assertEqual(SHORE_TARIFF_SOURCE.source_location, "Table 2")
        self.assertEqual(SHORE_TARIFF_SOURCE.classification, "scenario_not_measured")
        with self.assertRaises(ValueError):
            PriceSource("anything", "somewhere", "invented", "measured tariff")

    def test_hydrogen_and_explicit_shore_costs_are_raw_cny(self) -> None:
        from v2.economics import (
            ShoreEnergy,
            ShoreEnergyClassification,
            hydrogen_cost_cny,
            shore_energy_cost_cny,
        )

        self.assertEqual(hydrogen_cost_cny(2.0), 70.0)
        modeled = ShoreEnergy(10.0, ShoreEnergyClassification.MODELED)
        measured = ShoreEnergy(10.0, ShoreEnergyClassification.MEASURED)
        self.assertEqual(shore_energy_cost_cny(modeled), 11.0)
        self.assertEqual(shore_energy_cost_cny(measured), 11.0)
        with self.assertRaises(TypeError):
            ShoreEnergy(10.0, "measured")  # type: ignore[arg-type]
        for bad in (-1.0, math.nan, math.inf, True, "2"):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                hydrogen_cost_cny(bad)  # type: ignore[arg-type]

    def test_terminal_recharge_targets_episode_initial_soc_and_converter_is_gated(self) -> None:
        from v2.economics import (
            ShoreConverterCalibration,
            ShoreEnergyClassification,
            terminal_recharge_grid_energy,
            terminal_recharge_grid_energy_unverified,
        )
        from v2.models.battery_energy import formal_battery_efficiency

        efficiency = formal_battery_efficiency()
        below = terminal_recharge_grid_energy_unverified(
            episode_initial_soc=0.60,
            episode_end_soc=0.50,
            battery_capacity_kwh=100.0,
            battery_efficiency=efficiency,
            eta_shore_converter=0.80,
        )
        above = terminal_recharge_grid_energy_unverified(
            episode_initial_soc=0.60,
            episode_end_soc=0.70,
            battery_capacity_kwh=100.0,
            battery_efficiency=efficiency,
            eta_shore_converter=0.80,
        )
        self.assertAlmostEqual(below.energy_kwh, 10.0 / 0.95 / 0.80)
        self.assertIs(below.classification, ShoreEnergyClassification.MODELED)
        self.assertEqual(above.energy_kwh, 0.0)

        unresolved = ShoreConverterCalibration(
            0.95,
            "UNVERIFIED",
            "no approved vessel-specific converter calibration",
        )
        with self.assertRaisesRegex(ValueError, "NO-GO"):
            terminal_recharge_grid_energy(
                episode_initial_soc=0.60,
                episode_end_soc=0.50,
                battery_capacity_kwh=100.0,
                battery_efficiency=efficiency,
                converter_calibration=unresolved,
            )

    def test_missing_shore_tariff_is_a_gate_and_not_a_zero_cost_default(self) -> None:
        from v2.economics import (
            EconomicPriceCatalog,
            ShoreEnergy,
            ShoreEnergyClassification,
            shore_energy_cost_cny,
        )

        missing = EconomicPriceCatalog(
            hydrogen_cny_per_kg=35.0,
            fuel_cell_cny_per_kw=3500.0,
            battery_cny_per_kwh=2000.0,
            shore_cny_per_kwh=None,
        )
        with self.assertRaisesRegex(ValueError, "shore tariff"):
            shore_energy_cost_cny(
                ShoreEnergy(1.0, ShoreEnergyClassification.MODELED), prices=missing
            )

    def test_unverified_modeled_shore_energy_cannot_enter_formal_ledger(self) -> None:
        from v2.economics import (
            ShoreEnergy,
            ShoreEnergyClassification,
            build_formal_interval_ledger,
        )
        from v2.models.battery_degradation import BatteryLifetimeNormalization
        from v2.models.fuel_cell_degradation import (
            FC_SINGLE_CELL_VOLTAGE_BASIS,
            FuelCellLifetimeNormalization,
        )

        with self.assertRaisesRegex(ValueError, "shore-converter"):
            build_formal_interval_ledger(
                hydrogen_mass_kg=0.0,
                fuel_cell_voltage_loss_uv=0.0,
                fuel_cell_rated_kw=100.0,
                fuel_cell_normalization=FuelCellLifetimeNormalization(
                    0.7,
                    FC_SINGLE_CELL_VOLTAGE_BASIS,
                    "10.0000/unverified",
                    "unverified cell",
                ),
                battery_weighted_ah=0.0,
                battery_capacity_kwh=100.0,
                battery_normalization=BatteryLifetimeNormalization(
                    10_000.0,
                    "10.0000/unverified",
                    "weighted Ah",
                    "unverified battery",
                ),
                shore_energy=ShoreEnergy(
                    1.0, ShoreEnergyClassification.MODELED
                ),
            )

    def test_ledger_is_immutable_exact_unweighted_sum_and_negative_reward(self) -> None:
        from v2.economics import RawCnyIntervalLedger

        ledger = RawCnyIntervalLedger(
            h2_cost_cny=1.0,
            fuel_cell_degradation_cost_cny=2.0,
            battery_degradation_cost_cny=3.0,
            shore_cost_cny=4.0,
        )
        self.assertEqual(ledger.components_cny, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(ledger.total_cost_cny, 10.0)
        self.assertEqual(ledger.reward_cny, -10.0)
        with self.assertRaises(FrozenInstanceError):
            ledger.h2_cost_cny = 99.0  # type: ignore[misc]
        for bad in (-1.0, math.nan, math.inf, True, "1"):
            with self.subTest(bad=bad), self.assertRaises((TypeError, ValueError)):
                RawCnyIntervalLedger(bad, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RawCnyIntervalLedger(1.0e308, 1.0e308, 0.0, 0.0)

    def test_formal_degradation_rejects_raw_units_and_full_ledger_is_no_go(self) -> None:
        from v2.economics import build_formal_interval_ledger
        from v2.models.battery_degradation import BatteryLifetimeNormalization
        from v2.models.fuel_cell_degradation import (
            FC_SINGLE_CELL_VOLTAGE_BASIS,
            FuelCellLifetimeNormalization,
        )

        with self.assertRaises(TypeError):
            build_formal_interval_ledger(
                hydrogen_mass_kg=1.0,
                fuel_cell_voltage_loss_uv=10.0,
                fuel_cell_rated_kw=100.0,
                fuel_cell_normalization=500.0,  # type: ignore[arg-type]
                battery_weighted_ah=10.0,
                battery_capacity_kwh=100.0,
                battery_normalization=10_000.0,  # type: ignore[arg-type]
                shore_energy=None,
            )

        fc_unresolved = FuelCellLifetimeNormalization(
            0.7,
            FC_SINGLE_CELL_VOLTAGE_BASIS,
            "10.0000/unverified",
            "unverified cell",
        )
        batt_unresolved = BatteryLifetimeNormalization(
            10_000.0,
            "10.0000/unverified",
            "weighted Ah",
            "unverified battery",
        )
        with self.assertRaisesRegex(ValueError, "NO-GO"):
            build_formal_interval_ledger(
                hydrogen_mass_kg=1.0,
                fuel_cell_voltage_loss_uv=10.0,
                fuel_cell_rated_kw=100.0,
                fuel_cell_normalization=fc_unresolved,
                battery_weighted_ah=10.0,
                battery_capacity_kwh=100.0,
                battery_normalization=batt_unresolved,
                shore_energy=None,
            )

    def test_reward_scaling_is_train_calibrated_sealed_and_held_out_rejected(self) -> None:
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import (
            RawCnyIntervalLedger,
            RewardScaleCalibration,
            calibrate_reward_scale,
            scaled_reward,
        )

        train = DatasetProvenance("synthetic", "train-only", DataSplit.TRAIN)
        ledger = RawCnyIntervalLedger(10.0, 20.0, 30.0, 40.0)
        calibration = calibrate_reward_scale(
            (50.0, 150.0),
            provenance=train,
            audit_id="reward-scale-audit-001",
            reason="Train macro-interval raw-cost arithmetic mean",
        )
        self.assertEqual(calibration.scale_cny, 100.0)
        self.assertEqual(calibration.train_raw_costs_cny, (50.0, 150.0))
        self.assertEqual(calibration.sample_count, 2)
        self.assertEqual(calibration.derivation_rule, "positive_arithmetic_mean_v1")
        self.assertEqual(scaled_reward(ledger, calibration=calibration), -1.0)
        for split in (DataSplit.VALIDATION, DataSplit.TEST, DataSplit.UNKNOWN):
            provenance = DatasetProvenance("synthetic", split.value, split)
            with self.subTest(split=split), self.assertRaises(PermissionError):
                calibrate_reward_scale(
                    (100.0,),
                    provenance=provenance,
                    audit_id="forbidden-held-out",
                    reason="must fail before calibration",
                )

        with self.assertRaises(TypeError):
            RewardScaleCalibration(100.0, train)  # type: ignore[call-arg]

        forged = calibrate_reward_scale(
            (50.0, 150.0),
            provenance=train,
            audit_id="reward-scale-audit-001",
            reason="Train macro-interval raw-cost arithmetic mean",
        )
        object.__setattr__(forged, "train_raw_costs_cny", (1.0, 1.0))
        with self.assertRaises(ValueError):
            scaled_reward(ledger, calibration=forged)
        with self.assertRaises(TypeError):
            scaled_reward(ledger, calibration=100.0)  # type: ignore[arg-type]

        for values in ((0.0,), (-1.0,), (math.nan,), (True,), ("1",)):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                calibrate_reward_scale(
                    values,  # type: ignore[arg-type]
                    provenance=train,
                    audit_id="invalid-costs",
                    reason="must reject invalid calibration evidence",
                )
        with self.assertRaises(ValueError):
            calibrate_reward_scale(
                (1.0e308, 1.0e308),
                provenance=train,
                audit_id="overflow-costs",
                reason="must reject overflow",
            )

        tainted_provenance = DatasetProvenance(
            "synthetic", "train-to-test", DataSplit.TRAIN
        )
        tainted = calibrate_reward_scale(
            (100.0,),
            provenance=tainted_provenance,
            audit_id="tainted-provenance",
            reason="must detect provenance mutation",
        )
        object.__setattr__(tainted_provenance, "split", DataSplit.TEST)
        with self.assertRaises(PermissionError):
            scaled_reward(ledger, calibration=tainted)

    def test_reward_scale_rejects_subclasses_and_mutation_of_bound_evidence(self) -> None:
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import (
            RawCnyIntervalLedger,
            RewardScaleCalibration,
            calibrate_reward_scale,
            scaled_reward,
        )

        train = DatasetProvenance("synthetic", "train", DataSplit.TRAIN)
        ledger = RawCnyIntervalLedger(1.0, 0.0, 0.0, 0.0)
        calibration = calibrate_reward_scale(
            (1.0, 3.0),
            provenance=train,
            audit_id="immutable-evidence",
            reason="bind all calibration evidence",
        )
        for field, value in (
            ("scale_cny", 99.0),
            ("sample_count", 1),
            ("derivation_rule", "attacker_rule"),
            ("audit_id", "attacker-audit"),
            ("reason", "attacker reason"),
            ("digest", "0" * 64),
        ):
            forged = calibrate_reward_scale(
                (1.0, 3.0),
                provenance=train,
                audit_id="immutable-evidence",
                reason="bind all calibration evidence",
            )
            object.__setattr__(forged, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                scaled_reward(ledger, calibration=forged)

        forged_provenance = calibrate_reward_scale(
            (1.0, 3.0),
            provenance=train,
            audit_id="forged-provenance",
            reason="reject replacement provenance objects",
        )
        object.__setattr__(forged_provenance, "provenance", object())
        with self.assertRaises(TypeError):
            scaled_reward(ledger, calibration=forged_provenance)

        class CalibrationSubclass(RewardScaleCalibration):
            pass

        forged_subclass = object.__new__(CalibrationSubclass)
        for field in (
            "scale_cny",
            "train_raw_costs_cny",
            "sample_count",
            "provenance",
            "derivation_rule",
            "audit_id",
            "reason",
            "digest",
            "_seal",
        ):
            object.__setattr__(forged_subclass, field, getattr(calibration, field))
        with self.assertRaises(TypeError):
            scaled_reward(ledger, calibration=forged_subclass)

    def test_scaled_reward_rejects_nonfinite_quotient(self) -> None:
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import (
            RawCnyIntervalLedger,
            calibrate_reward_scale,
            scaled_reward,
        )

        calibration = calibrate_reward_scale(
            (5.0e-324,),
            provenance=DatasetProvenance("synthetic", "subnormal", DataSplit.TRAIN),
            audit_id="subnormal-scale",
            reason="exercise finite scaled-reward boundary",
        )
        ledger = RawCnyIntervalLedger(1.0e308, 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            scaled_reward(ledger, calibration=calibration)


if __name__ == "__main__":
    unittest.main()
