from __future__ import annotations

from dataclasses import replace
import inspect
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _require_attr(
    test: unittest.TestCase,
    module: object,
    name: str,
) -> object:
    test.assertTrue(hasattr(module, name), f"missing contract: {name}")
    return getattr(module, name)


class FuelCellIntervalCostTests(unittest.TestCase):
    def test_cumulative_life_state_boundaries(self) -> None:
        from v2.models import fuel_cell_degradation as model

        eol_uv = _require_attr(self, model, "FC_EOL_VOLTAGE_LOSS_UV")
        life_state = _require_attr(self, model, "fuel_cell_life_state")

        self.assertEqual(eol_uv, 70_000.0)
        cases = (
            (0.0, 0.0, 0.0, False),
            (35_000.0, 0.5, 0.5, False),
            (70_000.0, 1.0, 1.0, True),
            (80_000.0, 8.0 / 7.0, 1.0, True),
        )
        for cumulative_uv, expected_raw, expected_econ, expected_eol in cases:
            with self.subTest(cumulative_uv=cumulative_uv):
                state = life_state(cumulative_uv)
                self.assertAlmostEqual(state.raw_life_fraction, expected_raw)
                self.assertAlmostEqual(
                    state.economic_life_fraction,
                    expected_econ,
                )
                self.assertEqual(state.eol_reached, expected_eol)

    def test_crossing_eol_charges_only_unconsumed_fraction(self) -> None:
        from v2.models import fuel_cell_degradation as model

        interval_loss = _require_attr(
            self,
            model,
            "formal_fuel_cell_interval_life_loss",
        )
        interval_cost = _require_attr(
            self,
            model,
            "formal_fuel_cell_degradation_cost_cny",
        )

        increment = interval_loss(69_990.0, 70_010.0)
        expected_delta = 10.0 / 70_000.0
        self.assertAlmostEqual(
            increment.delta_economic_fraction,
            expected_delta,
        )
        self.assertTrue(increment.after.eol_reached)
        self.assertGreater(increment.after.raw_life_fraction, 1.0)
        self.assertEqual(increment.after.economic_life_fraction, 1.0)
        self.assertAlmostEqual(
            interval_cost(
                69_990.0,
                70_010.0,
                replacement_cost_cny=3_500.0 * 600.0,
            ),
            300.0,
        )

    def test_post_eol_increment_is_zero_and_cost_is_not_multiplied_by_eight(
        self,
    ) -> None:
        from v2.models import fuel_cell_degradation as model

        interval_loss = _require_attr(
            self,
            model,
            "formal_fuel_cell_interval_life_loss",
        )
        interval_cost = _require_attr(
            self,
            model,
            "formal_fuel_cell_degradation_cost_cny",
        )

        after_eol = interval_loss(80_000.0, 90_000.0)
        self.assertEqual(after_eol.delta_economic_fraction, 0.0)
        self.assertGreater(after_eol.after.raw_life_fraction, 1.0)
        self.assertTrue(after_eol.after.eol_reached)
        self.assertEqual(
            interval_cost(
                80_000.0,
                90_000.0,
                replacement_cost_cny=3_500.0 * 600.0,
            ),
            0.0,
        )

        full_life_cost = interval_cost(
            0.0,
            70_000.0,
            replacement_cost_cny=3_500.0 * 600.0,
        )
        self.assertEqual(full_life_cost, 2_100_000.0)
        self.assertNotEqual(full_life_cost, 3_500.0 * 600.0 * 8.0)
        with self.assertRaises(ValueError):
            interval_cost(
                0.0,
                70_000.0,
                replacement_cost_cny=3_500.0 * 600.0 * 8.0,
            )


class BatteryPlantAndIntervalCostTests(unittest.TestCase):
    def test_plant_capacity_current_and_ampere_hour_units(self) -> None:
        from v2.models import battery_degradation as model

        energy_kwh = _require_attr(self, model, "BATTERY_ENERGY_CAPACITY_KWH")
        voltage_v = _require_attr(self, model, "BATTERY_NOMINAL_VOLTAGE_V")
        capacity_ah = _require_attr(
            self,
            model,
            "BATTERY_NOMINAL_CHARGE_CAPACITY_AH",
        )
        current_1c_a = _require_attr(self, model, "BATTERY_CURRENT_REF_1C_A")

        expected_capacity_ah = 624_000.0 / 432.0
        self.assertEqual(energy_kwh, 624.0)
        self.assertEqual(voltage_v, 432.0)
        self.assertAlmostEqual(capacity_ah, expected_capacity_ah)
        self.assertAlmostEqual(current_1c_a, expected_capacity_ah)

        one_hour = model.battery_degradation_step(
            1.0,
            current_1c_a,
            3_600.0,
            current_1c_a,
        )
        one_second = model.battery_degradation_step(
            1.0,
            current_1c_a,
            1.0,
            current_1c_a,
        )
        self.assertAlmostEqual(one_hour.raw_ah, expected_capacity_ah)
        self.assertAlmostEqual(
            one_second.raw_ah,
            expected_capacity_ah / 3_600.0,
        )
        self.assertAlmostEqual(one_hour.raw_ah, one_second.raw_ah * 3_600.0)

    def test_lifetime_denominator_uses_the_approved_factor(self) -> None:
        from v2.models import battery_degradation as model

        factor = _require_attr(
            self,
            model,
            "BATTERY_LIFETIME_THROUGHPUT_FACTOR",
        )
        capacity_ah = _require_attr(
            self,
            model,
            "BATTERY_NOMINAL_CHARGE_CAPACITY_AH",
        )
        q_lifetime_ah = _require_attr(self, model, "BATTERY_LIFETIME_Q_AH")

        self.assertEqual(factor, 15_000.0)
        self.assertAlmostEqual(q_lifetime_ah, 15_000.0 * capacity_ah)
        self.assertAlmostEqual(q_lifetime_ah, 21_666_666.666666664)

    def test_secondary_provenance_cannot_masquerade_as_primary_or_measured(
        self,
    ) -> None:
        from v2.models import battery_degradation as model

        factory = _require_attr(
            self,
            model,
            "formal_battery_lifetime_normalization",
        )
        normalization = factory()
        self.assertEqual(
            normalization.provenance_classification,
            "literature-based lifetime-throughput modeling assumption",
        )
        self.assertEqual(
            normalization.evidence_basis,
            "secondary literature basis",
        )
        for forbidden in (
            "Three Gorges Hydrogen Boat 1 measured battery lifetime",
            "manufacturer specification",
            "Yang project configuration parameter",
        ):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                replace(normalization, provenance_classification=forbidden)

    def test_interval_increment_before_and_across_eol(self) -> None:
        from v2.models import battery_degradation as model

        factory = _require_attr(
            self,
            model,
            "formal_battery_lifetime_normalization",
        )
        interval_loss = _require_attr(
            self,
            model,
            "formal_battery_interval_life_loss",
        )
        interval_cost = _require_attr(
            self,
            model,
            "formal_battery_degradation_cost_cny",
        )
        normalization = factory()
        lifetime = normalization.q_lifetime_ah
        replacement_cost = 2_000.0 * 624.0

        quarter = interval_loss(
            0.0,
            0.25 * lifetime,
            normalization=normalization,
        )
        self.assertAlmostEqual(quarter.delta_economic_fraction, 0.25)
        self.assertAlmostEqual(
            interval_cost(
                0.0,
                0.25 * lifetime,
                replacement_cost_cny=replacement_cost,
                normalization=normalization,
            ),
            0.25 * replacement_cost,
        )

        crossing = interval_loss(
            0.999 * lifetime,
            1.001 * lifetime,
            normalization=normalization,
        )
        self.assertAlmostEqual(crossing.delta_economic_fraction, 0.001)
        self.assertGreater(crossing.after.raw_life_fraction, 1.0)
        self.assertEqual(crossing.after.economic_life_fraction, 1.0)
        self.assertTrue(crossing.after.eol_reached)
        self.assertAlmostEqual(
            interval_cost(
                0.999 * lifetime,
                1.001 * lifetime,
                replacement_cost_cny=replacement_cost,
                normalization=normalization,
            ),
            0.001 * replacement_cost,
        )
        with self.assertRaises(ValueError):
            interval_cost(
                0.0,
                lifetime,
                replacement_cost_cny=2.0 * replacement_cost,
                normalization=normalization,
            )

    def test_post_eol_raw_throughput_continues_but_interval_cost_is_zero(
        self,
    ) -> None:
        from v2.models import battery_degradation as model

        factory = _require_attr(
            self,
            model,
            "formal_battery_lifetime_normalization",
        )
        interval_loss = _require_attr(
            self,
            model,
            "formal_battery_interval_life_loss",
        )
        interval_cost = _require_attr(
            self,
            model,
            "formal_battery_degradation_cost_cny",
        )
        normalization = factory()
        lifetime = normalization.q_lifetime_ah
        replacement_cost = 2_000.0 * 624.0

        after_eol = interval_loss(
            1.1 * lifetime,
            1.2 * lifetime,
            normalization=normalization,
        )
        self.assertGreater(after_eol.after.raw_life_fraction, 1.0)
        self.assertEqual(after_eol.after.economic_life_fraction, 1.0)
        self.assertTrue(after_eol.after.eol_reached)
        self.assertEqual(after_eol.delta_economic_fraction, 0.0)
        self.assertEqual(
            interval_cost(
                1.1 * lifetime,
                1.2 * lifetime,
                replacement_cost_cny=replacement_cost,
                normalization=normalization,
            ),
            0.0,
        )


class _StateProvider:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> tuple[float, ...]:
        self.calls += 1
        return (float(self.calls), 0.0)


class _FiveLedgerBackend:
    def __init__(self, ledgers: tuple[object, ...]) -> None:
        self.ledgers = ledgers
        self.calls = 0

    def execute_mpc_step(self, weights: object) -> object:
        from v2.envs.multirate_weight_env import MPCExecutionResult

        del weights
        ledger = self.ledgers[self.calls]
        self.calls += 1
        return MPCExecutionResult(ledger=ledger, done=False)


class ShoreAndMacroLedgerTests(unittest.TestCase):
    def test_terminal_recharge_uses_one_aggregate_efficiency(self) -> None:
        from v2 import economics
        from v2.models.battery_energy import formal_battery_efficiency

        recharge = economics.terminal_recharge_grid_energy
        parameters = inspect.signature(recharge).parameters
        self.assertNotIn(
            "converter_calibration",
            parameters,
            "formal terminal recharge must not apply a second converter factor",
        )
        energy = recharge(
            episode_initial_soc=0.60,
            episode_end_soc=0.50,
            battery_capacity_kwh=624.0,
            battery_efficiency=formal_battery_efficiency(),
        )
        expected_grid_kwh = 62.4 / 0.95
        self.assertAlmostEqual(energy.energy_kwh, expected_grid_kwh)
        self.assertAlmostEqual(
            economics.shore_energy_cost_cny(energy),
            expected_grid_kwh * 1.10,
        )

    def test_five_interval_ledgers_sum_increments_without_double_counting(
        self,
    ) -> None:
        from v2.config import TimeScaleConfig
        from v2.dqn.action_space import ActionCandidate
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MultiRateWeightEnvironment

        ledgers = tuple(
            RawCnyIntervalLedger(
                h2_cost_cny=float(index),
                fuel_cell_degradation_cost_cny=10.0,
                battery_degradation_cost_cny=4.0,
                shore_cost_cny=5.0 if index == 5 else 0.0,
            )
            for index in range(1, 6)
        )
        backend = _FiveLedgerBackend(ledgers)
        environment = MultiRateWeightEnvironment(
            timescale=TimeScaleConfig(30.0, 5, 5),
            action_catalog=(ActionCandidate(2, 3, 5),),
            backend=backend,
            state_provider=_StateProvider(),
            synthetic_test_mode=True,
        )

        environment.reset()
        transition = environment.step("w_2_3_5")

        self.assertEqual(backend.calls, 5)
        self.assertEqual(
            transition.ledger.components_cny,
            (15.0, 50.0, 20.0, 5.0),
        )
        cumulative_fc_cost_recharged_each_step = math.fsum(
            (10.0, 20.0, 30.0, 40.0, 50.0)
        )
        self.assertNotEqual(
            transition.ledger.fuel_cell_degradation_cost_cny,
            cumulative_fc_cost_recharged_each_step,
        )
        self.assertEqual(transition.learning_reward, -90.0)
        self.assertEqual(transition.raw_economic_cost_cny, 90.0)
        self.assertEqual(transition.failure_penalty_score, 0.0)


class EconomicClosurePreflightTests(unittest.TestCase):
    def test_preflight_reports_verified_economics_despite_dataset_blocker(self) -> None:
        from v2.preflight import CalibrationStatus, assess_formal_training_preflight

        report = assess_formal_training_preflight()
        by_key = {check.key: check for check in report.checks}
        expected = {
            "fc_degradation_normalization": CalibrationStatus.VERIFIED,
            "battery_q_lifetime_normalization": CalibrationStatus.VERIFIED,
            "shore_charging_efficiency": CalibrationStatus.VERIFIED,
            "ts_mpc": CalibrationStatus.VERIFIED,
            "n_mpc": CalibrationStatus.VERIFIED,
            "dqn_switch_steps": CalibrationStatus.VERIFIED,
            "tau_lpf": CalibrationStatus.VERIFIED,
        }
        for key, status in expected.items():
            with self.subTest(key=key):
                self.assertIn(key, by_key)
                if key in by_key:
                    self.assertEqual(by_key[key].status, status)

        evidence_fragments = {
            "fc_degradation_normalization": (
                "literature/model verified",
                "not vessel-measured",
            ),
            "battery_q_lifetime_normalization": (
                "SECONDARY_LITERATURE / LITERATURE-CALIBRATED",
                "secondary literature basis",
            ),
            "shore_charging_efficiency": (
                "literature-based aggregate assumption",
                "not vessel-measured",
            ),
            "ts_mpc": ("30 s", "frozen nominal control interval"),
        }
        for key, fragments in evidence_fragments.items():
            with self.subTest(evidence=key):
                self.assertIn(key, by_key)
                if key in by_key:
                    for fragment in fragments:
                        self.assertIn(fragment, by_key[key].evidence)

        self.assertTrue(report.ready)
        self.assertEqual(report.formal_training, "GO")
        self.assertEqual(
            by_key["shore_mode_sidecar"].status,
            CalibrationStatus.VERIFIED,
        )
        self.assertEqual(
            by_key["curated_dataset_release"].status,
            CalibrationStatus.VERIFIED,
        )


if __name__ == "__main__":
    unittest.main()
