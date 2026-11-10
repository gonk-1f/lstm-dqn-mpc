"""Cross-check the TD3 fuel-cell path against the single-module source data."""

from pathlib import Path
import unittest

import numpy as np
from scipy.interpolate import PchipInterpolator

from .economic_ems import make_economic_env
from v2.models.fuel_cell_efficiency import (
    FC_DATA_PROVENANCE,
    formal_fuel_cell_efficiency_map,
    hydrogen_mass_from_map_kg,
)


# Independent reference transcribed from the frozen Sheet1!A2:B12 cells.
SOURCE_ROWS = (
    (0.0, 0.0),
    (11.3938, 63.8554),
    (22.3676, 62.4096),
    (35.9133, 60.4819),
    (49.116, 58.7952),
    (62.146, 58.0723),
    (74.8327, 56.8675),
    (87.1766, 56.1446),
    (98.3215, 54.9398),
    (109.123, 53.7349),
    (116.668, 52.0482),
)
MODULE_COUNT = 6
MODULE_RATING_KW = 100.0
LHV_KWH_PER_KG = 120.0 / 3.6
H2_PRICE_CNY_PER_KG = 21.9
DATA_ROOT = Path(__file__).resolve().parents[3] / "data/processed/operating_dataset_v6_1s"


def reference_module_curve():
    source_kw, source_eta_percent = np.asarray(SOURCE_ROWS).T
    source_eta = source_eta_percent / 100.0
    source = PchipInterpolator(source_kw, source_eta, extrapolate=False)
    in_rating = source_kw < MODULE_RATING_KW
    return PchipInterpolator(
        np.append(source_kw[in_rating], MODULE_RATING_KW),
        np.append(source_eta[in_rating], source(MODULE_RATING_KW)),
        extrapolate=False,
    )


def reference_hydrogen_kg(total_kw, duration_s, single_curve):
    module_kw = total_kw / MODULE_COUNT
    if module_kw == 0.0:
        return 0.0
    module_energy_kwh = module_kw * (duration_s / 3600.0)
    module_eta = float(single_curve(module_kw))
    return sum([module_energy_kwh / (module_eta * LHV_KWH_PER_KG)] * MODULE_COUNT)


class FuelCellAggregationTests(unittest.TestCase):
    def setUp(self):
        self.aggregate = formal_fuel_cell_efficiency_map()
        self.single = reference_module_curve()

    def test_workbook_provenance_and_operating_knots(self):
        self.assertEqual(FC_DATA_PROVENANCE.raw_points, SOURCE_ROWS)
        self.assertEqual((FC_DATA_PROVENANCE.worksheet, FC_DATA_PROVENANCE.cell_range), ("Sheet1", "A2:B12"))
        self.assertEqual(
            FC_DATA_PROVENANCE.workbook_sha256,
            "906a0383f6e427a938a8343e9fb1428bfea0f8e2766f5ac5449fea0ccbde4a21",
        )
        np.testing.assert_allclose(np.asarray(self.aggregate.power_kw) / MODULE_COUNT, self.single.x)
        self.assertEqual(self.aggregate.rated_power_kw / MODULE_COUNT, 100.0)
        self.assertAlmostEqual(68.3628 / MODULE_COUNT, 11.3938, places=12)
        self.assertAlmostEqual(self.aggregate.eta(68.3628), 0.638554, places=12)
        self.assertAlmostEqual(self.aggregate.eta(600.0), float(self.single(100.0)), places=14)
        self.assertAlmostEqual(self.aggregate.eta(600.0), 0.5476352421776669, places=14)
        self.assertNotIn(100.0, [power for power, _ in SOURCE_ROWS])
        self.assertAlmostEqual(self.aggregate.power_kw[-2] / MODULE_COUNT, 98.3215, places=12)
        self.assertTrue(all(6.0 * power not in self.aggregate.power_kw for power, _ in SOURCE_ROWS[-2:]))

    def test_efficiency_equivalence_and_interpolation_bounds(self):
        powers = np.linspace(0.0, 600.0, 1201)
        efficiencies = self.aggregate.eta(powers)
        np.testing.assert_allclose(efficiencies, self.single(powers / MODULE_COUNT), rtol=1e-14, atol=1e-15)
        self.assertEqual(efficiencies[0], 0.0)
        self.assertTrue(np.all(efficiencies[1:] > 0.0))
        self.assertLessEqual(float(efficiencies.max()), 0.638554 + 1e-15)

    def test_module_hydrogen_sum_and_one_second_units(self):
        for total_kw in (0.001, 30.0, 68.3628, 175.0, 300.0, 593.0, 600.0):
            for duration_s in (1.0, 3.5):
                with self.subTest(total_kw=total_kw, duration_s=duration_s):
                    expected = reference_hydrogen_kg(total_kw, duration_s, self.single)
                    self.assertAlmostEqual(hydrogen_mass_from_map_kg(total_kw, duration_s, self.aggregate), expected, places=14)
        measured_point_mass = 68.3628 / (3600.0 * 0.638554 * LHV_KWH_PER_KG)
        self.assertAlmostEqual(hydrogen_mass_from_map_kg(68.3628, 1.0, self.aggregate), measured_point_mass, places=14)

    def test_zero_power_returns_zero_and_out_of_domain_is_rejected(self):
        with np.errstate(divide="raise", invalid="raise"):
            self.assertEqual(hydrogen_mass_from_map_kg(0.0, 1.0, self.aggregate), 0.0)
        for power in (-0.001, 600.001):
            with self.subTest(power=power):
                with self.assertRaises(ValueError):
                    self.aggregate.eta(power)
                with self.assertRaises(ValueError):
                    hydrogen_mass_from_map_kg(power, 1.0, self.aggregate)

    def test_td3_net_bus_power_hydrogen_ledger_and_reward(self):
        env = make_economic_env(DATA_ROOT, fixed_train_id="parent_005_seg01")
        self.addCleanup(env.close)
        _, reset = env.reset(seed=42)
        self.assertEqual(reset["sample_id"], "parent_005_seg01")
        for action in (-1.0, 0.0):
            with self.subTest(action=action):
                _, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
                self.assertFalse(terminated or truncated)
                fc_kw = info["fc_actual_kw"]
                lower_kw, upper_kw = info["feasible_fc_lower_kw"], info["feasible_fc_upper_kw"]
                mapped_kw = lower_kw + (action + 1.0) * 0.5 * (upper_kw - lower_kw)
                self.assertAlmostEqual(fc_kw, mapped_kw, places=3)
                self.assertAlmostEqual(fc_kw + info["battery_actual_kw"], info["load_kw"], places=10)
                expected_mass = reference_hydrogen_kg(fc_kw, 1.0, self.single)
                expected_cost = H2_PRICE_CNY_PER_KG * expected_mass
                self.assertAlmostEqual(info["hydrogen_mass_kg"], expected_mass, places=14)
                self.assertAlmostEqual(info["h2_cost_cny"], expected_cost, places=13)
                self.assertEqual(info["reward_h2_cost_cny"], info["h2_cost_cny"])
                physical_cost = expected_cost + sum(
                    info[name] for name in ("fuel_cell_degradation_cost_cny", "battery_degradation_cost_cny", "shore_cost_cny")
                )
                self.assertAlmostEqual(info["physical_economic_cost_cny"], physical_cost, places=12)
                learning_cost = expected_cost + sum(
                    info[name] for name in (
                        "battery_equivalent_hydrogen_step_cny", "fuel_cell_transient_cost_cny",
                        "fuel_cell_low_load_cost_cny", "fuel_cell_high_load_cost_cny",
                        "battery_degradation_cost_cny",
                    )
                )
                self.assertAlmostEqual(reward, -0.001 * learning_cost, places=14)


if __name__ == "__main__":
    unittest.main()
