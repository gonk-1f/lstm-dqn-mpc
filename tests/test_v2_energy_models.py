from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


EXPECTED_RAW_POINTS = (
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


class FuelCellEfficiencyTests(unittest.TestCase):
    def test_formal_map_preserves_exact_numeric_source_and_provenance(self) -> None:
        from v2.models.fuel_cell_efficiency import (
            FC_DATA_RAW_POINTS,
            FC_DATA_WORKBOOK_SHA256,
            FC_EFFICIENCY_CALIBRATION_STATUS,
            calibrated_fuel_cell_efficiency_map,
        )

        efficiency_map = calibrated_fuel_cell_efficiency_map()

        self.assertEqual(FC_DATA_RAW_POINTS, EXPECTED_RAW_POINTS)
        self.assertEqual(FC_EFFICIENCY_CALIBRATION_STATUS, "SOURCE_BACKED")
        self.assertEqual(
            FC_DATA_WORKBOOK_SHA256,
            "906a0383f6e427a938a8343e9fb1428bfea0f8e2766f5ac5449fea0ccbde4a21",
        )
        self.assertEqual(efficiency_map.provenance.workbook_sha256, FC_DATA_WORKBOOK_SHA256)
        self.assertEqual(efficiency_map.provenance.worksheet, "Sheet1")
        self.assertEqual(efficiency_map.provenance.cell_range, "A2:B12")
        self.assertEqual(efficiency_map.provenance.raw_points, EXPECTED_RAW_POINTS)
        self.assertIn("FC_Data.xlsx", efficiency_map.provenance.workbook_path)
        self.assertIn("net system output", efficiency_map.provenance.source_columns)
        self.assertIn("efficiency percent", efficiency_map.provenance.source_columns)

    def test_source_100_kw_axis_maps_to_600_kw_without_scaling_eta(self) -> None:
        from scipy.interpolate import PchipInterpolator
        from v2.models.fuel_cell_efficiency import calibrated_fuel_cell_efficiency_map

        efficiency_map = calibrated_fuel_cell_efficiency_map()
        raw = np.asarray(EXPECTED_RAW_POINTS)
        expected_endpoint = float(PchipInterpolator(raw[:, 0], raw[:, 1])(100.0) / 100.0)

        self.assertEqual(efficiency_map.rated_power_kw, 600.0)
        self.assertAlmostEqual(float(efficiency_map.eta(600.0)), expected_endpoint, places=14)
        self.assertAlmostEqual(float(efficiency_map.eta(6.0 * 49.116)), 0.587952, places=14)
        self.assertLess(float(efficiency_map.eta(600.0)), 1.0)
        self.assertIn("load fraction", efficiency_map.provenance.axis_transform)
        self.assertIn("100 kW", efficiency_map.provenance.axis_transform)
        self.assertIn("600 kW", efficiency_map.provenance.axis_transform)
        self.assertIn("unchanged", efficiency_map.provenance.efficiency_transform)
        self.assertIn("above 100 kW", efficiency_map.provenance.endpoint_transform)

    def test_pchip_is_exact_at_points_and_bounded_between_adjacent_points(self) -> None:
        from v2.models.fuel_cell_efficiency import calibrated_fuel_cell_efficiency_map

        efficiency_map = calibrated_fuel_cell_efficiency_map()
        formal_power = np.asarray(efficiency_map.power_kw)
        formal_eta = np.asarray(efficiency_map.efficiencies)
        np.testing.assert_allclose(efficiency_map.eta(formal_power), formal_eta, rtol=0.0, atol=1e-14)

        for left, right, eta_left, eta_right in zip(
            formal_power[:-1], formal_power[1:], formal_eta[:-1], formal_eta[1:]
        ):
            sampled = efficiency_map.eta(np.linspace(left, right, 31))
            self.assertTrue(np.all(sampled >= min(eta_left, eta_right) - 1e-12))
            self.assertTrue(np.all(sampled <= max(eta_left, eta_right) + 1e-12))

    def test_zero_efficiency_is_preserved_only_at_zero_power(self) -> None:
        from v2.models.fuel_cell_efficiency import calibrated_fuel_cell_efficiency_map

        efficiency_map = calibrated_fuel_cell_efficiency_map()

        self.assertEqual(efficiency_map.eta(0.0), 0.0)
        self.assertGreater(float(efficiency_map.eta(1.0)), 0.0)

    def test_lhv_and_hydrogen_step_use_kwh_per_kg_and_hours(self) -> None:
        from v2.models.fuel_cell_efficiency import (
            LHV_H2_KWH_PER_KG,
            LHV_H2_MJ_PER_KG,
            hydrogen_mass_kg,
        )

        self.assertEqual(LHV_H2_MJ_PER_KG, 120.0)
        self.assertAlmostEqual(LHV_H2_KWH_PER_KG, 33.333333333333336)
        self.assertAlmostEqual(
            hydrogen_mass_kg(100.0, 3600.0, efficiency=0.5, rated_power_kw=600.0),
            6.0,
        )
        self.assertEqual(
            hydrogen_mass_kg(0.0, 30.0, efficiency=0.0, rated_power_kw=600.0),
            0.0,
        )

    def test_hydrogen_from_map_shortcuts_physical_zero(self) -> None:
        from v2.models.fuel_cell_efficiency import (
            calibrated_fuel_cell_efficiency_map,
            hydrogen_mass_from_map_kg,
        )

        efficiency_map = calibrated_fuel_cell_efficiency_map()

        self.assertEqual(hydrogen_mass_from_map_kg(0.0, 1.0, efficiency_map), 0.0)
        expected = 600.0 / (efficiency_map.eta(600.0) * (120.0 / 3.6))
        self.assertAlmostEqual(
            hydrogen_mass_from_map_kg(600.0, 3600.0, efficiency_map), expected
        )

    def test_fuel_cell_domain_and_invalid_inputs_are_rejected(self) -> None:
        from v2.models.fuel_cell_efficiency import (
            calibrated_fuel_cell_efficiency_map,
            hydrogen_mass_kg,
        )

        efficiency_map = calibrated_fuel_cell_efficiency_map()
        for bad_power in (-1.0, 600.0001, np.nan, np.inf, True, "1"):
            with self.subTest(power=bad_power), self.assertRaises((TypeError, ValueError)):
                efficiency_map.eta(bad_power)
        for arguments in (
            (-1.0, 1.0, 0.5, 600.0),
            (601.0, 1.0, 0.5, 600.0),
            (1.0, 0.0, 0.5, 600.0),
            (1.0, np.nan, 0.5, 600.0),
            (1.0, 1.0, 0.0, 600.0),
            (1.0, 1.0, 1.01, 600.0),
            (1.0, 1.0, 0.5, 0.0),
            (True, 1.0, 0.5, 600.0),
            (1.0, "1", 0.5, 600.0),
        ):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                hydrogen_mass_kg(
                    arguments[0],
                    arguments[1],
                    efficiency=arguments[2],
                    rated_power_kw=arguments[3],
                )


class BatteryEnergyTests(unittest.TestCase):
    def test_formal_factory_is_source_backed_at_095(self) -> None:
        from v2.models.battery_energy import (
            BATTERY_EFFICIENCY_CALIBRATION_STATUS,
            BATTERY_EFFICIENCY_SOURCE_DOI,
            formal_battery_efficiency,
        )

        efficiency = formal_battery_efficiency()

        self.assertEqual(efficiency.require_calibrated(), (0.95, 0.95))
        self.assertEqual(BATTERY_EFFICIENCY_CALIBRATION_STATUS, "SOURCE_BACKED")
        self.assertEqual(
            BATTERY_EFFICIENCY_SOURCE_DOI,
            "10.11930/j.issn.1004-9649.202507065",
        )
        self.assertEqual(efficiency.source_reference, BATTERY_EFFICIENCY_SOURCE_DOI)
        self.assertEqual(efficiency.source_location, "Table 3")

    def test_discharge_and_charge_have_exact_bus_to_battery_signs(self) -> None:
        from v2.models.battery_energy import next_soc

        self.assertAlmostEqual(
            next_soc(0.6, 100.0, 3600.0, 1000.0, eta_chg=0.8, eta_dis=0.8),
            0.475,
        )
        self.assertAlmostEqual(
            next_soc(0.5, -100.0, 3600.0, 1000.0, eta_chg=0.8, eta_dis=0.8),
            0.58,
        )

    def test_formal_efficiency_is_used_in_soc_dynamics_without_clamping(self) -> None:
        from v2.models.battery_energy import formal_battery_efficiency, next_soc

        eta_chg, eta_dis = formal_battery_efficiency().require_calibrated()

        self.assertAlmostEqual(
            next_soc(0.01, 100.0, 3600.0, 100.0, eta_chg=eta_chg, eta_dis=eta_dis),
            0.01 - 1.0 / 0.95,
        )
        self.assertAlmostEqual(
            next_soc(0.99, -100.0, 3600.0, 100.0, eta_chg=eta_chg, eta_dis=eta_dis),
            0.99 + 0.95,
        )

    def test_missing_or_unproven_efficiency_fails(self) -> None:
        from v2.models.battery_energy import BatteryEfficiency

        for efficiency in (
            BatteryEfficiency(),
            BatteryEfficiency(eta_chg=None, eta_dis=0.95, source_reference="paper"),
            BatteryEfficiency(eta_chg=0.95, eta_dis=0.95),
            BatteryEfficiency(
                eta_chg=0.95,
                eta_dis=0.95,
                source_reference="paper",
                source_location=None,
            ),
        ):
            with self.subTest(efficiency=efficiency), self.assertRaises(ValueError):
                efficiency.require_calibrated()

    def test_battery_inputs_reject_nonfinite_bool_and_coercible_text(self) -> None:
        from v2.models.battery_energy import BatteryEfficiency, next_soc

        for bad in (np.nan, np.inf, True, "0.95"):
            with self.subTest(efficiency=bad), self.assertRaises((TypeError, ValueError)):
                BatteryEfficiency(
                    eta_chg=bad,
                    eta_dis=0.95,
                    source_reference="paper",
                    source_location="table",
                ).require_calibrated()
            with self.subTest(dynamics=bad), self.assertRaises((TypeError, ValueError)):
                next_soc(bad, 0.0, 1.0, 1.0, eta_chg=0.95, eta_dis=0.95)

    def test_battery_duration_capacity_and_efficiency_domains_are_strict(self) -> None:
        from v2.models.battery_energy import next_soc

        for arguments in (
            (0.5, 0.0, 0.0, 1.0, 0.95, 0.95),
            (0.5, 0.0, 1.0, 0.0, 0.95, 0.95),
            (0.5, 0.0, 1.0, 1.0, 0.0, 0.95),
            (0.5, 0.0, 1.0, 1.0, 0.95, 1.01),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                next_soc(*arguments[:4], eta_chg=arguments[4], eta_dis=arguments[5])


class V2EnergyBoundaryTests(unittest.TestCase):
    def test_formal_v2_python_modules_do_not_reference_legacy_curve(self) -> None:
        forbidden = "d" + "p" + "0"
        for path in (ROOT / "src" / "v2").rglob("*.py"):
            with self.subTest(path=path):
                self.assertNotIn(forbidden, path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
