import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TestFcDp0Curve(unittest.TestCase):
    def setUp(self):
        self.data_path = ROOT / "data" / "fuel_cell" / "FC_Dp0_curve_for_Python.csv"

    def test_curve_file_is_available_with_required_columns(self):
        self.assertTrue(self.data_path.exists())
        df = pd.read_csv(self.data_path)

        self.assertEqual(list(df.columns), ["P_sys_kW", "mH2_g_s_Dp0", "eta_percent_Dp0"])
        self.assertAlmostEqual(float(df["P_sys_kW"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(df["P_sys_kW"].iloc[-1]), 100.0)

    def test_dp0_mapping_uses_relative_load_not_absolute_100kw_system(self):
        from mpc.solvers.fc_dp0_curve import eta_dp0, h2_rate_gps_dp0

        df = pd.read_csv(self.data_path)
        ratios = np.array([0.0, 0.1, 0.5, 1.0])
        p_total = 600.0 * ratios
        p_map = 100.0 * ratios
        expected_h2 = 6.0 * np.interp(p_map, df["P_sys_kW"], df["mH2_g_s_Dp0"])
        expected_eta = np.interp(p_map, df["P_sys_kW"], df["eta_percent_Dp0"]) / 100.0

        np.testing.assert_allclose(h2_rate_gps_dp0(p_total), expected_h2, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(eta_dp0(p_total), expected_eta, rtol=1e-10, atol=1e-12)
        self.assertEqual(float(h2_rate_gps_dp0(0.0)), 0.0)
        self.assertEqual(float(eta_dp0(0.0)), 0.0)

    def test_eta_rises_then_slowly_declines(self):
        from mpc.solvers.fc_dp0_curve import eta_dp0

        eta = eta_dp0(np.array([0.0, 60.0, 300.0, 600.0]))

        self.assertAlmostEqual(float(eta[0]), 0.0)
        self.assertGreater(float(eta[1]), 0.60)
        self.assertGreater(float(eta[1]), float(eta[2]))
        self.assertGreater(float(eta[2]), float(eta[3]))

    def test_forced_origin_quadratic_rate_fit_preserves_zero_and_stays_close(self):
        from mpc.solvers.fc_dp0_curve import dp0_quadratic_coefficients, h2_rate_gps_dp0_quadratic

        df = pd.read_csv(self.data_path)
        ratios = df["P_sys_kW"].to_numpy(dtype=float) / 100.0
        exact = 6.0 * df["mH2_g_s_Dp0"].to_numpy(dtype=float)
        approx = h2_rate_gps_dp0_quadratic(600.0 * ratios)
        a1, a2 = dp0_quadratic_coefficients()

        self.assertGreater(a1, 0.0)
        self.assertGreater(a2, 0.0)
        self.assertEqual(float(h2_rate_gps_dp0_quadratic(0.0)), 0.0)
        self.assertLess(float(np.sqrt(np.mean((approx - exact) ** 2))), 0.02)


if __name__ == "__main__":
    unittest.main()
