"""Existence checks use physical limits independently of any controller."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/utils/final_dataset_feasibility.py"
spec = importlib.util.spec_from_file_location("final_dataset_feasibility", MODULE_PATH)
audit_module = importlib.util.module_from_spec(spec) if MODULE_PATH.exists() else None
if audit_module is not None:
    sys.modules[spec.name] = audit_module
    spec.loader.exec_module(audit_module)


class FeasibilityAuditTests(unittest.TestCase):
    def audit(self, loads, **kwargs):
        self.assertIsNotNone(audit_module, "independent feasibility module must exist")
        return audit_module.audit_feasibility(loads, **kwargs)

    def assert_witness(self, loads, result):
        self.assertEqual(result["status"], "feasible")
        self.assertIs(result["feasible"], True)
        w = result["witness"]
        fc, batt, soc = (np.asarray(w[k]) for k in ("p_fc_kw", "p_batt_kw", "soc"))
        self.assertEqual(fc.shape, (len(loads) - 1,))
        np.testing.assert_allclose(fc + batt, loads[1:], atol=1e-7)
        np.testing.assert_allclose(soc, .55 - np.cumsum(batt) / (624 * 3600), atol=1e-10)
        self.assertGreaterEqual(fc.min(), -1e-6)
        self.assertLessEqual(fc.max(), 600 + 1e-6)
        self.assertGreaterEqual(batt.min(), -624 - 1e-6)
        self.assertLessEqual(batt.max(), 1248 + 1e-6)
        self.assertLessEqual(np.abs(np.diff(np.r_[np.clip(loads[0], 0, 600), fc])).max(), 48 + 1e-6)
        self.assertGreaterEqual(soc.min(), .2 - 1e-9)
        self.assertLessEqual(soc.max(), .8 + 1e-9)
        self.assertTrue(result["witness_validation"]["valid"])

    def test_constant_load_feasible_and_initial_sample_not_executed(self):
        loads = np.full(401, 280.)
        self.assert_witness(loads, self.audit(loads, return_witness=True))

    def test_initial_fc_ramp_can_make_power_feasible_load_infeasible(self):
        result = self.audit([0., 1400.])
        self.assertEqual(result["status"], "infeasible")
        self.assertIs(result["feasible"], False)
        self.assertEqual(result["solver_status"], 2)

    def test_lower_soc_energy_exhaustion_proven_infeasible(self):
        result = self.audit(np.full(7001, 800.))
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["solver_status"], 2)

    def test_upper_soc_regenerative_energy_proven_infeasible(self):
        result = self.audit(np.full(1001, -624.))
        self.assertEqual(result["status"], "infeasible")
        self.assertEqual(result["solver_status"], 2)

    def test_exact_lower_soc_endpoint_is_feasible(self):
        # 200 kW discharge for 3931.2 seconds worth of energy.
        loads = np.r_[600., np.full(3931, 800.), 640.]
        result = self.audit(loads, return_witness=True)
        self.assert_witness(loads, result)
        self.assertAlmostEqual(result["final_soc"], .2, places=9)

    def test_exact_upper_soc_endpoint_is_feasible(self):
        loads = np.r_[0., np.full(900, -624.)]
        result = self.audit(loads, return_witness=True)
        self.assert_witness(loads, result)
        self.assertAlmostEqual(result["final_soc"], .8, places=9)

    def test_long_40000_sample_audit_returns_checked_witness(self):
        loads = np.r_[250., 250. + 180 * np.sin(np.arange(39999) / 30)]
        self.assert_witness(loads, self.audit(loads, return_witness=True))

    def test_time_varying_energy_allocation_uses_complete_lp(self):
        # A single envelope mixing coefficient cannot serve both phases;
        # an actual feasible schedule must shift charging across time.
        loads = np.r_[100., np.full(9000, 100.), np.full(3000, 800.)]
        result = self.audit(loads, return_witness=True)
        self.assertEqual(result["solver_status"], 0)
        self.assert_witness(loads, result)

    def test_default_result_is_json_serializable_and_does_not_mutate_input(self):
        import json
        loads = np.array([100., 120., 300.])
        saved = loads.copy()
        result = self.audit(loads)
        self.assertNotIn("witness", result)
        json.dumps(result, allow_nan=False)
        np.testing.assert_array_equal(loads, saved)

    def test_solver_exception_is_unknown(self):
        self.assertIsNotNone(audit_module)
        with patch.object(audit_module, "linprog", side_effect=RuntimeError("solver unavailable")):
            result = self.audit([0., 1400.])
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["feasible"])
        self.assertIn("solver unavailable", result["solver_message"])

    def test_invalid_series_rejected_without_infeasibility_claim(self):
        for loads in ([], [200], [0, np.nan], [0, np.inf], [[0, 1]]):
            with self.subTest(loads=loads), self.assertRaises(ValueError):
                self.audit(loads)

    def test_solver_timeout_is_unknown_not_infeasible(self):
        self.assertIsNotNone(audit_module)
        with patch.object(audit_module, "linprog", return_value=SimpleNamespace(status=1, message="time limit", x=None)):
            result = self.audit([0., 1400.])
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["feasible"])

    def test_solver_success_without_valid_witness_is_unknown(self):
        self.assertIsNotNone(audit_module)
        with patch.object(audit_module, "linprog", return_value=SimpleNamespace(status=0, message="ok", x=np.zeros(2))):
            result = self.audit([0., 1400.])
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["feasible"])


if __name__ == "__main__":
    unittest.main()
