from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_mpc_1s_n6_soc_clamping_diagnostic import (
    SyntheticCase,
    build_case_matrix,
    build_constant_profile,
    build_pulse_profile,
    clamping_candidate_config,
)


class TestSocClampingDiagnosticContract(unittest.TestCase):
    def test_constant_profile_has_exact_state_samples(self) -> None:
        times, loads = build_constant_profile()

        self.assertEqual(times.dtype, np.dtype(float))
        self.assertEqual(loads.dtype, np.dtype(float))
        self.assertEqual(times.shape, (3601,))
        self.assertEqual(loads.shape, (3601,))
        np.testing.assert_array_equal(times, np.arange(3601, dtype=float))
        np.testing.assert_array_equal(loads, np.full(3601, 300.0, dtype=float))

    def test_pulse_profile_has_exact_state_time_boundaries(self) -> None:
        times, loads = build_pulse_profile()

        self.assertEqual(times.dtype, np.dtype(float))
        self.assertEqual(loads.dtype, np.dtype(float))
        self.assertEqual(times.shape, (3601,))
        self.assertEqual(loads.shape, (3601,))
        np.testing.assert_array_equal(times, np.arange(3601, dtype=float))
        self.assertEqual(loads[599], 300.0)
        self.assertEqual(loads[600], 450.0)
        self.assertEqual(loads[719], 450.0)
        self.assertEqual(loads[720], 300.0)
        self.assertEqual(int(np.count_nonzero(loads == 450.0)), 120)
        self.assertTrue(np.all(loads[(times < 600.0) | (times >= 720.0)] == 300.0))

    def test_pulse_profile_does_not_share_storage_with_constant_profiles(self) -> None:
        retained_times, retained_loads = build_constant_profile()
        pulse_times, pulse_loads = build_pulse_profile()

        self.assertFalse(np.shares_memory(retained_times, pulse_times))
        self.assertFalse(np.shares_memory(retained_loads, pulse_loads))
        pulse_loads[:] = -1.0
        np.testing.assert_array_equal(retained_times, np.arange(3601, dtype=float))
        np.testing.assert_array_equal(
            retained_loads,
            np.full(3601, 300.0, dtype=float),
        )
        _, fresh_constant_loads = build_constant_profile()
        np.testing.assert_array_equal(
            fresh_constant_loads,
            np.full(3601, 300.0, dtype=float),
        )

    def test_case_matrix_has_exact_ids_values_and_order(self) -> None:
        self.assertEqual(
            build_case_matrix(),
            [
                SyntheticCase("constant_soc053_qsoc10", "constant", "QSOC_10", 10.0, 0.53),
                SyntheticCase("constant_soc055_qsoc10", "constant", "QSOC_10", 10.0, 0.55),
                SyntheticCase("constant_soc057_qsoc10", "constant", "QSOC_10", 10.0, 0.57),
                SyntheticCase("constant_soc053_qsoc20", "constant", "QSOC_20", 20.0, 0.53),
                SyntheticCase("constant_soc055_qsoc20", "constant", "QSOC_20", 20.0, 0.55),
                SyntheticCase("constant_soc057_qsoc20", "constant", "QSOC_20", 20.0, 0.57),
                SyntheticCase("pulse_soc055_qsoc10", "pulse", "QSOC_10", 10.0, 0.55),
                SyntheticCase("pulse_soc055_qsoc20", "pulse", "QSOC_20", 20.0, 0.55),
            ],
        )

    def test_synthetic_case_is_frozen(self) -> None:
        case = build_case_matrix()[0]

        with self.assertRaises(FrozenInstanceError):
            case.q_soc = 20.0  # type: ignore[misc]

    def test_candidate_configs_change_only_q_soc_and_keep_frozen_parameters(self) -> None:
        q10 = asdict(clamping_candidate_config(10.0))
        q20 = asdict(clamping_candidate_config(20.0))

        self.assertNotIn("candidate_id", q10)
        self.assertEqual(q10.pop("q_soc"), 10.0)
        self.assertEqual(q20.pop("q_soc"), 20.0)
        self.assertEqual(q10, q20)
        self.assertEqual(
            q10,
            {
                "horizon": 6,
                "dt_seconds": 1.0,
                "battery_capacity_kwh": 693.0,
                "battery_charge_max_kw": 346.5,
                "battery_discharge_max_kw": 346.5,
                "battery_power_ref_kw": 346.5,
                "fuel_cell_min_kw": 0.0,
                "fuel_cell_max_kw": 560.0,
                "fuel_cell_ramp_rate_kw_per_s": 48.0,
                "fuel_cell_ramp_kw": None,
                "soc_min": 0.2,
                "soc_max": 0.8,
                "soc_band": 0.05,
                "objective_variant": "simplified_normalized_literature_v1",
                "q_h2": 0.5,
                "q_batt": 0.05,
                "q_ramp": 0.0,
                "q_terminal_soc": 0.0,
            },
        )

    def test_candidate_config_rejects_unsupported_q_soc_with_clear_error(self) -> None:
        unsupported_values = (
            5.0,
            17.5,
            9.999,
            10.001,
            19.999,
            20.001,
            float("inf"),
            float("-inf"),
            float("nan"),
        )

        for q_soc in unsupported_values:
            with self.subTest(q_soc=q_soc):
                with self.assertRaisesRegex(
                    ValueError,
                    r"q_soc must be one of \{10\.0, 20\.0\}",
                ):
                    clamping_candidate_config(q_soc)


if __name__ == "__main__":
    unittest.main()
