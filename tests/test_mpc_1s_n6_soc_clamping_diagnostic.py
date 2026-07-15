from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_mpc_1s_n6_soc_clamping_diagnostic import (
    SyntheticCase,
    _soc_code,
    annotate_correction_power,
    build_case_matrix,
    build_constant_profile,
    build_pulse_profile,
    clamping_candidate_config,
    longest_true_run,
    recovery_milestone,
    run_synthetic_case,
    steady_state_mask,
    summarize_window,
)


class TestSocClampingDiagnosticContract(unittest.TestCase):
    def test_soc_code_does_not_truncate_point_fifty_seven(self) -> None:
        self.assertEqual(_soc_code(0.53), "053")
        self.assertEqual(_soc_code(0.55), "055")
        self.assertEqual(_soc_code(0.57), "057")

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


class TestCorrectionPowerMetrics(unittest.TestCase):
    def test_annotation_uses_action_prestate_sign_and_does_not_mutate_input(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.57, 0.55, 0.53],
                "SOC_actual": [0.60, 0.50, 0.53, 0.57],
                "P_batt_actual_kw": [-10.0, 12.0, 50.0, 8.0],
            }
        )
        original = frame.copy(deep=True)

        annotated = annotate_correction_power(frame)

        pd.testing.assert_frame_equal(frame, original)
        self.assertIsNot(annotated, frame)
        np.testing.assert_allclose(
            annotated["soc_error_before"],
            [-0.02, 0.02, 0.0, -0.02],
            rtol=0.0,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            annotated["P_correction_kw"],
            [10.0, 12.0, 0.0, -8.0],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            annotated["positive_P_correction_kw"],
            [10.0, 12.0, 0.0, 0.0],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            annotated["active_near_reference_correction"].tolist(),
            [True, True, False, False],
        )

    def test_active_tolerance_and_power_threshold_use_inclusive_error_only(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [
                    0.53 - 1.0e-12,
                    0.57 + 1.0e-12,
                    np.nextafter(0.53 - 1.0e-12, -np.inf),
                    np.nextafter(
                        np.nextafter(0.57 + 1.0e-12, np.inf),
                        np.inf,
                    ),
                    0.53,
                    0.53,
                ],
                "P_batt_actual_kw": [-10.0, 10.0, -10.0, 10.0, -5.0, -5.1],
            }
        )

        annotated = annotate_correction_power(frame)

        self.assertEqual(
            annotated["active_near_reference_correction"].tolist(),
            [True, True, False, False, False, True],
        )

    def test_annotation_honors_a_finite_custom_soc_reference(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.52, 0.58],
                "P_batt_actual_kw": [-2.0, 3.0],
            }
        )

        annotated = annotate_correction_power(frame, soc_reference=0.60)

        np.testing.assert_allclose(annotated["soc_error_before"], [-0.08, -0.02])
        np.testing.assert_allclose(annotated["P_correction_kw"], [2.0, -3.0])

    def test_longest_true_run_handles_series_arrays_and_empty_masks(self) -> None:
        self.assertEqual(
            longest_true_run(pd.Series([False, True, True, False, True])),
            2,
        )
        self.assertEqual(
            longest_true_run(
                np.array([True, False, True, True, True, False], dtype=bool)
            ),
            3,
        )
        self.assertEqual(longest_true_run(np.array([], dtype=bool)), 0)
        self.assertEqual(longest_true_run(pd.Series([False, False])), 0)
        self.assertEqual(longest_true_run([True, True, True]), 3)

    def test_summary_uses_exact_kw_second_energy_units_and_population_soc_stats(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.53],
                "SOC_actual": [0.54, 0.56],
                "P_batt_actual_kw": [-1.0, 1.0],
                "P_fc_actual_kw": [2.0, 0.0],
                "load_actual_kw": [1.0, 1.0],
                "h2_kg_step": [0.1, 0.2],
            },
            index=[10, 11],
        )
        mask = pd.Series([True, True], index=frame.index)

        metrics = summarize_window(frame, mask, dt_seconds=1.0)

        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["duration_s"], 2.0)
        self.assertAlmostEqual(metrics["soc_min"], 0.54)
        self.assertAlmostEqual(metrics["soc_max"], 0.56)
        self.assertAlmostEqual(metrics["soc_range"], 0.02)
        self.assertAlmostEqual(metrics["soc_std"], 0.01)
        self.assertAlmostEqual(metrics["soc_final"], 0.56)
        self.assertAlmostEqual(metrics["mean_abs_soc_error"], 0.01)
        self.assertAlmostEqual(metrics["E_fc_surplus_kwh"], 1.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_charge_kwh"], 1.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_discharge_kwh"], 1.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_throughput_kwh"], 2.0 / 3600.0)
        self.assertAlmostEqual(
            metrics["E_batt_throughput_kwh"],
            metrics["E_batt_charge_kwh"] + metrics["E_batt_discharge_kwh"],
        )
        self.assertAlmostEqual(metrics["corrective_energy_kwh"], 1.0 / 3600.0)
        self.assertAlmostEqual(
            metrics["wrong_direction_energy_kwh"],
            1.0 / 3600.0,
        )
        self.assertAlmostEqual(metrics["H2_total_kg"], 0.3)
        self.assertAlmostEqual(metrics["mean_P_fc_actual_kw"], 1.0)
        self.assertAlmostEqual(metrics["mean_P_batt_actual_kw"], 0.0)
        self.assertAlmostEqual(metrics["mean_load_actual_kw"], 1.0)

    def test_summary_includes_zeros_in_positive_power_statistics(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.57, 0.55, 0.53],
                "SOC_actual": [0.531, 0.569, 0.55, 0.54],
                "P_batt_actual_kw": [-10.0, 12.0, 50.0, 8.0],
                "P_fc_actual_kw": [300.0, 300.0, 300.0, 300.0],
                "load_actual_kw": [300.0, 300.0, 300.0, 300.0],
                "h2_kg_step": [0.0, 0.0, 0.0, 0.0],
            }
        )

        metrics = summarize_window(frame, np.ones(4, dtype=bool))

        self.assertAlmostEqual(metrics["mean_positive_P_correction_kw"], 5.5)
        self.assertAlmostEqual(metrics["p95_positive_P_correction_kw"], 11.7)
        self.assertAlmostEqual(metrics["max_positive_P_correction_kw"], 12.0)
        self.assertAlmostEqual(metrics["ratio_active_correction"], 0.5)
        self.assertAlmostEqual(metrics["active_correction_s"], 2.0)
        self.assertAlmostEqual(metrics["longest_active_correction_s"], 2.0)
        self.assertAlmostEqual(metrics["corrective_energy_kwh"], 22.0 / 3600.0)
        self.assertAlmostEqual(metrics["wrong_direction_energy_kwh"], 8.0 / 3600.0)

    def test_summary_returns_finite_zero_positive_metrics_when_none_are_positive(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53],
                "SOC_actual": [0.52],
                "P_batt_actual_kw": [1.0],
                "P_fc_actual_kw": [1.0],
                "load_actual_kw": [1.0],
                "h2_kg_step": [0.0],
            }
        )

        metrics = summarize_window(frame, [True])

        for name in (
            "mean_positive_P_correction_kw",
            "p95_positive_P_correction_kw",
            "max_positive_P_correction_kw",
            "ratio_active_correction",
            "active_correction_s",
            "longest_active_correction_s",
            "corrective_energy_kwh",
        ):
            with self.subTest(name=name):
                self.assertTrue(np.isfinite(metrics[name]))
                self.assertEqual(metrics[name], 0.0)

    def test_summary_scales_power_energies_and_durations_but_not_step_hydrogen(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.53, 0.53],
                "SOC_actual": [0.531, 0.532, 0.533],
                "P_batt_actual_kw": [-10.0, -10.0, 8.0],
                "P_fc_actual_kw": [310.0, 310.0, 300.0],
                "load_actual_kw": [300.0, 300.0, 300.0],
                "h2_kg_step": [0.1, 0.2, 0.3],
            }
        )

        metrics = summarize_window(frame, [True, True, True], dt_seconds=2.0)

        self.assertEqual(metrics["duration_s"], 6.0)
        self.assertEqual(metrics["active_correction_s"], 4.0)
        self.assertEqual(metrics["longest_active_correction_s"], 4.0)
        self.assertAlmostEqual(metrics["corrective_energy_kwh"], 40.0 / 3600.0)
        self.assertAlmostEqual(metrics["wrong_direction_energy_kwh"], 16.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_fc_surplus_kwh"], 40.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_charge_kwh"], 40.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_discharge_kwh"], 16.0 / 3600.0)
        self.assertAlmostEqual(metrics["E_batt_throughput_kwh"], 56.0 / 3600.0)
        self.assertAlmostEqual(metrics["H2_total_kg"], 0.6)

    def test_summary_does_not_bridge_active_runs_across_unselected_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.53, 0.53],
                "SOC_actual": [0.531, 0.532, 0.533],
                "P_batt_actual_kw": [-10.0, -10.0, -10.0],
                "P_fc_actual_kw": [310.0, 310.0, 310.0],
                "load_actual_kw": [300.0, 300.0, 300.0],
                "h2_kg_step": [0.0, 0.0, 0.0],
            }
        )

        metrics = summarize_window(frame, [True, False, True])

        self.assertEqual(metrics["active_correction_s"], 2.0)
        self.assertEqual(metrics["longest_active_correction_s"], 1.0)

    def test_summary_rejects_empty_misaligned_or_invalid_duration_windows(self) -> None:
        frame = pd.DataFrame(
            {
                "SOC_before": [0.53, 0.54],
                "SOC_actual": [0.53, 0.54],
                "P_batt_actual_kw": [0.0, 0.0],
                "P_fc_actual_kw": [1.0, 1.0],
                "load_actual_kw": [1.0, 1.0],
                "h2_kg_step": [0.0, 0.0],
            },
            index=[10, 11],
        )

        with self.assertRaisesRegex(ValueError, "empty analysis window"):
            summarize_window(frame, [False, False])
        with self.assertRaisesRegex(ValueError, "mask.*length"):
            summarize_window(frame, [True])
        with self.assertRaisesRegex(ValueError, "mask.*index"):
            summarize_window(frame, pd.Series([True, False], index=[0, 1]))
        for dt_seconds in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(dt_seconds=dt_seconds):
                with self.assertRaisesRegex(ValueError, "dt_seconds"):
                    summarize_window(frame, [True, False], dt_seconds=dt_seconds)


class TestRecoveryMilestones(unittest.TestCase):
    def test_fractional_milestones_use_first_reach_and_detect_later_departure(self) -> None:
        times = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        soc = np.array([0.53, 0.534, 0.536, 0.539, 0.541, 0.53])

        self.assertEqual(
            recovery_milestone(
                times,
                soc,
                initial_soc=0.53,
                reduction_fraction=0.25,
            ),
            (20.0, False),
        )
        self.assertEqual(
            recovery_milestone(
                times,
                soc,
                initial_soc=0.53,
                reduction_fraction=0.50,
            ),
            (40.0, False),
        )

    def test_milestone_reports_never_reached_and_maintained_reach(self) -> None:
        self.assertEqual(
            recovery_milestone(
                [0.0, 10.0, 20.0],
                [0.53, 0.532, 0.534],
                initial_soc=0.53,
                reduction_fraction=0.25,
            ),
            (None, False),
        )
        self.assertEqual(
            recovery_milestone(
                [0.0, 10.0, 20.0, 30.0],
                [0.53, 0.536, 0.54, 0.545],
                initial_soc=0.53,
                reduction_fraction=0.50,
            ),
            (20.0, True),
        )

    def test_milestone_is_symmetric_and_honors_a_custom_reference(self) -> None:
        self.assertEqual(
            recovery_milestone(
                [0.0, 10.0, 20.0, 30.0],
                [0.57, 0.566, 0.564, 0.56],
                initial_soc=0.57,
                reduction_fraction=0.25,
            ),
            (20.0, True),
        )
        self.assertEqual(
            recovery_milestone(
                [0.0, 10.0, 20.0],
                [0.48, 0.49, 0.495],
                initial_soc=0.48,
                reduction_fraction=0.50,
                soc_reference=0.50,
            ),
            (10.0, True),
        )

    def test_milestone_validates_trajectory_and_arguments(self) -> None:
        invalid_calls = (
            (([], []), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0], [0.53, 0.54]), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0, 2.0, 1.0], [0.53, 0.54, 0.55]), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0, 1.0, 1.0], [0.53, 0.54, 0.55]), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0, float("nan")], [0.53, 0.54]), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0, 1.0], [0.53, float("inf")]), {"initial_soc": 0.53, "reduction_fraction": 0.25}),
            (([0.0], [0.53]), {"initial_soc": float("nan"), "reduction_fraction": 0.25}),
            (([0.0], [0.53]), {"initial_soc": 0.53, "reduction_fraction": -0.01}),
            (([0.0], [0.53]), {"initial_soc": 0.53, "reduction_fraction": 1.01}),
            (([0.0], [0.53]), {"initial_soc": 0.53, "reduction_fraction": float("nan")}),
            (([0.0], [0.53]), {"initial_soc": 0.53, "reduction_fraction": 0.25, "soc_reference": float("inf")}),
        )

        for args, kwargs in invalid_calls:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    recovery_milestone(*args, **kwargs)


class TestSteadyStateMask(unittest.TestCase):
    @staticmethod
    def constant_frame(sample_count: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "state_time_s": np.arange(sample_count, dtype=float),
                "load_actual_kw": np.full(sample_count, 300.0),
                "P_fc_actual_kw": np.full(sample_count, 300.0),
            }
        )

    def test_constant_region_obeys_full_window_and_start_end_boundaries(self) -> None:
        frame = self.constant_frame(250)

        mask = steady_state_mask(frame)

        self.assertIsInstance(mask, pd.Series)
        self.assertTrue(mask.index.equals(frame.index))
        self.assertFalse(mask.iloc[:60].any())
        self.assertTrue(mask.iloc[60])
        self.assertTrue(mask.iloc[189])
        self.assertFalse(mask.iloc[190:].any())
        self.assertTrue(mask.iloc[60:190].all())

    def test_jump_excludes_sixty_full_seconds_and_exact_boundary_is_eligible(self) -> None:
        frame = self.constant_frame(220)
        frame.loc[frame["state_time_s"] >= 80.0, "load_actual_kw"] = 320.0

        mask = steady_state_mask(
            frame,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertFalse(mask.iloc[58])
        self.assertTrue(mask.iloc[59])
        self.assertTrue(mask.iloc[79])
        self.assertFalse(mask.iloc[80:140].any())
        self.assertTrue(mask.iloc[140])

    def test_jump_uses_absolute_strict_threshold_and_configured_delay(self) -> None:
        frame = self.constant_frame(8)
        frame.loc[frame["state_time_s"] >= 2.0, "load_actual_kw"] = 290.0
        frame.loc[frame["state_time_s"] >= 4.0, "load_actual_kw"] = 270.0

        mask = steady_state_mask(
            frame,
            trailing_window_steps=1,
            post_jump_exclusion_s=2.0,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertTrue(mask.iloc[2])
        self.assertTrue(mask.iloc[3])
        self.assertFalse(mask.iloc[4:6].any())
        self.assertTrue(mask.iloc[6])

    def test_fc_at_exact_saturation_is_rejected(self) -> None:
        frame = self.constant_frame(120)
        frame.loc[80, "P_fc_actual_kw"] = 560.0

        mask = steady_state_mask(
            frame,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertTrue(mask.iloc[79])
        self.assertFalse(mask.iloc[80])
        self.assertTrue(mask.iloc[81])

    def test_slow_ramp_is_rejected_by_trailing_range_not_one_step_slope(self) -> None:
        frame = self.constant_frame(120)
        frame["load_actual_kw"] = 300.0 + 0.1 * frame["state_time_s"]
        self.assertLessEqual(
            float(frame["load_actual_kw"].diff().abs().max()),
            1.0,
        )

        mask = steady_state_mask(
            frame,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertFalse(mask.any())
        self.assertFalse(mask.iloc[59])
        self.assertFalse(mask.iloc[-1])

    def test_trailing_load_range_includes_exact_five_kw_boundary(self) -> None:
        frame = self.constant_frame(61)
        frame.loc[59, "load_actual_kw"] = 305.0
        frame.loc[60, "load_actual_kw"] = 305.1

        mask = steady_state_mask(
            frame,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertTrue(mask.iloc[59])
        self.assertFalse(mask.iloc[60])

    def test_state_time_takes_precedence_and_time_s_is_the_fallback(self) -> None:
        frame = self.constant_frame(4)
        frame["time_s"] = [3.0, 2.0, 1.0, 0.0]

        state_mask = steady_state_mask(
            frame,
            trailing_window_steps=2,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )
        fallback = frame.drop(columns=["state_time_s", "time_s"]).copy()
        fallback.insert(0, "time_s", np.arange(4, dtype=float))
        fallback.index = [10, 11, 12, 13]
        fallback_mask = steady_state_mask(
            fallback,
            trailing_window_steps=2,
            start_exclusion_s=0.0,
            end_exclusion_s=0.0,
        )

        self.assertEqual(state_mask.tolist(), [False, True, True, True])
        self.assertEqual(fallback_mask.tolist(), [False, True, True, True])
        self.assertTrue(fallback_mask.index.equals(fallback.index))

    def test_non_monotonic_or_non_one_second_state_times_are_rejected(self) -> None:
        for times in ([0.0, 2.0, 1.0], [0.0, 1.0, 3.0]):
            frame = pd.DataFrame(
                {
                    "state_time_s": times,
                    "load_actual_kw": [300.0, 300.0, 300.0],
                    "P_fc_actual_kw": [300.0, 300.0, 300.0],
                }
            )
            with self.subTest(times=times):
                with self.assertRaisesRegex(ValueError, "1 s"):
                    steady_state_mask(frame, trailing_window_steps=1)


class TestSyntheticCaseExecution(unittest.TestCase):
    @staticmethod
    def valid_runner_frames(initial_soc: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        state_times = np.arange(1, 3601, dtype=float)
        controls = pd.DataFrame(
            {
                "decision_time_s": state_times - 1.0,
                "time_s": state_times,
                "SOC_before": np.full(3600, initial_soc),
                "SOC_actual": np.full(3600, initial_soc),
                "prev_fc_actual_kw": np.full(3600, 300.0),
                "P_fc_actual_kw": np.full(3600, 300.0),
                "P_batt_actual_kw": np.zeros(3600),
                "load_actual_kw": np.full(3600, 300.0),
                "h2_kg_step": np.full(3600, 0.01),
                "success": np.ones(3600, dtype=bool),
            }
        )
        solver = pd.DataFrame({"success": np.ones(3600, dtype=bool)})
        return controls, solver

    def test_case_execution_passes_exact_profile_initial_state_and_config(self) -> None:
        case = next(
            item
            for item in build_case_matrix()
            if item.case_id == "constant_soc053_qsoc20"
        )
        controls, solver = self.valid_runner_frames(case.initial_soc)
        physical = {
            "closed_loop_complete": True,
            "physical_infeasible_point_count": 0,
            "solver_failure_count": 0,
        }

        with (
            patch(
                "run_mpc_1s_n6_soc_clamping_diagnostic.run_voyage",
                return_value=(controls, solver),
            ) as mocked_run,
            patch(
                "run_mpc_1s_n6_soc_clamping_diagnostic.build_candidate_metrics",
                return_value=(physical, pd.DataFrame(), pd.DataFrame()),
            ),
        ):
            trajectory, metrics = run_synthetic_case(case)

        kwargs = mocked_run.call_args.kwargs
        self.assertEqual(kwargs["voyage_id"], case.case_id)
        self.assertEqual(kwargs["candidate_id"], "QSOC_20")
        self.assertEqual(kwargs["initial_soc"], 0.53)
        self.assertEqual(kwargs["config"].q_soc, 20.0)
        np.testing.assert_array_equal(kwargs["times_s"], np.arange(3601, dtype=float))
        np.testing.assert_array_equal(kwargs["loads_kw"], np.full(3601, 300.0))
        self.assertEqual(len(trajectory), 3600)
        self.assertEqual(metrics["case_id"], case.case_id)
        self.assertEqual(metrics["initial_soc"], 0.53)

    def test_case_execution_rejects_incomplete_or_failed_control(self) -> None:
        case = build_case_matrix()[0]
        controls, solver = self.valid_runner_frames(case.initial_soc)

        for broken in (controls.iloc[:-1].copy(), controls.assign(success=False)):
            with self.subTest(row_count=len(broken)):
                with patch(
                    "run_mpc_1s_n6_soc_clamping_diagnostic.run_voyage",
                    return_value=(broken, solver),
                ):
                    with self.assertRaises(ValueError):
                        run_synthetic_case(case)


if __name__ == "__main__":
    unittest.main()
