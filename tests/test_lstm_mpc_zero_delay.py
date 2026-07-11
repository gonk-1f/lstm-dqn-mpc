from __future__ import annotations

import sys
import unittest
from pathlib import Path
import json
import hashlib
import tempfile

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
MAIN_ROOT = SRC_ROOT / "main"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from run_lstm_mpc_test import (  # noqa: E402
    CURRENT_FIXED_WEIGHT_SET,
    DEFAULT_WEIGHT_SETS,
    TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET,
    TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET,
    TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET,
    TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET,
    build_run_file_hashes,
    build_action_table,
    build_mpc_load_ref,
    clean_reporting_dict_for_weight_set,
    clean_reporting_frame_for_weight_set,
    clean_config_payload_for_weight_set,
    compute_closed_loop_metrics,
    load_weight_sets,
    make_debug_timing_row,
    make_run_id,
    mpc_config_from_weights,
    CONTROL_LAYER_SCOPE,
    REFERENCE_GENERATOR_CLASS,
)
from run_lstm_mpc_weight_sweep import recommend_weight_set  # noqa: E402
from mpc.controllers.reference_generator import CasadiReferenceGenerator  # noqa: E402
from mpc.solvers.casadi_solver import CasadiMPCConfig, _build_dual_objective_info, _build_single_objective_info  # noqa: E402
from mpc.solvers.fc_dp0_curve import h2_kg_step_dp0_quadratic  # noqa: E402


class TestLstmMpcZeroDelay(unittest.TestCase):
    def test_make_run_id_and_file_hashes_are_traceable(self) -> None:
        run_id = make_run_id(
            "dp0_total_load_qsoc500_qbatt0035_qramp3e-5",
            {"q_h2": 1.0, "q_soc": 500.0, "q_batt": 0.035, "q_ramp": 3e-5},
            timestamp="20260627_120000",
        )

        self.assertIn("20260627_120000", run_id)
        self.assertIn("qsoc500", run_id)
        self.assertIn("qbatt0p035", run_id)
        self.assertIn("qramp3em05", run_id)

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            png = output_dir / "fixed_mpc_p6_comparison.png"
            csv = output_dir / "fixed_mpc_p6_timeseries.csv"
            config = output_dir / "fixed_mpc_p6_config.json"
            png.write_bytes(b"png")
            csv.write_text("a\n1\n", encoding="utf-8")
            config.write_text("{}", encoding="utf-8")

            hashes = build_run_file_hashes(output_dir=output_dir, run_id=run_id)

        self.assertEqual(hashes["run_id"], run_id)
        self.assertEqual(hashes["fixed_mpc_p6_comparison.png"]["md5"], hashlib.md5(b"png").hexdigest())
        self.assertEqual(hashes["fixed_mpc_p6_timeseries.csv"]["exists"], True)

    def test_build_mpc_load_ref_uses_lstm_h1_to_h6_predictions(self) -> None:
        pred = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

        ref = build_mpc_load_ref(current_load_t=10.0, lstm_pred=pred, pred_horizon=6, mpc_horizon=6)

        self.assertEqual(ref.tolist(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0])


    def test_make_debug_timing_row_records_next_step_application(self) -> None:
        pred = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

        row = make_debug_timing_row(
            voyage_id=1,
            decision_index_t=18,
            history_len=18,
            pred_horizon=6,
            mpc_load_ref=build_mpc_load_ref(10.0, pred, pred_horizon=6, mpc_horizon=6),
            actual_load_t=10.0,
            pred_h1=11.0,
            actual_load_t_plus_1=10.5,
            history_available=True,
            lstm_available=True,
            forecast_source="lstm_h1_to_h6",
        )

        self.assertEqual(row["history_end_index"], row["decision_index_t"])
        self.assertEqual(row["lstm_forecast_start_index"], row["decision_index_t"] + 1)
        self.assertEqual(row["lstm_forecast_end_index"], row["decision_index_t"] + 6)
        self.assertEqual(row["mpc_ref_len"], 6)
        self.assertEqual(row["apply_index"], row["decision_index_t"] + 1)
        self.assertEqual(row["mpc_stage0_source"], "lstm_h1_to_h6")
        self.assertEqual(row["mpc_stage1_source"], "lstm_h1_to_h6")
        self.assertEqual(row["first_mpc_ref_load"], row["pred_h1"])


    def test_raw_terminal_soc_penalty_uses_dedicated_weight(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=1806.0,
            fuel_cell_max_kw=560.0,
            soc_min=0.2,
            soc_max=0.8,
            soc_target=0.65,
            use_raw_objective=True,
            raw_soc_squared=True,
            enable_terminal_soc_soft_penalty=True,
            q_soc=80.0,
            q_terminal_soc=400.0,
        )

        info = _build_dual_objective_info(
            p_fc_left_traj=np.array([20.0, 20.0]),
            p_fc_right_traj=np.array([20.0, 20.0]),
            p_batt_left_traj=np.array([0.0, 0.0]),
            p_batt_right_traj=np.array([0.0, 0.0]),
            soc_left_traj=np.array([0.55, 0.55, 0.55]),
            soc_right_traj=np.array([0.55, 0.55, 0.55]),
            prev_fc_left_kw=20.0,
            prev_fc_right_kw=20.0,
            config=cfg,
            terminal_load_left_kw=20.0,
            terminal_load_right_kw=20.0,
        )

        # ((0.55 - 0.65)^2 + (0.55 - 0.65)^2) / 2 * q_terminal_soc
        self.assertAlmostEqual(info["terminal_soc_term"], 4.0)


    def test_build_action_table_has_35_weight_actions_centered_on_balanced_v1(self) -> None:
        actions = build_action_table(
            base={"q_soc": 80.0, "q_fc": 0.001, "q_batt": 0.01, "q_ramp": 0.08, "q_terminal_soc": 400.0}
        )

        self.assertEqual(len(actions), 35)
        self.assertEqual(
            actions[0],
            {
                "action_id": 0,
                "q_soc": 65.0,
                "q_fc": 0.001,
                "q_batt": 0.01,
                "q_ramp": 0.04,
                "q_terminal_soc": 400.0,
            },
        )
        self.assertEqual(actions[-1]["q_soc"], 95.0)
        self.assertAlmostEqual(actions[-1]["q_ramp"], 0.12)

    def test_recommend_weight_set_filters_infeasible_sets_then_sorts_by_charge_sustaining_objectives(self) -> None:
        rows = [
            {
                "weight_set": "bad_soc",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.19,
                "mean_soc_max": 0.7,
                "mean_solver_success_rate": 1.0,
                "mean_fc_shutdown_time_after_load_zero_min": 0.0,
                "total_fc_idle_h2_consumption_kg": 0.0,
                "mean_h2_consumption_kg": 1.0,
                "mean_soc_terminal_error": 0.01,
                "mean_fc_ramp_mean_kw": 0.1,
                "mean_battery_throughput_kwh": 1.0,
                "mean_charge_sustaining_adjusted_h2_kg": 1.0,
                "mean_abs_soc_delta": 0.01,
            },
            {
                "weight_set": "h2_best",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_fc_shutdown_time_after_load_zero_min": 1.0,
                "total_fc_idle_h2_consumption_kg": 0.01,
                "mean_h2_consumption_kg": 1.0,
                "mean_soc_terminal_error": 0.02,
                "mean_fc_ramp_mean_kw": 0.1,
                "mean_battery_throughput_kwh": 5.0,
                "mean_charge_sustaining_adjusted_h2_kg": 1.1,
                "mean_abs_soc_delta": 0.01,
            },
            {
                "weight_set": "adjusted_h2_best",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_fc_shutdown_time_after_load_zero_min": 0.5,
                "total_fc_idle_h2_consumption_kg": 0.005,
                "mean_h2_consumption_kg": 1.1,
                "mean_soc_terminal_error": 0.01,
                "mean_fc_ramp_mean_kw": 0.5,
                "mean_battery_throughput_kwh": 1.0,
                "mean_charge_sustaining_adjusted_h2_kg": 0.9,
                "mean_abs_soc_delta": 0.01,
            },
        ]

        self.assertEqual(recommend_weight_set(rows), "adjusted_h2_best")

    def test_recommend_weight_set_rejects_soc_spending_candidate(self) -> None:
        from run_lstm_mpc_weight_sweep import recommend_weight_set

        rows = [
            {
                "weight_set": "soc_spending_low_h2",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_soc_terminal_error": 0.14,
                "mean_h2_consumption_kg": 0.2,
                "mean_charge_sustaining_adjusted_h2_kg": 12.0,
                "mean_battery_throughput_kwh": 250.0,
                "mean_fc_ramp_mean_kw": 0.02,
                "mean_abs_soc_delta": 0.14,
            },
            {
                "weight_set": "charge_sustaining_candidate",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_soc_terminal_error": 0.05,
                "mean_h2_consumption_kg": 8.0,
                "mean_charge_sustaining_adjusted_h2_kg": 12.5,
                "mean_battery_throughput_kwh": 100.0,
                "mean_fc_ramp_mean_kw": 1.0,
                "mean_abs_soc_delta": 0.05,
            },
        ]

        self.assertEqual(recommend_weight_set(rows), "charge_sustaining_candidate")

    def test_recommend_weight_set_uses_objective_ranking_without_rule_based_exclusions(self) -> None:
        rows = [
            {
                "weight_set": "dp0_h2_only_diag",
                "soc_reference_mode": "initial_soc",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_soc_terminal_error": 0.01,
                "mean_h2_consumption_kg": 0.5,
                "mean_battery_throughput_kwh": 1.0,
                "mean_fc_ramp_mean_kw": 0.1,
                "mean_charge_sustaining_adjusted_h2_kg": 0.5,
                "mean_abs_soc_delta": 0.01,
            },
            {
                "weight_set": "objective_best_even_if_behavior_flagged",
                "soc_reference_mode": "initial_soc",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_soc_terminal_error": 0.01,
                "mean_h2_consumption_kg": 0.6,
                "mean_battery_throughput_kwh": 1.5,
                "mean_fc_ramp_mean_kw": 0.2,
                "mean_charge_sustaining_adjusted_h2_kg": 0.6,
                "mean_abs_soc_delta": 0.01,
                "max_fc_kw": 320.0,
                "max_initial_battery_only_time_min": 20.0,
            },
            {
                "weight_set": "clean_but_worse_objective",
                "soc_reference_mode": "initial_soc",
                "total_unserved_energy_kwh": 0.0,
                "mean_soc_min": 0.21,
                "mean_soc_max": 0.79,
                "mean_solver_success_rate": 1.0,
                "mean_soc_terminal_error": 0.01,
                "mean_h2_consumption_kg": 1.0,
                "mean_battery_throughput_kwh": 1.0,
                "mean_fc_ramp_mean_kw": 0.1,
                "mean_charge_sustaining_adjusted_h2_kg": 1.0,
                "mean_abs_soc_delta": 0.01,
                "max_fc_kw": 220.0,
            },
        ]

        self.assertEqual(recommend_weight_set(rows), "objective_best_even_if_behavior_flagged")

    def test_dp0_baseline_uses_raw_physical_objective_and_requested_weights(self) -> None:
        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"])

        self.assertEqual(cfg.objective_mode, "raw_physical")
        self.assertFalse(cfg.use_raw_objective)
        self.assertFalse(cfg.use_dimensionless_objective)
        self.assertTrue(cfg.use_h2_mass_cost)
        self.assertFalse(cfg.normalize_h2_cost)
        self.assertFalse(cfg.enable_terminal_soc_soft_penalty)
        self.assertAlmostEqual(cfg.q_h2, 1.0)
        self.assertAlmostEqual(cfg.q_soc, 50.0)
        self.assertAlmostEqual(cfg.q_batt, 0.025)
        self.assertAlmostEqual(cfg.q_ramp, 1e-4)
        self.assertAlmostEqual(cfg.q_terminal_soc, 0.0)
        self.assertAlmostEqual(cfg.battery_capacity_kwh, 277.2)
        self.assertFalse(cfg.fuel_cell_ramp_constraint_enabled)
        self.assertAlmostEqual(cfg.soc_band, 0.0)
        self.assertAlmostEqual(cfg.terminal_soc_band, 0.0)

    def test_extra_json_weight_sets_are_preserved_and_rule_keys_filtered(self) -> None:
        raw = {
            "custom_external_weight_set": {
                **DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET],
                "low_load_fc_suppression": {"enabled": True},
                "soc_reserve": 0.55,
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mpc_weight_sets.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            loaded = load_weight_sets(path)

        self.assertIn("custom_external_weight_set", loaded)
        loaded_weights = loaded["custom_external_weight_set"]
        self.assertNotIn("low_load_fc_suppression", loaded_weights)
        self.assertNotIn("soc_reserve", loaded_weights)

        cfg = mpc_config_from_weights(loaded_weights)
        self.assertFalse(cfg.fuel_cell_ramp_constraint_enabled)

    def test_active_weight_sets_are_clean_literature_baseline_candidates(self) -> None:
        expected = {
            "dp0_raw_h2_soc_batt_ramp_nextstep_v1",
            "dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1",
            "dp0_total_load_qsoc400_qbatt003_qramp2e-5",
            "dp0_total_load_qsoc400_qbatt0035_qramp3e-5",
            "dp0_total_load_qsoc500_qbatt003_qramp3e-5",
            "dp0_total_load_qsoc500_qbatt0035_qramp3e-5",
            "dp0_total_load_no_reserve_no_terminal_qsoc600",
        }
        forbidden = {
            "low_load_fc_suppression",
            "soc_recovery_power_limit",
            "sustained_load_battery_discharge_limit",
            "fc_overproduction_limit",
            "low_load_fc_suppression_enabled",
            "soc_recovery_power_limit_enabled",
            "sustained_load_battery_discharge_limit_enabled",
            "fc_overproduction_limit_enabled",
        }

        self.assertEqual(set(DEFAULT_WEIGHT_SETS.keys()), expected)
        for key in expected:
            self.assertEqual(DEFAULT_WEIGHT_SETS[key]["objective_mode"], "raw_physical")
            self.assertTrue(DEFAULT_WEIGHT_SETS[key]["use_h2_mass_cost"])
            self.assertFalse(DEFAULT_WEIGHT_SETS[key]["normalize_h2_cost"])
            self.assertFalse(DEFAULT_WEIGHT_SETS[key]["use_dimensionless_objective"])
            self.assertFalse(DEFAULT_WEIGHT_SETS[key]["use_raw_objective"])
            self.assertAlmostEqual(DEFAULT_WEIGHT_SETS[key].get("q_terminal_soc", 0.0), 0.0)
            self.assertFalse(DEFAULT_WEIGHT_SETS[key].get("enable_terminal_soc_soft_penalty", False))
            self.assertFalse(DEFAULT_WEIGHT_SETS[key]["fuel_cell_ramp_constraint_enabled"])
            self.assertFalse(forbidden.intersection(DEFAULT_WEIGHT_SETS[key].keys()))
            self.assertNotIn("soc_reserve", DEFAULT_WEIGHT_SETS[key])
            self.assertNotIn("terminal_soc_band", DEFAULT_WEIGHT_SETS[key])
            self.assertNotIn("enable_terminal_soc_soft_penalty", DEFAULT_WEIGHT_SETS[key])
        self.assertAlmostEqual(DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"]["battery_capacity_kwh"], 277.2)
        self.assertAlmostEqual(DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"]["q_ramp"], 1e-4)
        self.assertAlmostEqual(DEFAULT_WEIGHT_SETS["dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1"]["battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(DEFAULT_WEIGHT_SETS["dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1"]["q_ramp"], 1e-4)
        candidate = DEFAULT_WEIGHT_SETS["dp0_total_load_qsoc400_qbatt0035_qramp3e-5"]
        self.assertAlmostEqual(candidate["q_h2"], 1.0)
        self.assertAlmostEqual(candidate["q_soc"], 400.0)
        self.assertAlmostEqual(candidate["q_fc"], 0.0)
        self.assertAlmostEqual(candidate["q_batt"], 0.035)
        self.assertAlmostEqual(candidate["q_ramp"], 3e-5)
        self.assertAlmostEqual(candidate["q_terminal_soc"], 0.0)
        self.assertAlmostEqual(candidate["battery_capacity_kwh"], 1806.0)
        self.assertFalse(candidate["fuel_cell_ramp_constraint_enabled"])
        q400_lower_fc_share = DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET]
        self.assertAlmostEqual(q400_lower_fc_share["q_h2"], 1.0)
        self.assertAlmostEqual(q400_lower_fc_share["q_soc"], 400.0)
        self.assertAlmostEqual(q400_lower_fc_share["q_fc"], 0.0)
        self.assertAlmostEqual(q400_lower_fc_share["q_batt"], 0.030)
        self.assertAlmostEqual(q400_lower_fc_share["q_ramp"], 2e-5)
        self.assertAlmostEqual(q400_lower_fc_share["q_terminal_soc"], 0.0)
        self.assertAlmostEqual(q400_lower_fc_share["battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(q400_lower_fc_share["soc_min"], 0.2)
        self.assertAlmostEqual(q400_lower_fc_share["soc_max"], 0.8)
        self.assertTrue(q400_lower_fc_share["raw_soc_squared"])
        self.assertFalse(q400_lower_fc_share["fuel_cell_ramp_constraint_enabled"])
        no_reserve = DEFAULT_WEIGHT_SETS[TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET]
        self.assertAlmostEqual(no_reserve["q_h2"], 1.0)
        self.assertAlmostEqual(no_reserve["q_soc"], 600.0)
        self.assertAlmostEqual(no_reserve["q_fc"], 0.0)
        self.assertAlmostEqual(no_reserve["q_batt"], 0.035)
        self.assertAlmostEqual(no_reserve["q_ramp"], 3e-5)
        self.assertAlmostEqual(no_reserve["q_terminal_soc"], 0.0)
        self.assertAlmostEqual(no_reserve["battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(no_reserve["soc_min"], 0.2)
        self.assertAlmostEqual(no_reserve["soc_max"], 0.8)
        self.assertTrue(no_reserve["raw_soc_squared"])
        self.assertFalse(no_reserve["fuel_cell_ramp_constraint_enabled"])
        self.assertNotIn("soc_reserve", no_reserve)
        self.assertNotIn("terminal_soc_band", no_reserve)
        self.assertNotIn("enable_terminal_soc_soft_penalty", no_reserve)
        q500 = DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET]
        self.assertAlmostEqual(q500["q_h2"], 1.0)
        self.assertAlmostEqual(q500["q_soc"], 500.0)
        self.assertAlmostEqual(q500["q_fc"], 0.0)
        self.assertAlmostEqual(q500["q_batt"], 0.030)
        self.assertAlmostEqual(q500["q_ramp"], 3e-5)
        self.assertAlmostEqual(q500["q_terminal_soc"], 0.0)
        self.assertAlmostEqual(q500["battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(q500["soc_min"], 0.2)
        self.assertAlmostEqual(q500["soc_max"], 0.8)
        self.assertTrue(q500["raw_soc_squared"])
        self.assertFalse(q500["fuel_cell_ramp_constraint_enabled"])
        self.assertNotIn("soc_reserve", q500)
        self.assertNotIn("terminal_soc_band", q500)
        self.assertNotIn("enable_terminal_soc_soft_penalty", q500)
        q500_qbatt35 = DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET]
        self.assertAlmostEqual(q500_qbatt35["q_h2"], 1.0)
        self.assertAlmostEqual(q500_qbatt35["q_soc"], 500.0)
        self.assertAlmostEqual(q500_qbatt35["q_fc"], 0.0)
        self.assertAlmostEqual(q500_qbatt35["q_batt"], 0.035)
        self.assertAlmostEqual(q500_qbatt35["q_ramp"], 3e-5)
        self.assertAlmostEqual(q500_qbatt35["q_terminal_soc"], 0.0)
        self.assertAlmostEqual(q500_qbatt35["battery_capacity_kwh"], 1806.0)
        self.assertAlmostEqual(q500_qbatt35["soc_min"], 0.2)
        self.assertAlmostEqual(q500_qbatt35["soc_max"], 0.8)
        self.assertTrue(q500_qbatt35["raw_soc_squared"])
        self.assertFalse(q500_qbatt35["fuel_cell_ramp_constraint_enabled"])
        self.assertNotIn("soc_reserve", q500_qbatt35)
        self.assertNotIn("terminal_soc_band", q500_qbatt35)
        self.assertNotIn("enable_terminal_soc_soft_penalty", q500_qbatt35)

    def test_no_reserve_weight_set_filters_reserve_and_terminal_reporting_columns(self) -> None:
        df = pd.DataFrame(
            {
                "SOC_start": [0.55],
                "SOC_end": [0.54],
                "soc_reserve": [0.55],
                "soc_below_reserve_duration_s": [30.0],
                "soc_below_reserve_min_gap": [-0.01],
                "q_terminal_soc": [0.0],
                "terminal_soc_band": [0.0],
                "weighted_terminal_soc_cost": [0.0],
                "fc_load_tracking_mae": [12.0],
            }
        )

        for weight_set in (
            TOTAL_LOAD_QSOC500_QBATT003_QRAMP3E5_WEIGHT_SET,
            TOTAL_LOAD_QSOC500_QBATT0035_QRAMP3E5_WEIGHT_SET,
        ):
            cleaned = clean_reporting_frame_for_weight_set(df, weight_set=weight_set)

            self.assertIn("SOC_start", cleaned.columns)
            self.assertIn("fc_load_tracking_mae", cleaned.columns)
            self.assertNotIn("soc_reserve", cleaned.columns)
            self.assertNotIn("soc_below_reserve_duration_s", cleaned.columns)
            self.assertNotIn("soc_below_reserve_min_gap", cleaned.columns)
            self.assertNotIn("q_terminal_soc", cleaned.columns)
            self.assertNotIn("terminal_soc_band", cleaned.columns)
            self.assertNotIn("weighted_terminal_soc_cost", cleaned.columns)

    def test_clean_reporting_dict_for_no_reserve_output_is_json_serializable(self) -> None:
        cleaned = clean_reporting_dict_for_weight_set(
            {
                "voyage_id": np.int64(1),
                "SOC_start": np.float64(0.55),
                "soc_reserve": np.float64(0.55),
                "terminal_soc_band": np.float64(0.0),
            },
            weight_set=TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET,
        )

        json.dumps(cleaned)
        self.assertEqual(cleaned["voyage_id"], 1)
        self.assertNotIn("soc_reserve", cleaned)

    def test_no_reserve_config_payload_hides_disabled_reserve_and_terminal_fields(self) -> None:
        payload = {
            "weights": {
                "q_terminal_soc": 0.0,
                "soc_reserve": 0.55,
                "terminal_soc_band": 0.0,
            },
            "effective_mpc_config": {
                "q_terminal_soc": 0.0,
                "soc_reserve": 0.55,
                "terminal_soc_band": 0.0,
                "enable_terminal_soc_soft_penalty": False,
            },
        }

        cleaned = clean_config_payload_for_weight_set(
            payload,
            weight_set=TOTAL_LOAD_NO_RESERVE_NO_TERMINAL_QSOC600_WEIGHT_SET,
        )

        self.assertEqual(cleaned["weights"]["q_terminal_soc"], 0.0)
        self.assertEqual(cleaned["effective_mpc_config"]["q_terminal_soc"], 0.0)
        self.assertNotIn("soc_reserve", cleaned["weights"])
        self.assertNotIn("terminal_soc_band", cleaned["weights"])
        self.assertNotIn("soc_reserve", cleaned["effective_mpc_config"])
        self.assertNotIn("terminal_soc_band", cleaned["effective_mpc_config"])
        self.assertNotIn("enable_terminal_soc_soft_penalty", cleaned["effective_mpc_config"])

    def test_closed_loop_metrics_report_reserve_and_fc_load_tracking(self) -> None:
        cfg = CasadiMPCConfig(
            prediction_horizon=2,
            dt_hours=30.0 / 3600.0,
            battery_capacity_kwh=1806.0,
            fuel_cell_max_kw=560.0,
            soc_reference_mode="initial_soc",
            soc_reserve=0.55,
            soc_band=0.0,
            objective_mode="raw_physical",
            use_h2_mass_cost=True,
            normalize_h2_cost=False,
            fuel_cell_ramp_constraint_enabled=False,
        )
        ts = pd.DataFrame(
            {
                "P_fc_kw": [100.0, 80.0, 120.0, 90.0],
                "P_batt_kw": [0.0, 40.0, -20.0, 0.0],
                "SOC": [0.56, 0.54, 0.53, 0.55],
                "SOC_before": [0.55, 0.56, 0.54, 0.53],
                "load_total_kw": [100.0, 120.0, 100.0, 90.0],
                "unserved_power_kw": [0.0, 0.0, 0.0, 0.0],
                "solver_success": [1, 1, 1, 1],
                "objective_value": [1.0, 1.0, 1.0, 1.0],
                "solve_ms": [10.0, 10.0, 10.0, 10.0],
                "file_name": ["synthetic.xlsx"] * 4,
            }
        )

        metrics = compute_closed_loop_metrics(
            ts,
            cfg,
            voyage_id=1,
            voyage_name="synthetic",
            weight_set="unit_test",
        )

        self.assertAlmostEqual(metrics["soc_below_reserve_duration_s"], 60.0)
        self.assertAlmostEqual(metrics["soc_below_reserve_min_gap"], -0.02)
        self.assertAlmostEqual(metrics["fc_above_load_energy_kwh"], 20.0 * cfg.dt_hours)
        self.assertAlmostEqual(metrics["fc_below_load_energy_kwh"], 40.0 * cfg.dt_hours)
        self.assertAlmostEqual(metrics["fc_load_tracking_mae"], 15.0)
        self.assertAlmostEqual(metrics["fc_load_tracking_bias"], -5.0)

    def test_default_fixed_weight_set_is_raw_physical_baseline(self) -> None:
        self.assertEqual(CURRENT_FIXED_WEIGHT_SET, "dp0_raw_h2_soc_batt_ramp_nextstep_v1")

    def test_rule_based_config_blocks_are_ignored_by_mpc_config(self) -> None:
        cfg = mpc_config_from_weights(
            {
                **DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"],
                "low_load_fc_suppression": {
                    "enabled": True,
                    "low_load_threshold_kw": 4.0,
                    "soc_allow_shutdown_threshold": 0.57,
                    "fc_idle_upper_kw": 1.5,
                },
                "soc_recovery_power_limit": {
                    "enabled": True,
                    "max_charge_power_kw": 80.0,
                },
                "sustained_load_battery_discharge_limit": {
                    "enabled": True,
                    "load_threshold_kw": 20.0,
                    "soc_margin_above_ref": 0.01,
                    "max_discharge_kw": 40.0,
                },
                "fc_overproduction_limit": {
                    "enabled": True,
                    "max_over_load_kw": 80.0,
                },
                "battery_throughput_penalty": {
                    "enabled": True,
                    "type": "absolute_power",
                    "normalization_kw": 300.0,
                },
            }
        )

        self.assertFalse(hasattr(cfg, "low_load_fc_suppression_enabled"))
        self.assertFalse(hasattr(cfg, "soc_recovery_power_limit_enabled"))
        self.assertFalse(hasattr(cfg, "sustained_load_battery_discharge_limit_enabled"))
        self.assertFalse(hasattr(cfg, "fc_overproduction_limit_enabled"))
        self.assertTrue(cfg.battery_throughput_penalty_enabled)
        self.assertEqual(cfg.battery_throughput_penalty_type, "absolute_power")
        self.assertAlmostEqual(cfg.battery_throughput_normalization_kw, 300.0)

    def test_lstm_mpc_entrypoint_uses_total_power_controller(self) -> None:
        self.assertEqual(CONTROL_LAYER_SCOPE, "total_power")
        self.assertIs(REFERENCE_GENERATOR_CLASS, CasadiReferenceGenerator)

    def test_dual_objective_info_reports_raw_physical_dp0_h2_breakdown(self) -> None:
        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"])

        info = _build_dual_objective_info(
            p_fc_left_traj=np.array([10.0, 20.0]),
            p_fc_right_traj=np.array([10.0, 20.0]),
            p_batt_left_traj=np.array([0.0, 0.0]),
            p_batt_right_traj=np.array([0.0, 0.0]),
            soc_left_traj=np.array([0.55, 0.55, 0.55]),
            soc_right_traj=np.array([0.55, 0.55, 0.55]),
            prev_fc_left_kw=10.0,
            prev_fc_right_kw=10.0,
            config=cfg,
            terminal_load_left_kw=20.0,
            terminal_load_right_kw=20.0,
        )

        self.assertEqual(info["raw_fc_cost_mode"], "h2_mass_kg")
        self.assertEqual(info["objective_scale_mode"], "raw_physical")
        self.assertTrue(info["use_h2_mass_cost"])
        self.assertGreater(info["raw_h2_cost"], 0.0)
        self.assertAlmostEqual(info["weighted_h2_cost"], cfg.q_h2 * info["raw_h2_cost"])
        self.assertIn("H2_step_kg_sum", info)

    def test_raw_physical_objective_info_exports_only_physical_cost_terms(self) -> None:
        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS["dp0_raw_h2_soc_batt_ramp_nextstep_v1"])
        p_fc = np.array([10.0, 20.0])
        p_batt = np.array([2.0, -3.0])
        soc = np.array([0.55, 0.56, 0.54])

        info = _build_single_objective_info(
            p_fc_traj=p_fc,
            p_bat_traj=p_batt,
            soc_traj=soc,
            prev_fc_kw=10.0,
            config=cfg,
            terminal_load_kw=20.0,
            soc_reference_value=0.55,
        )

        required = [
            "h2_mass_kg",
            "soc_cost_raw",
            "batt_throughput_kwh",
            "weighted_h2_cost",
            "weighted_soc_cost",
            "weighted_batt_cost",
            "ramp_cost_raw",
            "weighted_ramp_cost",
            "total_objective",
        ]
        for key in required:
            self.assertIn(key, info)
        effective_fields = {
            "effective_q_h2": cfg.q_h2,
            "effective_q_soc": cfg.q_soc,
            "effective_q_batt": cfg.q_batt,
            "effective_q_ramp": cfg.q_ramp,
            "effective_q_terminal_soc": cfg.q_terminal_soc,
            "effective_battery_capacity_kwh": cfg.battery_capacity_kwh,
            "effective_fuel_cell_ramp_constraint_enabled": cfg.fuel_cell_ramp_constraint_enabled,
            "effective_use_dimensionless_objective": cfg.use_dimensionless_objective,
            "effective_normalize_h2_cost": cfg.normalize_h2_cost,
            "effective_soc_band": cfg.soc_band,
            "effective_terminal_soc_band": cfg.terminal_soc_band,
        }
        for key, value in effective_fields.items():
            self.assertIn(key, info)
            self.assertEqual(info[key], value)
        expected_h2 = float(np.sum(h2_kg_step_dp0_quadratic(p_fc, dt_seconds=30.0, p_rated_total_kw=560.0)))
        expected_soc = float(np.sum((soc[1:] - 0.55) ** 2))
        expected_batt = float(np.sum(np.abs(p_batt)) * cfg.dt_hours)
        expected_ramp = float(np.sum(np.array([0.0, 10.0]) ** 2))

        self.assertAlmostEqual(info["h2_mass_kg"], expected_h2)
        self.assertAlmostEqual(info["soc_cost_raw"], expected_soc)
        self.assertAlmostEqual(info["batt_throughput_kwh"], expected_batt)
        self.assertAlmostEqual(info["ramp_cost_raw"], expected_ramp)
        self.assertAlmostEqual(info["weighted_h2_cost"], cfg.q_h2 * expected_h2)
        self.assertAlmostEqual(info["weighted_soc_cost"], cfg.q_soc * expected_soc)
        self.assertAlmostEqual(info["weighted_batt_cost"], cfg.q_batt * expected_batt)
        self.assertAlmostEqual(info["weighted_ramp_cost"], cfg.q_ramp * expected_ramp)
        self.assertAlmostEqual(info["weighted_terminal_soc_cost"], 0.0)
        self.assertAlmostEqual(
            info["total_objective"],
            (
                info["weighted_h2_cost"]
                + info["weighted_soc_cost"]
                + info["weighted_batt_cost"]
                + info["weighted_ramp_cost"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
