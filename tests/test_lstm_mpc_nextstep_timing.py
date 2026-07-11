from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_lstm_mpc_test import (  # noqa: E402
    CONTROL_APPLY_TIMING,
    CONTROL_TIMING,
    CURRENT_FIXED_WEIGHT_SET,
    DEFAULT_WEIGHT_SETS,
    TOTAL_LOAD_WEIGHT_SET,
    build_mpc_load_ref,
    execute_cached_fc_step,
    forecast_source_name,
    mpc_config_from_weights,
)
from run_lstm_mpc_total_load_test import build_parser as build_total_load_mpc_parser  # noqa: E402


class TestLstmMpcNextstepTiming(unittest.TestCase):
    def test_mpc_reference_uses_lstm_h1_to_h6_when_available(self) -> None:
        pred = np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0])

        ref = build_mpc_load_ref(99.0, pred, pred_horizon=6, mpc_horizon=6, lstm_available=True)

        np.testing.assert_array_equal(ref, pred)

    def test_mpc_reference_holds_current_load_when_lstm_unavailable(self) -> None:
        pred = np.full(6, np.nan)

        ref = build_mpc_load_ref(42.0, pred, pred_horizon=6, mpc_horizon=6, lstm_available=False)

        np.testing.assert_array_equal(ref, np.full(6, 42.0))
        self.assertEqual(
            forecast_source_name(history_available=False, lstm_available=False),
            "current_load_hold",
        )

    def test_default_nextstep_weight_set_and_config(self) -> None:
        self.assertEqual(CONTROL_TIMING, "one_step_ahead_lstm_mpc")
        self.assertEqual(CONTROL_APPLY_TIMING, "execute_cached_previous_mpc_command")
        self.assertEqual(CURRENT_FIXED_WEIGHT_SET, "dp0_raw_h2_soc_batt_ramp_nextstep_v1")

        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS[CURRENT_FIXED_WEIGHT_SET])

        self.assertEqual(cfg.objective_mode, "raw_physical")
        self.assertFalse(cfg.use_raw_objective)
        self.assertFalse(cfg.use_dimensionless_objective)
        self.assertTrue(cfg.use_h2_mass_cost)
        self.assertFalse(cfg.normalize_h2_cost)
        self.assertAlmostEqual(cfg.battery_capacity_kwh, 277.2)
        self.assertFalse(cfg.fuel_cell_ramp_constraint_enabled)
        self.assertAlmostEqual(cfg.fuel_cell_ramp_kw, 48.0)
        self.assertAlmostEqual(cfg.q_h2, 1.0)
        self.assertAlmostEqual(cfg.q_soc, 50.0)
        self.assertAlmostEqual(cfg.q_batt, 0.025)
        self.assertAlmostEqual(cfg.q_ramp, 1e-4)

    def test_total_load_weight_set_uses_1806kwh_and_disabled_hard_ramp(self) -> None:
        self.assertEqual(TOTAL_LOAD_WEIGHT_SET, "dp0_total_load_raw_h2_soc_batt_ramp_nextstep_v1")

        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS[TOTAL_LOAD_WEIGHT_SET])

        self.assertEqual(cfg.objective_mode, "raw_physical")
        self.assertFalse(cfg.use_dimensionless_objective)
        self.assertTrue(cfg.use_h2_mass_cost)
        self.assertFalse(cfg.normalize_h2_cost)
        self.assertAlmostEqual(cfg.battery_capacity_kwh, 1806.0)
        self.assertFalse(cfg.fuel_cell_ramp_constraint_enabled)
        self.assertAlmostEqual(cfg.q_h2, 1.0)
        self.assertAlmostEqual(cfg.q_soc, 300.0)
        self.assertAlmostEqual(cfg.q_batt, 0.020)
        self.assertAlmostEqual(cfg.q_ramp, 0.00010)

    def test_total_load_mpc_entry_defaults_do_not_overwrite_old_outputs(self) -> None:
        args = build_total_load_mpc_parser().parse_args([])

        self.assertEqual(args.weight_set, TOTAL_LOAD_WEIGHT_SET)
        self.assertIn("lstm_total_load_721", str(args.lstm_ckpt))
        self.assertIn("voyage_split_total_load_721.json", str(args.split_json))
        self.assertIn("total_load_66_segments.csv", str(args.source_csv))
        self.assertIn("outputs\\lstm_mpc_total_load_test", str(args.output_dir))

    def test_linear_interp_1s_mpc_entrypoint_is_deprecated(self) -> None:
        self.assertFalse((MAIN / "run_lstm_mpc_total_load_1s_test.py").exists())
        self.assertTrue(
            (MAIN / "_deprecated_run_lstm_mpc_total_load_1s_linear_interp_DO_NOT_USE.py").exists()
        )

    def test_actual_battery_power_is_current_load_minus_executed_fc(self) -> None:
        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS[CURRENT_FIXED_WEIGHT_SET])

        step = execute_cached_fc_step(actual_load_kw=72.0, fc_command_kw=50.0, soc_before=0.55, cfg=cfg)

        self.assertAlmostEqual(step["P_fc_executed_kw"], 50.0)
        self.assertAlmostEqual(step["P_batt_actual_kw"], 22.0)
        self.assertAlmostEqual(
            step["SOC_after"],
            0.55 - 22.0 * cfg.dt_hours / cfg.battery_capacity_kwh,
        )
        self.assertFalse(step["battery_discharge_limit_active"])
        self.assertFalse(step["battery_charge_limit_active"])


if __name__ == "__main__":
    unittest.main()
