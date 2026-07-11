from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc.solvers.casadi_solver import casadi_available  # noqa: E402
from run_lstm_mpc_test import (  # noqa: E402
    DEFAULT_WEIGHT_SETS,
    MPC_HORIZON,
    TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET,
    build_horizon_sensitivity_debug,
    build_solver_horizon_debug,
    build_timing_debug_first_steps,
    mpc_config_from_weights,
)


class TestLstmMpcNextstepHorizon(unittest.TestCase):
    def test_timing_debug_maps_lstm_h1_to_h6_to_mpc_stages(self) -> None:
        ts = pd.DataFrame(
            [
                {
                    "voyage_id": 1,
                    "decision_index_t": 0,
                    "time_h": 0.0,
                    "timestamp": "2024-01-01 00:00:00",
                    "history_available": False,
                    "lstm_available": False,
                    "forecast_source": "current_load_hold",
                    "load_total_kw": 42.0,
                    "P_fc_executed_kw": 40.0,
                    "P_batt_actual_kw": 2.0,
                    "P_fc_next_cmd_kw": 41.0,
                    "SOC_before": 0.55,
                    "SOC": 0.5499,
                    **{f"pred_h{h}": np.nan for h in range(1, MPC_HORIZON + 1)},
                    **{f"mpc_ref_load_stage{k}": 42.0 for k in range(MPC_HORIZON)},
                    **{f"P_fc_plan_stage{k}": 41.0 for k in range(MPC_HORIZON)},
                },
                {
                    "voyage_id": 1,
                    "decision_index_t": 18,
                    "time_h": 0.15,
                    "timestamp": "2024-01-01 00:09:00",
                    "history_available": True,
                    "lstm_available": True,
                    "forecast_source": "lstm_h1_to_h6",
                    "load_total_kw": 99.0,
                    "P_fc_executed_kw": 50.0,
                    "P_batt_actual_kw": 49.0,
                    "P_fc_next_cmd_kw": 60.0,
                    "SOC_before": 0.55,
                    "SOC": 0.5498,
                    **{f"pred_h{h}": float(10 + h) for h in range(1, MPC_HORIZON + 1)},
                    **{f"mpc_ref_load_stage{k}": float(11 + k) for k in range(MPC_HORIZON)},
                    **{f"P_fc_plan_stage{k}": float(60 + k) for k in range(MPC_HORIZON)},
                },
            ]
        )

        debug = build_timing_debug_first_steps(ts)

        lstm_row = debug.loc[debug["forecast_source"] == "lstm_h1_to_h6"].iloc[0]
        hold_row = debug.loc[debug["forecast_source"] == "current_load_hold"].iloc[0]
        self.assertTrue(bool(lstm_row["all_mpc_ref_match_lstm_h1_h6"]))
        self.assertTrue(bool(hold_row["all_mpc_ref_equal_current_load"]))
        for k in range(MPC_HORIZON):
            self.assertEqual(lstm_row[f"mpc_ref_load_stage{k}"], lstm_row[f"lstm_pred_h{k + 1}"])
            self.assertEqual(hold_row[f"mpc_ref_load_stage{k}"], hold_row["actual_load_t"])

    def test_solver_horizon_debug_reports_six_stage_mpc(self) -> None:
        cfg = mpc_config_from_weights(DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET])

        debug = build_solver_horizon_debug(cfg)

        self.assertEqual(debug["N"], 6)
        self.assertEqual(debug["decision_variable_length"], 6)
        self.assertEqual(debug["p_fc_decision_length"], 6)
        self.assertEqual(debug["p_batt_decision_length"], 6)
        self.assertEqual(debug["soc_prediction_steps"], 6)
        self.assertEqual(debug["objective_loop_range"], "0..5")
        self.assertEqual(debug["uses_stages"], [0, 1, 2, 3, 4, 5])
        self.assertTrue(debug["returns_only_first_control"])

    @unittest.skipUnless(casadi_available(), "CasADi is required for horizon sensitivity solve.")
    def test_horizon_sensitivity_changes_when_future_horizon_changes(self) -> None:
        weights = DEFAULT_WEIGHT_SETS[TOTAL_LOAD_QSOC400_QBATT003_QRAMP2E5_WEIGHT_SET]

        debug = build_horizon_sensitivity_debug(weights, current_soc=0.55, prev_fc_kw=100.0)

        self.assertEqual(set(debug["case_id"]), {"A_flat_100", "B_last_step_500", "C_h2_to_h6_500"})
        totals = debug.set_index("case_id")["objective_total"].astype(float)
        self.assertGreater(abs(totals["A_flat_100"] - totals["B_last_step_500"]), 1e-9)
        self.assertGreater(abs(totals["A_flat_100"] - totals["C_h2_to_h6_500"]), 1e-9)
        for k in range(MPC_HORIZON):
            self.assertIn(f"P_fc_plan_stage{k}", debug.columns)
        for k in range(MPC_HORIZON + 1):
            self.assertIn(f"SOC_plan_stage{k}", debug.columns)


if __name__ == "__main__":
    unittest.main()
