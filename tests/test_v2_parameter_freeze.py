from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FrozenBatteryLifetimeTests(unittest.TestCase):
    def test_factor_is_frozen_while_evidence_remains_secondary_literature(self) -> None:
        from v2.models import battery_degradation as model

        self.assertEqual(model.BATTERY_LIFETIME_THROUGHPUT_FACTOR, 15_000.0)
        self.assertAlmostEqual(
            model.BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
            624_000.0 / 432.0,
        )
        self.assertAlmostEqual(
            model.BATTERY_LIFETIME_Q_AH,
            15_000.0 * (624_000.0 / 432.0),
        )
        self.assertEqual(model.BATTERY_LIFETIME_CONFIGURATION_STATUS, "FROZEN")
        self.assertEqual(
            model.BATTERY_LIFETIME_EVIDENCE_STATUS,
            "SECONDARY_LITERATURE / LITERATURE-CALIBRATED",
        )
        self.assertEqual(
            model.BATTERY_LIFETIME_SENSITIVITY_FACTORS,
            (10_000.0, 15_000.0, 20_000.0),
        )


class FrozenTimeScaleTests(unittest.TestCase):
    def test_formal_baseline_freezes_n_m_and_tau_with_independent_semantics(self) -> None:
        from v2 import config as model

        self.assertEqual(model.TS_MPC_SECONDS, 30.0)
        self.assertEqual(model.N_MPC, 5)
        self.assertEqual(model.DQN_SWITCH_STEPS, 5)
        self.assertEqual(model.TAU_LPF_SECONDS, 90.0)
        self.assertEqual(
            model.FORMAL_TIMESCALE_CONFIGURATION_STATUS,
            "FROZEN_PROJECT_DESIGN",
        )
        self.assertEqual(model.N_MPC_EVIDENCE_STATUS, "PROJECT_DESIGN")
        self.assertEqual(
            model.DQN_SWITCH_STEPS_EVIDENCE_STATUS,
            "PROJECT_DESIGN",
        )
        self.assertEqual(model.TAU_LPF_EVIDENCE_STATUS, "PROJECT_DESIGN")

        baseline = model.TimeScaleConfig.formal_baseline()
        self.assertEqual(baseline.ts_mpc_seconds, 30.0)
        self.assertEqual(baseline.n_mpc, 5)
        self.assertEqual(baseline.dqn_switch_steps, 5)
        self.assertEqual(baseline.prediction_seconds, 150.0)
        self.assertEqual(baseline.switch_seconds, 150.0)
        self.assertEqual(baseline.mpc_solves_per_action, 5)
        self.assertNotEqual(
            baseline.n_mpc_semantics,
            baseline.dqn_switch_steps_semantics,
        )

    def test_lpf_alpha_uses_frozen_thirty_over_ninety_ratio(self) -> None:
        import math

        from v2.config import TAU_LPF_SECONDS, TS_MPC_SECONDS

        self.assertAlmostEqual(
            math.exp(-TS_MPC_SECONDS / TAU_LPF_SECONDS),
            math.exp(-30.0 / 90.0),
        )


class FrozenParameterPreflightTests(unittest.TestCase):
    def test_frozen_parameters_do_not_block_formal_training(self) -> None:
        from v2.data.raw_inventory import RawExcelInventory
        from v2.preflight import (
            CalibrationStatus,
            assess_data_preflight,
            assess_formal_training_preflight,
        )

        report = assess_formal_training_preflight()
        by_key = {check.key: check for check in report.checks}

        for key in (
            "battery_q_lifetime_normalization",
            "n_mpc",
            "dqn_switch_steps",
            "tau_lpf",
        ):
            with self.subTest(key=key):
                self.assertEqual(by_key[key].status, CalibrationStatus.VERIFIED)

        self.assertIn("SECONDARY_LITERATURE", by_key["battery_q_lifetime_normalization"].evidence)
        self.assertIn("FROZEN", by_key["battery_q_lifetime_normalization"].evidence)
        self.assertIn("FROZEN_PROJECT_DESIGN", by_key["n_mpc"].evidence)
        self.assertIn("FROZEN_PROJECT_DESIGN", by_key["dqn_switch_steps"].evidence)
        self.assertIn("FROZEN_PROJECT_DESIGN", by_key["tau_lpf"].evidence)
        self.assertIn("not vessel-measured", by_key["tau_lpf"].evidence)
        self.assertIn("not a unique optimum", by_key["tau_lpf"].evidence)

        self.assertEqual(
            tuple(check.key for check in report.checks if check.status is not CalibrationStatus.VERIFIED),
            ("final_dqn_state", "final_action_catalog"),
        )
        self.assertFalse(report.ready)
        self.assertEqual(report.formal_training, "NO-GO")

        data_report = assess_data_preflight(
            inventory=RawExcelInventory(root=ROOT, records=()),
            technical_specification=None,
        )
        self.assertEqual(
            {issue.code for issue in data_report.issues},
            {"missing_usable_raw_measurements", "missing_technical_specification"},
        )


if __name__ == "__main__":
    unittest.main()
