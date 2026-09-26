from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FormalPreflightTests(unittest.TestCase):
    def test_repository_preflight_checks_all_required_calibrations_and_audits(self) -> None:
        from v2.preflight import CalibrationStatus, assess_formal_training_preflight

        report = assess_formal_training_preflight()

        self.assertEqual(
            tuple(check.key for check in report.checks),
            (
                "eta_fc_curve",
                "eta_chg",
                "eta_dis",
                "fc_degradation_normalization",
                "battery_q_lifetime_normalization",
                "shore_charging_efficiency",
                "shore_electricity_price",
                "ts_mpc",
                "n_mpc",
                "dqn_switch_steps",
                "tau_lpf",
                "soc_deadband",
                "fc_aggregate_power_mapping",
                "final_dqn_state",
                "final_action_catalog",
                "objective_scale_comparability",
                "terminal_failure_policy",
                "curated_dataset_release",
                "shore_mode_sidecar",
            ),
        )
        self.assertTrue(report.ready)
        self.assertEqual(report.formal_training, "GO")
        self.assertEqual(report.checks[0].status, CalibrationStatus.VERIFIED)
        self.assertEqual(report.checks[1].status, CalibrationStatus.VERIFIED)
        self.assertEqual(report.checks[2].status, CalibrationStatus.VERIFIED)
        self.assertEqual(report.checks[3].status, CalibrationStatus.VERIFIED)
        self.assertEqual(report.checks[4].status, CalibrationStatus.VERIFIED)
        self.assertEqual(report.checks[5].status, CalibrationStatus.VERIFIED)
        by_key = {check.key: check for check in report.checks}
        for key in ("n_mpc", "dqn_switch_steps", "tau_lpf"):
            self.assertEqual(by_key[key].status, CalibrationStatus.VERIFIED)
        self.assertEqual(by_key["soc_deadband"].status, CalibrationStatus.VERIFIED)
        self.assertEqual(
            by_key["objective_scale_comparability"].status,
            CalibrationStatus.VERIFIED,
        )
        self.assertEqual(
            by_key["curated_dataset_release"].status,
            CalibrationStatus.VERIFIED,
        )
        self.assertIn("authenticated", by_key["curated_dataset_release"].evidence)
        self.assertIn("v2_s8_onboard_ais_v1", by_key["final_dqn_state"].evidence)
        self.assertEqual(
            by_key["shore_mode_sidecar"].status,
            CalibrationStatus.VERIFIED,
        )
        self.assertIn("Train=0", by_key["shore_mode_sidecar"].evidence)
        self.assertIn("Validation=0", by_key["shore_mode_sidecar"].evidence)
        self.assertTrue(all(check.evidence.strip() for check in report.checks))

    def test_formal_configuration_gate_accepts_authenticated_release(self) -> None:
        from v2.preflight import require_formal_training_ready

        report = require_formal_training_ready()

        self.assertTrue(report.ready)
        self.assertEqual(report.issues, ())

    def test_preflight_cli_reports_every_check_and_returns_go(self) -> None:
        from v2.main.run_preflight import main

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("FORMAL_TRAINING=GO", output.getvalue())
        self.assertEqual(
            sum(line.startswith("[") for line in output.getvalue().splitlines()),
            19,
        )
        self.assertIn("[VERIFIED] objective_scale_comparability", output.getvalue())
        self.assertIn("[VERIFIED] terminal_failure_policy", output.getvalue())

    def test_terminal_failure_audit_rejects_missing_or_tampered_evidence(self) -> None:
        from v2.preflight import CalibrationStatus, _failure_policy_evidence

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit_summary.json"
            status, evidence = _failure_policy_evidence(path)
            self.assertEqual(status, CalibrationStatus.NO_GO)
            self.assertIn("missing", evidence)

            source = ROOT / "outputs" / "v2_failure_penalty_audit" / "audit_summary.json"
            payload = json.loads(source.read_text(encoding="utf-8"))
            mutations = (
                ("test_payloads_opened", 1),
                ("reference_action_id", "w_1_1_8"),
                ("maximum_completed_raw_economic_cost_cny", 1.0),
                ("failure_penalty_score", 49_999.0),
            )
            for key, value in mutations:
                with self.subTest(key=key):
                    changed = dict(payload)
                    changed[key] = value
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    status, _ = _failure_policy_evidence(path)
                    self.assertEqual(status, CalibrationStatus.NO_GO)

    def test_timescale_cli_rejects_held_out_split_before_reading_payload(self) -> None:
        from v2.main.run_train_only_timescale_audit import main

        error = io.StringIO()
        with patch.object(Path, "read_text", side_effect=AssertionError("payload read")):
            with redirect_stderr(error):
                exit_code = main(
                    [
                        "--split",
                        "Validation",
                        "--train-json",
                        "must-not-open.json",
                        "--autocorrelation-lags",
                        "1",
                        "--rolling-window-samples",
                        "5",
                        "--change-threshold",
                        "1.0",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("Train-only", error.getvalue())

    def test_timescale_cli_runs_train_diagnostic_but_remains_no_go(self) -> None:
        from v2.main.run_train_only_timescale_audit import main

        with tempfile.TemporaryDirectory() as directory:
            payload = Path(directory) / "train.json"
            payload.write_text(json.dumps([float(index) for index in range(20)]))
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "--split",
                        "Train",
                        "--train-json",
                        str(payload),
                        "--provenance-id",
                        "sha256:synthetic-cli-fixture",
                        "--autocorrelation-lags",
                        "1,5",
                        "--rolling-window-samples",
                        "5",
                        "--change-threshold",
                        "1.0",
                    ]
                )

        self.assertEqual(exit_code, 2)
        rendered = json.loads(output.getvalue())
        self.assertEqual(rendered["formal_selection_status"], "NO-GO")
        self.assertEqual(rendered["candidate_switch_steps"], [5, 10])

    def test_preflight_report_matches_current_go_boundary(self) -> None:
        report = (ROOT / "docs" / "v2_preflight_report.md").read_text(encoding="utf-8")
        self.assertIn("FORMAL_TRAINING = GO", report)
        self.assertIn("30", report)
        self.assertIn("S8", report)
        self.assertIn("36-action", report)
        self.assertIn("Test payload", report)


if __name__ == "__main__":
    unittest.main()
