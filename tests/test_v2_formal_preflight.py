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
                "final_dqn_state",
                "final_action_catalog",
                "objective_scale_comparability",
            ),
        )
        self.assertFalse(report.ready)
        self.assertEqual(report.formal_training, "NO-GO")
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
        self.assertTrue(all(check.evidence.strip() for check in report.checks))

    def test_formal_gate_blocks_before_payload_or_data_provenance_access(self) -> None:
        from v2.preflight import FormalTrainingBlockedError, load_formal_train_payload

        accesses = 0

        def payload_loader() -> object:
            nonlocal accesses
            accesses += 1
            return object()

        with self.assertRaises(FormalTrainingBlockedError) as caught:
            load_formal_train_payload(
                split="Train",
                inventory=object(),
                technical_specification=object(),
                payload_loader=payload_loader,
            )

        self.assertEqual(accesses, 0)
        self.assertEqual(len(caught.exception.report.checks), 15)
        self.assertIn("FORMAL_TRAINING=NO-GO", str(caught.exception))

    def test_preflight_cli_reports_every_check_and_returns_no_go(self) -> None:
        from v2.main.run_preflight import main

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 2)
        self.assertIn("FORMAL_TRAINING=NO-GO", output.getvalue())
        self.assertEqual(
            sum(line.startswith("[") for line in output.getvalue().splitlines()),
            15,
        )
        self.assertIn("[VERIFIED] objective_scale_comparability", output.getvalue())

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

    def test_preflight_report_has_exactly_eighteen_numbered_sections(self) -> None:
        report = (ROOT / "docs" / "v2_preflight_report.md").read_text(encoding="utf-8")
        headings = [line for line in report.splitlines() if line.startswith("## ")]

        self.assertEqual(len(headings), 18)
        self.assertEqual(
            [heading.split(".", 1)[0] for heading in headings],
            [f"## {index}" for index in range(1, 19)],
        )
        self.assertIn("FORMAL_TRAINING = NO-GO", report)


if __name__ == "__main__":
    unittest.main()
