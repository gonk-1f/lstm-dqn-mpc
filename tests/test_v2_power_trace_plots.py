from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from v2.evaluation.formal_policy import EpisodePowerTrace


class TestPowerTracePlots(unittest.TestCase):
    @staticmethod
    def _trace(sample_id: str, *, completed: bool = True) -> EpisodePowerTrace:
        return EpisodePowerTrace(
            sample_id=sample_id,
            time_s=(0.0, 30.0, 60.0),
            load_power_kw=(100.0, 80.0, -60.0),
            fuel_cell_power_kw=(70.0, 60.0, 0.0),
            battery_bus_power_kw=(30.0, 20.0, -60.0),
            operating_mode=("onboard", "onboard", "shore_charging"),
            soc_time_s=(0.0, 30.0, 60.0, 90.0),
            soc=(0.60, 0.59, 0.585, 0.592),
            terminal_boundary_time_s=95.0,
            terminal_boundary_load_kw=0.0,
            completed=completed,
            failure_kind=None if completed else "physical_infeasibility",
        )

    def test_writes_one_png_per_episode_csv_manifest_and_html_index(self) -> None:
        from v2.evaluation.power_trace_plots import write_power_trace_plots

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plots"
            manifest = write_power_trace_plots(
                output,
                (self._trace("test_a"), self._trace("test_b", completed=False)),
                policy_id="H4_round_020",
            )

            self.assertEqual(manifest, output / "power_trace_manifest.csv")
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                ["index.html", "plots", "power_trace_manifest.csv"],
            )
            pngs = sorted((output / "plots").glob("*.png"))
            self.assertEqual([path.name for path in pngs], ["test_a.png", "test_b.png"])
            self.assertTrue(all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pngs))
            with manifest.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["sample_id"] for row in rows], ["test_a", "test_b"])
            self.assertEqual([row["status"] for row in rows], ["COMPLETE", "FAILED"])
            self.assertEqual([row["policy_id"] for row in rows], ["H4_round_020"] * 2)
            self.assertEqual([row["terminal_boundary_time_s"] for row in rows], ["95.0"] * 2)
            self.assertEqual([row["terminal_boundary_load_kw"] for row in rows], ["0.0"] * 2)
            self.assertEqual([row["unexecuted_tail_seconds"] for row in rows], ["5.0"] * 2)
            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("test_a", html)
            self.assertIn("physical_infeasibility", html)
            self.assertIn("plots/test_b.png", html)

    def test_terminal_boundary_curve_starts_after_last_executed_interval(self) -> None:
        from v2.evaluation.power_trace_plots import _terminal_boundary_curve

        time_s, load_kw = _terminal_boundary_curve(self._trace("test_a"))
        self.assertEqual(time_s, (90.0, 95.0))
        self.assertEqual(load_kw, (-60.0, 0.0))

    def test_validation_title_does_not_mislabel_plots_as_test(self) -> None:
        from v2.evaluation.power_trace_plots import write_power_trace_plots

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plots"
            write_power_trace_plots(
                output,
                (self._trace("validation_a"),),
                policy_id="H4_round_020_tau_180",
                artifact_title="H4 tau=180 s Validation power traces",
            )

            html = (output / "index.html").read_text(encoding="utf-8")
            self.assertIn("H4 tau=180 s Validation power traces", html)
            self.assertNotIn("Test power traces", html)

    def test_rejects_existing_destination_without_overwriting(self) -> None:
        from v2.evaluation.power_trace_plots import write_power_trace_plots

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "plots"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_power_trace_plots(
                    output,
                    (self._trace("test_a"),),
                    policy_id="H4_round_020",
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
