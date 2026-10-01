from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


class TestTauLpfDiagnostics(unittest.TestCase):
    def test_summary_reports_fc_variation_and_battery_burden(self) -> None:
        from v2.evaluation.formal_policy import (
            EpisodeEvaluation,
            EpisodePowerTrace,
            PolicyEvaluation,
        )
        from v2.evaluation.tau_lpf_screen import build_tau_screen_summary

        episode = EpisodeEvaluation(
            sample_id="validation_a",
            completed=True,
            failure_kind=None,
            transition_count=1,
            executed_mpc_steps=3,
            h2_cost_cny=10.0,
            fc_degradation_cost_cny=2.0,
            battery_degradation_cost_cny=3.0,
            shore_cost_cny=0.0,
            raw_economic_cost_cny=15.0,
            failure_penalty_score=0.0,
            learning_reward=-15.0,
            soc_min=0.55,
            soc_max=0.60,
            action_counts=(("w_8_1_1", 1),),
        )
        trace = EpisodePowerTrace(
            sample_id="validation_a",
            time_s=(0.0, 30.0, 60.0),
            load_power_kw=(100.0, 120.0, 80.0),
            fuel_cell_power_kw=(10.0, 30.0, 20.0),
            battery_bus_power_kw=(90.0, 90.0, 60.0),
            operating_mode=("onboard", "onboard", "onboard"),
            soc_time_s=(0.0, 30.0, 60.0, 90.0),
            soc=(0.60, 0.58, 0.56, 0.55),
            completed=True,
            failure_kind=None,
        )

        summary, rows = build_tau_screen_summary(
            tau_lpf_seconds=180.0,
            evaluation=PolicyEvaluation("H4_round_020_tau_180", (episode,)),
            traces=(trace,),
            checkpoint_trained_tau_seconds=90.0,
        )

        self.assertEqual(summary["screening_status"], "PRETRAIN_ENVIRONMENT_SCREEN_ONLY")
        self.assertEqual(summary["tau_lpf_seconds"], 180.0)
        self.assertEqual(summary["checkpoint_trained_tau_seconds"], 90.0)
        self.assertEqual(summary["completed_episodes"], 1)
        self.assertEqual(summary["raw_economic_cost_cny"], 15.0)
        self.assertEqual(summary["fc_total_variation_kw"], 40.0)
        self.assertAlmostEqual(summary["battery_absolute_energy_kwh"], 2.0)
        self.assertEqual(rows[0]["fc_max_abs_step_kw"], 20.0)
        self.assertEqual(rows[0]["soc_min"], 0.55)


class TestTauLpfValidationCli(unittest.TestCase):
    def test_uses_validation_for_180_and_300_and_never_opens_test(self) -> None:
        from v2.main import run_tau_lpf_validation_screen as module

        class Dataset:
            opened_test_payloads = 0

            def load_validation(self):
                return ("validation_episode",)

            def load_final_test(self, *args, **kwargs):
                raise AssertionError("Test must not be opened by tau screen")

            def split_episode_ids(self, split):
                self.split = split
                return ("train_episode",)

        dataset = Dataset()
        manifest = SimpleNamespace(
            selected_round=20,
            input_manifest_hashes=(
                ("ais", "a" * 64),
                ("modes", "b" * 64),
                ("power", "c" * 64),
            ),
        )
        evaluation = SimpleNamespace(
            policy_id="policy",
            episodes=(),
        )
        calls: list[float] = []

        def evaluate(*, episodes, policy, tau_lpf_seconds):
            self.assertEqual(episodes, ("validation_episode",))
            calls.append(tau_lpf_seconds)
            return evaluation, (SimpleNamespace(sample_id="validation_episode"),)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection"
            selection.mkdir()
            (selection / "best_validation.pt").write_bytes(b"checkpoint")
            output = root / "screen"
            with (
                mock.patch.object(module, "load_selection_outputs", return_value=manifest),
                mock.patch.object(module.FormalTrainingDataset, "open", return_value=dataset),
                mock.patch.object(module, "_manifest_hashes", return_value={
                    "ais": "a" * 64,
                    "modes": "b" * 64,
                    "power": "c" * 64,
                }),
                mock.patch.object(module, "load_evaluation_profile", return_value=(
                    SimpleNamespace(experiment_id="H4"),
                    object(),
                    {},
                )),
                mock.patch.object(
                    module,
                    "DqnAgent",
                    return_value=SimpleNamespace(greedy_action=lambda state: 0),
                ),
                mock.patch.object(module, "EpisodeShuffleSchedule", return_value=object()),
                mock.patch.object(module, "load_checkpoint", return_value=SimpleNamespace(
                    round_index=20,
                    episode_position=0,
                )),
                mock.patch.object(module, "evaluate_formal_policy_with_power_traces", side_effect=evaluate),
                mock.patch.object(module, "build_tau_screen_summary", side_effect=lambda **kwargs: ({
                    "tau_lpf_seconds": kwargs["tau_lpf_seconds"],
                    "completed_episodes": 1,
                    "episode_count": 1,
                    "raw_economic_cost_cny": 1.0,
                    "fc_total_variation_kw": 2.0,
                    "battery_absolute_energy_kwh": 3.0,
                }, ())),
                mock.patch.object(module, "write_tau_screen_outputs"),
                redirect_stdout(io.StringIO()),
            ):
                code = module.main([
                    "--selection-dir", str(selection),
                    "--output-dir", str(output),
                    "--reward-scale", str(root / "scale.json"),
                    "--taus", "180", "300",
                ])

        self.assertEqual(code, 0)
        self.assertEqual(calls, [180.0, 300.0])
        self.assertEqual(dataset.split, "train")
        self.assertEqual(dataset.opened_test_payloads, 0)


if __name__ == "__main__":
    unittest.main()
