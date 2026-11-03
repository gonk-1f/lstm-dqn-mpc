"""Bounded validation checks using real Validation data and a trained model."""
from __future__ import annotations

import importlib
import importlib.util
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

PACKAGE = "experiments.sb3_manual.ddpg_aggregate"


class EntryPointTests(unittest.TestCase):
    def test_module_is_importable(self):
        name = PACKAGE + ".validate"
        self.assertIsNotNone(importlib.util.find_spec(name))
        self.assertTrue(callable(importlib.import_module(name).run_validation))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.validation = importlib.import_module(PACKAGE + ".validate")

    def make_env(self):
        from .env import AggregateDDPGEnv
        env = AggregateDDPGEnv(self.validation.DEFAULT_DATA_ROOT, split="validation")
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def test_step_accounting_reuses_executed_ledger_without_physics_step(self):
        env = self.make_env()
        before = env.physical
        original_interval = env.physics.interval
        captured = []

        def capture(*args, **kwargs):
            result = original_interval(*args, **kwargs)
            captured.append(result)
            return result

        with mock.patch.object(env.physics, "interval", side_effect=capture):
            _, _, _, _, info = env.step([0.0])
        self.assertEqual(len(captured), 1)
        state_after, ledger, battery_kw = captured[0]
        with mock.patch.object(env.physics, "interval", side_effect=AssertionError("Extra physical step")):
            result = self.validation.step_accounting(env, before, info)
        self.assertIs(env.physical, state_after)
        self.assertEqual(env.index, 1)
        for field in ("h2_cost_cny", "fuel_cell_degradation_cost_cny",
                      "battery_degradation_cost_cny", "shore_cost_cny", "total_cost_cny"):
            self.assertAlmostEqual(result[field], getattr(ledger, field), places=12)
        self.assertGreaterEqual(result["hydrogen_mass_kg"], 0)
        self.assertAlmostEqual(result["battery_charge_bus_kwh"], max(-battery_kw, 0) / 3600)
        self.assertAlmostEqual(result["battery_discharge_bus_kwh"], max(battery_kw, 0) / 3600)

    def test_unexecuted_failure_has_no_physical_accounting(self):
        env = self.make_env()
        before = env.physical
        info = {"terminal_reason": "no_feasible_physical_power", "fc_actual_kw": 0.0}
        with mock.patch.object(self.validation, "hydrogen_mass_from_map_kg",
                               side_effect=AssertionError("Unexecuted hydrogen")):
            result = self.validation.step_accounting(env, before, info)
        self.assertTrue(result)
        self.assertTrue(all(value is None for value in result.values()))
        self.assertIs(env.physical, before)
        self.assertEqual(env.index, 0)

    def test_three_real_validation_steps_keep_model_unchanged_and_export(self):
        import torch
        from stable_baselines3 import DDPG
        model = self.validation.PROJECT_ROOT / (
            "outputs/ddpg_train_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip"
        )
        self.assertTrue(model.is_file(), "The completed formal model is required for this check")
        predict = DDPG.predict

        def deterministic_predict(*args, **kwargs):
            self.assertIs(kwargs.get("deterministic"), True)
            return predict(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(DDPG, "learn", side_effect=AssertionError("Validation learn()")), \
                 mock.patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Validation Adam")), \
                 mock.patch.object(DDPG, "predict", new=deterministic_predict):
                metrics = self.validation.run_validation(
                    model_path=model, output_root=Path(tmp), episodes=1,
                    max_steps_per_episode=3, device="cpu",
                )
            self.assertEqual(metrics["validation_voyages"], 1)
            self.assertFalse(metrics["validation_set_complete"])
            for key in ("parameters_unchanged", "optimizer_state_unchanged", "counters_unchanged",
                        "replay_unchanged", "no_gradients"):
                self.assertIs(metrics["integrity"][key], True, key)
            output = Path(metrics["output_dir"])
            with (output / "validation_step_data.csv").open(encoding="utf-8-sig", newline="") as handle:
                steps = list(csv.DictReader(handle))
            self.assertEqual([int(row["step_index"]) for row in steps], [1, 2, 3])
            self.assertTrue(all(row["step_executed"] == "True" for row in steps))
            self.assertEqual(steps[-1]["terminal_reason"], "validation_step_limit")
            self.assertEqual(steps[-1]["evaluation_truncated"], "True")
            self.assertGreater(float(steps[-1]["total_cost_cny"]), 0)
            with (output / "validation_summary.csv").open(encoding="utf-8-sig", newline="") as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]["full_voyage_economic_cost_cny"], "")
            saved = json.loads((output / "validation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["total_environment_steps"], 3)
            self.assertEqual(saved["total_executed_steps"], 3)
            self.assertIsNone(saved["mean_full_voyage_economic_cost_cny"])
            self.assertTrue(math.isfinite(saved["mean_episode_reward"]))
            for name in ("validation_reward_curve.png", "validation_power_curves.png",
                         "validation_soc_curves.png"):
                path = output / name
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
