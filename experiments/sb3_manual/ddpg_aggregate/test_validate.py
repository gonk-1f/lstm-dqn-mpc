"""Bounded validation checks using real data and an initialized current-reward model."""
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
                      "battery_degradation_cost_cny", "shore_cost_cny"):
            self.assertAlmostEqual(result[field], getattr(ledger, field), places=12)
        self.assertAlmostEqual(result["total_cost_cny"],
                               ledger.total_cost_cny + result["c_bat_eq_cny"], places=12)
        self.assertGreaterEqual(result["hydrogen_mass_kg"], 0)
        self.assertAlmostEqual(result["battery_charge_bus_kwh"], max(-battery_kw, 0) / 3600)
        self.assertAlmostEqual(result["battery_discharge_bus_kwh"], max(battery_kw, 0) / 3600)

    def test_unexecuted_failure_exports_shutdown_cost_without_physical_supply(self):
        env = self.make_env()
        before = env.physical
        from .economic_reward import shutdown_economic_components
        _, costs = shutdown_economic_components(env.physics, before)
        info = {**costs, "terminal_reason": "no_feasible_physical_power",
                "fc_actual_kw": 0.0, "step_executed": False}
        with mock.patch.object(env.physics, "interval",
                               side_effect=AssertionError("Unexecuted physical interval")):
            result = self.validation.step_accounting(env, before, info)
        self.assertTrue(result)
        self.assertEqual(result["total_cost_cny"], costs["total_cost_cny"])
        self.assertGreater(result["fuel_cell_start_stop_cost_cny"], 0.0)
        for name in ("hydrogen_mass_kg", "battery_charge_bus_kwh", "battery_discharge_bus_kwh",
                     "battery_degradation_cost_cny", "soc_deviation_penalty_cny"):
            self.assertEqual(result[name], 0.0)
        self.assertIs(env.physical, before)
        self.assertEqual(env.index, 0)

    def test_three_real_validation_steps_keep_model_unchanged_and_export(self):
        import torch
        from stable_baselines3 import DDPG
        predict = DDPG.predict

        def deterministic_predict(*args, **kwargs):
            self.assertIs(kwargs.get("deterministic"), True)
            return predict(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            fixture_env = self.make_env()
            torch.set_num_threads(1)
            fresh = DDPG("MlpPolicy", fixture_env, buffer_size=32, device="cpu", seed=42,
                         policy_kwargs=dict(net_arch=[16, 16]))
            fresh._ddpg_reward_identity = dict(fixture_env.reward_identity)
            self.assertEqual((fresh.num_timesteps, fresh._n_updates), (0, 0))
            model = Path(tmp) / "current_reward_model.zip"
            fresh.save(model)
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
            self.assertNotIn("soc_penalty", steps[0])
            self.assertNotIn("c_soc_cny", steps[0])
            self.assertTrue(all(math.isfinite(float(row["battery_current_a"])) for row in steps))
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
