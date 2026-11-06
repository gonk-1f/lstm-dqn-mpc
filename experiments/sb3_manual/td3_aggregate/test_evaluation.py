"""Bounded TD3 evaluation checks; never run full Validation8 or load Test data."""
from __future__ import annotations

import csv
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

PACKAGE = "experiments.sb3_manual.td3_aggregate"


class EvaluationEntryTests(unittest.TestCase):
    def test_unified_evaluator_exists(self):
        self.assertIsNotNone(importlib.util.find_spec(PACKAGE + ".evaluation"))

    def test_evaluator_import_avoids_ddpg_training_registration(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import " + PACKAGE + ".evaluation; "
            "assert 'experiments.sb3_manual.ddpg_aggregate.validate' not in sys.modules; "
            "assert 'experiments.sb3_manual.ddpg_aggregate.zoo_register' not in sys.modules; "
            "assert 'experiments.sb3_manual.ddpg_aggregate.test4_env' not in sys.modules",
        ], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class SummarySchemaTests(unittest.TestCase):
    def test_bounded_and_terminal_rows_have_the_same_reward_metadata_schema(self):
        # A minimal endpoint forces both boundaries without executing a full
        # voyage. Physics and all cost formulas remain outside this unit test.
        evaluation = importlib.import_module(PACKAGE + ".evaluation")

        class Endpoint:
            reward_identity = {"reward_version": "td3_economic_scaled_1e4_v1"}

            def __init__(self, terminal):
                self.unwrapped = self
                self.terminal = terminal

            def reset(self, seed=None):
                self.index = 0
                self.physical = SimpleNamespace(soc=0.6)
                return [0.0], {"sample_id": "voyage", "split": "validation"}

            def _episode_summary(self, completed, reason):
                return {"completed": completed, "terminal_reason": reason, "coverage": 1 / 10,
                        "sample_id": "voyage", "split": "validation"}

            def step(self, action):
                self.index += 1
                info = {name: 0.0 for name in evaluation.ACCOUNTING_FIELDS}
                info.update(raw_reward=-1.0, scaled_reward=-0.0001, reward_scale=0.0001,
                            battery_actual_kw=0.0, step_executed=True, requested_fc_kw=0.0,
                            fc_actual_kw=0.0, load_kw=0.0, terminal_reason="complete" if self.terminal else None)
                if self.terminal:
                    info["vessel_episode"] = {
                        **self._episode_summary(True, "complete"),
                        "algorithm_reward_version": self.reward_identity["reward_version"],
                    }
                return [0.0], -0.0001, self.terminal, False, info

        model = SimpleNamespace(predict=lambda obs, deterministic: ([0.0], None))
        summaries = []
        for terminal in (False, True):
            writer = csv.DictWriter(io.StringIO(), fieldnames=evaluation.STEP_FIELDS)
            summary, _ = evaluation._rollout(model, Endpoint(terminal), "voyage", 1, 42, 1, "validation", writer)
            summaries.append(summary)
        self.assertEqual(set(summaries[0]), set(summaries[1]))


class BoundedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluation = importlib.import_module(PACKAGE + ".evaluation")
        cls.wrapper = importlib.import_module(PACKAGE + ".reward_wrapper")

    def fixture_model(self, directory):
        import torch
        from stable_baselines3 import TD3

        torch.set_num_threads(1)
        env = self.wrapper.make_env(
            self.evaluation.DEFAULT_DATA_ROOT, split="train",
            fixed_train_id="parent_005_seg01",
        )
        self.addCleanup(env.close)
        model = TD3("MlpPolicy", env, buffer_size=32, device="cpu", seed=42,
                    policy_kwargs={"net_arch": [16, 16]})
        model._td3_reward_identity = dict(env.reward_identity)
        model.reward_scale = 0.0001
        path = Path(directory) / "model.zip"
        model.save(path)
        return path

    def test_bounded_validation_is_native_deterministic_and_preserves_raw_costs(self):
        import torch
        from stable_baselines3 import TD3, DDPG

        native_predict = TD3.predict
        native_load = TD3.load

        def predict(*args, **kwargs):
            self.assertIs(kwargs.get("deterministic"), True)
            return native_predict(*args, **kwargs)

        with tempfile.TemporaryDirectory() as temporary:
            path = self.fixture_model(temporary)
            previous_test_adapter = sys.modules.get("experiments.sb3_manual.ddpg_aggregate.test4_env")
            with mock.patch.object(TD3, "load", wraps=native_load) as load, \
                 mock.patch.object(TD3, "predict", new=predict), \
                 mock.patch.object(TD3, "learn", side_effect=AssertionError("evaluation learn")), \
                 mock.patch.object(DDPG, "load", side_effect=AssertionError("DDPG load")), \
                 mock.patch.object(torch.optim.Adam, "step", side_effect=AssertionError("evaluation optimizer")):
                metrics = self.evaluation.evaluate(
                    path, output_root=Path(temporary), episodes=1, max_steps=3,
                )
            self.assertEqual(load.call_args.kwargs["action_noise"], None)
            self.assertEqual(load.call_args.kwargs["device"], "cpu")
            self.assertEqual(metrics["split"], "validation")
            self.assertEqual(metrics["validation_voyages"], 1)
            self.assertFalse(metrics["validation_set_complete"])
            self.assertEqual(metrics["total_environment_steps"], 3)
            self.assertTrue(all(metrics["integrity"].values()))
            self.assertEqual(metrics["reward_scale"], 0.0001)
            summary = metrics["per_voyage"][0]
            self.assertAlmostEqual(summary["algorithm_reward_sum"], summary["raw_reward_sum"] / 10000)
            self.assertEqual(summary["episode_reward"], summary["raw_reward_sum"])
            self.assertGreater(summary["total_cost_cny"], 0)
            self.assertIsNone(summary["full_voyage_economic_cost_cny"])
            output = Path(metrics["output_dir"])
            with (output / "validation_step_data.csv").open(encoding="utf-8-sig", newline="") as handle:
                steps = list(csv.DictReader(handle))
            self.assertEqual(len(steps), 3)
            for row in steps:
                self.assertEqual(float(row["reward"]), float(row["raw_reward"]))
                self.assertEqual(float(row["scaled_reward"]), float(row["raw_reward"]) / 10000)
                self.assertEqual(float(row["reward_scale"]), 0.0001)
            self.assertEqual(steps[-1]["terminal_reason"], "validation_step_limit")
            self.assertEqual(steps[-1]["evaluation_truncated"], "True")
            saved = json.loads((output / "validation_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["per_voyage"], metrics["per_voyage"])
            for name in ("validation_reward_curve.png", "validation_power_curves.png", "validation_soc_curves.png"):
                self.assertEqual((output / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertIs(sys.modules.get("experiments.sb3_manual.ddpg_aggregate.test4_env"), previous_test_adapter)

    def test_fixed_train_diagnostic_records_raw_and_scaled_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.fixture_model(temporary)
            metrics = self.evaluation.evaluate(
                path, split="fixed_train", output_root=Path(temporary), max_steps=2, episode=20,
            )
            summary = metrics["summary"]
            self.assertEqual(metrics["split"], "train")
            self.assertEqual(summary["sample_id"], "parent_005_seg01")
            self.assertEqual(summary["total_steps"], 9258)
            self.assertEqual(summary["episode"], 20)
            self.assertEqual(summary["episode_length"], 2)
            self.assertTrue(summary["evaluation_truncated"])
            self.assertTrue(all(metrics["integrity"].values()))
            self.assertAlmostEqual(summary["algorithm_reward_sum"], summary["raw_reward_sum"] / 10000)

    def test_existing_evaluation_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sentinel = directory / "validation_metrics.json"
            sentinel.write_text("existing-result", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.evaluation.evaluate(directory / "missing-model.zip", output_dir=directory)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing-result")

    def test_vecnormalize_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "vecnormalize.pkl").write_bytes(b"must not load")
            with self.assertRaisesRegex(ValueError, "VecNormalize"):
                self.evaluation.evaluate(directory / "missing-model.zip", output_root=directory)


if __name__ == "__main__":
    unittest.main()
