"""Isolation checks for the simple-reward experiment's native Zoo pipeline."""
from pathlib import Path
import inspect
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import yaml


class SimplePipelineTests(unittest.TestCase):
    def test_formal_algorithm_parameters_and_episode_clocks_are_unchanged(self):
        directory = Path(__file__).parent
        original = yaml.safe_load((directory / "zoo_td3.yml").read_text(encoding="utf-8"))["AggregateShipTD3-v0"]
        simple = yaml.safe_load((directory / "zoo_td3_simple.yml").read_text(encoding="utf-8"))["AggregateShipTD3Simple-v0"]
        self.assertEqual({k: v for k, v in original.items() if k != "callback"},
                         {k: v for k, v in simple.items() if k != "callback"})
        original_callback = next(iter(original["callback"][0].values()))
        simple_callback = next(iter(simple["callback"][0].values()))
        self.assertEqual(original_callback, simple_callback)
        self.assertEqual(original["callback"][1:], simple["callback"][1:])

    def test_evaluation_requires_an_explicit_simple_reward_selection(self):
        from .evaluation import evaluate, argument_parser
        self.assertEqual(inspect.signature(evaluate).parameters["reward_mode"].default, "economic")
        args = argument_parser("check").parse_args(["--model-path", "model.zip", "--reward-mode", "simple"])
        self.assertEqual(args.reward_mode, "simple")

    def test_legacy_callback_defaults_stay_in_the_original_experiment(self):
        from .callbacks import ECONOMIC_FIELDS, TD3TrainingCallback
        self.assertEqual(TD3TrainingCallback.economic_fields, ECONOMIC_FIELDS)
        self.assertEqual(TD3TrainingCallback.evaluation_extra_args, ())
        self.assertEqual(TD3TrainingCallback.environment_id, "AggregateShipTD3-v0")

    def test_simple_resume_rejects_the_original_reward_before_model_loading(self):
        from .simple_training import validate_resume_metadata
        from .simple_reward import SIMPLE_REWARD_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.zip"
            for version, accepted in (("td3_economic_scaled_1e4_v1", False), (SIMPLE_REWARD_VERSION, True)):
                with ZipFile(path, "w") as archive:
                    archive.writestr("data", json.dumps({
                        "_td3_reward_identity": {"reward_version": version, "algorithm": "TD3", "reward_scale": 0.0001},
                        "reward_scale": 0.0001, "_td3_completed_episodes": 1, "_td3_successful_episodes": 1,
                    }))
                if accepted:
                    validate_resume_metadata(["--trained-agent", str(path)])
                else:
                    with self.assertRaisesRegex(ValueError, "simple-reward"):
                        validate_resume_metadata(["--trained-agent", str(path)])

    def test_pause_file_stops_only_at_a_real_episode_boundary(self):
        from stable_baselines3.common.logger import Logger
        from .callbacks import TD3TrainingCallback
        from .simple_callbacks import SimpleRewardTrainingCallback
        with tempfile.TemporaryDirectory() as tmp:
            callback = SimpleRewardTrainingCallback()
            callback.model = SimpleNamespace(logger=Logger(None, []))
            callback.pause_request = Path(tmp) / "pause_after_episode.request"
            callback.pause_request.touch()
            with patch.object(TD3TrainingCallback, "_on_step", return_value=True):
                callback.locals = {"dones": [False]}
                self.assertTrue(callback._on_step())
                callback.locals = {"dones": [True]}
                self.assertFalse(callback._on_step())

    def test_control_only_configuration_keeps_the_same_algorithm_and_episode_clocks(self):
        directory = Path(__file__).parent
        simple = yaml.safe_load((directory / "zoo_td3_simple.yml").read_text(encoding="utf-8"))["AggregateShipTD3Simple-v0"]
        control = yaml.safe_load((directory / "zoo_td3_control.yml").read_text(encoding="utf-8"))["AggregateShipTD3Control-v0"]
        self.assertEqual({k: v for k, v in simple.items() if k != "callback"},
                         {k: v for k, v in control.items() if k != "callback"})
        self.assertEqual(next(iter(simple["callback"][0].values())), next(iter(control["callback"][0].values())))
        self.assertEqual(simple["callback"][1:], control["callback"][1:])

    def test_control_resume_rejects_the_hydrogen_reward(self):
        from .simple_training import validate_resume_metadata
        from .simple_reward import SIMPLE_REWARD_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.zip"
            with ZipFile(path, "w") as archive:
                archive.writestr("data", json.dumps({
                    "_td3_reward_identity": {"reward_version": SIMPLE_REWARD_VERSION, "algorithm": "TD3", "reward_scale": 0.0001},
                    "reward_scale": 0.0001, "_td3_completed_episodes": 1, "_td3_successful_episodes": 1,
                }))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_resume_metadata(["--control-only", "--trained-agent", str(path)])


if __name__ == "__main__":
    unittest.main()
