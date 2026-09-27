from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestRewardScaleCalibration(unittest.TestCase):
    @staticmethod
    def _calibration():
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import calibrate_reward_scale

        return calibrate_reward_scale(
            (10.0, 20.0, 30.0),
            provenance=DatasetProvenance(
                "operating_dataset_zero_boundary_v2",
                "sha256:reward-scale-fixture",
                DataSplit.TRAIN,
            ),
            audit_id="fixture",
            reason="verify uniform scaling",
        )

    @staticmethod
    def _document():
        from v2.training.reward_scaling import build_reward_scale_document

        return build_reward_scale_document(
            calibration=TestRewardScaleCalibration._calibration(),
            reference_action_id="w_8_1_1",
            manifest_hashes={
                "power": "a" * 64,
                "ais": "b" * 64,
                "modes": "c" * 64,
            },
            train_segment_ids=("train-a",),
            test_payloads_opened=0,
        )

    def test_scaled_training_reward_divides_cost_and_failure_score_together(self) -> None:
        from v2.economics import scale_learning_reward

        self.assertAlmostEqual(
            scale_learning_reward(-50_030.0, calibration=self._calibration()),
            -2501.5,
        )

    def test_document_round_trip_and_held_out_rejection(self) -> None:
        from v2.analysis.action_screening import HeldOutSelectionError
        from v2.training.reward_scaling import load_reward_scale_document

        document = self._document()
        loaded = load_reward_scale_document(document)
        self.assertEqual(loaded.digest, self._calibration().digest)

        document["split"] = "validation"
        with self.assertRaises(HeldOutSelectionError):
            load_reward_scale_document(document)

    def test_document_rejects_cost_digest_manifest_and_action_mutation(self) -> None:
        from v2.training.reward_scaling import load_reward_scale_document

        mutations = (
            ("train_raw_interval_costs_cny", [10.0, 20.0, 31.0]),
            ("reference_action_id", "w_1_1_8"),
            (
                "input_manifest_sha256",
                {"power": "d" * 64, "ais": "b" * 64, "modes": "c" * 64},
            ),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                document = self._document()
                document[key] = value
                with self.assertRaises(ValueError):
                    load_reward_scale_document(document)

        missing = self._document()
        del missing["train_raw_interval_costs_cny"]
        with self.assertRaises((KeyError, ValueError)):
            load_reward_scale_document(missing)

        with self.assertRaises(ValueError):
            load_reward_scale_document(
                self._document(),
                expected_manifest_hashes={
                    "power": "f" * 64,
                    "ais": "b" * 64,
                    "modes": "c" * 64,
                },
            )

    def test_backend_retains_detached_ledger_per_physical_interval(self) -> None:
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.envs.formal_episode import FormalEpisodeBackend

        class Solver:
            def solve(self, **kwargs):
                soc = float(kwargs["current_soc"])

                class Command:
                    p_fc_kw = 0.0
                    p_batt_bus_kw = 0.0
                    predicted_next_soc = soc

                class Plan:
                    @staticmethod
                    def first_command():
                        return Command()

                return Plan()

        backend = FormalEpisodeBackend(
            load_kw=np.zeros(3),
            speed_kn=np.zeros(3),
            fc_power_kw=np.zeros(3),
            battery_bus_kw=np.zeros(3),
            operating_mode=("onboard",) * 3,
            mpc=Solver(),
        )
        weights = FINAL_DQN_ACTION_CATALOG[0].to_mpc_weights()
        for _ in range(3):
            backend.execute_mpc_step(weights)

        first_read = backend.interval_ledgers
        self.assertEqual(len(first_read), 3)
        self.assertEqual(first_read[0].total_cost_cny, 0.0)
        object.__setattr__(first_read[0], "h2_cost_cny", 999.0)
        self.assertEqual(backend.interval_ledgers[0].total_cost_cny, 0.0)

    def test_generator_uses_train_only_and_preserves_zero_cost_intervals(self) -> None:
        from v2.economics import RawCnyIntervalLedger
        from v2.main import run_reward_scale_calibration as runner

        dataset = mock.Mock()
        dataset.load_train.return_value = (
            SimpleNamespace(sample_id="train-a"),
            SimpleNamespace(sample_id="train-b"),
        )
        dataset.opened_test_payloads = 0

        backends = iter(
            (
                SimpleNamespace(
                    interval_ledgers=(
                        RawCnyIntervalLedger(0.0, 0.0, 0.0, 0.0),
                        RawCnyIntervalLedger(2.0, 0.0, 0.0, 0.0),
                    )
                ),
                SimpleNamespace(
                    interval_ledgers=(RawCnyIntervalLedger(4.0, 0.0, 0.0, 0.0),)
                ),
            )
        )

        class Environment:
            def reset(self):
                return (0.0,) * 90

            def step(self, action_id):
                return SimpleNamespace(done=True, failure_penalty_score=50_000.0)

        def environment(_episode):
            return next(backends), Environment()

        args = argparse.Namespace(
            power_root=Path("power"),
            ais_root=Path("ais"),
            mode_root=Path("modes"),
            output=Path("unused.json"),
        )
        hashes = {"power": "a" * 64, "ais": "b" * 64, "modes": "c" * 64}
        with mock.patch.object(
            runner.FormalTrainingDataset,
            "open",
            return_value=dataset,
        ), mock.patch.object(runner, "_environment", side_effect=environment), mock.patch.object(
            runner,
            "_manifest_hashes",
            return_value=hashes,
        ):
            document = runner.generate_calibration(args)

        dataset.load_train.assert_called_once_with()
        self.assertEqual(document["train_raw_interval_costs_cny"], [0.0, 2.0, 4.0])
        self.assertEqual(document["scale_cny"], 2.0)
        self.assertEqual(document["test_payloads_opened"], 0)
        self.assertNotEqual(document["scale_cny"], 50_000.0)

    def test_atomic_writer_replaces_destination_and_leaves_no_temp_file(self) -> None:
        from v2.main.run_reward_scale_calibration import _write_atomic

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "scale.json"
            _write_atomic(destination, self._document())
            self.assertTrue(destination.is_file())
            self.assertEqual(tuple(destination.parent.glob("*.tmp-*")), ())


if __name__ == "__main__":
    unittest.main()
