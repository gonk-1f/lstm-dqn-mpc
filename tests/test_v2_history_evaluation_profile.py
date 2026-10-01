from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest


HASHES = {"power": "a" * 64, "ais": "b" * 64, "modes": "c" * 64}


def _scale_document() -> dict[str, object]:
    from v2.analysis.action_screening import DataSplit, DatasetProvenance
    from v2.economics import calibrate_reward_scale
    from v2.training.reward_scaling import build_reward_scale_document

    calibration = calibrate_reward_scale(
        (10.0, 20.0, 30.0),
        provenance=DatasetProvenance(
            "operating_dataset_zero_boundary_v2",
            "sha256:h4-evaluation-fixture",
            DataSplit.TRAIN,
        ),
        audit_id="h4-evaluation-fixture",
        reason="verify profile-bound evaluation loading",
    )
    return build_reward_scale_document(
        calibration=calibration,
        reference_action_id="w_8_1_1",
        manifest_hashes=HASHES,
        train_segment_ids=("train-a",),
        test_payloads_opened=0,
    )


class TestHistoryEvaluationProfile(unittest.TestCase):
    def test_loads_h4_scaled_identity_and_forty_round_config(self) -> None:
        from v2.main.history_dqn_evaluation import load_evaluation_profile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scale.json"
            path.write_text(json.dumps(_scale_document()), encoding="utf-8")
            profile, config, identity = load_evaluation_profile(
                experiment_id="H4",
                reward_scale_path=path,
                expected_manifest_hashes=HASHES,
                rounds=40,
            )

        self.assertEqual(profile.experiment_id, "H4")
        self.assertEqual(config.experiment_id, "H4")
        self.assertEqual(config.learning_rate, 1.0e-3)
        self.assertEqual(config.reward_mode, "scaled")
        self.assertEqual(config.rounds, 40)
        self.assertEqual(
            identity,
            {
                "experiment_id": "H4",
                "reward_mode": "scaled",
                "reward_scaling_identity": profile.reward_scaling_identity,
            },
        )

    def test_scaled_profile_requires_matching_train_only_scale_document(self) -> None:
        from v2.main.history_dqn_evaluation import load_evaluation_profile

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "scale.json"
            path.write_text(json.dumps(_scale_document()), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_evaluation_profile(
                    experiment_id="H4",
                    reward_scale_path=path,
                    expected_manifest_hashes=HASHES | {"power": "d" * 64},
                    rounds=40,
                )
        with self.assertRaises(FileNotFoundError):
            load_evaluation_profile(
                experiment_id="H4",
                reward_scale_path=None,
                expected_manifest_hashes=HASHES,
                rounds=40,
            )

    def test_selection_and_test_parsers_accept_explicit_h4_profile(self) -> None:
        from v2.main import evaluate_formal_dqn_test, select_formal_dqn_checkpoint

        for parser in (
            select_formal_dqn_checkpoint._parser(),
            evaluate_formal_dqn_test._parser(),
        ):
            args = parser.parse_args(
                [
                    "--experiment",
                    "H4",
                    "--reward-scale",
                    "scale.json",
                    *(
                        ["--confirm-final-test", "FINAL_TEST_ONCE"]
                        if "confirm-final-test" in parser.format_help()
                        else []
                    ),
                ]
            )
            self.assertEqual(args.experiment, "H4")
            self.assertEqual(args.reward_scale, Path("scale.json"))


if __name__ == "__main__":
    unittest.main()
