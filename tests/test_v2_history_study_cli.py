from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestHistoryStudyCli(unittest.TestCase):
    @staticmethod
    def _calibration():
        from v2.analysis.action_screening import DataSplit, DatasetProvenance
        from v2.economics import calibrate_reward_scale

        return calibrate_reward_scale(
            (10.0, 20.0, 30.0),
            provenance=DatasetProvenance(
                "operating_dataset_zero_boundary_v2",
                "sha256:history-cli-fixture",
                DataSplit.TRAIN,
            ),
            audit_id="history-cli-fixture",
            reason="verify replay reward path",
        )

    def test_defaults_are_bounded_h1_pilot_and_experiment_output(self) -> None:
        from v2.main.train_history_dqn_study import _parser, _resolve_output_dir

        args = _parser().parse_args([])
        self.assertEqual(args.experiment, "H1")
        self.assertEqual(args.rounds, 10)
        self.assertEqual(args.seed, 42)
        self.assertEqual(_resolve_output_dir(args).name, "H1")

    def test_output_guard_requires_new_directory_or_exact_latest_resume(self) -> None:
        from v2.main.train_history_dqn_study import _guard_output_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "H2"
            _guard_output_directory(root, None)
            root.mkdir()
            (root / "other.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _guard_output_directory(root, None)
            with self.assertRaises(ValueError):
                _guard_output_directory(root, root / "other.txt")
            latest = root / "latest.pt"
            latest.write_bytes(b"checkpoint")
            _guard_output_directory(root, latest)

    def test_h1_rejects_scale_argument_and_scaled_profile_requires_document(self) -> None:
        from v2.main import train_history_dqn_study as module

        h1 = Namespace(experiment="H1", reward_scale=Path("scale.json"))
        with self.assertRaises(ValueError):
            module._load_profile(h1)
        h2 = Namespace(experiment="H2", reward_scale=Path("missing.json"))
        with self.assertRaises(FileNotFoundError):
            module._load_profile(h2)

    def test_profile_reward_enters_replay_while_raw_parts_remain_in_logs(self) -> None:
        from v2.data.formal_training_dataset import FormalEpisode
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND
        from v2.main import train_formal_dqn as trainer
        from v2.training.experiments import history_study_profile

        episode = FormalEpisode(
            "p",
            "train-a",
            "train",
            (pd.Timestamp("2024-01-01"),),
            np.asarray([0.0]),
            np.asarray([100.0]),
            np.asarray([3.0]),
            ("raw",),
            np.asarray([80.0]),
            np.asarray([20.0]),
            (OperatingMode.ONBOARD.value,),
            ("test",),
        )

        class Replay:
            def __init__(self):
                self.rows = []

            def append(self, *values):
                self.rows.append(values)

            def __len__(self):
                return len(self.rows)

        class Agent:
            instance = None

            def __init__(self, config, *, seed, device):
                self.config = config
                self.replay = Replay()
                self.optimizer_steps = 0
                Agent.instance = self

            def select_action(self, state, *, epsilon):
                return 0

        class Schedule:
            def __init__(self, episode_ids, *, seed):
                self.ids = tuple(episode_ids)

            def next_round(self):
                return self.ids

        class Backend:
            mode_counts = {mode: 0 for mode in OperatingMode}
            index = 1
            load_kw = np.asarray([100.0])
            previous_fc_kw = 0.0

        class Environment:
            def reset(self):
                return (0.0,) * 90

            def step(self, action_id):
                state = (0.0,) * 90
                return MacroTransition(
                    state=state,
                    action=FINAL_DQN_ACTION_CATALOG[0],
                    learning_reward=-50_030.0,
                    next_state=state,
                    done=True,
                    executed_mpc_steps=1,
                    ledger=RawCnyIntervalLedger(30.0, 0.0, 0.0, 0.0),
                    failure_penalty_score=50_000.0,
                    failure_kind=FORMAL_FAILURE_KIND,
                )

        args = Namespace(
            rounds=1,
            seed=42,
            device="cpu",
            output_dir=Path(tempfile.mkdtemp()),
            resume=None,
            log_every=1,
        )
        calibration = self._calibration()
        cases = (
            (history_study_profile("H1", None), None, -50_030.0),
            (history_study_profile("H2", calibration), calibration, -2501.5),
        )
        for profile, scale, expected in cases:
            with self.subTest(experiment=profile.experiment_id):
                output = io.StringIO()
                with (
                    mock.patch.object(trainer, "DqnAgent", Agent),
                    mock.patch.object(trainer, "EpisodeShuffleSchedule", Schedule),
                    mock.patch.object(
                        trainer,
                        "_environment",
                        return_value=(Backend(), Environment()),
                    ),
                    mock.patch.object(trainer, "save_checkpoint"),
                    mock.patch.object(
                        trainer,
                        "_evaluate_validation",
                        return_value=trainer.ValidationSummary.empty(),
                    ),
                    redirect_stdout(output),
                ):
                    trainer._train(
                        args,
                        (episode,),
                        (),
                        profile=profile,
                        calibration=scale,
                    )
                self.assertEqual(Agent.instance.replay.rows[0][2], expected)
                rendered = output.getvalue()
                self.assertIn("raw_economic_cost_cny=30.000000000", rendered)
                self.assertIn("failure_penalty_score=50000.000000000", rendered)
                self.assertIn(f"replay_reward={expected:.9f}", rendered)

if __name__ == "__main__":
    unittest.main()
