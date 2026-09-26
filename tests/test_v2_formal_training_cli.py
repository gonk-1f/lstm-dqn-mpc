from __future__ import annotations

from contextlib import redirect_stdout
from argparse import Namespace
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


class TestV2FormalTrainingCli(unittest.TestCase):
    def test_default_rounds_match_frozen_formal_baseline(self) -> None:
        from v2.main.train_formal_dqn import _parser

        args = _parser().parse_args([])
        self.assertEqual(args.rounds, 40)
        self.assertEqual(args.output_dir.name, "v2_formal_dqn_v3")

    def test_preflight_and_smoke_are_bounded_and_never_start_training(self) -> None:
        from v2.main.train_formal_dqn import main

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--preflight-only"])
        self.assertEqual(code, 0)
        self.assertIn("FORMAL_TRAINING=GO", output.getvalue())
        self.assertIn("train_macro_transitions=3721", output.getvalue())
        self.assertIn("train_unresolved_steps=0", output.getvalue())
        self.assertIn("validation_unresolved_steps=0", output.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(["--smoke-only", "--output-dir", directory])
            self.assertEqual(code, 0)
            rendered = output.getvalue()
            self.assertIn("SMOKE=PASS", rendered)
            self.assertIn("epsilon=", rendered)
            self.assertIn("greedy_rate=", rendered)
            self.assertIn("paused_shore_steps=", rendered)
            self.assertFalse((Path(directory) / "latest.pt").exists())

    def test_modes_are_mutually_exclusive(self) -> None:
        from v2.main.train_formal_dqn import main

        with self.assertRaises(SystemExit):
            main(["--preflight-only", "--smoke-only"])

    def test_train_records_failed_transition_and_continues_next_episode(self) -> None:
        from v2.data.formal_training_dataset import FormalEpisode
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND
        from v2.main import train_formal_dqn as module

        def episode(sample_id: str) -> FormalEpisode:
            return FormalEpisode(
                sample_id,
                sample_id,
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

        episodes = (episode("failed"), episode("completed"))
        created: list[str] = []

        class Replay:
            def __init__(self) -> None:
                self.rows = []

            def append(self, *values) -> None:
                self.rows.append(values)

            def __len__(self) -> int:
                return len(self.rows)

        class Agent:
            instance = None

            def __init__(self, config, *, seed, device) -> None:
                self.config = config
                self.replay = Replay()
                self.optimizer_steps = 0
                Agent.instance = self

            def select_action(self, state, *, epsilon) -> int:
                return 0

        class Schedule:
            def __init__(self, episode_ids, *, seed) -> None:
                self.episode_ids = tuple(episode_ids)

            def next_round(self):
                return self.episode_ids

        class Backend:
            def __init__(self) -> None:
                self.mode_counts = {mode: 0 for mode in OperatingMode}
                self.index = 1
                self.load_kw = np.asarray([100.0])
                self.previous_fc_kw = 0.0

        class Environment:
            def __init__(self, sample_id: str) -> None:
                self.sample_id = sample_id

            def reset(self):
                return (0.0,) * 8

            def step(self, action_id):
                failed = self.sample_id == "failed"
                ledger = RawCnyIntervalLedger(10.0, 0.0, 0.0, 0.0)
                return MacroTransition(
                    state=(0.0,) * 8,
                    action=FINAL_DQN_ACTION_CATALOG[0],
                    learning_reward=-50_010.0 if failed else -10.0,
                    next_state=(0.0,) * 8,
                    done=True,
                    executed_mpc_steps=0 if failed else 1,
                    ledger=ledger,
                    failure_penalty_score=50_000.0 if failed else 0.0,
                    failure_kind=FORMAL_FAILURE_KIND if failed else None,
                )

        def environment(value):
            created.append(value.sample_id)
            return Backend(), Environment(value.sample_id)

        args = Namespace(
            rounds=1,
            seed=42,
            device="cpu",
            output_dir=Path(tempfile.mkdtemp()),
            resume=None,
            log_every=1,
        )
        output = io.StringIO()
        with (
            mock.patch.object(module, "DqnAgent", Agent),
            mock.patch.object(module, "EpisodeShuffleSchedule", Schedule),
            mock.patch.object(module, "_environment", side_effect=environment),
            mock.patch.object(module, "save_checkpoint"),
            mock.patch.object(
                module,
                "_evaluate_validation",
                return_value=module.ValidationSummary.empty(),
            ),
            redirect_stdout(output),
        ):
            module._train(args, episodes, ())

        self.assertEqual(created, ["failed", "completed"])
        self.assertEqual(len(Agent.instance.replay), 2)
        self.assertTrue(Agent.instance.replay.rows[0][4])
        self.assertEqual(Agent.instance.replay.rows[0][2], -50_010.0)
        rendered = output.getvalue()
        self.assertIn("completed_episodes=1", rendered)
        self.assertIn("failed_episodes=1", rendered)
        self.assertIn("completion_rate=0.500000", rendered)

    def test_validation_uses_greedy_policy_without_learning_and_reports_score_parts(self) -> None:
        from v2.data.formal_training_dataset import FormalEpisode
        from v2.data.supervisory_rules import OperatingMode
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.economics import RawCnyIntervalLedger
        from v2.envs.multirate_weight_env import MacroTransition
        from v2.failure_policy import FORMAL_FAILURE_KIND
        from v2.main import train_formal_dqn as module

        episode = FormalEpisode(
            "p", "v", "validation", (pd.Timestamp("2024-01-01"),),
            np.asarray([0.0]), np.asarray([100.0]), np.asarray([3.0]), ("raw",),
            np.asarray([80.0]), np.asarray([20.0]),
            (OperatingMode.ONBOARD.value,), ("test",),
        )

        class Agent:
            def __init__(self) -> None:
                self.greedy_calls = 0
                self.optimizer_steps = 7
                self.replay = []

            def greedy_action(self, state) -> int:
                self.greedy_calls += 1
                return 0

        class Environment:
            def reset(self):
                return (0.0,) * 8

            def step(self, action_id):
                ledger = RawCnyIntervalLedger(12.0, 0.0, 0.0, 0.0)
                return MacroTransition(
                    state=(0.0,) * 8,
                    action=FINAL_DQN_ACTION_CATALOG[0],
                    learning_reward=-50_012.0,
                    next_state=(0.0,) * 8,
                    done=True,
                    executed_mpc_steps=0,
                    ledger=ledger,
                    failure_penalty_score=50_000.0,
                    failure_kind=FORMAL_FAILURE_KIND,
                )

        agent = Agent()
        with mock.patch.object(module, "_environment", return_value=(object(), Environment())):
            summary = module._evaluate_validation(agent, (episode,))

        self.assertEqual(agent.greedy_calls, 1)
        self.assertEqual(agent.optimizer_steps, 7)
        self.assertEqual(agent.replay, [])
        self.assertEqual(summary.raw_economic_cost_cny, 12.0)
        self.assertEqual(summary.failure_penalty_score, 50_000.0)
        self.assertEqual(summary.learning_reward, -50_012.0)
        self.assertEqual(summary.failed_episodes, 1)
        self.assertEqual(summary.completed_episodes, 0)
        self.assertEqual(summary.completion_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
