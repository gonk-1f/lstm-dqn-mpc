from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


class TestValidationSelectionCli(unittest.TestCase):
    def test_parser_defaults_are_the_frozen_formal_selection(self) -> None:
        from v2.main.select_formal_dqn_checkpoint import _parser

        args = _parser().parse_args([])
        self.assertEqual(args.checkpoint_dir.name, "v2_formal_dqn_v3")
        self.assertEqual(args.output_dir.name, "v2_formal_dqn_selection")
        self.assertEqual(args.first_round, 1)
        self.assertEqual(args.last_round, 40)
        self.assertEqual(args.device, "cpu")

    def test_visits_all_rounds_in_order_and_copies_validation_winner(self) -> None:
        from v2.evaluation.checkpoint_selection import AuthenticatedCheckpoint
        from v2.main import select_formal_dqn_checkpoint as module

        class Dataset:
            opened_test_payloads = 0

            def __init__(self) -> None:
                self.validation_calls = 0
                self.id_calls = 0

            def load_validation(self):
                self.validation_calls += 1
                return ("validation_episode",)

            def split_episode_ids(self, split):
                self.id_calls += 1
                self.asserted_split = split
                return ("train_a", "train_b")

        class Agent:
            def __init__(self, config, *, seed, device) -> None:
                self.round_index = None

            def greedy_action(self, state):
                return 0

        class Schedule:
            def __init__(self, episode_ids, *, seed) -> None:
                self.episode_ids = tuple(episode_ids)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            authenticated = []
            for round_index in range(1, 41):
                path = checkpoint_dir / f"round_{round_index:03d}.pt"
                path.write_bytes(f"checkpoint-{round_index}".encode("ascii"))
                authenticated.append(
                    AuthenticatedCheckpoint(
                        round_index,
                        path,
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            dataset = Dataset()
            loaded: list[int] = []

            def load_checkpoint(path, *, agent, schedule):
                round_index = int(Path(path).stem.removeprefix("round_"))
                loaded.append(round_index)
                agent.round_index = round_index
                return SimpleNamespace(round_index=round_index, episode_position=0)

            def evaluate(*, episodes, policy):
                self.assertEqual(episodes, ("validation_episode",))
                round_index = policy.agent.round_index
                raw = 1.0 if round_index == 7 else float(100 + round_index)
                return SimpleNamespace(
                    completed_episodes=8,
                    failed_episodes=0,
                    failure_penalty_score=0.0,
                    raw_economic_cost_cny=raw,
                    learning_reward=-raw,
                )

            output_dir = root / "selection"
            stdout = io.StringIO()
            with (
                mock.patch.object(module.FormalTrainingDataset, "open", return_value=dataset),
                mock.patch.object(
                    module,
                    "authenticate_checkpoint_candidates",
                    return_value=tuple(authenticated),
                ),
                mock.patch.object(module, "DqnAgent", Agent),
                mock.patch.object(module, "EpisodeShuffleSchedule", Schedule),
                mock.patch.object(module, "load_checkpoint", side_effect=load_checkpoint),
                mock.patch.object(module, "evaluate_formal_policy", side_effect=evaluate),
                mock.patch.object(module, "_input_manifest_hashes", return_value={
                    "power": "1" * 64,
                    "ais": "2" * 64,
                    "modes": "3" * 64,
                }),
                redirect_stdout(stdout),
            ):
                code = module.main(
                    [
                        "--checkpoint-dir", str(checkpoint_dir),
                        "--output-dir", str(output_dir),
                        "--device", "cpu",
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(loaded, list(range(1, 41)))
            self.assertEqual(dataset.validation_calls, 1)
            self.assertEqual(dataset.id_calls, 1)
            self.assertEqual(dataset.asserted_split, "train")
            self.assertEqual(dataset.opened_test_payloads, 0)
            self.assertEqual(
                (output_dir / "best_validation.pt").read_bytes(),
                (checkpoint_dir / "round_007.pt").read_bytes(),
            )
            self.assertIn("selected_round=7", stdout.getvalue())
            self.assertIn("test_payloads_opened=0", stdout.getvalue())

    def test_existing_output_fails_before_dataset_or_checkpoint_access(self) -> None:
        from v2.main import select_formal_dqn_checkpoint as module

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "selection"
            output_dir.mkdir()
            with (
                mock.patch.object(module.FormalTrainingDataset, "open") as open_dataset,
                mock.patch.object(module, "authenticate_checkpoint_candidates") as authenticate,
                self.assertRaises(FileExistsError),
            ):
                module.main(["--output-dir", str(output_dir)])
            open_dataset.assert_not_called()
            authenticate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
