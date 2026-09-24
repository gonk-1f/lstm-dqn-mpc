from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET = ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
AIS = ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_ais"


class TestFormalTrainingDataset(unittest.TestCase):
    def test_authenticates_current_train_and_ais_without_opening_test(self) -> None:
        from v2.data.formal_training_dataset import FormalTrainingDataset

        dataset = FormalTrainingDataset.open(DATASET, AIS)
        self.assertEqual(dataset.split_counts, {"train": 38, "validation": 10, "test": 5})
        self.assertEqual(dataset.train_supervisory_steps, 30_909)
        self.assertEqual(dataset.train_macro_transitions, 6_197)
        self.assertEqual(dataset.opened_test_payloads, 0)
        episodes = dataset.load_train()
        self.assertEqual(len(episodes), 38)
        self.assertEqual(sum(item.step_count for item in episodes), 30_909)
        self.assertTrue(all(item.speed_kn.shape == item.load_kw.shape for item in episodes))
        self.assertTrue(any((item.load_kw < -1.0).any() for item in episodes))
        self.assertEqual(dataset.opened_test_payloads, 0)
        with self.assertRaisesRegex(PermissionError, "Test"):
            dataset.load_split("test")

    def test_fixed_seed_shuffle_is_per_round_and_resumable(self) -> None:
        from v2.training.schedule import EpisodeShuffleSchedule

        identifiers = tuple(f"e{index:02d}" for index in range(38))
        first = EpisodeShuffleSchedule(identifiers, seed=42)
        round_1 = first.next_round()
        saved = first.state_dict()
        round_2 = first.next_round()
        self.assertNotEqual(round_1, round_2)
        self.assertEqual(set(round_1), set(identifiers))

        restored = EpisodeShuffleSchedule(identifiers, seed=999)
        restored.load_state_dict(saved)
        self.assertEqual(restored.next_round(), round_2)
        self.assertEqual(restored.rounds_emitted, 2)


if __name__ == "__main__":
    unittest.main()
