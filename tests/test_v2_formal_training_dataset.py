from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATASET = ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2"
AIS = ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_ais"
MODES = ROOT / "data" / "processed" / "operating_dataset_zero_boundary_v2_modes"


class TestFormalTrainingDataset(unittest.TestCase):
    def test_episode_macro_count_excludes_shore_pause_intervals(self) -> None:
        from v2.data.formal_training_dataset import FormalEpisode

        modes = (
            *("onboard",) * 4,
            *("shore_pending",) * 2,
            *("shore_charging",) * 8,
            *("onboard",) * 4,
        )
        size = len(modes)
        episode = FormalEpisode(
            parent="fixture",
            sample_id="fixture_001",
            split="train",
            timestamp=tuple(pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=30 * i) for i in range(size)),
            time_s=np.arange(size, dtype=float) * 30.0,
            load_kw=np.zeros(size),
            speed_kn=np.zeros(size),
            speed_provenance=("RAW_AIS",) * size,
            fc_power_kw=np.zeros(size),
            battery_bus_kw=np.zeros(size),
            operating_mode=modes,
            mode_reason=("fixture",) * size,
        )

        self.assertEqual(episode.macro_transition_count, 2)

    def test_authenticates_current_train_and_ais_without_opening_test(self) -> None:
        from v2.data.formal_training_dataset import FormalTrainingDataset

        dataset = FormalTrainingDataset.open(DATASET, AIS, MODES)
        self.assertEqual(dataset.split_counts, {"train": 30, "validation": 8, "test": 5})
        self.assertEqual(dataset.train_supervisory_steps, 23_590)
        self.assertEqual(
            dataset.unresolved_mode_counts,
            {"train": 0, "validation": 0, "test": 0},
        )
        self.assertGreater(dataset.train_macro_transitions, 0)
        self.assertLessEqual(dataset.train_macro_transitions, 4_718)
        self.assertEqual(dataset.opened_test_payloads, 0)
        episodes = dataset.load_train()
        self.assertEqual(len(episodes), 30)
        self.assertEqual(sum(item.step_count for item in episodes), 23_590)
        self.assertTrue(all(item.speed_kn.shape == item.load_kw.shape for item in episodes))
        self.assertTrue(all(item.fc_power_kw.shape == item.load_kw.shape for item in episodes))
        self.assertTrue(all(item.battery_bus_kw.shape == item.load_kw.shape for item in episodes))
        self.assertTrue(all(len(item.operating_mode) == item.step_count for item in episodes))
        self.assertTrue(
            {mode for item in episodes for mode in item.operating_mode}.issuperset(
                {"onboard", "shore_pending", "shore_charging"}
            )
        )
        self.assertTrue(any((item.load_kw < -1.0).any() for item in episodes))
        self.assertEqual(dataset.opened_test_payloads, 0)
        with self.assertRaisesRegex(PermissionError, "Test"):
            dataset.load_split("test")

    def test_fixed_seed_shuffle_is_per_round_and_resumable(self) -> None:
        from v2.training.schedule import EpisodeShuffleSchedule

        identifiers = tuple(f"e{index:02d}" for index in range(30))
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
