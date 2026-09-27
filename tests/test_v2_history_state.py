from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestFormalStateHistory(unittest.TestCase):
    def test_history_encodes_oldest_to_newest_with_tail_mask(self) -> None:
        from v2.dqn.history import FormalStateHistory

        history = FormalStateHistory()
        first = tuple(float(index) for index in range(8))
        current = tuple(float(index + 10) for index in range(8))
        history.commit(first)
        encoded = history.encode(current)

        self.assertEqual(len(encoded), 90)
        self.assertEqual(encoded[:64], (0.0,) * 64)
        self.assertEqual(encoded[64:72], first)
        self.assertEqual(encoded[72:80], current)
        self.assertEqual(encoded[80:], (0.0,) * 8 + (1.0, 1.0))

    def test_real_zero_frame_is_distinguished_by_mask(self) -> None:
        from v2.dqn.history import FormalStateHistory

        encoded = FormalStateHistory().encode((0.0,) * 8)
        self.assertEqual(encoded[:80], (0.0,) * 80)
        self.assertEqual(encoded[80:], (0.0,) * 9 + (1.0,))

    def test_encode_is_side_effect_free_and_reset_discards_history(self) -> None:
        from v2.dqn.history import FormalStateHistory

        history = FormalStateHistory()
        frame = (0.0,) * 8
        self.assertEqual(history.encode(frame), history.encode(frame))
        self.assertEqual(history.committed_count, 0)
        history.commit(frame)
        self.assertEqual(history.committed_count, 1)
        history.reset()
        self.assertEqual(history.committed_count, 0)
        self.assertEqual(history.encode(frame)[80:], (0.0,) * 9 + (1.0,))

    def test_retains_only_nine_past_frames_and_preserves_order(self) -> None:
        from v2.dqn.history import FormalStateHistory

        history = FormalStateHistory()
        frames = tuple((float(index),) * 8 for index in range(12))
        for frame in frames:
            history.commit(frame)

        current = (99.0,) * 8
        encoded = history.encode(current)
        expected_frames = frames[-9:] + (current,)
        self.assertEqual(history.committed_count, 9)
        self.assertEqual(
            encoded[:80],
            tuple(value for frame in expected_frames for value in frame),
        )
        self.assertEqual(encoded[80:], (1.0,) * 10)

    def test_rejects_noncanonical_frames(self) -> None:
        from v2.dqn.history import FormalStateHistory

        history = FormalStateHistory()
        bad_values = (
            (0.0,) * 7,
            [0.0] * 8,
            (False,) + (0.0,) * 7,
            (1,) + (0.0,) * 7,
            (math.nan,) + (0.0,) * 7,
            (math.inf,) + (0.0,) * 7,
            (-math.inf,) + (0.0,) * 7,
        )
        for value in bad_values:
            with self.subTest(value=repr(value)):
                with self.assertRaises((TypeError, ValueError)):
                    history.commit(value)  # type: ignore[arg-type]
                with self.assertRaises((TypeError, ValueError)):
                    history.encode(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
