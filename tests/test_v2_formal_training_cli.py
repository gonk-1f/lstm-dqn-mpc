from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestV2FormalTrainingCli(unittest.TestCase):
    def test_preflight_and_smoke_are_bounded_and_never_start_training(self) -> None:
        from v2.main.train_formal_dqn import main

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--preflight-only"])
        self.assertEqual(code, 2)
        self.assertIn("FORMAL_TRAINING=NO-GO", output.getvalue())
        self.assertIn("train_macro_transitions=5652", output.getvalue())
        self.assertIn("train_unresolved_steps=685", output.getvalue())
        self.assertIn("validation_unresolved_steps=230", output.getvalue())

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


if __name__ == "__main__":
    unittest.main()
