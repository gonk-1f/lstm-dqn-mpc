from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from diagnose_four_action_pretraining_soc_deadband import (  # noqa: E402
    REQUIRED_OUTPUT_FILENAMES,
    soc_deadband_violation,
)


class FourActionPretrainingSocDeadbandTests(unittest.TestCase):
    def test_diagnostic_declares_the_required_compact_outputs(self) -> None:
        self.assertEqual(
            REQUIRED_OUTPUT_FILENAMES,
            {
                "summary": "summary.json",
                "state_probe": "state_probe.csv",
                "pairwise": "pairwise_action_differences.csv",
                "regime": "regime_summary.csv",
                "rollout": "fixed_action_rollout_summary.csv",
            },
        )

    def test_diagnostic_uses_the_closed_soc_deadband(self) -> None:
        self.assertEqual(soc_deadband_violation(0.50), 0.0)
        self.assertEqual(soc_deadband_violation(0.55), 0.0)
        self.assertEqual(soc_deadband_violation(0.60), 0.0)
        self.assertAlmostEqual(soc_deadband_violation(0.49), 0.01)
        self.assertAlmostEqual(soc_deadband_violation(0.61), 0.01)


if __name__ == "__main__":
    unittest.main()
