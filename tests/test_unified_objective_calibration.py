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


from calibrate_unified_mpc_actions_train import (  # noqa: E402
    CANDIDATES,
    SCALE_REFERENCE_ACTION_IDS,
    WINDOWS,
    _validate_actions,
)
from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402
from mpc_solvers.formal_config import build_formal_mpc_config  # noqa: E402


class UnifiedObjectiveCalibrationTests(unittest.TestCase):
    def test_candidate_and_final_actions_are_nonnegative_and_sum_to_one(self) -> None:
        _validate_actions(CANDIDATES)
        _validate_actions(DQN_MPC_WEIGHT_ACTIONS)

    def test_scale_audit_uses_four_predeclared_semantic_endpoints(self) -> None:
        self.assertEqual(len(SCALE_REFERENCE_ACTION_IDS), 4)
        self.assertEqual(len(set(SCALE_REFERENCE_ACTION_IDS)), 4)

    def test_calibration_windows_are_train_only(self) -> None:
        self.assertGreaterEqual(len(WINDOWS), 7)
        self.assertTrue(all(segment_id.startswith("train_") for _, segment_id, *_ in WINDOWS))

    def test_continuous_soc_reference_configuration(self) -> None:
        config = build_formal_mpc_config()
        self.assertEqual(config.soc_reference, 0.55)
        self.assertEqual(config.soc_scale, 0.05)
        self.assertEqual(config.q_terminal_soc, 0.0)


if __name__ == "__main__":
    unittest.main()
