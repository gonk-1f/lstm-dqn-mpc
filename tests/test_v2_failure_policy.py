from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestV2FailurePolicy(unittest.TestCase):
    def test_formal_policy_is_a_separate_train_only_score(self) -> None:
        from v2.failure_policy import (
            FAILURE_CALIBRATION_ID,
            FAILURE_EVIDENCE_STATUS,
            FORMAL_FAILURE_KIND,
            FORMAL_FAILURE_POLICY,
            FORMAL_TERMINAL_FAILURE_PENALTY_SCORE,
            FormalFailurePolicy,
        )

        self.assertIs(type(FORMAL_FAILURE_POLICY), FormalFailurePolicy)
        self.assertEqual(FORMAL_TERMINAL_FAILURE_PENALTY_SCORE, 50_000.0)
        self.assertEqual(FORMAL_FAILURE_POLICY.penalty_score, 50_000.0)
        self.assertEqual(FORMAL_FAILURE_POLICY.failure_kind, FORMAL_FAILURE_KIND)
        self.assertEqual(FORMAL_FAILURE_KIND, "physical_mpc_infeasibility")
        self.assertEqual(
            FORMAL_FAILURE_POLICY.evidence_status,
            "DERIVED_TRAIN_ONLY / PROJECT_DESIGN",
        )
        self.assertEqual(FORMAL_FAILURE_POLICY.evidence_status, FAILURE_EVIDENCE_STATUS)
        self.assertEqual(FORMAL_FAILURE_POLICY.calibration_id, FAILURE_CALIBRATION_ID)
        self.assertEqual(
            FormalFailurePolicy.formal_baseline(), FORMAL_FAILURE_POLICY
        )

    def test_policy_rejects_noncanonical_or_nonfinite_values(self) -> None:
        from v2.failure_policy import FormalFailurePolicy

        canonical = (
            50_000.0,
            "physical_mpc_infeasibility",
            "DERIVED_TRAIN_ONLY / PROJECT_DESIGN",
            "v2_train_w_8_1_1_v1",
        )
        invalid = (
            (1.0, *canonical[1:]),
            (math.inf, *canonical[1:]),
            (50_000, *canonical[1:]),
            (canonical[0], "other", *canonical[2:]),
            (canonical[0], canonical[1], "MEASURED", canonical[3]),
            (*canonical[:3], "other"),
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                FormalFailurePolicy(*values)


if __name__ == "__main__":
    unittest.main()
