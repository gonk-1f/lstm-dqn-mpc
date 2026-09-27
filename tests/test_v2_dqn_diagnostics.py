from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class TestDqnDiagnostics(unittest.TestCase):
    def test_q_diagnostics_separate_common_mode_advantage_and_margin(self) -> None:
        from v2.training.diagnostics import summarize_q_values

        result = summarize_q_values(
            np.asarray([[100.0, 101.0, 99.0], [5.0, 5.0, 5.0]])
        )
        self.assertEqual(result.common_mode_mean, 52.5)
        self.assertAlmostEqual(result.centered_advantage_std, np.sqrt(2.0 / 6.0))
        self.assertEqual(result.top_two_margin_p50, 0.5)

    def test_action_summary_is_descriptive_and_canonically_ordered(self) -> None:
        from v2.dqn.action_space import FINAL_DQN_ACTION_CATALOG
        from v2.training.diagnostics import summarize_actions

        result = summarize_actions((2, 0, 1, 0), action_dim=36)
        self.assertEqual(result.unique_action_count, 3)
        self.assertEqual(result.max_action_share, 0.5)
        self.assertGreater(result.shannon_entropy, 0.0)
        self.assertFalse(hasattr(result, "selection_score"))
        self.assertEqual(
            result.action_counts,
            tuple(
                (FINAL_DQN_ACTION_CATALOG[index].action_id, count)
                for index, count in ((0, 2), (1, 1), (2, 1))
            ),
        )

    def test_optimization_summary_reports_td_percentiles_and_gradient(self) -> None:
        from v2.training.diagnostics import DqnOptimizationDiagnostics

        result = DqnOptimizationDiagnostics.from_tensors(
            loss=torch.tensor(1.25),
            td_error=torch.tensor([-1.0, 2.0, -10.0, 4.0]),
            q_values=torch.tensor([[3.0, 1.0], [5.0, 5.0]]),
            gradient_norm=torch.tensor(5.0),
        )
        self.assertEqual(result.loss, 1.25)
        self.assertEqual(result.td_abs_p50, 3.0)
        self.assertAlmostEqual(result.td_abs_p95, 9.1)
        self.assertEqual(result.td_abs_max, 10.0)
        self.assertEqual(result.gradient_norm_preclip, 5.0)
        self.assertEqual(result.q_common_mean, 3.5)
        self.assertEqual(result.q_margin_p50, 1.0)

    def test_rejects_empty_nonfinite_and_invalid_action_inputs(self) -> None:
        from v2.training.diagnostics import summarize_actions, summarize_q_values

        for values in (
            np.empty((0, 3)),
            np.empty((2, 0)),
            np.asarray([[1.0, np.nan]]),
            np.asarray([[1.0, np.inf]]),
            np.asarray([1.0, 2.0]),
        ):
            with self.subTest(shape=values.shape):
                with self.assertRaises(ValueError):
                    summarize_q_values(values)
        for actions in ((), (True,), (-1,), (36,), (1.0,)):
            with self.subTest(actions=actions):
                with self.assertRaises((TypeError, ValueError)):
                    summarize_actions(actions, action_dim=36)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
