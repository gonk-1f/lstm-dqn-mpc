from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


import run_dqn_mpc_causal_training as formal_training  # noqa: E402
import test_dqn_mpc_causal as validation_artifacts  # noqa: E402
import train_dqn_mpc_mlp as training  # noqa: E402
from dqn.agents.dqn_agent import DQNTrainConfig  # noqa: E402


class ValidationQDiagnosticsTests(unittest.TestCase):
    def _agent(self):
        runtime = training.create_training_runtime(
            DQNTrainConfig(device="cpu", warmup_steps=10)
        )
        with torch.no_grad():
            for parameter in runtime.agent.q_net.parameters():
                parameter.zero_()
            runtime.agent.q_net.layers[-1].bias.copy_(
                torch.tensor([0.1, 0.2, 0.3, 0.4])
            )
        return runtime.agent

    def test_validation_trace_records_greedy_q_values_and_gap(self) -> None:
        result, trace = validation_artifacts.run_test_episode(
            voyage_id="fixture",
            loads_kw=np.asarray([220.0, 225.0, 230.0]),
            base_config=training.build_formal_mpc_config(),
            agent=self._agent(),
        )
        self.assertTrue(result["completed"])
        self.assertTrue(
            {"q_A0", "q_A1", "q_A2", "q_A3", "q_best", "q_second", "q_gap", "action_id"}.issubset(trace.columns)
        )
        self.assertTrue((trace["action_id"] == 3).all())
        np.testing.assert_allclose(trace["q_best"], 0.4)
        np.testing.assert_allclose(trace["q_second"], 0.3)
        np.testing.assert_allclose(trace["q_gap"], 0.1)

    def test_validation_summary_reports_gap_and_regime_action_fractions(self) -> None:
        trace = pd.DataFrame(
            {
                "action_id": [0, 1, 2, 3],
                "q_gap": [0.0, 1.0e-7, 0.2, 0.4],
                "soc_before": [0.49, 0.55, 0.61, 0.55],
                "load_delta_kw": [60.0, -60.0, 0.0, 0.0],
                "current_load_kw": [100.0, 300.0, 500.0, 700.0],
            }
        )
        summary = formal_training.summarize_validation_traces([trace])
        self.assertEqual(summary["action_counts"], {"A0": 1, "A1": 1, "A2": 1, "A3": 1})
        self.assertAlmostEqual(summary["q_gap"]["median"], 0.10000005)
        self.assertEqual(summary["q_gap"]["near_zero_count"], 2)
        self.assertEqual(summary["regime_action_fractions"]["soc"]["soc_lt_0_50"]["A0"], 1.0)
        self.assertEqual(summary["regime_action_fractions"]["transition"]["rapid_rise"]["A0"], 1.0)


if __name__ == "__main__":
    unittest.main()
