from __future__ import annotations

import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN = SRC / "main"
for path in (SRC, MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank  # noqa: E402
from mpc_solvers.formal_config import (  # noqa: E402
    SOC_SOFT_MAX,
    SOC_SOFT_MIN,
    SOC_SOFT_SCALE,
    build_formal_mpc_config,
)
from mpc_solvers.mpc_qp_formulation import build_qp_problem  # noqa: E402


class MpcSocDeadbandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = build_formal_mpc_config()

    def _soc_only_problem(self):
        config = replace(
            self.base_config,
            q_h2=0.0,
            q_batt=0.0,
            q_soc=1.0,
            q_fc_var=0.0,
        )
        return config, build_qp_problem(
            config,
            load_forecast_kw=np.zeros(config.horizon),
            current_soc=0.55,
            prev_fc_kw=0.0,
            soc_reference=0.55,
        )

    @staticmethod
    def _objective(problem, values: np.ndarray) -> float:
        return float(0.5 * values @ problem.P @ values + problem.q @ values)

    def _deadband_objective(self, soc: float) -> float:
        config, problem = self._soc_only_problem()
        values = np.zeros(problem.P.shape[0], dtype=float)
        values[2 * config.horizon : 3 * config.horizon + 1] = soc
        violation = max(0.0, SOC_SOFT_MIN - soc, soc - SOC_SOFT_MAX)
        values[3 * config.horizon + 1 :] = violation
        return self._objective(problem, values)

    def test_deadband_cost_is_zero_inside_closed_working_range(self) -> None:
        for soc in (0.50, 0.55, 0.60):
            with self.subTest(soc=soc):
                self.assertAlmostEqual(self._deadband_objective(soc), 0.0, places=12)

    def test_deadband_cost_is_positive_outside_working_range(self) -> None:
        self.assertGreater(self._deadband_objective(0.49), 0.0)
        self.assertGreater(self._deadband_objective(0.61), 0.0)

    def test_equal_distance_from_either_boundary_has_equal_normalized_cost(self) -> None:
        self.assertAlmostEqual(
            self._deadband_objective(0.49),
            self._deadband_objective(0.61),
            places=12,
        )

    def test_actions_only_vary_the_four_objective_weights(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(DQN_MPC_WEIGHT_ACTIONS[0])),
            ("action_id", "q_h2", "q_batt", "q_soc", "q_fc_var", "name"),
        )
        self.assertEqual(
            DQN_MPC_WEIGHT_ACTIONS[2].as_tuple(),
            (0.25, 0.45, 200.0, 8.0),
        )

    def test_all_actions_share_convex_deadband_qp_and_solve(self) -> None:
        bank = MpcWeightSolverBank(self.base_config)
        expected_a = bank._entries[0].A.toarray()
        for action in DQN_MPC_WEIGHT_ACTIONS:
            with self.subTest(action=action.name):
                entry = bank._entries[action.action_id]
                self.assertEqual(entry.config.soc_soft_min, SOC_SOFT_MIN)
                self.assertEqual(entry.config.soc_soft_max, SOC_SOFT_MAX)
                self.assertEqual(entry.config.soc_band, SOC_SOFT_SCALE)
                self.assertEqual(entry.P.shape, (25, 25))
                np.testing.assert_allclose(entry.A.toarray(), expected_a)
                result, _ = bank.solve(
                    action_id=action.action_id,
                    load_forecast_kw=np.full(self.base_config.horizon, 200.0),
                    current_soc=0.49,
                    prev_fc_kw=180.0,
                    soc_reference=0.55,
                )
                self.assertTrue(str(result.info.status).lower().startswith("solved"))

    def test_frozen_physical_configuration_is_unchanged(self) -> None:
        config = self.base_config
        self.assertEqual(config.horizon, 6)
        self.assertEqual(config.dt_seconds, 1.0)
        self.assertEqual(config.fuel_cell_max_kw, 600.0)
        self.assertEqual(config.fuel_cell_ramp_rate_kw_per_s, 48.0)
        self.assertEqual(config.battery_capacity_kwh, 624.0)
        self.assertEqual((config.soc_min, config.soc_max), (0.2, 0.8))


if __name__ == "__main__":
    unittest.main()
