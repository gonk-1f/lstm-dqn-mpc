from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
MAIN_ROOT = SRC_ROOT / "main"

for path in (SRC_ROOT, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402
from mpc_solvers.formal_config import build_formal_mpc_config  # noqa: E402
from mpc_solvers.mpc_qp_formulation import build_qp_problem  # noqa: E402
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank  # noqa: E402


class A2SocRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = build_formal_mpc_config()
        self.actions = {action.action_id: action for action in DQN_MPC_WEIGHT_ACTIONS}

    def _config_for_action(self, action_id: int):
        action = self.actions[action_id]
        return replace(
            self.base_config,
            q_h2=action.q_h2,
            q_batt=action.q_batt,
            q_soc=action.q_soc,
            q_fc_var=action.q_fc_var,
            soc_penalty_mode=action.soc_penalty_mode,
        )

    def _soc_only_problem(self, action_id: int):
        config = replace(
            self._config_for_action(action_id),
            q_h2=0.0,
            q_batt=0.0,
            q_fc_var=0.0,
        )
        problem = build_qp_problem(
            config,
            load_forecast_kw=np.zeros(config.horizon),
            current_soc=0.55,
            prev_fc_kw=0.0,
            soc_reference=0.55,
        )
        return config, problem

    @staticmethod
    def _objective(problem, values: np.ndarray) -> float:
        return float(0.5 * values @ problem.P @ values + problem.q @ values)

    def test_action_modes_and_numeric_weights_are_exact(self) -> None:
        self.assertEqual(len(DQN_MPC_WEIGHT_ACTIONS), 4)
        self.assertEqual(self.actions[2].as_tuple(), (0.25, 0.45, 200.0, 8.0))
        self.assertEqual(self.actions[2].soc_penalty_mode, "deficit_only")
        for action_id in (0, 1, 3):
            self.assertEqual(self.actions[action_id].soc_penalty_mode, "symmetric")

    def test_a2_deficit_cost_is_positive_below_reference(self) -> None:
        config, problem = self._soc_only_problem(2)
        values = np.zeros(problem.P.shape[0], dtype=float)
        soc_start = 2 * config.horizon
        deficit_start = 3 * config.horizon + 1
        values[soc_start : soc_start + config.horizon + 1] = 0.54
        values[deficit_start:] = 0.01
        expected = config.q_soc * config.horizon * (0.01 / config.soc_band) ** 2
        self.assertAlmostEqual(self._objective(problem, values), expected, places=10)

    def test_a2_deficit_cost_is_zero_at_reference(self) -> None:
        config, problem = self._soc_only_problem(2)
        values = np.zeros(problem.P.shape[0], dtype=float)
        soc_start = 2 * config.horizon
        values[soc_start : soc_start + config.horizon + 1] = 0.55
        self.assertAlmostEqual(self._objective(problem, values), 0.0, places=12)

    def test_a2_deficit_cost_is_zero_above_reference(self) -> None:
        config, problem = self._soc_only_problem(2)
        values = np.zeros(problem.P.shape[0], dtype=float)
        soc_start = 2 * config.horizon
        values[soc_start : soc_start + config.horizon + 1] = 0.56
        self.assertAlmostEqual(self._objective(problem, values), 0.0, places=12)

    def test_symmetric_actions_keep_soc_quadratic_and_shared_qp_structure(self) -> None:
        symmetric_config, symmetric = self._soc_only_problem(0)
        deficit_config, deficit = self._soc_only_problem(2)
        self.assertEqual(symmetric.P.shape, deficit.P.shape)
        self.assertEqual(symmetric.A.shape, deficit.A.shape)
        self.assertTrue(np.array_equal(symmetric.P.indptr, deficit.P.indptr))
        self.assertTrue(np.array_equal(symmetric.P.indices, deficit.P.indices))
        self.assertTrue(np.array_equal(symmetric.A.indptr, deficit.A.indptr))
        self.assertTrue(np.array_equal(symmetric.A.indices, deficit.A.indices))

        soc_indices = np.arange(
            2 * symmetric_config.horizon + 1,
            3 * symmetric_config.horizon + 1,
        )
        deficit_indices = np.arange(
            3 * deficit_config.horizon + 1,
            4 * deficit_config.horizon + 1,
        )
        symmetric_diag = symmetric.P.diagonal()
        deficit_diag = deficit.P.diagonal()
        self.assertTrue(np.all(symmetric_diag[soc_indices] > 0.0))
        self.assertTrue(np.all(symmetric_diag[deficit_indices] == 0.0))
        self.assertTrue(np.all(deficit_diag[soc_indices] == 0.0))
        self.assertTrue(np.all(deficit_diag[deficit_indices] > 0.0))

    def test_a2_qp_remains_convex_and_osqp_solvable(self) -> None:
        bank = MpcWeightSolverBank(self.base_config)
        result, _ = bank.solve(
            action_id=2,
            load_forecast_kw=np.full(self.base_config.horizon, 200.0),
            current_soc=0.54,
            prev_fc_kw=180.0,
            soc_reference=0.55,
        )
        self.assertTrue(str(result.info.status).lower().startswith("solved"))
        self.assertEqual(len(result.x), 4 * self.base_config.horizon + 1)

    def test_physical_configuration_is_unchanged(self) -> None:
        config = self.base_config
        self.assertEqual(config.horizon, 6)
        self.assertEqual(config.dt_seconds, 1.0)
        self.assertEqual(config.fuel_cell_max_kw, 600.0)
        self.assertEqual(config.fuel_cell_ramp_rate_kw_per_s, 48.0)
        self.assertEqual(config.battery_capacity_kwh, 624.0)
        self.assertEqual(config.battery_charge_max_kw, 624.0)
        self.assertEqual(config.battery_discharge_max_kw, 1248.0)
        self.assertEqual((config.soc_min, config.soc_max), (0.2, 0.8))


if __name__ == "__main__":
    unittest.main()
