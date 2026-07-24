from __future__ import annotations

import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"
for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dqn.utils.action_mapper import DQN_MPC_WEIGHT_ACTIONS  # noqa: E402
from mpc_solvers.dqn_mpc_solver_bank import (  # noqa: E402
    MpcWeightSolverBank,
    _OSQP_SETTINGS,
)
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    build_qp_problem,
    resolved_ramp_kw_per_step,
)
from mpc_solvers.osqp_runtime import _try_import_osqp  # noqa: E402

WEIGHT_FIELDS = {"q_h2", "q_batt", "q_soc", "q_fc_var"}
POWER_TOLERANCE_KW = 0.1
SOC_TOLERANCE = 2.0e-5


def candidate_c_base_config() -> QpMpcConfig:
    return QpMpcConfig(
        horizon=6,
        dt_seconds=1.0,
        battery_capacity_kwh=624.0,
        battery_charge_max_kw=624.0,
        battery_discharge_max_kw=1248.0,
        battery_power_ref_kw=624.0,
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=560.0,
        fuel_cell_ramp_rate_kw_per_s=48.0,
        fuel_cell_ramp_kw=None,
        soc_min=0.2,
        soc_max=0.8,
        soc_band=0.05,
        objective_variant="n6_h2_batt_soc_fcvar_normalized_v1",
        q_h2=0.25,
        q_batt=0.40,
        q_soc=12.0,
        q_fc_var=20.0,
        q_ramp=0.0,
        q_terminal_soc=0.0,
    )


class TestDqnMpcSolverBank(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = candidate_c_base_config()
        self.bank = MpcWeightSolverBank(self.base_config)
        self.load_forecast_kw = np.array([200.0, 215.0, 230.0, 245.0, 235.0, 225.0])
        self.current_soc = 0.55
        self.prev_fc_kw = 200.0
        self.soc_reference = 0.55

    def _problem(self, config: QpMpcConfig):
        return build_qp_problem(
            config,
            load_forecast_kw=self.load_forecast_kw,
            current_soc=self.current_soc,
            prev_fc_kw=self.prev_fc_kw,
            soc_reference=self.soc_reference,
            include_diagnostics=False,
        )

    def _assert_physical_constraints(self, config: QpMpcConfig, solution: np.ndarray) -> None:
        horizon = int(config.horizon)
        values = np.asarray(solution, dtype=float).reshape(-1)
        self.assertEqual(len(values), 3 * horizon + 1)
        p_fc = values[:horizon]
        p_batt = values[horizon : 2 * horizon]
        soc = values[2 * horizon :]

        np.testing.assert_allclose(
            p_fc + p_batt,
            self.load_forecast_kw,
            rtol=0.0,
            atol=POWER_TOLERANCE_KW,
        )
        self.assertGreaterEqual(
            float(p_fc.min()),
            float(config.fuel_cell_min_kw) - POWER_TOLERANCE_KW,
        )
        self.assertLessEqual(
            float(p_fc.max()),
            float(config.fuel_cell_max_kw) + POWER_TOLERANCE_KW,
        )
        self.assertGreaterEqual(
            float(p_batt.min()),
            -float(config.battery_charge_max_kw) - POWER_TOLERANCE_KW,
        )
        self.assertLessEqual(
            float(p_batt.max()),
            float(config.battery_discharge_max_kw) + POWER_TOLERANCE_KW,
        )
        self.assertGreaterEqual(float(soc.min()), float(config.soc_min) - SOC_TOLERANCE)
        self.assertLessEqual(float(soc.max()), float(config.soc_max) + SOC_TOLERANCE)
        self.assertAlmostEqual(float(soc[0]), self.current_soc, delta=SOC_TOLERANCE)

        expected_next_soc = soc[:-1] - (
            p_batt
            * float(config.dt_seconds)
            / (3600.0 * float(config.battery_capacity_kwh))
        )
        np.testing.assert_allclose(
            soc[1:],
            expected_next_soc,
            rtol=0.0,
            atol=SOC_TOLERANCE,
        )

        ramp_limit = float(resolved_ramp_kw_per_step(config))
        self.assertLessEqual(
            abs(float(p_fc[0]) - self.prev_fc_kw),
            ramp_limit + POWER_TOLERANCE_KW,
        )
        self.assertTrue(
            np.all(np.abs(np.diff(p_fc)) <= ramp_limit + POWER_TOLERANCE_KW)
        )

    def test_entries_configs_weights_and_solver_objects_are_independent(self) -> None:
        entries = self.bank._entries
        self.assertEqual(len(entries), 7)
        self.assertEqual(list(entries), list(range(7)))
        self.assertEqual(len({id(entry.config) for entry in entries.values()}), 7)
        self.assertEqual(len({id(entry.solver) for entry in entries.values()}), 7)
        self.assertEqual(len({id(entry.A) for entry in entries.values()}), 1)

        config_field_names = [field.name for field in fields(QpMpcConfig)]
        for action in DQN_MPC_WEIGHT_ACTIONS:
            entry = entries[action.action_id]
            self.assertEqual(entry.action, action)
            self.assertEqual(
                (
                    entry.config.q_h2,
                    entry.config.q_batt,
                    entry.config.q_soc,
                    entry.config.q_fc_var,
                ),
                action.as_tuple(),
            )
            for name in config_field_names:
                if name not in WEIGHT_FIELDS:
                    self.assertEqual(getattr(entry.config, name), getattr(self.base_config, name))

    def test_each_action_has_its_own_weighted_p_and_common_a(self) -> None:
        entries = self.bank._entries
        reference_a = entries[0].A
        p_payloads: set[bytes] = set()

        for action_id, entry in entries.items():
            expected = self._problem(entry.config)
            np.testing.assert_allclose(entry.P.toarray(), expected.P.toarray())
            np.testing.assert_allclose(entry.A.toarray(), expected.A.toarray())
            np.testing.assert_allclose(entry.A.toarray(), reference_a.toarray())
            p_payloads.add(entry.P.toarray().tobytes())
            self.assertEqual(action_id, entry.action.action_id)

        self.assertEqual(len(p_payloads), 7)

    def test_all_actions_solve_and_satisfy_physical_constraints(self) -> None:
        for action_id in range(7):
            with self.subTest(action_id=action_id):
                result, solve_ms = self.bank.solve(
                    action_id=action_id,
                    load_forecast_kw=self.load_forecast_kw,
                    current_soc=self.current_soc,
                    prev_fc_kw=self.prev_fc_kw,
                    soc_reference=self.soc_reference,
                )
                self.assertTrue(str(result.info.status).lower().startswith("solved"))
                self.assertIsNotNone(result.x)
                self.assertGreaterEqual(solve_ms, 0.0)
                self._assert_physical_constraints(
                    self.bank._entries[action_id].config,
                    np.asarray(result.x, dtype=float),
                )

    def test_action_zero_matches_direct_candidate_c_solve(self) -> None:
        self.assertEqual(
            _OSQP_SETTINGS,
            {
                "verbose": False,
                "polishing": True,
                "warm_starting": True,
                "eps_abs": 1.0e-5,
                "eps_rel": 1.0e-5,
                "max_iter": 20000,
                "adaptive_rho": True,
            },
        )
        action = DQN_MPC_WEIGHT_ACTIONS[0]
        direct_config = replace(
            self.base_config,
            q_h2=action.q_h2,
            q_batt=action.q_batt,
            q_soc=action.q_soc,
            q_fc_var=action.q_fc_var,
        )
        direct_problem = self._problem(direct_config)
        osqp_module, import_error = _try_import_osqp()
        self.assertIsNotNone(osqp_module, import_error)
        direct_solver = osqp_module.OSQP()
        direct_solver.setup(
            P=direct_problem.P,
            q=direct_problem.q,
            A=direct_problem.A,
            l=direct_problem.l,
            u=direct_problem.u,
            **_OSQP_SETTINGS,
        )
        direct_result = direct_solver.solve()

        bank_result, _ = self.bank.solve(
            action_id=0,
            load_forecast_kw=self.load_forecast_kw,
            current_soc=self.current_soc,
            prev_fc_kw=self.prev_fc_kw,
            soc_reference=self.soc_reference,
        )

        self.assertTrue(str(direct_result.info.status).lower().startswith("solved"))
        self.assertTrue(str(bank_result.info.status).lower().startswith("solved"))
        np.testing.assert_allclose(
            np.asarray(bank_result.x, dtype=float),
            np.asarray(direct_result.x, dtype=float),
            rtol=0.0,
            atol=POWER_TOLERANCE_KW,
        )

    def test_repeated_solve_keeps_same_solver_object(self) -> None:
        solver = self.bank._entries[3].solver
        for _ in range(2):
            result, _ = self.bank.solve(
                action_id=3,
                load_forecast_kw=self.load_forecast_kw,
                current_soc=self.current_soc,
                prev_fc_kw=self.prev_fc_kw,
                soc_reference=self.soc_reference,
            )
            self.assertTrue(str(result.info.status).lower().startswith("solved"))
            self.assertIs(self.bank._entries[3].solver, solver)

    def test_invalid_action_id_raises(self) -> None:
        for action_id in (-1, 7):
            with self.subTest(action_id=action_id):
                with self.assertRaises((IndexError, ValueError)):
                    self.bank.solve(
                        action_id=action_id,
                        load_forecast_kw=self.load_forecast_kw,
                        current_soc=self.current_soc,
                        prev_fc_kw=self.prev_fc_kw,
                        soc_reference=self.soc_reference,
                    )


if __name__ == "__main__":
    unittest.main()
