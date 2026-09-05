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
from mpc_solvers.formal_config import build_formal_mpc_config  # noqa: E402
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    build_qp_problem,
    resolved_ramp_kw_per_step,
)
from mpc_solvers.n6_qp_scaling import (  # noqa: E402
    _setup_n6_osqp_solver,
    scale_n6_qp_problem,
)
from mpc_solvers.osqp_runtime import _try_import_osqp  # noqa: E402

ACTION_CONFIG_FIELDS = {
    "q_h2",
    "q_batt",
    "q_soc",
    "q_fc_var",
}
POWER_TOLERANCE_KW = 0.1
SOC_TOLERANCE = 2.0e-5


def formal_base_config() -> QpMpcConfig:
    return build_formal_mpc_config()


class TestDqnMpcSolverBank(unittest.TestCase):
    def setUp(self) -> None:
        self.base_config = formal_base_config()
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
        self.assertEqual(len(values), 4 * horizon + 1)
        p_fc = values[:horizon]
        p_batt = values[horizon : 2 * horizon]
        soc = values[2 * horizon : 3 * horizon + 1]
        violation = values[3 * horizon + 1 :]

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

        self.assertGreaterEqual(float(violation.min()), -SOC_TOLERANCE)
        np.testing.assert_allclose(
            violation,
            np.maximum(
                0.0,
                np.maximum(
                    float(config.soc_soft_min) - soc[1:],
                    soc[1:] - float(config.soc_soft_max),
                ),
            ),
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
        action_count = len(DQN_MPC_WEIGHT_ACTIONS)
        entries = self.bank._entries
        self.assertEqual(len(entries), action_count)
        self.assertEqual(list(entries), list(range(action_count)))
        self.assertEqual(
            len({id(entry.config) for entry in entries.values()}),
            action_count,
        )
        self.assertEqual(
            len({id(entry.solver) for entry in entries.values()}),
            action_count,
        )
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
                if name not in ACTION_CONFIG_FIELDS:
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

        self.assertEqual(
            len(p_payloads),
            len(DQN_MPC_WEIGHT_ACTIONS),
        )

    def test_all_actions_solve_and_satisfy_physical_constraints(self) -> None:
        for action_id in range(len(DQN_MPC_WEIGHT_ACTIONS)):
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

    def test_action_zero_matches_direct_formal_solve(self) -> None:
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

        # A0 必须保持正式 nominal 权重。
        action = DQN_MPC_WEIGHT_ACTIONS[0]

        self.assertEqual(
            action.as_tuple(),
            (0.25, 0.40, 12.0, 20.0),
        )

        direct_config = replace(
            self.base_config,
            q_h2=action.q_h2,
            q_batt=action.q_batt,
            q_soc=action.q_soc,
            q_fc_var=action.q_fc_var,
        )

        # 先构造与正式 nominal action 相同的物理 QP。
        direct_problem = self._problem(direct_config)

        # 再走正式 N=6 MPC 使用的同一套仿射缩放链路。
        scaled_problem, transform = scale_n6_qp_problem(
            direct_problem,
            config=direct_config,
        )

        osqp_module, import_error = _try_import_osqp()
        self.assertIsNotNone(osqp_module, import_error)

        direct_solver = _setup_n6_osqp_solver(
            osqp_module,
            scaled_problem,
        )

        direct_result = direct_solver.solve(raise_error=False)

        self.assertTrue(
            str(direct_result.info.status).lower().startswith("solved")
        )
        self.assertIsNotNone(direct_result.x)

        direct_scaled_solution = np.asarray(
            direct_result.x,
            dtype=float,
        ).reshape(-1)

        direct_physical_solution = transform.to_physical(
            direct_scaled_solution
        )

        # Solver bank A0：
        # 内部同样应在缩放空间求解，返回时恢复为物理解。
        bank_result, _ = self.bank.solve(
            action_id=0,
            load_forecast_kw=self.load_forecast_kw,
            current_soc=self.current_soc,
            prev_fc_kw=self.prev_fc_kw,
            soc_reference=self.soc_reference,
        )

        self.assertTrue(
            str(bank_result.info.status).lower().startswith("solved")
        )
        self.assertIsNotNone(bank_result.x)

        bank_physical_solution = np.asarray(
            bank_result.x,
            dtype=float,
        ).reshape(-1)

        horizon = int(direct_config.horizon)

        # 1. 完整物理决策向量必须一致。
        np.testing.assert_allclose(
            bank_physical_solution,
            direct_physical_solution,
            rtol=0.0,
            atol=POWER_TOLERANCE_KW,
        )

        # 决策变量顺序：
        # [P_fc(0:N), P_batt(0:N), SOC(0:N+1), SOC_band_violation(0:N)]
        bank_p_fc = bank_physical_solution[:horizon]
        bank_p_batt = bank_physical_solution[horizon: 2 * horizon]
        bank_soc = bank_physical_solution[2 * horizon: 3 * horizon + 1]

        direct_p_fc = direct_physical_solution[:horizon]
        direct_p_batt = direct_physical_solution[horizon: 2 * horizon]
        direct_soc = direct_physical_solution[2 * horizon: 3 * horizon + 1]

        # 2. 第一时刻燃料电池功率。
        self.assertAlmostEqual(
            float(bank_p_fc[0]),
            float(direct_p_fc[0]),
            delta=POWER_TOLERANCE_KW,
        )

        # 3. 第一时刻电池功率。
        self.assertAlmostEqual(
            float(bank_p_batt[0]),
            float(direct_p_batt[0]),
            delta=POWER_TOLERANCE_KW,
        )

        # 4. 执行第一步后的 SOC。
        self.assertAlmostEqual(
            float(bank_soc[1]),
            float(direct_soc[1]),
            delta=SOC_TOLERANCE,
        )

        # 5. A0 结果必须满足功率平衡。
        np.testing.assert_allclose(
            bank_p_fc + bank_p_batt,
            self.load_forecast_kw,
            rtol=0.0,
            atol=POWER_TOLERANCE_KW,
        )

        # 6. 确认 bank 确实保留了缩放空间解，
        #    且恢复后就是当前返回的物理解。
        self.assertTrue(hasattr(bank_result, "x_scaled"))
        self.assertTrue(hasattr(bank_result, "x_physical"))

        np.testing.assert_allclose(
            np.asarray(bank_result.x_physical, dtype=float),
            bank_physical_solution,
            rtol=0.0,
            atol=1.0e-12,
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
