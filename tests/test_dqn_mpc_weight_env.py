from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAIN_ROOT = SRC / "main"

for path in (SRC, MAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from dqn.utils.reward import (  # noqa: E402
    calculate_mpc_weight_reward,
)
from dqn.utils.state_builder import (  # noqa: E402
    DQN_MPC_STATE_DIM,
    build_dqn_mpc_state,
)
from envs.dqn_mpc_weight_env import (  # noqa: E402
    DqnMpcWeightEnv,
    validate_executed_battery_power_kw,
)
from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    resolved_ramp_kw_per_step,
)
from run_mpc_1s_n6_four_objective_sensitivity import (  # noqa: E402
    N6_STATE_COMMIT_TOLERANCES,
    build_sensitivity_cases,
    four_objective_config,
)

FORMAL_TEST_DATA_PATH = (
    ROOT
    / "outputs"
    / "mpc_solver_benchmark_1s"
    / "data"
    / "test_voyages_spline_1s.parquet"
)
REGRESSION_VOYAGE_ID = "voyage_064"


def candidate_c_config() -> QpMpcConfig:
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
        objective_variant=(
            "n6_h2_batt_soc_fcvar_normalized_v1"
        ),
        q_h2=0.25,
        q_batt=0.40,
        q_soc=12.0,
        q_fc_var=20.0,
        q_ramp=0.0,
        q_terminal_soc=0.0,
    )


class TestDqnMpcWeightEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.loads = np.asarray(
            [
                200.0,
                215.0,
                230.0,
                245.0,
                260.0,
                250.0,
                240.0,
                230.0,
                220.0,
                210.0,
            ],
            dtype=float,
        )

        self.config = candidate_c_config()

        self.env = DqnMpcWeightEnv(
            loads_kw=self.loads,
            base_config=self.config,
            initial_soc=0.55,
        )

    def test_reset_returns_expected_initial_state(
        self,
    ) -> None:
        state = self.env.reset()

        self.assertEqual(
            state.shape,
            (DQN_MPC_STATE_DIM,),
        )

        self.assertEqual(
            state.dtype,
            np.float32,
        )

        expected = build_dqn_mpc_state(
            current_soc=0.55,
            previous_fc_kw=200.0,
            previous_batt_kw=0.0,
            load_history_kw=self.loads[:1],
        )

        np.testing.assert_allclose(
            state,
            expected,
            rtol=0.0,
            atol=1.0e-7,
        )

    def test_one_step_advances_exactly_one_sample(
        self,
    ) -> None:
        next_state, reward, done, info = (
            self.env.step(0)
        )

        self.assertEqual(
            self.env.decision_index,
            1,
        )

        self.assertEqual(
            info["decision_index"],
            0,
        )

        self.assertEqual(
            info["execution_index"],
            1,
        )

        self.assertAlmostEqual(
            info["load_actual_kw"],
            self.loads[1],
            places=12,
        )

        self.assertEqual(
            next_state.shape,
            (7,),
        )

        self.assertTrue(
            np.isfinite(reward)
        )

        self.assertFalse(done)

    def test_mpc_forecast_holds_current_load(self) -> None:
        forecast = self.env._future_window(2)
        np.testing.assert_array_equal(
            forecast,
            np.full(6, self.loads[2]),
        )

    def test_executed_battery_power_bounds_fail_fast(self) -> None:
        validate_executed_battery_power_kw(
            p_batt_kw=0.0,
            charge_max_kw=624.0,
            discharge_max_kw=1248.0,
        )
        with self.assertRaises(ValueError):
            validate_executed_battery_power_kw(
                p_batt_kw=1248.1,
                charge_max_kw=624.0,
                discharge_max_kw=1248.0,
            )
        with self.assertRaises(ValueError):
            validate_executed_battery_power_kw(
                p_batt_kw=-624.1,
                charge_max_kw=624.0,
                discharge_max_kw=1248.0,
            )

    def test_executed_power_balance_and_soc_update(
        self,
    ) -> None:
        _, _, _, info = self.env.step(0)

        self.assertAlmostEqual(
            info["p_fc_kw"]
            + info["p_batt_kw"],
            info["load_actual_kw"],
            places=10,
        )

        expected_soc = (
            info["soc_before"]
            - info["p_batt_kw"]
            * self.config.dt_seconds
            / 3600.0
            / self.config.battery_capacity_kwh
        )

        self.assertAlmostEqual(
            info["soc_after"],
            expected_soc,
            places=12,
        )

        self.assertAlmostEqual(
            info["power_balance_residual_kw"],
            0.0,
            places=10,
        )

    def test_reward_matches_fixed_reward_function(
        self,
    ) -> None:
        _, reward, _, info = self.env.step(0)

        expected_reward, _ = (
            calculate_mpc_weight_reward(
                p_fc_kw=info["p_fc_kw"],
                p_batt_kw=info["p_batt_kw"],
                next_soc=info["soc_after"],
                previous_fc_kw=info[
                    "p_fc_prev_kw"
                ],
                soc_before=info["soc_before"],
                load_delta_kw=0.0,
            )
        )

        self.assertAlmostEqual(
            reward,
            expected_reward,
            places=12,
        )

    def test_reward_uses_observed_backward_load_delta(self) -> None:
        self.env.step(0)
        _, reward, _, info = self.env.step(0)
        expected_reward, _ = calculate_mpc_weight_reward(
            p_fc_kw=info["p_fc_kw"],
            p_batt_kw=info["p_batt_kw"],
            next_soc=info["soc_after"],
            previous_fc_kw=info["p_fc_prev_kw"],
            soc_before=info["soc_before"],
            load_delta_kw=self.loads[1] - self.loads[0],
        )
        self.assertAlmostEqual(reward, expected_reward, places=12)

    def test_fixed_action_zero_completes_episode(
        self,
    ) -> None:
        state = self.env.reset()

        step_count = 0
        done = False

        while not done:
            state, reward, done, info = (
                self.env.step(0)
            )

            self.assertEqual(
                state.shape,
                (7,),
            )

            self.assertTrue(
                np.isfinite(reward)
            )

            self.assertTrue(
                str(
                    info["solver_status"]
                )
                .lower()
                .startswith("solved")
            )

            step_count += 1

        self.assertEqual(
            step_count,
            len(self.loads) - 1,
        )

        self.assertEqual(
            self.env.decision_index,
            len(self.loads) - 1,
        )

        with self.assertRaises(RuntimeError):
            self.env.step(0)

    def test_action_zero_matches_persistence_reference(
            self,
    ) -> None:
        loads_kw = self.loads.copy()
        config = four_objective_config(build_sensitivity_cases()[0])

        # Independent causal/persistence MPC reference:
        # at decision t, the N=6 forecast is [P_t, ..., P_t].
        solver_bank = MpcWeightSolverBank(config)

        reference_fc: list[float] = []
        reference_batt: list[float] = []
        reference_soc: list[float] = []
        reference_load: list[float] = []

        current_soc = 0.55
        previous_fc = float(
            np.clip(
                loads_kw[0],
                config.fuel_cell_min_kw,
                config.fuel_cell_max_kw,
            )
        )

        for decision_index in range(len(loads_kw) - 1):
            execution_index = decision_index + 1
            current_load = float(loads_kw[decision_index])

            load_forecast = np.full(
                int(config.horizon),
                current_load,
                dtype=float,
            )

            result, _ = solver_bank.solve(
                action_id=0,
                load_forecast_kw=load_forecast,
                current_soc=current_soc,
                prev_fc_kw=previous_fc,
                soc_reference=0.55,
            )

            self.assertTrue(
                str(result.info.status).lower().startswith("solved")
            )
            self.assertIsNotNone(result.x)

            solution = np.asarray(result.x, dtype=float).reshape(-1)

            p_fc = float(solution[0])
            load_actual = float(loads_kw[execution_index])
            p_batt = load_actual - p_fc

            soc_after = current_soc - (
                    p_batt
                    * float(config.dt_seconds)
                    / 3600.0
                    / float(config.battery_capacity_kwh)
            )

            reference_load.append(load_actual)
            reference_fc.append(p_fc)
            reference_batt.append(p_batt)
            reference_soc.append(soc_after)

            previous_fc = p_fc
            current_soc = soc_after

        env = DqnMpcWeightEnv(
            loads_kw=loads_kw,
            base_config=config,
            initial_soc=0.55,
        )

        actual_rows: list[dict[str, object]] = []
        done = False
        while not done:
            _, _, done, info = env.step(0)
            actual_rows.append(info)

        actual_load = np.asarray(
            [float(row["load_actual_kw"]) for row in actual_rows]
        )
        actual_fc = np.asarray(
            [float(row["p_fc_kw"]) for row in actual_rows]
        )
        actual_batt = np.asarray(
            [float(row["p_batt_kw"]) for row in actual_rows]
        )
        actual_soc = np.asarray(
            [float(row["soc_after"]) for row in actual_rows]
        )

        power_tolerance_kw = float(
            N6_STATE_COMMIT_TOLERANCES["power_bound_kw"]
        )
        soc_tolerance = float(
            N6_STATE_COMMIT_TOLERANCES["soc"]
        )

        np.testing.assert_allclose(
            actual_load,
            np.asarray(reference_load),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            actual_fc,
            np.asarray(reference_fc),
            rtol=0.0,
            atol=power_tolerance_kw,
        )
        np.testing.assert_allclose(
            actual_batt,
            np.asarray(reference_batt),
            rtol=0.0,
            atol=power_tolerance_kw,
        )
        np.testing.assert_allclose(
            actual_soc,
            np.asarray(reference_soc),
            rtol=0.0,
            atol=soc_tolerance,
        )

    def test_reset_restores_episode_state(
        self,
    ) -> None:
        initial_state = self.env.reset().copy()

        self.env.step(0)
        self.env.step(3)

        reset_state = self.env.reset()

        np.testing.assert_allclose(
            reset_state,
            initial_state,
            rtol=0.0,
            atol=1.0e-7,
        )

        self.assertEqual(
            self.env.decision_index,
            0,
        )

        self.assertAlmostEqual(
            self.env.current_soc,
            0.55,
            places=12,
        )

        self.assertFalse(
            self.env.done
        )


if __name__ == "__main__":
    unittest.main()
