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
)
from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    resolved_ramp_kw_per_step,
)
from run_mpc_1s_n6_four_objective_sensitivity import (  # noqa: E402
    N6_STATE_COMMIT_TOLERANCES,
    build_sensitivity_cases,
    four_objective_config,
    run_voyage,
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
            current_load_kw=200.0,
            previous_load_kw=200.0,
            future_load_kw=self.loads[1:7],
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
            (11,),
        )

        self.assertTrue(
            np.isfinite(reward)
        )

        self.assertFalse(done)

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
            )
        )

        self.assertAlmostEqual(
            reward,
            expected_reward,
            places=12,
        )

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
                (11,),
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

    def test_action_zero_full_voyage_matches_formal_candidate_c(
        self,
    ) -> None:
        frame = pd.read_parquet(
            FORMAL_TEST_DATA_PATH,
            columns=[
                "voyage_id",
                "time_s",
                "load_total_kw",
            ],
        )
        voyage = (
            frame.loc[
                frame["voyage_id"].astype(str)
                == REGRESSION_VOYAGE_ID
            ]
            .sort_values("time_s", kind="stable")
            .reset_index(drop=True)
        )
        self.assertFalse(voyage.empty)

        loads_kw = voyage["load_total_kw"].to_numpy(
            dtype=float
        )
        times_s = voyage["time_s"].to_numpy(dtype=float)
        case = build_sensitivity_cases()[0]
        config = four_objective_config(case)

        reference, reference_solver = run_voyage(
            voyage_id=REGRESSION_VOYAGE_ID,
            loads_kw=loads_kw,
            times_s=times_s,
            case=case,
            config=config,
            initial_soc=0.55,
        )
        self.assertTrue(reference["success"].all())
        self.assertTrue(reference_solver["success"].all())

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

        self.assertEqual(
            len(actual_rows),
            len(loads_kw) - 1,
        )
        self.assertEqual(len(actual_rows), len(reference))

        actual_decision_indices = np.asarray(
            [
                int(row["decision_index"])
                for row in actual_rows
            ],
            dtype=int,
        )
        actual_execution_indices = np.asarray(
            [
                int(row["execution_index"])
                for row in actual_rows
            ],
            dtype=int,
        )
        np.testing.assert_array_equal(
            actual_decision_indices,
            reference["decision_index"].to_numpy(dtype=int),
        )
        np.testing.assert_array_equal(
            actual_execution_indices,
            reference["execution_index"].to_numpy(dtype=int),
        )

        actual_load = np.asarray(
            [
                float(row["load_actual_kw"])
                for row in actual_rows
            ]
        )
        actual_fc = np.asarray(
            [
                float(row["p_fc_kw"])
                for row in actual_rows
            ]
        )
        actual_batt = np.asarray(
            [
                float(row["p_batt_kw"])
                for row in actual_rows
            ]
        )
        actual_soc = np.asarray(
            [
                float(row["soc_after"])
                for row in actual_rows
            ]
        )

        reference_load = reference[
            "load_actual_kw"
        ].to_numpy(dtype=float)
        reference_fc = reference[
            "P_fc_actual_kw"
        ].to_numpy(dtype=float)
        reference_batt = reference[
            "P_batt_actual_kw"
        ].to_numpy(dtype=float)
        reference_soc = reference[
            "SOC_actual"
        ].to_numpy(dtype=float)

        power_tolerance_kw = float(
            N6_STATE_COMMIT_TOLERANCES[
                "power_bound_kw"
            ]
        )
        soc_tolerance = float(
            N6_STATE_COMMIT_TOLERANCES["soc"]
        )
        np.testing.assert_allclose(
            actual_load,
            reference_load,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            actual_fc,
            reference_fc,
            rtol=0.0,
            atol=power_tolerance_kw,
        )
        np.testing.assert_allclose(
            actual_batt,
            reference_batt,
            rtol=0.0,
            atol=power_tolerance_kw,
        )
        np.testing.assert_allclose(
            actual_soc,
            reference_soc,
            rtol=0.0,
            atol=soc_tolerance,
        )

        actual_solver_status = np.asarray(
            [
                str(row["solver_status"])
                for row in actual_rows
            ]
        )
        reference_solver_status = reference[
            "solver_status"
        ].astype(str).to_numpy()
        np.testing.assert_array_equal(
            actual_solver_status,
            reference_solver_status,
        )
        self.assertTrue(
            all(
                status.lower().startswith("solved")
                for status in actual_solver_status
            )
        )

        balance_tolerance_kw = float(
            N6_STATE_COMMIT_TOLERANCES[
                "actual_balance_kw"
            ]
        )
        np.testing.assert_allclose(
            actual_fc + actual_batt,
            actual_load,
            rtol=0.0,
            atol=balance_tolerance_kw,
        )

        initial_fc_kw = float(
            np.clip(
                loads_kw[0],
                config.fuel_cell_min_kw,
                config.fuel_cell_max_kw,
            )
        )
        fc_delta_kw = np.diff(
            np.concatenate(
                [
                    np.asarray([initial_fc_kw]),
                    actual_fc,
                ]
            )
        )
        self.assertLessEqual(
            float(np.max(np.abs(fc_delta_kw))),
            float(resolved_ramp_kw_per_step(config))
            + float(
                N6_STATE_COMMIT_TOLERANCES["ramp_kw"]
            ),
        )
        self.assertGreaterEqual(
            float(actual_fc.min()),
            float(config.fuel_cell_min_kw)
            - power_tolerance_kw,
        )
        self.assertLessEqual(
            float(actual_fc.max()),
            float(config.fuel_cell_max_kw)
            + power_tolerance_kw,
        )
        self.assertGreaterEqual(
            float(actual_batt.min()),
            -float(config.battery_charge_max_kw)
            - power_tolerance_kw,
        )
        self.assertLessEqual(
            float(actual_batt.max()),
            float(config.battery_discharge_max_kw)
            + power_tolerance_kw,
        )
        self.assertGreaterEqual(
            float(actual_soc.min()),
            float(config.soc_min) - soc_tolerance,
        )
        self.assertLessEqual(
            float(actual_soc.max()),
            float(config.soc_max) + soc_tolerance,
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
