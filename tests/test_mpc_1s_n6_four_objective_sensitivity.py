from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src" / "main"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mpc_solvers.mpc_qp_formulation import (  # noqa: E402
    QpMpcConfig,
    build_qp_problem,
    h2_quadratic_kg_step_coefficients,
    hessian_min_eigenvalue,
)

OBJECTIVE_VARIANT = "n6_h2_batt_soc_fcvar_normalized_v1"
SOC_REFERENCE = 0.55


def objective_config(*, horizon: int = 3, **weights: float) -> QpMpcConfig:
    values = {"q_h2": 0.0, "q_batt": 0.0, "q_soc": 0.0, "q_fc_var": 0.0}
    values.update(weights)
    return QpMpcConfig(
        horizon=horizon,
        dt_seconds=1.0,
        battery_capacity_kwh=693.0,
        battery_charge_max_kw=346.5,
        battery_discharge_max_kw=346.5,
        battery_power_ref_kw=346.5,
        fuel_cell_min_kw=0.0,
        fuel_cell_max_kw=560.0,
        fuel_cell_ramp_rate_kw_per_s=48.0,
        soc_min=0.2,
        soc_max=0.8,
        soc_band=0.05,
        objective_variant=OBJECTIVE_VARIANT,
        q_ramp=0.0,
        q_terminal_soc=0.0,
        **values,
    )


def problem(config: QpMpcConfig, *, prev_fc_kw: float = 200.0):
    load = np.linspace(250.0, 300.0, config.horizon)
    return build_qp_problem(
        config,
        load_forecast_kw=load,
        current_soc=0.54,
        prev_fc_kw=prev_fc_kw,
        soc_reference=SOC_REFERENCE,
    )


class TestFourObjectiveQp(unittest.TestCase):
    def test_each_weight_changes_only_its_exact_osqp_coefficients(self) -> None:
        horizon = 3
        prev_fc_kw = 123.0
        cfg = objective_config(horizon=horizon)
        base = problem(cfg, prev_fc_kw=prev_fc_kw)
        h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(cfg)
        h2_ref = h2_quad * 560.0**2 + h2_linear * 560.0

        expected: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for weight_name, indices, diagonal, linear in (
            ("q_h2", range(0, horizon), 2.0 * h2_quad / h2_ref, h2_linear / h2_ref),
            ("q_batt", range(horizon, 2 * horizon), 2.0 / 346.5**2, 0.0),
            (
                "q_soc",
                range(2 * horizon + 1, 3 * horizon + 1),
                2.0 / 0.05**2,
                -2.0 * SOC_REFERENCE / 0.05**2,
            ),
        ):
            wanted_p = np.zeros_like(base.P.toarray())
            wanted_q = np.zeros_like(base.q)
            for index in indices:
                wanted_p[index, index] = diagonal
                wanted_q[index] = linear
            expected[weight_name] = wanted_p, wanted_q

        fc_var_weight = 1.0 / 48.0**2
        wanted_fc_var_p = np.zeros_like(base.P.toarray())
        wanted_fc_var_p[:horizon, :horizon] = np.array(
            [
                [4.0 * fc_var_weight, -2.0 * fc_var_weight, 0.0],
                [-2.0 * fc_var_weight, 4.0 * fc_var_weight, -2.0 * fc_var_weight],
                [0.0, -2.0 * fc_var_weight, 2.0 * fc_var_weight],
            ]
        )
        wanted_fc_var_q = np.zeros_like(base.q)
        wanted_fc_var_q[0] = -2.0 * fc_var_weight * prev_fc_kw
        expected["q_fc_var"] = wanted_fc_var_p, wanted_fc_var_q

        for weight_name, (wanted_p, wanted_q) in expected.items():
            changed = problem(
                objective_config(horizon=horizon, **{weight_name: 1.0}),
                prev_fc_kw=prev_fc_kw,
            )
            np.testing.assert_allclose(
                changed.P.toarray() - base.P.toarray(), wanted_p, rtol=0.0, atol=1e-12
            )
            np.testing.assert_allclose(changed.q - base.q, wanted_q, rtol=0.0, atol=1e-12)

    def test_fc_variation_has_prev_fc_first_term_and_adjacent_terms(self) -> None:
        prev_fc_kw = 123.0
        qp = problem(objective_config(horizon=3, q_fc_var=1.0), prev_fc_kw=prev_fc_kw)
        weight = 1.0 / 48.0**2
        expected = np.array(
            [
                [4.0 * weight, -2.0 * weight, 0.0],
                [-2.0 * weight, 4.0 * weight, -2.0 * weight],
                [0.0, -2.0 * weight, 2.0 * weight],
            ]
        )
        np.testing.assert_allclose(qp.P.toarray()[:3, :3], expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            qp.q[:3], np.array([-2.0 * weight * prev_fc_kw, 0.0, 0.0]), rtol=0.0, atol=1e-12
        )

    def test_baseline_metadata_has_exact_terms_descriptions_and_references(self) -> None:
        cfg = objective_config(q_h2=1.0, q_batt=1.0, q_soc=1.0, q_fc_var=1.0)
        qp = problem(cfg)

        self.assertEqual(qp.metadata["objective_variant"], OBJECTIVE_VARIANT)
        self.assertEqual(
            qp.metadata["objective_terms"],
            [
                "H2_norm",
                "Batt_power_sq_norm",
                "SOC_tracking_sq_norm",
                "FC_variation_sq_norm",
            ],
        )
        self.assertEqual(
            qp.metadata["objective_term_descriptions"],
            {
                "H2_norm": "sum(k=0..N-1) m_H2(P_fc[k]) / m_H2(560 kW, 1 s)",
                "Batt_power_sq_norm": "sum(k=0..N-1) (P_batt[k] / 346.5 kW)^2",
                "SOC_tracking_sq_norm": "sum(k=1..N) ((SOC[k] - SOC_ref) / 0.05)^2",
                "FC_variation_sq_norm": (
                    "((P_fc[0] - P_fc_prev) / 48 kW)^2 + "
                    "sum(k=1..N-1) ((P_fc[k] - P_fc[k-1]) / 48 kW)^2"
                ),
            },
        )
        self.assertAlmostEqual(qp.metadata["h2_reference_kg_per_step"], 0.00883945296644347)
        self.assertEqual(qp.metadata["battery_power_ref_kw"], 346.5)
        self.assertEqual(qp.metadata["soc_reference"], SOC_REFERENCE)
        self.assertEqual(qp.metadata["soc_band"], 0.05)
        self.assertEqual(qp.metadata["fuel_cell_variation_ref_kw_per_step"], 48.0)
        self.assertEqual(
            [qp.metadata[name] for name in ("q_h2", "q_batt", "q_soc", "q_fc_var")],
            [1.0, 1.0, 1.0, 1.0],
        )
        self.assertTrue(qp.metadata["objective_uses_term_normalization"])
        self.assertTrue(qp.metadata["soc_cost_in_objective"])
        self.assertEqual(qp.metadata["battery_cost_form"], "normalized (P_batt / P_batt_ref)^2")
        self.assertEqual(
            qp.metadata["fuel_cell_variation_cost_form"],
            "((P_fc[0] - P_fc_prev) / 48)^2 and ((P_fc[k] - P_fc[k-1]) / 48)^2",
        )
        self.assertEqual(
            qp.metadata["ignored_objective_weight_fields"], ["q_ramp", "q_terminal_soc"]
        )
        self.assertFalse(qp.metadata["terminal_soc_cost_in_objective"])
        self.assertFalse(qp.metadata["slack_cost_in_objective"])
        self.assertFalse(qp.metadata["extra_ramp_cost_in_objective"])

        h2_only = problem(objective_config(q_h2=1.0))
        self.assertGreater(h2_only.q[0], 0.0)

    def test_nondefault_references_keep_metadata_descriptions_truthful(self) -> None:
        cfg = replace(
            objective_config(q_h2=1.0, q_batt=1.0, q_soc=1.0, q_fc_var=1.0),
            dt_seconds=2.5,
            fuel_cell_max_kw=420.0,
            battery_power_ref_kw=111.25,
            soc_band=0.0375,
            fuel_cell_ramp_kw=17.5,
        )
        qp = problem(cfg)
        h2_quad, h2_linear, _, _ = h2_quadratic_kg_step_coefficients(cfg)
        h2_reference = h2_quad * 420.0**2 + h2_linear * 420.0

        self.assertAlmostEqual(qp.metadata["h2_reference_kg_per_step"], h2_reference)
        self.assertEqual(qp.metadata["battery_power_ref_kw"], 111.25)
        self.assertEqual(qp.metadata["soc_band"], 0.0375)
        self.assertEqual(qp.metadata["fuel_cell_variation_ref_kw_per_step"], 17.5)
        self.assertEqual(
            qp.metadata["fuel_cell_variation_cost_form"],
            "((P_fc[0] - P_fc_prev) / 17.5)^2 and ((P_fc[k] - P_fc[k-1]) / 17.5)^2",
        )
        self.assertEqual(
            qp.metadata["objective_term_descriptions"],
            {
                "H2_norm": "sum(k=0..N-1) m_H2(P_fc[k]) / m_H2(420 kW, 2.5 s)",
                "Batt_power_sq_norm": "sum(k=0..N-1) (P_batt[k] / 111.25 kW)^2",
                "SOC_tracking_sq_norm": "sum(k=1..N) ((SOC[k] - SOC_ref) / 0.0375)^2",
                "FC_variation_sq_norm": (
                    "((P_fc[0] - P_fc_prev) / 17.5 kW)^2 + "
                    "sum(k=1..N-1) ((P_fc[k] - P_fc[k-1]) / 17.5 kW)^2"
                ),
            },
        )

    def test_legacy_soft_weights_are_ignored_and_constraints_do_not_change(self) -> None:
        cfg = objective_config(q_h2=1.0, q_batt=1.0, q_soc=1.0, q_fc_var=1.0)
        base = problem(cfg)
        ignored = problem(replace(cfg, q_ramp=999.0, q_terminal_soc=888.0))
        np.testing.assert_allclose(base.P.toarray(), ignored.P.toarray())
        np.testing.assert_allclose(base.q, ignored.q)
        np.testing.assert_allclose(base.A.toarray(), ignored.A.toarray())
        np.testing.assert_allclose(base.l, ignored.l)
        np.testing.assert_allclose(base.u, ignored.u)

    def test_weight_changes_preserve_constraints_and_hessian_psd(self) -> None:
        baseline_cfg = objective_config(q_h2=1.0, q_batt=1.0, q_soc=1.0, q_fc_var=1.0)
        baseline = problem(baseline_cfg)
        for name in ("q_h2", "q_batt", "q_soc", "q_fc_var"):
            changed = problem(replace(baseline_cfg, **{name: 4.0}))
            np.testing.assert_allclose(changed.A.toarray(), baseline.A.toarray())
            np.testing.assert_allclose(changed.l, baseline.l)
            np.testing.assert_allclose(changed.u, baseline.u)
            self.assertGreaterEqual(hessian_min_eigenvalue(changed), -1e-10)

    def test_power_balance_soc_dynamics_and_hard_ramp_remain_exact(self) -> None:
        cfg = objective_config(
            horizon=2, q_h2=1.0, q_batt=1.0, q_soc=1.0, q_fc_var=1.0
        )
        qp = problem(cfg, prev_fc_kw=200.0)
        matrix = qp.A.toarray()

        self.assertEqual(qp.metadata["variable_order"], "P_fc[0:N], P_batt[0:N], SOC[0:N+1]")
        self.assertEqual(qp.metadata["n_variables"], 7)
        self.assertEqual(qp.metadata["n_constraints"], 14)

        soc_coefficient = 1.0 / (3600.0 * 693.0)
        np.testing.assert_array_equal(
            matrix[8], [0.0, 0.0, soc_coefficient, 0.0, -1.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            matrix[9], [0.0, 0.0, 0.0, soc_coefficient, 0.0, -1.0, 1.0]
        )
        np.testing.assert_array_equal(qp.l[8:10], [0.0, 0.0])
        np.testing.assert_array_equal(qp.u[8:10], [0.0, 0.0])

        np.testing.assert_array_equal(matrix[10], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(matrix[11], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(qp.l[10:12], [250.0, 300.0])
        np.testing.assert_array_equal(qp.u[10:12], [250.0, 300.0])

        np.testing.assert_array_equal(matrix[12], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual((qp.l[12], qp.u[12]), (152.0, 248.0))
        np.testing.assert_array_equal(matrix[13], [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual((qp.l[13], qp.u[13]), (-48.0, 48.0))

    def test_obsolete_n6_objective_variants_are_rejected(self) -> None:
        for variant in (
            "n6_h2_fc_variation_battery_v1",
            "n6_h2_fc_variation_battery_unnormalized_v1",
        ):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, f"unsupported objective_variant: {variant}"):
                    problem(replace(objective_config(), objective_variant=variant))


class TestSensitivityRunnerContract(unittest.TestCase):
    def test_exact_17_cases_are_unique_and_in_deterministic_order(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            SensitivityCase,
            build_sensitivity_cases,
        )

        cases = build_sensitivity_cases()
        expected_ids = [
            "baseline_1_1_1_1",
            "q_h2_0p25",
            "q_h2_0p5",
            "q_h2_2",
            "q_h2_4",
            "q_batt_0p25",
            "q_batt_0p5",
            "q_batt_2",
            "q_batt_4",
            "q_soc_0p25",
            "q_soc_0p5",
            "q_soc_2",
            "q_soc_4",
            "q_fc_var_0p25",
            "q_fc_var_0p5",
            "q_fc_var_2",
            "q_fc_var_4",
        ]
        self.assertEqual([case.config_id for case in cases], expected_ids)
        self.assertEqual(len(cases), 17)
        self.assertEqual(
            len({(case.q_h2, case.q_batt, case.q_soc, case.q_fc_var) for case in cases}),
            17,
        )
        self.assertEqual(
            [case.varied_weight for case in cases],
            [None] + [name for name in ("q_h2", "q_batt", "q_soc", "q_fc_var") for _ in range(4)],
        )
        self.assertEqual(
            [case.weight_value for case in cases],
            [1.0] + [value for _ in range(4) for value in (0.25, 0.5, 2.0, 4.0)],
        )
        self.assertTrue(SensitivityCase.__dataclass_params__.frozen)

    def test_default_paths_and_fixed_experiment_constants_are_exact(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        self.assertEqual(module.OBJECTIVE_VARIANT, OBJECTIVE_VARIANT)
        self.assertEqual(module.N6_HORIZON, 6)
        self.assertEqual(module.N6_DT_SECONDS, 1.0)
        self.assertEqual(
            module.EXPECTED_TEST_VOYAGES,
            tuple(f"voyage_{index:03d}" for index in range(60, 67)),
        )
        self.assertEqual(
            module.DEFAULT_INPUT_PATH,
            ROOT / "outputs" / "mpc_solver_benchmark_1s" / "data" / "test_voyages_spline_1s.parquet",
        )
        self.assertTrue(module.DEFAULT_INPUT_PATH.is_file())
        self.assertEqual(
            module.DEFAULT_OUTPUT_DIR,
            ROOT / "outputs" / "mpc_1s_n6_four_objective_sensitivity",
        )
        self.assertEqual(
            module.DEFAULT_SUMMARY_REPORT,
            ROOT / "reports" / "mpc_1s_n6_four_objective_sensitivity_summary.md",
        )
        self.assertEqual(
            module.DEFAULT_TABLE_REPORT,
            ROOT / "reports" / "mpc_1s_n6_four_objective_sensitivity_table.csv",
        )

    def test_baseline_config_freezes_n6_physics_and_four_weights(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            build_sensitivity_cases,
            four_objective_config,
        )

        config = four_objective_config(build_sensitivity_cases()[0])
        self.assertEqual((config.horizon, config.dt_seconds), (6, 1.0))
        self.assertEqual(config.battery_capacity_kwh, 693.0)
        self.assertEqual(
            (
                config.battery_charge_max_kw,
                config.battery_discharge_max_kw,
                config.battery_power_ref_kw,
            ),
            (346.5, 346.5, 346.5),
        )
        self.assertEqual((config.fuel_cell_min_kw, config.fuel_cell_max_kw), (0.0, 560.0))
        self.assertEqual(config.fuel_cell_ramp_rate_kw_per_s, 48.0)
        self.assertIsNone(config.fuel_cell_ramp_kw)
        self.assertEqual((config.soc_min, config.soc_max, config.soc_band), (0.2, 0.8, 0.05))
        self.assertEqual(config.objective_variant, OBJECTIVE_VARIANT)
        self.assertEqual(
            (config.q_h2, config.q_batt, config.q_soc, config.q_fc_var),
            (1.0, 1.0, 1.0, 1.0),
        )
        self.assertEqual((config.q_ramp, config.q_terminal_soc), (0.0, 0.0))

    def test_future_window_is_t_plus_1_through_t_plus_6_with_end_padding(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import ideal_future_window

        loads = np.arange(8, dtype=float)
        np.testing.assert_array_equal(
            ideal_future_window(loads, decision_index=0),
            np.array([1, 2, 3, 4, 5, 6], dtype=float),
        )
        np.testing.assert_array_equal(
            ideal_future_window(loads, decision_index=6),
            np.array([7, 7, 7, 7, 7, 7], dtype=float),
        )
        with self.assertRaisesRegex(ValueError, "future execution sample"):
            ideal_future_window(loads, decision_index=7)

    def test_first_action_uses_actual_battery_power_to_update_soc(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            build_sensitivity_cases,
            extract_first_step,
            four_objective_config,
        )

        config = four_objective_config(build_sensitivity_cases()[0])
        solution = np.zeros(19, dtype=float)
        solution[0] = 110.0
        solution[6] = -7.0
        solution[13] = 0.550003
        applied = extract_first_step(
            solution,
            config=config,
            load_actual_kw=100.0,
            current_soc=0.55,
        )
        self.assertEqual(applied["P_fc_plan_kw"], 110.0)
        self.assertEqual(applied["P_fc_actual_kw"], 110.0)
        self.assertEqual(applied["P_batt_plan_kw"], -7.0)
        self.assertEqual(applied["P_batt_actual_kw"], -10.0)
        self.assertEqual(applied["SOC_predicted"], 0.550003)
        self.assertAlmostEqual(applied["SOC_actual"], 0.55 + 10.0 / (3600.0 * 693.0))

    def test_affine_transform_roundtrip_constraints_bounds_and_objective(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            build_sensitivity_cases,
            four_objective_config,
            scale_n6_qp_problem,
        )

        config = four_objective_config(build_sensitivity_cases()[0])
        physical_problem = build_qp_problem(
            config,
            load_forecast_kw=np.full(6, 300.0),
            current_soc=0.54,
            prev_fc_kw=200.0,
            soc_reference=SOC_REFERENCE,
        )
        scaled_problem, transform = scale_n6_qp_problem(physical_problem, config=config)
        physical = np.r_[
            np.full(6, 250.0),
            np.full(6, 50.0),
            np.linspace(0.54, 0.5399, 7),
        ]
        normalized = transform.to_normalized(physical)

        np.testing.assert_array_equal(
            transform.variable_scale,
            np.r_[np.full(6, 560.0), np.full(6, 346.5), np.full(7, 0.05)],
        )
        np.testing.assert_array_equal(
            transform.variable_offset,
            np.r_[np.zeros(12), np.full(7, SOC_REFERENCE)],
        )
        np.testing.assert_allclose(transform.to_physical(normalized), physical, atol=1.0e-12)
        np.testing.assert_allclose(
            np.asarray(scaled_problem.A @ normalized).reshape(-1),
            transform.row_scale
            * (
                np.asarray(physical_problem.A @ physical).reshape(-1)
                - transform.constraint_offset
            ),
            atol=1.0e-12,
        )
        scaled_lower, scaled_upper = transform.transform_bounds(
            physical_problem.l, physical_problem.u
        )
        np.testing.assert_allclose(scaled_problem.l, scaled_lower, atol=1.0e-12)
        np.testing.assert_allclose(scaled_problem.u, scaled_upper, atol=1.0e-12)
        physical_value = 0.5 * float(physical @ (physical_problem.P @ physical)) + float(
            physical_problem.q @ physical
        )
        scaled_value = (
            0.5 * float(normalized @ (scaled_problem.P @ normalized))
            + float(scaled_problem.q @ normalized)
            + transform.objective_constant
        )
        self.assertAlmostEqual(physical_value, scaled_value, places=9)

    def test_scaled_linear_refresh_matches_fresh_previous_fc_problem(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            build_sensitivity_cases,
            four_objective_config,
            scale_n6_qp_problem,
            scaled_linear_for_previous_fc,
        )

        config = four_objective_config(build_sensitivity_cases()[0])
        loads = np.full(6, 300.0)
        base_problem = build_qp_problem(
            config,
            load_forecast_kw=loads,
            current_soc=0.54,
            prev_fc_kw=200.0,
            soc_reference=SOC_REFERENCE,
        )
        scaled_base, transform = scale_n6_qp_problem(base_problem, config=config)
        refreshed = scaled_linear_for_previous_fc(
            scaled_base.q,
            config=config,
            transform=transform,
            base_previous_fc_kw=200.0,
            previous_fc_kw=210.0,
        )
        fresh_problem = build_qp_problem(
            config,
            load_forecast_kw=loads,
            current_soc=0.54,
            prev_fc_kw=210.0,
            soc_reference=SOC_REFERENCE,
        )
        scaled_fresh, _ = scale_n6_qp_problem(fresh_problem, config=config)
        np.testing.assert_allclose(refreshed, scaled_fresh.q, rtol=0.0, atol=1.0e-12)

    def test_osqp_settings_are_exact_and_have_no_fixed_adaptive_interval(self) -> None:
        from run_mpc_1s_n6_four_objective_sensitivity import (
            N6_OSQP_SETTINGS,
            N6_STATE_COMMIT_TOLERANCES,
        )

        self.assertEqual(
            N6_OSQP_SETTINGS,
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
        self.assertNotIn("adaptive_rho_interval", N6_OSQP_SETTINGS)
        self.assertEqual(N6_STATE_COMMIT_TOLERANCES["soc"], 1.0e-5)

    def test_solved_candidate_outside_commit_tolerance_is_not_applied(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        solved_result = SimpleNamespace(
            x=np.zeros(19, dtype=float),
            info=SimpleNamespace(
                status="solved inaccurate",
                iter=50,
                prim_res=1.0e-5,
                dual_res=1.0e-5,
            ),
        )
        rejected_step = {
            "P_fc_plan_kw": 700.0,
            "P_batt_plan_kw": -590.0,
            "SOC_predicted": 0.550236,
            "P_fc_actual_kw": 700.0,
            "P_batt_actual_kw": -590.0,
            "SOC_actual": 0.550236,
        }
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(solved_result, 0.2),
            ) as solve,
            patch.object(module, "extract_first_step", return_value=rejected_step),
        ):
            controls, solver_rows = module.run_voyage(
                voyage_id="voyage_test",
                loads_kw=np.array([100.0, 110.0, 120.0]),
                times_s=np.array([0.0, 1.0, 2.0]),
                case=case,
                config=config,
            )

        self.assertEqual(solve.call_count, 1)
        self.assertEqual(len(controls), 1)
        self.assertFalse(bool(controls.iloc[0]["success"]))
        self.assertTrue(np.isnan(controls.iloc[0]["SOC_actual"]))
        self.assertGreater(controls.iloc[0]["fc_bound_residual_kw"], 0.1)
        self.assertIn("commit tolerance", controls.iloc[0]["status"])
        self.assertFalse(bool(solver_rows.iloc[0]["success"]))
        self.assertTrue(bool(solver_rows.iloc[0]["solved_inaccurate"]))

    def test_state_commit_rejects_nonfinite_solution_but_accepts_osqp_scale_soc_residual(
        self,
    ) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        nonfinite_result = SimpleNamespace(
            x=np.full(19, np.nan),
            info=SimpleNamespace(
                status="solved",
                iter=25,
                prim_res=1.0e-8,
                dual_res=1.0e-8,
            ),
        )
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(nonfinite_result, 0.1),
            ),
            patch.object(module, "extract_first_step") as extract,
        ):
            rejected, _ = module.run_voyage(
                voyage_id="voyage_test",
                loads_kw=np.array([100.0, 110.0]),
                times_s=np.array([0.0, 1.0]),
                case=case,
                config=config,
            )
        extract.assert_not_called()
        self.assertFalse(bool(rejected.iloc[0]["success"]))
        self.assertEqual(rejected.iloc[0]["rejection_reason"], "nonfinite_solution")
        self.assertTrue(np.isnan(rejected.iloc[0]["SOC_actual"]))

        solved_result = SimpleNamespace(
            x=np.zeros(19, dtype=float),
            info=SimpleNamespace(
                status="solved",
                iter=25,
                prim_res=1.0e-8,
                dual_res=1.0e-8,
            ),
        )
        accepted_step = {
            "P_fc_plan_kw": 110.0,
            "P_batt_plan_kw": 0.0,
            "SOC_predicted": 0.2 - 5.86e-6,
            "P_fc_actual_kw": 110.0,
            "P_batt_actual_kw": 0.0,
            "SOC_actual": 0.2 - 5.86e-6,
        }
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(solved_result, 0.1),
            ),
            patch.object(module, "extract_first_step", return_value=accepted_step),
        ):
            accepted, _ = module.run_voyage(
                voyage_id="voyage_test",
                loads_kw=np.array([100.0, 110.0]),
                times_s=np.array([0.0, 1.0]),
                case=case,
                config=config,
                initial_soc=0.2,
            )
        self.assertTrue(bool(accepted.iloc[0]["success"]))
        self.assertGreater(accepted.iloc[0]["soc_bound_residual"], module.N6_TOLERANCES["soc"])
        self.assertLess(accepted.iloc[0]["soc_bound_residual"], module.N6_STATE_COMMIT_TOLERANCES["soc"])
        self.assertAlmostEqual(accepted.iloc[0]["SOC_actual"], 0.2 - 5.86e-6)

    def test_run_voyage_uses_fixed_soc_reference_and_stops_on_final_failure(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        infeasible_result = SimpleNamespace(
            x=None,
            info=SimpleNamespace(
                status="primal infeasible",
                iter=100,
                prim_res=0.1,
                dual_res=0.2,
            ),
        )
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                return_value=(infeasible_result, 1.0),
            ) as solve,
            patch.object(module, "build_qp_problem", wraps=module.build_qp_problem) as build,
        ):
            controls, solver_rows = module.run_voyage(
                voyage_id="voyage_test",
                loads_kw=np.array([100.0, 110.0, 120.0, 130.0]),
                times_s=np.array([0.0, 1.0, 2.0, 3.0]),
                case=case,
                config=config,
            )

        self.assertEqual(solve.call_count, 1)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.call_args.kwargs["soc_reference"], SOC_REFERENCE)
        self.assertEqual(len(controls), 1)
        self.assertEqual(len(solver_rows), 1)
        self.assertEqual(controls.iloc[0]["config_id"], case.config_id)
        self.assertEqual(controls.iloc[0]["SOC_before"], SOC_REFERENCE)
        self.assertEqual(controls.iloc[0]["prev_fc_actual_kw"], 100.0)
        self.assertFalse(bool(controls.iloc[0]["success"]))
        self.assertTrue(np.isnan(controls.iloc[0]["SOC_actual"]))
        self.assertEqual(controls.iloc[0]["voyage_expected_steps"], 3)
        self.assertEqual(solver_rows.iloc[0]["status"], "primal infeasible")

    def test_run_voyage_rejects_nonfinite_or_out_of_bounds_initial_soc(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        for initial_soc in (float("nan"), float("inf"), 0.19, 0.81):
            with self.subTest(initial_soc=initial_soc):
                with (
                    patch.object(module, "_try_import_osqp") as try_import,
                    self.assertRaisesRegex(ValueError, "initial_soc"),
                ):
                    module.run_voyage(
                        voyage_id="voyage_test",
                        loads_kw=np.array([100.0, 110.0]),
                        times_s=np.array([0.0, 1.0]),
                        case=case,
                        config=config,
                        initial_soc=initial_soc,
                    )
                try_import.assert_not_called()

    def test_run_voyage_rejects_nonpositive_or_noninteger_max_steps(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        for max_steps in (0, -1, 1.5, True):
            with self.subTest(max_steps=max_steps):
                with (
                    patch.object(module, "_try_import_osqp") as try_import,
                    self.assertRaisesRegex(ValueError, "max_steps"),
                ):
                    module.run_voyage(
                        voyage_id="voyage_test",
                        loads_kw=np.array([100.0, 110.0]),
                        times_s=np.array([0.0, 1.0]),
                        case=case,
                        config=config,
                        max_steps=max_steps,
                    )
                try_import.assert_not_called()

    def test_max_iter_cold_restart_can_succeed_without_counting_initial_attempt_solved(
        self,
    ) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        max_iter_result = SimpleNamespace(
            x=np.zeros(19, dtype=float),
            info=SimpleNamespace(
                status="maximum iterations reached",
                iter=20000,
                prim_res=0.1,
                dual_res=0.2,
            ),
        )
        normalized_solution = np.zeros(19, dtype=float)
        normalized_solution[0] = 110.0 / 560.0
        solved_result = SimpleNamespace(
            x=normalized_solution,
            info=SimpleNamespace(
                status="solved",
                iter=75,
                prim_res=1.0e-8,
                dual_res=2.0e-8,
            ),
        )

        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(
                module,
                "_setup_n6_osqp_solver",
                side_effect=["warm_solver", "cold_solver"],
            ) as setup,
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                side_effect=[(max_iter_result, 2.0), (solved_result, 0.5)],
            ),
        ):
            controls, solver_rows = module.run_voyage(
                voyage_id="voyage_test",
                loads_kw=np.array([100.0, 110.0]),
                times_s=np.array([0.0, 1.0]),
                case=case,
                config=config,
            )

        self.assertEqual(setup.call_count, 2)
        self.assertTrue(bool(controls.iloc[0]["success"]))
        self.assertEqual(solver_rows.iloc[0]["initial_status"], "maximum iterations reached")
        self.assertEqual(solver_rows.iloc[0]["status"], "solved")
        self.assertTrue(bool(solver_rows.iloc[0]["max_iter_reached"]))
        self.assertTrue(bool(solver_rows.iloc[0]["cold_restart_used"]))
        self.assertTrue(bool(solver_rows.iloc[0]["cold_restart_succeeded"]))
        self.assertEqual(solver_rows.iloc[0]["attempt_count"], 2)

    def test_two_successful_steps_refresh_previous_fc_soc_and_osqp_updates(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.SensitivityCase(
            config_id="synthetic_weights",
            varied_weight=None,
            weight_value=1.0,
            q_h2=0.25,
            q_batt=0.5,
            q_soc=2.0,
            q_fc_var=4.0,
        )
        config = module.four_objective_config(case)
        solved_result = SimpleNamespace(
            x=np.zeros(19, dtype=float),
            info=SimpleNamespace(
                status="solved",
                iter=25,
                prim_res=1.0e-8,
                dual_res=1.0e-8,
            ),
        )
        applied_steps = [
            {
                "P_fc_plan_kw": 110.0,
                "P_batt_plan_kw": 50.0,
                "SOC_predicted": 0.54,
                "P_fc_actual_kw": 110.0,
                "P_batt_actual_kw": 50.0,
                "SOC_actual": 0.54,
            },
            {
                "P_fc_plan_kw": 130.0,
                "P_batt_plan_kw": -20.0,
                "SOC_predicted": 0.55,
                "P_fc_actual_kw": 130.0,
                "P_batt_actual_kw": -20.0,
                "SOC_actual": 0.55,
            },
        ]

        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(module, "_setup_n6_osqp_solver", return_value="solver"),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                side_effect=[(solved_result, 0.1), (solved_result, 0.2)],
            ) as solve,
            patch.object(
                module,
                "extract_first_step",
                side_effect=applied_steps,
            ) as extract,
        ):
            controls, solver_rows = module.run_voyage(
                voyage_id="voyage_synthetic",
                loads_kw=np.array([100.0, 160.0, 110.0]),
                times_s=np.array([0.0, 1.0, 2.0]),
                case=case,
                config=config,
            )

        self.assertTrue(controls["success"].all())
        self.assertTrue(solver_rows["success"].all())
        self.assertEqual(list(controls["prev_fc_actual_kw"]), [100.0, 110.0])
        self.assertEqual(list(controls["SOC_before"]), [0.55, 0.54])
        self.assertEqual(extract.call_args_list[0].kwargs["current_soc"], 0.55)
        self.assertEqual(extract.call_args_list[1].kwargs["current_soc"], 0.54)

        first_update = solve.call_args_list[0].kwargs
        second_update = solve.call_args_list[1].kwargs
        self.assertFalse(np.array_equal(first_update["linear"], second_update["linear"]))
        self.assertFalse(np.array_equal(first_update["lower"], second_update["lower"]))
        self.assertFalse(np.array_equal(first_update["upper"], second_update["upper"]))

        h2_reference = module.physical_h2_kg_step(config, config.fuel_cell_max_kw)
        expected_raw = {
            "p_batt_sq_kw2_step": [50.0**2, (-20.0) ** 2],
            "soc_error_sq_step": [(0.54 - 0.55) ** 2, 0.0],
            "fc_delta_sq_kw2_step": [10.0**2, 20.0**2],
        }
        for column, expected in expected_raw.items():
            np.testing.assert_allclose(controls[column], expected, rtol=0.0, atol=1.0e-15)
        np.testing.assert_allclose(
            controls["J_h2_norm_step"],
            controls["h2_kg_step"] / h2_reference,
        )
        np.testing.assert_allclose(
            controls["J_batt_norm_step"],
            np.array(expected_raw["p_batt_sq_kw2_step"]) / 346.5**2,
        )
        np.testing.assert_allclose(
            controls["J_soc_norm_step"],
            np.array(expected_raw["soc_error_sq_step"]) / 0.05**2,
        )
        np.testing.assert_allclose(
            controls["J_fc_var_norm_step"],
            np.array(expected_raw["fc_delta_sq_kw2_step"]) / 48.0**2,
        )
        weighted_columns = {
            "weighted_h2_contribution_step": ("J_h2_norm_step", case.q_h2),
            "weighted_batt_contribution_step": ("J_batt_norm_step", case.q_batt),
            "weighted_soc_contribution_step": ("J_soc_norm_step", case.q_soc),
            "weighted_fc_var_contribution_step": ("J_fc_var_norm_step", case.q_fc_var),
        }
        for weighted, (normalized, weight) in weighted_columns.items():
            np.testing.assert_allclose(controls[weighted], controls[normalized] * weight)
        np.testing.assert_allclose(
            controls["total_weighted_objective_step"],
            controls[list(weighted_columns)].sum(axis=1),
        )
        for normalized, cumulative in (
            ("J_h2_norm_step", "cum_J_h2_norm"),
            ("J_batt_norm_step", "cum_J_batt_norm"),
            ("J_soc_norm_step", "cum_J_soc_norm"),
            ("J_fc_var_norm_step", "cum_J_fc_var_norm"),
        ):
            np.testing.assert_allclose(controls[cumulative], controls[normalized].cumsum())


class TestSensitivityMetrics(unittest.TestCase):
    @staticmethod
    def _synthetic_two_step_frames(
        module,
        case,
        config,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        h2_steps = np.array([0.001, 0.002], dtype=float)
        batt_sq = np.array([50.0**2, (-20.0) ** 2], dtype=float)
        soc_sq = np.array([(0.54 - 0.55) ** 2, 0.0], dtype=float)
        fc_delta_sq = np.array([10.0**2, 20.0**2], dtype=float)
        h2_reference = module.physical_h2_kg_step(config, config.fuel_cell_max_kw)
        normalized = {
            "J_h2_norm_step": h2_steps / h2_reference,
            "J_batt_norm_step": batt_sq / config.battery_power_ref_kw**2,
            "J_soc_norm_step": soc_sq / config.soc_band**2,
            "J_fc_var_norm_step": fc_delta_sq
            / module.resolved_ramp_kw_per_step(config) ** 2,
        }
        controls = pd.DataFrame(
            {
                "config_id": case.config_id,
                "voyage_id": "voyage_synthetic",
                "voyage_expected_steps": 2,
                "decision_index": [0, 1],
                "execution_index": [1, 2],
                "SOC_before": [0.55, 0.54],
                "SOC_actual": [0.54, 0.55],
                "P_fc_actual_kw": [110.0, 130.0],
                "P_batt_actual_kw": [50.0, -20.0],
                "fc_delta_actual_kw": [10.0, 20.0],
                "actual_balance_residual_kw": [0.0, 0.01],
                "h2_kg_step": h2_steps,
                "p_batt_sq_kw2_step": batt_sq,
                "soc_error_sq_step": soc_sq,
                "fc_delta_sq_kw2_step": fc_delta_sq,
                **normalized,
                "success": [True, True],
            }
        )
        solver_rows = pd.DataFrame(
            {
                "voyage_id": "voyage_synthetic",
                "decision_index": ["0", "1"],
                "execution_index": [1.0, 2.0],
                "status": ["solved", "solved"],
                "success": [1, 1],
                "max_iter_reached": [False, False],
                "solve_ms": [1.0, 3.0],
                "time_s": [1.0, 2.0],
            }
        )
        return controls, solver_rows

    def test_voyage_metrics_sum_raw_normalized_and_weighted_objectives(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.SensitivityCase(
            config_id="synthetic_weights",
            varied_weight=None,
            weight_value=1.0,
            q_h2=0.25,
            q_batt=0.5,
            q_soc=2.0,
            q_fc_var=4.0,
        )
        config = module.four_objective_config(case)
        controls, solver_rows = self._synthetic_two_step_frames(module, case, config)

        metrics = module.build_voyage_metrics(
            controls,
            solver_rows,
            case=case,
            config=config,
        )

        self.assertTrue(metrics["completed"])
        self.assertTrue(metrics["metrics_comparable"])
        self.assertEqual(metrics["sum_p_batt_sq_kw2"], 50.0**2 + (-20.0) ** 2)
        self.assertAlmostEqual(metrics["sum_soc_error_sq"], (0.54 - 0.55) ** 2)
        self.assertEqual(metrics["sum_fc_delta_sq_kw2"], 10.0**2 + 20.0**2)
        self.assertAlmostEqual(
            metrics["J_batt_norm"], metrics["sum_p_batt_sq_kw2"] / 346.5**2
        )
        self.assertAlmostEqual(
            metrics["J_soc_norm"], metrics["sum_soc_error_sq"] / 0.05**2
        )
        self.assertAlmostEqual(
            metrics["J_fc_var_norm"], metrics["sum_fc_delta_sq_kw2"] / 48.0**2
        )
        self.assertAlmostEqual(
            metrics["total_weighted_objective"],
            metrics["weighted_h2_contribution"]
            + metrics["weighted_batt_contribution"]
            + metrics["weighted_soc_contribution"]
            + metrics["weighted_fc_var_contribution"],
        )
        self.assertEqual(
            (metrics["expected_step_count"], metrics["attempted_step_count"]),
            (2, 2),
        )
        self.assertEqual(metrics["applied_step_count"], 2)
        self.assertEqual((metrics["initial_soc"], metrics["final_soc"]), (0.55, 0.55))
        self.assertEqual((metrics["max_fc_kw"], metrics["min_fc_kw"]), (130.0, 110.0))
        self.assertEqual(metrics["max_batt_discharge_kw"], 50.0)
        self.assertEqual(metrics["max_batt_charge_kw"], 20.0)

    def test_voyage_metrics_reject_missing_control_rows_as_noncomparable(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        controls, solver_rows = self._synthetic_two_step_frames(module, case, config)

        metrics = module.build_voyage_metrics(
            controls.iloc[:1].copy(),
            solver_rows,
            case=case,
            config=config,
        )

        self.assertFalse(metrics["completed"])
        self.assertFalse(metrics["metrics_comparable"])

    def test_voyage_metrics_require_control_solver_row_alignment(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        mismatch_cases = {
            "voyage_id": "other_voyage",
            "decision_index": "99",
            "execution_index": 99,
            "success": 0,
        }
        for column, bad_value in mismatch_cases.items():
            with self.subTest(column=column):
                controls, solver_rows = self._synthetic_two_step_frames(
                    module, case, config
                )
                solver_rows.loc[1, column] = bad_value

                metrics = module.build_voyage_metrics(
                    controls,
                    solver_rows,
                    case=case,
                    config=config,
                )

                self.assertFalse(metrics["completed"])
                self.assertFalse(metrics["metrics_comparable"])

    def test_final_max_iter_failure_is_recorded_as_nan_and_prefix_noncomparable(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        normalized_solution = np.zeros(19, dtype=float)
        normalized_solution[0] = 100.0 / 560.0
        solved_result = SimpleNamespace(
            x=normalized_solution,
            info=SimpleNamespace(
                status="solved",
                iter=25,
                prim_res=1.0e-8,
                dual_res=1.0e-8,
            ),
        )
        max_iter_result = SimpleNamespace(
            x=np.zeros(19, dtype=float),
            info=SimpleNamespace(
                status="maximum iterations reached",
                iter=20000,
                prim_res=0.1,
                dual_res=0.2,
            ),
        )
        first_soc = 0.55 - 10.0 / (3600.0 * 693.0)
        first_applied = {
            "P_fc_plan_kw": 100.0,
            "P_batt_plan_kw": 10.0,
            "SOC_predicted": first_soc,
            "P_fc_actual_kw": 100.0,
            "P_batt_actual_kw": 10.0,
            "SOC_actual": first_soc,
        }
        with (
            patch.object(module, "_try_import_osqp", return_value=(object(), None)),
            patch.object(
                module,
                "_setup_n6_osqp_solver",
                side_effect=["warm_solver", "cold_solver"],
            ),
            patch.object(
                module,
                "_solve_with_persistent_osqp",
                side_effect=[
                    (solved_result, 0.5),
                    (max_iter_result, 1.0),
                    (max_iter_result, 2.0),
                ],
            ) as solve,
            patch.object(module, "extract_first_step", return_value=first_applied),
        ):
            controls, solver_rows = module.run_voyage(
                voyage_id="voyage_failed",
                loads_kw=np.array([100.0, 110.0, 120.0]),
                times_s=np.array([0.0, 1.0, 2.0]),
                case=case,
                config=config,
            )

        self.assertEqual(solve.call_count, 3)
        self.assertEqual((len(controls), len(solver_rows)), (2, 2))
        self.assertTrue(bool(controls.iloc[0]["success"]))
        self.assertFalse(bool(controls.iloc[1]["success"]))
        self.assertEqual(controls.iloc[1]["SOC_before"], controls.iloc[0]["SOC_actual"])
        self.assertEqual(solver_rows.iloc[1]["attempt_count"], 2)
        self.assertFalse(bool(solver_rows.iloc[1]["cold_restart_succeeded"]))
        failed_nan_columns = [
            "P_fc_plan_kw",
            "P_batt_plan_kw",
            "SOC_predicted",
            "P_fc_actual_kw",
            "P_batt_actual_kw",
            "SOC_actual",
            "h2_kg_step",
            "p_batt_sq_kw2_step",
            "soc_error_sq_step",
            "fc_delta_sq_kw2_step",
            "J_h2_norm_step",
            "J_batt_norm_step",
            "J_soc_norm_step",
            "J_fc_var_norm_step",
            "weighted_h2_contribution_step",
            "weighted_batt_contribution_step",
            "weighted_soc_contribution_step",
            "weighted_fc_var_contribution_step",
            "total_weighted_objective_step",
            "cum_J_h2_norm",
            "cum_J_batt_norm",
            "cum_J_soc_norm",
            "cum_J_fc_var_norm",
        ]
        self.assertTrue(controls.loc[1, failed_nan_columns].isna().all())

        metrics = module.build_voyage_metrics(
            controls,
            solver_rows,
            case=case,
            config=config,
        )
        self.assertFalse(metrics["completed"])
        self.assertFalse(metrics["metrics_comparable"])
        self.assertEqual(metrics["solver_failure_count"], 1)
        self.assertEqual(metrics["max_iter_count"], 1)
        self.assertEqual(metrics["applied_step_count"], 1)
        self.assertEqual(metrics["final_soc"], controls.iloc[0]["SOC_actual"])
        self.assertAlmostEqual(
            metrics["total_weighted_objective"],
            controls.iloc[0]["total_weighted_objective_step"],
        )

    def test_configuration_summary_sums_metrics_uses_combined_timing_and_has_no_rank(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        controls, solver_rows = self._synthetic_two_step_frames(module, case, config)
        first = module.build_voyage_metrics(
            controls,
            solver_rows,
            case=case,
            config=config,
        )
        second = dict(first)
        second.update(
            {
                "voyage_id": "voyage_failed",
                "completed": False,
                "metrics_comparable": False,
                "solver_failure_count": 1,
                "primal_infeasible_count": 1,
                "max_iter_count": 1,
                "attempted_step_count": 1,
                "applied_step_count": 1,
            }
        )
        voyage_metrics = pd.DataFrame([first, second])
        combined_solver_rows = pd.DataFrame(
            {
                "solve_ms": [1.0, 9.0, 5.0],
                "success": [True, True, False],
            }
        )

        summary = module.build_configuration_summary(
            voyage_metrics,
            combined_solver_rows,
            case=case,
        )

        self.assertEqual(summary["voyage_count"], 2)
        self.assertEqual(summary["completed_voyage_count"], 1)
        self.assertEqual(summary["completion_rate"], 0.5)
        self.assertFalse(summary["metrics_comparable"])
        self.assertEqual(summary["mean_solve_time_ms"], 5.0)
        self.assertEqual(summary["max_solve_time_ms"], 9.0)
        self.assertAlmostEqual(summary["p95_solve_time_ms"], 8.6)
        for name in (
            "solver_failure_count",
            "primal_infeasible_count",
            "max_iter_count",
            "total_h2_kg",
            "sum_p_batt_sq_kw2",
            "sum_soc_error_sq",
            "sum_fc_delta_sq_kw2",
            "J_h2_norm",
            "J_batt_norm",
            "J_soc_norm",
            "J_fc_var_norm",
            "weighted_h2_contribution",
            "weighted_batt_contribution",
            "weighted_soc_contribution",
            "weighted_fc_var_contribution",
            "total_weighted_objective",
        ):
            self.assertAlmostEqual(summary[name], voyage_metrics[name].sum())
        for forbidden in ("selected", "score", "rank", "winner", "best"):
            self.assertFalse(any(forbidden in name.lower() for name in summary))

    def test_evaluate_configuration_groups_voyages_in_sorted_order(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        data = pd.DataFrame(
            {
                "voyage_id": ["voyage_b", "voyage_b", "voyage_a", "voyage_a"],
                "time_s": [0.0, 1.0, 0.0, 1.0],
                "load_total_kw": [200.0, 210.0, 100.0, 110.0],
            }
        )

        def fake_run_voyage(**kwargs):
            voyage_id = kwargs["voyage_id"]
            return (
                pd.DataFrame({"voyage_id": [voyage_id], "marker": ["control"]}),
                pd.DataFrame({"voyage_id": [voyage_id], "marker": ["solver"]}),
            )

        def fake_metrics(controls, solver_rows, **_kwargs):
            self.assertEqual(controls.iloc[0]["voyage_id"], solver_rows.iloc[0]["voyage_id"])
            return {"voyage_id": controls.iloc[0]["voyage_id"], "completed": True}

        with (
            patch.object(module, "run_voyage", side_effect=fake_run_voyage) as run,
            patch.object(module, "build_voyage_metrics", side_effect=fake_metrics) as metrics,
            patch.object(
                module,
                "build_configuration_summary",
                return_value={"config_id": case.config_id},
            ) as summarize,
        ):
            result = module.evaluate_configuration(case, data=data)

        self.assertEqual(
            [call.kwargs["voyage_id"] for call in run.call_args_list],
            ["voyage_a", "voyage_b"],
        )
        self.assertEqual(metrics.call_count, 2)
        summarize.assert_called_once()
        self.assertEqual(list(result["controls"]["voyage_id"]), ["voyage_a", "voyage_b"])
        self.assertEqual(
            list(result["solver_rows"]["voyage_id"]), ["voyage_a", "voyage_b"]
        )
        self.assertEqual(
            list(result["voyage_metrics"]["voyage_id"]), ["voyage_a", "voyage_b"]
        )
        self.assertIs(result["case"], case)
        self.assertEqual(result["summary"], {"config_id": case.config_id})


class TestSensitivityArtifactsAndCli(unittest.TestCase):
    @staticmethod
    def _synthetic_result(module, voyages: tuple[str, ...]) -> dict:
        case = module.build_sensitivity_cases()[0]
        config = module.four_objective_config(case)
        metrics_rows = []
        control_rows = []
        solver_rows = []
        for index, voyage_id in enumerate(voyages):
            total_h2 = 0.01 + index * 0.001
            batt_sq = 100.0 + index
            soc_sq = 0.001 + index * 0.0001
            fc_sq = 25.0 + index
            j_h2 = total_h2 / module.physical_h2_kg_step(
                config, config.fuel_cell_max_kw
            )
            j_batt = batt_sq / config.battery_power_ref_kw**2
            j_soc = soc_sq / config.soc_band**2
            j_fc = fc_sq / module.resolved_ramp_kw_per_step(config) ** 2
            solve_ms = 1.0 + index
            metrics_rows.append(
                {
                    "config_id": case.config_id,
                    "q_h2": case.q_h2,
                    "q_batt": case.q_batt,
                    "q_soc": case.q_soc,
                    "q_fc_var": case.q_fc_var,
                    "voyage_id": voyage_id,
                    "completed": True,
                    "solver_failure_count": 0,
                    "primal_infeasible_count": 0,
                    "max_iter_count": 0,
                    "first_failure_time_s": np.nan,
                    "mean_solve_time_ms": solve_ms,
                    "p95_solve_time_ms": solve_ms,
                    "max_solve_time_ms": solve_ms,
                    "initial_soc": 0.55,
                    "final_soc": 0.549 - index * 0.0001,
                    "delta_soc": -0.001 - index * 0.0001,
                    "min_soc": 0.549 - index * 0.0001,
                    "max_soc": 0.55,
                    "max_power_balance_residual_kw": 0.0,
                    "max_fc_ramp_kw_per_step": 10.0,
                    "max_fc_kw": 210.0,
                    "min_fc_kw": 190.0,
                    "max_batt_discharge_kw": 20.0,
                    "max_batt_charge_kw": 5.0,
                    "total_h2_kg": total_h2,
                    "sum_p_batt_sq_kw2": batt_sq,
                    "sum_soc_error_sq": soc_sq,
                    "sum_fc_delta_sq_kw2": fc_sq,
                    "J_h2_norm": j_h2,
                    "J_batt_norm": j_batt,
                    "J_soc_norm": j_soc,
                    "J_fc_var_norm": j_fc,
                    "weighted_h2_contribution": j_h2,
                    "weighted_batt_contribution": j_batt,
                    "weighted_soc_contribution": j_soc,
                    "weighted_fc_var_contribution": j_fc,
                    "total_weighted_objective": j_h2 + j_batt + j_soc + j_fc,
                    "expected_step_count": 2,
                    "attempted_step_count": 2,
                    "applied_step_count": 2,
                    "first_failure_status": "",
                    "metrics_comparable": True,
                }
            )
            for step in range(2):
                control_rows.append(
                    {
                        "voyage_id": voyage_id,
                        "time_s": float(step + 1),
                        "load_actual_kw": 200.0 + step,
                        "P_fc_actual_kw": 190.0 + step,
                        "P_batt_actual_kw": 10.0,
                        "SOC_actual": 0.55 - (step + 1) * 0.0001,
                        "cum_J_h2_norm": 0.1 * (step + 1),
                        "cum_J_batt_norm": 0.2 * (step + 1),
                        "cum_J_soc_norm": 0.3 * (step + 1),
                        "cum_J_fc_var_norm": 0.4 * (step + 1),
                        "success": True,
                        "status": "solved",
                    }
                )
            solver_rows.extend([{"solve_ms": solve_ms}, {"solve_ms": solve_ms}])
        voyage_metrics = pd.DataFrame(metrics_rows)
        solver_frame = pd.DataFrame(solver_rows)
        return {
            "case": case,
            "config": config,
            "controls": pd.DataFrame(control_rows),
            "solver_rows": solver_frame,
            "voyage_metrics": voyage_metrics,
            "summary": module.build_configuration_summary(
                voyage_metrics, solver_frame, case=case
            ),
        }

    @classmethod
    def _summary_table(cls, module) -> pd.DataFrame:
        baseline = cls._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)[
            "summary"
        ]
        rows = []
        for index, case in enumerate(module.build_sensitivity_cases()):
            row = dict(baseline)
            row.update(
                {
                    "config_id": case.config_id,
                    "varied_weight": case.varied_weight or "baseline",
                    "weight_value": case.weight_value,
                    "q_h2": case.q_h2,
                    "q_batt": case.q_batt,
                    "q_soc": case.q_soc,
                    "q_fc_var": case.q_fc_var,
                    "total_h2_kg": baseline["total_h2_kg"] + index,
                    "sum_p_batt_sq_kw2": baseline["sum_p_batt_sq_kw2"] + index,
                    "sum_soc_error_sq": baseline["sum_soc_error_sq"] + index,
                    "sum_fc_delta_sq_kw2": baseline["sum_fc_delta_sq_kw2"] + index,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _fake_plot(_frame, path, **_kwargs) -> None:
        Path(path).write_bytes(b"plot")

    def test_prepare_case_dir_guards_paths_and_overwrites_only_exact_case(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        case = module.build_sensitivity_cases()[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "output"
            case_dir = module.prepare_case_dir(root, case, overwrite=False)
            marker = case_dir / "old.txt"
            marker.write_text("old", encoding="utf-8")
            sibling = root / "keep"
            sibling.mkdir()
            (sibling / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                module.prepare_case_dir(root, case, overwrite=False)
            recreated = module.prepare_case_dir(root, case, overwrite=True)
            self.assertEqual(recreated, case_dir)
            self.assertFalse(marker.exists())
            self.assertTrue((sibling / "keep.txt").is_file())

            escaped = replace(case, config_id="../escaped")
            with self.assertRaisesRegex(ValueError, "escaped"):
                module.prepare_case_dir(root, escaped, overwrite=False)
            diagnostic = module.prepare_case_dir(
                root,
                replace(case, config_id="diagnostic_case"),
                overwrite=False,
                diagnostic_voyage="voyage_060",
            )
            self.assertEqual(
                diagnostic.parent,
                (root / "diagnostics" / "voyage_060").resolve(),
            )

    def test_configuration_artifacts_are_compact_and_metadata_is_complete(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"frozen-input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "artifacts",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )
            self.assertEqual(
                {path.name for path in case_dir.iterdir()},
                {"config.json", "voyage_metrics.csv", "plots"},
            )
            self.assertEqual(
                {path.name for path in (case_dir / "plots").iterdir()},
                {
                    f"{voyage}_power_soc_objectives.png"
                    for voyage in module.EXPECTED_TEST_VOYAGES
                },
            )
            metadata = json.loads((case_dir / "config.json").read_text("utf-8"))
            self.assertEqual(metadata["config_id"], "baseline_1_1_1_1")
            self.assertEqual(set(metadata["weights"]), set(module.WEIGHT_NAMES))
            self.assertEqual(metadata["model"]["objective_variant"], module.OBJECTIVE_VARIANT)
            self.assertEqual(metadata["soc_reference"], 0.55)
            self.assertEqual(metadata["qp_metadata"]["objective_variant"], module.OBJECTIVE_VARIANT)
            self.assertFalse(Path(metadata["input_path"]).is_absolute())
            self.assertEqual(metadata["input_sha256"], module.sha256_file(input_path))
            self.assertEqual(len(metadata["implementation_sha256"]), 64)
            self.assertEqual(metadata["voyages"], list(module.EXPECTED_TEST_VOYAGES))
            self.assertTrue(metadata["formal_complete"])
            self.assertFalse(metadata["lstm_used"])
            self.assertFalse(metadata["dqn_used"])
            self.assertEqual(
                metadata["forecast"],
                "t+1..t+6 actual natural-clipped spline load where available",
            )
            self.assertEqual(
                metadata["forecast_tail_policy"],
                "same-voyage final sample edge-hold; never crosses voyage boundary",
            )
            self.assertEqual(metadata["state_commit_tolerances"]["soc"], 1.0e-5)
            self.assertTrue(metadata["first_move_only"])
            self.assertEqual(metadata["configuration_summary"], result["summary"])
            written_metrics = pd.read_csv(case_dir / "voyage_metrics.csv")
            self.assertIn("configuration_p95_solve_time_ms", written_metrics.columns)
            np.testing.assert_allclose(
                written_metrics["configuration_p95_solve_time_ms"],
                result["summary"]["p95_solve_time_ms"],
                rtol=0.0,
                atol=0.0,
            )

    def test_implementation_hash_covers_runtime_qp_and_hydrogen_dependencies(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        paths = []

        def fake_hash(path):
            paths.append(Path(path).resolve())
            return "0" * 64

        with patch.object(module, "sha256_file", side_effect=fake_hash):
            digest = module._implementation_sha256()

        self.assertEqual(len(digest), 64)
        expected = {
            Path(module.__file__).resolve(),
            module.REPO_ROOT / "src/main/mpc_solvers/mpc_qp_formulation.py",
            module.REPO_ROOT / "src/main/benchmark_mpc_qp_osqp_1s.py",
            module.REPO_ROOT / "src/mpc/solvers/fc_dp0_curve.py",
            module.REPO_ROOT / "data/fuel_cell/FC_Dp0_curve_for_Python.csv",
        }
        self.assertEqual(set(paths), {path.resolve() for path in expected})

    def test_diagnostic_artifact_has_one_plot_and_is_incomplete(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, ("voyage_060",))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage="voyage_060",
                )
            self.assertEqual(
                case_dir.parent,
                (root / "output" / "diagnostics" / "voyage_060").resolve(),
            )
            self.assertEqual(len(list((case_dir / "plots").iterdir())), 1)
            self.assertFalse(
                json.loads((case_dir / "config.json").read_text("utf-8"))[
                    "formal_complete"
                ]
            )

    def test_load_matching_case_round_trips_summary_and_rejects_stale_metadata(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )
            loaded = module.load_matching_case(
                case_dir,
                case=result["case"],
                input_path=input_path,
                expected_voyages=module.EXPECTED_TEST_VOYAGES,
                formal_complete=True,
            )
            self.assertEqual(loaded, result["summary"])

            metadata_path = case_dir / "config.json"
            original = json.loads(metadata_path.read_text("utf-8"))
            mutations = {
                "weight": lambda value: value["weights"].__setitem__("q_h2", 0.25),
                "objective": lambda value: value["model"].__setitem__(
                    "objective_variant", "stale"
                ),
                "input": lambda value: value.__setitem__("input_sha256", "0" * 64),
                "implementation": lambda value: value.__setitem__(
                    "implementation_sha256", "0" * 64
                ),
                "complete": lambda value: value.__setitem__("formal_complete", False),
                "voyages": lambda value: value.__setitem__(
                    "voyages", list(reversed(value["voyages"]))
                ),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    changed = json.loads(json.dumps(original))
                    mutate(changed)
                    metadata_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        module.load_matching_case(
                            case_dir,
                            case=result["case"],
                            input_path=input_path,
                            expected_voyages=module.EXPECTED_TEST_VOYAGES,
                            formal_complete=True,
                        )
            metadata_path.write_text(json.dumps(original), encoding="utf-8")

            metrics_path = case_dir / "voyage_metrics.csv"
            original_metrics = metrics_path.read_text("utf-8")
            result["voyage_metrics"].iloc[:-1].to_csv(metrics_path, index=False)
            with self.assertRaises(ValueError):
                module.load_matching_case(
                    case_dir,
                    case=result["case"],
                    input_path=input_path,
                    expected_voyages=module.EXPECTED_TEST_VOYAGES,
                    formal_complete=True,
                )
            metrics_path.write_text(original_metrics, encoding="utf-8")
            plot = case_dir / "plots" / "voyage_066_power_soc_objectives.png"
            plot.unlink()
            with self.assertRaises(ValueError):
                module.load_matching_case(
                    case_dir,
                    case=result["case"],
                    input_path=input_path,
                    expected_voyages=module.EXPECTED_TEST_VOYAGES,
                    formal_complete=True,
                )

    def test_reuse_cross_checks_exact_p95_between_csv_and_json(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )
            metadata_path = case_dir / "config.json"
            metrics_path = case_dir / "voyage_metrics.csv"
            original_metadata = json.loads(metadata_path.read_text("utf-8"))
            original_metrics = pd.read_csv(metrics_path)

            changed_metadata = json.loads(json.dumps(original_metadata))
            changed_metadata["configuration_summary"]["p95_solve_time_ms"] += 1.0
            metadata_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "p95"):
                module.load_matching_case(
                    case_dir,
                    case=result["case"],
                    input_path=input_path,
                    expected_voyages=module.EXPECTED_TEST_VOYAGES,
                    formal_complete=True,
                )
            metadata_path.write_text(json.dumps(original_metadata), encoding="utf-8")

            for name, bad_value in (("inconsistent", 99.0), ("infinite", np.inf)):
                with self.subTest(name=name):
                    changed_metrics = original_metrics.copy()
                    changed_metrics.loc[0, "configuration_p95_solve_time_ms"] = bad_value
                    changed_metrics.to_csv(metrics_path, index=False)
                    with self.assertRaisesRegex(ValueError, "p95"):
                        module.load_matching_case(
                            case_dir,
                            case=result["case"],
                            input_path=input_path,
                            expected_voyages=module.EXPECTED_TEST_VOYAGES,
                            formal_complete=True,
                        )

    def test_reuse_preserves_long_p95_through_csv_round_trip(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        exact_p95 = 0.46425501350313453
        result["summary"]["p95_solve_time_ms"] = exact_p95
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )

            loaded = module.load_matching_case(
                case_dir,
                case=result["case"],
                input_path=input_path,
                expected_voyages=module.EXPECTED_TEST_VOYAGES,
                formal_complete=True,
            )

        self.assertEqual(loaded["p95_solve_time_ms"], exact_p95)

    def test_reuse_rejects_missing_nonnumeric_infinite_or_negative_metrics(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )
            metrics_path = case_dir / "voyage_metrics.csv"
            original = pd.read_csv(metrics_path)
            mutations = (
                ("solver_failure_count", np.nan, "finite numeric"),
                ("mean_solve_time_ms", "not-a-number", "finite numeric"),
                ("initial_soc", np.inf, "finite numeric"),
                ("expected_step_count", np.inf, "finite numeric"),
                ("attempted_step_count", -1, "non-negative"),
                ("applied_step_count", np.nan, "finite numeric"),
            )
            for column, bad_value, message in mutations:
                with self.subTest(column=column):
                    changed = original.copy()
                    if isinstance(bad_value, str):
                        changed[column] = changed[column].astype(object)
                    elif not np.isfinite(float(bad_value)):
                        changed[column] = changed[column].astype(float)
                    changed.loc[0, column] = bad_value
                    changed.to_csv(metrics_path, index=False)
                    with self.assertRaisesRegex(ValueError, message):
                        module.load_matching_case(
                            case_dir,
                            case=result["case"],
                            input_path=input_path,
                            expected_voyages=module.EXPECTED_TEST_VOYAGES,
                            formal_complete=True,
                        )

    def test_reuse_allows_conditional_physical_nan_when_no_step_was_applied(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        result = self._synthetic_result(module, module.EXPECTED_TEST_VOYAGES)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")
            with patch.object(module, "write_voyage_plot", side_effect=self._fake_plot):
                case_dir = module.write_configuration_artifacts(
                    result,
                    output_dir=root / "output",
                    input_path=input_path,
                    overwrite=False,
                    diagnostic_voyage=None,
                )
            metrics_path = case_dir / "voyage_metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metrics.loc[0, "applied_step_count"] = 0
            for column in (
                "max_power_balance_residual_kw",
                "max_fc_ramp_kw_per_step",
                "max_fc_kw",
                "min_fc_kw",
                "max_batt_discharge_kw",
                "max_batt_charge_kw",
            ):
                metrics.loc[0, column] = np.nan
            metrics.to_csv(metrics_path, index=False)

            loaded = module.load_matching_case(
                case_dir,
                case=result["case"],
                input_path=input_path,
                expected_voyages=module.EXPECTED_TEST_VOYAGES,
                formal_complete=True,
            )
            self.assertEqual(loaded, result["summary"])

    def test_voyage_plot_has_four_axes_and_marks_failure_on_every_axis(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        frame = self._synthetic_result(module, ("voyage_060",))["controls"]
        frame.loc[1, "success"] = False
        frame.loc[1, "status"] = "maximum iterations reached"
        captured = []
        original_subplots = module.plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append((figure, axes))
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module.plt, "subplots", side_effect=capture
        ):
            output = Path(temporary) / "plot.png"
            module.write_voyage_plot(frame, output, config_id="baseline_1_1_1_1")
            self.assertTrue(output.is_file())
        figure, axes = captured[0]
        self.assertEqual(len(figure.axes), 4)
        for axis in np.asarray(axes).reshape(-1):
            self.assertTrue(
                any(
                    len(np.asarray(line.get_xdata()).reshape(-1)) == 2
                    and np.allclose(line.get_xdata(), [2.0, 2.0])
                    for line in axis.lines
                )
            )

    def test_voyage_plot_closes_figure_when_save_fails(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        frame = self._synthetic_result(module, ("voyage_060",))["controls"]
        module.plt.close("all")
        before = module.plt.get_fignums()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "matplotlib.figure.Figure.savefig", side_effect=RuntimeError("save failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                module.write_voyage_plot(
                    frame,
                    Path(temporary) / "plot.png",
                    config_id="baseline_1_1_1_1",
                )
        self.assertEqual(module.plt.get_fignums(), before)

    def test_summary_outputs_have_exact_schema_paths_and_no_selection_fields(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        captured = []
        original_subplots = module.plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module.plt, "subplots", side_effect=capture
        ):
            root = Path(temporary)
            module.write_summary_figures(table, root / "summary")
            self.assertEqual(
                {path.name for path in (root / "summary").iterdir()},
                {f"{name}_sensitivity.png" for name in module.WEIGHT_NAMES},
            )
            self.assertEqual([len(figure.axes) for figure in captured], [6] * 4)
            table_path = root / "table.csv"
            report_path = root / "summary.md"
            module.write_summary_table(table, table_path)
            module.write_summary_report(table, report_path)
            written = pd.read_csv(table_path)
            self.assertEqual(len(written), 17)
            self.assertEqual(len(written[list(module.WEIGHT_NAMES)].drop_duplicates()), 17)
            for forbidden in ("selected", "score", "rank", "winner", "best"):
                self.assertFalse(any(forbidden in name.lower() for name in written.columns))
            report = report_path.read_text("utf-8").lower()
            self.assertIn("offline oracle", report)
            self.assertIn("no lstm", report)
            self.assertIn("no dqn", report)
            self.assertIn("no automatic best", report)
            self.assertIn("edge-hold", report)
            with self.assertRaises(ValueError):
                module.write_summary_table(table.assign(score=1.0), root / "bad.csv")

    def test_summary_report_requires_explicit_voyage_count_columns(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        with tempfile.TemporaryDirectory() as temporary:
            for column in ("voyage_count", "completed_voyage_count"):
                with self.subTest(column=column), self.assertRaisesRegex(
                    ValueError, column
                ):
                    module.write_summary_report(
                        table.drop(columns=column),
                        Path(temporary) / "summary.md",
                    )

    def test_complete_summary_report_marks_truncated_rows_and_keeps_manual_boundary(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        truncated = table["config_id"].eq("q_h2_2")
        table.loc[truncated, "completed_voyage_count"] = 6
        table.loc[truncated, "completion_rate"] = 6.0 / 7.0
        table.loc[truncated, "metrics_comparable"] = False
        table.loc[truncated, "solver_failure_count"] = 1
        table.loc[truncated, "max_iter_count"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "summary.md"
            module.write_summary_report(table, report_path)
            report = report_path.read_text("utf-8")

        self.assertIn("q_h2_2", report)
        self.assertIn("截断", report)
        self.assertIn("不完整配置的累计量", report)
        self.assertIn("不自动生成建议搜索区间", report)
        self.assertNotIn("`q_h2:[0.25,0.5]`", report)
        self.assertIn("accepted fixed weight: none", report)

    def test_summary_report_reports_completion_boundaries_without_selecting_intervals(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        completion = {
            "q_h2_2": 6,
            "q_h2_4": 1,
            "q_batt_0p25": 5,
            "q_batt_0p5": 6,
            "q_soc_0p25": 6,
            "q_soc_0p5": 6,
        }
        for config_id, completed in completion.items():
            match = table["config_id"].eq(config_id)
            table.loc[match, "completed_voyage_count"] = completed
            table.loc[match, "completion_rate"] = completed / 7.0
            table.loc[match, "metrics_comparable"] = False
            table.loc[match, "solver_failure_count"] = 7 - completed
            table.loc[match, "max_iter_count"] = 7 - completed

        def set_values(config_id, **values):
            match = table["config_id"].eq(config_id)
            for field, value in values.items():
                table.loc[match, field] = value

        set_values("q_h2_0p25", min_soc=0.299)
        set_values("q_h2_0p5", min_soc=0.199995)
        set_values(
            "baseline_1_1_1_1",
            final_soc_mean=0.288,
            sum_p_batt_sq_kw2=437.5e6,
            sum_soc_error_sq=4672.0,
            sum_fc_delta_sq_kw2=130390.0,
        )
        set_values(
            "q_batt_2", final_soc_mean=0.380, sum_p_batt_sq_kw2=167.4e6
        )
        set_values(
            "q_batt_4", final_soc_mean=0.436, sum_p_batt_sq_kw2=77.6e6
        )
        set_values(
            "q_soc_2", final_soc_mean=0.305, sum_soc_error_sq=4428.0
        )
        set_values(
            "q_soc_4", final_soc_mean=0.335, sum_soc_error_sq=3803.0
        )
        for config_id, fc_delta, batt_sq in (
            ("q_fc_var_0p25", 158339.0, 424.4e6),
            ("q_fc_var_0p5", 149543.0, 429.8e6),
            ("q_fc_var_2", 100803.0, 453.6e6),
            ("q_fc_var_4", 64072.0, 485.2e6),
        ):
            set_values(
                config_id,
                sum_fc_delta_sq_kw2=fc_delta,
                sum_p_batt_sq_kw2=batt_sq,
            )

        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "summary.md"
            module.write_summary_report(table, report_path)
            report = report_path.read_text("utf-8")

        self.assertIn("q_h2:[1,2]", report)
        self.assertIn("q_batt:[0.5,1]", report)
        self.assertIn("q_soc:[0.5,1]", report)
        self.assertIn("q_fc_var: tested [0.25,4] has no completion boundary", report)
        self.assertNotIn("建议审阅", report)
        self.assertIn("不自动生成建议搜索区间", report)

    def test_baseline_only_report_has_conditional_behavior_and_soc_upper_audit(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module).iloc[[0]].copy()
        table.loc[:, "initial_soc"] = 0.55
        table.loc[:, "final_soc_mean"] = 0.56
        table.loc[:, "max_soc"] = 0.800002
        table.loc[:, "max_batt_charge_kw"] = 0.0
        with tempfile.TemporaryDirectory() as temporary:
            report_path = Path(temporary) / "summary.md"
            module.write_summary_report(table, report_path)
            report = report_path.read_text("utf-8")

        self.assertIn("## 全 1 baseline", report)
        self.assertIn("平均 SOC 净上升", report)
        self.assertIn("未出现充电功率", report)
        self.assertNotIn("平均 SOC 明显净下降", report)
        self.assertIn("2.000e-06", report)
        self.assertIn("完整 17 配置矩阵不存在", report)

    def test_report_and_figures_reject_inconsistent_comparability(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        mismatch = table["config_id"].eq("q_h2_2")
        table.loc[mismatch, "completed_voyage_count"] = 6
        table.loc[mismatch, "completion_rate"] = 6.0 / 7.0
        table.loc[mismatch, "metrics_comparable"] = True
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "metrics_comparable"):
                module.write_summary_report(table, Path(temporary) / "summary.md")
            with self.assertRaisesRegex(ValueError, "metrics_comparable"):
                module.write_summary_figures(table, Path(temporary) / "summary")

    def test_summary_figures_omit_truncated_physical_totals(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        truncated = table["config_id"].isin(
            ("q_h2_2", "q_soc_0p25", "q_soc_0p5")
        )
        table.loc[truncated, "metrics_comparable"] = False
        table.loc[truncated, "completed_voyage_count"] = 6
        table.loc[truncated, "completion_rate"] = 6.0 / 7.0
        captured = []
        original_subplots = module.plt.subplots

        def capture(*args, **kwargs):
            figure, axes = original_subplots(*args, **kwargs)
            captured.append(figure)
            return figure, axes

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            module.plt, "subplots", side_effect=capture
        ):
            module.write_summary_figures(table, Path(temporary) / "summary")

        q_h2_figure = captured[0]
        for axis in q_h2_figure.axes[:5]:
            for line in axis.lines:
                y_values = np.asarray(line.get_ydata(), dtype=float)
                self.assertTrue(np.isnan(y_values[3]))
        completion = np.asarray(q_h2_figure.axes[5].lines[0].get_ydata(), dtype=float)
        self.assertAlmostEqual(completion[3], 6.0 / 7.0)
        q_h2_legend = q_h2_figure.axes[5].get_legend()
        self.assertIsNotNone(q_h2_legend)
        self.assertTrue(
            any("truncated" in text.get_text().lower() for text in q_h2_legend.get_texts())
        )

        q_soc_figure = captured[2]
        completion_axis = q_soc_figure.axes[5]
        legend = completion_axis.get_legend()
        self.assertIsNotNone(legend)
        self.assertTrue(
            any("truncated" in text.get_text().lower() for text in legend.get_texts())
        )
        marked_offsets = np.asarray(completion_axis.collections[0].get_offsets(), dtype=float)
        self.assertTrue(
            np.allclose(marked_offsets, [[0.25, 6.0 / 7.0], [0.5, 6.0 / 7.0]])
        )
        q_soc_figure.canvas.draw()
        renderer = q_soc_figure.canvas.get_renderer()
        legend_bounds = legend.get_window_extent(renderer=renderer)
        axis_bounds = completion_axis.get_window_extent(renderer=renderer)
        self.assertGreaterEqual(legend_bounds.x0, axis_bounds.x0)
        self.assertGreaterEqual(legend_bounds.y0, axis_bounds.y0)
        self.assertLessEqual(legend_bounds.x1, axis_bounds.x1)
        self.assertLessEqual(legend_bounds.y1, axis_bounds.y1)

    def test_summary_figure_closes_figure_when_save_fails(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        module.plt.close("all")
        before = module.plt.get_fignums()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "matplotlib.figure.Figure.savefig", side_effect=RuntimeError("save failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "save failed"):
                module.write_summary_figures(table, Path(temporary) / "summary")
        self.assertEqual(module.plt.get_fignums(), before)

    def test_summary_figures_reject_unknown_entry_without_deleting_or_plotting(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        table = self._summary_table(module)
        with tempfile.TemporaryDirectory() as temporary:
            summary_dir = Path(temporary) / "summary"
            summary_dir.mkdir()
            stale = summary_dir / "stale_extra.png"
            stale.write_bytes(b"stale")

            with (
                patch.object(module.plt, "subplots") as subplots,
                self.assertRaisesRegex(ValueError, "unexpected"),
            ):
                module.write_summary_figures(table, summary_dir)

            subplots.assert_not_called()
            self.assertEqual({path.name for path in summary_dir.iterdir()}, {stale.name})
            self.assertEqual(stale.read_bytes(), b"stale")

    def test_parser_is_minimal_required_and_mutually_exclusive(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        parser = module.build_parser()
        baseline = parser.parse_args(
            ["--baseline", "--voyage", "voyage_060", "--output-dir", "custom", "--overwrite"]
        )
        self.assertTrue(baseline.baseline)
        self.assertFalse(baseline.one_factor)
        self.assertEqual(baseline.voyage, "voyage_060")
        self.assertEqual(baseline.output_dir, Path("custom"))
        self.assertTrue(baseline.overwrite)
        for arguments in (
            [],
            ["--baseline", "--one-factor"],
            ["--baseline", "--voyage", "voyage_999"],
            ["--baseline", "--input", "input.parquet"],
            ["--baseline", "--max-steps", "1"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(arguments)

    def test_run_experiment_sequences_modes_reuses_baseline_and_skips_reports_for_diagnostic(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        formal_data = pd.DataFrame(
            {
                "voyage_id": list(module.EXPECTED_TEST_VOYAGES),
                "time_s": np.zeros(7),
                "load_total_kw": np.ones(7),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            input_path.write_bytes(b"input")

            def evaluate(case, *, data):
                voyages = tuple(sorted(data["voyage_id"].astype(str).unique()))
                result = self._synthetic_result(module, voyages)
                result["case"] = case
                result["config"] = module.four_objective_config(case)
                result["summary"].update(
                    {
                        "config_id": case.config_id,
                        "varied_weight": case.varied_weight or "baseline",
                        "weight_value": case.weight_value,
                        **{name: getattr(case, name) for name in module.WEIGHT_NAMES},
                    }
                )
                return result

            with (
                patch.object(module, "load_spline_test_data", return_value=formal_data),
                patch.object(module, "evaluate_configuration", side_effect=evaluate) as run,
                patch.object(module, "write_configuration_artifacts"),
                patch.object(module, "write_summary_table") as table_writer,
                patch.object(module, "write_summary_report") as report_writer,
                patch.object(module, "write_summary_figures") as figure_writer,
            ):
                table = module.run_experiment(
                    mode="one-factor",
                    input_path=input_path,
                    output_dir=root / "output",
                )
            self.assertEqual(len(table), 17)
            self.assertEqual(
                [call.args[0].config_id for call in run.call_args_list],
                [case.config_id for case in module.build_sensitivity_cases()],
            )
            table_writer.assert_called_once()
            report_writer.assert_called_once()
            figure_writer.assert_called_once_with(table, root / "output" / "summary")

            baseline_dir = root / "reuse" / "baseline_1_1_1_1"
            baseline_dir.mkdir(parents=True)
            expected_summary = self._synthetic_result(
                module, module.EXPECTED_TEST_VOYAGES
            )["summary"]
            with (
                patch.object(module, "load_spline_test_data", return_value=formal_data),
                patch.object(module, "load_matching_case", return_value=expected_summary) as reuse,
                patch.object(module, "evaluate_configuration") as evaluate_mock,
                patch.object(module, "write_summary_table") as baseline_table,
                patch.object(module, "write_summary_report") as baseline_report,
                patch.object(module, "write_summary_figures") as baseline_figures,
            ):
                reused = module.run_experiment(
                    mode="baseline",
                    input_path=input_path,
                    output_dir=root / "reuse",
                )
            self.assertEqual(reused.iloc[0].to_dict(), expected_summary)
            reuse.assert_called_once()
            evaluate_mock.assert_not_called()
            baseline_table.assert_not_called()
            baseline_report.assert_not_called()
            baseline_figures.assert_not_called()

            (root / "reuse" / "q_h2_0p25").mkdir()
            with (
                patch.object(module, "load_spline_test_data", return_value=formal_data),
                patch.object(module, "evaluate_configuration") as unsafe_evaluate,
                self.assertRaisesRegex(ValueError, "one-factor evidence"),
            ):
                module.run_experiment(
                    mode="baseline",
                    input_path=input_path,
                    output_dir=root / "reuse",
                    overwrite=True,
                )
            unsafe_evaluate.assert_not_called()

            with (
                patch.object(module, "load_spline_test_data", return_value=formal_data),
                patch.object(module, "evaluate_configuration", side_effect=evaluate),
                patch.object(module, "write_configuration_artifacts") as artifact_writer,
                patch.object(module, "write_summary_table") as diagnostic_table,
                patch.object(module, "write_summary_report") as diagnostic_report,
                patch.object(module, "write_summary_figures") as diagnostic_figures,
            ):
                diagnostic = module.run_experiment(
                    mode="baseline",
                    input_path=input_path,
                    output_dir=root / "diagnostic",
                    voyage_id="voyage_060",
                )
            self.assertEqual(len(diagnostic), 1)
            self.assertEqual(artifact_writer.call_args.kwargs["diagnostic_voyage"], "voyage_060")
            diagnostic_table.assert_not_called()
            diagnostic_report.assert_not_called()
            diagnostic_figures.assert_not_called()

    def test_main_maps_exact_mode_to_run_experiment(self) -> None:
        import run_mpc_1s_n6_four_objective_sensitivity as module

        with patch.object(module, "run_experiment") as run:
            module.main(["--baseline", "--voyage", "voyage_060"])
        run.assert_called_once_with(
            mode="baseline",
            output_dir=module.DEFAULT_OUTPUT_DIR,
            voyage_id="voyage_060",
            overwrite=False,
        )


if __name__ == "__main__":
    unittest.main()
