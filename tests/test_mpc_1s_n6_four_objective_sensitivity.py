from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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
        from run_mpc_1s_n6_four_objective_sensitivity import N6_OSQP_SETTINGS

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


if __name__ == "__main__":
    unittest.main()
