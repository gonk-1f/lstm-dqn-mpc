from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

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
        np.testing.assert_allclose(
            matrix[8, [2, 4, 5]], [1.0 / (3600.0 * 693.0), -1.0, 1.0]
        )
        np.testing.assert_allclose(matrix[10, [0, 2]], [1.0, 1.0])
        self.assertEqual((qp.l[10], qp.u[10]), (250.0, 250.0))
        np.testing.assert_allclose(matrix[12, :2], [1.0, 0.0])
        self.assertEqual((qp.l[12], qp.u[12]), (152.0, 248.0))
        np.testing.assert_allclose(matrix[13, :2], [-1.0, 1.0])
        self.assertEqual((qp.l[13], qp.u[13]), (-48.0, 48.0))

    def test_obsolete_n6_objective_variants_are_rejected(self) -> None:
        for variant in (
            "n6_h2_fc_variation_battery_v1",
            "n6_h2_fc_variation_battery_unnormalized_v1",
        ):
            with self.subTest(variant=variant):
                with self.assertRaisesRegex(ValueError, f"unsupported objective_variant: {variant}"):
                    problem(replace(objective_config(), objective_variant=variant))


if __name__ == "__main__":
    unittest.main()
