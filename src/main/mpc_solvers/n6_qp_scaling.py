from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .mpc_qp_formulation import (
    QpMpcConfig,
    QpProblem,
    resolved_ramp_kw_per_step,
)


OBJECTIVE_VARIANT = "n6_h2_batt_soc_fcvar_normalized_v1"
N6_HORIZON = 6
FIXED_SOC_REFERENCE = 0.55

N6_OSQP_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "polishing": True,
    "warm_starting": True,
    "eps_abs": 1.0e-5,
    "eps_rel": 1.0e-5,
    "max_iter": 20000,
    "adaptive_rho": True,
}


@dataclass(frozen=True)
class N6QpTransform:
    variable_scale: np.ndarray
    variable_offset: np.ndarray
    row_scale: np.ndarray
    constraint_offset: np.ndarray
    objective_constant: float

    def to_normalized(self, physical: np.ndarray) -> np.ndarray:
        values = np.asarray(physical, dtype=float).reshape(-1)
        return (values - self.variable_offset) / self.variable_scale

    def to_physical(self, normalized: np.ndarray) -> np.ndarray:
        values = np.asarray(normalized, dtype=float).reshape(-1)
        return self.variable_offset + self.variable_scale * values

    def transform_bounds(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_values = np.asarray(lower, dtype=float).reshape(-1)
        upper_values = np.asarray(upper, dtype=float).reshape(-1)

        return (
            self.row_scale * (lower_values - self.constraint_offset),
            self.row_scale * (upper_values - self.constraint_offset),
        )


def scale_n6_qp_problem(
    problem: QpProblem,
    *,
    config: QpMpcConfig,
) -> tuple[QpProblem, N6QpTransform]:
    horizon = int(config.horizon)

    if horizon != N6_HORIZON:
        raise ValueError(
            f"N=6 scaling requires horizon={N6_HORIZON}, got {horizon}"
        )

    expected_variables = 3 * horizon + 1
    expected_constraints = 6 * horizon + 2

    if problem.P.shape != (expected_variables, expected_variables):
        raise ValueError("unexpected N=6 QP variable dimensions")

    if problem.A.shape != (expected_constraints, expected_variables):
        raise ValueError("unexpected N=6 QP constraint dimensions")

    fuel_cell_scale = float(config.fuel_cell_max_kw)
    battery_scale = float(config.battery_power_ref_kw)
    soc_scale = float(config.soc_band)
    ramp_scale = max(float(resolved_ramp_kw_per_step(config)), 1.0)

    variable_scale = np.concatenate(
        [
            np.full(horizon, fuel_cell_scale),
            np.full(horizon, battery_scale),
            np.full(horizon + 1, soc_scale),
        ]
    )

    variable_offset = np.concatenate(
        [
            np.zeros(2 * horizon),
            np.full(horizon + 1, FIXED_SOC_REFERENCE),
        ]
    )

    row_scale = np.concatenate(
        [
            np.full(horizon, 1.0 / fuel_cell_scale),
            np.full(horizon, 1.0 / battery_scale),
            np.full(horizon + 1, 1.0 / soc_scale),
            np.full(1 + horizon, 1.0 / soc_scale),
            np.full(horizon, 1.0 / fuel_cell_scale),
            np.full(horizon, 1.0 / ramp_scale),
        ]
    )

    scale_matrix = sparse.diags(variable_scale, format="csc")
    row_matrix = sparse.diags(row_scale, format="csc")

    constraint_offset = np.asarray(
        problem.A @ variable_offset,
        dtype=float,
    ).reshape(-1)

    scaled_p = (scale_matrix @ problem.P @ scale_matrix).tocsc()

    scaled_q = variable_scale * (
        np.asarray(
            problem.P @ variable_offset,
            dtype=float,
        ).reshape(-1)
        + problem.q
    )

    scaled_a = (row_matrix @ problem.A @ scale_matrix).tocsc()

    scaled_l = row_scale * (
        problem.l - constraint_offset
    )

    scaled_u = row_scale * (
        problem.u - constraint_offset
    )

    objective_constant = float(
        0.5 * variable_offset @ problem.P @ variable_offset
        + problem.q @ variable_offset
    )

    transform = N6QpTransform(
        variable_scale=variable_scale,
        variable_offset=variable_offset,
        row_scale=row_scale,
        constraint_offset=constraint_offset,
        objective_constant=objective_constant,
    )

    metadata = {
        **problem.metadata,
        "osqp_affine_scaling": True,
        "osqp_variable_scale": variable_scale.tolist(),
        "osqp_variable_offset": variable_offset.tolist(),
        "osqp_constraint_row_scale": row_scale.tolist(),
    }

    return (
        QpProblem(
            P=scaled_p,
            q=scaled_q,
            A=scaled_a,
            l=scaled_l,
            u=scaled_u,
            metadata=metadata,
        ),
        transform,
    )


def _setup_n6_osqp_solver(
    osqp_module: Any,
    problem: QpProblem,
) -> Any:
    solver = osqp_module.OSQP()

    solver.setup(
        P=problem.P,
        q=problem.q,
        A=problem.A,
        l=problem.l,
        u=problem.u,
        **N6_OSQP_SETTINGS,
    )

    return solver


def scaled_linear_for_previous_fc(
    base_scaled_linear: np.ndarray,
    *,
    config: QpMpcConfig,
    transform: N6QpTransform,
    base_previous_fc_kw: float,
    previous_fc_kw: float,
) -> np.ndarray:
    if str(config.objective_variant) != OBJECTIVE_VARIANT:
        raise ValueError(
            "rolling linear refresh requires "
            f"objective_variant={OBJECTIVE_VARIANT}"
        )

    linear = np.asarray(
        base_scaled_linear,
        dtype=float,
    ).copy()

    ramp_kw_per_step = float(
        resolved_ramp_kw_per_step(config)
    )

    delta_physical = (
        -2.0
        * float(config.q_fc_var)
        * (
            float(previous_fc_kw)
            - float(base_previous_fc_kw)
        )
        / ramp_kw_per_step**2
    )

    linear[0] += (
        float(transform.variable_scale[0])
        * delta_physical
    )

    return linear