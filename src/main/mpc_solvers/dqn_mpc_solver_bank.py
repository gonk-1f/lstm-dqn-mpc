from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import sparse

from dqn.utils.action_mapper import (
    DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
    get_weight_action,
)

from .mpc_qp_formulation import QpMpcConfig, build_qp_problem
from .osqp_runtime import (
    _qp_bounds_for_step,
    _solve_with_persistent_osqp,
    _try_import_osqp,
)

_OSQP_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "polishing": True,
    "warm_starting": True,
    "eps_abs": 1.0e-5,
    "eps_rel": 1.0e-5,
    "max_iter": 20000,
    "adaptive_rho": True,
}


@dataclass(frozen=True)
class _SolverEntry:
    action: MPCWeightAction
    config: QpMpcConfig
    P: sparse.csc_matrix
    A: sparse.csc_matrix
    solver: Any


class MpcWeightSolverBank:
    def __init__(self, base_config: QpMpcConfig) -> None:
        osqp_module, import_error = _try_import_osqp()
        if osqp_module is None:
            raise RuntimeError(f"Cannot import osqp: {import_error}")

        horizon = int(base_config.horizon)
        setup_load = np.full(horizon, float(base_config.fuel_cell_min_kw), dtype=float)
        setup_soc = 0.5 * (float(base_config.soc_min) + float(base_config.soc_max))
        setup_prev_fc = float(base_config.fuel_cell_min_kw)
        entries: dict[int, _SolverEntry] = {}
        common_a: sparse.csc_matrix | None = None

        for action in DQN_MPC_WEIGHT_ACTIONS:
            config = replace(
                base_config,
                q_h2=action.q_h2,
                q_batt=action.q_batt,
                q_soc=action.q_soc,
                q_fc_var=action.q_fc_var,
            )
            problem = build_qp_problem(
                config,
                load_forecast_kw=setup_load,
                current_soc=setup_soc,
                prev_fc_kw=setup_prev_fc,
                soc_reference=setup_soc,
                include_diagnostics=False,
            )
            if common_a is None:
                common_a = problem.A
            solver = osqp_module.OSQP()
            solver.setup(
                P=problem.P,
                q=problem.q,
                A=common_a,
                l=problem.l,
                u=problem.u,
                **_OSQP_SETTINGS,
            )
            entries[action.action_id] = _SolverEntry(
                action=action,
                config=config,
                P=problem.P,
                A=common_a,
                solver=solver,
            )

        self._entries = entries

    def solve(
        self,
        action_id: int,
        load_forecast_kw: np.ndarray | list[float],
        current_soc: float,
        prev_fc_kw: float,
        soc_reference: float,
    ) -> tuple[Any, float]:
        action = get_weight_action(action_id)
        entry = self._entries[action.action_id]
        problem = build_qp_problem(
            entry.config,
            load_forecast_kw=load_forecast_kw,
            current_soc=current_soc,
            prev_fc_kw=prev_fc_kw,
            soc_reference=soc_reference,
            include_diagnostics=False,
        )
        lower, upper = _qp_bounds_for_step(
            entry.config,
            load_forecast_kw=load_forecast_kw,
            current_soc=current_soc,
            prev_fc_kw=prev_fc_kw,
        )
        return _solve_with_persistent_osqp(
            entry.solver,
            lower=lower,
            upper=upper,
            linear=problem.q,
        )
