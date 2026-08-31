from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from dqn.utils.action_mapper import (
    DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
)

from .mpc_qp_formulation import (
    QpMpcConfig,
    build_qp_problem,
)

from .n6_qp_scaling import (
    FIXED_SOC_REFERENCE,
    N6_OSQP_SETTINGS,
    N6QpTransform,
    _setup_n6_osqp_solver,
    scale_n6_qp_problem,
    scaled_linear_for_previous_fc,
)

from .osqp_runtime import (
    _qp_bounds_for_step,
    _solve_with_persistent_osqp,
    _try_import_osqp,
)


# 暂时保留旧名称，避免现有测试导入失败。
# 唯一真实设置定义位于 n6_qp_scaling.py。
_OSQP_SETTINGS = N6_OSQP_SETTINGS


@dataclass(frozen=True)
class _SolverEntry:
    action: MPCWeightAction
    config: QpMpcConfig

    # 物理空间矩阵，保留用于检查和回归测试。
    P: sparse.csc_matrix
    A: sparse.csc_matrix

    # 实际交给OSQP的缩放空间矩阵。
    scaled_P: sparse.csc_matrix
    scaled_A: sparse.csc_matrix

    solver: Any
    transform: N6QpTransform
    setup_previous_fc_kw: float
    base_scaled_linear: np.ndarray


class MpcWeightSolverBank:
    def __init__(
        self,
        base_config: QpMpcConfig,
        *,
        actions: Sequence[MPCWeightAction] = DQN_MPC_WEIGHT_ACTIONS,
    ) -> None:
        osqp_module, import_error = _try_import_osqp()

        if osqp_module is None:
            raise RuntimeError(f"Cannot import osqp: {import_error}")

        resolved_actions = tuple(actions)
        if not resolved_actions:
            raise ValueError("solver bank requires at least one action")

        action_ids = [action.action_id for action in resolved_actions]
        if any(type(action_id) is not int or action_id < 0 for action_id in action_ids):
            raise ValueError("action IDs must be non-negative integers")
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("solver bank action IDs must be unique")

        action_names = [str(action.name) for action in resolved_actions]
        if any(not name for name in action_names):
            raise ValueError("solver bank action names must be non-empty")
        if len(action_names) != len(set(action_names)):
            raise ValueError("solver bank action names must be unique")

        weight_rows = np.asarray(
            [action.as_tuple() for action in resolved_actions],
            dtype=float,
        )
        if not np.all(np.isfinite(weight_rows)):
            raise ValueError("solver bank action weights must be finite")
        if np.any(weight_rows < 0.0):
            raise ValueError("solver bank action weights must be non-negative")

        horizon = int(base_config.horizon)

        setup_load = np.full(
            horizon,
            float(base_config.fuel_cell_min_kw),
            dtype=float,
        )

        setup_soc = float(FIXED_SOC_REFERENCE)
        setup_prev_fc = float(base_config.fuel_cell_min_kw)

        entries: dict[int, _SolverEntry] = {}
        common_physical_a: sparse.csc_matrix | None = None

        for action in resolved_actions:
            config = replace(
                base_config,
                q_h2=action.q_h2,
                q_batt=action.q_batt,
                q_soc=action.q_soc,
                q_fc_var=action.q_fc_var,
                soc_penalty_mode=action.soc_penalty_mode,
            )

            physical_problem = build_qp_problem(
                config,
                load_forecast_kw=setup_load,
                current_soc=setup_soc,
                prev_fc_kw=setup_prev_fc,
                soc_reference=FIXED_SOC_REFERENCE,
                include_diagnostics=False,
            )

            scaled_problem, transform = scale_n6_qp_problem(
                physical_problem,
                config=config,
            )

            solver = _setup_n6_osqp_solver(
                osqp_module,
                scaled_problem,
            )

            if common_physical_a is None:
                common_physical_a = physical_problem.A

            entries[action.action_id] = _SolverEntry(
                action=action,
                config=config,
                P=physical_problem.P,
                A=common_physical_a,
                scaled_P=scaled_problem.P,
                scaled_A=scaled_problem.A,
                solver=solver,
                transform=transform,
                setup_previous_fc_kw=setup_prev_fc,
                base_scaled_linear=scaled_problem.q.copy(),
            )

        self._entries = entries
        self.actions = resolved_actions

    def solve(
        self,
        action_id: int,
        load_forecast_kw: np.ndarray | list[float],
        current_soc: float,
        prev_fc_kw: float,
        soc_reference: float,
    ) -> tuple[Any, float]:
        if type(action_id) is not int:
            raise ValueError("action_id must be an integer")
        try:
            entry = self._entries[action_id]
        except KeyError as error:
            raise IndexError(
                f"action_id {action_id} is not present in this solver bank"
            ) from error

        if not np.isclose(
            float(soc_reference),
            float(FIXED_SOC_REFERENCE),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "solver bank currently requires "
                f"soc_reference={FIXED_SOC_REFERENCE}"
            )

        lower_physical, upper_physical = _qp_bounds_for_step(
            entry.config,
            load_forecast_kw=load_forecast_kw,
            current_soc=current_soc,
            prev_fc_kw=prev_fc_kw,
        )

        lower_scaled, upper_scaled = entry.transform.transform_bounds(
            lower_physical,
            upper_physical,
        )

        linear_scaled = scaled_linear_for_previous_fc(
            entry.base_scaled_linear,
            config=entry.config,
            transform=entry.transform,
            base_previous_fc_kw=entry.setup_previous_fc_kw,
            previous_fc_kw=prev_fc_kw,
        )

        result, solve_ms = _solve_with_persistent_osqp(
            entry.solver,
            lower=lower_scaled,
            upper=upper_scaled,
            linear=linear_scaled,
        )

        # OSQP内部解属于缩放空间。
        # 为保持当前solve接口兼容，将result.x恢复为物理解；
        # 同时保留x_scaled和x_physical供后续环境使用。
        if result.x is not None:
            scaled_solution = np.asarray(
                result.x,
                dtype=float,
            ).reshape(-1).copy()

            physical_solution = entry.transform.to_physical(
                scaled_solution
            )

            result.x_scaled = scaled_solution
            result.x_physical = physical_solution
            result.x = physical_solution

        return result, float(solve_ms)
