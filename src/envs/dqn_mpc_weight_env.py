from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# mpc_solvers currently lives under src/main/.
# Add that existing package root without importing src/main.py.
SRC_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = SRC_ROOT / "main"

if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))


from dqn.utils.reward import calculate_mpc_weight_reward
from dqn.utils.action_mapper import (
    DQN_MPC_WEIGHT_ACTIONS,
    MPCWeightAction,
)
from dqn.utils.state_builder import (
    DQN_MPC_PREVIEW_STEPS,
    SOC_REFERENCE,
    build_dqn_mpc_state,
)

from mpc_solvers.dqn_mpc_solver_bank import MpcWeightSolverBank
from mpc_solvers.mpc_qp_formulation import QpMpcConfig


class MpcSolveFailure(RuntimeError):
    """Structured, non-fallback MPC failure at one environment decision."""

    def __init__(
        self,
        *,
        action_id: int,
        decision_index: int,
        execution_index: int,
        solver_status: str,
        solve_ms: float,
        current_soc: float,
        previous_fc_kw: float,
        future_load_kw: Sequence[float],
        iterations: int | None,
        primal_residual: float | None,
        dual_residual: float | None,
    ) -> None:
        super().__init__(
            "MPC solve failed: "
            f"action_id={action_id}, "
            f"decision_index={decision_index}, "
            f"status={solver_status}"
        )
        self.action_id = int(action_id)
        self.decision_index = int(decision_index)
        self.execution_index = int(execution_index)
        self.solver_status = str(solver_status)
        self.solve_ms = float(solve_ms)
        self.current_soc = float(current_soc)
        self.previous_fc_kw = float(previous_fc_kw)
        self.future_load_kw = tuple(
            float(value) for value in future_load_kw
        )
        self.iterations = (
            None if iterations is None else int(iterations)
        )
        self.primal_residual = (
            None
            if primal_residual is None
            else float(primal_residual)
        )
        self.dual_residual = (
            None
            if dual_residual is None
            else float(dual_residual)
        )


class DqnMpcWeightEnv:
    """
    Single-voyage environment for DQN-based MPC weight selection.

    Time convention
    ---------------
    At decision index t:

        state s_t contains:
            current load P_load[t]
            previous executed FC/battery power
            current SOC
            future load preview P_load[t+1 : t+7]

        action a_t selects one MPC four-weight tuple.

        MPC optimizes the next six execution samples:
            t+1, ..., t+6

        Only the first MPC control is executed at:
            t+1

        The environment then advances to state:
            s_{t+1}

    One voyage is one episode.

    This environment does not contain a DQN agent, replay buffer,
    exploration policy, training logic, fallback controller, or
    additional heuristic reward/penalty.
    """

    def __init__(
        self,
        *,
        loads_kw: Sequence[float] | np.ndarray,
        base_config: QpMpcConfig,
        initial_soc: float = SOC_REFERENCE,
        actions: Sequence[MPCWeightAction] = DQN_MPC_WEIGHT_ACTIONS,
    ) -> None:
        loads = np.asarray(
            loads_kw,
            dtype=np.float64,
        ).reshape(-1)

        if loads.size < 2:
            raise ValueError(
                "loads_kw must contain at least two samples"
            )

        if not np.all(np.isfinite(loads)):
            raise ValueError(
                "loads_kw must contain only finite values"
            )

        if int(base_config.horizon) != DQN_MPC_PREVIEW_STEPS:
            raise ValueError(
                "DQN-MPC environment requires "
                f"horizon={DQN_MPC_PREVIEW_STEPS}"
            )

        if not np.isclose(
            float(base_config.dt_seconds),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "DQN-MPC environment currently requires dt_seconds=1.0"
            )

        initial_soc_value = float(initial_soc)

        if not np.isfinite(initial_soc_value):
            raise ValueError("initial_soc must be finite")

        if not (
            float(base_config.soc_min)
            <= initial_soc_value
            <= float(base_config.soc_max)
        ):
            raise ValueError(
                "initial_soc must lie within MPC SOC bounds"
            )

        self.loads_kw = loads
        self.base_config = base_config
        self.initial_soc = initial_soc_value
        self.actions = tuple(actions)

        self.solver_bank = MpcWeightSolverBank(
            base_config,
            actions=self.actions,
        )

        self.decision_index = 0
        self.current_soc = initial_soc_value
        self.previous_fc_kw = 0.0
        self.previous_batt_kw = 0.0
        self.done = False

        self.reset()

    def _future_window(
        self,
        decision_index: int,
    ) -> np.ndarray:
        """
        Return t+1 ... t+6 load preview.

        Near the end of a voyage, repeat the final measured load
        so that the MPC horizon remains exactly N=6, matching the
        fixed-horizon Candidate C convention.
        """

        index = int(decision_index)
        start = index + 1

        preview = self.loads_kw[
            start : start + DQN_MPC_PREVIEW_STEPS
        ]

        if preview.size == 0:
            return np.full(
                DQN_MPC_PREVIEW_STEPS,
                float(self.loads_kw[-1]),
                dtype=np.float64,
            )

        if preview.size < DQN_MPC_PREVIEW_STEPS:
            preview = np.pad(
                preview,
                (
                    0,
                    DQN_MPC_PREVIEW_STEPS - preview.size,
                ),
                mode="edge",
            )

        return np.asarray(
            preview,
            dtype=np.float64,
        )

    def _build_state(self) -> np.ndarray:
        index = int(self.decision_index)

        current_load_kw = float(self.loads_kw[index])

        previous_load_kw = (
            float(self.loads_kw[index - 1])
            if index > 0
            else current_load_kw
        )

        return build_dqn_mpc_state(
            current_soc=self.current_soc,
            previous_fc_kw=self.previous_fc_kw,
            previous_batt_kw=self.previous_batt_kw,
            current_load_kw=current_load_kw,
            previous_load_kw=previous_load_kw,
            future_load_kw=self._future_window(index),
        )

    def reset(self) -> np.ndarray:
        """
        Reset to the beginning of the current voyage.

        Initial FC power follows the existing Candidate C runner:
        clip the first load to the FC physical range.

        Initial battery power is the residual required by power
        balance at the first measured sample.
        """

        self.decision_index = 0
        self.current_soc = float(self.initial_soc)

        initial_load_kw = float(self.loads_kw[0])

        self.previous_fc_kw = float(
            np.clip(
                initial_load_kw,
                float(self.base_config.fuel_cell_min_kw),
                float(self.base_config.fuel_cell_max_kw),
            )
        )

        self.previous_batt_kw = (
            initial_load_kw - self.previous_fc_kw
        )

        self.done = False

        return self._build_state()

    def step(
        self,
        action_id: int,
    ) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute one DQN-MPC interaction.

        The selected action chooses one complete four-weight MPC
        configuration. Only the first control move is applied.
        """

        if self.done:
            raise RuntimeError(
                "episode is finished; call reset() before step()"
            )
        if type(action_id) is not int:
            raise ValueError("action_id must be an integer")

        decision_index = int(self.decision_index)
        execution_index = decision_index + 1

        load_forecast_kw = self._future_window(
            decision_index
        )

        load_actual_kw = float(
            self.loads_kw[execution_index]
        )

        soc_before = float(self.current_soc)
        previous_fc_before = float(
            self.previous_fc_kw
        )

        result, solve_ms = self.solver_bank.solve(
            action_id=action_id,
            load_forecast_kw=load_forecast_kw,
            current_soc=soc_before,
            prev_fc_kw=previous_fc_before,
            soc_reference=SOC_REFERENCE,
        )

        solver_status = str(result.info.status)

        if (
            not solver_status.lower().startswith("solved")
            or result.x is None
        ):
            info = result.info
            failure_status = solver_status
            if (
                solver_status.lower().startswith("solved")
                and result.x is None
            ):
                failure_status = (
                    f"{solver_status}; invalid solution vector is missing"
                )
            raise MpcSolveFailure(
                action_id=action_id,
                decision_index=decision_index,
                execution_index=execution_index,
                solver_status=failure_status,
                solve_ms=solve_ms,
                current_soc=soc_before,
                previous_fc_kw=previous_fc_before,
                future_load_kw=load_forecast_kw,
                iterations=getattr(info, "iter", None),
                primal_residual=getattr(info, "prim_res", None),
                dual_residual=getattr(info, "dual_res", None),
            )

        solution = np.asarray(
            result.x,
            dtype=np.float64,
        ).reshape(-1)

        horizon = int(self.base_config.horizon)
        expected_size = 3 * horizon + 1

        if solution.size != expected_size:
            raise RuntimeError(
                "unexpected MPC solution size: "
                f"{solution.size} != {expected_size}"
            )

        if not np.all(np.isfinite(solution)):
            raise RuntimeError(
                "MPC solution contains non-finite values"
            )

        # Physical decision-variable order:
        # [P_fc[0:N], P_batt[0:N], SOC[0:N+1]]
        p_fc_plan_kw = float(solution[0])
        p_batt_plan_kw = float(solution[horizon])
        soc_predicted = float(
            solution[2 * horizon + 1]
        )

        # Match the formal Candidate C execution rule:
        # FC follows the first MPC decision.
        # Battery closes the actual power balance.
        p_fc_actual_kw = p_fc_plan_kw
        p_batt_actual_kw = (
            load_actual_kw - p_fc_actual_kw
        )

        next_soc = soc_before - (
            p_batt_actual_kw
            * float(self.base_config.dt_seconds)
            / 3600.0
            / float(
                self.base_config.battery_capacity_kwh
            )
        )

        reward, reward_info = (
            calculate_mpc_weight_reward(
                p_fc_kw=p_fc_actual_kw,
                p_batt_kw=p_batt_actual_kw,
                next_soc=next_soc,
                previous_fc_kw=previous_fc_before,
            )
        )

        # Commit the executed physical state.
        self.decision_index = execution_index
        self.current_soc = float(next_soc)
        self.previous_fc_kw = float(
            p_fc_actual_kw
        )
        self.previous_batt_kw = float(
            p_batt_actual_kw
        )

        self.done = bool(
            self.decision_index
            >= self.loads_kw.size - 1
        )

        next_state = self._build_state()

        info: dict[str, Any] = {
            "action_id": int(action_id),
            "decision_index": decision_index,
            "execution_index": execution_index,
            "load_actual_kw": load_actual_kw,
            "p_fc_prev_kw": previous_fc_before,
            "p_fc_plan_kw": p_fc_plan_kw,
            "p_batt_plan_kw": p_batt_plan_kw,
            "soc_predicted": soc_predicted,
            "p_fc_kw": p_fc_actual_kw,
            "p_batt_kw": p_batt_actual_kw,
            "soc_before": soc_before,
            "soc_after": float(next_soc),
            "power_balance_residual_kw": float(
                p_fc_actual_kw
                + p_batt_actual_kw
                - load_actual_kw
            ),
            "soc_prediction_residual": float(
                soc_predicted - next_soc
            ),
            "solver_status": solver_status,
            "solve_ms": float(solve_ms),
            "reward_terms": reward_info,
        }

        return (
            next_state,
            float(reward),
            self.done,
            info,
        )
