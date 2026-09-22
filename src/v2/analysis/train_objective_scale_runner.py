"""Representative-case selection and MPC adapter for the Train scale audit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from numbers import Real
from typing import Sequence

from ..config import PlantConfig, TimeScaleConfig
from ..control.causal_base_load import CausalBaseLoadFilter
from ..control.nonlinear_mpc import (
    DELTA_P_FC_OBJECTIVE_SCALE_KW,
    MPCConfig,
    NonlinearMPC,
    P_FC_OBJECTIVE_SCALE_KW,
    SOC_HARD_MAX,
    SOC_HARD_MIN,
    SOC_OBJECTIVE_SCALE,
    SOC_WORKING_HIGH,
    SOC_WORKING_LOW,
)
from ..dqn.action_space import ActionCandidate, CANDIDATE_ACTION_BANK
from ..models.battery_energy import formal_battery_efficiency
from .action_screening import DatasetProvenance
from .objective_scale_audit import (
    BehaviorTolerance,
    ObjectiveAuditCase,
    ObjectiveScaleAuditResult,
    run_objective_scale_audit,
)


AUDIT_TAU_LPF_SECONDS = 90.0


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real non-bool scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class AuditReadyState:
    parent_id: str
    timestamp: datetime
    p_load_kw: float
    soc_system: float
    previous_p_fc_total_kw: float
    history_loads_kw: tuple[float, ...]
    absolute_load_change_kw: float

    def __post_init__(self) -> None:
        if type(self.parent_id) is not str or not self.parent_id.strip():
            raise ValueError("parent_id must be a nonempty exact string")
        if type(self.timestamp) is not datetime or self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be a timezone-aware exact datetime")
        for name in (
            "p_load_kw",
            "soc_system",
            "previous_p_fc_total_kw",
            "absolute_load_change_kw",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.p_load_kw < 0.0 or self.absolute_load_change_kw < 0.0:
            raise ValueError("load and absolute load change must be nonnegative")
        if not 0.0 <= self.soc_system <= 1.0:
            raise ValueError("soc_system must lie in [0, 1]")
        if type(self.history_loads_kw) is not tuple:
            raise TypeError("history_loads_kw must be an exact tuple")
        history = tuple(_finite(value, "history load") for value in self.history_loads_kw)
        if any(value < 0.0 for value in history):
            raise ValueError("history loads must be nonnegative")
        object.__setattr__(self, "history_loads_kw", history)


def _load_bin(load_kw: float) -> int:
    if load_kw < 200.0:
        return 0
    if load_kw < 400.0:
        return 1
    return 2


def select_representative_cases(
    states: Sequence[AuditReadyState],
) -> tuple[ObjectiveAuditCase, ...]:
    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise TypeError("states must be a sequence")
    checked = tuple(states)
    if any(type(state) is not AuditReadyState for state in checked):
        raise TypeError("states must contain exact AuditReadyState values")
    usable = tuple(
        sorted(
            (state for state in checked if SOC_HARD_MIN <= state.soc_system <= SOC_HARD_MAX),
            key=lambda state: (state.parent_id, state.timestamp),
        )
    )
    if not usable:
        return ()

    selected: list[AuditReadyState] = []
    for bin_id in range(3):
        group = tuple(state for state in usable if _load_bin(state.p_load_kw) == bin_id)
        if not group:
            continue
        steady = min(
            group,
            key=lambda state: (
                state.absolute_load_change_kw,
                state.parent_id,
                state.timestamp,
            ),
        )
        rapid = max(
            group,
            key=lambda state: (
                state.absolute_load_change_kw,
                -state.timestamp.timestamp(),
            ),
        )
        for state in (steady, rapid):
            if state not in selected:
                selected.append(state)

    soc_bands = (
        lambda state: state.soc_system < 0.4,
        lambda state: 0.4 <= state.soc_system <= 0.6,
        lambda state: state.soc_system > 0.6,
    )
    for belongs in soc_bands:
        available = tuple(state for state in usable if belongs(state))
        if available and not any(belongs(state) for state in selected):
            addition = min(
                available,
                key=lambda state: (
                    state.absolute_load_change_kw,
                    state.parent_id,
                    state.timestamp,
                ),
            )
            if len(selected) < 7:
                selected.append(addition)

    selected.sort(key=lambda state: (state.parent_id, state.timestamp))
    return tuple(
        ObjectiveAuditCase(
            case_id=f"{state.parent_id}/{state.timestamp.isoformat()}",
            order=index,
            payload=(
                state.p_load_kw,
                state.soc_system,
                state.previous_p_fc_total_kw,
                *state.history_loads_kw,
            ),
        )
        for index, state in enumerate(selected)
    )


def _mpc_config() -> MPCConfig:
    plant = PlantConfig.research_simulation()
    return MPCConfig(
        timescale=TimeScaleConfig.provisional(),
        fuel_cell_rated_kw=plant.fuel_cell_rated_total_kw,
        battery_capacity_kwh=plant.battery_nominal_energy_kwh,
        battery_efficiency=formal_battery_efficiency(),
        battery_charge_min_kw=plant.battery_charge_min_kw,
        battery_discharge_max_kw=plant.battery_discharge_max_kw,
        soc_min=SOC_HARD_MIN,
        soc_max=SOC_HARD_MAX,
        soc_deadband_low=SOC_WORKING_LOW,
        soc_deadband_high=SOC_WORKING_HIGH,
        p_fc_scale_kw=P_FC_OBJECTIVE_SCALE_KW,
        delta_p_fc_scale_kw=DELTA_P_FC_OBJECTIVE_SCALE_KW,
        soc_scale=SOC_OBJECTIVE_SCALE,
        fuel_cell_ramp_kw_per_step=None,
    )


def solve_audit_case(case: ObjectiveAuditCase, action: ActionCandidate):
    if type(case) is not ObjectiveAuditCase or type(action) is not ActionCandidate:
        raise TypeError("case and action must be exact audit values")
    if len(case.payload) < 3:
        raise ValueError("audit case payload is incomplete")
    load, soc, previous_fc, *history = case.payload
    config = _mpc_config()
    base_filter = CausalBaseLoadFilter(
        sample_seconds=config.timescale.ts_mpc_seconds,
        tau_seconds=AUDIT_TAU_LPF_SECONDS,
    )
    for historical_load in history:
        base_filter.commit(historical_load)
    return NonlinearMPC(config).solve(
        observed_load_kw=load,
        current_soc=soc,
        previous_executed_p_fc_kw=previous_fc,
        weights=action.to_mpc_weights(),
        base_load_filter=base_filter,
    )


def run_train_objective_scale_audit(
    states: Sequence[AuditReadyState],
    provenance: DatasetProvenance,
) -> ObjectiveScaleAuditResult:
    cases = select_representative_cases(states)
    if not cases:
        raise ValueError("no hard-bound-valid representative Train cases")
    return run_objective_scale_audit(
        provenance=provenance,
        cases_loader=lambda: cases,
        actions=CANDIDATE_ACTION_BANK,
        solver_runner=solve_audit_case,
        behavior_tolerance=BehaviorTolerance(1e-12, 1e-6, 1e-10),
    )


__all__ = [
    "AUDIT_TAU_LPF_SECONDS",
    "AuditReadyState",
    "run_train_objective_scale_audit",
    "select_representative_cases",
    "solve_audit_case",
]
