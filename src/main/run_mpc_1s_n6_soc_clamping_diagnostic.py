from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc_solvers.mpc_qp_formulation import QpMpcConfig
from run_mpc_1s_n6_qsoc_feasibility import qsoc_candidate_config


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    profile_kind: str
    candidate_id: str
    q_soc: float
    initial_soc: float


def build_constant_profile() -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(3601, dtype=float)
    loads = np.full(times.shape, 300.0, dtype=float)
    return times, loads


def build_pulse_profile() -> tuple[np.ndarray, np.ndarray]:
    times, constant_loads = build_constant_profile()
    loads = constant_loads.copy()
    loads[(times >= 600.0) & (times < 720.0)] = 450.0
    return times, loads


def clamping_candidate_config(q_soc: float) -> QpMpcConfig:
    normalized_q_soc = float(q_soc)
    if not np.isfinite(normalized_q_soc) or normalized_q_soc not in (10.0, 20.0):
        raise ValueError("q_soc must be one of {10.0, 20.0}")
    candidate_id = {
        10.0: "QSOC_10",
        20.0: "QSOC_20",
    }[normalized_q_soc]
    return qsoc_candidate_config(candidate_id)


def build_case_matrix() -> list[SyntheticCase]:
    constant_cases = [
        SyntheticCase(
            case_id=f"constant_soc{int(round(initial_soc * 100)):03d}_qsoc{int(q_soc)}",
            profile_kind="constant",
            candidate_id=f"QSOC_{int(q_soc)}",
            q_soc=q_soc,
            initial_soc=initial_soc,
        )
        for q_soc in (10.0, 20.0)
        for initial_soc in (0.53, 0.55, 0.57)
    ]
    pulse_cases = [
        SyntheticCase(
            case_id=f"pulse_soc055_qsoc{int(q_soc)}",
            profile_kind="pulse",
            candidate_id=f"QSOC_{int(q_soc)}",
            q_soc=q_soc,
            initial_soc=0.55,
        )
        for q_soc in (10.0, 20.0)
    ]
    return constant_cases + pulse_cases
