from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pandas as pd

from mpc.controllers.reference_generator import (
    CasadiReferenceGenerator,
    DualSideCasadiReferenceGenerator,
    DualSideScipyReferenceGenerator,
    HeuristicReferenceGenerator,
    ScipyReferenceGenerator,
    attach_reference_columns,
)
from mpc.solvers.casadi_solver import CasadiMPCConfig, casadi_available
from mpc.solvers.scipy_solver import scipy_available


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_casadi_mpc_config(
    project_root: str | Path | None = None,
    base_config: CasadiMPCConfig | None = None,
    **overrides,
) -> CasadiMPCConfig:
    config = base_config or CasadiMPCConfig()
    if project_root is None:
        project_root = Path.cwd()

    try:
        from utils.config_loader import load_project_config
    except ModuleNotFoundError:
        mpc_config = {}
    else:
        mpc_config = load_project_config(project_root).get("mpc", {})

    updates = {}
    if "horizon_steps" in mpc_config:
        updates["prediction_horizon"] = int(mpc_config["horizon_steps"])
    if "sample_time_seconds" in mpc_config:
        updates["dt_hours"] = float(mpc_config["sample_time_seconds"]) / 3600.0
    if "soc_ref" in mpc_config:
        updates["soc_target"] = float(mpc_config["soc_ref"])
    if "enable_battery_use_in_mpc" in mpc_config:
        updates["enable_battery_use_in_mpc"] = _as_bool(mpc_config["enable_battery_use_in_mpc"])

    objective_weights = mpc_config.get("objective_weights", {})
    objective_mapping = {
        "q1": "q1_fuel_cost",
        "q2": "q2_fc_smooth",
        "q3": "q3_soc_dev",
        "q4": "q4_battery_use",
        "q5": "q5_soc_terminal",
    }
    for source_key, target_key in objective_mapping.items():
        if source_key in objective_weights:
            updates[target_key] = float(objective_weights[source_key])

    updates.update(overrides)
    valid_updates = {key: value for key, value in updates.items() if hasattr(config, key)}
    return replace(config, **valid_updates)


class UpperMPCController:
    """Upper-layer rolling MPC reference generator.

    The controller prefers strict constrained solvers. CasADi/IPOPT is used if
    installed; otherwise SciPy/SLSQP is used so the project does not silently
    fall back to the heuristic rule during normal vessel-data runs.
    """

    def __init__(self, prefer_casadi: bool = True, casadi_config: CasadiMPCConfig | None = None):
        self.uses_casadi = False
        self.mode = "heuristic"
        self.casadi_config = casadi_config or build_casadi_mpc_config()
        self.reference_generator = HeuristicReferenceGenerator()
        if prefer_casadi and casadi_available():
            try:
                self.reference_generator = DualSideCasadiReferenceGenerator(self.casadi_config)
                self.uses_casadi = True
                self.mode = "casadi_mpc_dual_side"
            except Exception:
                try:
                    self.reference_generator = CasadiReferenceGenerator(self.casadi_config)
                    self.uses_casadi = True
                    self.mode = "casadi_mpc"
                except Exception:
                    self.reference_generator = HeuristicReferenceGenerator()
                    self.uses_casadi = False
                    self.mode = "heuristic_fallback"
        elif scipy_available():
            try:
                self.reference_generator = DualSideScipyReferenceGenerator(self.casadi_config)
                self.uses_casadi = False
                self.mode = "scipy_mpc_dual_side"
            except Exception:
                try:
                    self.reference_generator = ScipyReferenceGenerator(self.casadi_config)
                    self.uses_casadi = False
                    self.mode = "scipy_mpc"
                except Exception:
                    self.reference_generator = HeuristicReferenceGenerator()
                    self.uses_casadi = False
                    self.mode = "heuristic_fallback"

    def annotate_dataset(self, dataset: pd.DataFrame, solve_stride: int = 1) -> pd.DataFrame:
        return attach_reference_columns(
            dataset,
            generator=self.reference_generator,
            source_name=self.mode,
            solve_stride=solve_stride,
        )

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "uses_casadi": self.uses_casadi,
            "uses_strict_mpc": self.mode in {"casadi_mpc_dual_side", "casadi_mpc", "scipy_mpc_dual_side", "scipy_mpc"},
            "application": "hydrogen_vessel_microgrid",
            "grid_buy_sell_model": False,
            "side_reference_coordination": True,
            "casadi_config": asdict(self.casadi_config),
        }
