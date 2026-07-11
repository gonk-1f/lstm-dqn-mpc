from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mpc.solvers.fc_dp0_curve import (
    CURVE_SOURCE_LABEL,
    dp0_quadratic_coefficients,
    eta_dp0,
    h2_kg_step_dp0,
    h2_kg_step_dp0_quadratic,
    h2_rate_gps_dp0,
)


P_CURVE_RATIO = np.array([0.10, 0.20, 0.40, 0.60, 0.80, 1.00], dtype=float)
ETA_CURVE_DP0 = np.array([0.58, 0.56, 0.53, 0.50, 0.46, 0.42], dtype=float)
LHV_H2_KWH_PER_KG_DEFAULT = 33.3
FC_ETA_POLY_A2_DEFAULT = -0.03503869
FC_ETA_POLY_A1_DEFAULT = -0.13600602
FC_ETA_POLY_A0_DEFAULT = 0.59150903
FC_ETA_MIN_DEFAULT = 0.30
FC_ETA_MAX_DEFAULT = 0.60
H2_ZERO_POWER_EPS_KW = 1.0e-6


def _clip(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))


def fc_efficiency_from_ratio(
    p_fc_kw: np.ndarray | float,
    p_fc_rated_kw: float = 560.0,
    eta_min: float = FC_ETA_MIN_DEFAULT,
    eta_max: float = FC_ETA_MAX_DEFAULT,
    fc_eta_poly_a2: float = FC_ETA_POLY_A2_DEFAULT,
    fc_eta_poly_a1: float = FC_ETA_POLY_A1_DEFAULT,
    fc_eta_poly_a0: float = FC_ETA_POLY_A0_DEFAULT,
    eps_kw: float = H2_ZERO_POWER_EPS_KW,
) -> np.ndarray:
    """Fresh Dp=0 FC efficiency from the exported CSV curve."""

    del eta_min, eta_max, fc_eta_poly_a2, fc_eta_poly_a1, fc_eta_poly_a0, eps_kw
    return eta_dp0(p_fc_kw, p_rated_total_kw=p_fc_rated_kw)


def h2_consumption_kg(
    p_fc_kw: np.ndarray | float,
    dt_seconds: float = 30.0,
    p_fc_rated_kw: float = 560.0,
    lhv_h2_kwh_per_kg: float = LHV_H2_KWH_PER_KG_DEFAULT,
    eta_min: float = FC_ETA_MIN_DEFAULT,
    eta_max: float = FC_ETA_MAX_DEFAULT,
    fc_eta_poly_a2: float = FC_ETA_POLY_A2_DEFAULT,
    fc_eta_poly_a1: float = FC_ETA_POLY_A1_DEFAULT,
    fc_eta_poly_a0: float = FC_ETA_POLY_A0_DEFAULT,
    eps_kw: float = H2_ZERO_POWER_EPS_KW,
) -> np.ndarray:
    """Hydrogen mass estimate from the exported Dp=0 curve."""

    del lhv_h2_kwh_per_kg, eta_min, eta_max, fc_eta_poly_a2, fc_eta_poly_a1, fc_eta_poly_a0, eps_kw
    return h2_kg_step_dp0(p_fc_kw, dt_seconds=dt_seconds, p_rated_total_kw=p_fc_rated_kw)


def h2_reference_kg_per_step(
    p_fc_rated_kw: float = 560.0,
    dt_seconds: float = 30.0,
    lhv_h2_kwh_per_kg: float = LHV_H2_KWH_PER_KG_DEFAULT,
    eta_min: float = FC_ETA_MIN_DEFAULT,
    eta_max: float = FC_ETA_MAX_DEFAULT,
    fc_eta_poly_a2: float = FC_ETA_POLY_A2_DEFAULT,
    fc_eta_poly_a1: float = FC_ETA_POLY_A1_DEFAULT,
    fc_eta_poly_a0: float = FC_ETA_POLY_A0_DEFAULT,
) -> float:
    del lhv_h2_kwh_per_kg, eta_min, eta_max, fc_eta_poly_a2, fc_eta_poly_a1, fc_eta_poly_a0
    return float(h2_kg_step_dp0(float(p_fc_rated_kw), dt_seconds=float(dt_seconds), p_rated_total_kw=float(p_fc_rated_kw)))


def h2_sanity_check_at_figure_rated(
    figure_rated_kw: float = 100.0,
    lhv_h2_kwh_per_kg: float = LHV_H2_KWH_PER_KG_DEFAULT,
    eta_min: float = FC_ETA_MIN_DEFAULT,
    eta_max: float = FC_ETA_MAX_DEFAULT,
    fc_eta_poly_a2: float = FC_ETA_POLY_A2_DEFAULT,
    fc_eta_poly_a1: float = FC_ETA_POLY_A1_DEFAULT,
    fc_eta_poly_a0: float = FC_ETA_POLY_A0_DEFAULT,
) -> dict[str, float | bool]:
    del lhv_h2_kwh_per_kg, eta_min, eta_max, fc_eta_poly_a2, fc_eta_poly_a1, fc_eta_poly_a0
    eta_rated = float(eta_dp0(float(figure_rated_kw), p_rated_total_kw=float(figure_rated_kw)))
    mdot_gps = float(h2_rate_gps_dp0(float(figure_rated_kw), p_rated_total_kw=float(figure_rated_kw)))
    return {
        "eta_rated": eta_rated,
        "sanity_check_mdot_gps_at_100kW": mdot_gps,
        "sanity_check_uses_exported_dp0_curve": True,
    }


def _eta_poly_at_ratio(config: "CasadiMPCConfig", ratio: float) -> float:
    p_fc_kw = float(config.fuel_cell_max_kw) * float(ratio)
    eta = float(eta_dp0(p_fc_kw, p_rated_total_kw=float(config.fuel_cell_max_kw)))
    return min(float(config.fc_eta_max), max(float(config.fc_eta_min), eta))


def _h2_ref_kg_per_step(config: "CasadiMPCConfig") -> float:
    return h2_reference_kg_per_step(
        p_fc_rated_kw=float(config.fuel_cell_max_kw),
        dt_seconds=float(config.dt_hours) * 3600.0,
    )


def _effective_config_info(config: "CasadiMPCConfig") -> dict[str, float | bool | str]:
    return {
        "effective_q_h2": float(config.q_h2),
        "effective_q_soc": float(config.q_soc),
        "effective_q_batt": float(config.q_batt),
        "effective_q_ramp": float(config.q_ramp),
        "effective_q_terminal_soc": float(_terminal_soc_weight(config)),
        "effective_battery_capacity_kwh": float(config.battery_capacity_kwh),
        "effective_fuel_cell_ramp_constraint_enabled": bool(config.fuel_cell_ramp_constraint_enabled),
        "effective_use_dimensionless_objective": bool(config.use_dimensionless_objective),
        "effective_normalize_h2_cost": bool(config.normalize_h2_cost),
        "effective_soc_band": float(config.soc_band),
        "effective_terminal_soc_band": float(config.terminal_soc_band),
    }


def casadi_available() -> bool:
    try:
        import casadi  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@dataclass
class CasadiMPCConfig:
    prediction_horizon: int = 8
    dt_hours: float = 1.0 / 60.0
    battery_capacity_kwh: float = 1067.0
    battery_charge_max_kw: float = 350.0
    battery_discharge_max_kw: float = 350.0
    fuel_cell_min_kw: float = 0.0
    fuel_cell_max_kw: float = 240.0
    fuel_cell_ramp_kw: float = 40.0
    fuel_cell_ramp_constraint_enabled: bool = True
    initial_fc_mode: str = "initial_current_load"
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_target: float = 0.65
    soc_reference_mode: str = "fixed_target"
    soc_reserve: float = 0.55
    terminal_soc_band: float = 0.02
    # Legacy weights (normalized objective)
    q1_fuel_cost: float = 0.40
    q2_fc_smooth: float = 0.30
    q3_soc_dev: float = 1.00
    q4_battery_use: float = 8.00
    q5_soc_terminal: float = 0.00
    enable_battery_use_in_mpc: bool = True
    ipopt_max_iter: int = 200
    # Raw objective mode (no normalization, MW for power², kW² for ramp)
    use_raw_objective: bool = False
    raw_soc_squared: bool = True  # True=(SOC-SOC_ref)², False=|SOC-SOC_ref|
    raw_fc_energy_linear: bool = False  # True=q_fc*sum(P_fc)*dt_h, False=q_fc*sum(P_fc²)
    enable_terminal_soc_soft_penalty: bool = False
    objective_mode: str = "legacy"
    use_dimensionless_objective: bool = False
    use_h2_mass_cost: bool = False
    normalize_h2_cost: bool = True
    lhv_h2_kwh_per_kg: float = LHV_H2_KWH_PER_KG_DEFAULT
    fc_eta_poly_a2: float = FC_ETA_POLY_A2_DEFAULT
    fc_eta_poly_a1: float = FC_ETA_POLY_A1_DEFAULT
    fc_eta_poly_a0: float = FC_ETA_POLY_A0_DEFAULT
    fc_eta_min: float = FC_ETA_MIN_DEFAULT
    fc_eta_max: float = FC_ETA_MAX_DEFAULT
    soc_band: float = 0.05
    q_h2: float = 1.0
    q_soc: float = 5.0        # weight on SOC deviation
    q_fc: float = 100.0       # weight on P_fc²
    q_batt: float = 1000.0    # weight on P_batt²
    q_ramp: float = 0.1       # weight on ΔP_fc²
    q_terminal_soc: float | None = None
    enable_fc_post_filter: bool = False
    battery_throughput_penalty_enabled: bool = False
    battery_throughput_penalty_type: str = "absolute_power"
    battery_throughput_normalization_kw: float = 350.0
    soc_penalty_type: str = "symmetric_tracking"
    ipopt_tol: float = 1e-5


def _terminal_soc_weight(config: CasadiMPCConfig) -> float:
    return float(config.q_soc if config.q_terminal_soc is None else config.q_terminal_soc)


def _validated_soc_reference_mode(config: CasadiMPCConfig) -> str:
    mode = str(config.soc_reference_mode).strip().lower()
    if mode not in {"fixed_target", "initial_soc", "reserve_only"}:
        raise ValueError(f"Unsupported soc_reference_mode: {config.soc_reference_mode!r}")
    return mode


def _validated_soc_penalty_type(config: CasadiMPCConfig) -> str:
    penalty_type = str(config.soc_penalty_type).strip().lower()
    if penalty_type not in {"symmetric_tracking", "lower_deviation_only", "reserve_only"}:
        raise ValueError(f"Unsupported soc_penalty_type: {config.soc_penalty_type!r}")
    return penalty_type


def resolve_soc_reference(
    config: CasadiMPCConfig,
    current_soc: float,
    soc_reference_value: float | None = None,
) -> float:
    mode = _validated_soc_reference_mode(config)
    if mode == "fixed_target":
        return float(config.soc_target)
    if mode == "reserve_only":
        return float(config.soc_reserve)
    return float(current_soc if soc_reference_value is None else soc_reference_value)


def _deadband_square(error: np.ndarray | float, band: float, *, normalize: bool) -> np.ndarray:
    excess = np.maximum(np.abs(np.asarray(error, dtype=float)) - max(float(band), 0.0), 0.0)
    if normalize:
        excess = excess / max(float(band), 1e-6)
    return excess**2


def compute_soc_stage_cost(
    config: CasadiMPCConfig,
    soc: np.ndarray | float,
    soc_reference_value: float,
    *,
    normalize: bool = False,
) -> np.ndarray | float:
    mode = _validated_soc_reference_mode(config)
    penalty_type = _validated_soc_penalty_type(config)
    soc_values = np.asarray(soc, dtype=float)
    if mode == "reserve_only" or penalty_type == "reserve_only":
        cost = np.maximum(float(config.soc_reserve) - soc_values, 0.0) ** 2
    elif penalty_type == "lower_deviation_only":
        excess = np.maximum(float(soc_reference_value) - soc_values - max(float(config.soc_band), 0.0), 0.0)
        if normalize:
            excess = excess / max(float(config.soc_band), 1e-6)
        cost = excess**2
    else:
        cost = _deadband_square(soc_values - float(soc_reference_value), config.soc_band, normalize=normalize)
    return float(cost) if cost.ndim == 0 else cost


def compute_soc_terminal_cost(
    config: CasadiMPCConfig,
    terminal_soc: float,
    soc_reference_value: float,
    *,
    normalize: bool = False,
) -> float:
    mode = _validated_soc_reference_mode(config)
    penalty_type = _validated_soc_penalty_type(config)
    if mode == "reserve_only" or penalty_type == "reserve_only":
        return float(max(float(config.soc_reserve) - float(terminal_soc), 0.0) ** 2)
    if penalty_type == "lower_deviation_only":
        excess = max(
            float(soc_reference_value) - float(terminal_soc) - max(float(config.terminal_soc_band), 0.0),
            0.0,
        )
        if normalize:
            excess = excess / max(float(config.terminal_soc_band), 1e-6)
        return float(excess**2)
    return float(
        _deadband_square(
            float(terminal_soc) - float(soc_reference_value),
            config.terminal_soc_band,
            normalize=normalize,
        )
    )


@dataclass
class CasadiMPCResult:
    fuel_cell_ref_kw: float
    battery_ref_kw: float
    predicted_soc: float
    soc_ref: float
    fuel_cell_ref_traj_kw: list[float]
    battery_ref_traj_kw: list[float]
    soc_pred_traj: list[float]
    objective_value: float
    objective_info: dict
    success: bool


@dataclass
class DualSideCasadiMPCResult:
    fuel_cell_ref_left_kw: float
    fuel_cell_ref_right_kw: float
    battery_ref_left_kw: float
    battery_ref_right_kw: float
    predicted_soc_left: float
    predicted_soc_right: float
    soc_ref: float
    fuel_cell_ref_left_traj_kw: list[float]
    fuel_cell_ref_right_traj_kw: list[float]
    battery_ref_left_traj_kw: list[float]
    battery_ref_right_traj_kw: list[float]
    soc_pred_left_traj: list[float]
    soc_pred_right_traj: list[float]
    objective_value: float
    objective_info: dict
    success: bool


class ShipCasadiMPC:
    """Upper-layer MPC solved by CasADi/IPOPT.

    This is the phase-1 single-load, single-battery, single-fuel-cell model.
    The class is kept in `src/mpc` so it can replace the heuristic generator
    without changing the surrounding hierarchical interfaces.
    """

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not casadi_available():
            raise ModuleNotFoundError("casadi is not available in the current interpreter.")
        self.config = config or CasadiMPCConfig()
        self._last_solution: np.ndarray | None = None
        self._build_solver()

    def _build_solver(self) -> None:
        from casadi import SX, fabs, fmax, fmin, nlpsol, sqrt, vertcat

        cfg = self.config
        N = cfg.prediction_horizon
        raw_physical_objective = str(cfg.objective_mode).strip().lower() == "raw_physical"
        dp0_a1 = dp0_a2 = None
        if (raw_physical_objective or cfg.use_dimensionless_objective) and cfg.use_h2_mass_cost:
            dp0_a1, dp0_a2 = dp0_quadratic_coefficients()
        soc = SX.sym("soc", N + 1)
        p_batt = SX.sym("p_batt", N)
        p_fc = SX.sym("p_fc", N)
        p_batt_abs = SX.sym("p_batt_abs", N) if raw_physical_objective else None
        load = SX.sym("load", N)
        terminal_load = SX.sym("terminal_load")
        prev_fc = SX.sym("prev_fc")
        soc0 = SX.sym("soc0")
        soc_reference = SX.sym("soc_reference")

        x = vertcat(soc, p_batt, p_fc, p_batt_abs) if raw_physical_objective else vertcat(soc, p_batt, p_fc)
        p = vertcat(load, terminal_load, prev_fc, soc0, soc_reference)

        g = []
        lbg = []
        ubg = []
        J = 0
        battery_abs_max = max(float(cfg.battery_charge_max_kw), float(cfg.battery_discharge_max_kw))
        soc_reference_mode = _validated_soc_reference_mode(cfg)
        soc_penalty_type = _validated_soc_penalty_type(cfg)

        def soc_penalty(value, band: float, *, normalize: bool):
            if soc_reference_mode == "reserve_only" or soc_penalty_type == "reserve_only":
                return fmax(float(cfg.soc_reserve) - value, 0.0) ** 2
            if soc_penalty_type == "lower_deviation_only":
                excess = fmax(soc_reference - value - max(float(band), 0.0), 0.0)
            else:
                excess = fmax(fabs(value - soc_reference) - max(float(band), 0.0), 0.0)
            if normalize:
                excess = excess / max(float(band), 1e-6)
            return excess**2

        g.append(soc[0] - soc0)
        lbg.append(0.0)
        ubg.append(0.0)

        for k in range(N):
            next_soc = soc[k] - p_batt[k] * cfg.dt_hours / cfg.battery_capacity_kwh
            g.append(soc[k + 1] - next_soc)
            lbg.append(0.0)
            ubg.append(0.0)

            g.append(p_fc[k] + p_batt[k] - load[k])
            lbg.append(0.0)
            ubg.append(0.0)

            delta_fc = p_fc[k] - (prev_fc if k == 0 else p_fc[k - 1])
            if cfg.fuel_cell_ramp_constraint_enabled:
                g.append(delta_fc)
                lbg.append(-cfg.fuel_cell_ramp_kw)
                ubg.append(cfg.fuel_cell_ramp_kw)

            if raw_physical_objective:
                g.append(p_batt_abs[k] - p_batt[k])
                lbg.append(0.0)
                ubg.append(2.0 * battery_abs_max)
                g.append(p_batt_abs[k] + p_batt[k])
                lbg.append(0.0)
                ubg.append(2.0 * battery_abs_max)

                ratio = p_fc[k] / max(cfg.fuel_cell_max_kw, 1e-6)
                h2_rate_gps = (cfg.fuel_cell_max_kw / 100.0) * (dp0_a1 * ratio + dp0_a2 * ratio**2)
                h2_mass = h2_rate_gps * (cfg.dt_hours * 3600.0) / 1000.0
                batt_energy_kwh = p_batt_abs[k] * cfg.dt_hours
                soc_error = soc[k + 1] - soc_reference
                J += cfg.q_h2 * h2_mass
                J += cfg.q_soc * soc_error**2
                if cfg.enable_battery_use_in_mpc:
                    J += cfg.q_batt * batt_energy_kwh
                J += cfg.q_ramp * delta_fc**2
            elif cfg.use_raw_objective:
                if cfg.raw_fc_energy_linear:
                    J += cfg.q_fc * p_fc[k] * cfg.dt_hours
                else:
                    J += cfg.q_fc * p_fc[k] ** 2
                J += cfg.q_ramp * delta_fc**2
                J += cfg.q_soc * soc_penalty(soc[k + 1], cfg.soc_band, normalize=False)
                if cfg.enable_battery_use_in_mpc:
                    if cfg.battery_throughput_penalty_enabled and cfg.battery_throughput_penalty_type == "absolute_power":
                        batt_use = sqrt(p_batt[k] ** 2 + 1e-6) / max(cfg.battery_throughput_normalization_kw, 1e-6)
                        J += cfg.q_batt * batt_use
                    else:
                        J += cfg.q_batt * p_batt[k] ** 2
            elif cfg.use_dimensionless_objective:
                if cfg.use_h2_mass_cost:
                    ratio = fmin(fmax(p_fc[k] / max(cfg.fuel_cell_max_kw, 1e-6), 0.0), 1.0)
                    h2_rate_gps = (cfg.fuel_cell_max_kw / 100.0) * (dp0_a1 * ratio + dp0_a2 * ratio**2)
                    h2_mass = h2_rate_gps * (cfg.dt_hours * 3600.0) / 1000.0
                    h2_ref = max(_h2_ref_kg_per_step(cfg), 1e-12)
                    h2_cost = h2_mass / h2_ref if cfg.normalize_h2_cost else h2_mass
                    J += cfg.q_h2 * h2_cost
                else:
                    J += cfg.q_fc * p_fc[k] / max(cfg.fuel_cell_max_kw, 1e-6)
                ramp_norm = delta_fc / max(cfg.fuel_cell_ramp_kw, 1e-6)
                J += cfg.q_ramp * ramp_norm**2
                J += cfg.q_soc * soc_penalty(soc[k + 1], cfg.soc_band, normalize=True)
                if cfg.enable_battery_use_in_mpc:
                    if cfg.battery_throughput_penalty_enabled and cfg.battery_throughput_penalty_type == "absolute_power":
                        batt_norm = sqrt(p_batt[k] ** 2 + 1e-6) / max(
                            cfg.battery_throughput_normalization_kw, 1e-6
                        )
                        J += cfg.q_batt * batt_norm
                    else:
                        batt_norm = p_batt[k] / max(cfg.battery_discharge_max_kw, 1e-6)
                        J += cfg.q_batt * batt_norm**2
            else:
                fuel_cost_norm = p_fc[k] / max(cfg.fuel_cell_max_kw, 1e-6)
                fc_smooth_norm = sqrt(delta_fc**2 + 1e-6) / max(cfg.fuel_cell_max_kw, 1e-6)
                soc_dev_norm = sqrt(soc_penalty(soc[k + 1], cfg.soc_band, normalize=False) + 1e-12) / max(
                    cfg.soc_max - cfg.soc_min, 1e-6
                )
                battery_use_norm = (p_batt[k] / max(cfg.battery_discharge_max_kw, 1e-6)) ** 2

                J += cfg.q1_fuel_cost * fuel_cost_norm
                J += cfg.q2_fc_smooth * fc_smooth_norm
                J += cfg.q3_soc_dev * soc_dev_norm
                if cfg.enable_battery_use_in_mpc:
                    J += cfg.q4_battery_use * battery_use_norm
        terminal_soc = soc[N]
        terminal_soc_norm = sqrt(soc_penalty(terminal_soc, cfg.terminal_soc_band, normalize=False) + 1e-12) / max(
            cfg.soc_max - cfg.soc_min, 1e-6
        )
        if raw_physical_objective:
            pass
        elif cfg.use_raw_objective:
            if cfg.enable_terminal_soc_soft_penalty:
                J += _terminal_soc_weight(cfg) * soc_penalty(
                    terminal_soc, cfg.terminal_soc_band, normalize=False
                )
        elif cfg.use_dimensionless_objective:
            if cfg.enable_terminal_soc_soft_penalty:
                J += _terminal_soc_weight(cfg) * soc_penalty(
                    terminal_soc, cfg.terminal_soc_band, normalize=True
                )
        else:
            J += cfg.q5_soc_terminal * terminal_soc_norm

        lbx = []
        ubx = []
        lbx.extend([cfg.soc_min] * (N + 1))
        ubx.extend([cfg.soc_max] * (N + 1))
        lbx.extend([-cfg.battery_charge_max_kw] * N)
        ubx.extend([cfg.battery_discharge_max_kw] * N)
        lbx.extend([cfg.fuel_cell_min_kw] * N)
        ubx.extend([cfg.fuel_cell_max_kw] * N)
        if raw_physical_objective:
            lbx.extend([0.0] * N)
            ubx.extend([battery_abs_max] * N)

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(cfg.ipopt_max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "ipopt.acceptable_tol": 1e-4,
            "print_time": 0,
        }
        self.solver = nlpsol("ship_upper_mpc", "ipopt", {"f": J, "x": x, "g": vertcat(*g), "p": p}, opts)
        self.lbx = np.asarray(lbx, dtype=float)
        self.ubx = np.asarray(ubx, dtype=float)
        self.lbg = np.asarray(lbg, dtype=float)
        self.ubg = np.asarray(ubg, dtype=float)

    def solve(
        self,
        current_soc: float,
        prev_fc_kw: float,
        load_forecast_kw: np.ndarray,
        terminal_load_kw: float | None = None,
        soc_reference_value: float | None = None,
    ) -> CasadiMPCResult:
        cfg = self.config
        load_forecast_kw = np.asarray(load_forecast_kw, dtype=float).reshape(-1)
        if load_forecast_kw.shape[0] != cfg.prediction_horizon:
            raise ValueError(f"Expected horizon {cfg.prediction_horizon}, got {load_forecast_kw.shape[0]}")

        current_soc = _clip(current_soc, cfg.soc_min, cfg.soc_max)
        resolved_soc_reference = resolve_soc_reference(cfg, current_soc, soc_reference_value)
        prev_fc_kw = _clip(prev_fc_kw, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw)
        if self._last_solution is None:
            x0 = np.zeros_like(self.lbx)
            x0[: cfg.prediction_horizon + 1] = current_soc
            start = cfg.prediction_horizon + 1
            x0[start : start + cfg.prediction_horizon] = np.clip(
                load_forecast_kw * 0.3,
                -cfg.battery_charge_max_kw,
                cfg.battery_discharge_max_kw,
            )
            start += cfg.prediction_horizon
            x0[start : start + cfg.prediction_horizon] = np.clip(
                load_forecast_kw * 0.7,
                cfg.fuel_cell_min_kw,
                cfg.fuel_cell_max_kw,
            )
            if str(cfg.objective_mode).strip().lower() == "raw_physical":
                start += cfg.prediction_horizon
                p_batt_start = cfg.prediction_horizon + 1
                x0[start : start + cfg.prediction_horizon] = np.abs(
                    x0[p_batt_start : p_batt_start + cfg.prediction_horizon]
                )
        else:
            x0 = self._last_solution.copy()
            x0[: cfg.prediction_horizon + 1] = np.clip(current_soc, cfg.soc_min, cfg.soc_max)

        solve_lbx = self.lbx.copy()
        solve_ubx = self.ubx.copy()
        p_batt_offset = cfg.prediction_horizon + 1
        p_fc_offset = p_batt_offset + cfg.prediction_horizon
        x0 = np.minimum(np.maximum(x0, solve_lbx), solve_ubx)

        terminal_load_kw = float(load_forecast_kw[-1] if terminal_load_kw is None else terminal_load_kw)
        params = np.concatenate(
            [load_forecast_kw, [terminal_load_kw, prev_fc_kw, current_soc, resolved_soc_reference]]
        )
        solution = self.solver(x0=x0, lbx=solve_lbx, ubx=solve_ubx, lbg=self.lbg, ubg=self.ubg, p=params)
        stats = self.solver.stats()
        values = np.asarray(solution["x"]).reshape(-1)
        self._last_solution = values.copy()

        offset = 0
        soc = values[offset : offset + cfg.prediction_horizon + 1]
        offset += cfg.prediction_horizon + 1
        p_batt = values[offset : offset + cfg.prediction_horizon]
        offset += cfg.prediction_horizon
        p_fc = values[offset : offset + cfg.prediction_horizon]

        objective_info = _build_single_objective_info(
            p_fc_traj=p_fc,
            p_bat_traj=p_batt,
            soc_traj=soc,
            prev_fc_kw=prev_fc_kw,
            config=cfg,
            terminal_load_kw=terminal_load_kw,
            soc_reference_value=resolved_soc_reference,
        )
        objective_info = {
            **objective_info,
            "enable_fc_post_filter": bool(cfg.enable_fc_post_filter),
            "soc_reference_mode": str(cfg.soc_reference_mode),
            "soc_penalty_type": str(cfg.soc_penalty_type),
            "soc_ref_value": float(resolved_soc_reference),
            "soc_reserve": float(cfg.soc_reserve),
            "P_fc_upper_bound": float(solve_ubx[p_fc_offset]),
            "P_fc_lower_bound": float(solve_lbx[p_fc_offset]),
            "battery_discharge_upper_bound": float(solve_ubx[p_batt_offset]),
            "battery_charge_upper_bound": float(-solve_lbx[p_batt_offset]),
            "battery_discharge_limit_active": bool(
                solve_ubx[p_batt_offset] < float(cfg.battery_discharge_max_kw) - 1e-9
            ),
            "battery_charge_limit_active": bool(
                solve_lbx[p_batt_offset] > -float(cfg.battery_charge_max_kw) + 1e-9
            ),
            "solver_message": str(stats.get("return_status", "")),
        }

        return CasadiMPCResult(
            fuel_cell_ref_kw=float(p_fc[0]),
            battery_ref_kw=float(p_batt[0]),
            predicted_soc=float(soc[1]),
            soc_ref=float(resolved_soc_reference),
            fuel_cell_ref_traj_kw=[float(v) for v in p_fc],
            battery_ref_traj_kw=[float(v) for v in p_batt],
            soc_pred_traj=[float(v) for v in soc],
            objective_value=float(solution["f"]),
            objective_info=objective_info,
            success=bool(stats.get("success", True)),
        )


class ShipDualSideCasadiMPC:
    """Dual-side upper MPC with left/right battery and fuel-cell references."""

    def __init__(self, config: CasadiMPCConfig | None = None):
        if not casadi_available():
            raise ModuleNotFoundError("casadi is not available in the current interpreter.")
        self.config = config or CasadiMPCConfig()
        self._last_solution: np.ndarray | None = None
        self._build_solver()

    def _build_solver(self) -> None:
        from casadi import SX, fmax, fmin, nlpsol, sqrt, vertcat

        cfg = self.config
        N = cfg.prediction_horizon
        soc_left = SX.sym("soc_left", N + 1)
        soc_right = SX.sym("soc_right", N + 1)
        p_batt_left = SX.sym("p_batt_left", N)
        p_batt_right = SX.sym("p_batt_right", N)
        p_fc_left = SX.sym("p_fc_left", N)
        p_fc_right = SX.sym("p_fc_right", N)
        load_left = SX.sym("load_left", N)
        load_right = SX.sym("load_right", N)
        terminal_load_left = SX.sym("terminal_load_left")
        terminal_load_right = SX.sym("terminal_load_right")
        prev_fc_left = SX.sym("prev_fc_left")
        prev_fc_right = SX.sym("prev_fc_right")
        soc0_left = SX.sym("soc0_left")
        soc0_right = SX.sym("soc0_right")

        x = vertcat(
            soc_left,
            soc_right,
            p_batt_left,
            p_batt_right,
            p_fc_left,
            p_fc_right,
        )
        p = vertcat(
            load_left,
            load_right,
            terminal_load_left,
            terminal_load_right,
            prev_fc_left,
            prev_fc_right,
            soc0_left,
            soc0_right,
        )

        g = []
        lbg = []
        ubg = []
        J = 0

        g += [soc_left[0] - soc0_left, soc_right[0] - soc0_right]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

        if cfg.use_dimensionless_objective and cfg.use_h2_mass_cost:
            dp0_a1, dp0_a2 = dp0_quadratic_coefficients()
        else:
            dp0_a1, dp0_a2 = 0.0, 0.0
        half_capacity = cfg.battery_capacity_kwh / 2.0
        half_fc_max = cfg.fuel_cell_max_kw / 2.0
        half_charge_max = cfg.battery_charge_max_kw / 2.0
        half_discharge_max = cfg.battery_discharge_max_kw / 2.0
        half_ramp = cfg.fuel_cell_ramp_kw / 2.0

        for k in range(N):
            next_soc_left = soc_left[k] - p_batt_left[k] * cfg.dt_hours / half_capacity
            next_soc_right = soc_right[k] - p_batt_right[k] * cfg.dt_hours / half_capacity
            g += [soc_left[k + 1] - next_soc_left, soc_right[k + 1] - next_soc_right]
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

            g += [
                p_fc_left[k] + p_batt_left[k] - load_left[k],
                p_fc_right[k] + p_batt_right[k] - load_right[k],
            ]
            lbg += [0.0, 0.0]
            ubg += [0.0, 0.0]

            delta_fc_left = p_fc_left[k] - (prev_fc_left if k == 0 else p_fc_left[k - 1])
            delta_fc_right = p_fc_right[k] - (prev_fc_right if k == 0 else p_fc_right[k - 1])
            g += [delta_fc_left, delta_fc_right]
            lbg += [-half_ramp, -half_ramp]
            ubg += [half_ramp, half_ramp]

            if cfg.use_raw_objective:
                batt_kw2 = p_batt_left[k] ** 2 + p_batt_right[k] ** 2
                ramp_kw2 = delta_fc_left ** 2 + delta_fc_right ** 2
                if cfg.raw_soc_squared:
                    soc_dev = (soc_left[k + 1] - cfg.soc_target) ** 2 + \
                              (soc_right[k + 1] - cfg.soc_target) ** 2
                else:
                    soc_dev = sqrt((soc_left[k + 1] - cfg.soc_target) ** 2 + 1e-6) + \
                              sqrt((soc_right[k + 1] - cfg.soc_target) ** 2 + 1e-6)
                J += cfg.q_soc * soc_dev / 2.0
                if cfg.raw_fc_energy_linear:
                    J += cfg.q_fc * (p_fc_left[k] + p_fc_right[k]) * cfg.dt_hours
                else:
                    fc_kw2 = p_fc_left[k] ** 2 + p_fc_right[k] ** 2
                    J += cfg.q_fc * fc_kw2 / 2.0
                if cfg.battery_throughput_penalty_enabled and cfg.battery_throughput_penalty_type == "absolute_power":
                    battery_use_norm = (
                        sqrt(p_batt_left[k] ** 2 + 1e-6) + sqrt(p_batt_right[k] ** 2 + 1e-6)
                    ) / max(cfg.battery_throughput_normalization_kw, 1e-6)
                    J += cfg.q_batt * battery_use_norm
                else:
                    J += cfg.q_batt * batt_kw2 / 2.0
                J += cfg.q_ramp * ramp_kw2 / 2.0
            elif cfg.use_dimensionless_objective:
                total_fc = p_fc_left[k] + p_fc_right[k]
                if cfg.use_h2_mass_cost:
                    ratio = fmin(fmax(total_fc / max(cfg.fuel_cell_max_kw, 1e-6), 0.0), 1.0)
                    h2_rate_gps = (cfg.fuel_cell_max_kw / 100.0) * (dp0_a1 * ratio + dp0_a2 * ratio**2)
                    h2_mass = h2_rate_gps * (cfg.dt_hours * 3600.0) / 1000.0
                    h2_ref = max(_h2_ref_kg_per_step(cfg), 1e-12)
                    fuel_cost = h2_mass / h2_ref if cfg.normalize_h2_cost else h2_mass
                    J += cfg.q_h2 * fuel_cost
                else:
                    fuel_cost = total_fc / max(cfg.fuel_cell_max_kw, 1e-6)
                    J += cfg.q_fc * fuel_cost
                fc_smooth_norm = (sqrt(delta_fc_left**2 + 1e-6) + sqrt(delta_fc_right**2 + 1e-6)) / max(
                    cfg.fuel_cell_ramp_kw, 1e-6
                )
                soc_dev_norm = (
                    sqrt((soc_left[k + 1] - cfg.soc_target) ** 2 + 1e-6)
                    + sqrt((soc_right[k + 1] - cfg.soc_target) ** 2 + 1e-6)
                ) / (2.0 * max(cfg.soc_band, 1e-6))
                if cfg.battery_throughput_penalty_enabled and cfg.battery_throughput_penalty_type == "absolute_power":
                    battery_use_norm = (
                        sqrt(p_batt_left[k] ** 2 + 1e-6) + sqrt(p_batt_right[k] ** 2 + 1e-6)
                    ) / max(cfg.battery_throughput_normalization_kw, 1e-6)
                else:
                    battery_use_norm = (
                        (p_batt_left[k] / max(half_discharge_max, 1e-6)) ** 2
                        + (p_batt_right[k] / max(half_discharge_max, 1e-6)) ** 2
                    ) / 2.0

                J += cfg.q_ramp * fc_smooth_norm**2
                J += cfg.q_soc * soc_dev_norm**2
                if cfg.enable_battery_use_in_mpc:
                    J += cfg.q_batt * battery_use_norm
            else:
                fuel_cost_norm = (p_fc_left[k] + p_fc_right[k]) / max(cfg.fuel_cell_max_kw, 1e-6)
                fc_smooth_norm = (sqrt(delta_fc_left**2 + 1e-6) + sqrt(delta_fc_right**2 + 1e-6)) / max(
                    cfg.fuel_cell_max_kw, 1e-6
                )
                soc_dev_norm = (
                    sqrt((soc_left[k + 1] - cfg.soc_target) ** 2 + 1e-6)
                    + sqrt((soc_right[k + 1] - cfg.soc_target) ** 2 + 1e-6)
                ) / (2.0 * max(cfg.soc_max - cfg.soc_min, 1e-6))
                battery_use_norm = (
                    (p_batt_left[k] / max(half_discharge_max, 1e-6)) ** 2
                    + (p_batt_right[k] / max(half_discharge_max, 1e-6)) ** 2
                ) / 2.0

                J += cfg.q1_fuel_cost * fuel_cost_norm
                J += cfg.q2_fc_smooth * fc_smooth_norm
                J += cfg.q3_soc_dev * soc_dev_norm
                if cfg.enable_battery_use_in_mpc:
                    J += cfg.q4_battery_use * battery_use_norm
        terminal_soc_left = soc_left[N]
        terminal_soc_right = soc_right[N]
        if cfg.use_raw_objective:
            if cfg.raw_soc_squared:
                terminal_soc_dev = (terminal_soc_left - cfg.soc_target) ** 2 + (
                    terminal_soc_right - cfg.soc_target
                ) ** 2
            else:
                terminal_soc_dev = sqrt((terminal_soc_left - cfg.soc_target) ** 2 + 1e-6) + sqrt(
                    (terminal_soc_right - cfg.soc_target) ** 2 + 1e-6
                )
            if cfg.enable_terminal_soc_soft_penalty:
                J += _terminal_soc_weight(cfg) * terminal_soc_dev / 2.0
        elif cfg.use_dimensionless_objective:
            terminal_soc_norm = (
                sqrt((terminal_soc_left - cfg.soc_target) ** 2 + 1e-6)
                + sqrt((terminal_soc_right - cfg.soc_target) ** 2 + 1e-6)
            ) / (2.0 * max(cfg.soc_band, 1e-6))
            if cfg.enable_terminal_soc_soft_penalty:
                J += _terminal_soc_weight(cfg) * terminal_soc_norm**2
        else:
            terminal_soc_norm = (
                sqrt((terminal_soc_left - cfg.soc_target) ** 2 + 1e-6)
                + sqrt((terminal_soc_right - cfg.soc_target) ** 2 + 1e-6)
            ) / (2.0 * max(cfg.soc_max - cfg.soc_min, 1e-6))
            J += cfg.q5_soc_terminal * terminal_soc_norm

        lbx = []
        ubx = []
        lbx.extend([cfg.soc_min] * (N + 1))
        ubx.extend([cfg.soc_max] * (N + 1))
        lbx.extend([cfg.soc_min] * (N + 1))
        ubx.extend([cfg.soc_max] * (N + 1))
        lbx.extend([-half_charge_max] * N)
        ubx.extend([half_discharge_max] * N)
        lbx.extend([-half_charge_max] * N)
        ubx.extend([half_discharge_max] * N)
        lbx.extend([cfg.fuel_cell_min_kw] * N)
        ubx.extend([half_fc_max] * N)
        lbx.extend([cfg.fuel_cell_min_kw] * N)
        ubx.extend([half_fc_max] * N)

        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": int(cfg.ipopt_max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "ipopt.acceptable_tol": 1e-4,
            "print_time": 0,
        }
        self.solver = nlpsol("ship_dual_side_mpc", "ipopt", {"f": J, "x": x, "g": vertcat(*g), "p": p}, opts)
        self.lbx = np.asarray(lbx, dtype=float)
        self.ubx = np.asarray(ubx, dtype=float)
        self.lbg = np.asarray(lbg, dtype=float)
        self.ubg = np.asarray(ubg, dtype=float)

    def solve(
        self,
        soc_left_0: float,
        soc_right_0: float,
        prev_fc_left_kw: float,
        prev_fc_right_kw: float,
        load_left_forecast_kw: np.ndarray,
        load_right_forecast_kw: np.ndarray,
        terminal_load_left_kw: float | None = None,
        terminal_load_right_kw: float | None = None,
    ) -> DualSideCasadiMPCResult:
        cfg = self.config
        load_left_forecast_kw = np.asarray(load_left_forecast_kw, dtype=float).reshape(-1)
        load_right_forecast_kw = np.asarray(load_right_forecast_kw, dtype=float).reshape(-1)
        if load_left_forecast_kw.shape[0] != cfg.prediction_horizon or load_right_forecast_kw.shape[0] != cfg.prediction_horizon:
            raise ValueError("Dual-side load forecasts must match prediction horizon.")

        half_fc_max = cfg.fuel_cell_max_kw / 2.0
        soc_left_0 = _clip(soc_left_0, cfg.soc_min, cfg.soc_max)
        soc_right_0 = _clip(soc_right_0, cfg.soc_min, cfg.soc_max)
        prev_fc_left_kw = _clip(prev_fc_left_kw, cfg.fuel_cell_min_kw, half_fc_max)
        prev_fc_right_kw = _clip(prev_fc_right_kw, cfg.fuel_cell_min_kw, half_fc_max)
        if self._last_solution is None:
            x0 = np.zeros_like(self.lbx)
            offset = 0
            x0[offset : offset + cfg.prediction_horizon + 1] = soc_left_0
            offset += cfg.prediction_horizon + 1
            x0[offset : offset + cfg.prediction_horizon + 1] = soc_right_0
            offset += cfg.prediction_horizon + 1
            x0[offset : offset + cfg.prediction_horizon] = np.clip(
                load_left_forecast_kw * 0.3, -cfg.battery_charge_max_kw / 2.0, cfg.battery_discharge_max_kw / 2.0
            )
            offset += cfg.prediction_horizon
            x0[offset : offset + cfg.prediction_horizon] = np.clip(
                load_right_forecast_kw * 0.3, -cfg.battery_charge_max_kw / 2.0, cfg.battery_discharge_max_kw / 2.0
            )
            offset += cfg.prediction_horizon
            x0[offset : offset + cfg.prediction_horizon] = np.clip(
                load_left_forecast_kw * 0.7, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw / 2.0
            )
            offset += cfg.prediction_horizon
            x0[offset : offset + cfg.prediction_horizon] = np.clip(
                load_right_forecast_kw * 0.7, cfg.fuel_cell_min_kw, cfg.fuel_cell_max_kw / 2.0
            )
        else:
            x0 = self._last_solution.copy()
            offset = 0
            x0[offset : offset + cfg.prediction_horizon + 1] = np.clip(soc_left_0, cfg.soc_min, cfg.soc_max)
            offset += cfg.prediction_horizon + 1
            x0[offset : offset + cfg.prediction_horizon + 1] = np.clip(soc_right_0, cfg.soc_min, cfg.soc_max)

        terminal_load_left_kw = float(
            load_left_forecast_kw[-1] if terminal_load_left_kw is None else terminal_load_left_kw
        )
        terminal_load_right_kw = float(
            load_right_forecast_kw[-1] if terminal_load_right_kw is None else terminal_load_right_kw
        )
        params = np.concatenate(
            [
                load_left_forecast_kw,
                load_right_forecast_kw,
                [terminal_load_left_kw, terminal_load_right_kw],
                [prev_fc_left_kw, prev_fc_right_kw, soc_left_0, soc_right_0],
            ]
        )
        solution = self.solver(x0=x0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=params)
        stats = self.solver.stats()
        values = np.asarray(solution["x"]).reshape(-1)
        self._last_solution = values.copy()

        offset = 0
        soc_left = values[offset : offset + cfg.prediction_horizon + 1]
        offset += cfg.prediction_horizon + 1
        soc_right = values[offset : offset + cfg.prediction_horizon + 1]
        offset += cfg.prediction_horizon + 1
        p_batt_left = values[offset : offset + cfg.prediction_horizon]
        offset += cfg.prediction_horizon
        p_batt_right = values[offset : offset + cfg.prediction_horizon]
        offset += cfg.prediction_horizon
        p_fc_left = values[offset : offset + cfg.prediction_horizon]
        offset += cfg.prediction_horizon
        p_fc_right = values[offset : offset + cfg.prediction_horizon]

        objective_info = _build_dual_objective_info(
            p_fc_left_traj=p_fc_left,
            p_fc_right_traj=p_fc_right,
            p_batt_left_traj=p_batt_left,
            p_batt_right_traj=p_batt_right,
            soc_left_traj=soc_left,
            soc_right_traj=soc_right,
            prev_fc_left_kw=prev_fc_left_kw,
            prev_fc_right_kw=prev_fc_right_kw,
            config=cfg,
            terminal_load_left_kw=terminal_load_left_kw,
            terminal_load_right_kw=terminal_load_right_kw,
        )
        objective_info = {**objective_info, "solver_message": str(stats.get("return_status", ""))}

        return DualSideCasadiMPCResult(
            fuel_cell_ref_left_kw=float(p_fc_left[0]),
            fuel_cell_ref_right_kw=float(p_fc_right[0]),
            battery_ref_left_kw=float(p_batt_left[0]),
            battery_ref_right_kw=float(p_batt_right[0]),
            predicted_soc_left=float(soc_left[1]),
            predicted_soc_right=float(soc_right[1]),
            soc_ref=float(cfg.soc_target),
            fuel_cell_ref_left_traj_kw=[float(v) for v in p_fc_left],
            fuel_cell_ref_right_traj_kw=[float(v) for v in p_fc_right],
            battery_ref_left_traj_kw=[float(v) for v in p_batt_left],
            battery_ref_right_traj_kw=[float(v) for v in p_batt_right],
            soc_pred_left_traj=[float(v) for v in soc_left],
            soc_pred_right_traj=[float(v) for v in soc_right],
            objective_value=float(solution["f"]),
            objective_info=objective_info,
            success=bool(stats.get("success", True)),
        )


def _build_single_objective_info(
    p_fc_traj: np.ndarray,
    p_bat_traj: np.ndarray,
    soc_traj: np.ndarray,
    prev_fc_kw: float,
    config: CasadiMPCConfig,
    terminal_load_kw: float | None = None,
    soc_reference_value: float | None = None,
) -> dict:
    fc_scale = max(config.fuel_cell_max_kw, 1e-6)
    bat_scale = max(config.battery_discharge_max_kw, 1e-6)
    soc_span = max(config.soc_max - config.soc_min, 1e-6)
    dt_seconds = float(config.dt_hours) * 3600.0
    delta_fc = np.diff(np.concatenate([[prev_fc_kw], p_fc_traj]))
    fuel_cost_norm = p_fc_traj / fc_scale
    fc_smooth_norm = np.abs(delta_fc) / (
        max(config.fuel_cell_ramp_kw, 1e-6) if config.use_dimensionless_objective else fc_scale
    )
    eta_fc = fc_efficiency_from_ratio(
        p_fc_traj,
        p_fc_rated_kw=float(config.fuel_cell_max_kw),
    )
    h2_kg = h2_kg_step_dp0_quadratic(
        p_fc_traj,
        dt_seconds=dt_seconds,
        p_rated_total_kw=float(config.fuel_cell_max_kw),
    )
    m_h2_ref = _h2_ref_kg_per_step(config)
    terminal_soc = float(soc_traj[-1])
    soc_ref = resolve_soc_reference(config, float(soc_traj[0]), soc_reference_value)
    soc_stage_cost_raw = np.asarray(
        compute_soc_stage_cost(config, soc_traj[1:], soc_ref, normalize=False),
        dtype=float,
    )
    soc_stage_cost_norm = np.asarray(
        compute_soc_stage_cost(config, soc_traj[1:], soc_ref, normalize=config.use_dimensionless_objective),
        dtype=float,
    )
    terminal_soc_cost_raw = compute_soc_terminal_cost(config, terminal_soc, soc_ref, normalize=False)
    terminal_soc_cost_norm_value = compute_soc_terminal_cost(
        config,
        terminal_soc,
        soc_ref,
        normalize=config.use_dimensionless_objective,
    )
    soc_dev_norm = np.sqrt(np.maximum(soc_stage_cost_norm, 0.0))
    terminal_soc_norm = float(np.sqrt(max(terminal_soc_cost_norm_value, 0.0)))
    if config.battery_throughput_penalty_enabled and config.battery_throughput_penalty_type == "absolute_power":
        battery_use_norm = np.abs(p_bat_traj) / max(config.battery_throughput_normalization_kw, 1e-6)
        batt_cost_raw = float(np.sum(np.abs(p_bat_traj)))
    else:
        battery_use_norm = (p_bat_traj / bat_scale) ** 2
        batt_cost_raw = float(np.sum(p_bat_traj**2))
    batt_cost_norm = float(np.sum(battery_use_norm)) if config.enable_battery_use_in_mpc else 0.0
    batt_throughput_kwh = float(np.sum(np.abs(p_bat_traj)) * config.dt_hours)
    raw_physical_objective = str(config.objective_mode).strip().lower() == "raw_physical"
    if raw_physical_objective:
        h2_cost_raw = float(np.sum(h2_kg))
        h2_cost_norm = float("nan")
        raw_h2_cost = h2_cost_raw
        fuel_cost_term = float(config.q_h2 * h2_cost_raw)
        raw_fc_cost_mode = "h2_mass_kg"
        objective_scale_mode = "raw_physical"

        soc_stage_physical = (np.asarray(soc_traj[1:], dtype=float) - float(soc_ref)) ** 2
        soc_cost_raw = float(np.sum(soc_stage_physical))
        raw_soc_cost = soc_cost_raw
        soc_cost_norm = float("nan")
        soc_dev_norm = np.sqrt(np.maximum(soc_stage_physical, 0.0))
        soc_dev_term = float(config.q_soc * raw_soc_cost)

        batt_cost_raw = batt_throughput_kwh
        batt_cost_norm = float("nan")
        battery_use_norm = np.full_like(p_bat_traj, np.nan, dtype=float)
        battery_use_term = float(config.q_batt * batt_throughput_kwh if config.enable_battery_use_in_mpc else 0.0)

        ramp_cost_raw = float(np.sum(delta_fc**2))
        raw_ramp_cost = ramp_cost_raw
        ramp_cost_norm = float("nan")
        fc_smooth_term = float(config.q_ramp * raw_ramp_cost)
        terminal_soc_cost_raw = 0.0
        terminal_soc_cost_norm = float("nan")
        terminal_soc_norm = float("nan")
        terminal_soc_term = 0.0
    elif config.use_raw_objective:
        if config.raw_fc_energy_linear:
            raw_h2_cost = float(np.sum(p_fc_traj * config.dt_hours))
            fuel_cost_term = float(config.q_fc * raw_h2_cost)
            raw_fc_cost_mode = "linear_energy"
        else:
            raw_h2_cost = float(np.sum(p_fc_traj**2))
            fuel_cost_term = float(config.q_fc * raw_h2_cost)
            raw_fc_cost_mode = "power_squared"
        ramp_cost_raw = float(np.sum(delta_fc**2))
        raw_ramp_cost = ramp_cost_raw
        fc_smooth_term = float(config.q_ramp * raw_ramp_cost)
        terminal_soc_raw = float(terminal_soc_cost_raw)
        soc_cost_raw = float(np.sum(soc_stage_cost_raw))
        raw_soc_cost = soc_cost_raw
        soc_dev_term = float(config.q_soc * raw_soc_cost)
        battery_use_term = float(config.q_batt * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_term = float(
            _terminal_soc_weight(config) * terminal_soc_raw if config.enable_terminal_soc_soft_penalty else 0.0
        )
        h2_cost_raw = raw_h2_cost
        h2_cost_norm = raw_h2_cost
        soc_cost_norm = raw_soc_cost
        ramp_cost_norm = raw_ramp_cost
        terminal_soc_cost_norm = float(terminal_soc_raw)
        objective_scale_mode = "raw"
    elif config.use_dimensionless_objective:
        if config.use_h2_mass_cost:
            raw_fc_cost_mode = "h2_mass_normalized" if config.normalize_h2_cost else "h2_mass_kg"
            h2_term = h2_kg / max(float(m_h2_ref), 1e-12) if config.normalize_h2_cost else h2_kg
            h2_cost_raw = float(np.sum(h2_kg))
            h2_cost_norm = float(np.sum(h2_term))
            raw_h2_cost = h2_cost_norm
            fuel_cost_term = float(config.q_h2 * raw_h2_cost)
            fuel_cost_norm = h2_term
            objective_scale_mode = "h2_dimensionless" if config.normalize_h2_cost else "h2_mass"
        else:
            raw_fc_cost_mode = "dimensionless_power"
            h2_cost_raw = float(np.sum(p_fc_traj))
            h2_cost_norm = float(np.sum(fuel_cost_norm))
            raw_h2_cost = h2_cost_norm
            fuel_cost_term = float(config.q_fc * raw_h2_cost)
            objective_scale_mode = "dimensionless"
        ramp_cost_raw = float(np.sum(delta_fc**2))
        raw_ramp_cost = float(np.sum(fc_smooth_norm**2))
        soc_cost_raw = float(np.sum(soc_stage_cost_raw))
        raw_soc_cost = float(np.sum(soc_stage_cost_norm))
        ramp_cost_norm = raw_ramp_cost
        soc_cost_norm = raw_soc_cost
        fc_smooth_term = float(config.q_ramp * raw_ramp_cost)
        soc_dev_term = float(config.q_soc * raw_soc_cost)
        battery_use_term = float(config.q_batt * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_cost_norm = float(terminal_soc_cost_norm_value)
        terminal_soc_term = float(
            _terminal_soc_weight(config) * terminal_soc_cost_norm if config.enable_terminal_soc_soft_penalty else 0.0
        )
    else:
        raw_fc_cost_mode = "normalized_power"
        h2_cost_raw = float(np.sum(p_fc_traj))
        h2_cost_norm = float(np.sum(fuel_cost_norm))
        raw_h2_cost = h2_cost_norm
        ramp_cost_raw = float(np.sum(np.abs(delta_fc)))
        raw_ramp_cost = float(np.sum(fc_smooth_norm))
        soc_cost_raw = float(np.sum(soc_stage_cost_raw))
        raw_soc_cost = float(np.sum(soc_dev_norm))
        ramp_cost_norm = raw_ramp_cost
        soc_cost_norm = raw_soc_cost
        fuel_cost_term = float(config.q1_fuel_cost * raw_h2_cost)
        fc_smooth_term = float(config.q2_fc_smooth * raw_ramp_cost)
        soc_dev_term = float(config.q3_soc_dev * raw_soc_cost)
        battery_use_term = float(config.q4_battery_use * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_cost_norm = float(terminal_soc_norm)
        terminal_soc_term = float(config.q5_soc_terminal * terminal_soc_cost_norm)
        objective_scale_mode = "legacy_normalized"
    total_mpc_cost = fuel_cost_term + fc_smooth_term + soc_dev_term + battery_use_term + terminal_soc_term
    finite_eta = eta_fc[np.isfinite(eta_fc)]
    return {
        **_effective_config_info(config),
        "fuel_cost_norm_mean": float(np.mean(fuel_cost_norm)),
        "fc_smooth_norm_mean": float(np.mean(fc_smooth_norm)),
        "soc_dev_norm_mean": float(np.mean(soc_dev_norm)),
        "terminal_soc_norm": float(terminal_soc_norm),
        "battery_use_norm_mean": float(np.mean(battery_use_norm)),
        "fuel_cost_term": fuel_cost_term,
        "fc_smooth_term": fc_smooth_term,
        "soc_dev_term": soc_dev_term,
        "h2_cost_raw": float(h2_cost_raw),
        "h2_cost_raw_kg": float(h2_cost_raw if config.use_h2_mass_cost else np.sum(h2_kg)),
        "h2_mass_kg": float(np.sum(h2_kg)),
        "h2_cost_norm": float(h2_cost_norm),
        "weighted_h2_cost": fuel_cost_term,
        "soc_cost_raw": float(soc_cost_raw),
        "soc_cost_norm": float(soc_cost_norm),
        "weighted_soc_cost": soc_dev_term,
        "ramp_cost_raw": float(ramp_cost_raw),
        "ramp_cost_norm": float(ramp_cost_norm),
        "weighted_ramp_cost": fc_smooth_term,
        "batt_cost_raw": float(batt_cost_raw if config.enable_battery_use_in_mpc else 0.0),
        "batt_cost_norm": float(batt_cost_norm),
        "batt_throughput_kwh": float(batt_throughput_kwh),
        "weighted_batt_cost": battery_use_term,
        "terminal_soc_cost_raw": float(terminal_soc_cost_raw if config.enable_terminal_soc_soft_penalty else 0.0),
        "terminal_soc_cost_norm": float(terminal_soc_cost_norm if config.enable_terminal_soc_soft_penalty else 0.0),
        "weighted_terminal_soc_cost": terminal_soc_term,
        "raw_h2_cost": raw_h2_cost,
        "raw_soc_cost": raw_soc_cost,
        "raw_ramp_cost": raw_ramp_cost,
        "battery_use_term": battery_use_term,
        "terminal_soc_term": terminal_soc_term,
        "q_terminal_soc_effective": _terminal_soc_weight(config),
        "total_mpc_cost": total_mpc_cost,
        "soc_reference_mode": str(config.soc_reference_mode),
        "soc_penalty_type": str(config.soc_penalty_type),
        "soc_ref_value": float(soc_ref),
        "soc_reserve": float(config.soc_reserve),
        "terminal_soc_band": float(config.terminal_soc_band),
        "soc_terminal_error": float(abs(terminal_soc - soc_ref)),
        "enable_battery_use_in_mpc": bool(config.enable_battery_use_in_mpc),
        "enable_terminal_soc_soft_penalty": bool(config.enable_terminal_soc_soft_penalty),
        "raw_fc_cost_mode": raw_fc_cost_mode,
        "objective_scale_mode": objective_scale_mode,
        "objective_mode": str(config.objective_mode),
        "use_h2_mass_cost": bool(config.use_h2_mass_cost),
        "normalize_h2_cost": bool(config.normalize_h2_cost),
        "fuel_cell_ramp_constraint_enabled": bool(config.fuel_cell_ramp_constraint_enabled),
        "fuel_cell_ramp_kw": float(config.fuel_cell_ramp_kw),
        "m_H2_ref_kg_per_step": float(m_h2_ref),
        "P_fc_rated_kw": float(config.fuel_cell_max_kw),
        "eta_rated": float(_eta_poly_at_ratio(config, 1.0)),
        "dt_seconds": float(dt_seconds),
        "H2_step_kg_sum": float(np.sum(h2_kg)),
        "average_eta_fc": float(np.mean(finite_eta)) if finite_eta.size else float("nan"),
        "min_eta_fc": float(np.min(finite_eta)) if finite_eta.size else float("nan"),
        "max_eta_fc": float(np.max(finite_eta)) if finite_eta.size else float("nan"),
        "fc_efficiency_curve_source": CURVE_SOURCE_LABEL,
        "fc_efficiency_curve_note": (
            "fresh D_p=0 efficiency curve exported from MATLAB CSV; optimizer uses a forced-origin quadratic fit"
        ),
        "battery_throughput_penalty_enabled": bool(config.battery_throughput_penalty_enabled),
        "battery_throughput_penalty_type": str(config.battery_throughput_penalty_type),
        "battery_throughput_normalization_kw": float(config.battery_throughput_normalization_kw),
        "total_objective": total_mpc_cost,
        "objective_value": total_mpc_cost,
    }


def _build_dual_objective_info(
    p_fc_left_traj: np.ndarray,
    p_fc_right_traj: np.ndarray,
    p_batt_left_traj: np.ndarray,
    p_batt_right_traj: np.ndarray,
    soc_left_traj: np.ndarray,
    soc_right_traj: np.ndarray,
    prev_fc_left_kw: float,
    prev_fc_right_kw: float,
    config: CasadiMPCConfig,
    terminal_load_left_kw: float | None = None,
    terminal_load_right_kw: float | None = None,
) -> dict:
    fc_scale = max(config.fuel_cell_max_kw, 1e-6)
    bat_scale = max(config.battery_discharge_max_kw / 2.0, 1e-6)
    soc_span = max(config.soc_max - config.soc_min, 1e-6)
    dt_seconds = float(config.dt_hours) * 3600.0
    total_fc_traj = p_fc_left_traj + p_fc_right_traj
    delta_fc_left = np.diff(np.concatenate([[prev_fc_left_kw], p_fc_left_traj]))
    delta_fc_right = np.diff(np.concatenate([[prev_fc_right_kw], p_fc_right_traj]))
    fuel_cost_norm = (p_fc_left_traj + p_fc_right_traj) / fc_scale
    fc_smooth_norm = (np.abs(delta_fc_left) + np.abs(delta_fc_right)) / fc_scale
    soc_dev_norm = (np.abs(soc_left_traj[1:] - config.soc_target) + np.abs(soc_right_traj[1:] - config.soc_target)) / (
        2.0 * soc_span
    )
    terminal_soc_left = float(soc_left_traj[-1])
    terminal_soc_right = float(soc_right_traj[-1])
    terminal_soc_norm = (
        abs(terminal_soc_left - config.soc_target) + abs(terminal_soc_right - config.soc_target)
    ) / (2.0 * soc_span)
    if config.battery_throughput_penalty_enabled and config.battery_throughput_penalty_type == "absolute_power":
        battery_use_norm = (np.abs(p_batt_left_traj) + np.abs(p_batt_right_traj)) / max(
            config.battery_throughput_normalization_kw, 1e-6
        )
        batt_cost_raw = float(np.sum(np.abs(p_batt_left_traj) + np.abs(p_batt_right_traj)))
    else:
        battery_use_norm = (
            (p_batt_left_traj / bat_scale) ** 2 + (p_batt_right_traj / bat_scale) ** 2
        ) / 2.0
        batt_cost_raw = float(np.sum((p_batt_left_traj**2 + p_batt_right_traj**2) / 2.0))
    batt_cost_norm = float(np.sum(battery_use_norm)) if config.enable_battery_use_in_mpc else 0.0
    terminal_soc_cost_raw = float(
        (abs(terminal_soc_left - config.soc_target) + abs(terminal_soc_right - config.soc_target)) / 2.0
    )
    eta_fc = fc_efficiency_from_ratio(
        total_fc_traj,
        p_fc_rated_kw=float(config.fuel_cell_max_kw),
    )
    h2_kg = h2_kg_step_dp0_quadratic(
        total_fc_traj,
        dt_seconds=dt_seconds,
        p_rated_total_kw=float(config.fuel_cell_max_kw),
    )
    m_h2_ref = _h2_ref_kg_per_step(config)
    batt_throughput_kwh = float(np.sum(np.abs(p_batt_left_traj) + np.abs(p_batt_right_traj)) * config.dt_hours)
    raw_physical_objective = str(config.objective_mode).strip().lower() == "raw_physical"
    if raw_physical_objective:
        raw_fc_cost_mode = "h2_mass_kg"
        objective_scale_mode = "raw_physical"
        h2_cost_raw = float(np.sum(h2_kg))
        h2_cost_norm = float("nan")
        raw_h2_cost = h2_cost_raw
        fuel_cost_term = float(config.q_h2 * raw_h2_cost)
        soc_left_ref = float(soc_left_traj[0])
        soc_right_ref = float(soc_right_traj[0])
        soc_dev_raw = ((soc_left_traj[1:] - soc_left_ref) ** 2 + (soc_right_traj[1:] - soc_right_ref) ** 2) / 2.0
        soc_cost_raw = float(np.sum(soc_dev_raw))
        raw_soc_cost = soc_cost_raw
        soc_cost_norm = float("nan")
        soc_dev_norm = np.sqrt(np.maximum(soc_dev_raw, 0.0))
        soc_dev_term = float(config.q_soc * raw_soc_cost)
        batt_cost_raw = batt_throughput_kwh
        batt_cost_norm = float("nan")
        battery_use_norm = np.full_like(total_fc_traj, np.nan, dtype=float)
        battery_use_term = float(config.q_batt * batt_throughput_kwh if config.enable_battery_use_in_mpc else 0.0)
        ramp_cost_raw = 0.0
        raw_ramp_cost = 0.0
        ramp_cost_norm = float("nan")
        fc_smooth_term = 0.0
        terminal_soc_cost_raw = 0.0
        terminal_soc_cost_norm = float("nan")
        terminal_soc_norm = float("nan")
        terminal_soc_term = 0.0
    elif config.use_raw_objective:
        batt_kw2 = p_batt_left_traj**2 + p_batt_right_traj**2
        ramp_kw2 = delta_fc_left**2 + delta_fc_right**2
        if config.raw_soc_squared:
            soc_dev_raw = (soc_left_traj[1:] - config.soc_target) ** 2 + (
                soc_right_traj[1:] - config.soc_target
            ) ** 2
            terminal_soc_raw = (terminal_soc_left - config.soc_target) ** 2 + (
                terminal_soc_right - config.soc_target
            ) ** 2
        else:
            soc_dev_raw = np.abs(soc_left_traj[1:] - config.soc_target) + np.abs(
                soc_right_traj[1:] - config.soc_target
            )
            terminal_soc_raw = abs(terminal_soc_left - config.soc_target) + abs(
                terminal_soc_right - config.soc_target
            )
        if config.raw_fc_energy_linear:
            fuel_cost_term = float(np.sum(config.q_fc * (p_fc_left_traj + p_fc_right_traj) * config.dt_hours))
            raw_h2_cost = float(np.sum((p_fc_left_traj + p_fc_right_traj) * config.dt_hours))
            raw_fc_cost_mode = "linear_energy"
        else:
            fc_kw2 = p_fc_left_traj**2 + p_fc_right_traj**2
            fuel_cost_term = float(np.sum(config.q_fc * fc_kw2 / 2.0))
            raw_h2_cost = float(np.sum(fc_kw2 / 2.0))
            raw_fc_cost_mode = "power_squared"
        ramp_cost_raw = float(np.sum(ramp_kw2 / 2.0))
        fc_smooth_term = float(np.sum(config.q_ramp * ramp_kw2 / 2.0))
        soc_dev_term = float(np.sum(config.q_soc * soc_dev_raw / 2.0))
        raw_ramp_cost = ramp_cost_raw
        raw_soc_cost = float(np.sum(soc_dev_raw / 2.0))
        soc_cost_raw = float(
            np.sum(
                (
                    np.abs(soc_left_traj[1:] - config.soc_target)
                    + np.abs(soc_right_traj[1:] - config.soc_target)
                )
                / 2.0
            )
        )
        battery_use_term = float(config.q_batt * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_term = float(
            _terminal_soc_weight(config) * terminal_soc_raw / 2.0
            if config.enable_terminal_soc_soft_penalty
            else 0.0
        )
        h2_cost_raw = raw_h2_cost
        h2_cost_norm = raw_h2_cost
        soc_cost_norm = raw_soc_cost
        ramp_cost_norm = raw_ramp_cost
        terminal_soc_cost_norm = float(terminal_soc_raw / 2.0)
        objective_scale_mode = "raw"
    elif config.use_dimensionless_objective:
        if config.use_h2_mass_cost:
            raw_fc_cost_mode = "h2_mass_normalized" if config.normalize_h2_cost else "h2_mass_kg"
            h2_term = h2_kg / max(float(m_h2_ref), 1e-12) if config.normalize_h2_cost else h2_kg
            h2_cost_raw = float(np.sum(h2_kg))
            h2_cost_norm = float(np.sum(h2_term))
            raw_h2_cost = h2_cost_norm
            fuel_cost_term = float(config.q_h2 * raw_h2_cost)
            fuel_cost_norm = h2_term
            objective_scale_mode = "h2_dimensionless" if config.normalize_h2_cost else "h2_mass"
        else:
            raw_fc_cost_mode = "dimensionless_power"
            h2_cost_raw = float(np.sum(total_fc_traj))
            h2_cost_norm = float(np.sum(fuel_cost_norm))
            raw_h2_cost = h2_cost_norm
            fuel_cost_term = float(config.q_fc * raw_h2_cost)
            objective_scale_mode = "dimensionless"
        fc_smooth_norm = (np.abs(delta_fc_left) + np.abs(delta_fc_right)) / max(config.fuel_cell_ramp_kw, 1e-6)
        soc_dev_norm = (
            np.abs(soc_left_traj[1:] - config.soc_target) + np.abs(soc_right_traj[1:] - config.soc_target)
        ) / (2.0 * max(config.soc_band, 1e-6))
        ramp_cost_raw = float(np.sum((delta_fc_left**2 + delta_fc_right**2) / 2.0))
        raw_ramp_cost = float(np.sum(fc_smooth_norm**2))
        soc_cost_raw = float(
            np.sum(
                (
                    np.abs(soc_left_traj[1:] - config.soc_target)
                    + np.abs(soc_right_traj[1:] - config.soc_target)
                )
                / 2.0
            )
        )
        raw_soc_cost = float(np.sum(soc_dev_norm**2))
        ramp_cost_norm = raw_ramp_cost
        soc_cost_norm = raw_soc_cost
        fc_smooth_term = float(config.q_ramp * raw_ramp_cost)
        soc_dev_term = float(config.q_soc * raw_soc_cost)
        battery_use_term = float(config.q_batt * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_norm = (
            abs(terminal_soc_left - config.soc_target) + abs(terminal_soc_right - config.soc_target)
        ) / (2.0 * max(config.soc_band, 1e-6))
        terminal_soc_cost_norm = float(terminal_soc_norm**2)
        terminal_soc_term = float(
            _terminal_soc_weight(config) * terminal_soc_cost_norm if config.enable_terminal_soc_soft_penalty else 0.0
        )
    else:
        raw_fc_cost_mode = "normalized_power"
        h2_cost_raw = float(np.sum(total_fc_traj))
        h2_cost_norm = float(np.sum(fuel_cost_norm))
        raw_h2_cost = h2_cost_norm
        ramp_cost_raw = float(np.sum((np.abs(delta_fc_left) + np.abs(delta_fc_right)) / 2.0))
        raw_ramp_cost = float(np.sum(fc_smooth_norm))
        soc_cost_raw = float(
            np.sum(
                (
                    np.abs(soc_left_traj[1:] - config.soc_target)
                    + np.abs(soc_right_traj[1:] - config.soc_target)
                )
                / 2.0
            )
        )
        raw_soc_cost = float(np.sum(soc_dev_norm))
        ramp_cost_norm = raw_ramp_cost
        soc_cost_norm = raw_soc_cost
        fuel_cost_term = float(np.sum(config.q1_fuel_cost * fuel_cost_norm))
        fc_smooth_term = float(np.sum(config.q2_fc_smooth * fc_smooth_norm))
        soc_dev_term = float(np.sum(config.q3_soc_dev * soc_dev_norm))
        battery_use_term = float(config.q4_battery_use * batt_cost_norm if config.enable_battery_use_in_mpc else 0.0)
        terminal_soc_cost_norm = float(terminal_soc_norm)
        terminal_soc_term = float(config.q5_soc_terminal * terminal_soc_cost_norm)
        objective_scale_mode = "legacy_normalized"
    total_mpc_cost = fuel_cost_term + fc_smooth_term + soc_dev_term + battery_use_term + terminal_soc_term
    finite_eta = eta_fc[np.isfinite(eta_fc)]
    return {
        **_effective_config_info(config),
        "fuel_cost_norm_mean": float(np.mean(fuel_cost_norm)),
        "fc_smooth_norm_mean": float(np.mean(fc_smooth_norm)),
        "soc_dev_norm_mean": float(np.mean(soc_dev_norm)),
        "terminal_soc_norm": float(terminal_soc_norm),
        "battery_use_norm_mean": float(np.mean(battery_use_norm)),
        "fuel_cost_term": fuel_cost_term,
        "fc_smooth_term": fc_smooth_term,
        "soc_dev_term": soc_dev_term,
        "h2_cost_raw": float(h2_cost_raw),
        "h2_mass_kg": float(np.sum(h2_kg)),
        "h2_cost_norm": float(h2_cost_norm),
        "weighted_h2_cost": fuel_cost_term,
        "soc_cost_raw": float(soc_cost_raw),
        "soc_cost_norm": float(soc_cost_norm),
        "weighted_soc_cost": soc_dev_term,
        "ramp_cost_raw": float(ramp_cost_raw),
        "ramp_cost_norm": float(ramp_cost_norm),
        "weighted_ramp_cost": fc_smooth_term,
        "batt_cost_raw": float(batt_cost_raw if config.enable_battery_use_in_mpc else 0.0),
        "batt_cost_norm": float(batt_cost_norm),
        "batt_throughput_kwh": float(batt_throughput_kwh),
        "weighted_batt_cost": battery_use_term,
        "terminal_soc_cost_raw": float(terminal_soc_cost_raw if config.enable_terminal_soc_soft_penalty else 0.0),
        "terminal_soc_cost_norm": float(terminal_soc_cost_norm if config.enable_terminal_soc_soft_penalty else 0.0),
        "weighted_terminal_soc_cost": terminal_soc_term,
        "raw_h2_cost": raw_h2_cost,
        "raw_soc_cost": raw_soc_cost,
        "raw_ramp_cost": raw_ramp_cost,
        "battery_use_term": battery_use_term,
        "terminal_soc_term": terminal_soc_term,
        "q_terminal_soc_effective": _terminal_soc_weight(config),
        "total_mpc_cost": total_mpc_cost,
        "enable_battery_use_in_mpc": bool(config.enable_battery_use_in_mpc),
        "enable_terminal_soc_soft_penalty": bool(config.enable_terminal_soc_soft_penalty),
        "raw_fc_cost_mode": raw_fc_cost_mode,
        "objective_scale_mode": objective_scale_mode,
        "objective_mode": str(config.objective_mode),
        "use_h2_mass_cost": bool(config.use_h2_mass_cost),
        "normalize_h2_cost": bool(config.normalize_h2_cost),
        "m_H2_ref_kg_per_step": float(m_h2_ref),
        "P_fc_rated_kw": float(config.fuel_cell_max_kw),
        "eta_rated": float(_eta_poly_at_ratio(config, 1.0)),
        "dt_seconds": float(dt_seconds),
        "H2_step_kg_sum": float(np.sum(h2_kg)),
        "average_eta_fc": float(np.mean(finite_eta)) if finite_eta.size else float("nan"),
        "min_eta_fc": float(np.min(finite_eta)) if finite_eta.size else float("nan"),
        "max_eta_fc": float(np.max(finite_eta)) if finite_eta.size else float("nan"),
        "fc_efficiency_curve_source": CURVE_SOURCE_LABEL,
        "fc_efficiency_curve_note": (
            "fresh D_p=0 efficiency curve exported from MATLAB CSV; optimizer uses a forced-origin quadratic fit"
        ),
        "battery_throughput_penalty_enabled": bool(config.battery_throughput_penalty_enabled),
        "battery_throughput_penalty_type": str(config.battery_throughput_penalty_type),
        "battery_throughput_normalization_kw": float(config.battery_throughput_normalization_kw),
        "total_objective": total_mpc_cost,
        "objective_value": total_mpc_cost,
    }


def solve_upper_mpc(current_state: dict, load_prediction: list[float] | np.ndarray, refs: dict | None, params: CasadiMPCConfig | None):
    config = params or CasadiMPCConfig()
    if refs and "soc_ref" in refs:
        config = CasadiMPCConfig(**{**config.__dict__, "soc_target": float(refs["soc_ref"])})
    solver = ShipCasadiMPC(config)
    return solver.solve(
        current_soc=float(current_state["soc"]),
        prev_fc_kw=float(current_state.get("previous_fuel_cell_kw", 0.0)),
        load_forecast_kw=np.asarray(load_prediction, dtype=float),
    )
