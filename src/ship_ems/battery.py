
"""
Ship Energy Management - Lithium Battery Model.

Four equivalent clusters; 624 kWh aggregate capacity.
SOC limits: 0.20--0.80.

Degradation:
  Zhou et al. (2025), Eqs. (12)--(14):
      dQ_weighted = F(SOC) * G(I) * |I| * dt
  Project lifetime normalization:
      Q_life = 15000 * Q_nominal

The 15000 factor is the previously frozen project baseline, calibrated from
secondary literature. It is not a fixed parameter stated by Zhou or Yang.
Yang et al. (2026) instead use a DOD-dependent ACC(D) cycle-life model.

Power convention: positive = discharge; negative = charge.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


# =========================================================
# 0. Project lifetime calibration
# =========================================================

BATTERY_LIFETIME_THROUGHPUT_FACTOR = 15_000.0

# Sensitivity analysis parameters.
BATTERY_LIFETIME_SENSITIVITY_FACTORS = (
    10_000.0,
    15_000.0,
    20_000.0,
)

BATTERY_LIFETIME_CALIBRATION_STATUS = (
    "FROZEN_PROJECT_BASELINE_SECONDARY_LITERATURE_CALIBRATION"
)


# =========================================================
# 1. Battery parameters
# =========================================================

@dataclass(frozen=True)
class BatteryConfig:
    """624 kWh aggregate lithium battery represented by four clusters."""

    cluster_count: int = 4

    capacity_kwh: float = 624.0

    nominal_voltage_v: float = 432.0

    max_charge_power_kw: float = 624.0

    max_discharge_power_kw: float = 624.0

    charge_efficiency: float = 0.95

    discharge_efficiency: float = 0.95

    initial_soc: float = 0.65

    soc_min: float = 0.20

    soc_max: float = 0.80

    def __post_init__(self):

        if (
            isinstance(self.cluster_count, bool)
            or not isinstance(self.cluster_count, int)
            or self.cluster_count <= 0
        ):
            raise ValueError(
                "cluster_count must be a positive integer."
            )

        for name in (
            "capacity_kwh",
            "nominal_voltage_v",
            "max_charge_power_kw",
            "max_discharge_power_kw",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"{name} must be finite and positive."
                )

        for name in (
            "charge_efficiency",
            "discharge_efficiency",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0 < float(value) <= 1
            ):
                raise ValueError(
                    f"{name} must lie in (0, 1]."
                )

        for name in (
            "initial_soc",
            "soc_min",
            "soc_max",
        ):
            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"{name} must be finite."
                )

        if not (
            0 <= self.soc_min
            < self.soc_max
            <= 1
        ):
            raise ValueError(
                "Invalid SOC boundaries."
            )

        if not (
            self.soc_min
            <= self.initial_soc
            <= self.soc_max
        ):
            raise ValueError(
                "Initial SOC outside limits."
            )

    @property
    def cluster_capacity_kwh(self) -> float:

        return (
            self.capacity_kwh
            / self.cluster_count
        )

    @property
    def cluster_max_charge_kw(self) -> float:

        return (
            self.max_charge_power_kw
            / self.cluster_count
        )

    @property
    def cluster_max_discharge_kw(self) -> float:

        return (
            self.max_discharge_power_kw
            / self.cluster_count
        )

    @property
    def nominal_capacity_ah(self) -> float:
        """Aggregate nominal capacity in Ah."""

        return (
            self.capacity_kwh
            * 1000.0
            / self.nominal_voltage_v
        )

    @property
    def current_reference_a(self) -> float:
        """Aggregate 1C reference current in A."""

        return self.nominal_capacity_ah

    @property
    def lifetime_capacity_ah(self) -> float:
        """Q_life = 15000 * Q_nominal, in weighted-throughput Ah."""

        return (
            BATTERY_LIFETIME_THROUGHPUT_FACTOR
            * self.nominal_capacity_ah
        )


# =========================================================
# 2. Battery step result
# =========================================================

@dataclass(frozen=True)
class BatteryStepResult:

    soc_before: float

    soc_after: float

    power_kw: float

    cluster_power_kw: float

    current_a: float

    cluster_current_a: float

    f_soc: float

    g_current: float

    weighted_aging_increment_ah: float

    degradation_before: float

    degradation_after: float

    degradation_increment: float


# =========================================================
# 3. Lithium battery
# =========================================================

class LithiumBattery:
    """
    Aggregate lithium battery model.

    The four clusters share power equally and have synchronized SOC.

    Zhou's weighted Ah throughput is converted to an economic life
    fraction using the project's frozen 15000 throughput factor.
    """

    def __init__(
        self,
        config: BatteryConfig | None = None
    ):

        if (
            config is not None
            and type(config) is not BatteryConfig
        ):
            raise TypeError(
                "config must be a BatteryConfig instance."
            )

        self.config = config or BatteryConfig()

        self.reset()

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    def reset(
        self,
        initial_soc: float | None = None
    ):
        """Reset SOC and voyage-level throughput accounts."""

        soc = (
            self.config.initial_soc
            if initial_soc is None
            else float(initial_soc)
        )

        if not math.isfinite(soc):
            raise ValueError(
                "Initial SOC must be finite."
            )

        if not (
            self.config.soc_min
            <= soc
            <= self.config.soc_max
        ):
            raise ValueError(
                "Initial SOC outside battery limits."
            )

        self.soc = soc

        self.initial_soc = soc

        # Cumulative Zhou stress-weighted throughput.
        # Unit: Ah.
        self.weighted_aging_throughput_ah = 0.0

        # Unweighted throughput for diagnostics.
        self.raw_aging_throughput_ah = 0.0

        self.elapsed_time_s = 0.0

    # -----------------------------------------------------
    # Degradation indicators
    # -----------------------------------------------------

    @property
    def raw_degradation_fraction(self) -> float:
        """
        Unclipped cumulative life consumption.

        raw = weighted Ah / (15000 * nominal Ah).
        """

        return (
            self.weighted_aging_throughput_ah
            / self.config.lifetime_capacity_ah
        )

    def degradation_fraction(self) -> float:
        """Economic life fraction clipped to [0, 1]."""

        return min(
            max(self.raw_degradation_fraction, 0.0),
            1.0,
        )

    # -----------------------------------------------------
    # Feasible battery power interval
    # -----------------------------------------------------

    def get_power_bounds(
        self,
        dt_s: float = 1.0
    ) -> tuple[float, float]:
        """
        Return:

            (minimum charging power, maximum discharging power)

        Negative power means charging.
        Positive power means discharging.
        """

        dt_s = float(dt_s)

        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError(
                "dt_s must be finite and positive."
            )

        cfg = self.config

        dt_h = dt_s / 3600.0

        # Discharge SOC update:
        # SOC_next = SOC - P_dis * dt / (E * eta_dis)

        discharge_soc_limit = (
            (self.soc - cfg.soc_min)
            * cfg.capacity_kwh
            * cfg.discharge_efficiency
            / dt_h
        )

        # Charge SOC update:
        # SOC_next = SOC + |P_chg| * dt * eta_chg / E

        charge_soc_limit = (
            (cfg.soc_max - self.soc)
            * cfg.capacity_kwh
            / (
                dt_h
                * cfg.charge_efficiency
            )
        )

        max_discharge = min(
            cfg.max_discharge_power_kw,
            max(0.0, discharge_soc_limit),
        )

        max_charge = min(
            cfg.max_charge_power_kw,
            max(0.0, charge_soc_limit),
        )

        return -max_charge, max_discharge

    # -----------------------------------------------------
    # Current calculation
    # -----------------------------------------------------

    def calculate_current(
        self,
        soc_before: float,
        soc_after: float,
        dt_s: float,
    ) -> float:
        """
        Recover signed cell-side current from SOC change.

        I = (SOC_before - SOC_after)
            * Q_nominal * 3600 / dt_s

        Positive current: discharge.
        Negative current: charge.

        The SOC transition already includes charge/discharge
        efficiency, so efficiency is not applied a second time.
        """

        soc_before = float(soc_before)

        soc_after = float(soc_after)

        dt_s = float(dt_s)

        if not all(
            map(
                math.isfinite,
                (soc_before, soc_after, dt_s)
            )
        ):
            raise ValueError(
                "SOC values and dt_s must be finite."
            )

        if dt_s <= 0:
            raise ValueError(
                "dt_s must be positive."
            )

        current_a = (
            (soc_before - soc_after)
            * self.config.nominal_capacity_ah
            * 3600.0
            / dt_s
        )

        if not math.isfinite(current_a):
            raise ValueError(
                "Calculated battery current is non-finite."
            )

        return current_a

    # -----------------------------------------------------
    # Zhou aging weighting factors
    # -----------------------------------------------------

    def calculate_aging_factors(
        self,
        soc: float,
        current_a: float,
    ) -> tuple[float, float]:
        """
        Zhou Eqs. (13)-(14).

        F(SOC) = 1 + 3.25 * (1 - SOC)^2

        G(I) = 1 + 0.45 * I / I_nom, I >= 0

        G(I) = 1 + 0.55 * |I| / I_nom, I < 0
        """

        soc = float(soc)

        current_a = float(current_a)

        if (
            not math.isfinite(soc)
            or not 0 <= soc <= 1
        ):
            raise ValueError(
                "soc must be finite and lie in [0, 1]."
            )

        if not math.isfinite(current_a):
            raise ValueError(
                "current_a must be finite."
            )

        f_soc = (
            1.0
            + 3.25 * (1.0 - soc) ** 2
        )

        i_ref = self.config.current_reference_a

        if current_a >= 0:

            g_current = (
                1.0
                + 0.45 * current_a / i_ref
            )

        else:

            g_current = (
                1.0
                + 0.55 * abs(current_a) / i_ref
            )

        return f_soc, g_current

    # -----------------------------------------------------
    # One-step battery update
    # -----------------------------------------------------

    def step(
        self,
        power_kw: float,
        dt_s: float = 1.0,
    ) -> BatteryStepResult:
        """
        Execute one battery step without implicit power clipping.

        P > 0: discharge.
        P < 0: charge.

        Current for the aging calculation is recovered from the
        actual SOC transition, including charge/discharge efficiency.
        """

        power_kw = float(power_kw)

        dt_s = float(dt_s)

        if not math.isfinite(power_kw):
            raise ValueError(
                "Battery power must be finite."
            )

        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError(
                "dt_s must be finite and positive."
            )

        cfg = self.config

        p_min, p_max = self.get_power_bounds(dt_s)

        tolerance = 1e-8

        if (
            power_kw < p_min - tolerance
            or power_kw > p_max + tolerance
        ):
            raise ValueError(
                f"Battery power infeasible: "
                f"{power_kw:.6f} kW, "
                f"allowed [{p_min:.6f}, {p_max:.6f}] kW."
            )

        soc_before = self.soc

        dt_h = dt_s / 3600.0

        # =================================================
        # 1. SOC update
        # =================================================

        if power_kw >= 0:

            # Discharging.
            # Cell-side energy = bus-side energy / eta_dis.

            soc_after = (
                soc_before
                - power_kw * dt_h
                / (
                    cfg.capacity_kwh
                    * cfg.discharge_efficiency
                )
            )

        else:

            # Charging.
            # Stored cell-side energy = eta_chg * bus energy.

            soc_after = (
                soc_before
                - power_kw * dt_h
                * cfg.charge_efficiency
                / cfg.capacity_kwh
            )

        if (
            soc_after < cfg.soc_min - tolerance
            or soc_after > cfg.soc_max + tolerance
        ):
            raise RuntimeError(
                "Battery SOC update violated physical limits."
            )

        # Remove only floating-point excursions within tolerance.
        soc_after = min(
            max(soc_after, cfg.soc_min),
            cfg.soc_max,
        )

        # =================================================
        # 2. Recover cell current from SOC change
        # =================================================

        current_a = self.calculate_current(
            soc_before=soc_before,
            soc_after=soc_after,
            dt_s=dt_s,
        )

        # =================================================
        # 3. Zhou SOC/current weighting factors
        # =================================================

        f_soc, g_current = (
            self.calculate_aging_factors(
                soc=soc_before,
                current_a=current_a,
            )
        )

        # =================================================
        # 4. Weighted aging throughput
        # =================================================

        aging_increment_ah = (
            f_soc
            * g_current
            * abs(current_a)
            * dt_h
        )

        raw_aging_increment_ah = (
            abs(current_a) * dt_h
        )

        if (
            not math.isfinite(aging_increment_ah)
            or aging_increment_ah < 0
        ):
            raise ValueError(
                "Invalid weighted aging increment."
            )

        aging_before_ah = (
            self.weighted_aging_throughput_ah
        )

        # =================================================
        # 5. Convert weighted throughput to life consumption
        # =================================================

        q_lifetime_ah = cfg.lifetime_capacity_ah

        degradation_before = min(
            max(
                aging_before_ah / q_lifetime_ah,
                0.0,
            ),
            1.0,
        )

        self.weighted_aging_throughput_ah += (
            aging_increment_ah
        )

        self.raw_aging_throughput_ah += (
            raw_aging_increment_ah
        )

        degradation_after = min(
            max(
                self.weighted_aging_throughput_ah
                / q_lifetime_ah,
                0.0,
            ),
            1.0,
        )

        # Only the newly consumed life fraction is charged.
        degradation_increment = max(
            degradation_after - degradation_before,
            0.0,
        )

        # =================================================
        # 6. Update physical state
        # =================================================

        self.soc = soc_after

        self.elapsed_time_s += dt_s

        return BatteryStepResult(

            soc_before=soc_before,

            soc_after=soc_after,

            power_kw=power_kw,

            cluster_power_kw=(
                power_kw / cfg.cluster_count
            ),

            current_a=current_a,

            cluster_current_a=(
                current_a / cfg.cluster_count
            ),

            f_soc=f_soc,

            g_current=g_current,

            weighted_aging_increment_ah=(
                aging_increment_ah
            ),

            degradation_before=(
                degradation_before
            ),

            degradation_after=(
                degradation_after
            ),

            degradation_increment=(
                degradation_increment
            ),
        )
