
"""
Ship EMS Cost Calculator.

Step reward:
    Hydrogen consumption
    + FC operating degradation
    + Battery degradation
    + Per-step SOC squared penalty

No additional terminal SOC penalty.

Voyage economic cost:
    Actual hydrogen
    + FC operating degradation
    + FC start-stop degradation
    + Battery degradation
    + Battery equivalent hydrogen

Battery equivalent hydrogen and FC start-stop
degradation are excluded from the step reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .battery import (
    BatteryStepResult,
    LithiumBattery,
)

from .config import (
    CONFIG,
    ShipConfig,
)

from .fuel_cell import (
    D_START_STOP_UV_PER_CYCLE,
    FuelCellSystem,
)


# =========================================================
# 1. Default FC average efficiency
# =========================================================

# Frozen project parameter. This is the previously agreed value and is
# intentionally not recalculated from the later quadrature implementation.
DEFAULT_ETA_FC_AVG = 0.5755641791


# =========================================================
# 2. Step cost
# =========================================================

@dataclass(frozen=True)
class StepCost:

    hydrogen_kg: float

    hydrogen_cost_rmb: float

    fc_degradation_increment: float

    fc_degradation_cost_rmb: float

    battery_degradation_increment: float

    battery_degradation_cost_rmb: float

    @property
    def reward_cost_rmb(self):

        return (
            self.hydrogen_cost_rmb
            + self.fc_degradation_cost_rmb
            + self.battery_degradation_cost_rmb
        )


# =========================================================
# 3. Voyage cost
# =========================================================

@dataclass(frozen=True)
class VoyageCost:

    hydrogen_kg: float

    hydrogen_cost_rmb: float

    fc_operating_degradation_cost_rmb: float

    fc_start_stop_degradation_cost_rmb: float

    fc_total_degradation_cost_rmb: float

    fc_degradation_indicator: float

    battery_degradation_increment: float

    battery_degradation_indicator: float

    battery_degradation_cost_rmb: float

    battery_equivalent_hydrogen_kg: float

    battery_equivalent_hydrogen_cost_rmb: float

    total_equivalent_hydrogen_kg: float

    total_equivalent_hydrogen_cost_rmb: float

    total_economic_cost_rmb: float


# =========================================================
# 4. Cost calculator
# =========================================================

class CostCalculator:

    def __init__(
        self,
        ship_config: ShipConfig | None = None,
        eta_fc_avg: float | None = None,
    ):

        self.config = ship_config or CONFIG

        self.eta_fc_avg = (
            DEFAULT_ETA_FC_AVG
            if eta_fc_avg is None
            else float(eta_fc_avg)
        )

        if not math.isfinite(self.eta_fc_avg):
            raise ValueError(
                "Invalid average FC efficiency."
            )

        if not 0 < self.eta_fc_avg <= 1:
            raise ValueError(
                "eta_fc_avg must be in (0, 1]."
            )

        if self.config.hydrogen_lhv_mj_per_kg <= 0:
            raise ValueError("Invalid hydrogen LHV.")

        if self.config.hydrogen_price_rmb_per_kg <= 0:
            raise ValueError("Invalid hydrogen price.")

        if self.config.battery_replacement_cost_rmb_per_kwh <= 0:
            raise ValueError("Invalid battery replacement cost.")

    # =====================================================
    # 5. Single-step operating costs
    # =====================================================

    def calculate_step_costs(
        self,
        fc_result: dict,
        battery_result: BatteryStepResult,
        fuel_cell: FuelCellSystem,
        battery: LithiumBattery,
    ) -> StepCost:
        """
        Single-step costs.

        FC start-stop degradation must be zero here.
        Battery equivalent hydrogen is excluded.
        """

        if int(fc_result.get("start_stop_cycles", 0)) != 0:

            raise ValueError(
                "FC start-stop degradation must not "
                "enter the step reward."
            )

        # -------------------------------------------------
        # 1. Hydrogen consumption
        # -------------------------------------------------

        hydrogen_kg = float(
            fc_result["hydrogen_kg"]
        )

        hydrogen_cost = (
            hydrogen_kg
            * self.config.hydrogen_price_rmb_per_kg
        )

        # -------------------------------------------------
        # 2. FC operating degradation
        # -------------------------------------------------

        fc_degradation_increment = float(
            fc_result["delta_degradation_ratio"]
        )

        fc_degradation_cost = (
            fc_degradation_increment
            * fuel_cell.params.replacement_cost_rmb
        )

        # -------------------------------------------------
        # 3. Battery degradation
        # -------------------------------------------------

        # Use the bounded indicator increment from battery.py.
        battery_degradation_increment = float(
            battery_result.degradation_increment
        )

        battery_degradation_cost = (
            battery_degradation_increment
            * self.config.battery_replacement_cost_rmb_per_kwh
            * battery.config.capacity_kwh
        )

        values = (
            hydrogen_kg,
            hydrogen_cost,
            fc_degradation_increment,
            fc_degradation_cost,
            battery_degradation_increment,
            battery_degradation_cost,
        )

        if not all(
            math.isfinite(float(v))
            for v in values
        ):
            raise ValueError(
                "Non-finite operating cost."
            )

        if any(float(v) < 0 for v in values):
            raise ValueError(
                "Negative operating cost detected."
            )

        return StepCost(

            hydrogen_kg=hydrogen_kg,

            hydrogen_cost_rmb=hydrogen_cost,

            fc_degradation_increment=(
                fc_degradation_increment
            ),

            fc_degradation_cost_rmb=(
                fc_degradation_cost
            ),

            battery_degradation_increment=(
                battery_degradation_increment
            ),

            battery_degradation_cost_rmb=(
                battery_degradation_cost
            )
        )

    # =====================================================
    # 6. TD3 reward
    # =====================================================

    def calculate_reward(
            self,
            step_costs: StepCost,
            next_soc: float,
            alpha: float = 0.001,
            beta_step: float = 0.5,
    ) -> float:
        """
        Calculate TD3 reward at each executed environment step.

        Reward:
            r_t = -alpha * (
                C_H2,t
                + C_FC_deg,t
                + C_LB_deg,t
                + beta_step * (SOC_next - SOC_target)**2
            )

        SOC penalty:
            Applied at every normal environment step.

        Terminal SOC penalty:
            Disabled. No additional terminal penalty.

        Excluded from reward:
            Battery equivalent hydrogen consumption.
            Fuel-cell start-stop degradation.

        These excluded costs remain in the independent
        voyage economic cost ledger.
        """

        alpha = float(alpha)
        beta_step = float(beta_step)
        next_soc = float(next_soc)

        # ------------------------------------------------
        # 1. Validate reward parameters
        # ------------------------------------------------

        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError(
                "Invalid alpha."
            )

        if not math.isfinite(beta_step) or beta_step < 0:
            raise ValueError(
                "Invalid beta_step."
            )

        if not math.isfinite(next_soc):
            raise ValueError(
                "Invalid next_soc."
            )

        # ------------------------------------------------
        # 2. Economic operating costs
        # ------------------------------------------------

        economic_cost = float(
            step_costs.reward_cost_rmb
        )

        # ------------------------------------------------
        # 3. Per-step SOC penalty
        # ------------------------------------------------

        soc_error = (
                next_soc
                - self.config.soc_target
        )

        soc_penalty = (
                beta_step
                * soc_error ** 2
        )

        # ------------------------------------------------
        # 4. Final step reward
        # ------------------------------------------------

        reward = (
                -alpha
                * (economic_cost + soc_penalty)
        )

        if not math.isfinite(reward):
            raise ValueError(
                "Non-finite reward."
            )

        return float(reward)

    # =====================================================
    # 7. FC start-stop degradation cost
    # =====================================================

    @staticmethod
    def calculate_start_stop_cost(
        fuel_cell: FuelCellSystem,
        cycles: int = 1,
    ) -> float:
        """
        Calculate start-stop degradation separately.

        This method does not modify the FC state.
        It prevents double counting in the step reward.
        """

        if (
            int(cycles) != cycles
            or cycles < 0
        ):
            raise ValueError(
                "Invalid start-stop cycle count."
            )

        cycles = int(cycles)

        if cycles == 0:
            return 0.0

        eol_loss = (
            fuel_cell.params.eol_voltage_loss_uv
        )

        voltage_before = (
            fuel_cell.voltage_loss_uv
        )

        voltage_after = (
            voltage_before
            + D_START_STOP_UV_PER_CYCLE * cycles
        )

        degradation_before = min(
            voltage_before / eol_loss,
            1.0
        )

        degradation_after = min(
            voltage_after / eol_loss,
            1.0
        )

        degradation_increment = max(
            degradation_after - degradation_before,
            0.0
        )

        return (
            degradation_increment
            * fuel_cell.params.replacement_cost_rmb
        )

    # =====================================================
    # 8. Complete voyage economic accounting
    # =====================================================

    def calculate_voyage_costs(
        self,
        step_costs: Sequence[StepCost],
        fuel_cell: FuelCellSystem,
        battery: LithiumBattery,
        initial_soc: float,
        final_soc: float,
        start_stop_cycles: int = 1,
    ) -> VoyageCost:
        """
        Complete voyage economic cost.

        Battery equivalent hydrogen:

        m_H2_eq =
            (SOC_initial - SOC_final) * E_battery
            / (eta_fc_avg * LHV_H2)

        SOC decrease:
            Positive equivalent hydrogen.

        SOC increase:
            Negative equivalent hydrogen.

        One FC start-stop cycle is accounted for here,
        not in the step reward.
        """

        if not step_costs:
            raise ValueError(
                "Empty voyage cost records."
            )

        initial_soc = float(initial_soc)

        final_soc = float(final_soc)

        cfg = battery.config

        if not (
            cfg.soc_min
            <= initial_soc
            <= cfg.soc_max
        ):
            raise ValueError("Invalid initial SOC.")

        if not (
            cfg.soc_min
            <= final_soc
            <= cfg.soc_max
        ):
            raise ValueError("Invalid final SOC.")

        if fuel_cell.total_start_stop_cycles != 0:

            raise ValueError(
                "Start-stop degradation has already "
                "been counted in the FC model."
            )

        # The episode step records must reconcile with the two physical
        # degradation accumulators before voyage-level costs are reported.
        recorded_hydrogen_kg = sum(x.hydrogen_kg for x in step_costs)
        if not math.isclose(
            recorded_hydrogen_kg,
            fuel_cell.total_hydrogen_kg,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Voyage hydrogen ledger does not match FC accumulator."
            )

        recorded_fc_operating_fraction = sum(
            x.fc_degradation_increment for x in step_costs
        )
        model_fc_operating_fraction = min(
            fuel_cell.voltage_loss_uv
            / fuel_cell.params.eol_voltage_loss_uv,
            1.0,
        )
        if not math.isclose(
            recorded_fc_operating_fraction,
            model_fc_operating_fraction,
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "FC operating-degradation ledger does not reconcile."
            )

        recorded_battery_fraction = sum(
            x.battery_degradation_increment for x in step_costs
        )
        model_battery_fraction = battery.degradation_fraction()
        if not math.isclose(
            recorded_battery_fraction,
            model_battery_fraction,
            rel_tol=1e-8,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "Battery degradation ledger does not reconcile."
            )

        # -------------------------------------------------
        # 1. Actual hydrogen
        # -------------------------------------------------

        hydrogen_kg = sum(
            x.hydrogen_kg
            for x in step_costs
        )

        hydrogen_cost = (
            hydrogen_kg
            * self.config.hydrogen_price_rmb_per_kg
        )

        # -------------------------------------------------
        # 2. FC operating degradation
        # -------------------------------------------------

        fc_operating_cost = sum(
            x.fc_degradation_cost_rmb
            for x in step_costs
        )

        # -------------------------------------------------
        # 3. FC start-stop degradation
        # -------------------------------------------------

        fc_start_stop_cost = (
            self.calculate_start_stop_cost(
                fuel_cell,
                cycles=start_stop_cycles,
            )
        )

        fc_total_degradation_cost = (
            fc_operating_cost
            + fc_start_stop_cost
        )

        fc_eol_loss = (
            fuel_cell.params.eol_voltage_loss_uv
        )

        fc_final_voltage_loss = (
            fuel_cell.voltage_loss_uv
            + D_START_STOP_UV_PER_CYCLE
            * int(start_stop_cycles)
        )

        fc_degradation_indicator = min(
            fc_final_voltage_loss / fc_eol_loss,
            1.0
        )

        # -------------------------------------------------
        # 4. Battery degradation
        # -------------------------------------------------

        battery_degradation_indicator = (
            battery.degradation_fraction()
        )

        battery_degradation_increment = sum(
            x.battery_degradation_increment
            for x in step_costs
        )

        battery_degradation_cost = (
            battery_degradation_indicator
            * self.config.battery_replacement_cost_rmb_per_kwh
            * battery.config.capacity_kwh
        )

        # -------------------------------------------------
        # 5. Battery equivalent hydrogen
        # -------------------------------------------------

        hydrogen_lhv_kwh_per_kg = (
            self.config.hydrogen_lhv_mj_per_kg
            / 3.6
        )

        battery_equivalent_hydrogen_kg = (
            (initial_soc - final_soc)
            * battery.config.capacity_kwh
            / (
                self.eta_fc_avg
                * hydrogen_lhv_kwh_per_kg
            )
        )

        battery_equivalent_hydrogen_cost = (
            battery_equivalent_hydrogen_kg
            * self.config.hydrogen_price_rmb_per_kg
        )

        # -------------------------------------------------
        # 6. Total equivalent hydrogen
        # -------------------------------------------------

        total_equivalent_hydrogen_kg = (
            hydrogen_kg
            + battery_equivalent_hydrogen_kg
        )

        total_equivalent_hydrogen_cost = (
            total_equivalent_hydrogen_kg
            * self.config.hydrogen_price_rmb_per_kg
        )

        # -------------------------------------------------
        # 7. Total voyage economic cost
        # -------------------------------------------------

        total_economic_cost = (
            total_equivalent_hydrogen_cost
            + fc_total_degradation_cost
            + battery_degradation_cost
        )

        return VoyageCost(

            hydrogen_kg=hydrogen_kg,

            hydrogen_cost_rmb=hydrogen_cost,

            fc_operating_degradation_cost_rmb=(
                fc_operating_cost
            ),

            fc_start_stop_degradation_cost_rmb=(
                fc_start_stop_cost
            ),

            fc_total_degradation_cost_rmb=(
                fc_total_degradation_cost
            ),

            fc_degradation_indicator=(
                fc_degradation_indicator
            ),

            battery_degradation_increment=(
                battery_degradation_increment
            ),

            battery_degradation_indicator=(
                battery_degradation_indicator
            ),

            battery_degradation_cost_rmb=(
                battery_degradation_cost
            ),

            battery_equivalent_hydrogen_kg=(
                battery_equivalent_hydrogen_kg
            ),

            battery_equivalent_hydrogen_cost_rmb=(
                battery_equivalent_hydrogen_cost
            ),

            total_equivalent_hydrogen_kg=(
                total_equivalent_hydrogen_kg
            ),

            total_equivalent_hydrogen_cost_rmb=(
                total_equivalent_hydrogen_cost
            ),

            total_economic_cost_rmb=(
                total_economic_cost
            )
        )
