
"""
Ship EMS configuration.

Fuel cell:
    6 stacks, 100 kW gross rated power per stack.

Battery:
    4 equivalent clusters, 624 kWh total capacity.

No shore power.
No supercapacitor.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShipConfig:

    # =====================================================
    # 1. Fuel cell system
    # =====================================================

    fc_stack_count: int = 6

    # Per-stack gross rated power.
    fc_stack_rated_gross_kw: float = 100.0

    # Fuel cell replacement economics.
    fc_replacement_cost_rmb_per_kw: float = 3500.0

    fc_replacement_fraction: float = 0.50

    # =====================================================
    # 2. Lithium battery system
    # =====================================================

    battery_cluster_count: int = 4

    battery_total_capacity_kwh: float = 624.0

    battery_rated_voltage_v: float = 432.0

    # 1C charge/discharge power limits.
    battery_max_charge_kw: float = 624.0

    battery_max_discharge_kw: float = 624.0

    eta_charge: float = 0.95

    eta_discharge: float = 0.95

    # SOC constraints.
    soc_initial: float = 0.65

    soc_min: float = 0.20

    soc_max: float = 0.80

    soc_target: float = 0.50

    # =====================================================
    # 3. Hydrogen
    # =====================================================

    hydrogen_lhv_mj_per_kg: float = 120.0

    hydrogen_price_rmb_per_kg: float = 21.9

    # =====================================================
    # 4. Shore power
    # =====================================================

    shore_power_enabled: bool = False

    # =====================================================
    # 5. Battery replacement economics
    # =====================================================

    battery_replacement_cost_rmb_per_kwh: float = 2000.0

    # =====================================================
    # 6. Derived parameters
    # =====================================================

    @property
    def fc_total_rated_gross_kw(self) -> float:

        return (
            self.fc_stack_count
            * self.fc_stack_rated_gross_kw
        )

    @property
    def battery_cluster_capacity_kwh(self) -> float:

        return (
            self.battery_total_capacity_kwh
            / self.battery_cluster_count
        )

    @property
    def battery_cluster_max_charge_kw(self) -> float:

        return (
            self.battery_max_charge_kw
            / self.battery_cluster_count
        )

    @property
    def battery_cluster_max_discharge_kw(self) -> float:

        return (
            self.battery_max_discharge_kw
            / self.battery_cluster_count
        )

    @property
    def battery_nominal_capacity_ah(self) -> float:

        return (
            self.battery_total_capacity_kwh
            * 1000.0
            / self.battery_rated_voltage_v
        )

    @property
    def fc_replacement_cost_rmb(self) -> float:

        return (
            self.fc_total_rated_gross_kw
            * self.fc_replacement_cost_rmb_per_kw
            * self.fc_replacement_fraction
        )

    @property
    def battery_replacement_cost_rmb(self) -> float:

        return (
            self.battery_total_capacity_kwh
            * self.battery_replacement_cost_rmb_per_kwh
        )

    # =====================================================
    # 7. Validation
    # =====================================================

    def validate(self):

        if self.fc_stack_count <= 0:
            raise ValueError("Invalid FC stack count.")

        if self.fc_stack_rated_gross_kw <= 0:
            raise ValueError("Invalid FC rated power.")

        if self.battery_cluster_count <= 0:
            raise ValueError("Invalid battery cluster count.")

        if self.battery_total_capacity_kwh <= 0:
            raise ValueError("Invalid battery capacity.")

        if self.battery_rated_voltage_v <= 0:
            raise ValueError("Invalid battery voltage.")

        if not (
            0 < self.eta_charge <= 1
            and 0 < self.eta_discharge <= 1
        ):
            raise ValueError("Invalid battery efficiency.")

        if not (
            0 < self.soc_min
            < self.soc_initial
            < self.soc_max
            < 1
        ):
            raise ValueError("Invalid SOC configuration.")

        if not (
            self.soc_min
            <= self.soc_target
            <= self.soc_max
        ):
            raise ValueError("Invalid SOC target.")

        if self.hydrogen_lhv_mj_per_kg <= 0:
            raise ValueError("Invalid hydrogen LHV.")

        if self.hydrogen_price_rmb_per_kg <= 0:
            raise ValueError("Invalid hydrogen price.")

        if self.battery_max_charge_kw <= 0:
            raise ValueError("Invalid battery charge power.")

        if self.battery_max_discharge_kw <= 0:
            raise ValueError("Invalid battery discharge power.")

        if self.fc_replacement_cost_rmb_per_kw <= 0:
            raise ValueError("Invalid FC replacement cost.")

        if self.battery_replacement_cost_rmb_per_kwh <= 0:
            raise ValueError("Invalid battery replacement cost.")


CONFIG = ShipConfig()

CONFIG.validate()
