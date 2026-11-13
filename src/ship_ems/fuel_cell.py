
"""
Ship Energy Management - Fuel Cell Model

Configuration:
    6 x 100 kW gross-rated fuel cell stacks.

Power:
    Gross power = Net power + Auxiliary power.

EMS:
    All FC power commands use total net output power.

Degradation:
    Zhou four-condition voltage degradation model.

ECSA:
    Not used.
"""

from dataclasses import dataclass

import numpy as np


# =========================================================
# 1. Characteristic curves
# =========================================================

# Per-stack net power (kW)

FC_NET_POWER_KW = np.array([
    0.0000, 11.3938, 22.3676, 35.9133,
    49.1160, 62.1460, 74.8327, 87.1766,
    98.3215, 109.1230, 116.6680
], dtype=float)


# Net efficiency (%)

FC_EFFICIENCY_PERCENT = np.array([
    0.0000, 63.8554, 62.4096, 60.4819,
    58.7952, 58.0723, 56.8675, 56.1446,
    54.9398, 53.7349, 52.0482
], dtype=float)


# Cell voltage (V)

FC_CELL_VOLTAGE = np.array([
    1.010650, 0.880189, 0.849129, 0.827103,
    0.811097, 0.799108, 0.788121, 0.777134,
    0.765139, 0.754147, 0.744157
], dtype=float)


# Auxiliary power (kW)

FC_AUX_POWER_KW = np.array([
    0.000000, 0.722892, 2.168670, 3.373490,
    4.819280, 7.469880, 10.602400, 14.457800,
    16.626500, 19.036100, 20.963900
], dtype=float)


# Hydrogen flow (g/min per stack)

FC_H2_FLOW_G_MIN = np.array([
    0.00000, 9.15663, 18.55420, 30.84340,
    43.37350, 55.66270, 68.19280, 80.72290,
    93.25300, 105.78300, 116.86700
], dtype=float)


# Gross stack power (kW)

FC_GROSS_POWER_KW = np.array([
    0.000000, 12.116692, 24.536270,
    39.286790, 53.935280, 69.615880,
    85.435100, 101.634400, 114.948000,
    128.159100, 137.631900
], dtype=float)


# =========================================================
# 2. Zhou four-condition degradation parameters
# =========================================================

D_LOW_UV_PER_H = 10.17

D_HIGH_UV_PER_H = 11.74

D_TRANSIENT_UV_PER_KW = 0.0441

D_START_STOP_UV_PER_CYCLE = 23.91


# =========================================================
# 3. Fuel cell parameters
# =========================================================

@dataclass(frozen=True)
class FuelCellParameters:

    n_stacks: int = 6

    rated_stack_gross_power_kw: float = 100.0

    hydrogen_price_rmb_per_kg: float = 21.9

    # Low-load threshold per stack.
    low_power_threshold_kw: float = 30.0

    # Mylonopoulos et al. (2024), section 4.3, sets the EOL criterion
    # used with the Zhou degradation rates above. DOI: 10.1016/j.ijhydene.2024.10.235.
    initial_cell_voltage_v: float = 0.95

    eol_voltage_drop_ratio: float = 0.15

    # Replacement economics.
    fc_cost_rmb_per_kw: float = 3500.0

    replacement_fraction: float = 0.50

    def __post_init__(self):

        if (
            isinstance(self.n_stacks, bool)
            or not isinstance(self.n_stacks, int)
            or self.n_stacks <= 0
        ):
            raise ValueError("n_stacks must be a positive integer.")

        positive_parameters = (
            self.rated_stack_gross_power_kw,
            self.hydrogen_price_rmb_per_kg,
            self.initial_cell_voltage_v,
            self.fc_cost_rmb_per_kw,
        )
        if not all(
            np.isfinite(float(value)) and float(value) > 0
            for value in positive_parameters
        ):
            raise ValueError(
                "FC rated power, hydrogen price, voltage, and cost "
                "must be finite and positive."
            )

        if (
            self.rated_stack_gross_power_kw
            > FC_GROSS_POWER_KW[-1]
        ):
            raise ValueError(
                "Rated gross power exceeds measured curve."
            )

        if (
            not np.isfinite(self.low_power_threshold_kw)
            or self.low_power_threshold_kw < 0
            or self.low_power_threshold_kw
            >= self.max_stack_net_power_kw
        ):
            raise ValueError("Invalid FC low-load threshold.")

        if not (
            np.isfinite(self.eol_voltage_drop_ratio)
            and 0 < self.eol_voltage_drop_ratio < 1
        ):
            raise ValueError("Invalid FC EOL voltage ratio.")

        if (
            not np.isfinite(self.replacement_fraction)
            or not 0 < self.replacement_fraction <= 1
        ):
            raise ValueError("Invalid FC replacement fraction.")

    @property
    def max_stack_net_power_kw(self):

        return float(
            np.interp(
                self.rated_stack_gross_power_kw,
                FC_GROSS_POWER_KW,
                FC_NET_POWER_KW
            )
        )

    @property
    def rated_stack_net_power_kw(self):
        """Backward-compatible alias."""

        return self.max_stack_net_power_kw

    @property
    def max_total_net_power_kw(self):

        return (
            self.n_stacks
            * self.max_stack_net_power_kw
        )

    @property
    def rated_total_gross_power_kw(self):

        return (
            self.n_stacks
            * self.rated_stack_gross_power_kw
        )

    @property
    def eol_voltage_loss_uv(self):

        return (
            self.initial_cell_voltage_v
            * self.eol_voltage_drop_ratio
            * 1e6
        )

    @property
    def replacement_cost_rmb(self):

        return (
            self.rated_total_gross_power_kw
            * self.fc_cost_rmb_per_kw
            * self.replacement_fraction
        )


# =========================================================
# 4. Power result
# =========================================================

@dataclass(frozen=True)
class FuelCellPowerResult:

    net_power_kw: float

    gross_power_kw: float

    auxiliary_power_kw: float

    stack_net_power_kw: float

    stack_gross_power_kw: float

    stack_auxiliary_power_kw: float

    efficiency: float

    cell_voltage_v: float

    stack_h2_flow_g_min: float

    hydrogen_rate_kg_s: float


# =========================================================
# 5. Fuel cell system
# =========================================================

class FuelCellSystem:

    def __init__(self, params=None):

        self.params = (
            params or FuelCellParameters()
        )

        curve_arrays = (
            FC_NET_POWER_KW,
            FC_EFFICIENCY_PERCENT,
            FC_CELL_VOLTAGE,
            FC_AUX_POWER_KW,
            FC_H2_FLOW_G_MIN,
            FC_GROSS_POWER_KW,
        )
        if len({len(x) for x in curve_arrays}) != 1:
            raise ValueError("FC characteristic curve lengths mismatch.")

        if not all(np.all(np.isfinite(x)) for x in curve_arrays):
            raise ValueError("FC characteristic curve contains non-finite data.")

        if (
            np.any(np.diff(FC_NET_POWER_KW) <= 0)
            or np.any(np.diff(FC_GROSS_POWER_KW) <= 0)
        ):
            raise ValueError("FC power axes must be strictly increasing.")

        if not np.allclose(
            FC_GROSS_POWER_KW,
            FC_NET_POWER_KW + FC_AUX_POWER_KW,
            atol=2e-5,
            rtol=0.0
        ):
            raise ValueError(
                "FC characteristic data invalid: "
                "gross != net + auxiliary."
            )

        self.reset()

    @property
    def max_net_power_kw(self):

        return self.params.max_total_net_power_kw

    @property
    def max_gross_power_kw(self):

        return self.params.rated_total_gross_power_kw

    # -----------------------------------------------------
    # Reset
    # -----------------------------------------------------

    def reset(self, initial_is_on=True):

        self.previous_stack_net_power_kw = 0.0

        self.previous_is_on = bool(initial_is_on)

        self.voltage_loss_uv = 0.0

        self.degradation_ratio = 0.0

        self.total_low_power_hours = 0.0

        self.total_high_power_hours = 0.0

        self.total_power_variation_kw = 0.0

        self.total_start_stop_cycles = 0

        self.total_hydrogen_kg = 0.0

        self.total_hydrogen_cost_rmb = 0.0

        self.total_degradation_cost_rmb = 0.0

    # -----------------------------------------------------
    # Interpolation
    # -----------------------------------------------------

    @staticmethod
    def _interp(power_kw, values):

        p = float(power_kw)

        if not np.isfinite(p) or p < -1e-9:
            raise ValueError(
                "Invalid FC interpolation power."
            )

        if p > FC_NET_POWER_KW[-1] + 1e-9:
            raise ValueError(
                "FC power exceeds measured curve."
            )

        return float(
            np.interp(
                max(p, 0.0),
                FC_NET_POWER_KW,
                values
            )
        )

    # -----------------------------------------------------
    # Power and hydrogen calculation
    # -----------------------------------------------------

    def calculate_power(self, net_total_power_kw):

        p_total = float(net_total_power_kw)

        p_max = self.max_net_power_kw

        if (
            not np.isfinite(p_total)
            or p_total < -1e-9
            or p_total > p_max + 1e-9
        ):
            raise ValueError(
                f"Net FC power must be within "
                f"[0, {p_max:.6f}] kW."
            )

        n = self.params.n_stacks

        # Equal net-power distribution.
        p_stack = max(p_total, 0.0) / n

        eta = (
            self._interp(
                p_stack,
                FC_EFFICIENCY_PERCENT
            )
            / 100.0
        )

        voltage = self._interp(
            p_stack,
            FC_CELL_VOLTAGE
        )

        p_aux_stack = self._interp(
            p_stack,
            FC_AUX_POWER_KW
        )

        p_gross_stack = self._interp(
            p_stack,
            FC_GROSS_POWER_KW
        )

        h2_stack_g_min = self._interp(
            p_stack,
            FC_H2_FLOW_G_MIN
        )

        # g/min per stack -> kg/s for the complete system.
        h2_rate_total_kg_s = (
            h2_stack_g_min * n / 60000.0
        )

        return FuelCellPowerResult(

            net_power_kw=p_stack * n,

            gross_power_kw=p_gross_stack * n,

            auxiliary_power_kw=p_aux_stack * n,

            stack_net_power_kw=p_stack,

            stack_gross_power_kw=p_gross_stack,

            stack_auxiliary_power_kw=p_aux_stack,

            efficiency=eta,

            cell_voltage_v=voltage,

            stack_h2_flow_g_min=h2_stack_g_min,

            hydrogen_rate_kg_s=h2_rate_total_kg_s
        )

    # -----------------------------------------------------
    # One-step operation and degradation
    # -----------------------------------------------------

    def step(
        self,
        net_total_power_kw,
        dt_s,
        is_on=True,
        start_stop_cycles=None
    ):

        dt_s = float(dt_s)

        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Invalid dt_s.")

        is_on = bool(is_on)

        p_total = float(net_total_power_kw)

        if not is_on and abs(p_total) > 1e-9:
            raise ValueError(
                "Shutdown FC must have zero output."
            )

        power = self.calculate_power(p_total)

        p_stack = power.stack_net_power_kw

        dt_hours = dt_s / 3600.0

        # =================================================
        # Condition 1 and 2: Low/high-load operation
        # =================================================

        low_hours = 0.0

        high_hours = 0.0

        if is_on:

            if p_stack < self.params.low_power_threshold_kw:

                low_hours = dt_hours

            else:

                high_hours = dt_hours

        # =================================================
        # Condition 3: Load variation
        # =================================================

        # Shutdown power reduction is not counted as an
        # additional transient degradation term.
        if is_on:

            delta_p_stack_kw = abs(
                p_stack - self.previous_stack_net_power_kw
            )

        else:

            delta_p_stack_kw = 0.0

        # =================================================
        # Condition 4: Start-stop
        # =================================================

        if start_stop_cycles is None:

            n_start_stop = int(
                self.previous_is_on and not is_on
            )

        else:

            if (
                int(start_stop_cycles) != start_stop_cycles
                or start_stop_cycles < 0
            ):
                raise ValueError(
                    "Invalid start-stop cycle count."
                )

            n_start_stop = int(start_stop_cycles)

        # =================================================
        # Four-condition voltage degradation
        # =================================================

        delta_v_low_uv = (
            D_LOW_UV_PER_H * low_hours
        )

        delta_v_high_uv = (
            D_HIGH_UV_PER_H * high_hours
        )

        delta_v_transient_uv = (
            D_TRANSIENT_UV_PER_KW
            * delta_p_stack_kw
        )

        delta_v_start_stop_uv = (
            D_START_STOP_UV_PER_CYCLE
            * n_start_stop
        )

        delta_voltage_loss_uv = (
            delta_v_low_uv
            + delta_v_high_uv
            + delta_v_transient_uv
            + delta_v_start_stop_uv
        )

        eol_loss = self.params.eol_voltage_loss_uv

        degradation_before = min(
            self.voltage_loss_uv / eol_loss,
            1.0
        )

        self.voltage_loss_uv += (
            delta_voltage_loss_uv
        )

        degradation_after = min(
            self.voltage_loss_uv / eol_loss,
            1.0
        )

        degradation_increment = max(
            degradation_after - degradation_before,
            0.0
        )

        self.degradation_ratio = degradation_after

        # =================================================
        # Economic cost
        # =================================================

        degradation_cost_rmb = (
            degradation_increment
            * self.params.replacement_cost_rmb
        )

        hydrogen_kg = (
            power.hydrogen_rate_kg_s * dt_s
        )

        hydrogen_cost_rmb = (
            hydrogen_kg
            * self.params.hydrogen_price_rmb_per_kg
        )

        # =================================================
        # Update statistics
        # =================================================

        self.total_low_power_hours += low_hours

        self.total_high_power_hours += high_hours

        self.total_power_variation_kw += (
            delta_p_stack_kw
        )

        self.total_start_stop_cycles += n_start_stop

        # Each quantity is accumulated exactly once.
        self.total_hydrogen_kg += hydrogen_kg

        self.total_hydrogen_cost_rmb += hydrogen_cost_rmb

        self.total_degradation_cost_rmb += (
            degradation_cost_rmb
        )

        self.previous_stack_net_power_kw = p_stack

        self.previous_is_on = is_on

        # =================================================
        # Return
        # =================================================

        return {
            "cumulative_hydrogen_kg": self.total_hydrogen_kg,

            "cumulative_hydrogen_cost_rmb": self.total_hydrogen_cost_rmb,

            "cumulative_degradation_cost_rmb": self.total_degradation_cost_rmb,

            "net_power_kw": power.net_power_kw,

            "gross_power_kw": power.gross_power_kw,

            "auxiliary_power_kw": power.auxiliary_power_kw,

            "stack_net_power_kw": power.stack_net_power_kw,

            "stack_gross_power_kw": power.stack_gross_power_kw,

            "stack_auxiliary_power_kw": (
                power.stack_auxiliary_power_kw
            ),

            "efficiency": power.efficiency,

            "cell_voltage_v": power.cell_voltage_v,

            "stack_h2_flow_g_min": (
                power.stack_h2_flow_g_min
            ),

            "hydrogen_rate_kg_s": (
                power.hydrogen_rate_kg_s
            ),

            "hydrogen_kg": hydrogen_kg,

            "hydrogen_cost_rmb": hydrogen_cost_rmb,

            "low_power_hours": low_hours,

            "high_power_hours": high_hours,

            "power_variation_kw": delta_p_stack_kw,

            "start_stop_cycles": n_start_stop,

            "delta_v_low_uv": delta_v_low_uv,

            "delta_v_high_uv": delta_v_high_uv,

            "delta_v_transient_uv": delta_v_transient_uv,

            "delta_v_start_stop_uv": (
                delta_v_start_stop_uv
            ),

            "delta_voltage_loss_uv": (
                delta_voltage_loss_uv
            ),

            "cumulative_voltage_loss_uv": (
                self.voltage_loss_uv
            ),

            "degradation_ratio": (
                self.degradation_ratio
            ),

            "degradation_pct": (
                self.degradation_ratio * 100.0
            ),

            "delta_degradation_ratio": (
                degradation_increment
            ),

            "degradation_cost_rmb": (
                degradation_cost_rmb
            ),

            "remaining_life_ratio": max(
                1.0 - self.degradation_ratio,
                0.0
            ),

            "eol_reached": (
                self.degradation_ratio >= 1.0 - 1e-12
            )
        }
