
"""
Ship Energy Management Environment.

Gymnasium API.

Hybrid system:
    Fuel Cell + Lithium Battery.

State:
    [SOC,
     P_load / P_FC_max,
     P_FC_previous / P_FC_max,
     delta_P_load / P_FC_max,
     time_progress]

Action:
    Total FC net power increment:
    [-60, +60] kW per environment step.

Power balance:
    P_load = P_FC_net + P_battery

Battery:
    Positive power = discharge.
    Negative power = charge.

Episode:
    One complete voyage segment.

Reward:
    Hydrogen cost.
    FC operating degradation cost.
    Battery degradation cost.
    Terminal SOC squared deviation.

Excluded from step reward:
    FC start-stop degradation.
    Battery equivalent hydrogen.

FC start-stop degradation is accounted for
once at voyage-level economic accounting.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

import gymnasium as gym

from gymnasium import spaces

from .config import CONFIG

from .fuel_cell import (
    FuelCellSystem,
    FuelCellParameters,
)

from .battery import (
    LithiumBattery,
    BatteryConfig,
)

from .costs import (
    CostCalculator,
    StepCost,
)


class ShipEMSEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        load_profiles: Sequence,
        sample_ids: Sequence[str] | None = None,
        fuel_cell_params: FuelCellParameters | None = None,
        battery_config: BatteryConfig | None = None,
        dt_s: float = 1.0,
        max_fc_delta_kw: float = 60.0,
        random_voyage: bool = True,
        reward_fn: Callable[[dict], float] | None = None,
        cost_calculator: CostCalculator | None = None,
        reward_alpha: float | None = 0.001,
        soc_terminal_weight: float | None = 20000.0,
        soc_target: float = CONFIG.soc_target,
        failure_penalty: float = -20.0,
    ):

        super().__init__()

        # =================================================
        # 1. Basic parameters
        # =================================================

        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Invalid dt_s.")

        if (
            not np.isfinite(max_fc_delta_kw)
            or max_fc_delta_kw <= 0
        ):
            raise ValueError("Invalid FC action range.")

        if not np.isfinite(soc_target):
            raise ValueError("Invalid SOC target.")

        if (
            not np.isfinite(failure_penalty)
            or failure_penalty > 0
        ):
            raise ValueError("Invalid failure penalty.")

        if (reward_alpha is None) != (
            soc_terminal_weight is None
        ):
            raise ValueError(
                "reward_alpha and soc_terminal_weight "
                "must be configured together."
            )

        if reward_alpha is not None:

            if (
                not np.isfinite(reward_alpha)
                or reward_alpha < 0
            ):
                raise ValueError("Invalid reward_alpha.")

            if (
                not np.isfinite(soc_terminal_weight)
                or soc_terminal_weight < 0
            ):
                raise ValueError(
                    "Invalid SOC terminal weight."
                )

        if (
            reward_fn is not None
            and reward_alpha is not None
        ):
            raise ValueError(
                "Use either reward_fn or built-in reward "
                "parameters, not both."
            )

        self.dt_s = float(dt_s)

        self.max_fc_delta_kw = float(
            max_fc_delta_kw
        )

        self.random_voyage = bool(
            random_voyage
        )

        self.reward_fn = reward_fn

        self.reward_alpha = (
            None
            if reward_alpha is None
            else float(reward_alpha)
        )

        self.soc_terminal_weight = (
            None
            if soc_terminal_weight is None
            else float(soc_terminal_weight)
        )

        self.soc_target = float(soc_target)

        self.failure_penalty = float(
            failure_penalty
        )

        # =================================================
        # 2. Physical models
        # =================================================

        fc_params = (
            fuel_cell_params
            or FuelCellParameters()
        )

        lb_config = (
            battery_config
            or BatteryConfig()
        )

        self.fuel_cell = FuelCellSystem(
            fc_params
        )

        self.battery = LithiumBattery(
            lb_config
        )

        # Important:
        # FC maximum power means maximum NET power,
        # not total gross rated power.

        self.fc_max_kw = (
            self.fuel_cell.max_net_power_kw
        )

        self.fc_gross_max_kw = (
            self.fuel_cell.max_gross_power_kw
        )

        if not (
            self.battery.config.soc_min
            <= self.soc_target
            <= self.battery.config.soc_max
        ):
            raise ValueError(
                "SOC target outside battery limits."
            )

        # =================================================
        # 3. Cost calculator
        # =================================================

        if cost_calculator is None:

            # Use the project's frozen eta_fc_avg=0.5755641791.
            self.cost_calculator = CostCalculator()

        else:

            self.cost_calculator = cost_calculator

        cost_cfg = self.cost_calculator.config

        # -------------------------------------------------
        # Configuration consistency
        # -------------------------------------------------

        if (
            self.fuel_cell.params.n_stacks
            != cost_cfg.fc_stack_count
        ):
            raise ValueError(
                "FC stack count mismatch."
            )

        if not np.isclose(
            self.fuel_cell.params.rated_stack_gross_power_kw,
            cost_cfg.fc_stack_rated_gross_kw,
        ):
            raise ValueError(
                "FC gross rated power mismatch."
            )

        if not np.isclose(
            self.fuel_cell.params.replacement_cost_rmb,
            cost_cfg.fc_replacement_cost_rmb,
        ):
            raise ValueError(
                "FC replacement cost mismatch."
            )

        if not np.isclose(
            self.battery.config.capacity_kwh,
            cost_cfg.battery_total_capacity_kwh,
        ):
            raise ValueError(
                "Battery capacity mismatch."
            )

        battery_config_checks = (
            (
                self.battery.config.cluster_count,
                cost_cfg.battery_cluster_count,
                "Battery cluster count mismatch.",
            ),
            (
                self.battery.config.nominal_voltage_v,
                cost_cfg.battery_rated_voltage_v,
                "Battery nominal voltage mismatch.",
            ),
            (
                self.battery.config.charge_efficiency,
                cost_cfg.eta_charge,
                "Battery charge efficiency mismatch.",
            ),
            (
                self.battery.config.discharge_efficiency,
                cost_cfg.eta_discharge,
                "Battery discharge efficiency mismatch.",
            ),
            (
                self.battery.config.initial_soc,
                cost_cfg.soc_initial,
                "Battery initial SOC mismatch.",
            ),
            (
                self.battery.config.soc_min,
                cost_cfg.soc_min,
                "Battery SOC lower limit mismatch.",
            ),
            (
                self.battery.config.soc_max,
                cost_cfg.soc_max,
                "Battery SOC upper limit mismatch.",
            ),
        )

        for actual, expected, message in battery_config_checks:
            if not np.isclose(actual, expected):
                raise ValueError(message)

        if not np.isclose(
            self.battery.config.max_charge_power_kw,
            cost_cfg.battery_max_charge_kw,
        ):
            raise ValueError(
                "Battery charge power mismatch."
            )

        if not np.isclose(
            self.battery.config.max_discharge_power_kw,
            cost_cfg.battery_max_discharge_kw,
        ):
            raise ValueError(
                "Battery discharge power mismatch."
            )

        if not np.isclose(
            self.fuel_cell.params.hydrogen_price_rmb_per_kg,
            cost_cfg.hydrogen_price_rmb_per_kg,
        ):
            raise ValueError(
                "Hydrogen price mismatch."
            )

        if not np.isclose(
            self.soc_target,
            cost_cfg.soc_target,
        ):
            raise ValueError(
                "SOC target mismatch."
            )

        # =================================================
        # 4. Load voyage profiles
        # =================================================

        if hasattr(load_profiles, "columns"):

            if "load_total_kw" not in load_profiles.columns:
                raise ValueError(
                    "DataFrame must contain load_total_kw."
                )

            raw_profiles = [load_profiles]

        elif isinstance(load_profiles, np.ndarray):

            if load_profiles.ndim == 1:

                raw_profiles = [
                    load_profiles
                ]

            elif load_profiles.ndim == 2:

                raw_profiles = list(
                    load_profiles
                )

            else:

                raise ValueError(
                    "load_profiles must be 1D or 2D."
                )

        else:

            raw_profiles = list(
                load_profiles
            )

            if (
                raw_profiles
                and all(
                    np.isscalar(x)
                    for x in raw_profiles
                )
            ):
                raw_profiles = [
                    np.asarray(
                        raw_profiles,
                        dtype=np.float64
                    )
                ]

        if not raw_profiles:
            raise ValueError(
                "No voyage profiles provided."
            )

        self.load_profiles = []

        for profile in raw_profiles:

            if hasattr(profile, "columns"):

                if "load_total_kw" not in profile.columns:
                    raise ValueError(
                        "Voyage DataFrame must contain "
                        "load_total_kw."
                    )

                # This environment uses one scalar dt_s for every transition.
                # If the loader supplied per-row intervals, reject irregular or
                # mismatched sampling instead of silently ignoring real timing.
                if "dt_to_next_s" in profile.columns:
                    profile_dt = profile[
                        "dt_to_next_s"
                    ].to_numpy(dtype=np.float64)[:-1]

                    if (
                        profile_dt.size != len(profile) - 1
                        or not np.all(np.isfinite(profile_dt))
                        or np.any(profile_dt <= 0)
                        or not np.allclose(
                            profile_dt,
                            self.dt_s,
                            rtol=1e-6,
                            atol=1e-6,
                        )
                    ):
                        raise ValueError(
                            "Voyage sampling intervals do not match the "
                            "environment's fixed dt_s. Use uniform data "
                            "or a variable-step environment."
                        )

                arr = profile[
                    "load_total_kw"
                ].to_numpy(dtype=np.float64)

            else:

                arr = np.asarray(
                    profile,
                    dtype=np.float64
                )

            if arr.ndim != 1 or len(arr) < 2:
                raise ValueError(
                    "Each voyage must contain at least "
                    "two samples."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Invalid load values."
                )

            if np.any(arr < 0):
                raise ValueError(
                    "Negative ship load detected."
                )

            self.load_profiles.append(
                arr.copy()
            )

        self.num_voyages = len(
            self.load_profiles
        )

        if sample_ids is None:

            self.sample_ids = [
                f"voyage_{i:03d}"
                for i in range(self.num_voyages)
            ]

        else:

            if len(sample_ids) != self.num_voyages:
                raise ValueError(
                    "sample_ids length mismatch."
                )

            self.sample_ids = list(
                sample_ids
            )

        # =================================================
        # 5. Action space
        # =================================================

        self.action_space = spaces.Box(
            low=np.array(
                [-self.max_fc_delta_kw],
                dtype=np.float32
            ),
            high=np.array(
                [self.max_fc_delta_kw],
                dtype=np.float32
            ),
            dtype=np.float32
        )

        # =================================================
        # 6. Observation space
        # =================================================

        self.observation_space = spaces.Box(

            low=np.array(
                [
                    self.battery.config.soc_min,
                    -np.inf,
                    0.0,
                    -np.inf,
                    0.0,
                ],
                dtype=np.float32
            ),

            high=np.array(
                [
                    self.battery.config.soc_max,
                    np.inf,
                    1.0,
                    np.inf,
                    1.0,
                ],
                dtype=np.float32
            ),

            dtype=np.float32
        )

        # =================================================
        # 7. Episode variables
        # =================================================

        self.voyage_index = 0

        self.voyage_id = None

        self.load_profile = None

        self.step_index = 0

        self.fc_power_kw = 0.0

        self.initial_soc = None

        self.episode_step_costs: list[StepCost] = []

        self.voyage_costs = None

        self._episode_done = True

    # =====================================================
    # 8. Observation
    # =====================================================

    def _get_obs(self):

        n = len(
            self.load_profile
        )

        idx = min(
            self.step_index,
            n - 1
        )

        load_kw = float(
            self.load_profile[idx]
        )

        if self.step_index == 0:

            delta_load_kw = 0.0

        else:

            delta_load_kw = float(
                self.load_profile[idx]
                - self.load_profile[idx - 1]
            )

        time_progress = min(
            self.step_index / (n - 1),
            1.0
        )

        return np.array(
            [
                self.battery.soc,

                load_kw / self.fc_max_kw,

                self.fc_power_kw / self.fc_max_kw,

                delta_load_kw / self.fc_max_kw,

                time_progress,
            ],
            dtype=np.float32
        )

    # =====================================================
    # 9. Reward readiness
    # =====================================================

    @property
    def reward_ready(self):

        return (
            self.reward_fn is not None
            or (
                self.reward_alpha is not None
                and self.soc_terminal_weight is not None
            )
        )

    # =====================================================
    # 10. Step costs
    # =====================================================

    def _calculate_step_cost(
        self,
        fc_result,
        battery_result,
    ):

        return (
            self.cost_calculator.calculate_step_costs(

                fc_result=fc_result,

                battery_result=battery_result,

                fuel_cell=self.fuel_cell,

                battery=self.battery,
            )
        )

    # =====================================================
    # 11. Reward
    # =====================================================

    def _calculate_reward(
        self,
        info,
        step_cost=None,
    ):

        if info.get("failure", False):

            return self.failure_penalty

        if self.reward_fn is not None:

            reward = float(
                self.reward_fn(info)
            )

            if not np.isfinite(reward):
                raise ValueError(
                    "Non-finite reward."
                )

            return reward

        if self.reward_alpha is None:

            return 0.0

        if step_cost is None:

            raise RuntimeError(
                "Missing step cost."
            )

        # H2 + FC operating degradation
        # + battery degradation.

        reward = (
            -self.reward_alpha
            * step_cost.reward_cost_rmb
        )

        # Terminal SOC squared penalty.

        if info.get("terminated", False):

            soc_error = (
                self.battery.soc
                - self.soc_target
            )

            # SOC terminal penalty is expressed on the same raw-cost scale
            # as the economic terms, so apply reward_alpha to it as well.
            reward -= (
                self.reward_alpha
                * self.soc_terminal_weight
                * soc_error ** 2
            )

        if not np.isfinite(reward):
            raise ValueError(
                "Non-finite reward."
            )

        return float(reward)

    # =====================================================
    # 12. Voyage economic accounting
    # =====================================================

    def _calculate_voyage_costs(self):

        return (
            self.cost_calculator.calculate_voyage_costs(

                step_costs=self.episode_step_costs,

                fuel_cell=self.fuel_cell,

                battery=self.battery,

                initial_soc=float(self.initial_soc),

                final_soc=float(self.battery.soc),

                # One start-stop cycle per complete voyage.
                start_stop_cycles=1,
            )
        )

    def _add_voyage_economic_info(
        self,
        info,
        voyage_costs,
    ):

        info.update({

            "initial_soc": float(
                self.initial_soc
            ),

            "final_soc": float(
                self.battery.soc
            ),

            "soc_terminal_error": (
                self.battery.soc
                - self.soc_target
            ),

            "soc_terminal_abs_error": abs(
                self.battery.soc
                - self.soc_target
            ),

            "soc_terminal_penalty": (
                0.0
                if self.soc_terminal_weight is None
                else (
                    self.soc_terminal_weight
                    * (
                        self.battery.soc
                        - self.soc_target
                    ) ** 2
                )
            ),

            # Actual hydrogen.
            "voyage_hydrogen_kg": (
                voyage_costs.hydrogen_kg
            ),

            "voyage_hydrogen_cost_rmb": (
                voyage_costs.hydrogen_cost_rmb
            ),

            # FC degradation.
            "fc_operating_degradation_cost_rmb": (
                voyage_costs.fc_operating_degradation_cost_rmb
            ),

            "fc_start_stop_degradation_cost_rmb": (
                voyage_costs.fc_start_stop_degradation_cost_rmb
            ),

            "fc_total_degradation_cost_rmb": (
                voyage_costs.fc_total_degradation_cost_rmb
            ),

            # Voyage-level FC indicator includes the separately booked
            # end-of-voyage stop event. The per-step FC indicator above
            # remains the operating-only cumulative indicator.
            "voyage_fc_degradation_indicator": (
                voyage_costs.fc_degradation_indicator
            ),

            "voyage_fc_start_stop_cycles": 1,

            # Voyage-level battery totals use distinct keys. They must not
            # overwrite the last step's incremental cost fields.
            "voyage_battery_degradation_increment": (
                voyage_costs.battery_degradation_increment
            ),

            "voyage_battery_degradation_indicator": (
                voyage_costs.battery_degradation_indicator
            ),

            "battery_raw_degradation_indicator": (
                self.battery.raw_degradation_fraction
            ),

            "voyage_battery_degradation_cost_rmb": (
                voyage_costs.battery_degradation_cost_rmb
            ),

            # Battery equivalent hydrogen.
            "battery_equivalent_hydrogen_kg": (
                voyage_costs.battery_equivalent_hydrogen_kg
            ),

            "battery_equivalent_hydrogen_cost_rmb": (
                voyage_costs.battery_equivalent_hydrogen_cost_rmb
            ),

            # Total equivalent hydrogen.
            "total_equivalent_hydrogen_kg": (
                voyage_costs.total_equivalent_hydrogen_kg
            ),

            "total_equivalent_hydrogen_cost_rmb": (
                voyage_costs.total_equivalent_hydrogen_cost_rmb
            ),

            # Total economic cost.
            "total_economic_cost_rmb": (
                voyage_costs.total_economic_cost_rmb
            ),

        })

    # =====================================================
    # 13. Reset
    # =====================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        options = options or {}

        if "voyage_index" in options:

            idx = int(
                options["voyage_index"]
            )

            if not 0 <= idx < self.num_voyages:
                raise ValueError(
                    "Invalid voyage_index."
                )

        elif self.random_voyage:

            idx = int(
                self.np_random.integers(
                    self.num_voyages
                )
            )

        else:

            idx = 0

        self.voyage_index = idx

        self.voyage_id = (
            self.sample_ids[idx]
        )

        self.load_profile = (
            self.load_profiles[idx]
        )

        # FC remains ON in standby at zero output.
        # No artificial ON/OFF transition is generated
        # during normal environment steps.

        self.fuel_cell.reset(
            initial_is_on=True
        )

        self.battery.reset()

        self.initial_soc = float(
            self.battery.soc
        )

        self.step_index = 0

        self.fc_power_kw = 0.0

        self.episode_step_costs.clear()

        self.voyage_costs = None

        self._episode_done = False

        obs = self._get_obs()

        info = {

            "voyage_id": self.voyage_id,

            "voyage_index": self.voyage_index,

            "num_samples": len(
                self.load_profile
            ),

            "episode_length": (
                len(self.load_profile) - 1
            ),

            "initial_soc": self.initial_soc,

            "initial_fc_net_kw": 0.0,

            "fc_max_net_power_kw": (
                self.fc_max_kw
            ),

            "fc_max_gross_power_kw": (
                self.fc_gross_max_kw
            ),

            "soc_target": self.soc_target,

            "reward_ready": self.reward_ready,

            "economic_cost_ready": True,
        }

        return obs, info

    # =====================================================
    # 14. Environment step
    # =====================================================

    def step(self, action):

        if self._episode_done:

            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        # -------------------------------------------------
        # 14.1 Action validation
        # -------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float64
        ).reshape(-1)

        if action.size != 1:
            raise ValueError(
                "Action must have shape (1,)."
            )

        if not np.all(np.isfinite(action)):
            raise ValueError(
                "Invalid action."
            )

        delta_fc_requested = float(
            np.clip(
                action[0],
                -self.max_fc_delta_kw,
                self.max_fc_delta_kw
            )
        )

        previous_fc_kw = float(
            self.fc_power_kw
        )

        requested_fc_kw = (
            previous_fc_kw
            + delta_fc_requested
        )

        requested_fc_clipped_kw = float(
            np.clip(
                requested_fc_kw,
                0.0,
                self.fc_max_kw
            )
        )

        # -------------------------------------------------
        # 14.2 Current ship load
        # -------------------------------------------------

        if (
            self.step_index
            >= len(self.load_profile) - 1
        ):
            raise RuntimeError(
                "No remaining voyage transition."
            )

        load_kw = float(
            self.load_profile[self.step_index]
        )

        # -------------------------------------------------
        # 14.3 Battery physical feasible range
        # -------------------------------------------------

        battery_min_kw, battery_max_kw = (
            self.battery.get_power_bounds(
                self.dt_s
            )
        )

        # P_battery = P_load - P_FC_net.

        fc_physical_min = max(
            0.0,
            load_kw - battery_max_kw
        )

        fc_physical_max = min(
            self.fc_max_kw,
            load_kw - battery_min_kw
        )

        # -------------------------------------------------
        # 14.4 FC action increment constraints
        # -------------------------------------------------

        fc_action_min = max(
            0.0,
            previous_fc_kw - self.max_fc_delta_kw
        )

        fc_action_max = min(
            self.fc_max_kw,
            previous_fc_kw + self.max_fc_delta_kw
        )

        fc_feasible_min = max(
            fc_physical_min,
            fc_action_min
        )

        fc_feasible_max = min(
            fc_physical_max,
            fc_action_max
        )

        # -------------------------------------------------
        # 14.5 Infeasible operating condition
        # -------------------------------------------------

        if (
            fc_feasible_min
            > fc_feasible_max + 1e-8
        ):

            self._episode_done = True

            info = {

                "voyage_id": self.voyage_id,

                "voyage_index": self.voyage_index,

                "step_index": self.step_index,

                "load_kw": load_kw,

                "requested_action_delta_kw": (
                    delta_fc_requested
                ),

                "requested_fc_kw": (
                    requested_fc_clipped_kw
                ),

                "fc_actual_kw": None,

                "battery_actual_kw": None,

                "soc": float(
                    self.battery.soc
                ),

                "feasible_fc_lower_kw": (
                    fc_feasible_min
                ),

                "feasible_fc_upper_kw": (
                    fc_feasible_max
                ),

                "battery_feasible_lower_kw": (
                    battery_min_kw
                ),

                "battery_feasible_upper_kw": (
                    battery_max_kw
                ),

                "failure": True,

                "terminal_reason": (
                    "no_feasible_power"
                ),

                "terminated": True,

                "truncated": False,

                "reward_ready": self.reward_ready,

                "economic_cost_ready": False,
            }

            reward = self._calculate_reward(
                info
            )

            return (
                self._get_obs(),
                reward,
                True,
                False,
                info
            )

        # -------------------------------------------------
        # 14.6 Project action into feasible FC interval
        # -------------------------------------------------

        fc_actual_kw = float(
            np.clip(
                requested_fc_clipped_kw,
                fc_feasible_min,
                fc_feasible_max
            )
        )

        delta_fc_actual_kw = (
            fc_actual_kw - previous_fc_kw
        )

        projected = (
            abs(
                fc_actual_kw
                - requested_fc_clipped_kw
            ) > 1e-8
        )

        # -------------------------------------------------
        # 14.7 Battery residual power
        # -------------------------------------------------

        battery_actual_kw = (
            load_kw - fc_actual_kw
        )

        # -------------------------------------------------
        # 14.8 Fuel cell
        # -------------------------------------------------

        fc_result = self.fuel_cell.step(

            net_total_power_kw=fc_actual_kw,

            dt_s=self.dt_s,

            is_on=True,

            # No start-stop degradation in step reward.
            start_stop_cycles=0,
        )

        # -------------------------------------------------
        # 14.9 Battery
        # -------------------------------------------------

        battery_result = self.battery.step(

            power_kw=battery_actual_kw,

            dt_s=self.dt_s,
        )

        # -------------------------------------------------
        # 14.10 Update state
        # -------------------------------------------------

        self.fc_power_kw = fc_actual_kw

        self.step_index += 1

        terminated = (
            self.step_index
            >= len(self.load_profile) - 1
        )

        self._episode_done = terminated

        # -------------------------------------------------
        # 14.11 Step costs
        # -------------------------------------------------

        step_cost = self._calculate_step_cost(
            fc_result=fc_result,
            battery_result=battery_result,
        )

        self.episode_step_costs.append(
            step_cost
        )

        if int(fc_result["start_stop_cycles"]) != 0:

            raise RuntimeError(
                "Unexpected FC start-stop degradation "
                "in step reward."
            )

        # -------------------------------------------------
        # 14.12 Information
        # -------------------------------------------------

        info = {

            "voyage_id": self.voyage_id,

            "voyage_index": self.voyage_index,

            "step_index": self.step_index - 1,

            "load_kw": load_kw,

            # Action.
            "requested_action_delta_kw": (
                delta_fc_requested
            ),

            "requested_fc_kw": (
                requested_fc_clipped_kw
            ),

            "fc_actual_kw": (
                fc_actual_kw
            ),

            "delta_fc_kw": (
                delta_fc_actual_kw
            ),

            "projected": projected,

            # Battery.
            "battery_actual_kw": (
                battery_actual_kw
            ),

            # FC and battery feasible ranges.
            "feasible_fc_lower_kw": (
                fc_feasible_min
            ),

            "feasible_fc_upper_kw": (
                fc_feasible_max
            ),

            "battery_feasible_lower_kw": (
                battery_min_kw
            ),

            "battery_feasible_upper_kw": (
                battery_max_kw
            ),

            # Power balance.
            "power_balance_error_kw": (
                fc_actual_kw
                + battery_actual_kw
                - load_kw
            ),

            # FC gross/net/auxiliary power.
            "fc_net_power_kw": (
                float(fc_result["net_power_kw"])
            ),

            "fc_gross_power_kw": (
                float(fc_result["gross_power_kw"])
            ),

            "fc_auxiliary_power_kw": (
                float(fc_result["auxiliary_power_kw"])
            ),

            # SOC.
            "soc_before": (
                float(battery_result.soc_before)
            ),

            "soc": (
                float(battery_result.soc_after)
            ),

            "soc_delta_from_initial": (
                float(self.battery.soc)
                - float(self.initial_soc)
            ),

            "soc_target": self.soc_target,

            # Hydrogen.
            "hydrogen_kg": (
                step_cost.hydrogen_kg
            ),

            "hydrogen_cost_rmb": (
                step_cost.hydrogen_cost_rmb
            ),

            "fc_efficiency": (
                float(fc_result["efficiency"])
            ),

            "fc_cell_voltage_v": (
                float(fc_result["cell_voltage_v"])
            ),

            # FC four-condition degradation.
            "fc_low_power_hours": (
                float(fc_result["low_power_hours"])
            ),

            "fc_high_power_hours": (
                float(fc_result["high_power_hours"])
            ),

            "fc_power_variation_kw": (
                float(fc_result["power_variation_kw"])
            ),

            "fc_delta_v_low_uv": (
                float(fc_result["delta_v_low_uv"])
            ),

            "fc_delta_v_high_uv": (
                float(fc_result["delta_v_high_uv"])
            ),

            "fc_delta_v_transient_uv": (
                float(fc_result["delta_v_transient_uv"])
            ),

            "fc_delta_v_start_stop_uv": (
                float(fc_result["delta_v_start_stop_uv"])
            ),

            "fc_delta_voltage_loss_uv": (
                float(fc_result["delta_voltage_loss_uv"])
            ),

            "fc_cumulative_voltage_loss_uv": (
                float(fc_result["cumulative_voltage_loss_uv"])
            ),

            "fc_degradation_ratio": (
                float(fc_result["degradation_ratio"])
            ),

            "fc_degradation_pct": (
                float(fc_result["degradation_pct"])
            ),

            "fc_degradation_increment": (
                float(step_cost.fc_degradation_increment)
            ),

            "fc_degradation_cost_rmb": (
                float(step_cost.fc_degradation_cost_rmb)
            ),

            "fc_start_stop_cycles": (
                int(fc_result["start_stop_cycles"])
            ),

            # Battery degradation.
            "battery_current_a": (
                float(battery_result.current_a)
            ),

            "battery_cluster_current_a": (
                float(battery_result.cluster_current_a)
            ),

            "battery_f_soc": (
                float(battery_result.f_soc)
            ),

            "battery_g_current": (
                float(battery_result.g_current)
            ),

            "battery_weighted_aging_increment_ah": (
                float(
                    battery_result.weighted_aging_increment_ah
                )
            ),

            "battery_degradation_indicator": (
                float(self.battery.degradation_fraction())
            ),

            "battery_raw_degradation_indicator": (
                float(self.battery.raw_degradation_fraction)
            ),

            "battery_degradation_increment": (
                float(step_cost.battery_degradation_increment)
            ),

            "battery_degradation_cost_rmb": (
                float(step_cost.battery_degradation_cost_rmb)
            ),

            # Reward operating costs.
            "step_reward_cost_rmb": (
                float(step_cost.reward_cost_rmb)
            ),

            "terminated": terminated,

            "truncated": False,

            "terminal_reason": (
                "voyage_end" if terminated else None
            ),

            "failure": False,

            "reward_ready": self.reward_ready,

            "economic_cost_ready": True,
        }

        # -------------------------------------------------
        # 14.13 Voyage-level economic accounting
        # -------------------------------------------------

        if terminated:

            self.voyage_costs = (
                self._calculate_voyage_costs()
            )

            self._add_voyage_economic_info(
                info=info,
                voyage_costs=self.voyage_costs,
            )

        # -------------------------------------------------
        # 14.14 Reward
        # -------------------------------------------------

        reward = self._calculate_reward(
            info=info,
            step_cost=step_cost,
        )

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            info
        )

    # =====================================================
    # 15. Close
    # =====================================================

    def close(self):
        pass
