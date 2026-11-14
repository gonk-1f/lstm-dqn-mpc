
"""Ship energy management environment (fuel cell + lithium battery).

The action selects a relative position in the current feasible FC net-power
interval.

Default reward when no custom reward is supplied:
    -reward_alpha * (H2 cost + FC degradation cost + LB degradation cost
                     + soc_step_weight * (SOC_next - soc_target)**2)

There is no additional terminal SOC penalty. Power infeasibility retains
its standalone failure penalty.
"""

from __future__ import annotations

from typing import Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .battery import BatteryConfig, LithiumBattery
from .config import CONFIG
from .costs import CostCalculator, StepCost
from .fuel_cell import FuelCellParameters, FuelCellSystem


class ShipEMSEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        load_profiles: Sequence,
        sample_ids: Sequence[str] | None = None,
        fuel_cell_params: FuelCellParameters | None = None,
        battery_config: BatteryConfig | None = None,
        dt_s: float = 1.0,
        random_voyage: bool = True,
        reward_fn: Callable[[dict], float] | None = None,
        cost_calculator: CostCalculator | None = None,
        reward_alpha: float | None = 0.001,
        soc_step_weight: float | None = 0.5,
        soc_target: float = CONFIG.soc_target,
        failure_penalty: float = -20.0,
        *,
        train_window_steps: int | None = None,
    ):
        super().__init__()

        if not np.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Invalid dt_s.")

        if not np.isfinite(soc_target):
            raise ValueError("Invalid SOC target.")

        if not np.isfinite(failure_penalty) or failure_penalty > 0:
            raise ValueError("Invalid failure penalty.")

        if (reward_alpha is None) != (soc_step_weight is None):
            raise ValueError(
                "reward_alpha and soc_step_weight must be configured together."
            )

        if reward_alpha is not None:
            if not np.isfinite(reward_alpha) or reward_alpha < 0:
                raise ValueError("Invalid reward_alpha.")

            if not np.isfinite(soc_step_weight) or soc_step_weight < 0:
                raise ValueError("Invalid SOC step weight.")

        if reward_fn is not None and reward_alpha is not None:
            raise ValueError(
                "Use either reward_fn or built-in reward parameters, not both."
            )

        self.dt_s = float(dt_s)
        self.random_voyage = bool(random_voyage)

        self.reward_fn = reward_fn
        self.reward_alpha = (
            None if reward_alpha is None else float(reward_alpha)
        )
        self.soc_step_weight = (
            None if soc_step_weight is None else float(soc_step_weight)
        )
        self.soc_target = float(soc_target)
        self.failure_penalty = float(failure_penalty)

        # ------------------------------------------------
        # Physical models
        # ------------------------------------------------

        self.fuel_cell = FuelCellSystem(
            fuel_cell_params or FuelCellParameters()
        )
        self.battery = LithiumBattery(
            battery_config or BatteryConfig()
        )

        self.fc_max_kw = self.fuel_cell.max_net_power_kw
        self.fc_gross_max_kw = self.fuel_cell.max_gross_power_kw
        self.load_scale_kw = (
            self.fc_max_kw + self.battery.config.max_discharge_power_kw
        )

        if not (
            self.battery.config.soc_min
            <= self.soc_target
            <= self.battery.config.soc_max
        ):
            raise ValueError("SOC target outside battery limits.")

        self.cost_calculator = (
            cost_calculator
            if cost_calculator is not None
            else CostCalculator()
        )
        cost_cfg = self.cost_calculator.config

        # ------------------------------------------------
        # Configuration consistency
        # ------------------------------------------------

        if self.fuel_cell.params.n_stacks != cost_cfg.fc_stack_count:
            raise ValueError("FC stack count mismatch.")

        if not np.isclose(
            self.fuel_cell.params.rated_stack_gross_power_kw,
            cost_cfg.fc_stack_rated_gross_kw,
        ):
            raise ValueError("FC gross rated power mismatch.")

        if not np.isclose(
            self.fuel_cell.params.replacement_cost_rmb,
            cost_cfg.fc_replacement_cost_rmb,
        ):
            raise ValueError("FC replacement cost mismatch.")

        if not np.isclose(
            self.battery.config.capacity_kwh,
            cost_cfg.battery_total_capacity_kwh,
        ):
            raise ValueError("Battery capacity mismatch.")

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
            raise ValueError("Battery charge power mismatch.")

        if not np.isclose(
            self.battery.config.max_discharge_power_kw,
            cost_cfg.battery_max_discharge_kw,
        ):
            raise ValueError("Battery discharge power mismatch.")

        if not np.isclose(
            self.fuel_cell.params.hydrogen_price_rmb_per_kg,
            cost_cfg.hydrogen_price_rmb_per_kg,
        ):
            raise ValueError("Hydrogen price mismatch.")

        if not np.isclose(self.soc_target, cost_cfg.soc_target):
            raise ValueError("SOC target mismatch.")

        # ------------------------------------------------
        # Load voyage profiles
        # ------------------------------------------------

        if hasattr(load_profiles, "columns"):
            if "load_total_kw" not in load_profiles.columns:
                raise ValueError(
                    "DataFrame must contain load_total_kw."
                )
            raw_profiles = [load_profiles]

        elif isinstance(load_profiles, np.ndarray):
            if load_profiles.ndim == 1:
                raw_profiles = [load_profiles]
            elif load_profiles.ndim == 2:
                raw_profiles = list(load_profiles)
            else:
                raise ValueError("load_profiles must be 1D or 2D.")

        else:
            raw_profiles = list(load_profiles)
            if raw_profiles and all(
                np.isscalar(x) for x in raw_profiles
            ):
                raw_profiles = [
                    np.asarray(raw_profiles, dtype=np.float64)
                ]

        if not raw_profiles:
            raise ValueError("No voyage profiles provided.")

        self.load_profiles = []

        for profile in raw_profiles:
            if hasattr(profile, "columns"):
                if "load_total_kw" not in profile.columns:
                    raise ValueError(
                        "Voyage DataFrame must contain load_total_kw."
                    )

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
                arr = np.asarray(profile, dtype=np.float64)

            if arr.ndim != 1 or len(arr) < 2:
                raise ValueError(
                    "Each voyage must contain at least two samples."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError("Invalid load values.")

            if np.any(arr < 0):
                raise ValueError("Negative ship load detected.")

            self.load_profiles.append(arr.copy())

        self.num_voyages = len(self.load_profiles)
        if train_window_steps is not None and not (
            0 < train_window_steps < min(map(len, self.load_profiles))
        ):
            raise ValueError("Training window must fit every supplied voyage.")
        self.train_window_steps = train_window_steps

        if sample_ids is None:
            self.sample_ids = [
                f"voyage_{i:03d}"
                for i in range(self.num_voyages)
            ]
        else:
            if len(sample_ids) != self.num_voyages:
                raise ValueError("sample_ids length mismatch.")
            self.sample_ids = list(sample_ids)

        # ------------------------------------------------
        # Action space
        # ------------------------------------------------

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # ------------------------------------------------
        # Observation space
        # ------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # ------------------------------------------------
        # Episode state
        # ------------------------------------------------

        self.voyage_index = 0
        self.voyage_id = None
        self.load_profile = None
        self.step_index = 0
        self.fc_power_kw = 0.0
        self.initial_soc = None

        self.episode_step_costs: list[StepCost] = []
        self.voyage_costs = None
        self._episode_done = True

    # ====================================================
    # Observation
    # ====================================================

    def _get_obs(self):
        idx = min(self.step_index, len(self.load_profile) - 1)
        observation = np.array(
            [
                self.fc_power_kw / self.fc_max_kw,
                self.load_profile[idx] / self.load_scale_kw,
                self.battery.soc,
            ],
            dtype=np.float32,
        )
        return np.clip(observation, 0.0, 1.0)

    # ====================================================
    # Reward readiness
    # ====================================================

    @property
    def reward_ready(self):
        return (
            self.reward_fn is not None
            or (
                self.reward_alpha is not None
                and self.soc_step_weight is not None
            )
        )

    # ====================================================
    # Step cost
    # ====================================================

    def _calculate_step_cost(self, fc_result, battery_result):
        return self.cost_calculator.calculate_step_costs(
            fc_result=fc_result,
            battery_result=battery_result,
            fuel_cell=self.fuel_cell,
            battery=self.battery,
            dt_s=self.dt_s,
        )

    # ====================================================
    # Reward calculation
    # ====================================================

    def _calculate_reward(self, info, step_cost=None):
        """
        Normal step:
            r = -alpha * (
                H2_cost + FC_deg_cost + LB_deg_cost
                + beta_step * (SOC_next - target)^2
            )

        Final normal step:
            Same formula, without any extra terminal penalty.

        Infeasible failure:
            Fixed failure penalty, replacing normal reward.
        """

        if info.get("failure", False):
            return self.failure_penalty

        if self.reward_fn is not None:
            reward = float(self.reward_fn(info))

            if not np.isfinite(reward):
                raise ValueError("Non-finite reward.")

            return reward

        if self.reward_alpha is None:
            return 0.0

        if step_cost is None:
            raise RuntimeError("Missing step cost.")

        # Actual SOC after executing the current battery step.
        soc_error = (
            self.battery.soc - self.soc_target
        )

        soc_penalty = (
            self.soc_step_weight * soc_error ** 2
        )

        reward = -self.reward_alpha * (
            step_cost.reward_cost_rmb
            + soc_penalty
        )

        if not np.isfinite(reward):
            raise ValueError("Non-finite reward.")

        return float(reward)

    # ====================================================
    # Voyage economic accounting
    # ====================================================

    def _calculate_voyage_costs(self, start_stop_cycles: int):
        return self.cost_calculator.calculate_voyage_costs(
            step_costs=self.episode_step_costs,
            fuel_cell=self.fuel_cell,
            battery=self.battery,
            initial_soc=float(self.initial_soc),
            final_soc=float(self.battery.soc),
            start_stop_cycles=start_stop_cycles,
        )

    def _add_voyage_economic_info(self, info, voyage_costs, start_stop_cycles: int):
        info.update({
            "initial_soc": float(self.initial_soc),
            "final_soc": float(self.battery.soc),

            "soc_terminal_error": (
                self.battery.soc - self.soc_target
            ),

            "soc_terminal_abs_error": abs(
                self.battery.soc - self.soc_target
            ),

            # Compatibility with earlier evaluation tables.
            # No terminal SOC penalty is applied.
            "soc_terminal_penalty": 0.0,

            "voyage_hydrogen_kg": (
                voyage_costs.hydrogen_kg
            ),

            "voyage_hydrogen_cost_rmb": (
                voyage_costs.hydrogen_cost_rmb
            ),

            "fc_operating_degradation_cost_rmb": (
                voyage_costs.fc_operating_degradation_cost_rmb
            ),

            "fc_start_stop_degradation_cost_rmb": (
                voyage_costs.fc_start_stop_degradation_cost_rmb
            ),

            "fc_total_degradation_cost_rmb": (
                voyage_costs.fc_total_degradation_cost_rmb
            ),

            "voyage_fc_degradation_indicator": (
                voyage_costs.fc_degradation_indicator
            ),

            "voyage_fc_start_stop_cycles": start_stop_cycles,

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

            "battery_equivalent_hydrogen_kg": (
                voyage_costs.battery_equivalent_hydrogen_kg
            ),

            "battery_equivalent_hydrogen_cost_rmb": (
                voyage_costs.battery_equivalent_hydrogen_cost_rmb
            ),

            "total_equivalent_hydrogen_kg": (
                voyage_costs.total_equivalent_hydrogen_kg
            ),

            "total_equivalent_hydrogen_cost_rmb": (
                voyage_costs.total_equivalent_hydrogen_cost_rmb
            ),

            "total_economic_cost_rmb": (
                voyage_costs.total_economic_cost_rmb
            ),
        })

    # ====================================================
    # Reset
    # ====================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "voyage_index" in options:
            idx = int(options["voyage_index"])

            if not 0 <= idx < self.num_voyages:
                raise ValueError("Invalid voyage_index.")

        elif self.random_voyage:
            idx = int(
                self.np_random.integers(self.num_voyages)
            )

        else:
            idx = 0

        self.voyage_index = idx
        self.voyage_id = self.sample_ids[idx]
        profile = self.load_profiles[idx]
        self.load_profile = profile
        self._window_info = {}
        if self.train_window_steps is not None:
            start = int(self.np_random.integers(len(profile) - self.train_window_steps))
            end = start + self.train_window_steps
            self.load_profile = profile[start:end + 1]
            self._window_info = {
                "episode_scope": "training_window",
                "window_start_step": start,
                "window_end_step": end,
            }

        # FC stays on in standby at zero power.
        self.fuel_cell.reset(initial_is_on=True)
        self.battery.reset()

        self.initial_soc = float(self.battery.soc)
        self.step_index = 0
        self.fc_power_kw = 0.0

        self.episode_step_costs.clear()
        self.voyage_costs = None
        self._episode_done = False

        info = {
            **self._window_info,
            "voyage_id": self.voyage_id,
            "voyage_index": self.voyage_index,
            "num_samples": len(self.load_profile),
            "episode_length": len(self.load_profile) - 1,
            "initial_soc": self.initial_soc,
            "initial_fc_net_kw": 0.0,
            "fc_max_net_power_kw": self.fc_max_kw,
            "fc_max_gross_power_kw": self.fc_gross_max_kw,
            "soc_target": self.soc_target,
            "reward_ready": self.reward_ready,
            "economic_cost_ready": True,
        }

        return self._get_obs(), info

    # ====================================================
    # Environment step
    # ====================================================

    def step(self, action):
        if self._episode_done:
            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        # ------------------------------------------------
        # 1. Validate action
        # ------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float64,
        ).reshape(-1)

        if action.size != 1:
            raise ValueError("Action must have shape (1,).")

        if not np.all(np.isfinite(action)):
            raise ValueError("Invalid action.")

        action_fraction = float(np.clip(action[0], 0.0, 1.0))

        previous_fc_kw = float(self.fc_power_kw)

        if self.step_index >= len(self.load_profile) - 1:
            raise RuntimeError(
                "No remaining voyage transition."
            )

        # ------------------------------------------------
        # 2. Current ship load
        # ------------------------------------------------

        load_kw = float(
            self.load_profile[self.step_index]
        )

        # ------------------------------------------------
        # 3. Physical feasible intervals
        # ------------------------------------------------

        battery_min_kw, battery_max_kw = (
            self.battery.get_power_bounds(self.dt_s)
        )

        # Power balance:
        # P_battery = P_load - P_FC_net

        fc_feasible_min = max(
            0.0,
            load_kw - battery_max_kw,
        )

        fc_feasible_max = min(
            self.fc_max_kw,
            load_kw - battery_min_kw,
        )

        # ------------------------------------------------
        # 4. Infeasible operating condition
        # ------------------------------------------------

        if fc_feasible_min > fc_feasible_max + 1e-8:
            self._episode_done = True

            info = {
                **self._window_info,
                "voyage_id": self.voyage_id,
                "voyage_index": self.voyage_index,
                "step_index": self.step_index,

                "load_kw": load_kw,

                "requested_action_fraction": action_fraction,
                "requested_fc_kw": None,

                "fc_actual_kw": None,
                "battery_actual_kw": None,

                "soc": float(self.battery.soc),

                "feasible_fc_lower_kw": fc_feasible_min,
                "feasible_fc_upper_kw": fc_feasible_max,

                "battery_feasible_lower_kw": battery_min_kw,
                "battery_feasible_upper_kw": battery_max_kw,

                "failure": True,
                "terminal_reason": "no_feasible_power",

                "terminated": True,
                "truncated": False,

                "reward_ready": self.reward_ready,
                "economic_cost_ready": False,
            }

            reward = self._calculate_reward(info)

            return (
                self._get_obs(),
                reward,
                True,
                False,
                info,
            )

        # ------------------------------------------------
        # 5. Map the action directly into the feasible FC interval
        # ------------------------------------------------

        # Collapse an interval inverted only by the rounding tolerance.
        fc_feasible_min = min(fc_feasible_min, fc_feasible_max)
        requested_fc_kw = fc_feasible_min + action_fraction * (
            fc_feasible_max - fc_feasible_min
        )
        fc_actual_kw = requested_fc_kw

        delta_fc_actual_kw = (
            fc_actual_kw - previous_fc_kw
        )

        # ------------------------------------------------
        # 6. Battery residual power
        # ------------------------------------------------

        battery_actual_kw = (
            load_kw - fc_actual_kw
        )

        # ------------------------------------------------
        # 7. Fuel-cell update
        # ------------------------------------------------

        fc_result = self.fuel_cell.step(
            net_total_power_kw=fc_actual_kw,
            dt_s=self.dt_s,
            is_on=True,
            start_stop_cycles=0,
        )

        # ------------------------------------------------
        # 8. Battery update
        # ------------------------------------------------

        battery_result = self.battery.step(
            power_kw=battery_actual_kw,
            dt_s=self.dt_s,
        )

        # ------------------------------------------------
        # 9. Update state
        # ------------------------------------------------

        self.fc_power_kw = fc_actual_kw
        self.step_index += 1

        terminated = self.step_index >= len(self.load_profile) - 1
        self._episode_done = terminated
        terminal_reason = None
        if terminated:
            terminal_reason = (
                "training_window_end" if self.train_window_steps is not None else "voyage_end"
            )

        # ------------------------------------------------
        # 10. Step operating costs
        # ------------------------------------------------

        step_cost = self._calculate_step_cost(
            fc_result=fc_result,
            battery_result=battery_result,
        )

        self.episode_step_costs.append(step_cost)

        if int(fc_result["start_stop_cycles"]) != 0:
            raise RuntimeError(
                "Unexpected FC start-stop degradation "
                "in step reward."
            )

        # ------------------------------------------------
        # 11. Per-step SOC penalty for diagnostics
        # ------------------------------------------------

        soc_penalty_unscaled = (
            0.0
            if self.soc_step_weight is None
            else float(
                self.soc_step_weight
                * (
                    self.battery.soc
                    - self.soc_target
                ) ** 2
            )
        )

        # ------------------------------------------------
        # 12. Step information
        # ------------------------------------------------

        info = {
            **self._window_info,
            "voyage_id": self.voyage_id,
            "voyage_index": self.voyage_index,
            "step_index": self.step_index - 1,

            "load_kw": load_kw,

            # Actions
            "requested_action_fraction": action_fraction,
            "requested_fc_kw": requested_fc_kw,
            "fc_actual_kw": fc_actual_kw,
            "delta_fc_kw": delta_fc_actual_kw,
            "projected": False,

            # Battery power
            "battery_actual_kw": battery_actual_kw,

            # Feasibility intervals
            "feasible_fc_lower_kw": fc_feasible_min,
            "feasible_fc_upper_kw": fc_feasible_max,
            "battery_feasible_lower_kw": battery_min_kw,
            "battery_feasible_upper_kw": battery_max_kw,

            # Power balance
            "power_balance_error_kw": (
                fc_actual_kw + battery_actual_kw - load_kw
            ),

            # FC gross/net power
            "fc_net_power_kw": float(
                fc_result["net_power_kw"]
            ),

            "fc_gross_power_kw": float(
                fc_result["gross_power_kw"]
            ),

            "fc_auxiliary_power_kw": float(
                fc_result["auxiliary_power_kw"]
            ),

            # Battery SOC
            "soc_before": float(
                battery_result.soc_before
            ),

            "soc_after": float(battery_result.soc_after),
            "soc": float(
                battery_result.soc_after
            ),

            "soc_delta_from_initial": (
                float(self.battery.soc)
                - float(self.initial_soc)
            ),

            "soc_target": self.soc_target,

            # Hydrogen consumption
            "hydrogen_kg": step_cost.hydrogen_kg,
            "hydrogen_cost_rmb": step_cost.hydrogen_cost_rmb,

            "fc_efficiency": float(
                fc_result["efficiency"]
            ),

            "fc_cell_voltage_v": float(
                fc_result["cell_voltage_v"]
            ),

            # FC degradation
            "fc_low_power_hours": float(
                fc_result["low_power_hours"]
            ),

            "fc_high_power_hours": float(
                fc_result["high_power_hours"]
            ),

            "fc_power_variation_kw": float(
                fc_result["power_variation_kw"]
            ),

            "fc_delta_v_low_uv": float(
                fc_result["delta_v_low_uv"]
            ),

            "fc_delta_v_high_uv": float(
                fc_result["delta_v_high_uv"]
            ),

            "fc_delta_v_transient_uv": float(
                fc_result["delta_v_transient_uv"]
            ),

            "fc_delta_v_start_stop_uv": float(
                fc_result["delta_v_start_stop_uv"]
            ),

            "fc_delta_voltage_loss_uv": float(
                fc_result["delta_voltage_loss_uv"]
            ),

            "fc_cumulative_voltage_loss_uv": float(
                fc_result["cumulative_voltage_loss_uv"]
            ),

            "fc_degradation_ratio": float(
                fc_result["degradation_ratio"]
            ),

            "fc_degradation_pct": float(
                fc_result["degradation_pct"]
            ),

            "fc_degradation_increment": float(
                step_cost.fc_degradation_increment
            ),

            "fc_degradation_cost_rmb": float(
                step_cost.fc_degradation_cost_rmb
            ),

            "fc_start_stop_cycles": int(
                fc_result["start_stop_cycles"]
            ),

            # Battery degradation
            "battery_current_a": float(
                battery_result.current_a
            ),

            "battery_cluster_current_a": float(
                battery_result.cluster_current_a
            ),

            "battery_f_soc": float(
                battery_result.f_soc
            ),

            "battery_g_current": float(
                battery_result.g_current
            ),

            "battery_weighted_aging_increment_ah": float(
                battery_result.weighted_aging_increment_ah
            ),

            "battery_degradation_indicator": float(
                self.battery.degradation_fraction()
            ),

            "battery_raw_degradation_indicator": float(
                self.battery.raw_degradation_fraction
            ),

            "battery_degradation_increment": float(
                step_cost.battery_degradation_increment
            ),

            "battery_degradation_cost_rmb": float(
                step_cost.battery_degradation_cost_rmb
            ),

            # Base operating costs and signed battery energy adjustment
            "step_reward_cost_rmb": float(
                step_cost.reward_cost_rmb
            ),
            "step_battery_equivalent_hydrogen_kg": (
                step_cost.battery_equivalent_hydrogen_kg
            ),
            "step_battery_equivalent_hydrogen_cost_rmb": (
                step_cost.battery_equivalent_hydrogen_cost_rmb
            ),
            "battery_equivalent_soc_factor": step_cost.battery_equivalent_soc_factor,
            "battery_equivalent_s": step_cost.battery_equivalent_s,

            # SOC penalty used by the reward function, independent of costs.
            # This is NOT actual economic cost.
            "step_soc_penalty_unscaled": (
                soc_penalty_unscaled
            ),

            # Episode status
            "terminated": terminated,
            "truncated": False,
            "terminal_reason": terminal_reason,

            "failure": False,
            "reward_ready": self.reward_ready,
            "economic_cost_ready": True,
        }

        # ------------------------------------------------
        # 13. Voyage economic accounting
        # ------------------------------------------------

        if terminated:
            # Only full-voyage episodes record a complete physical cycle.
            start_stop_cycles = int(self.train_window_steps is None)
            self.voyage_costs = self._calculate_voyage_costs(start_stop_cycles)
            self._add_voyage_economic_info(
                info=info,
                voyage_costs=self.voyage_costs,
                start_stop_cycles=start_stop_cycles,
            )

        # ------------------------------------------------
        # 14. Final reward
        # ------------------------------------------------

        reward = self._calculate_reward(
            info=info,
            step_cost=step_cost,
        )

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            info,
        )

    # ====================================================
    # Close
    # ====================================================

    def close(self):
        pass
