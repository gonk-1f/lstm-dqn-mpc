"""State-dependent feasible-interval action and voyage-progress observation wrappers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from v2.data.supervisory_rules import normalize_onboard_load_kw

FC_RATED_KW = 600.0
ACTION_MAPPING_VERSION = "feasible_interval_affine_v1"
PREVIOUS_FC_ACTION_MAPPING_VERSION = "previous_fc_centered_v2"
OBSERVATION_VERSION = "soc_load_prev_fc_load_delta_voyage_progress_fraction_v2"


class VoyageProgressObservationWrapper(gym.ObservationWrapper):
    """Replace elapsed hours by normalized voyage progress in the fifth state slot.

    This is a finite-horizon feature: the route/voyage schedule is assumed to be
    known to the supervisory controller. It preserves the existing 5-D shape.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        high = env.observation_space.high.copy()
        high[4] = 1.0
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.copy(), high=high, dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        base = self.env.unwrapped
        if base.episode is None:
            raise RuntimeError("Voyage progress is unavailable before reset")
        total_steps = len(base.episode.load_kw)
        if total_steps < 1:
            raise RuntimeError("A voyage must contain at least one physical step")
        result = np.asarray(observation, dtype=np.float32).copy()
        result[4] = np.float32(np.clip(base.index / total_steps, 0.0, 1.0))
        return result


class FeasibleIntervalActionWrapper(gym.ActionWrapper):
    """Map normalized Actor output directly into the current physical feasible interval.

    The legacy mapping centers on the interval midpoint. The v6 mapping uses
    the previous executed FC power as its center after the first interval.
    """

    def __init__(self, env: gym.Env, *, action_mapping: str = ACTION_MAPPING_VERSION):
        super().__init__(env)
        if action_mapping not in (ACTION_MAPPING_VERSION, PREVIOUS_FC_ACTION_MAPPING_VERSION):
            raise ValueError(f"Unknown feasible-interval action mapping: {action_mapping}")
        self.action_mapping = action_mapping
        self.last_mapping: dict | None = None

    def _map(self, action) -> tuple[np.ndarray, dict]:
        value = np.asarray(action, dtype=np.float32).reshape(-1)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"Expected one finite normalized action, got {value}")
        if not -1.000001 <= float(value[0]) <= 1.000001:
            raise ValueError(f"Normalized action is outside [-1, 1]: {value}")
        a = float(np.clip(value[0], -1.0, 1.0))
        base = self.env.unwrapped
        if base.episode is None or base.terminated:
            raise RuntimeError("Action mapping requires an active voyage")
        load_kw = normalize_onboard_load_kw(float(base.episode.load_kw[base.index]))
        lower_kw, upper_kw = base._physically_feasible_fc_range(load_kw)
        reference_kw = FC_RATED_KW * (a + 1.0) / 2.0
        previous_centered = self.action_mapping == PREVIOUS_FC_ACTION_MAPPING_VERSION
        center_kw = None
        feasible = lower_kw <= upper_kw + 1e-8
        if feasible:
            width_kw = max(0.0, upper_kw - lower_kw)
            # When the physical range collapses within its round-off tolerance,
            # use the same upper endpoint as the underlying actuator.
            mapping_lower_kw = min(lower_kw, upper_kw)
            if previous_centered and base.index > 0:
                center_kw = float(np.clip(base.physical.previous_fc_kw, mapping_lower_kw, upper_kw))
                span_kw = center_kw - mapping_lower_kw if a < 0.0 else upper_kw - center_kw
                mapped_kw = center_kw + a * span_kw
            else:
                center_kw = mapping_lower_kw + 0.5 * width_kw
                mapped_kw = mapping_lower_kw + (a + 1.0) * 0.5 * width_kw
            inner_action = 2.0 * mapped_kw / FC_RATED_KW - 1.0
            inner = np.array([inner_action], dtype=np.float32)
        else:
            # The inner environment will execute its existing fail-closed path.
            # No fabricated feasible setpoint is sent on an infeasible state.
            width_kw = 0.0
            mapped_kw = None
            inner = np.array([a], dtype=np.float32)
        metadata = {
            "actor_action": a,
            "actor_reference_fc_kw": reference_kw,
            "action_mapped_fc_kw": mapped_kw,
            "action_mapping": self.action_mapping,
            "feasible_interval_width_kw": width_kw,
            "feasible_fc_lower_kw": lower_kw,
            "feasible_fc_upper_kw": upper_kw,
        }
        if previous_centered:
            metadata.update(
                previous_fc_kw=float(base.physical.previous_fc_kw),
                action_center_fc_kw=center_kw,
            )
        self.last_mapping = metadata
        return inner, metadata

    def action(self, action) -> np.ndarray:
        inner, _ = self._map(action)
        return inner

    def step(self, action):
        inner_action, metadata = self._map(action)
        observation, reward, terminated, truncated, info = self.env.step(inner_action)
        info = {**info, **metadata}
        if metadata["action_mapped_fc_kw"] is not None and info.get("step_executed"):
            mismatch = abs(float(info["fc_actual_kw"]) - metadata["action_mapped_fc_kw"])
            if mismatch > 1e-3:
                raise RuntimeError(
                    "Feasible-interval action mapping disagrees with the physical actuator "
                    f"by {mismatch:.6g} kW"
                )
            # Any residual below tolerance is float32 round-off, not a real projection.
            info["projected"] = False
        if "vessel_episode" in info:
            info["vessel_episode"] = {**info["vessel_episode"], "power_projection_fraction": 0.0}
        return observation, reward, terminated, truncated, info
