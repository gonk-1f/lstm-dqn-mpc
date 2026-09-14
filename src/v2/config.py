from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class TimeScaleConfig:
    """Independent lower-MPC and upper-DQN timing parameters.

    ``n_mpc`` is a prediction-horizon length. ``dqn_switch_steps`` is the
    count of real receding-horizon control periods for which an action is held.
    They deliberately remain distinct even when their provisional values match.
    """

    ts_mpc_seconds: float
    n_mpc: int
    dqn_switch_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.ts_mpc_seconds, bool)
            or not isinstance(self.ts_mpc_seconds, Real)
            or not math.isfinite(self.ts_mpc_seconds)
            or self.ts_mpc_seconds <= 0
        ):
            raise ValueError("ts_mpc_seconds must be finite and positive")
        object.__setattr__(self, "ts_mpc_seconds", float(self.ts_mpc_seconds))
        if type(self.n_mpc) is not int or self.n_mpc <= 0:
            raise ValueError("n_mpc must be a positive integer")
        if type(self.dqn_switch_steps) is not int or self.dqn_switch_steps <= 0:
            raise ValueError("dqn_switch_steps must be a positive integer")

    @classmethod
    def provisional(cls) -> "TimeScaleConfig":
        return cls(ts_mpc_seconds=30.0, n_mpc=5, dqn_switch_steps=5)

    @property
    def prediction_seconds(self) -> float:
        return float(self.ts_mpc_seconds) * self.n_mpc

    @property
    def switch_seconds(self) -> float:
        return float(self.ts_mpc_seconds) * self.dqn_switch_steps

    @property
    def mpc_solves_per_action(self) -> int:
        return self.dqn_switch_steps

    @property
    def n_mpc_semantics(self) -> str:
        return "future MPC prediction steps per rolling solve"

    @property
    def dqn_switch_steps_semantics(self) -> str:
        return "executed MPC control periods per selected DQN action"


@dataclass(frozen=True)
class PlantConfig:
    fuel_cell_rated_total_kw: float
    battery_nominal_energy_kwh: float
    source_type: str
    source_reference: str

    @classmethod
    def research_simulation(cls) -> "PlantConfig":
        return cls(
            fuel_cell_rated_total_kw=600.0,
            battery_nominal_energy_kwh=624.0,
            source_type="research_simulation",
            source_reference=(
                "Yang et al., Ocean Engineering (2026), "
                "DOI 10.1016/j.oceaneng.2026.125687"
            ),
        )

    @classmethod
    def project_configuration(cls) -> "PlantConfig":
        """Backward-compatible name for the approved research simulation."""

        return cls.research_simulation()


@dataclass(frozen=True)
class RealVesselSpecification:
    fuel_cell_rated_total_kw: float
    fuel_cell_module_count: int
    fuel_cell_module_rated_kw: float
    battery_nominal_energy_kwh: float
    battery_cluster_count: int
    battery_rated_output_min_kw: float
    battery_rated_voltage_v: float
    operation_ends_with_shore_charging: bool
    source_type: str
    source_reference: str

    @classmethod
    def from_authoritative_specification(cls) -> "RealVesselSpecification":
        return cls(
            fuel_cell_rated_total_kw=560.0,
            fuel_cell_module_count=8,
            fuel_cell_module_rated_kw=70.0,
            battery_nominal_energy_kwh=1806.0,
            battery_cluster_count=12,
            battery_rated_output_min_kw=900.0,
            battery_rated_voltage_v=537.6,
            operation_ends_with_shore_charging=True,
            source_type="real_vessel_technical_specification",
            source_reference=(
                "rightpdf_“三峡氢舟1号”动力系统系统技术规格书 - "
                "V1_word2pdf.pdf"
            ),
        )
