"""Signed SOC-equivalent hydrogen evaluation for complete independent voyages.

This evaluator never changes physical states, interval ledgers or rewards.
The FC average efficiency is an explicit experiment parameter, not a fitted
quantity or a default inferred from a particular controller trajectory.
"""

from dataclasses import dataclass, field
from math import fsum, isfinite
from numbers import Real

from v2.economics import HYDROGEN_PRICE_CNY_PER_KG, RawCnyIntervalLedger
from v2.models.fuel_cell_efficiency import (
    FC_DATA_AXIS_TRANSFORM, FC_DATA_EFFICIENCY_TRANSFORM, FC_DATA_RANGE,
    FC_DATA_SHEET, FC_DATA_WORKBOOK_PATH, FC_DATA_WORKBOOK_SHA256,
    LHV_H2_KWH_PER_KG, formal_fuel_cell_efficiency_map,
)


EVALUATION_VERSION = 'zhou_soc_equivalent_hydrogen_v1'
BATTERY_NOMINAL_ENERGY_KWH = 624.0


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    result = float(value)
    if not isfinite(result):
        raise ValueError(f'{name} must be a finite real number')
    return result


@dataclass(frozen=True)
class ZhouEvaluationConfig:
    fc_average_efficiency: float
    _curve_reference: tuple[tuple[float, ...], tuple[float, ...]] | None = field(
        default=None, init=False, repr=False)

    def __post_init__(self):
        eta = _finite_real(self.fc_average_efficiency, 'fc_average_efficiency')
        if not 0.0 < eta <= 1.0:
            raise ValueError('fc_average_efficiency must satisfy 0 < eta <= 1')
        object.__setattr__(self, 'fc_average_efficiency', eta)

    @classmethod
    def from_formal_curve(cls):
        """Freeze the equal-weight mean over the sixty positive FC actions."""
        curve = formal_fuel_cell_efficiency_map()
        powers = tuple(float(power) for power in range(10, 601, 10))
        efficiencies = tuple(_finite_real(curve.eta(power), 'formal curve efficiency')
                             for power in powers)
        if any(not 0.0 < eta <= 1.0 for eta in efficiencies):
            raise ValueError('each positive-action formal curve efficiency must lie in (0, 1]')
        result = cls(fc_average_efficiency=fsum(efficiencies) / 60)
        object.__setattr__(result, '_curve_reference', (powers, efficiencies))
        return result

    def identity(self) -> dict[str, object]:
        """Bind the formula and every conversion parameter to run metadata."""
        identity = {
            'evaluation_version': EVALUATION_VERSION,
            'formula': ('m_H2_equivalent = m_H2_actual + '
                        '(SOC_initial - SOC_terminal) * battery_nominal_energy_kwh / '
                        '(fc_average_efficiency * hydrogen_lhv_kwh_per_kg); '
                        'total_cost_cny = m_H2_equivalent * hydrogen_price_cny_per_kg '
                        '+ original_ledger_fc_degradation_cost_cny '
                        '+ original_ledger_battery_degradation_cost_cny'),
            'soc_initial': 0.6,
            'fc_average_efficiency': self.fc_average_efficiency,
            'fc_average_efficiency_source': 'explicit run parameter; no default or trajectory fitting',
            'battery_nominal_energy_kwh': BATTERY_NOMINAL_ENERGY_KWH,
            'battery_energy_source': 'existing formal plant capacity',
            'hydrogen_lhv_kwh_per_kg': LHV_H2_KWH_PER_KG,
            'hydrogen_lhv_source': 'existing formal fuel-cell model: 120 MJ/kg',
            'hydrogen_price_cny_per_kg': HYDROGEN_PRICE_CNY_PER_KG,
            'hydrogen_price_source': 'existing formal price catalog',
            'soc_correction': 'signed; no clipping; no additional battery efficiency factor',
            'degradation_costs': 'unchanged original formal interval ledger, including EOL caps',
            'evaluation_scope': 'complete independent voyages only; never training reward',
            'raw_load_handling': 'retain raw load; execute existing normalize_onboard_load_kw deadband',
            'terminal_fc_accounting': 'no synthetic shutdown action; final executed FC state retained',
            'fc_start_stop_accounting': ('existing model: 0-to-positive charges one startup cycle; '
                                         'positive-to-0 does not add another complete startup cycle'),
            'actual_hydrogen_method': 'original instantaneous formal fuel-cell efficiency curve',
            'fc_curve_source': {
                'workbook_path': FC_DATA_WORKBOOK_PATH,
                'workbook_sha256': FC_DATA_WORKBOOK_SHA256,
                'sheet': FC_DATA_SHEET, 'range': FC_DATA_RANGE,
                'power_axis_mapping': FC_DATA_AXIS_TRANSFORM,
                'efficiency_mapping': FC_DATA_EFFICIENCY_TRANSFORM,
                'interpolation': 'existing formal shape-preserving PCHIP efficiency map',
            },
        }
        if self._curve_reference is not None:
            powers, efficiencies = self._curve_reference
            identity.update(
                fc_average_efficiency_source='frozen formal-curve arithmetic mean; project comparison parameter',
                fc_reference_method='math.fsum(eta(P) for P in range(10, 601, 10)) / 60',
                fc_reference_count=60,
                fc_reference_power_kw=list(powers),
                fc_reference_efficiencies=list(efficiencies),
                fc_reference_excludes_zero_power=True,
                fc_reference_is_vessel_calibration=False,
            )
        return identity


@dataclass(frozen=True)
class ZhouEconomicEvaluation:
    actual_hydrogen_mass_kg: float
    soc_equivalent_hydrogen_mass_kg: float
    equivalent_hydrogen_mass_kg: float
    equivalent_hydrogen_cost_cny: float
    fuel_cell_degradation_cost_cny: float
    battery_degradation_cost_cny: float
    total_cost_cny: float
    soc_initial: float
    soc_terminal: float
    fc_average_efficiency: float
    battery_nominal_energy_kwh: float = BATTERY_NOMINAL_ENERGY_KWH
    hydrogen_lhv_kwh_per_kg: float = LHV_H2_KWH_PER_KG
    hydrogen_price_cny_per_kg: float = HYDROGEN_PRICE_CNY_PER_KG
    evaluation_version: str = EVALUATION_VERSION


def evaluate_zhou_economics(
    ledger: RawCnyIntervalLedger, *, soc_initial: float, soc_terminal: float,
    config: ZhouEvaluationConfig,
) -> ZhouEconomicEvaluation:
    """Evaluate the original complete ledger with exactly one SOC correction."""
    if type(ledger) is not RawCnyIntervalLedger:
        raise TypeError('independent economics requires the original formal interval ledger')
    if type(config) is not ZhouEvaluationConfig:
        raise TypeError('independent economics requires an explicit ZhouEvaluationConfig')
    initial = _finite_real(soc_initial, 'soc_initial')
    terminal = _finite_real(soc_terminal, 'soc_terminal')
    if initial != 0.6 or not 0.2 <= terminal <= 0.8:
        raise ValueError('independent evaluation requires initial SOC 0.6 and a feasible terminal SOC')
    if ledger.shore_cost_cny != 0.0:
        raise ValueError('independent evaluation cannot include shore settlement costs')
    actual_mass = ledger.h2_cost_cny / HYDROGEN_PRICE_CNY_PER_KG
    correction = ((initial - terminal) * BATTERY_NOMINAL_ENERGY_KWH
                  / (config.fc_average_efficiency * LHV_H2_KWH_PER_KG))
    equivalent_mass = fsum((actual_mass, correction))
    hydrogen_cost = equivalent_mass * HYDROGEN_PRICE_CNY_PER_KG
    total = fsum((hydrogen_cost, ledger.fuel_cell_degradation_cost_cny,
                  ledger.battery_degradation_cost_cny))
    if not all(isfinite(value) for value in (correction, equivalent_mass, hydrogen_cost, total)):
        raise ValueError('independent economic evaluation must remain finite')
    return ZhouEconomicEvaluation(
        actual_mass, correction, equivalent_mass, hydrogen_cost,
        ledger.fuel_cell_degradation_cost_cny, ledger.battery_degradation_cost_cny,
        total, initial, terminal, config.fc_average_efficiency,
    )
