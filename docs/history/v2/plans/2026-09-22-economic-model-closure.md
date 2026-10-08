# Economic Model Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert cumulative fuel-cell and battery degradation diagnostics into non-duplicating 30-second interval costs, close the approved shore-energy boundary, and report the resulting calibration statuses without enabling formal training.

**Architecture:** Fuel-cell and battery modules own cumulative life-state and clipped interval-increment calculations. `v2.economics` multiplies only interval economic fractions by fixed aggregate replacement costs and places those increments in `RawCnyIntervalLedger`; `MultiRateWeightEnvironment` remains a pure sum of interval ledgers. Preflight reports the evidence class of each calibration separately from overall training readiness.

**Tech Stack:** Python 3 dataclasses and enums, `unittest`, existing v2 economics and multirate environment modules.

---

### Task 1: Prove the complete contract is red

**Files:**
- Create: `tests/test_v2_economic_interval_semantics.py`

- [ ] **Step 1: Add focused fuel-cell, battery, shore, macro-ledger, and preflight tests**

Define one test per observable contract. Use an assertion-based `_require_attr`
helper so an absent production symbol is a test failure with a precise message,
not an import error. Cover these exact cases:

```python
def _require_attr(test: unittest.TestCase, module: object, name: str) -> object:
    test.assertTrue(hasattr(module, name), f"missing contract: {name}")
    return getattr(module, name)
```

Add these exact test methods and assertion datasets:

- `FuelCellIntervalCostTests.test_cumulative_life_state_boundaries` checks
  `(loss, raw, econ, eol)` values `(0,0,0,False)`,
  `(35000,0.5,0.5,False)`, `(70000,1,1,True)`, and
  `(80000,8/7,1,True)`.
- `FuelCellIntervalCostTests.test_crossing_eol_charges_only_unconsumed_fraction`
  checks `69990 -> 70010` gives delta `10/70000` and cost `300 CNY`.
- `FuelCellIntervalCostTests.test_post_eol_increment_is_zero_and_cost_is_not_multiplied_by_eight`
  checks `80000 -> 90000` costs zero and `0 -> 70000` costs exactly
  `3500 * 600`, not `3500 * 600 * 8`.
- `BatteryIntervalCostTests.test_plant_capacity_current_and_ampere_hour_units`
  checks `624000/432`, the identical 1C current, one hour of raw Ah, and one
  second divided by 3600.
- `BatteryIntervalCostTests.test_lifetime_denominator_and_secondary_provenance`
  checks `15000 * 624000/432` and rejects the three forbidden provenance roles.
- `BatteryIntervalCostTests.test_interval_increment_before_across_and_after_eol`
  checks an ordinary quarter-life increment, `0.999 -> 1.001` clipping, and a
  post-EOL zero increment while the raw fraction continues above one.
- `ShoreAndMacroLedgerTests.test_terminal_recharge_uses_one_aggregate_efficiency`
  checks `0.10 * 624 / 0.95` and the resulting `* 1.10` cost.
- `ShoreAndMacroLedgerTests.test_five_interval_ledgers_sum_increments_without_double_counting`
  injects five exact ledgers, checks each summed component, and asserts the FC
  macro component is the sum of five increments rather than the sum of their
  cumulative prefixes.
- `EconomicClosurePreflightTests.test_preflight_reports_evidence_classes_and_remains_no_go`
  checks FC, battery, shore, Ts, N, M, tau, and overall training statuses.

- [ ] **Step 2: Run the focused module and verify the red state**

Run:

```powershell
python -m unittest tests.test_v2_economic_interval_semantics -v
```

Expected: the macro-ledger characterization may already pass because the
environment correctly sums interval ledgers. All new normalization, interval
cost, shore-signature, and preflight-status contracts fail for explicit missing
symbols, obsolete signatures, or obsolete statuses. Do not edit production
code in this step.

### Task 2: Implement fuel-cell cumulative and interval semantics

**Files:**
- Modify: `src/v2/models/fuel_cell_degradation.py`
- Test: `tests/test_v2_economic_interval_semantics.py`
- Test: `tests/test_v2_degradation_models.py`

- [ ] **Step 1: Add exact aggregate model constants and result records**

```python
FC_EOL_VOLTAGE_LOSS_UV = 70_000.0
FC_LIFETIME_NORMALIZATION_STATUS = "VERIFIED"
FC_LIFETIME_EVIDENCE_CLASS = "literature/model verified; not vessel-measured"


@dataclass(frozen=True)
class FuelCellLifeState:
    cumulative_voltage_loss_uv: float
    raw_life_fraction: float
    economic_life_fraction: float
    eol_reached: bool


@dataclass(frozen=True)
class FuelCellLifeIncrement:
    before: FuelCellLifeState
    after: FuelCellLifeState
    delta_economic_fraction: float
```

- [ ] **Step 2: Implement cumulative state and monotonic clipped increment**

```python
def fuel_cell_life_state(cumulative_voltage_loss_uv: float) -> FuelCellLifeState:
    loss = _strict_scalar(cumulative_voltage_loss_uv, "cumulative_voltage_loss_uv")
    if loss < 0.0:
        raise ValueError("cumulative_voltage_loss_uv must be non-negative")
    raw = loss / FC_EOL_VOLTAGE_LOSS_UV
    econ = min(max(raw, 0.0), 1.0)
    return FuelCellLifeState(loss, raw, econ, raw >= 1.0)


def formal_fuel_cell_interval_life_loss(
    cumulative_voltage_loss_before_uv: float,
    cumulative_voltage_loss_after_uv: float,
) -> FuelCellLifeIncrement:
    before = fuel_cell_life_state(cumulative_voltage_loss_before_uv)
    after = fuel_cell_life_state(cumulative_voltage_loss_after_uv)
    if after.cumulative_voltage_loss_uv < before.cumulative_voltage_loss_uv:
        raise ValueError("cumulative fuel-cell voltage loss must not decrease")
    delta = after.economic_life_fraction - before.economic_life_fraction
    return FuelCellLifeIncrement(before, after, delta)
```

- [ ] **Step 3: Convert formal fuel-cell cost to an interval boundary**

```python
def formal_fuel_cell_degradation_cost_cny(
    cumulative_voltage_loss_before_uv: float,
    cumulative_voltage_loss_after_uv: float,
    *,
    replacement_cost_cny: float,
) -> float:
    increment = formal_fuel_cell_interval_life_loss(
        cumulative_voltage_loss_before_uv,
        cumulative_voltage_loss_after_uv,
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    return increment.delta_economic_fraction * price
```

- [ ] **Step 4: Run fuel-cell focused and legacy degradation tests**

```powershell
python -m unittest tests.test_v2_economic_interval_semantics.FuelCellIntervalCostTests tests.test_v2_degradation_models.FuelCellDegradationTests -v
```

Expected: all listed tests pass after obsolete fail-closed assertions are
updated to the approved aggregate-equivalent contract.

### Task 3: Implement battery plant, provenance, and interval semantics

**Files:**
- Modify: `src/v2/models/battery_degradation.py`
- Test: `tests/test_v2_economic_interval_semantics.py`
- Test: `tests/test_v2_degradation_models.py`

- [ ] **Step 1: Add exact plant and literature-calibration constants**

```python
BATTERY_ENERGY_CAPACITY_KWH = 624.0
BATTERY_NOMINAL_VOLTAGE_V = 432.0
BATTERY_NOMINAL_CHARGE_CAPACITY_AH = 624_000.0 / 432.0
BATTERY_CURRENT_REF_1C_A = BATTERY_NOMINAL_CHARGE_CAPACITY_AH
BATTERY_LIFETIME_THROUGHPUT_FACTOR = 15_000.0
BATTERY_LIFETIME_Q_AH = (
    BATTERY_LIFETIME_THROUGHPUT_FACTOR
    * BATTERY_NOMINAL_CHARGE_CAPACITY_AH
)
BATTERY_LIFETIME_SENSITIVITY_FACTORS = (10_000.0, 15_000.0, 20_000.0)
BATTERY_LIFETIME_NORMALIZATION_STATUS = "PROVISIONAL / LITERATURE-CALIBRATED"
BATTERY_LIFETIME_PROVENANCE_CLASSIFICATION = (
    "literature-based lifetime-throughput modeling assumption"
)
BATTERY_LIFETIME_EVIDENCE_BASIS = "secondary literature basis"
```

- [ ] **Step 2: Replace the unresolved record with an exact provenance record**

```python
@dataclass(frozen=True)
class BatteryLifetimeNormalization:
    q_lifetime_ah: float
    throughput_factor: float
    nominal_charge_capacity_ah: float
    provenance_classification: str
    evidence_basis: str
    applicability: str

    def require_literature_calibrated(self) -> "BatteryLifetimeNormalization":
        return self


def formal_battery_lifetime_normalization() -> BatteryLifetimeNormalization:
    return BatteryLifetimeNormalization(
        BATTERY_LIFETIME_Q_AH,
        BATTERY_LIFETIME_THROUGHPUT_FACTOR,
        BATTERY_NOMINAL_CHARGE_CAPACITY_AH,
        BATTERY_LIFETIME_PROVENANCE_CLASSIFICATION,
        BATTERY_LIFETIME_EVIDENCE_BASIS,
        "provisional project battery lifetime normalization; not vessel measured",
    )
```

`BatteryLifetimeNormalization.__post_init__` must require these exact numeric
derivations and exact provenance strings. Values labeled measured,
manufacturer specification, or Yang project parameter raise `ValueError`.

- [ ] **Step 3: Add cumulative state and monotonic clipped interval increment**

```python
@dataclass(frozen=True)
class BatteryLifeState:
    cumulative_weighted_ah: float
    raw_life_fraction: float
    economic_life_fraction: float
    eol_reached: bool


@dataclass(frozen=True)
class BatteryLifeIncrement:
    before: BatteryLifeState
    after: BatteryLifeState
    delta_economic_fraction: float


def formal_battery_interval_life_loss(
    cumulative_weighted_ah_before: float,
    cumulative_weighted_ah_after: float,
    *,
    normalization: BatteryLifetimeNormalization,
) -> BatteryLifeIncrement:
    if type(normalization) is not BatteryLifetimeNormalization:
        raise TypeError("normalization must use the exact provenance-bearing type")
    checked = normalization.require_literature_calibrated()
    before_ah = _strict_scalar(
        cumulative_weighted_ah_before,
        "cumulative_weighted_ah_before",
    )
    after_ah = _strict_scalar(
        cumulative_weighted_ah_after,
        "cumulative_weighted_ah_after",
    )
    if before_ah < 0.0 or after_ah < before_ah:
        raise ValueError("cumulative weighted Ah must be non-negative and monotonic")
    before_raw = before_ah / checked.q_lifetime_ah
    after_raw = after_ah / checked.q_lifetime_ah
    before = BatteryLifeState(
        before_ah,
        before_raw,
        min(before_raw, 1.0),
        before_raw >= 1.0,
    )
    after = BatteryLifeState(
        after_ah,
        after_raw,
        min(after_raw, 1.0),
        after_raw >= 1.0,
    )
    return BatteryLifeIncrement(
        before,
        after,
        after.economic_life_fraction - before.economic_life_fraction,
    )
```

- [ ] **Step 4: Convert formal battery cost to an interval boundary**

```python
def formal_battery_degradation_cost_cny(
    cumulative_weighted_ah_before: float,
    cumulative_weighted_ah_after: float,
    *,
    replacement_cost_cny: float,
    normalization: BatteryLifetimeNormalization,
) -> float:
    increment = formal_battery_interval_life_loss(
        cumulative_weighted_ah_before,
        cumulative_weighted_ah_after,
        normalization=normalization,
    )
    price = _strict_scalar(replacement_cost_cny, "replacement_cost_cny")
    if price < 0.0:
        raise ValueError("replacement_cost_cny must be non-negative")
    return increment.delta_economic_fraction * price
```

- [ ] **Step 5: Run battery focused and legacy degradation tests**

```powershell
python -m unittest tests.test_v2_economic_interval_semantics.BatteryIntervalCostTests tests.test_v2_degradation_models.BatteryDegradationTests -v
```

Expected: all listed tests pass, including explicit `/3600`, provenance
misclassification rejection, EOL crossing, and post-EOL zero cost.

### Task 4: Close shore and formal interval-ledger boundaries

**Files:**
- Modify: `src/v2/economics.py`
- Modify: `tests/test_v2_state_and_economics.py`
- Test: `tests/test_v2_economic_interval_semantics.py`

- [ ] **Step 1: Remove the obsolete second converter efficiency**

```python
SHORE_CHARGING_EFFICIENCY_STATUS = "VERIFIED"
SHORE_CHARGING_EFFICIENCY_EVIDENCE = (
    "literature-based aggregate assumption; not vessel-measured"
)


def terminal_recharge_grid_energy(
    *,
    episode_initial_soc: float,
    episode_end_soc: float,
    battery_capacity_kwh: float,
    battery_efficiency: BatteryEfficiency,
) -> ShoreEnergy:
    initial = _finite_scalar(episode_initial_soc, "episode_initial_soc")
    end = _finite_scalar(episode_end_soc, "episode_end_soc")
    capacity = _positive_scalar(battery_capacity_kwh, "battery_capacity_kwh")
    if not 0.0 <= initial <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("SOC values must lie in [0, 1]")
    efficiency = BatteryEfficiency.require_source_backed(battery_efficiency)
    battery_side_needed_kwh = max(0.0, initial - end) * capacity
    return ShoreEnergy(
        battery_side_needed_kwh / efficiency.eta_chg,
        ShoreEnergyClassification.MODELED,
    )
```

- [ ] **Step 2: Change the formal ledger to before/after cumulative inputs**

```python
def build_formal_interval_ledger(
    *,
    hydrogen_mass_kg: float,
    fuel_cell_cumulative_voltage_loss_before_uv: float,
    fuel_cell_cumulative_voltage_loss_after_uv: float,
    fuel_cell_rated_kw: float,
    battery_cumulative_weighted_ah_before: float,
    battery_cumulative_weighted_ah_after: float,
    battery_capacity_kwh: float,
    battery_normalization: BatteryLifetimeNormalization,
    shore_energy: ShoreEnergy | None,
    prices: EconomicPriceCatalog = FORMAL_PRICE_CATALOG,
) -> RawCnyIntervalLedger:
    catalog = _validate_prices(prices)
    fc_rated = _positive_scalar(fuel_cell_rated_kw, "fuel_cell_rated_kw")
    battery_capacity = _positive_scalar(
        battery_capacity_kwh,
        "battery_capacity_kwh",
    )
    if fc_rated != 600.0:
        raise ValueError("fuel_cell_rated_kw must equal the approved 600 kW rating")
    if battery_capacity != 624.0:
        raise ValueError("battery_capacity_kwh must equal the approved 624 kWh capacity")
    h2_cost = hydrogen_cost_cny(hydrogen_mass_kg, prices=catalog)
    fc_cost = formal_fuel_cell_degradation_cost_cny(
        fuel_cell_cumulative_voltage_loss_before_uv,
        fuel_cell_cumulative_voltage_loss_after_uv,
        replacement_cost_cny=fc_rated * catalog.fuel_cell_cny_per_kw,
    )
    battery_cost = formal_battery_degradation_cost_cny(
        battery_cumulative_weighted_ah_before,
        battery_cumulative_weighted_ah_after,
        replacement_cost_cny=battery_capacity * catalog.battery_cny_per_kwh,
        normalization=battery_normalization,
    )
    shore_cost = (
        0.0
        if shore_energy is None
        else shore_energy_cost_cny(shore_energy, prices=catalog)
    )
    return RawCnyIntervalLedger(h2_cost, fc_cost, battery_cost, shore_cost)
```

Fuel-cell cost is
`formal_fuel_cell_degradation_cost_cny(before, after,
replacement_cost_cny=3500 * 600)`. Battery cost is
`formal_battery_degradation_cost_cny(before, after,
replacement_cost_cny=2000 * 624, normalization=normalization)`.

- [ ] **Step 3: Run shore, ledger, and macro tests**

```powershell
python -m unittest tests.test_v2_economic_interval_semantics.ShoreAndMacroLedgerTests tests.test_v2_state_and_economics.EconomicCostTests tests.test_v2_multirate_env -v
```

Expected: terminal recharge is `62.4 / 0.95`, each ledger contains only interval
increments, and five ledgers sum without repeated cumulative charges.

### Task 5: Update preflight and time-index evidence

**Files:**
- Modify: `src/v2/preflight.py`
- Modify: `src/v2/control/nonlinear_mpc.py`
- Modify: `tests/test_v2_formal_preflight.py`
- Test: `tests/test_v2_economic_interval_semantics.py`

- [ ] **Step 1: Report the approved evidence classes**

Set `fc_degradation_normalization` to `VERIFIED`,
`battery_q_lifetime_normalization` to `PROVISIONAL`, add an explicit verified
`shore_charging_efficiency` check, and set `ts_mpc` to `VERIFIED`. Keep `n_mpc`,
`dqn_switch_steps`, and `tau_lpf` as `PROVISIONAL`; keep final state and action
catalog blockers unchanged so `formal_training` remains `NO-GO`.

```python
FormalCalibrationCheck(
    "battery_q_lifetime_normalization",
    CalibrationStatus.PROVISIONAL,
    "PROVISIONAL / LITERATURE-CALIBRATED; secondary literature basis; not vessel measured",
)
```

- [ ] **Step 2: Clarify existing MPC indices without changing equations**

Add comments stating that `k` indexes 30-second MPC samples, `N=5` is the
provisional horizon length, and the DQN environment currently holds weights for
`M=5` actual MPC executions. Do not change control calculations.

- [ ] **Step 3: Run focused and formal-preflight tests**

```powershell
python -m unittest tests.test_v2_economic_interval_semantics.EconomicClosurePreflightTests tests.test_v2_formal_preflight -v
```

Expected: evidence statuses match the specification and formal training remains
NO-GO.

### Task 6: Update documentation and verify the repository

**Files:**
- Modify: `docs/v2_fc_degradation_model.md`
- Modify: `docs/v2_battery_degradation_model.md`
- Modify: `docs/v2_economic_parameters.md`
- Modify: `docs/v2_preflight_report.md`
- Create: `docs/v2_remaining_training_work.md`

- [ ] **Step 1: Document interval-cost semantics and provenance limits**

Record the before/after equations, clipped deltas, replacement-charge bounds,
secondary-literature battery status, single 0.95 shore factor, and the explicit
statement that literature/model verification is not vessel measurement.

- [ ] **Step 2: Record remaining work without executing it**

List dataset build/split, state audit, action screening, sensitivity, and formal
training as future blocked work. Do not invoke any of them.

- [ ] **Step 3: Run full verification**

```powershell
python -m unittest discover -s tests -p 'test_v2_*.py'
python -m unittest tests.test_v2_nonlinear_mpc.NonlinearMPCTests.test_scipy_backend_smoke -v
python -m compileall -q src/v2 tests
git diff --check
```

Expected: zero failures, solver smoke passes, compile/import succeeds, and Git
reports no whitespace errors.

- [ ] **Step 4: Confirm prohibited work was not performed**

Inspect the diff and command history. There must be no dataset build/split,
action screening, DQN state audit, sensitivity run, or formal training output.
