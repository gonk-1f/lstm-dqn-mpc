# Economic Model Closure Design

## Scope

Close the fuel-cell, battery, and terminal-shore economic-model boundaries that
remain after the accepted Train-only objective-scale audit. This change freezes
explicit modeling assumptions and their provenance; it does not run dataset
construction, state or action screening, sensitivity studies, DQN training, or
formal training.

Formal training remains **NO-GO** until every independent preflight blocker is
closed. Passing the tests in this change verifies code contracts only; it does
not turn literature-calibrated assumptions into vessel measurements.

## Design principles

- Keep raw physical accumulators separate from economic life fractions.
- Expose an unclipped diagnostic life fraction and a clipped economic fraction.
- Report end of life explicitly instead of hiding it in a clipped value.
- Charge replacement cost at most once per modeled component lifetime.
- Preserve exact units at every boundary; ampere-seconds and ampere-hours must
  never be interchanged implicitly.
- Bind every promoted parameter to an explicit source role and applicability
  statement. A source for a formula is not automatically a source for a vessel
  parameter.
- Reject forged, subclassed, incomplete, non-finite, Boolean, or text-valued
  calibration payloads at formal boundaries.

## Fuel-cell normalization

The fuel-cell degradation model remains one aggregate-equivalent plant model.
It does not instantiate eight independent degradation states and does not
multiply or divide raw voltage loss or replacement cost by stack count.

The approved end-of-life denominator is:

```text
FC_EOL_VOLTAGE_LOSS_UV = 70_000.0

D_fc_raw_before
    = cumulative_voltage_loss_before_uv / FC_EOL_VOLTAGE_LOSS_UV
D_fc_raw_after
    = cumulative_voltage_loss_after_uv / FC_EOL_VOLTAGE_LOSS_UV

D_fc_econ_before = clip(D_fc_raw_before, 0, 1)
D_fc_econ_after = clip(D_fc_raw_after, 0, 1)

delta_D_fc_econ = D_fc_econ_after - D_fc_econ_before
delta_D_fc_econ >= 0

C_fc_deg_interval
    = delta_D_fc_econ * 3500 CNY/kW * 600 kW

fc_eol_reached = D_fc_raw_after >= 1
```

The cumulative raw and economic fractions remain diagnostic state. Only the
non-negative clipped difference enters the current 30-second interval ledger.
An interval that crosses EOL charges only the remaining fraction up to one; an
interval starting at or above EOL charges zero even while raw voltage loss
continues to accumulate. The formal boundary rejects a decreasing cumulative
input rather than creating a negative cost.

The 600 kW rating is the aggregate plant rating. Without a modeled replacement
and state reset, the sum of all interval fuel-cell degradation charges is
bounded by `3500 * 600 = 2_100_000 CNY`, not eight times that amount. A ledger
must never multiply the cumulative `D_fc_econ_after` by replacement price on
every interval.

The existing voltage-loss coefficients retain their current source roles.
The aggregate-equivalent representation is a modeling abstraction for the EMS,
not a claim that eight physical stacks age identically. The observed 243 V
system value is retained only as measured sanity provenance and is not used as
the lifetime denominator. The formal fuel-cell normalization status becomes
`VERIFIED` for this explicitly defined aggregate-equivalent model.

## Battery plant parameters and units

Freeze the following plant values and exact derivations:

```text
BATTERY_ENERGY_CAPACITY_KWH = 624.0
BATTERY_NOMINAL_VOLTAGE_V = 432.0

BATTERY_NOMINAL_CHARGE_CAPACITY_AH
    = 624_000 Wh / 432 V
    = 1444.4444444444443 Ah

BATTERY_CURRENT_REF_1C_A
    = BATTERY_NOMINAL_CHARGE_CAPACITY_AH
    = 1444.4444444444443 A
```

These values define energy capacity, nominal charge capacity, and the current
reference used by the stress model. They do not by themselves constitute a
measured lifetime calibration.

The existing per-step degradation equation remains:

```text
F(SOC) = 1 + 3.25 * (1 - SOC)^2

G(I) = 1 + 0.45 * I / I_ref       when I >= 0
G(I) = 1 + 0.55 * abs(I) / I_ref  when I < 0

Q_raw_Ah = abs(I_A) * dt_seconds / 3600
Q_weighted_Ah = Q_raw_Ah * F(SOC) * G(I)
```

Positive current continues to mean discharge. The cumulative degradation
account stores ampere-hours. Any input accumulator expressed in ampere-seconds
must be divided by 3600 before this boundary, or normalized by an explicitly
ampere-second denominator; mixed-unit calculations are forbidden.

## Battery lifetime-throughput assumption

The approved lifetime-throughput factor is:

```text
BATTERY_LIFETIME_THROUGHPUT_FACTOR = 15_000.0

Q_lifetime_Ah
    = BATTERY_LIFETIME_THROUGHPUT_FACTOR
      * BATTERY_NOMINAL_CHARGE_CAPACITY_AH
    = 21_666_666.666666664 Ah
```

Its provenance classification is exactly
`literature-based lifetime-throughput modeling assumption`. The implementation
and documentation must explicitly state that it is **not**:

- a measured lifetime for Three Gorges Hydrogen Boat 1;
- a manufacturer specification; or
- a Yang project configuration parameter.

The current source audit confirms the weighted-Ah stress formulation through
the Kwon/Zhou literature chain, but has not established a defensible direct
primary source for the numeric factor 15000 that is applicable to this vessel
battery. Until such a source is located, the factor must be labeled
`secondary literature basis`; no direct-source DOI may be invented.

The implementation retains a future sensitivity hook containing exactly
`(10_000.0, 15_000.0, 20_000.0)`. This change does not execute sensitivity
analysis and the hook must not choose a value dynamically.

Battery life and cost are:

```text
D_batt_raw_before
    = cumulative_weighted_Ah_before / Q_lifetime_Ah
D_batt_raw_after
    = cumulative_weighted_Ah_after / Q_lifetime_Ah

D_batt_econ_before = clip(D_batt_raw_before, 0, 1)
D_batt_econ_after = clip(D_batt_raw_after, 0, 1)

delta_D_batt_econ = D_batt_econ_after - D_batt_econ_before
delta_D_batt_econ >= 0

C_batt_deg_interval
    = delta_D_batt_econ * 2000 CNY/kWh * 624 kWh

battery_eol_reached = D_batt_raw_after >= 1
```

Only the non-negative clipped difference enters the current 30-second interval
ledger. Crossing EOL charges only the uncharged remainder, while further raw
weighted-Ah accumulation after EOL has zero economic increment. Decreasing
cumulative weighted Ah is invalid. Without a modeled replacement and state
reset, the sum of all interval battery degradation charges is bounded by
`2000 * 624 = 1_248_000 CNY`; cumulative economic loss must not be charged
again on later intervals.

The battery lifetime-normalization status becomes
`PROVISIONAL / LITERATURE-CALIBRATED`, never `VERIFIED` or measured.

## Terminal shore recharge

Terminal recharge targets the episode's initial SOC and applies one aggregate
charge-path efficiency:

```text
E_battery_needed_kWh
    = max(0, episode_initial_soc - episode_end_soc) * 624 kWh

E_grid_kWh = E_battery_needed_kWh / 0.95
C_shore = E_grid_kWh * 1.10 CNY/kWh
```

The approved `0.95` is the single aggregate charging efficiency at this
economic boundary. The implementation removes the extra shore-converter gate
and does not divide again by battery charge efficiency. The `1.10 CNY/kWh`
price remains a literature-backed peak-tariff scenario, not a measured wharf
contract price. Measured and modeled shore-energy records remain distinct.

## Formal interval ledger

The raw-CNY interval ledger continues to contain exactly four unweighted
components:

```text
C_total = C_H2 + C_fc_deg + C_batt_deg + C_shore_if_incurred
reward_cny = -C_total
```

Each physical MPC execution returns one ledger for its actual 30-second
interval. That ledger accepts only:

- hydrogen consumed in the current interval;
- incremental fuel-cell economic lifetime cost in the current interval;
- incremental battery economic lifetime cost in the current interval; and
- shore energy cost incurred in the current interval or terminal recharge.

The formal ledger accepts the exact approved fuel-cell normalization and the
exact provisional battery normalization. It derives degradation charges from
before/after cumulative raw states and uses only the clipped cumulative-fraction
differences. Cumulative raw degradation, cumulative raw life fraction, clipped
cumulative economic fraction, and EOL flags remain available as diagnostic
state but are not ledger cost inputs.

`MultiRateWeightEnvironment` sums the four components across its `M` physical
MPC executions. Therefore its macro ledger is economically correct only when
each input ledger contains interval increments. Passing cumulative degradation
cost into each step would double count prior lifetime consumption and is a
contract violation. No `0.3/0.4/0.3` weights are added.

## Preflight and status reporting

The preflight report changes only the following entries:

- fuel-cell degradation normalization: `VERIFIED` for the approved
  aggregate-equivalent 70,000 microvolt model;
- battery lifetime-throughput normalization:
  `PROVISIONAL / LITERATURE-CALIBRATED` with secondary-literature provenance;
- shore charging efficiency: `VERIFIED` as one aggregate 0.95 factor; and
- `Ts_mpc = 30 s`: frozen as the project control interval.

`N = 5`, `M = 5`, and `tau_LPF = 90 s` remain provisional. Dataset readiness,
the final DQN state, and the final action catalog remain unresolved or NO-GO as
already recorded. Therefore `FORMAL_TRAINING` remains `NO-GO`.

## MPC time-index clarification

Only comments and documentation are added where needed to make the existing
time contract explicit: `k` indexes 30-second MPC samples, the horizon contains
`N = 5` such samples, and the DQN switching interval uses `M = 5` samples in the
current provisional baseline. Control behavior and objective equations are not
changed.

## Validation strategy

Implementation follows test-first red-green cycles covering:

- fuel-cell raw/economic/EOL behavior below, at, and above 70,000 microvolts;
- fuel-cell incremental cost before EOL, across EOL, and after EOL;
- aggregate 600 kW fuel-cell replacement cost with no eightfold charge;
- exact battery Ah and 1C derivations from 624 kWh and 432 V;
- ampere-hour integration including the mandatory `/3600` conversion;
- battery raw/economic/EOL behavior and the exact lifetime denominator;
- battery incremental cost before EOL, across EOL, and after EOL;
- immutable sensitivity candidates with no sensitivity execution;
- rejection of provenance that misclassifies 15000 as measured, manufacturer,
  or project configuration data;
- terminal recharge using exactly one 0.95 factor;
- formal ledger integration and maximum replacement-cost bounds;
- five-step macro-ledger summation without cumulative-cost double counting;
- preflight statuses and continued formal-training NO-GO; and
- regression protection for existing MPC objectives and control behavior.

The final verification set includes the focused degradation/economics tests,
all `test_v2_*.py` tests, solver smoke tests, compile/import checks, and
`git diff --check`.
