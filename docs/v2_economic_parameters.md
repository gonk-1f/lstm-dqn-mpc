# v2 operating-state and economic parameters

## Status

`REWARD_VERSION` is `macro_interval_real_economic_cost_v1`.

Formal training remains **NO-GO**. The following blockers are intentional:

- the candidate DQN state has not passed Train-only distribution, correlation,
  redundancy, and sensitivity audits;
- the aggregate fuel-cell lifetime normalization is not verified;
- the battery lifetime-throughput normalization is not verified; and
- no verified shore-converter efficiency is available for modeled terminal
  recharge.

Passing unit tests confirms formulas and gates. It does not promote any of
these unresolved quantities to formal calibration.

## Candidate operating state

The candidate state has nine semantic groups but ten flattened scalars. The
fuel-cell group contains two scalars: current and previous fuel-cell power.

| Group | Flattened feature | Definition |
|---|---|---|
| SOC | `soc` | current SOC fraction |
| Fuel-cell power | `fuel_cell_power_fraction` | current `P_fc / P_fc_rated` |
| Fuel-cell power | `previous_fuel_cell_power_fraction` | previous causal `P_fc / P_fc_rated` |
| Battery power | `battery_power_fraction` | current `P_batt / battery_power_scale` |
| Load | `load_power_fraction` | current `P_load / load_power_scale` |
| Recent mean load | `recent_load_mean_fraction` | arithmetic window mean divided by `load_power_scale` |
| Recent load standard deviation | `recent_load_population_std_fraction` | population standard deviation (`ddof=0`) divided by `load_power_scale` |
| Recent load trend | `recent_load_window_trend_fraction` | least-squares slope in kW/s times `window_seconds`, divided by `load_power_scale` |
| Causal base load | `causal_base_load_fraction` | current causal base load divided by its explicit scale |
| Recent SOC change | `recent_delta_soc` | current SOC minus the oldest SOC in the window |

`OperatingHistorySample` records are frozen, finite physical records with
timestamps in seconds. The input history is an immutable tuple with strictly
increasing timestamps. The window is the inclusive physical-time interval
`[current_time_seconds - window_seconds, current_time_seconds]`; it is never a
last-N-samples window. Minutes must be converted explicitly to seconds by the
caller. Records after `current_time_seconds` are never used. The builder
requires an exact current-time sample and at least two samples within the
window, so it cannot invent either previous fuel-cell power or a trend.

All power denominators are explicit, finite, and positive. SOC and delta-SOC
remain fractions. `CANDIDATE_STATE_STATUS` is `NO-GO` until the stated
Train-only audits are completed without using Validation or Test data.

## Source-backed price scenario

The fixed values are:

| Quantity | Value | Source and role |
|---|---:|---|
| Hydrogen | 35 CNY/kg | DOI `10.3390/jmse13010034`, unit-price source only |
| Fuel-cell equipment | 3500 CNY/kW | DOI `10.3390/jmse13010034`, unit-price source only |
| Battery equipment | 2000 CNY/kWh | DOI `10.3390/jmse13010034`, unit-price source only |
| Shore electricity | 1.10 CNY/kWh | DOI `10.11930/j.issn.1004-9649.202507065`, Table 3 |

The first DOI is not the source for a 600 kW fuel-cell rating or a 624 kWh
battery capacity. Those plant values must retain their own provenance.

The shore price is a user-approved **peak-tariff scenario**, not a measured
actual wharf tariff. `ShoreEnergyClassification` distinguishes `MEASURED` from
`MODELED`; the code never infers that a channel is measured. Missing prices
raise an error rather than becoming zero.

## Terminal recharge

The episode-specific target is exactly `episode_initial_soc`:

```text
E_battery_needed_kWh = max(0, episode_initial_soc - episode_end_soc)
                       * battery_capacity_kWh

E_grid_kWh = E_battery_needed_kWh / eta_chg / eta_shore_converter
```

`eta_chg` must come from the exact Task 3 `BatteryEfficiency` calibration and
therefore equals 0.95. There is no approved `eta_shore_converter`. The formal
`terminal_recharge_grid_energy` boundary is consequently NO-GO. The distinctly
named `terminal_recharge_grid_energy_unverified` function is pure arithmetic
for synthetic tests only and requires the converter efficiency explicitly.
Its modeled output is rejected by the formal interval-ledger boundary.

## Raw-CNY interval ledger and reward

`RawCnyIntervalLedger` contains exactly four non-negative, finite components:

```text
C_total = C_H2 + C_FC_deg + C_Batt_deg + C_shore_if_incurred
reward_cny = -C_total
```

There are no `0.3/0.4/0.3` coefficients and no other artificial component
weights. Raw CNY components are retained for logging.

Fuel-cell degradation cost may only use a verified relative life loss in
`[0, 1]` multiplied by `3500 CNY/kW * rated_kW`. Battery degradation cost may
only use a verified relative life loss in `[0, 1]` multiplied by
`2000 CNY/kWh * capacity_kWh`. The current Task 4 quantities—raw microvolts and
raw or stress-weighted ampere-hours—are not relative life fractions.
`build_formal_interval_ledger` therefore delegates to the Task 4 formal
normalization boundaries, which currently fail closed. The raw ledger class is
an immutable accounting value, not evidence that its degradation components
have passed formal normalization.

Reward scaling has no default `C_ref`. `calibrate_reward_scale` computes a
positive arithmetic mean from an immutable tuple of Train raw-CNY interval
costs and requires an exact Task 6 `DatasetProvenance` whose split is
`DataSplit.TRAIN`. Validation, Test, and Unknown provenance are rejected. The
resulting `RewardScaleCalibration` is tamper-evident, and scaled reward is
`reward_cny / scale_cny`.
