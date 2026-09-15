# v2 Aggregate Fuel-Cell Degradation Model

## Version, scope, and evidence

The exact model version is `aggregate_four_condition_voltage_loss_v1`. It is
an aggregate fuel-cell voltage-loss accounting model. The numerical
coefficients are sourced independently from DOI
`10.1016/j.ijhydene.2024.02.349`:

| Condition | Coefficient | Unit |
|---|---:|---|
| Low-load runtime | 10.17 | microvolt/hour |
| High-load runtime | 11.74 | microvolt/hour |
| Transient | 0.0441 | microvolt per absolute delta-kW |
| Start/stop | 23.91 | microvolt per aggregate cycle |

DOI `10.3390/jmse13010034` supports the four-condition accounting structure;
it is not represented as the independent source of the numerical
coefficients. Keeping these source roles separate prevents a structural
reference from being mistaken for coefficient calibration.

This model treats the whole fuel-cell plant as one aggregate source. The
hysteresis tracker therefore estimates aggregate ON/OFF transitions. A counted
start is not a measurement or reconstruction of starts across the real eight
stacks.

## Raw microvolt accounting

For an executed interval of length `dt_hours`, the raw loss is

\[
\Delta V_{\mu V}=\Delta V_{low}+\Delta V_{high}
 +0.0441\left|P_t-P_{t-1}\right|+23.91N_{start/stop}.
\]

While the aggregate fuel cell is ON, exactly one runtime term applies:

\[
\Delta V_{runtime}=\begin{cases}
10.17\,\Delta t_{hours}, & 0\leq P_t<0.8P_{rated},\\
11.74\,\Delta t_{hours}, & 0.8P_{rated}\leq P_t\leq P_{rated}.
\end{cases}
\]

High load begins at exactly `0.8 * rated_power_kw`. An OFF interval accrues no
low- or high-runtime loss. The transient term is still the absolute change in
executed aggregate power; it is distinct from any MPC smoothing objective and
is not suppressed merely because the new ON/OFF state is OFF. All returned
components and cumulative values remain in microvolts.

Negative power, power above the explicit rated power, non-positive duration,
non-positive rated power, non-finite values, booleans, and numeric-looking text
are rejected.

## Aggregate ON/OFF hysteresis and dwell

The state tracker requires
`0 <= p_off_threshold_kw < p_on_threshold_kw <= rated_power_kw` and a positive
dwell time. Its transitions are inclusive at the thresholds:

- OFF becomes ON only after `power_kw >= p_on_threshold_kw` continuously for
  the full dwell; that transition counts one aggregate start.
- ON becomes OFF only after `power_kw <= p_off_threshold_kw` continuously for
  the full dwell.
- A sample inside the open hysteresis band holds the current state and resets
  any pending contrary dwell.

Power supplied to every update is checked against the explicit zero-to-rated
domain. The tracker reports per-update starts/stops and cumulative aggregate
counts.

## Lifetime normalization and cost gate

The stated relative normalization is

\[
D_{fc}=\frac{\Delta V}{0.1V_{init}}.
\]

Here `Delta V` and `V_init` must use the same voltage unit. The implementation's
explicitly synthetic helper converts input microvolts to volts before applying
the formula. However, neither a calibrated value for `V_init` nor whether it is
a cell, stack, or aggregate-system voltage is currently established. The
formal status is therefore exactly `NO-GO`; no formal default or formal factory
exists.

A proposed formal normalization record must carry the initial voltage, exact
voltage basis, source DOI, and system applicability. Even a complete-looking
record is rejected because no authoritative record has been approved. Bare
numeric normalization and subclasses are also rejected. Consequently, raw
microvolt loss cannot be multiplied by fuel-cell replacement price. The formal
relative-life and degradation-to-CNY entrypoints fail before any such
multiplication. This document does not invent an initial voltage, lifetime, or
equipment-cost conversion.
