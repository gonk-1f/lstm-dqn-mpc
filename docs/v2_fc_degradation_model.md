# v2 Fuel-Cell Degradation Model

## Version, scope, and evidence

The exact model version is `aggregate_four_condition_voltage_loss_v1`. The
four-condition structure is retained for the aggregate controller, but the
numerical voltage-loss coefficients are per-cell values whose transient term
uses the individual fuel-cell/reference-unit power reported in Table 3 of DOI
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

The coefficient evaluator therefore accepts only an explicitly named
source-compatible reference-unit power trace. It does not accept the 600 kW
aggregate plant trace. The plant-level hysteresis tracker separately estimates
aggregate ON/OFF transitions. A counted start is only an aggregate proxy; it
is not a measurement or reconstruction of starts across the real eight stacks.

## Raw microvolt accounting

For a source-compatible reference-unit executed interval of length `dt_hours`,
the raw single-cell voltage loss is

\[
\Delta V_{\mu V}=\Delta V_{low}+\Delta V_{high}
 +0.0441\left|P_{ref,t}-P_{ref,t-1}\right|+23.91N_{start/stop}.
\]

While the aggregate fuel cell is ON, exactly one runtime term applies:

\[
\Delta V_{runtime}=\begin{cases}
10.17\,\Delta t_{hours}, & 0\leq P_{ref,t}<0.8P_{ref,rated},\\
11.74\,\Delta t_{hours}, & 0.8P_{ref,rated}\leq P_{ref,t}\leq P_{ref,rated}.
\end{cases}
\]

High load begins at exactly `0.8 * reference_rated_power_kw`. An OFF interval
accrues no low- or high-runtime loss. The transient term is the absolute change
in executed source-compatible reference-unit power; it is distinct from any
MPC smoothing objective and is not suppressed merely because the new ON/OFF
state is OFF. All returned components and cumulative values remain in
microvolts of single-cell voltage loss.

The raw API is named `reference_unit_voltage_loss_step_uv`, and its power
arguments carry `reference_` names. Negative power, power above the explicit
reference-unit rated power, non-positive duration, non-positive rated power,
non-finite values, booleans, and numeric-looking text are rejected.
Result and cumulative-account construction also require finite derived runtime
and total loss, not merely finite individual components. Account updates
validate every prospective component plus the prospective runtime and total
before mutation, so cross-component overflow cannot partially update a ledger.

## Aggregate-power mapping gate

The repository has no sourced count, topology, or calibrated ratio that maps
the 600 kW aggregate command to the Table 3 reference-unit power. Multiplying or
dividing aggregate power by an assumed number of stacks, cells, or parallel
units would invent a calibration and can materially mis-scale the transient
term. Consequently:

- the former ambiguous `fc_voltage_loss_step_uv` API is not exported;
- `AggregateFcOnOffTracker` may consume aggregate power solely for ON/OFF
  hysteresis and dwell accounting; and
- `formal_aggregate_fc_voltage_loss_step_uv` requires a provenance-bearing
  mapping record, but every such record currently fails closed with mapping
  status `NO-GO`.

There is no formal aggregate-power-to-reference-unit conversion and no
unverified aggregate degradation proxy.

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

Here the cited basis is single-cell voltage: `Delta V` is a single-cell loss and
`V_init` must be a single-cell initial voltage in the same physical basis. The
implementation's explicitly synthetic helper converts input microvolts to
volts before applying the formula and requires the exact basis label
`single-cell voltage`; stack and aggregate-system bases are rejected. No
applicable numeric single-cell `V_init` has been approved. The formal status is
therefore exactly `NO-GO`; no formal default or formal factory exists.

A proposed formal normalization record must carry the single-cell initial
voltage, exact single-cell basis, source DOI, and applicability. Even a
complete-looking record is rejected because no authoritative numeric record
has been approved. Bare numeric normalization and subclasses are also rejected.
Consequently, raw microvolt loss cannot be multiplied by fuel-cell replacement
price. The formal relative-life and degradation-to-CNY entrypoints fail before
any such multiplication. This document does not invent an initial voltage,
power mapping, lifetime, or equipment-cost conversion.
