# v2 Battery Degradation Model

## Version, scope, and evidence

The exact model version is `soc_current_weighted_throughput_v1`. Its SOC- and
current-stress equations are sourced from DOI `10.3390/en14133810`.

Positive battery current and power mean discharge; negative current and power
mean charge. This sign convention matches the v2 energy model, but the charge
and discharge efficiencies `eta_chg` and `eta_dis` do not enter degradation
accounting.

## Step equations and units

For `SOC` in `[0, 1]`, current `I` in amperes, positive nominal current
`I_nom` in amperes, and a positive step duration `dt_hours`,

\[
F(SOC)=1+3.25(1-SOC)^2,
\]

\[
G(I)=\begin{cases}
1+0.45I/I_{nom}, & I\geq 0\quad\text{(discharge)},\\
1+0.55|I|/I_{nom}, & I<0\quad\text{(charge)},
\end{cases}
\]

\[
Q_{raw}=|I|\Delta t_{hours}\quad[\mathrm{Ah}],
\qquad
Q_{weighted}=Q_{raw}F(SOC)G(I)\quad[\mathrm{weighted\ Ah}].
\]

Each step exposes raw Ah, weighted Ah, SOC stress, and current stress. The
cumulative account sums raw and weighted Ah only; it does not imply a lifetime
fraction. SOC outside `[0, 1]`, non-positive `I_nom`, non-positive duration,
non-finite values, booleans, and numeric-looking text are rejected.
Result records and initial cumulative-account fields apply the same strict
validation. Cumulative updates compute and validate the complete prospective
state before mutation, so floating-point overflow is rejected without a
partial ledger update.

## Plant values, lifetime throughput, and provenance

`Q_nominal` describes battery capacity. `Q_lifetime` would be a lifetime
weighted-throughput denominator. They are distinct physical quantities, and
the nominal capacity must never be substituted for the lifetime denominator.

The frozen project plant values are `624 kWh` and `432 V`, giving

\[
Q_{nominal}=624000/432=1444.444444\ldots\;Ah,
\]

and the formal 1C current reference is the same numeric value in amperes. The
step implementation converts ampere-seconds to ampere-hours with `/3600`.

The frozen formal-baseline lifetime assumption is

\[
Q_{lifetime}=15000\,Q_{nominal}=21666666.666666664\;Ah.
\]

Its exact classification is
`literature-based lifetime-throughput modeling assumption`, with
`secondary literature basis`. It is not a measured Three Gorges Hydrogen Boat
1 lifetime, manufacturer specification, or Yang project parameter. Its
configuration status is `FROZEN`; its evidence status remains
`SECONDARY_LITERATURE / LITERATURE-CALIBRATED`. Configuration freezing does not
promote it to measured evidence. The immutable future sensitivity candidates
are `{10000, 15000, 20000}`; sensitivity is optional and does not block formal
training.

## Cumulative diagnostics and interval economic cost

For cumulative weighted Ah before and after an interval,

\[
D_{raw}=Q_{weighted}/Q_{lifetime},\qquad D_{econ}=clip(D_{raw},0,1),
\]

\[
\Delta D_{econ}=D_{econ,after}-D_{econ,before}.
\]

Cumulative weighted Ah must be monotonic. `D_raw_after >= 1` reports EOL.
The current interval cost is

\[
C_{Batt,interval}=\Delta D_{econ}\times2000\times624\;CNY.
\]

Crossing EOL charges only the remaining fraction and post-EOL raw throughput
continues with zero economic increment. Without replacement/reset, cumulative
interval charges are bounded by `1,248,000 CNY`. Bare denominators and forged
provenance records remain rejected.
