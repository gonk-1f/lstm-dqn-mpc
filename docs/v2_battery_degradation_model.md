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

## Nominal capacity is not lifetime throughput

`Q_nominal` describes battery capacity. `Q_lifetime` would be a lifetime
weighted-throughput denominator. They are distinct physical quantities, and
the nominal capacity must never be substituted for the lifetime denominator.

A valid formal `Q_lifetime` calibration would need all of the following:

- a source DOI;
- an explicit unit consistent with weighted Ah;
- battery chemistry applicability;
- system/configuration applicability.

No such authoritative calibration is currently available. No denominator is
inferred from nominal capacity or invented from the stress-model paper. The
formal battery lifetime-normalization status is exactly `NO-GO`, and there is
no formal default or formal factory.

The helper named `battery_relative_life_loss_unverified` can divide weighted Ah
by a caller-supplied positive number for synthetic checks only. Its name and
separate API prevent that bare number from masquerading as formal calibration.
The formal relative-life entrypoint requires the exact provenance-bearing type
and still rejects it while the calibration remains unresolved; subclasses and
complete-looking forged records cannot bypass the gate.

Equipment replacement prices may be represented at a future cost boundary,
but neither raw Ah nor weighted Ah is a currency. The formal degradation-to-CNY
entrypoint first requires verified lifetime normalization and therefore fails
closed before any multiplication by price. This document supplies neither a
fabricated lifetime throughput nor a degradation cost.
