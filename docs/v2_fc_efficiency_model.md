# v2 Fuel-Cell Efficiency, Hydrogen, and Battery Energy Model

## Calibration status

Fuel-cell system efficiency and battery charge/discharge efficiency are now
**SOURCE_BACKED**. They are no longer reasons to block formal training. The
later lifetime, state, action, dataset, and integrated-runtime gates have since
been frozen or verified; current readiness is recorded in
`docs/v2_preflight_report.md`.

## Fuel-cell numeric source

The formal source is the user-authorized numeric workbook, which supersedes any
image digitization:

- Workbook: `C:\Users\20883\OneDrive\Desktop\氢耗\FC_Data.xlsx`
- SHA-256: `906a0383f6e427a938a8343e9fb1428bfea0f8e2766f5ac5449fea0ccbde4a21`
- Worksheet and range: `Sheet1`, `A2:B12`
- Column A: net system output, kW
- Column B: system efficiency, percent points

The values below are embedded in the model so tests and runtime are hermetic and
do not depend on the external workbook.

| Source output (kW) | System efficiency (%) |
|---:|---:|
| 0 | 0 |
| 11.3938 | 63.8554 |
| 22.3676 | 62.4096 |
| 35.9133 | 60.4819 |
| 49.116 | 58.7952 |
| 62.146 | 58.0723 |
| 74.8327 | 56.8675 |
| 87.1766 | 56.1446 |
| 98.3215 | 54.9398 |
| 109.123 | 53.7349 |
| 116.668 | 52.0482 |

The source is treated as a 100 kW system characteristic. Only the horizontal
axis is mapped to the 600 kW research aggregate by equal load fraction:

\[
\lambda=\frac{P_{fc}}{600\ \mathrm{kW}}
       =\frac{P_{source}}{100\ \mathrm{kW}}.
\]

Efficiency is converted from percent points to a fraction and otherwise remains
unchanged; it is never multiplied by six. The 109.123 and 116.668 kW source
points lie outside the 100 kW characteristic domain. They are retained as source
provenance and used only as shape-preserving PCHIP support to interpolate the
endpoint at exactly 100 kW. That endpoint is 54.76352421776668%, which maps to
600 kW. The formal interpolation domain is exactly 0 through 600 kW, and queries
outside it fail instead of extrapolating.

The map preserves the measured zero efficiency at zero output. It does not
invent a positive idle efficiency. PCHIP is exact at the retained points and
shape-preserving between them; no polynomial fit or constant-efficiency default
is used. Formal hydrogen accounting revalidates the complete workbook path,
hexadecimal SHA-256, worksheet, range, source-column meanings, raw points, all
transformation statements, rated power, formal power knots, and efficiency
knots. A generic or altered map can still be used for interpolation experiments,
but it cannot enter formal hydrogen accounting. Formal evaluation also rebuilds
PCHIP from the revalidated canonical knots on each call; the map's cached,
mutable interpolation object is therefore not trusted for hydrogen mass.

The workbook's old `m_h2` column is not a formal input. If interpreted as g/min,
it implies an effective heating value of approximately 115.1--116.9 MJ/kg,
which conflicts with the adopted 120 MJ/kg LHV. It is recorded only as a
cross-check conflict and does not override the efficiency/LHV model.

## Hydrogen accounting

The formal lower heating value and one-step hydrogen mass are

\[
LHV_{H_2}=120\ \mathrm{MJ/kg}=33.333333333\ldots\ \mathrm{kWh/kg},
\]

\[
m_{H_2,t}=\frac{P_{fc,t}\,\Delta t_{hours}}
                 {\eta_{fc}(P_{fc,t})\,LHV_{H_2}}\quad[\mathrm{kg}].
\]

At zero output, the implementation returns exactly zero before division, so the
source-backed value `eta_fc(0) = 0` never creates `0/0`. Positive power requires
strictly positive efficiency. Power outside 0 through rated output, non-finite
or non-numeric inputs, non-positive duration, and invalid efficiency all fail
explicitly. The generic formula helper is named with an `_unverified` suffix and
is only a low-level pure-math test utility; the formal public boundary consumes
and verifies the complete authoritative efficiency map.

## Battery energy accounting

The formal charge and discharge efficiencies are both 0.95. Source: DOI
`10.11930/j.issn.1004-9649.202507065`, Table 3. The formal factory returns these
values with the DOI and table location attached. Formal validation requires all
four values to match exactly; missing values, 0.8 efficiencies, arbitrary source
strings, altered table locations, and subclasses cannot pass. Formal `next_soc`
accepts only this verified object, so callers cannot bypass provenance by passing
bare efficiencies. Exact built-in `float` and `str` fields are required, so
comparison-overriding subclasses cannot impersonate the canonical values.
Synthetic sign/dynamics checks use the explicitly named
`next_soc_unverified` pure-math helper instead.

With positive bus power defined as discharge, battery-side power is

\[
P_{cell}=\begin{cases}
P_{batt,bus}/\eta_{dis}, & P_{batt,bus}>0,\\
\eta_{chg}P_{batt,bus}, & P_{batt,bus}<0.
\end{cases}
\]

SOC advances as

\[
SOC_{t+1}=SOC_t-\frac{P_{cell}\Delta t_{hours}}{E_{nominal}}.
\]

The energy model does not clamp SOC; feasibility belongs to the controller.
These efficiencies are used only in energy/SOC dynamics and are never applied
to battery degradation or lifetime normalization.
