# v2 Train-only objective-scale audit

## 1. Final objective definitions

For horizon length `N`:

`J_base = (1/N) sum_i ((P_fc(i)-P_base(i))/600 kW)^2`

`J_smooth = (1/N) sum_i ((P_fc(i)-P_fc(i-1))/600 kW)^2`

The first smoothness difference uses the previously executed FC power. For SOC,
`d_SOC=0.40-SOC` below 0.40, zero on the inclusive interval `[0.40,0.60]`, and
`SOC-0.60` above 0.60:

`J_SOC = (1/N) sum_i (d_SOC(SOC(i))/0.60)^2`.

The DQN action remains three positive weights summing to one. Weight simplex and
fixed objective normalization have different roles and are not interchangeable.

## 2. Why the fixed scales are 600 / 600 / 0.60

`P_fc_scale=600 kW` is the aggregate rated power of the selected research
simulation configuration. `Delta_P_fc_scale=600 kW` uses the same rated-power
scale for numerical comparability; the old v1 `48 kW` one-second quantity is no
longer used as an objective denominator. `SOC_scale=0.60` is the width of the
hard physical interval, `0.80-0.20`.

The FC hard ramp constraint is independent. Code still requires an explicit
positive `fuel_cell_ramp_kw_per_step`; there is no formal source-backed default.
The solver smoke fixture currently uses `100 kW/step`, which is test input rather
than a calibrated plant fact. This change neither adopts `48 kW/step` nor creates
`48*30=1440 kW/step`.

## 3. SOC hard and soft regions

- Hard physical constraint: `0.20 <= SOC <= 0.80`.
- Soft zero-penalty working band: `0.40 <= SOC <= 0.60`.

The working band is not added to the SLSQP constraint set. SOC values outside it
but inside the hard interval remain physically feasible and receive a soft
quadratic penalty.

## 4. Train-only evidence boundary

`run_objective_scale_audit` rejects Validation, Test, Unknown, old dataset
versions, and forged provenance before invoking the lazy state loader or solver
runner. Each accepted Train state is solved with the complete canonical 36-action
tenth-grid bank in canonical order; subsets and reordered banks are rejected. The returned exact `MPCPlan`
must contain a successful five-step plan whose total objective matches the
requested action weights.

Validation/Test cannot select normalization constants, thresholds, or action
weights. The audit never applies a recommended rescaling automatically.

## 5. Objective statistics

The required per-term output is `count`, population `mean/std`, `P50`, `P90`,
`P95`, `P99`, and `max`. SOC additionally reports `Pr(J_SOC>0)` and the four
conditional positive percentiles.

Current formal Train result: **not available**. The repository has no accepted
raw Train operating-cycle payload and no final action catalog. Consequently no
actual Train MPC solve set was created in this increment; counts, means,
percentiles, and maxima are reported as `N/A`, not replaced by synthetic values.

## 6. Active-P95 scale ratio

The audit uses full-distribution P95 for `J_base` and `J_smooth`, and
`P95(J_SOC | J_SOC>0)` for SOC. It computes
`max(P95_active)/min(P95_active)`. The declared project engineering rule is:

- ratio `<=5`: GO;
- `5<ratio<10`: WARNING;
- ratio `>=10`: NO-GO.

These limits are engineering review targets, not theoretical constants. Current
ratio: `N/A`; objective-scale status: **NO-GO**, because positive-SOC and other
representative Train solve evidence is absent.

## 7. Weighted contribution dominance

For every ordered term pair the audit records actual `q_i*J_i` distributions
and the count/rate of samples satisfying `0.1*J_i > 0.7*J_j`, including affected
state/action IDs. Current result: `N/A`; it is therefore not known whether one
term at weight 0.1 still persistently dominates another at weight 0.7.

## 8. Behavioral responsiveness

Per Train state, the audit compares different actions using raw components,
first FC command, first battery command, and the complete predicted SOC path.
Numerically indistinguishable action pairs under explicit tolerances are listed
as behavioral redundancy. Current result: `N/A`; no claim about action
responsiveness is supported, and the action catalog is not redesigned here.

## 9. Recommendation boundary

When a completed Train audit is NO-GO, its active P95 values may be recorded as
candidate fixed constants `c_i` for a later reviewed proposal
`J_i_final=J_i/c_i`. This increment does not have the evidence required to
compute such constants and does not modify the method beyond the approved
600/600/0.60 definitions.

## 10. Final decision

`OBJECTIVE_SCALE_AUDIT = NO-GO` and `FORMAL_TRAINING = NO-GO`. Unit tests use
synthetic plans only to verify formulas, split guards, statistics, contribution
logic, redundancy detection, and sealing; they are not experimental results.
