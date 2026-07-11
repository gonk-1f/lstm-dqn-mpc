# Simplified Normalized Objective Check

Scope: formal 1 s OSQP-QP objective variant for the offline natural-clipped spline benchmark. This does not modify the 30 s CasADi/IPOPT mainline, train LSTM, or train DQN.

## Variant

- Name: `simplified_normalized_literature_v1`
- Objective: `sum(q_h2 * H2_norm + q_soc * SOC_norm + q_batt * Batt_norm)`
- Removed from the objective: fuel-cell ramp soft penalty and terminal SOC penalty.
- Ramp remains as a hard constraint: `48 kW/s = 48.0 kW/step`.

## Fixed Physical Denominators

- `P_fc_max = 560.0 kW`.
- `P_batt_ref = 346.5 kW`, matching the 0.5C power for `E_batt = 693.0 kWh`.
- `SOC_band = 0.05`.
- `m_H2_ref = alpha * 560^2 + beta * 560 = 0.008839452966 kg/step`.

These denominators are fixed physical scales. They are not test set max, min, mean, voyage statistics, or spline-data-derived statistics, so this check does not introduce data leakage.

## Why 346.5 kW Replaces 138.6 kW

- The specification-book full system is about `1806 kWh` and about `900 kW`, which is about `0.5C`.
- The current formal scaled pack uses `10 x 69.3 kWh = 693 kWh`.
- Therefore `693 kWh * 0.5C = 346.5 kW` is used for both the battery bound and the battery normalization denominator.
- The old `277.2 kWh / 138.6 kW` basis is retained only as a legacy assumption in historical outputs, not as the formal battery basis for this run.

## Convexity

- Hessian minimum eigenvalue: `0.0`.
- Convex QP flag: `True`.
- The objective is a nonnegative weighted sum of convex quadratic terms and a linear H2 term. Linear equality/inequality constraints preserve convex QP form.

## Example Normalized Ranges

### H2_norm

| P_fc_kw | H2_norm |
|---:|---:|
| 0.0 | 0.000000000 |
| 100.0 | 0.157141028 |
| 280.0 | 0.463475144 |
| 560.0 | 1.000000000 |

### SOC_norm

| SOC_deviation | SOC_norm |
|---:|---:|
| 0.000 | 0.000000000 |
| 0.010 | 0.040000000 |
| 0.030 | 0.360000000 |
| 0.050 | 1.000000000 |

### Batt_norm

| P_batt_kw | Batt_norm |
|---:|---:|
| 0.0 | 0.000000000 |
| 10.0 | 0.000832901 |
| 50.0 | 0.020822532 |
| 346.5 | 1.000000000 |

## Difference From Legacy Raw Objective

- Legacy raw objective multiplied physical units directly, for example `q_batt * P_batt^2` and optional `q_ramp * delta_P_fc^2`.
- The formal variant uses fixed, dimensionless physical normalizers and keeps only H2, SOC, and battery-use penalty terms.
- This makes fixed weights easier to interpret and more suitable as a baseline before future DQN dynamic weighting, without adding DQN training here.

## Source Coefficients

- alpha/quad coefficient in kg per 1 s step: `4.118109055436559e-09`.
- beta/linear coefficient in kg per 1 s step: `1.3478596369033152e-05`.
- Dp0 forced-origin source coefficients: a1=`1.3478596369033151`, a2=`0.23061410710444727`.
