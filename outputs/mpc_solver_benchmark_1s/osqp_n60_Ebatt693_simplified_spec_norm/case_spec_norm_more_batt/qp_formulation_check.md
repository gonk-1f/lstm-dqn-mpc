# 1 s QP Formulation Check

Scope: parallel benchmark formulation only. This file does not modify the current 30 s CasADi/IPOPT mainline.

## Dimensions

- Horizon: `60`
- Sample time: `1.0 s`
- Variables: `181`
- Constraints: `362`
- Variable order: `P_fc[0:N], P_batt[0:N], SOC[0:N+1]`

## Ramp Unit

- Source ramp rate: `48.0 kW/s`
- Solver ramp bound: `48.0 kW/step`
- Conversion rule: `fuel_cell_ramp_rate_kw_per_s multiplied by dt_seconds`

## Hydrogen Quadratic

- Dp0 CSV: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\data\fuel_cell\FC_Dp0_curve_for_Python.csv`
- Forced-origin fit a1: `1.3478596369033151`
- Forced-origin fit a2: `0.23061410710444727`
- kg/step quadratic coefficient: `4.118109055436559e-09`
- kg/step linear coefficient: `1.3478596369033152e-05`

## Convexity

- Hessian minimum eigenvalue: `0.0`
- Convex QP flag: `True`

## JSON Metadata

```json
{
  "horizon": 60,
  "dt_seconds": 1.0,
  "variable_order": "P_fc[0:N], P_batt[0:N], SOC[0:N+1]",
  "n_variables": 181,
  "n_constraints": 362,
  "fuel_cell_ramp_rate_kw_per_s": 48.0,
  "fuel_cell_ramp_kw_per_step": 48.0,
  "fuel_cell_ramp_kw_explicit_override": false,
  "fuel_cell_ramp_source": "fuel_cell_ramp_rate_kw_per_s multiplied by dt_seconds",
  "objective_variant": "simplified_normalized_literature_v1",
  "objective_terms": [
    "H2_norm",
    "SOC_norm",
    "Batt_norm"
  ],
  "battery_power_ref_kw": 346.5,
  "soc_band": 0.05,
  "h2_reference_kg_per_step": 0.00883945296644347,
  "h2_curve_csv": "C:\\Users\\20883\\OneDrive\\Desktop\\microgrid-mpc-master\\data\\fuel_cell\\FC_Dp0_curve_for_Python.csv",
  "dp0_forced_origin_a1": 1.3478596369033151,
  "dp0_forced_origin_a2": 0.23061410710444727,
  "h2_kg_step_quad_coeff": 4.118109055436559e-09,
  "h2_kg_step_linear_coeff": 1.3478596369033152e-05,
  "hessian_min_eigenvalue": 0.0,
  "convex_qp": true,
  "diagnostics_computed": true,
  "battery_cost_form": "normalized (P_batt / P_batt_ref)^2"
}
```
