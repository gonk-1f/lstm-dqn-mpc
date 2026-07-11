# Charge-Sustaining H2-MPC Design

## Goal

Replace fixed `SOC=0.65` tracking in the current total-power fixed H2-MPC with voyage-level charge-sustaining and reserve-based SOC management. Preserve the standard zero-delay LSTM-MPC timing and the Dp0 total-hydrogen-mass objective. Do not train or evaluate DQN.

## Root Cause

The current single-system CasADi MPC uses `cfg.soc_target` directly in every stage SOC term and in the optional terminal SOC term. The rolling `current_soc` is only the initial state constraint. With `soc_target=0.65`, `q_terminal_soc=20`, and terminal penalty enabled, the optimizer correctly minimizes the configured objective by raising FC power and charging the battery toward 0.65. Solver success therefore does not imply physically suitable control.

## Timing Invariant

The control reference remains exactly six stages:

```python
mpc_load_ref = np.concatenate([[current_load_t], lstm_pred[:5]])
```

Stage 0 is measured load at decision time. Stages 1 through 5 are LSTM `h1` through `h5`. LSTM, checkpoint, voyage split, and nonnegative forecast projection are unchanged.

## SOC Reference Modes

`CasadiMPCConfig` gains:

```python
soc_reference_mode: str = "fixed_target"
soc_reserve: float = 0.55
terminal_soc_band: float = 0.02
```

Supported modes:

- `fixed_target`: reference is `soc_target`; retained for diagnosis only.
- `initial_soc`: reference is the SOC at the beginning of the current voyage. The runner passes this constant value to every rolling solve; it must not drift with `current_soc`.
- `reserve_only`: no symmetric tracking. Only `max(soc_reserve - SOC, 0)` is penalized.

For `initial_soc` and `fixed_target`, stage cost uses normalized deadband-square:

```python
excess = max(abs(SOC - soc_ref) - soc_band, 0)
cost = (excess / max(soc_band, eps)) ** 2
```

For `reserve_only`, stage cost is the requested one-sided `deficit**2`.

Terminal cost follows the same mode and uses `terminal_soc_band`. The default recommended baseline disables terminal SOC cost.

## SOC Recovery Charge Limit

The signed battery convention remains:

- positive `P_batt`: discharge
- negative `P_batt`: charge

When recovery limiting is enabled and current SOC is at or above reserve, the rolling solve applies:

```python
P_batt[k] >= -max_charge_power_kw
```

for every stage. If current SOC is below reserve, the normal battery charge bound remains available for safety recovery. The default candidate limit is `80 kW`.

## H2 Objective

The objective continues to use the Dp0 fitted total hydrogen mass per stage:

```python
h2_mass_kg = h2_rate_g_s(P_fc) * dt_seconds / 1000
```

It does not minimize specific consumption or inverse efficiency. Outputs include `h2_cost_raw_kg`, normalized H2 cost, and weighted H2 cost.

## Candidates And Recommendation

The old `dp0_batt_penalty_v1` behavior is preserved as `dp0_batt_penalty_v1_old_fixed_target_diag`. New candidates use `initial_soc` or `reserve_only`, Dp0 mass cost, and the recovery charge limit. Recommendation excludes diagnostic names and all `fixed_target` sets.

A candidate is ineligible when any voyage violates solver success, energy balance, SOC bounds, high FC duration, high battery-charge duration, or unjustified first-10-minute SOC rise. Eligible candidates are ranked by total H2, charge-sustaining adjusted H2, battery throughput, FC ramp, and terminal SOC deviation.

## Outputs

Each sweep weight set produces four voyage CSV outputs and three figures per voyage: power/SOC, objective decomposition, and first-20-minute SOC-reference behavior. The diagnosis report compares the old fixed-target case against the new modes and names the recommended fixed baseline without claiming DQN effectiveness.

## Testing

Regression tests cover all three reference modes, voyage-level reference persistence, zero-delay timing, recovery charge bounds, H2 mass semantics, diagnostic exclusion, abnormal-behavior rejection, and required metrics.

