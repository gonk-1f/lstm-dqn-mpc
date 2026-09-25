# V2 Shore-Event and S8 Finalization Design

## Goal

Finalize the causal DQN state and shore-event semantics before formal v2
training. DQN and MPC operate only while the vessel is onboard. Confirmed or
causally pending shore charging pauses both controllers while physical time,
SOC, battery degradation, and shore cost continue.

## Scope

This change preserves the frozen 36-action catalog, `Ts=30 s`, `N=5`, `M=5`,
the MPC objective, the persistence load forecast, and the interval-incremental
FC/battery degradation economics. It does not start long-running training and
does not change Train/Validation/Test parent allocation.

## Authenticated operating-mode sidecar

The existing signed load and AIS speed files are insufficient to authenticate
shore charging. A versioned sidecar shall be generated on the exact frozen 30 s
axis from the raw eight FC channels, twelve BMS clusters, and AIS telemetry.
Each row retains:

- timestamp and model time;
- aggregate FC output;
- aggregate battery bus power, positive for discharge and negative for charge;
- current causal AIS speed and its provenance;
- per-source completeness/freshness/conflict evidence;
- the causal operating mode and classification reason; and
- source/model hashes in the sidecar manifest.

The frozen project detector uses current and past evidence only:

- near-static AIS: `speed_kn <= 0.1`;
- aggregate FC near off: `abs(p_fc_total_kw) <= 8.0`;
- aggregate BMS charging: `p_batt_bus_kw < -1.0`;
- three consecutive valid 30 s candidate samples confirm shore charging.

The first two causal candidates are `SHORE_PENDING`; they already stop DQN and
MPC. The third and subsequent consecutive candidates are `SHORE_CHARGING`.
A candidate run shorter than three samples is written as `UNRESOLVED`, as are
quality-invalid or contradictory rows. Formal preflight fails closed when
Train or Validation contains `UNRESOLVED`. A positive or zero onboard load is
not shore solely because speed is zero. Negative battery power is not shore
when the FC/onboard evidence contradicts the composite signature.

The detector is a project-design classifier. The thresholds are not described
as vessel-measured or literature-optimal values.

## Frozen S8 state

DQN is queried only at an ONBOARD decision boundary. Its exact state order is:

1. `soc`;
2. `causal_base_load_fraction = P_base / 600`;
3. `load_residual_fraction = (P_load - P_base) / 600`;
4. `recent_load_population_std_fraction = std(P_load) / 600`;
5. `recent_load_window_trend_fraction = slope(P_load) * 150 / 600`;
6. `fuel_cell_power_fraction = P_fc / 600`;
7. `fuel_cell_delta_fraction = (P_fc - P_fc_previous) / 600`;
8. `speed_fraction = speed_kn / 20`.

The fixed scales are project configuration, are not episode statistics, and
are not clipped. State construction may read only samples with timestamps not
later than the current boundary. The MPC continues to use its frozen causal
persistence predictor and never receives future measured load.

## Event-driven macro transition

One DQN action controls at most five actual ONBOARD rolling MPC solves. Physical
intervals in `SHORE_PENDING` or `SHORE_CHARGING` do not increment the MPC count,
DQN global step, epsilon schedule, replay size, or optimizer update count.

If shore begins after the third solve, the remaining two solves are cancelled.
The environment advances every shore interval, accumulates its ledger, and
closes the outstanding transition only at the next ONBOARD decision boundary
or at episode termination. Shore never creates a synthetic DQN action or replay
transition. If the fifth solve is immediately followed by shore, that shore
event is still attached to the same outstanding transition.

The transition reward is the negative sum of all incremental raw-CNY ledgers:

```text
-(H2 + FC degradation + onboard battery degradation
  + shore electricity + shore battery degradation)
```

`executed_mpc_steps` counts only actual MPC solves and therefore lies in
`1..5` for every emitted formal transition.

## Shore physics and economics

During confirmed or pending shore charging:

- `P_fc = 0` and hydrogen consumption is zero;
- simulated SOC remains continuous and is never overwritten by measured SOC;
- the raw BMS aggregate charging trace is treated as a modeled battery-side
  charging-capacity profile;
- accepted battery-side power is limited by that profile, the current simulated
  SOC, the `0.60` project charging target, the `0.80` hard bound, and interval
  duration;
- grid energy equals accepted battery energy divided once by `eta_chg=0.95`;
- shore cost equals grid energy times `1.10 CNY/kWh`; and
- the existing formal battery degradation model receives the actual accepted
  battery current.

If a shore interval brings FC power from positive to zero, the existing FC
start/stop degradation model records the real stop event. No later shore sample
repeats that event.

The zero-boundary dataset removes terminal shore tails. A single explicitly
modeled terminal recharge to the fixed episode initial SOC remains attached to
the last transition. When a future payload contains a real terminal shore
event, the runner must not add a second recharge for energy already delivered.

## Re-entry policy

At shore exit, simulated SOC and cumulative raw degradation remain continuous.
The causal base-load filter and recent onboard load history are reset. The first
re-entry S8 uses current load as base, zero residual/std/trend, previous FC zero,
current FC zero before the first new MPC command, zero delta-FC, and the current
causal AIS speed. Shore power never enters the onboard load statistics.

## Discount semantics

The formal objective is undiscounted finite-episode total CNY. The frozen
training baseline therefore uses `gamma=1.0`. This avoids applying one fixed
discount to transitions whose physical durations differ. A duration-aware SMDP
discount is outside this baseline.

## Training gate

Formal training remains `NO-GO` until all of the following are verified:

- authenticated mode-sidecar identity and hashes;
- zero unresolved Train/Validation mode rows;
- S8 schema/checkpoint compatibility;
- event-driven transition and controller-pause tests;
- causal-state and persistence-forecast leakage tests;
- focused, complete v2, solver/integration, compile/import, and diff checks.

No Test payload may be opened by preflight, training, validation, or model
selection.
