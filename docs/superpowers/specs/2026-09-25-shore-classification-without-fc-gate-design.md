# Shore Classification Without an FC-Telemetry Gate

## Objective

Classify shore charging from trustworthy vessel speed and battery charging
power. Recorded vessel fuel-cell power remains available for audit but does
not determine the operating mode. During a modeled shore interval, the
controller executes no DQN decision or MPC solve and commands zero fuel-cell
power.

## Classification contract

A 30 s supervisory sample is a shore candidate when all of the following are
true:

- the composite AIS/BMS evidence passes the existing quality checks;
- `speed_kn <= 0.1`;
- `p_batt_bus_kw < -1.0`, where negative bus power denotes battery charging.

The value of recorded `p_fc_total_kw` does not affect this decision. The
existing three-consecutive-sample confirmation remains frozen. The first two
confirmed samples are `shore_pending`; subsequent confirmed samples are
`shore_charging`. Both modes use the same shore interlock and pause DQN/MPC.
A candidate run shorter than three samples continues to fail closed as
`unresolved`.

## Simulation contract

For `shore_pending` and `shore_charging` intervals:

- do not request a DQN action and do not call the MPC solver;
- execute `p_fc_kw = 0` irrespective of recorded FC telemetry;
- derive accepted battery charging power from the recorded battery-bus
  charging profile, subject to the existing battery power and SOC limits;
- update SOC from accepted battery-side energy;
- update battery degradation from the executed battery current;
- compute shore grid energy once using the frozen aggregate charging
  efficiency and include its cost in the interval ledger/reward;
- produce no hydrogen cost and no FC running degradation for the shore
  interval. Any start/stop transition accounting already defined by the FC
  degradation model remains unchanged.

Raw FC telemetry must not be overwritten. It remains in the mode sidecar as
diagnostic evidence, separate from the simulated zero-FC command.

## Dataset effects

Rebuild only the mode sidecar from the already curated power and AIS datasets.
Do not change dataset membership, Train/Validation/Test assignments, raw power
payloads, AIS payloads, or interpolation. Under the approved rule, the 110
previously unresolved samples in `zero_boundary_029` are expected to join the
already confirmed contiguous shore event as `shore_charging`. The event keeps
its original two leading `shore_pending` samples.

## Tests and verification

Test-first coverage must demonstrate that:

1. Three stationary charging samples classify as shore even when recorded FC
   power is nonzero.
2. Moving charging samples remain onboard when their reconstructed load is
   valid.
3. A shore interval does not call MPC, executes zero FC power, and updates SOC,
   battery degradation, and shore cost from battery charging power.
4. The rebuilt mode sidecar has zero unresolved samples for
   `zero_boundary_029`, without changing power/AIS identities or Test data.

After implementation, run focused tests, all v2 tests, the complete test
suite, formal solver smoke, compile/import checks, and `git diff --check`.

## Non-goals

This change does not alter the DQN state, action catalog, MPC objective,
objective normalization, degradation formulas, dataset split, power/AIS
interpolation, or formal training itself.
