# v2 terminal failure penalty calibration

The formal baseline uses a separate terminal failure score of `50,000.0` for
deterministic `PhysicalInfeasibilityError` outcomes. Its evidence classification
is `DERIVED_TRAIN_ONLY / PROJECT_DESIGN`; it is not a measured vessel cost,
manufacturer value, replacement price, or component of the CNY ledger.

The authenticated audit at
`outputs/v2_failure_penalty_audit/audit_summary.json` ran all 30 Train episodes
with the fixed catalog action `w_8_1_1`. It opened zero Test payloads. Results:

- 29 completed episodes;
- one physical failure: `zero_boundary_015`;
- maximum completed raw economic cost: `20,779.575664249034 CNY` on
  `zero_boundary_046`;
- failure penalty: `50,000.0`, more than twice that Train reference maximum.

The audit is bound to the current power/AIS/mode manifest hashes, frozen
36-action catalog digest, failure-policy identity, per-episode score identity,
and canonical result digest. Preflight fails closed if any bound field changes.

Runtime semantics are:

```text
success: learning_reward = -raw_economic_cost_cny
failure: learning_reward = -raw_economic_cost_cny - 50000
```

A physical failure emits one terminal replay transition, including any valid
interval ledgers already executed in that macro step. Numerical solver errors,
non-finite values, malformed backend results, and programming errors remain
fatal and emit no replay transition.
