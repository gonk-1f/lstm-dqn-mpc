# V2 DQN State Audit Design

## Goal

Select a low-dimensional, causal, physically interpretable DQN state for the
current v2 multi-rate DQN-WMPC architecture. The audit must use only the
currently active formal Train split and must not start training, inspect held-out
data for feature selection, or modify the action catalog.

## Authoritative Scope

- Code under `src/v2/` is authoritative for the controller, environment,
  economics, plant models, and current candidate state.
- `data/processed/operating_dataset_zero_boundary_v2/` is the authoritative
  formal operating dataset because it is the default root used by
  `src/utils/formal_operating_dataset.py`.
- Only the 38 rows whose `split` is `train` in
  `metadata/sample_manifest.csv` may select raw parent records or contribute
  observations to feature statistics, correlations, regime analysis, or state
  recommendations.
- Validation and Test files must not be opened by the audit runner.
- The raw telemetry root is used only to reconstruct measured FC, battery, and
  SOC values inside the approved Train parent and timestamp boundaries.
- V1 and its historical 84-action design are outside scope.

## Alternatives Considered

### Measurement-first reconstruction (selected)

Use the active Train manifest as a strict parent/time whitelist. Reconstruct
the eight-FC total, twelve-cluster battery total, and system SOC from causal raw
telemetry alignment. Compute load-history features from the formal Train load
series. This preserves measured operating information without making the audit
depend on a candidate MPC action.

### Fixed-controller rollout (rejected)

Generating FC and SOC trajectories with one fixed action would make the feature
distribution depend on an arbitrarily selected MPC weight triple. That creates
circular evidence for a state whose purpose is to choose those weights.

### Load-only audit (rejected)

The formal segment CSVs contain timestamp, elapsed seconds, and total load only.
They cannot support evidence about SOC, FC dynamics, battery redundancy, or the
Markov sufficiency of the proposed state.

## Sampling and Causal Feature Construction

The audit operates at the frozen supervisory period `Ts = 30 s`.

1. Load only Train manifest rows and verify that every selected source path is
   under `train/`.
2. Restrict raw telemetry discovery to the selected Train parent identifiers.
3. Restrict every parent to its formal Train start and end timestamps.
4. Use the existing v2 causal alignment rules: no future sample, no source-row
   reuse, and at most 10 seconds freshness.
5. Require all eight FC channels and all twelve battery channels for measured
   power features. SOC is the arithmetic mean of the twelve available cluster
   SOC values, following the existing supervisory audit convention.
6. Mark incomplete or stale observations as ineligible. Do not interpolate new
   FC, battery, or SOC values for this audit.
7. Sample the frozen formal load at 30-second supervisory instants. Existing
   dataset construction remains unchanged; the audit adds no interpolation and
   deletes no data.
8. Analyze operational samples only. Negative formal load and boundary-only
   zero-load points are reported separately and excluded from candidate DQN
   state statistics because the current nonlinear MPC rejects negative observed
   load and the state is intended for active vessel operation.

The causal LPF uses the frozen values:

```text
Ts = 30 s
tau_LPF = 90 s
alpha = exp(-30 / 90)
P_base(k) = alpha P_base(k-1) + (1-alpha) P_load(k)
```

Recent mean, population standard deviation, and least-squares trend use the
inclusive 150-second interval `[t-150 s, t]`. This duration equals the frozen
DQN macro interval and is a project-design choice, not a Train-optimized
hyperparameter. Each segment initializes its own LPF and history; no state is
carried across parent or segment boundaries.

## Audited Physical Features

The audit builds the current ten scalars and the proposed replacements in
physical units before applying frozen normalization:

- current SOC;
- current and previous measured FC total power;
- measured battery discharge-positive total power;
- formal total load;
- recent load mean, population standard deviation, and trend;
- causal base load;
- recent SOC change over the 150-second window;
- `delta_load = P_load - P_base`;
- `delta_fc = P_fc(k) - P_fc(k-1)`.

Power-balance redundancy is reported in two distinct forms:

- the environment identity `P_batt = P_load - P_fc`, which is exact by model
  construction;
- the measured residual `P_load - P_fc - P_batt` after causal raw alignment,
  which may contain measurement and interpolation mismatch and does not turn
  battery power into an independent simulated state.

## Frozen Normalization for Comparison

Feature selection is performed in physical units and normalization never uses
Train minima or maxima.

- `SOC`: unchanged in `[0, 1]`;
- `P_fc` and `delta_fc`: divide by the v2 research plant FC rating, 600 kW;
- load, base load, load residual, load mean, and load standard deviation:
  divide by 600 kW;
- battery power: divide by the existing explicit v2 battery power scale used by
  the candidate state audit;
- load trend: `trend_kw_per_s * 150 s / 600 kW`;
- recent SOC change: raw SOC fraction difference.

The report must distinguish this research-simulation normalization from the
real-vessel 560 kW technical specification. No normalization may be adjusted
from Validation or Test observations.

## Statistical Evidence

For every eligible candidate and replacement feature, produce Train-only:

- count, missing count, minimum, maximum, mean, population standard deviation,
  P1, P5, P50, P95, and P99;
- Pearson and Spearman correlation matrices;
- exact or near-deterministic relationship checks;
- near-zero-variance checks with the threshold and units stated explicitly;
- redundancy evidence for battery power, recent load mean, previous FC power,
  FC delta, and recent SOC change.

Regime labels are derived from Train data only and are descriptive rather than
training labels. Rise, fall, and steady states use the sign and magnitude of the
causal 150-second trend. Volatility uses the causal load standard deviation.
SOC and FC low/high groups use fixed physical thresholds already present in v2
when available; otherwise the report presents continuous/bin summaries without
inventing a frozen controller threshold. Any Train quantile used solely for a
descriptive table is recorded and must not become a production threshold.

## State Comparisons

The report compares, without training a DQN:

- `S10`: the current ten-scalar candidate;
- `S7`: SOC, base load, load residual, load standard deviation, load trend,
  current FC power, and FC delta;
- `S6-A`: S7 without FC delta;
- `S6-B`: S7 without load standard deviation;
- a minimum defensible state selected from causal sufficiency and physical
  control relevance.

Comparison criteria are causality, online availability, leakage safety,
relationship to `q_base`, `q_smooth`, and `q_soc`, approximate Markov
sufficiency, dimensional economy, deterministic redundancy, and regime
discrimination. Correlation alone cannot select or remove a feature.

## Markov Audit

The code audit covers `src/v2/control/`, `src/v2/envs/`, `src/v2/economics.py`,
and `src/v2/models/`. It inventories every remembered value and classifies it as:

1. represented by the candidate state;
2. solver-only numerical state that does not alter the physical transition or
   reward definition;
3. episode bookkeeping whose omission is justified under the formal episode
   initialization and EOL assumptions;
4. physical or reward state that must be added for approximate Markov behavior.

The inventory explicitly covers causal LPF state, previous executed FC power,
cumulative FC voltage loss, cumulative weighted battery Ah, previous DQN action,
MPC warm start, terminal recharge accounting, and the macro-step position if it
can affect transition or reward.

SOH features are not automatically added. They are required only if the formal
training episode can approach the EOL clipping boundary or if marginal reward
or transition changes materially with cumulative lifetime state. Otherwise the
report documents the initialization and distance-to-EOL evidence supporting
their omission.

## Outputs

Create `outputs/v2_dqn_state_audit/` containing:

- `audit_manifest.json`: active dataset identity, Train-only input hashes,
  frozen constants, sample accounting, and command provenance;
- `feature_statistics.csv`;
- `pearson_correlation.csv`;
- `spearman_correlation.csv`;
- `power_balance_residuals.csv`;
- `regime_summary.csv`;
- `state_comparison.csv`;
- compact correlation, distribution, and regime-discrimination figures.

Create `docs/v2_dqn_state_audit.md` containing the requested ten-feature
KEEP/REMOVE/REPLACE/CONDITIONAL table, final schema, normalization, history,
redundancy conclusions, Markov inventory, Train-only evidence, minimum state,
and an explicit conclusion on S7.

## Implementation Boundaries

- Add a focused v2 analysis module, a v2 command-line runner, and focused tests.
- Do not modify `src/v2/dqn/state.py` in this audit phase.
- Do not change preflight status, action candidates, MPC objectives, economic
  formulas, formal dataset contents, or split manifests.
- Do not start DQN training or solver-based action screening.
- Do not read Validation or Test segment files during audit execution.

## Verification

Tests must prove that the runner rejects held-out manifest rows, reads only
Train paths, preserves causal alignment, constructs the 150-second history
without crossing segment boundaries, uses fixed normalization, and produces
deterministic outputs from fixture telemetry. Final verification includes the
focused tests, all v2 tests, compile/import checks, and `git diff --check`.
