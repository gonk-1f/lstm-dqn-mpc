# v2 Train-only objective-scale audit

Audit date: 2026-09-22
Branch: `refactor/multiscale-dqn-wmpc-v2`

## 1. Decision

`OBJECTIVE_SCALE_AUDIT_READY = YES`

`OBJECTIVE_SCALE_GATE = PASS`

The audit used original timestamped telemetry from the 46 parents frozen as
Train in `parent_split_manifest.csv`. Strict causal construction produced 1,211
audit-ready supervisory states from 20 Train parents. Six deterministic
representative cases were solved with all 36 canonical simplex actions, for 216
accepted MPC solves.

The three active P95 values are `0.0492065`, `0.0376662`, and `0.0688487`.
Their `scale_ratio` is `1.827863`, which is below the project PASS boundary of
5. The fixed `600 / 600 / 0.60` normalization therefore passes the declared
global objective-scale comparability criterion on this Train audit.

This PASS is qualified. Ordered minimum-vs-maximum-weight dominance reaches
`46.76%` for `J_SOC > J_base` and `46.30%` for `J_SOC > J_smooth`. The three
terms are globally comparable at active P95, but there is clear local,
state/action-conditioned dominance. The audit does not justify claiming that
every simplex weight remains equally influential in every operating state.

No objective formula, normalization constant, SOC band, hard boundary, DQN
simplex action, or action catalog was changed. No DQN training was run.

## 2. Frozen audit contract

For horizon length `N=5`, the unweighted components remain:

`J_base = (1/N) sum_i ((P_fc(i)-P_base(i))/600 kW)^2`

`J_smooth = (1/N) sum_i ((P_fc(i)-P_fc(i-1))/600 kW)^2`

`J_SOC = (1/N) sum_i (d_SOC(SOC(i))/0.60)^2`

where `d_SOC=0` within `[0.40,0.60]`, is the distance to `0.40` below the
working band, and is the distance to `0.60` above it. The hard SOC interval is
unchanged at `[0.20,0.80]`.

The audit used the research-simulation battery bounds `-624/+1248 kW`, the
30 s MPC scale, `tau_LPF=90 s`, and no hard FC ramp. Those timescale values are
now frozen project-design baseline configuration; this historical audit is not
evidence that they are vessel-calibrated or uniquely optimal. Disabling the hard
ramp does not disable `J_smooth`.

## 3. Train-only eligibility rules

Only manifest-declared Train parents were read. Validation, Test, old 1 s
interpolated data, and derived `aligned_30s` tables were not used as audit
states.

The frozen rules are:

- `FRESHNESS_CAP_SECONDS = 10.0`; accepted matches satisfy
  `0 <= supervisory_timestamp - source_timestamp <= 10 s`.
- Matching is causal, one-to-one, and never uses future samples or stale
  forward-fill.
- All eight FC channels, all twelve BMS cluster channels, and AIS speed must be
  complete and fresh.
- `SPEED_ZERO_TOLERANCE_KN = 0.1` and `FC_ZERO_TOLERANCE_KW = 8.0` are
  provisional Train-derived audit rules, not equipment limits.
- Shore classification requires at least two consecutive, gap-free samples
  with near-zero speed, near-zero total FC power, and negative total battery
  power.
- A complete fresh state with speed above `0.1 kn` and nonnegative reconstructed
  balance is `sailing_island`; ambiguous, contradictory, stale, incomplete, or
  long-gap-contaminated states are `unknown`.
- Only `sailing_island` may use
  `P_load = P_fc_total + P_batt_total`, with battery discharge positive.

The raw loader collapses exact within-channel duplicates before strict state
construction. The state builder observed zero remaining exact duplicate rows
and zero conflicting duplicate timestamps.

## 4. Supervisory-state result

| Item | Count |
|---|---:|
| Train parents read | 46 |
| candidate supervisory states | 33,115 |
| `sailing_island` | 6,175 |
| `shore_connected` | 566 |
| `unknown` | 26,374 |
| audit-ready states | 1,211 |
| Train parents contributing audit-ready states | 20 |
| representative cases | 6 |
| canonical actions per case | 36 |
| accepted MPC solves | 216 |

SOC coverage among the 1,211 audit-ready states was:

| SOC band | Count |
|---|---:|
| `< 0.40` | 0 |
| `[0.40,0.60]` | 134 |
| `> 0.60` | 1,077 |

The deterministic representative set covered every available hard-bound-valid
SOC band, low/medium/high load, and steady/rapid load change. No eligible
below-0.40 state existed, so the audit contains positive SOC penalties from the
upper side only; no lower-side behavior is inferred.

Representative case IDs:

1. `3月27日07_00_3月27日11_00/2024-03-27T08:35:29+08:00`
2. `3月27日07_00_3月27日11_00/2024-03-27T08:42:59+08:00`
3. `3月28日08_00_3月28日11_00/2024-03-28T09:48:31+08:00`
4. `4月18日12_00_4月18日18_00/2024-04-18T14:25:43+08:00`
5. `4月23日13_00_4月23日18_00/2024-04-23T17:29:22+08:00`
6. `5月9日08_00_5月9日17_00/2024-05-09T09:49:08+08:00`

Provenance ID:
`sha256:8d3c927af075b70b167a5d3e35235de1bbe3d02a2da9ff21016d5d77747fa4db`.

Result digest:
`691719dedacbc7053728b09d21bd20597ae7253c45b1250b1c514a07992087d5`.

## 5. Unweighted objective statistics

| Objective | count | mean | P50 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `J_base` | 216 | 0.0103392 | 0.00328228 | 0.0302056 | 0.0492065 | 0.0759031 | 0.0851657 |
| `J_smooth` | 216 | 0.0101138 | 0.00329250 | 0.0347131 | 0.0376662 | 0.0448716 | 0.0486550 |
| `J_SOC` | 216 | 0.0480326 | 0.0524876 | 0.0675149 | 0.0686387 | 0.0695259 | 0.0697423 |

SOC activity statistics:

| Statistic | Result |
|---|---:|
| `count(J_SOC > 0)` | 180 |
| `Pr(J_SOC > 0)` | 0.833333 |
| `P50(J_SOC \| J_SOC > 0)` | 0.0534461 |
| `P90(J_SOC \| J_SOC > 0)` | 0.0676663 |
| `P95(J_SOC \| J_SOC > 0)` | 0.0688487 |
| `P99(J_SOC \| J_SOC > 0)` | 0.0695701 |

## 6. Active P95 and scale ratio

| Objective | active P95 source | active P95 |
|---|---|---:|
| `J_base` | full-distribution P95 | 0.0492065 |
| `J_smooth` | full-distribution P95 | 0.0376662 |
| `J_SOC` | positive-only P95 | 0.0688487 |

`scale_ratio = 0.0688487 / 0.0376662 = 1.827863`.

The project criterion is:

- `scale_ratio <= 5`: PASS;
- `5 < scale_ratio < 10`: WARNING;
- `scale_ratio >= 10`: NO-GO.

Result: **PASS**.

## 7. Minimum-vs-maximum-weight dominance

For every ordered pair, the event is `0.1 * J_i > 0.7 * J_j`. Rates use all
216 accepted state/action observations.

| Dominant term `i` | Dominated term `j` | Count | Proportion |
|---|---|---:|---:|
| `J_base` | `J_smooth` | 26 | 12.04% |
| `J_base` | `J_SOC` | 15 | 6.94% |
| `J_smooth` | `J_base` | 69 | 31.94% |
| `J_smooth` | `J_SOC` | 36 | 16.67% |
| `J_SOC` | `J_base` | 101 | 46.76% |
| `J_SOC` | `J_smooth` | 100 | 46.30% |

There is clear dominance in this representative audit, especially from the SOC
term against the other terms. This does not contradict the active-P95 PASS:
active P95 tests global scale comparability, while dominance is a local
state/action test and is amplified when another objective is near zero.

No project threshold was approved for converting a dominance proportion into a
different gate result, so these rates are reported as a material warning and do
not overwrite the declared active-P95 gate.

## 8. Conclusion and boundary

The `600 / 600 / 0.60` normalization is sufficient to pass the project's
current global objective-scale criterion on the audited Train subset. It is not
sufficient evidence that DQN weight changes will always have balanced local
influence, because the ordered dominance rates are substantial and the sample
contains no eligible SOC-below-0.40 state.

Therefore:

- objective-scale comparability: **PASS / VERIFIED**;
- obvious local dominance: **YES**;
- automatic normalization change: **NO**;
- new scaling coefficients introduced: **NO**;
- formal DQN training authorized by this report: **NO**; other independent
  preflight gates remain unresolved or NO-GO.
