# Objective-audit Train data rules

## Scope

Freeze only the data-eligibility rules needed by the existing Train-only MPC
objective-scale audit. MPC objectives, `600/600/0.60` normalization, SOC bands,
the 36-action candidate simplex, and DQN training remain unchanged.

## Frozen Train-only rules

- `FRESHNESS_CAP_SECONDS = 10.0`. A match is usable only when
  `0 <= supervisory_timestamp - source_timestamp <= 10 s`. Matching is causal,
  one-to-one, and never forward-fills an older source after the cap.
- `SPEED_ZERO_TOLERANCE_KN = 0.1`, the minimum positive AIS increment observed
  in the Train parents.
- `FC_ZERO_TOLERANCE_KW = 8.0`, the Train-only P99 of aggregate FC power while
  all eight FC channels report non-running.
- `LONG_GAP_SECONDS = 45.0`, inherited from the prior Train clock audit as
  `1.5 * 30 s`.
- Operating mode is exactly one of `sailing_island`, `shore_connected`, or
  `unknown`.
- Shore requires at least two consecutive, gap-free complete states satisfying
  `speed <= 0.1 kn`, `abs(P_fc_total) <= 8 kW`, and `P_batt_total < 0`.
- Sailing requires a complete fresh state, `speed > 0.1 kn`, and nonnegative
  reconstructed balance. Every stationary non-shore, contradictory, missing,
  stale, conflicting, or gap-contaminated state is `unknown`.
- Only `sailing_island` may use
  `P_load = P_fc_total + P_batt_total`, with battery discharge positive.

## Audit data flow

Read only parent folders whose frozen manifest split is Train. Within every
equipment channel, remove exact timestamp duplicates and invalidate conflicting
duplicates. Use the deduplicated left-FC-1 clock without rounding. Align the
other seven FC channels, twelve BMS channels, and AIS speed using the latest
unused source timestamp not later than the supervisory timestamp. Reject stale
or long-gap-contaminated states before classification.

Select a small deterministic subset across Train-only load level, load-change
intensity, and available SOC bands, constrained to the hard SOC interval. Replay
the 90 s causal base-load filter only along preceding eligible supervisory
history. Run each selected case with all 36 canonical actions and record the
existing audit statistics. If no compliant cases survive, do not call MPC.

## Evidence boundary

The 10 s cap, 0.1 kn speed tolerance, and 8 kW FC tolerance are provisional
project audit rules derived only from Train timestamps and sensor distributions.
They are not equipment limits. Validation/Test and old 1 s interpolated data are
never read.
