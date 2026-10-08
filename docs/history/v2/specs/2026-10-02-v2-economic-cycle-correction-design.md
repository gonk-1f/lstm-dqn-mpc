# v2 Economic and Start-Cycle Correction Design

## Approved baseline

- Count one fuel-cell start/stop cycle only on an `OFF -> ON` transition.
- Keep the exact on/off predicate `power_kw > 0`; do not add an unsupported threshold.
- Set hydrogen price to `21.9 CNY/kg`, sourced from Yang et al. (2026), Table 6.
- Keep fuel-cell equipment price at `3500 CNY/kW` and apply the approved stack
  replacement fraction `0.5`, so the aggregate replacement charge is
  `3500 * 600 * 0.5 = 1,050,000 CNY`.
- Preserve incremental interval life-loss charging and EOL clipping.
- Bump the reward and fuel-cell degradation semantic identities so old reward
  scales and checkpoints fail closed.
- Regenerate the authenticated Train-only reward scale without opening Test.

## Retired pilots

P1 and P2 are retired completely: remove their experiment profiles, CLI choices,
focused tests, design/plan documents, and output directories. Generic epsilon
schedule and replay-capacity configuration remain because H1-H4 use those
checkpoint-bound fields.

## Verification

Use red-green tests for start-only cycle counting, price provenance, hydrogen
cost, replacement cost, interval EOL cap, and semantic-version rejection. Then
run focused tests, all v2 tests, solver smoke, compile/import, regenerated reward
scale authentication, and `git diff --check`.
