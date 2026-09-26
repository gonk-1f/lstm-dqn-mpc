# v2 Train-only objective-scale audit

Audit date: 2026-09-26

Dataset: `operating_dataset_zero_boundary_v2`

Result artifact: `outputs/v2_objective_scale_audit/audit_summary.json`

## Decision

`OBJECTIVE_SCALE_AUDIT_READY = YES`

`OBJECTIVE_SCALE_GATE = PASS`

The audit was recomputed after the approved ten-segment exclusion. It used the
current 30 Train segments only, their exact timestamp boundaries, the current
raw-source inventory, and the frozen 36-action catalog. Validation and Test did
not participate in case selection.

Strict raw FC/BMS/AIS eligibility produced 1,834 audit-ready states from 12
Train parents. Six deterministic representative cases were solved with all 36
actions, yielding 216 accepted MPC solves.

The active P95 objective magnitudes are:

- `J_base = 0.0317947064`
- `J_smooth = 0.0393886386`
- active `J_SOC = 0.0590044542`

Thus `scale_ratio = 1.8557949056`, below the frozen PASS boundary of 5. The
existing `600 kW / 600 kW / 0.60` normalization remains acceptable. This is a
global magnitude-comparability result; it does not claim that every action has
equal local influence in every state.

## Frozen contract

For `N=5` and `Ts=30 s`:

`J_base = (1/N) sum_i ((P_fc(i)-P_base(i))/600 kW)^2`

`J_smooth = (1/N) sum_i ((P_fc(i)-P_fc(i-1))/600 kW)^2`

`J_SOC = (1/N) sum_i (d_SOC(SOC(i))/0.60)^2`

where `d_SOC=0` inside `[0.40,0.60]`, with hard SOC bounds `[0.20,0.80]`.
No objective formula, normalization, SOC band, action, or MPC horizon was
changed by this audit.

## Authentication

- Train segment count: 30
- Current power-manifest SHA-256:
  `413f2c8e60487ab43f5d24d179d0cbe20697eb1204ab6521151044a03dddf029`
- Current raw-source inventory SHA-256:
  `aa3df3eb3e1b8b94b3169cb14914b5fd3785bf42bce5dce54385fd076506df89`
- Provenance digest:
  `sha256:8439717d3a3ad0d257cb4e22291077d71cffb6d08d538bf2278a495e6859cdaa`
- Accepted result digest:
  `3a7243751169a118ec62079a4f76c9b6391413777ebbc42f24a117bc9a47605e`

The integrated preflight requires these identities to remain exact. Any change
to the dataset manifests, raw-source inventory, result digest, objective model,
or accepted status fails closed and requires this audit to be rerun.
