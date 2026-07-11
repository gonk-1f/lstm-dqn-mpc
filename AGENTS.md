# AGENTS.md

## Operating Rule

- Answer and act from verified project facts, not user preference.
- Do not appease the user: correct wrong assumptions directly and state uncertainty when evidence is missing.
- Do not fabricate vessel parameters, model behavior, metrics, or experimental conclusions.
- Keep context small; read only the status files and directly relevant files before acting.
- After meaningful work, overwrite-update `project_status.md`, `next_steps.md`, and `thread.md`.

## Project

- Domain: shipboard fuel-cell / lithium-battery hybrid microgrid energy management.
- Goal: engineering-oriented hierarchical EMS, not a toy RL demo.
- Current paper label: `Proposed MPC-KAN-DQN`.
- Current stage: fixed-weight; dynamic weights are not implemented.
- Formal current evaluation scenario: 7-2-1 voyage split with four complete test voyages.
- Low-SOC results: stress test / appendix only.

## Current Architecture

1. LSTM forecast provides rolling load horizon.
   - current training entrypoint: `src/main/run_train_lstm_721.py`.
   - current checkpoint: `outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/best_lstm_load_predictor.pt`.
   - current checkpoint is the reproducible `validation_MAE`-selected retrain promoted from `outputs/lstm_721_retrain/`.
   - previous better-performing but non-reproducible checkpoint is backed up under `outputs/lstm_721/candidate_asym_weighted_huber_delta10/checkpoints/candidate_asym_weighted_huber_delta10/backup_before_reproducible_retrain_20260617_200348/`.
   - retraining defaults to `outputs/lstm_721_retrain/` and does not overwrite the formal checkpoint unless `--overwrite_current` is explicitly passed.
   - retraining optimizes weighted asymmetric Huber loss but selects the best checkpoint by `validation_MAE`.
   - current test outputs: `outputs/lstm_test/`.
   - forecasts entering MPC are nonnegative projected.

2. Upper MPC outputs total references.
   - current fixed LSTM-MPC entrypoint: `src/main/run_lstm_mpc_test.py`.
   - horizon `N=6`, sample time `30 s`.
   - stage 0 uses actual measured load at decision time; stages 1..5 use LSTM future predictions.
   - outputs total `P_fc_ref` and `P_batt_ref`.
   - current fixed LSTM-MPC is total-power control only; do not describe it as left/right or dual-side energy management.
   - objective terms are set through `outputs/config/mpc_weight_sets.json`.

3. KAN-DQN dynamic weighting is future work.
   - Do not train DQN until the fixed LSTM-MPC baseline on the four-voyage test set is accepted.
   - KAN is the Q-network type, not a separate controller layer.
   - DQN must not directly output left/right device powers unless a new design explicitly changes the controller interface.

4. Execution layer enforces physics.
   - physical projection enforces power balance and bounds.
   - allocator splits FC symmetrically by default.
   - battery split uses limited soft SOC-balance bias.

## Current Weights

- Fixed MPC weight sets live in `outputs/config/mpc_weight_sets.json`.
- Current fixed baseline set: `dp0_baseline_v1`.
- Current baseline weights: `q_h2=0.95`, `q_soc=2.80`, `q_fc=0.0`, `q_batt=0.0`, `q_ramp=0.005`, `soc_band=0.10`, terminal SOC soft penalty disabled.
- Additional named sets remain available for comparison: `balanced_v1`, `economy_v1`, and `safety_v1`.
- DQN reward/weight logic is not current experimental evidence.

## Paper Metric Policy

- Do not cite `paper_score_no_ramp` as a paper result; it is internal auxiliary selection score only.
- Do not cite `total_cost` as economic money; it is environment reward/cost accumulation.
- Main paper metrics should be physical:
  - FC use in current fixed LSTM-MPC: report Dp0 curve-based `H2_total_kg`.
  - FC fluctuation: `P_fc_std`.
  - SOC maintenance: `SOC_end_minus_start` and/or target-band/terminal SOC.
  - SOC transition smoothness: add `soc_slope_std = std(diff(SOC))`; optional `soc_step_max_abs`.
  - Battery use: `battery_discharge_kwh`, `battery_throughput_kwh`.
- Battery throughput is battery use / cycling proxy, not validated battery lifetime.
- `P_fc_ramp_mean`, `P_fc_ramp_max`, and `action_switch_count` are diagnostics only.
- Current H2 model uses the imported fresh `D_p=0` fuel-cell curve in `data/fuel_cell/FC_Dp0_curve_for_Python.csv` through `src/mpc/solvers/fc_dp0_curve.py`.
- The curve maps by relative load against default total FC rating `560 kW`; do not treat the figure's `P_sys=100 kW` as the ship system size.

## Key Paths

- current concise status: `project_status.md`, `next_steps.md`, `thread.md`.

## Work Discipline

- Prefer `rg`/targeted reads; avoid broad scans and old archived results unless needed.
- Preserve modular boundaries: MPC, DQN, allocator, environment, plotting/reporting.
- Keep units and sign conventions explicit.
- Use reversible, narrow edits; do not delete major project parts without clear approval.
- Validate with the smallest useful command before claiming completion.
- For document work, render DOCX outputs and inspect page images before delivery.
