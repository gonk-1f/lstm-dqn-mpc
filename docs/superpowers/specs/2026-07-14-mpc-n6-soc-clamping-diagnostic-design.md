# N=6 MPC Near-Reference SOC Clamping Diagnostic Design

**Date:** 2026-07-14

**Approval:** The user explicitly instructed execution of this bounded diagnostic. The supplied contract fixes the model, case matrix, metrics, plots, three allowed labels, output paths, verification commands, commit message, and Git push workflow. It also forbids full-voyage reruns, formal configuration changes, LSTM/DQN work, and historical-result deletion.

## Question and decision boundary

The experiment asks one question: with `N=6`, `q_soc=20`, a steady 300 kW load, and SOC only `0.02` away from `SOC_ref=0.55`, does the controller sustain material power transfer solely to force SOC back to the reference, creating avoidable FC surplus, battery cycling, and hydrogen use?

The result must use exactly one reviewed label:

- `no_evidence_of_SOC_clamping`
- `moderate_SOC_clamping`
- `excessive_SOC_clamping`

This is a diagnostic label, not a fixed-weight selection. No provisional or accepted MPC configuration is created.

## Existing evidence and protected artifacts

The existing `QSOC_20` seven-voyage artifacts are background evidence only and will not be rerun or rewritten. Their retained metrics support these bounded statements:

- `voyage_063`: `soc_min=0.4536597780149933`, `final_soc=0.5301867605402605`; it has the deepest transient SOC sag, but it is not the worst net-change voyage.
- `voyage_065`: `soc_min=0.4970619651267146`, `final_soc=0.5301868382239626`.
- Recovery following high load, FC saturation, or a load drop cannot by itself prove SOC clamping.
- Short charging caused by the 48 kW/s hard FC ramp after a load drop is transient behavior, not long-term clamping evidence.

No `q_soc=17.5` result exists, so no such supplemental column will be created.

The following existing paths are protected and must remain byte-for-byte untouched by the new runner:

- `outputs/mpc_1s_n6_qsoc_feasibility/`
- `outputs/mpc_1s_n6_weight_selection/`
- `reports/mpc_1s_n6_qsoc_feasibility_summary.md`
- `reports/mpc_1s_n6_qsoc_feasibility_table.csv`
- every historical N=60, LSTM, DQN, and 10 ms path

## Architecture choice

Three approaches were considered:

1. **Selected: isolated synthetic wrapper around the verified N=6 runner.** Add `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py`, call `run_voyage()` with in-memory profiles and explicit `qsoc_candidate_config()`, and write only the new output/report paths. This preserves the already fingerprinted controller and solver implementation.
2. **Rejected: extend the existing q_soc full-voyage CLI.** Its normal lifecycle invalidates combined reports and resets candidate directories, creating an unacceptable risk of modifying retained `QSOC_10/20` evidence.
3. **Rejected: copy the OSQP rolling loop.** A second solver loop could drift in timing, bounds, scaling, recovery, or SOC-update semantics and would make the comparison scientifically weaker.

The new module owns only synthetic profile construction, case orchestration, clamping-specific metrics, plots, provenance, and report rendering. It does not modify the QP formulation, shared runner, q_soc candidate definitions, or `configs/`.

## Frozen controller contract

Both weights use the same existing explicit configuration, differing only in `candidate_id` and `model.q_soc`:

| parameter | value |
|---|---:|
| `dt` | 1 s |
| prediction/control horizon | 6 steps |
| applied action | first step only |
| forecast | synthetic future `t+1..t+6`; no LSTM |
| `P_fc_min/max` | 0 / 560 kW |
| FC hard ramp | 48 kW/step |
| `E_batt` | 693 kWh |
| `P_batt_min/max` | -346.5 / 346.5 kW |
| `P_batt_ref` | 346.5 kW |
| `SOC_ref/min/max` | 0.55 / 0.2 / 0.8 |
| `SOC_band` | 0.05 |
| `q_h2` | 0.5 |
| `q_batt` | 0.05 |
| `q_ramp` | 0 |
| `q_terminal_soc` | 0 |
| compared `q_soc` | 10, 20 |

`SOC_band=0.05` is the SOC objective normalization scale, not a deadband or hard clamp. `q_ramp=0` removes a ramp cost but does not remove the 48 kW/s hard ramp constraint.

## Synthetic cases and timing

The exact case matrix contains eight runs:

- constant 300 kW: `q_soc={10,20} × SOC_0={0.53,0.55,0.57}`
- pulse: `q_soc={10,20} × SOC_0={0.55}`

Every case has `P_fc_prev=300 kW`. The existing `run_voyage()` initializes previous FC power as the clipped first load; the wrapper must verify `loads[0]=300` and verify the first control row records `prev_fc_actual_kw=300`.

Each profile contains 3601 state samples at `0..3600 s`, producing exactly 3600 applied one-second actions. The controller decision at `t` uses `t+1..t+6`; the result row records both `decision_time_s=t` and executed-state `time_s=t+1`.

The load schedule is defined on executed-state time:

- constant: 300 kW for every sample;
- pulse: 300 kW for `0 <= time < 600`, 450 kW for `600 <= time < 720`, and 300 kW for `time >= 720`.

Thus the pulse contributes exactly 120 applied high-load samples and 5 kWh of additional load energy relative to the constant profile. Because the controller sees six future synthetic points, this is an offline ideal-foresight diagnostic, not a causal step-response experiment.

## Analysis windows

Total hydrogen and energy metrics always cover all 3600 applied steps. Exclusion windows affect only long-term clamping interpretation.

- Constant full response: all 3600 steps.
- Constant initial transient: the first 60 applied steps (`decision_time_s < 60`) are reported separately.
- Constant long-term window: `decision_time_s >= 60` (3540 steps).
- Pulse high-load window: executed-state `600 <= time_s < 720` (120 steps).
- Pulse post-drop transient: `720 <= time_s < 780` (60 steps); excluded from long-term clamping judgment.
- Pulse post-transient long-term window: `time_s >= 780`.

For optional real-voyage support, a tested steady-state mask requires a trailing 60 s load range no greater than 5 kW, at least 60 s since the latest load jump greater than 10 kW, FC not saturated at 560 kW, and exclusion of configured voyage-start/end ramp windows. A single-step `abs(delta_load)<=1 kW/s` test is not used.

## Metric semantics

The action-direction metric uses the state before the action:

```text
e_SOC(t) = SOC_before(t) - 0.55
P_correction(t) = sign(e_SOC(t)) * P_batt_actual(t)
```

Under the repository sign convention, `P_batt>0` is discharge and `P_batt<0` is charge. Therefore `P_correction>0` moves SOC toward 0.55; a negative value moves it away. At exactly zero error, `P_correction=0` even if another objective causes battery dispatch.

For each full and long-term window, report:

- `SOC_range`, `SOC_std`, `SOC_final`, `mean_abs_SOC_error`;
- the mean, p95, and maximum of the positive component `max(P_correction,0)` over all samples;
- `ratio_active_correction = mean(P_correction>5 kW and abs(e_SOC)<=0.02+1e-12)`;
- active-correction seconds and longest continuous active-correction duration;
- corrective and wrong-direction energy from the positive and negative components of `P_correction`;
- `E_fc_surplus`, battery charge/discharge/throughput, and `H2_total`;
- solver completeness and physical residual checks inherited from `build_candidate_metrics()`.

`E_fc_surplus` and battery charge energy are mathematically the same when exact power balance holds, so they are reported as correlated views and must not be counted as two independent pieces of evidence.

For constant `SOC_0=0.53` and `0.57`, reconstruct the state trajectory including `t=0` and report SOC at 300, 900, 1800, and 3600 s. The 25% and 50% recovery times are the first state times satisfying:

```text
abs(SOC(t)-0.55) <= (1-fraction) * 0.02
```

Unreached milestones are stored as null and rendered `not reached`. The report also states whether a reached milestone remains satisfied through the end, preventing a transient reference crossing from being misreported as sustained recovery.

## Relative comparisons

For constant near-reference cases, the same-`q_soc`, `SOC_0=0.55` constant case is the baseline for:

- `H2_excess`
- `battery_throughput_excess`
- `fc_surplus_energy_excess`

Matched q effects are `metric(q20)-metric(q10)` for the same profile and initial SOC. Pulse effects are compared to the same-q constant `SOC_0=0.55` baseline, and a difference-in-differences row isolates whether q20 changes the pulse increment relative to q10.

The raw pulse-minus-constant hydrogen increment is not called clamping excess because the pulse itself adds 5 kWh of physical load demand.

## Classification method

The code produces deterministic numerical evidence and behavior flags, but it does not invent thresholds for the contract's undefined qualitative words “sustained”, “material”, “clear”, “limited”, or “reasonable”. After all eight runs pass completeness and physical checks, the evidence is reviewed and written to `diagnostic_decision.json`; report generation validates that the decision uses one allowed label and includes an explicit reason for every qualitative predicate.

- `excessive_SOC_clamping` requires all five user-specified evidence groups simultaneously: sustained near-reference correction on both sides, long-term FC/load deviation, continuous charge/discharge, clearly larger q20 energy/H2 costs, and no benefit beyond tighter SOC.
- `moderate_SOC_clamping` requires faster q20 recovery on both sides plus some active correction, limited paired costs, and a physically reasonable complete pulse response, while not satisfying every excessive condition.
- `no_evidence_of_SOC_clamping` is used when neither complete evidence combination is present.

The decision additionally records wrong-direction drift, asymmetric response, overshoot, and reference crossing so that “no clamping evidence” is never misrepresented as “the controller is otherwise optimal”. If any run is incomplete, paired configs differ outside q_soc, or the qualitative evidence is not reviewed, classification status is `insufficient_evidence` and no three-way label is emitted.

## Artifacts

The runner writes only:

- `outputs/mpc_1s_n6_soc_clamping_diagnostic/run_metadata.json`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/case_metrics.csv`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/window_metrics.csv`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/comparison_metrics.csv`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/synthetic_trajectories.parquet`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/diagnostic_decision.json`
- `outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/*.png`
- `reports/mpc_1s_n6_soc_clamping_diagnostic.md`
- `reports/mpc_1s_n6_soc_clamping_metrics.csv`

At least five figures are generated: the three constant initial-SOC comparisons, the pulse comparison, and a q_soc metric comparison. Every synthetic figure contains the literal annotation `diagnostic synthetic profile` and never calls the profile a real voyage.

Run metadata records the exact case matrix, frozen configs, implementation/data hashes, runtime versions, generation ID, source Git revision, and the fact that no external load dataset or LSTM was used.

## Failure handling and tests

No result, decision, or report is accepted unless every case has exactly 3600 attempted/applied successful steps, zero physical-infeasible points, and matching q10/q20 configs outside `candidate_id` and `model.q_soc`.

Tests must first fail for the missing diagnostic API, then cover:

- exact constant and pulse arrays, including boundaries 599/600/719/720;
- exact eight-case matrix and initial SOC values;
- q10/q20 configs differing only in q_soc and identity;
- initial FC power 300 kW;
- steady-state filtering and 60 s exclusions;
- correction-power sign and near-reference active ratio;
- kW-to-kWh energy units and non-double-counting note;
- 25%/50% first-reach and sustained-recovery behavior;
- rejection of incomplete/physically invalid cases;
- synthetic labels and isolated artifact paths.

Verification is `python -m compileall src`, the focused `unittest` module, the complete `unittest` suite, `--report-only`, `git diff --check`, and an independent code/numerical review before commit and push.
