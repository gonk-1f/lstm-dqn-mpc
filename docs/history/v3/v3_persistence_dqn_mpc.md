# Experimental persistence DQN–MPC path

Archive note: the generated `outputs/v3_persistence_calibration` artifacts remain in commit `2e70008` on branch `refactor/multiscale-dqn-wmpc-v2`. Artifact paths below describe that archived revision and are absent from the direct-power branch.

The active v3 control path uses 30 s observations and a five-step economic MPC. At decision boundary `k`, it forecasts `(L_k, L_k, L_k, L_k, L_k)` and previews the LPF reference without changing the committed LPF state. The FC command is the first MPC output; the following real load is revealed later, so the battery carries its forecast error. No LSTM checkpoint is loaded on this path. The earlier LSTM study remains in `v3_predictive_control.md` as diagnostic evidence.

## DQN timing and state

The v3 DQN chooses the two independent nonnegative weights `(lambda_ref, lambda_SOC)` every **30 s**. Its persistence state has ten values:

`[SOC, L/600, LPF/600, P_FC/600, delta_P_FC/600, e_k/600, e_(k-1)/600, e_(k-2)/600, prior_lambda_ref, prior_lambda_SOC]`.

Here `e_k=L_k-L_(k-1)` is the most recent one-step persistence error, and missing startup errors are zero-padded. The five forecast entries in the older 15-value LSTM state would all duplicate `L_k`, so they are omitted. The previous experimental 15-value forecast-aware path is still available only when a forecaster is explicitly supplied. Its checkpoint is not compatible with the ten-value path; no v3 DQN policy is trained yet.

`PredictiveController.begin_voyage()` is required before the first persistence-mode observation of every voyage. It inserts a **virtual 0 kW control boundary** with no executed interval or economic charge. The first real ONBOARD load after departure is taken from data, even if it is nonzero; its forecast error is charged to the battery. Small negative source loads in the formal `[-1, 0)` kW deadband are normalized to zero. After a shore interval, physical SOC and degradation accounts carry forward, while LPF, DQN history, old weights and FC prior power reset. The controller keeps the FC at zero during shore.

## ONBOARD and shore ledgers

On ONBOARD intervals, the first predicted and actual FC commands are equal. Therefore their hydrogen and FC-degradation costs are equal. Battery power, SOC and battery degradation may differ when the next real load differs from the held-current forecast. The DQN reward uses the **actual** four-component cost ledger, not the MPC plan ledger.

Shore is outside the five-step MPC objective. When no scheduled shore event is known in the prediction window, planned shore cost is zero. On shore, no DQN action and no MPC solve are made. `finish_terminal_decision(actual_onboard_load, shore_charge_requests)` executes the last ONBOARD command, settles every 30 s shore request, then returns a single terminal transition with `done=True` and reward equal to the negative sum of the last ONBOARD and shore actual ledgers. The transition records requested and accepted charge powers. The first shore step sets FC power to zero and accounts for its shutdown transient exactly once; the formal aggregate cycle counter charges its start/stop-cycle term on the next `0 -> positive` start, not on both edges.

The frozen dataset's negative `p_batt_bus_kw` values are treated as **available battery-side charging requests**, because accepting all of several shore traces would exceed the battery's physical capacity. Accepted power is bounded by the 624 kW charge limit and the existing formal shore target SOC of 0.6. Only accepted energy changes SOC, causes battery degradation, and incurs modeled grid cost. For an accepted battery-side magnitude `P` over 30 s, `E_battery=P*30/3600` kWh and `E_grid=E_battery/0.95` kWh. The 1.10 CNY/kWh shore tariff is applied to modeled `E_grid`. Requested and accepted power traces are both returned in `ShoreSettlement`. Shore charging does not silently exceed the SOC target. The grid energy is modeled from the battery-side channel; the dataset does not provide a directly measured grid meter reading.

At the new voyage boundary, `begin_voyage()` keeps the post-shore SOC and cumulative degradation but seeds zero load, zero FC power and a fresh LPF. The next real ONBOARD sample may jump from that virtual zero. No artificial zero is written to the dataset.

## Explicit upper-level mode gate and episode replay

`v3.episode_replay.replay_episode` reads the authenticated `operating_mode` column before each physical row. `onboard` is the only route that calls the policy and solves MPC. Both `shore_pending` and `shore_charging` are shore routes: FC is zero, no policy or MPC call is made, and the negative recorded battery power is treated as a bounded charge request. `unresolved` fails before any action is taken. A shore block closes the preceding ONBOARD transition and its costs enter that terminal reward; a new ONBOARD run begins with a virtual zero-load control boundary and the actual post-shore SOC. If the dataset ends without a shore block, the last ONBOARD transition is terminal without inventing a shore charging interval.

`v3.closed_loop_screen` uses this replay for a bounded fixed-action Train pilot. Each selected case is a **complete ONBOARD run plus its following shore block**, with a logged source episode/index and the same stated initial SOC for every candidate. Runs selected from the middle of a source episode are standalone counterfactuals; earlier source-episode costs and SOC are not replayed. The pilot records actual cost components, accepted shore charge, SOC extrema, FC power changes, physical failures, and per-action solve time. It reads only Train payloads and checks the calibration artifact against the current dataset manifest hashes.

The 2026-10-06 pilot evaluated the response-screened 16 actions on two complete Train runs, each starting from SOC 0.6:

| Source run | ONBOARD / shore rows | Low-cost behavior | High-cost diagnostic |
| --- | ---: | --- | --- |
| `zero_boundary_017:621` | 7 / 24 | Actions 0, 1, 2, 3, 6, 7 and 11 leave FC off; total cost 11.76 CNY | Action 12 starts FC; total cost 393.98 CNY, of which 383.78 CNY is FC degradation |
| `zero_boundary_046:344` | 28 / 222 | Action 3 costs 513.59 CNY with one FC start | Action 0 costs 2298.20 CNY with six FC starts and 2168.38 CNY FC degradation |

All 32 case-action replays completed without battery-power or SOC bound failure; no Test payload was opened. Requested shore battery energy was 33.06 and 335.06 kWh respectively, but the 0.6 SOC cap admitted only 4.40–7.34 and 10.13–73.50 kWh across actions. Every case ended at SOC 0.6. The full per-action component costs and trajectories are in `outputs/v3_persistence_calibration/closed_loop_train_pilot.json`.

This pilot exposes an important model mismatch: the SLSQP objective uses smooth FC runtime/transient proxies and omits the formal discontinuous start charge, while the actual ledger charges about 358.65 CNY per FC start. Under low reference weights, the optimizer sometimes commands only 0.23–3.70 kW, repeatedly crossing zero and incurring six formal starts in the second run. The fixed-action rankings from two short runs are diagnostic; they do not establish a final DQN action catalog. Exact startup treatment or an explicit FC minimum-on-power rule, followed by closed-loop re-screening, is needed before large-scale DQN training. Solver time also varies greatly: this two-run, 32-action pilot took about 1008 s summed across actions.

From `src`, reproduce the pilot with `..\venv\Scripts\python.exe -m v3.closed_loop_screen`. The default catalog is the response-screened 16-action Train artifact; both default source runs and the standardized initial SOC are recorded in the result.

## Comparison with the formal v2 M5 baseline

| Aspect | Formal v2 M5 | Experimental v3 persistence |
| --- | --- | --- |
| MPC time and horizon | 30 s, 5 steps | 30 s, 5 steps |
| DQN weight hold | 5 physical steps (150 s) | 1 physical step (30 s) |
| First command information | Current interval load is supplied to the MPC solve and command | Load through `k` is supplied; the command is executed against the subsequently revealed `k+1` load |
| DQN action | Three normalized weights | Two independent nonnegative weights |
| DQN state | 90-value S8 history and mask | 10-value current physical state and load changes |
| Load forecast | Current-load persistence | Current-load persistence |
| MPC objective | Base tracking, FC smoothing, SOC deadband | Hydrogen, FC/battery degradation proxies, LPF tracking, SOC center |
| Reward | Negative actual cost aggregated over a macro interval | Negative actual cost per step; terminal reward also settles shore |

Thus the two methods differ in more than objective and reward. The underlying reward concept, `-actual economic cost`, already exists in v2; the notable differences are the information available before each command, DQN timing and where shore cost closes the replay transition. A comparison isolating the objective would require matched observation timing, state, action and DQN cadence.

## Train-only objective and action-scale calibration (2026-10-06)

`python -m v3.action_calibration` now evaluates the **same unweighted objective terms** used inside the v3 SLSQP solver. It opens only the 30 Train episodes, builds causal 180 s LPF states for ONBOARD samples, and takes 240 deterministic load-quantile decision origins. For this preliminary calibration, the reference policy sets each FC forecast power to the LPF reference clipped to 0–600 kW, starts at SOC 0.5, and approximates the previous FC command by the previous clipped LPF value. This is a scale anchor, **not a closed-loop v3 policy replay**.

The median positive five-step economic numerator is `C_nom = 8.9276331723 CNY` (239 positive cases and one zero-cost case). Its Train reference-policy p10/p90 values are approximately 1.927/32.110 CNY. The Train p95 absolute 30 s load change is 65.969 kW, yielding a material five-step reference penalty of 0.0604424. The median five-step SOC penalty from SOC 0.4 and 0.6 snapshots is 5.0. These scales give independent nominal weight bases `lambda_ref = 16.54468` and `lambda_soc = 0.2`.

An initial numerically balanced 4-by-4 grid used reference levels `{0, 4.13617, 16.54468, 66.17872}` and SOC levels `{0, 0.05, 0.2, 0.8}`. All 16 actions solved on all 20 Train pilot states, but 18 action pairs were nearly identical under a maximum 5 kW first-FC and 0.005 SOC-path difference criterion. Equal *absolute term magnitudes* therefore did not produce distinct control actions: much of the SOC penalty is a state-dependent offset that a five-step FC decision cannot change.

The current **response-screened candidate** grid keeps the reference levels and tries SOC levels `{0, 0.8, 3.2, 12.8}`. All 16 actions again solved on all 20 pilot states. Near-identical pairs fell to two; median first-FC spread across actions rose from 15.6 to 53.0 kW, with a maximum of 196.3 kW. The largest SOC weight is intentionally a strong-priority action and can make the weighted SOC term much larger than the normalized economic term in absolute value. The outcome is a candidate catalog, not a demonstrated optimal one. Closed-loop Train/Validation screening must still assess actual total cost, constraint hits, action use, and switching before DQN training or final action selection.

Reproducible artifacts: `outputs/v3_persistence_calibration/action_calibration.json` (numeric-balance grid) and `outputs/v3_persistence_calibration/action_calibration_response16.json` (response-screened grid). Both record input-manifest hashes, selected Train origins, per-action pilot outcomes, and `test_payloads_opened = 0`.

From the repository's `src` directory, reproduce them with:

```powershell
..\venv\Scripts\python.exe -m v3.action_calibration
..\venv\Scripts\python.exe -m v3.action_calibration --soc-response-factor 16 --output ..\outputs\v3_persistence_calibration\action_calibration_response16.json
```

This path remains an experimental controller and accounting contract. DQN reward-scale calibration, an end-to-end formal episode trainer, closed-loop Validation selection, and final held-out evaluation have not been completed.
