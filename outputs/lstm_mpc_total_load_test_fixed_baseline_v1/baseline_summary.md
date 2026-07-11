# Clean Fixed LSTM-H2-MPC Baseline v1

## Configuration

- Weight set: `dp0_total_load_qsoc400_qbatt003_qramp2e-5`
- Weights: `{"q_h2": 1.0, "q_soc": 400.0, "q_fc": 0.0, "q_batt": 0.03, "q_ramp": 2e-05, "q_terminal_soc": 0.0}`
- Load definition: `fuel_cell_total_kw + battery_total_kw`
- LSTM checkpoint: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\lstm_total_load_721\checkpoints\best_lstm_load_predictor.pt`
- Split file: `C:\Users\20883\OneDrive\Desktop\microgrid-mpc-master\outputs\config\voyage_split_total_load_721.json`
- Battery capacity: `1806.0 kWh`
- Fuel-cell hard ramp constraint enabled: `False`
- SOC limits: `0.2` to `0.8`
- SOC reference mode: `initial_soc`

## Timing

- Control timing: `one_step_ahead_lstm_mpc`
- Control apply timing: `execute_cached_previous_mpc_command`
- When LSTM is available, MPC load reference is exactly `[pred_h1, pred_h2, pred_h3, pred_h4, pred_h5, pred_h6]`.
- Before LSTM history is available, MPC uses a six-step current-load hold.
- The measured current load is used for cached-command execution and SOC update, not as MPC stage 0 when LSTM is available.

## Objective

```text
J = sum_k [
    q_h2   * mH2_kg[k]
  + q_soc  * (SOC[k+1] - SOC_ref)^2
  + q_batt * E_batt_kwh[k]
  + q_ramp * (P_fc[k] - P_fc[k-1])^2
]
```

The Dp0 hydrogen term uses the imported fresh fuel-cell curve. No reserve penalty, terminal SOC penalty, SOC deadband, normalized objective, or rule-based load/SOC limits are active in this fixed baseline.

## Test Voyages

- `voyage_060`
- `voyage_061`
- `voyage_062`
- `voyage_063`
- `voyage_064`
- `voyage_065`
- `voyage_066`

## Forecast Metrics

```text
voyage_id   RMSE_h1    MAE_h1   WAPE_h1    RMSE_h6    MAE_h6   WAPE_h6
        1 50.973502 20.641144  9.352964  97.705845 46.898249 20.936285
        2 18.920725 10.913430  5.933646  33.135866 18.910141 10.130713
        3 50.900548 21.522002 11.246557  96.209373 52.867821 27.271501
        4 45.068539 20.684148  7.726811  72.884461 40.403424 15.022121
        5 28.545486 13.945842  9.257952  73.538101 39.147813 26.001650
        6 46.221925 19.917893 11.314795 102.101810 54.261174 30.372280
        7 26.166835 13.028738  6.731249  72.472057 37.066402 18.869424
      all 41.659162 18.194507  8.403912  79.617153 41.202687 18.834018
```

## Closed-Loop Metrics

```text
 voyage_id  duration_h  SOC_start  SOC_end  SOC_min  SOC_max  SOC_delta  H2_total_kg  charge_sustaining_adjusted_H2  battery_throughput_kwh  fc_above_load_energy_kwh  fc_below_load_energy_kwh  fc_load_tracking_mae  fc_ramp_mean_kw  fc_ramp_max_kw  solver_success_rate
         1    2.966667       0.55 0.525491 0.518096 0.550118  -0.024509    30.863807                      33.230560              127.900655                 41.754059                 86.146596             43.112580         8.146554       60.754006                  1.0
         2    2.991667       0.55 0.529610 0.524589 0.550046  -0.020390    25.566505                      27.504417               70.358838                 16.766850                 53.591988             23.518275         5.221311       34.433464                  1.0
         3    2.966667       0.55 0.527216 0.519002 0.550005  -0.022784    26.017555                      28.167102              131.222465                 45.037207                 86.185258             44.232292         8.170866       60.179292                  1.0
         4    9.000000       0.55 0.528541 0.491190 0.550026  -0.021459   126.869714                     128.979748              305.093398                133.095955                171.997444             33.899266         6.549334       55.800790                  1.0
         5    1.991667       0.55 0.529026 0.523927 0.550003  -0.020974    12.714048                      14.704095               68.814001                 15.373415                 53.440586             34.550963         6.174129       59.818030                  1.0
         6    2.991667       0.55 0.524434 0.510796 0.550050  -0.025566    24.493835                      26.984196              141.366301                 47.626746                 93.739555             47.253360         7.322101       59.459322                  1.0
         7    2.991667       0.55 0.526325 0.518332 0.550048  -0.023675    27.350076                      29.655443              101.689298                 29.465740                 72.223558             33.990852         6.785359       51.335249                  1.0
```

## Objective Breakdown

```text
 voyage_id  h2_mass_kg_sum  batt_throughput_kwh_sum  weighted_h2_cost_sum  weighted_soc_cost_sum  weighted_batt_cost_sum  weighted_ramp_cost_sum  total_objective_sum
         1      180.264172               538.378925            180.264172             480.442604               16.151368                2.709375           679.567519
         2      149.995247               369.607764            149.995247             395.652687               11.088233                0.839664           557.575832
         3      151.241398               534.245644            151.241398             429.039891               16.027369                2.482029           598.790687
         4      738.054733              1172.319708            738.054733            2748.502378               35.169591                6.567911          3528.294613
         5       73.500610               379.963861             73.500610             253.190011               11.398916                1.256106           339.345643
         6      142.194363               553.058562            142.194363             587.834629               16.591757                2.640497           749.261246
         7      159.270269               497.697868            159.270269             486.608755               14.930936                2.400482           663.210442
```

## Fixed Baseline Selection Note

`q_soc=400, q_batt=0.030, q_ramp=0.00002` is retained as the current fixed baseline because it reduced the excessive fuel-cell share observed in higher-SOC-weight runs while preserving charge-sustaining behavior better than lower battery-throughput penalties. This is a fixed-weight baseline, not evidence for dynamic-weight KAN-DQN.

## Limitations

Fixed weights cannot adapt to voyage-dependent operating phases. Remaining fuel-cell over/under-production and SOC drift must be handled by objective-weight design or the future dynamic weighting layer, not by if-load or if-SOC rules.

## Future Dynamic Weighting

The future SineKAN-DQN stage should adjust MPC objective weights online while keeping this same physical MPC interface: LSTM h1-h6 forecast in, total P_fc/P_batt reference out.

A clean literature-consistent LSTM-H2-MPC baseline has been restored, using only physical constraints and objective-function-based optimization terms.
