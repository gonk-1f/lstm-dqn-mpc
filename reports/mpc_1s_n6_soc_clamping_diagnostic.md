# N=6 MPC near-reference SOC clamping diagnostic

## Conclusion

**Diagnostic label: `no_evidence_of_SOC_clamping`.**

Under the fixed `dt=1 s`, `N=6`, terminal-free setup, `q_soc=20` does not force SOC deviations of only `±0.02` back to `0.55` through sustained fuel-cell overproduction and battery charging/discharging. This label addresses only the specified clamping mechanism. It does **not** mean the controller is optimal: the synthetic cases show downward drift, asymmetric response, and overshoot followed by continued drift.

All results below use an **offline ideal-foresight diagnostic synthetic profile**. They are not measured voyages, LSTM forecasts, or DQN results.

## Fixed experiment

- `dt=1 s`, `N=6`, apply only the first MPC action.
- FC: `0..560 kW`, ramp `48 kW/step`, initial FC power `300 kW`.
- Battery: `693 kWh`, `P_batt=-346.5..346.5 kW`, reference `346.5 kW`.
- SOC: reference `0.55`, bounds `0.2..0.8`, normalization band `0.05`.
- Weights: `q_h2=0.5`, `q_batt=0.05`, `q_ramp=0`, `q_terminal_soc=0`.
- Only `q_soc=10` and `q_soc=20` are compared.
- Six constant-300 kW cases plus two 300-450-300 kW pulse cases; 3,600 applied steps per case and 28,800 total.
- All eight runs completed with zero solver failures and zero physical-infeasibility points.

## Constant-load evidence

| Initial SOC | q_soc | SOC at 300 s | SOC at 900 s | SOC final | 25% / 50% first reach | Mean positive correction | Active-near-reference ratio | FC surplus | Battery throughput | H2 total |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0.53 | 10 | 0.505107 | 0.488033 | 0.483664 | not reached / not reached | 0.000 kW | 0.0000 | 0.000 kWh | 32.111 kWh | 14.1194 kg |
| 0.53 | 20 | 0.519872 | 0.516963 | 0.516912 | not reached / not reached | 0.000 kW | 0.0000 | 0.000 kWh | 9.070 kWh | 15.3810 kg |
| 0.55 | 10 | 0.515895 | 0.491347 | 0.483664 | not applicable | 0.000 kW | 0.0000 | 0.000 kWh | 45.971 kWh | 13.3852 kg |
| 0.55 | 20 | 0.524835 | 0.517100 | 0.516912 | not applicable | 0.000 kW | 0.0000 | 0.000 kWh | 22.930 kWh | 14.6407 kg |
| 0.57 | 10 | 0.534242 | 0.495347 | 0.483664 | 45 s / 86 s | 13.863 kW | 0.04694 | 0.000 kWh | 59.831 kWh | 12.6511 kg |
| 0.57 | 20 | 0.534561 | 0.517491 | 0.516912 | 45 s / 86 s | 13.863 kW | 0.04694 | 0.000 kWh | 36.790 kWh | 13.9065 kg |

Key observations:

1. At `SOC0=0.53`, neither weight produces positive correction power. The battery continues discharging, so SOC moves away from `0.55`; `q_soc=20` ends at `0.516912` and never reaches either recovery milestone.
2. At `SOC0=0.57`, q10 and q20 have the same 25% and 50% first-reach times (`45 s`, `86 s`), the same `169 s` active-near-reference duration, and the same correction-power statistics to numerical precision. Both then cross below `0.55` and continue drifting; neither first reach is sustained.
3. All six constant cases have zero FC-surplus energy and zero battery-charge energy. Therefore the observed q20 behavior is reduced battery discharge, not FC overproduction followed by battery charging.
4. Relative to q10 at the same initial SOC, q20 adds about `1.255..1.262 kg` H2 but reduces battery throughput by about `23.04 kWh`. This is a different steady power split; it is not evidence of extra charge/discharge throughput used to clamp SOC.
5. Relative to each weight's own `SOC0=0.55` baseline, the H2 changes for `SOC0=0.53` are very similar (`+0.7342 kg` at q10 and `+0.7403 kg` at q20), as are those for `SOC0=0.57` (`-0.7341 kg` for both).

## Pulse evidence and 60 s transient exclusion

The pulse adds exactly `5.0 kWh` of load demand. The first `60 s` after the load drops (`720 <= t < 780 s`) is marked as the ramp-transient region and excluded from the long-term clamping judgment.

| Metric | q_soc=10 | q_soc=20 |
|---|---:|---:|
| SOC at 720 s | 0.493479 | 0.515794 |
| SOC at 780 s | 0.492036 | 0.516166 |
| SOC final | 0.483664 | 0.516912 |
| Full-run mean positive correction | 0.000 kW | 0.837 kW |
| Active-near-reference ratio | 0.0000 | 0.0000 |
| Full-run FC surplus | 0.000 kWh | 0.837 kWh |
| Post-transient FC surplus | 0.000 kWh | 0.520 kWh |
| Battery throughput | 45.971 kWh | 24.605 kWh |
| H2 excess vs same-q constant baseline | 0.2831 kg | 0.2915 kg |

After `t=780 s`, q20 charges gently: battery power is about `-12.59 kW` at 780 s and `-5.60 kW` at 900 s, then decays toward zero. SOC rises only from `0.516166` to `0.516912`, not toward `0.55`. The q20-minus-q10 pulse H2 increment after removing their different constant baselines is only about `0.0084 kg`; the full pulse H2 increments must not be mislabelled as clamping cost because the pulse itself adds 5 kWh of demand. FC surplus and battery charge are the same power-balance transfer and are not counted as independent evidence.

## Why the label is not moderate or excessive

- **Moderate is rejected:** q20 is not faster than q10 on both sides. At `SOC0=0.53`, neither reaches 25% or 50%; at `SOC0=0.57`, their first-reach times are identical.
- **Excessive is rejected:** there is no positive correction at `SOC0=0.53`; constant cases have no FC surplus or charging; q20 does not increase total battery throughput; and the observed SOC tightening is accompanied by drift below the reference rather than sustained return to `0.55`.
- The correct diagnostic is therefore `no_evidence_of_SOC_clamping`, with behavior flags `wrong_direction_drift=true`, `asymmetric_response=true`, and `overshoot_then_drift=true`.

## Existing q_soc=20 voyage background (not rerun)

The retained seven-voyage figures remain background evidence only. Their stable-load regions visually show `P_fc` close to load and `P_batt` close to zero. `voyage_063` reaches `SOC_min=0.453660` and finishes at `0.530187`; `voyage_065` reaches `SOC_min=0.497062` and finishes at `0.530187`. These dynamics show that q20 does not hold voyage SOC at exactly `0.55`. High-load/560 kW saturation recovery and the first ramp-limited interval after a load drop are not treated as clamping evidence.

No q17.5 result existed, so none was added. The q5/q10/q20 seven-voyage study, N=60 results, and all historical outputs were left unchanged.

## Figures and files

- [Constant SOC0=0.53](../outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/constant_soc053_comparison.png)
- [Constant SOC0=0.55](../outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/constant_soc055_comparison.png)
- [Constant SOC0=0.57](../outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/constant_soc057_comparison.png)
- [Pulse comparison](../outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/pulse_comparison.png)
- [Metric comparison](../outputs/mpc_1s_n6_soc_clamping_diagnostic/plots/metric_comparison.png)
- [Case metrics CSV](mpc_1s_n6_soc_clamping_metrics.csv)
- [Synthetic trajectories](../outputs/mpc_1s_n6_soc_clamping_diagnostic/synthetic_trajectories.parquet)

## Scope and configuration status

- Formal MPC configuration modified: **false**.
- Final weight selected: **false**.
- LSTM or DQN started: **false**.
- Existing q_soc/N=60 artifacts modified or deleted: **false**.
- Verification: `python -m compileall src` passed; focused diagnostic tests were `32/32`; the complete suite was `206/206`; `git diff --check` passed. Commit/push synchronization is reported in the final handoff.
