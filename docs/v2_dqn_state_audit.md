# V2 DQN 状态空间审核

## 审核范围

本审核只使用 `operating_dataset_zero_boundary_v2/train` 的 38 个航段和 23122 个 eligible 30 s 状态点。Validation/Test 航段 CSV 打开数为 0。原始 FC/BMS 遥测由 Train parent 与时间边界双重白名单限制。本轮未运行 DQN 训练或动作筛选。

## 当前 10 维候选状态

| current_feature | decision | weight_link | reason |
| --- | --- | --- | --- |
| soc | KEEP | q_soc | 直接描述电池能量裕量。 |
| fuel_cell_power_fraction | KEEP | q_base / q_smooth / economic reward | FC 工作点影响功率分配、效率与退化。 |
| previous_fuel_cell_power_fraction | REPLACE | q_smooth | 改为 delta P_fc；与当前 P_fc 联合可精确恢复前一时刻功率，且控制语义更直接。 |
| battery_power_fraction | REMOVE | none independently | 环境中 P_batt=P_load-P_fc，是确定性冗余。 |
| load_power_fraction | REPLACE | q_base / q_smooth | 用 P_base 与 P_load-P_base 分离低频基础负荷和瞬时峰谷。 |
| recent_load_mean_fraction | REMOVE | q_base already covered | P_base 是下层控制器真实动态状态；Train Pearson=0.9983，Spearman=0.9982。 |
| recent_load_population_std_fraction | KEEP | q_smooth | 因果波动强度提供瞬时 residual 之外的信息。 |
| recent_load_window_trend_fraction | KEEP | q_base / q_smooth | 区分增载、减载与稳态，决定 FC 跟随和电池缓冲需求。 |
| causal_base_load_fraction | KEEP | q_base | 它既是 LPF 必要记忆，也是 J_base 的直接参考。 |
| recent_delta_soc | REMOVE | q_soc already covered by SOC | 它主要是电池功率的时间积分结果；与当前电池功率的 Train Pearson=-0.9288。 |

## 推荐 S7 schema

| order | feature | definition | normalization | memory |
| --- | --- | --- | --- | --- |
| 1 | soc | SOC(k) | 保持原始 fraction | 当前值 |
| 2 | causal_base_load_fraction | P_base(k) | 除以 600 kW | LPF 内部状态 |
| 3 | load_residual_fraction | P_load(k)-P_base(k) | 除以 600 kW | 当前值 |
| 4 | recent_load_population_std_fraction | [k-150 s,k] 内 P_load 的总体标准差 | 除以 600 kW | 150 s 因果历史 |
| 5 | recent_load_window_trend_fraction | [k-150 s,k] 内负荷最小二乘趋势 | trend*150 s/600 kW | 150 s 因果历史 |
| 6 | fuel_cell_power_fraction | P_fc(k) | 除以 600 kW | 当前值 |
| 7 | fuel_cell_delta_fraction | P_fc(k)-P_fc(k-1) | 除以 600 kW | 前一执行 FC 功率 |

所有尺度都是固定物理尺度，不使用 Train min-max。600 kW 是冻结的 v2 research-simulation plant rating，不是实船技术规格中的 560 kW；归一化结果允许超出 [-1,1]，不得裁剪。

## Train-only 数值证据

| feature | representation | count | missing_count | min | max | mean | std | p01 | p05 | p50 | p95 | p99 | near_zero_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soc | normalized | 23122 | 0 | 0.36725 | 0.975917 | 0.703403 | 0.129718 | 0.446684 | 0.501083 | 0.697583 | 0.91875 | 0.957233 | False |
| fuel_cell_power_fraction | normalized | 23122 | 0 | 0 | 0.728333 | 0.185903 | 0.218812 | 0 | 0 | 0.08 | 0.656667 | 0.71 | False |
| recent_load_population_std_fraction | normalized | 23122 | 0 | 0 | 0.719936 | 0.0297593 | 0.0566685 | 2.06471e-06 | 0.000731231 | 0.00916734 | 0.149497 | 0.284263 | False |
| recent_load_window_trend_fraction | normalized | 23122 | 0 | -1.8035 | 1.73048 | -0.000729415 | 0.144915 | -0.557643 | -0.164185 | -2.07036e-06 | 0.180048 | 0.511944 | False |
| causal_base_load_fraction | normalized | 23122 | 0 | 2.17072e-05 | 1.83207 | 0.391682 | 0.347097 | 9.06818e-05 | 0.0555524 | 0.264721 | 1.06666 | 1.34705 | False |
| load_residual_fraction | normalized | 23122 | 0 | -1.11909 | 0.713432 | -0.000395907 | 0.0646944 | -0.243871 | -0.0713136 | -0.000379425 | 0.0762733 | 0.215658 | False |
| fuel_cell_delta_fraction | normalized | 23122 | 0 | -0.383333 | 0.24 | 0.000105239 | 0.01378 | -0.0233333 | -0.00333333 | 0 | 0.00333333 | 0.0383333 | False |

仿真环境中的功率平衡是精确恒等式。实测重建残差最大绝对值为 1.13687e-13 kW；该量来自同源功率重建，不能作为 battery_power 独立信息的证据。

## 冗余分析

| relationship | left_feature | right_feature | model_status | pearson | spearman | measured_max_abs_residual_kw | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| environment_power_balance | battery_power_kw | load_power_kw-fc_power_kw | EXACT_IDENTITY |  |  | 1.13687e-13 | Battery power is deterministic in the simulated environment; measured residual reflects telemetry/alignment mismatch. |
| recent_load_mean_vs_causal_base | recent_load_mean_kw | base_load_kw | EMPIRICAL_CORRELATION | 0.998298 | 0.998151 |  | Train-only descriptive evidence |
| current_fc_vs_previous_fc | fc_power_kw | previous_fc_power_kw | LINEAR_REPARAMETERIZATION_WITH_DELTA | 0.998017 | 0.99731 |  | Train-only descriptive evidence |
| recent_delta_soc_vs_battery_power | recent_delta_soc | battery_power_kw | EMPIRICAL_CORRELATION | -0.928849 | -0.947182 |  | Train-only descriptive evidence |

## 运行工况区分能力

| regime | count | fraction | threshold | threshold_unit | threshold_status | mean_soc | mean_load_kw | mean_fc_kw | mean_delta_load_kw | mean_delta_fc_kw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steady_load | 11561 | 0.5 | 0.0500281 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.689787 | 178.717 | 91.1961 | -0.237631 | 0.101462 |
| load_rise | 5783 | 0.250108 | 0.0500281 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.732408 | 330.082 | 132.686 | 25.6319 | 1.06 |
| load_fall | 5778 | 0.249892 | -0.0500281 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.701617 | 251.536 | 131.088 | -26.1292 | -1.01125 |
| high_volatility | 5781 | 0.250022 | 13.1036 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.731853 | 323.582 | 145.874 | -0.599694 | -0.0544888 |
| low_soc | 24 | 0.00103797 | 0.4 | fraction | FROZEN_CONTROLLER_BOUND | 0.382462 | 343.802 | 63.0833 | -9.6402 | 3.04167 |
| high_soc | 17276 | 0.747167 | 0.6 | fraction | FROZEN_CONTROLLER_BOUND | 0.758545 | 255.366 | 114.54 | 0.550969 | 0.154376 |
| fc_low_load | 9899 | 0.42812 | 0 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.717021 | 134.597 | 0 | 0.676304 | -0.02475 |
| fc_high_load | 5892 | 0.254822 | 189 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.706682 | 417.678 | 303.627 | -1.56458 | 0.384759 |

表中的 trend、volatility 和 FC 阈值仅用于 Train 描述，不是生产策略阈值，也没有利用 held-out 数据拟合。

## 状态结构消融比较

| state_id | dimension | features | purpose |
| --- | --- | --- | --- |
| S10 | 10 | soc<br>fuel_cell_power_fraction<br>previous_fuel_cell_power_fraction<br>battery_power_fraction<br>load_power_fraction<br>recent_load_mean_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>causal_base_load_fraction<br>recent_delta_soc | current candidate |
| S7 | 7 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction | full proposed |
| S6-A | 6 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction | remove FC delta |
| S6-B | 6 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction | remove load standard deviation |
| MINIMUM | 5 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction | minimum defensible physical state |

这些比较是结构与因果信息消融。由于本轮禁止训练，不能声称 S7、S6-A 或 S6-B 的回报性能优劣。

## Markov 性审核

| memory_id | classification | state_treatment | code_evidence | reason |
| --- | --- | --- | --- | --- |
| battery_soc | PHYSICAL_TRANSITION_STATE | KEEP soc | src/v2/models/battery_energy.py:next_soc | SOC directly changes the next physical state and q_soc trade-off. |
| causal_lpf_state | CONTROLLER_DYNAMIC_STATE | KEEP causal_base_load_fraction | src/v2/control/causal_base_load.py:CausalBaseLoadFilter._observed_base_kw | The next base reference depends on the committed LPF state. |
| previous_executed_fc_power | CONTROLLER_DYNAMIC_STATE | COVER with current FC plus delta FC | src/v2/control/nonlinear_mpc.py:NonlinearMPC.solve | P_fc(k) and delta P_fc(k) recover P_fc(k-1) exactly. |
| fc_on_off_state | DEGRADATION_DYNAMIC_STATE | COVER by current FC operating state | src/v2/models/fuel_cell_degradation.py:FuelCellVoltageLossTracker.is_on | Start/stop status is observable from the executed FC on/off condition. |
| cumulative_fc_voltage_loss | REWARD_CLIPPING_STATE | CONDITIONAL: omit only with reset and far-from-EOL invariant | src/v2/models/fuel_cell_degradation.py:fuel_cell_economic_life_increment | Below EOL, marginal interval cost is independent of the cumulative level; clipping changes it near EOL. |
| cumulative_battery_weighted_ah | REWARD_CLIPPING_STATE | CONDITIONAL: omit only with reset and far-from-EOL invariant | src/v2/models/battery_degradation.py:battery_economic_life_increment | Below EOL, marginal interval cost is independent of the cumulative level; clipping changes it near EOL. |
| cumulative_start_stop_counts | DIAGNOSTIC_ACCOUNTING | OMIT | src/v2/models/fuel_cell_degradation.py:FuelCellVoltageLossTracker | Counts diagnose accumulated loss; the next increment depends on current on/off transition, not the count. |
| previous_dqn_action | POLICY_HISTORY | OMIT while no switching penalty or action-rate constraint exists | src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment.step | The environment holds the selected action for M steps but defines no reward or transition term from the prior action. |
| mpc_warm_start | NUMERICAL_SOLVER_STATE | OMIT; enforce solver robustness separately | src/v2/control/nonlinear_mpc.py:shifted_warm_start | Warm start is an optional initial guess, not a physical state; local-solution sensitivity remains a solver audit concern. |
| terminal_recharge_initial_soc | EPISODE_ACCOUNTING_STATE | OMIT only when initial SOC is frozen per episode | src/v2/economics.py:terminal_recharge_grid_energy | Terminal shore cost depends on initial minus final SOC; a variable initial SOC would need state or explicit episode context. |
| macro_step_position | MULTIRATE_EXECUTION_STATE | OMIT at DQN decision boundaries | src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment.step | The DQN is queried only at macro boundaries; the M-step countdown is internal during action execution. |
| environment_done_failed_replay_history | SOFTWARE_BOOKKEEPING | OMIT | src/v2/envs/multirate_weight_env.py:MultiRateWeightEnvironment | Done/failure gates and replay history do not define a continuing physical state presented for another action. |

只有在每个训练 episode 都重置累计退化、且 preflight 上界证明 episode 远离 EOL clipping 时，累计 FC/Battery lifetime fraction 才可省略；否则必须增加两项 clipped lifetime state。MPC warm start 不进入 DQN state，但 integrated solver robustness 必须证明不同初值不会导致实质不同的执行命令。terminal recharge 还要求冻结 episode initial SOC。

## Minimum defensible state

最小可辩护状态为 `[SOC, P_base, P_load-P_base, P_fc, delta_P_fc]`（S5）。它保留电池能量裕量、LPF 记忆、瞬时峰谷、FC 工作点与 FC 动态，但删除显式 volatility 和 trend，因此只适合作为论文消融基线，不应作为首选正式状态。

## 证据边界与局限性

实船 Train SOC 中有 6118/23122（26.46%）位于 v2 仿真硬区间 [0.20, 0.80] 之外。这些实测 SOC/FC 数据用于判断特征覆盖与区分力，不代表未来仿真策略的 state-visitation distribution。正式环境仍将依据模型转移生成 SOC 与 FC 轨迹。功率平衡残差接近零是因为 formal load 与 FC/BMS 功率同源构造，不是独立传感器验证。

## S7 明确结论

建议采用 proposed 7-dimensional state 作为 v2 正式 baseline：各特征均为因果、在 DQN 决策边界可获得、与 q_base/q_smooth/q_soc 有明确关系，并删除 S10 中确定性的 battery/load/FC 重复信息。S6-A 删除 delta_P_fc 后削弱 q_smooth 的直接动态信息；S6-B 删除 load_std 后失去与 trend 低相关的波动强度信息。因此二者仅作为消融，不优先于 S7。本审核不修改 `CANDIDATE_STATE_STATUS`；正式 freeze 仍需另行实现 schema，并落实 episode reset、EOL distance、terminal initial SOC 与 integrated solver robustness 合同。
