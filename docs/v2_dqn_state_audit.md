# V2 DQN 状态空间审核

## 审核范围

本审核只使用 `operating_dataset_zero_boundary_v2/train` 的 30 个航段和 9846 个 eligible 30 s 状态点。Validation/Test 航段 CSV 打开数为 0。原始 FC/BMS 遥测由 Train parent 与时间边界双重白名单限制。本轮未运行 DQN 训练或动作筛选。

formal ONBOARD 轴共有 18448 个点；其中 9846 个点（53.37%）具备严格因果、12 簇齐全且不复用原始行的实测 SOC proxy。无可用 proxy 行的航段为 ['zero_boundary_011', 'zero_boundary_034']。这只限制实测分布证据覆盖率，不会删除 formal 训练轴上的 ONBOARD 点；正式环境 SOC 由模型递推。

## 当前 10 维候选状态

| current_feature | decision | weight_link | reason |
| --- | --- | --- | --- |
| soc | KEEP | q_soc | 直接描述电池能量裕量。 |
| fuel_cell_power_fraction | KEEP | q_base / q_smooth / economic reward | FC 工作点影响功率分配、效率与退化。 |
| previous_fuel_cell_power_fraction | REPLACE | q_smooth | 改为 delta P_fc；与当前 P_fc 联合可精确恢复前一时刻功率，且控制语义更直接。 |
| battery_power_fraction | REMOVE | none independently | 环境中 P_batt=P_load-P_fc，是确定性冗余。 |
| load_power_fraction | REPLACE | q_base / q_smooth | 用 P_base 与 P_load-P_base 分离低频基础负荷和瞬时峰谷。 |
| recent_load_mean_fraction | REMOVE | q_base already covered | P_base 是下层控制器真实动态状态；Train Pearson=0.9980，Spearman=0.9980。 |
| recent_load_population_std_fraction | KEEP | q_smooth | 因果波动强度提供瞬时 residual 之外的信息。 |
| recent_load_window_trend_fraction | KEEP | q_base / q_smooth | 区分增载、减载与稳态，决定 FC 跟随和电池缓冲需求。 |
| causal_base_load_fraction | KEEP | q_base | 它既是 LPF 必要记忆，也是 J_base 的直接参考。 |
| recent_delta_soc | REMOVE | q_soc already covered by SOC | 它主要是电池功率的时间积分结果；与当前电池功率的 Train Pearson=-0.9252。 |
| speed_fraction | ADD | shore interlock / operating context | AIS 航速区分在航与靠泊上下文；岸电区间不进入 DQN 决策。 |

## 冻结 S8 schema

| order | feature | definition | normalization | memory |
| --- | --- | --- | --- | --- |
| 1 | soc | SOC(k) | 保持原始 fraction | 当前值 |
| 2 | causal_base_load_fraction | P_base(k) | 除以 600 kW | LPF 内部状态 |
| 3 | load_residual_fraction | P_load(k)-P_base(k) | 除以 600 kW | 当前值 |
| 4 | recent_load_population_std_fraction | [k-150 s,k] 内 P_load 的总体标准差 | 除以 600 kW | 150 s 因果历史 |
| 5 | recent_load_window_trend_fraction | [k-150 s,k] 内负荷最小二乘趋势 | trend*150 s/600 kW | 150 s 因果历史 |
| 6 | fuel_cell_power_fraction | P_fc(k) | 除以 600 kW | 当前值 |
| 7 | fuel_cell_delta_fraction | P_fc(k)-P_fc(k-1) | 除以 600 kW | 前一执行 FC 功率 |
| 8 | speed_fraction | v(k) | 除以 20 kn | 当前 AIS 航速 |

所有尺度都是固定物理尺度，不使用 Train min-max。600 kW 是冻结的 v2 research-simulation plant rating，不是实船技术规格中的 560 kW；归一化结果允许超出 [-1,1]，不得裁剪。

## Train-only 数值证据

| feature | representation | count | missing_count | min | max | mean | std | p01 | p05 | p50 | p95 | p99 | near_zero_variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soc | normalized | 9846 | 0 | 0.4255 | 0.979333 | 0.748715 | 0.125825 | 0.464992 | 0.516271 | 0.767375 | 0.942583 | 0.96975 | False |
| fuel_cell_power_fraction | normalized | 9846 | 0 | 0 | 0.728333 | 0.20905 | 0.255457 | 0 | 0 | 0 | 0.665 | 0.7 | False |
| recent_load_population_std_fraction | normalized | 9846 | 0 | 0 | 0.526825 | 0.039654 | 0.065614 | 0 | 0 | 0.0132082 | 0.186599 | 0.314868 | False |
| recent_load_window_trend_fraction | normalized | 9846 | 0 | -1.40784 | 1.86229 | 0.00169007 | 0.175599 | -0.630028 | -0.265026 | 0 | 0.268405 | 0.593844 | False |
| causal_base_load_fraction | normalized | 9846 | 0 | 0 | 1.67311 | 0.502886 | 0.379593 | 1.75694e-66 | 7.25931e-16 | 0.501191 | 1.1551 | 1.37068 | False |
| load_residual_fraction | normalized | 9846 | 0 | -0.692076 | 0.600927 | 0.000586349 | 0.0774187 | -0.285108 | -0.107749 | -4.0977e-39 | 0.109132 | 0.250805 | False |
| fuel_cell_delta_fraction | normalized | 9846 | 0 | -0.3 | 0.204679 | 0.00052961 | 0.0132608 | -0.02 | -0.00333333 | 0 | 0.005 | 0.0459167 | False |
| speed_fraction | normalized | 9846 | 0 | 0 | 0.765 | 0.287181 | 0.222741 | 0 | 0 | 0.34 | 0.605 | 0.68 | False |

仿真环境中的功率平衡是精确恒等式。实测重建残差最大绝对值为 0.980549 kW；该量来自同源功率重建，不能作为 battery_power 独立信息的证据。

## 冗余分析

| relationship | left_feature | right_feature | model_status | pearson | spearman | measured_max_abs_residual_kw | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| environment_power_balance | battery_power_kw | load_power_kw-fc_power_kw | EXACT_IDENTITY |  |  | 0.980549 | Battery power is deterministic in the simulated environment; measured residual reflects telemetry/alignment mismatch. |
| recent_load_mean_vs_causal_base | recent_load_mean_kw | base_load_kw | EMPIRICAL_CORRELATION | 0.998039 | 0.997952 |  | Train-only descriptive evidence |
| current_fc_vs_previous_fc | fc_power_kw | previous_fc_power_kw | LINEAR_REPARAMETERIZATION_WITH_DELTA | 0.998652 | 0.996756 |  | Train-only descriptive evidence |
| recent_delta_soc_vs_battery_power | recent_delta_soc | battery_power_kw | EMPIRICAL_CORRELATION | -0.92525 | -0.930843 |  | Train-only descriptive evidence |

## 运行工况区分能力

| regime | count | fraction | threshold | threshold_unit | threshold_status | mean_soc | mean_load_kw | mean_fc_kw | mean_delta_load_kw | mean_delta_fc_kw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| steady_load | 4923 | 0.5 | 0.0666665 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.748521 | 247.449 | 95.9598 | -0.0629149 | 0.319707 |
| load_rise | 2555 | 0.259496 | 0.0666665 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.772259 | 400.578 | 148.362 | 34.6007 | 1.16322 |
| load_fall | 2368 | 0.240504 | -0.0666665 | kW/s | DESCRIPTIVE_TRAIN_ONLY | 0.723713 | 309.393 | 161.953 | -35.7395 | -0.598484 |
| high_volatility | 2462 | 0.250051 | 21.4843 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.763955 | 340.519 | 152.579 | 0.119276 | 0.240085 |
| low_soc | 0 | 0 | 0.4 | fraction | FROZEN_CONTROLLER_BOUND |  |  |  |  |  |
| high_soc | 8515 | 0.864818 | 0.6 | fraction | FROZEN_CONTROLLER_BOUND | 0.782456 | 301.54 | 115.126 | 1.3494 | 0.396254 |
| fc_low_load | 5136 | 0.521633 | 0 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.772715 | 166.343 | 0 | 2.18637 | -0.0215002 |
| fc_high_load | 2482 | 0.252082 | 265 | kW | DESCRIPTIVE_TRAIN_ONLY | 0.726072 | 515.042 | 354.724 | -3.71824 | 0.623913 |

表中的 trend、volatility 和 FC 阈值仅用于 Train 描述，不是生产策略阈值，也没有利用 held-out 数据拟合。

## 状态结构消融比较

| state_id | dimension | features | purpose |
| --- | --- | --- | --- |
| S10 | 10 | soc<br>fuel_cell_power_fraction<br>previous_fuel_cell_power_fraction<br>battery_power_fraction<br>load_power_fraction<br>recent_load_mean_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>causal_base_load_fraction<br>recent_delta_soc | current candidate |
| S8 | 8 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction<br>speed_fraction | frozen formal baseline |
| S7-NO-SPEED | 7 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_population_std_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction | ablation without AIS speed |
| S7-NO-STD | 7 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>recent_load_window_trend_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction<br>speed_fraction | remove load standard deviation |
| MINIMUM | 5 | soc<br>causal_base_load_fraction<br>load_residual_fraction<br>fuel_cell_power_fraction<br>fuel_cell_delta_fraction | minimum defensible physical state |

这些比较是结构与因果信息消融。S8 是已冻结正式 baseline；无航速版本仅作为消融，不参与当前正式训练。

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

实船 Train SOC 中有 3537/9846（35.92%）位于 v2 仿真硬区间 [0.20, 0.80] 之外。这些实测 SOC/FC 数据用于判断特征覆盖与区分力，不代表未来仿真策略的 state-visitation distribution。正式环境仍将依据模型转移生成 SOC 与 FC 轨迹。功率平衡残差接近零是因为 formal load 与 FC/BMS 功率同源构造，不是独立传感器验证。

## S8 最终结论

冻结的八维 S8 与生产 `FORMAL_STATE_FEATURE_NAMES` 完全一致。前七维保留 SOC、LPF 记忆、负荷残差/波动/趋势及 FC 工作点动态；`speed_fraction` 提供 AIS 在航上下文。DQN 只在 ONBOARD 决策边界读取 S8，shore_pending/shore_charging 会重置控制历史并暂停 DQN/MPC。累计退化账户逐 episode 重置；保守上界为 FC=0.770930、battery=0.034480，均低于 EOL=1，因此 clipped lifetime 在当前 formal episode 内不可达，累计退化无需进入 S8。
