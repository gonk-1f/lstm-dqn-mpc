# v2 formal-training preflight report

## 1. cleanup summary — VERIFIED

v1 方法说明与诊断文档已移入 `docs/archive_v1/`，删除/保留范围记录在
`docs/v2_cleanup_manifest.md`。该状态只证明代码与文档边界已清理，不证明 v2
数据或模型已完成标定。

最终对比 v1 基线 commit `ba81281da8236f0816d352a51ccfd7d176baabaf`：共删除
852 个已跟踪文件，全部位于清单列出的 15 个 v1 `outputs/` 目录；另有
2 个忽略的 v1 输出树和 2 个空占位目录按清单处理。10 个已跟踪文档以
100% rename 移入 `docs/archive_v1/`，另保留并归档 1 个原未跟踪的论文复核文档。
完整路径、单目录文件数、恢复 commit 与本地 recovery cache 位置均以
`docs/v2_cleanup_manifest.md` 为权威记录；未发现清单外的已跟踪文件删除。

## 2. code architecture — VERIFIED

当前实现按 `data`、`models`、`control`、`dqn`、`envs`、`economics`、
`analysis`、`training` 和 `main` 分层。`src/v2/contracts.py` 固定八个版本标识；
v2 artifact 必须精确匹配这些语义，v1 checkpoint/replay 不能恢复。

## 3. exact mathematical model — VERIFIED

下层目标版本为 `fc_base_smooth_soc_deadband_mean_v2`：三项正权重分别约束 FC 对因果
基准负载的跟踪、FC 功率变化和平滑 SOC deadband 罚项。上层版本为
`macro_interval_real_economic_cost_v1`，宏区间账本只累计氢耗、FC 退化、电池
退化和实际发生的岸电成本，reward 为负的原始 CNY 总成本：

`J = q_base*J_base + q_smooth*J_smooth + q_soc*J_soc`，

`J_base = (1/N) sum_i ((P_fc[i]-P_base_hat[i])/600)^2`，

`J_smooth = (1/N) sum_i (Delta P_fc[i]/600)^2`，

`J_soc = (1/N) sum_i phi(SOC[i])`，其中 `phi` 是 `[0.40,0.60]` 外到最近
边界距离除以 `0.60` 后的平方、区间内为零；SOC 物理硬边界仍为 `[0.20,0.80]`；

`reward_macro = -(C_H2 + C_FC_deg + C_Batt_deg + C_shore_if_incurred)`。

功率平衡为 `P_batt_bus[i] = P_load_hat[i] - P_fc[i]`，正值表示放电。精确定义
与单位见 `docs/method_v2_multiscale_dqn_wmpc.md` 和
`docs/v2_economic_parameters.md`。
“VERIFIED”只表示代码合同已明确，不表示所有数值校准已完成。

## 4. raw-data support — NO-GO

objective-scale audit 已从冻结清单中的 46 个 Train parent 读取原始时间戳遥测，
并形成 1,211 个严格合规 supervisory states；这足以支持该窄范围审计。仓库仍未
形成可用于完整正式训练的 episode/state/reward payload 合同与全链路证据，因此本项
继续 **NO-GO**。不得读取 Validation/Test 或使用历史处理结果弥补正式训练缺口。

## 5. FC efficiency source — VERIFIED

`eta_fc(P)` 来自用户授权 `FC_Data.xlsx` 的 `Sheet1!A2:B12`，文件 SHA-256、11
个原始点和横向 `100 kW -> 600 kW` 映射已冻结在
`docs/v2_fc_efficiency_model.md`。效率只从百分数转为比例，不乘以 6，域外不外推。

## 6. Battery efficiencies source — VERIFIED

`eta_chg = eta_dis = 0.95`，来源为 DOI
`10.11930/j.issn.1004-9649.202507065` Table 3。来源、位置和值必须同时匹配；
这两个效率只进入能量/SOC 动力学，不进入退化归一化。

## 7. degradation equations — MIXED

FC 四工况电压损失结构和电池 SOC/电流加权吞吐方程已编码并带来源，分别见
`docs/v2_fc_degradation_model.md` 与 `docs/v2_battery_degradation_model.md`。
FC 经济归一化采用 70,000 microvolt aggregate-equivalent EOL，状态为
**VERIFIED literature/model**，不是 vessel-measured。电池采用
`15000 * (624000/432) Ah`；configuration status 为 **FROZEN**，evidence status
仍为 **SECONDARY_LITERATURE / LITERATURE-CALIBRATED**，不是 vessel measured、
manufacturer specification 或 Yang measured parameter。
两者均按 before/after clipped cumulative fraction 的差值生成 interval CNY，跨越
EOL 只补剩余寿命，EOL 后不重复收费。FC 原始功率到来源参考单元的映射适用性仍未
解决，但不再冒充经济归一化 blocker。

## 8. economic prices — VERIFIED

固定场景采用氢气 `35 CNY/kg`、FC `3500 CNY/kW`、电池 `2000 CNY/kWh` 和
岸电 `1.10 CNY/kWh`。来源角色见 `docs/v2_economic_parameters.md`。价格已冻结
且 interval degradation cost 已可按第 7 节合同计算。Battery lifetime factor 已冻结
为 formal baseline，但其 secondary-literature evidence classification 保持不变。

## 9. shore-data status — VERIFIED ASSUMPTION / NOT MEASURED

`1.10 CNY/kWh` 是 `scenario_not_measured` 的峰时电价场景，不是项目码头实测
电价。终端补能使用一次 aggregate `0.95` 文献假设，不再叠加第二个岸电变换器
效率。该代码合同为 **VERIFIED literature-based aggregate assumption**，但不代表
实船岸电计量或合同电价。

## 10. plant parameter provenance — VERIFIED

`docs/v2_plant_configuration.md` 明确分离 600 kW/624 kWh 研究仿真配置与
560 kW/约 1806 kWh 的真实船舶规格记录。两组值不得拼接为同一正式设备；
当前“VERIFIED”仅指来源分类和分离规则可追溯。

## 11. MPC Ts / N configuration — FROZEN

`Ts_MPC=30 s` 已冻结为 nominal control interval，并由 Train 原始时钟审计支持，
状态为 **VERIFIED**。`N=5` 已冻结为 `FROZEN_PROJECT_DESIGN`，每次 solve 预测
5 个 30 s supervisory steps，即 150 s；它不是文献证明或全局优化得到的唯一最优值。

## 12. DQN M configuration — FROZEN

`M=5` 已冻结为 `FROZEN_PROJECT_DESIGN`：同一个 DQN-selected action 保持 5 次
真实 rolling MPC solve，即 150 s macro interval。`M` 与 MPC 内部 prediction horizon
`N` 语义独立。历史 Train-only `M in {5,10}` 诊断不构成最优性证明，也不再阻塞
formal training。

## 13. candidate action behavior analysis — NO-GO

36 个正十分位三权重候选已确定性生成，但没有在可用真实 Train 工况上完成可行性、
行为向量、Pareto、去重和聚类链。合成单元测试不构成候选行为证据。

`docs/v2_objective_scale_audit.md` 已使用 46 个清单内 Train parent 构建 1,211 个
合规 supervisory states，并对 6 个代表 case 与完整 36 个候选 action 完成 216 次
实际求解。active-P95 量级比为 `1.827863`，因此
`objective_scale_comparability` gate 为 **VERIFIED**。但最大有序 dominance 比例
达到 `46.76%`，属于明确的局部行为警告；该结果不等同于最终 action catalog 的
行为筛选，candidate action behavior analysis 仍为 **NO-GO**。

## 14. final K — NO-GO

`FINAL_DQN_ACTION_CATALOG` 仍为 `None`，最终代表数 `K` 未冻结。候选库不能作为
最终 DQN action catalog，也不能通过手填 K 绕过筛选证据。

## 15. solver robustness — NO-GO

配对 cold/warm 审计接口已定义成功率与仅双方成功案例上的速度比，但没有覆盖真实
Train case 和最终 catalog 的重复求解记录。因此不能声称热启动更快、更可靠，
也不能把接口测试当作 solver audit。

## 16. unit/contract test results — VERIFIED

2026-09-22 本次参数冻结收口后，当前工作树已通过：focused parameter/contract
tests `58/58`、全部 v2 tests `208/208`、N=5 物理约束/首步执行与
cold/shifted-warm 确定性 solver smoke `2/2`、compile/import 和
`git diff --check`。真实审计使用原始 Train 遥测并完成 216 次 MPC 求解；
合成单元测试仍只用于证明统计、门禁和数据规则实现，不替代该真实结果。

## 17. remaining unsupported assumptions — UNRESOLVED

尚未获得或冻结：正式 Train dataset/episode payload、FC 聚合功率到来源参考单元的
raw-model 映射、原船/正式训练适用的电池充放电功率边界、
FC 每步爬坡限制、最终 state、reward scale、最终 action catalog/K、
最终 catalog 的完整真实案例 solver robustness audit，以及独立的完整
behavioral-responsiveness 筛选。objective-scale 与 weighted-dominance 的 Train-only
结果已经归档；不得用其替代仍缺失的证据，也不得用研究仿真值、合成夹具或
held-out 表现补齐这些门槛。

`tau_LPF=90 s` 已冻结为 `FROZEN_PROJECT_DESIGN`，在 `Ts=30 s` 下
`alpha=exp(-30/90)`；它有 LPF/FC-low-frequency 文献结构支持，但不是实船标定值或
唯一最优值。FC hard ramp 允许 disabled，未使用旧 `48 kW/step`。审计 battery bounds `[-624,+1248] kW` 来自
Yang et al. (2026) Table 6 的研究配置，不能提升为原船或正式训练硬件边界。

## 18. formal training GO/NO-GO — NO-GO

代码在读取训练 payload 前检查 15 项：`eta_fc(P)`、`eta_chg`、`eta_dis`、FC 退化
归一化、电池 `Q_lifetime`、岸电充电效率、岸电价格、`Ts_MPC`、`N`、`M`、
`tau_LPF`、SOC deadband、最终 DQN state、最终 action catalog，以及
`objective_scale_comparability`。其中 battery lifetime、N、M 与 tau 已按冻结配置
通过 gate，但 evidence classification 仍独立保留。最终 state/action 仍非
`VERIFIED`；正式 dataset/episode payload 与最终集成 solver robustness 也未关闭，
结论为：

`FORMAL_TRAINING = NO-GO`

本报告没有执行真实训练或 Validation/Test 评估，也没有执行最终 catalog 的正式
solver robustness audit。
