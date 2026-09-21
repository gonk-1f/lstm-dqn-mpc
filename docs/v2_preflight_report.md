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

`docs/v2_raw_excel_inventory.md` 与 `docs/v2_data_provenance.md` 没有登记可用于正式
v2 工况循环的、已审计且带时间戳的原始 Train 测量。现有候选文件不能因扩展名或
历史处理结果自动升级为 raw fact；不得读取 Validation/Test 来弥补此缺口。

## 5. FC efficiency source — VERIFIED

`eta_fc(P)` 来自用户授权 `FC_Data.xlsx` 的 `Sheet1!A2:B12`，文件 SHA-256、11
个原始点和横向 `100 kW -> 600 kW` 映射已冻结在
`docs/v2_fc_efficiency_model.md`。效率只从百分数转为比例，不乘以 6，域外不外推。

## 6. Battery efficiencies source — VERIFIED

`eta_chg = eta_dis = 0.95`，来源为 DOI
`10.11930/j.issn.1004-9649.202507065` Table 3。来源、位置和值必须同时匹配；
这两个效率只进入能量/SOC 动力学，不进入退化归一化。

## 7. degradation equations — NO-GO

FC 四工况电压损失结构和电池 SOC/电流加权吞吐方程已编码并带来源，分别见
`docs/v2_fc_degradation_model.md` 与 `docs/v2_battery_degradation_model.md`。
但 FC 缺少适用的单电池初始电压及聚合功率映射，电池缺少权威
`Q_lifetime`；两类 raw loss 都不能转换为正式寿命比例或 CNY。

## 8. economic prices — VERIFIED

固定场景采用氢气 `35 CNY/kg`、FC `3500 CNY/kW`、电池 `2000 CNY/kWh` 和
岸电 `1.10 CNY/kWh`。来源角色见 `docs/v2_economic_parameters.md`。价格已冻结
不等于退化成本可计算；后者仍受第 7 节归一化 gate 阻止。

## 9. shore-data status — NO-GO

`1.10 CNY/kWh` 是 `scenario_not_measured` 的峰时电价场景，不是项目码头实测
电价。仓库也没有已批准的岸电变换器效率，因此终端补能量的正式计算继续失败关闭。

## 10. plant parameter provenance — VERIFIED

`docs/v2_plant_configuration.md` 明确分离 600 kW/624 kWh 研究仿真配置与
560 kW/约 1806 kWh 的真实船舶规格记录。两组值不得拼接为同一正式设备；
当前“VERIFIED”仅指来源分类和分离规则可追溯。

## 11. MPC Ts / N evidence — PROVISIONAL

`Ts_MPC=30 s`、`N=5` 只是用户批准的待审计基线。当前没有真实 Train 时间序列
证据证明相关长度或求解鲁棒性支持该组合，不能称为正式选择。

## 12. DQN M evidence — PROVISIONAL

Train-only 敏感性范围严格限定为 `M in {5,10}`，且 `N=5` 固定不联动。
`src/v2/analysis/timescale_audit.py` 只生成诊断快照，不填写正式 M；当前不能声称
`M=5` 或 `M=10` 更优。

## 13. candidate action behavior analysis — NO-GO

36 个正十分位三权重候选已确定性生成，但没有在可用真实 Train 工况上完成可行性、
行为向量、Pareto、去重和聚类链。合成单元测试不构成候选行为证据。

`docs/v2_objective_scale_audit.md` 定义了 Train-only actual-solve 接口、active-P95
量级比、`0.1*J_i > 0.7*J_j` 支配统计和 behavioral redundancy 检查。由于没有
可接受的真实 Train 工况和最终 catalog，当前所有数值结果均为 `N/A`，新的
`objective_scale_comparability` gate 为 **NO-GO**。

## 14. final K — NO-GO

`FINAL_DQN_ACTION_CATALOG` 仍为 `None`，最终代表数 `K` 未冻结。候选库不能作为
最终 DQN action catalog，也不能通过手填 K 绕过筛选证据。

## 15. solver robustness — NO-GO

配对 cold/warm 审计接口已定义成功率与仅双方成功案例上的速度比，但没有覆盖真实
Train case 和最终 catalog 的重复求解记录。因此不能声称热启动更快、更可靠，
也不能把接口测试当作 solver audit。

## 16. unit/contract test results — VERIFIED

2026-09-21 本增量的 objective/MPC、objective-scale audit、Train-only guard 和
preflight 定向集合为 `47/47` 通过；其中新增 objective-scale audit 为 `6/6`。
最终全仓套件只运行一次，`381/381` 通过，用时 `143.057 s`。

求解器 smoke 单独复核 `N=5` 物理约束/首步执行和显式 cold/shifted-warm
确定性，`2/2` 通过，用时 `0.643 s`。`python -m compileall -q src tests`、导入
边界扫描和 `git diff --check` 均通过。合成 objective audit 只证明统计和门禁实现，
不构成真实 Train objective-scale 数值结果。

## 17. remaining unsupported assumptions — UNRESOLVED

尚未获得或冻结：可用真实 Train 工况、FC 聚合功率到来源参考单元的映射、FC 寿命
归一化、电池 `Q_lifetime`、岸电变换效率、正式 `Ts_MPC/N/M/tau_LPF`、电池充放电
功率边界、FC 每步爬坡限制、最终 state、reward scale、最终 action catalog/K、
完整真实案例 solver audit，以及真实 Train objective-scale/weighted-dominance/
behavioral-responsiveness 结果。不得用研究仿真值、合成夹具或 held-out 表现替代这些证据。

## 18. formal training GO/NO-GO — NO-GO

代码在读取训练 payload 前逐项检查原需求实际列出的全部 13 项：`eta_fc(P)`、
`eta_chg`、`eta_dis`、FC 退化归一化、电池 `Q_lifetime`、岸电价格、`Ts_MPC`、
`N`、`M`、`tau_LPF`、SOC deadband、最终 DQN state、最终 action catalog。
本增量另增加第 14 项 `objective_scale_comparability` gate。当前有任一非
`VERIFIED` 项即失败关闭，结论为：

`FORMAL_TRAINING = NO-GO`

本报告没有执行真实训练、Validation/Test 评估或真实工况 solver audit。
