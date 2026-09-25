# v2 formal-training preflight report

更新日期：2026-09-25。当前配置与集成实现状态为：

`FORMAL_TRAINING = NO-GO`

NO-GO 的当前原因是岸电模式证据尚未全部闭合，不是时间尺度、退化归一化、
S8 或36动作未冻结。

## 已冻结配置

- `Ts=30 s`：verified nominal control interval；
- `N_MPC=5`：`FROZEN_PROJECT_DESIGN`，150 s prediction horizon；
- `DQN_SWITCH_STEPS=5`：`FROZEN_PROJECT_DESIGN`；仅计算真实 ONBOARD rolling
  MPC solves，岸电物理步不计入 M；
- `TAU_LPF_SECONDS=90`：`FROZEN_PROJECT_DESIGN`；
- battery lifetime factor `15000`：配置冻结，证据仍为
  `SECONDARY_LITERATURE / LITERATURE-CALIBRATED`；
- FC lifetime normalization：70,000 microvolt aggregate-equivalent EOL，
  `VERIFIED literature/model`，不是 vessel-measured；
- DQN state：ONBOARD-only S8 `FROZEN_PROJECT_BASELINE`；
- DQN actions：完整 36-action positive tenth-grid catalog，
  `FROZEN_PROJECT_BASELINE`，不是筛选后全局最优集合；
- formal economic return：`gamma=1.0`，有限 episode 未折扣总 CNY。

## 数据与 mode sidecar

正式 power 数据与 AIS sidecar 保持 38 Train、10 Validation、5 Test。新增
`operating_dataset_zero_boundary_v2_modes`，在完全相同的30 s 时间轴上保留：

- 8 FC 聚合功率；
- 12 BMS 聚合电池母线功率，放电为正、充电为负；
- frozen `load_total_kw` 与组件重构残差；
- AIS speed 与 provenance；
- completeness、freshness、duplicate conflict；
- `ONBOARD`、`SHORE_PENDING`、`SHORE_CHARGING`、`UNRESOLVED`。

岸电候选要求 AIS 近静止、FC 总功率处于8 kW项目停机容差内、电池充电超过
1 kW，且数据质量有效。连续3个30 s候选点确认岸电；前两个点以因果
`SHORE_PENDING` 暂停控制。短候选段、质量失败或负总功率但复合证据不充分的点
保持 `UNRESOLVED`。

当前真实计数：

- Train：28,041 ONBOARD，32 SHORE_PENDING，2,151 SHORE_CHARGING，
  685 UNRESOLVED；
- Validation：4,946 ONBOARD，10 SHORE_PENDING，1,081 SHORE_CHARGING，
  230 UNRESOLVED；
- Test：2,765 ONBOARD，0 UNRESOLVED。

因此不能把所有负总功率点自动改写成岸电。当前5,652个 Train ONBOARD macro
候选只是基于已知模式的诊断量；unresolved 审核后必须重新计算最终训练量。

## S8 与事件驱动 transition

S8 顺序为：

`[SOC, base/600, residual/600, std/600, trend*150/600, FC/600, deltaFC/600, speed/20]`

只在 ONBOARD 决策边界构造。岸电期间 DQN action、MPC solve、epsilon、global
step、replay 和 gradient update 均暂停；SOC、岸电费用、电池退化及必要的 FC
停机退化继续逐30 s更新。岸电结束后清空 LPF 与 onboard load history，保留 SOC
和累计退化，并以当前负荷/航速构造因果 cold-start S8。

一个 action 最多执行5次 ONBOARD MPC。如果第3次后进入岸电，transition 的
`executed_mpc_steps=3`；岸电期间全部增量 ledger 归入该 transition，在下一次
ONBOARD 或 episode 结束时闭合。

## 经济与退化合同

reward 是氢耗、FC经济寿命增量、电池经济寿命增量和岸电费用的负原始 CNY
总和。FC 与 battery 均使用 clipped cumulative fraction 的 before/after 差值，
跨 EOL 只补剩余寿命，EOL 后继续保留 raw diagnostics 但不重复收 replacement
cost。

岸电原始 BMS 曲线作为电池侧充电能力轨迹；实际吸收功率受仿真 SOC、0.60
目标、0.80硬上限和区间长度约束。grid energy 为实际电池侧能量除以一次
`eta_chg=0.95`，再按1.10 CNY/kWh计费。

## 当前门禁与允许命令

```powershell
$env:PYTHONPATH="src"
python -m v2.main.train_formal_dqn --preflight-only
python -m v2.main.train_formal_dqn --smoke-only
```

`--preflight-only` 应返回退出码2和 `FORMAL_TRAINING=NO-GO`；`--smoke-only`
只执行有界事件链、无梯度且不写正式 checkpoint。685/230 unresolved mode 点完成
审核或采用明确、可审计的排除/分段政策之前，不得运行正式长训练。

Train、Validation 与 Test payload 均已建立并保持严格隔离；其中 Test payload
不参与模式阈值确定、状态/动作设计、训练或超参数选择。当前 NO-GO 仅表示
Train/Validation 的 unresolved mode 证据尚未收口，不得把 Test 无 unresolved
误解为可以绕过该门禁。
