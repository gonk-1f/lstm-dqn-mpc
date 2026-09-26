# v2 formal-training preflight report

更新日期：2026-09-26。当前配置、数据与集成门禁状态为：

`FORMAL_TRAINING = GO`

## 已冻结合同

- `Ts=30 s`，`N_MPC=5`（150 s prediction horizon）；
- `DQN_SWITCH_STEPS=5`（同一 action 最多保持 5 次真实 ONBOARD MPC solve）；
- `TAU_LPF_SECONDS=90`；
- ONBOARD-only AIS-aware S8；
- 完整 36-action positive tenth-grid catalog；
- episode return 使用 `gamma=1.0`；正常 transition 的 learning reward 为真实
  interval CNY 增量成本之负值；
- 可证明的物理不可行终止当前 episode，另加 50,000 分失败惩罚；该分数不是 CNY，
  不进入 formal economic ledger；
- 岸电期间暂停 DQN/MPC，FC 强制为 0，SOC、电池退化和岸电成本继续更新。

## 当前数据身份

- Power：30 Train / 8 Validation / 5 Test；
- Train：23,590 个 30 s 点，其中 18,448 ONBOARD、88 SHORE_PENDING、
  5,054 SHORE_CHARGING、0 UNRESOLVED；
- Validation：3,261 ONBOARD、28 SHORE_PENDING、1,334 SHORE_CHARGING、
  0 UNRESOLVED；
- Test：2,728 ONBOARD、6 SHORE_PENDING、31 SHORE_CHARGING、0 UNRESOLVED；
- Train macro transitions：3,721。

岸电分类仅依赖质量有效的 AIS 近零航速与 BMS 充电证据；原始 FC 只保留为诊断，
不作为岸电 gate。ONBOARD 的 `[-1,0) kW` 数值死区在进入状态、LPF 和 MPC 前归零，
小于 `-1 kW` 则 fail closed。

## S8 状态审计

冻结顺序为：

`[SOC, base/600, residual/600, std/600, trend*150/600, FC/600, deltaFC/600, speed/20]`

审计包绑定当前 power/AIS/mode 三份 manifest、30 个 Train segment 的 SHA-256、
生产 schema digest 及全部审计 artifact hash。Validation/Test payload 未用于状态审计。

严格的 12 簇 BMS 因果 SOC proxy 覆盖 9,846/18,448 个 ONBOARD 点和 28/30 个
Train 航段；缺少 proxy 的 `zero_boundary_011`、`zero_boundary_034` 仍完整保留在
formal 训练轴中，因为正式环境 SOC 由模型递推，不由该 proxy 驱动。

累计退化未进入 S8 的条件也已闭合：逐 episode 重置后，保守最坏上界为
FC raw lifetime fraction `0.770929631`、battery `0.03448025`，均低于 EOL=1，
因此当前 episode 内不会进入 clipped-cost 的隐藏状态区。

## Objective 与 solver 门禁

当前 30-Train-segment objective audit 使用 6 个确定性代表工况 × 36 actions =
216 次 solve；active-P95 `scale_ratio=1.855794906`，状态为 PASS。结果 digest 与
当前 sample/source manifests 均由 preflight 认证。

Train-only terminal-failure audit 以固定动作 `w_8_1_1` 运行全部 30 个 Train
航段：29 个完成，`zero_boundary_015` 因硬 SOC 可达性失败；已完成航段最大原始
经济成本为 `20,779.575664249 CNY`（`zero_boundary_046`）。50,000 分失败惩罚
高于该参考最大值两倍，证据分类为 `DERIVED_TRAIN_ONLY / PROJECT_DESIGN`。
审计未打开 Test payload，并由当前 power/AIS/mode manifests、36-action digest 和
结果 digest 共同认证。

正式入口的 live preflight 还会：

- 认证 Train/Validation power、AIS、mode payload；
- 验证所有 Train S8 有限且维数为 8；
- 对首、中、末三个动作执行可重复性 solver 检查；
- 确认 Test payload 未被训练/模型选择打开。
- 认证 terminal-failure penalty 的 Train-only 标定及其独立于经济 ledger 的语义。

## 运行边界

预检：

```powershell
$env:PYTHONPATH=(Resolve-Path "src").Path
python -X utf8 -u -m v2.main.train_formal_dqn --preflight-only
```

集成 smoke（无梯度、无 checkpoint）：

```powershell
python -X utf8 -u -m v2.main.train_formal_dqn --smoke-only
```

正式训练入口独立于旧 84-action 脚本；Train 每轮用固定 seed 重新打乱，
Validation 不打乱且不训练，Test 不参与训练或模型选择。
