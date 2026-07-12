# Project Status

> 当前状态唯一入口。`thread.md`、`project_status.md` 和 `next_steps.md` 是历史上下文，不代表全部当前事实。

## Active framework

目标论文框架为：66 航段船舶负荷 -> LSTM direct multi-output 预测 -> `N=6` 凸 QP-MPC -> OSQP -> SineKAN-DQN 选择 `q_h2/q_soc/q_batt` -> 按实际施加的电池功率更新闭环 SOC。

当前仓库的组成模块已存在，但目标端到端链路尚未统一。`N=60` OSQP 路径是使用离线 1 s 重构负荷的求解器/控制 benchmark；30 s LSTM-MPC 路径使用 CasADi/IPOPT 和另一套历史参数。两者都不是已验收的目标 `N=6` OSQP 闭环。

## Active data

- 原始依据：`total_load_excels/` 中 66 个 30 s 实船航段。
- 正式划分：按航次开始时间顺序分为 train/validation/test = 46/13/7，清单在 `outputs/config/voyage_split_total_load_721.json`。
- 负荷定义：`total_load_fc_plus_batt_kw = fuel_cell_total_kw + battery_total_kw`；代码会检查该恒等式。
- 1 s benchmark 数据：按航次对 30 s 负荷做 natural cubic spline，再裁剪到非负；属于离线、使用未来端点的重构数据，不是在线实测。
- 10 ms 辅助数据：由两个 1 ms 原始工作簿直接每 10 行抽点；32,000 行，负载约 0.10–37.46 kW，只用于高频 LSTM 预测诊断。
- 窗口不得跨航次/原子序列；scaler 只在训练数据拟合。

## Active parameters

| 参数 | 当前候选/边界 | 说明 |
| --- | --- | --- |
| `P_fc_max` | 560 kW | 目标船舶主线 |
| `E_batt` | 693 kWh | 旧 277.2/1806 kWh 配置已失效 |
| `P_batt_max` | 346.5 kW | 充放电绝对边界 |
| `P_batt_ref` | 346.5 kW | 归一化参考 |
| `SOC_ref/min/max` | 0.55 / 0.2 / 0.8 | 目标 QP benchmark |
| `SOC_band` | 0.05 | 当前 693 kWh benchmark 事实；本轮未修改 |
| FC ramp | 48 kW/s | QP 中为硬约束 |
| LSTM | history 30, prediction 6 | direct multi-output |
| 正式候选 horizon | `N=6` | 尚无统一 OSQP 闭环入口 |
| benchmark horizon | `N=60` | 仅离线 benchmark |

暂定 benchmark 权重为 `q_h2=0.5`、`q_soc=2.0`、`q_batt=0.05`、`q_ramp=0`、`q_terminal_soc=0`。它不是最终论文权重，也不是仓库全局默认值。

## Active modules

- **30 s data/LSTM：** 数据构建、46/13/7 划分、direct multi-output LSTM、checkpoint 和逐 horizon 指标已存在。
- **1 s spline/LSTM：** 重构、物理诊断、简单基线和固定 Task C 训练结果已存在；仅作辅助/离线证据。
- **QP-MPC/OSQP：** 归一化凸 QP、固定稀疏结构、边界更新、warm start、求解器统计和 `N=60` benchmark 已存在。
- **CasADi/IPOPT：** 30 s LSTM-MPC 历史/支持性路径已存在，与目标 OSQP 主线参数不统一。
- **DQN：** replay、target network、Double DQN、epsilon、奖励、动作映射及 MLP/KAN/SineKAN 网络实现已存在，但目标环境/动作未统一。
- **SineKAN：** 项目 wrapper 会导入 `SineKAN-main/` 的第三方实现；许可证待核验。

## Current validated results

以下是仓库可复查产物，不等于本轮重新运行训练或正式论文验收：

- `outputs/lstm_total_load_721/` 保存 30 s、66 航段 LSTM checkpoint 与 test/horizon 指标；7 个 test 航次聚合的 h1 指标约为 RMSE 41.94 kW、MAE 18.18 kW、WAPE 8.40%。
- 1 s Task C (`history=30`, `prediction=6`) 的 LSTM 在 test 上 h1/h6 MAE 约为 1.79/3.85 kW；current-hold 为 0.60/3.57 kW，last-slope 为 0.04/0.80 kW。LSTM 未超过简单基线，因此尚不能作为预测优势证据。
- `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` 保存 7 个 test 航次上的 693 kWh OSQP benchmark。候选权重仍被报告标为暂定且存在数值约束验收问题，未被正式接受。
- 已核对的 CLI 入口均可显示 `--help`；完整训练与 benchmark 本轮未重跑。
- 当前单元测试基线：141 项全部通过；10 ms audit 已严格校验 assignment key 必须恰为 train/validation/test，并覆盖缺失 key 与额外 key 场景。

## Provisional items

- `q_h2=0.5, q_soc=2.0, q_batt=0.05` 仅为 `N=60` 离线 benchmark 工程候选。
- `SOC_band=0.05` 是当前 benchmark 事实，但仍需连同候选闭环行为一起验收。
- 1 s spline 仅适合离线算法/求解器研究；若论文声称在线 1 s 预测，必须换用因果可得数据或重新定义实验。
- SineKAN 是目标 Q 网络，但尚无目标环境下的最终结果；MLP-DQN 和 KAN-DQN 也未完成公平对照。
- 30 s CasADi LSTM-MPC 与 `N=60` OSQP benchmark 仅作迁移依据，不应被拼接成已完成主链路。

## Blocking issues

1. 尚无把 LSTM 6 步预测、`N=6` QP、OSQP、实际负荷反馈和实际电池功率 SOC 更新统一起来的正式闭环入口。
2. 固定 `N=6` QP-MPC 基线及其权重尚未在 7 个 test 航次上通过物理、性能、失败回退和实时性验收。
3. 目标 DQN 状态/动作/奖励/环境尚未冻结；现有脚本仍选择 `q_ramp` 或直接功率动作，并使用 1806/1067 kWh 等旧参数。
4. 1 s LSTM 没有超过简单基线，且 spline 数据非因果，不能支撑在线预测优势结论。
5. 依赖清单不完整、第三方 SineKAN 许可证未核验、多个跟踪文件含本地绝对路径，干净环境复现尚未闭合。

## Next priority tasks

1. **P0：** 定义并实现唯一的 `N=6` OSQP LSTM-MPC 闭环，明确 stage 时序、预测输入、求解失败回退和 SOC 实际更新。
2. **P0：** 在 7 个 test 航次上验收固定权重基线；未通过前不训练 DQN。
3. **P1：** 冻结仅选择 `q_h2/q_soc/q_batt` 的动作表、与动作权重无自指关系的物理奖励，以及统一的目标环境。
4. **P1：** 在相同预算和种子下实现 MLP-DQN 与 SineKAN-DQN 公平比较；KAN-DQN 仅在资源允许时加入。
5. **P2：** 设计固定/动态权重、真实/预测负荷和计算实时性的论文实验矩阵。
6. **P3：** 锁定依赖、迁移绝对路径、建立 CI 和输出保留策略。

完整任务和验收条件见 `docs/UNFINISHED_TASKS.md`。

## Last updated commit

- 状态审计日期：2026-07-12。
- 审计基线：`23ea106` (`首次上传项目代码`)。
- 本文件的初始化版本由包含它的文档提交标识；使用 `git log -1 --oneline` 获取该提交。文档不得保存一个会因自我修改而失效的 commit hash。
