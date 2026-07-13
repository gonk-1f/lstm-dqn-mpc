# Project Status

> 当前状态唯一入口。`thread.md`、`project_status.md` 和 `next_steps.md` 是历史上下文，不代表全部当前事实。

## Active framework

目标论文框架为：66 航段船舶负荷 -> LSTM direct multi-output 预测 -> `N=6` 凸 QP-MPC -> OSQP -> SineKAN-DQN 选择 `q_h2/q_soc/q_batt` -> 按实际施加的电池功率更新闭环 SOC。

当前仓库已有独立 `N=6` OSQP 离线滚动入口，但它使用样条未来真实点作为 ideal foresight，没有接入 LSTM。`N=60` OSQP 路径降为历史 solver/performance benchmark；30 s LSTM-MPC 路径使用 CasADi/IPOPT 和另一套历史参数。目标端到端 LSTM-OSQP-DQN 链路仍未统一。

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
| `SOC_band` | `q_soc`-only 诊断固定为 0.05；旧 C 为 0.075 | 不得把旧 C 与新诊断混报 |
| FC ramp | 48 kW/s | QP 中为硬约束 |
| LSTM | history 30, prediction 6 | direct multi-output |
| 固定权重选择 horizon | `N=6` | A–D 均拒绝；后续 `q_soc=20` 通过结构可行性门禁但未正式接受 |
| historical benchmark horizon | `N=60` | 仅历史 solver/performance benchmark |

当前没有 provisional 或 accepted 固定权重。A–D 四组仍是已拒绝的历史候选。严格限定的新诊断固定 `q_h2=0.5`、`q_batt=0.05`、`SOC_band=0.05`、`q_ramp=0`、`q_terminal_soc=0`，只改变 `q_soc={5,10,20}`；其中 `q_soc=20` 是可行性见证，不是自动选出的最终权重。

## Active modules

- **30 s data/LSTM：** 数据构建、46/13/7 划分、direct multi-output LSTM、checkpoint 和逐 horizon 指标已存在。
- **1 s spline/LSTM：** 重构、物理诊断、简单基线和固定 Task C 训练结果已存在；仅作辅助/离线证据。
- **QP-MPC/OSQP：** 归一化凸 QP、等价仿射数值缩放、固定稀疏结构、边界更新、warm start、冷重启诊断、失败终止、`N=6` ideal-foresight runner、`q_soc`-only 诊断入口和历史 `N=60` benchmark 已存在。
- **CasADi/IPOPT：** 30 s LSTM-MPC 历史/支持性路径已存在，与目标 OSQP 主线参数不统一。
- **DQN：** replay、target network、Double DQN、epsilon、奖励、动作映射及 MLP/KAN/SineKAN 网络实现已存在，但目标环境/动作未统一。
- **SineKAN：** 项目 wrapper 会导入 `SineKAN-main/` 的第三方实现；许可证待核验。

## Current validated results

以下是仓库可复查产物，不等于本轮重新运行训练或正式论文验收：

- `outputs/lstm_total_load_721/` 保存 30 s、66 航段 LSTM checkpoint 与 test/horizon 指标；7 个 test 航次聚合的 h1 指标约为 RMSE 41.94 kW、MAE 18.18 kW、WAPE 8.40%。
- 1 s Task C (`history=30`, `prediction=6`) 的 LSTM 在 test 上 h1/h6 MAE 约为 1.79/3.85 kW；current-hold 为 0.60/3.57 kW，last-slope 为 0.04/0.80 kW。LSTM 未超过简单基线，因此尚不能作为预测优势证据。
- `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` 保存 7 个 test 航次上的 693 kWh OSQP benchmark。候选权重仍被报告标为暂定且存在数值约束验收问题，未被正式接受。
- `outputs/mpc_1s_n6_weight_selection/` 保存 2026-07-13 的 A–D 四候选离线 ideal-foresight 结果。闭环覆盖率 A/B/C/D 分别为 0.723573/0.714511/0.648608/0.735204；最坏航段 SOC 净变化分别为 -0.349873/-0.349808/-0.350000/-0.349695。A/B/D 各有一个未完成航段，C 有两个；人工结论为 `no_candidate_selected`。
- `outputs/mpc_1s_n6_qsoc_feasibility/` 保存严格限定的 `q_soc={5,10,20}` 诊断。三组均完成 7 航段、93,030 个应用步，solver failure 和物理不可行点均为 0；最坏航段 SOC 净变化依次为 -0.106636/-0.047082/-0.021640，只有 `q_soc=20` 通过 `-0.03` 门禁。它的总体氢耗为 284.452 kg、电池吞吐为 594.583 kWh、FC 高于负荷比例为 0.357336、p99/max 求解时间为 0.140/45.850 ms。
- N=6 runner 在每个时刻使用 `t+1..t+6`、只执行第一步，以实际功率平衡计算电池功率和 SOC；最终求解失败时终止该航段，不用 NaN/冻结状态伪造后续闭环。
- 已核对的 CLI 入口均可显示 `--help`；本轮完整运行了三组 `q_soc` 候选，但没有重跑训练、A–D 或历史 N=60 benchmark。
- 本轮最终 `python -m compileall src` 通过，完整单元测试为 174 项全部通过；10 ms audit 仍严格校验 assignment key 必须恰为 train/validation/test。

## Provisional items

- 当前没有 provisional 固定权重；`configs/benchmarks/mpc_1s_n6_provisional.*` 未创建。`q_soc=20` 只作为 `weight_only_sufficient` 的可行性见证，仍需审查 35.7336% FC 高于负荷、氢耗和正式在线预测边界。
- A–D 的参数和结果只代表被拒绝的有限候选集，不得把 D 的较高覆盖率解释为 least-bad 保留依据。
- `N=60` 权重和产物只作历史 benchmark，紧凑入口见 `outputs/mpc_1s_n6_weight_selection/N60_HISTORICAL_BENCHMARK.md`。
- 1 s spline 仅适合离线算法/求解器研究；若论文声称在线 1 s 预测，必须换用因果可得数据或重新定义实验。
- SineKAN 是目标 Q 网络，但尚无目标环境下的最终结果；MLP-DQN 和 KAN-DQN 也未完成公平对照。
- 30 s CasADi LSTM-MPC 与 `N=60` OSQP benchmark 仅作迁移依据，不应被拼接成已完成主链路。

## Blocking issues

1. 已有 ideal-foresight `N=6` 执行路径，但尚无把 LSTM 6 步预测接入该路径的正式闭环入口。
2. `q_soc=20` 已通过离线 ideal-foresight 的全航段/物理/SOC/求解门禁，但尚未完成正式工程接受，也没有接入因果 LSTM 预测；在明确接受固定基线前不得训练正式 DQN。
3. 目标 DQN 状态/动作/奖励/环境尚未冻结；现有脚本仍选择 `q_ramp` 或直接功率动作，并使用 1806/1067 kWh 等旧参数。
4. 1 s LSTM 没有超过简单基线，且 spline 数据非因果，不能支撑在线预测优势结论。
5. 依赖清单不完整、第三方 SineKAN 许可证未核验、多个跟踪文件含本地绝对路径，干净环境复现尚未闭合。

## Next priority tasks

1. **P0：** 对 `q_soc=20` 可行性见证完成正式工程审查，重点判断 FC 高于负荷、氢耗与电池使用是否可接受；只在证据充分时创建 provisional/accepted 配置，不使用 least-bad。
2. **P0：** 将因果可用的 6 步预测 provider 和经过测试的确定性失败回退接入现有 `N=6` 时序/物理执行路径。
3. **P1：** 冻结仅选择 `q_h2/q_soc/q_batt` 的动作表、与动作权重无自指关系的物理奖励，以及统一的目标环境。
4. **P1：** 在相同预算和种子下实现 MLP-DQN 与 SineKAN-DQN 公平比较；KAN-DQN 仅在资源允许时加入。
5. **P2：** 设计固定/动态权重、真实/预测负荷和计算实时性的论文实验矩阵。
6. **P3：** 锁定依赖、迁移绝对路径、建立 CI 和输出保留策略。

完整任务和验收条件见 `docs/UNFINISHED_TASKS.md`。

## Last updated commit

- 状态审计日期：2026-07-13。
- 使用 `git log -1 --oneline` 获取包含本状态文件的提交；文档不保存会因自我修改而失效的 commit hash。
