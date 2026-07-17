# Project Status

> 当前状态唯一入口。`thread.md`、`project_status.md` 和 `next_steps.md` 是历史上下文，不代表全部当前事实。

## Active framework

目标论文框架为：66 航段船舶负荷 -> LSTM direct multi-output 预测 -> `N=6` 凸 QP-MPC -> OSQP -> SineKAN-DQN 选择 `q_h2/q_soc/q_batt` -> 按实际施加的电池功率更新闭环 SOC。

当前唯一的 `N=6` OSQP 离线滚动入口是 `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`。它使用样条未来真实点作为 offline oracle/ideal foresight，没有接入 LSTM 或 DQN，并且每次只执行第一步。`N=60` OSQP 路径降为历史 solver/performance benchmark；30 s LSTM-MPC 路径使用 CasADi/IPOPT 和另一套历史参数。目标端到端 LSTM-OSQP-DQN 链路仍未统一。

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
| `SOC_band` | 0.05 | 四目标 SOC tracking 归一化参考 |
| FC ramp | 48 kW/s | QP 中为硬约束 |
| LSTM | history 30, prediction 6 | direct multi-output |
| 四目标灵敏度 horizon | `N=6` | baseline 与 one-factor 结果均未运行 |
| baseline 权重 | `q_h2=q_batt=q_soc=q_fc_var=1` | 先运行并人工审查，不自动接受 |
| one-factor 矩阵 | 每项取 `0.25, 0.5, 1, 2, 4` | 共享全 1 baseline，共 17 个唯一配置 |
| historical benchmark horizon | `N=60` | 仅历史 solver/performance benchmark |

当前没有 provisional 或 accepted 固定权重。活动目标仅含 `H2_norm`、`Batt_power_sq_norm`、`SOC_tracking_sq_norm` 与 `FC_variation_sq_norm` 四项；参考值依次为 `m_H2(560 kW, 1 s)=0.00883945296644347 kg/step`、`346.5 kW`、`SOC_ref=0.55` 与 `SOC_band=0.05`、`48 kW/step`。`q_ramp=q_terminal_soc=0`，不加入额外软 ramp 或 terminal SOC 目标。

## Active modules

- **30 s data/LSTM：** 数据构建、46/13/7 划分、direct multi-output LSTM、checkpoint 和逐 horizon 指标已存在。
- **1 s spline/LSTM：** 重构、物理诊断、简单基线和固定 Task C 训练结果已存在；仅作辅助/离线证据。
- **QP-MPC/OSQP：** 归一化四目标凸 QP、等价仿射数值缩放、固定稀疏结构、边界更新、warm start、失败终止、唯一 `N=6` offline-oracle runner/focused test 和历史 `N=60` benchmark 已存在。
- **CasADi/IPOPT：** 30 s LSTM-MPC 历史/支持性路径已存在，与目标 OSQP 主线参数不统一。
- **DQN：** replay、target network、Double DQN、epsilon、奖励、动作映射及 MLP/KAN/SineKAN 网络实现已存在，但目标环境/动作未统一。
- **SineKAN：** 项目 wrapper 会导入 `SineKAN-main/` 的第三方实现；许可证待核验。

## Current validated results

以下是仓库可复查产物，不等于本轮重新运行训练或正式论文验收：

- `outputs/lstm_total_load_721/` 保存 30 s、66 航段 LSTM checkpoint 与 test/horizon 指标；7 个 test 航次聚合的 h1 指标约为 RMSE 41.94 kW、MAE 18.18 kW、WAPE 8.40%。
- 1 s Task C (`history=30`, `prediction=6`) 的 LSTM 在 test 上 h1/h6 MAE 约为 1.79/3.85 kW；current-hold 为 0.60/3.57 kW，last-slope 为 0.04/0.80 kW。LSTM 未超过简单基线，因此尚不能作为预测优势证据。
- `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` 保存 7 个 test 航次上的 693 kWh OSQP benchmark。全部候选均未通过门禁，决策为 `NONE_ACCEPTED`，各候选均为 `accepted=false`；不同候选分别存在 SOC、约束或 solver 问题，因此没有 accepted baseline。
- `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` 与 `tests/test_mpc_1s_n6_four_objective_sensitivity.py` 冻结了 baseline-first 的 17 配置、`t+1..t+6`、第一步执行、实际功率/SOC 更新和无自动最优选择的契约。
- `outputs/mpc_1s_n6_four_objective_sensitivity/`、`reports/mpc_1s_n6_four_objective_sensitivity_summary.md` 与 `reports/mpc_1s_n6_four_objective_sensitivity_table.csv` 当前均不存在；全 1 baseline 和完整 17 配置 one-factor 结果均为 **未运行**，因此没有可报告的完成率、数值趋势或推荐区间。
- `N=6` runner 在每个时刻使用 `t+1..t+6`、只执行第一步，以实际功率平衡计算电池功率和 SOC；最终求解失败时终止该航段，不用 NaN/冻结状态伪造后续闭环。
- 本轮未运行新的正式实验，也未重跑 LSTM/DQN 或历史 `N=60` benchmark。清理后的 focused test 为 43/43 通过，保留 `N=60` benchmark test 为 17/17 通过。

## Provisional items

- 当前没有 provisional 固定权重；`configs/benchmarks/mpc_1s_n6_provisional.*` 未创建。四目标 baseline/one-factor 尚未运行，不能先验指定趋势或候选。
- 17 配置矩阵只提供逐项物理灵敏度；runner 禁止自动 best/score/rank/winner。完整结果运行后仍须人工审查 SOC、氢耗、电池使用、FC 波动、约束和求解状态，才能决定下一步区间或接受/拒绝。
- `N=60` 权重和产物只作历史 benchmark，保留证据直接位于 `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/`，不依赖任何已删除的 `N=6` 指针。
- 1 s spline 仅适合离线算法/求解器研究；若论文声称在线 1 s 预测，必须换用因果可得数据或重新定义实验。
- SineKAN 是目标 Q 网络，但尚无目标环境下的最终结果；MLP-DQN 和 KAN-DQN 也未完成公平对照。
- 30 s CasADi LSTM-MPC 与 `N=60` OSQP benchmark 仅作迁移依据，不应被拼接成已完成主链路。

## Blocking issues

1. 已有 ideal-foresight `N=6` 执行路径，但尚无把 LSTM 6 步预测接入该路径的正式闭环入口。
2. 四目标全 1 baseline 与 17 配置 one-factor 尚未运行和人工审查；在依据物理指标明确接受固定基线前，不得训练正式 DQN。
3. 目标 DQN 状态/动作/奖励/环境尚未冻结；现有脚本仍选择 `q_ramp` 或直接功率动作，并使用 1806/1067 kWh 等旧参数。
4. 1 s LSTM 没有超过简单基线，且 spline 数据非因果，不能支撑在线预测优势结论。
5. 依赖清单不完整、第三方 SineKAN 许可证未核验、多个跟踪文件含本地绝对路径，干净环境复现尚未闭合。

## Next priority tasks

1. **P0：** 先运行并审查全 1 baseline，再运行完整 17 配置 one-factor；只根据逐航次物理证据决定下一搜索区间或 accepted/rejected，不使用自动综合分数选 best。
2. **P0：** 将因果可用的 6 步预测 provider 和经过测试的确定性失败回退接入现有 `N=6` 时序/物理执行路径。
3. **P1：** 冻结仅选择 `q_h2/q_soc/q_batt` 的动作表、与动作权重无自指关系的物理奖励，以及统一的目标环境。
4. **P1：** 在相同预算和种子下实现 MLP-DQN 与 SineKAN-DQN 公平比较；KAN-DQN 仅在资源允许时加入。
5. **P2：** 设计固定/动态权重、真实/预测负荷和计算实时性的论文实验矩阵。
6. **P3：** 锁定依赖、迁移绝对路径、建立 CI 和输出保留策略。

完整任务和验收条件见 `docs/UNFINISHED_TASKS.md`。

## Last updated commit

- 状态审计日期：2026-07-17。
- 使用 `git log -1 --oneline` 获取包含本状态文件的提交；文档不保存会因自我修改而失效的 commit hash。
