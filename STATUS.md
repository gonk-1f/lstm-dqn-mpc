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
| 四目标灵敏度 horizon | `N=6` | baseline 与完整 17-case one-factor 矩阵均已正式运行 |
| baseline 权重 | `q_h2=q_batt=q_soc=q_fc_var=1` | 已运行和人工审查，不自动接受，当前没有 accepted 权重 |
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
- `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` 与 `tests/test_mpc_1s_n6_four_objective_sensitivity.py` 冻结了 baseline-first 的 17 配置、可获得的 `t+1..t+6`、航段尾部同航段末样本 edge-hold、第一步执行、实际功率/SOC 更新和无自动最优选择的契约。
- 全 1 baseline 产物已写入 `outputs/mpc_1s_n6_four_objective_sensitivity/baseline_1_1_1_1/`、`reports/mpc_1s_n6_four_objective_sensitivity_summary.md` 与 `reports/mpc_1s_n6_four_objective_sensitivity_table.csv`。7/7 航次完成，93030/93030/93030 个 expected/attempted/applied 步，solver failure、primal infeasible 和 max-iter 均为 0，`formal_complete=true`。
- baseline 总氢耗为 `218.448931 kg`；平均 SOC 从 `0.55` 降至 `0.287935`，平均变化 `-0.262065`，全局最低值为 `0.199997215`；平均/95 分位/最大求解时间为 `0.178247/0.472075/14.804800 ms`。归一化/加权四项总量为 H2 `24712.946847`、battery `3643.932321`、SOC `1869104.558203`、FC variation `56.592723`，总和 `1897518.030093`。
- 七张逐航次图均显示 SOC 从 `0.55` 下降且没有恢复；燃料电池承担主要负荷，电池主要放电补差。`voyage_063` 约 9400 s 到达 SOC 下界后电池近零、FC 几乎独自跟随负荷；最低 SOC 相对 `0.2` 的残差 `2.7846e-6` 超过 runner 声明的 `1e-6` 容差，分类为小幅数值约束容差超限，而不是 solver failure。
- 完整 17-case one-factor 矩阵已运行：17 个唯一配置、119 个配置-航段，其中 107 个完整、12 个最终失败；累计记录 1 个 primal-infeasible 和 24 个 max-iter 事件。完成率不足 1 的配置只保留失败前缀，汇总 CSV 用 `metrics_comparable=false` 明确禁止横向比较累计物理量。
- 单因素事实：`q_h2=2/4`、`q_batt=0.25/0.5`、`q_soc=0.25/0.5` 存在不完整航段；`q_fc_var` 四个变体均完成。降低 `q_h2` 或提高 `q_batt/q_soc` 可减缓 SOC 下降；提高 `q_fc_var` 可降低 FC 变化平方量，但会增加电池平方量和功率尖峰。所有完整配置的平均 final SOC 均低于 0.55，没有 accepted 固定权重。
- `N=6` runner 在每个时刻使用可获得的 `t+1..t+6`（尾部同航段末样本 edge-hold）、只执行第一步，以实际功率平衡计算电池功率和 SOC；只有 OSQP 状态、有限性和首步 state-commit 残差门禁全部通过才更新状态。本轮门禁拒绝为 0；最终失败时立即终止，不用 NaN/冻结状态伪造后续闭环。
- 本轮没有运行 LSTM/DQN 或重跑历史 `N=60` 数值实验；只执行了其保留回归测试。四目标 focused test 为 52/52 通过，保留 `N=60` benchmark test 为 17/17 通过，完整 suite 为 193/193 通过。

## Provisional items

- 当前没有 provisional 或 accepted 固定权重；`configs/benchmarks/mpc_1s_n6_provisional.*` 未创建。17 配置矩阵只提供逐项物理灵敏度，runner 禁止自动 best/score/rank/winner。
- 已完成结果与图表审查。下一轮人工审阅区间为 `q_h2:[0.25,0.5]`、`q_batt:[2,4]`、`q_soc:[2,4]`；`q_fc_var` 在已测 `[0.25,4]` 内只有连续权衡、没有完成率转折，无法由本轮数据确认更窄区间。这些边界不是新实验授权，也不是 optimum/accepted 结论；联合筛选仍等待人工决定。
- `N=60` 权重和产物只作历史 benchmark，保留证据直接位于 `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/`，不依赖任何已删除的 `N=6` 指针。
- 1 s spline 仅适合离线算法/求解器研究；若论文声称在线 1 s 预测，必须换用因果可得数据或重新定义实验。
- SineKAN 是目标 Q 网络，但尚无目标环境下的最终结果；MLP-DQN 和 KAN-DQN 也未完成公平对照。
- 30 s CasADi LSTM-MPC 与 `N=60` OSQP benchmark 仅作迁移依据，不应被拼接成已完成主链路。

## Blocking issues

1. 已有 ideal-foresight `N=6` 执行路径，但尚无把 LSTM 6 步预测接入该路径的正式闭环入口。
2. 四目标全 1 baseline 与完整 17-case one-factor 已运行和人工审查，但没有配置被接受：所有完整配置的平均 final SOC 低于 0.55，若干配置在 SOC 下界附近失败，且存在小幅 SOC 容差超限；在人工明确接受固定权重前不得训练正式 DQN。
3. 目标 DQN 状态/动作/奖励/环境尚未冻结；现有脚本仍选择 `q_ramp` 或直接功率动作，并使用 1806/1067 kWh 等旧参数。
4. 1 s LSTM 没有超过简单基线，且 spline 数据非因果，不能支撑在线预测优势结论。
5. 依赖清单不完整、第三方 SineKAN 许可证未核验、多个跟踪文件含本地绝对路径，干净环境复现尚未闭合。

## Next priority tasks

1. **P0：** baseline 与完整 17-case one-factor 已完成；等待人工审阅报告中的物理权衡与建议区间，再决定是否授权联合筛选或接受/拒绝固定权重，不使用自动综合分数选 best。
2. **P0：** 将因果可用的 6 步预测 provider 和经过测试的确定性失败回退接入现有 `N=6` 时序/物理执行路径。
3. **P1：** 冻结仅选择 `q_h2/q_soc/q_batt` 的动作表、与动作权重无自指关系的物理奖励，以及统一的目标环境。
4. **P1：** 在相同预算和种子下实现 MLP-DQN 与 SineKAN-DQN 公平比较；KAN-DQN 仅在资源允许时加入。
5. **P2：** 设计固定/动态权重、真实/预测负荷和计算实时性的论文实验矩阵。
6. **P3：** 锁定依赖、迁移绝对路径、建立 CI 和输出保留策略。

完整任务和验收条件见 `docs/UNFINISHED_TASKS.md`。

## Last updated commit

- 状态审计日期：2026-07-18。
- 使用 `git log -1 --oneline` 获取包含本状态文件的提交；文档不保存会因自我修改而失效的 commit hash。
