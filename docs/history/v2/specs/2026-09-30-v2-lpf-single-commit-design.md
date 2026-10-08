# v2 LPF 单次提交修复设计

日期：2026-09-30

## 1. 问题与目标

正式 v2 环境名义配置为 `Ts = 30 s`、`tau_LPF = 180 s`，因此一阶因果低通滤波器每个真实监督控制步应只执行一次状态更新：

```text
alpha = exp(-30 / 180)
x_k = alpha * x_(k-1) + (1 - alpha) * load_k
```

当前调用链在一次成功的 `FormalEpisodeBackend.execute_mpc_step()` 中，对同一个 `load_k` 提交了两次滤波器状态：`NonlinearMPC.solve()` 提交一次，Backend 又提交一次。实际递推因子因而成为 `alpha^2 = exp(-30 / 90)`，使名义 180 s 的滤波响应等效接近 90 s。

本轮目标是恢复“一个真实 30 s interval 对应一次 LPF 状态提交”的合同，使代码执行语义与冻结配置一致。修复后重新生成受影响的训练前证据，并重新训练 H4。动作集中度和 DQN 超参数不在本轮调整。

## 2. 选择的接口方案

采用方案 A：求解与状态提交分离。

- `CausalBaseLoadFilter.preview()` 继续提供无副作用预测。
- `NonlinearMPC.solve()` 只读取滤波器状态并生成计划，不提交滤波器状态。
- `FormalEpisodeBackend.execute_mpc_step()` 仅在求解成功、首个控制量通过独立校验并被真实执行时，对当前负载提交一次。
- 需要模拟连续真实执行的审计或校准调用方，必须在每次成功执行后显式提交；只做候选求解、失败求解或敏感性探测的调用不得提交。

不采用“solver 内提交、Backend 不提交”的最小修补，因为求解器调用本身不等于控制量已执行，隐式修改外部状态会继续混淆规划与环境状态转移。

## 3. 运行时数据流

每个 ONBOARD 30 s interval 的顺序固定为：

1. Backend 读取当前 `load_k`、SOC 和上一时刻 FC 功率。
2. MPC 通过 `preview(load_k, horizon=5)` 构造因果持久性负载预测和基础负载参考。
3. MPC 求解并返回计划，但不修改滤波器。
4. Backend 校验并执行 `plan.first_command()`。
5. Backend 对 `load_k` 调用一次 `commit()`。
6. Backend 更新历史状态、SOC、退化账户和 interval ledger。

若求解失败、返回计划无效或首个控制量校验失败，则本 interval 不执行，LPF 状态保持不变。

SHORE interval 继续重置 onboard 历史和滤波器，不新增 DQN/MPC step，也不受本次接口修复影响。

## 4. 合同和版本隔离

控制语义中新增明确的 LPF 更新合同标识，例如：

```text
base_load_filter_update_version = causal_single_commit_per_executed_interval_v1
```

该字段进入 checkpoint、reward-scale、审计和选择产物已经使用的 `control_semantics` 身份校验。旧 checkpoint 即使写有 `tau_lpf_seconds = 180`，也不得与修复后的环境恢复或混用。

`TAU_LPF_SECONDS` 保持 `180.0`。MPC 三项目标公式、目标归一化、36 动作目录、DQN 状态、学习率、探索策略、经济模型和数据集均不修改。

## 5. TDD 验收合同

先新增红测试并确认它因当前重复提交而失败，再修改生产代码。

必要测试包括：

1. `NonlinearMPC.solve()` 成功时不改变传入滤波器的 `observed_base_kw`。
2. 求解失败或无效计划时滤波器同样不改变。
3. 一个成功的 Backend ONBOARD interval 只产生一次递推：
   `after = alpha * before + (1 - alpha) * load`。
4. 连续两个真实 interval 恰好产生两次递推，不得按四次递推计算。
5. 30 s、180 s 配置下的实际递推因子保持 `exp(-30/180)`，不得成为其平方。
6. SHORE interval 不提交 onboard LPF 状态，并保持既有重置语义。
7. 新 `control_semantics` 字段使缺少该字段的旧 checkpoint/reward-scale 身份不兼容。
8. 所有需要连续执行的审计调用方在成功执行后显式提交，结果保持有限且可复现。

## 6. 受影响产物与清理范围

以下产物由重复更新的运行链路生成，移入 Windows 回收站，不永久删除：

- `outputs/v2_history_dqn_study/H4_tau180/`
- `outputs/v2_history_dqn_study/H4_tau180_selection_40/`
- `outputs/v2_history_dqn_study/H4_tau180_test/`
- `outputs/v2_history_dqn_study/H4_tau180_test_power_plots/`
- `outputs/v2_history_dqn_study/tau_lpf_validation_screen/`
- 当前 `reward_scale_calibration.json`
- 与旧控制语义绑定的状态审计、目标尺度审计和失败惩罚审计派生产物

保留以下历史对照：

- `outputs/v2_history_dqn_study/H4_round20_test/`
- `outputs/v2_history_dqn_study/H4_round20_test_power_plots/`
- `outputs/v2_history_dqn_study/reward_scale_calibration_tau90_backup.json`

清理前必须逐项解析并验证绝对路径位于仓库的 `outputs` 目录内。

## 7. 证据重建和验证

修复后使用现有 Train/Validation 数据重新生成必要证据，不打开 Test：

1. 状态审计及其 manifest；
2. 目标尺度审计；
3. 失败惩罚审计；
4. Train-only reward-scale calibration；
5. formal preflight；
6. H4 smoke。

代码验证顺序：

- LPF focused tests；
- 相关 formal episode、MPC、reward-scale 和身份合同测试；
- 全部 v2 tests；
- solver smoke；
- compile/import；
- `git diff --check`。

只有 preflight 恢复为 GO 且 smoke 通过后，才给出新的 H4 40轮训练命令和恢复命令。

## 8. 训练后评估边界

新模型训练完成后先在 Validation 中重新选择 checkpoint，并审计：

- FC total variation、mean/p95/max absolute step；
- battery absolute energy、RMS power 和退化成本；
- SOC范围和物理可行性；
- 动作种类、占比、状态到动作映射和Q值间隔。

本轮不因旧 Test 中的 018 图修改DQN超参数。旧 Test 已经被查看，任何受该反馈影响的新模型结果必须标为 post-hoc/exploratory；若要重新声明无偏最终测试，需要预先封存新的未见测试数据。
