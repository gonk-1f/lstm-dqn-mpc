# V4 方案 A 正式训练前算法冻结与诊断补充

基于 `feat/direct-power-mlp-ddqn` 的方案 A 保守门控提交 `9a9cebad0ddfae054c8143a9f09eb1bbb7c4884e`。本次只审查现有算法并增加只读诊断；未运行 40 轮正式训练，未读取 Test，未改冻结数据、物理模型、经济账本、奖励、网络、优化器、回放采样或控制规则。

## 算法冻结核对

- [MLP 网络](../src/dqn/networks/mlp_qnet.py)为 `8→128→64→61`；[V4 智能体](../src/v4/dqn.py)具有在线/目标网络、ε-greedy、FIFO 经济经验回放、Adam、Smooth L1 和梯度裁剪。Double DQN 目标在下一状态策略候选掩码内用**在线网络选动作、目标网络评估**；终止不 bootstrap。`n_step=1` 为本轮拟用配置。
- [方案 A](../src/v4/control.py)先求未改的物理掩码，再用此前 FC 功率形成策略候选；当前执行与 TD 后继掩码采用同一规则。它没有修改 `MLPQNetwork`、奖励或 TD 损失。
- `failure_terminal_quota=2` 只影响带明确失败终止标签的经济 replay 抽样；SOC 分段软惩罚、电池价值时间重分配及终端反向校正、辅助 outcome 网络均为旧机制。[`select_power`](../src/v4/dqn.py)只读取在线 Q；[经济 `learn`](../src/v4/dqn.py)只读取在线/目标 Q，**没有读取 outcome 网络**。Outcome 训练标签与更新是独立路径。
- 下一次正式对照须从头初始化：`rounds=40, seed=42, MLP 8–128–64–61, Adam lr=1e-4, batch=64, replay_capacity=100000, gamma=1, epsilon 1→0.05, reward_feedback=redistributed, reward_scale=0.001, beta_soc=500, failure_penalty_scale=1, failure_terminal_quota=2, cadence=replay32, target_interval=500 hard sync, n_step=1, episode_credit_scope=voyage`。固定 Train30/Validation8，只有 greedy Train 30/30 后评估 Validation；两者全完成后才比较 Validation 可比费用。Test 始终封存。正式训练尚未获本次授权。

## 新增只读诊断

[正式入口](../src/v4/feedback_study.py)新增可选 `--diagnostic-rounds 1,10,20,30,40`。未指定时关闭，保留旧实验调用的输出口径；启用时仅在指定轮次结束后，使用当轮 greedy Train 中已经执行的 `_015`、`_017`、`_044` 决策状态。每样本最多 16 条，优先抽取低 SOC 且负载≥600 kW、连续至少 3 步 FC≤100 kW 且负载≥600 kW、SOC≥0.79、受迫停机，以及失败末端附近；没有这些状态时保留首/末决策。类别阈值只决定**记录哪些状态**，不参与控制或模型选择。

每条记录含完整 8 维状态、当前物理/策略候选掩码、61 个在线 Q、实际选中动作及合法动作 Q 排序。Q 单位是 `reward_scale` 后的训练奖励等价值，**不是实际人民币费用**。Q 值以与 `select_power` 相同的单状态前向形状计算，避免批量矩阵运算的末位舍入差异。

每个选定轮次在 `diagnostic_checkpoints/round_NNN.pt` 保存在线和目标网络状态，以及源码提交、完整超参数、manifest SHA256、经济更新/目标同步/经验插入与环境步数；相应的 `round_NNN_decisions.json` 只存上述少量状态。诊断文件标记 `diagnostic_only`，与 `best_agent.pt` 分离；最佳模型仍完全由原 Train/Validation 资格和费用规则选出。诊断 checkpoint 只用于重现 Q 值，不含优化器与 replay，**不能当作可续训 checkpoint**。

诊断在 `torch.inference_mode()` 下读取网络；不调用 `select_power`、`remember`、`learn`、`sync_target` 或 outcome 更新，不抽随机数。未打开诊断时不额外计算实际状态 Q、不生成诊断目录；报告新增 `diagnostic_configuration` 与 `diagnostic_artifacts` 字段（关闭时后者为空），旧字段及其口径保持不变。

## 验收与限制

定向合成对照使用同一 seed，分别关闭/打开第 1 轮诊断，逐项比较经济 replay、动作轨迹、在线/目标/outcome 网络、三种随机数流、优化器与目标同步计数、greedy Train/Validation 和 `best_agent.pt` 资格；诊断 checkpoint 重新载入后复现所记录 Q 值。另覆盖受迫停机和四类优先状态标签、61 动作排名、Test 封存。已有门控、n-step、失败配额、奖励、岸电回归一起执行。

诊断是有限状态抽样，不能代表整条航段所有 Q 排名；若某重点航段在首个动作前就失败，没有可记录的实际决策，也不会生成虚假 Q 轨迹。保守门控仍不能保证 FC 持续输出或全航段完成，需后续独立正式训练验证。

PowerShell 定向回归：先设置 `$env:PYTHONPATH='src;tests'`，再执行 `python -m pytest tests/test_v4_decision_diagnostics.py tests/test_v4_scheme_a_gate.py tests/test_v4_control.py tests/test_v4_dqn.py tests/test_v4_n_step.py tests/test_v4_failure_td.py tests/test_v4_reward_feedback.py tests/test_v4_scaled_training_entry.py tests/test_v4_feedback_monitoring.py -q`，结果 **116 passed**。诊断文件在补充全局 PyTorch RNG、CLI 参数转发及未合格轮次/Test 封存检查后单独复跑：**5 passed**。两组均为合成/定向测试；没有正式 Train 40 轮运行，也没有 Test 评估。
