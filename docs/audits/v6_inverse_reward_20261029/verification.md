# 验证、变更边界和交付状态

基点 `814994e`；独立修改前审计提交为 `52af7e6`。所有生产源码、训练数据和历史 `docs/results/` 保持基点内容。没有把未解决的设计伪装成已完成的生产功能。

## 已完成检查

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 新审计、能量价值数学案例、倒数与终止参考 | 57 passed，10.79 s | [audit_regression.txt](audit_regression.txt) |
| 保留环境、Target、经济、调度、Optuna 和 Mask 回归 | 80 passed，56.92 s | [retained_regression.txt](retained_regression.txt) |
| 合计 | 137 passed，最终 0 failed | [verification.json](verification.json) |
| 依赖、help、审计 Smoke、候选数值 CLI | 全部 exit 0 | [cli_verification.json](cli_verification.json) |
| 新真实 Train 审计 | 35×32 前缀步；最短航段两策略各 3272 步正常完成，无学习 | [pre_audit.json](pre_audit.json) 与逐步 CSV |
| 源码与数据保护 | `git diff 814994e -- src data docs/results` 为空 | 本分支 Git 差异与 preservation.json |

新测试命令：

```powershell
& $v6Python -m pytest tests/test_inverse_pre_audit.py tests/test_realtime_value_design.py tests/test_inverse_reward_analysis.py -q
```

保留回归命令：

```powershell
& $v6Python -m pytest tests/test_v6_direct_environment.py tests/test_v6_target_env_steps.py tests/test_v6_economic_reward_audit.py tests/test_v4_independent_economics.py tests/test_v6_episode_schedule.py tests/test_v6_episode_optuna.py tests/test_sb3_masked_dqn.py -q
```

这不是运行全仓库所有历史测试，也不是“新 inverse 模式验收通过”。审计参考函数试写时，TDD 发现先求终止尾步倒数再替 −1 会在有限零/负值或极小尾成本处错误抛异常，已在测试内参考函数修正为正常前缀求逆、真实末步直接替换；非有限账本仍报错。没有改生产失败处理。

## 数值容差与数据缺失

- 零/负费用分类按实际符号比较，没有 epsilon、abs 或 clipping。
- 标准分布使用 NumPy 默认线性分位数；缺失值与零严格分开。
- EOL 负例固定值检验绝对容差 `1e-14` CNY；物理平衡/固定 SOC telescope 固定短轨迹检查 `1e-12`；充放电循环 SOC 容差 `1e-15`、费用 `1e-11`。
- 库存/势函数数学案例使用显式 pytest 近似比较及原 FC 曲线，不宣称全部浮点逐 bit 相等。
- 已归档 Train 累计汇总与稀疏记录不足以复原全部历史逐步成本分布或累计倒数。JSON 保留 null，新固定策略轨迹另列，不冒充旧模型数据。
- 完整实时模型费用未计算，因为放电估值/初始价值/价值状态未选定；充电分支条件差额不等于完整新模型结果。

## 短时执行和计时

主审计轨迹为 1120 个前缀步加 6544 个最短航段步；另有合成/边界、循环及独立 35×1 步审计 Smoke。它们都不学习新策略，不构成正式训练。两条完整轨迹运行内分别约 7.849/8.380 秒；单独 CLI Smoke 进程为约 41.160 秒，包含加载全部 Train、构造与因果窗口等开销，不能用作未来 NN 训练速度预测。

没有新奖励训练 Loss、TDMAE、模型改善或实测 13 动作 Q 排名，因为生产门禁未通过，未进行这些实验。已有 TD 算子测试只证明终止 −1 不 bootstrap 的算术能力。没有正式 500 episode、长时间 Optuna、Test 评估或新奖励模型/Replay/SQLite 交付。

## 新增/修改文件与保留项

新增审计脚本 `scripts/audit_inverse_economics.py`、三个审计测试文件及本独立审计目录。没有新增生产 `inverse_economic`、实时价值模型或 metadata 版本，也没有修改旧失败分数。

500 episode、有放回 35-Train 抽样、1 秒 direct 环境、32:1 cadence、原 batch/default/network、10000 正式步 Target、普通 masked one-step DQN、gamma=1、Huber、35/8/4 数据划分均保持原实现。旧 reward、checkpoint 含义和 Optuna 入口没有悄悄改变。

继续上线需要先明确正常 C=0 的 reward 定义、选择有依据的放电/初始储能价值模型及状态表达，并接受或重新设计倒数目标与真实费用目标的差异。当前不提供不存在的训练、Validation、单 Trial、搜索/恢复或最优参数重训命令；这些待决项与可执行审计命令见 [README.md](README.md)。
