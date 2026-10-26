# V6 学习率 1e-4 运行与逐轮 Q 报告

本次授权配置为 Adam 学习率 `0.0001`、SOC 权重 `0.2`、FC 波动权重 `0.05`。
其余保持标准 DQN、单步 TD、1 秒执行、13 档 `[-60,60] kW` 实际增量、seed 42、
50,000 条 Train 暖身经验和 8 次 Train 完整遍历。观察结果不触发算法或参数自动修改。

每轮 runner 保存结果后调用 `scripts/report_v6.py RUN_DIRECTORY`。报告脚本只处理已有记录，
不创建网络、优化器或环境运行；Test 不读取。`write_round_summary(root, entry, previous_entry=None, config=None)`
也可直接调用，返回摘要字典并写入 `round_NN_summary.json` 和 `round_NN_summary.md`。

每轮报告 Train 完成数 `/35`、Greedy Validation 完成数 `/8`、实际环境步和 Adam/Target 计数。
失败分别按 SOC 上边界、下边界、内区和原始失败原因统计；边界侧与增量动作不可达是交叉分类，不能相加。
FC 变载和相邻正反向次数按实际执行物理小时折算，不能直接比较不同长度失败前缀的原始计数。

经济报告区分全部已执行区间、仅失败前缀、仅完整航段子集。列出实际 FC 氢耗质量、氢成本、
FC 退化、电池退化和原四项账本总额，并另列有符号的电池等效氢修正与修正后经济成本。
SOC、波动和失败惩罚不属于真实经济支出；只有固定划分全部航段完成，才给固定全套完整成本。
旧记录未直接保存氢耗质量时，使用项目正式氢价 `21.9 CNY/kg` 从实际氢成本换算。

`entry['q_diagnostics']` 保留四类证据：

- `training_td_fit`：在线训练 draw 的 Q、TD 目标和拟合误差统计。
- `checkpoint_replay_fit`：当前 checkpoint 对已有 Replay 样本的只读拟合；不执行优化。
- `fixed_probes`：固定状态的全部 13 个 Q、动作排名和合法性；`file` 指向完整记录。
- `train_failure_value`：已有 Train 失败轨迹的累计实际回报、当前 Q 与真实采样次数；`file` 指向完整记录。

训练期间 TD 目标随 Online/Target 参数变化，**TD target 不是 Q\***。较小拟合误差只代表对对应目标的拟合，
不能推出策略更优。每轮同时展示 Validation 完成、SOC 边界失败与功率波动；若误差下降而这些指标未改善，
报告明确指出。旧 Validation 固定探针仅供报告，不回流 Replay；单个状态的 Q 或排名不能独立支持整体结论。
失败轨迹实际回报来自已发生的行为策略，也不是最优 Q 的真值。
各经验类别的 Markdown 表直接列出 Q、TD 目标及有符号 `target − Q` 残差的均值和范围，
并同时显示 Replay 类别总量与测量数。分层样本和 TD 目标随轮次变化，weighted MAE 的变化仅作描述，
不能解释为对 Q* 的拟合改善。

旧运行缺失的 Q 诊断标为“未记录”，不补造历史统计。旧失败记录仅在缺少终止 FC 字段时保留原 Replay 读取方式；
新运行使用 `previous_fc_terminal_kw`，避免依赖环形 Replay 中可能已覆盖的旧经验。
