# v4 分阶段 beta / update cadence 对照计划

**Goal:** 先按每轮 greedy 资格与经济成本选择 beta，再在该 beta 下公平比较三种更新调度。

**Architecture:** 复用 v4 的物理回放、奖励、经验存储和 Double-DQN；新增监控训练入口和额度调度。评估恢复训练 RNG，不写 replay，不更新网络。阶段 1 target 仍按轮同步；阶段 2 的 A/B/C 全部按实际经济 optimizer 次数每 1000 次同步。各独立运行从 seed42 新初始化，不从最佳模型续训。

**Fixed:** gamma1、batch64、Adam1e-4、ε1→0.05、8-128-64-61 MLP、61个动作、SOC phi 和正式经济/岸电/末端结算；Test 封存。CPU 线程配置保留当前值；可并行计算独立配置，不共享模型、回放或随机数。

- [x] 在 `tests/test_v4_monitored_training.py` 验证 greedy 不消耗训练 RNG、失败 Train 跳过 Validation、最佳资格与最低成本选模，以及 replay 容量饱和后的累计插入计数。
- [x] 在 `tests/test_v4_experiment_schedule.py` 验证 8/16 条有效插入预算、跨 episode/round 余数保留、实际 optimizer 后每 1000 次 target 同步，以及 A 保持每完成样本16次更新。
- [x] 在 `src/v4/experiment_schedule.py` 实现调度与 checkpoint 资格判断；在 `src/v4/monitored_training.py` 实现共享训练/greedy监控/保存。
- [x] 在 `src/v4/staged_study.py` 实现阶段屏障、独立配置运行、曲线与表；记录源码提交和 manifest 哈希，输出每轮报告。
- [x] 验证阶段1 beta250 的训练权重/回放/RNG轨迹与原训练相同；监控只新增诊断与最佳轮次选模。
- [x] 阶段1：250/500/1000/2000，各40轮，保留资格正确的最低 Validation 可比费用 checkpoint；输出全部每轮曲线/表。
- [ ] 若存在合格 beta，选择最低合格费用者；阶段2从相同seed新初始化执行 A/B/C，target统一1000经济更新。额度包含 bootstrap 新入 replay 条目，单列 bootstrap 和探索更新；失败未完成 suffix 不进入 economic replay 的语义保持。
- [ ] 若阶段1无合格 beta，阶段2不运行，明确未满足前提，不能推荐未经实验的 cadence。
- [x] 核验阶段1 best checkpoint 实际重放、费用口径、计数与 Test0；阶段2仅保留已有合格checkpoint，未在暂停后追加模型评估。

## 用户暂停（2026-10-08）

按最新指令停止全部训练并归档所有现存结果。阶段1已完整结束，选择beta500；阶段2 A完成18个监控轮次，在第19轮停止，B完成19个监控轮次，在第20轮停止，C未运行。完整三组cadence对照仍未完成，不再执行。六份保存模型只有online权重，没有optimizer/replay/RNG续训状态。

归档保留完整/部分轮次、原始日志、中断及资源错误证据、权重和已有轨迹；报告中单列中断轮次计数未知的限制。此次不推荐未经完整对照的cadence或KAN进入条件。
