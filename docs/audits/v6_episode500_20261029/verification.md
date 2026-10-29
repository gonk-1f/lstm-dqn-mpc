# 交付验证与剩余决策

2026-10-29；基点 `b212483`，新分支 `feat/v6-sb3-economic-episode500`。

代码分为 `b61d414`（共享 direct 环境与 Target 时钟）、`780fd9b`（完整 episode 训练器）、`e4730f6`（共用 Optuna）三个提交；本文件所属交付提交另存审计工具、回归和运行证据。发布前再次确认 PR #1 尚未合并，因此新 PR 指向 Optuna 审计集成分支。

## 实际检查

完整命令输出为 [regression.txt](regression.txt)，摘要和 SHA256 为 [regression.json](regression.json)：**286 passed，283 subtests passed，164.30 秒**。范围为全部新 episode/单线程/Target/Train-only 审计/Optuna 测试，加受影响的旧 SB3、v4/v5/v6 控制、经济、数据、冻结参数和 Optuna 回归；不是全仓库所有历史测试。

| 需求 | 实际证据 |
| --- | --- |
| 双奖励经济/物理一致性 | 新环境逐步对照旧执行器/Queue，包含功率、SOC、氢气、退化、reward、Mask、终止原因；固定短轨迹与旧经济回归通过 |
| 单线程真实数据等价与速度 | 672 对真实 Train step 对照；32/64/128 步各 3 次，中位耗时比 1.4167/1.4341/1.3000；原始数据见 environment_benchmark.json |
| 500 episode i.i.d. 调度 | 35 个 ID、500 次独立均匀有放回抽样与独立 Random.choice 精确对照；第 83 次快照恢复后剩余顺序、计数一致；全局 RNG 不受影响 |
| epsilon episode 单位 | 第 1/400/401/500 episode 对应 1/.05/.05/.05 |
| 32:1 与 Target | 真 SB3 小网络实际 Adam，10000/20000 正式步分别记录 312/624 次先前 Adam；20000 步结束共 625 Adam，预热不计时钟，新旧同步不同时生效 |
| 自然边界与真实 Train | 最短 Train `parent_046_seg02` 完整 3272 步、102 Adam，另计 32 步热身；没有在 32 的倍数处截断 |
| 暂停恢复 | 多个 warmup/正式/32 步边界的模型 tensor、优化器、Replay、RNG、顺序、Target 与 JSONL 精确比较；真实 direct CLI 和活动 Trial 另行验证 |
| Validation/Test | 独立 Greedy env、inference、RNG 还原，Adam/Replay/训练计数不变；拒绝混入 Test split；真实短验证不打开真实 Val/Test payload |
| 轻日志与保留 | 无 Adam 时 Loss/TDMAE 为 null；不保存完整 Q/minibatch/逐秒轨迹；保留 best 和 latest N，孤儿 checkpoint generation 回归通过 |
| Optuna 共用与预算 | 同 runner、32:1、纯经济、环境步 Target；SQLite、sampler、活动配置、孤儿 Trial、固定预算、参数导出与 fresh 目录初始化测试 |
| CLI | help、双奖励 smoke、训练暂停恢复、inspect、compare、实际 Trial、完成后恢复、活动 Trial 暂停恢复、导出参数、新模型 smoke 均 exit 0，见 cli_verification/ |

真实 Train 有界检查的 [real_episode_verification.json](real_episode_verification.json)：只打开一条 Train payload，32 个真实 warmup 步加 3272 个正式步，102 次 Adam，完整运行内耗时 **18.105 秒**（不含解释器导入）。累计奖励 `-21001.8842316763`，完整经济成本 `21001.8842316763`，恒等残差在 `1e-7` 容差内。该次仅用 2 条各 8 秒的合成 Validation 检查流程；没有执行正式 8 条 Validation，也不代表完整 35 条 Train 的控制性能或全负载平均吞吐。

CLI 纯经济/历史对照的微型运行各 3 个 64 秒合成 episode，192 个正式步、6 次 Adam。纯经济实际暂停于 64 步热身后第 1 个正式步，再恢复到完整结束。Optuna 微型 Trial 完成 2 个 64 秒 episode，128 正式步、4 次 Adam；另一次在 33 正式步、1 Adam 后暂停，恢复复用同一个 Trial，未新增分配。

验证用新模型、Replay 和 SQLite 仅位于忽略的 outputs/ 临时运行目录。Git 中的 cli_verification/ 仅有 CSV/JSON/JSONL/终端文本，不包含模型、Replay 或数据库。

## 故障边界

独立审查确认了 checkpoint 写入被强制终止后 generation 重名的问题，已先复现两个失败案例再修复。现在编号避开完整及 .writing 孤儿，latest 成功写入后才清理受限目录；递归删除前验证 resolved 目标仍在 checkpoints 内。正常 Ctrl+C 与上述特定孤儿场景已测试；没有声明任意断电瞬间所有文件都能恢复。

Optuna 的支持暂停方式为正常 Ctrl+C 或 `--stop-after-steps`。若在 SQLite ask/active-marker/tell 等任意硬终止窗口留下孤立状态，会明确拒绝继续分配，而不是静默替换 Trial。模型重新训练只导入参数 JSON；新目录初始化不会把 Trial 已训练模型当作独立重训结果。

## 失败代价是正式运行前的待确认项

当前 `10132.66` 原值保留，独立任务失败分数没有被混入真实经济账本。Train-only 既有行为重评分的 122/125 对照存在失败回报更高；这不证明新策略主动失败。旧参考按当前 split 映射有非 Train 身份，因此不能宣传成当前划分重新认证的 Train-only 标定。

候选标定、数值敏感性及三条尚无归档完成证据的 Train 已列在 [failure_penalty_audit.md](failure_penalty_audit.md)。正式 CLI 默认拒绝启动；`--acknowledge-failure-penalty` 是调用者确认接受继承值风险的标记，不是自动选定新分数，也不解决其尺度问题。

## 预算与下一步

500 episode 在 seed42、全完成假设下为 6119941 正式步。短微测推算的物理采集、Adam、热身物理 step 和十次 Validation 物理 step 合计约 4.789 小时；100/150 episode Trial 同类工作项约 .990/1.473 小时。它们漏掉策略推断、完整 Replay、reset、报告、checkpoint 等，失败还会改变长度与行为，不能当作可靠总时长。详见 [environment_benchmark.md](environment_benchmark.md)。

先确认失败代价，再由用户运行 [README.md](README.md) 中单 Trial 的真实计时命令，决定正式预算。500 episode、真实 100/150 episode Trial、长搜索与完整独立重训没有在交付过程中自动启动；Test 未加载或评估。已有审计、冻结基线与历史结果保留。
