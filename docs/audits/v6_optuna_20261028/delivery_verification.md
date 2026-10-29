# 最终验证与基准差异

交付日期：2026-10-29。基准：`544908d`；代码集成：`201b033`；工作分支：`feat/v6-sb3-optuna-audit`。
本阶段只交付可审查代码、审计和微型流程证据，没有启动正式训练或重新运行 Test。

## 验证结果

| 范围 | 结果 | 证据 |
| --- | --- | --- |
| SB3 Mask/TD、环境、checkpoint、物理、经济、冻结参数、数据及保留 CLI 回归 | 143 passed，285 subtests passed，96.93 s | [delivery_regression.json](delivery_regression.json)、[delivery_regression.txt](delivery_regression.txt) |
| Optuna 评价、恢复、SQLite/sampler、预算与计时回归 | 49 passed，90.07 s | [delivery_optuna_regression.json](delivery_optuna_regression.json)、[delivery_optuna_regression.txt](delivery_optuna_regression.txt) |
| V6 默认奖励合成 CLI | 4 步热身、16 次 Adam、2 次 Greedy Validation，完成；恢复后仍为 1 个 Trial | [smoke_verification.json](smoke_verification.json)、[baseline 结果](smoke_results/baseline/trial_0000/result.json) |
| 纯经济奖励合成 CLI | 4 步热身、4 次 Adam、2 次 Greedy Validation，完成；恢复后仍为 1 个 Trial | [economic 结果](smoke_results/economic/trial_0000/result.json) |
| Validation 只读语义 | 上述每次评价的 Adam 增量均为 0 | 两份 result.json 的 evaluation_history |
| 正式/全局环境保护 | 安装前后包清单摘要保持一致；隔离环境 pip check 为 exit 0 | [delivery_environment_verification.json](delivery_environment_verification.json) |
| 其他工作树及冻结文件保护 | 其他 9 个 worktree 的 HEAD、分支、status/路径不变；保护路径相对基准差异为空 | [delivery_preservation.json](delivery_preservation.json) |

两组 pytest 互不重复，共 192 项测试及额外 285 个子测试。这是与本次改动相关的选定回归集，未声称运行全仓库的所有历史测试。
回归只使用固定短轨迹或合成负载，不执行正式完整 35 航段训练及 Test 控制评价。
合成 smoke 复用真实物理执行器，但不读取正式负载；`test_payloads_opened=0` 字段另有 loader 路径审查及缺失 Test 文件的隔离回归支持，不能单独作为 I/O 证据。

CLI 首次运行/完成后恢复的进程墙钟分别为：V6 64.93/17.11 s，纯经济 28.58/12.93 s。
这些时间包含解释器导入、模型和数据库初始化；首次运行还与回归并行，受机器负载影响，不能据此预测完整航段训练速度。
分阶段计时与纯合成范围预算示例保存在各自的 `performance.json`、`budget_example.json`。真实数据预算的测量命令见 [optuna_usage.md](optuna_usage.md)。

## 修复与独立审查

暂停回归覆盖 warmup 中途/末端、非评价步、评价边界及最终 Adam 步；恢复后的网络参数和评价历史与连续运行一致。
Validation checkpoint 已保存而 Optuna 上报/剪枝尚未完成时，恢复先补齐该决策，再继续采集。
无训练生命周期测试覆盖活动配置被改、当前源码或 Train/Validation 文件被改、sampler 缺失、孤立 RUNNING、失败/剪枝预算、完成后重复恢复及 sampler 序列一致性。

预算工具先复现并修复非法参数、缺计时、旧 profile 与残留报告不一致，以及失败训练调用被误计为实际 Adam 的问题。
现在要求可信的 Trial 结果、实际更新数和计时调用数一致；不一致记录返回不可估算。

独立静态审查核对完成数/覆盖率/完整成本的排序、动作 Mask、Test 隔离、暂停恢复顺序、预算、配置身份和序列化。
最后确认的预算计数问题已按失败回归修复；审查范围内没有其余已确认的 P1/P2。该结论不是对任意断电写盘瞬间或未执行正式训练性能的保证。

## 相对 `544908d` 的改动

| 阶段 | 提交 | 具体变化 |
| --- | --- | --- |
| 1：清理与依赖 | `0b0642a` | 新增静态清单、worktree 保护证据与 pytest 入口；可逆归档 2 个历史脚本和 1 个历史测试，删除数为 0 |
| 2：经济审计 | `1e080f6` | 新增离线四模式审计工具、固定短轨迹回归、既有 Train/Val 重评分和经济解释 |
| 3：工具集成 | `201b033` | 新增 5 个 `v6_sb3` 模块、4 个 Optuna 测试文件、固定依赖与兼容证据；忽略新虚拟环境目录 |
| 4：交付 | 本文所属提交 | 汇总实际验证、运行命令、经济/Q 拟合风险与推荐公式、Smoke 报告和完整变更清单 |

阶段 1–3 的逐文件改动见 [baseline_to_integration_changes.csv](baseline_to_integration_changes.csv)。
`src/v6_sb3/training.py`、`masked_dqn.py`、`environment.py`、`curriculum.py` 和 `evaluation.py` 保持基准内容；`src/v2`、`src/v3`、`src/v4`、训练数据和 `docs/results/` 的受控文件也没有改动。
新训练入口从独立目录和配置开始，不跨奖励载入旧 Replay。Zoo 的官方参考源码只作为兼容审计证据保存在文档目录，不进入生产导入链。

## 审查后的决策

先评审 [经济审计](economic_reward_audit.md) 第 7、9 节，决定是否批准以 `zhou_economic_only` 为独立对照及如何验证固定失败分数。
若批准正式实验，先执行已记录的有限计时任务，再确定 Trial 数和更新预算。合成 smoke 只能证明接入可运行，不能证明奖励方案、超参数或控制性能改善。
新环境使用 `--system-site-packages` 借用既有科学包，未静默升级正式环境；它不是完全封闭的跨机器重建环境，具体限制及锁文件核对方法见 [兼容报告](zoo_optuna_compatibility.md)。
