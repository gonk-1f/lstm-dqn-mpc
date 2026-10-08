# 文档索引：当前 v4 与历史资料

## 当前主线

1. [v4 方法与默认配置](v4_direct_power.md)：当前正式入口、8维状态、61动作、奖励与经济账本。
2. [失败轨迹经济 TD 修复](v4_failure_economic_td_2026-10-08.md)：d111c99 的实现和合成证据。
3. [本次仓库整理与依赖审计](v4_repository_audit_2026-10-08.md)：迁移、删除、保留、链接与哈希。
4. [正式训练前数值诊断](v4_td_numerical_diagnostic_2026-10-08.md)：仅固定合成 replay 的缩放对照，未改正式默认。
5. [统一奖励缩放接入与正式训练命令](v4_reward_scale_training_2026-10-08.md)：默认仍为1，用户将显式运行0.001的单组40轮；本次实现验证未启动该训练。

当前 `v4.feedback_study` 默认 beta500、redistributed、replay32、target500、batch64、γ1、n1、epsilon1→0.05、失败惩罚倍率1、奖励缩放1。
只读查看 [入口代码](../src/v4/feedback_study.py) 或运行 `--help`。正式训练需要显式轮数和新目录；本次没有启动。

## 实验和验证证据

- [旧40轮及中断原始归档](results/v4_staged_beta_cadence_seed42/README.md)：既有日志、图表与 best checkpoint，保持原路径/字节。
- [奖励重分配契约](results/v4_reward_feedback_implementation_20261008/reward_contract_verification.json)。
- [失败 TD 验证](results/v4_failure_economic_td_20261008/verification.json)与[合成动作排序证据](results/v4_failure_economic_td_20261008/learning_order_evidence.json)。
- [仓库整理与回归核验证据](results/v4_repository_td_audit_20261008/README.md)及[当前默认奖励模式合成数值结果](results/v4_td_scale_diagnostic_20261008/redistributed/summary.json)，属于诊断而非正式训练。

## 模型、数据和历史

- [当前共享模型与数据依据](reference/README.md)：FC/Battery物理、退化、经济、数据来源；v2文档仍有当前依赖。
- [历史总索引](history/README.md)：v2/v3/v4 旧路线、研究资料和已完成计划。
- [v1 原归档](archive_v1/)：保持旧 Git 恢复与来源证据。
- [原清理manifest](v2_cleanup_manifest.md)、[2026-10-07清理记录](history/v4/v4_cleanup_2026-10-07.md)及[当日inventory](v4_cleanup_inventory_2026-10-07.json)。

旧训练入口是历史复现工具，参数和失败语义可能与当前不同。相对链接和图片已审计；历史文字中的已删除本地生成路径另列清单，不伪造不存在的图、成本或新模型评估。
Train/Validation/Test 划分、正式数据与已审核原始结果不改动。Test 不参与本次数值诊断。
