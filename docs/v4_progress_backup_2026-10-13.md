# V4进度备份说明（2026-10-13）

- 正式进度分支：`feat/direct-power-mlp-ddqn`。保存既有SOC物理掩码一致性修复、实验标识/种子接口、源码哈希、续训诊断归档、Replay覆盖与FC变化统计，以及研究报告和小型诊断附件。
- PER候选备份分支：`feat/v4-per-prep`。基于本次正式分支提交保存；默认`uniform`，显式选择`prioritized`才启用。正式分支的经济Q采样未因此切换成PER。
- PER候选包括独立诊断脚本，不加入FC门控、强制出力或新奖励；134项检查是此前候选准备阶段的结果，本次提交仅运行差异和Python语法检查，没有重新训练。
- `docs/results/v4_per_jointfit_state_followup_20261013/independent_probe_models.pt`约418 KB，是冻结目标诊断副本，明确标记非正式checkpoint；不是`best_agent.pt`，不能用于正式续训。
- 两项联合诊断见[报告](v4_per_jointfit_state_diagnostic_2026-10-13.md)。小规模拟合没有证明PER或时间状态改善完整经济控制，正式状态仍为7维。
- 本地清理释放约37.97 GB，项目约45.50→7.53 GB。两组40轮输出已有完整Git归档；其他旧逐轮JSON已无损压缩，原始最终轮轨迹及26个正式checkpoint保留。详细清理清单位于本地`outputs/cleanup_20261013/cleanup_manifest.json`。
- `outputs/`继续被Git忽略。完整A/B训练状态、Replay、逐轮压缩轨迹以及清理清单没有因此上传GitHub；提交源码和报告不等于完整实验备份。
- 历史checkpoint的源码提交/哈希核对保持严格。本次提交改变HEAD，历史权重仍可用于匹配版本的只读诊断，不应按新提交直接续训或切换到PER。
- 原始数据、正式标签、Train/Validation/Test划分均未改动。本次没有读取Test或启动训练。

较早准备报告记录的是当时未提交状态；当前备份状态以本说明和远端对应分支为准。
