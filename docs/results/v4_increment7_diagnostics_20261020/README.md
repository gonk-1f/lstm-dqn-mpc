# 2026-10-18至20日DDQN诊断归档

本目录保存最近checkpoint、更新时序、奖励分项/缩放、动作mask、状态信息、损失/PER、统计对账与频繁变载反事实诊断。它是研究证据归档，不代表算法已经修复或消融已经完成。

## 报告索引

- [checkpoint及反事实](checkpoint_audit_round100/checkpoint_audit.md)
- [Q学习根因复核](checkpoint_audit_round100/q_learning_root_cause_review.md)
- [冻结共同拟合历史诊断](checkpoint_audit_round100/jointfit_history_20261019/frozen_jointfit_history_report.md)
- [奖励分项及缩放](reward_scale_audit_20261019/reward_scale_audit.md)
- [动作可达性及物理mask](action_mask_audit_20261019/action_mask_audit.md)
- [三维状态信息](state_information_audit_20261020/state_information_audit.md)
- [损失与PER](loss_per_audit_20261020/loss_per_hyperparameter_audit.md)
- [统计口径对账及基线状态](statistical_reconciliation_20261020/reconciliation_and_baseline_status.md)
- [最新逐状态变载/保持比较及反事实](fc_switching_causal_audit_20261020/fc_switching_root_cause.md)

当前分支feat/v4-increment7-zhou-td1。旧100轮、交错更新R81及早期基线状态报告分别保留其真实运行身份和记录时间；不能混作同一模型或最终100轮结论。奖励缩放、状态扩维、MSE等长训练对照尚未因审计而执行。

## 文件格式和复算

超过1 MB的CSV/JSON等文本使用gzip无损压缩；小文件及图表保留原文件名。CSV可直接通过`pandas.read_csv('文件.csv.gz')`读取；JSON通过`gzip.open(..., 'rt', encoding='utf-8')`读取。正文提及的未压缩文件名，在本目录中可能对应同名`.gz`版本。所有源文件SHA、归档SHA、原始字节数和压缩字节数见`archive_manifest.json`，每个gzip均已读取解压内容校验。

源诊断脚本的路径仍以原outputs布局为上下文；复算时使用原工作区或按manifest恢复布局。本次不运行这些脚本、训练或测试。`runtime_worktree_changes.patch.gz`保存诊断所对应但尚未提交的本地交错更新源码及定向检查差异，便于追溯；本提交不将该补丁应用到源码。

checkpoint/Replay/模型权重仍保存在本地原实验及诊断目录，本Git归档不包含权重。提交推送不能替代大型训练产物的备份。

## 本次本地清理

`cleanup_plan.json`逐文件限定清理范围：仅两组历史绝对功率实验（含独立航段实验的第100轮续训目录）的早期逐轮轨迹。保留第1、10、20、…、90、99、100轮中实际存在的轨迹，并无损压缩未压缩的保留文件。报告、指标、全部checkpoint、数据集、源码、旧增量100轮诊断基准和运行中交错更新目录均保留；DDPG/TD3不动。

旧实验摘要另保存在`cleanup_retained_summaries/`。删除清单中的非关键早期轨迹没有上传Git，清理后不能从本次Git归档恢复这些全量轨迹。执行结果及实际释放空间以后续`cleanup_result.json`为准。

## 正在运行训练的源码身份

现有续训校验严格比较source_commit与源码SHA。本次归档提交会推进Git HEAD，但不会改变正在运行进程的源码或参数。若该运行中断，不能直接在新HEAD下声称可原样恢复；应使用原提交dc3c1947bc4dc400710896dfe3acca2e9ae3f54d加归档的runtime_worktree_changes.patch.gz还原到隔离工作树，再按原身份恢复。不要改checkpoint元数据绕过检查。
