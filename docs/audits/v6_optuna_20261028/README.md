# V6 SB3 审计与独立 Optuna 接入

基准为 `feat/v6-sb3-baseline` 的 `544908d`，工作分支为 `feat/v6-sb3-optuna-audit`。
本交付先保留冻结实验，再提供独立搜索入口。正式训练预算和奖励方案需审查后确定。

## 阅读顺序与交付清单

| 内容 | 文件 |
| --- | --- |
| 阶段计划及执行边界 | [implementation_plan.md](implementation_plan.md) |
| 依赖、入口、保留/归档/删除依据 | [cleanup_report.md](cleanup_report.md)、[cleanup_manifest.json](cleanup_manifest.json) |
| 清理前后 Python 文件清单 | [file_inventory_before.csv](file_inventory_before.csv)、[file_inventory_after.csv](file_inventory_after.csv) |
| 导入及调用关系 | [dependency_edges_after.csv](dependency_edges_after.csv)、[dependency_entrypoints.csv](dependency_entrypoints.csv) |
| 其他 worktree 保护检查 | [worktree_preservation.json](worktree_preservation.json) |
| 四种奖励、经济账本与退化审计 | [economic_reward_audit.md](economic_reward_audit.md) |
| 既有轨迹的离线重评分 | [offline_reward_comparison.csv](offline_reward_comparison.csv)、[reward_source_manifest.json](reward_source_manifest.json) |
| Zoo/Optuna 版本、官方源码及安装证据 | [zoo_optuna_compatibility.md](zoo_optuna_compatibility.md)、[raw_compatibility/](raw_compatibility/) |
| 搜索空间、评价、恢复与运行说明 | [optuna_usage.md](optuna_usage.md) |
| 最终验证及与基准的差异 | [delivery_verification.md](delivery_verification.md) |

阶段 1 提交为 `0b0642a`，阶段 2 提交为 `1e080f6`。阶段 3 和阶段 4 分别提交适配实现与交付记录。
阶段 3 代码提交为 `201b033`；最终验证于 2026-10-29 完成，沿用本任务起始日期的审计目录。
早期报告中的“未 commit”描述各子任务产出时的状态；最终状态以交付验证和 Git 历史为准。

## 奖励方案的结论

冻结 `v6_energy_soc_ramp` 的实现排除了真实 FC 和电池退化费用，训练目标与完整经济评价存在已证实的差异。
在独立完整航段、`gamma=1`、SOC 折氢只结算一次等条件下，`zhou_economic_only` 的累计回报等于完整经济评价的负值；失败分数单独处理。
既有 Validation 轨迹的最大浮点恒等残差为 `1.164e-10 CNY`，这证明核算关系，不证明新策略更优。

建议先评审纯经济模式与当前失败分数的尺度，再批准独立的 Train/Validation 实验。
新入口默认仍使用冻结 V6 奖励，纯经济模式必须显式选择；两者拥有独立目录、配置和 checkpoint 身份。
没有自动调整奖励权重、物理参数或经济系数。
