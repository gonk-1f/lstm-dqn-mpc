# V6 小负载审计与 1/(1+C) 奖励

基线为 PR #3 的 `2c2577ecd25d41b684a2c82cd345b6d3ad843def`，新分支为 `feat/v6-sb3-one-plus-reward`。负载审计和设计报告先独立提交，生产实现及最终验证另行提交。

## 修改前证据

- [负载来源、死区与完整 Train 分布](load_audit.md)，[机器可读结果](load_audit.json)，[全部小负载节点及来源分列的物理/成本值](load_small_nodes.csv)。统计覆盖全部 35 Train；1 kW 仅为描述性分箱上界，不是新增执行阈值。
- [接口预审计](interface_audit.md)：模式、恢复、Validation、checkpoint、参数导入导出、比较的隔离。
- [奖励与真实失败语义设计](reward_design.md)：使用现有经济模型的 CNY 成本；未实施前次待决的实时 SOC 库存估值。
- [执行计划](../../superpowers/plans/2026-10-29-one-plus-economic-reward.md)，[历史产物修改前清单](existing_artifact_inventory_before.json)。

`parent_015_seg01` 第 1 秒的数值已由冻结 Train 原始通道生成链逐点重现：它是 PCHIP 1 秒重构节点，不是该时刻独立测得的负载。已有正负 1 kW 边界吸附规则只用于选取航段边界；运行时的 `[-1,0]` 归零规则不等于正负对称死区。因此本次保留物理负载数据和规范化函数，不新增正负载死区。若要将内部小正值也归零，仍需单独的数据契约变更依据。

已有固定诊断轨迹只覆盖部分小负载样本。CSV 将其与独立单秒反事实核算分列：后者每行均固定 fresh SOC=.6、FC=0、全新老化状态，不串联成轨迹；没有动作的 terminal boundary 只作元数据，不制造 transition。不得把反事实值或固定策略结果称为训练后 DQN 轨迹。

## 实现与验证状态

此索引的修改前版本尚未宣告生产实现或测试通过。完成后的实际命令、输出、Smoke 边界、改动摘要和后续命令将记录在 `verification.md` 与 `commands.md`；正式 500 episode、长 Optuna 搜索及 Test 均不在本次执行授权范围。

历史 checkpoint 与 study 的源码哈希校验保持严格。历史产物不迁移或重写；需要续训旧产物时，应回到其原始提交和依赖。新模式使用独立新目录与身份，不能载入旧 Replay 作为其训练状态。
