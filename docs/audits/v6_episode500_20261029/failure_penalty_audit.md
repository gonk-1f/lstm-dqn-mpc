# V6 500 episode：Train-only 失败分数尺度审计

日期：2026-10-29。**正式训练前必须确认沿用 `F=10132.660087898294` 的适用性。当前值保持不变；本审计没有提出自动替换值，也没有执行正式训练。** 新 CLI 的显式 acknowledge 是接受已知风险的记录，不能证明该分数已经重新标定。

## 证据范围与缺失性

只读现有 `docs/results/sb3_dqn_1s_20261028/raw/round_01..08_train_samples.csv`，共 280 条既有逐航段累计汇总，以及 8 个 `round_*_train_failure_prefixes.csv.gz`，共 47 组、14,241 条稀疏观察。用冻结 split manifest 校验全部汇总恰好各含当前 35 条 Train；未打开 Validation/Test 航段或其成本。来源路径及 SHA256 见 `failure_penalty_source_manifest.json`。

这里有两种不同的证据：逐航段 CSV 保存真实执行器当时累计的氢气、FC 退化、电池退化、SOC 折氢与失败分数，所以可以离线重新计算**已归档行为**的完整/前缀经济回报；各分项与累计成本的最大核对残差为 0 CNY。它们没有被本次独立重放核验。失败逐步 CSV 只保存首步、少数较早时刻和尾段，缺中间动作与两项退化 ledger，不能用稀疏行求和替代累计成本，也不能从中恢复完整经济单步分布。机器结果保留这些字段为 `null`。

32/35 个 Train ID 至少有一次归档完成。`parent_015_seg01`、`parent_017_seg01`、`parent_036_seg01` 没有完成证据；不能为其填写完成经济成本。没有重放 35 条 Train，没有从 checkpoint 推断控制，也没有运行长航段控制任务。

## 旧参考为何不能直接认证新奖励

`src/v4/failure_penalty_reference.json` 的来源 SHA256 与旧 `best_profiles.json` 文件一致：`3e1fae46266d99629116b52156840541a33136be92f9f4e68e1e8b03256a0642`。本工具对该旧混合归档只校验文件字节哈希，没有解码其 Validation 部分或读取任何新轨迹。沿用分数是旧 30 条 `zero_boundary_*` 完成样本 comparable ledger 的 nearest-rank p95，范围 1,343.16–10,468.94，p95 为 10,132.66。它不是当前 35 条 `parent_*`、1 秒动作粒度的重新标定。

旧 comparable 账本口径是氢气与退化等 physical ledger；新 `zhou_economic_only` 还含 signed SOC 等效成本，且逐秒动作可改变启动事件数及真实退化累计，不能把两个参考集合直接互换。

**还有 split 迁移问题：** 仅按 manifest 的父样本身份对照，旧 30 个参考 ID 中，21 个属于当前 Train，8 个属于当前 Validation，1 个属于当前 Test（`zero_boundary_013`）。这是旧参考的历史来源，不是本次使用 Test 调参；但是，“旧文件写着 Train”不足以称该常数已经按当前冻结 split 做过 Train-only 标定。本次新候选的计算仅使用当前 Train 归档汇总。没有利用这 9 个当前非 Train 样本的成本重算任何候选。

## Train 经济尺度

下表费用均排除独立失败分数；分位数使用 nearest rank，median 为标准中位数。

| 既有 Train 行为 | 数量 | 最小 / CNY | 中位 / CNY | p95 / CNY | 最大 / CNY |
|---|---:|---:|---:|---:|---:|
| 完成航段完整经济成本 | 233 | 26,829.82 | 161,569.02 | 468,782.35 | 1,051,677.98 |
| 失败真实执行前缀经济成本 | 47 | 13,581.77 | 47,498.68 | 148,228.58 | 281,513.13 |
| 每个有完成记录的 ID 取已知最低完整成本 | 32 | 26,829.82 | 97,053.23 | 319,318.24 | 697,457.67 |

233 条完成记录全部高于现有 F；这本身不证明失败更划算，因为还必须加入失败前缀成本。47 条失败中 27 条为 `soc_limited`、20 条为 `increment_action_limited`；没有归档 `structural_power` 失败。

真实短前缀的单步经济证据另外保存在 `environment_benchmark.json`：最短 Train `parent_046_seg02`，固定合法保持策略、仅 32 秒，FC 一直关闭，成本单步 0.06742653–0.06940929 CNY，总成本 2.1888949570 CNY，回报为其负值，恒等残差为 0。它没有启动事件，不能代表其他 Train 状态、完整航段单步分布或失败前缀。一般实现量级中，FC 一次关闭到正功率的原始启动退化为 358.65 CNY，另有线性变载及运行成本；寿命达到 EOL 后实际 capped 费用可能降低。不能把每一步都当成 0.07 CNY，也不能把每次动作都当成一次启动。

## 失败定义、位置与不可控性

依据 `src/v4/control.py`、`src/v4/failure_replay.py` 和新 direct adapter 共用契约：

- `soc_limited`：全功率网格在电池功率约束内有解，但结合当前 SOC 后物理可行集为空。
- `structural_power`：即使暂不考虑 SOC，全功率网格与电池功率范围也无法平衡当前 normalized load；在该时刻不能靠功率策略解决。现有聚合范围为 FC 0..600 kW、电池 -624..1248 kW，正负载大于 1848 kW 是明确的结构性超限例子。当前 Train manifest 最大峰值为 1103.8994 kW，且归档没有该类失败；这一 metadata/归档结论不排除 SOC 或增量限制。
- `increment_action_limited`：物理功率集合非空，但与当前 FC 的 13 个 ±60 kW 增量动作交集为空。它取决于历史状态及负载变化，不能笼统断言总能由当下策略避免。SOC 失败也不能仅凭分类就认证存在全航段可达策略。

失败发生在尚未执行的下一决策 F。`prepare_failure_replay` 把罚分仅扣在 **F−1 的最后一个真实 transition**：`reward_last = base_reward_last - F_penalty`，`done=True`，successor mask 清空；不为失败动作造物理账单或虚假 transition。若首决策即失败，属于 event-only，没有真实 transition 可以承载罚分，也不产生 TD 样本。失败 TD target 因为 terminal 不 bootstrap，但仍是 `base_reward_last - F_penalty`，不能恒写为 `-F_penalty`。

## 已存在的回报排序风险

在 gamma=1 下，同一 ID 的完成轨迹成本 C、失败执行前缀成本 P，对应经济重评分为 `R_complete=-C` 与 `R_fail=-(P+F)`。当 `C-P > F` 时，失败回报更高。

对当前归档中同一 Train ID、相同初始任务契约而不同轮次行为，共能配对 125 组；**122/125 组在现有 F 下，失败经济回报高于完成经济回报**。逐组来源与数值在 `train_same_id_failure_comparisons.csv`，不是跨不同航段拿低前缀比较高完整成本。

一个具体 Train 例子：`parent_037_seg01`，第 1 轮失败覆盖 53.20%，前缀成本 P=206,355.0887；第 7 轮完成成本 C=1,051,677.9761。现有 F 下，失败重评分 −216,487.7488，完成重评分 −1,051,677.9761；相差 835,190.2273。两条行为的 break-even F 是 845,322.8874 CNY 等效奖励单位。

这证明**在既有真实行为的经济重评分中存在该排序风险**，不证明某个新训练策略主动谋求失败，不是同一已执行前缀之后强行完成的反事实，不证明完成轨迹最优，也不是新纯经济训练的实测结果。历史行为本来由旧奖励产生。

## 备选标定与敏感性建议，均未应用

| 仅由当前 Train 推导的对照候选 | 候选 F / CNY 等效奖励单位 | 125 组中失败仍获更高回报 |
|---|---:|---:|
| 现有继承值，仅作对照 | 10,132.66 | 122 |
| 已知完成成本 p95 | 468,782.35 | 16 |
| 已知同 ID 正向成本差最大值 | 845,322.89 | 0（最大差那组为打平） |
| 已知完成成本最大值 | 1,051,677.98 | 0 |

这些值仅是**待决策方案**。p95 不保证阻止所有已观察排序；最大差是样本内打平阈值，严格劣于完成还需要明确的 margin；完成最大值也不是任意未来策略的上界。重复轮次让同一 ID 多次进入分布，而按 ID 的最低已知完成成本只覆盖 32 个 ID。不能把这些样本依赖阈值说成全 35 航段的最优标定。

可选流程是先固定 Train 的可比较完成参考，分别评审 global p95、global maximum 和同 ID 的成本差方案，补齐无完成记录的 3 个 Train 可达性证据，再制定与任务价值相符的独立失败代价。若要形式保证，需要推导任务物理/寿命/时域下的上界及 signed SOC 成本下界；不应把本表最大值当作数学上界。分 ID 分数还会改变统一任务失败代价的语义，需单独批准设计。

建议先用本工具离线敏感性表排除不合适尺度；获明确批准后，才对少量固定 Train seed 做预先限定预算的候选对照，观察完成率、覆盖率、Train 经济成本与失败末步 TD 残差/梯度裁剪频度，保持 reward 其余部分、物理系数和 SOC Mask 不变。Validation 只按既定层级选模，不用于算罚分；Test 完全隔离。大的 F 会增大终止 TD 目标幅度，Huber/clipping 不证明其优化稳定，因此也不能静默采用约百万元方案。

## 复现

在项目根目录 PowerShell，均为已验证 CLI：

```powershell
.venv-optuna\Scripts\python.exe scripts\audit_episode_failure_penalty.py --help
.venv-optuna\Scripts\python.exe scripts\audit_episode_failure_penalty.py
.venv-optuna\Scripts\python.exe scripts\benchmark_v6_environment.py --help
.venv-optuna\Scripts\python.exe scripts\benchmark_v6_environment.py --steps 32 64 128 --repeats 3 --measure-adam --adam-repeats 3 --torch-threads 1
```

机器证据：`failure_penalty_audit.json`、`failure_penalty_source_manifest.json`、`train_economic_summary_evidence.csv`、`train_same_id_failure_comparisons.csv`、`train_sparse_failure_evidence.json`。本次保留所有未能从归档证明的缺失值，未修改冻结物理、经济参数或失败分数。
