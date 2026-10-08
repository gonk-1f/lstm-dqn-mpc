# v4 仓库整理与依赖审计（2026-10-08）

基于 `d111c99`。本次审查 docs/outputs/src/tests 与历史入口，不改写 Git 历史，不启动正式训练，不读取正式 Test payload；正式物理模型、经济账本和默认训练算法保持不变。
先阅读并遵循 [上次清理记录](history/v4/v4_cleanup_2026-10-07.md)，没有重复删除已移除的旧支线。

## 文档分类与入口

迁移 67 个文档：v2历史方法/研究8、已执行计划29、旧spec16、共享模型依据4、数据来源2、v3诊断2、v4历史报告6。
根首页和当前方法页更新为 `v4.feedback_study` 的实际行为；更新前的两页另存为历史快照。
当前入口默认仍为 beta500、redistributed、replay32、target500、batch64、γ1、n1、epsilon1→0.05、失败倍率1、未缩放奖励。
旧页中纯经济/outcome-only/episode16/round-sync 等配置只属于其历史阶段，不作为新默认。

新增 [文档总索引](README.md)、[历史索引](history/README.md)、[模型与数据依据](reference/README.md)。
所有 `docs/results/`、`docs/figures/`、`docs/archive_v1/` 及旧清理manifest保留原路径；不修改已有原始日志、结果、checkpoint或证据哈希。
迁移只改必要相对链接，不改历史实验结论；33处相对目的地完成重映射，旧MD字节未变的文件保留原SHA，链接调整文件记录前后SHA。

## 保留与依赖依据

保留原有156个源码文件、91个测试文件及必要基础代码；本次增加独立诊断脚本/测试，不改既有生产实现。
全v4静态依赖闭包包含62个本地Python文件，其中包含v2/v3数据、物理、经济、MPC、MLP/KAN兼容包；静态可达不等于执行MPC或读取Test。
保留17个被旧MPC预检/训练入口读取的校准输入，逐文件Git对象和SHA核对通过，不因位于outputs而删除。
保留MPC分支、main、当前v4分支及两个现有工作树，没有额外旧支线可删。

以下文档仍在旧根路径：

- `v2_preflight_report.md`：必要回归直接读取。
- `v2_dqn_state_audit.md`：旧生成入口输出路径。
- `v2_fc_efficiency_model.md`：生产来源记录引用。
- `v2_cleanup_manifest.md` 与 `v4_cleanup_inventory_2026-10-07.json`：历史恢复与审计依据。

本地40轮基线保留 `report.json` 的防覆盖作用，以及未找到归档匹配的checkpoint、episode_metrics、profiles、train.log。
本地旧Q排序证据与最终归档不相同，保留，不用名字相似推断冗余。

## 限定清理

已删除4组可重建的合成测试basetemp：240文件、25,050,034字节，每个文件都有已有synthetic/tmp_path测试生成器依据。
已删除104个本地结果副本：24,679,979字节，与保留的正式归档逐文件SHA256相同。其中93个对应staged raw、11个对应基线PNG；正式原件仍保留。
以上合计344文件、49,730,013字节。删除使用单一PowerShell `-LiteralPath`，先验证绝对路径在当前工作树明确子目录内，且文件为已忽略/未跟踪项；没有对data使用清理命令。
剩余245个Python缓存、5个pytest缓存及本次66个合成回归临时文件（合计316文件、11,331,493字节）已验证为可再生且已忽略/未跟踪，但自动审批拒绝删除命令，仅返回 `blocked by policy`。这些文件实际仍存在；没有继续尝试删除，列入暂缓清单。该限制不影响提交后的Git status。

## 暂缓，不删除

以下八个入口无静态入站导入，但有CLI、历史来源、sidecar或审计用途。静态无入站不能证明冗余：

- `src/main/audit_fixed_physical_l2_84_train.py`
- `src/main/audit_rms_l2_84_train.py`
- `src/main/audit_rms_marginals_84_train.py`
- `src/main/build_operating_segment_dataset.py`
- `src/v2/main/build_ais_speed_sidecar.py`
- `src/v2/main/build_shore_mode_sidecar.py`
- `src/v2/main/run_failure_penalty_audit.py`
- `src/v3/test_diagnostic.py`

旧v4/v2/v3训练入口仍保留复现，不代替当前单配置入口；不存在额外需要删除的旧分支/工作树。
静态审计不能完全穷尽用户外部命令和反射行为；有争议的源文件与测试保持原状。

## 链接、证据与验证范围

基线85个Markdown文件：39个本地链接/图片，断链0、未定义显式引用0。
历史文字中的输出路径另列：54处由旧清理清单确认删除、6处在其他分支归档、43处在当前工作树缺失/未生成；它们不是Markdown链接，不伪造输出来使审计看起来通过。
23个外部文献URL仅列目录，本次未进行网络可用性或论文内容复核。
最终链接、回归、文件数量和哈希结果见本报告的核验结果，以及机器可读审计。


## 最终核验

- 最终97份README/docs Markdown的193处本地链接/图片均有效，断链0；7份新索引117处目的地核验通过。
- 基线383项保护SHA256全部一致：共享src/tests、历史结果、图片和必要证据没有变化。
- 5个正式数据/划分manifest与清理前完全一致，只读取元数据做哈希；不读取Train/Validation/Test功率轨迹。
- 17个校准输入共5,193,717字节仍保留，逐文件Git对象和SHA匹配。
- 删除的104个本地副本全部有相同SHA的正式归档原件，归档原件字节不变；另240个已删除synthetic临时文件可由原测试重建。
- 67次移动与2个页面快照核验通过：撤销33处链接目的地改写后恢复清理前字节；所有旧证据目录保持不变。
- 主回归182项、330子测试通过（42.16s）；后续最终CLI/非零实际SHORE契约10项通过（4.54s），其中2项已在主回归中覆盖，合计190个不同测试。测试不调用正式轨迹加载器。
- 数值对照六个配置逐批采样索引完全相同，每配置602次经济更新、1次调度Target同步、剩余credit6；只是合成replay诊断，没有正式environment rollout。
- 外部URL只分类，不宣称外网可用；源文件“无入站导入”不作为删除理由。

最终链接检查结果及每个文件的哈希在[完整核验记录](results/v4_repository_td_audit_20261008/final_verification.json)与[文档复核](results/v4_repository_td_audit_20261008/final_docs_review.json)中。

## 文件数与字节变化

以下为工作树文件统计，不代表所有文件都被Git跟踪；`outputs`采用与清理前相同口径，排除本次仓库审计子目录，该子目录另列。
`docs`增长来自本次诊断CSV/JSON、图表、审计证据与索引，历史文档移动本身不删除资料。
目录表是完成提交前的快照；机器清单有逐文件大小和采集时间，文件夹元数据不计入。

<!-- inventory-table-start -->
| 范围 | 清理前文件数 | 完成后文件数 | 清理前字节 | 完成后字节 | 字节变化 |
|---|---:|---:|---:|---:|---:|
| docs | 221 | 261 | 28002144 | 37442492 | +9440348 |
| outputs（排除本次仓库审计） | 367 | 33 | 57064238 | 12381814 | -44682424 |
| src（含未删缓存） | 300 | 300 | 4195814 | 4195814 | +0 |
| tests（含未删缓存） | 190 | 192 | 2847363 | 2929325 | +81962 |
| .pytest_cache | 5 | 5 | 21554 | 23747 | +2193 |
<!-- inventory-table-end -->

本次额外的 `scripts` 诊断脚本及缓存、本地仓库审计目录，分别统计于完整核验记录；不混入上述旧范围的删除收益。
实际删除344文件、49,730,013字节（47.43MiB）；这不包括审批拒绝的316文件。没有任何src/test/data正式文件删除。

## 机器可读证据

[证据总索引](results/v4_repository_td_audit_20261008/README.md)集中列出清理前inventory、迁移目的地与前后SHA、依赖图、执行删除清单、暂缓删除清单、回归日志及最终核验。

- [逐文件依赖/保留/暂缓依据](results/v4_repository_td_audit_20261008/dependency_audit.json)
- [67个文档迁移与链接重映射](results/v4_repository_td_audit_20261008/documentation_moves.json)
- [实际删除344文件清单](results/v4_repository_td_audit_20261008/approved_deletions.json)
- [审批拒绝、仍保留的临时文件](results/v4_repository_td_audit_20261008/final_temporary_cleanup.json)
- [生产源码不变及诊断实现复核](results/v4_repository_td_audit_20261008/final_numerical_review.json)
- [回归日志](results/v4_repository_td_audit_20261008/regression.log)与[最终CLI/SHORE契约日志](results/v4_repository_td_audit_20261008/final_cli_shore_contract.log)

本次不提交新的正式40轮性能结论。未缩放仍是正式默认；是否采用诊断建议的统一0.001由后续对照决定，详见[数值报告](v4_td_numerical_diagnostic_2026-10-08.md)。
