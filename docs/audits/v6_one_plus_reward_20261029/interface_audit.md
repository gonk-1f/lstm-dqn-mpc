# `inverse_one_plus_economic` 接口预审计

日期：2026-10-29。只读基线：`2c2577ecd25d41b684a2c82cd345b6d3ad843def`。本报告依据源码阅读，不表示新模式已经实现或测试通过；本审计没有运行训练、Validation rollout 或 Test，也没有修改生产源码。

## 结论

新模式应只接入现有单线程 `DirectVesselEnv` / episode runner 路径，保留旧模式默认值、经济账本和训练时钟。新增正常奖励为按 CNY 数值计算的 `1/(1+C_t)`；明确物理失败时，在返回、记录及 Replay 插入前，将最后一个真实 transition 的训练奖励替换为 `-1`。这不修改 `executed_ledger`、SOC、动作或经济评价，不调用旧金额型失败扣分来实现新模式。

主要风险不是增加一个 CLI choice，而是下游仍硬编码旧模式、导入时丢掉奖励身份，以及把“旧模式语义兼容”误称为“历史 checkpoint 可跨源码版本恢复”。下列建议属于待实现修改点。

## 1. 直接环境与失败处理

源码位置：`src/v6_sb3/direct_environment.py:24`（模式集合）、`:141`（实际 reward 计算）、`:175`（失败转换）、`:202`（snapshot）、`:213`（restore）。

- 保留默认 `zhou_economic_only`，在 V6 层新增独立模式和明确的 reward contract。新模式的 `C_t` 使用同一步实际 ledger 的氢气、FC 退化、电池退化，加上既有 signed SOC 等效氢耗成本；不引入 SOC/ramp 辅助项，不读取失败金额作为经济成本。经济模型仍为冻结固定效率口径，不能把此次奖励改造标记成实时效率模型已经落地。
- 当前 `onboard_reward_components` 并不支持新 mode。不要将新名字透传后期待其自动采用完整经济成本；可在 V6 中显式取得既有 `zhou_economic_only` 的经济分项，再按新合同计算奖励。无需为此改变 V4 历史 replay 的模式集合。
- 先取得真实 ledger，校验分项和总成本有限且 `C>-1`，计算有定义的基准奖励，再按真实后继可行集决定是否物理终止。后继物理无动作时，直接构造同一真实 transition 的终止视图：`reward=-1`、`done=True`、空后继 Mask、正确 `failure_*` reason；`failure_penalty_equivalent_cny=0`，不经过 `base_reward-F`，不添加新 transition，不修改原 ledger。
- 用户验收第 5 项确定：**所有 `C<=-1` 均须报错，包括物理失败的最后真实步。** 非有限 ledger 也必须报错，不能被 `-1` 掩盖。`original_reward` 因此始终保存已通过定义域校验的 `1/(1+C)` 基准；先验证定义域，再执行物理失败替换。首决策无动作继续抛 `InfeasibleReset`，仅记录 event，不造 TD 样本。
- 正常完成保留该步的正常奖励；人为暂停保留 live state，不改成 terminal。动作 Mask 和正常物理失败分类沿用现有实现。
- 新公式在 `C=0` 时为 `1`，不得误用此前纯 `1/C` 的 `C>0` 审计门禁。确定的定义域是有限 `C>-1`，还需验证正常奖励有限。不得静默 `abs`、epsilon、clip 或更改合法动作。在 `C>=0` 子域上，正常奖励位于 `(0,1]`；合法的 `-1<C<0` 会产生大于 1 的奖励。

建议新模式 snapshot 记录完整 reward contract（公式版本、CNY 数值单位、成本组成、经济模型身份、末步替换与无 bootstrap 规则），restore 必须逐项一致。保留旧 snapshot 既有字段的含义；不要让新模式的 snapshot 仍把旧 `10132.66` 作为实际生效的失败规则。可以为旧模式继续保留历史字段，为新模式新增明确身份字段。

## 2. Episode runner、Validation 与统计

源码位置：`src/v6_sb3/episode_training.py:39`（config）、`:97`（环境工厂）、`:211`（Validation）、`:258`（入口 gate）、`:289`（恢复）、`:543`（CLI）、`:568`（resume/import）；`src/v6_sb3/episode_metrics.py:17`；`src/v6_sb3/search.py:20`。

1. config 和 CLI 同时允许显式新 mode，默认仍为旧 mode；`_make_env(config)` 继续用于训练和 Greedy Validation，避免两套 reward executor。
2. 旧 `acknowledge_failure_penalty` 仅对旧金额失败模式生效。新 mode 不应要求用户接受一个实际未使用的 `10132.66`，也不应以该 flag 改变新模式 `-1`。run API、fresh CLI、resume CLI、Optuna gate 都要一致处理。
3. `EpisodeMetrics` 已将 `return_sum` 与真实 `economic_cost` 分开，适合复用。确保末步 reward 已在 observe/replay 之前替换，因此 Return 只包含一次 `-1`，经济累计仍包含该真实执行步成本。新 reward 的统计命名不得延用 `_cny` 来暗示其是经济金额；现有兼容字段 `DirectTransition.reward_cny` 可保留，但需注明其在新模式承载训练分数。
4. 增加轻量 reward min/max/mean、正常转移数、物理 terminal 替换数及必要的成本/奖励错误计数；这些聚合状态应随 `EpisodeMetrics.snapshot()` 恢复，避免暂停后统计断裂。无须逐秒保存 13 个 Q 值或 Replay 全量日志。
5. Validation 继续按完成数、覆盖率、固定完整集合真实 CNY 费用选模，不按新 Return 选模。不改 `validation_objective`，不把失败前缀低费用当完整航段成本。保留 RNG/counters/Replay 的现有隔离检查。
6. CLI `--resume` 目前会用磁盘 configuration 整体替换解析后的 options。若用户显式传了冲突 `--reward-mode`，当前结构会静默忽略它；应在覆盖前拒绝显式冲突，而没有显式 mode 时从磁盘恢复。不要把 argparse 的默认旧 mode 当成用户显式冲突。

不改变：500 episodes、有放回 Train 调度、1 秒物理步、`train_freq=32`、`gradient_steps=1`、默认 batch 64、正式环境步 Target 时钟、`gamma=1`、普通单步 MaskedDQN/Huber。

## 3. Optuna 的三处硬编码与独立 Study

源码位置：`src/v6_sb3/episode_tuning.py:76`（`trial_config`）、`:92`（run API）、`:109`（identity）、`:203`（export）、`:218`（CLI）。

- 新增显式 `reward_mode` 参数并保持旧默认；`trial_config` 必须从已验证 Study identity 读取 mode，不能继续写死 `zhou_economic_only`。
- Study identity 记录 mode、reward contract、经济模型身份及失败语义；Trial config、user attrs、selected/export 文件携带相同身份。pruning 继续只报告 Validation score。
- `export_best` 当前把 `reward_mode` 写死为旧模式；必须改为读取 Study identity，不能靠调用 CLI 时的默认值。若需要不同 Study name，为新 mode 分配新名并记录 identity，旧名保持可读。仅目录隔离且每目录有独立 SQLite、完整 mode identity 校验也足以避免共享 Study，无须为此重命名历史 Study。
- fresh 必须新目录；resume 在创建/续写 Trial 或加载 Replay 之前严格验证 Study/Trial/reward identity，跨 mode 必须拒绝，不能以 `--resume` 给已有 Trial 改奖励。
- 保留当前 episode budget / 32:1 路径，不把旧的 transition-budget 搜索机制迁移到新 mode。

## 4. 参数导出/导入不能丢身份

当前 `episode_training.py:575-580` 先提取 `model_parameters` 或 `parameters`，然后只校验超参数名。其结果是 envelope 中的 `reward_mode`、`target_sync_unit` 等元数据被丢掉。新模式导出的参数可能在默认旧奖励下重新训练，且不报错；从旧 Adam-update 时钟搜索导出的 Target 数值也可能被误当作正式环境步。

建议在解包前校验结构化导出的 schema、reward mode/contract、经济模型身份、`target_sync_unit='formal_env_steps'`、选择来源为 Validation。新旧 mode 或 Target 单位冲突时明确拒绝。导出新 schema 时携带这些字段及源 Study/Trial；旧 schema 和旧纯超参数字典保留已支持的用法，不悄悄更改默认奖励。若允许新 mode 使用没有身份的纯超参数字典，必须要求 mode 显式启用并明确该字典仅提供数值超参数；不能把其描述成已按新 mode 选出的最优结果。

源模型/Replay 绝不从 best-parameter export 加载；新训练使用新目录、新网络和新 Replay。预算、训练采样顺序与选参来源不应由超参数文件静默覆盖。

## 5. Checkpoint、源码身份与历史结果恢复

源码位置：`src/v6_sb3/identity.py:30-41`；`episode_training.py:164-188`、`:290-302`；`episode_tuning.py:115-119`；旧 `tuning.py:380-384`。

**已核实的限制：** `execution_identity` 对 `src` 下所有 `.py` 文件取哈希，并记录依赖、经济配置和旧失败参数。添加一个新 reward 模块也会改变 `source_files` 集合。已有 checkpoint 保存完整 identity，resume 按字典相等严格比较；因此在此次新源码下，即使选旧 mode，既有旧 checkpoint/Study 也会因源码身份不一致而拒绝恢复。这是当前安全约束的预期结果，不是旧奖励行为发生变化。

最小安全交付应区分：

- 旧 mode 在新版本可以新建运行并在相同源码/依赖下恢复，历史归档仍可检查与经济比较。
- 已由旧源码创建的运行，继续使用其记录的原提交、相同依赖和原运行目录恢复。不要原地改写历史配置、identity、Replay 或 checkpoint。
- 不添加忽略源码哈希、自动接受所有旧 schema 或仅比较 `reward_mode` 的宽松恢复开关。若用户要求跨源码版本原地精确续训，这是单独的迁移设计与验证范围，不能由“兼容旧模式”自动推导；需明确兼容提交/文件集合与等价证据后再设计迁移。

新 mode 的 identity 应显式记录实际生效的 reward/failure 合同，而不是只保存旧 `FailurePenalty.from_reference()`。保留 source/dependency/data 严格校验。checkpoint 的 `state.pkl`、环境 snapshot、运行 configuration/identity 必须一致；可在 checkpoint summary 或模型元数据中加入 mode，便于单独拿到 `model.zip` 时辨认其训练目标，不能靠文件名猜测。

## 6. 运行比较默认严格校验合同

源码位置：`episode_training.py:550-560`。当前 `--compare-runs` 只检查 Validation 样本内容哈希是否一致，随后直接比较已保存 score。

本次确定的接口策略为：默认严格拒绝 `reward_mode`、`reward_contract` 或 `economics` 不一致的运行，避免旧结果误用；同时继续验证 Validation 样本一致。输出应展示合同身份，不得比较 Return 的绝对数值或将其显示成 CNY。当前任务不新增跨模式比较 flag；跨奖励的经济对照如后续需要，另行设计并明确评价合同。整份训练源码 identity 不必完全相同才能进行同合同的参数对照，但奖励、经济评价及 score 规则必须可验证一致。

## 7. 旧 `tuning.py` 入口的最小处理

源码位置：`src/v6_sb3/tuning.py:49-66`、`:143-176`、`:355-363`、`:464-479`。

旧 `execute_experiment` 对除 `v6_energy_soc_ramp` 外的 mode 全部走硬编码 `zhou_economic_only`。只扩大 CLI choice 会导致新名字实际执行旧 reward。其历史搜索还是 transition budget、`train_freq=1` 与 Queue curriculum，不满足本次 episode runner 机制。

建议旧入口保留两种旧 mode，对未知/新 mode 在开始工作前显式拒绝，并提示使用现有 `--episode-search` 分派到新增支持新 mode 的 `episode_tuning`。该分派会转交参数，因此 episode parser 必须支持 `--reward-mode`。不要为了接入一个奖励重写旧 curriculum、旧 checkpoint 或旧搜索时钟。直接调用旧 `execute_experiment`/`run_study` API 也应拒绝新 mode，防止绕过 CLI silently fallback。

## 8. 必要测试建议与接口 RED 证据

| 范围 | 能证明接口隔离的必要断言 |
|---|---|
| 新正常奖励 | 实际执行正成本与零成本；费用 ledger/SOC/Mask 同旧经济模式；`C=0 -> r=1`；定义域外及非有限值明确报错，不改变可行动作 |
| 真实终止 | successor 无动作且成本已通过有限 `C>-1` 校验时，仅最后真实步 `-1`，original_reward 保留基准、旧失败金额为 0、账本保留、空后继 Mask，masked TD target 恰为 `-1`；末步 `C<=-1` 仍报错；正常完成不获额外 `-1` |
| event / 异常 | 首决策无动作没有 Replay item；非有限 ledger、文件错误和程序异常不转换成普通失败；pause 不增加 terminal |
| 模式身份 | 新旧 mode snapshot 互相拒绝；同 mode/合同恢复后 reward 与成本连续；formula/成本单位/经济参数任一改变均拒绝 |
| runner | 新 mode 训练和 `_validation` 都由同一 config 创建 Direct env；Return 与 EconomicCost 分离；Validation 不插 Replay、不改 RNG/更新计数 |
| Optuna wiring | mock runner 捕获 Trial mode，pause/resume mode 保持一致；跨 mode Study resume 在任何新 Trial/训练之前失败；export 使用真实 Study mode |
| 参数 envelope | 新 export 导入匹配 mode；新到旧、旧到新 mode 冲突、非 formal-env-step Target 单位、未知 schema 明确失败；旧支持格式仍可用 |
| CLI resume | 未显式 mode 从磁盘恢复；显式冲突 mode 不被静默覆盖；新 mode 不要求旧失败罚分 acknowledge，旧 mode 仍要求 |
| 历史源码 | 一次 source hash 改变即拒绝已有 identity；不能因旧模式默认值相同而绕过。保留现有 exact-resume/节拍回归 |
| compare | reward_mode/reward_contract/economics 任一不同均拒绝；Validation 或 score 规则不同也拒绝；同合同对照不读 Return 排名；不提供跨模式覆盖 flag |
| 旧搜索入口 | 旧 CLI/API 对新 mode 明确拒绝；`--episode-search --reward-mode ...` 正确分派；两种旧 mode 行为仍保留 |

上述多数接口测试可以使用 mock runner、临时目录、SQLite 与合成单步环境完成，无须正式训练或读取 Test。短训练/真实 Train smoke 由主任务执行，本只读审计未执行。

已新增 `tests/test_v6_one_plus_optuna.py`，仅使用 mock episode runner 和临时 SQLite，未改生产源码。RED 命令：

```powershell
.venv-optuna/Scripts/python.exe -m pytest tests/test_v6_one_plus_optuna.py -q --tb=line
```

实际结果：`8 failed, 1 passed in 4.91s`，exit 1；完整输出为 `interface_optuna_red.txt`。7 个新 API 检查因 `run_study` 尚不接受 `reward_mode` 而失败，1 个新 CLI 检查因 parser 尚不接受 `--reward-mode` 而失败；旧默认 Study/v1 导出的兼容性检查通过。这是实现前的预期 RED，不是功能交付通过证据。该测试中的 runner 只返回固定暂停/Validation 结果，没有网络、Replay 更新或船舶训练。
