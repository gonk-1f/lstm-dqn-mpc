# 最终交付与验证

2026-10-29，基于 `2c2577e`。先提交独立审计 `c517553`，其时 `src/`、数据及旧结果与基线无差异，再实现生产奖励。审计脚本后补原始文件哈希与显式反事实步数上限，不改变审计数值或物理数据。

## 来源结论与未决事项

[负载审计](load_audit.md) 从当前 Train 父组原始通道重做对齐、内部缺口 natural cubic、零边界裁剪和 1 s PCHIP，10116 节点与冻结文件的最大绝对差为 0。目标第 1 秒来自 PCHIP 首段多项式，不能称为该时刻直接测量，也没有证据将它归因为浮点舍入残差。

完整 35 Train：457562 控制间隔和 35 terminal 元数据节点；`abs(raw load)<=1 kW` 有 5382 控制节点、32 terminal 节点。该 1 kW 是描述性统计上界。5414 行逐样本表保留原值、normalize 值及三类互不混淆的来源：185 节点可匹配 source component，522 节点可匹配既有固定策略诊断，5382 控制节点有独立 fresh SOC=.6/FC0 单秒反事实。缺失诊断不补造，terminal 无反事实动作。

已有对称 ±1 kW 规则仅吸附选中的数据边界；执行期 `normalize_onboard_load_kw` 仍明确保留任何正值、仅将 `[-1,0]` 归零。没有可信依据把内部正负载统一归零，因此**未修改原始数据、规范化规则或掩码**。内部小正值是否应引入死区仍是单独待决的数据契约问题；需要测量精度/独立负载证据或明确项目规则，不能为了奖励数值而自行设置阈值。

## 生产改动

新独立模式为 `inverse_one_plus_economic`；默认仍是旧模式。正常真实转移严格为 `1.0/(1.0+total_cost_cny)`，输入是 CNY 数值；无分转换、C0、tanh、epsilon、clip、附加偏移或奖励缩放。

| 成本 / CNY | 正常奖励 |
|---:|---:|
| 0 | 1 |
| 6.836828188453796e-7 | 0.9999993163176486 |
| 1 | 0.5 |
| -0.0012402552831242142 | 1.0012417954264625 |

成本和分母必须有限，分母严格大于 0，奖励必须有限。`C=-1`、更小值、NaN、±Inf 明确失败；异常是程序/经济输入错误，不能转为物理失败或插入 Replay。失败末步也先校验该域，再将最后一个真实执行 transition 的训练奖励替换为 `-1`；`done=True`、空后继 mask、无额外动作、无终止附加奖励。正常完成只保留正常公式。失败 −1 不进入 executed ledger 或 CNY 汇总，`failure_penalty_equivalent_cny=0`。新路径不读取旧 F 引用、不调用旧扣罚重放；旧模式保留原公式与旧失败处理。

| 代码位置 | 作用 |
|---|---|
| `src/v6_sb3/rewards.py:25`、`:37`、`:56` | 原账本+signed SOC 的 fsum、精确映射和版本合同 |
| `src/v6_sb3/direct_environment.py:130`、`:206`、`:221` | 一次真实执行、末步替换、成本字段、快照身份与恢复 |
| `src/v6_sb3/episode_metrics.py:17` | Return 与物理经济账本分列；新模式使用同一 fsum |
| `src/v6_sb3/episode_training.py:540`、`:569` | 参数导入身份和结果比较门禁 |
| `src/v6_sb3/episode_tuning.py:104`、`:225` | 新模式 Study `v6_one_plus_episode_dqn`、Trial 接线、v2 导出 |
| `src/v6_sb3/identity.py:26` | 新模式成本单位/公式/失败规则、源码/依赖/数据/经济身份；不读取旧 F |
| `src/v6_sb3/tuning.py` | 旧 step-budget 入口显式拒绝新模式，要求走 episode-search |

恢复核对完整配置、源码/依赖/数据身份、环境快照和已保存模型的模式合同。Validation 使用同一模式环境，独立保存合同，仍按真实 CNY 和完成优先规则选模型，不把 Return 当经济成本。参数导入要求匹配的新 v2 模式、经济口径、Validation 来源、Target 时钟和当前源码哈希；旧或无标签参数不能用于新模式。导入记录保存启动时已校验的 JSON 快照，源文件随后被覆盖/删除也不会记错来源。新模式 Smoke 重训保留导出的模型参数，仅缩小 episode/warmup/Validation 预算，超出 Smoke 小模型上限明确拒绝。比较默认拒绝不同模式、奖励合同、经济合同或 Validation 数据。

旧源码版本产物不能跨源码哈希无条件恢复；本次没有放宽校验、迁移或改写旧 checkpoint/study。旧模式公式兼容与历史二进制跨版本续训不是同一承诺。历史续训应使用原提交及依赖。

## 实际测试

最终命令见 [commands.md](commands.md)。最终源码选定回归集：**283 passed, 3 subtests passed in 157.01s，exit 0**，完整输出 [final_verification.txt](final_verification.txt)。范围包含全部 `test_v6*.py`、新奖励测试、此前三组审计测试、supervisory 和独立经济测试；不是声称运行了整个仓库所有历史测试。

验收点由 `test_one_plus_reward.py`、`test_v6_one_plus_runner.py`、`test_v6_one_plus_optuna.py` 覆盖：四个指定数值、异常域、真实 SOC/结构/增量失败、正常完成、无伪转移、账本不变、旧 F 读取禁止、快照、精确 live resume、模型/参数/Study 隔离、Validation 无训练写入、CLI 模式冲突、导出重训及比较。真实 SB3 Replay 测试证明失败只插入一条实际末步、reward=-1、done/空 mask，并在巨大后继 Q 下 TD target 仍为 -1；异常成本注入测试证明 Replay 插入数为 0。

保留了开发中的失败证据：奖励/runner/Optuna 首次 RED；首次宽回归的两个失败来自测试未固定 FC-off，启动费抵消了注入的负 SOC 项，已固定测试动作而未放宽生产域检查；Smoke 导入参数覆盖和二次读取来源的问题都有独立 RED 复现并修复。对应 `*_red.txt`、`regression.txt`、`regression_final.txt` 为中间记录，最终结果以 `final_verification.txt` 为准。文本日志仅统一行尾和尾随空白，没有改变结果。

## 有界 CLI Smoke

最后一次在最终源码下执行以下完整周期，8 条子命令均 exit 0：基线暂停/恢复、单 Trial 暂停/恢复、导出、全新重训、比较和 inspect。每个命令的实际参数、stdout/stderr、耗时和退出码在 [cli_final.json](cli_final.json)，计数/身份断言见 [cli_final_summary.json](cli_final_summary.json)。此前也保留一轮收尾前 Smoke 记录，不能与最终周期混算。

| 最终周期 | 完整合成 episodes | warmup steps | formal steps | Adam | Replay 插入 |
|---|---:|---:|---:|---:|---:|
| 基线 | 3 | 64 | 192 | 6 | 256 |
| 单 Trial | 2 | 4 | 128 | 4 | 132 |
| 全新重训 | 3 | 64 | 192 | 6 | 256 |

最终重训实参保留导出的 Target=32 formal steps、buffer=256；新模型与 Replay 从零初始化。Validation 也只用合成 8 秒航段，真实 Validation/Test payload 读取均为 0。这验证执行链与恢复合同，不代表正式训练性能或策略经济性。

[保留核对](preservation.json)：跟踪的数据、底层 v2/v3/v4 物理经济代码、历史 outputs/docs/results 均无基线差异；1669 个既有产物的大小和修改时间均未变。此清单是大小/mtime 核对，不冒称全部旧二进制已逐字节重哈希。新运行二进制仅在独立 `.tmp`，不提交模型、Replay 或 SQLite。

## 口径与后续边界

本次只改变奖励映射，仍使用原固定参考效率 SOC 经济模型。此前实时库存初始价值、状态表示等设计未实施。负成本的已知 EOL 边界是模型边界测试，不声称其在全新当前 Train 轨迹中已发生。允许 `-1<C<0` 时奖励大于 1；新映射仍不等价于最小化累计 CNY，也不构成长航段策略质量证明。

未执行正式 500 episode、长 Optuna 或 Test。内部正负载死区保持待决。后续终端命令已经提供，但正式执行仍需另行明确授权。
