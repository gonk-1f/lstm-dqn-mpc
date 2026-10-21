# V5暂停时的已提交轮次历史分析

- 实验ID：`V5-ECON4-DELTA7-TD1-PER-INTERLEAVED-S42`。
- 训练源码提交：`34712730a4423baad09549d96e261c2f2a8de028`；源码树SHA：`06a3700e1e0c7c06427ea0cbd743f33f381df6d8bda319bc4b68d59e8142f184`。
- 依据暂停后完整checkpoint内的 `committed_checkpoint_report.json`；完整轮次1—32，计划100轮。第33轮未闭合，不混入正式统计。源报告 `run_status=running` 为被手动停止前保存的旧值，不能据此判断进程仍运行。
- 本分析仅读取小型报告及元数据，未加载轨迹、模型权重、Replay，未训练、未访问Test。

## 1. 完成率与稳定性

| 完整轮次 | 探索Train | Greedy Train | Greedy Validation |
|---:|---:|---:|---:|
| 1 | 5/35 | 1/35 | 1/8 |
| 18 | 10/35 | 20/35 | 4/8 |
| 22 | 9/35 | 26/35 | 4/8 |
| 25 | 15/35 | 27/35 | 7/8 |
| 26 | 16/35 | 10/35 | 1/8 |
| 29 | 19/35 | 7/35 | 1/8 |
| 30 | 14/35 | 17/35 | 1/8 |
| 31 | 13/35 | 24/35 | 5/8 |
| 32 | 22/35 | 29/35 | 5/8 |

- Greedy Train最高为第32轮29/35（82.86%）；Validation最高为第25轮7/8（87.50%），当轮Train为27/35。
- 32轮中没有35/35 Train、没有8/8 Validation，没有合格checkpoint；`best_checkpoint=null`。第25轮不是合格最佳模型。
- 第25轮后Validation曾降至连续多轮1/8，末轮为5/8。不能据单轮回升宣称收敛。

## 2. 第32轮FC变载和失败

| 指标（均为Greedy） | Train | Validation |
|---|---:|---:|
| 完成数 | 29/35 | 5/8 |
| 已执行步数（含失败前缀） | 12333 | 3423 |
| 平均绝对FC变载 kW/步 | 29.403984 | 27.665371 |
| 非零变载比例 | 81.442% | 85.326% |
| 连续反向次数/相邻非零动作组 | 5132/9988 | 1305/2900 |
| 上述反向比例 | 51.382% | 45.000% |
| 启动次数（含失败前缀） | 94 | 34 |
| SOC受限失败 | 6 | 3 |
| 增量不可达失败 | 0 | 0 |
| 已执行SOC最小值 | 0.200465420 | 0.200830461 |

第32轮既有大量非零变载/反向动作，也有SOC受限失败；符合用户要求的暂停条件。反向定义为同一航段内连续两个非零实际FC增量异号，不跨保持动作、样本边界或终止。

末轮失败样本（位置沿用正式程序0基row编号）：

- Train `parent_008_seg01: row 74 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Train `parent_010_seg02: row 228 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Train `parent_015_seg01: row 136 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Train `parent_017_seg01: row 125 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Train `parent_036_seg01: row 85 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Train `parent_050_seg01: row 271 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Validation `parent_002_seg01: row 111 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Validation `parent_016_seg01: row 261 (onboard): no feasible FC action for actual battery power and SOC bounds`。
- Validation `parent_044_seg01: row 181 (onboard): no feasible FC action for actual battery power and SOC bounds`。

## 3. 经济与训练奖励的口径

- 没有任何轮次完成全部固定Train/Validation集合，`comparable_evaluable_cost_cny` 全部为null；因此不比较不同完成集合的成本总数。
- `round_summary.csv` 分开保留完整成功子集成本、实际执行前缀成本和独立失败训练信号；低累计成本不代表供电或经济更优。
- 以下只分解末轮实际执行的经济账本（含失败前缀），用于识别已执行成本来源，不是完整航段经济性能排名。

| 第32轮实际执行账本，元 | Train | Validation |
|---|---:|---:|
| 实际氢耗 | 30477.727827 | 9171.552609 |
| 有符号SOC等效氢耗 | 2191.250449 | 1090.643153 |
| FC退化 | 86881.915750 | 26529.108375 |
| 电池退化 | 4025.289015 | 1238.566887 |
| 上述四分项合计 | 123576.183041 | 38029.871023 |
| FC退化在上述合计中占比 | 70.306% | 69.759% |
| 独立失败训练惩罚（原始奖励单位；不是真实费用） | 60795.960527 | 30397.980264 |

- 正常经济奖励与四分项负和的历史最大绝对对账误差为 2.91e-11 原始单位；SOC辅助项和额外变载项均未加入。

## 4. 同一运行内计数对账

| 范围 | 实际ONBOARD步数 | Replay插入 | Adam更新 | Target软更新 | UTD |
|---|---:|---:|---:|---:|---:|
| Bootstrap | 14995 | 14995 | 468 | 468 | 0.031210403 |
| 正式已闭合32轮 | 232838 | 232838 | 7276 | 7276 | 0.031249195 |
| 含Bootstrap总计 | 247833 | 247833 | 7744 | 7744 | 0.031246848 |

- 初始Target完整复制1次；总软更新7744次，按Adam更新而非环境步更新。Greedy评价的277009步不进入训练UTD分母。
- 正式更新调用路径：航段内部回调7246次、真实失败终止收尾30次、航段结束集中更新0次。
- 正式每条探索训练样本平均更新6.496429次；剩余transition credit=25。

## 5. 能确认与不能确认的结论

- 已确认：暂停时供电资格不达标；FC频繁变载/反向仍明显；完整轮次计数与约1/32 UTD对账。
- 已确认：第32轮Greedy失败均为SOC受限，不应混称增量动作缺口；历史其他轮次仍存在增量不可达，CSV按原因分列。
- 尚未证明：纯经济奖励或新增负载差分已改善最终策略；本次只运行至32/100轮，不能表述为100轮训练失败。
- 仅靠这些历史汇总无法确定Q排序偏差的训练根因，也不能由高FC退化费用直接推出应加惩罚或更改超参数。
- Test继续封存，未计算Test指标。

## 产物

- `history_summary.json`：身份、计数、末轮与最佳完成轮次结构化摘要。
- `round_summary.csv`：32轮×探索/Greedy Train/Greedy Validation，共96行，成本和失败口径明确分列。
- `training_pause_curves.png`：完成率、FC变载及反向比例、失败原因和分类TD MAE。
