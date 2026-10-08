# V4 奖励缩放 40 轮运行归档（2026-10-08）

运行：`v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104`。
来源提交：`c382943f8ddea75624dc08d75445ca4a304b95ba`；运行元数据记录启动时工作树干净。
`report.json` 记录 `run_status=completed`、`completed_training=true`，共完成 40 轮，耗时 2338.047 秒。

## 配置

| 项目 | 实际值 |
| --- | --- |
| 奖励缩放 `reward_scale` | 0.001 |
| SOC 软惩罚系数 `beta_soc` | 500 |
| 学习率 / 折扣 `gamma` | 0.0001 / 1.0 |
| 经济 Q 更新节奏 | `replay32` |
| 目标网络同步 | 每 500 次经济 Q 优化器更新 |
| `n_step` / 批大小 / 种子 | 1 / 64 / 42 |
| 探索率 | 第 1 轮 1.0，降至第 40 轮 0.05 |
| Replay 容量 | 100000 |
| 初始化 | 新种子权重、空 replay、新优化器；未加载 checkpoint |

完整配置见 [run_metadata.json](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/run_metadata.json)。

## 完成率与评估状态

| 项目 | 结果 |
| --- | --- |
| greedy Train 最高完成率 | 第 9、10 轮：22/30（73.33%） |
| 第 40 轮 exploratory Train | 15/30（50.00%） |
| 第 40 轮 greedy Train | 12/30（40.00%） |
| Validation | 40 轮全部跳过，原因均为 `greedy_train_incomplete` |
| 合格轮次 / best checkpoint | 无 / `null` |
| Test | `test_payloads_opened=0`，未评估，无 Test 完成率 |

没有任何一轮 greedy Train 达到 30/30。完整拆分成本字段为 `null`，本报告不据此比较经济性。

日志末尾记录：

```text
MONITOR round=40 exploratory=15/30 greedy_train=12/30 validation=SKIPPED cost=None updates=15476 best_round=None
```

## 实际执行计数

| 项目 | 计数 |
| --- | ---: |
| 正式训练环境步 / 策略调用 | 476973 / 476973 |
| Bootstrap 环境步 | 18273 |
| greedy 评估环境步 | 358165 |
| 经济 replay 累计插入 | 495246 |
| 成功 / 失败来源的经济 replay 插入 | 431434 / 63812 |
| 失败终止插入 | 468 |
| 经济 Q 优化器更新 | 15476（Bootstrap 571 + 正式训练 14905） |
| Outcome 优化器更新 | 19680 |
| 目标网络同步调用 | 31（初始化复制 1 + 定期同步 30） |
| 剩余 transition credit | 14 |

经济 replay 插入数等于 Bootstrap 与正式训练环境步之和。计数为累计执行次数，不能解读为唯一样本数。

失败分类按运行记录累计如下：

| 阶段 | `soc_limited` | `structural_power` | `execution_error` |
| --- | ---: | ---: | ---: |
| Bootstrap | 1 | 0 | 0 |
| 40 轮 exploratory Train | 467 | 0 | 0 |
| 40 轮 greedy Train 评估 | 698 | 0 | 0 |

## 第 40 轮监控记录

| 指标 | exploratory Train | greedy Train |
| --- | ---: | ---: |
| 已执行 ONBOARD 步数 | 10788 | 8465 |
| 平均 SOC | 0.547575 | 0.443466 |
| SOC < 0.4 占比 | 32.86% | 45.35% |
| 0.4 ≤ SOC ≤ 0.6 占比 | 21.81% | 41.25% |
| FC 为 0 的步数占比 | 35.34% | 49.64% |
| FC 平均功率（kW） | 138.579 | 122.907 |
| FC 启动次数 | 112 | 34 |

以上 SOC 与 FC 指标覆盖已执行 ONBOARD 步，包括失败前缀；不含 SHORE。
第 40 轮共进行 337 次经济 Q 更新、采样 21568 个 transition：TD MAE 为 0.0286417（缩放奖励单位），对应原奖励等价单位 28.6417；失败终止采样 31 次，其 TD MAE 为 9.56071（缩放奖励单位）。平均 Smooth L1 loss 为 0.0135532，梯度裁剪次数为 0。奖励与 TD 的等价单位不代表实际支出。

## 归档文件

源目录共有 93 个产物：80 个逐轮轨迹 JSON 和 13 个报告、日志及图表文件。
80 个轨迹 JSON 使用 gzip 无损压缩，归档文件名为原名追加 `.gz`；其余文件保留原始字节。
解压后的轨迹内容及 SHA-256 校验信息见 [archive_manifest.json](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/archive_manifest.json)。

- [归档说明](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/README.md)
- [完整报告](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/report.json)、[逐轮 CSV](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/round_metrics.csv)、[训练日志](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/train.log)
- [完成率曲线](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/completion_rates.png)、[TD 与梯度](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/td_and_gradients.png)、[SOC 与探索率](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/soc_and_epsilon.png)、[FC 与更新计数](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/fc_and_update_counts.png)
- [末轮完成轨迹示例](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/final_round_040_greedy_train_completed_trajectory.png)、[末轮失败轨迹示例](results/v4_scale001_beta500_replay32_target500_seed42_40r_20261008_220104/raw/final_round_040_greedy_train_failed_trajectory.png)

本报告依据该次运行的元数据、完整报告、逐轮指标与日志，记录已产生的结果。
