# v4 失败终止经验配额2：40轮正式训练

源码 `bed325354cf0760738b6e722711ecf990f05cca4`，分支 `feat/direct-power-mlp-ddqn`，启动时工作区干净。与[旧40轮基线](v4_scaled_reward_40r_2026-10-08.md)相比，训练配置唯一主动变化为 `failure_terminal_quota=2`；数据manifest哈希相同。网络、奖励、学习率、replay32、target500、seed42及Train/Validation/Test划分均未改。原始本地输出保留于 `outputs/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/`。

## 结果

| 指标 | 旧均匀采样 | 配额2 |
| --- | ---: | ---: |
| 最高greedy Train完成数 | 22/30 | 27/30（第3、25、26、37轮） |
| 第40轮greedy Train完成数 | 12/30 | 25/30 |
| 第31–40轮平均完成数 | 12/30 | 24.9/30 |
| 40轮累计greedy Train SOC受限失败 | 698 | 276 |
| 第40轮失败终止TD采样 | 31/21568（0.144%） | 942/30144（3.125%） |
| 第40轮失败终止TD MAE | 9.561 | 1.382 |
| 第40轮总体TD MAE | 0.0286 | 0.1287 |

TD统计均使用缩放系数0.001后的训练奖励单位。新实验40轮合计抽中失败终止经验32800次，旧实验821次；新实验结束时经验池仍有37条失败终止经验。总体TD MAE的采样分布已改变，不能据此直接判断策略质量。新实验总经济Q更新16990次、旧实验15476次；这是相同replay32调度下执行轨迹和进入replay的经验量不同所致。两组预填充均为571次更新。

第40轮全部执行过的ONBOARD步统计：FC=0占比从49.64%降至17.41%，平均FC功率从122.91升至250.88 kW，SOC<0.4占比从45.35%降至0.63%。同时SOC≥0.79占比从8.29%升至78.01%，FC启动次数从34增至326次，平均绝对FC功率变化从8.64升至46.18 kW。供电完成率改善伴随长期贴近SOC上限和频繁启停，尚不能认为整体控制效果已合格。

第40轮完成子集的实际账本仅供诊断：旧实验完成12个Train航段，本次完成25个。相应氢耗为5595/33751元、FC退化12158/201119元、电池退化1797/2033元、实际岸电55/21元，另有模拟末端补能3053/0元。**由于完成样本集合不同，这些费用不能用于判断经济性改善或恶化。**两组均无30/30的greedy Train轮次，Validation全部按门槛跳过；无合格checkpoint，Test读取数为0。

后续应优先诊断失败风险向早期动作的TD传播及高SOC、频繁启停现象。现有结果不支持单凭TD误差下降继续提高采样配额，也没有证明MLP容量不足。本实验没有修改算法或启动其他训练。

## 可复核文件

完整归档见[目录说明](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/README.md)和[逐文件SHA256清单](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/archive_manifest.json)。逐轮[指标](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/round_metrics.csv)、[报告](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/report.json)、[元数据](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/run_metadata.json)、[控制台日志](results/v4_scale001_beta500_replay32_target500_quota2_seed42_40r_20261009/raw/train.log)和全部80个逐轮轨迹均保留。
