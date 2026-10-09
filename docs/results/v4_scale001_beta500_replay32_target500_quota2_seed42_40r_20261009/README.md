# 失败终止配额2：40轮结果归档

[结果分析](../../v4_failure_terminal_quota2_40r_2026-10-09.md)；[逐文件归档清单](archive_manifest.json)。源码提交 `bed325354cf0760738b6e722711ecf990f05cca4`，训练完成40轮，Test读取数为0。

`raw/` 保存原报告、逐轮指标、元数据、日志和图片。80个完整探索/greedy Train逐轮轨迹以独立 `.json.gz` 无损压缩；其余13个文件逐字节复制。原始93个文件总计2278753606字节，归档载荷131498770字节。清单记录每个原文件、归档文件及解压结果的SHA-256，全部与原件一致；原始本地输出未删除或覆盖。

例如[第40轮greedy Train轨迹](raw/round_040_train_trajectories.json.gz)可用 Python `gzip.open(path, 'rt', encoding='utf-8')`读取。[完成率曲线](raw/completion_rates.png)、[SOC与epsilon曲线](raw/soc_and_epsilon.png)、[TD曲线](raw/td_and_gradients.png)、[FC与更新次数曲线](raw/fc_and_update_counts.png)及第40轮[完成](raw/final_round_040_greedy_train_completed_trajectory.png)/[失败](raw/final_round_040_greedy_train_failed_trajectory.png)代表轨迹均在归档中。

最高greedy Train为27/30，末轮25/30；无轮次达到30/30，Validation未执行，没有合格best checkpoint。训练结果不可作为完整Train经济成本比较。
