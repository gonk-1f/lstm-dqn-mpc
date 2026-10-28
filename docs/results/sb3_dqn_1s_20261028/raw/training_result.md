# SB3 DQN 对照结果

状态：completed；已完成 8/8 遍。
固定每遍35条Train；实际总步数随策略改变，不宣称与旧V6更新总量相同。
Test payload读取0。TD目标不是Q*；Train包含随机探索，Greedy Validation不进入Replay。

|遍数|Train完成|Greedy完成|FC变载/h|相邻反向/h|详情|
|---|---:|---:|---:|---:|---|
|1|18/35|7/8|3444.113|1901.231|[报告](round_01_summary.md)|
|2|28/35|7/8|130.020|29.965|[报告](round_02_summary.md)|
|3|30/35|7/8|2496.003|1006.134|[报告](round_03_summary.md)|
|4|31/35|7/8|2695.032|2063.037|[报告](round_04_summary.md)|
|5|31/35|8/8|3282.325|1449.971|[报告](round_05_summary.md)|
|6|32/35|7/8|3316.401|2344.169|[报告](round_06_summary.md)|
|7|31/35|8/8|3200.039|2058.227|[报告](round_07_summary.md)|
|8|32/35|5/8|3066.590|2360.878|[报告](round_08_summary.md)|

每轮reference_comparison记录V6第2/8轮与V5第32轮，只在共同完整航段上比较经济成本。
完整逐秒Validation CSV.gz含全部13Q与合法排序；模型、Replay和随机状态在models/round_NN。
若TD误差下降而失败或变载没有改善，不能把Loss变化解释成控制改善。
