# V6 1 秒数据处理报告

直接复用冻结数据集已有的 canonical 1 秒轨迹，43 个 gzip 文件原样复制；新增插值点 0。
Train/Validation/Test 保持 35/8/4；只打开并复制 Train 和 Validation，Test 仅保留冻结元数据。
逐点对照全部字段的原始文本，时间戳、负载、source_time_s、source_1s_row 全部完全一致。
所有原始锚点负载最大绝对偏差为 0；无不规则间隔或缺失区间。
已有处理轨迹不属于新增的真实 1 Hz 实测数据。终点节点保留，但不产生额外动作。

源目录：`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v5-economic4-ddqn\data\processed\operating_dataset_independent_v2`
目标目录：`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-dqn-1s\data\processed\operating_dataset_v6_1s`

每条航段的完整来源与目标路径、源行号、起止时间戳见同目录 data_processing_report.csv。

| 航段 | 划分 | 处理前节点 | 处理后节点 | 时长(s) | 原始锚点 | 最大偏差(kW) | 不规则/缺失 |
|---|---|---:|---:|---:|---:|---:|---|
| parent_002_seg01 | validation | 12169 | 12169 | 12168 | 12169 | 0 | 无/无 |
| parent_003_seg01 | train | 21481 | 21481 | 21480 | 21481 | 0 | 无/无 |
| parent_005_seg01 | train | 9259 | 9259 | 9258 | 9259 | 0 | 无/无 |
| parent_006_seg01 | train | 10551 | 10551 | 10550 | 10551 | 0 | 无/无 |
| parent_007_seg01 | train | 8449 | 8449 | 8448 | 8449 | 0 | 无/无 |
| parent_008_seg01 | train | 22857 | 22857 | 22856 | 22857 | 0 | 无/无 |
| parent_009_seg01 | train | 9481 | 9481 | 9480 | 9481 | 0 | 无/无 |
| parent_009_seg02 | train | 7756 | 7756 | 7755 | 7756 | 0 | 无/无 |
| parent_010_seg01 | train | 11191 | 11191 | 11190 | 11191 | 0 | 无/无 |
| parent_010_seg02 | train | 15782 | 15782 | 15781 | 15782 | 0 | 无/无 |
| parent_011_seg01 | validation | 23197 | 23197 | 23196 | 23197 | 0 | 无/无 |
| parent_012_seg01 | train | 17874 | 17874 | 17873 | 17874 | 0 | 无/无 |
| parent_015_seg01 | train | 10116 | 10116 | 10115 | 10116 | 0 | 无/无 |
| parent_016_seg01 | validation | 21071 | 21071 | 21070 | 21071 | 0 | 无/无 |
| parent_017_seg01 | train | 29324 | 29324 | 29323 | 29324 | 0 | 无/无 |
| parent_018_seg01 | train | 27311 | 27311 | 27310 | 27311 | 0 | 无/无 |
| parent_019_seg01 | train | 10471 | 10471 | 10470 | 10471 | 0 | 无/无 |
| parent_019_seg02 | train | 8869 | 8869 | 8868 | 8869 | 0 | 无/无 |
| parent_020_seg01 | validation | 3559 | 3559 | 3558 | 3559 | 0 | 无/无 |
| parent_021_seg01 | train | 6997 | 6997 | 6996 | 6997 | 0 | 无/无 |
| parent_022_seg01 | train | 5491 | 5491 | 5490 | 5491 | 0 | 无/无 |
| parent_029_seg01 | train | 10621 | 10621 | 10620 | 10621 | 0 | 无/无 |
| parent_029_seg02 | train | 5222 | 5222 | 5221 | 5222 | 0 | 无/无 |
| parent_033_seg01 | validation | 11071 | 11071 | 11070 | 11071 | 0 | 无/无 |
| parent_034_seg01 | train | 9721 | 9721 | 9720 | 9721 | 0 | 无/无 |
| parent_035_seg01 | train | 9301 | 9301 | 9300 | 9301 | 0 | 无/无 |
| parent_036_seg01 | train | 6662 | 6662 | 6661 | 6662 | 0 | 无/无 |
| parent_037_seg01 | train | 36661 | 36661 | 36660 | 36661 | 0 | 无/无 |
| parent_038_seg01 | train | 21092 | 21092 | 21091 | 21092 | 0 | 无/无 |
| parent_040_seg01 | validation | 15541 | 15541 | 15540 | 15541 | 0 | 无/无 |
| parent_041_seg01 | train | 19262 | 19262 | 19261 | 19262 | 0 | 无/无 |
| parent_042_seg01 | train | 18422 | 18422 | 18421 | 18422 | 0 | 无/无 |
| parent_043_seg01 | train | 11762 | 11762 | 11761 | 11762 | 0 | 无/无 |
| parent_044_seg01 | validation | 24482 | 24482 | 24481 | 24482 | 0 | 无/无 |
| parent_045_seg01 | validation | 32680 | 32680 | 32679 | 32680 | 0 | 无/无 |
| parent_046_seg02 | train | 3273 | 3273 | 3272 | 3273 | 0 | 无/无 |
| parent_047_seg01 | train | 5881 | 5881 | 5880 | 5881 | 0 | 无/无 |
| parent_048_seg01 | train | 8581 | 8581 | 8580 | 8581 | 0 | 无/无 |
| parent_049_seg01 | train | 8071 | 8071 | 8070 | 8071 | 0 | 无/无 |
| parent_050_seg01 | train | 30692 | 30692 | 30691 | 30692 | 0 | 无/无 |
| parent_051_seg01 | train | 4351 | 4351 | 4350 | 4351 | 0 | 无/无 |
| parent_052_seg01 | train | 7111 | 7111 | 7110 | 7111 | 0 | 无/无 |
| parent_053_seg01 | train | 7651 | 7651 | 7650 | 7651 | 0 | 无/无 |
