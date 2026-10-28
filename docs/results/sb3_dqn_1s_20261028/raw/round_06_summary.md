# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 6 轮报告

Train 完成 32/35；Greedy Validation 完成 7/8。
实际 Adam 更新 425983（累计 2285025）；Target 复制 426（含初始化累计 2286）。
Epsilon 0.208333 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|32/35|0|3|0|0|3|0|2609.546|1224.480|
|Greedy Validation|7/8|0|1|0|1|0|0|3316.401|2344.169|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1689.447837|36998.907638|7057709.393167|4165.202599|7098873.503404|2513.135312|7101386.638716|
|Train/仅失败前缀|3|87.868768|1924.326029|158596.785333|338.671110|160859.782472|861.199855|161720.982327|
|Train/仅完整航段子集|32|1601.579069|35074.581609|6899112.607834|3826.531489|6938013.720932|1651.935456|6939665.656389|
|Validation/全部已执行区间|8|541.855390|11866.633046|2861049.218709|1156.162464|2874072.014219|548.975511|2874620.989730|
|Validation/仅失败前缀|1|34.773771|761.545579|67421.021208|104.257276|68286.824063|286.968550|68573.792613|
|Validation/仅完整航段子集|7|507.081620|11105.087467|2793628.197500|1051.905188|2805785.190156|262.006961|2806047.197117|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|26076353|-53.620057 [-3174.256348, 333.485199]|-53.575664 [-3178.433594, 340.719757]|0.044393 [-202.283203, 216.993103]|1.143638|4.115720|8192 / uniform_reservoir_estimate|
|normal_terminal|—|2245|-14.744367 [-214.436340, 92.072319]|-0.087545 [-0.370775, -0.000707]|14.656822 [-92.162743, 214.163712]|31.041516|86.586708|2245 / exact_all_draws|
|failure_terminal|—|248|-3034.743086 [-3734.143799, -1993.960083]|-10134.865305 [-10135.000977, -10134.686523]|-7100.122226 [-8141.012695, -6400.711914]|7100.122226|8096.074390|248 / exact_all_draws|
|failure_prefix|—|1184066|-814.035845 [-3743.043457, 227.788025]|-814.109827 [-3736.271973, 227.551346]|-0.073982 [-152.455750, 150.352966]|3.832182|14.776406|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：248；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4553；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_06_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|964511|8192|-46.354345 [-2672.065918, 248.269638]|-45.870969 [-2648.460938, 256.997894]|0.483376 [-56.872559, 201.707153]|1.193959|4.317018|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|78|78|0.646521 [-119.641052, 91.235352]|-0.075743 [-0.360890, -0.000707]|-0.722264 [-91.325775, 119.498680]|30.584957|75.669673|78 / exact_on_saved_diagnostic_sample|
|failure_terminal|8|8|-3203.488312 [-3727.109863, -2160.051758]|-10134.874634 [-10135.000977, -10134.686523]|-6931.386353 [-7974.634766, -6407.745605]|6931.386353|7938.043359|8 / exact_on_saved_diagnostic_sample|
|failure_prefix|35403|8192|-850.590012 [-3726.575684, 200.394913]|-850.517616 [-3727.334229, 199.866806]|0.072396 [-149.278687, 149.358826]|3.925685|14.822058|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16470。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_06.json)

```json
{
  "file": "q_probes/round_06.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_06_train_failure_values.csv.gz)

```json
{
  "file": "round_06_train_failure_values.csv.gz",
  "prefix_file": "round_06_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 909,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -2066.848876953125,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -8068.037841796875,
      "terminal_mae": 8068.037841796875,
      "prefix_td_mae": 6.924374561436129,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1420.9940185546875,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8713.563598632812,
      "terminal_mae": 8713.563598632812,
      "prefix_td_mae": 2.2836719032944433,
      "full_q_rows": 303
    }
  ],
  "interpretation": "Only actual Train transitions; read-only checkpoint inference. Observed return is the realized epsilon-greedy trajectory return, not Q* or a counterfactual benefit. TD targets use current target-network masked max; they are moving learning targets. The first upper/lower failures are fixed for checkpoint comparisons; no report rows enter Replay."
}
```

## 解释限制与变化

- TD target 不是 Q*。训练时 Online/Target 参数变化会改变目标；拟合误差仅衡量对对应时刻 bootstrap 目标的拟合。
- Loss、TD 误差或单个状态 Q 值不能单独证明控制改善；同时检查完整航段完成数、边界失败和物理时间尺度的功率波动。
- 固定 Validation 探针仅用于报告，不加入 Replay、不训练；Test 不读取。
- 失败前缀成本只覆盖实际执行区间；完整航段子集成本不是固定全套成本。SOC/ramp/失败惩罚不计入经济账本。
- SOC 上/下边界计数与增量动作不可达计数是交叉分类，不可相加作为失败总数。
- 在线 draw 与 checkpoint 分层诊断样本的分布不同；加权 MAE 仅在同一统计口径内作轮间比较，并同时保留各类别误差。
- 分层诊断样本和 TD 目标均随轮次变化；weighted MAE 的变化仅作描述，不能解释为对 Q* 的拟合改善。
- 实际失败轨迹回报来自当时行为策略，不是最优 Q 的真值；未知终局的历史 draw 分类不会在航段失败后追改。
- 失败终止诊断样本平均 target−Q=-6931.386353；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-8068.037842，终止MAE=8068.037842。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8713.563599，终止MAE=8713.563599。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": -1,
  "validation_boundary_failures_delta": 1,
  "validation_changes_per_hour_delta": 34.076214397184685,
  "validation_reversals_per_hour_delta": 894.1978196744221,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -0.2145258377523378
}
```
