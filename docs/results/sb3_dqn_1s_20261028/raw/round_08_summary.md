# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 8 轮报告

Train 完成 32/35；Greedy Validation 完成 5/8。
实际 Adam 更新 425995（累计 3128257）；Target 复制 426（含初始化累计 3129）。
Epsilon 0.050000 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|32/35|0|3|0|0|3|0|2523.756|1371.920|
|Greedy Validation|5/8|0|3|0|0|3|0|3066.590|2360.878|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1704.805081|37335.231265|9430628.776584|3820.223806|9471784.231655|2094.351474|9473878.583129|
|Train/仅失败前缀|3|89.744298|1965.400122|194351.994000|339.853233|196657.247354|861.064009|197518.311363|
|Train/仅完整航段子集|32|1615.060783|35369.831144|9236276.782584|3480.370574|9275126.984301|1233.287465|9276360.271766|
|Validation/全部已执行区间|8|422.495590|9252.653419|2745610.836333|937.311128|2755800.800881|1076.825867|2756877.626748|
|Validation/仅失败前缀|3|81.276216|1779.949141|526079.678750|279.469370|528139.097261|861.195217|529000.292478|
|Validation/仅完整航段子集|5|341.219373|7472.704279|2219531.157583|657.841758|2227661.703620|215.630649|2227877.334270|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|26196229|-32.439673 [-3643.544434, 293.151733]|-32.410418 [-3637.946045, 291.633301]|0.029255 [-198.077362, 254.876434]|1.214968|4.327081|8192 / uniform_reservoir_estimate|
|normal_terminal|—|2049|-1.923367 [-174.648941, 149.319046]|-0.070515 [-0.381195, -0.000800]|1.852852 [-149.409470, 174.450485]|48.208384|116.050520|2049 / exact_all_draws|
|failure_terminal|—|213|-3333.971970 [-4107.805176, -2088.824951]|-10134.861301 [-10135.000977, -10134.678711]|-6800.889314 [-8045.853516, -6027.003418]|6800.889314|7838.685645|213 / exact_all_draws|
|failure_prefix|—|1065189|-858.971346 [-4251.606445, 222.598129]|-859.096939 [-4250.923340, 222.492752]|-0.125593 [-177.445435, 154.245911]|4.261190|15.579846|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：213；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4968；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_08_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|963841|8192|-23.182932 [-2716.634521, 229.610031]|-22.779422 [-2733.405518, 229.205078]|0.403510 [-66.124268, 225.634521]|1.259564|4.367888|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|72|72|3.969899 [-161.953201, 150.386902]|-0.072914 [-0.381195, -0.000800]|-4.042814 [-150.477325, 161.754745]|56.880884|124.575785|72 / exact_on_saved_diagnostic_sample|
|failure_terminal|7|7|-3240.624128 [-4070.397705, -2131.591553]|-10134.817801 [-10134.958008, -10134.678711]|-6894.193708 [-8003.086914, -6064.560547]|6894.193708|7945.797461|7 / exact_on_saved_diagnostic_sample|
|failure_prefix|36080|8192|-871.908080 [-4189.451660, 227.591949]|-873.004808 [-4197.431641, 226.232788]|-1.096728 [-113.584900, 135.835297]|4.265884|15.084962|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16463。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_08.json)

```json
{
  "file": "q_probes/round_08.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_08_train_failure_values.csv.gz)

```json
{
  "file": "round_08_train_failure_values.csv.gz",
  "prefix_file": "round_08_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 909,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -2256.47314453125,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -7878.41357421875,
      "terminal_mae": 7878.41357421875,
      "prefix_td_mae": 6.784194365242459,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1440.229736328125,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8694.327880859375,
      "terminal_mae": 8694.327880859375,
      "prefix_td_mae": 10.99615304201644,
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
- 失败终止诊断样本平均 target−Q=-6894.193708；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-7878.413574，终止MAE=7878.413574。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8694.327881，终止MAE=8694.327881。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": -3,
  "validation_boundary_failures_delta": 3,
  "validation_changes_per_hour_delta": -133.44918837434307,
  "validation_reversals_per_hour_delta": 302.65106176488825,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -1.0817871540013355
}
```
