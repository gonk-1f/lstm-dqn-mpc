# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 3 轮报告

Train 完成 30/35；Greedy Validation 完成 7/8。
实际 Adam 更新 396351（累计 1036746）；Target 复制 396（含初始化累计 1037）。
Epsilon 0.683333 → 0.525000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|30/35|0|5|0|2|3|0|3141.788|1440.523|
|Greedy Validation|7/8|0|1|0|0|1|0|2496.003|1006.134|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1504.723171|32953.437437|5181616.145000|3744.213667|5218313.796104|2689.879486|5221003.675590|
|Train/仅失败前缀|5|128.831530|2821.410504|251614.625792|496.243822|254932.280117|1435.187385|256367.467502|
|Train/仅完整航段子集|30|1375.891641|30132.026933|4930001.519209|3247.969846|4963381.515987|1254.692101|4964636.208088|
|Validation/全部已执行区间|8|560.699769|12279.324936|1510161.545167|1010.528278|1523451.398381|577.150670|1524028.549051|
|Validation/仅失败前缀|1|51.931721|1137.304699|39210.353417|96.883225|40444.541341|287.022397|40731.563738|
|Validation/仅完整航段子集|7|508.768047|11142.020237|1470951.191750|913.645053|1483006.857040|290.128273|1483296.985313|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|19493955|-108.187238 [-2003.570679, 103.276611]|-108.154585 [-2002.386841, 103.191238]|0.032653 [-115.815063, 117.991028]|0.754245|2.510086|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1825|-65.057179 [-818.960205, 47.978100]|-0.230466 [-1.310338, -0.012874]|64.826712 [-48.013870, 817.649841]|75.060708|415.796667|1825 / exact_all_draws|
|failure_terminal|—|916|-1334.350512 [-2318.381348, -731.455566]|-10134.737666 [-10135.122070, -10134.480469]|-8800.387163 [-9403.380859, -7816.472168]|8800.387163|9285.199707|916 / exact_all_draws|
|failure_prefix|—|5869768|-314.055319 [-2344.752197, 111.622643]|-314.041873 [-2344.104004, 111.175613]|0.013446 [-237.390564, 123.445862]|1.070970|3.651329|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：916；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：3639；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_03_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|845882|8192|-97.458417 [-1848.469116, 75.373055]|-97.135004 [-1814.282349, 76.107147]|0.323413 [-93.969482, 43.721527]|0.841088|3.091154|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|76|76|-55.849839 [-819.917969, 49.495846]|-0.196755 [-1.246376, -0.012874]|55.653082 [-49.531616, 818.671570]|69.443036|322.889618|76 / exact_on_saved_diagnostic_sample|
|failure_terminal|23|23|-1695.949789 [-2354.646729, -984.742859]|-10134.745117 [-10135.067383, -10134.480469]|-8438.795304 [-9149.755859, -7780.207031]|8438.795304|8957.001660|23 / exact_on_saved_diagnostic_sample|
|failure_prefix|154019|8192|-391.263893 [-2351.047363, 80.827568]|-390.520309 [-2346.855469, 84.024864]|0.743585 [-62.588745, 93.854126]|1.638107|5.830606|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16483。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_03.json)

```json
{
  "file": "q_probes/round_03.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_03_train_failure_values.csv.gz)

```json
{
  "file": "round_03_train_failure_values.csv.gz",
  "prefix_file": "round_03_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1515,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -1588.7193603515625,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -8546.167358398438,
      "terminal_mae": 8546.167358398438,
      "prefix_td_mae": 2.8191643266488384,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1216.7838134765625,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8917.773803710938,
      "terminal_mae": 8917.773803710938,
      "prefix_td_mae": 0.44557551200816176,
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
- 失败终止诊断样本平均 target−Q=-8438.795304；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-8546.167358，终止MAE=8546.167358。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8917.773804，终止MAE=8917.773804。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 0,
  "validation_boundary_failures_delta": 0,
  "validation_changes_per_hour_delta": 2365.982918636938,
  "validation_reversals_per_hour_delta": 976.1693850807101,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -2.976232854852473
}
```
