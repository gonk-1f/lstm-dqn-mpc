# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 4 轮报告

Train 完成 31/35；Greedy Validation 完成 7/8。
实际 Adam 更新 416987（累计 1453733）；Target 复制 417（含初始化累计 1454）。
Epsilon 0.525000 → 0.366667。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|31/35|0|4|0|2|2|0|3035.698|1377.418|
|Greedy Validation|7/8|0|1|0|0|1|0|2695.032|2063.037|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1610.739916|35275.204162|6313185.702459|3729.883602|6352190.790223|2476.751728|6354667.541951|
|Train/仅失败前缀|4|119.369910|2614.201026|237628.823667|424.836188|240667.860881|1148.048553|241815.909434|
|Train/仅完整航段子集|31|1491.370006|32661.003137|6075556.878792|3305.047414|6111522.929343|1328.703175|6112851.632518|
|Validation/全部已执行区间|8|546.848895|11975.990803|598887.862917|1030.284986|611894.138706|609.899594|612504.038301|
|Validation/仅失败前缀|1|49.900663|1092.824530|10998.705917|102.387435|12193.917882|287.034207|12480.952088|
|Validation/仅完整航段子集|7|496.948232|10883.166274|587889.157000|927.897551|599700.220825|322.865388|600023.086213|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|24179041|-87.870266 [-2614.895508, 120.716576]|-87.831021 [-2615.699707, 120.264038]|0.039245 [-165.683105, 158.543396]|0.948655|3.085065|8192 / uniform_reservoir_estimate|
|normal_terminal|—|2017|-27.840730 [-873.937378, 68.773026]|-0.152981 [-1.246376, -0.000707]|27.687748 [-68.792168, 872.690979]|45.359061|133.230438|2017 / exact_all_draws|
|failure_terminal|—|402|-2122.611766 [-3093.736328, -988.583618]|-10134.788156 [-10134.972656, -10134.482422]|-8012.176390 [-9145.915039, -7041.117188]|8012.176390|8859.832373|402 / exact_all_draws|
|failure_prefix|—|2505708|-508.143087 [-3105.032715, 153.604675]|-508.139499 [-3098.721191, 151.754929]|0.003588 [-231.535767, 152.586670]|1.809451|6.845776|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：402；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4041；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_04_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|942367|8192|-77.227340 [-2101.541016, 105.482864]|-77.437429 [-2104.471191, 103.749687]|-0.210089 [-85.474304, 126.785034]|1.041094|3.491268|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|77|77|-9.555384 [-807.933167, 68.651917]|-0.118038 [-1.076227, -0.000707]|9.437347 [-68.671059, 806.856934]|35.865298|67.423756|77 / exact_on_saved_diagnostic_sample|
|failure_terminal|12|12|-2391.532583 [-3111.760010, -1874.088623]|-10134.812256 [-10134.972656, -10134.664062]|-7743.279582 [-8260.875000, -7023.148438]|7743.279582|8238.810547|12 / exact_on_saved_diagnostic_sample|
|failure_prefix|57544|8192|-672.623073 [-3106.169189, 154.514359]|-671.964748 [-3104.362305, 149.428589]|0.658325 [-101.854614, 112.361084]|2.816018|9.893622|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16473。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_04.json)

```json
{
  "file": "q_probes/round_04.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_04_train_failure_values.csv.gz)

```json
{
  "file": "round_04_train_failure_values.csv.gz",
  "prefix_file": "round_04_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1212,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -1953.92041015625,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -8180.96630859375,
      "terminal_mae": 8180.96630859375,
      "prefix_td_mae": 3.912641095799326,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1379.8822021484375,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8754.675415039062,
      "terminal_mae": 8754.675415039062,
      "prefix_td_mae": 1.681514411572589,
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
- 失败终止诊断样本平均 target−Q=-7743.279582；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-8180.966309，终止MAE=8180.966309。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8754.675415，终止MAE=8754.675415。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 0,
  "validation_boundary_failures_delta": 0,
  "validation_changes_per_hour_delta": 199.0291411928647,
  "validation_reversals_per_hour_delta": 1056.9022515449751,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -5.601154317532287
}
```
