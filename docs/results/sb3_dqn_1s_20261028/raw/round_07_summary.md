# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 7 轮报告

Train 完成 31/35；Greedy Validation 完成 8/8。
实际 Adam 更新 417237（累计 2702262）；Target 复制 417（含初始化累计 2703）。
Epsilon 0.050000 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|31/35|0|4|0|1|3|0|2583.970|1295.764|
|Greedy Validation|8/8|0|0|0|0|0|0|3200.039|2058.227|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1631.675535|35733.694221|8014331.165375|3942.902421|8054007.762017|2472.755743|8056480.517760|
|Train/仅失败前缀|4|125.456489|2747.497118|297616.641208|437.613381|300801.751708|1148.265724|301950.017432|
|Train/仅完整航段子集|31|1506.219046|32986.197103|7716714.524167|3505.289039|7753206.010309|1324.490018|7754530.500328|
|Validation/全部已执行区间|8|618.445484|13543.956109|3025417.416292|1036.380356|3039997.752757|19.012311|3040016.765068|
|Validation/仅失败前缀|0|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|
|Validation/仅完整航段子集|8|618.445484|13543.956109|3025417.416292|1036.380356|3039997.752757|19.012311|3040016.765068|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：3040016.765068 CNY。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|25630143|-42.307719 [-3656.682617, 391.455231]|-42.268555 [-3657.249023, 390.011536]|0.039164 [-209.446350, 246.101807]|1.237071|4.365494|8192 / uniform_reservoir_estimate|
|normal_terminal|—|2073|0.157463 [-143.325836, 107.142380]|-0.072029 [-0.360890, -0.000707]|-0.229492 [-107.232803, 143.183472]|34.820254|87.359004|2073 / exact_all_draws|
|failure_terminal|—|202|-3120.796540 [-4018.031982, -2167.566895]|-10134.862803 [-10135.000977, -10134.686523]|-7014.066271 [-7967.119629, -6116.776367]|7014.066271|7874.772876|202 / exact_all_draws|
|failure_prefix|—|1070750|-862.354171 [-4161.487793, 250.197586]|-862.436656 [-4162.301758, 248.894272]|-0.082485 [-174.728027, 160.746948]|4.748755|16.995642|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：202；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4755；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_07_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|953488|8192|-35.079986 [-3060.662842, 210.411530]|-35.810956 [-3054.433594, 211.415176]|-0.730970 [-79.453491, 133.348694]|1.466047|4.359498|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|75|75|-3.649335 [-172.341568, 109.847572]|-0.067236 [-0.360890, -0.000800]|3.582098 [-109.937996, 172.143112]|42.062306|95.393729|75 / exact_on_saved_diagnostic_sample|
|failure_terminal|10|10|-3420.092383 [-4050.360596, -2392.095703]|-10134.862109 [-10135.000977, -10134.693359]|-6714.769775 [-7742.597656, -6084.448242]|6714.769775|7719.792456|10 / exact_on_saved_diagnostic_sample|
|failure_prefix|46427|8192|-869.516599 [-4073.514404, 171.819855]|-870.614793 [-4091.643799, 172.885513]|-1.098194 [-167.129761, 139.298645]|4.047674|14.603241|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16469。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_07.json)

```json
{
  "file": "q_probes/round_07.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_07_train_failure_values.csv.gz)

```json
{
  "file": "round_07_train_failure_values.csv.gz",
  "prefix_file": "round_07_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1212,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -2326.973388671875,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -7807.913330078125,
      "terminal_mae": 7807.913330078125,
      "prefix_td_mae": 8.408363759122937,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1458.12060546875,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8676.43701171875,
      "terminal_mae": 8676.43701171875,
      "prefix_td_mae": 1.649659213640832,
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
- 失败终止诊断样本平均 target−Q=-6714.769775；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-7807.913330，终止MAE=7807.913330。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8676.437012，终止MAE=8676.437012。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 1,
  "validation_boundary_failures_delta": -1,
  "validation_changes_per_hour_delta": -116.36221487018884,
  "validation_reversals_per_hour_delta": -285.94250881341577,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 0.9533053783800876
}
```
