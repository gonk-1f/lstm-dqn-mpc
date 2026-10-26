# V6 第 6 轮报告

Train 完成 28/35；Greedy Validation 完成 3/8。
实际 Adam 更新 367207（累计 2146754）；Target 复制 367（含初始化累计 2147）。
Epsilon 0.208333 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|28/35|0|7|0|4|3|0|2753.840|1336.122|
|Greedy Validation|3/8|0|5|0|4|1|0|3050.295|2010.376|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1329.097992|29107.246032|6441154.437250|3807.717119|6474069.400401|3798.318964|6477867.719365|
|Train/仅失败前缀|7|156.278144|3422.491348|568875.237000|788.377320|573086.105667|2008.962511|575095.068178|
|Train/仅完整航段子集|28|1172.819849|25684.754685|5872279.200250|3019.339799|5900983.294734|1789.356453|5902772.651187|
|Validation/全部已执行区间|8|187.688332|4110.374476|1258036.578417|1094.203164|1263241.156056|1672.663820|1264913.819876|
|Validation/仅失败前缀|5|62.595356|1370.838296|696201.538375|513.086223|698085.462894|1434.928327|699520.391221|
|Validation/仅完整航段子集|3|125.092976|2739.536179|561835.040042|581.116941|565155.693162|237.735493|565393.428656|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|21846747|-32.405743 [-3329.055664, 331.424713]|-32.332545 [-3326.468994, 330.361145]|0.073198 [-287.986664, 246.277649]|1.423627|5.423149|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1789|59.241242 [-550.041016, 154.018143]|-0.081048 [-0.709794, 0.000947]|-59.322290 [-154.115738, 549.331238]|69.336702|114.769846|1789 / exact_all_draws|
|failure_terminal|—|334|-2650.465859 [-3664.264893, -1561.145264]|-10134.840005 [-10135.117188, -10134.605469]|-7484.374170 [-8573.459961, -6470.852539]|7484.374170|8558.944873|334 / exact_all_draws|
|failure_prefix|—|1652378|-752.117827 [-3727.389648, 207.481964]|-752.186486 [-3721.655762, 206.109085]|-0.068659 [-325.900757, 197.757996]|4.706516|17.405472|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：334；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4523；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_06_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|913637|8192|-19.046300 [-2232.360596, 162.241394]|-18.879467 [-2251.225586, 162.636673]|0.166833 [-62.975754, 124.352417]|1.323450|4.987835|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|79|79|65.993411 [-144.695389, 158.170212]|-0.067476 [-0.403382, 0.000947]|-66.060887 [-158.267807, 144.292007]|69.892028|119.461106|79 / exact_on_saved_diagnostic_sample|
|failure_terminal|18|18|-2732.112745 [-3654.336670, -1575.521362]|-10134.834418 [-10135.117188, -10134.605469]|-7402.721653 [-8559.083984, -6480.780273]|7402.721653|8398.259668|18 / exact_on_saved_diagnostic_sample|
|failure_prefix|86266|8192|-732.581811 [-3627.980713, 174.845993]|-732.819700 [-3631.533691, 184.111816]|-0.237890 [-174.313843, 166.463135]|4.372845|16.946954|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16481。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_06.json)

```json
{
  "file": "q_probes/round_06.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_06_train_failure_values.csv.gz)

```json
{
  "file": "round_06_train_failure_values.csv.gz",
  "prefix_file": "round_06_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 2121,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -3640.686767578125,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -6494.339599609375,
      "terminal_mae": 6494.339599609375,
      "prefix_td_mae": 2.43796693410305,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1213.5484619140625,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -8920.980834960938,
      "terminal_mae": 8920.980834960938,
      "prefix_td_mae": 3.634439733644195,
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
- 失败终止诊断样本平均 target−Q=-7402.721653；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-6494.339600，终止MAE=6494.339600。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-8920.980835，终止MAE=8920.980835。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": -2,
  "validation_boundary_failures_delta": 2,
  "validation_changes_per_hour_delta": 2821.0409475216343,
  "validation_reversals_per_hour_delta": 1935.6199120835408,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 1.8678955794326573
}
```
