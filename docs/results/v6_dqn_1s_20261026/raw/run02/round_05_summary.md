# V6 第 5 轮报告

Train 完成 28/35；Greedy Validation 完成 5/8。
实际 Adam 更新 358243（累计 1779547）；Target 复制 358（含初始化累计 1780）。
Epsilon 0.366667 → 0.208333。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|28/35|0|7|0|4|3|0|2939.687|1363.806|
|Greedy Validation|5/8|0|3|0|1|2|0|229.254|74.756|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1328.718776|29098.941204|6478669.248917|3437.741335|6511205.931456|3515.156280|6514721.087736|
|Train/仅失败前缀|7|186.407930|4082.333671|529235.788833|751.627683|534069.750187|2009.151369|536078.901556|
|Train/仅完整航段子集|28|1142.310846|25016.607533|5949433.460084|2686.113652|5977136.181269|1506.004911|5978642.186180|
|Validation/全部已执行区间|8|379.851284|8318.743116|229039.275375|1283.698042|238641.716533|1343.685827|239985.402360|
|Validation/仅失败前缀|3|50.701376|1110.360132|2409.703292|298.744042|3818.807466|861.135420|4679.942886|
|Validation/仅完整航段子集|5|329.149908|7208.382985|226629.572083|984.954000|234822.909067|482.550407|235305.459475|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|21422710|-47.690122 [-3069.743164, 438.019470]|-47.624966 [-3064.134766, 437.766418]|0.065156 [-275.884125, 198.301483]|1.341704|5.120238|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1754|55.267990 [-541.923889, 136.121521]|-0.107343 [-0.709794, -0.001130]|-55.375334 [-136.432510, 541.214111]|73.532101|107.681300|1754 / exact_all_draws|
|failure_terminal|—|310|-2403.546950 [-3360.818359, -1531.240356]|-10134.830642 [-10134.955078, -10134.605469]|-7731.283674 [-8603.365234, -6774.082031]|7731.283674|8483.706982|310 / exact_all_draws|
|failure_prefix|—|1502778|-746.068077 [-3417.765625, 343.712219]|-746.043123 [-3412.645508, 341.014343]|0.024954 [-280.145538, 188.605652]|4.038049|14.646533|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：310；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4189；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_05_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|932275|8192|-34.480268 [-2338.419189, 287.197449]|-34.214062 [-2347.400879, 292.438904]|0.266206 [-153.015015, 116.181305]|1.327458|5.409255|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|76|76|54.353019 [-538.340515, 121.251869]|-0.097591 [-0.709794, -0.001130]|-54.450610 [-121.265976, 537.630737]|73.148524|108.201071|76 / exact_on_saved_diagnostic_sample|
|failure_terminal|14|14|-2503.199393 [-3372.238770, -1576.393066]|-10134.838518 [-10134.955078, -10134.605469]|-7631.639265 [-8558.212891, -6762.661621]|7631.639265|8444.682520|14 / exact_on_saved_diagnostic_sample|
|failure_prefix|67635|8192|-738.597351 [-3391.608887, 179.088181]|-738.182060 [-3393.794434, 179.136597]|0.415291 [-162.318237, 143.558044]|3.821688|13.584622|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16474。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_05.json)

```json
{
  "file": "q_probes/round_05.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_05_train_failure_values.csv.gz)

```json
{
  "file": "round_05_train_failure_values.csv.gz",
  "prefix_file": "round_05_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 2121,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -3370.6298828125,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -6764.396484375,
      "terminal_mae": 6764.396484375,
      "prefix_td_mae": 3.364062075583351,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1228.4932861328125,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -8906.036010742188,
      "terminal_mae": 8906.036010742188,
      "prefix_td_mae": 4.31100352710446,
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
- 失败终止诊断样本平均 target−Q=-7631.639265；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-6764.396484，终止MAE=6764.396484。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-8906.036011，终止MAE=8906.036011。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 1,
  "validation_boundary_failures_delta": -1,
  "validation_changes_per_hour_delta": -3007.290885608133,
  "validation_reversals_per_hour_delta": -1465.2546266717404,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 0.558628571483208
}
```
