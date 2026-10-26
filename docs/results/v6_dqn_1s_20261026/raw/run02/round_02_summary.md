# V6 第 2 轮报告

Train 完成 28/35；Greedy Validation 完成 7/8。
实际 Adam 更新 358449（累计 632575）；Target 复制 358（含初始化累计 633）。
Epsilon 0.841667 → 0.683333。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|28/35|1|6|0|3|4|0|3220.857|1515.439|
|Greedy Validation|7/8|0|1|0|0|1|0|44.060|3.396|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1382.522114|30277.234297|3383290.406125|3871.119983|3417438.760405|2840.328713|3420279.089118|
|Train/仅失败前缀|7|216.614373|4743.854759|515914.695667|812.314154|521470.864580|1578.627485|523049.492065|
|Train/仅完整航段子集|28|1165.907741|25533.379538|2867375.710458|3058.805829|2895967.895825|1261.701228|2897229.597053|
|Validation/全部已执行区间|8|528.159864|11566.701014|25468.334042|810.174809|37845.209865|905.413565|38750.623430|
|Validation/仅失败前缀|1|48.189518|1055.350451|1807.363833|83.104435|2945.818719|287.038774|3232.857493|
|Validation/仅完整航段子集|7|479.970345|10511.350563|23660.970208|727.070374|34899.391145|618.374791|35517.765937|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|15215879|-108.009762 [-1122.568115, 33.691830]|-107.994741 [-1121.974121, 32.964916]|0.015021 [-43.731995, 48.606018]|0.423507|1.269977|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1601|-64.390639 [-416.919250, 20.796591]|-0.298142 [-0.885807, -0.007850]|64.092496 [-20.922659, 416.033447]|65.927372|285.309753|1601 / exact_all_draws|
|failure_terminal|—|1147|-780.413399 [-1548.994507, -420.675568]|-10134.743374 [-10135.089844, -10134.486328]|-9354.329972 [-9714.152344, -8586.028320]|9354.329972|9666.742480|1147 / exact_all_draws|
|failure_prefix|—|7722109|-214.942843 [-1554.484497, 26.674646]|-214.947202 [-1560.144287, 26.316385]|-0.004359 [-107.542480, 57.693054]|0.489349|1.497244|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：1147；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：2581；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_02_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|502329|8192|-114.825542 [-1114.637817, 26.383640]|-114.833105 [-1117.220825, 26.326283]|-0.007563 [-24.049072, 47.919250]|0.600219|1.850939|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|50|50|-57.546299 [-420.938446, 20.994867]|-0.236321 [-0.885807, -0.007850]|57.309978 [-21.021645, 420.052643]|61.244544|335.046338|50 / exact_on_saved_diagnostic_sample|
|failure_terminal|27|27|-1106.127369 [-1571.937622, -782.421265]|-10134.761900 [-10135.089844, -10134.486328]|-9028.634440 [-9352.108398, -8563.084961]|9028.634440|9334.748535|27 / exact_on_saved_diagnostic_sample|
|failure_prefix|180169|8192|-292.231946 [-1505.123535, 25.373379]|-292.273557 [-1506.103516, 25.675594]|-0.041611 [-83.184326, 32.460754]|0.835162|2.561008|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 682575；本次只读测量 16461。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_02.json)

```json
{
  "file": "q_probes/round_02.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_02_train_failure_values.csv.gz)

```json
{
  "file": "round_02_train_failure_values.csv.gz",
  "prefix_file": "round_02_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 2121,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 151,
      "terminal_q": -1514.0223388671875,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -8621.004028320312,
      "terminal_mae": 8621.004028320312,
      "prefix_td_mae": 0.8232725029749586,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 164,
      "terminal_q": -782.4212646484375,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -9352.108032226562,
      "terminal_mae": 9352.108032226562,
      "prefix_td_mae": 0.7853954390974234,
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
- 失败终止诊断样本平均 target−Q=-9028.634440；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 151 次；终止 target−Q=-8621.004028，终止MAE=8621.004028。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 164 次；终止 target−Q=-9352.108032，终止MAE=9352.108032。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 0,
  "validation_boundary_failures_delta": 0,
  "validation_changes_per_hour_delta": -1601.8974697039168,
  "validation_reversals_per_hour_delta": -1461.3101993433736,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 3.6043836457058003
}
```
