# V6 第 7 轮报告

Train 完成 25/35；Greedy Validation 完成 6/8。
实际 Adam 更新 323398（累计 2470152）；Target 复制 324（含初始化累计 2471）。
Epsilon 0.050000 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|25/35|0|10|0|3|7|0|2647.475|1359.537|
|Greedy Validation|6/8|0|2|0|0|2|0|3491.036|1902.733|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1143.671871|25046.413981|6605298.308876|3833.371504|6634178.094361|4569.956619|6638748.050980|
|Train/仅失败前缀|10|235.496802|5157.379967|1064981.194125|1046.223395|1071184.797487|2870.370702|1074055.168189|
|Train/仅完整航段子集|25|908.175069|19889.034014|5540317.114751|2787.148109|5562993.296874|1699.585917|5564692.882791|
|Validation/全部已执行区间|8|435.528178|9538.067104|2185742.900126|1369.378375|2196650.345605|976.614730|2197626.960335|
|Validation/仅失败前缀|2|49.973111|1094.411135|345103.011250|207.815273|346405.237658|574.171362|346979.409020|
|Validation/仅完整航段子集|6|385.555067|8443.655969|1840639.888876|1161.563102|1850245.107947|402.443368|1850647.551315|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|18675280|-14.992878 [-3434.864746, 250.542389]|-14.911727 [-3439.757568, 247.207794]|0.081151 [-289.321411, 266.528748]|1.507289|5.463255|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1665|73.766372 [-8.413704, 180.601807]|-0.050180 [-0.343292, 0.000947]|-73.816552 [-180.699402, 8.283703]|74.010451|124.432962|1665 / exact_all_draws|
|failure_terminal|—|415|-2772.858139 [-3895.304688, -1586.203613]|-10134.827139 [-10135.117188, -10134.605469]|-7361.968985 [-8548.402344, -6239.593750]|7361.968985|8341.188672|415 / exact_all_draws|
|failure_prefix|—|2020112|-706.663469 [-4154.450684, 218.700363]|-706.733082 [-4134.525391, 219.002884]|-0.069614 [-604.589294, 313.863708]|4.837854|17.817239|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：415；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4938；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_07_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|888467|8192|-16.805603 [-2667.208008, 201.059891]|-16.649109 [-2622.595215, 201.818665]|0.156494 [-152.459473, 150.569550]|1.599261|6.414117|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|77|77|78.170865 [-7.503130, 180.962448]|-0.041512 [-0.224807, 0.000947]|-78.212377 [-181.060043, 7.373130]|78.403887|129.671609|77 / exact_on_saved_diagnostic_sample|
|failure_terminal|23|23|-2852.946087 [-3887.964111, -1898.100708]|-10134.840948 [-10135.117188, -10134.679688]|-7281.894871 [-8236.678711, -6246.934570]|7281.894871|8117.468604|23 / exact_on_saved_diagnostic_sample|
|failure_prefix|111433|8192|-691.599874 [-4002.031982, 214.797409]|-692.064713 [-4011.117432, 217.294739]|-0.464839 [-146.788574, 200.091064]|4.402967|16.252136|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16484。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_07.json)

```json
{
  "file": "q_probes/round_07.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_07_train_failure_values.csv.gz)

```json
{
  "file": "round_07_train_failure_values.csv.gz",
  "prefix_file": "round_07_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 3030,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -3830.46044921875,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -6304.56591796875,
      "terminal_mae": 6304.56591796875,
      "prefix_td_mae": 5.64982778031305,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1158.66552734375,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -8975.86376953125,
      "terminal_mae": 8975.86376953125,
      "prefix_td_mae": 10.451237861683827,
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
- 失败终止诊断样本平均 target−Q=-7281.894871；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-6304.565918，终止MAE=6304.565918。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-8975.863770，终止MAE=8975.863770。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 3,
  "validation_boundary_failures_delta": -3,
  "validation_changes_per_hour_delta": 440.74053947950324,
  "validation_reversals_per_hour_delta": -107.64322339432806,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 2.2581088955641953
}
```
