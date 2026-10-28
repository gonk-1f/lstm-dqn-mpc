# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 1 轮报告

Train 完成 18/35；Greedy Validation 完成 7/8。
实际 Adam 更新 274332（累计 274332）；Target 复制 274（含初始化累计 275）。
Epsilon 1.000000 → 0.841667。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|18/35|8|9|0|12|5|0|3275.907|1552.701|
|Greedy Validation|7/8|0|1|0|0|1|0|3444.113|1901.231|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1112.059197|24354.096416|2065748.868208|4228.431062|2094331.395687|1705.960263|2096037.355950|
|Train/仅失败前缀|17|484.016558|10599.962625|863306.546917|2098.215834|876004.725376|1435.275569|877440.000946|
|Train/仅完整航段子集|18|628.042639|13754.133791|1202442.321292|2130.215228|1218326.670310|270.684694|1218597.355004|
|Validation/全部已执行区间|8|528.426863|11572.548294|1718968.542000|673.391069|1731214.481364|826.696203|1732041.177567|
|Validation/仅失败前缀|1|44.221796|968.457337|67424.971458|92.275923|68485.704718|287.089243|68772.793962|
|Validation/仅完整航段子集|7|484.205067|10604.090957|1651543.570542|581.115146|1662728.776645|539.606960|1663268.383606|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|7446199|-53.633574 [-495.646637, 22.764416]|-53.628213 [-495.551056, 22.859892]|0.005361 [-10.119968, 14.174377]|0.184849|0.543333|8192 / uniform_reservoir_estimate|
|normal_terminal|—|806|-52.168681 [-270.497040, 8.621291]|-0.529409 [-1.310338, -0.035770]|51.639272 [-8.657061, 269.186707]|52.213313|184.102657|806 / exact_all_draws|
|failure_terminal|—|1549|-218.578826 [-633.142517, -0.368376]|-10134.740509 [-10135.122070, -10134.480469]|-9916.161685 [-10134.239258, -9501.979492]|9916.161685|10108.457422|1549 / exact_all_draws|
|failure_prefix|—|10108694|-69.086743 [-629.846985, 7.904484]|-69.089924 [-630.456604, 7.839914]|-0.003180 [-23.362396, 15.261536]|0.167939|0.503589|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：1549；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：1549；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_01_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|179078|8192|-84.822546 [-493.906677, 21.464272]|-84.676334 [-494.246796, 21.656248]|0.146212 [-6.595337, 12.167091]|0.307000|0.834988|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|20|20|-71.078232 [-276.462738, 8.304823]|-0.399972 [-1.310338, -0.035770]|70.678261 [-8.340592, 275.152405]|71.613755|262.629063|20 / exact_on_saved_diagnostic_sample|
|failure_terminal|21|21|-474.129072 [-641.194397, -391.058655]|-10134.728237 [-10135.122070, -10134.480469]|-9660.599098 [-9743.777344, -9493.927734]|9660.599098|9741.115234|21 / exact_on_saved_diagnostic_sample|
|failure_prefix|145213|8192|-136.177836 [-638.204041, 5.977456]|-136.063099 [-638.557983, 6.245031]|0.114736 [-9.592682, 12.428253]|0.302009|0.781372|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 324332；本次只读测量 16425。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_01.json)

```json
{
  "file": "q_probes/round_01.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_01_train_failure_values.csv.gz)

```json
{
  "file": "round_01_train_failure_values.csv.gz",
  "prefix_file": "round_01_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 5151,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 98,
      "terminal_q": -515.5277099609375,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -9619.359008789062,
      "terminal_mae": 9619.359008789062,
      "prefix_td_mae": 0.42290078725246405,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 104,
      "terminal_q": -408.4767150878906,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -9726.08090209961,
      "terminal_mae": 9726.08090209961,
      "prefix_td_mae": 0.2213070092611755,
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
- 失败终止诊断样本平均 target−Q=-9660.599098；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 98 次；终止 target−Q=-9619.359009，终止MAE=9619.359009。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 104 次；终止 target−Q=-9726.080902，终止MAE=9726.080902。前缀与终止拟合及Validation需同时查看。
- 这是本运行首轮，没有轮间基线；不能据本轮 loss 或单状态 Q 宣称改善。
