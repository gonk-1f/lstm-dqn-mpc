# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 5 轮报告

Train 完成 31/35；Greedy Validation 完成 8/8。
实际 Adam 更新 405309（累计 1859042）；Target 复制 406（含初始化累计 1860）。
Epsilon 0.366667 → 0.208333。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|31/35|0|4|0|0|4|0|2867.812|1321.525|
|Greedy Validation|8/8|0|0|0|0|0|0|3282.325|1449.971|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1595.281831|34936.672088|6560144.770542|3926.330719|6599007.773349|2537.317355|6601545.090705|
|Train/仅失败前缀|4|98.577283|2158.842500|225483.474417|431.745329|228074.062245|1148.269414|229222.331659|
|Train/仅完整航段子集|31|1496.704547|32777.829589|6334661.296125|3494.585390|6370933.711104|1389.047941|6372322.759045|
|Validation/全部已执行区间|8|608.937557|13335.732495|1564363.120958|1104.967266|1578803.820719|220.228769|1579024.049488|
|Validation/仅失败前缀|0|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|0.000000|
|Validation/仅完整航段子集|8|608.937557|13335.732495|1564363.120958|1104.967266|1578803.820719|220.228769|1579024.049488|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：1579024.049488 CNY。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|24642293|-71.528844 [-3284.354492, 150.175491]|-71.483454 [-3286.641602, 147.824539]|0.045390 [-185.148682, 187.284851]|1.047780|3.498102|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1942|-16.032638 [-895.829468, 82.813919]|-0.110013 [-1.076227, -0.000707]|15.922625 [-82.833061, 894.753235]|36.702168|77.703509|1942 / exact_all_draws|
|failure_terminal|—|264|-2500.851387 [-3498.972168, -1922.077393]|-10134.796098 [-10134.972656, -10134.664062]|-7633.944721 [-8212.609375, -6635.883301]|7633.944721|8178.118140|264 / exact_all_draws|
|failure_prefix|—|1295277|-735.381771 [-3510.964844, 189.400116]|-735.389616 [-3510.705566, 188.176010]|-0.007844 [-131.406128, 147.086060]|3.059849|11.794917|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：264；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4305；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_05_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|957572|8192|-64.192972 [-2341.520996, 135.083862]|-63.610649 [-2339.737793, 137.282654]|0.582323 [-79.380188, 68.709717]|1.147352|3.705035|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|76|76|-28.398587 [-894.953064, 61.257954]|-0.106179 [-1.076227, -0.000707]|28.292408 [-61.258659, 893.876831]|40.230098|91.595541|76 / exact_on_saved_diagnostic_sample|
|failure_terminal|9|9|-2915.585422 [-3518.061523, -2024.661133]|-10134.844510 [-10134.972656, -10134.686523]|-7219.259115 [-8110.311523, -6616.793945]|7219.259115|8085.668945|9 / exact_on_saved_diagnostic_sample|
|failure_prefix|42343|8192|-773.784840 [-3509.595947, 157.412048]|-772.086272 [-3505.287109, 157.961853]|1.698568 [-85.261414, 117.218628]|3.158434|11.192465|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16469。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_05.json)

```json
{
  "file": "q_probes/round_05.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_05_train_failure_values.csv.gz)

```json
{
  "file": "round_05_train_failure_values.csv.gz",
  "prefix_file": "round_05_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1212,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 171,
      "terminal_q": -2079.2021484375,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -8055.6845703125,
      "terminal_mae": 8055.6845703125,
      "prefix_td_mae": 5.140805705493649,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 183,
      "terminal_q": -1418.135009765625,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -8716.422607421875,
      "terminal_mae": 8716.422607421875,
      "prefix_td_mae": 1.00293221378958,
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
- 失败终止诊断样本平均 target−Q=-7219.259115；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 171 次；终止 target−Q=-8055.684570，终止MAE=8055.684570。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 183 次；终止 target−Q=-8716.422607，终止MAE=8716.422607。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 1,
  "validation_boundary_failures_delta": -1,
  "validation_changes_per_hour_delta": 587.2931550371163,
  "validation_reversals_per_hour_delta": -613.0651967698191,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -1.4538657852324226
}
```
