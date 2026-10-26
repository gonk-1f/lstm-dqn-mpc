# V6 第 8 轮报告

Train 完成 25/35；Greedy Validation 完成 3/8。
实际 Adam 更新 312007（累计 2782159）；Target 复制 312（含初始化累计 2783）。
Epsilon 0.050000 → 0.050000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|25/35|0|10|0|4|6|0|2633.583|1372.469|
|Greedy Validation|3/8|0|5|0|5|0|0|1144.241|663.136|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1057.926555|23168.591562|7518367.971792|4078.099999|7545614.663353|4455.261726|7550069.925078|
|Train/仅失败前缀|10|193.878537|4245.939968|1029797.811083|1055.978240|1035099.729291|2870.308486|1037970.037777|
|Train/仅完整航段子集|25|864.048018|18922.651594|6488570.160709|3022.121759|6510514.934062|1584.953240|6512099.887301|
|Validation/全部已执行区间|8|251.703123|5512.298397|613684.407667|1150.979100|620347.685164|1681.540160|622029.225324|
|Validation/仅失败前缀|5|87.912589|1925.285694|279649.214875|626.262366|282200.762935|1434.747533|283635.510468|
|Validation/仅完整航段子集|3|163.790534|3587.012703|334035.192792|524.716734|338146.922229|246.792627|338393.714855|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|17486989|-3.290938 [-3578.713135, 276.013885]|-3.197408 [-3581.021240, 273.570129]|0.093530 [-312.864319, 316.055542]|1.823921|7.257729|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1599|89.689316 [-7.393042, 210.998749]|-0.040284 [-0.399589, 0.000947]|-89.729600 [-211.096344, 7.263041]|89.784240|177.477615|1599 / exact_all_draws|
|failure_terminal|—|558|-2881.401877 [-4051.926514, -1885.979492]|-10134.829697 [-10135.120117, -10134.644531]|-7253.427818 [-8248.788086, -6082.971680]|7253.427818|8173.549316|558 / exact_all_draws|
|failure_prefix|—|2479302|-692.452636 [-4854.358398, 254.851059]|-692.561870 [-4849.998047, 255.358948]|-0.109234 [-667.756104, 388.338959]|4.714531|17.191405|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：558；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：5496；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_08_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|867005|8192|6.059989 [-2608.506348, 225.647552]|6.858602 [-2600.133789, 225.897247]|0.798614 [-157.527588, 189.944458]|2.170170|7.694991|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|78|78|97.536128 [28.150200, 210.958923]|-0.039457 [-0.399589, 0.000947]|-97.575586 [-211.056519, -28.223827]|97.575586|185.857571|78 / exact_on_saved_diagnostic_sample|
|failure_terminal|27|27|-2881.479574 [-4035.688965, -1931.718384]|-10134.811089 [-10135.120117, -10134.644531]|-7253.331525 [-8203.048828, -6099.209473]|7253.331525|8118.256885|27 / exact_on_saved_diagnostic_sample|
|failure_prefix|132890|8192|-681.744140 [-4157.218750, 213.803452]|-681.494267 [-4114.322754, 213.544754]|0.249873 [-665.819336, 188.983521]|4.941310|18.137827|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16489。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_08.json)

```json
{
  "file": "q_probes/round_08.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_08_train_failure_values.csv.gz)

```json
{
  "file": "round_08_train_failure_values.csv.gz",
  "prefix_file": "round_08_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 3030,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -3997.195068359375,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -6137.831298828125,
      "terminal_mae": 6137.831298828125,
      "prefix_td_mae": 3.948023701345684,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1238.502685546875,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -8896.026611328125,
      "terminal_mae": 8896.026611328125,
      "prefix_td_mae": 9.54974870492291,
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
- 失败终止诊断样本平均 target−Q=-7253.331525；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-6137.831299，终止MAE=6137.831299。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-8896.026611，终止MAE=8896.026611。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": -3,
  "validation_boundary_failures_delta": 3,
  "validation_changes_per_hour_delta": -2346.7951583006425,
  "validation_reversals_per_hour_delta": -1239.5967796962036,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 2.362157313830444
}
```
