# V6 第 4 轮报告

Train 完成 31/35；Greedy Validation 完成 4/8。
实际 Adam 更新 405150（累计 1421304）；Target 复制 405（含初始化累计 1422）。
Epsilon 0.525000 → 0.366667。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|31/35|0|4|0|3|1|0|3052.301|1420.709|
|Greedy Validation|4/8|0|4|0|3|1|0|3236.545|1540.011|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1557.146776|34101.514402|6023749.602084|3681.631281|6061532.747766|2860.172399|6064392.920166|
|Train/仅失败前缀|4|94.700979|2073.951444|177313.720167|413.077308|179800.748919|1148.029109|180948.778028|
|Train/仅完整航段子集|31|1462.445797|32027.562957|5846435.881917|3268.553973|5881731.998847|1712.143291|5883444.142138|
|Validation/全部已执行区间|8|273.910114|5998.631499|988230.544583|1034.730162|995263.906244|1577.206721|996841.112965|
|Validation/仅失败前缀|4|73.862007|1617.577947|331350.103667|391.405861|333359.087474|1148.195384|334507.282858|
|Validation/仅完整航段子集|4|200.048107|4381.053552|656880.440917|643.324302|661904.818770|429.011337|662333.830107|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|23207751|-80.752702 [-2667.182129, 380.649292]|-80.707548 [-2665.644287, 380.079071]|0.045154 [-235.876358, 185.044006]|1.125736|3.712466|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1946|12.090617 [-611.841370, 105.112350]|-0.137866 [-0.885807, -0.001130]|-12.228483 [-105.423340, 610.955566]|56.688085|130.562309|1946 / exact_all_draws|
|failure_terminal|—|458|-2068.370523 [-2985.050049, -1264.510742]|-10134.832219 [-10135.089844, -10134.496094]|-8066.461692 [-8870.264648, -7149.804688]|8066.461692|8727.381152|458 / exact_all_draws|
|failure_prefix|—|2719445|-508.555453 [-2981.920898, 289.917023]|-508.552613 [-2983.582520, 288.875916]|0.002840 [-272.954346, 158.582947]|2.248565|7.801717|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：458；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：3879；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_04_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|936834|8192|-60.981944 [-2120.314453, 259.233154]|-60.701056 [-2086.799072, 265.908752]|0.280889 [-60.960510, 80.762939]|1.236429|4.616433|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|78|78|49.845670 [-452.461487, 104.780075]|-0.104779 [-0.709794, -0.001130]|-49.950449 [-105.091064, 451.751678]|65.613667|93.707260|78 / exact_on_saved_diagnostic_sample|
|failure_terminal|13|13|-2332.386268 [-2984.175049, -1643.850830]|-10134.817082 [-10134.955078, -10134.664062]|-7802.430814 [-8490.813477, -7150.679688]|7802.430814|8469.178320|13 / exact_on_saved_diagnostic_sample|
|failure_prefix|63075|8192|-669.206484 [-2973.733643, 293.199310]|-669.324835 [-2976.180908, 291.802734]|-0.118351 [-157.361206, 142.446899]|3.504849|12.386308|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16475。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_04.json)

```json
{
  "file": "q_probes/round_04.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_04_train_failure_values.csv.gz)

```json
{
  "file": "round_04_train_failure_values.csv.gz",
  "prefix_file": "round_04_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1212,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -2987.90966796875,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -7147.11669921875,
      "terminal_mae": 7147.11669921875,
      "prefix_td_mae": 1.804651070904258,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1123.15869140625,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -9011.37060546875,
      "terminal_mae": 9011.37060546875,
      "prefix_td_mae": 3.6723755590173583,
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
- 失败终止诊断样本平均 target−Q=-7802.430814；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-7147.116699，终止MAE=7147.116699。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-9011.370605，终止MAE=9011.370605。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": -3,
  "validation_boundary_failures_delta": 3,
  "validation_changes_per_hour_delta": 2148.2098168060193,
  "validation_reversals_per_hour_delta": 857.8549590688685,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -6.641740622035293
}
```
