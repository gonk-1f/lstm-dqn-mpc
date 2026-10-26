# V6 第 3 轮报告

Train 完成 29/35；Greedy Validation 完成 7/8。
实际 Adam 更新 383579（累计 1016154）；Target 复制 384（含初始化累计 1017）。
Epsilon 0.683333 → 0.525000。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|29/35|0|6|0|2|4|0|3150.004|1462.491|
|Greedy Validation|7/8|0|1|0|0|1|0|1088.336|682.156|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1427.725296|31267.183980|5163072.987042|3358.642621|5197698.813644|3328.967611|5201027.781254|
|Train/仅失败前缀|6|167.461430|3667.405322|274857.351458|582.265118|279107.021899|1722.359252|280829.381150|
|Train/仅完整航段子集|29|1260.263866|27599.778658|4888215.635584|2776.377504|4918591.791745|1606.608359|4920198.400104|
|Validation/全部已执行区间|8|537.964456|11781.421594|360694.686708|858.981962|373335.090264|683.121126|374018.211390|
|Validation/仅失败前缀|1|46.563780|1019.746790|18490.199167|92.230924|19602.176881|287.059449|19889.236331|
|Validation/仅完整航段子集|7|491.400676|10761.674803|342204.487542|766.751038|353732.913383|396.061677|354128.975060|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|18983278|-104.538226 [-1867.415405, 72.728844]|-104.503244 [-1867.210205, 73.072327]|0.034982 [-125.467682, 121.269897]|0.788093|2.634006|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1821|-43.591160 [-598.423828, 68.159225]|-0.214666 [-0.885807, -0.007850]|43.376494 [-68.456230, 597.538025]|62.716727|394.059479|1821 / exact_all_draws|
|failure_terminal|—|840|-1387.655906 [-2682.361328, -786.379822]|-10134.769364 [-10135.089844, -10134.486328]|-8747.113451 [-9348.149414, -7452.661133]|8747.113451|9237.976123|840 / exact_all_draws|
|failure_prefix|—|5563117|-345.680493 [-2700.202881, 62.337990]|-345.679830 [-2700.186768, 62.063370]|0.000663 [-238.397034, 141.811890]|1.286293|4.235260|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：840；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：3421；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_03_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|828481|8192|-94.369965 [-1565.861572, 65.254890]|-94.202750 [-1558.166016, 65.459595]|0.167215 [-50.114258, 71.218933]|0.911611|3.306545|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|76|76|-37.243635 [-599.722961, 56.430695]|-0.194154 [-0.885807, -0.007850]|37.049480 [-56.741688, 598.837158]|69.117161|421.728401|76 / exact_on_saved_diagnostic_sample|
|failure_terminal|27|27|-1650.200950 [-2332.428711, -939.138794]|-10134.759042 [-10135.089844, -10134.486328]|-8484.558124 [-9195.390625, -7802.425781]|8484.558124|8960.291211|27 / exact_on_saved_diagnostic_sample|
|failure_prefix|171416|8192|-412.024430 [-2351.012939, 60.979031]|-411.977445 [-2371.949463, 61.050674]|0.046985 [-107.901001, 83.218140]|1.610645|5.784550|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16487。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_03.json)

```json
{
  "file": "q_probes/round_03.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_03_train_failure_values.csv.gz)

```json
{
  "file": "round_03_train_failure_values.csv.gz",
  "prefix_file": "round_03_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 1818,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -2332.52685546875,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -7802.49951171875,
      "terminal_mae": 7802.49951171875,
      "prefix_td_mae": 1.4256166369709748,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -939.1387939453125,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -9195.390502929688,
      "terminal_mae": 9195.390502929688,
      "prefix_td_mae": 2.1562049815196866,
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
- 失败终止诊断样本平均 target−Q=-8484.558124；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-7802.499512，终止MAE=7802.499512。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-9195.390503，终止MAE=9195.390503。前缀与终止拟合及Validation需同时查看。
- TD 拟合误差下降，但 Greedy Validation 完成数未提高。
- TD 拟合误差下降，但上下 SOC 边界失败总数未减少。
- TD 拟合误差下降，但 FC 每小时变载次数未改善。
- TD 拟合误差下降，但 FC 每小时相邻反向次数未改善。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 0,
  "validation_boundary_failures_delta": 0,
  "validation_changes_per_hour_delta": 1044.2754703160583,
  "validation_reversals_per_hour_delta": 678.7599572522732,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": -0.2428672754840111
}
```
