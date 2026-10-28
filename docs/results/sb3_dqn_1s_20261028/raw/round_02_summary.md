# Stable-Baselines3 DQN 对照：固定8遍预算

# V6 第 2 轮报告

Train 完成 28/35；Greedy Validation 完成 7/8。
实际 Adam 更新 366063（累计 640395）；Target 复制 366（含初始化累计 641）。
Epsilon 0.841667 → 0.683333。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|28/35|1|6|0|3|4|0|3222.449|1514.946|
|Greedy Validation|7/8|0|1|0|0|1|0|130.020|29.965|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1423.583818|31176.485604|3657863.715833|4084.557081|3693124.758518|2619.284637|3695744.043155|
|Train/仅失败前缀|7|201.167524|4405.568774|471514.384583|813.918548|476733.871905|1578.653671|478312.525576|
|Train/仅完整航段子集|28|1222.416294|26770.916830|3186349.331250|3270.638533|3216390.886613|1040.630965|3217431.517579|
|Validation/全部已执行区间|8|549.437141|12032.673393|24396.398375|949.258616|37378.330384|453.221541|37831.551925|
|Validation/仅失败前缀|1|48.147466|1054.429502|5517.721875|100.675134|6672.826511|287.033080|6959.859591|
|Validation/仅完整航段子集|7|501.289675|10978.243891|18878.676500|848.583482|30705.503873|166.188461|30871.692334|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|14859627|-97.370145 [-1258.870605, 45.509087]|-97.356891 [-1253.825439, 44.844692]|0.013254 [-46.189331, 59.313049]|0.421727|1.308021|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1550|-78.656723 [-578.314636, 28.727905]|-0.324398 [-1.310338, -0.012874]|78.332325 [-28.763676, 577.004272]|80.494615|375.853804|1550 / exact_all_draws|
|failure_terminal|—|1174|-753.659163 [-1508.668701, -393.321320]|-10134.732730 [-10135.122070, -10134.480469]|-9381.073560 [-9741.514648, -8626.159180]|9381.073560|9695.906055|1174 / exact_all_draws|
|failure_prefix|—|8565681|-190.279117 [-1509.895630, 32.190380]|-190.277482 [-1508.751343, 32.405033]|0.001636 [-114.053711, 60.749023]|0.459029|1.333261|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：1174；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：2723；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_02_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|495955|8192|-106.804188 [-1197.569458, 43.281532]|-107.051479 [-1203.473267, 43.396446]|-0.247291 [-15.576145, 19.556274]|0.587056|1.801484|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|48|48|-71.524511 [-581.482910, 29.260193]|-0.255807 [-1.310338, -0.012874]|71.268703 [-29.295963, 580.172546]|75.659980|338.211113|48 / exact_on_saved_diagnostic_sample|
|failure_terminal|28|28|-1059.242698 [-1513.608398, -731.367981]|-10134.742188 [-10135.122070, -10134.480469]|-9075.499477 [-9403.467773, -8621.513672]|9075.499477|9373.565381|28 / exact_on_saved_diagnostic_sample|
|failure_prefix|194364|8192|-255.427646 [-1511.922485, 32.853035]|-255.701572 [-1512.236694, 32.788498]|-0.273926 [-105.339508, 59.604309]|0.708857|2.052625|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 690395；本次只读测量 16460。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/q_probes/round_02.json)

```json
{
  "file": "q_probes/round_02.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-sb3-baseline/outputs/sb3_dqn_1s_seed42_run01/round_02_train_failure_values.csv.gz)

```json
{
  "file": "round_02_train_failure_values.csv.gz",
  "prefix_file": "round_02_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 2121,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_035_seg01",
      "origin_round": 1,
      "terminal_replay_id": 57891,
      "actual_training_draws": 144,
      "terminal_q": -1100.6844482421875,
      "terminal_target": -10134.88671875,
      "terminal_signed_residual": -9034.202270507812,
      "terminal_mae": 9034.202270507812,
      "prefix_td_mae": 1.2502016736971622,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 64423,
      "actual_training_draws": 153,
      "terminal_q": -848.2364501953125,
      "terminal_target": -10134.5576171875,
      "terminal_signed_residual": -9286.321166992188,
      "terminal_mae": 9286.321166992188,
      "prefix_td_mae": 0.6137880742155164,
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
- 失败终止诊断样本平均 target−Q=-9075.499477；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_035_seg01 的终止经验累计实际抽样 144 次；终止 target−Q=-9034.202271，终止MAE=9034.202271。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 153 次；终止 target−Q=-9286.321167，终止MAE=9286.321167。前缀与终止拟合及Validation需同时查看。

相对上一轮的记录变化：
```json
{
  "validation_completed_delta": 0,
  "validation_boundary_failures_delta": 0,
  "validation_changes_per_hour_delta": -3314.092883007793,
  "validation_reversals_per_hour_delta": -1871.265916722773,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 3.561478979392618
}
```
