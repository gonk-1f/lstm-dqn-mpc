# V6 第 1 轮报告

Train 完成 20/35；Greedy Validation 完成 7/8。
实际 Adam 更新 274126（累计 274126）；Target 复制 274（含初始化累计 275）。
Epsilon 1.000000 → 0.841667。

|划分|完成|上边界失败|下边界失败|内区失败|增量不可达|SOC不可行|结构功率失败|FC变载/h|相邻反向/h|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Train|20/35|7|8|0|13|2|0|3273.641|1549.193|
|Greedy Validation|7/8|0|1|0|0|1|0|1645.958|1464.706|

费率以实际执行秒数折算；失败前缀长度变化会影响跨轮比较。

|划分/成本范围|航段数|实际FC氢耗(kg)|氢成本(CNY)|FC退化(CNY)|电池退化(CNY)|四项账本总额(CNY)|电池等效氢修正(CNY)|修正后经济成本(CNY)|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Train/全部已执行区间|35|1094.421254|23967.825465|2022804.721625|4233.552718|2051006.099808|1673.830457|2052679.930265|
|Train/仅失败前缀|15|405.926623|8889.793040|647051.764500|1706.883462|657648.441002|1291.658561|658940.099563|
|Train/仅完整航段子集|20|688.494631|15078.032425|1375752.957125|2526.669256|1393357.658806|382.171896|1393739.830702|
|Validation/全部已执行区间|8|544.829389|11931.763611|313411.058250|728.028546|326070.850407|522.995266|326593.845673|
|Validation/仅失败前缀|1|51.547325|1128.886416|19879.220708|86.600942|21094.708066|287.089750|21381.797816|
|Validation/仅完整航段子集|7|493.282064|10802.877195|293531.837542|641.427604|304976.142341|235.905515|305212.047856|

Train 固定全套完整成本：未提供，存在未完成航段。
Validation 固定全套完整成本：未提供，存在未完成航段。

四项账本包括 FC 氢成本、FC 退化、电池退化和岸电成本；本任务岸电关闭。
电池等效氢修正保留正负号，另列于实际账本之外。旧记录若未直接保存 FC 氢耗质量，由正式氢价 21.9 CNY/kg 换算。

## Q 与 TD 拟合证据

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|9345598|-60.920864 [-503.137177, 11.321260]|-60.917642 [-502.991760, 11.093120]|0.003221 [-7.819469, 12.536545]|0.175551|0.534805|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1042|-36.163654 [-193.271378, 6.163786]|-0.388040 [-0.885807, -0.026777]|35.775615 [-6.390186, 192.385574]|36.276286|131.535636|1042 / exact_all_draws|
|failure_terminal|—|1434|-213.135120 [-600.588318, -0.800148]|-10134.722982 [-10135.026367, -10134.486328]|-9921.587876 [-10134.222656, -9534.438477]|9921.587876|10112.060449|1434 / exact_all_draws|
|failure_prefix|—|8195990|-79.608178 [-599.140625, 8.576562]|-79.611478 [-599.221252, 8.278689]|-0.003301 [-19.974121, 10.578430]|0.157760|0.487551|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：1434；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：1434；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_01_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|196882|8192|-95.876424 [-488.846100, 7.718309]|-95.911337 [-488.842773, 7.238863]|-0.034913 [-3.315229, 12.915329]|0.263632|0.766295|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|22|22|-58.039393 [-197.231064, 7.098032]|-0.356908 [-0.885807, -0.026777]|57.682485 [-7.324432, 196.345261]|58.951210|181.779103|22 / exact_on_saved_diagnostic_sample|
|failure_terminal|20|20|-478.289153 [-602.118042, -414.876282]|-10134.715918 [-10135.026367, -10134.486328]|-9656.426611 [-9719.952148, -9532.908203]|9656.426611|9714.611182|20 / exact_on_saved_diagnostic_sample|
|failure_prefix|127202|8192|-150.684227 [-602.531921, 8.306583]|-150.778257 [-602.379883, 8.561418]|-0.094030 [-8.872963, 11.202515]|0.275077|0.753809|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 324126；本次只读测量 16426。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_01.json)

```json
{
  "file": "q_probes/round_01.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_01_train_failure_values.csv.gz)

```json
{
  "file": "round_01_train_failure_values.csv.gz",
  "prefix_file": "round_01_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 4545,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 87,
      "terminal_q": -602.1180419921875,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -9532.908325195312,
      "terminal_mae": 9532.908325195312,
      "prefix_td_mae": 0.38235790682154774,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 112,
      "terminal_q": -420.1988220214844,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -9714.330474853516,
      "terminal_mae": 9714.330474853516,
      "prefix_td_mae": 0.40025076960885764,
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
- 失败终止诊断样本平均 target−Q=-9656.426611；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 87 次；终止 target−Q=-9532.908325，终止MAE=9532.908325。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 112 次；终止 target−Q=-9714.330475，终止MAE=9714.330475。前缀与终止拟合及Validation需同时查看。
- 这是本运行首轮，没有轮间基线；不能据本轮 loss 或单状态 Q 宣称改善。
