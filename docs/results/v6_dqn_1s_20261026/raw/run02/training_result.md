# V6 single-configuration training result

Status: completed; completed traversals: 8/8.
Warmup: 50000 genuine transitions, 0 Adam updates.
Formal training: 2782159 executed transitions; Adam updates: 2782159; Target copies (including initialization): 2783.
Test payloads opened: 0. This report describes this run's recorded configuration.

|Traversal|Train /35|Validation /8|Train steps|Adam|Target copies|Epsilon start/end|
|---|---:|---:|---:|---:|---:|---|
|1|20|7|274126|274126|274|1.000000/0.841667|
|2|28|7|358449|358449|358|0.841667/0.683333|
|3|29|7|383579|383579|384|0.683333/0.525000|
|4|31|4|405150|405150|405|0.525000/0.366667|
|5|28|5|358243|358243|358|0.366667/0.208333|
|6|28|3|367207|367207|367|0.208333/0.050000|
|7|25|6|323398|323398|324|0.050000/0.050000|
|8|25|3|312007|312007|312|0.050000/0.050000|

Research candidate: traversal 1.
Selection prioritizes Validation completion count, then duration coverage; full-set cost is used when all eight complete.
Incomplete-prefix costs are never represented as a fixed complete-set cost.
TD targets are bootstrapped training targets, not Q*. Online/Target updates change those targets during training.
Loss, TD fitting error and a single probe Q do not establish control improvement; evaluate completion, boundary failures and FC fluctuations together.
Each round_NN_summary.md/.json reports Train/Validation failure sides, actual hydrogen and economic components, failed prefixes, completed subsets and recorded Q diagnostics.
Fixed Validation probes are reporting only and never enter Replay or training.

## Comparison with V5 round32

Different time resolution, action availability, state, objective and training budget prevent a single-factor causal comparison.
Change/reversal rates use actual physical hours; per-step event counts are not compared directly.
Jointly completed Validation IDs at the final traversal: 3.
Same-ID complete economic cost: V5 14180.327670 CNY; V6 338393.714855 CNY.

Final recorded Validation completion: 3/8; V5 round32:5/8.
V6 FC changes/hour: 1144.240815; adjacent reversals/hour: 663.136021.
These rates include actual failed prefixes and describe different policy paths; same-ID complete costs are reported separately.
Each completed traversal preserves round_NN_failure_details.csv and .json with Train/Validation failure states and all 13 candidate legality flags.
Train failures occurred under epsilon-greedy exploration; per-action random-choice flags were not stored, so a random action cannot be identified as their cause.

Validation CSV.gz and PNG files contain load, FC power, battery power, SOC, FC increments,
all13 action Q values, legal hold rank, separate reward terms and cumulative reward terms.
Fixed probes preserve V5 physical SOC/FC/time and add V6 causal history. They have a different input/output
definition and reward unit from V5; raw numerical Q values must not be compared as identical targets.
See fixed_probe_history.json and v5_v6_validation_comparison.csv.

Data reuse preserved every canonical original anchor exactly. No new real1Hz measurements were created.
The ±60kW/s action capability remains a simulation assumption.

## Existing fixed physical probes in V6 coordinates

|Sample|Time/s|Initial hold rank /13|Latest hold rank /13|Greedy increment/kW|Greedy minus hold Q|
|---|---:|---:|---:|---:|---:|
|parent_011_seg01|750|1|6|-50|0.517723|
|parent_011_seg01|780|1|6|-20|0.301819|
|parent_045_seg01|870|1|1|0|0.000000|
|parent_045_seg01|900|1|1|0|0.000000|

## Per-round Q fitting and control evidence

## Traversal 1: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_01_summary.md)

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

Interpretation and control consistency:

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

## Traversal 2: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_02_summary.md)

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

Interpretation and control consistency:

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

## Traversal 3: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_03_summary.md)

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

Interpretation and control consistency:

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

## Traversal 4: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_04_summary.md)

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

Interpretation and control consistency:

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

## Traversal 5: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_05_summary.md)

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

Interpretation and control consistency:

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

## Traversal 6: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_06_summary.md)

### 在线训练 draw 统计

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|—|21846747|-32.405743 [-3329.055664, 331.424713]|-32.332545 [-3326.468994, 330.361145]|0.073198 [-287.986664, 246.277649]|1.423627|5.423149|8192 / uniform_reservoir_estimate|
|normal_terminal|—|1789|59.241242 [-550.041016, 154.018143]|-0.081048 [-0.709794, 0.000947]|-59.322290 [-154.115738, 549.331238]|69.336702|114.769846|1789 / exact_all_draws|
|failure_terminal|—|334|-2650.465859 [-3664.264893, -1561.145264]|-10134.840005 [-10135.117188, -10134.605469]|-7484.374170 [-8573.459961, -6470.852539]|7484.374170|8558.944873|334 / exact_all_draws|
|failure_prefix|—|1652378|-752.117827 [-3727.389648, 207.481964]|-752.186486 [-3721.655762, 206.109085]|-0.068659 [-325.900757, 197.757996]|4.706516|17.405472|8192 / uniform_reservoir_estimate|

统计范围：actual optimizer minibatch draws before Adam; count includes repeat draws。
本窗口失败终止实际 draw 次数：334；分 replay ID 的完整计数保留在本轮 JSON。
累计失败终止实际 draw 次数：4523；分 replay ID 的完整计数保留在本轮 JSON。
A streaming prefix remains normal_nonterminal until its voyage fails; historical draws retain their label at draw time.

### 当前 checkpoint 对只读 Replay 样本的拟合

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_06_replay_td_fit.csv.gz)

|经验类别|Replay类别总量|实际draw/诊断样本数|Q均值 [最小,最大]|TD目标均值 [最小,最大]|有符号残差均值 [最小,最大]|MAE|绝对误差p95|p95计数/方式|
|---|---:|---:|---|---|---|---:|---:|---|
|normal_nonterminal|913637|8192|-19.046300 [-2232.360596, 162.241394]|-18.879467 [-2251.225586, 162.636673]|0.166833 [-62.975754, 124.352417]|1.323450|4.987835|8192 / exact_on_saved_diagnostic_sample|
|normal_terminal|79|79|65.993411 [-144.695389, 158.170212]|-0.067476 [-0.403382, 0.000947]|-66.060887 [-158.267807, 144.292007]|69.892028|119.461106|79 / exact_on_saved_diagnostic_sample|
|failure_terminal|18|18|-2732.112745 [-3654.336670, -1575.521362]|-10134.834418 [-10135.117188, -10134.605469]|-7402.721653 [-8559.083984, -6480.780273]|7402.721653|8398.259668|18 / exact_on_saved_diagnostic_sample|
|failure_prefix|86266|8192|-732.581811 [-3627.980713, 174.845993]|-732.819700 [-3631.533691, 184.111816]|-0.237890 [-174.313843, 166.463135]|4.372845|16.946954|8192 / exact_on_saved_diagnostic_sample|

统计范围：read-only stratified replay diagnostic; measured_count is not the replay population or actual optimizer draws。
Replay 经验 1000000；本次只读测量 16481。

### 固定全 13 动作 Q 排名

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/q_probes/round_06.json)

```json
{
  "file": "q_probes/round_06.json",
  "count": 44
}
```

### 既有 Train 失败轨迹实际回报 / 当前 Q / 采样次数

[完整记录](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_06_train_failure_values.csv.gz)

```json
{
  "file": "round_06_train_failure_values.csv.gz",
  "prefix_file": "round_06_train_failure_prefixes.csv.gz",
  "recorded_prefix_rows": 2121,
  "probes_count": 606,
  "events": [
    {
      "boundary_side": "lower",
      "sample_id": "parent_015_seg01",
      "origin_round": 1,
      "terminal_replay_id": 63649,
      "actual_training_draws": 186,
      "terminal_q": -3640.686767578125,
      "terminal_target": -10135.0263671875,
      "terminal_signed_residual": -6494.339599609375,
      "terminal_mae": 6494.339599609375,
      "prefix_td_mae": 2.43796693410305,
      "full_q_rows": 303
    },
    {
      "boundary_side": "upper",
      "sample_id": "parent_010_seg01",
      "origin_round": 1,
      "terminal_replay_id": 66204,
      "actual_training_draws": 189,
      "terminal_q": -1213.5484619140625,
      "terminal_target": -10134.529296875,
      "terminal_signed_residual": -8920.980834960938,
      "terminal_mae": 8920.980834960938,
      "prefix_td_mae": 3.634439733644195,
      "full_q_rows": 303
    }
  ],
  "interpretation": "Only actual Train transitions; read-only checkpoint inference. Observed return is the realized epsilon-greedy trajectory return, not Q* or a counterfactual benefit. TD targets use current target-network masked max; they are moving learning targets. The first upper/lower failures are fixed for checkpoint comparisons; no report rows enter Replay."
}
```

Interpretation and control consistency:

- TD target 不是 Q*。训练时 Online/Target 参数变化会改变目标；拟合误差仅衡量对对应时刻 bootstrap 目标的拟合。
- Loss、TD 误差或单个状态 Q 值不能单独证明控制改善；同时检查完整航段完成数、边界失败和物理时间尺度的功率波动。
- 固定 Validation 探针仅用于报告，不加入 Replay、不训练；Test 不读取。
- 失败前缀成本只覆盖实际执行区间；完整航段子集成本不是固定全套成本。SOC/ramp/失败惩罚不计入经济账本。
- SOC 上/下边界计数与增量动作不可达计数是交叉分类，不可相加作为失败总数。
- 在线 draw 与 checkpoint 分层诊断样本的分布不同；加权 MAE 仅在同一统计口径内作轮间比较，并同时保留各类别误差。
- 分层诊断样本和 TD 目标均随轮次变化；weighted MAE 的变化仅作描述，不能解释为对 Q* 的拟合改善。
- 实际失败轨迹回报来自当时行为策略，不是最优 Q 的真值；未知终局的历史 draw 分类不会在航段失败后追改。
- 失败终止诊断样本平均 target−Q=-7402.721653；Q仍偏高，平均低估已观测失败代价。该判断针对已执行失败动作，不将此前行为轨迹回报当作Q*。
- 固定Train lower失败 parent_015_seg01 的终止经验累计实际抽样 186 次；终止 target−Q=-6494.339600，终止MAE=6494.339600。前缀与终止拟合及Validation需同时查看。
- 固定Train upper失败 parent_010_seg01 的终止经验累计实际抽样 189 次；终止 target−Q=-8920.980835，终止MAE=8920.980835。前缀与终止拟合及Validation需同时查看。

```json
{
  "validation_completed_delta": -2,
  "validation_boundary_failures_delta": 2,
  "validation_changes_per_hour_delta": 2821.0409475216343,
  "validation_reversals_per_hour_delta": 1935.6199120835408,
  "td_error_source": "checkpoint_replay_fit",
  "td_error_metric": "sample_weighted_mae",
  "td_error_delta": 1.8678955794326573
}
```

## Traversal 7: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_07_summary.md)

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

Interpretation and control consistency:

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

## Traversal 8: recorded evidence

[Control, cost and Q summary](C:/Users/20883/OneDrive/Desktop/lstm-dqn-mpc/lstm-dqn-mpc/.worktrees/v6-dqn-1s/outputs/v6_dqn_1s_seed42_20261025_run02/round_08_summary.md)

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

Interpretation and control consistency:

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

