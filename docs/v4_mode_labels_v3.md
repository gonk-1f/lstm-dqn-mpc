# v4 工况标签 v3

新标签：`data/processed/operating_dataset_zero_boundary_v2_modes_v3/`。旧模式侧车、30 s 主功率数据、AIS、样本时间戳和 Train/Validation/Test 划分均保留。新样本 CSV 只含时间轴、`mode`、`mode_reason` 和可选负载覆盖列；v4 加载器从经 SHA-256 校验的旧侧车读取 FC／电池功率。新 manifest 与 policy 绑定旧侧车 manifest，旧训练 checkpoint 因 manifest 不同不能续训。

规则：FC 正出力且供电净负载非负属于 ONBOARD 候选；FC 零出力而由电池供电或零负载的边界也可为 ONBOARD。SHORE 是**模型化补能机会**，要求 FC≤1 kW、电池功率<−1 kW、AIS≤0.1 kn、数据有效且至少连续三个 30 s 点。头两个点标为 `shore_pending`，不作 DQN 决策、不更新补能 SOC；第三点起标为 `shore_charging`，才确认模型化补能。持续不足三点的零 FC 充电、负净负载冲突及原始 FC 快照覆盖不足均为 UNKNOWN。没有实测岸电接入证明。

| 划分 | ONBOARD 点 | SHORE_PENDING 点 | SHORE_CHARGING 点 | UNKNOWN 点 |
|---|---:|---:|---:|---:|
| Train | 21427 | 10 | 1949 | 204 |
| Validation | 3524 | 10 | 1081 | 8 |
| Test 标签 | 2765 | 0 | 0 | 0 |

旧 Train 的 44 个 SHORE 区间、5142 点，现在分别为 ONBOARD 3009、模型化 SHORE 1959、UNKNOWN 174；没有 FC>1 kW 的点保留在 SHORE。新增 ONBOARD 行的现存负载均有效，无须覆盖主数据负载。

v4 遇 UNKNOWN 即在首个未知行前截断该**整条样本**；UNKNOWN 不是成功终止，也不是物理失败，不产生新动作、虚构补能或供电失败惩罚。UNKNOWN 之后不跨接 SOC 或 Bellman 回报。正式入口默认 `--unknown-replay-mode observed_nstep`：只将前缀中完整8步奖励且真实后继决策已观测的窗口，以 `done=False`、结果 `unresolved` 写入经济 replay；接触数据缺口的尾窗舍弃。`discard` 可复现先前舍弃未闭合前缀的规则。已确认模型SHORE前后的回报仍连续。

**9 个 Train 与 1 个 Validation 样本含 UNKNOWN**。固定完整可评价集合由标签自动识别，目前为21/7；每轮Greedy Validation独立于Train完成数执行。最佳模型要求固定完整Train与Validation样本全部完成，同时含UNKNOWN样本已观测前缀无物理失败；不将截断成本纳入完整成本比较。前缀进入Replay不改变评价资格。详见[前缀回放修复说明](v4_observed_prefix_replay_fix_2026-10-12.md)。
