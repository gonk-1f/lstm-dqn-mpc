# Project Status

## Repository cleanup

- 删除 6 个已跟踪 `.idea` 文件和危险的 `c-d.cmd`。
- 删除 11 个高置信失效源码文件：旧 direct-control DQN 训练/评估链、空壳实验文件、未引用且导入即访问外网的 `SolcastAPI.py`、以及仅服务于已删除旧环境的 `performance.py`。
- 删除旧 `outputs/dqn_mpc_mlp_10k_baseline/` 中 15 个混合 v1/v2/v3、hard-voyage 和临时诊断产物。
- 将 v2/v3 的当前结论压缩为 `outputs/action_space/v2_summary.json` 和 `v3_summary.json`；原始历史仍可由 Git 追溯。
- 新增统一的 `src/main/evaluate_dqn_mpc_action_space.py`，替代重复的 state probe、固定动作覆盖和失败汇总逻辑。
- `.gitignore` 仅增加 scratch/debug/tmp/非最终 checkpoint/IDE 缓存规则；正式 summary、最终 checkpoint 和必要产物仍可跟踪。
- 不以“当前不可达”作为唯一删除依据；用途不明确的历史研究模块继续保留。

## 正式工作流

- 数据划分：train/validation/test = 46/13/7；test 固定为 `voyage_060`–`voyage_066`。
- 固定基准：`src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline`。
- MPC：`dt=1 s`、`N=6`、未来 `t+1..t+6` offline-oracle 预览、每次只执行第一步。
- Candidate C：`(q_h2, q_batt, q_soc, q_fc_var)=(0.25, 0.40, 12, 20)`。
- DQN：11 维状态、完整四权重离散动作、固定四项评价奖励、持久化 OSQP solver bank。
- Solver failure 显式记录，无静默 fallback；OSQP 明确保留非 solved 状态对象。

## v2

- 固定动作覆盖：350/413 成功，63 个 primal infeasible，0 个 max-iter。
- A6 在 52 个可完成航段中最佳 50 次，正常 SOC 代表状态的首步 FC 平均总跨度仅 `0.323 kW`。
- 结论：数值不同但控制行为塌缩，拒绝。

## v3

- 固定动作覆盖：320/413 成功，79 个 primal infeasible，14 个 max-iter。
- A5 最佳 37 次、A4 最佳 14 次，单动作全面支配减弱。
- A2/A3 首步 FC 平均差仅 `0.096 kW`，仍局部冗余。
- 8 个 train/validation 航段全动作失败，A6 集中出现 14 次 max-iter。
- 结论：区分度改善，但覆盖、冗余和数值稳定性仍未通过。

## 功能型五动作候选

| 动作 | 权重 | 角色 |
| --- | --- | --- |
| A0 | `(0.25, 0.40, 12, 20)` | Nominal / Candidate C |
| A1 | `(0.40, 0.25, 8, 8)` | Hydrogen Economy |
| A2 | `(0.25, 0.45, 36, 15)` | SOC Regulation |
| A3 | `(0.15, 0.80, 12, 1)` | Fast FC Response |
| A4 | `(0.25, 0.10, 12, 40)` | FC Smoothing |

- 第一版 A4 为 `(0.20, 0.25, 12, 30)`，局部探针显示它与 A0 过近。
- 只进行一次受控修正，得到上表 A4；其余四个候选不变。
- 五个角色已覆盖当前识别出的控制轴，没有证据支持增加第六动作。

## Acceptance

- 代表状态：9 个唯一真实负荷窗口 × 3 个 SOC 区间 × 3 个上一时刻 FC 区间，共 81 个状态。
- test access audit：`voyage_060`–`voyage_066` 轨迹访问为空。
- 状态探针：405/405 solved；奖励赢家 A0/A1/A4 分别为 24/3/54，最高占比 `66.7%`。
- A0/A4 的首步 FC 平均/最大差为 `0.389/0.953 kW`，horizon FC RMS 平均差 `0.903 kW`，仍被判为行为冗余。
- 全航段覆盖：59×5=295 组；240 solved、54 primal infeasible、1 maximum-iterations。
- A0–A4 完成数：48、48、50、51、43。
- A4 的唯一 max-iter 位于 `voyage_044`、decision index 4060；A4 覆盖低于 A0。
- 全动作失败航段：`voyage_016`、`021`、`024`、`041`、`044`、`045`、`053`、`054`。
- 本轮决定：`FAIL`。
- 失败原因：A0/A4 冗余、A4 max-iter、A4 低于 Candidate C 覆盖、8 个全动作失败航段。
- 未冻结新动作空间；`src/dqn/utils/action_mapper.py` 仍保留 v3，未把失败候选写入生产映射。

详细状态探针、覆盖和验收理由见：

- `outputs/action_space/final_state_probes.csv`
- `outputs/action_space/final_coverage.csv`
- `outputs/action_space/final_summary.json`

## Training

- 正式 MLP-DQN：未开始。
- 正式 checkpoint：无。
- KAN/SineKAN/LSTM：本轮未接入。
- 同航段工况—动作分析：未执行，因为动作空间验收未通过。

## Limitations

- 1 s 负荷来自 30 s 数据的离线三次样条重构，不是在线实测预测。
- 当前没有通过验收的 `FINAL ACTION SPACE`。
- 五动作候选仍存在固定航段覆盖和 solver 稳定性问题；最终数值以 `final_summary.json` 为准。
- 当前依赖未锁定，`pytest` 在现有环境不可用。
- 部分历史研究模块尚未完成独立用途审计，因此没有凭不可达性批量删除。
