# 基于凸 QP-MPC 与 DQN 权重选择的船舶能量管理

## 项目定位

本项目研究燃料电池/锂电池混合动力船舶的上层能量管理。固定基准使用 `N=6`、`dt=1 s` 的凸 QP-MPC；DQN 的目标接口是在每个决策时刻选择一套完整 MPC 四权重，而不是直接输出燃料电池或电池功率。

当前 DQN-MPC 仍属于 offline-oracle / ideal-foresight benchmark：状态和 MPC 均使用同航段未来 `t+1..t+6` 的真实 1 s 样条负荷。它不是在线 LSTM-DQN-MPC，也不能把样条 1 s 数据解释成真实在线 1 s 实测数据。

```text
11 维状态
  -> DQN 选择 (q_h2, q_batt, q_soc, q_fc_var)
  -> 对应的持久化 OSQP solver
  -> MPC 执行第一步 P_fc / P_batt
  -> 更新 SOC 和下一状态
```

## 当前结论

- 固定 Candidate C 基准保持 `(0.25, 0.40, 12.0, 20.0)`。
- 当前源码中的 7 动作表是 v3 历史开发版本，尚未通过正式训练前验收。
- 本轮按控制功能构造并验证了 5 动作候选；它完成了状态探针和 59 个 train/validation 航段的固定动作覆盖，但验收结果为 `FAIL`。
- 因此没有冻结新的 `FINAL ACTION SPACE`，没有修改生产动作表，也没有启动正式 MLP-DQN 训练。
- `voyage_060`–`voyage_066` 的轨迹未用于动作设计、修正、覆盖、训练或验证。

## 数据与划分

| 数据 | 路径 | 用途与边界 |
| --- | --- | --- |
| 30 s 原始实船航段 | `total_load_excels/` | 原始来源，不修改 |
| 正式 1 s 数据 | `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` | 由各设备通道离线三次样条重构 |
| 1 ms / 10 ms 数据 | `data/millisecond_1ms/`、`data/millisecond_10ms/` | 独立台架数据线，不是当前 MPC 默认输入 |
| 活动航段划分 | `outputs/config/voyage_split_total_load_721.json` | chronological 46/13/7 |

活动划分为：

- train：`voyage_001`–`voyage_046`
- validation：`voyage_047`–`voyage_059`
- test：`voyage_060`–`voyage_066`

详细数据来源、字段和限制见 `docs/DATA_PROVENANCE.md`。

## 正式 MPC

固定基准入口：

```powershell
python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline
```

主要物理配置：

| 项目 | 数值 |
| --- | --- |
| 采样与时域 | `dt=1 s`，`N=6` |
| 燃料电池 | `0..560 kW`，硬爬坡 `48 kW/s` |
| 电池 | `624 kWh`，充电下限 `-624 kW`，放电上限 `+1248 kW` |
| SOC | 参考 `0.55`，硬边界 `[0.2, 0.8]`，归一化带宽 `0.05` |
| 执行 | 使用 `t+1..t+6` 预览，航段尾 edge-hold，每轮只执行第一步 |

四目标为：

```text
J = q_h2     * Σ m_H2(P_fc) / m_H2(560 kW)
  + q_batt   * Σ (P_batt / 624 kW)^2
  + q_soc    * Σ ((SOC - 0.55) / 0.05)^2
  + q_fc_var * Σ (ΔP_fc / 48 kW)^2
```

QP 定义、正式数值缩放和 OSQP 运行层分别位于：

- `src/main/mpc_solvers/mpc_qp_formulation.py`
- `src/main/mpc_solvers/n6_qp_scaling.py`
- `src/main/mpc_solvers/osqp_runtime.py`

## DQN 状态、动作和奖励

11 维状态顺序固定为：

```text
SOC
上一时刻 P_fc
上一时刻 P_batt
当前负荷
当前负荷变化
未来 6 步负荷
```

归一化后不做 clip。实现位于 `src/dqn/utils/state_builder.py`。

固定评价奖励不随所选动作改变，只包含：

```text
0.25 * H2
0.40 * battery power square
12.0 * SOC tracking
20.0 * FC variation
```

没有 SOC guard、终端 SOC、动作切换惩罚、额外约束惩罚或启发式动作规则。实现位于 `src/dqn/utils/reward.py`。

当前生产映射仍是未通过验收的 v3：

| 动作 | 名称 | `(q_h2, q_batt, q_soc, q_fc_var)` |
| --- | --- | --- |
| A0 | candidate_C | `(0.25, 0.40, 12, 20)` |
| A1 | hydrogen_economy | `(0.60, 0.15, 4, 2)` |
| A2 | balanced | `(0.25, 0.50, 20, 12)` |
| A3 | soc_maintenance | `(0.20, 0.45, 28, 18)` |
| A4 | strong_soc_recovery | `(0.30, 0.45, 50, 18)` |
| A5 | fast_fc_response | `(0.15, 0.80, 12, 1)` |
| A6 | fc_smoothing | `(0.15, 0.15, 8, 50)` |

## 功能型五动作候选

本轮验证的候选没有写入 `action_mapper.py`，因为它未通过冻结门槛：

| 候选动作 | 权重 | 控制角色 |
| --- | --- | --- |
| A0 | `(0.25, 0.40, 12, 20)` | Nominal / Candidate C |
| A1 | `(0.40, 0.25, 8, 8)` | Hydrogen Economy |
| A2 | `(0.25, 0.45, 36, 15)` | 双向 SOC Regulation |
| A3 | `(0.15, 0.80, 12, 1)` | Fast FC Response |
| A4 | `(0.25, 0.10, 12, 40)` | FC Smoothing / Battery Buffer |

只进行了一次受控修正：A4 从 `(0.20, 0.25, 12, 30)` 改为上表数值，以增加动态轴上的分离。没有增加第六动作，也没有进行网格、随机或贝叶斯搜索。

## 动作空间评估

统一评估入口：

```powershell
python src/main/evaluate_dqn_mpc_action_space.py `
  --actions-json <candidate-actions.json> `
  --output-dir outputs/action_space `
  --prefix final
```

评估器只允许读取 train/validation，且在轨迹 I/O 前锁定拒绝 `voyage_060`–`voyage_066`。它统一生成：

- 81 个代表状态 × 全部候选动作的状态探针；
- train/validation 全航段固定动作覆盖；
- solver 状态分类；
- 物理指标、奖励及四项奖励分解；
- 动作差异、状态条件赢家和验收结论。

验收失败会阻止动作冻结和正式训练，不存在静默回退到 A0。

本轮结果：

- 状态探针 405/405 solved；奖励赢家为 A0 24、A1 3、A4 54，最高占比 `66.7%`，没有触发 90% 全面支配门槛。
- A0/A4 的首步 FC 最大差仅 `0.953 kW`，horizon FC RMS 平均差 `0.903 kW`，仍触发行为冗余门槛。
- 固定动作覆盖 240/295 完成，另有 54 个 primal infeasible 和 1 个 maximum-iterations。
- A0–A4 的完成航段数分别为 48、48、50、51、43；A4 低于 Candidate C。
- 8 个航段全动作失败：`voyage_016`、`021`、`024`、`041`、`044`、`045`、`053`、`054`。
- 最终验收为 `FAIL`，因此停止在训练门之前。

## 保留输出

| 路径 | 内容 |
| --- | --- |
| `outputs/mpc_1s_n6_candidate_C/` | 固定 Candidate C 的 7 个测试航段基准 |
| `outputs/action_space/v2_summary.json` | v2 塌缩的最小历史证据 |
| `outputs/action_space/v3_summary.json` | v3 改善但未通过的最小历史证据 |
| `outputs/action_space/final_state_probes.csv` | 本轮五动作状态探针 |
| `outputs/action_space/final_coverage.csv` | 本轮 59×5 固定动作覆盖 |
| `outputs/action_space/final_summary.json` | 本轮验收、数据访问审计和追溯信息 |

旧 `outputs/dqn_mpc_mlp_10k_baseline/` 混合目录中的 v1/v2/v3、hard-voyage 和临时诊断文件已由上述最小证据替代。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `src/main/` | 数据构建、固定 MPC、动作空间评估、MLP-DQN 训练入口 |
| `src/main/mpc_solvers/` | QP、缩放、OSQP 和动作 solver bank |
| `src/envs/dqn_mpc_weight_env.py` | 单航段 DQN-MPC 闭环环境 |
| `src/dqn/` | 动作、状态、奖励、MLP agent、回放和探索策略 |
| `tests/` | 数据、MPC 和 DQN-MPC 回归测试 |
| `outputs/` | 正式数据、基准和最小实验凭据 |

历史 direct-control DQN 入口、空壳实验文件、危险外部 API 脚本、IDE 文件及旧重复动作实验产物已清理。仅凭“当前入口不可达”但用途仍不明确的研究模块没有被猜测删除。

## 验证

项目当前以 `unittest` 为可用测试入口：

```powershell
python -m compileall -q src
python -m unittest discover -s tests -p "test_*.py"
git diff --check
```

当前环境未安装 `pytest`，因此没有为本轮工作临时安装或修改依赖。

## 训练边界

`src/main/train_dqn_mpc_mlp.py` 保留为正式 MLP-DQN 训练/验证入口，但本轮验收为 `FAIL`，所以没有启动训练、没有生成正式 checkpoint，也没有开展同航段动作时序分析。KAN、SineKAN 和 LSTM 均未进入本轮流程。

## 已知限制

- 1 s 样条数据是离线重构，依赖未来 30 s 节点，不具备在线因果性。
- Candidate C 是固定基准，不代表全局最优。
- 当前没有通过验收的最终 DQN 动作空间，也没有正式收敛模型。
- 动作空间仍存在覆盖/数值稳定性问题；详见 `outputs/action_space/final_summary.json`。
- 依赖未锁定，部分历史 manifest 含旧绝对路径。
- `SineKAN-main/` 的第三方许可兼容性仍需在发布前核验。
