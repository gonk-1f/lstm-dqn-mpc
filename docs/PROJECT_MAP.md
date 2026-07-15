# Project Map

## Status labels

- **active**：当前目标主线直接需要，且代码/数据入口可定位。
- **auxiliary**：用于诊断、对照或迁移，不是论文主链路默认入口。
- **historical**：保留研究过程和可追溯性，不代表当前参数或结论。
- **deprecated**：已明确不应继续使用。
- **uncertain**：用途、来源或验收状态仍需核验。

“active”只表示与当前方向相关，不等于模块已经通过端到端验收。

## Top-level structure

```text
.
|-- src/
|   |-- forecasting/          # LSTM、特征、scaler、窗口和指标
|   |-- main/                 # 数据、训练、MPC、benchmark CLI
|   |   `-- mpc_solvers/      # 当前凸 QP 形式
|   |-- mpc/                  # 历史 CasADi/IPOPT 与物理组件
|   |-- dqn/                  # agent、replay、policy、reward、Q 网络
|   `-- envs/                 # 多个历史 DQN/船舶环境
|-- configs/                  # 旧通用配置，部分参数已失效
|-- data/                     # 燃料电池曲线、毫秒/10 ms 数据等
|-- total_load_excels/        # 66 个原始 30 s 航段
|-- outputs/                  # 数据构建、模型、报告、benchmark 产物
|-- tests/                    # unittest 测试
|-- docs/                     # 设计、协议和接管审计文档
|-- SineKAN-main/             # 第三方 SineKAN 源码/notebook 副本
|-- reports/                  # 较早的 LSTM 报告
`-- tmp/, .codex_tmp/         # 已跟踪/本地临时产物，待后续清理
```

## Main entrypoints

| 路径 | 作用 | 状态 | 备注 |
| --- | --- | --- | --- |
| `src/main/build_total_load_dataset_721.py` | 读取 66 个 30 s Excel，检查负荷恒等式并生成 46/13/7 数据 | active | 默认不跨航次；split/scaler 规则写入 manifest |
| `src/main/run_train_lstm_total_load_721.py` | 66 航段 30 s direct multi-output LSTM 训练/测试 | active | history 18、prediction 6、30 s；是原始实测路线 |
| `src/main/run_train_lstm_721.py` | 较早的 35 航段/7-2-1 LSTM 流程 | historical | 保留的 delta10 checkpoint 属于此路线，不等同 66 航段流程 |
| `src/main/build_spline_1s_diagnostics.py` | 逐航次生成 natural/not-a-knot 1 s spline 并审计 | auxiliary | natural-clipped 被后续离线实验使用；非因果 |
| `src/main/run_lstm_spline_1s_hparam_search.py` | 1 s direct multi-output LSTM 与 baseline 对比 | auxiliary | 当前 Task C 未超过简单 baseline |
| `src/main/build_mpc_solver_benchmark_1s_data.py` | 构建 7 个 test 航次的 1 s benchmark 输入 | auxiliary | 入口来自离线 spline |
| `src/main/benchmark_mpc_qp_osqp_1s.py` | OSQP `N=60` 求解器/控制 benchmark | historical | 只保留 solver/performance 证据；不再作为默认配置或继续权重搜索 |
| `src/main/run_mpc_1s_n6_weight_selection.py` | `N=6` ideal-foresight 固定权重实验 | active/auxiliary | `t+1..t+6`、只执行第一步、实际 SOC 更新；A–D 已运行但无候选通过，不是 LSTM 闭环 |
| `src/main/run_mpc_1s_n6_qsoc_feasibility.py` | `N=6` 的 `q_soc`-only 结构诊断 | active/auxiliary | 固定其他权重/结构；同代输入/实现/运行时指纹防止产物混用；`q_soc=20` 为唯一可行性见证；不自动创建正式配置或启动 DQN |
| `src/main/run_mpc_1s_n6_soc_clamping_diagnostic.py` | `N=6` 近参考 SOC 合成诊断 | active/auxiliary | 只比较 q10/q20 的 6 个恒载和 2 个脉冲工况；结论为 `no_evidence_of_SOC_clamping`，不选择正式权重 |
| `src/main/run_lstm_mpc_test.py` | 30 s CasADi/IPOPT LSTM-MPC | historical | 有预测与实际反馈逻辑，但物理参数/求解器不是目标主线 |
| `src/main/run_lstm_mpc_total_load_test.py` | 上述 30 s 流程的总负荷包装入口 | auxiliary | 可作为迁移参考 |
| `src/main/run_lstm_mpc_weight_sweep.py` | 历史固定权重搜索 | historical | 不应继续扩展为无约束大搜索 |
| `src/main/build_millisecond_10ms_dataset.py` | 1 ms 原始数据直接抽点到 10 ms | auxiliary | 默认源路径含本地绝对路径，CLI 可覆盖 |
| `src/main/audit_millisecond_10ms_dataset.py` | 10 ms manifest/数据审计 | auxiliary | 严格校验 assignment key、逐序列窗口、时间步和来源行 |
| `src/main/run_lstm_millisecond_10ms_search.py` | 高频 direct multi-output LSTM 与 baseline | auxiliary | 未发现正式 checkpoint/结果目录 |
| `src/main/run_train_dqn.py` | SineKAN Double-DQN 历史训练入口 | historical | 选择 `q_soc/q_ramp`，使用 1806 kWh、30 s CasADi 环境；不符合目标接口 |
| `src/main/run_test_dqn.py` | 历史 DQN 测试入口 | historical | 无目标环境 checkpoint 可验收 |
| `src/main/_deprecated_*linear_interp*` | 1 s 线性插值旧路径 | deprecated | 文件名和测试均明确标记 DO_NOT_USE |

## LSTM modules

| 路径/产物 | 类型 | 状态 | 审计结论 |
| --- | --- | --- | --- |
| `src/forecasting/lstm_load_predictor.py` | 通用 PyTorch LSTM、scaler、checkpoint、horizon 指标 | active | head 一次输出完整预测向量，是 direct multi-output，不是 recursive |
| `src/forecasting/feature_pipeline.py` | 训练/验证/测试特征与窗口 | active | 应持续强制按 voyage 分组和 train-only scaler |
| `src/forecasting/ensemble_lstm_provider.py` | 多模型预测 provider | uncertain | 不是当前目标主链路入口，需在闭环统一时决定是否保留 |
| `outputs/lstm_total_load_721/` | 66 航段 30 s 模型、指标、checkpoint | active | 原始实测预测证据；默认 history 18、prediction 6 |
| `outputs/lstm_721/` | 较早 35 航段 retained delta10 模型 | historical | 可复现边界与当前 66 航段线不同，不能混报 |
| `outputs/lstm_spline_1s_hparam_search/` | 1 s Task C 与 baseline | auxiliary | history 30、prediction 6；LSTM 未超过 last-slope |
| `src/forecasting/millisecond_multistep_lstm.py` | 10 ms LSTM、train-only scaler、按序列窗口、baseline | auxiliary | current-hold、last-slope、local linear trend 均已实现 |
| `src/main/run_spline_1s_lstm_diagnostic.py` | 旧 spline 可预测性诊断 | historical | 与后来的 fixed Task C 区分 |

没有发现当前主线使用 recursive 多步预测的正式入口；三个主要 LSTM 路线均以 direct vector 输出为核心。

## MPC and OSQP modules

| 路径/功能 | 状态 | 审计结论 |
| --- | --- | --- |
| `src/main/mpc_solvers/mpc_qp_formulation.py` | active | 构造线性约束、SOC 动力学、归一化凸二次目标；默认参数包含 693 kWh/346.5 kW/560 kW/48 kW/s |
| `src/main/benchmark_mpc_qp_osqp_1s.py` 内的持久 OSQP workspace | historical/supporting | N=60 原始设置为 `eps_abs=eps_rel=1e-4`、`max_iter=4000`；N=6 runner 只复用边界/计时等支持函数 |
| `src/main/run_mpc_1s_n6_weight_selection.py` 的 N=6 workspace | active/auxiliary | 严格等价仿射缩放，`eps_abs=eps_rel=1e-5`、`max_iter=10000`、固定 rho 更新间隔；max_iter 可冷重启同一 QP，不构造控制 fallback |
| `src/main/run_mpc_1s_n6_qsoc_feasibility.py` | active/auxiliary | 通过显式配置复用同一 N=6 workspace，校验候选 config 指纹和有限残差；仅测试 `q_soc={5,10,20}` |
| 独立可部署 OSQP controller wrapper | uncertain/missing | 当前 N=6 入口是离线实验 runner；尚未封装预测 provider 和确定性控制失败回退 |
| `outputs/mpc_1s_n6_weight_selection/` | active evidence | A–D 配置、逐航段/总体指标、solver 统计、约束审计、曲线与人工拒绝决策；无 provisional 配置 |
| `outputs/mpc_1s_n6_qsoc_feasibility/` | active evidence | 三个 `q_soc` 配置、逐航段/总体指标、solver 统计、约束审计、曲线和结构诊断；`QSOC_20` 为 witness，非 accepted 配置 |
| `outputs/mpc_1s_n6_soc_clamping_diagnostic/` | active evidence | 8 个 synthetic ideal-foresight 工况的轨迹、指标和 5 组图；不是实船/LSTM/DQN 结果 |
| `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` | historical | 693 kWh `N=60` 离线 benchmark 证据；紧凑指针在 N=6 输出目录中 |
| `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt277p2_simplified_spec_norm/` | historical | 旧容量研究；不得作为当前物理配置 |
| `src/mpc/` 与 `src/main/run_lstm_mpc_test.py` | historical/supporting | CasADi/IPOPT、氢耗曲线、回退和历史 SOC 设计；需择取可迁移逻辑，不应直接当 OSQP 主线 |
| `outputs/lstm_mpc_total_load_test_fixed_baseline_v1/` | historical/supporting | 30 s 固定基线产物，使用不同参数体系 |
| `configs/mpc.yaml`, `configs/ship_system.yaml` | historical | 含 horizon 18、SOC 0.65、1806 kWh/350 kW 等旧参数 |

QP 中 `P_fc + P_batt = load`，SOC 由实际电池功率更新，FC ramp 是硬约束。简化 objective 保留氢耗/SOC/电池使用项；`q_ramp=0` 和 `q_terminal_soc=0` 时不含相应软目标。N=6 离线实验在最终求解失败处终止航段并标记不完整；尚无可部署的控制回退动作。

## DQN and Q-network modules

| 路径 | 状态 | 审计结论 |
| --- | --- | --- |
| `src/dqn/agents/dqn_agent.py` | active component | replay、target network、可选 Double DQN 和优化逻辑存在 |
| `src/dqn/memory/replay_buffer.py` | active component | 经验回放实现存在 |
| `src/dqn/policies/epsilon_greedy.py` | active component | epsilon 策略实现存在 |
| `src/dqn/utils/action_mapper.py` | historical/incompatible | 权重动作包含 `q_fc/q_ramp` 等并要求 18 步预测；不符合目标三权重 `N=6` 接口 |
| `src/dqn/utils/reward.py` | auxiliary | 物理归一化奖励不是 MPC objective 的直接复用，但 SOC placeholder 和目标环境接口仍需冻结 |
| `src/envs/ship_env_simple.py` | historical | 直接功率动作，1067 kWh/350 kW；不是目标设计 |
| `src/envs/ship_env_dual_side.py` | historical | 左右侧功率/trim 动作，1806 kWh/350 kW；不是目标设计 |
| `src/main/run_train_dqn.py` | historical/incompatible | SineKAN、Double DQN 训练流程完整，但动作选 `q_soc/q_ramp`，使用旧 30 s CasADi 环境 |
| `src/dqn/networks/mlp_qnet.py` | active candidate | 可作为公平基线，但尚无目标环境结果 |
| `src/dqn/networks/sine_kan_qnet.py` | active candidate | 实际导入第三方 SineKAN，支持 normalizer/embedding/dueling；尚无目标环境结果 |
| `src/dqn/networks/kan_qnet.py` | auxiliary | 依赖外部 `pykan`；不是占位，但尚未在目标环境验证 |
| `src/dqn/networks/kan_v2_qnet.py` | auxiliary | 本地紧凑 KAN-style 实现；尚未在目标环境验证 |
| `src/dqn/networks/factory.py` | active component | 支持 MLP、KAN、KAN-v2、SineKAN 工厂选择 |
| `SineKAN-main/` | third-party/uncertain license | 完整源码和 8 个 notebook；无 LICENSE/包元数据，发布前必须核验 |

未发现 `outputs/checkpoints/dqn/` 或目标环境下的正式 DQN checkpoint/评价产物。现有测试目录也没有 DQN 专项单元测试。

## Data processing and provenance

| 路径 | 状态 | 内容 |
| --- | --- | --- |
| `total_load_excels/` | active, must keep | 66 个 30 s 原始实船 Excel |
| `outputs/total_load_dataset_build/` | active | 66 航段合并数据和构建输出 |
| `outputs/config/voyage_split_total_load_721.json` | active | 46/13/7 权威航次映射 |
| `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` | auxiliary | 66 个逐航次 1 s natural-clipped CSV 和 manifest |
| `outputs/mpc_solver_benchmark_1s/data/` | auxiliary | 7 个 test 航次的 spline benchmark 输入与数据检查 |
| `data/millisecond_1ms/` | auxiliary, must keep | 两个项目内 1 ms 原始工作簿和 source manifest |
| `data/millisecond_10ms/` | auxiliary | 直接抽点后的 32,000 行数据 |
| `outputs/config/millisecond_10ms_split_721.json` | auxiliary | 19 个原子序列的 10 ms 划分与 hash |
| `data/fuel_cell/FC_Dp0_curve_for_Python.csv` | active/supporting | Dp=0 燃料电池相对负载到氢耗映射 |

详见 `docs/DATA_PROVENANCE.md`。

## Tests

| 测试组 | 覆盖 | 状态 |
| --- | --- | --- |
| `tests/test_total_load_dataset_721.py` | 30 s 数据构建、划分、边界 | active |
| `tests/test_spline_1s_diagnostics.py` | spline 构建和物理标记 | active |
| `tests/test_lstm_spline_1s_hparam_search.py` | 1 s LSTM/baseline 逻辑 | auxiliary |
| `tests/test_mpc_solver_benchmark_1s.py` | QP/OSQP benchmark | active |
| `tests/test_mpc_1s_n6_qsoc_feasibility.py` | 新候选冻结、显式配置注入、门禁、config 指纹、产物失效 | active |
| `tests/test_lstm_mpc_*` | 30 s CasADi 时序、horizon、zero-delay | historical/supporting |
| `tests/test_millisecond_*` | 10 ms 构建、审计、模型 | auxiliary；split key 缺失/额外场景已有回归覆盖 |
| `tests/test_mpc_*` | 初始 dispatch、ramp 开关、SOC reference | supporting |
| `tests/_deprecated_*linear_interp*` | 旧线性插值 | deprecated |
| DQN 专项测试 | missing | 目标状态/动作/奖励/网络 parity 尚无测试 |

## Configuration hierarchy

1. `STATUS.md`：当前科学事实和阻塞项的唯一入口。
2. 数据 manifest：`outputs/config/voyage_split_total_load_721.json` 与 `outputs/config/millisecond_10ms_split_721.json`。
3. 运行时 CLI 参数和随输出保存的 `run_config.json`/`solver_config.json`：具体实验事实。
4. `configs/*.yaml` 与 `outputs/config/mpc_weight_sets.json`：历史配置集合；使用前必须核对容量、SOC、时域和求解器，不能默认视为 active。
5. `thread.md`、`project_status.md`、`next_steps.md`：历史上下文，不是当前唯一事实源。

## Historical experiment areas

| 类别 | 位置示例 | 状态 | 与论文关系 |
| --- | --- | --- | --- |
| 277.2 kWh OSQP | `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt277p2_simplified_spec_norm/` | historical | diagnostic/supporting |
| 693 kWh OSQP | `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` | active benchmark | supporting, not final baseline |
| 30 s CasADi fixed baseline | `outputs/lstm_mpc_total_load_test_fixed_baseline_v1/` | historical | supporting |
| 1 s raw/natural/not-a-knot/clipped spline | `outputs/spline_1s_diagnostics/` | auxiliary/historical mix | diagnostic |
| 1 s LSTM hparam | `outputs/lstm_spline_1s_hparam_search/` | auxiliary | diagnostic; negative comparison result |
| 线性插值 1 s | `_deprecated_*linear_interp*` 脚本与测试 | deprecated | not acceptable for core claims |
| 毫秒/10 ms | `data/millisecond_*`, corresponding scripts/tests | auxiliary | prediction latency/accuracy only |
| `raw_weight_retune`, `weight_sensitivity`, terminal/ramp/reserve/reference 实验 | 历史脚本、报告和状态文件中的记录 | historical/uncertain | diagnostic; exact retained artifact set must be inventoried before deletion |

历史内容的保留/归档建议见 `docs/CLEANUP_CANDIDATES.md`。
