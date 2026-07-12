# Unfinished Tasks

## Prioritization rule

- **P0**：阻止目标 LSTM-QP-MPC 闭环成立的问题。
- **P1**：在任何正式 DQN 训练前必须完成的问题。
- **P2**：形成论文证据所需的实验。
- **P3**：工程、复现和辅助研究整理。

任务按依赖排序。固定 `N=6` MPC 未通过验收前，不应启动正式 DQN 训练；1 s 离线 spline 不得替代真实在线因果数据结论。

## P0 — Close the fixed LSTM-QP-MPC loop

### P0-1 Freeze the formal timing and data contract

- **task:** 明确正式 1 s 实验的数据来源、decision time、LSTM history、h1–h6 预测含义、MPC stage 0–5 映射、实际负荷反馈和 SOC 更新时间。
- **reason:** 当前 `N=60` benchmark 直接使用未来 spline 行，30 s CasADi 路径使用另一套时序；离线 spline 又依赖未来端点。若不先冻结因果边界，后续闭环和论文表述都不可验证。
- **affected_files:** `src/main/run_lstm_mpc_test.py`, `src/main/benchmark_mpc_qp_osqp_1s.py`, `src/main/build_mpc_solver_benchmark_1s_data.py`, `docs/DATA_PROVENANCE.md`, future formal config/spec.
- **acceptance_criteria:** 一份版本化 contract 明确每个向量元素对应的物理时间；测试可检测 off-by-one、zero-delay 和未来数据泄露；正式数据被标记为 measured/causal 或 offline/reconstructed；论文主实验不把未来 spline 端点写成在线可用信息。
- **dependencies:** none.

### P0-2 Implement one reusable `N=6` OSQP closed-loop entry

- **task:** 将凸 QP 和 persistent OSQP workspace 封装为正式控制器，接收 6 步 LSTM 预测、当前 SOC/FC 状态，输出第一步功率并滚动更新。
- **reason:** 现有 QP/OSQP 只在 `N=60` benchmark 中闭合；30 s LSTM-MPC 使用 CasADi/IPOPT，不能证明目标框架已完成。
- **affected_files:** `src/main/mpc_solvers/mpc_qp_formulation.py`, `src/main/benchmark_mpc_qp_osqp_1s.py`, new/selected controller module under `src/`, new CLI under `src/main/`, relevant tests.
- **acceptance_criteria:** horizon 固定为 6；固定稀疏结构、参数更新和 warm start 被保留；每步执行的 `P_fc/P_batt` 满足功率平衡、设备边界、SOC 边界和 48 kW/s ramp；SOC 用实际施加的 `P_batt` 更新；逐步记录 status、iterations、residual、solve time 和约束残差；具有经过测试的确定性失败回退，不用 NaN 继续控制。
- **dependencies:** P0-1.

### P0-3 Validate and accept one fixed-weight baseline

- **task:** 在正式 7 个 test 航次上验收 `693 kWh / 346.5 kW`、`N=6` 固定 QP-MPC；决定保留、调整或拒绝暂定 `q_h2=0.5, q_soc=2.0, q_batt=0.05`。
- **reason:** 当前权重来自 `N=60` 离线 benchmark，结果仍有数值约束验收问题，不能直接成为论文基线或 DQN 动作中心。
- **affected_files:** formal MPC config, fixed-baseline runner, output report schema, `STATUS.md`, MPC tests.
- **acceptance_criteria:** 实验前注册 hard-constraint tolerance、求解成功率、fallback、SOC、氢耗、电池吞吐和 1 s 实时性门槛；逐航次和聚合结果都保存；所有 hard constraints 在容差内；对任何失败航次给出可复现 case；形成明确的 accepted/rejected 决策，而不是只按加权总分选择。
- **dependencies:** P0-2.

### P0-4 Remove stale parameters from the active execution path

- **task:** 为目标入口建立唯一配置源，阻止 277.2/1067/1806 kWh、138.6/350 kW、SOC 0.65、horizon 18 等历史值进入正式闭环。
- **reason:** `configs/`, `src/envs/` 和历史脚本存在多套互不兼容参数；仅更新文档无法避免误运行。
- **affected_files:** `configs/mpc.yaml`, `configs/ship_system.yaml`, `configs/dqn.yaml`, formal controller/env config, configuration validation tests.
- **acceptance_criteria:** 正式 CLI 保存解析后的完整配置；启动时校验容量、功率、SOC、采样间隔和 horizon；目标测试断言 `560/693/346.5/0.55/0.2/0.8/N=6`；历史入口显式标记 historical 或要求单独配置，不静默回退到旧值。
- **dependencies:** P0-1, P0-2.

## P1 — Freeze the DQN problem before training

### P1-1 Define the three-weight action space

- **task:** 只为 `q_h2`、`q_soc`、`q_batt` 定义有限、可解释、始终保持 QP 凸性的动作表；移除目标路径中的 `q_ramp`、terminal、直接功率和左右侧动作。
- **reason:** 现有 `action_mapper.py`、`run_train_dqn.py` 和环境使用不同动作语义，无法比较或复现。
- **affected_files:** `src/dqn/utils/action_mapper.py`, `outputs/config/dqn_mpc_action_table.json`, formal DQN config, action tests.
- **acceptance_criteria:** 每个 action ID 唯一映射三项非负权重；所有动作保持同一硬约束和 horizon；动作表版本/hash 随 checkpoint 保存；测试覆盖边界、重复项、无效权重和 deterministic mapping。
- **dependencies:** P0-3.

### P1-2 Freeze an action-independent physical reward

- **task:** 用氢耗、SOC 安全/维持、电池吞吐、求解失败和必要的动作切换项定义奖励，固定归一化和裁剪规则。
- **reason:** 若奖励直接使用随 action 权重变化的 MPC objective，agent 可通过改变计分尺而非改善物理行为获益。现有 reward 未直接复用 objective，但仍有 SOC placeholder 和不同环境接口。
- **affected_files:** `src/dqn/utils/reward.py`, formal environment, DQN config, reward unit tests.
- **acceptance_criteria:** 同一物理 transition 在不同 action 标签下产生相同基础物理成本；单位、参考值、裁剪和 terminal/failure penalty 有测试；reward 分量逐步记录；论文指标不以训练 reward 代替。
- **dependencies:** P0-3, P1-1.

### P1-3 Build the real LSTM-OSQP DQN environment

- **task:** 环境每步使用正式 LSTM 预测和 `N=6` OSQP 控制器，DQN 只选权重，实际执行第一步功率并更新 SOC。
- **reason:** `ship_env_simple.py` 和 `ship_env_dual_side.py` 直接控制功率且使用旧容量；`run_train_dqn.py` 使用 30 s CasADi 和 `q_soc/q_ramp`，都不是真正目标环境。
- **affected_files:** `src/envs/`, formal controller/provider adapters, `src/main/run_train_dqn.py` or a new explicit entry, integration tests.
- **acceptance_criteria:** state/action/reward/transition schema 版本化；无 test voyage 训练泄露；每一步可追溯 forecast、action、QP status、applied power、SOC 和 reward；与固定权重 controller 共享同一物理执行路径；失败回退与 P0-2 完全一致。
- **dependencies:** P0-2, P1-1, P1-2.

### P1-4 Establish tested and reproducible DQN training

- **task:** 补齐 target sync、Double DQN target、replay warm-up、epsilon schedule、checkpoint resume、seed 和 evaluation-only 测试，冻结训练协议。
- **reason:** 组件代码存在，但测试目录没有 DQN 专项测试，也没有目标环境 checkpoint。
- **affected_files:** `src/dqn/agents/`, `src/dqn/memory/`, `src/dqn/policies/`, formal train/eval entrypoints, new `tests/test_dqn_*.py`.
- **acceptance_criteria:** 小型 deterministic environment 上有可重复 smoke test；训练和评价数据严格分离；checkpoint 保存网络类型、optimizer、replay/step、action table、config、seed 和 commit；evaluation 禁用 exploration；失败恢复不改变实验身份。
- **dependencies:** P1-3.

### P1-5 Prepare fair MLP and SineKAN baselines

- **task:** 让 MLP-DQN 与 SineKAN-DQN 通过同一 network factory 接受相同 state/action、训练步数、replay、optimizer 策略、seed 集和评价航次；KAN-DQN 作为可选第三组。
- **reason:** 网络实现存在，但没有目标环境下可比较产物；参数量和计算成本不控制时，结果不公平。
- **affected_files:** `src/dqn/networks/`, factory/config, DQN runner, model summary and evaluation reports.
- **acceptance_criteria:** 保存参数量、FLOPs/推理时间或等价计算成本；除 Q 网络结构外训练协议一致；至少多个预注册 seed；逐 seed 和置信区间均报告；没有只挑最好 seed 的结论。
- **dependencies:** P1-4, third-party license decision for SineKAN.

## P2 — Produce paper evidence

### P2-1 Quantify forecast-to-control impact

- **task:** 比较可用预测、current-hold/last-slope、以及仅用于上界的 perfect-future 条件对闭环行为的影响。
- **reason:** 1 s LSTM 当前不优于简单 baseline；单独的预测 MAE 不能证明控制价值。
- **affected_files:** formal forecast provider interface, fixed controller evaluation, experiment configs/reports.
- **acceptance_criteria:** 相同航次、初始状态和 controller 配置；分别报告预测指标和物理闭环指标；perfect-future 明确标为 oracle；结论反映统计不确定性。
- **dependencies:** P0-3 and a causal/declared forecast source.

### P2-2 Compare fixed MPC and dynamic weighting

- **task:** 在相同 7 个 test 航次上比较 accepted fixed QP-MPC、MLP-DQN 调权和 SineKAN-DQN 调权。
- **reason:** 这是论文方法有效性的核心证据，目前尚不存在。
- **affected_files:** formal evaluation runner, result schema, report/plot pipeline.
- **acceptance_criteria:** 报告氢耗、`SOC_end-start`/band violations、SOC step/slope、电池放电与 throughput、FC 波动、求解时间、失败/fallback 和 action 分布；保留逐航次结果；不把内部 reward 或任意加权总分当经济成本。
- **dependencies:** P0-3, P1-5.

### P2-3 Run network architecture comparison and ablations

- **task:** 对 MLP、SineKAN、可选 KAN 做同协议比较，并消融 forecast、动态权重和关键 state/reward 分量。
- **reason:** 需要区分网络结构收益、动态调权收益和预测输入收益。
- **affected_files:** DQN configs, evaluation scripts, statistical analysis/reporting.
- **acceptance_criteria:** 预注册 seed 和主要指标；同数据预算/训练预算；报告均值、离散度和失败 run；同时给出网络参数量与推理时间；负结果不丢弃。
- **dependencies:** P2-2.

### P2-4 Validate robustness and realtime behavior

- **task:** 对预测误差、SOC 初值、负载边界、求解器扰动和 out-of-distribution 航次做压力测试。
- **reason:** nominal solver success 不足以证明工程可用性。
- **affected_files:** scenario generator, formal controller/env, evaluation reports.
- **acceptance_criteria:** 压力场景与正常 test 完全分开；预定义安全边界；报告 worst case、fallback 和 p50/p95/p99 solve time；给出不满足实时/安全约束的条件。
- **dependencies:** P2-2.

## P3 — Engineering, provenance, and auxiliary work

### P3-1 Fix exact split-key validation

- **task:** 修复 10 ms audit，要求 `assignments` key 集合严格等于 `train/validation/test`，并返回稳定的 `ValueError` 诊断。
- **reason:** 当前缺失 key 被忽略，额外 key 触发 `KeyError`；导致 4 项单元测试失败并阻止仓库全绿。
- **affected_files:** `src/main/audit_millisecond_10ms_dataset.py`, `tests/test_millisecond_10ms_audit.py`.
- **acceptance_criteria:** missing_train、missing_validation、missing_test、extra_holdout 四种场景均通过；完整测试全绿；错误消息列出 missing/extra 集合。
- **dependencies:** none. This documentation-only round does not authorize the Python fix.

### P3-2 Lock dependencies and add CI

- **task:** 用 `pyproject.toml`/lock 或受控 requirements 明确 Python 和核心包版本，修正 `sklearn` 包名并补齐 OSQP/Parquet/Optuna 等依赖，建立 Windows/Linux 至少一条 CI。
- **reason:** 当前 `requirements.txt` 不完整，无法保证干净环境安装；无 CI 防止测试/路径回归。
- **affected_files:** `requirements.txt`, future `pyproject.toml`/lock, CI workflow, README installation section.
- **acceptance_criteria:** 空环境一条命令安装；CLI `--help`、数据小样 smoke 和单元测试在 CI 通过；依赖版本、平台限制和可选训练组清晰。
- **dependencies:** P3-1 for all-green baseline.

### P3-3 Remove local absolute-path assumptions

- **task:** 将代码默认值、manifest 和 run config 中的个人桌面/旧工作区绝对路径迁移为仓库相对路径或显式 CLI/config 输入。
- **reason:** 当前数据构建、报告和保留输出含不可移植路径，阻止他人复现并可能暴露个人目录。
- **affected_files:** `configs/base.yaml`, `src/main/build_millisecond_10ms_dataset.py`, `src/main/build_total_load_dataset_721.py`, `src/main/validate_fc_dp0_curve.py`, selected docs/reports/manifests/run configs.
- **acceptance_criteria:** `rg` 审计不再在 active 代码/README/STATUS/新文档中发现个人绝对路径；旧产物若必须保留，另存 sanitized metadata 并保留 hash；从任意 clone 路径可运行 smoke test。
- **dependencies:** define which outputs are authoritative.

### P3-4 Resolve third-party code and licensing

- **task:** 核验 SineKAN 和外部 KAN 的上游 commit、许可证、引用与再分发条件；决定保留完整副本、submodule/vendor 记录或最小实现。
- **reason:** `SineKAN-main/` 没有 LICENSE/包元数据，不能在发布前假定可自由再分发。
- **affected_files:** `SineKAN-main/`, `src/dqn/networks/sine_kan_qnet.py`, KAN modules, future NOTICE/CITATION files.
- **acceptance_criteria:** 上游 URL/commit/hash 和许可证可验证；项目分发方式合规；最小实现有 parity test；notebook 与正式运行代码边界清楚。
- **dependencies:** legal/source verification; required before publishing SineKAN results.

### P3-5 Establish output and repository retention policy

- **task:** 区分原始不可替代数据、权威小型摘要、可重建大型输出、历史诊断和临时工具文件，再经授权执行归档/外部存储/Git LFS 或删除。
- **reason:** `outputs/` 约 1.4 GiB 且大量文件已跟踪；`.codex_tmp/`, `tmp/`, `.idea/` 也有已跟踪内容，增加 clone 和审阅成本。
- **affected_files:** `.gitignore`, `outputs/`, `.codex_tmp/`, `tmp/`, `.idea/`, `docs/CLEANUP_CANDIDATES.md`.
- **acceptance_criteria:** 每个保留实验有 config、manifest、summary、关键曲线和生成命令；可重建巨型逐步 CSV 不重复进 Git；任何删除先备份并经用户确认；仓库无凭据和个人缓存。
- **dependencies:** paper evidence inventory and explicit cleanup approval.

### P3-6 Complete or deliberately close the 10 ms auxiliary study

- **task:** 修复审计后，以预注册小规模协议运行 10 ms LSTM/baseline，或明确将其关闭为代码可用但无论文结论的辅助路线。
- **reason:** 当前有数据、代码和测试，但无正式 checkpoint/报告；继续悬置会混淆项目范围。
- **affected_files:** `src/main/run_lstm_millisecond_10ms_search.py`, `src/forecasting/millisecond_multistep_lstm.py`, auxiliary outputs/report.
- **acceptance_criteria:** 若运行，保存数据 hash、split、seed、配置、逐 horizon baseline 对比和延迟；明确不接入船舶 MPC/DQN。若关闭，在 STATUS/PROJECT_MAP 标记并停止产生新输出。
- **dependencies:** P3-1, P3-2; no dependency on core P0/P1 work.
