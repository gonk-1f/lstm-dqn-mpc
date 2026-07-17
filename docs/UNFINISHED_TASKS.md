# Unfinished Tasks

## Prioritization rule

- **P0**：阻止目标 LSTM-QP-MPC 闭环成立的问题。
- **P1**：在任何正式 DQN 训练前必须完成的问题。
- **P2**：形成论文证据所需的实验。
- **P3**：工程、复现和辅助研究整理。

任务按依赖排序。固定 `N=6` MPC 未被正式接受前，不应启动正式 DQN 训练；1 s 离线 spline 不得替代真实在线因果数据结论。

## P0 — Close the fixed LSTM-QP-MPC loop

### P0-1 Freeze the formal timing and data contract

- **task:** 明确正式 1 s 实验的数据来源、decision time、LSTM history、h1–h6 预测含义、MPC stage 0–5 映射、实际负荷反馈和 SOC 更新时间。
- **reason:** 离线 ideal-foresight `N=6` 路径已用测试固定为 `t+1..t+6`、只执行第一步和实际 SOC 更新，但正式 LSTM provider 的 history/h1–h6 映射、可用时刻和因果数据仍未接入。`N=60` 已降为历史 benchmark。
- **affected_files:** formal forecast-provider/controller config, `src/main/run_lstm_mpc_test.py`, `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`, `tests/test_mpc_1s_n6_four_objective_sensitivity.py`, `docs/DATA_PROVENANCE.md`.
- **acceptance_criteria:** 一份版本化 contract 明确每个向量元素对应的物理时间；测试可检测 off-by-one、zero-delay 和未来数据泄露；正式数据被标记为 measured/causal 或 offline/reconstructed；论文主实验不把未来 spline 端点写成在线可用信息。
- **dependencies:** none.

### P0-2 Implement one reusable `N=6` OSQP closed-loop entry

- **task:** 从已验证的 ideal-foresight runner 提取可复用正式控制器，接收 6 步 forecast provider 输出、当前 SOC/FC 状态，输出第一步功率并提供确定性失败回退。
- **reason:** `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` 已实现正确滚动时序、等价数值缩放、实际功率/SOC 更新和失败终止，但它是 offline-oracle 实验入口，不接 LSTM；最终求解失败时只能终止航段，尚不能部署。
- **affected_files:** `src/main/mpc_solvers/mpc_qp_formulation.py`, `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`, new controller/provider module under `src/`, formal CLI and integration tests.
- **acceptance_criteria:** horizon 固定为 6；固定稀疏结构、参数更新和 warm start 被保留；每步执行的 `P_fc/P_batt` 满足功率平衡、设备边界、SOC 边界和 48 kW/s ramp；SOC 用实际施加的 `P_batt` 更新；逐步记录 status、iterations、residual、solve time 和约束残差；具有经过测试的确定性失败回退，不用 NaN 继续控制。
- **dependencies:** P0-1.

### P0-3 Run the four-objective matrix and manually accept or reject a fixed baseline

- **task:** 先运行四权重全 1 baseline，再运行完整 17 配置 one-factor；逐航次审查功率分配、氢耗、电池使用、SOC、FC variation、硬约束和求解状态，只在证据充分时形成 provisional/accepted 固定基线。
- **reason:** 当前 objective 已固定为 `H2_norm`、`Batt_power_sq_norm`、`SOC_tracking_sq_norm`、`FC_variation_sq_norm`，参考值分别为 `0.00883945296644347 kg/step`、`346.5 kW`、`SOC_ref=0.55`/`SOC_band=0.05`、`48 kW/step`。baseline 为四权重全 1；每项 one-factor 值为 `0.25,0.5,1,2,4`。但是 baseline 与完整 17 配置结果均未运行，不能预先宣称趋势、推荐区间或最佳权重。
- **affected_files:** `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`, `tests/test_mpc_1s_n6_four_objective_sensitivity.py`, `outputs/mpc_1s_n6_four_objective_sensitivity/`, `reports/mpc_1s_n6_four_objective_sensitivity_summary.md`, `reports/mpc_1s_n6_four_objective_sensitivity_table.csv`, formal MPC config, `STATUS.md`.
- **acceptance_criteria:** `--baseline` 先完成 7 个固定 test 航段，`--one-factor` 再形成 baseline-first 的 17 个唯一配置；复核 hard-constraint tolerance、SOC、氢耗、电池吞吐、FC variation/surplus、完成率和 1 s 实时性；明确 offline oracle 到因果预测的外推边界；基于物理指标形成显式 accepted/rejected 或证据支持的下一区间，不使用自动 best/score/rank/winner、加权总分或 least-bad 替代人工门禁；正式接受前保持 DQN 阻塞。
- **dependencies:** P0-1.

### P0-4 Remove stale parameters from the active execution path

- **task:** 为目标入口建立唯一配置源，阻止 277.2/1067/1806 kWh、138.6/350 kW、SOC 0.65、horizon 18 等历史值进入正式闭环。
- **reason:** `configs/`, `src/envs/` 和历史脚本存在多套互不兼容参数；仅更新文档无法避免误运行。
- **affected_files:** `configs/mpc.yaml`, `configs/ship_system.yaml`, `configs/dqn.yaml`, formal controller/env config, configuration validation tests.
- **acceptance_criteria:** 正式 CLI 保存解析后的完整配置；启动时校验容量、功率、SOC、采样间隔和 horizon；目标测试断言 `560/693/346.5/0.55/0.2/0.8/N=6`；历史入口显式标记 historical 或要求单独配置，不静默回退到旧值。
- **dependencies:** P0-1, P0-2.

## P1 — Freeze the DQN problem before training

### P1-1 Define the three-weight action space

- **task:** 在 P0-3 人工冻结固定 baseline（包括 `q_fc_var`）后，只为 `q_h2`、`q_soc`、`q_batt` 定义有限、可解释、始终保持 QP 凸性的 DQN 动作表；移除目标路径中的 `q_ramp`、terminal、直接功率和左右侧动作。
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

### P3-1 Lock dependencies and add CI

- **task:** 用 `pyproject.toml`/lock 或受控 requirements 明确 Python 和核心包版本，修正 `sklearn` 包名并补齐 OSQP/Parquet/Optuna 等依赖，建立 Windows/Linux 至少一条 CI。
- **reason:** 当前 `requirements.txt` 不完整，无法保证干净环境安装；无 CI 防止测试/路径回归。
- **affected_files:** `requirements.txt`, future `pyproject.toml`/lock, CI workflow, README installation section.
- **acceptance_criteria:** 空环境一条命令安装；CLI `--help`、数据小样 smoke 和单元测试在 CI 通过；依赖版本、平台限制和可选训练组清晰。
- **dependencies:** none; the current local unit-test baseline is all green.

### P3-2 Remove local absolute-path assumptions

- **task:** 将代码默认值、manifest 和 run config 中的个人桌面/旧工作区绝对路径迁移为仓库相对路径或显式 CLI/config 输入。
- **reason:** 当前数据构建、报告和保留输出含不可移植路径，阻止他人复现并可能暴露个人目录。
- **affected_files:** `configs/base.yaml`, `src/main/build_millisecond_10ms_dataset.py`, `src/main/build_total_load_dataset_721.py`, `src/main/validate_fc_dp0_curve.py`, selected docs/reports/manifests/run configs.
- **acceptance_criteria:** `rg` 审计不再在 active 代码/README/STATUS/新文档中发现个人绝对路径；旧产物若必须保留，另存 sanitized metadata 并保留 hash；从任意 clone 路径可运行 smoke test。
- **dependencies:** define which outputs are authoritative.

### P3-3 Resolve third-party code and licensing

- **task:** 核验 SineKAN 和外部 KAN 的上游 commit、许可证、引用与再分发条件；决定保留完整副本、submodule/vendor 记录或最小实现。
- **reason:** `SineKAN-main/` 没有 LICENSE/包元数据，不能在发布前假定可自由再分发。
- **affected_files:** `SineKAN-main/`, `src/dqn/networks/sine_kan_qnet.py`, KAN modules, future NOTICE/CITATION files.
- **acceptance_criteria:** 上游 URL/commit/hash 和许可证可验证；项目分发方式合规；最小实现有 parity test；notebook 与正式运行代码边界清楚。
- **dependencies:** legal/source verification; required before publishing SineKAN results.

### P3-4 Establish output and repository retention policy

- **task:** 区分原始不可替代数据、权威小型摘要、可重建大型输出、历史诊断和临时工具文件，再经授权执行归档/外部存储/Git LFS 或删除。
- **reason:** `outputs/` 约 1.4 GiB 且大量文件已跟踪；`.codex_tmp/`, `tmp/`, `.idea/` 也有已跟踪内容，增加 clone 和审阅成本。
- **affected_files:** `.gitignore`, `outputs/`, `.codex_tmp/`, `tmp/`, `.idea/`, `docs/CLEANUP_CANDIDATES.md`.
- **acceptance_criteria:** 每个保留实验有 config、manifest、summary、关键曲线和生成命令；可重建巨型逐步 CSV 不重复进 Git；任何删除先备份并经用户确认；仓库无凭据和个人缓存。
- **dependencies:** paper evidence inventory and explicit cleanup approval.

### P3-5 Complete or deliberately close the 10 ms auxiliary study

- **task:** 以预注册小规模协议运行 10 ms LSTM/baseline，或明确将其关闭为代码可用但无论文结论的辅助路线。
- **reason:** 当前有数据、代码和测试，但无正式 checkpoint/报告；继续悬置会混淆项目范围。
- **affected_files:** `src/main/run_lstm_millisecond_10ms_search.py`, `src/forecasting/millisecond_multistep_lstm.py`, auxiliary outputs/report.
- **acceptance_criteria:** 若运行，保存数据 hash、split、seed、配置、逐 horizon baseline 对比和延迟；明确不接入船舶 MPC/DQN。若关闭，在 STATUS/PROJECT_MAP 标记并停止产生新输出。
- **dependencies:** P3-1; no dependency on core P0/P1 work.
