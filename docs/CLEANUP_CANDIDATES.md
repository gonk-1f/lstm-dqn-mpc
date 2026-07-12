# Cleanup Candidates

## Safety rule

本文件是审计清单，不是删除授权。本轮不得依据它执行 `git rm`、删除/移动目录、重写 Git 历史或清空 outputs。任何清理都必须先确认权威产物、可重建性、备份位置、论文引用和用户授权。

扫描日期为 2026-07-12。排除 `.git/` 和本地虚拟环境后，工作树约 2,884 个文件、1.73 GB；`outputs/` 约 1.40 GB，是主要体量来源。数字是接管时快照，后续会变化。

## Can be ignored for future files

以下属于本地工具/编辑器产物，适合在 `.gitignore` 中阻止新增：

| Path/pattern | Current observation | Recommendation |
| --- | --- | --- |
| `.codex_tmp/` | 约 1,041 个文件已被 Git 跟踪，目录约 52 MiB | 新文件直接忽略；已跟踪文件在单独清理任务中审计后再 `git rm --cached` |
| `tmp/` | 10 个文件已跟踪，约 6 MiB | 新文件直接忽略；先确认是否包含唯一实验摘要 |
| `.idea/` | 6 个文件已跟踪 | 新文件直接忽略；如团队不共享 IDE 配置，再单独取消跟踪 |
| `.vscode/` | 现有规则已忽略个人设置 | 保持；若未来需要团队配置，应仅白名单明确文件 |
| `venv/`, `.venv/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `*.log`, `.env` | 已有忽略规则 | 保持；不得提交凭据 |

本轮不添加全局 `*.pt`/`*.pth` 忽略：仓库已有 7 个被跟踪的模型文件，部分可能是保留复现证据。checkpoint 应在 P3 输出策略中按“权威模型、可重建模型、临时 epoch”分类，再决定 Git LFS/制品存储/精确 ignore。

## Must keep unless an explicit migration is approved

| Category | Paths | Reason |
| --- | --- | --- |
| 源码和测试 | `src/`, `tests/` | 当前实现和行为证据；历史测试也帮助识别旧接口 |
| 当前文档 | `README.md`, `AGENTS.md`, `STATUS.md`, `docs/` | 项目接管、科学边界和后续执行入口 |
| 30 s 原始数据 | `total_load_excels/` | 66 航段实船原始来源；不可用 spline 替代 |
| 毫秒原始副本 | `data/millisecond_1ms/raw/` 与 manifest | 辅助数据 lineage；清理前需外部备份和 hash |
| 数据划分 | `outputs/config/voyage_split_total_load_721.json`, `outputs/config/millisecond_10ms_split_721.json` | 防止 split 漂移和数据泄露 |
| FC 曲线 | `data/fuel_cell/FC_Dp0_curve_for_Python.csv` | 氢耗计算依据 |
| 关键摘要/配置 | 每个保留实验的 manifest、run/solver config、summary、逐航次/逐 horizon 指标 | 论文复现所需最小证据 |
| 第三方代码（暂时） | `SineKAN-main/` | 当前 wrapper 实际导入；许可证/最小化方案未决，不能先删 |

## Large data and generated outputs

| Candidate | Observation | Proposed action | Approval needed |
| --- | --- | --- | --- |
| `outputs/spline_1s_diagnostics/data/cubic_spline_1s_natural_clipped.csv` | 约 337 MiB 的合并 CSV；当前未被 Git 跟踪且已有明确 ignore | keep ignored；确认可由逐航次文件/脚本重建后可删除本地副本 | yes |
| `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` | 66 个逐航次 1 s CSV；OSQP 数据可由此构建 | 保留 manifest/脚本；评估将大数据移到 release/LFS/制品存储 | yes |
| `outputs/mpc_solver_benchmark_1s/**/control_all_steps.csv` 等逐步 CSV | 多个 14–25 MiB 文件，可由配置和输入重算 | 保留 accepted run 的摘要/失败 cases/代表轨迹；其余 `summarize_only` 后归档 | yes |
| `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt277p2_simplified_spec_norm/` | 旧 277.2 kWh/138.6 kW 大型结果树 | `move_to_archive` 或外部制品；主仓只保留 decision/config/hash | yes |
| `outputs/mpc_solver_benchmark_1s/osqp_n60_Ebatt693_simplified_spec_norm/` | 当前 `N=60` 诊断/候选结果 | 在正式 `N=6` 基线完成前 keep；之后只保留论文相关 case 和摘要 | yes |
| `outputs/lstm_mpc_total_load_test_fixed_baseline_v1/` | 30 s CasADi、1806/277.2 kWh 混合历史元数据 | `summarize_only` 后归档；保留迁移所需行为/指标 | yes |
| `outputs/lstm_721/` 与 `outputs/lstm_total_load_721/` | 两条不同数据线的 checkpoint/结果 | 先明确哪条作为 30 s 正式基线；不要互相覆盖；旧线归档 | yes |
| `outputs/lstm_spline_1s_hparam_search/` | 负结果但科学上有价值 | 保留最终 report/config/metrics；中间 checkpoint 可在 hash/备份后删 | yes |
| `data/millisecond_10ms/` | 可由项目内 1 ms 原始副本重建 | 保留 manifest/脚本/hash；CSV 是否进 Git 由辅助实验决策决定 | yes |

## Historical experiments classification

| Experiment family | status | relation_to_paper | recommended_action | Rationale |
| --- | --- | --- | --- | --- |
| 277.2 kWh / 350 kW CasADi variants | historical | supporting/diagnostic | summarize_only, then move_to_archive | 与当前 693/346.5 主线不一致；保留控制行为教训 |
| 277.2 kWh / 138.6 kW OSQP | historical | diagnostic | move_to_archive | 已被 693/346.5 basis 取代 |
| 693 kWh / 346.5 kW `N=60` OSQP | active benchmark | supporting | keep | 当前求解器/离线控制证据，但不是正式 `N=6` 基线 |
| 1806 kWh / 350 kW 30 s CasADi | historical | supporting | summarize_only | 可迁移时序/回退/指标，不是目标物理参数 |
| `raw_weight_retune` | historical | diagnostic | summarize_only | 已从 formal benchmark CLI 工作流移除；代码/历史产物仍可追溯 |
| `weight_sensitivity` | historical | diagnostic | summarize_only | 同上；不继续无边界搜索 |
| `physical_baseline_v2` | historical/uncertain artifact coverage | diagnostic | investigate, then summarize_only | 当前 1 s formal entry 明确未引入；先定位全部残留 |
| SOC reserve slack | historical/uncertain artifact coverage | diagnostic | investigate, then summarize_only | 不属于当前简化 QP objective |
| FC low-frequency/reference tracking | historical/uncertain artifact coverage | diagnostic | investigate, then summarize_only | 不属于当前 formal 1 s entry |
| terminal SOC penalty | historical | diagnostic | summarize_only | 当前暂定 QP `q_terminal_soc=0`；测试仍覆盖历史行为 |
| ramp soft penalty | historical | diagnostic | summarize_only | 当前 48 kW/s 是硬约束、`q_ramp=0` |
| linear interpolation 1 s | deprecated | unrelated to core evidence | delete_after_backup | 脚本/测试已标 DO_NOT_USE；先保留迁移说明和 Git 历史 |
| natural spline raw | historical/auxiliary | diagnostic | summarize_only | 存在负值/过冲与未来端点问题 |
| not-a-knot spline | historical/auxiliary | diagnostic | summarize_only | 只作边界条件对照 |
| natural-clipped spline | auxiliary | supporting | keep | 当前离线 LSTM/OSQP 输入，必须保留非因果标签 |
| 1 s LSTM hyperparameter search | auxiliary | diagnostic | keep summary, archive intermediates | LSTM 未超过简单 baseline，是必要负结果 |
| OSQP solver benchmark | active benchmark | supporting | keep | 固定稀疏、warm start、实时性与约束证据 |
| IPOPT/CasADi comparison | historical | supporting | summarize_only | 迁移求解器/闭环逻辑，但参数体系不同 |
| 电池功率边界审计 | supporting | supporting | keep | 物理一致性证据 |
| 氢耗曲线审计 | supporting | supporting | keep | 物理指标来源证据 |
| 毫秒/10 ms 实验 | auxiliary | diagnostic | keep code/provenance; decide study | 不接入船舶 MPC/DQN 主线 |

## Local absolute paths to migrate

以下仓库相对文件已发现或高度确定含个人桌面、旧工作区或机器专用路径。处理时应保留 lineage/hash，不应简单全局替换后丢失来源：

### Active or executable code/config

- `configs/base.yaml`：raw data root。
- `src/main/build_millisecond_10ms_dataset.py`：默认两个 source workbook。
- `src/main/build_total_load_dataset_721.py`：可选 AIS root 由 home/Desktop 构造。
- `src/main/validate_fc_dp0_curve.py`：验证图片默认路径。

### Documentation/history

- `docs/dqn_formulation.md`。
- `reports/lstm_721.md`。
- `thread.md`, `project_status.md`, `next_steps.md` 中的机器命令或旧上下文。
- `docs/superpowers/specs/` 和 `docs/superpowers/plans/` 中的旧设计定位信息。

### Generated metadata

- `data/millisecond_1ms/source_manifest.json`。
- `data/millisecond_10ms/dataset_manifest.json`, `data/millisecond_10ms/independent_audit.json`。
- `outputs/lstm_total_load_721/config.json`, `run_config.json` 和 checkpoint metadata。
- `outputs/lstm_721/**/best_lstm_load_predictor.json`, promotion/backup manifests。
- `outputs/lstm_mpc_total_load_test_fixed_baseline_v1/*.json`。
- `outputs/spline_1s_diagnostics/spline_build_summary.json`。
- `outputs/lstm_spline_1s_hparam_search/**/run_summary.json`。
- `outputs/mpc_solver_benchmark_1s/data/voyage_split_spline_1s_total_load_721.json` 与若干 `qp_formulation_check.md`。

建议为 active 新运行生成相对路径和内容 hash；历史只读产物可保存 sanitized copy，不要改写原始 evidence 后仍称 hash 未变。

## Duplicate and stale implementations

| Area | Candidate overlap | Recommendation |
| --- | --- | --- |
| 30 s LSTM | `run_train_lstm_721.py` vs `run_train_lstm_total_load_721.py` | 保留二者直到 35/66 航段论文用途明确；命名/README 强制区分 |
| 1 s LSTM | deprecated linear interpolation、old diagnostic、fixed Task C | 线性插值 deprecated；保留一个 spline diagnostic 和一个可配置正式入口 |
| MPC | generic SciPy/CasADi controllers、30 s runner、OSQP benchmark | 提取目标 `N=6` OSQP controller；其他标 historical/supporting，不立即删 |
| DQN env | simple、dual-side、`run_train_dqn.py` 内嵌权重控制 | 新建/选择一个目标环境后归档旧接口；先用测试记录差异 |
| KAN | external pykan、KAN-v2、SineKAN | 公平比较与许可证决策后决定最小支持集合 |
| Status files | `thread.md`, `project_status.md`, `next_steps.md`, `STATUS.md` | `STATUS.md` 唯一 active；其余冻结为历史，不再同步写流水账 |

## Stale parameter configurations

以下文件必须在使用前调查，不能作为当前默认事实：

- `configs/mpc.yaml`：历史 horizon/SOC/权重。
- `configs/ship_system.yaml`：1806 kWh/350 kW 等历史物理值。
- `configs/dqn.yaml`：旧 q 范围/环境接口。
- `outputs/config/mpc_weight_sets.json`：包含 277.2 kWh 和多组历史 CasADi 权重。
- `outputs/config/dqn_mpc_action_table.json`：包含 `q_terminal_soc`/旧动作维度。
- `src/envs/ship_env_simple.py` 与 `src/envs/ship_env_dual_side.py`：1067/1806 kWh、直接功率动作。
- `src/main/run_train_dqn.py`：1806 kWh、`q_soc/q_ramp` 动作和 30 s CasADi 控制。

推荐动作是 `investigate` -> 为目标入口添加严格配置校验 -> 将旧配置重命名/归档。不得直接把暂定 benchmark 权重覆盖到这些全局文件。

## Third-party code

`SineKAN-main/` 含第三方源码、README 和 8 个 notebook，项目 wrapper 目前直接导入 `sine_kan.py`。目录未发现 LICENSE 或固定上游 commit。建议顺序：

1. 核验上游仓库、论文、commit 和许可证。
2. 记录该副本 hash，并为项目 wrapper 添加 parity test。
3. 根据许可证决定保留 vendor 副本、使用依赖/submodule，或提取带 attribution 的最小实现。
4. notebook 若与正式运行无关，可在备份和授权后移到外部 archive。

在完成以上步骤前，recommended action 是 **keep and investigate**，不是删除。

## Proposed cleanup sequence

1. 冻结权威数据、模型、benchmark 和论文引用清单。
2. 为每个可重建产物保存生成命令、config、input hash、commit、summary 和代表性轨迹。
3. 外部备份并校验 hash。
4. 单独提交取消跟踪本地临时目录；不要与算法修改混合。
5. 将历史结果迁到 archive/release/LFS 或对象存储，再更新链接。
6. 仅在用户明确批准后删除重复逐步输出或 deprecated 文件。
7. 最后用 fresh clone/CI 验证核心工作流与文档路径。
