# V6 依赖盘点与历史 DQN 脚本清理报告

## 实际回归结果

清理前，三个SB3回归文件合计10项通过（31.80秒）。清理后串行运行SB3环境、Mask/TD、
checkpoint恢复，以及共享FC/电池退化、能量模型、经济区间、参数冻结、独立经济/数据、失败TD，
并显式运行已归档的历史CLI测试：**127 passed，285 subtests passed，59.57秒**。
命令和范围见`phase1_regression.json`。没有执行完整120文件套件，不将本结果表述为全部历史测试通过。
上述回归使用固定fixture或临时数据，没有正式训练或重跑Test。

基线：`544908d67ae9c36f8101ba5a27d290eaf4c49dd2`。分支：`feat/v6-sb3-optuna-audit`。
本报告只覆盖阶段 1。归档范围经过主线程确认，只移动 3 个明确的历史 DQN 专用文件，
没有删除源码、依赖、数据、模型或训练结果，也没有修改其他 worktree。

## 扫描范围与结果

| 项目 | 结果 |
| --- | ---: |
| 基线 Git 受控 Python 文件 | 348 |
| `docs/results/` 历史 Python 快照，仅登记路径、不打开内容 | 35 |
| 历史源码/依赖入口 artifact 总数，另含 1 个 requirements 快照 | 36 |
| 其余受控 Python 源码，全部 AST 解析成功 | 313 |
| 经归档路径映射后仍存在的上述源码 | 313 / 313 |
| 获批归档 | 3 |
| 删除 | 0 |
| 非本次工作树的状态对比 | 9 / 9 一致 |

`inventory_before.json` 和 `file_inventory_before.csv` 保存清理前清单；
`inventory_after.json` 和 `file_inventory_after.csv` 保存清理后清单。
运行中的其他代理在同一工作树新增的代码也单独登记为 `tracked=false`，不伪称它们属于基线。
最新 after 快照共解析 319 个 Python 文件，其中 310 个仍位于 Git 原路径，
3 个基线文件位于新归档路径，另外 6 个为并行工作新增代码。
前后 AST 解析错误均为 0。

依赖边同时提供 CSV 与 JSON：`dependency_edges_before.*` 为 5,833 条，
`dependency_edges_after.*` 为 5,861 条。边数变化也包含同一工作树中并行新增代码的影响，
不能将全部差额归因于归档。依赖声明和 pytest 入口另见 `dependency_entrypoints.*`。
清理前受控源码的依赖边按 Git 基线内容重建，保留原先取得的清理前 worktree 状态快照。

## 生产依赖、回归与保留边界

以 `src/v6_sb3/` 的当前模块和 `scripts/start_v6_sb3.ps1`、
`scripts/build_v6_dataset.py`、`scripts/report_v6.py`、`scripts/finish_v6_report.ps1`
为入口，计算包含函数内导入、包初始化与字面脚本引用的保守闭包。
基线闭包有 56 个文件；其中明确包含以下共享实现：

- `src/v2/` 的物理参数、经济账本、电池/燃料电池退化、控制和状态构建。
- `src/v3/control.py`、`episode_replay.py`、`shore.py` 的物理执行与结算。
- `src/v4/` 的动作 mask、DQN、失败回放、独立经济口径、V6 数据、TD 和 Q 诊断。
- `src/dqn/networks/` 的共享网络工厂和实现。

因此，不能因模块名包含 `v2`、`v3` 或 `v4` 就删除旧版本源码。
这次归档未修改这些生产模块，也未删除它们的依赖。
after 闭包为 58 个文件：新增的 `src/v6_sb3/performance.py`、
`src/v6_sb3/search.py` 属于主任务并行开发；原有 56 个闭包文件一个未少。

机器清单按以下类别区分用途：

| 类别 | 解释与处置 |
| --- | --- |
| `v6_production_closure` | 可从选定 V6 入口静态到达；保留。 |
| `current_regression` | 继续保留的默认回归测试；包括旧模块上的共享合同回归。 |
| `historical_experiment` | 已确认历史训练/研究入口；除本次 3 项归档外，继续原地保留。 |
| `unused_unconfirmed` | 不在该闭包内，但无法证明未使用；全部保留。 |
| `dependency_entry` / `audit_tool` | 依赖/测试入口与只读审计工具。 |
| 历史 artifact | `docs/results/` 的冻结源码/配置；只登记文件名。 |

基线的 121 个 `tests/*.py` 中只归档 1 个，其余 120 个继续留在默认测试目录。
三个 SB3 测试 `test_sb3_environment.py`、`test_sb3_masked_dqn.py`、
`test_sb3_training_integration.py` 全部保留。
物理、经济、数据完整性、checkpoint/resume、mask 与 TD 测试都未被本次移动。
例如 `test_v2_dqn_experiments.py`、`test_v2_m10_ablation.py` 虽属历史实验，
但包含 checkpoint 兼容检查，因此继续保留。

## 归档与恢复映射

| 原路径 | 归档路径 | 理由与内容变化 |
| --- | --- | --- |
| `tests/test_v2_history_study_cli.py` | `tests/history/test_v2_history_study_cli.py` | 历史 H1/H2 study CLI 与训练路由；仅将 `ROOT` 的 `parents[1]` 改为 `parents[2]`。 |
| `scripts/v4_per_frozen_probe.py` | `scripts/history/v4_per_frozen_probe.py` | V4 PER frozen-target 离线拟合研究；源码未编辑。 |
| `scripts/v4_per_jointfit_state_probe.py` | `scripts/history/v4_per_jointfit_state_probe.py` | V4 PER 7d/clock 联合拟合及小样本 Train 研究；源码未编辑。 |

两个脚本均没有 `__file__`、`ROOT` 或相对脚本目录加载其他文件的逻辑，
原有参数仍相对调用工作目录解释，无需改写根目录。
已将归档文件与基线源码比较：脚本内容一致，只存在 Git LF 与工作区 CRLF 的既有差异；
归档测试除上述 ROOT 单行适配外内容一致。
扫描到的源码/脚本中没有依赖这 3 个旧路径的运行时入边；
审计工具中的映射字符串单独作为清理记录存在。
历史文档中的旧命令保留原貌，不改写历史结果或设计记录。

完整保留、归档、删除清单及回滚映射见 `cleanup_manifest.json`；
平面归档表见 `cleanup_files.csv`。删除清单为空。
恢复时将 3 个文件移回原路径，并将该历史测试的 ROOT 层数恢复为 1。

## 默认测试入口

新增 `pytest.ini`：

```ini
[pytest]
testpaths = tests
pythonpath = src src/main
norecursedirs = history .git .venv .venv-* .worktrees
```

这样默认 `pytest` 从测试目录收集，并排除 `history`，同时明确原有源码导入路径。
`src/main/test_*.py` 与 `src/v3/test_diagnostic.py` 是带 `main` 的历史命令入口，
静态检查没有发现它们定义 pytest 测试函数，因此不把这些 CLI 纳入默认收集。
需要复查归档测试时仍可显式运行：

```powershell
python -m pytest tests/history/test_v2_history_study_cli.py -q
```

本清理任务没有运行 pytest、训练、Test 或历史离线拟合脚本。
当前回归和必要 smoke 由主线程统一运行，结果由主线程汇总；
本报告只声明已完成的静态检查与文件保留核验。

## 其他工作树保护证据

`worktrees_before.json` 与 `worktrees_after.json` 记录 10 个 worktree 的 HEAD、
分支、全部 status 和 untracked 文件名。`worktree_preservation.json` 对除本工作树外的
9 个工作树逐一比较，HEAD、分支、状态与文件名列表全部一致。

| 工作树 | 有状态的路径数 | 其中 untracked | DDPG/TD3 相关未提交路径 |
| --- | ---: | ---: | ---: |
| 主工作树 | 0 | 0 | 0 |
| `direct-power-mlp` | 1,153 | 1,107 | 38 |
| `v4-energy-soc-per` | 12 | 7 | 0 |
| `v4-increment7-zhou` | 7 | 2 | 0 |
| `v4-per-prep` | 33 | 10 | 0 |
| `v5-2x2-control` | 6 | 5 | 0 |
| `v5-all-legal-bellman` | 11 | 10 | 0 |
| `v5-economic4-ddqn` | 0 | 0 | 0 |
| `v6-dqn-1s` | 0 | 0 | 0 |

这些比较是状态与文件名层面的保护证据，不代表逐字节内容证明。
未打开其他工作树的未提交源码内容，没有重置、清理、移动或暂存其中任何文件，
也未计算大结果、模型、Replay 或数据 hash。

## 静态分析局限与复查方式

- AST 会记录函数内、可选和类型检查导入；被静态到达不等于实际执行。
- 记录 `importlib`、`__import__`、`runpy` 和子进程调用，识别字面路径、
  PowerShell `-m` 与 Python 字面参数向量。`v6_sb3.training` 动态加载
  `scripts/report_v6.py`、旧 V6 子进程调用报告脚本的引用均已计入。
- 动态拼接模块名、变量别名、插件注册、环境控制的 `sys.path`、`exec/eval`
  无法保证完整解析，保留原表达式供人工复查。
- 字符串中的路径和命令可能只是文档、错误消息或元数据；边会标记为候选引用。
  外部或未解析引用不会被当成“可删除”证据。
- 模块解析按 `src`、旧 `src/main`、测试和脚本同目录约定建别名；
  实际导入优先级、同名冲突和 monkeypatch 仍可能不同。
- 无入边或不在 V6 闭包内均不能证明未使用。本次没有据此删文件。
- 只读取受控及显式登记的新增 Python 源码、脚本与依赖配置；
  没有导入生产模块，没有打开任何 Train/Validation/Test 数据 payload、checkpoint、
  Replay 或结果内容。`docs/results/` 中的冻结源码也不解析。

工具为 `scripts/audit_v6_dependencies.py`，使用 Python 标准库。
清理前应在移动前生成 `--phase before`；清理后可重新执行：

```powershell
python -B scripts/audit_v6_dependencies.py --phase after
```

工具只写本工作树 `docs/audits/` 下的审计文件，不执行归档或 Git 修改。
本任务未 commit；阶段 1 的精确暂存路径见 `stage1_submission_files.txt`，
不包括并行经济审计/Optuna 实现源码或它们的测试。
