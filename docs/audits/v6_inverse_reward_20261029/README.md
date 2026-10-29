# 实时 SOC 估值与直接倒数奖励：审计交付

基点：PR #2 的 `814994e`。分支：`feat/v6-sb3-inverse-reward-audit`。检查远端时 PR #1、#2 均尚未合并；本次 PR 指向 `feat/v6-sb3-economic-episode500`。

**结论：本分支交付审计，不开放生产 `inverse_economic`。** 用户明确要求发生不可消除的冲突时停止相应修改。本次在真实 Train 合法非终止动作中发现零成本，且当前约束禁止通过 epsilon、偏移、裁剪、取绝对值、禁动作或伪造失败来定义倒数。因此未修改生产经济模型、奖励、旧失败分数、训练或 Optuna 接口，也没有为未实现功能编造运行命令。

## 阅读顺序

| 内容 | 文件 |
| --- | --- |
| 审计先行与停止边界 | [execution_scope.md](execution_scope.md) |
| 修改前源码、单位、分项、历史证据缺失与实测统计 | [pre_audit.md](pre_audit.md)、[pre_audit.json](pre_audit.json) |
| 实时充电、FC-off 放电、库存及边际估值方案 | [realtime_efficiency_design.md](realtime_efficiency_design.md) |
| 倒数、终止 −1、数值与长期目标分析 | [inverse_reward_analysis.md](inverse_reward_analysis.md) |
| 数值反例、35 条 Train 短前缀、两条完整审计轨迹 | [boundary_cases.csv](boundary_cases.csv)、[train_prefix_steps.csv](train_prefix_steps.csv)、[train_full_steps.csv](train_full_steps.csv) |
| 实际检查、保留范围与后续接口状态 | [verification.md](verification.md) |

## 可核对的关键证据

1. 新生成的 35 条 Train 各 32 步，共 1120 个固定合法动作，其中 **35 个成本为零**，分布在 **27 个 Train ID**，全部为正常非终止动作。该样本没有负成本，不代表完整合法域严格为正。最小正成本为 `6.836828188453796e-7 CNY`，对应倒数约 `1462666.564722 CNY^-1`。
2. 同一最短航段的两条新审计轨迹均正常完成 3272 步且每步成本为正：`hold_300` 的累计成本为 `798.420595`、累计倒数为 `29280.013533`；`load_follow` 为 `1074.639200`、`145747.460165`。直接倒数偏好后者，虽然其当前定义的完整经济成本更高。两者末端 SOC 不同，费用已按原合同包含 signed SOC 修正；不是声称末端 SOC 相同，也不是新实时模型的比较。
3. FC=70 kW、load=0 的充电例子中，固定参考值抵扣/实际氢耗费用约为 `1.061992`。旧 SOC 项是参考价值势差，因此这不等于原物理能量不守恒或源码重复计费；它说明该项不能直接解释为本步实际充电氢气分配。
4. 历史氢成本库存法可做到分配充电氢气不超过实际耗氢，FC 关闭放电也不除以 eta(0)。但必须确定初始储能价值、能量池定义及新增库存状态的观测/恢复方式。相同 history5 可对应不同库存价值和不同即时成本，不能悄悄假定已有输入完整表达它。

1120 个前缀与 6544 个完整轨迹步是**本次新生成的有界审计数据**，不是既有已训练控制器的轨迹。没有学习或更新模型。既有 Train 累计汇总与稀疏失败记录不能复原全部 35 条历史轨迹的逐步完整成本及累计倒数；这部分保持缺失，而非从汇总或 Validation 补算。

## 已验证的 PowerShell 命令

设置环境与依赖检查：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH = Join-Path (Get-Location).Path 'src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONUTF8 = '1'
$v6Python = Join-Path (Get-Location).Path '.venv-optuna\Scripts\python.exe'
& $v6Python -m pip check
& $v6Python scripts/audit_inverse_economics.py --help
```

审计 Smoke（35 条 Train 各 1 步；**不是新奖励训练**，单独输出以免覆盖已提交证据）：

```powershell
& $v6Python scripts/audit_inverse_economics.py --prefix-steps 1 --output 'outputs/inverse_audit_smoke'
```

复现全部有界审计（最多 35×32 + 2×3272 步，无学习）：

```powershell
& $v6Python scripts/audit_inverse_economics.py --prefix-steps 32 --full-shortest --output 'outputs/inverse_audit_reproduction'
```

查看已提交的经济审计与数学方案数值：

```powershell
$inverseAudit = Get-Content 'docs/audits/v6_inverse_reward_20261029/pre_audit.json' -Raw | ConvertFrom-Json
$inverseAudit.new_prefix_distributions.all
$inverseAudit.full_runs | Select-Object policy, steps, completed, economic_cost_cny, offline_inverse_sum_before_terminal_replacement
& $v6Python tests/test_realtime_value_design.py
```

运行本次审计测试：

```powershell
& $v6Python -m pytest tests/test_inverse_pre_audit.py tests/test_realtime_value_design.py tests/test_inverse_reward_analysis.py -q
```

## 请求的训练命令为何没有交付

| 请求 | 当前状态 |
| --- | --- |
| `inverse_economic` 训练 Smoke | 未接入生产；以上只有严格标记的审计 Smoke |
| 实时模型生产配置 | 尚未选定初始价值、出库估值与状态约定；文档/数学案例不等于已部署模型 |
| 新奖励 500 episode、暂停/恢复、Validation | 审计门禁未通过，因此没有该模式的可运行命令 |
| 新奖励单 Trial、Optuna 搜索/恢复 | 同上，未启用任何隐藏 fallback |
| 新奖励最佳参数导出、独立重训/比较 | 没有新奖励 study 或训练结果，不伪造参数、checkpoint 或比较 |

既有 PR #2 命令及历史模式仍原样保留；它们运行的是旧奖励，不能充当本次新模式命令。没有运行正式 500 episode、长搜索或 Test。

## 继续改造前需要确定的设计

- **零成本正常转移的奖励定义。** 严格 `1/C`、保留这些合法动作和允许 `C=0` 三项不能同时得到处处有定义的普通实数奖励。必须由用户明确修改其中的设计条件；本次未擅自放宽。
- **放电储能的价值约定及必要状态。** 若选历史成本库存，须确定初始能量池/成本基准及是否允许把库存价值加入观测；若选边际补能/完整势函数，须确定有依据的补能参考或价格过程，并明确重估项。当前 eta(P_FC) 本身不能唯一决定这些选择。
- **接受倒数与原经济目标不等价。** 即使解决定义域问题，正成本时最大化累计倒数也不是最小化累计费用；固定终止 −1 也不保证所有策略比较中完成优先。不能用新的隐藏惩罚来消除这些已知局限。
