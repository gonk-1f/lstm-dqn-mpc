# AGENTS.md

## Repository boundary

- 唯一有效的 Git 工作目录是包含本文件和 `.git/` 的内层仓库根目录。外层目录只是容器，不得把它初始化为仓库，也不得在其中新增、移动或复制项目文件。
- 旧 `microgrid-mpc-master` 已归档并默认只读。除非用户明确指定迁移某个文件，否则不得读取后修改、同步或复用其中的工作树。
- 禁止 `git init`、移动/复制/删除 `.git`、强制推送、`git reset --hard`、`git clean -fd` 或覆盖整个工作树。

## Professional evidence rule

- 从已验证的项目事实出发，不迎合错误假设；证据不足时明确说“不确定/待核验”。
- 不虚构船舶参数、数据来源、模型行为、指标或实验结论。求解器 success 不等于控制行为合格。
- 优先读取 `STATUS.md` 和任务直接相关文件，必要时再检查历史产物；不凭文件名推断用途。

## Start-of-task checks

在修改前从仓库根目录运行并核对：

```powershell
git rev-parse --show-toplevel
git status --short --branch
git remote -v
git branch --show-current
```

常规任务开始时先 `git pull --rebase origin main`；若工作树不干净、远端/分支异常或发生冲突，停止有风险的 Git 操作并报告，不得自行覆盖用户修改。

## Scientific source of truth

- `STATUS.md` 是当前状态的唯一入口；`thread.md`、`project_status.md`、`next_steps.md` 仅保留历史上下文，不继续追加为事实源。
- 目标论文主线是：实船负荷 -> LSTM 6 步预测 -> `N=6` 凸 QP-MPC -> OSQP -> SineKAN-DQN 选择 `q_h2/q_soc/q_batt` -> 用实际施加功率更新 SOC。
- `N=60` 仅为 1 s 离线求解器/控制 benchmark，不得与正式 `N=6` 预测时域混用。
- 1 s natural-clipped 数据由 30 s 航段离线样条重构并使用未来端点，不得描述为真实在线 1 s 实测数据。
- 当前物理候选为 `P_fc_max=560 kW`、`E_batt=693 kWh`、`|P_batt|<=346.5 kW`、`SOC_ref=0.55`、`SOC_min=0.2`、`SOC_max=0.8`、FC ramp `48 kW/s`。
- `q_h2=0.5, q_soc=2.0, q_batt=0.05, q_ramp=0, q_terminal_soc=0` 只是 `N=60` 离线 benchmark 的暂定候选，不得自动提升为全局或论文正式权重。
- 旧 `277.2 kWh`、`1806 kWh`、直接功率动作、左右侧功率动作和选择 `q_ramp` 的 DQN 分支均按历史/待迁移处理，除非任务明确要求审计它们。
- 固定 `N=6` LSTM-QP-MPC 未被正式接受前，不得启动正式 DQN 训练。KAN/SineKAN 是 Q 网络类型，不是独立控制层。
- 不得凭文件名、求解器 success 或已有输出宣称科学结论；核对代码、配置、数据来源、闭环行为和指标后再下结论。

## Paper metric policy

- 论文主指标使用物理量：Dp0 曲线氢耗、`P_fc_std`、`SOC_end_minus_start`/SOC band、`soc_slope_std`/`soc_step_max_abs`、电池放电与 throughput、求解时间和失败/fallback。
- `battery_throughput_kwh` 只是电池使用/循环 proxy，不是经验证的寿命指标；ramp 和 action switch 主要作为诊断量。
- 不把 `paper_score_no_ramp` 当论文结果，不把环境累计 `total_cost` 当货币成本，也不以训练 reward 替代物理评价。
- 氢耗使用 `data/fuel_cell/FC_Dp0_curve_for_Python.csv` 和 `src/mpc/solvers/fc_dp0_curve.py`；曲线按 560 kW 总 FC 额定功率的相对负载映射，不把来源图中的 100 kW 当船舶系统规模。

## Change discipline

- 只修改当前任务涉及的文件，保留用户已有修改；优先使用 `rg` 和定向读取。
- 不擅自改变容量、功率边界、SOC 范围、`SOC_band`、数据划分、MPC 权重或控制时序。
- 不删除源代码、历史结果、第三方代码或大文件；清理先记录在 `docs/CLEANUP_CANDIDATES.md`，取得明确授权后再执行。
- 不提交 `venv/`、`.venv/`、缓存、日志、临时目录、编辑器个人设置、凭据或无关大型输出。已被 Git 跟踪但应忽略的内容先记录，不在无关任务中执行 `git rm`。
- 新代码应保持模块边界、单位和功率正负号清晰，并为关键数据/控制不变量增加测试。

## Verification and Git completion

- 修改代码后运行最小相关测试；完成完整功能后运行相称的回归测试。文档任务至少运行 `git diff --check`、路径/命令/参数一致性检查和 `git status --short`。
- 在声称完成前读取实际命令输出。测试失败、实验未完成、Git 冲突、远端异常或验证不足时不得 push；保留本地修改并报告具体阻塞。
- 只暂存本任务文件，使用明确 commit 信息。提交/推送前再次获取远端并确认没有需要处理的更新；禁止 force push。
- 每次结束必须报告：工作目录、分支/远端、修改文件、验证命令与结果、commit hash、push 状态、远端是否更新，以及仍未解决的阻塞项。
