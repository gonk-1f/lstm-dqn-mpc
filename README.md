# 基于 LSTM、凸 QP-MPC 与 SineKAN-DQN 的燃料电池/锂电池混合动力船舶能量管理

## Research objective

本项目面向论文级船舶上层能量管理研究：利用负荷预测与滚动优化，在燃料电池功率、电池功率、氢耗、SOC 安全和计算实时性之间建立可复现实验。研究范围止于上层功率参考与 SOC 状态更新，不包含 DC/DC 开关、PWM、电流内环、母线电压环或电机电磁暂态控制。

仓库同时保留了多条历史研究分支。下文把“目标论文框架”“当前可验证的离线模块”和“尚未打通的环节”分开描述；存在代码或输出不等于已经形成可用于论文结论的端到端证据。

## Method overview

```text
load data
  -> LSTM multi-step forecasting
  -> convex QP-MPC
  -> OSQP solve
  -> SineKAN-DQN weight selection
  -> applied power and closed-loop SOC update
```

目标控制接口中，DQN 只选择 MPC 的 `q_h2`、`q_soc`、`q_batt`，不直接输出燃料电池或电池功率。当前仓库尚未把上述目标接口统一成一条正式可运行链路；已有 DQN 环境包含旧参数和不同动作定义，不能直接视为目标实现。

## Data lines

| 数据线 | 来源与采样 | 当前用途 | 事实边界 |
| --- | --- | --- | --- |
| 原始 30 s 实船航段 | `total_load_excels/` 中 66 个 Excel 航段 | 原始依据、30 s LSTM、历史 CasADi LSTM-MPC | 实测数据；按航次分组，不应跨航次建窗 |
| 三次样条 1 s 数据 | 由 30 s 航段逐航次离线重构，保存在 `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` | 1 s 预测诊断、OSQP 求解器和离线控制 benchmark | 使用未来端点，不能称为真实在线 1 s 实测数据 |
| 毫秒原始数据与 10 ms 抽点数据 | `data/millisecond_1ms/` 与 `data/millisecond_10ms/` | 高频短时 LSTM 辅助实验 | 负载尺度与船舶主线不同，不是 MPC/DQN 默认数据源 |

详细来源、字段、划分和泄露边界见 `docs/DATA_PROVENANCE.md`。

## Active scientific configuration

下表是当前论文目标/1 s 离线 benchmark 的科学配置边界，并不表示端到端框架已经验收。

| 项目 | 当前约定 |
| --- | --- |
| 航段 | 66 个；按时间顺序划分为 train/validation/test = 46/13/7 |
| benchmark 采样间隔 | 1 s（离线 spline 重构） |
| 1 s 数据 | 30 s 实船数据的 natural cubic spline 离线重构并做非负裁剪 |
| LSTM | direct multi-output；history = 30，prediction = 6 |
| 燃料电池 | `P_fc_max = 560 kW` |
| 电池 | `E_batt = 693 kWh`，`|P_batt| <= 346.5 kW`，归一化参考 `P_batt_ref = 346.5 kW` |
| SOC | `SOC_ref = 0.55`，`SOC_min = 0.2`，`SOC_max = 0.8`，当前 693 kWh benchmark 的 `SOC_band = 0.05` |
| 燃料电池爬坡 | `48 kW/s` 硬约束 |
| MPC 求解 | 凸 QP，OSQP，固定稀疏结构、参数更新、warm start 与等价数值缩放 |
| `N=6` 四目标灵敏度 | offline oracle 使用真实样条点 `t+1..t+6`，每次只执行第一步；不使用 LSTM 或 DQN |
| DQN 目标接口 | SineKAN Q 网络选择 `q_h2`、`q_soc`、`q_batt` |

`N=60` 只作为历史 1 s 离线 OSQP solver/performance benchmark；论文目标 LSTM 预测时域是 `N=6`，二者不得作为同一个默认配置。当前唯一的正式 `N=6` 离线入口是 `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`，其归一化目标为：

```text
J = q_h2    * sum[k=0..5] m_H2(P_fc[k]) / m_H2(560 kW, 1 s)
  + q_batt  * sum[k=0..5] (P_batt[k] / 346.5 kW)^2
  + q_soc   * sum[k=1..6] ((SOC[k] - 0.55) / 0.05)^2
  + q_fc_var * (((P_fc[0] - P_fc_prev) / 48 kW)^2
                + sum[k=1..5] ((P_fc[k] - P_fc[k-1]) / 48 kW)^2)
```

氢耗项使用单一参考 `m_H2(560 kW, 1 s)=0.00883945296644347 kg/step`；其余三个归一化参考是 `346.5 kW` 电池功率、`SOC_ref=0.55` 与 `SOC_band=0.05`、以及 `48 kW/step` 燃料电池变化量。baseline 为 `q_h2=q_batt=q_soc=q_fc_var=1`；one-factor 矩阵对每一项分别使用 `0.25, 0.5, 1, 2, 4`，共享全 1 baseline，共 17 个唯一配置。该流程不自动计算 best/score/rank/winner，也不接受最终权重；baseline 和完整 17 配置当前均为 **未运行**。

## Repository structure

| 路径 | 内容 |
| --- | --- |
| `src/forecasting/` | 30 s、1 s 和 10 ms LSTM、scaler、窗口与评价逻辑 |
| `src/main/` | 数据构建、训练、诊断、CasADi LSTM-MPC 和 OSQP benchmark 入口 |
| `src/main/mpc_solvers/` | 1 s 凸 QP 形式与约束结构 |
| `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` offline-oracle baseline/one-factor runner；固定 17 配置、第一步执行、指标、图和报告 |
| `tests/test_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` 四目标 focused test；冻结目标、时序、物理更新、产物与 CLI 契约 |
| `src/mpc/` | 历史 CasADi/IPOPT 控制器、燃料电池氢耗曲线等组件 |
| `src/dqn/` | DQN agent、动作映射、奖励和 MLP/KAN/SineKAN Q 网络 |
| `src/envs/` | 多个历史 DQN/船舶环境；尚未统一为目标 N=6 QP-MPC 环境 |
| `configs/` | 旧通用配置；部分容量、SOC 和时域参数已过期，不是当前唯一事实来源 |
| `outputs/config/` | 可复查的数据划分、动作表和历史权重配置 |
| `outputs/` | 模型、报告和 benchmark 产物；包含当前证据与历史结果 |
| `tests/` | 数据、LSTM、MPC、OSQP 和闭环相关单元测试 |
| `SineKAN-main/` | 复制的第三方 SineKAN 代码和 notebook，许可证仍待核验 |
| `docs/` | 接口说明、实验协议及本轮建立的项目地图和审计文档 |

模块级状态见 `docs/PROJECT_MAP.md`；当前唯一状态入口是 `STATUS.md`。

## Installation

本仓库当前没有锁文件或 `pyproject.toml`。`requirements.txt` 是旧依赖清单：它使用已不推荐的包名 `sklearn`，且漏列代码实际使用的 `osqp`、`pyarrow` 和 `optuna`，因此不能把它宣称为已验证的一键复现环境。

已审计环境使用 Python 3.11。新环境可先创建并安装当前代码直接涉及的依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install casadi numpy pandas matplotlib xlrd scipy seaborn progressbar2 statsmodels scikit-learn PyYAML torch osqp pyarrow optuna
```

安装后先运行测试；依赖锁定和干净环境验证仍是 P3 任务。

## Main workflows

以下入口均来自仓库实际脚本。先用 `--help` 核对参数和输出位置；训练和历史完整 benchmark 可能耗时。

| 工作流 | 可核对入口 | 状态 |
| --- | --- | --- |
| 66 航段数据构建/检查 | `python src/main/build_total_load_dataset_721.py --help` | 可用；默认 46/13/7 |
| 1 s spline 数据诊断 | `python src/main/build_spline_1s_diagnostics.py --help` | 可用；仅离线重构 |
| 10 ms 数据审计 | `python src/main/audit_millisecond_10ms_dataset.py --help` | 可用；严格校验 train/validation/test assignment key 集合 |
| 30 s LSTM 训练与测试汇总 | `python src/main/run_train_lstm_total_load_721.py --help` | 可用；已有 checkpoint 和逐 horizon 指标 |
| 1 s LSTM 诊断 | `python src/main/run_lstm_spline_1s_hparam_search.py --help` | 辅助实验；现有 LSTM 未超过简单基线 |
| 1 s OSQP `N=60` benchmark | `python src/main/benchmark_mpc_qp_osqp_1s.py --help` | historical；不再作为默认配置或继续搜索 |
| `N=6` 四目标全 1 baseline | `python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline` | **未运行**；offline oracle；不接 LSTM/DQN；只执行第一步 |
| `N=6` 四目标 17 配置 one-factor | `python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --one-factor` | **未运行**；不自动选择 best 或最终权重 |
| 30 s CasADi LSTM-MPC | `python src/main/run_lstm_mpc_total_load_test.py --help` | 历史/支持性链路，参数体系与目标 OSQP 主线不同 |
| 目标 1 s LSTM + `N=6` OSQP 闭环 | 尚无统一入口 | **not yet unified**；现有 N=6 入口仅使用理想预知负荷 |
| 目标 DQN 训练与公平比较 | 无可接受入口 | **not yet unified**；现有脚本仍使用旧环境/动作/参数 |

通用测试命令：

```powershell
python -m unittest discover -s tests -v
```

本轮已运行的测试结果见 `STATUS.md`：`N=6` focused test 为 43/43，保留 `N=60` benchmark test 为 17/17；完整 suite 尚未重跑，将在 Task 9 执行。测试范围和仍缺失的 DQN 专项覆盖见 `docs/PROJECT_MAP.md`。

## Current status

- 66 航段的 30 s 数据读取、46/13/7 划分和 30 s direct multi-output LSTM 路径已存在。
- natural-clipped 1 s 离线数据、1 s LSTM 诊断、历史 `N=60` benchmark，以及唯一的 `N=6` 四目标 runner/focused test 均已存在。
- 现有 1 s LSTM 在保留测试集上没有超过 current-hold/last-slope 等简单基线，因此不能作为正式预测优势证据。
- 全 1 baseline 与 17 配置 one-factor 正式结果均 **未运行**；`outputs/mpc_1s_n6_four_objective_sensitivity/` 和两份 `reports/mpc_1s_n6_four_objective_sensitivity_*` 当前不存在，不能报告数值或趋势。
- 目标 `N=6` LSTM-OSQP 在线闭环、正式接受的固定权重、仅选三项 MPC 权重的 DQN 环境、SineKAN-DQN/MLP-DQN 公平比较均未完成。
- 旧 `277.2 kWh`、`1806 kWh` 及不同动作空间仍存在于历史实现和输出中，但已不代表当前目标配置。

## Known limitations

- 1 s spline 数据依赖相邻 30 s 节点，是离线重构，不具备在线因果性。
- 四目标 baseline 和 one-factor 矩阵尚未运行；在读取完整物理指标和逐航次曲线并完成人工审查前，不能声称任何趋势、下一搜索区间或最终最优权重。
- LSTM 的 6 步预测尚未接入已验证的 `N=6` 时序执行路径。
- OSQP benchmark 的求解失败路径只记录失败，尚未形成可部署的控制回退策略。
- 现有 DQN 分支互不兼容，尚未形成目标状态、动作、奖励和闭环环境。
- SineKAN-DQN 尚无目标环境下的最终训练结果，也没有与 MLP-DQN/KAN-DQN 的同预算、多随机种子公平对比。
- 当前依赖清单不完整，仓库包含大量已跟踪输出和本地工具临时产物。

## Reproducibility

- 30 s 航次划分：`outputs/config/voyage_split_total_load_721.json`。
- 10 ms 原子序列划分：`outputs/config/millisecond_10ms_split_721.json`，seed `20260710`，scaler 仅拟合训练行。
- 30 s LSTM 默认 seed 为 `42`；1 s Task C 诊断 seed 为 `123`。
- 训练窗口必须按航次/原子序列构造，禁止跨边界；scaler 只在训练集拟合。
- 已存在的主要产物位于 `outputs/lstm_total_load_721/`、`outputs/lstm_spline_1s_hparam_search/` 和 `outputs/mpc_solver_benchmark_1s/`。`N=6` 四目标运行将写入 `outputs/mpc_1s_n6_four_objective_sensitivity/`、`reports/mpc_1s_n6_four_objective_sensitivity_summary.md` 与 `reports/mpc_1s_n6_four_objective_sensitivity_table.csv`；三者当前均不存在。
- 运行前保存配置、随机种子、Git commit、输入 manifest 和逐航次/逐 horizon 指标；不要只保留聚合图。
- 本轮 focused 与保留 `N=60` 回归测试状态见 `STATUS.md`；依赖未锁定、历史绝对路径和第三方许可证问题仍使“干净环境完全可复现”结论不成立。

## Citation and third-party code

`SineKAN-main/` 来自第三方 SineKAN 项目，目录内 README 指向论文预印本和上游仓库；项目网络 `src/dqn/networks/sine_kan_qnet.py` 会导入其中的实现。该副本没有发现许可证文件，许可证兼容性和最终引用格式必须在发布或精简前核验。

仓库同时包含传统 MLP、外部 `pykan` 风格 KAN 和本地 KAN-v2 Q 网络实现。任何论文比较都应在相同状态、动作、奖励、训练步数、随机种子集合与评价航次上进行；当前尚无满足该条件的最终比较结果。
