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
| 原始 30 s 实船航段 | `total_load_excels/` 中 66 个 Excel 航段 | 原始依据与 1 s 数据构建 | 实测数据；按航次分组，不应跨航次建窗 |
| 三次样条 1 s 数据 | 由 30 s 航段逐航次离线重构，保存在 `outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/` | OSQP 离线控制输入 | 使用未来端点，不能称为真实在线 1 s 实测数据 |
| 毫秒原始数据与 10 ms 抽点数据 | `data/millisecond_1ms/` 与 `data/millisecond_10ms/` | 数据处理与审计 | 负载尺度与船舶主线不同，不是 MPC/DQN 默认数据源 |

详细来源、字段、划分和泄露边界见 `docs/DATA_PROVENANCE.md`。

## Active scientific configuration

下表是当前论文目标/1 s 离线 benchmark 的科学配置边界，并不表示端到端框架已经验收。

| 项目 | 当前约定 |
| --- | --- |
| 航段 | 66 个原始航段；审计排除 16 个异常航段后，50 个正常航段按时间顺序划分为 train/validation/test = 35/10/5 |
| benchmark 采样间隔 | 1 s（离线 spline 重构） |
| 1 s 数据 | 30 s 实船数据的 natural cubic spline 离线重构并做非负裁剪 |
| LSTM | direct multi-output；history = 30，prediction = 6 |
| 燃料电池 | `P_fc_max = 560 kW` |
| 电池 | `E_batt = 693 kWh`，`|P_batt| <= 346.5 kW`，归一化参考 `P_batt_ref = 346.5 kW` |
| SOC | `SOC_ref = 0.55`，`SOC_min = 0.2`，`SOC_max = 0.8`，当前 693 kWh benchmark 的 `SOC_band = 0.05` |
| 燃料电池爬坡 | `48 kW/s` 硬约束 |
| MPC 求解 | 凸 QP，OSQP，固定稀疏结构、参数更新、warm start 与等价数值缩放 |
| `N=6` 四目标固定控制 | offline oracle 使用可获得的真实样条点 `t+1..t+6`，航段尾部同航段末样本 edge-hold，每次只执行第一步；固定 candidate_C，不使用 LSTM 或 DQN |
| DQN 目标接口 | SineKAN Q 网络选择 `q_h2`、`q_soc`、`q_batt` |

历史 `N=60` 结果已清理；`benchmark_mpc_qp_osqp_1s.py` 仅因当前入口直接复用其中的 OSQP 辅助函数而保留。当前唯一的正式 `N=6` 离线入口是 `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`，其归一化目标为：

```text
J = q_h2    * sum[k=0..5] m_H2(P_fc[k]) / m_H2(560 kW, 1 s)
  + q_batt  * sum[k=0..5] (P_batt[k] / 346.5 kW)^2
  + q_soc   * sum[k=1..6] ((SOC[k] - 0.55) / 0.05)^2
  + q_fc_var * (((P_fc[0] - P_fc_prev) / 48 kW)^2
                + sum[k=1..5] ((P_fc[k] - P_fc[k-1]) / 48 kW)^2)
```

氢耗项使用单一参考 `m_H2(560 kW, 1 s)=0.00883945296644347 kg/step`；其余三个归一化参考是 `346.5 kW` 电池功率、`SOC_ref=0.55` 与 `SOC_band=0.05`、以及 `48 kW/step` 燃料电池变化量。唯一固定权重为 `q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。旧 17 组 one-factor 配置、CLI 分支和产物已删除。

## Repository structure

| 路径 | 内容 |
| --- | --- |
| `src/forecasting/` | 30 s、1 s 和 10 ms LSTM、scaler、窗口与评价逻辑 |
| `src/main/` | 30 s/1 s/毫秒数据构建与审计，以及固定 `N=6` OSQP 入口 |
| `src/main/mpc_solvers/` | 1 s 凸 QP 形式与约束结构 |
| `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` offline-oracle runner；只运行固定 candidate_C |
| `tests/test_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` 四目标 focused test；冻结目标、时序、物理更新、产物与 CLI 契约 |
| `src/mpc/` | 历史 CasADi/IPOPT 控制器、燃料电池氢耗曲线等组件 |
| `src/dqn/` | DQN agent、动作映射、奖励和 MLP/KAN/SineKAN Q 网络 |
| `src/envs/` | 多个历史 DQN/船舶环境；尚未统一为目标 N=6 QP-MPC 环境 |
| `configs/` | 旧通用配置；部分容量、SOC 和时域参数已过期，不是当前唯一事实来源 |
| `outputs/config/` | 两份保留的数据划分 JSON |
| `outputs/mpc_1s_n6_candidate_C/` | 当前固定权重在新测试集 5 航段上的结果与图 |
| `tests/` | 保留数据构建、数据审计和当前 MPC 直接相关测试 |
| `SineKAN-main/` | 复制的第三方 SineKAN 代码和 notebook，许可证仍待核验 |
| `docs/` | 数据来源、接口说明和数据审计文档 |

当前状态入口是 `STATUS.md`。

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

以下入口均来自当前保留脚本。

| 工作流 | 可核对入口 | 状态 |
| --- | --- | --- |
| 66 航段数据构建/检查 | `python src/main/build_total_load_dataset_721.py --help` | 可用；原始构建默认 46/13/7，活动审计划分以 JSON 的 35/10/5 为准 |
| 1 s spline 数据诊断 | `python src/main/build_spline_1s_diagnostics.py --help` | 可用；仅离线重构 |
| 10 ms 数据审计 | `python src/main/audit_millisecond_10ms_dataset.py --help` | 可用；严格校验 train/validation/test assignment key 集合 |
| 固定 candidate_C `N=6` 四目标 MPC | `python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline` | 仅固定权重；offline oracle，不接 LSTM/DQN，只执行第一步 |

本轮最低验证命令：

```powershell
python -m compileall src
python -m unittest tests.test_mpc_1s_n6_four_objective_sensitivity tests.test_mpc_solver_benchmark_1s
```

本轮验证结果见 `STATUS.md`；按清理任务要求不运行全仓库长时间测试或重新执行权重实验。

## Current status

- 66 航段的 30 s 原始数据和 natural-clipped 1 s 离线数据均保留；16 个明确异常航段只从活动划分中排除，未删除源文件。
- 唯一 `N=6` runner 只暴露 candidate_C：`q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。
- candidate_C 保存新测试集 5/5 航段指标和 5 张正式功率/SOC 图；求解失败、primal infeasible 和 max-iter 均为 0。
- 旧 A/B、17 组 one-factor、N=60 结果、smoke/临时产物和已废弃入口已清理。
- 全部 66 航段审计见 `docs/VOYAGE_DATA_QUALITY_AUDIT.md`；`voyage_060` 与另外 15 个明确异常航段均不进入训练、验证或测试。

## Known limitations

- 1 s spline 数据依赖相邻 30 s 节点，是离线重构，不具备在线因果性。
- candidate_C 是当前固定配置，不代表已证明全局最优；本轮没有进行任何权重搜索。
- LSTM 的 6 步预测尚未接入已验证的 `N=6` 时序执行路径。
- OSQP benchmark 的求解失败路径只记录失败，尚未形成可部署的控制回退策略。
- 现有 DQN 分支互不兼容，尚未形成目标状态、动作、奖励和闭环环境。
- SineKAN-DQN 尚无目标环境下的最终训练结果，也没有与 MLP-DQN/KAN-DQN 的同预算、多随机种子公平对比。
- 当前依赖清单仍不完整，部分保留 manifest 含旧绝对路径。

## Reproducibility

- 30 s 航次划分：`outputs/config/voyage_split_total_load_721.json`。
- 10 ms 原子序列划分：`outputs/config/millisecond_10ms_split_721.json`，seed `20260710`，scaler 仅拟合训练行。
- 1 s benchmark 输入：`outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`。
- 固定 MPC 产物：`outputs/mpc_1s_n6_candidate_C/`。
- candidate_C 配置记录当前源码内容、活动划分和当前 parquet 三类 SHA-256；实际测试航段为 `voyage_061, 063, 064, 065, 066`。
- 依赖未锁定、历史绝对路径和第三方许可证问题仍使“干净环境完全可复现”结论不成立。

## Citation and third-party code

`SineKAN-main/` 来自第三方 SineKAN 项目，目录内 README 指向论文预印本和上游仓库；项目网络 `src/dqn/networks/sine_kan_qnet.py` 会导入其中的实现。该副本没有发现许可证文件，许可证兼容性和最终引用格式必须在发布或精简前核验。

仓库同时包含传统 MLP、外部 `pykan` 风格 KAN 和本地 KAN-v2 Q 网络实现。任何论文比较都应在相同状态、动作、奖励、训练步数、随机种子集合与评价航次上进行；当前尚无满足该条件的最终比较结果。
