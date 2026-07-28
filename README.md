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

目标控制接口中，DQN 只选择 MPC 的完整四权重组合 `q_h2`、`q_batt`、`q_soc`、`q_fc_var`，不直接输出燃料电池或电池功率。当前已实现 7 动作映射、11 维状态、固定评价奖励、7 个持久化 OSQP solver、单航段闭环环境和 MLP-DQN 训练/验证入口，但尚无完成并验收的正式 DQN 训练模型。固定 candidate_C 基准仍直接使用未来 `t+1..t+6` 的真实 1 s 样条负荷，不使用 LSTM 或 DQN。

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
| 航段 | 66 个航段按时间顺序划分为 train/validation/test = 46/13/7；测试集为 `voyage_060` 至 `voyage_066` |
| benchmark 采样间隔 | 1 s（离线 spline 重构） |
| 1 s 数据 | 30 s 实船数据的 natural cubic spline 离线重构并做非负裁剪 |
| LSTM | 保留 direct multi-output 预测模块；当前正式 MPC 不使用 LSTM |
| 燃料电池 | `P_fc_max = 560 kW` |
| 电池 | `E_batt = 624 kWh`，充电下限 `-624 kW`，放电上限 `+1248 kW`，归一化参考 `P_batt_ref = 624 kW`；`P_batt > 0` 表示放电 |
| SOC | `SOC_ref = 0.55`，`SOC_min = 0.2`，`SOC_max = 0.8`，`SOC_band = 0.05` |
| 燃料电池爬坡 | `48 kW/s` 硬约束 |
| MPC 求解 | 凸 QP，OSQP，固定稀疏结构、参数更新、warm start 与等价数值缩放 |
| `N=6` 四目标固定控制 | offline oracle 使用可获得的真实样条点 `t+1..t+6`，航段尾部同航段末样本 edge-hold，每次只执行第一步；固定 candidate_C，不使用 LSTM 或 DQN |
| DQN 目标接口 | A0–A6 分别对应一套完整四权重组合；动作映射、11 维状态、固定奖励、solver bank、闭环环境和 MLP 训练入口均已实现，尚未验收正式训练模型 |

历史 `N=60` 结果已清理；正式入口使用 `src/main/mpc_solvers/osqp_runtime.py` 中的 OSQP 运行辅助函数。当前唯一的正式 `N=6` 离线入口是 `src/main/run_mpc_1s_n6_four_objective_sensitivity.py`，其归一化目标为：

```text
J = q_h2    * sum[k=0..5] m_H2(P_fc[k]) / m_H2(560 kW, 1 s)
  + q_batt  * sum[k=0..5] (P_batt[k] / 624 kW)^2
  + q_soc   * sum[k=1..6] ((SOC[k] - 0.55) / 0.05)^2
  + q_fc_var * (((P_fc[0] - P_fc_prev) / 48 kW)^2
                + sum[k=1..5] ((P_fc[k] - P_fc[k-1]) / 48 kW)^2)
```

氢耗项使用单一参考 `m_H2(560 kW, 1 s)=0.00883945296644347 kg/step`；其余三个归一化参考是 `624 kW` 电池功率、`SOC_ref=0.55` 与 `SOC_band=0.05`、以及 `48 kW/step` 燃料电池变化量。固定 candidate_C baseline runner 的权重为 `q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。旧 17 组 one-factor 配置、CLI 分支和产物已删除。

当前 DQN-MPC 离散动作表保持 7 个动作，权重顺序统一为 `(q_h2, q_batt, q_soc, q_fc_var)`：

| 动作 | 名称 | `q_h2` | `q_batt` | `q_soc` | `q_fc_var` |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | candidate_C | 0.25 | 0.40 | 12 | 20 |
| A1 | fast_fc_response | 0.25 | 0.60 | 12 | 5 |
| A2 | soc_recovery_30 | 0.15 | 0.35 | 30 | 12 |
| A3 | soc_recovery_40 | 0.15 | 0.35 | 40 | 12 |
| A4 | soc_recovery_fast | 0.15 | 0.35 | 30 | 5 |
| A5 | soc_recovery | 0.15 | 0.35 | 20 | 12 |
| A6 | strong_soc_recovery | 0.10 | 0.30 | 40 | 5 |

旧 `fixed_action_coverage.*` 使用旧动作表；新 `fixed_action_coverage_v2.*` 使用上表。A2/A3/A4/A6 的编号语义已经改变，跨版本比较必须同时核对名称和四权重，不能只按 action ID 合并。

## Repository structure

| 路径 | 内容 |
| --- | --- |
| `src/forecasting/` | 30 s、1 s 和 10 ms LSTM、scaler、窗口与评价逻辑 |
| `src/main/` | 30 s/1 s/毫秒数据构建与审计、固定 `N=6` OSQP 入口及 MLP-DQN–MPC 训练/验证入口 |
| `src/main/mpc_solvers/` | 1 s 凸 QP 形式、数值缩放、OSQP 运行层和 7 动作持久化 solver bank |
| `src/main/run_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` offline-oracle runner；只运行固定 candidate_C |
| `tests/test_mpc_1s_n6_four_objective_sensitivity.py` | 唯一 `N=6` 四目标 focused test；冻结目标、时序、物理更新、产物与 CLI 契约 |
| `src/mpc/` | 历史 CasADi/IPOPT 控制器、燃料电池氢耗曲线等组件 |
| `src/dqn/` | DQN agent、7 动作映射、11 维状态、固定奖励和 MLP/KAN/SineKAN Q 网络 |
| `src/envs/` | `dqn_mpc_weight_env.py` 实现四权重选择闭环环境；MPC 执行第一控制步并更新实际功率与 SOC |
| `configs/` | 旧通用配置；部分容量、SOC 和时域参数已过期，不是当前唯一事实来源 |
| `outputs/config/` | 两份保留的数据划分 JSON |
| `outputs/mpc_1s_n6_candidate_C/` | 当前固定权重在测试集 7 航段上的结果与图 |
| `outputs/dqn_mpc_mlp_10k_baseline/` | 非测试集固定动作覆盖、困难航段物理可行性与动作重设计诊断 |
| `tests/` | 数据构建、数据审计、MPC，以及 DQN 动作、状态、奖励、solver bank、环境和 MLP 集成测试 |
| `SineKAN-main/` | 复制的第三方 SineKAN 运行源码和 README，许可证仍待核验 |
| `docs/` | 数据来源、接口说明和数据审计文档 |

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
| 66 航段数据构建/检查 | `python src/main/build_total_load_dataset_721.py --help` | 可用；活动划分为 46/13/7 |
| 1 s spline 数据诊断 | `python src/main/build_spline_1s_diagnostics.py --help` | 可用；仅离线重构 |
| 10 ms 数据审计 | `python src/main/audit_millisecond_10ms_dataset.py --help` | 可用；严格校验 train/validation/test assignment key 集合 |
| 固定 candidate_C `N=6` 四目标 MPC | `python src/main/run_mpc_1s_n6_four_objective_sensitivity.py --baseline` | 仅固定权重；offline oracle，不接 LSTM/DQN，只执行第一步 |
| MLP-DQN–MPC 训练/验证 | `src/main/train_dqn_mpc_mlp.py` | 入口已实现；尚无完成并验收的正式训练模型，不得读取 test 轨迹调参 |

当前 DQN-MPC 关键回归测试：

```powershell
python -m unittest tests.test_dqn_weight_action_table
python -m unittest tests.test_dqn_mpc_solver_bank
python -m unittest tests.test_dqn_mpc_mlp_smoke
python -m unittest tests.test_dqn_mpc_weight_env
```

## Current status

- 66 航段的 30 s 原始数据和 `device_channel_natural_spline_1s` 离线数据均保留；活动划分为 46/13/7。
- 唯一 `N=6` runner 只暴露 candidate_C：`q_h2=0.25`、`q_batt=0.4`、`q_soc=12.0`、`q_fc_var=20.0`。
- candidate_C 保存测试集 7/7 航段指标和 7 张正式功率/SOC 图；求解失败、primal infeasible 和 max-iter 均为 0。
- 旧 A/B、17 组 one-factor、N=60 结果、smoke/临时产物和已废弃入口已清理。
- 当前测试集为 `voyage_060` 至 `voyage_066`；`voyage_060` 不再按旧 BDM 掉零口径排除。
- DQN-MPC 已打通 7 动作、11 维状态、固定评价奖励、持久化 solver bank、闭环环境和 MLP 训练/验证入口，但本轮没有训练 DQN。
- 新动作表在 46 个训练航段和 13 个验证航段上完成 59×7=413 组固定动作检查：350 组完整运行，63 组在首次 `primal infeasible` 停止。
- 9 个旧版全动作失败航段中，A6 救回 `voyage_011`，A3 救回 `voyage_024`；全动作失败航段由 9 个降至 7 个。
- 在 52 个至少有一个动作完整运行的航段中，A6 按固定 candidate_C reward 成为最佳动作 50 次，A3 为 2 次，其他动作为 0 次。该结果说明当前动作空间仍高度偏向 A6，不能据此宣称 DQN 自适应策略已经成立。

## Known limitations

- 1 s spline 数据依赖相邻 30 s 节点，是离线重构，不具备在线因果性。
- candidate_C 是当前固定配置，不代表已证明全局最优；当前分支只在非测试困难航段上进行了有边界的动作表定向重设计，不是全局权重搜索。
- LSTM 的 6 步预测尚未接入已验证的 `N=6` 时序执行路径。
- OSQP benchmark 的求解失败路径只记录失败，尚未形成可部署的控制回退策略。
- DQN-MPC 代码路径已经实现，但尚无完成并验收的正式训练模型；7 个固定动作仍有 7 个非测试航段全部失败，且 A6 在固定评价中占据 50/52 次最佳，动作区分度不足。
- SineKAN-DQN 尚无目标环境下的最终训练结果，也没有与 MLP-DQN/KAN-DQN 的同预算、多随机种子公平对比。
- 当前依赖清单仍不完整，部分保留 manifest 含旧绝对路径。

## Reproducibility

- 30 s 航次划分：`outputs/config/voyage_split_total_load_721.json`。
- 10 ms 原子序列划分：`outputs/config/millisecond_10ms_split_721.json`，seed `20260710`，scaler 仅拟合训练行。
- 1 s benchmark 输入：`outputs/mpc_solver_benchmark_1s/data/test_voyages_spline_1s.parquet`。
- 固定 MPC 产物：`outputs/mpc_1s_n6_candidate_C/`。
- candidate_C 配置记录生成时的源码内容、活动划分和当前 parquet 三类 SHA-256；实际测试航段为 `voyage_060` 至 `voyage_066`。
- 旧动作表覆盖：`outputs/dqn_mpc_mlp_10k_baseline/fixed_action_coverage.csv` 与 `fixed_action_coverage_summary.json`。
- 新动作表覆盖：`outputs/dqn_mpc_mlp_10k_baseline/fixed_action_coverage_v2.csv` 与 `fixed_action_coverage_v2_summary.json`。
- 困难航段证据：同目录下 `hard_voyage_physical_feasibility.*`、`hard_voyage_dynamic_reference.*` 和 `hard_voyage_weight_redesign.*`。
- 动作重设计和 59×7 覆盖仅使用 train/validation 航段，没有读取 test 轨迹；现有环境回归测试单独使用 `voyage_064` 验证 A0 与正式 candidate_C 链路的一致性。
- 依赖未锁定、历史绝对路径和第三方许可证问题仍使“干净环境完全可复现”结论不成立。

## Citation and third-party code

`SineKAN-main/` 来自第三方 SineKAN 项目，目录内 README 指向论文预印本和上游仓库；项目网络 `src/dqn/networks/sine_kan_qnet.py` 会导入其中的 `sine_kan.py`。该副本没有发现许可证文件，许可证兼容性和最终引用格式必须在发布或精简前核验。

仓库同时包含传统 MLP、外部 `pykan` 风格 KAN 和本地 KAN-v2 Q 网络实现。任何论文比较都应在相同状态、动作、奖励、训练步数、随机种子集合与评价航次上进行；当前尚无满足该条件的最终比较结果。
