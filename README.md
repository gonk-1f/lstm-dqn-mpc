# Causal DQN-MPC 船舶混合能源管理

当前正式方法为 **`executed_closed_loop_reward`**：DQN 从固定 84 组 MPC 权重中选择一组，MPC 分配燃料电池与电池功率，RL 使用统一的实际执行闭环性能评分。代码已更新；本轮没有训练 DQN，也没有读取真实 Validation/Test 负荷。失败惩罚尚待标定，正式训练入口默认阻止运行。

## 状态、动作与时序

状态仅包含当前/历史测量，固定顺序如下。所有输入分量均无量纲；历史不足 10/60 个点时只使用已经获得的样本。

| 维度 | 定义 | 含义 |
|---|---|---|
| 1 | `(SOC_t - 0.55) / 0.05` | 当前 SOC 相对参考值的偏差 |
| 2 | `P_fc,t-1 / 600` | 上次执行的 FC 功率 |
| 3 | `P_batt,t-1 / 624` | 上次执行的电池功率，放电为正、充电为负 |
| 4 | `P_load,t / 600` | 当前测得的总负荷 |
| 5 | `(P_load,t - P_load,t-1) / 48` | 1 s 内负荷变化 |
| 6 | 最近 10 s 平均负荷 `/ 600` | 短期历史负荷水平 |
| 7 | 最近 60 s 平均负荷 `/ 600` | 较长期历史负荷水平 |

正式 MLP 为 `7 -> 128 -> 64 -> 84`，输出每个动作的 Q 值；KAN 仍是可替换后端，不改变动作和 reward 语义。

四个权重固定按 `(q_H, q_B, q_S, q_F)` 排列，代码字段为 `(q_h2, q_batt, q_soc, q_fc_var)`，分别作用于 MPC 内部预测的氢耗、电池功率平方、SOC 偏差平方和 FC 功率变化平方。

$$n_H+n_B+n_S+n_F=10,\quad n_i\ge1,\quad q_i=n_i/10.$$

正整数四元组数为 $\binom{9}{3}=84$，每个 $n_i\le7$。按 `(n_H,n_B,n_S,n_F)` 字典序生成，ID 为 0–83；A0=`(0.1,0.1,0.1,0.7)`，A83=`(0.7,0.1,0.1,0.1)`。这是完整整数网格，不是经 Pareto 优化或已证明安全的动作集。构造和求和约束均使用整数，不用浮点相等筛选；权重的数学和精确为 1，求解器浮点表示允许机器舍入误差。

动作表版本：`positive_integer_simplex_10_lexicographic_v1`。模型、训练状态和 replay 保存完整 action table、整数权重、reward version 与 `switch_seconds`。旧 4 动作 checkpoint/replay **不能继续训练当前方法**；即使旧 replay 中动作 ID 恰好在 0–83 之间，也不能静默加载。变更排序、reward 或缺失语义元数据均拒绝加载。

MPC 的 `Ts=1 s`、`N=6`，每秒重求解，只执行第一步。DQN 的 `T_sw=1 s`，每秒重新选权重。保留仓库原有因果推进规则：数组索引 `t` 的状态观察当前负荷，MPC 使用该负荷作 `t+1...t+6` 的 persistence forecast，随后环境执行到下一测量点；下一点真实负荷仅用于模拟执行中的电池功率平衡，不进入此前的 state/MPC forecast。

```text
state_t -> action_t -> q_action -> N=6 MPC
        -> 执行首步FC，电池补足实际负荷差
        -> 计算SOC_next并检查现有物理约束
        -> 计算本次实际执行reward -> state_next -> replay
```

公式里的 `P_fc,t/P_batt,t` 指 transition t 实际执行的功率；日志明确区分 `decision_index=t` 和 `execution_index=t+1`。这一区别不会把未来真实负荷提供给 DQN。当前不启用 LSTM，不改数据划分或 MPC 输入时序。

## 实际执行 reward

$$h_t=\frac{\dot m_{H_2}(P_{fc,t})}{h_E},\quad h_E=\dot m_{H_2}(600),$$
$$b_t=\frac{P_{batt,t}}{624},\quad s_t=\frac{SOC_{t+1}-0.55}{0.05},\quad f_t=\frac{P_{fc,t}-P_{fc,t-1}}{48},$$
$$\boxed{\ell_t=h_t+\tfrac12b_t^2+\tfrac12s_t^2+\tfrac12f_t^2,\qquad r_t=-\ell_t.}$$

这里的标量 `s_t` 是 reward 的 SOC 项，不是 7 维 RL 状态向量。`h,b,s,f,ell,r` 都无量纲。所有 84 个动作共享完全相同的评分；动作权重仅影响 MPC 产生怎样的控制，不进入 reward。`raw_mpc_objective` 和 `mpc_objective_terms` 仍记录为诊断量，未来 6 步计划、预测 SOC 和动作自身 $J_a^*$ 不参与正式 reward。

| 参数/符号 | 单位与含义 | 当前依据及限制 |
|---|---|---|
| $P_{fc,t}$ | kW，当前 transition 实际执行的 FC 功率 | 采用 MPC 首步输出 |
| $P_{fc,t-1}$ | kW，前次已执行 FC 功率 | 构造实际功率变化，初值按首个负荷及 FC 上下限初始化 |
| $P_{batt,t}$ | kW，实际电池功率 | 实际负荷减实际 FC 功率，正放电/负充电 |
| $\dot m_{H_2}$ | g/s，FC 耗氢速率 | 复用 `data/fuel_cell/FC_Dp0_curve_for_Python.csv` 的 Dp0 插值曲线；600 kW 总额定功率映射到原 100 kW 曲线并按容量缩放 |
| 600 | kW，FC 当前额定/最大功率配置，兼作经济参考点 | $h_E\approx9.5146445568$ g/s，为固定量；不随动作或状态重算参考尺度 |
| $h_t$ | 无量纲，实际耗氢相对满参考功率耗氢的比例 | 对**耗氢速率线性**，不再平方；并不声称耗氢与 FC 功率线性 |
| 624 | kW，电池功率参考 | 当前容量 624 kWh，约 1C 对应 624 kW；沿用工程参考 |
| $b_t^2$ | 无量纲，`battery_power_stress_proxy` | 电池充放电功率幅值代理，充/放电对称；不是寿命或热模型认证的最终连续应力阈值 |
| 0.55 | SOC 比例，55%，初始/运行参考 | 当前控制设计选择，不是硬边界或已证明经济最优点 |
| 0.05 | SOC 比例，5 个百分点，$\sigma_S$ | 偏差达 5 个百分点时 $s_t^2=1$；沿用 Train-only 边际审计支持的工程尺度 |
| $s_t^2$ | 无量纲，下一步实际 SOC 偏差平方 | 对 0.55 两侧对称；0.50 与 0.60 的贡献相同 |
| 48 | kW/step（本轮 step=1 s），来自 48 kW/s FC ramp 基准 | 当前配置的物理变化率参考；$f_t^2$ 对升/降功率对称 |
| $1/2$ | 无量纲，三项二次代价的固定系数 | 本轮指定的混合效用约定；它确实影响三项相对线性氢耗的权重，不是经设备验证的天然常数 |
| $T_s,T_{sw}$ | s，分别为 MPC 求解/执行周期与 DQN 选权重周期 | 均为 1；先固定一秒 transition，便于单独验证 reward 更新 |
| $N$ | 6 个预测步，即 6 s | 只影响 MPC 预测，reward 不跨 6 步评分 |
| $\gamma$ | 0.99，无量纲折扣因子 | 沿用 1 s RL 决策配置，表示下一步价值乘 0.99；不是设备物理参数 |
| $d_t$ | 布尔 terminal 标记 | episode 结束或已标定的物理执行失败时，禁止 bootstrap |

MPC 内部仍用 Dp0 的凸二次拟合来维持 QP 结构；reward 用原插值曲线评估实际 FC 功率。两者模型层次不同，不能把 MPC 中的氢耗目标值替代为真实执行评分。

保留的 Train-only SOC 边际审计表明：在旧预测 RMS/L2 评分下，0.05 主要降低低 SOC 状态的绝对价值，未系统性垄断同状态 84 动作的实际控制选择。它支持暂时保留尺度，**不等于已经验证新首步线性氢耗 reward 的训练效果**。原报告与数据链见 [清理归档](docs/diagnostic_cleanup_20260913.md)。后续若获得设备依据明确的 $P_{dis,cont}$、$P_{chg,cont}$，再独立研究超额应力；本轮不虚构阈值。

Bellman target 不变：

$$y_t=r_t+0.99(1-d_t)\max_{a'}Q_{target}(state_{t+1},a').$$

不实现 SMDP 或 $\gamma^{T_{sw}}$。5/10/20 s 权重保持是未来独立消融，不是本次代码功能。

## 为什么选择这个版本

- 不再使用各动作自身 $J_a^*$：不同权重的内部目标不是统一的外部性能标尺，权重和为 1 也不能解决这个问题。
- 暂不使用 ideal/reference：状态相关参考和跨度会改变同一物理表现的计分尺度，并依赖额外求解与参考可行性；当前选择固定可解释尺度。
- 暂不采用全 Gaussian：会把经济氢耗也纳入平方/指数变换，增加饱和效应与尺度解释负担。文献使用 Gaussian 不代表本任务必须沿用。
- 不使用 potential shaping、action-id bonus 或动作均衡奖励：当前目标是直接评估已执行结果，没有授权增加这些额外偏好。

这些是本项目的设计取舍，不是本轮通过新训练得到的优越性结论。

## 物理约束与 failure

现有配置不变：FC 0–600 kW，FC ramp 48 kW/s；电池容量 624 kWh，功率 -624–+1248 kW；SOC hard bounds 0.20–0.80。SOC 连续参考 0.55，尺度 0.05；`q_terminal_soc=0`。求解后、状态提交前检查实际功率、SOC、功率平衡及 ramp，不增加事后 clamp/fallback。

`terminal_failure_penalty` 定义为**正的终止代价**，物理执行失败的奖励为其负值。默认 `None`，`failure_penalty_calibration` 默认 `None`；两者缺失或数值不合法时，正式训练入口在加载数据前报错。该值尚待 **Train-only reward/Q/TD 尺度标定**，本轮不设默认数值。

分类为：

- `physical_execution_violation`：首步实际结果违反现有硬约束；使用显式标定的 terminal penalty，状态不提交，terminal transition 不 bootstrap。
- `forecast_qp_infeasible`：求解器报告预测 QP primal infeasible，不能自动等同于已执行物理失败；停止并保留诊断，不写伪造 reward/replay。
- `numerical_solver_failure`：未收敛、迭代上限等；抛出结构化异常中止当前运行，不把它当动作造成的物理失败，不写失败 transition。已有成功 transition 可保留。

## 历史基线与复现

| 方法标签 | 实现/公式 | 用途 |
|---|---|---|
| `legacy_yuan_like_self_cost` | `src/dqn/utils/legacy_reward.py`，项目旧式 $1/(1+J_a^*)$ | 保留最接近 FCHEV 应用方向的自身 MPC 代价奖励对照；Yuan 原式为 $1/C$，并非完全复刻 |
| `legacy_predicted_common_reward` | 同文件，$1/(1+\sqrt{(H/6)^2+B/6+S/6+F/6})$ | 保留统一预测 RMS/L2 评分消融；输入是旧 6 步未加权 H/B/S/F 累积量 |
| `executed_closed_loop_reward` | `src/dqn/utils/reward.py` | 当前唯一正式主方法 |

旧四动作表为 `LEGACY_FOUR_WEIGHT_ACTIONS`，历史 audit/calibration 脚本显式使用冻结定义或自身网格；不能因为当前默认表改变而悄悄改变旧实验含义。旧 ideal/reference、fixed-physical-L2、RMS 和边际审计脚本及必要结果链保留，见 [逐项 KEEP/DELETE 清单](docs/diagnostic_cleanup_20260913.md)。复现旧已提交控制器可在独立检出中使用清单记载的旧 Git commit；勿用旧 checkpoint 恢复本版正式训练。

## 数据与入口

正式数据：`data/processed/operating_dataset_final/`。权威清单为 `metadata/sample_manifest.csv`、`metadata/parent_split_manifest.csv`。同一 parent voyage 不跨 split；本轮保留 110 Train、27 Validation、8 Test segments，未修改数据或读取真实 Validation/Test 负荷。原始设备记录与预处理数据、manifest 及引用文件均保留。

安装依赖：`python -m pip install -r requirements.txt`。

正式入口：`python src/main/run_dqn_mpc_causal_training.py`。**当前会因未标定 penalty 而停止，这是预期行为。** 标定完成后通过 `--terminal-failure-penalty` 和 `--failure-penalty-calibration` 显式传入实际标定值与证据标识；恢复还需 `--resume-training-state`，配置与语义必须一致。不能随意填一个数绕过标定。推理模型也保存并恢复该配置，独立 replay 会拒绝不同的失败惩罚配置。

新输出目录：`outputs/dqn_mpc_{mlp|kan}_executed_reward_84_v1_formal_rounds/`。每轮保存含语义元数据的推理模型和完整训练状态（online/target、optimizer、replay、epsilon、计数及 RNG）。MLP/KAN 目录和 checkpoint 不混用。独立评估入口 `src/main/test_dqn_mpc_causal.py` 默认要求新命名空间的 round 2 checkpoint。

固定对照入口 `src/main/test_mpc_nominal_causal.py` 当前固定 **grid A0=(0.1,0.1,0.1,0.7)**，不再叫旧 balanced；输出到 `outputs/mpc_fixed_grid_a0_executed_reward_v1_test/`。固定对照的正式评估入口同样要求显式传入上述两个 failure 参数，并在读取数据前校验。这些入口本轮均未执行正式训练或 held-out 评估。

训练默认仍为 MSE TD loss、学习率 `5e-4`（Adam 步长）、`batch_size=64`（每次更新抽取64条transition）、replay 容量 `300000`、seed `42`、warmup `5000`（先积累经验）、target hard-sync 间隔 `500` 步；epsilon 从 `1.0` 按 `0.99999813` 衰减至下限 `0.05`。它们是既有训练超参数，不是设备物理依据或本轮重新标定结果。

## 原文参考与适用范围

1. Zarrouki, Spanakakis & Betz (2024), *A Safe Reinforcement Learning driven Weights-varying Model Predictive Control for Autonomous Vehicle Motion Control*，[原文 §V-A/B、式8](https://arxiv.org/html/2402.02624v1)：离散 Pareto 权重与切换区间实测 RMS reward。其 PPO、Gaussian 和预优化安全集不照搬。
2. Zarrouki et al. (2021), *Weights-varying MPC for Autonomous Vehicle Guidance: a Deep Reinforcement Learning Approach*，[DOI](https://doi.org/10.23919/ECC54610.2021.9655042)：外部 tracking/comfort 多目标 reward，式4/5为 Gaussian，并非本项目混合线性/二次公式。
3. Haspolat & Yalcin (2023), *Energy Management of P2 Hybrid Electric Vehicle Based on Event-Triggered Nonlinear Model Predictive Control and Deep Q Network*，[原文 §4.2](https://doi.org/10.3390/wevj14060135)：DQN 调整 NMPC 权重与效率评价；其事件触发机制未引入本轮。
4. Mehndiratta, Camci & Kayacan (2018), *Automated Tuning of Nonlinear Model Predictive Controller by Reinforcement Learning*，[DOI](https://doi.org/10.1109/IROS.2018.8594350)：按工程误差标准评估权重，式14–16、表II/III；其飞行误差阈值不能替代船舶设备依据。
5. Sun et al. (2024), *Adaptive parameterized model predictive control based on reinforcement learning: A synthesis framework*，[原文 §3.2.2、式13](https://repository.tudelft.nl/file/File_5f230ded-4f1a-456f-b3e4-4337cd8f90ec)：RL 区间测量状态与已执行输入性能评分，明确不同时间尺度。
6. Zhai et al. (2026), *Model-based MPC with adaptive weights for tracked vehicle trajectory tracking*，[原文](https://doi.org/10.1177/16878132261426615)：DQN 在线调整 MPC 权重；履带车跟踪实验不是船舶能量管理参数认证。
7. Yuan et al. (2025), *Energy management and performance improvement for fuel cell hybrid electric vehicle with reinforcement learning-based dual model predictive control*，[DOI](https://doi.org/10.1016/j.ijhydene.2025.151770)：已核对用户本地 PDF 第9页 §3.2.2 式51，奖励是各 MPC 代价倒数。作为重要对照保留，不声称它采用本项目的统一首步 reward。

七篇原文的核对位置与差异详见 [文献核对记录](docs/executed_reward_references_20260913.md)。当前公式是本项目明确选择的综合设计，不是某篇论文的逐式复刻；尤其 `1/2` 相对氢耗的权重没有在这些原文中获得统一设备依据，本轮按要求保留。

## 验证

`tests/test_executed_reward_contract.py` 检查动作/网络、首步公式、未来计划不变性、二次对称、氢耗速率线性、模型/replay 语义拒绝和数值失败隔离。其余单元测试使用合成夹具检查 transition、优化器和恢复过程；这不构成正式 DQN 训练实验。

```powershell
python -X utf8 -m unittest discover -s tests -p 'test_executed_reward_contract.py' -v
python -X utf8 -m unittest discover -s tests -v
python -X utf8 -m compileall -q src tests
git diff --check
```

本轮完整测试另加文件读取拦截，禁止打开真实 Validation/Test payload；最终验证结果见 [执行记录](docs/executed_reward_verification_20260913.md)。
