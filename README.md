# lstm-dqn-mpc

## 基于 DQN-MPC 的混合动力船舶能量管理

本项目面向燃料电池—锂电池混合动力船舶能量管理问题，构建基于 **DQN 高层策略选择 + 凸 QP-MPC 底层功率分配** 的分层能量管理方法。

当前活动版本暂不接入 LSTM 负荷预测，采用完全因果的 DQN 状态输入。MPC 在每个 1 s 控制时刻采用当前负荷保持（current-load persistence）构造 6 s 预测时域，并以滚动优化方式执行第一步控制量。

当前版本已经完成：

- 四动作 DQN-MPC 控制策略设计；
- 600 kW 燃料电池参数统一；
- 7 维因果 DQN 状态构建；
- 工况敏感公共奖励函数设计；
- DQN 正式训练；
- 13 条 validation 航段验证；
- 7 条独立 test 航段测试；
- 测试航段逐秒 trace 输出；
- 功率分配与 SOC 时序图输出。

---

# 1. 系统结构

当前控制结构为：

```text
当前/历史运行状态
        ↓
       DQN
        ↓
选择 MPC 权重动作 A0-A3
        ↓
   QP-MPC (N=6)
        ↓
燃料电池 / 电池功率分配
        ↓
只执行当前第一步控制
        ↓
下一秒重新观测并滚动优化
```

DQN 不直接输出燃料电池功率或电池功率，而是在每个决策时刻选择一组 MPC 目标函数权重。

MPC 根据所选动作对应的权重完成底层功率优化。

---

# 2. 当前物理参数

## 2.1 燃料电池系统

- 最大功率：600 kW
- 最小功率：0 kW
- 最大功率变化率：48 kW/s

## 2.2 锂电池系统

- 额定容量：624 kWh
- 最大充电功率：624 kW
- 最大放电功率：1248 kW
- SOC reference：0.55
- SOC hard lower bound：0.20
- SOC hard upper bound：0.80

当前符号约定：

```text
P_batt > 0：电池放电
P_batt < 0：电池充电
```

## 2.3 MPC

- Sample time：1 s
- Prediction horizon：N = 6
- Optimization method：convex QP / OSQP

---

# 3. 无 LSTM 条件下的 MPC 负荷处理

当前版本不使用 LSTM，也不读取真实未来负荷。

在时刻 `t`，MPC 使用当前实际负荷构造 6 步 persistence forecast：

\[
\hat P_{load,t+k|t}=P_{load,t},
\qquad k=1,\ldots,6
\]

即：

```text
[P_t, P_t, P_t, P_t, P_t, P_t]
```

MPC 求解完成后只执行第一步控制。

到下一秒 `t+1`：

1. 获取新的实际负荷；
2. 更新系统状态；
3. 重新构造新的 6 步负荷预测；
4. 重新求解 MPC。

因此，负荷并不是整个航段保持恒定，而只是每一个 MPC 预测窗口内部采用当前负荷保持。

该设置用于建立无负荷预测模型条件下的 causal DQN-MPC baseline。

---

# 4. DQN 状态空间

当前 DQN state 为 **7 维纯因果状态**。

\[
s_t =
[
s_1,s_2,\ldots,s_7
]
\]

具体定义：

| Dimension | State |
| --- | --- |
| 1 | `(SOC - 0.55) / 0.05` |
| 2 | `previous_FC / 600` |
| 3 | `previous_battery / 624` |
| 4 | `current_load / 600` |
| 5 | `(P_load[t] - P_load[t-1]) / 48` |
| 6 | `mean(P_load[max(0,t-9):t+1]) / 600` |
| 7 | `mean(P_load[max(0,t-59):t+1]) / 600` |

其中：

- 第 5 维描述当前负荷变化趋势；
- 第 6 维描述最近约 10 s 的短时平均负荷；
- 第 7 维描述最近约 60 s 的中短期平均负荷。

航段开始阶段如果历史长度不足 10 s 或 60 s，则只使用当前已经存在的历史样本计算平均值。

当前 state **不包含任何真实未来负荷**。

---

# 5. DQN 动作空间

DQN 当前包含 4 个离散动作。

每一个动作对应一组 MPC 目标函数权重：

\[
(q_{h2},q_{batt},q_{soc},q_{fcvar})
\]

| Action | Name | Weights | Intended role |
| --- | --- | --- | --- |
| A0 | nominal | `(0.25, 0.40, 12, 20)` | 正常、平稳运行 |
| A1 | hydrogen economy | `(0.40, 0.25, 8, 8)` | 氢耗经济性 |
| A2 | SOC regulation | `(0.25, 0.45, 200, 8)` | SOC 调节与能量恢复 |
| A3 | fast FC response | `(0.15, 0.80, 12, 8)` | 快速负荷变化下提高 FC 响应能力 |

四个动作不是直接的功率指令。

DQN 的作用是根据当前运行状态决定：

> 当前工况下应该采用哪一种 MPC 优化偏好。

---

# 6. MPC 目标函数

单个 MPC 动作对应的优化目标为：

\[
J_{MPC}
=
\sum_{k=0}^{N-1}
\left[
q_{h2}H_k
+
q_{batt}B_k
+
q_{soc}S_k
\right]
+
q_{fcvar}F
\]

其中主要包括：

- 燃料电池氢耗；
- 电池功率使用；
- SOC reference tracking；
- 燃料电池功率变化。

不同动作通过不同权重产生不同的底层功率分配行为。

---

# 7. DQN 公共奖励函数

DQN 的四个动作使用完全相同的公共 reward。

Reward 不包含任何 `action_id` 直接奖励。

定义：

\[
r_t=-J_t
\]

其中：

\[
J_t =
0.25H_t
+
0.40B_t
+
12S_t
+
360g_{low}D_t
+
4g_{rise}R_t
+
20(1-g_{rise})(1-g_{low})F_t
\]

---

## 7.1 氢耗项

\[
H_t=
\frac{m_{H2}(P_{fc,t})}
{m_{H2}(600)}
\]

用于评价燃料电池实际氢耗。

---

## 7.2 电池功率项

\[
B_t=
\left(
\frac{P_{batt,t}}{624}
\right)^2
\]

用于抑制过大的电池充放电功率。

---

## 7.3 SOC 基础调节项

\[
S_t=
\left(
\frac{SOC_{t+1}-0.55}
{0.05}
\right)^2
\]

用于维持 SOC 在 reference 附近运行。

---

## 7.4 低 SOC 放电惩罚

\[
D_t=
\left(
\frac{\max(P_{batt,t},0)}
{624}
\right)^2
\]

SOC 门控：

\[
g_{low}
=
clip
\left(
\frac{0.55-SOC_{before}}
{0.55-0.20},
0,1
\right)
\]

因此：

- SOC 接近 0.55 时，该项作用较弱；
- SOC 越接近 0.20，继续使用电池放电的代价越大。

该项用于体现 SOC regulation 行为的实际价值。

---

## 7.5 快速升负荷门控

负荷变化使用当前已观测的后向差分：

\[
\Delta P_{load,t}
=
P_{load,t}-P_{load,t-1}
\]

TRAIN 数据正负荷增量 P99 为：

\[
\Delta P_{rise,ref}
=
6.116713499697\ \text{kW/s}
\]

定义：

\[
g_{rise}
=
clip
\left(
\frac{\max(\Delta P_{load,t},0)}
{6.116713499697},
0,1
\right)
\]

该参数用于判断当前负荷增长相对于训练数据是否属于快速增长工况。

---

## 7.6 FC 快速响应不足项

FC 实际变化：

\[
\Delta P_{fc,t}
=
P_{fc,t}-P_{fc,t-1}
\]

FC 物理最大 ramp 为：

\[
48\ \text{kW/s}
\]

期望可达响应：

\[
\Delta P_{fc,desired}
=
\min
\left(
\max(\Delta P_{load,t},0),
48
\right)
\]

响应不足定义：

\[
R_t=
\left[
\frac{
\max(
\Delta P_{fc,desired}
-
\max(\Delta P_{fc,t},0),
0)
}{48}
\right]^2
\]

负荷快速上升时，如果 FC 没有充分利用允许的 ramp，则产生额外惩罚。

---

## 7.7 FC 平滑项

\[
F_t=
\left(
\frac{\Delta P_{fc,t}}
{48}
\right)^2
\]

实际权重为：

\[
20(1-g_{rise})(1-g_{low})F_t
\]

因此：

- 正常 SOC + 平稳负荷：强调 FC 平滑；
- 低 SOC：允许 FC 更积极地调整以保护电池；
- 快速升负荷：允许 FC 快速响应负荷变化。

---

## 7.8 求解失败

如果 MPC 出现 solver failure：

```text
terminal reward = -620
```

该 transition 作为负反馈参与 DQN 学习。

---

# 8. DQN 网络

当前 DQN 使用 MLP：

```text
Input: 7
   ↓
128
   ↓
64
   ↓
Output: 4
```

即：

```text
7 → 128 → 64 → 4
```

输出对应：

```text
Q(s,A0)
Q(s,A1)
Q(s,A2)
Q(s,A3)
```

当前版本未采用：

- Double DQN
- Dueling DQN
- KAN
- SineKAN

---

# 9. 数据集划分

当前共有 66 条 voyage。

固定划分：

| Dataset | Voyages | Count |
| --- | --- | ---: |
| Train | voyage_001–voyage_046 | 46 |
| Validation | voyage_047–voyage_059 | 13 |
| Test | voyage_060–voyage_066 | 7 |

Test 集在模型设计、训练以及 validation 阶段不参与使用。

---

# 10. 正式 DQN 训练

当前冻结模型：

```text
outputs/
└── dqn_mpc_mlp_causal_1epoch_20260820/
    └── model_final.pt
```

本轮正式训练规模约为：

```text
1,245,456 training steps
```

训练过程中允许 exploration 导致部分 episode failure。

Solver failure 使用 `-620` terminal reward 作为学习信号。

因此 training 阶段不要求所有 episode 100% 完成。

---

# 11. Validation 结果

Validation：

```text
voyage_047 – voyage_059
```

共 13 条航段。

结果：

```text
Completed: 12 / 13
Success rate: 92.31%
```

唯一未完成航段：

```text
voyage_054
```

`voyage_054` 的失败机理已经定位为：

1. 航段存在长时间高负荷区域；
2. 燃料电池长时间达到 600 kW 上限；
3. 负荷持续高于燃料电池最大功率；
4. 电池持续提供功率缺口；
5. SOC 最终接近 0.20；
6. MPC 出现 primal infeasible。

因此该航段目前作为 causal DQN-MPC 的已知边界工况保留，不从数据集中删除。

---

# 12. Independent Test

冻结模型完成 validation 后，不再修改模型参数，随后对：

```text
voyage_060 – voyage_066
```

进行第一次独立 test。

测试使用：

```text
greedy DQN policy
```

即：

\[
a_t=\arg\max_a Q(s_t,a)
\]

测试过程中：

- 不进行 exploration；
- 不更新神经网络；
- 不进行再训练。

测试结果：

```text
Completed: 7 / 7
Success rate: 100%
Solver failures: 0
```

---

# 13. Test SOC 结果

| Voyage | Minimum SOC | Final SOC |
| --- | ---: | ---: |
| voyage_060 | 0.531647 | 0.547406 |
| voyage_061 | 0.543507 | 0.549463 |
| voyage_062 | 0.537566 | 0.548247 |
| voyage_063 | 0.398705 | 0.547958 |
| voyage_064 | 0.540398 | 0.549397 |
| voyage_065 | 0.495405 | 0.547712 |
| voyage_066 | 0.540415 | 0.549969 |

最低 SOC 出现在：

```text
voyage_063
```

其最低值为：

\[
SOC_{min}=0.398705
\]

仍明显高于：

\[
SOC_{hard,min}=0.20
\]

---

# 14. Test 动作分布

7 条 test 航段总动作次数：

| Action | Count | Fraction |
| --- | ---: | ---: |
| A0 nominal | 2253 | 2.42% |
| A1 hydrogen economy | 24127 | 25.89% |
| A2 SOC regulation | 63919 | 68.60% |
| A3 fast FC response | 2879 | 3.09% |

当前冻结策略主要采用 A2 SOC regulation，同时根据工况使用其他三个动作。

A3 主要用于较少出现的快速负荷变化工况，因此其总体选择比例较低。

---

# 15. Test 功率与 SOC 图

正式 test 会为每条测试航段保存逐秒 trace 和时序图。

输出结构：

```text
outputs/
└── dqn_mpc_mlp_causal_1epoch_20260820/
    └── formal_test/
        ├── test_by_voyage.csv
        ├── test_summary.json
        ├── traces/
        │   ├── voyage_060_trace.csv
        │   ├── voyage_061_trace.csv
        │   ├── ...
        │   └── voyage_066_trace.csv
        └── plots/
            ├── voyage_060_power.png
            ├── voyage_060_soc.png
            ├── ...
            ├── voyage_066_power.png
            └── voyage_066_soc.png
```

功率图包括：

- Load
- Fuel cell
- Battery

满足功率平衡：

\[
P_{load}=P_{fc}+P_{batt}
\]

SOC 图包括：

- SOC trajectory
- SOC reference = 0.55

其中 `voyage_063` 为当前 test 中 SOC 变化最明显的典型高负荷航段。

在该航段高负荷阶段，燃料电池达到约 600 kW 上限，电池承担额外负荷，因此 SOC 明显下降；当负荷降低后，燃料电池输出高于当前负荷，电池进入充电状态，SOC 随后恢复至参考值附近。

---

# 16. PyCharm 运行入口

## 16.1 正式训练

```text
src/main/run_dqn_mpc_causal_training.py
```

运行后：

```text
Train
  ↓
Validation
  ↓
model_final.pt
```

---

## 16.2 独立 Validation

```text
src/main/validate_dqn_mpc_causal.py
```

用于加载已有冻结模型重新运行：

```text
voyage_047 – voyage_059
```

不会重新训练模型。

---

## 16.3 独立 Test

```text
src/main/test_dqn_mpc_causal.py
```

用于加载冻结模型运行：

```text
voyage_060 – voyage_066
```

并自动输出：

- test summary
- per-voyage results
- step-by-step traces
- power allocation plots
- SOC trajectory plots

---

# 17. 当前研究结论

当前 causal DQN-MPC 已完成完整的：

```text
Train
  ↓
Validation
  ↓
Independent Test
```

流程。

当前结果：

```text
Validation: 12 / 13 = 92.31%
Independent Test: 7 / 7 = 100%
```

独立 test 中：

- 7 条航段全部完成；
- solver failure = 0；
- 所有测试航段 SOC 均保持在硬约束范围内；
- 最低测试 SOC 为 0.398705；
- DQN 能根据运行状态在四种 MPC 权重模式之间进行选择；
- 功率分配满足燃料电池—电池混合供能逻辑。

---

# 18. 当前模型边界

当前版本仍存在以下边界：

1. 暂未接入 LSTM；
2. MPC 采用 current-load persistence forecast；
3. DQN 只使用当前与历史信息；
4. `voyage_054` 在 validation 中仍会因长时间高负荷导致 SOC 接近下限并产生 MPC infeasible；
5. 当前 action distribution 中 A2 SOC regulation 占比较高，后续可进一步分析其状态依赖性；
6. 当前结果用于验证 DQN-MPC 分层能量管理策略的可行性，不代表所有可能航行工况均已覆盖。

---

# 19. 当前活动版本

当前主要开发分支：

```text
feat/dqn-four-action-diagnostic
```

主要模型配置：

```text
FC max:          600 kW
Battery:         624 kWh
MPC horizon:     6 s
DQN state:       7
DQN actions:     4
Train voyages:   46
Validation:      13
Test:            7
```

当前冻结 checkpoint：

```text
outputs/dqn_mpc_mlp_causal_1epoch_20260820/model_final.pt
```
