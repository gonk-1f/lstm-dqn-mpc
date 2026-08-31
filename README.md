# Causal DQN-MPC 船舶混合能源管理

本仓库保留论文最终控制主线：由因果 7 维状态驱动 DQN 选择四个 MPC 权重动作，再由 `N=6` 的凸 QP-MPC 完成燃料电池与电池功率分配。正式默认 Q-network 为 MLP；KAN 作为可替换后端，共用同一套状态、奖励、动作、回放、训练循环、环境和 MPC。

## 数据来源与划分

原始航段位于 `total_load_excels/`。正式 1 s 负荷数据位于：

```text
outputs/spline_1s_diagnostics/data/natural_clipped_by_voyage/
```

1 s 数据由每条航段的 30 s 设备通道离线做 natural cubic spline 重构，并对建模负荷列做非负裁剪。该变换使用区间两端节点，因此只用于离线论文实验；不能被描述为在线可获得的预测信号。通道构造与审计细节见 `docs/CLUSTER_BASED_TOTAL_LOAD_REBUILD.md`。

固定按航段划分，不跨航段采样：

| Split | Voyage | 数量 |
| --- | --- | ---: |
| Train | `voyage_001`–`voyage_046` | 46 |
| Validation | `voyage_047`–`voyage_059` | 13 |
| Test | `voyage_060`–`voyage_066` | 7 |

权威划分文件为 `outputs/config/voyage_split_total_load_721.json`。训练与 validation 不读取 test 航段。

## 控制结构

每秒执行一次：

```text
当前与历史测量
  -> causal 7维 state
  -> DQN action (A0...A3)
  -> N=6 persistence-forecast QP-MPC
  -> P_fc 与 P_batt
  -> SOC 更新
```

状态顺序为：

1. `(SOC_t - 0.55) / 0.05`
2. `P_fc,t-1 / 600`
3. `P_batt,t-1 / 624`
4. `P_load,t / 600`
5. `(P_load,t - P_load,t-1) / 48`
6. 最近 10 s 平均负荷 `/ 600`
7. 最近 60 s 平均负荷 `/ 600`

MPC 只使用当前测得负荷在 6 个预测步上的 persistence forecast，不使用 LSTM 或未来真实负荷。

## Q-network 后端

正式默认 MLP：

```text
7 -> 128 -> 64 -> 4
```

统一 factory 接口：

```python
DQNTrainConfig(network_type="mlp")  # 正式默认
DQNTrainConfig(network_type="kan")  # 自包含 Torch KAN
```

KAN 只替换 Q-network；不修改 environment、reward、state builder、replay buffer、action mapping、MPC solver 或训练循环。当前 KAN 不依赖外部 pykan 或独立 SineKAN 工程。

## 四个 MPC 动作

`MPCWeightAction.as_tuple()` 始终返回 `(q_h2, q_batt, q_soc, q_fcvar)`，动作 ID 和数量固定：

| Action | 权重 | SOC penalty |
| --- | --- | --- |
| A0 nominal | `(0.25, 0.40, 12, 20)` | symmetric |
| A1 hydrogen economy | `(0.40, 0.25, 8, 8)` | symmetric |
| A2 SOC recovery | `(0.25, 0.45, 200, 8)` | deficit-only |
| A3 fast FC response | `(0.15, 0.80, 12, 8)` | symmetric |

A2 的 SOC 项为：

```text
d_k >= 0.55 - SOC_k
d_k >= 0
J_soc,A2 = 200 * sum_k (d_k / 0.05)^2
```

因此 `SOC < 0.55` 时 A2 推动恢复；`SOC >= 0.55` 时 A2 的 SOC 项为零，不会为了压回参考值主动制造反向能量搬移。A0、A1、A3 仍使用原对称 SOC 跟踪项。

所有动作使用相同的 slack-QP 变量与稀疏约束结构。非 A2 动作把 deficit 变量固定为零。实现保持凸性并由 OSQP 求解，没有求解器外 clamp。

## 物理配置

| 参数 | 值 |
| --- | ---: |
| MPC horizon | 6 s |
| 控制周期 | 1 s |
| FC 功率范围 | 0–600 kW |
| FC hard ramp | 48 kW/s |
| Battery capacity | 624 kWh |
| Battery 功率范围 | -624–+1248 kW |
| SOC 初值/参考值 | 0.55 |
| SOC hard bounds | 0.20–0.80 |

## Common reward

DQN 的 common reward 与 MPC 动作权重分离，所有动作使用同一评价函数。它包含固定的氢耗、电池功率、对称 SOC、低 SOC 放电、快速负荷上升响应和 FC 变化六项。A2 的 deficit-only 修改只发生在 MPC objective，未修改 reward 系数或 terminal failure reward。

```text
r_t = -J_t
J_t = 0.25 H_t + 0.40 B_t + 12 S_t
    + 360 g_low D_t + 4 g_rise R_t + q_F,t F_t
q_F,t = 8 + 12 (1 - g_rise) (1 - g_low)
```

其中 `H_t` 为相对 600 kW 参考点的 Dp0 氢耗，`B_t=(P_batt/624)^2`，`S_t=((SOC_t+1-0.55)/0.05)^2`；`D_t`、`R_t` 与 `F_t` 分别为低 SOC 放电、快速升负荷响应不足和 FC 变化归一化项。求解失败的 terminal reward 仍为 `-620`。

## 正式入口

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

构建/核验数据：

```powershell
python src/main/build_total_load_dataset_721.py
python src/main/build_spline_1s_diagnostics.py
```

正式 MLP 训练与 validation（两轮完整 train voyages）：

```powershell
python src/main/run_dqn_mpc_causal_training.py
```

新 MLP 输出目录：

```text
outputs/dqn_mpc_mlp_causal_deficit_a2_formal_rounds/
```

未来 KAN 输出必须使用：

```text
outputs/dqn_mpc_kan_causal_deficit_a2_formal_rounds/
```

两种后端不得混用 checkpoint。独立 test 默认只接受新 MLP namespace 中的 `round_2/model_round2.pt`；文件不存在或路径属于另一后端时会明确报错，不会 fallback：

```powershell
python src/main/test_dqn_mpc_causal.py
```

Fixed A0 基线入口：

```powershell
python src/main/test_mpc_nominal_causal.py
```

控制器对比入口：

```powershell
python src/main/compare_mpc_vs_dqn.py
```

## 自动化验证

本仓库保留针对以下契约的 focused tests：causal 7 维状态、四动作映射、common reward、MLP/KAN factory 与 greedy action、replay/update、正式 split、validation 无学习副作用、QP 物理约束、A2 deficit-only 语义、统一 OSQP 结构及 checkpoint 隔离。

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前仓库不包含与新 A2 语义兼容的正式 DQN checkpoint。必须先重新训练，再运行独立 test；旧对称 A2 checkpoint 不得复用。
