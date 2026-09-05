# Causal DQN-MPC 船舶混合能源管理

本仓库保留论文最终控制主线：由因果 7 维状态驱动 DQN 选择四个 MPC 权重动作，再由 `N=6` 的凸 QP-MPC 完成燃料电池与电池功率分配。正式默认 Q-network 为 MLP；KAN 作为可替换后端，共用同一套状态、奖励、动作、回放、训练循环、环境和 MPC。

## 数据来源与划分

原始约30 s设备 CSV 位于桌面“氢舟一号”（只读）。正式 1 s 负荷数据位于：

```text
data/processed/operating_segments_1s_rebuilt/
```

先在原始层审核20路正式通道、排除无效与外部充电、提取混合动力 operating segment；再于每段统一30 s参考节点先求 `P_fc_total + P_batt_total`，对总负荷使用 PCHIP 重构到1 s。正式列仅为 `timestamp`、`time_s`、`load_total_kw`；不对设备通道分别插值，也不对负荷无条件裁剪。

segment 继承 parent voyage 划分，不跨 parent 泄漏：

| Split | Parent voyage | 当前 segment 数量 |
| --- | --- | ---: |
| Train | 46 parent voyages | 144 |
| Validation | 13 parent voyages | 23 |
| Test | 7 parent voyages | 10 |

权威划分文件为 `data/processed/operating_segments_1s_rebuilt/split_manifest.csv`。最终数据共 66 个 parent voyages、177 个 segments、1,114,037 条 1 s 样本；训练与 validation 不读取 test segment，每段为独立 episode，初始 SOC 固定0.55。旧 natural cubic spline 数据已废弃。

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

| Action | 权重 |
| --- | --- | --- |
| A0 nominal | `(0.25, 0.40, 12, 20)` |
| A1 hydrogen economy | `(0.40, 0.25, 8, 8)` |
| A2 SOC recovery | `(0.25, 0.45, 200, 8)` |
| A3 fast FC response | `(0.15, 0.80, 12, 8)` |

四个动作使用完全相同的 MPC 目标函数与物理约束；DQN 只选择
`(q_h2, q_batt, q_soc, q_fcvar)`。SOC 软工作区间统一为 `[0.50, 0.60]`：

```text
d_k >= 0.50 - SOC_k
d_k >= SOC_k - 0.60
d_k >= 0
J_soc = q_soc * sum_k (d_k / 0.05)^2
```

因此区间内 SOC 代价为零；区间外按到最近边界的归一化平方距离惩罚。所有动作使用相同的 `SOC_band_violation` 辅助变量和稀疏约束结构。实现保持凸性并由 OSQP 求解，没有求解器外 clamp。

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

DQN 的 common reward 与 MPC 动作权重分离。MLP 和 KAN backend、A0～A3 所有动作均使用同一个固定四项评价函数：

```text
r_t = -J_t
J_t = 0.25 H_t + 0.40 B_t
    + Phi_SOC(SOC_t+1) + 20 F_t
```

其中：

- `H_t=m_H2(P_fc,t)/m_H2(600)`：normalized hydrogen consumption；
- `B_t=(P_batt,t/624)^2`：normalized battery power penalty；
- `F_t=((P_fc,t-P_fc,t-1)/48)^2`：fuel-cell power variation penalty；
- `Phi_SOC`：SOC soft working-range penalty。

```text
Phi_SOC(SOC) = ((0.50 - SOC) / 0.05)^2,  SOC < 0.50
               0,                       0.50 <= SOC <= 0.60
               ((SOC - 0.60) / 0.05)^2, SOC > 0.60
```

reward 与 MPC 的 soft SOC range 均为 `0.50～0.60`，MPC 的 hard SOC constraints 为 `0.20～0.80`。SOC 位于 soft range 内时不要求实时跟踪 `0.55`。`SOC_ref=0.55` 仍用于系统初始/参考 SOC，但不构成 MPC 或 common reward 的逐步跟踪项。求解失败的 terminal reward 保持 `-620`。

## 正式入口

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

构建/核验数据：

```powershell
python src/main/build_rebuilt_operating_segment_dataset.py
```

正式 MLP 训练与 validation（两轮完整 train voyages）：

```powershell
python src/main/run_dqn_mpc_causal_training.py
```

新 MLP 输出目录：

```text
outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds/
```

未来 KAN 输出必须使用：

```text
outputs/dqn_mpc_kan_causal_soc_deadband_formal_rounds/
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

本仓库保留针对以下契约的 focused tests：causal 7 维状态、四动作映射、common reward、MLP/KAN factory 与 greedy action、replay/update、正式 split、validation 无学习副作用、QP 物理约束、统一 SOC deadband 语义、统一 OSQP 结构及 checkpoint 隔离。

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前仓库不包含与新 A2 语义兼容的正式 DQN checkpoint。必须先重新训练，再运行独立 test；旧对称 A2 checkpoint 不得复用。
