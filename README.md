# Causal DQN-MPC 船舶混合能源管理

本仓库保留论文最终控制主线：由因果 7 维状态驱动 DQN 选择四个 MPC 权重动作，再由 `N=6` 的凸 QP-MPC 完成燃料电池与电池功率分配。正式默认 Q-network 为 MLP；KAN 作为可替换后端，共用同一套状态、奖励、动作、回放、训练循环、环境和 MPC。

## 数据来源与划分

原始约30 s设备 CSV 位于桌面“氢舟一号”（只读）。正式 1 s 负荷数据位于：

```text
data/processed/operating_dataset_final/
```

先在原始层审核 8 路 FC 与 12 个 battery cluster，按
`P_batt_cluster=-U*I/1000` 重构电池功率，再计算
`P_load=P_fc_total+P_batt_total`。排除外部充电、异常和物理不可行区间后，
只对已接受航段的总负荷使用 PCHIP 重构到 1 s，不跨缺口插值。
`total_load_excels/` 的电池量来自 BDM，不属于正式数据链路。

segment 继承 parent voyage 划分，不跨 parent 泄漏：

| Split | Parent voyage | 当前 segment 数量 |
| --- | --- | ---: |
| Train | 20 used parents | 20 |
| Validation | 6 used parents | 6 |
| Test | 7 used parents | 8 |

冻结的 parent 角色仍为 46/13/7；通过正式语义和物理筛选后，共使用 33 个
parent，形成 34 条正式样本和 208,418 个 1 s 点。权威样本清单为
`data/processed/operating_dataset_final/metadata/sample_manifest.csv`，parent
角色清单为 `metadata/parent_split_manifest.csv`。所有正式样本均已通过独立
物理可行性审计；训练和 validation 不读取 Test，每条样本作为独立 episode，
控制仿真初始 SOC 为 0.55。

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

正式 DQN 训练配置使用 MSE TD loss、`gamma=0.99`、`batch_size=64`、replay buffer 容量 `300000` 和 `seed=42`。

## 四个 MPC 动作

`MPCWeightAction.as_tuple()` 始终返回 `(q_h2, q_batt, q_soc, q_fcvar)`，动作 ID 和数量固定：

| Action | 权重 |
| --- | --- | --- |
| A0 balanced | `(0.20, 0.50, 40, 16)` |
| A1 hydrogen economy | `(0.40, 0.25, 8, 8)` |
| A2 FC smoothing | `(0.25, 0.50, 30, 40)` |
| A3 SOC protection | `(0.15, 0.80, 120, 8)` |

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

## DQN reward

DQN 奖励直接来自所选动作在同一次 MPC 求解中的完整 `N=6` 最优目标值。MPC 仍使用该动作的原始四项权重求解；仅在生成奖励时用权重和统一数值尺度：

```text
J_bar = J_MPC* / (q_h2 + q_batt + q_soc + q_fcvar)
r_t = 1 / (1 + J_bar)
```

四个动作的权重和依次为 `56.70 / 16.65 / 70.75 / 128.95`。完整目标包含原有归一化的 horizon hydrogen、battery power、SOC deadband 和 FC variation 四项。奖励计算直接复用本次求解的物理解，不重复求解 MPC，也不使用额外的固定公共评价权重。成功奖励满足 `0 < r_t <= 1`；求解失败的 terminal reward 保持 `-620`。

MPC 的 soft SOC range 仍为 `0.50～0.60`，hard SOC constraints 仍为 `0.20～0.80`。`SOC_ref=0.55` 继续用于系统初始/参考 SOC，不构成额外终端目标项。

正式 DQN discount factor 为 `gamma=0.99`，Bellman target 保持标准形式
`r + gamma * (1-done) * max Q_target(next_state)`。

## 正式入口

安装依赖：

```powershell
python -m pip install -r requirements.txt
```

构建/核验数据：

```powershell
python -X utf8 -B src/main/build_final_operating_dataset.py `
  --raw-root "C:/Users/20883/OneDrive/Desktop/氢舟一号" `
  --output-root data/processed/operating_dataset_final_repeat
```

正式 MLP 训练与 validation（两轮完整 train voyages）：

```powershell
python src/main/run_dqn_mpc_causal_training.py
```

每轮同时写入纯推理 `model_roundX.pt` 与原子保存的
`training_state_roundX.pt`。后者包含 online/target network、optimizer、
epsilon、global step、update count、完整 replay buffer 及 Python/NumPy/PyTorch
RNG 状态，可在关机后继续下一轮：

```powershell
python src/main/run_dqn_mpc_causal_training.py `
  --resume-training-state outputs/dqn_mpc_mlp_causal_soc_deadband_formal_rounds/round_1/training_state_round1.pt
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

Fixed A0 balanced 基线入口：

```powershell
python src/main/test_mpc_nominal_causal.py
```

控制器对比入口：

```powershell
python src/main/compare_mpc_vs_dqn.py
```

## 自动化验证

本仓库保留针对以下契约的 focused tests：causal 7 维状态、四动作映射、所选动作 MPC objective reward、MLP/KAN factory 与 greedy action、replay/update、正式 split、validation 无学习副作用、QP 物理约束、统一 SOC deadband 语义、统一 OSQP 结构及 checkpoint 隔离。

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

当前仓库不包含与新 A2 语义兼容的正式 DQN checkpoint。必须先重新训练，再运行独立 test；旧对称 A2 checkpoint 不得复用。
