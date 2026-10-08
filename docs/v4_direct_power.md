# v4 当前方法：直接功率 MLP Double-DQN

本页依据 `d111c99` 整理。旧方法页及旧首页保存在 [历史 v4 索引](history/v4/README.md)。
本次只整理文档与数值诊断，不修改训练默认值、物理模型或经济账本，不启动正式训练。

## 控制与模式

每个 30 s ONBOARD 决策先观察当前负载，再从 61 个动作 `0,10,…,600 kW` 选择 FC 功率。
`P_bat=P_load-P_FC`，正值放电、负值充电。单步 mask 排除立即违反电池功率或下一步 SOC `[0.2,0.8]` 的动作；不能保证整航段未来可行。
SHORE_PENDING/SHORE_CHARGING 由 operating_mode 决定，不是 DQN 动作。
岸电暂停 DQN，FC 关闭；按实际充电请求及 SOC0.6 上限接受充电，更新正式 SOC、电池退化及购电费用。
再次出港重置负载/FC 控制历史，保留 SOC/累计退化；首步仍用实测负载。当前无 LSTM、LPF、MPC 求解或 KAN 控制。

8 维状态依次为 SOC、当前负载/600、当前与上一负载差/600、上一与更早负载差/600、上一 FC 功率/600、累计 FC 经济寿命比例、累计电池经济寿命比例、出港标志。选动作时不读取未来负载。

## 经济账本与训练奖励

四项实际费用保持：

\[
C_{actual}=C_{H_2}+C_{FC,deg}+C_{Bat,deg}+C_{shore,electricity}.
\]

FC 启停/变载属于正式 FC 退化，不额外增加波动或停机费用。
正常航段后真实岸电费用附着到上一 ONBOARD 终止 transition，SHORE 无 DQN 动作、无额外 SOC shaping。
无真实后续岸电且终端 SOC<0.6 时，使用既有岸充模型计算 **MODELED terminal settlement**：等效补能电费及充电退化；不追加控制点、不改变真实 SOC、不冒充实测交易。

普通 ONBOARD 原奖励为：

\[
r^{old}_k=-\left(C_{H_2,k}+C_{FC,deg,k}+C_{Bat,deg,k}+\beta_{soc}\phi(SOC_{k+1}^{actual})\right).
\]

`phi(s)` 为到 `[0.4,0.6]` 的平方距离，区间内为零。正常终止仍包含原有真实岸电或 MODELED 结算。
SOC shaping 不是实际经济支出，单独统计。默认奖励重分配从既有正式参数生成：

\[
B(s)=\frac{c_{shore}E_{bat}}{\eta_{chg}}(0.6-s),\qquad
r^{shift}_k=r^{old}_k-\left[B(SOC_{k+1})-B(SOC_k)\right].
\]

每个完整 ONBOARD 段最后增加 `B(SOC_T)-B(SOC_0)`，保持全段累计奖励不变。
使用最后 ONBOARD 动作后的实际 SOC，不能使用岸充后的 SOC，也不跨航段抵消。
即时价值调整、终端校正与失败惩罚均不进入实际经济账本。

## 失败经验进入经济 Q

真实物理 mask 为空、且已有实际执行动作时，最后一个 transition 构造成显式失败 terminal：`done=True`、空下一 mask、禁止 bootstrap。
保留原物理状态、动作顺序和实际账本；失败不触发 SHORE/MODELED 终端结算。

`C_fail = failure_penalty_scale × 10132.660087898294`。
参考值来自冻结历史 30 个已完成 Train 样本可比经济成本的 nearest-rank P95；默认倍率 1，量纲为人民币等价奖励单位，非实际支出。
重分配失败末步同样闭合已执行后缀势函数，因此两种模式累计都为 `sum(r_old)-C_fail`。
首动作前失败只记录事件，不向前一成功航段收费；SOC 受限、结构性容量不足和程序异常分别统计。
失败 terminal 不计正常完成。有限惩罚不能保证航段完成，outcome 仅保留诊断用途。
详见 [失败 TD 实施报告](v4_failure_economic_td_2026-10-08.md)。

## 当前默认入口配置

下表指 `python -m v4.feedback_study`，不指旧训练入口或通用函数的兼容默认值。

| 配置 | 默认值 |
|---|---|
| 网络 / 动作 | MLP 8–128–64–61、ReLU / 0:10:600 kW |
| beta_soc / reward-feedback | 500 / redistributed |
| failure-penalty-scale / reward-scale | 1 / 1，`--reward-scale` 可显式配置 |
| γ / n-step | 1 / 1 |
| Adam 学习率 / batch | 1e-4 / 64 |
| replay capacity | 100000 |
| 更新额度 | 每实际新增 32 条经济 replay 给予 1 次更新，余数跨轮保留 |
| Target | 每 500 次经济 optimizer update hard sync，初始化复制另列 |
| Loss / clip | Smooth L1 / 梯度范数上限 10 |
| seed / epsilon | 42 / 1.0→0.05 |

成功/失败经验均按实际插入数给额度，FIFO 满后插入继续累计。
下一状态 mask、Double-DQN 在线选动作/目标网络估值保持原实现。

统一缩放在经济replay插入处执行一次，默认1；原reward组成、终端校正和n-step回报先按原奖励单位计算。outcome与真实经济账本不缩放。Q/TD同时记录训练单位和除以reward_scale后的原单位，详见[接入验证与40轮命令](v4_reward_scale_training_2026-10-08.md)。本次实现验证没有执行正式40轮。

## 运行与选模边界

帮助命令不启动训练：

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m v4.feedback_study --help
```

正式训练必须显式指定轮数和新目录。每轮完整 greedy Train 全完成后才评估 Validation；不写 replay、不更新网络，恢复训练 RNG。
只有 Train30/30、Validation8/8 的 checkpoint 才按 Validation 可比经济费用选模，Test 封存。
尚无本次失败 TD 修复后的正式性能结论；旧 beta/cadence 结果只作为 [历史记录](history/v4/README.md)。

源码：[入口](../src/v4/feedback_study.py)、[控制](../src/v4/control.py)、[经济 Q](../src/v4/dqn.py)、[失败视图](../src/v4/failure_replay.py)、[更新额度](../src/v4/experiment_schedule.py)。
