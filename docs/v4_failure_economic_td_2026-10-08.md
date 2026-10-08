# v4：失败航段进入经济 Q 的 TD 学习

## 范围与结论

基于 `feat/direct-power-mlp-ddqn` 的远端/本地 `3c6d59bf370177a63c503c4abf778a1bff9e6149`。
本次只修改实现，运行单元测试、契约测试和合成验证。没有启动正式训练，没有打开 Test。
上一阶段的正式训练、中断日志、图表和 best checkpoint 均保留。

未完成 ONBOARD 后缀现在能进入经济 replay，并通过经济 Q 的 Bellman target 影响 online 网络。
独立 outcome 网络仍保留诊断用途；新机制不依赖它参与动作选择。
这不是正式完成率改善的证据，也不构成供电完成的数学保证。

机器可读证据：

- [verification.json](results/v4_failure_economic_td_20261008/verification.json)
- [learning_order_evidence.json](results/v4_failure_economic_td_20261008/learning_order_evidence.json)

## 修改文件

| 文件 | 作用 |
|---|---|
| `src/v4/failure_replay.py`（新增） | 对真实失败构造只用于学习的终止视图；冻结惩罚参数、闭合电池势函数 |
| `src/v4/failure_penalty_reference.json`（新增） | 30 个历史已完成 Train 样本的经济尺度及来源 SHA256 |
| `src/v4/control.py` | 失败原因、实际失败观察状态、终止原因及正常完成谓词；实际运行轨迹仍保留原貌 |
| `src/v4/reward_feedback.py` | 正常/失败轨迹共同使用的势函数终端校正 |
| `src/v4/dqn.py` | 失败经验插入、来源计数、严格禁止终止 bootstrap、失败终止 TD 误差统计 |
| `src/v4/monitored_training.py` | 预填充/探索阶段接入失败经济经验；greedy 只读报告和每轮分类计数 |
| `src/v4/feedback_study.py` | 当前单配置入口启用失败 TD；新增 `--failure-penalty-scale` |
| `src/v4/diagnostics.py` | 训练惩罚与经济账本分别输出，明确正常/失败终止 |
| `src/v4/train.py` | 正常航段完成数排除失败 terminal，保留先前已完成的航段 |
| `src/v4/staged_study.py` | 历史复现显式关闭新协议；不覆盖旧归档 |
| `tests/test_v4_failure_td.py`（新增） | 失败分支、奖励恒等式、TD target、排序改变、边界与异常测试 |

当前入口 `v4.feedback_study` 默认仍为 replay32、target500、batch64、γ=1、n=1、beta_soc=500。
网络仍为 8–128–64–61，FC 动作为 0:10:600 kW，Adam=1e-4、Smooth L1、梯度裁剪=10。
物理模型、8 维状态、SOC 硬约束、action mask、epsilon、数据划分均未改变。
兼容用途的通用监控函数保留旧调度默认值；旧 `v4.train` 训练入口和 staged 历史复现不是本次新协议的正式入口。

## 失败终止语义

只处理控制器确实观察到单步物理动作 mask 为空的 `no_feasible_action`。
模型或策略抛出的异常，即使异常类型名含 `NoFeasibleFCActionError`，也不能代替物理空 mask 证据。
非有限 Q 值和模型计算异常记为 `execution_error`，不制造失败惩罚或失败经验。

处理步骤：

1. 找到当前样本内最后一个正常完成的 ONBOARD 航段；以前的正常 transition 完全保留。
2. 当前失败后缀有已执行动作时，保留原状态、动作、顺序、原始奖励与经济账本。
3. 仅替换最后一个已有 transition 的学习字段：`done=True`、空下一动作 mask、失败标签、势函数校正、一次惩罚。
4. 下一状态仍为真实观察到的失败状态；不创建虚假的全零状态或未执行动作。
5. 当前后缀为空时只记录事件。岸充后首步失败不得向前一个成功航段分配惩罚。

`terminal_reason` 使用 `completed`、`failure_soc_limited`、`failure_structural_power`。
失败 terminal 不计正常完成数，不产生 completed-episode 更新额度，不满足 checkpoint 完成率门槛。
失败时不会执行真实 SHORE 或 MODELED terminal settlement；正常完成时保持原有结算。

### 物理失败分类

在原有 61 动作网格上，先只看 FC/Battery 功率边界，再看 SOC 可行性：

- `structural_power`：连功率边界都没有满足的动作，例如负载 2000 kW 超过当前 600+1248 kW 总能力。
- `soc_limited`：有功率可分配的动作，但全部受 SOC 约束排除。

两类分别统计。有真实已执行后缀的两类 `no_feasible_action` 均按用户要求保留失败 TD，使用明确的同一惩罚协议。
结构性失败不能解释为通过 FC 策略可消除，报告 `controllable_by_power_policy=False`。
仅凭 SOC 分类也不能证明整条航段可完成，因此其该字段为未知，而非自动标记可控。
若未来正式数据存在结构性失败，应先审查数据/系统容量，不能期待惩罚修复容量不足。

## 奖励公式与累计一致性

令 `r_old,k` 为已有奖励：实际经济费用、SOC 软惩罚及正常终止结算的原有组合。
正常 ONBOARD 主体仍为：

\[
r^{old}_k=-\left(C_{H_2,k}+C_{FC,deg,k}+C_{Bat,deg,k}+\beta_{soc}\phi(SOC_{k+1})\right).
\]

沿用正式参数读取的电池价值：

\[
B(s)=\frac{c_{shore}E_{bat}}{\eta_{chg}}(0.6-s),
\qquad \Delta B_k=B(SOC_{k+1}^{actual})-B(SOC_k).
\]

重分配模式每步 `r_shifted,k = r_old,k - ΔB_k`。
失败发生后，设当前后缀已执行 m 步，其最后真实执行后 SOC 为 `SOC_m`，该后缀起始 SOC 为 `SOC_0`：

\[
\kappa_{fail}=B(SOC_m)-B(SOC_0).
\]

最后一个已执行动作的经济 replay 奖励为：

\[
r^{replay}_{m-1}=\begin{cases}
r^{old}_{m-1}-C_{fail}, & \text{原奖励模式},\\
r^{old}_{m-1}-\Delta B_{m-1}+\kappa_{fail}-C_{fail}, & \text{重分配模式}.
\end{cases}
\]

因此对同一已执行失败后缀：

\[
\sum r^{original,replay}=\sum r^{redistributed,replay}=\sum r^{old}-C_{fail}.
\]

校正只闭合电池价值的时间重分配，不是末端充电，不收费，不改变物理 SOC。
不使用之后 SHORE 的 SOC；不同 ONBOARD 段各自闭合，不跨 SHORE 抵消。

Double-DQN 的非终止部分仍为在线网络选动作、目标网络估值：

\[
y_k=\begin{cases}
r^{replay}_k, & done_k=1,\\
r^{replay}_k+\gamma Q_{target}\left(s_{k+1},\arg\max_{a\in A_{k+1}}Q_{online}(s_{k+1},a)\right), & done_k=0.
\end{cases}
\]

终止分支显式返回零 continuation，避免 `0 × NaN` 污染 target。

## 惩罚尺度与敏感性方案

定义 `C_fail = λ_fail × C_ref`，`λ_fail` 为可配置无量纲参数，默认 1。
`C_ref` 冻结为历史 beta500 best checkpoint 的 **30 个完成 Train 样本可比经济成本的 nearest-rank P95**：

\[
C_{ref}=10132.660087898294.
\]

来源 `docs/results/v4_staged_beta_cadence_seed42/raw/stage1/beta_500/best_profiles.json` 的 `train` 项。
源文件 SHA256：`3e1fae46266d99629116b52156840541a33136be92f9f4e68e1e8b03256a0642`。
来源结果提交为 `b35c8408ad1258c089dcc72df9a7c0e5e285d4f5`，已由 `4a425b4` 归档。
JSON 保存逐样本值和来源，不依赖新训练，不使用 Validation/Test 选定尺度。
尺度包括原有实际四项费用及必要的 modeled settlement，排除 SOC shaping。

该参数量纲为 **CNY-equivalent reward units，非实际人民币支出**。
失败罚项在日志单独标为 `failure_penalty_equivalent_cny`，绝不加入经济账本四项或 terminal settlement。
这是按已有完整样本账单量级确定的初始尺度，不是最优值或保证所有失败比成功更差的上界。
样本长度、历史策略与 SOC shaping 会影响合适尺度；保持参数冻结和完整完成资格门槛。

固定合成失败轨迹的纯算术核对（未训练敏感性配置）：

| λ_fail | C_fail | 原奖励含惩罚累计 | 重分配含惩罚累计 |
|---:|---:|---:|---:|
| 0（消融） | 0 | -21.8750495903 | -21.8750495903 |
| 0.5 | 5066.3300439491 | -5088.2050935394 | -5088.2050935394 |
| 1 | 10132.6600878983 | -10154.5351374886 | -10154.5351374886 |
| 2 | 20265.3201757966 | -20287.1952253869 | -20287.1952253869 |

这四种核对真实账本完全相同，势函数校正均为 6.0941828255，累计差异均为零。
未来若获准做敏感性分析，应固定其他参数，只改变 λ，并比较完成率、合格 Validation 成本、失败终止 TD 误差和梯度裁剪影响。
原奖励/重分配两组的 λ、冻结参考及失败协议必须完全相同，不能各自选惩罚制造多变量对照。

## 实际验证结果

相关回归 **170 passed，330 subtests passed，121.45 s**。
新增文件覆盖 17 个收集的测试案例，另复用已有 reward、n-step、调度、物理/退化、岸充及只读监控测试。
未运行正式 Train/Validation rollout；包含合成数据集的一轮入口检查，不是正式训练结果。

- 同一 `.2135` 初始 SOC、负载 `[600,1000,0]`：首动作 0 kW 导致后续 SOC 无可行动作；首动作 600 kW 的固定策略可完成。
- 失败 transition 进入经济 replay；`done=True`、空 mask，即使无用 next-Q 为 NaN，TD target 仍等于本步含惩罚 reward。
- 原/重分配失败轨迹累计相同，惩罚只计一次，重复转换拒绝。
- 正常轨迹 A–H（含不同长度/初始 SOC，共 10 个固定轨迹）的实际与 modeled 账本、原/重分配累计奖励，与上一提交归档严格相等。
- 另从 `3c6d59b` 加载旧控制器执行这 10 个轨迹的两种奖励模式，共 20 次跨提交核对：旧 transition 的全部字段，以及实际 SOC 序列、最终物理状态和结算字段，逐项严格相等。
- 多 ONBOARD 边界、岸充后首步失败、结构性容量失败、程序异常和非有限模型输出均覆盖。
- greedy 评估不插经验、不更新网络/优化器、恢复 RNG，失败不计正常完成。

### replay32/target500 计数核对

| 验证 | 成功插入 | 失败插入 | 合计插入 | 经济更新 | 调度 Target 同步 | 剩余 credit |
|---|---:|---:|---:|---:|---:|---:|
| 小型合成训练入口 | 69 | 41 | 110 | 3（预填充 2，探索 1） | 0 | 14 |
| 固定合成 replay 重复学习 | 14022 | 5248 | 19270 | 602 | 1 | 6 |

前者有 1 条失败 terminal，后者有 5248 条；均为真实动作的学习视图。
更新额度是实际插入数，失败 suffix 不再被丢弃；FIFO 满后插入计数及跨轮 credit 的既有契约仍通过。
Target 总调用数分别 1/3：前者 1 次初始化；后者 2 次初始化复制（额外一次仅用于测试的起始偏好）+ 500-update 同步 1 次。
小型入口保留旧 outcome 更新，实际 32 次，但不触发 target sync。经济 Q 独立学习证明中 outcome 更新为 0 且权重完全不变。

### Online Q 动作排序证据

构造全部 61 个首动作分支；4 个会失败，57 个能沿固定后续策略完成。
测试为保证失败样本出现，重复失败分支；测试起始 FC=0 bias 加 1，仅为受控初始偏好，不修改生产初始化。
使用原 MLP、Adam1e-4、batch64、clip10、γ1、n1、replay32、target500，进行 602 次合成经济更新：

| 项目 | 更新前 | 更新后 |
|---|---:|---:|
| Q(s,FC=0) | 0.7418578863 | -18.4606533051 |
| Q(s,FC=600) | -0.1029864177 | -18.2440910339 |
| Q(0)−Q(600) | 0.8448443040 | -0.2165622711 |
| 全部物理可行动作下 greedy | 0 kW | 330 kW |

更新后的实际 greedy 合成执行为 `[330,330,330]`，3 个真实动作均完成，未访问 outcome 选择策略。
失败终止 TD 采样数 10488，平均绝对 TD 误差约 10424.55。
这表示网络已改变动作排序，但 Q 数值尚未收敛；不能用该小测试宣布完成率或经济性提高。

## 监控、归档与限制

每轮报告新增失败样本数、SOC/结构性/程序异常计数、失败 terminal replay 数、成功/失败插入数及累计比例。
沿用经济 optimizer/target 累计计数、greedy Train/Validation 完成率、SOC 分档、FC=0/启动、四项经济成本和 modeled settlement。
TD 统计增加 `failure_terminal` 子项（样本数、均值、MAE、RMS、最大误差）；固定 61 动作 Q 值诊断继续保留。
失败样本的部分已执行费用用于单独诊断，未完成集合的完整可比成本仍为 null，不能伪装成低成本成功样本。

已独立复核 93 个旧 raw 归档文件 SHA256 和 5 个正式 metadata manifest SHA256；全部匹配。
旧结果、v2/v3、正式数据均无 git 差异。合成测试 Test 读取计数为 0；原正式 Test 未访问。

局限：

- 有限失败惩罚只是 soft learning signal，不能替代整航段完成的硬资格检查。
- n=1 仍需多次 bootstrap 把末端失败价值向更早动作传播；8 维状态不包含未来完整负载，可能有状态混叠。
- SOC 无可行动作不自动证明过去某个动作具有可避免的因果责任；结构性容量不足更不能靠学习解决。
- 错误/首步无动作事件不生成失败 TD；它们需要独立的数据或模型诊断。
- 参考 P95 是历史完成样本的经济尺度，SOC shaping 不在其中。正式性能和尺度敏感性仍待验证。
- 合成重复学习是数据路径和排序证明，失败 TD 残差大，没有完成率或经济改进结论。

## 下一步最小对照（尚未执行）

获准后，只做 `original` 与 `redistributed` 两组；同 seed42、beta500、λ_fail=1、replay32、target500、batch64、γ1、n1、物理 mask 与冻结参考。
使用独立新目录和完全一致的训练预算、预填充规则、每轮 greedy 门槛；测试集继续封存。
先对照完成率及 failure TD/动作排序，随后比较符合 Train30/30、Validation8/8 资格的经济成本。
若需要 λ 的敏感性分析，作为独立实验设计，不与奖励重分配同时变动。本次不自动启动这些实验。
