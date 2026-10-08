# v4 奖励时间重分配与更新调度：实现验证

基于 `feat/direct-power-mlp-ddqn` 的 `4a425b4`。本次只验证实现，未启动正式训练，未加载正式Train/Validation功率数据，未打开Test。合成数据测试最多2轮；其完成数不能解释为正式30/30、8/8改善。上一阶段40轮与中断归档全部保留。

## 1. 代码与公式

新增 `src/v4/reward_feedback.py` 从现有正式参数生成：

```text
K = FORMAL_PRICE_CATALOG.shore_cny_per_kwh
    * accountant.plant.battery_nominal_energy_kwh
    / accountant.efficiency.require_calibrated()[0]
B(s) = K * (SHORE_TARGET_SOC - s)
```

当前K为722.5263157895元/SOC，参考SOC为0.6。参数来源分别为既有v2经济目录、PlantConfig、已校准电池效率及v3岸充目标，新增实现没有复制1.10、624或0.95常量。

`src/v4/control.py` 对每个连续ONBOARD段独立保存起始SOC：

```text
r_original = -(actual economic costs + beta_soc*phi(actual_SOC_next))
             # 最后一步还沿用既有真实SHORE或MODELED结算
immediate_battery_energy_adjustment = -(B(actual_SOC_next) - B(SOC_before))
terminal_correction = B(actual_onboard_SOC_T) - B(voyage_SOC_0)  # 只在该段最后一步
r_new = r_original + immediate_battery_energy_adjustment + terminal_correction
```

phi仍为低于0.4或高于0.6时到工作区间的平方距离，区间内为0；默认对照beta为500。校正使用`next_physical.soc`，而非岸电更新后的`physical.soc`。下一ONBOARD段重新保存岸充后的真实起始SOC。失败未完成后缀不加完整段校正。

实际账本、船上充电的氢耗/FC退化/电池退化、真实SHORE费用、MODELED充电退化及末端结算调用保持原样。即时反馈与校正均不进入四项经济账本，SOC变化不再重复应用充放电效率。

每段有：

```text
sum(r_new-r_original)
 = -(B(SOC_T)-B(SOC_0)) + (B(SOC_T)-B(SOC_0)) = 0
```

因此这是奖励时间重分配。它不改变完整航段的总优化目标，也不提供新的供电完成性保证。

## 2. 更新调度与n-step

`experiment_schedule.py`支持episode16、replay32、replay16；历史replay8保留用于旧协议复现。后两种按真实经济replay累计插入数给予额度，容量饱和后仍计新插入，余数跨航段和轮次保留。初始化target复制单列，其后按实际经济optimizer更新计数hard sync；新入口可选择250/500/1000，默认500。outcome更新不触发同步。

旧协议曾按“完整样本”而非样本内部每个ONBOARD段给予16次更新。为避免静默改变基线，`run_monitored_training`的旧调用默认仍为`episode_credit_scope=sample`。新入口默认`voyage`，每个真正完成的ONBOARD段给予16次额度，包括后来失败样本内已完整结束的前段；无动作的失败段不获得episode额度。配置和实际计数均写入报告。

`dqn.py`支持n=1/8，默认1。完整段的第k个经济replay条目为：

```text
R = sum(gamma**j * r_new[k+j], j=0..m-1)
y = R + gamma**m * Q_target(s_next, argmax_feasible Q_online(s_next))
m = min(n, 当前完整ONBOARD段剩余步数)
```

段尾缩短；done和next-action mask来自最后一个实际transition，不跨SHORE或其他航段，也不接受缺少终端的失败后缀。每个原决策仍对应一条replay条目，故n-step不膨胀入库额度。gamma仍为1，MLP 8-128-64-61、61动作、Adam、Smooth L1和梯度裁剪10不变。

终端费用在1-step下仍需通过采样与TD逐步向前传播。n=8只把已完成段的真实终端费用与校正直接覆盖到最多8个先前决策，不能补救未进入经济replay的失败后缀。本次未将n=8混入任何正式对照。

## 3. 监控与入口

`monitored_training.py`保留每轮greedy Train，以及Train全完成后才执行Validation的规则。新入口预先校验完整30/8样本，只有两集合全完成才保存经济合格checkpoint，按不含软惩罚的Validation可比经济费用选择。评估不写replay、不更新网络，恢复agent/outcome/Torch训练随机流。

每轮记录四项实际成本、MODELED四项及费用、SOC软惩罚、原奖励/即时调整/校正/新奖励；保存探索和greedy的FC/Battery/SOC逐步轨迹及原经济账本。完成样本的总经济费用与失败保留前缀费用分开解释，不把失败截断后的较低部分费用视为全集合比较结果。失败样本前导SHORE若未附着到保留transition，其费用不能由这些transition统计恢复，报告明确标注该范围。

记录SOC四档、均值/最小/终端值、FC=0、FC启动数、实际保留的ONBOARD transition数、policy调用数、经济replay入库、bootstrap与后续经济更新、target复制、TD loss/error均值/绝对均值/RMS/最大值。

`diagnostics.py`使用三个固定合成状态（低SOC/中高负载、工作SOC/高负载、高SOC/低负载），每轮记录61个Q值、完整动作排序、可行集合、贪婪动作及FC=0排名。它们用于观察同一状态的排序演化，不代表训练状态的频率分布，也不构成安全保证。

新入口为 `python -m v4.feedback_study`，必须显式给出`--output-dir`和`--rounds`，只运行一个配置，不创建并行任务。支持`--reward-feedback original/redistributed`、`--cadence episode16/replay32/replay16`、`--target-interval 250/500/1000`、`--n-step 1/8`。默认beta500、redistributed、replay32、target500、n1只是初始配置，不表示replay32的效果已优于其他调度。

旧`staged_study`保留历史训练协议；输出路径保护阻止它及新入口写入上一阶段归档（包括`--resume`）。要复现旧协议必须选择新的输出目录。正式训练命令本次未执行。

## 4. 契约与合成验证

测试覆盖A模拟补能、B高于0.6终端SOC、C真实SHORE、D两段ONBOARD夹SHORE、E船上FC充电、F跨0.4/0.6、G不变SOC，以及H的不同长度/初始SOC；逐项核对四项实际账本严格相等、MODELED及真实SHORE分类不变、即时符号和每段累计奖励恒等。

人民币奖励恒等式容差为`rel=1e-12, abs=1e-8`；物理状态及各RawCnyIntervalLedger以严格相等核对。Float32 TD目标测试使用`abs=1e-6`，与双精度人民币账本容差区分。

纯合成数据的实际计数如下，batch64、target间隔250；没有用这些数字判断算法经济性：

| 调度 | 合成轮数 | 总经济replay新增 | Bootstrap经济更新 | 后续经济更新 | 总经济更新 | 剩余credit | Target复制（含初始化） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| episode16 | 1 | 240 | 480 | 480 | 960 | 0 | 4 |
| replay32 | 2 | 360 | 3 | 8 | 11 | 8 | 1 |
| replay16 | 2 | 360 | 7 | 15 | 22 | 8 | 1 |

另测250/500/1000同步间隔、FIFO饱和计数、数据未够batch时额度保留、跨轮余数、n-step段尾mask、拒绝无边界拼接，以及新CLI在合成数据上的n8单轮调用。旧路径训练网络权重、replay和随机流的回归一致性测试保持通过。

最终相关测试为 **153 passed，330 subtests passed**。10条固定轨迹的累计奖励最大绝对误差为 **1.1368683772161603e-13**；各四项实际账本与MODELED账本严格相等。

结果清单见 [契约记录](results/v4_reward_feedback_implementation_20261008/reward_contract_verification.json)及[测试与计数核验](results/v4_reward_feedback_implementation_20261008/verification.json)。上一阶段93个原始归档文件SHA256及5个数据manifest哈希再次核验，不覆盖旧报告、图、日志或best权重。报告中的`formal_training_optimizer_updates`仅沿用“bootstrap之后训练阶段”的字段命名，本次对应的验证运行全部为合成数据。

## 5. 失败轨迹仍未解决的学习问题

失败样本中已结束的前段进入经济Q；实际未完成后缀继续仅进入outcome，经济Q仍学不到该后缀的长期供电失败后果。即时反馈、n-step和更频繁更新均不改变这个数据排除事实。outcome仍不参与动作选择；其整段结果标签也不是每个早期动作的因果责任。

修复了“岸电后下一段首步就无可行动作”时，把上一段已完成轨迹传成`failed=True`而令校验报错的问题。现在前段按成功结果保留；尚无已执行动作的新失败段单独记录事件计数，不创建假动作或假terminal。

独立解决方向应先评估现有完成性标签/模型在Train与Validation上的判别及校准，再讨论把完成性作为与经济Q分开的约束学习信号。是否参与决策、如何分配失败责任需要独立实验与确认；本次没有添加失败人民币罚款或安全网络。

奖励校正需要控制器保存段起始SOC；多段运行时该量未新增到既有8维网络状态中。不同起始SOC下的终端反馈含有历史常数，这是后续分析的一个状态表达限制，当前按用户要求未改状态/网络。

## 6. 下一步最小对照（等待确认）

先只比较原奖励和重分配奖励两组：固定beta500、同一replay32调度、target500、n1、相同seed/训练预算/epsilon进度和greedy选模规则，从头独立初始化。预算及epsilon衰减进度在启动前确认，不直接把新短程结果当成归档40轮的同配对照。

随后若需要，固定奖励再比较episode16/replay32/replay16；n8放到单独的一因素实验。失败后缀的完成性信号问题亦须单独处理。本次提交后等待用户确认，不自动启动这些实验。
