# inverse_one_plus_economic 环境与失败语义设计审计

审计基础：`2c2577e`；分支：`feat/v6-sb3-one-plus-reward`。本文写于生产实现之前，仅核对源码、数学域及已有合成测试夹具。未运行训练、真实航段或 Test。实现须等待本次独立审计提交完成。

## 1. 精确定义与数值域

新增独立模式 `inverse_one_plus_economic`。保留既有固定参考效率 SOC 等效价值；不引入实时效率库存，也不改物理账本。

正常执行转移的金额成本为：

\[
C_t=\operatorname{fsum}(C_{H2,t},C_{FCdeg,t},C_{Batdeg,t},C_{SOCeq,t}).
\]

本任务为无岸电独立航段，原账本 `shore_cost_cny=0`。SOC 项为原来的有符号参考价值，不是新增氢气物理流；不重复添加终端 SOC 成本、退化或额外 SOC/ramp 惩罚。

将 C 的数值按元计入指定映射：

\[
d_t=1+C_t[\mathrm{CNY}],\qquad r_t=1/d_t.
\]

“1”是此次明确选定的数值映射常数，不能把该式描述为单位无关或累计真实成本最小化的等价变换。元的数值应进入实验身份；不能换成分后使用同一公式。

所有真实转移先检查：C 为有限实数，d 为有限且严格正的实数，最终奖励有限。`C=-1`、`C<-1`、NaN、Inf 均明确报错，不能 epsilon、abs、clip、重采样或转成物理失败。真实失败末步同样先检查这个域，不能以 −1 掩盖异常成本。

| C（元） | d | 正常奖励 |
|---:|---:|---:|
| 0 | 1 | 1 |
| 1 | 2 | 0.5 |
| 3 | 4 | 0.25 |
| −0.5 | 0.5 | 2 |
| −1 | 0 | 异常 |
| −2 | −1 | 异常 |

因此正常奖励仅在 `C>=0` 时位于 `(0,1]`；当 `−1<C<0` 时奖励大于 1，且 C 从右侧接近 −1 时无上界。原来的 idle 零成本可以合法计算为 1，但这不消除其他负成本或长期目标排序风险。不得把“增加 1”表述为全状态域稳定性证明。

## 2. 现有调用链与最小修改边界

`DirectVesselEnv.step` 每次只调用一次 `EconomicMPC.interval`，得到实际下一物理状态、原金额账本及电池功率，再调用 `onboard_reward_components`。后者已有 `zhou_economic_only` 分支将原账本成本与固定参考 SOC 项相加，且不添加 SOC/ramp 惩罚。

建议新 `src/v6_sb3/rewards.py` 负责金额分项校验、`fsum`、分母检查、精确映射及稳定的模式身份。新模式在环境中复用 `zhou_economic_only` 的 SOC 项计算；不把新模式塞进所有 v4 历史调用者，也不改 v4 物理执行、经济公式或失败重放函数。新 helper 和 episode metrics 必须使用同一组原成本分项；保留 `EconomicCost` 金额与 `Return` 强化学习回报的区分。

`DirectTransition.reward_cny` 是既有字段名，本模式实际承载无量纲学习奖励。为避免误读，模式身份和诊断字段须显式说明 reward unit；成本仍从原 ledger 与 `soc_equivalent_hydrogen_cost_cny` 获得，不能由 `−reward_cny` 推算。不为本次新增模式全面重命名历史 dataclass 字段。

## 3. 真实失败必须替换最后真实步

原环境在真实执行后检查下一物理/策略候选动作集合。若存在执行过的末步、航段未完成且后继候选为空，会调用 `prepare_failure_replay`。该旧函数在最后真实步上减去旧金额惩罚 F，并设置 `done=True`、清空后继动作、标记 `failure_<cause>`。这符合“末步终止”，但其奖励公式不符合新模式。

新模式应独立处理已分类的真实无动作失败：在验证该实际末步成本及分母后，将它的训练奖励**替换为 −1**；保持该步原物理状态、动作、实际账本、SOC 项、时间、执行步数不变。设置 `done=True`、`next_feasible_actions=()`、相应 `failure_<cause>`，保留正常映射结果作为诊断 `original_reward`。不先给正常奖励再附加 −1，不额外生成失败动作或转移，不计入金额成本。

新模式完全不调用 `FailurePenalty.from_reference` 或 `prepare_failure_replay`，不依靠把旧 F 乘零伪装成独立实现。构造时若用户传入非 None 的旧 `penalty` 参数，应明确拒绝；新模式 `self.penalty=None`。旧两种模式保持原有 F 读取、奖励及重放路径。

| 事件 | 新模式行为 |
|---|---|
| 正常完成最后一秒 | 正常 `1/(1+C)`；`terminated=True` |
| 后继因 SOC 无合法动作 | 最后真实步奖励替换 −1；真实失败终止 |
| 后继因增量动作限制无候选 | 最后真实步奖励替换 −1；记录 `increment_action_limited` |
| 后继负载超过总功率结构上限 | 最后真实步奖励替换 −1；记录 `structural_power` |
| reset 即无动作 | 保留 `InfeasibleReset`，零真实步、无伪造奖励 |
| 用户传入 mask 外动作 | 抛异常，不能推进物理或伪造失败 |
| 数据/文件/数值异常 | 抛异常，不能转换为 −1 |
| 暂停训练预算 | 保存真实前缀；不制造终止步或 −1 |

`terminated=True` 应使 SB3 replay 的 terminal done 阻断 bootstrap；`truncated=False` 保持当前完整航段语义。必须用 replay/TD 测试验证，不能只检查 Gym 返回值。

## 4. checkpoint 与旧模式隔离

环境 snapshot 新模式应记录 `reward_mode`、公式/版本/金额单位/失败替换规则，以及 `failure_penalty=None`。恢复时必须检查这些身份；新模式与历史模式互相恢复应失败。同模式恢复不得读取旧 F、重放前缀或执行新动作。

旧模式 snapshot 格式和环境结果尽量保持原行为；需要新增字段时应清楚限定新模式身份，不使旧快照因无关字段被悄悄重新解释。训练 config、run 目录、Optuna Study、checkpoint 还需由对应接口实现进行隔离，不能只依赖环境中的模式字符串。

## 5. 可复用的合成夹具与计划测试

`tests/test_v6_direct_environment.py` 已提供 SimpleNamespace 独立航段夹具：`load_kw/time_s/dt_s` 明确为每秒，并覆盖原两模式与冻结 `replay_episode` oracle 的逐步相等性。

新增 `tests/test_one_plus_reward.py` 应覆盖以下独立合同：

1. helper 正常正成本、零成本、`−1<C<0` 的精确结果与单步排序；`−1/更小值/NaN/Inf` 拒绝；失败 override 前仍执行成本/分母检查。
2. 一秒 `load=0, FC=0`：金额成本零，正常完成奖励为 1；不是失败、无 −1。
3. FC 运行充电、运行放电、停机放电：与旧 `zhou_economic_only` 相同动作序列的状态、原 ledger、SOC 项、mask 逐步一致，仅正常学习奖励变为指定映射。
4. `(100,1400)` 负载序列、首步 FC=60：后继增量动作失败；`(100,1900)`：后继结构无解；两者执行一次 interval、只有一条真实转移、奖励为 −1、旧金额惩罚为零、不可再次 step。
5. 低 SOC 附近可通过合法 snapshot 构造短合成 SOC 失败，避免为了触发失败运行 1600 秒；断言与原 mask/SOC 数学一致。
6. 首行 `(1400,)` 或 `(1900,)` 保留无真实转移的 reset 事件；异常退出、非法 action、损坏输入不能伪装成 physical failure。
7. 将 `FailurePenalty.from_reference` 和 `prepare_failure_replay` mock 为“调用即抛错”，新模式完成、失败及 snapshot/restore 均不触发；显式旧 penalty 参数拒绝。
8. 新模式正常前缀与失败边界 snapshot/restore 精确；身份不匹配拒绝；恢复不增加 accountant.interval 调用数。
9. 旧 `tests/test_v6_direct_environment.py` 全部保留通过，验证旧模式结果未改变；金额 metrics 在新模式下仍按 ledger+SOC 统计，不把 return 当成本。

本报告记录的是实施前设计；没有将计划测试写成已经通过的结果。实施后须另报实际运行命令、测试数量、失败项和必要的 smoke 结果。
