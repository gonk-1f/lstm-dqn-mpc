# v2 时间尺度与求解器可靠性审计

## 当前结论

正式 baseline 已冻结以下项目设计配置，但正式训练仍为 **NO-GO**：

- `Ts_MPC = 30 s`；
- 下层预测长度 `N_MPC = 5`；
- 上层动作保持步数 `M = 5`；
- `tau_LPF = 90 s`。

`N_MPC` 与 `M` 的语义不同。前者是一次滚动优化向未来预测的步数，后者是一个
DQN 权重动作实际保持并执行的 MPC 周期数。在 `Ts=30 s` 下，两者数值都为 5，
分别形成 150 s prediction horizon 和 150 s macro interval。配置状态为
`FROZEN_PROJECT_DESIGN`；证据状态为 `PROJECT_DESIGN`，不声称数据优选、文献全局
最优或实船标定。既有审计接口只保留为非阻塞诊断快照。

## Train-only 边界

以下方法选择或校准集中列入 `SelectionParameter`，全部只允许使用带有当前 `DATASET_VERSION`、非空来源标识和精确 `DataSplit.TRAIN` 的 `DatasetProvenance`：

- `N_MPC`；
- `dqn_switch_steps`（`M`）；
- `tau_LPF`；
- SOC deadband；
- state schema；
- action catalog；
- reward scale；
- objective normalization。

Validation、Test 和 Unknown 只能在方法冻结后用于评估，不能回流选择上述项目。入口接收惰性的 payload loader，并在调用 loader、读取样本或运行求解器之前校验 split 与 provenance。字符串 split、枚举仿制品、旧数据版本、注入字段和修改后的 provenance 均失败关闭。

该软件边界能阻止直接把 held-out payload 送入审计函数，但不能证明实验人员从未在系统外查看 held-out 结果。数据权限、实验登记和人工复核仍是必要的治理措施。

## 时间序列诊断定义

`run_timescale_audit` 防御性复制有限实数序列，并记录以下量：

1. 自相关：对声明的正整数 lag `l`，使用全序列均值和全序列中心平方和，
   `acf(l) = sum[t=l..T-1]((x[t]-mean)(x[t-l]-mean)) / sum[t=0..T-1](x[t]-mean)^2`。
   常数序列的分母为零时定义为 0，而不是产生 NaN。
2. 滚动方差：长度为 `W` 的每个完整窗口使用总体方差
   `sum((x-mean_window)^2) / W`，不对首尾补值。
3. 变化点：当相邻样本满足 `abs(x[t]-x[t-1]) >= threshold` 时记录索引 `t`。
4. 工况持续时间：由序列起点、变化点和序列终点形成分段，分别记录样本数和乘以 `Ts_MPC` 后的秒数。

这些统计量用于描述相关长度、局部波动和工况停留时间，不自动推出控制或决策时间尺度。变化阈值、lag 和窗口也是方法设定，正式使用前必须在 Train 上预先登记并说明物理含义。

## `M={5,10}` 敏感性

对每个固定候选 `M`，审计只使用从序列起点开始的完整、不重叠宏区间，记录：

- 完整宏区间数量；
- 每个宏区间的均值；
- 各区间总体方差的平均值；
- 未进入完整区间的尾部样本数。

任何尾部截断都显式入账，不能静默丢弃。当前接口不允许增加、删除或重排候选，
也不允许改变 `N_MPC=5`。该诊断不再负责冻结 M，也不阻塞 formal training；未来
论文若报告 sensitivity，仍只能使用 Train 数据且不得由 Validation/Test 改变 baseline。

## 配对冷启动/热启动审计

`run_paired_solver_audit` 要求 case 使用唯一 `case_id`、非负 seed 和从 0 开始的连续 order。对每个 case，按同一 payload、seed 和 order 依次请求 cold 与 warm 运行。runner 必须回报相同的 case/seed/order 和正确 start mode；接口不会替错误元数据“纠正标签”。输入 case、运行观测和最终结果均被复制成不可变快照。

结果分别报告：

- cold 成功数与成功率；
- warm 成功数与成功率；
- cold/warm 同时成功的配对数；
- 仅在双方都成功的 case 上计算 `cold_seconds / warm_seconds` 的中位数，并列出实际进入速度比较的 case ID。

失败运行绝不进入速度统计，因此“热启动成功率更高/更低”和“热启动更快/更慢”是两个独立结论，不构造混合总分。计时必须为有限正数，迭代数、status、seed 和 order 使用非布尔整数，success 使用精确布尔值。

## 密封与可复核性

两个顶层审计结果都只能由审计函数创建。结果包含确定性 SHA-256 digest；`validate()` 会重新检查 exact types、有限值、固定候选、配对身份、派生汇总和 digest。调用方对原列表或 runner 返回对象的后续修改不会改变已保存结果；通过 `object.__setattr__` 篡改冻结对象会在复核时被发现。

digest 证明的是同一序列化内容得到同一标识，不是数据真实性证明。正式归档还必须保存原始 Train 数据清单、文件哈希、预处理版本、case 构造规则、求解器及硬件/软件环境。

## 尚缺证据

仓库当前没有满足 v2 原始数据门槛的真实、时间戳化 Train 工况循环，也没有覆盖正式 action catalog 和真实工况 case 的重复配对求解报告。因此目前不能声称：

- `N_MPC=5` 已由数据选定；
- `M=5` 优于 `M=10`；
- warm start 在正式案例上更可靠或更快；
- `N=5`、`M=5` 或 `tau_LPF=90 s` 是唯一最优值或实船标定值；
- state、action catalog、reward scale 或 objective normalization 已完成全部
  Train-only 审计（SOC deadband 已由方法定义固定，但未因此解除其他 gate）。

N、M 与 tau 的 frozen configuration 已通过 preflight，未升级其 evidence 等级。
正式训练因 dataset/episode payload、最终 state、最终 action catalog/K 和最终集成
solver robustness 仍保持 **NO-GO**。
