# v4 UNKNOWN 截断与选模口径

正式入口 `src/v4/feedback_study.py` 每轮均以 ε=0 评价全部 Train 和 Validation 样本，Train 结果不再决定是否运行 Validation。按当前 v3 工况标签自动识别不含 UNKNOWN 的完整可评价集合；含 UNKNOWN 的样本仍运行至首个未知点，以记录截断前是否发生物理失败及已执行步数。

当前冻结标签核查为 Train 21/30、Validation 7/8 可完整评价；这些数量由标签计算，代码没有写死 21 或 7。

报告分别保存完整可评价样本数、完成数、物理失败数、UNKNOWN 样本数、实际数据截断数和 UNKNOWN 前执行的 ONBOARD 步数。Validation 可比费用只汇总固定完整集合；其中任一样本未完成时费用记为 `null`。合格 checkpoint 要求 Train 和 Validation 的固定完整集合都完成、所有含 UNKNOWN 样本的已观测前缀无物理失败，再以固定 Validation 集合的完整费用选最优。程序执行错误不能取得资格。

UNKNOWN 属于观测截断，不是供电失败或成功终止。其未闭合尾部不进入经济 Q replay，不生成终端补能或失败罚款；仅此前有真实成功终止的闭合前缀可保留。每轮及 bootstrap 记录因此舍弃的经济 Q transition 数量。同一样本内已确认 SHORE 前后的正常 8-step 回报保持连续，不能跨 UNKNOWN 接续。旧 checkpoint 与 v3 标签任务不兼容，也不用于续训。
