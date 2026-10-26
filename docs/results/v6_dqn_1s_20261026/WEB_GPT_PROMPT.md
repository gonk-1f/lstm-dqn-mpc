# 给网页版GPT的V6证据审查入口

请查看仓库 `gonk-1f/lstm-dqn-mpc` 的 **`feat/v6-dqn-1s`分支**，入口为 `docs/results/v6_dqn_1s_20261026/README.md`。本次正式DQN已完成8轮，不要将仓库历史V4/V5或run01配置混入run02结论。

先读 [ANALYSIS_CONTROL.md](ANALYSIS_CONTROL.md)、[ANALYSIS_Q.md](ANALYSIS_Q.md)、[逐轮指标](browser/round_metrics.csv)、[TD统计](browser/td_fit_all_rounds.csv)和[失败前最后决策](browser/validation_last_executed_before_failure.csv)。再沿源码与原始记录核实，不能仅转述已有分析。

重点问题：

1. 当前代码是否存在具体实现错误？检查1秒时间索引、状态推进、电池正负方向/SOC积分、奖励量纲、动作mask、失败惩罚附着时序、终止停止bootstrap、Online梯度与Target纯硬复制。没有发现错误不等于证明整个实现正确。
2. 为什么失败终止MAE从9656降到7253，Greedy Validation却从7/8变成3/8？结合Uniform Replay、Huber beta1的有界样本梯度、gamma1与失败终止真实采样比例分析。区分源码可证机制、数据支持的解释和需要额外实验才能验证的因果。
3. 对第8轮002/016/033/044/045分别核对失败前300秒、末次13动作Q排序、SOC及物理可行FC范围。区分FC持续偏低、调整过晚、负载趋势与最终1秒跳变；不能笼统归因于Q网络或档位数。
4. 查固定高SOC探针002/F−5后期为何偏好+60，及实际低SOC状态为何仍保持0/低FC或继续下降。固定旧轨迹探针不等于当前策略实际访问状态；无合法动作状态不应强行排序合法Greedy。
5. 按共同完整的020/040两条比较8轮经济成本/FC变载，避免把失败截短造成的低累计成本解释成省钱。所有轮次固定8条全套完整成本均未形成。
6. 给出最少必要的下一版建议与精确源码位置。优先区分一步mask的可恢复性缺口与终止价值边界拟合不足。若扩大最大FC增量，明确这是仿真假设变化，不能冒充实船爬坡标定。

读取资料时：`browser/`纯文本CSV可直接看；所有压缩完整轨迹、各轮checkpoint、Replay样本及功率图在 `raw/run02/`。原始13动作Q/合法排名见 `browser/q_all_13_actions_all_checkpoints.csv` 和 `raw/run02/q_probes/round_NN.json`；每个轨迹CSV.gz也保留每秒全部13Q。

请勿将TD target或探索行为实际累计回报当作Q*。不要用Validation构造训练目标，不读取Test，不补造未保存的完整Train轨迹。只提出建议，不自动修改算法或启动新实验。
