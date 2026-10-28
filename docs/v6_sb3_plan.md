# V6 SB3 对照：实现计划与训练边界

基准实际HEAD：`fce060ca12fd842d8496aba8d6f922eae179075e`；基准工作区干净。
新工作树分支：`feat/v6-sb3-baseline`。SB3固定官方v2.9.0 / `8908708f10c8ff29759c67f55c8acb56cab27463`。

用户已明确：**本次只准备实现和必要固定样本检查，正式训练由用户在PyCharm终端启动**。
预算选择为固定8遍，每遍35条Train各执行一次；失败时刻随策略变化，总步数/更新数允许变化。
这属于按遍数匹配，不冒充与旧V6恰好2,782,159次更新相同。

1. 实读官方SB3与CleanRL源码，记录14项差异；Tianshou只作后续PER参考。
2. 用有界队列把现有V6 execute物理执行器适配为Gym步骤，不修改原环境。
3. 继承真实SB3 DQN/Replay/OffPolicy收集闭环，扩展动作与TD掩码、按Adam1000次硬Target。
4. 匹配50,000条无Adam warmup、原航段顺序、按pass/episode的epsilon日程及其余run02参数。
5. 逐pass输出控制/Q/TD/成本对照，复用原诊断与真实Train失败前缀，Validation只推断。
6. 每pass保存SB3模型、完整Replay、优化器与随机状态、课程位置。Ctrl+C请求在完整步骤后保存；中途物理状态按已保存动作前缀重建。
7. 只运行少量固定动作、固定Q和短人工航段闭环检查，不执行正式航段训练或读取Test。
8. 交付明确的PyCharm PowerShell启动/恢复命令。终端只显示启动与8个轮次摘要，不启用进度条/TensorBoard；细节流式写文件。

不加入PER、Double-DQN、额外可恢复性约束或新奖励；不修改第三方仓库、旧V6模型和归档。
