# PER 独立候选准备（2026-10-13）

本工作副本基于运行中的 B 组源码快照；快照 SHA256 在 `v4_per_base_snapshot.json`。不修改运行中的工作树，不提交或推送，不启动正式训练。

## 范围与顺序

1. 添加 FIFO 兼容、稳定 Replay ID 的比例优先采样索引，使用 sum/min tree；先验证采样、淘汰、恢复。
2. 在 Agent 中添加显式 `uniform/prioritized` 选项。默认 uniform 的 RNG、损失和采样保持原状。
3. 保留现有失败终止配额及不足时回退规则；普通池使用有放回 PER。每个样本的 Huber 损失分别乘 IS 权重。
4. 保留全部 TD、n-step、奖励、动作掩码、优化器频率和网络设置；只添加采样配置、优先级保存与诊断。
5. 补正式入口配置和兼容性核对；少量定向测试及独立冻结目标比较；最终核对 live 源码未变。

## 候选算法定义

参考 Schaul et al., *Prioritized Experience Replay*, §3.3–3.4，<https://arxiv.org/abs/1511.05952>。

普通池：`p_i=(abs(TD_i)+epsilon)^alpha`, `P_i=p_i/sum(p)`。候选 `alpha=0.6`、`beta_IS=1`、`epsilon=1e-6` 均可配置，不表示已确定正式参数。

保留原失败配额抽样。普通池大小为 `N_o`，未归一化权重 `(N_o*P_i)^(-beta_IS)`。取池级共同上界 `M=max(1,(N_o*P_min)^(-beta_IS))`，普通权重除以 M，失败池权重也为 `1/M`。beta_IS=1 时恢复的是原固定配额目标的期望（共同正比例缩放），不是全 Replay 均匀目标。配额0时全 Replay 使用 PER。

`loss=mean(weight_i * SmoothL1(Q_i,y_i,reduction='none'))`。不能先求 batch 平均再乘权重。优先级使用未加权 TD，重复抽样 ID 取最大绝对 TD。新样本先赋历史最大优先级。FIFO 删除时同步清除优先级，避免旧引用。

## 验收边界

Batch64、replay32、网络、学习率、奖励及物理约束不改；不添加 SOC/FC 门控，不覆盖 Q 排名。候选默认关闭。保存并恢复优先级和 RNG，拒绝不匹配的采样配置。小规模 TD 拟合结果仅说明采样与优化行为，不能证明闭环完成率或经济性改善。
