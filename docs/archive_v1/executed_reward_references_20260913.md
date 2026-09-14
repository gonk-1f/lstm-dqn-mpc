# 原文核对记录 — 2026-09-13

本轮核对正文而非仅二手摘要。这里只记录与设计有关的位置和边界，不把跨领域研究当作船舶设备参数认证。600/624/0.55/0.05/48 来自当前工程配置与已有项目审计；1/2 是本轮指定的效用系数，不声称七篇论文给出一致的这些数值。

| 原文 | 核对位置 | 可支持的思想 | 不应外推的结论 |
|---|---|---|---|
| Zarrouki et al. 2024 | [arXiv 正文](https://arxiv.org/html/2402.02624v1)，§V-A、§V-B、式8及随后 RMS 定义 | 每个离散 action 对应一组预优化 Pareto 权重；reward 用两个切换时刻之间的实测误差 | 本项目84网格未经其 BO/Pareto 安全筛选，不能继承安全或最优保证；原文用 PPO/MOG |
| Zarrouki et al. 2021 | [作者公开全文](https://www.researchgate.net/publication/357540616_Weights-varying_MPC_for_Autonomous_Vehicle_Guidance_a_Deep_Reinforcement_Learning_Approach)，式4、5 | tracking/comfort 的外部多目标指标；Gaussian 中各指标有中心和尺度 | 其 Gaussian 不是本轮线性氢耗+三项半二次公式 |
| Haspolat & Yalcin 2023 | [全文 PDF](https://www.researchgate.net/publication/371098049_Energy_Management_of_P2_Hybrid_Electric_Vehicle_Based_on_Event-Triggered_Nonlinear_Model_Predictive_Control_and_Deep_Q_Network/fulltext/647221e1a25e543829d0b80a/Energy-Management-of-P2-Hybrid-Electric-Vehicle-Based-on-Event-Triggered-Nonlinear-Model-Predictive-Control-and-Deep-Q-Network.pdf)，§4.2，PDF第17–18页 | DQN 训练 NMPC 功率权重，以效率表现构造反馈 | 其 P2 HEV、电机/发动机效率、事件触发策略和阈值不能直接用作本项目电池连续应力阈值 |
| Mehndiratta et al. 2018 | [作者公开全文](https://www.researchgate.net/publication/330591563_Automated_Tuning_of_Nonlinear_Model_Predictive_Controller_by_Reinforcement_Learning)，§III、式14–16、表II/III | 按位置误差、误差变化、jerk、稳态误差及应用阈值评价 NMPC 权重 | 飞行应用的误差阈值不能转换成未经设备依据的充放电阈值 |
| Sun et al. 2024 | [TU Delft 原文 PDF](https://repository.tudelft.nl/file/File_5f230ded-4f1a-456f-b3e4-4337cd8f90ec)，§3.2、式13（PDF第6页） | RL 周期中 measured states 与 implemented PMPC inputs 组成外部性能评价；分别定义多种时间尺度 | 原文支持时间尺度显式建模，不要求本轮必须改T_sw或实施SMDP |
| Zhai et al. 2026 | [出版商全文](https://journals.sagepub.com/doi/10.1177/16878132261426615)，MPC weight adaptive adjustment method based on DQN 一节及算法 | DQN 根据状态调节 MPC 权重，循环求 QP 和更新 Q 网络 | 履带车轨迹跟踪效果不能视为船舶能量管理的测试结果 |
| Yuan et al. 2025 | 用户本地原文 PDF，第9页 §3.2.2，式51；[DOI](https://doi.org/10.1016/j.ijhydene.2025.151770) | 双层 MPC 接收 Q-learning 权重；两奖励分别为对应最优代价的倒数 1/C | 原文是各自 MPC 预测最优序列的代价反馈，不能说是本项目统一实际首步评分；项目 legacy 的 1/(1+J) 也不等同原式 |

出版商 Yuan 页面本次在线访问失败，因此使用用户已提供的本地论文原件重新核对，没有把失败的网络页面说成已读取。原件 SHA-256：`b49986cdce335eb0b212ea8ea5c9943ffcc1f7c4f147a813c122606890578c87`。

## 工程决定

当前主方法采用统一执行评分。保留三个方法标签供后续消融，不宣称当前版本已经训练收敛或优于旧方法。0.05 的保留依据是既有 Train-only 预测 RMS/L2 边际审计，只能作为暂不扫描的依据，不能代替本次 reward 的新闭环训练验证。

文献核对未发现足以把 battery 连续充放电应力阈值、terminal failure penalty 或本轮三个 1/2 系数视为设备已认证参数的依据；因此阈值不添加、failure penalty不填值，1/2按确定公式保留。
