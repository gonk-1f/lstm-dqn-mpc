# 船舶强化学习实验

本工作树保留当前实验 experiments/sb3_manual/ddpg_aggregate/，使用原生 Stable-Baselines3 DDPG。

- 训练入口：experiments/sb3_manual/ddpg_aggregate/train.py
- 独立评估入口：experiments/sb3_manual/ddpg_aggregate/evaluate.py
- data/ 及 Train / Validation / Test 划分保持不变。
- outputs/ddpg_*、模型、检查点、训练日志及 TensorBoard 全部保留。
- Python 虚拟环境和已安装软件包保持原样。

DDPG 直接及传递引用的28个公共源码文件继续保留在 src/v2/、src/v3/、src/v4/ 原位置，未修改或迁移。
部分旧名称的 control、analysis、dqn 模块由包初始化间接导入，仍是当前运行依赖。

最新 SB3 / RL Zoo DQN 位于独立的 v6-zoo-vessel 工作树和 feat/v6-zoo-vessel 分支。
该工作树未参与本次清理；没有复制、合并或改动其他工作树。
