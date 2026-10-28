# 船舶燃料电池与锂电池能量管理

**当前工作分支 `feat/v6-sb3-baseline`：官方Stable-Baselines3 2.9.0 DQN对照已完成8遍Train，共3,128,257次正式Adam更新。**
**[奖励曲线与完整训练结果](docs/results/sb3_dqn_1s_20261028/README.md)**：Validation完成数为7、7、7、7、8、7、8、5；按预定规则选择第5轮，末轮回退，尚无稳定收敛证据。
见[训练与Test终端命令](docs/v6_sb3_pycharm.md)、[14项源码差异审查](docs/v6_sb3_source_audit.md)。
本分支的基准是下述已完成的V6 run02；新增实现位于`src/v6_sb3/`，原V6源码和历史结果保持原样。

当前分支 `feat/v6-dqn-1s` 使用 **1秒普通DQN、单步TD和13档FC实际增量**。
学习率0.0001，SOC权重0.2，FC波动权重0.05；8轮正式训练已完成，末轮Greedy Validation完成3/8，控制效果尚未稳定。
物理、退化和经济模型复用v2/v3。最大增量±60 kW/s是仿真假设。

## V6源码与完整训练资料

- **[完整资料与阅读入口](docs/results/v6_dqn_1s_20261026/README.md)**：所有逐轮checkpoint、完整Replay断点、64条Validation轨迹、64幅功率/SOC图、逐轮日志及可直接浏览的13动作Q排序。
- [控制与逐航段失败分析](docs/results/v6_dqn_1s_20261026/ANALYSIS_CONTROL.md)、[Q价值与TD拟合分析](docs/results/v6_dqn_1s_20261026/ANALYSIS_Q.md)。
- [给网页版GPT的审查说明](docs/results/v6_dqn_1s_20261026/WEB_GPT_PROMPT.md)、[实际训练入口](src/v4/v6_training.py)、[当前配置及记录口径](docs/v6_lr1e4_q_reporting.md)。

以下V4入口保留作历史方法与共享源码参考，不代表V6当前配置。

## 历史V4入口与文档

- [文档总索引](docs/README.md)：当前方法、实施报告、模型依据及历史资料分类。
- [v4 方法与默认配置](docs/v4_direct_power.md)：奖励、失败终止、经济账本与选模资格。
- [失败经验进入经济 Q 的实施验证](docs/v4_failure_economic_td_2026-10-08.md)：d111c99 的修复与合成证据。
- [最新0.001奖励缩放的正式40轮结果](docs/v4_scaled_reward_40r_2026-10-08.md)：完成性仍未达标，保留逐轮日志、指标、图表及全量轨迹归档。
- 当前单配置入口为 [feedback_study.py](src/v4/feedback_study.py)，默认 beta500、redistributed、replay32、target500、batch64、γ1、n1、epsilon 1→0.05。`--reward-scale` 默认 1；[本次训练准备与PyCharm命令](docs/v4_reward_scale_training_2026-10-08.md)显式选0.001。

查看入口帮助不加载正式轨迹、不运行训练：

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
python -m v4.feedback_study --help
```

训练必须显式选择轮数及全新输出目录。每轮 greedy Train 全完成才评估 Validation；只有 Train30/30、Validation8/8 才具备按经济费用选模资格。
Test 不参与训练或选模。实现验证不能当成正式完成率或经济性改善；本次整理与数值诊断未启动正式训练。

## 仓库布局

| 路径 | 用途 |
|---|---|
| `src/v4/` | V6普通DQN与训练记录；保留V4/V5 Double-DQN及共享控制依赖 |
| `src/v2/`、`src/v3/` | 保留 MPC 与当前共享物理/经济/数据依赖 |
| `src/dqn/networks/` | MLP/KAN 实现，当前只启用 MLP |
| `tests/` | 当前与共享依赖回归；文件较旧不代表可删除 |
| `docs/results/`、`docs/figures/` | 已审核实验、评估和原始归档证据 |
| `docs/history/` | 旧报告、研究资料和已执行计划；不代表当前默认值 |
| `outputs/` | 本地生成物已归档清理；17个旧版正式校准/预检输入保留，不能整体删除 |
| `data/` | 正式数据和冻结划分，本次不改动 |

旧 `v4.train`、`v4.review`、`v4.staged_study` 和 v2/v3 入口保留用于历史复现。
它们的默认配置或失败处理可能不同，不能代替当前 `v4.feedback_study` 的单变量对照。
MPC 分支、默认主分支及两个现有工作树保留，不改写 Git 历史。
