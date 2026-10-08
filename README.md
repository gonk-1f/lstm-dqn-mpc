# 船舶燃料电池与锂电池能量管理

当前分支 `feat/direct-power-mlp-ddqn` 使用 **30 s 直接功率 MLP Double-DQN**。
FC 动作为 0–600 kW、间隔 10 kW，电池承担剩余负载；单步物理 mask 保持功率与 SOC 硬边界。
物理、退化和经济模型复用 v2/v3，当前控制器不调用 MPC 求解、不启用 KAN。

## 当前入口与文档

- [文档总索引](docs/README.md)：当前方法、实施报告、模型依据及历史资料分类。
- [v4 方法与默认配置](docs/v4_direct_power.md)：奖励、失败终止、经济账本与选模资格。
- [失败经验进入经济 Q 的实施验证](docs/v4_failure_economic_td_2026-10-08.md)：d111c99 的修复与合成证据。
- 当前单配置入口为 [feedback_study.py](src/v4/feedback_study.py)，默认 beta500、redistributed、replay32、target500、batch64、γ1、n1、epsilon 1→0.05。奖励缩放仍为 1。

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
| `src/v4/` | 当前控制、Double-DQN、奖励重分配、失败 TD、训练监控 |
| `src/v2/`、`src/v3/` | 保留 MPC 与当前共享物理/经济/数据依赖 |
| `src/dqn/networks/` | MLP/KAN 实现，当前只启用 MLP |
| `tests/` | 当前与共享依赖回归；文件较旧不代表可删除 |
| `docs/results/`、`docs/figures/` | 已审核实验、评估和原始归档证据 |
| `docs/history/` | 旧报告、研究资料和已执行计划；不代表当前默认值 |
| `outputs/` | 本地生成物及必要校准输入，不能整体清空 |
| `data/` | 正式数据和冻结划分，本次不改动 |

旧 `v4.train`、`v4.review`、`v4.staged_study` 和 v2/v3 入口保留用于历史复现。
它们的默认配置或失败处理可能不同，不能代替当前 `v4.feedback_study` 的单变量对照。
MPC 分支、默认主分支及两个现有工作树保留，不改写 Git 历史。
