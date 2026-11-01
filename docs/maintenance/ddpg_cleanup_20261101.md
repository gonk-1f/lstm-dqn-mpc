# DDPG 清理记录：2026-11-01

## 范围与基准

- 工作树：`C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline`。
- 分支：`feat/sb3-ddpg-aggregate`，未切换分支。
- 执行前 HEAD 与 fetch 后远端均为 `90187aa0b8ee74f4902083c9b1138435427bd362`；`git rev-list --left-right --count HEAD...origin/feat/sb3-ddpg-aggregate` 输出 `0 0`。
- 执行前 `git status --porcelain=v1 -uall` 无输出。
- 不改动 DDPG 算法、配置、数据、虚拟环境、其他工作树；未运行训练、Optuna 或数据重建。
- 已有稀疏检出将 `docs/results/` 的 391 个跟踪文件排除在磁盘工作区之外；`docs/audits/` 本地也不存在。本次未删除这些历史记录、未调整稀疏检出。

## 实际删除的旧脚本和配置

| 路径 | 删除依据 |
| --- | --- |
| `experiments/sb3_manual/test_masked_dqn.py` | 旧 V6 MaskedDQN 物理前缀测试，引用已删除的 `v6_sb3`，当前 DDPG 不引用 |
| `experiments/sb3_manual/benchmark_cpu_gpu.py` | 旧 MaskedDQN CPU/GPU 基准工具，当前 DDPG 不引用 |
| `experiments/sb3_manual/compare_models.py` | CartPole DQN 模型比较工具，当前 DDPG 不引用 |
| `experiments/sb3_manual/dqn_custom.yml` | CartPole-v1 DQN 配置，当前 DDPG 不引用 |

检索范围为当前 `experiments/`、`src/`、README 与配置中的文件名、MaskedDQN、smoke、sanity 引用。未发现当前 DDPG 对上述文件的引用，也未发现其他需要删除的有效测试脚本。保留 DDPG 仍依赖的 `src/v2/`、`src/v3/`、`src/v4/` 模块原位置。

## 冒烟输出：尝试删除但未执行

| 路径 | 文件数 | 字节 | 实际结果 |
| --- | ---: | ---: | --- |
| `outputs/ddpg_smoke_001/` | 1 | 58 | 自动审批拒绝，目录仍在 |
| `outputs/ddpg_smoke_002/` | 4 | 3,949,023 | 自动审批拒绝，目录仍在 |
| `outputs/ddpg_zoo_smoke/` | 5 | 3,951,270 | 自动审批拒绝，目录仍在 |

对这三个明确授权目录执行原生 PowerShell `Remove-Item -LiteralPath ... -Recurse -Force`，在进程创建前返回 `rejected: blocked by policy`，未执行任何输出删除。未使用其他解释器或工具绕过限制。本次实际删除输出目录数为 **0**；待删除合计 10 个文件、7,900,351 字节（7.534 MiB）。其中冒烟模型属于待删除项，不能将它们报告为已删除。

## 检查后保留的验证与日志

- `outputs/ddpg_val_intermediate/`：与 `ddpg_val_after_36ep/` 的全部 8 条航段指标不同。两者完成率均为 7/8，但 mean_coverage 分别为 0.9078775376822842、0.9107675340059638，SOC、reward_sum 等也不同；不是被最终结果完全覆盖的重复副本。
- `outputs/ddpg_validation_001/`：8 条验证中完成 2 条，mean_coverage 为 0.44906656673591405。JSON 未记录模型来源；仅凭早期时间戳不足以认定其没有独立价值，保留待补充来源。
- `outputs/ddpg_validation_plots/`：含 8 份独有逐步轨迹 CSV、8 张 PNG 和汇总 CSV/JSON。汇总结果对应保留的 `ddpg_zoo_500ep` 模型。删除整个目录会丢失轨迹 CSV，故完整保留。
- `outputs/ddpg_zoo_tensorboard/`：两个 event 文件含 `rollout/ep_len_mean`、`rollout/ep_rew_mean`、`time/fps`、`train/actor_loss`、`train/critic_loss`、`train/learning_rate`。最后标量步数分别为 177216 和 410380；保留这些训练曲线。
- 检查后删除的重复验证与日志：**无**。

## 必须保留的模型、Replay 与 Checkpoint

下列文件均存在，且清理前后 SHA-256 一致。

| 文件 | 字节 | 保留原因 |
| --- | ---: | --- |
| `outputs/ddpg_train_50k/ddpg_aggregate.zip` | 3,948,864 | 正式训练模型，保留作评估与对照 |
| `outputs/ddpg_zoo_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip` | 3,949,514 | 正式训练模型，保留作评估与对照 |
| `outputs/ddpg_zoo_500ep/ddpg/AggregateShipDDPG-v0_1/replay_buffer.pkl` | 40,001,549 | 恢复训练的 Replay Buffer |
| `outputs/ddpg_zoo_500ep/ddpg/AggregateShipDDPG-v0_1/rl_model_100000_steps.zip` | 3,949,181 | 有效阶段 Checkpoint，保留作恢复与实验对照 |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/AggregateShipDDPG-v0.zip` | 3,950,338 | 正式训练模型，保留作评估与对照 |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/replay_buffer.pkl` | 40,001,549 | 恢复训练的 Replay Buffer |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/rl_model_100000_steps.zip` | 3,949,181 | 有效阶段 Checkpoint，保留作恢复与实验对照 |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/rl_model_200000_steps.zip` | 3,949,594 | 有效阶段 Checkpoint，保留作恢复与实验对照 |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/rl_model_300000_steps.zip` | 3,949,886 | 有效阶段 Checkpoint，保留作恢复与实验对照 |
| `outputs/ddpg_zoo_resume/ddpg/AggregateShipDDPG-v0_2/rl_model_400000_steps.zip` | 3,950,298 | 有效阶段 Checkpoint，保留作恢复与实验对照 |

同时完整保留 `outputs/ddpg_train_50k/`、`outputs/ddpg_val_50k/`、`outputs/ddpg_val_after_36ep/`、`outputs/ddpg_zoo_500ep/`、`outputs/ddpg_zoo_resume/` 中的监视日志、配置和验证结果。没有修改或移动任何 Replay Buffer。

## 输出目录清单与空间

操作前已生成逐文件 CSV 清单：`C:\Users\20883\AppData\Local\Temp\ddpg_cleanup_before_20261101_152809.csv`（59 个文件，含相对路径、字节、扩展名、最后修改时间；位于本机 TEMP，不加入 Git）。目录汇总如下，时间为本机北京时间。

| outputs 下目录 | 文件数 | 字节 | 文件类型 | 最后修改时间 |
| --- | ---: | ---: | --- | --- |
| `ddpg_smoke_001` | 1 | 58 | .csv × 1 | 2026-10-30T17:37:29 |
| `ddpg_smoke_002` | 4 | 3,949,023 | TensorBoard event × 1, .csv × 1, .json × 1, .zip × 1 | 2026-10-30T17:38:57 |
| `ddpg_train_50k` | 4 | 3,951,899 | TensorBoard event × 1, .csv × 1, .json × 1, .zip × 1 | 2026-10-30T18:37:30 |
| `ddpg_val_50k` | 2 | 6,308 | .csv × 1, .json × 1 | 2026-10-30T18:50:45 |
| `ddpg_val_after_36ep` | 2 | 6,182 | .csv × 1, .json × 1 | 2026-11-01T12:35:35 |
| `ddpg_val_intermediate` | 2 | 6,226 | .csv × 1, .json × 1 | 2026-10-31T13:02:59 |
| `ddpg_validation_001` | 2 | 6,800 | .csv × 1, .json × 1 | 2026-10-30T17:40:50 |
| `ddpg_validation_plots` | 18 | 20,675,422 | .csv × 9, .json × 1, .png × 8 | 2026-10-31T13:12:38 |
| `ddpg_zoo_500ep` | 7 | 47,903,545 | .csv × 1, .pkl × 1, .txt × 1, .yml × 2, .zip × 2 | 2026-10-31T02:23:25 |
| `ddpg_zoo_resume` | 10 | 59,754,871 | .csv × 1, .pkl × 1, .txt × 1, .yml × 2, .zip × 5 | 2026-11-01T01:48:11 |
| `ddpg_zoo_smoke` | 5 | 3,951,270 | .csv × 1, .txt × 1, .yml × 2, .zip × 1 | 2026-10-30T19:07:22 |
| `ddpg_zoo_tensorboard` | 2 | 17,728 | TensorBoard event × 2 | 2026-11-01T01:41:43 |

- 清理前 outputs：140,229,332 字节（133.733 MiB），59 个文件。
- 清理后 outputs：140,229,332 字节（133.733 MiB），59 个文件。
- outputs 实际释放：**0 字节**；不能把预期可释放空间写成实际释放。
- C 盘空闲：操作前 125,391,859,712 字节，操作后 125,388,660,736 字节；净减少 3,198,976 字节。该全盘读数受本机其他写入、临时清单及 Python 缓存影响，不作为清理收益。

## 验证

使用当前工作树的 `.venv/Scripts/python.exe`（Python 3.11.5）；当前进程的 PYTHONPATH 设置为本工作树 `src` 和根目录。

| 命令或检查 | 结果 |
| --- | --- |
| `python -m compileall -q experiments/sb3_manual/ddpg_aggregate` | exit 0 |
| `python -m experiments.sb3_manual.ddpg_aggregate.train --help` | exit 0，正常显示 CLI |
| `python -m experiments.sb3_manual.ddpg_aggregate.evaluate --help` | exit 0，正常显示 CLI |
| 受保护文件 SHA-256 对比 | 1029 个现存文件全部一致，包括核心代码、数据集、保留模型及 Checkpoint |
| `git diff --stat HEAD -- src data experiments/sb3_manual/ddpg_aggregate docs/audits docs/results` | 无输出，保护路径无变更 |
| `data/processed/operating_dataset_v6_1s` | 仍存在 |
| 三个待删除冒烟目录存在性 | 全部仍存在，删除未完成 |
| `git diff --check` 与 `git diff --cached --check` | 均 exit 0；首次暂存检查发现文末多余空行，移除后重新检查通过 |

提交范围限上述四个删除项和本记录；不暂存 outputs，不使用 `git add -A`，不 force push。
