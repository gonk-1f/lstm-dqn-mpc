# Zoo 正式 Episode 预算与探索调度验收（2026-10-30）

本次基于 PR #5 的 `7df880d1aea073e664918cde36389e97370b159c`，在 `.worktrees/v6-zoo-vessel` / `feat/v6-zoo-vessel` 完成。PR #5 开发前为 open/draft，head 与本地一致。只扩展 Zoo 独立路线；没有修改旧 V6 runner、物理模型、奖励、13动作、动态mask、数据划分、第三方包或旧实验。

## 结果与实际运行范围

166 项单元、集成及保留回归通过，退出码0；真实 Train 的3个正式Episode、一个官方Zoo Episode Trial、4096步旧模式回归及保存模型的独立8Val评估全部退出0。没有启动 Zoo 的500Episode正式训练、多Trial长搜索或Test。旧工作树的既有后台训练自行继续；本任务没有向它发送暂停/停止/恢复操作。

| 实际命令路线 | Warmup | 跨界后续步 | 正式环境步 | 正式Episode | Adam | 总Train步 | 完整8Val |
|---|---:|---:|---:|---:|---:|---:|---|
| Train `--n-episodes 3`，完整正式配置 | 50000 | 442 | 21761 | 3（3成功/0失败） | 680 | 71761 | 1/8完成 |
| 官方Optuna `--n-episodes 1 --n-trials 1` | 50000 | 367 | 8815 | 1（1成功/0失败） | 275 | 58815 | 1/8完成 |
| 原timestep `--n-timesteps 4096 --bounded` | 64 | 2704 | 4032 | 不按Episode停止 | 126 | 4096 | 0/8完成 |
| 保存后的3Episode最终模型独立评估 | 不增加 | 不增加 | 不增加 | 不增加 | 不增加 | 不增加 | 1/8完成，逐样本与最终inline评估一致 |

timestep行的正式Episode计数为0：其第一条跨warmup航段未计数，步预算在下一条航段中间结束；这保持原路线行为。表中“正式环境步”包含跨warmup边界航段在warmup之后的经验，绝不丢弃；“正式Episode”只包含后续完整运行。

3Episode未使用`--bounded`，实际网络[256,256]、batch64、Replay1000000、learning_starts50000、32:1更新、Target10000、gamma1、CPU。Optuna保留原sampler，实际抽得 learning_rate=0.00004745232567282912、batch32、buffer1000000、network_width128、Target5000；warmup50000和32:1未改变。Trial0状态 **COMPLETE**，目标值 **1.133028858324654**，不是mean reward。PRUNED/FAIL由辅助边界测试覆盖，不声称真实船舶Trial发生了剪枝；CLI默认pruner仍为none。

Train、Trial和独立评估的8Val完成率均为0.125，平均覆盖率0.2660577166493081，整组完整经济成本为`None`。这验证预算、接口与存储语义，不表示策略达到合格控制效果。

## 实现位置与时序

| 文件及入口 | 作用 |
|---|---|
| `experiments/sb3_manual/zoo_vessel/launch.py:12` / `resolve_budget:48` | 明确互斥预算；选择`--n-episodes`省略值时为500；预算/探索独立默认与显式选择；非法组合在数据/输出IO前报错 |
| `algorithm.py:89` | 复用SB3线性数学，以已完成正式Episode计算epsilon；每航段固定，warmup/bridge为1 |
| `algorithm.py:96` / `collect_rollouts:113` / `_sample_action:145` | 委托原生learn/collector；在下一个动作采样之前退出采集，保留最后terminal及native episode bookkeeping |
| `algorithm.py:153` / `_on_step:188` / `train:199` | Replay写入之后计数；原formal Target时钟；32步边界门控Adam；再通知Episode观察器 |
| `callbacks.py:27` / `_after_episode_boundary:227` | 独立8Val、独立mask、RNG与训练计数保护；每50Episode和最终N，在terminal已存且应有Adam完成之后评估 |
| `callbacks.py:119` / `_record_training_summary:173` | 原生Logger、CSV、TensorBoard、每1000步进度、单一末表汇总 |
| `manager.py:33` / `objective:207` | 自然最大horizon保护上限；保留官方objective/Trial管理，记录实际状态与参数 |
| `env.py` 的 `max_episode_steps` | 只读split-local航段长度元数据；不改变DirectVesselEnv物理执行 |

原生顺序为 `env.step → callback.on_step → Replay写入 → Target/epsilon时钟 → native terminal bookkeeping`。适配层不在最后terminal的pre-storage callback返回False；在下次采样前结束当前collector，原生learn仍负责调用train。formal步未满32时跳过额外Adam，恰好满32保留正常更新。随后执行Episode Validation；下一次collector在预算完成时直接返回，不再执行动作。

Episode模式`learn`得到的保护上限是 `LS + (N+1)*max_train_voyage_steps + 1`。本次N3为196641、Trial N1为123321；实际在71761/58815步结束。上限不充当Episode预算，提前命中明确FAIL。官方Zoo报告文件名仍可能包含该内部上限；实际步数应读取各`run_summary.json`，不能把文件名当成训练步数。

## 真实3Episode边界证据

warmup在总步50000结束，原航段自然继续到50442；跨界442步保持epsilon1、正常Replay/学习，但不计正式Episode。后续三次记录如下：

| 正式Episode | 航段 | 真实执行步 | 整段epsilon | 终止时总步 | 终止后Adam |
|---:|---|---:|---:|---:|---:|
| 1 | parent_007_seg01 | 8448 | 1.0 | 58890 | 277 |
| 2 | parent_029_seg02 | 5221 | 0.6041666666666667 | 64111 | 440 |
| 3 | parent_053_seg01 | 7650 | 0.20833333333333337 | 71761 | 680 |

短预算的epsilon目标也随N缩放。500目标的100/200/300/400完成事件分别为.7625/.525/.2875/.05，第401～500段保持.05，由辅助事件测试验证；未实际训练400/500条真实航段。

71761条经验全部保存，最后一条为真实terminal、后继mask为空，不执行第4个正式Episode动作。21761除32为680余1，所以没有第681次Adam。Target仅在formal10000/20000同步，复制时分别已完成312/624次Adam；formal10000当步不触发Adam，formal20000的复制先于同一步第625次Adam。总同步计数2，含初始网络复制为3。

Replay共12个真实terminal，其中4个物理失败来自warmup/跨界部分，正式3次均成功。独立复核71749个非reset动作都满足上一条经验的后继mask；reset动作还经过环境实时合法性检查。所有terminal用极大后继Q验证仍为原terminal reward，无bootstrap；失败reward为-1，legacy failure_penalty全零。权重、Adam状态和观测/奖励均有限。

## 原生输出与保存模型

实际原生终端表格保存在 [3Episode末表](native_episode3_final.txt)、[Trial末表](native_episode_trial1_final.txt)、[timestep末表](native_timestep4096_final.txt)，来自真实stdout截取，没有重绘。3Episode收尾实际字段摘录：

```text
|    buffer_size                       | 71761                                |
|    pending_transition                | 0                                    |
|    exploration_rate                  | 0.05                                 |
|    completed_formal_episodes         | 3                                    |
|    formal_env_steps                  | 21761                                |
|    n_updates                         | 680                                  |
|    warmup_bridge_env_steps           | 442                                  |
|    warmup_env_steps                  | 50000                                |
|    evaluated_adam_updates            | 680                                  |
|    evaluated_env_steps               | 71761                                |
|    evaluated_formal_episodes         | 3                                    |
```

原生`time/episodes=12`包含warmup航段，正式预算字段恰好3。CSV分别有72/59/5条进度行（Train3/Trial/timestep），覆盖每1000步及唯一收尾，没有重复最终步。TensorBoard真实events经EventAccumulator读取；Loss/TDMAE/counters/Validation均有记录，成本None在CSV空值、TensorBoard不写伪零。

最佳与最终模型分别保存：普通训练在 `dqn_vessel/VesselEnergy-v0_1/best_model.zip` 与 `VesselEnergy-v0.zip`；Trial在 `trials/trial_0/best_model.zip` 与 `final_model.zip`。普通训练同时保存`replay_buffer.pkl`。本次最终Episode评估对应680次Adam，加载最终模型独立评估与其逐样本结果相同。timestep仍保留旧pre-storage评估：4096处inline actor为125次Adam、最终模型为126次，报告没有混称。

## 单元/集成/回归实际命令

在Zoo工作树，使用其`.venv-zoo`，设置`PYTHONPATH=<worktree>\src;<worktree>`，`PYTHONUTF8=1`及OMP/MKL/OPENBLAS线程为1。版本实测Zoo2.9.0、SB3/contrib2.9.2、Gym1.2.3、Torch2.8.0+cu128、Optuna4.9.0、TensorBoard2.21.0，device始终CPU；`pip check`无损坏依赖。见 [版本](runtime_versions.txt)、[pip check](pip_check.txt)。

```powershell
.venv-zoo\Scripts\python.exe -m pytest tests/test_zoo_vessel_env.py tests/test_zoo_algorithm.py tests/test_zoo_episode_algorithm.py tests/test_zoo_callbacks.py tests/test_zoo_episode_callbacks.py tests/test_zoo_manager.py tests/test_zoo_episode_manager.py tests/test_zoo_timestep_baseline.py tests/test_v6_direct_environment.py tests/test_one_plus_reward.py tests/test_sb3_masked_dqn.py tests/test_v6_target_env_steps.py -q --tb=short
```

实际 **166 passed in 67.03s**，exit0；完整短输出见 [final_unit.txt](final_unit.txt)。包含：互斥CLI、warmup精确/跨界、3Episode停止、terminal入Replay/零bootstrap、31/32/33/64/65步尾块、Target时钟、整段epsilon、400/500事件、50～500评估事件、Val隔离及RNG保护、原生stdout/CSV/TB、旧timestep及官方objective剪枝/失败状态。另直接执行`git show 7df880d:.../algorithm.py`的历史算法，与新版timestep同seed比较完整权重、Adam状态、Replay和时钟，全部逐值相同；本机没有跳过该测试。

TDD先后记录：算法新增测试初次27项因缺接口失败；manager初次5 failed/10 passed；callbacks初次3 failed/1 passed。后续pruning误COMPLETE、末表summary缺列等先复现失败再修正。只读最大horizon元数据的独立RED摘录保存在 [max_horizon_red.txt](max_horizon_red.txt)。最终统一回归结果以上述实测166项为准。

## 真实有界命令、耗时与证据

下列是实际执行参数；`$d`是旧工作树只读的 `data/processed/operating_dataset_v6_1s`，`$b`是新工作树 `outputs/zoo_vessel_dev/episode_acceptance_20261030`。重跑必须换全新目录，不复用已有结果。

```powershell
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir "$b/episode3" --n-episodes 3 --exploration-schedule episode --seed 42
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir "$b/timestep4096" --n-timesteps 4096 --eval-freq 4096 --bounded --seed 42
& $py -u -m experiments.sb3_manual.zoo_vessel.launch optimize --data-root $d --output-dir "$b/episode_trial1" --n-episodes 1 --exploration-schedule episode --n-trials 1 --seed 42
& $py -u -m experiments.sb3_manual.zoo_vessel.launch evaluate --data-root $d --output-dir "$b/independent_eval" --model "$b/episode3/dqn_vessel/VesselEnergy-v0_1/VesselEnergy-v0.zip"
```

实际墙钟分别为390.495、138.541、387.103、129.159秒，全部exit0。部分独立进程并行，不能把它们相加当总任务耗时，也不将该数据当单进程性能基准。Train/timestep/独立eval stderr为空。Trial stderr有Zoo默认关闭TensorBoard提示、Optuna multivariate实验性提示及正常Study/COMPLETE日志，无训练错误；适配层的实际TensorBoard events已单独验证。见 [实际汇总与完整本机日志路径](run_results.json)、[Optuna stderr](optuna_stderr.txt)。

产物检查命令：

```powershell
& $py tests/verify_zoo_episode_outputs.py --train "$b/episode3" --optimization "$b/episode_trial1" --timestep "$b/timestep4096" --evaluation "$b/independent_eval" --output docs/audits/v6_zoo_episode_20261030/artifacts_verification.json
```

实际exit0，`passed=true`。脚本不训练、不评估、不读取数据payload，SQLite以只读模式打开。检查了模型/Replay、Adam/Target、history、动作mask、terminal、CSV/TensorBoard、真实Study COMPLETE状态、参数和目标值，并核对最终模型独立评估。见 [完整结构化产物证据](artifacts_verification.json)、[命令输出](artifacts_check.txt)。

## 隔离、交付与限制

1. `git diff 7df880d -- src data outputs`为空；新增改动仅Zoo adapter、README/YAML、测试、计划和轻量文本证据。旧工作树HEAD仍17f3da8，用户CartPole/Zoo staged/untracked文件保留。本任务不写旧源码、checkpoint或identity；旧训练自身产生的自然checkpoint更新继续发生。
2. Train/Val固定加载器及测试只允许35/8，`dataset_identity`仅hash Train/Val payload与split/manifest元数据；未读取Test payload。没有额外开启操作系统级I/O跟踪。
3. 模型保存/加载和独立Validation受支持；完整物理live-state、RNG、航段游标续训仍未实现，Episode模型已有训练步时再次`learn()`会明确拒绝。不能拿此checkpoint冒称可精确继续半条航段。
4. Study恢复不恢复TPE随机状态；中断RUNNING Trial不做伪恢复。每Study只运行一个launcher，默认CLI不主动剪枝，PRUNED机制另有官方objective集成测试。
5. Episode+显式timestep探索的数学基准是记录的保护上限；timestep+episode探索另有独立Episode目标。二者均不冒称与默认500Episode/episode调度等价。正常timestep行为保持原生语义。
6. 模型、Replay、events、SQLite及全量运行日志均留在本机ignored输出，未提交大文件。完整五类Windows命令在 [README](../../../experiments/sb3_manual/zoo_vessel/README.md)。提交推送后停止本次开发，不自动启动用户的正式Zoo训练或搜索。
