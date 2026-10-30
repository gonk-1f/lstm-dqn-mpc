# V6 船舶环境的官方 Zoo 独立路线

本目录基于 PR #4 / `17f3da8`，在独立工作树 `v6-zoo-vessel`、分支 `feat/v6-zoo-vessel` 中开发。未修改原工作树的 `src`、环境、CartPole/Zoo 产物或训练 checkpoint，也没有调用旧 `episode_training` / `episode_tuning` 训练器。

实际运行环境为 **RL Baselines3 Zoo 2.9.0、SB3/contrib 2.9.2、Gymnasium 1.2.3、PyTorch 2.8.0+cu128、Optuna 4.9.0、TensorBoard 2.21.0**。SB3/contrib 2.9.2 与用户新装 Zoo 所在 `.venv` 的版本一致；另建 `.venv-zoo`，未改动原来的 `.venv` 或 `.venv-optuna`。训练强制 CPU、单 Torch 线程。

## 官方与适配层的边界

| 功能 | 实际提供者 |
|---|---|
| 超参数读取/预处理、日志目录、Monitor/VecEnv、模型构造与保存 | 官方 Zoo `ExperimentManager` |
| 实际收集循环、Replay 插入、优化 dispatch、线性探索数学 | SB3 `model.learn()` / OffPolicyAlgorithm |
| 实际 HPO 调度、Optuna Study、Trial 调用与报告 | 官方 Zoo `hyperparameters_optimization()` / `objective()` |
| DQN 网络、masked Replay、合法后继 TD、原 Huber/Adam | 原 `v6_sb3.masked_dqn.MaskedDQN`，由新增类继承 |
| 固定 Train35 / Validation8、独立选段、Gym 注册与轻量账本信息 | `env.py`，每秒物理执行完全委托原 `DirectVesselEnv` |
| 掩码预测、精确 warmup 分块、正式 Episode 预算、两种探索进度、formal Target 时钟、TDMAE | `algorithm.py` |
| 每1000步观察、post-update Episode Validation、完整8Val完成优先经济评分、Trial回调桥 | `callbacks.py` |
| 作用域 `dqn_vessel` 算法/sampler/converter 注册与明确启动命令 | `manager.py` / `launch.py` |

作用域注册不替换标准 `dqn`，也不修改 site-packages；进程结束或异常时还原注册与 Trial 回调。不能仅注册 Gym 环境后改用标准 DQN，它仍然不认识动作掩码。此路线的启动器实际调用官方 manager 和原生 `model.learn()`，不是旧 V6 训练器的包装。

官方依据：[Zoo v2.9.0 ExperimentManager](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.9.0/rl_zoo3/exp_manager.py)、[SB3 v2.9.2 OffPolicyAlgorithm](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/off_policy_algorithm.py)。实现核对了本机同版本源码；Episode 扩展与实际测试见 [2026-10-30 验收报告](../../../docs/audits/v6_zoo_episode_20261030/verification.md)，首次接入证据保留在 [原验收报告](../../../docs/audits/v6_zoo_vessel_20261029/verification.md)。

## 不变的环境与经济语义

Gym ID 为 `VesselEnergy-v0`，5维状态、`Discrete(13)`。Train 每次 reset 从固定35条航段中 seeded 均匀有放回选择；Validation 单独加载固定8条，评估前重置顺序，逐 ID 检查恰好覆盖一次。两个 split 构造后不可改，不能通过 reset options 覆写。仅读取选定 split 的 payload；不加载 Test。

所有物理、FC退化、SOC、signed SOC 等效成本和动作可行集合来自原执行器。正常奖励严格 `1/(1+C_CNY)`；真实失败仅将最后真实步替换为 −1，经济账本不扣 −1；正常结束无额外奖励。非法动作明确报错，数据/数值故障不伪装成物理失败或普通剪枝。

`action_masks()` 同时服务 warmup、epsilon-greedy 和 greedy；Replay 保留真实后继 mask，TD 仅对合法后继取最大值，真实终止不 bootstrap。独立评估显式传入 **eval_env 自己的 masks**；算法也提供 `evaluation_mask_source(eval_env)` 供 SB3 `evaluate_policy` 使用。

## 预算、探索率与 Target

入口要求明确选择一个互斥预算：

| 预算 | 结束条件 | 默认探索调度 | 默认 Validation |
|---|---|---|---|
| `--n-episodes 500` 或 `--n-episodes` | 恰好500个完整正式 Train Episode 自然终止后 | `episode` | 每50个正式Episode及最终Episode，均评估全部8Val |
| `--n-timesteps N` | 原生 SB3 总 Train 步预算，含warmup | `timestep` | 按 `--eval-freq` 的环境步间隔 |

一个 Episode 是一条抽选 Train 航段从 reset 到自然完成或真实物理失败的运行。成功和失败都计一次；它不是全部35条 Train 的一遍，也不等于成功跑完整条航段。35条 Train 继续均匀有放回抽样，seed可复现。没有预算参数、同时指定两个预算或非正预算都会在读数据前报错。

Episode 模式的500次全部从 warmup 结束后的自然 reset 开始。warmup固定为50000个实际环境步，不计正式Episode；若50000落在某条航段中间，该航段完整继续，不制造done或reset，且整条不计正式Episode。它在warmup之后的transition仍完整入Replay、参与正常学习，并计入 `formal_env_steps` 和 `warmup_bridge_env_steps`。到第500个正式terminal，最后transition已存Replay及真实done/mask、应有Adam完成后才停止，不执行第501个Episode的动作。DummyVecEnv正常自动reset的预选段不等于执行下一段动作。

Episode 模式仍向原生 `learn(total_timesteps)` 传保护上限：`learning_starts + (N+1)*max_train_voyage_steps + 1`，只依赖自然航段最大长度，涵盖最多一个跨warmup航段。`--max-env-steps`可显式收紧保护上限；提前触发而未完成N个正式Episode会明确 FAIL，不报告 COMPLETE。上限不是平均长度换算的Episode预算。选择 Episode 预算时显式的 `--n-timesteps` 被拒绝。

timestep 模式保留原生预算行为：rollout可能使实际步数略超过请求值，并可能结束于航段中间，不制造终止；以保存的 `num_timesteps` 为准。它不等价于500个完整正式Episode。

默认 YAML 使用 warmup=50000、train_freq=32、gradient_steps=1。适配器将最后一个不足32步的 warmup 块精确切到50000，随后才按32个 **formal** transition 一次 Adam；不偷偷把 warmup 改成50016。Target 仅按已入 Replay 的 formal 环境步同步，不计 warmup/Validation，并在同一步的 Adam 之前复制；`target_sync_unit`、`formal_env_steps`、`target_sync_history` 随模型保存，CSV 中另记 `vessel/formal_env_steps`。

两种预算均保持正式32:1更新。Episode终止导致的不足32步采集块不触发额外Adam；后续采集补足原32步边界，正好满32步时保留该次更新。Target在每10000个已存储formal步同步，发生于同一步的Adam之前。epsilon、episode计数和Validation不会改变Target时钟。

`--exploration-schedule timestep|episode` 与预算是两个独立选择。默认按上表，显式选择不会被覆盖：

- `episode`：复用SB3线性调度数学，以已完成正式Episode作为进度，每段固定其开始时的epsilon。500目标时 `max(.05, 1-.95*completed/400)`；首段1.0，第100/200/300/400次结束后为.7625/.525/.2875/.05，后100次保持.05。warmup和跨界航段始终1.0。
- `timestep`：保留PR #5原始按实际总步数（含warmup）的SB3调度；参数仍为initial1.0、final.05、fraction.8，Episode内会随步下降。
- 若显式将 timestep预算与episode探索组合，`--exploration-episode-target`指定epsilon目标，默认500，并不增加Episode停止条件。
- 若显式将 Episode预算与timestep探索组合，探索进度采用记录的内部步数保护上限；它不具备“第400次降至.05”的性质，修改保护上限也会改变该显式组合的timestep探索进度。

正式默认超参数不变：网络[256,256]、batch64、Replay1000000、learning_starts50000、train_freq32、gradient_steps1、Target10000、gamma1.0、CPU。Optuna保留已有sampler及官方调度，每个Trial使用同一完整Episode预算和warmup；不把Episode换成固定步数。实际步数、Adam、Validation、参数和 COMPLETE/PRUNED/FAIL 写入各Trial汇总。Episode模式评估频率由 `--eval-every-episodes`（默认50）和最终N控制，官方 `--n-evaluations` 不覆盖此规则。

## 原生表格、CSV 和 TensorBoard

使用的始终是 **SB3 Logger**、`HumanOutputFormat`、`CSVOutputFormat` 和 `TensorBoardOutputFormat`，由 `model.dump_logs()` 渲染官方终端表格，没有另写替代打印器。关闭每若干 episode 自动 dump，由 callback 每1000环境步和最终收尾 dump；未完成长航段也有进度。

保留原生 `rollout/ep_len_mean`、`rollout/ep_rew_mean`、`rollout/exploration_rate`、`time/fps`、`time/total_timesteps`、`train/loss`、`train/n_updates`。在首个 episode 结束或首个 Adam 前，本来尚不存在的统计不伪造为0。新增 `train/td_mae`、`replay/buffer_size`、当前航段/SOC/FC功率，以及**本航段累计** FC启动次数与 CNY成本。

另记 `train/completed_formal_episodes`、`train/target_episodes`、`train/formal_env_steps`、`train/adam_updates`、warmup及bridge步数、预算/探索模式、正式Train成功/失败次数和比例。Train成功率与8Val完成率分别记录；未开始正式Episode时成功率留空。收尾原生表格和 `run_summary.json` 记录实际计数、Target同步次数、最终epsilon、最终Validation、最佳模型与最终模型路径。

原生 `time/episodes` 和 `rollout/ep_len_mean` 保留SB3口径，包含warmup期间终止的航段；正式预算只看 `train/completed_formal_episodes`。例如本次3Episode验收的原生总episode数为12，其中正式预算恰好3次，不能把两个计数混用。

原生 callback 在当前 transition 入 Replay **之前**调用，所以1000步表格可显示 buffer=999；额外的 `replay/pending_transition=1` 明确说明此时序，收尾表格再显示最终一致计数。日志/Validation不改变动作、奖励、Replay采样、Adam、Target或探索率；测试对照了有/无这些回调时完整权重、Replay、RNG和计数逐值一致。

Validation指标始终来自独立8Val，不是训练完成率。优化先比较真实完成数量，未全完成时再比较覆盖率；全部8条完成后才比较完整 CNY。未完成整组时经济成本为 `None`：CSV空值、TensorBoard不写虚构成本scalar，同时记录 `validation/economic_cost_available=0`。`last_mean_reward` 只为匹配官方 TrialEvalCallback 的接口名，内部是明确标注的完成优先 score，未伪装成 mean reward 日志。

Episode模式评估由post-storage/post-update hook触发，最终第500次评估包含最后所有应有Adam。每次保存对应正式Episode、总步、formal步、Replay及Adam，验证期间不推进这些计数；Trial剪枝也发生在已提交的边界。timestep模式保留原来callback的pre-storage评估时序，两者不混称。

官方 HPO 默认关闭 TensorBoard；适配回调在每个 Trial 的 `native_logs` 目录重新配置原生三种 writer，因此那条官方提示不代表 Trial 没有生成 TensorBoard。

## Windows / PyCharm PowerShell 命令

在 PyCharm 使用新工作树和 `.venv-zoo` 解释器。准备命令：

```powershell
$z='C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-zoo-vessel'
$d='C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\data\processed\operating_dataset_v6_1s'
Set-Location -LiteralPath $z
$py=Join-Path $z '.venv-zoo\Scripts\python.exe'
$env:PYTHONPATH="$z\src;$z"
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
# 本机已建好环境；在新的克隆中才需要创建/安装，不能对旧运行环境执行安装。
# D:\py\Python3\python.exe -m venv --system-site-packages .venv-zoo
# & $py -m pip install -r experiments/sb3_manual/zoo_vessel/requirements.txt
& $py -m pip check
$tag=Get-Date -Format 'yyyyMMdd_HHmmss'
```

以下五条路线均由用户自行执行，输出目录必须新建且为空。正式500Episode训练，默认完整模型、50000步warmup和每50Episode Validation：

```powershell
$train=Join-Path $z "outputs\zoo_vessel_dev\user_episode500_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir $train --n-episodes 500 --exploration-schedule episode --eval-every-episodes 50 --seed 42
```

真实 Train 的3个正式Episode短预算验证（保留正式超参数与完整warmup，也评估全部8Val）：

```powershell
$short=Join-Path $z "outputs\zoo_vessel_dev\user_episode3_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir $short --n-episodes 3 --exploration-schedule episode --seed 42
```

Zoo 500Episode多Trial搜索，**长时间任务，仅用户决定启动**：

```powershell
$study=Join-Path $z "outputs\zoo_vessel_dev\user_episode_study_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch optimize --data-root $d --output-dir $study --n-episodes 500 --exploration-schedule episode --eval-every-episodes 50 --n-trials 20 --study-name vessel_episode500 --seed 42
```

保留的原生timestep训练命令。此例 `--bounded` 显式使用warmup64、网络[32,32]、batch32、Replay8192、Target128，仅用于小规模接口验收，不代表正式超参数；Validation仍完整8条：

```powershell
$train=Join-Path $z "outputs\zoo_vessel_dev\user_train_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir $train --n-timesteps 4096 --eval-freq 4096 --bounded
```

保留的timestep Optuna短搜索命令：

```powershell
$study=Join-Path $z "outputs\zoo_vessel_dev\user_study_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch optimize --data-root $d --output-dir $study --n-timesteps 1024 --n-trials 1 --n-evaluations 1 --bounded
```

独立加载最终模型、完整评估8Val：

```powershell
$model=Join-Path $train 'dqn_vessel\VesselEnergy-v0_1\VesselEnergy-v0.zip'
$evaluation=Join-Path $z "outputs\zoo_vessel_dev\user_eval_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch evaluate --data-root $d --model $model --output-dir $evaluation
```

查看 CSV 和启动 TensorBoard（由用户自行执行）：

```powershell
Get-ChildItem -LiteralPath $train -Recurse -Filter progress.csv
& $py -m tensorboard.main --logdir (Join-Path $train 'tensorboard') --port 6006
```

`--bounded` 在Episode模式也可用于更小的接口实验（最多3个正式Episode、单Trial），会明确缩小warmup/模型；它不是正式500Episode配置。正式实验移除该flag；上述3Episode命令已保留正式配置。普通参数来自 `dqn_vessel.yml`，不支持替换成无掩码DQN。

Study可用相同命令加 `--resume-study` 接续已完成Trial的研究，但不支持精确恢复中断的RUNNING Trial。源码/预算/调度/数据/奖励身份不同的Study或输出会被拒绝，不复用旧结果。保存模型可独立评估；此路线未提供完整物理live-state精确续训。前台Ctrl+C会中断并报未完成，不能承诺原V6 runner式checkpoint恢复；请保留输出并使用新的实验目录。

## 验收与明确限制

此前接入验收的4096步/单Trial证据保留。新增Episode验收实际命令、结束数量、warmup桥接、Replay/Adam/Target、8Val、原生表格/CSV/TensorBoard及独立加载结果见本次验收报告。辅助模拟验证500个事件的epsilon与每50次评估，不实际启动正式500Episode。

1. Gym注册、PassiveEnvChecker及合法动作的基本接口通过。通用主动 `check_env` 会随机采无掩码动作，不适用于本合同；未为了通过该checker而放宽物理约束。
2. 原生 Zoo 保存模型/Replay，不保存完整物理live-state/RNG/选段游标以提供旧runner式精确续训。独立评估受支持；不能承诺中航段精确恢复或旧checkpoint混用。
3. timestep Validation仍在当前步存储/下一Adam前；Episode Validation在terminal存储/应有Adam后。最佳Validation模型和最终训练模型分别保存、分别标记，不以其中一个替代另一个。
4. Study身份严格比较源码、适配、经济口径、数据和预算；`--resume-study` 不恢复TPE随机状态，也不是跨进程锁。每个Study一次只运行一个launcher，`n_jobs=1`。
5. 500Episode指500次完整正式尝试，包含真实失败；不承诺500条航段全部成功。没有启动正式500Episode、长搜索或Test。最终实验由用户自行运行。
