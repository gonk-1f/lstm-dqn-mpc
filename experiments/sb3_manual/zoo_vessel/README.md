# V6 船舶环境的官方 Zoo 独立路线

本目录基于 PR #4 / `17f3da8`，在独立工作树 `v6-zoo-vessel`、分支 `feat/v6-zoo-vessel` 中开发。未修改原工作树的 `src`、环境、CartPole/Zoo 产物或训练 checkpoint，也没有调用旧 `episode_training` / `episode_tuning` 训练器。

实际运行环境为 **RL Baselines3 Zoo 2.9.0、SB3/contrib 2.9.2、Gymnasium 1.2.3、PyTorch 2.8.0+cu128、Optuna 4.9.0、TensorBoard 2.21.0**。SB3/contrib 2.9.2 与用户新装 Zoo 所在 `.venv` 的版本一致；另建 `.venv-zoo`，未改动原来的 `.venv` 或 `.venv-optuna`。训练强制 CPU、单 Torch 线程。

## 官方与适配层的边界

| 功能 | 实际提供者 |
|---|---|
| 超参数读取/预处理、日志目录、Monitor/VecEnv、模型构造与保存 | 官方 Zoo `ExperimentManager` |
| 实际收集循环、Replay 插入、优化 dispatch、native epsilon schedule | SB3 `model.learn()` / OffPolicyAlgorithm |
| 实际 HPO 调度、Optuna Study、Trial 调用与报告 | 官方 Zoo `hyperparameters_optimization()` / `objective()` |
| DQN 网络、masked Replay、合法后继 TD、原 Huber/Adam | 原 `v6_sb3.masked_dqn.MaskedDQN`，由新增类继承 |
| 固定 Train35 / Validation8、独立选段、Gym 注册与轻量账本信息 | `env.py`，每秒物理执行完全委托原 `DirectVesselEnv` |
| 原生 learn 的掩码预测、精确 warmup 分块、formal Target 时钟、TDMAE | `algorithm.py` |
| 每1000步观察、完整8Val完成优先经济评分、Trial回调桥 | `callbacks.py` |
| 作用域 `dqn_vessel` 算法/sampler/converter 注册与明确启动命令 | `manager.py` / `launch.py` |

作用域注册不替换标准 `dqn`，也不修改 site-packages；进程结束或异常时还原注册与 Trial 回调。不能仅注册 Gym 环境后改用标准 DQN，它仍然不认识动作掩码。此路线的启动器实际调用官方 manager 和原生 `model.learn()`，不是旧 V6 训练器的包装。

官方依据：[Zoo v2.9.0 ExperimentManager](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.9.0/rl_zoo3/exp_manager.py)、[SB3 v2.9.2 OffPolicyAlgorithm](https://github.com/DLR-RM/stable-baselines3/blob/v2.9.2/stable_baselines3/common/off_policy_algorithm.py)。实现核对了本机同版本源码；测试证据见 [独立验收报告](../../../../docs/audits/v6_zoo_vessel_20261029/verification.md)。

## 不变的环境与经济语义

Gym ID 为 `VesselEnergy-v0`，5维状态、`Discrete(13)`。Train 每次 reset 从固定35条航段中 seeded 均匀有放回选择；Validation 单独加载固定8条，评估前重置顺序，逐 ID 检查恰好覆盖一次。两个 split 构造后不可改，不能通过 reset options 覆写。仅读取选定 split 的 payload；不加载 Test。

所有物理、FC退化、SOC、signed SOC 等效成本和动作可行集合来自原执行器。正常奖励严格 `1/(1+C_CNY)`；真实失败仅将最后真实步替换为 −1，经济账本不扣 −1；正常结束无额外奖励。非法动作明确报错，数据/数值故障不伪装成物理失败或普通剪枝。

`action_masks()` 同时服务 warmup、epsilon-greedy 和 greedy；Replay 保留真实后继 mask，TD 仅对合法后继取最大值，真实终止不 bootstrap。独立评估显式传入 **eval_env 自己的 masks**；算法也提供 `evaluation_mask_source(eval_env)` 供 SB3 `evaluate_policy` 使用。

## 预算、探索率与 Target

这是**原生 timestep 路线**，不是500完整episode路线。`--n-timesteps` 包含 warmup，达到预算时可能仍处于某条真实航段；没有人为 TimeLimit、伪终止、补奖励或近似500 episode换算。原生 rollout 可能使实际步数略超过请求值，最终以保存的 `num_timesteps` 为准。CLI 必须显式指定步数，不接受 `--episodes`。

默认 YAML 使用 warmup=50000、train_freq=32、gradient_steps=1。适配器将最后一个不足32步的 warmup 块精确切到50000，随后才按32个 **formal** transition 一次 Adam；不偷偷把 warmup 改成50016。Target 仅按已入 Replay 的 formal 环境步同步，不计 warmup/Validation，并在同一步的 Adam 之前复制；`target_sync_unit`、`formal_env_steps`、`target_sync_history` 随模型保存，CSV 中另记 `vessel/formal_env_steps`。

原 `MaskedDQN._on_step()` 不更新 epsilon，因为原 runner 按 episode 设置它；新适配恢复 SB3 按总 timesteps 的 exploration schedule，参数来自 YAML 的 `exploration_fraction` / `exploration_final_eps`。因此这条路线的探索率不能冒称与500 episode旧路线相同。调参固定 gamma=1、32:1更新比例，仅采样学习率、batch、buffer、网络宽度及 formal Target interval。

## 原生表格、CSV 和 TensorBoard

使用的始终是 **SB3 Logger**、`HumanOutputFormat`、`CSVOutputFormat` 和 `TensorBoardOutputFormat`，由 `model.dump_logs()` 渲染官方终端表格，没有另写替代打印器。关闭每若干 episode 自动 dump，由 callback 每1000环境步和最终收尾 dump；未完成长航段也有进度。

保留原生 `rollout/ep_len_mean`、`rollout/ep_rew_mean`、`rollout/exploration_rate`、`time/fps`、`time/total_timesteps`、`train/loss`、`train/n_updates`。在首个 episode 结束或首个 Adam 前，本来尚不存在的统计不伪造为0。新增 `train/td_mae`、`replay/buffer_size`、当前航段/SOC/FC功率，以及**本航段累计** FC启动次数与 CNY成本。

原生 callback 在当前 transition 入 Replay **之前**调用，所以1000步表格可显示 buffer=999；额外的 `replay/pending_transition=1` 明确说明此时序，收尾表格再显示最终一致计数。日志/Validation不改变动作、奖励、Replay采样、Adam、Target或探索率；测试对照了有/无这些回调时完整权重、Replay、RNG和计数逐值一致。

Validation指标始终来自独立8Val，不是训练完成率。优化先比较真实完成数量，未全完成时再比较覆盖率；全部8条完成后才比较完整 CNY。未完成整组时经济成本为 `None`：CSV空值、TensorBoard不写虚构成本scalar，同时记录 `validation/economic_cost_available=0`。`last_mean_reward` 只为匹配官方 TrialEvalCallback 的接口名，内部是明确标注的完成优先 score，未伪装成 mean reward 日志。

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

已验证的小预算训练命令，使用真实 Train，`--bounded` 仅缩小模型/训练预算，不缩短或替换 Validation：

```powershell
$train=Join-Path $z "outputs\zoo_vessel_dev\user_train_$tag"
& $py -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir $train --n-timesteps 4096 --eval-freq 4096 --bounded
```

Zoo Optuna 短搜索命令：

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

扩大实验由用户自行决定：移除 `--bounded`，显式设置 timesteps/评估间隔/Trial数量，并使用全新输出目录。普通模型参数来自 `dqn_vessel.yml`；此 CLI 不支持任意替换算法或无掩码DQN。Study可用相同命令加 `--resume-study` 接续已完成Trial的研究，但不支持精确恢复中断的RUNNING Trial。

## 验收与明确限制

真实有界验收已运行4096步训练、一个1024步Trial、独立加载评估；每次都实际尝试全部8Val。短训完成率为0，不代表策略质量通过；完整经济成本保持 `None`，没有写成零。详见验收报告与真实原生表格。

1. Gym注册、PassiveEnvChecker及合法动作的基本接口通过。通用主动 `check_env` 会随机采无掩码动作，不适用于本合同；未为了通过该checker而放宽物理约束。
2. 原生 Zoo 保存模型/Replay，不保存完整物理live-state/RNG/选段游标以提供旧runner式精确续训。独立评估受支持；不能承诺中航段精确恢复或旧checkpoint混用。
3. Validation在原生 callback 的当前步存储/下一Adam之前评估。4,096步的 inline best 对应125次Adam，最终模型为126次；报告单独评估最终模型，不能混称二者结果。
4. Study身份严格比较源码、适配、经济口径、数据和预算；`--resume-study` 不恢复TPE随机状态，也不是跨进程锁。每个Study一次只运行一个launcher，`n_jobs=1`。
5. 未实现500完整episode目标；本路线明确定义为原生总步预算。未自动启动正式训练、长搜索或Test。最终实验由用户自行运行。
