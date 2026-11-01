# DDPG 固定参数与原生 Zoo Optuna

## 固定配置

基准提交：`b45e6e472e8fb1538207b73e80e15fd6d986c52d`。Python 3.11.5、SB3 2.9.2、RL Zoo 2.9.0、Optuna 4.9.0、PyTorch 2.8.0+cu128；未安装或更改依赖。

| 参数 | 原值 | 本次固定值 |
| --- | --- | --- |
| learning_rate | 0.001 | 0.001 |
| gamma | 0.99 | 0.999 |
| batch_size | 256 | 64 |
| buffer_size | 1000000 | 1000000 |
| learning_starts | 100 | 5000 |
| train_freq | 1 | 32 |
| gradient_steps | 1 | 1 |
| tau | 0.005 | 0.005 |
| NormalActionNoise sigma | 0.1 | 0.1 |
| policy_kwargs | 未显式指定网络 | dict(net_arch=[256, 256]) |

原生 DDPG 实际构造结果（CPU，仅构造、不训练）：

- Actor 与 Actor target：3 → 256 → ReLU → 256 → ReLU → 1 → Tanh。
- Critic 与 Critic target：状态+动作共 4 维 → 256 → ReLU → 256 → ReLU → 1；单 Critic，Adam 优化器。
- SB3 DDPG 内部复用 `TD3Policy` 类并设置 `n_critics=1`，日志中的该类名不代表切换了算法。
- 实际 `train_freq=TrainFreq(32, STEP)`、`gradient_steps=1`；不会转换成 32:32。
- 单环境、1 秒周期、原有 device 选择逻辑保持不变。
- `train.py --total-timesteps` 默认仍为 **1000**，正式训练务必显式传预算。
- 正式 `zoo_ddpg.yml` 仍为 500000 步，并保留原有 `max_episodes=484` 回调与数据路径。它不代表保证完成 500 episode。

## 原生 Zoo 与本地适配边界

`zoo_optuna.py` 先替换 `rl_zoo3.hyperparams_opt.HYPERPARAMS_SAMPLER["ddpg"]`，再导入原生 `rl_zoo3.train.train`；不修改 site-packages。

原生调用链保持为：

```text
rl_zoo3.train.train()
  -> ExperimentManager.setup_experiment()
  -> ExperimentManager.hyperparameters_optimization()
  -> 原生 TPE / NopPruner / SQLite Study（maximize, load_if_exists=True）
  -> ExperimentManager.objective()
  -> 原生 SB3 DDPG.model.learn()
  -> 原生 TrialEvalCallback
```

适配器只搜索 tau（0.005～0.15，对数）和 noise_std（0.03～0.12，均匀）；返回字典只有 `tau` 与 `action_noise=NormalActionNoise(...)`。Optuna trial.params 只包含 tau、noise_std。未调用默认 `sample_td3_params` 或 `convert_offpolicy_params`，因此固定参数继续来自 YAML。

`zoo_trial_logging.TrialParametersCallback` 在训练开始时读取已构造模型，通过原生 Trial 的 user_attrs 保存实际参数，同时打印一次并写入原生评估回调的目录 `effective_parameters.json`。它不采样、不建 Study、不创建目标函数、不调用 learn、不修改模型、不干预 step 或更新次数。

## Train 预算与 Validation8

核对 35 个真实 Train gzip CSV 的行数，扣除表头及末尾边界点后，步数与 manifest.duration_s 全部一致：

- 最短 3272 步，中位数 9720 步，最长 36660 步。
- Trial 预算：`ceil(8 * 9720 / 32) * 32 = 77760` 步。
- 这是约 8 个典型航段的等效预算，不保证完成 8 个完整 episode。
- 独立 `zoo_ddpg_optuna.yml` 使用 77760 步；不包含 episode 提前停止回调。
- 原生两次评估频率为 `77760 / 2 = 38880`，单环境下在 38880、77760 步执行，均与 32 步收集周期对齐。
- Train 仍按原有随机抽样。Validation 原有 reset 已确定性顺序轮转，不需要修改 env.py。
- `--eval-env-kwargs` 必须指定 split=validation；8 次连续评估将遍历 8 条不同 Validation 航段，即便 VecEnv auto-reset 使下一轮从不同游标位置开始。
- 测试通过 reset 检查完整一轮及偏移后一轮均为 8 个不同航段；未执行整航段评估，也未读取 Test payload。

目标是原生 `TrialEvalCallback.last_mean_reward`，即**第二次 Validation8 的平均 episode reward，maximize**。当前奖励仍为 SOC 偏差罚项、FC 功率变化罚项的负值及失败惩罚；不是直接最小化氢耗或经济成本。奖励、SOC 权重、参考值、状态、动作、物理模型与数据划分均未修改。

## Windows PowerShell 搜索命令（交付命令，未执行搜索）

```powershell
Set-Location -LiteralPath 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
$env:PYTHONPATH = "$PWD\src;$PWD"

& '.\.venv\Scripts\python.exe' -u -m experiments.sb3_manual.ddpg_aggregate.zoo_optuna `
  --algo ddpg --env AggregateShipDDPG-v0 `
  --gym-packages experiments.sb3_manual.ddpg_aggregate.zoo_register `
  --conf-file experiments/sb3_manual/ddpg_aggregate/zoo_ddpg_optuna.yml `
  --optimize-hyperparameters --n-timesteps 77760 `
  --sampler tpe --pruner none --n-startup-trials 10 `
  --max-total-trials 20 --n-jobs 1 --seed 42 `
  --eval-episodes 8 --n-eval-envs 1 --n-evaluations 2 `
  --eval-env-kwargs "data_root:'data/processed/operating_dataset_v6_1s'" "split:'validation'" `
  --storage "sqlite:///outputs/ddpg_optuna/study.db" `
  --study-name ddpg_tau_noise_v1 `
  --optimization-log-path outputs/ddpg_optuna/evaluations `
  --log-folder outputs/ddpg_optuna/logs `
  --device auto --no-optim-plots
```

原样重跑使用同一 SQLite Study，Zoo 原生 `load_if_exists=True` 和 `max-total-trials=20` 控制正常重启的累计预算，不会直接新增另一批 20 个 Trial。不要改变固定参数后复用此 Study。

原生行为需要区分：

- `--seed 42` 设置 Zoo 全局随机数、环境和 TPE；Zoo 2.9.0 原生 objective 构造每个 trial 模型时传 `seed=None`。不额外改写此行为，也不声称每 Trial 都单独固定为 model.seed=42。
- Zoo 的最大 Trial 统计包含 COMPLETE、RUNNING、PRUNED，不包含 FAIL；并非对所有历史失败尝试的绝对条数上限。
- Zoo 在 HPO 中禁用 TensorBoard，评估记录使用每 Trial 的 evaluations.npz；本项目不另建训练/Study 日志系统。
- 原生 best_model.zip 保存该 Trial 两次评估中较高回报的模型；Study 目标仍取第二次评估回报，二者不能混为同一个快照。
- 普通 Zoo 的 `--study-name/--trial-id` 参数导入仍使用其默认 DDPG converter，要求更宽搜索空间中的字段；不要用它直接导入本项目仅两参数的 Study。参数依据下面 JSON 与保留的基线 YAML 读取。

## 搜索产物（由用户运行后生成）

- `outputs/ddpg_optuna/study.db`：原生 SQLite Study、tau/noise_std、结果、有效参数 user_attrs。
- `outputs/ddpg_optuna/evaluations/trial_N/evaluations.npz`：原生两轮评估。
- `outputs/ddpg_optuna/evaluations/trial_N/best_model.zip`：原生每 Trial 最佳模型。
- `outputs/ddpg_optuna/evaluations/trial_N/effective_parameters.json`：实际生效配置、网络、噪声、Train/Validation split。
- `outputs/ddpg_optuna/logs/ddpg/report_*.csv/.pkl`：原生搜索汇总；终端输出原生最优 Trial。
- 以上目录被 Git 忽略，模型和数据库不提交。

## 配置验收命令

在上述工作目录与 PYTHONPATH 下执行：

```powershell
& '.\.venv\Scripts\python.exe' -m compileall -q experiments/sb3_manual/ddpg_aggregate
& '.\.venv\Scripts\python.exe' -m experiments.sb3_manual.ddpg_aggregate.train --help
& '.\.venv\Scripts\python.exe' -m experiments.sb3_manual.ddpg_aggregate.zoo_optuna --help
& '.\.venv\Scripts\python.exe' -m rl_zoo3.train --help
& '.\.venv\Scripts\python.exe' -m unittest experiments.sb3_manual.ddpg_aggregate.test_zoo_optuna -v
git diff --check
```

实际配置单元检查：7 项全部通过（73.285 秒）。覆盖两份 YAML、原生 sampler 注册、两参数分布边界、32:1 更新配置、真实模型的四个网络、原生 TPE/NopPruner、独立 Validation8 轮转、实际参数 JSON/user_attrs。

compileall 与上列三个 `--help` 命令均 exit 0。对照基准的 env.py、evaluate.py、zoo_register.py、src/ 和 data/ 差异检查无输出（exit 0）。唯一提示是原生 TPE 对 multivariate 参数的 ExperimentalWarning，不影响本次配置验收。

验证只执行原生 ExperimentManager 的 setup、环境 reset、模型构造和回调初始化，环境步=0、Adam 更新=0、Replay=0。未执行 hyperparameters_optimization、未创建 Study、未启动 Trial 或正式训练；不能把配置验收说成已完成搜索。
