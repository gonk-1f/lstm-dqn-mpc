# V6 阶段 3：RL Zoo / Optuna 固定版本兼容审查

审查日期：2026-10-28。范围为官方发行依赖、DQN 调参源代码、独立环境安装及导入检查。本报告未运行任何训练、Optuna trial/search、Gym 环境实例或 Test 评估；项目 runner 的验证由主任务另行记录。

## 结论与固定组合

在保留正式 **SB3 2.9.0 / Gymnasium 1.2.3 / Python 3.11.5 / PyTorch 2.8.0+cu128** 的条件下，采用 **RL Zoo 2.9.1 + SB3-Contrib 2.9.0 + Optuna 4.9.0**。声明依赖满足、实际导入成功、`pip check` 成功；这些检查证明安装和接口可用，不证明项目调参或训练结果正确。

| 包 | 固定版本 | 官方发行约束及本次判断 |
| --- | --- | --- |
| stable-baselines3 | 2.9.0 | 官方 wheel 元数据要求 Python >=3.10、torch >=2.8；保留正式版本。 |
| gymnasium | 1.2.3 | 符合上述 SB3 及 Zoo 2.9.1 的声明范围，沿用现有包。 |
| torch | 2.8.0+cu128 | 沿用既有科学环境 CUDA 构建，未安装或升级；项目运行仍须显式 `device="cpu"`。 |
| sb3-contrib | 2.9.0 | `stable_baselines3>=2.9.0,<3.0`；与固定 SB3 相容。 |
| rl-zoo3 | 2.9.1 | Python >=3.10；`sb3_contrib>=2.9.0,<3.0`、`gymnasium>=0.29.1,<2.0`、`optuna>=3.0`。 |
| optuna | 4.9.0 | Python >=3.9；使用本报告固定版本的 Trial / study / RDB API。 |

依据：[SB3 2.9.0 PyPI 元数据](https://pypi.org/pypi/stable-baselines3/2.9.0/json)、[Zoo 2.9.1 PyPI 元数据](https://pypi.org/pypi/rl-zoo3/2.9.1/json)、[Contrib 2.9.0 PyPI 元数据](https://pypi.org/pypi/sb3-contrib/2.9.0/json)、[Optuna 4.9.0 PyPI 元数据](https://pypi.org/pypi/optuna/4.9.0/json)。相关 JSON 已保存在 `raw_compatibility/`。

截至审查日，Zoo 2.9.2 已于 2026-10-26 发布，但要求 `sb3_contrib>=2.9.2`，会迫使本次固定的 SB3 系列升级，因此不采用。发行包 README 中泛化的 master 安装说明不能代替固定 wheel 的依赖元数据；本次未安装 master、预发行版本或 Zoo extras。依据：[Zoo 2.9.2 PyPI 元数据](https://pypi.org/pypi/rl-zoo3/2.9.2/json)、[官方发行说明](https://github.com/DLR-RM/rl-baselines3-zoo/releases/tag/v2.9.2)。

## 官方源代码定位与文件校验

GitHub tag 通过官方 API 解析为以下 commit，而非按 master 推断：

| 项目 | tag | commit |
| --- | --- | --- |
| RL Zoo | v2.9.1 | `ecfecc9ed2460c46e8357413a471bcb2b7600087` |
| SB3-Contrib | v2.9.0 | `075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb` |
| Optuna | v4.9.0 | `4db42e31c24b200e52595df9d4c00e2cdeefea2b` |

官方定位：[Zoo commit](https://github.com/DLR-RM/rl-baselines3-zoo/commit/ecfecc9ed2460c46e8357413a471bcb2b7600087)、[Contrib commit](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/commit/075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb)、[Optuna commit](https://github.com/optuna/optuna/commit/4db42e31c24b200e52595df9d4c00e2cdeefea2b)。

`raw_compatibility/official_manifest.json` 记录官方 URL、commit、下载源文件 SHA-256 和已安装源文件比较结果；下载的 Zoo sampler、converter、callback、experiment manager 与所装 wheel 中对应代码一致。保留了 Zoo MIT LICENSE。`setup.py` 和文档不是安装后的运行模块，其比较值为 null；它们仍有独立源文件摘要。

pip 使用机器既有的清华 PyPI 镜像配置。为校验来源，又从官方 PyPI/files.pythonhosted.org 下载四个核心 wheel 并验证发布 SHA-256：Zoo 21、Contrib 43、Optuna 219、SB3 64 个已安装 Python 源文件与对应官方 wheel 全部一致。21 个新装传递依赖的 pip 归档摘要也逐一匹配官方 PyPI 对应发行的摘要；详见 `raw_compatibility/package_verification.json`。没有修改 pip 全局配置。

## 独立环境与实际路径

可用解释器：

```text
C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\.venv-optuna\Scripts\python.exe
```

新环境使用 `--system-site-packages`，从 `D:\py\Python3` **只读借用**既有 Torch、Gymnasium、NumPy、Pandas、SciPy、OSQP 和 Optuna 的部分传递依赖；四个核心运行包均在新环境中精确安装为本地包，其中 Optuna 本地版本遮蔽全局同版本。新 venv 不继承正式 `.venv` 自己的 site-packages，SB3 已在新环境单独安装。

| 包组 | 本次实际导入目录 |
| --- | --- |
| stable_baselines3 / sb3_contrib / rl_zoo3 / optuna | `<worktree>\.venv-optuna\Lib\site-packages\` |
| torch / gymnasium / numpy / pandas / scipy / osqp / sqlalchemy | `D:\py\Python3\Lib\site-packages\` |

`raw_compatibility/import_check.json` 给出了每个包的完整绝对 `__file__`、distribution 路径、实际版本、解释器以及 API 签名。检测到 `torch.cuda.is_available()==True`；这仅是硬件/安装探测，本次未使用 GPU 训练，也未把正式 CPU 执行要求改为自动设备选择。

这种方案隔离新安装和版本遮蔽，但**不是完全封闭的环境**：未来全局科学包发生变化可能影响新 venv。因此正式使用前应核对锁文件和实际路径；重建到其他机器需要另行提供匹配的科学包及 PyTorch CUDA wheel 来源，不能把本锁文件理解为跨平台一键重建包仓库。该限制是本次为避免重新下载 CUDA 包而采用的明确权衡。

复现本机隔离安装步骤（在 worktree 根目录执行）：

```powershell
& '.\.venv\Scripts\python.exe' -m venv --system-site-packages '.venv-optuna'
& '.\.venv-optuna\Scripts\python.exe' -m pip install --no-deps --ignore-installed stable-baselines3==2.9.0 optuna==4.9.0 sb3-contrib==2.9.0 rl-zoo3==2.9.1
& '.\.venv-optuna\Scripts\python.exe' -m pip install -r requirements-optuna.txt
& '.\.venv-optuna\Scripts\python.exe' -m pip check
```

`requirements-optuna.txt` 自动引用 `requirements-optuna-lock.txt` constraints 和现有 `requirements-sb3.txt`。锁文件固定本机 Python 3.11 / Windows 的 **63 包运行依赖闭包**，含 `torch==2.8.0+cu128`；它不是包含所有无关全局工具的 freeze。实际完整 `pip freeze --all`、本地包 `pip freeze --all --local` 和安装解析报告分别保存在 `raw_compatibility/`。

正式 `.venv` 与 `D:\py\Python3` 的安装前后 `pip freeze --all` 逐字一致，SHA-256 分别为：

```text
formal: 9319ff230325e7561bacf073bcb4811a825e43497161f812ae56a95a9bfa4f30
global: a0a46c15b0c436432fa8c8e3582017ac19a0b5850c964d533ea8bc0a82c02a09
```

三套解释器的 `pip check` 均为 exit 0 / `No broken requirements found.`；63 个锁定版本均匹配已安装版本。证据：`raw_compatibility/environment_verification.json` 及 `*_before_freeze.txt`、`*_after_freeze.txt`、`*_pip_check.txt`。本次所有安装命令均显式使用 `.venv-optuna\Scripts\python.exe`，未向正式或全局解释器执行安装、升级、卸载。

初次完整导入发生 `MemoryError`，调用链是 Zoo → Hugging Face → requests → chardet → `langthaimodel.py` 的文件加载。错误后资源采样显示可用物理内存 1,417,088 KiB、可用虚拟提交空间 1,822,764 KiB。随后单独 chardet 导入成功，再次单进程完整导入也成功，期间没有更换包版本。资源压力是合理解释，但错误后采样不能证明异常瞬间的确切根因；本报告不将该次异常归类为依赖版本冲突。准确记录见 `raw_compatibility/import_resource_note.json`。

## 固定 Zoo 2.9.1 的 DQN 搜索空间

审查对象为 [sample_dqn_params 与 convert_offpolicy_params](https://github.com/DLR-RM/rl-baselines3-zoo/blob/ecfecc9ed2460c46e8357413a471bcb2b7600087/rl_zoo3/hyperparams_opt.py#L352)，运行接口为：

```python
sample_dqn_params(trial, n_actions, n_envs, additional_args)
# additional_args 至少需要 using_her_replay_buffer；普通 DQN 设 False。
```

`n_actions` / `n_envs` 在普通 DQN sampler 内没有改变下表范围。HER、QR-DQN 分支不属于本次普通 DQN 基线。

| Optuna 记录的参数 | 固定源码范围 | 传给 SB3 的参数/转换 |
| --- | --- | --- |
| one_minus_gamma | float `[0.0001, 0.03]`，log | `gamma=1-one_minus_gamma`，即 `[0.97, 0.9999]` |
| batch_size_pow | int `[2, 11]` | `batch_size=2**batch_size_pow`，即 `4..2048` 的 2 次幂 |
| learning_rate | float `[1e-5, 0.002]`，log | 同名 |
| train_freq | int `[1, 10]` | 同名 |
| subsample_steps | int `[1, min(train_freq,8)]` | 不直接传给 SB3；`gradient_steps=max(train_freq//subsample_steps,1)` |
| exploration_final_eps | float `[0, 0.2]` | 同名 |
| exploration_fraction | float `[0, 0.5]` | 同名 |
| target_update_interval | int `[1, 20000]`，log | 同名 |
| net_arch | categorical `small, medium, big` | `policy_kwargs.net_arch` 为 `[64,64]`、`[256,256]`、`[400,300]` |

注意源码关于 batch size 的注释写了从 `2**5` 开始，但可执行语句是 `suggest_int(..., 2, 11)`；本表采用可执行代码，不沿用错误注释。`gamma` 和展开后的 `batch_size` 也保存为 trial user attributes。Converter 返回的是 SB3 kwargs，不能把 `trial.params` 中的变换前参数直接作为 DQN 构造参数。

`buffer_size`、`learning_starts`、`tau`、`max_grad_norm` 等没有在该 DQN sampler 中搜索；Zoo 用 YAML 加 SB3 默认值补齐未搜索参数。本项目若缩小空间、固定 `gradient_steps=1`、修改批量大小或使用不同网络结构，应明确标注为项目适配，而不能声称与 Zoo 原始搜索空间完全相同。官方 [dqn.yml](https://github.com/DLR-RM/rl-baselines3-zoo/blob/ecfecc9ed2460c46e8357413a471bcb2b7600087/hyperparams/dqn.yml) 的未列出环境 default 只有 `n_timesteps=1e6` 与 `policy=MlpPolicy`，不包含本项目 MPC 环境的专属参数。

## 目标、sampler、pruner 与资源释放

固定 [ExperimentManager.objective / hyperparameters_optimization](https://github.com/DLR-RM/rl-baselines3-zoo/blob/ecfecc9ed2460c46e8357413a471bcb2b7600087/rl_zoo3/exp_manager.py#L777) 的官方行为如下，不能与本项目物理目标混淆：

1. 将 sampler 结果覆盖基线参数，为 trial 创建训练和评估环境；算法构造显式 `seed=None`，没有固定该 trial 的训练随机种子。
2. 评估间隔为 `max(int(n_timesteps/n_evaluations)//n_envs,1)` 个 callback 调用。固定 [TrialEvalCallback](https://github.com/DLR-RM/rl-baselines3-zoo/blob/ecfecc9ed2460c46e8357413a471bcb2b7600087/rl_zoo3/callbacks.py#L16) 每次评估后报告 `last_mean_reward`，`step` 是从 1 开始的评估序号，而非环境 timestep。
3. 回调调用 `trial.should_prune()`，被剪枝时返回 False 停止 learn，之后 objective 抛 `TrialPruned`。最终目标为**最后一次评估平均 episode reward**，study 使用 `direction="maximize"`；不是历史最好 reward，也不是多训练 seed 平均指标。
4. `random` 使用带 seed 的 RandomSampler；`tpe` 使用 `TPESampler(n_startup_trials=..., seed=..., multivariate=True)`。可选 auto 需要额外 optunahub，本次未安装。
5. `median` 为 `MedianPruner(n_startup_trials=..., n_warmup_steps=n_evaluations//3)`；`halving` 为 `SuccessiveHalvingPruner(min_resource=1,reduction_factor=4,min_early_stopping_rate=0)`；`none` 为 NopPruner。Zoo 同版本调参文档要求 halving 配置 `n_jobs>1`。
6. 正常 learn 结束或 `AssertionError`/`ValueError` 时关闭训练/评估环境，后两类错误被当作 TrialPruned。该版本没有覆盖全部异常路径的 finally，也未传 `gc_after_trial=True`。不能把 Zoo 2.9.2 的清理修复反推到 2.9.1。

本项目实施建议：用 Validation 的既定物理指标作为目标，并显式使用相符的 minimize/maximize；保持正式物理环境、数据划分和配置固定；保证每个 trial 的训练/评估 seed 可追踪；把中间 report 的 step 定义与 pruner 的资源单位对齐；对异常使用 finally 关闭资源。若 objective 包含多 seed，不能将单个 seed 的值冒充聚合后的 trial 目标。以上为本项目方法建议，不是 Zoo 2.9.1 已提供的功能。

## 持久化与恢复的边界

Zoo 2.9.1 使用 `create_study(..., load_if_exists=True)`：`.log` 路径转换为 `JournalStorage(JournalFileBackend(...))`，其他 storage 字符串交给 Optuna。恢复同一 study 需要固定 study 名和 storage；默认 `storage=None` 是不持久化的内存存储。依据：[Zoo 固定 commit](https://github.com/DLR-RM/rl-baselines3-zoo/blob/ecfecc9ed2460c46e8357413a471bcb2b7600087/rl_zoo3/exp_manager.py#L912)、[Optuna 4.9.0 create_study](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.create_study.html)。

Optuna 保存 trial 参数、目标和元数据，但不会自动保存 DQN 模型、优化器、replay buffer、环境状态或随机数状态。因此恢复 study 不等于从训练中断处继续。重启时构造相同 seed 的新 sampler 也不保证完整恢复已消耗的 sampler 随机状态。依据：[Optuna 4.9.0 FAQ](https://optuna.readthedocs.io/en/v4.9.0/faq.html)。

本项目计划采用 **SQLite + n_jobs=1 + ask/tell**：应使用单个活动写入者，恢复前校验配置/数据/目标/版本签名。`load_if_exists` 只处理 study 名冲突，不替代这些校验。Optuna 不推荐 SQLite 并行优化；OneDrive 路径也不能被据此视为跨机器可并发的数据库服务。依据：[Optuna 4.9.0 RDBStorage](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.storages.RDBStorage.html)。

若恢复已有 RUNNING trial 的访问，固定版本的 `Trial(study, trial_id)` 中 `trial_id` 是存储 ID，不能误用展示用的 `trial.number`。恢复的对象可读取已有参数与中间值，但是否重启训练、读取何种 checkpoint、如何处理未完成产物，仍须由项目 runner 明确实现和验证。使用 storage 或 Study 的私有字段时应记录对 Optuna 4.9.0 的版本依赖。本审查只核对接口，不声称项目恢复路径已经通过实测。

`RDBStorage` 的 heartbeat 机制适用于 `Study.optimize()`，**不适用于 ask/tell**。Optuna 4.9.0 新名为 `heartbeat_stale_trial_callback` 和 `RetryHeartbeatStaleTrialCallback`；`failed_trial_callback` 已自 4.9.0 标记弃用。自动重试会创建重试 trial，不等于恢复原 DQN 的内存状态。本项目 ask/tell 路径不应依赖 heartbeat 自动恢复。依据：[RDBStorage 4.9.0](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.storages.RDBStorage.html)、[4.9.0 callback 源码](https://github.com/optuna/optuna/blob/4db42e31c24b200e52595df9d4c00e2cdeefea2b/optuna/storages/_callbacks.py)。

如未来改用 Journal，4.9.0 默认文件锁使用 symlink；Windows 如缺少 symlink 权限，应显式选 `JournalFileOpenLock`。本次未为了探测权限而创建额外 journal/study。依据：[JournalFileBackend 4.9.0](https://optuna.readthedocs.io/en/v4.9.0/reference/generated/optuna.storages.journal.JournalFileBackend.html)、[固定版本锁实现](https://github.com/optuna/optuna/blob/4db42e31c24b200e52595df9d4c00e2cdeefea2b/optuna/storages/journal/_file.py)。

## 交付与验证范围

交付 `requirements-optuna.txt`、`requirements-optuna-lock.txt`、本报告，以及 `raw_compatibility/` 的官方元数据/源文件摘要、安装解析、全量及本地 freeze、导入检查、依赖检查和正式环境前后对比。未改动 `src/v6_sb3` 或 tests；未提交 git commit。

本报告能确认固定依赖和所列接口可用。正式 runner 的 trial 目标、Validation 隔离、trial 失败分类、模型保存/恢复、剪枝和预算语义仍需要项目级检查；不应把安装/导入成功等同于这些行为已经验证。
