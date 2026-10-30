# Zoo 船舶独立路线验收

基础提交 `17f3da8`，独立工作树 `v6-zoo-vessel`，分支 `feat/v6-zoo-vessel`。开发完成后停止，未自动运行正式500 episode或长搜索。模型、Replay、TensorBoard二进制及SQLite仅留本机 ignored outputs，Git仅保存代码、小型文本证据和JSON/CSV。

## 实际环境与隔离

核查用户新装 `.venv` 为 Zoo2.9.0 / SB3与contrib2.9.2；此前正式训练 `.venv-optuna` 为 Zoo2.9.1 / SB32.9.0。此次在新工作树另建 `.venv-zoo`，固定用户Zoo组合并补TensorBoard2.21.0，`pip check` 输出 `No broken requirements found.`。完整版本见 [verification.json](verification.json)。

底层 `src` 和数据与17f3da8无diff，原工作树仍在原分支/HEAD，用户 staged/untracked CartPole/benchmark 文件原样保留。原正式训练此前已在100000实际Train步预算处自然暂停：11个已结束episode、50000 warmup、50000 formal、1562 Adam；该训练目录最后修改时间为2026-10-29 16:49:20北京时间，早于本任务。此次没有恢复它或写其checkpoint。证据见 [isolation.json](isolation.json)。

## 单元与保留回归

最终实际命令：

```powershell
.venv-zoo\Scripts\python.exe -m pytest tests/test_zoo_vessel_env.py tests/test_zoo_algorithm.py tests/test_zoo_callbacks.py tests/test_zoo_manager.py tests/test_v6_direct_environment.py tests/test_one_plus_reward.py tests/test_sb3_masked_dqn.py tests/test_v6_target_env_steps.py -q --tb=line
```

结果 **104 passed in 55.69s，exit0**，见 [final_unit.txt](final_unit.txt)。覆盖环境固定35/8和选段种子、非法动作、正常/失败奖励账本、native learn、Replay后继mask、terminal零bootstrap、epsilon调度、非整32的warmup、formal Target时钟、独立eval mask与RNG、官方manager/Optuna作用域和输出身份。

额外对照有/无日志与完整合成Validation回调的相同seed训练：最终Replay动作/奖励/mask、完整网络/Target权重、NumPy RNG、Adam计数和探索率逐值一致。3项日志/Validation测试通过，见 [logging_invariance.txt](logging_invariance.txt)。这些是合成单元测试；下节才是真实船舶数据测试。开发中的缺接口/observer时序RED也保留，未把它们当作最终通过结果。

## 实际官方 Zoo 有界训练

三条CLI均为真实入口，均 exit0；工作目录为新工作树。共享数据根只读指向原工作树的冻结 `data/processed/operating_dataset_v6_1s`。

```powershell
$d='C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\data\processed\operating_dataset_v6_1s'
.venv-zoo\Scripts\python.exe -u -m experiments.sb3_manual.zoo_vessel.launch train --data-root $d --output-dir outputs/zoo_vessel_dev/real_train_4096 --n-timesteps 4096 --eval-freq 4096 --bounded
.venv-zoo\Scripts\python.exe -u -m experiments.sb3_manual.zoo_vessel.launch evaluate --data-root $d --output-dir outputs/zoo_vessel_dev/independent_eval --model outputs/zoo_vessel_dev/real_train_4096/dqn_vessel/VesselEnergy-v0_1/VesselEnergy-v0.zip
.venv-zoo\Scripts\python.exe -u -m experiments.sb3_manual.zoo_vessel.launch optimize --data-root $d --output-dir outputs/zoo_vessel_dev/real_optuna_one_trial --n-timesteps 1024 --bounded --n-trials 1 --n-evaluations 1
```

已有产物目录不能复用；直接复现请使用手册中的新时间戳目录。[完整运行命令与官方/适配边界](../../../experiments/sb3_manual/zoo_vessel/README.md)。

| 运行 | 实际Train总步（含warmup） | warmup | formal | Adam | Replay |
|---|---:|---:|---:|---:|---:|
| 官方manager训练 | 4096 | 64 | 4032 | 126 | 4096 |
| 官方Optuna Trial0 | 1024 | 64 | 960 | 30 | 1024 |

`--bounded`只把模型改为32×32、buffer8192、batch32、warmup64、Target128 formal步，仍为真实35Train，仍完整尝试8条真实Validation，没有伪造/截断航段，也没有将训练当Validation。原生 `model.learn` 的真实回放/优化/掩码路径全部执行，非旧episode训练器。

训练日志：[real_train_4096_console.txt](real_train_4096_console.txt)。保存/加载后的Final模型独立评估：[independent_validation.json](independent_validation.json)。官方Optuna实际产生 **1个COMPLETE Trial**，value=0.07792733312109677，等于其完整8Val的完成优先score；非平均Reward单指标。原始报告见 [real_optuna_console.txt](real_optuna_console.txt)。

## 实际原生表格与文件

以下摘自本次真实4096步输出；完整原生表格见 [native_table_final.txt](native_table_final.txt)，没有用自定义print模拟：

```text
| replay/                    |                  |
|    buffer_size             | 4096             |
|    pending_transition      | 0                |
| rollout/                   |                  |
|    ep_len_mean             | 2.77e+03         |
|    ep_rew_mean             | 746              |
|    exploration_rate        | 0.05             |
| time/                      |                  |
|    fps                     | 57               |
|    total_timesteps         | 4096             |
| train/                     |                  |
|    loss                    | 0.0802           |
|    n_updates               | 126              |
|    td_mae                  | 0.343            |
| validation/                |                  |
|    completion_rate         | 0                |
|    economic_cost_cny       | None             |
|    evaluated_adam_updates  | 125              |
| vessel/                    |                  |
|    fc_power_kw             | 580              |
|    formal_env_steps        | 4032             |
|    sample_id               | parent_046_seg02 |
|    soc                     | 0.796            |
```

每1000步分别输出1000/2000/3000/4000，最后4096步收尾；CSV无重复/逆序时间点。1000步处Replay999、`pending_transition=1` 是原生callback先于存储的真实时序，最终4096条一致。并非少插入或日志伪报。

训练与Trial均实际生成TensorBoard events，验证了loss、TDMAE、n_updates、SOC、Validation完成率/覆盖率的scalar；完整经济成本不可比较时没有成本scalar伪零。原生小型CSV副本：[native_train_progress.csv](native_train_progress.csv)、[native_trial_progress.csv](native_trial_progress.csv)。真实event文件保留本机outputs，不提交二进制。

## 保存、加载、计数与经济验证

实际执行：

```powershell
.venv-zoo\Scripts\python.exe tests/verify_zoo_real_outputs.py
```

结果 `passed=true`，见 [artifacts_final_check.txt](artifacts_final_check.txt)、[artifacts_verification.json](artifacts_verification.json)。验证读取真实model/Replay/CSV/TensorBoard/SQLite，未执行学习或读取数据payload：

- 保存模型/权重/奖励有限，CPU加载，Replay未覆盖，4096条与总步一致。
- 4094个非reset动作使用上一Replay的后继mask独立验证合法；其余reset动作和所有真实动作均通过环境运行时严格合法性门禁，训练正常退出。
- 实际出现1个物理失败terminal，reward=-1、后继mask空、无旧金额惩罚；巨大后继Q下该真实末步TD仍无bootstrap。
- Target每128个formal存储步复制，history的Adam位置与32:1严格一致；warmup不计Target。
- 普通训练104项测试还验证非整32warmup不改变原V6 formal时钟；模型/Replay/优化计数未靠表格推断。
- 独立加载最终模型完整评估8个Val ID，评估自身mask、无训练Replay/Adam写入、无Test。

## 结果解读与限制

本次短训只证明接口链可运行，不证明策略已学好：inline best八Val完成0/8，coverage约0.1315；最终126次Adam模型独立评估0/8，coverage=0.11645380917683287；Trial同样0/8，coverage=0.15585466624219355。整组完整经济成本都为null；失败前缀CNY单列，不能充作完整费用。

inline Validation发生于原生callback的当前step入库/下一Adam之前：4096步时其权重对应125次Adam、Replay4095；最终保存模型为126次Adam、Replay4096，因此两次评估不同是预期边界，不应混称相同模型。字段已显式记录。

Gym注册和合法动作的基本接口/PassiveEnvChecker通过；通用主动无掩码随机checker不适用。原生timesteps不等于500完整episodes；budget可能在航段中间结束但不制造terminal。新路线未实现旧runner式物理live-state精确恢复；Study恢复不恢复TPE RNG，也没有跨进程单writer锁，限定每Study一次一个launcher。所有限制和可执行用户命令见手册。

本任务没有启动正式500 episode、长Optuna搜索或Test，没有修改燃料电池退化、物理/动作掩码、经济奖励或数据划分。代码与证据整理后停止，最终实验由用户自行启动。
