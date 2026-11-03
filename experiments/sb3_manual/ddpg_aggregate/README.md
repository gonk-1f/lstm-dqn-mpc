# DDPG 固定航段500 Episode实验

当前路线使用 SB3 原生 DDPG、RL Zoo 原生训练/Optuna。FC 为 600 kW 聚合系统，
电池为 624 kWh 聚合系统；不分堆、不分簇。物理模型、动作/状态和冻结数据划分不变。

## 奖励与成本

版本：`ddpg_economic_tanh_v1`。

- `C_H2`：现有实际氢耗乘现有氢价 21.9 CNY/kg。
- `C_FC_deg`、`C_LB_deg`：直接使用现有物理 interval 返回的退化账本，不重复推进物理或另加寿命惩罚。
- `C_bat_eq = (SOC_before-SOC_next)*E_bat*p_H2/(eta_eq*LHV_H2)`。
  `E_bat=624 kWh`、`LHV_H2=120/3.6 kWh/kg` 均取现有模型。
  放电为正成本，充电为负抵扣；SOC 差已包含现有充放电效率，不重复计 DC/DC 或电池效率。
- FC 开启时 `eta_eq=eta_FC(P_FC_actual)`。FC 关闭时，使用现有系统 PCHIP 曲线
  正功率范围的最大效率；程序从曲线计算为 `0.6385540000000001`，不手填替代效率。
  每步导出 `fc_efficiency_actual` 和 `equivalent_fc_efficiency`。

```text
economic_cost = C_H2 + C_FC_deg + C_LB_deg + C_bat_eq
C_SOC = lambda_soc * (SOC_next - 0.5)**2
r = economic_cost + C_SOC
reward = tanh(1/r)       # 正常执行，r=0 时取右极限 reward=1
reward = -1             # 原有真实物理不可行终止
```

不对 SOC 平方偏差做区间归一化或再乘时间步长。独立 FC ramp 惩罚已经删除；
Ramp RMS 继续作为诊断指标，原 CSV 的 ramp penalty 字段固定为0。
负值或非有限总成本明确报错，不取绝对值、不裁剪、不加 epsilon。
失败奖励不记入经济成本；未执行失败步不制造电池功率、氢耗或时间。

CSV/JSON 的 `total_cost_cny` 是四项经济成本；`c_soc_cny` 为 SOC 惩罚，
`reward_total_cost_cny` 才是奖励公式中的 r。不再额外做终端 SOC 补能计价，
避免重复计算电池能量。FC 退化仍是现有聚合代理，未声称逐堆实测寿命。

## 固定 Train35 标定

正式基线模型：
`outputs/ddpg_train_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip`，
SHA256 `caf67c28eea863c01c8d6f9a8f7e38ea27b2f3ce924fe1d12dc4a843551f100f`。
固定遍历 Train35 各一次，deterministic=True，未读取 Validation/Test payload。

- 计划457562秒，实际448434个物理步；33航段完成。
- parent_015_seg01：4564/10115步后失败；parent_036_seg01：3084/6661步后失败。
  标定保留35条航段的全部实际执行步，不为失败调用或未执行尾段补造成本。
- 对全部实际步的四项经济成本取中位数（不包含SOC惩罚，不是35个航段均值的中位数）：
  **C_typ = 0.17429534153424198 CNY**。
- **lambda_soc = 8.0**。本次按用户要求由17改为8；原始比例估算仍为
  `100*C_typ = 17.4295`，`C_typ` 与原标定记录不变，8不是重新标定的结果。
- 标定最小四项经济成本0.04301742089671926 CNY；没有负总成本。
- 3个CPU推理worker用时118.80秒，模型、优化器、计数及Replay保持不变。

固定值及基线/数据/原始结果SHA在 `economic_reward_config.json`。
训练只读取该固定文件，不在线重算权重。日志记录 C_typ、lambda 和实际使用的等效效率。
原始收集与命令：
`outputs/ddpg_reward_calibration/collection_20261103_133754_684927/`。
最终计算另存该目录的 `calibration_result.json` 和 `calibration_result.log`。
`calibrate_reward.py` 仅负责离线收集，标定原始NPZ/日志不提交Git。

## 直接运行

在实际 DDPG 工作树用项目 `.venv` 解释器，打开根目录脚本，直接点击PyCharm绿色运行按钮：

| 用途 | 根目录脚本 |
|---|---|
| 正式500 Episode训练 | run_ddpg_train.py |
| 固定训练航段确定性诊断 | run_ddpg_fixed_eval.py |
| 独立Validation8 | run_ddpg_validate.py |
| 授权后的独立Test4 | run_ddpg_test.py |
| 原生Zoo Optuna搜索 | run_ddpg_optuna.py |

本次开发不启动正式训练。固定航段诊断、Validation/Test默认指向本次训练输出的首个模型；
该模型要在用户后续训练完成后才存在。
也可通过原模块的 `--model-path` 明确选择模型。
旧策略可以在新奖励下单独重新评估，输出明确标记 legacy；
旧奖励checkpoint不能恢复到新奖励训练，不混用旧奖励的评估结果。

原生训练参数：
gamma=.999、learning_rate=.001、batch_size=64、learning_starts=5000、
train_freq=32、gradient_steps=1、tau=.045、normal noise sigma=.1、
network=[256,256]、buffer_size=1000000、device=cuda、n_envs=1、seed=42。
Zoo仍以20000000环境步作安全上限，原生 StopTrainingOnMaxEpisodes=500。
每轮固定从 Train35 中的 **parent_005_seg01** 第0秒开始，SOC=0.6、FC=0 kW。
实测该航段长9258秒，500条完整航段最多4629000训练环境步，安全上限不会提前截断。
真实物理失败终止也计为一个已结束Episode，不能把500次结束等同于500次成功完成。
构造环境时 `fixed_train_id=None` 仍保留随机Train35机制；Validation/Test不使用固定训练ID。

新训练输出：`outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v1/ddpg/AggregateShipDDPG-v0_N/`。
初次运行N=1；Zoo后续运行使用新编号，不覆盖历史模型。若运行编号改变，应通过模块的
`--model-path` 指定实际模型，或同步根目录三个评估入口的路径，不能误评估旧的N=1模型。
独立输出分别为 `outputs/ddpg_fixed_parent_005_evaluation_economic_v1`、
`outputs/ddpg_fixed_parent_005_validation_economic_v1`、`outputs/ddpg_fixed_parent_005_test_economic_v1`，
每次评估再创建带时间戳的独立子目录。
历史模型、Replay、Trial18及全部旧输出保留。

PyCharm使用当前项目 `.venv` 解释器，工作目录为本DDPG工作树根目录，
直接绿色运行 `run_ddpg_train.py`。等价PowerShell命令（仅供用户确认后执行）：

```powershell
& '.\.venv\Scripts\python.exe' '.\run_ddpg_train.py'
```

该脚本仍通过 `sys.executable -m rl_zoo3.train` 调用原生Zoo，使用正式YAML、
CUDA、`--eval-freq -1` 和 `--save-replay-buffer`；500轮周期验证由Episode回调负责。
`train.py` 是次要的原生SB3入口，已同步固定ID、tau、noise、两种周期回调及最终Replay保存；
正式实验优先使用根目录Zoo入口。

## 每20 Episode固定训练航段诊断

在20、40、…、500次真实Episode结束时，`FixedTrainEvaluationCallback` 保存独立checkpoint，
以同一解释器启动独立CPU子进程，加载模型副本，对parent_005_seg01从头确定性运行。
使用 `DDPG.load(action_noise=None)`、`predict(deterministic=True)` 和 `torch.inference_mode()`，
不执行learn、梯度或Target更新，不接触训练Replay，不重置父进程Python/NumPy/PyTorch随机状态。
当前正式配置没有VecNormalize，直接复用训练环境的状态和动作映射。

训练运行目录下的 `fixed_train_evaluation/` 保存：

- `training_episodes.csv`：带探索噪声的每轮训练奖励、平均每步奖励、调用步数、实际物理步数、完成状态、coverage和终止原因。
- `episode_0020/model.zip` 等25个独立模型和checkpoint时钟元数据；不在每个节点重复保存Replay。
- 各节点 `evaluation/fixed_train_*/fixed_train_summary.csv`、`fixed_train_metrics.json`、`fixed_train_step_data.csv`。
- 累计 `fixed_train_evaluation.csv/json`：轮数、航段ID、累计/平均每步奖励、长度、完成状态、coverage、终止原因、
  四项经济成本及总成本、SOC初/终/最小/最大、启停次数和FC功率变化RMS。

失败前缀成本仅为已执行成本，完整航段经济成本字段留空；未执行失败步不伪造电池功率或时间。
固定航段诊断属于训练过程诊断，不能作为独立泛化结果。训练噪声奖励、固定航段无噪声诊断、
Validation8泛化结果、最终Test4测试结果分开保存和比较；不使用Test4挑选模型。

训练后独立诊断可绿色运行 `run_ddpg_fixed_eval.py`；指定其他checkpoint用：

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.sb3_manual.ddpg_aggregate.fixed_train_eval --model-path '实际模型路径.zip' --episode 500
```

## 每100 Episode独立Validation8

回调按实际 `dones` 计数，在100、200、300、400、500触发，绝不用环境步近似轮数。
在当前Zoo运行目录的 `periodic_validation/episode_0100/` 等节点保存独立 model.zip，
随后以同解释器启动独立CPU验证子进程，确定性遍历原Validation8，不调用Test。

验证子进程加载模型副本，不更新Actor/Critic/Target或Replay，也不重置训练进程的
随机数或线程设置。保留每节点CSV/JSON、逐步成本与功率/SOC图，另有累计
`periodic_validation.csv/json`，记录完成率、平均累计奖励、四项经济成本及SOC表现。
失败前缀费用不冒充完整航段费用。

时钟保持SB3原口径：回调位于终止环境步之后、该transition入Replay及该rollout的
Adam更新之前，checkpoint_metadata.json明确记录该阶段；不额外做Adam/Target更新。
自定义完成轮数随checkpoint保存，恢复时继续计数；每个节点禁止覆盖。
正式训练结束仍由Zoo按原选项保存一次最终Replay，不在每个验证节点重复保存Replay。

## Optuna隔离

仍实际调用 Zoo ExperimentManager、原生Optuna及SB3 learn；不改site-packages。
搜索空间仍仅tau [.005,.06]（log）和noise_std [.07,.15]，其余历史搜索配置不变。
原77760步预算、两次Validation8评估、TPE/NopPruner、单job、最多20trial保持不变。
正式500轮的周期回调不加入Optuna YAML。

新Study：`ddpg_tau_noise_economic_tanh_v1`。
新存储：`sqlite:///outputs/ddpg_optuna_economic_v1/study.db`。
入口拒绝旧奖励Study名称；trial/study/参数导出均记录固定reward_identity，
不允许沿用无身份或不同权重的Study。旧Study数据库及Trial18均未修改。
本轮没有执行搜索或创建新Study。
参数导出以新Study的 `effective_parameters.json` 及 reward_identity 为准；
原生全量 DDPG `--trial-id` converter 不适用于这里仅含tau/noise的搜索参数，未新增转换逻辑。

本次不修改Optuna脚本、搜索YAML、数据库或历史Trial。历史搜索使用lambda_soc=17，
不能作为lambda_soc=8的同一Study继续搜索；现有奖励身份检查会拒绝混用，历史结果保留。
固定ID和tau=.045、noise=.1仅用于本次正式固定航段实验，不改历史随机Train35搜索配置。

## 本次修改与限定验收

修改范围：环境增加固定Train ID；正式YAML和次要train入口同步tau/noise；
经济配置仅将SOC权重17改为8；三个原根目录入口同步新模型/输出路径；
新增固定航段评估模块及根目录入口；更新本README和相关配置/奖励断言，新增固定航段测试。
物理模型、数据集、Validation/Test评估实现、历史Optuna脚本/YAML/数据库均未修改。

相关测试分批执行，19项通过：`test_fixed_train` 10项、`test_periodic_validation` 4项、
`test_test4` 数据适配3项、`test_economic_reward` 固定配置断言1项、
`test_zoo_optuna.NativeConfigurationTests.test_fixed_configs_and_budget` 1项。
最后一项只构建配置，不运行Optuna；其历史FixedTrial的noise_std=.06低于现有采样下限，
产生一条已存在的UserWarning，不影响本次配置检查，未扩大范围修改该旧fixture。

真实Train有界检查：小模型4步/4次Adam更新及独立子进程3步推理；
原生Zoo正式配置另跑32步，Replay=32、Adam=0（未达learning_starts=5000）。
20轮/100轮回调以模拟结束事件验证，分别触发25次/5次；没有执行完整Validation8或Test4推理。
编译、固定评估CLI `--help`、`git diff --check` 通过；未启动正式500轮训练。

相关测试复现（项目根目录PowerShell；无需运行正式入口）：

```powershell
$pkg = 'experiments.sb3_manual.ddpg_aggregate'
& '.\.venv\Scripts\python.exe' -m unittest "$pkg.test_fixed_train" "$pkg.test_periodic_validation" "$pkg.test_test4" "$pkg.test_economic_reward.EconomicRewardTests.test_fixed_config_is_loaded_only_on_request_and_identity_is_stable" "$pkg.test_zoo_optuna.NativeConfigurationTests.test_fixed_configs_and_budget" -v
git diff --check
```
