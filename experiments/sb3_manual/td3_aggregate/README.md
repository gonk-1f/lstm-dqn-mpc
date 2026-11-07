# 原生 SB3 TD3 聚合船舶实验

## 独立简化奖励对照实验

最新的纯控制奖励入口为 `run_td3_train_control.py`，配置 `zoo_td3_control.yml`，环境
`AggregateShipTD3Control-v0`。正常奖励只保留
`-(0.0025*(P_FC_actual-P_FC_previous_actual)**2 + 10*(SOC_next-0.5)**2)`，
仍除以10000；失败仍为-20000。实际氢耗继续记入物理账本，但不参与该版本奖励。
训练从新模型/新Replay开始，输出独立保存到 `outputs/td3_control_fc_soc_500ep/`，
其余TD3参数、500次episode预算及20/100轮评估流程不变。独立评估使用
`--reward-mode control`；暂停方式仍为在具体run目录创建 `pause_after_episode.request`。
恢复使用 `run_td3_train_control.py --trained-agent "本版本模型完整路径"`。
不允许用含氢耗奖励模型或其Replay继续训练纯控制奖励版本。

下述 `run_td3_train_simple.py` 保留上一版含氢耗的三项奖励实验，方便复现历史记录。

PyCharm直接运行根目录 `run_td3_train_simple.py`，使用原生Zoo TD3、独立环境
`AggregateShipTD3Simple-v0` 和 `zoo_td3_simple.yml`。新实验从头开始；不能加载
旧经济奖励模型或旧Replay。原入口和 `zoo_td3.yml` 保持原设置。

用户确认恢复历史平方波动项，本次原始奖励为：

`raw_reward = -(C_H2 + 0.0025*(P_FC_actual-P_FC_previous_actual)**2 + 10*(SOC_next-0.5)**2)`。

系数追溯提交 `49082cdbb45fcbd964420b36e6d0263682783370` 的DDPG环境；
原历史奖励只含FC平方波动和SOC，本次按要求额外加入实际氢耗成本，故并非完全复现
旧实验，也不使用绝对值项。失败仍为 `-20000`，所有奖励仍除以10000再交给TD3。
最后真实负载区间按该步实际功率变化计算一次；零时长关闭事件只保留原物理账本，
不增加额外关闭波动惩罚或SOC项。

电池等效氢耗、FC和电池退化均不进入此训练奖励，但完整经济账本继续保存。
FC额定功率、投影、SOC动力学、状态/动作及全部TD3参数均与原配置相同；仍固定
`parent_005_seg01`、500次实际episode结束事件，每20轮固定Train诊断、每100轮Validation8。
不读取Test；失败episode不等同于完整成功航段。

```powershell
.\.venv\Scripts\python.exe run_td3_train_simple.py
```

结果单独保存到 `outputs/td3_simple_h2_fc_soc_500ep/td3/AggregateShipTD3Simple-v0_N/`。
`training_episodes.csv`逐轮保存回报、三项奖励组成、失败覆盖奖励、长度、完成状态和
SOC初末变化；`reward_curves.png`及`reward_curve_summary.json`每20轮和结束时更新。
保留SB3原生终端、CSV和TensorBoard日志。两套奖励定义不同，比较时应同时查看独立
确定性评估的完成率、SOC及同口径完整经济账本，不能直接比较两种原始奖励的数值大小。

安全暂停：在具体run目录创建空文件 `pause_after_episode.request`，当前episode结束后
Zoo自动保存模型及Replay退出。恢复前移除该请求文件，并指定本实验对应模型：

```powershell
.\.venv\Scripts\python.exe run_td3_train_simple.py --trained-agent "完整路径/AggregateShipTD3Simple-v0.zip"
```

Zoo会从模型所在目录加载对应Replay；禁止将其他实验的Replay复制进该目录。
单独评估新模型时，沿用原入口并显式选择新奖励，例如：

```powershell
.\.venv\Scripts\python.exe -m experiments.sb3_manual.td3_aggregate.fixed_train_eval --model-path "完整路径/model.zip" --reward-mode simple
.\.venv\Scripts\python.exe -m experiments.sb3_manual.td3_aggregate.validate --model-path "完整路径/model.zip" --reward-mode simple
```

本实验使用 **Stable-Baselines3 2.9.2 原生 `TD3`** 和 **RL Zoo 2.9.2** 的
`rl_zoo3.train` / `ExperimentManager` / `model.learn()`。不替换 `TD3.train()`，
不使用自定义 Huber。TD3 的三个改进是双 Critic 取较小目标值、延迟 Actor/目标网络
更新、目标动作平滑。实现和默认值见 [SB3 官方 TD3 文档](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)。

## 环境与奖励

注册 ID：`AggregateShipTD3-v0`。`reward_wrapper.make_env()` 直接包装现有
`ddpg_aggregate.env.AggregateDDPGEnv`；不复制物理模型、不修改 DDPG 文件。
训练固定 `parent_005_seg01`（9258个1秒区间），Train35/Validation8 与原项目相同。
初始 SOC=0.6、目标 SOC=0.5、FC=600 kW、电池=624 kWh、5维状态、1维连续
动作 `[-1,1]`、物理投影和无48 kW/s约束均保持原样；状态不读取未来负载。

原奖励保留六项经济成本及 `10*(SOC_next-0.5)**2`，启停循环只入完整经济账本。
包装器只执行 `scaled_reward = raw_reward / 10000.0`，失败 `-20000 → -2`。
所有正常/失败/末步奖励使用相同比例。无 VecNormalize、clipping、tanh 或倒数。

- `reward_sum`、`raw_reward_sum`：未缩放原奖励，保留与 DDPG 的比较口径。
- `algorithm_reward_sum`：传入 TD3 的缩放奖励累计值。
- 所有 `*_cost_cny` 和 SOC 惩罚记录仍为原值；SOC 惩罚不是真实设备成本。
- 每步 `info`/评估 CSV 导出 `raw_reward`、`scaled_reward`、`reward_scale=0.0001`。
- 模型保存 `_td3_reward_identity`、`reward_scale` 和累计 episode 计数。
  新奖励身份为 `td3_economic_scaled_1e4_v1`，同时保留基础奖励版本和配置哈希。
  训练入口续训检查和所有评估均拒绝缺失/不匹配身份或比例的模型。

## 正式配置与来源

| 参数 | 值 | 来源 |
| --- | --- | --- |
| policy / activation | MlpPolicy / ReLU | SB3 TD3 |
| learning_rate | 0.001 | Wu表1；正文写0.0001，存在冲突 |
| gamma | 1.0 | Wu |
| buffer_size | 1000000 | Wu |
| batch_size | 32 | Wu |
| tau | 0.001 | Wu参考值，调用频率不同 |
| net_arch | [256,256]，Actor和两个Critic | Wu层宽、SB3双Critic |
| learning_starts | 1000 | 本项目工程设置 |
| train_freq / gradient_steps | 32 / 1 | 当前TD3实验设置 |
| policy_delay | 2 | SB3原生默认 |
| target_policy_noise / target_noise_clip | 0.2 / 0.5 | SB3原生默认 |
| noise_type / noise_std | normal / 0.1 | 本次实验设置 |
| seed / n_envs / device | 42 / 1 / cuda | 本次实验设置 |
| episode预算 / n_timesteps上限 | 500 / 20000000 | 项目预算 |

Wu 2021 使用 **DDQN**。已核对其PDF第9页正文、第11页表1：正文学习率为0.0001，
表1为0.001；本实验按任务选0.001。当前train_freq=32是本次明确指定的实验设置。
预热后每32个环境步同时更新Critic1和Critic2一次，每两次Critic更新才更新Actor及两套
目标网络，即Actor更新间隔64个环境步。learning_starts=1000时，首次Critic更新在第1024步，
首次Actor/目标网络更新在第1056步；之后分别按32/64步间隔更新。
因此tau相同也不代表与Wu或现有DDPG有相同的实际目标更新速度。
[Wu原文](https://discovery.ucl.ac.uk/10135003/1/manuscript_accepted.pdf)。

500按真实episode结束事件计数，包含物理失败；不能解释为500条成功完整航段。
原生 `StopTrainingOnMaxEpisodes` 和 TD3 回调共同执行该预算；固定环境步数仅是安全上限。

两算法的物理任务、原始奖励和人民币账本一致，但本次网络、批量、更新频率、tau及
奖励数值尺度也不同，因此它是两套指定配置的比较，不能将性能差异全部归因于算法。

## 直接运行

PyCharm解释器选本工作树 `.venv\Scripts\python.exe`，工作目录选项目根目录。
四个根目录脚本均可绿色三角运行：

| 任务 | 脚本 |
| --- | --- |
| 正式训练 | `run_td3_train.py` |
| 固定Train航段确定性诊断 | `run_td3_fixed_eval.py` |
| 独立Validation8 | `run_td3_validate.py` |
| 最终Test4（仅选定模型后执行） | `run_td3_test.py` |

评估脚本默认选择 `outputs/td3_train_500ep/td3/` 下最近保存的最终TD3模型；
也可在PyCharm Parameters或终端传 `--model-path` 指定 checkpoint。没有模型时明确报错，
不会回退加载DDPG或短测试模型。

在本工作树根目录的PowerShell直接运行官方训练入口：

```powershell
.\.venv\Scripts\python.exe -u -m rl_zoo3.train --algo td3 --env AggregateShipTD3-v0 --gym-packages experiments.sb3_manual.td3_aggregate.env_registration --conf-file experiments/sb3_manual/td3_aggregate/zoo_td3.yml --device cuda --eval-freq -1 --seed 42 --log-folder outputs/td3_train_500ep --tensorboard-log outputs/td3_train_500ep/tensorboard --log-interval -2 --save-replay-buffer
```

也可执行 `.\.venv\Scripts\python.exe run_td3_train.py`，该入口使用当前解释器和相同
参数启动官方模块，继承控制台输出。Ctrl+C后等待Zoo保存模型和Replay再退出；入口不会
因为自身收到Ctrl+C而杀死正在保存的子进程。不要用强制停止代替保存。

```powershell
.\.venv\Scripts\python.exe run_td3_fixed_eval.py --model-path outputs/td3_train_500ep/td3/AggregateShipTD3-v0_1/AggregateShipTD3-v0.zip
.\.venv\Scripts\python.exe run_td3_validate.py --model-path outputs/td3_train_500ep/td3/AggregateShipTD3-v0_1/AggregateShipTD3-v0.zip
.\.venv\Scripts\python.exe run_td3_test.py --model-path outputs/td3_train_500ep/td3/AggregateShipTD3-v0_1/AggregateShipTD3-v0.zip
```

每次评估使用独立带时间戳目录，默认完整运行指定航段，`deterministic=True`、无探索噪声、
无learn或梯度更新。`--max-steps`仅供有界检查，结果明确标记不完整，不充当完整验证成绩。
Validation固定顺序8条，Test固定顺序4条；Test适配器仅在显式最终测试时导入。

## 周期评估与日志

每20个episode保存固定Train诊断checkpoint并启动独立CPU评估进程；每100个episode
另保存checkpoint并完整执行Validation8。不向训练Replay加入评估数据，不在训练进程加载
第二个模型，不改变训练随机数或优化器。Test不进入周期评估。

每个独立Zoo run位于 `outputs/td3_train_500ep/td3/AggregateShipTD3-v0_N/`：

- `training_episodes.csv`：原始/算法回报、SOC、功率、完成状态和成本。
- `fixed_train_evaluation/episode_0020/` 等：模型、元数据、逐秒CSV及固定航段汇总。
- `periodic_validation/episode_0100/` 等：模型、Validation8逐秒CSV/汇总/图像/JSON。
- `progress.csv`、TensorBoard events、原生终端表格：每1000步及episode结束输出。
- `AggregateShipTD3-v0.zip`、`replay_buffer.pkl`：Zoo正常结束或Ctrl+C后保存。

保留原生 `train/actor_loss`、`critic_loss`、`n_updates`、`rollout/ep_len_mean`、
`ep_rew_mean`和`time/fps`；增加Actor动作统计/饱和率、双Q统计、Replay数量、请求/实际
FC功率、电池功率、投影比例、SOC、累计完成率及原始人民币经济分项。
首个episode结束前没有真实episode均值，日志不会填虚构值。
回调每次只调用一次原生`dump_logs()`；`--log-interval -2`避免另外的episode自动dump。
`train/*`表示最近已完成更新；每步回调checkpoint发生在当前转移进入Replay及本轮更新之前，
元数据明确记录该时钟。周期checkpoint不重复保存Replay。

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir outputs/td3_train_500ep
```

## 已执行的有界验收

原生TD3/Gym注册、5维状态/1维动作、真实Train reset/step、正常/失败/终端奖励缩放、
成本/动力学不变、模型身份拒绝、保存加载、20/100/500 episode回调触发均已检查。
以下为先前train_freq=1时的有界验收记录；本次频率修改只做配置核对，未重新训练。
64步单元短训产生32次Critic/16次Actor更新；另通过官方Zoo CLI运行96步，产生
64次Critic/32次Actor更新，退出码0，原生表格、CSV和TensorBoard均有实际记录。
短训参数仅以测试实例或CLI覆盖，正式YAML未改为测试值。

安装证据和有界CLI结果见 `outputs/td3_setup_checks/`，官方96步运行原始命令与stdout/stderr
见 `outputs/td3_native_zoo_check_20261105_230804/`。这些是开发检查，不是正式实验。
本次没有启动500轮训练、完整Validation8、Test4或Optuna，也没有提交/推送Git。

实际环境：Python3.11.5、PyTorch2.8.0+cu128/CUDA12.8、Gymnasium1.2.3、SB3/Zoo均2.9.2。
Zoo为`third_party/rl-baselines3-zoo/`的editable安装；SB3仍使用原安装wheel，官方同版本
只读源码在`third_party/stable-baselines3/`，原生TD3源码已核对一致。
Zoo新版依赖`sb3-team-hf-integration`，已替换与其Hub版本要求冲突的旧`huggingface-sb3`，
`pip check`通过；PyTorch、SB3、NumPy和Gymnasium版本保持原样。
