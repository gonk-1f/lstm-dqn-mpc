# DDPG 聚合经济奖励与周期验证

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
- **lambda_soc = 100*C_typ = 17.429534153424196**。
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
| 独立Validation8 | run_ddpg_validate.py |
| 授权后的独立Test4 | run_ddpg_test.py |
| 原生Zoo Optuna搜索 | run_ddpg_optuna.py |

本次开发没有执行这四个正式入口。验证/Test默认指向新奖励训练输出的首个模型；
新奖励训练尚未开始，故该默认模型要在用户后续训练完成后才存在。
也可通过原模块的 `--model-path` 明确选择模型。
旧策略可以在新奖励下单独重新评估，输出明确标记 legacy；
旧奖励checkpoint不能恢复到新奖励训练，不混用旧奖励的评估结果。

原生训练参数：
gamma=.999、learning_rate=.001、batch_size=64、learning_starts=5000、
train_freq=32、gradient_steps=1、tau=.03、normal noise sigma=.12、
network=[256,256]、buffer_size=1000000、device=cuda。
Zoo仍以20000000环境步作安全上限，原生 StopTrainingOnMaxEpisodes=500。
每轮从原Train35随机选一条完整航段；真实物理失败终止也计为一个已结束Episode。

新训练输出：`outputs/ddpg_train_500ep_economic_v1/ddpg/AggregateShipDDPG-v0_N/`。
独立评估输出使用 `ddpg_validation_economic_v1`、`ddpg_test_economic_v1`。
历史模型、Replay、Trial18及全部旧输出保留。

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
搜索空间仍仅tau [.005,.15]（log）和noise_std [.03,.12]，其余算法配置不变。
原77760步预算、两次Validation8评估、TPE/NopPruner、单job、最多20trial保持不变。
正式500轮的周期回调不加入Optuna YAML。

新Study：`ddpg_tau_noise_economic_tanh_v1`。
新存储：`sqlite:///outputs/ddpg_optuna_economic_v1/study.db`。
入口拒绝旧奖励Study名称；trial/study/参数导出均记录固定reward_identity，
不允许沿用无身份或不同权重的Study。旧Study数据库及Trial18均未修改。
本轮没有执行搜索或创建新Study。
参数导出以新Study的 `effective_parameters.json` 及 reward_identity 为准；
原生全量 DDPG `--trial-id` converter 不适用于这里仅含tau/noise的搜索参数，未新增转换逻辑。
