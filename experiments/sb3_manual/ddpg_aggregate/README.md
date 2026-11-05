# 聚合燃料电池–锂电池 SB3 DDPG

版本：`ddpg_economic_env_cycle_soc_v6`。模型仍为一个600 kW聚合燃料电池系统
和一个624 kWh聚合锂电池系统，动作映射、SOC动力学和35/8/4数据划分不变。

## 五维状态与 Critic 损失

观测按以下顺序输入网络：
`[SOC_t, P_load,t/600, P_FC,t-1/600, (P_load,t-P_load,t-1)/600, t_elapsed/3600]`。
FC功率是上一已执行时刻的实际功率；首步负载变化量为0。
负载变化只使用当前及上一秒负载，不读取未来负载。运行时间以小时为单位，
不除以航段总时长，也不需要预计到港时间。
Actor输入由3维变为5维；Critic输入为5维状态与1维动作，共6维。

正式Zoo入口和备用`train.py`均使用继承SB3 DDPG的`HuberDDPG`，
仅将Critic的TD误差损失改为Huber（delta=1）。Actor目标、Adam更新次数、
Target软更新时钟、探索噪声和其余训练参数保持原样；batch_size仍为64。
新增状态与损失配置写入模型身份，新实验须从头训练，不能续训旧三维状态模型。

## 运行周期、负奖励与完整经济账本

```text
reward_economic_cost = C_H2 + C_bat_eq + C_LB_deg + C_FC_transient + C_FC_low + C_FC_high
soc_penalty = 10 * (SOC_next - 0.5)**2
reward = -(reward_economic_cost + soc_penalty)  # 仅真实执行步
reward = -20000            # 原有真实物理不可行失败
total_cost = C_H2 + C_bat_eq + C_LB_deg + C_FC_deg  # 完整账本，FC仍含启停
```

奖励不使用tanh、倒数或C_ref缩放；有限有符号合计直接取负，
不取绝对值、不裁剪、不加epsilon。非有限值明确报错，完整经济账本原有的负值检查不变。
SOC惩罚只在真实执行步计算一次，不额外乘时间步长，不进入完整经济成本。
电池退化模型内部的SOC应力函数和原SOC物理边界仍保留。

环境在reset时管理FC启动；运行期为on/idle，0 kW仅表示零出力，
不意味着物理关机，0→正功率不再反复产生启动循环。航段结束时环境关闭FC/LB，
在同一终止时刻归零，两者不增加运行时间，SOC保留。每次航段只在关闭时结算一次
23.91 μV/cycle；该成本仅进入完整经济账本，仍遵守原EOL封顶。
正常完成时，关闭造成的FC变载成本进入最后真实执行步的奖励，
不制造额外动作、Replay转移或一秒航段。
真实物理失败也关闭设备并保存关闭成本，但该未执行调用的训练奖励固定为-20000，
不再叠加经济奖励或SOC惩罚，不回写先前Replay；失败奖励自身不进入物理经济账本。
因此失败航段的实际Return不能机械地等同于`-sum(reward_total_cost_cny)`。

- `C_H2`：原物理账本的实际氢耗成本，保留现有氢价、效率曲线及LHV。
- `C_FC_deg`：四工况电压损失按下述寿命阈值换算的更换成本。
- 奖励仅纳入FC变载、低载、高载三项；环境周期的启停成本只从奖励中排除，
  四工况系数、累计电压损失和完整退化账本保留，运行中的零出力仍按on/idle计价。
  FC各成本按本步原始电压损失占比分摊同一EOL裁剪后的账本成本，
  不从账本硬减固定启停金额。
- `C_LB_deg`：原加权Ah吞吐模型，以物理模型实际电流计价。
- `C_bat_eq`：有符号电芯能量变化除以固定`eta_fc_avg`及LHV后乘氢价。
  放电为正，充电为负；SOC变化已含原充放电效率，不重复计效率或DC/DC损耗。
  无论FC开关状态，电池等效项均使用`eta_fc_avg=0.5755641791002982`；
  FC实际氢耗继续采用当前功率点效率。参考Zhou等(2025)公式(25)，固定`k=1`。

该固定效率按`integral(P dP)/integral(P/eta(P) dP)`计算，采用现有曲线的9个
正功率节点（含既有600 kW插值端点）作梯形积分，区间为68.3628–600 kW。
假设区间内每kW均匀加权，不把节点简单等权，也不把0–68.3628 kW低功率插值
或关闭状态纳入参考运行区间。数据来自`FC_Data.xlsx`的`Sheet1!A2:B12`，
沿用既有100→600 kW映射；节点、积分值及源文件SHA记录在奖励JSON配置中。
已有Train35轨迹来自旧策略且含两个失败前缀，没有代表性FC运行分布的明确依据，
因此没有将其作为运行时间权重。该值是曲线参考平均效率，并非实测运行平均。
`C_ref=0.14112952687229602`仅作为历史标定元数据保留，不参与v6奖励计算。
配置SHA及成本口径身份会区分旧模型，旧瞬时效率模型不能直接续训或混用评估。

`total_cost_cny`为含环境FC周期结算的完整经济总值，不含SOC惩罚；
`reward_economic_cost_cny`为六项经济成本，`soc_deviation_penalty_cny`为独立SOC惩罚，
`reward_total_cost_cny`为两者之和。正常末步中的变载/周期结算已包含在对应成本中，
`shutdown_*_cost_cny`只作事件明细，不能再重复相加。
逐秒记录`fuel_cell_transient_cost_cny`、`fuel_cell_low_load_cost_cny`、
`fuel_cell_high_load_cost_cny`、`fuel_cell_start_stop_cost_cny`，以及
`fc_efficiency_actual`、`equivalent_fc_efficiency`和`battery_current_a`。
真实区间的`fc_actual_kw/battery_actual_kw`不会因关机改写为0；关闭后的0值另存
`post_shutdown_fc_kw/post_shutdown_battery_kw`，并标记`shutdown_duration_s=0`。
功率图在同一时刻另画关闭点/竖线，保持最后真实区间的功率平衡；不补造末端负载或时间。
FC Ramp RMS仅统计真实执行步；末状态及`last_interval_*`分别说明关闭后状态和最后实际出力。

## 退化模型

FC保留Zhou四工况系数：低负载10.17 μV/h、高负载11.74 μV/h、
变负载0.0441 μV/kW、启停23.91 μV/cycle。
采用Mylonopoulos的0.95 V初始单电池电压下降15%作为报废标准：
`0.95 × 0.15 × 10^6 = 142500 μV`。
设备价格、50%更换比例和600 kW聚合功率映射不变。
现有80%高负载分界明确属于项目建模假设，不作为Zhou原文参数。

LB保留Zhou的F(SOC)、G(I)、加权Ah吞吐、15000倍额定容量寿命归一化及设备价。
退化使用物理电池更新中的实际电流，避免用未经效率处理的母线功率直接除以432 V。
不改变环境原有SOC更新、充放电效率、容量或物理功率约束。

## 历史Train35参考成本标定

`C_ref`固定存于`economic_reward_config.json`的`c_ref`，从现有正式基线
在Train35各航段实际执行的一秒轨迹重新核算上述四项经济成本后取全体步中位数。
历史核算按142500 μV寿命阈值和实际电池电流重新计价，
得到 **C_ref = 0.14112952687229602 CNY**。35条Train航段共448434个实际物理秒，
其中33条完整结束，另2条只使用真实失败前缀；负、零和非有限总成本样本均为0。
重新核算报告：`outputs/ddpg_reward_recalibration_v3/recalibration_20261104_160123_698139/recalibration_result.json`。
基线模型为`outputs/ddpg_train_500ep/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip`，
原始轨迹来自`outputs/ddpg_reward_calibration/collection_20261103_133754_684927/`。
模型、数据manifest、原始轨迹和重新核算报告的SHA及各航段覆盖情况保存在奖励配置中。
只使用Train35，不使用Validation8或Test4；真实失败调用和未执行尾段不补造成本。
这些记录仅保留历史标定来源；v6训练和评估不使用该参考值计算奖励。

原标定工具命令（历史核算用途，v6训练前无需执行）：

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.sb3_manual.ddpg_aggregate.recalibrate_costs --collection-report 'outputs/ddpg_reward_calibration/collection_20261103_133754_684927/calibration_collection.json' --write-config
```

该命令会创建新的重核算报告并写入奖励配置；
训练及恢复期间应保持该配置不变，避免改变模型对应的标定身份。

## 直接运行入口

打开实际DDPG工作树，选择项目`.venv`解释器，在PyCharm绿色运行：

| 用途 | 根目录脚本 |
|---|---|
| 正式500 Episode训练 | run_ddpg_train.py |
| 每20轮同航段诊断的独立入口 | run_ddpg_fixed_eval.py |
| 独立Validation8 | run_ddpg_validate.py |
| 最终授权后的Test4 | run_ddpg_test.py |
| 原生Zoo Optuna | run_ddpg_optuna.py |

正式训练仍调用`sys.executable -m rl_zoo3.train`，加载`zoo_ddpg.yml`。
正式参数：gamma=1、learning_rate=.001、batch_size=64、learning_starts=5000、
train_freq=32、gradient_steps=1、tau=.01、normal noise_std=.1、
Actor/Critic隐藏层均为[128,128]、buffer_size=1000000、device=cuda、n_envs=1、seed=42。
安全上限20000000环境步，实际500次Episode结束后停止。每次固定
`parent_005_seg01`，SOC=0.6、FC=0 kW且物理状态为on/idle，从第0秒开始；
真实失败也计一次Episode结束。
取消FC的48 kW/s硬爬坡约束，保留0–600 kW额定范围、电池功率及SOC边界；
FC变载退化仍按实际功率变化计算。模型身份记录无爬坡限制，防止旧约束模型混用。
备用`train.py`同步gamma=1，其余既有算法参数不变，不作为本次正式入口。

新训练目录：`outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_N/`。
初次N=1，后续Zoo使用新编号，不覆盖模型。三个独立评估入口默认指向N=1最终模型，
如编号不同请通过相应模块的`--model-path`选择实际checkpoint。新模型在用户后续训练后才存在。
独立输出根分别为`outputs/ddpg_fixed_parent_005_evaluation_economic_v3`、
`outputs/ddpg_fixed_parent_005_validation_economic_v3`、`outputs/ddpg_fixed_parent_005_test_economic_v3`，
每次另建时间戳目录。历史模型、Replay及搜索结果不作为本次初始条件。

新训练不传`--trained-agent`，不载入旧模型或旧Replay。完整奖励身份包含版本、
`c_ref`、配置SHA及成本模型定义；恢复、Validation、Test和Optuna均拒绝旧身份或无身份模型。
同版本、相同完整身份的checkpoint才能用于后续恢复与评估；v5模型和Replay不能用于v6续训，
不重算或重写历史Replay奖励。

```powershell
& '.\.venv\Scripts\python.exe' '.\run_ddpg_train.py'
```

以上仅是运行命令，本次开发不启动正式500轮训练或Optuna搜索。

## 周期诊断与验证

每20次真实Episode结束保存checkpoint，并在独立CPU子进程中确定性运行固定Train航段。
每100次真实Episode结束完整执行Validation8，节点为100、200、300、400、500。
使用`DDPG.load(action_noise=None)`、`predict(deterministic=True)`和`torch.inference_mode()`，
不更新Actor、Critic、Target或Replay，不重置训练进程的随机状态。Test4不参与周期验证。

`fixed_train_evaluation/`保留带噪训练Episode汇总、各20轮模型和逐步CSV，
`periodic_validation/`保留各100轮checkpoint、Validation8逐航段CSV/JSON及功率/SOC图。
评价先看完成率、coverage，再比较完整航段经济成本；失败前缀成本不冒充完整航段成本。
评估子进程直接写入`episode_XXXX/`以避免Windows路径过长，不覆盖已有逐步文件。
回调时钟仍为环境步返回后、Replay插入与本rollout的Adam更新前，元数据明确标注。
最终Replay只由Zoo按原选项保存，不在每个节点重复保存。

## Optuna隔离

仍调用Zoo原生ExperimentManager及Optuna，搜索空间不变：
tau [.005,.06]（log）、noise_std [.07,.15]；其余配置和77760步预算、
两次Validation8、TPE/NopPruner、单job及最多20trial不变。
新Study为`ddpg_tau_noise_economic_env_cycle_soc_v6_obs5_huber1`，
新存储为`sqlite:///outputs/ddpg_optuna_economic_v3/study.db`。
Study、trial和导出的`effective_parameters.json`保存同一奖励身份；
旧Study和不同标定身份不能混用。此改动不运行搜索。

## 检查

完整测试包括奖励数值/版本隔离、FC四工况与寿命阈值、LB应力/实际电流、
固定航段配置、周期触发、训练/验证分离、深Windows路径及保存加载。
有界测试只使用临时小模型，部分执行数步Train/Validation；Test4适配测试只检查数据与reset，
不运行完整Test4模型推理。标定单独只使用Train35。

```powershell
$pkg = 'experiments.sb3_manual.ddpg_aggregate'
$modules = Get-ChildItem '.\experiments\sb3_manual\ddpg_aggregate' -Filter 'test_*.py' | ForEach-Object { "$pkg.$($_.BaseName)" }
& '.\.venv\Scripts\python.exe' -m unittest @modules -v
git diff --check
```
