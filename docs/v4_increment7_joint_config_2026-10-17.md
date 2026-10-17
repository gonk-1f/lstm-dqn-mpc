# 独立航段：单步TD、7档增量动作与Zhou联合奖励

## 隔离与配置

- 基准：`a765dd151a099d278725e62907deb63499020843`。
- 新分支：`feat/v4-increment7-zhou-td1`；工作树：`.worktrees/v4-increment7-zhou`。
- 原`feat/direct-power-mlp-ddqn`工作树、正在运行的绝对功率实验及历史产物未改动。
- 固定数据：`operating_dataset_independent_v2`，`splits_v2_exclude046_seg01`，Train35/Validation8/Test4；未加载Test。
- 新组合：`state_mode=minimal3`、`action_mode=increment7`、`reward_cost_mode=zhou_joint`、`n_step=1`。
- 状态严格为`[SOC, current_load/600, previous_actual_FC/600]`；网络`3–128–128–7`，ReLU。
- Gamma1、Adam0.001、Batch64、Replay300000、replay32、Target软更新tau0.001、reward_scale0.001、seed42、100轮及原epsilon调度不变。
- PER保持alpha0.6、beta1、epsilon1e-6、真实失败终止配额2；失败惩罚与完整供电选模资格不变。

这组同时改变TD步数、动作表示和正常奖励组成，属于新的组合方案，不能把以后效果的变化单独归因于其中一项。

## 动作与一步目标

Q输出索引0至6依次对应`[-60,-30,-10,0,10,30,60] kW`。
实际FC功率为上一实际FC功率加所选增量。初始FC为0，第一步仅可能执行0/10/30/60 kW。

现有环境与`select_power()`接口继续传递实际绝对FC功率；Agent根据当前状态中的上一FC功率转换到增量索引。环境轨迹同时记录`action_mode/action_index/action_delta_kw`，进入Replay前检查一致性。下一动作掩码根据`next_state`中的实际FC功率重新映射，不能沿用当前步索引。

物理mask先核对0～600 kW、电池充放电功率及下一SOC，再与7档增量候选取交集。没有裁剪或绝对动作回退。若物理绝对mask非空而增量集合为空，单独标记`increment_action_limited`，沿原真实失败流程处理；若第一步即失败，不虚构已执行动作。

一步目标复用原Masked Double-DQN：

`y = scaled_reward + (1-done)*Q_target(next_state, argmax_legal Q_online(next_state))`

真实终止不bootstrap，最后一个真实动作承担一次既有失败惩罚；不制造岸电、末端充电或虚拟动作。

## 奖励与经济对账

每个真实执行步先以未缩放单位计算：

```text
C_SOC_equ_step = (SOC_before-SOC_next)*624/(eta_ref*LHV_H2)*21.9
C_H2_equ_step = original_actual_hydrogen_cost + C_SOC_equ_step
raw_reward = -(C_H2_equ_step + original_FC_degradation + original_battery_degradation
               + 200*(SOC_next-0.6)^2 + (0.11025/60)*actual_delta_FC^2)
Replay_reward = 0.001*raw_reward
```

- `eta_ref=0.5711999752720877`：沿用正式效率曲线10～600 kW、间隔10 kW的60点算术平均；LHV沿用120 MJ/kg，即33.333333333333336 kWh/kg。
- 二次项系数`0.0018375`奖励等效单位/kW²；10/30/60 kW变化分别对应0.18375/1.65375/6.615未缩放单位。
- SOC修正保留正负号，使用实际SOC差，不重复乘充放电效率。它逐步望远镜求和到整航段的唯一SOC等效氢耗修正，不在终端再次收费。
- FC原有变载退化仍属于真实经济模型。二次变载项及SOC偏差项仅用于控制性能奖励，不能加到实际经济账本或选模成本中。
- 整航段Zhou评价仍是等效氢耗费用＋原FC退化＋原电池退化。原四项物理账本保留，独立航段岸电项为0。
- 旧评价配置身份保持不变；新实验元数据另外明确记录逐步SOC等效项和二次惩罚公式、动作身份、网络维度、n_step及源码哈希。不能使用旧61输出checkpoint续训7输出模型。

## 可达性与必要检查

明细：`docs/results/v4_increment7_reachability_20261017.csv`；复算：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& 'D:\py\Python3\python.exe' scripts/check_v4_increment_reachability.py
```

方法在61个实际FC档位维护SOC外包络，并另保留真实到达点及前驱。包络非空本身不作为可行证明；成功结论必须回溯完整合法动作序列，并由正式环境逐步执行验证。使用完整负载轨迹进行离线搜索，因此属于非因果存在性见证，不是给在线DDQN提供未来输入，也不是推荐控制器或经济最优轨迹。

- Train35/35、Validation8/8全部取得完整正式环境见证，共20,062个控制步。
- 所有首步FC≤60 kW，所有相邻FC变化≤60 kW；没有裁剪或回退。
- 见证SOC总范围`[0.20000005601758686,0.7999999781332527]`，无越界。该搜索接近边界，不代表具有稳健控制余量。
- 无岸电、末端补能；Test payload读取数0。
- 首遍正式环境验证对每一步额外断言7档增量；另对`parent_015_seg01`、最长`parent_037_seg01`及短`parent_046_seg02`通过新增增量接口复核，动作序列、SOC及Zhou总账完全一致；逐步经济累计与整航段评价差异最大1.82×10⁻¹²元，详情见CSV。
- 定向检查26项通过：增量索引/探索/Greedy、物理mask、一步DDQN目标与终止屏蔽、一次反向更新、真实失败仅一次惩罚、逐步等效氢耗及短尾步对账、checkpoint互拒/恢复身份、旧独立绝对功率入口及诊断兼容。

保持旧的61档绝对功率和原奖励模式可用，不改物理模型、经济参数及数据文件。没有启动正式训练。

## PyCharm PowerShell启动命令

使用新工作树，不要在仍在运行的旧工作树内启动本组。新输出目录必须为空；已有同名结果时改用新的目录名。

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v4-increment7-zhou'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$ddqnPython = 'D:\py\Python3\python.exe'
$runDir = Join-Path (Get-Location) 'outputs\v4_independent35_m3_delta7_zhou_td1_per_seed42_100r'
$ddqnArgs = @(
  '--dataset-mode','independent',
  '--data-root','data/processed/operating_dataset_independent_v2',
  '--state-mode','minimal3','--action-mode','increment7',
  '--reward-cost-mode','zhou_joint','--n-step','1',
  '--rounds','100','--seed','42','--beta-soc','200',
  '--reward-scale','0.001','--reward-feedback','original',
  '--replay-sampling','prioritized','--per-alpha','0.6','--per-beta','1','--per-epsilon','1e-6',
  '--cadence','replay32','--failure-terminal-quota','2','--failure-penalty-scale','1',
  '--episode-credit-scope','voyage',
  '--experiment-id','V4-INDEP35-M3-DELTA7-ZHOU-TD1-PER-S42',
  '--diagnostic-rounds','1,10,20,30,40,50,60,70,80,90,100'
)
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs
```

从完整轮次checkpoint恢复：重开终端后先重新执行上面的目录、PYTHONPATH、解释器、`runDir`和`ddqnArgs`变量定义，然后仅执行下面这一行，不再执行启动行：

```powershell
& $ddqnPython -m v4.feedback_study --output-dir $runDir @ddqnArgs --resume-from (Join-Path $runDir 'training_state_latest.pt')
```

恢复要求源码、动作模式、奖励、数据manifest、超参数和诊断轮次身份一致。从最新完整轮次继续，未完成轮次不作为已完成进度。开始新实验不传`--resume-from`，由现有入口新建网络、Adam和空Replay。

## 文件范围

- `src/v4/actions.py`：共享动作身份和合法增量交集。
- `src/v4/dqn.py`：7输出、实际FC与增量索引映射、Replay与加载校验。
- `src/v4/control.py`、`independent_economics.py`：执行mask、逐步Zhou等效费用、联合奖励。
- `src/v4/failure_replay.py`：增量受限失败沿用原一次性失败学习。
- `src/v4/feedback_study.py`、`monitored_training.py`、`train.py`：入口、实际参数传递、元数据与共享评价接口。
- `src/v4/diagnostics.py`、`decision_diagnostics.py`：7动作Q索引及轨迹字段。
- 三个小型定向测试文件、一个可复算脚本、43行可达性CSV及本文；没有数据重建或历史输出改写。
