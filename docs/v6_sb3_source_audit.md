# V6与官方SB3 DQN：源码审查及对照边界

本报告记录训练前源码审查。正式SB3对照训练由用户在PyCharm终端启动；这里不把待训练的结果写成已完成。没有读取Test payload，没有引入PER、Double-DQN、额外可恢复性约束或新奖励。

## 1. 实际版本和审查范围

|对象|实际版本/commit|取得方式及边界|
|---|---|---|
|当前项目基准|`fce060ca12fd842d8496aba8d6f922eae179075e`，`feat/v6-dqn-1s`|从实际Git HEAD确认；新工作树为`feat/v6-sb3-baseline`，基准工作区干净|
|Stable-Baselines3|`2.9.0`，`8908708f10c8ff29759c67f55c8acb56cab27463`|官方仓库tag `v2.9.0`浅检出，逐项读取下列源码；不是自写同名训练器|
|CleanRL|源码声明`2.0.0b1`，`fe8d8a03c41a7ef5b523e2e354bd01c363e786bb`|官方仓库实际HEAD浅检出；只审查，不安装或执行它的训练器|
|Tianshou|源码声明`2.0.1`，`f2402056b03acafeedf2518a5088db55add004ab`|官方HEAD及该commit的`buffer_base.py`、`prio.py`、`pyproject.toml`实际下载；仅作后续PER参考|

第三方源码位于工作树之外的`C:/Users/20883/.codex/references/`，未复制整个第三方仓库进入项目。机器路径只是本次审查位置；以下commit永久链接才是可移植的来源。版本机器可读记录见[sb3_reference_versions.json](sb3_reference_versions.json)。

官方SB3声明Python≥3.10、Torch≥2.8且<3.0、Gymnasium≥0.29.1且<2.0，来源是[该commit的setup.py](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/setup.py#L81-L90)。训练环境的实际解释器/依赖必须由本项目启动记录另行保存，不能用源码依赖范围替代安装版本。

本次已在独立工作树的`.venv`中安装SB3 2.9.0。该venv使用`--system-site-packages`复用原解释器`D:/py/Python3/python.exe`的Python3.11.5、Torch2.8.0+cu128；本轮训练设备仍显式为CPU。SB3只安装到这个venv，没有改动原V6的全局环境。

已逐文件比较安装wheel与上述官方commit的五个目标文件：`dqn/dqn.py`、`dqn/policies.py`、`common/buffers.py`、`common/off_policy_algorithm.py`、`common/torch_layers.py`。按统一换行读取后，源码文本全部相同，证据见[sb3_installed_source_match.json](sb3_installed_source_match.json)。这证明本次导入的目标实现对应所审查版本；不是仅记录了一个未实际使用的仓库地址。

主要来源：

- [SB3 DQN算法](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/dqn/dqn.py)：`_setup_model:146`、`_on_step:171`、`train:187`、`predict:233`。
- [SB3 DQN策略和Q网络](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/dqn/policies.py)：`QNetwork:18`、`DQNPolicy._build:153`。
- [SB3 Replay](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/common/buffers.py)：`BaseBuffer.sample:106`、`ReplayBuffer:158`、`add:247`、`_get_samples:307`。
- [SB3离策略训练闭环](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/common/off_policy_algorithm.py)：`_setup_model:174`、`learn:312`、`_sample_action:367`、`_store_transition:449`、`collect_rollouts:514`。
- [SB3网络层构造](https://github.com/DLR-RM/stable-baselines3/blob/8908708f10c8ff29759c67f55c8acb56cab27463/stable_baselines3/common/torch_layers.py#L108-L180)：`create_mlp`。
- [官方DQN文档](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)：用于接口和算法说明；本文行号与默认值以实际检出的2.9.0源码为准。
- [CleanRL DQN](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/dqn.py)：不是本轮训练入口。

## 2. 已确认的V6实际算法

正式run02入口是[`v4.v6_training.run`](../src/v4/v6_training.py#L261)，实例化[`DirectPowerDQN`](../src/v4/dqn.py#L1072)，调用[`masked_dqn_targets`](../src/v4/dqn.py#L60)。目标直接对Target网络的下一状态合法动作取最大值：

`target = reward + gamma * max_legal Q_target(next_state)`，但终止经验的续值为0。

这里**没有**使用Online网络选择下一动作后再到Target取值。因此当前V6是普通DQN、单步TD；同一文件上方保留的`DirectPowerDDQN`和`masked_double_dqn_targets`属于历史路径，不能当作run02实际执行算法。

run02正式配置来自已提交的[configuration.json](results/v6_dqn_1s_20261026/raw/run02/configuration.json)，也与[`v6_training.CONFIG`](../src/v4/v6_training.py#L32)逐项一致：5→256→256→13、ReLU、CPU、γ1、Adam 1e-4/betas(.9,.999)/eps1e-8、Huber beta1、clip10、Uniform Replay 1e6、batch64、预填充50,000条且零Adam、每个正式执行步一次更新、每1000次Adam纯硬Target、奖励缩放1、SOC权重.2、FC波动权重.05、失败惩罚10132.660087898294。

## 3. 十四项差异表

“差异”不自动等于bug。下表分别标记算法定义、工程实现、环境适配、参数/预算差异；新增适配必须把必要的V6语义带入SB3的真实闭环。

|项目|V6实际实现及位置|官方SB3 2.9.0实际实现|分类、对照处理|
|---|---|---|---|
|1. Online/Target初始化|`DirectPowerDQN.__init__:1087`：创建两个MLP，Target显式`requires_grad=False`、eval，然后`sync_target:1197`复制Online；每网络70,669可训练参数，另有10个非训练归一化buffer且未启用归一化|`DQNPolicy._build:153–173`分别创建两个Q网络、复制state_dict、Target设eval；Adam只接收`q_net.parameters()`。默认[64,64]，不是[256,256]；Target没有单独显式冻结，但目标计算在no_grad中|参数/工程。显式配置[256,256]与ReLU；Target不进优化器，目标不建图。eval与冻结是不同概念；原生未显式冻结本身不构成Target被更新的证据|
|2. Replay存储/采样/容量|`_UniformDQNReplay:999`为固定NumPy FIFO环；states/next_states float32、动作int64、done float32、next mask bool；`sample:1056`用独立Python RNG的`random.sample`，batch内无放回|`ReplayBuffer:158`固定NumPy环，`add:247`复制实际转移；`BaseBuffer.sample:106`用`np.random.randint`，batch内可重复；单环境时容量1e6就是1e6条。默认超时字段参与done处理|工程采样差异。两者都是Uniform边缘采样，但不是同一批次分布或随机流；保持SB3原生有放回采样并明确记录，不把旧Replay搬来冒充SB3|
|3. 合法动作mask|`select_power:1123`在合法索引中Greedy或随机；`remember:1139`存next mask；mask来自原物理计算与13档增量交集|`QNetwork._predict:68`对全部动作argmax；`DQN.predict:233`随机从整个Discrete空间；Replay没有船舶mask字段|环境适配。必须扩展Greedy、epsilon随机、warmup随机及Replay/TD全部路径。仅在env外过滤动作会改变行为，不能算完成适配|
|4. 当前动作Q提取|`DirectPowerDQN.learn:1170`：`online(states)`后按实际动作索引gather|`DQN.train:187–217`：`q_net(observations)`后同样gather记录动作|算法一致。动作索引0…12必须稳定映射−60…+60 kW，不能把绝对FC功率当索引|
|5. 下一状态TD目标|`masked_dqn_targets:60`对Target合法下一动作取最大，γ1、n1|`DQN.train:197–209`用Target全部动作max；支持n-step，但默认n1；默认γ.99|环境适配+参数。扩展为合法max并显式γ1/n1；仍保持普通DQN，不引入Online选Target估的Double-DQN|
|6. 终止是否bootstrap|V6 `torch.where(done==0, gamma*maximum, 0)`使终止续值归零；空mask只允许终止；失败惩罚通过`prepare_failure_replay:59`加到最后真实动作一次|原生使用`reward + (1-done)*discount*next_max`；`ReplayBuffer._get_samples:320–322`对TimeLimit截断清除done以允许bootstrap；`_store_transition:449`用info里的terminal_observation替换VecEnv自动reset后的观测|环境适配。空terminal mask若先得到−inf再乘0会NaN，必须安全归零。控制失败及V6有限航段正常结束为terminated；不得误标TimeLimit截断。必须存真实terminal observation/mask而非新航段reset信息|
|7. Huber/SmoothL1|`learn:1181`显式beta1，mean reduction|`DQN.train:217`使用PyTorch smooth_l1默认beta1、mean|算法一致；保持奖励缩放1。大失败目标与Huber饱和可能影响拟合，属于后续分析，本轮不能偷偷改MSE或beta|
|8. backward/clip/Adam顺序|`learn:1185–1190`清Online梯度、backward、clip Online范数10并拒绝非有限梯度、Adam.step|`DQN.train:221–225`清优化器梯度、backward、clip整个policy参数范数10、Adam.step；Target因no_grad没有梯度|工程差异。有效梯度都来自Online；显式冻结Target后clip Online可保持意图。V6的有限值检查更明确，不能因此声称SB3没有实际更新|
|9. Adam参数|`__init__:1101`显式lr1e-4、betas(.9,.999)、eps1e-8|`DQNPolicy._build:169`默认Adam，lr来自schedule；betas/eps未覆盖时由Torch默认提供|参数匹配。对照仍显式给出相同值和常量学习率，避免依赖默认或误启用学习率衰减|
|10. Target更新计数|初始复制计1；之后`learn:1193`每1000次实际Adam后复制；`target_soft_update_calls=0`|`DQN._on_step:171`在collect_rollouts每环境步后按`_n_calls`复制，使用tau；默认间隔10000环境步，且warmup也调用_on_step|参数/调度差异。关闭原生环境步复制，改为每1000次实际Adam硬复制。只把target_update_interval设1000仍会在warmup复制，并且复制发生在该步Adam之前，不能称对齐|
|11. epsilon-greedy|`v6_training.epsilon_at:62`按`(pass−1+episode_index/35)/6`从1降到.05，每航段固定；`select_power`的Python RNG也用于Replay采样|`DQN._on_step:184`按总environment timestep进度调用LinearSchedule，`predict`用NumPy全局RNG，Discrete空间有自己的RNG|参数/工程。由runner按真实pass/episode设置V6相同epsilon；不采用默认exploration_fraction=.1。seed相同不表示跨框架动作序列相同|
|12. warmup/训练频率/预算|`v6_training.remember:335`每条正式闭合Train转移入Replay后learn一次；warmup精确50,000、不更新；8次35条遍历；旧总正式更新2,782,159|2.9.0实际默认learning_starts=100、train_freq=4、batch32；`OffPolicyAlgorithm.learn:348`在`num_timesteps>learning_starts`后更新；train_freq和gradient_steps分别控制收集/更新|参数/预算。显式50,000、train_freq1、gradient_steps1、batch64；warmup后的第一条正式转移应获得第1次Adam。固定8遍与固定总步数存在下节所述冲突，不能同时宣称严格相同|
|13. checkpoint/恢复|`training_state:1201`保存Online/Target/Adam/Replay、agent Python RNG、Torch RNG、TD诊断及计数；`v6_training.run:261`轮边界恢复|`BaseAlgorithm.save:819`和DQN `_get_torch_save_params:284`保存policy与optimizer；Replay需另调`save_replay_buffer:217`。加载种子不等于恢复全局RNG；普通模型zip不含完整物理env快照|工程。需要额外保存Replay、np/Python/Torch RNG、探索/计数/数据次序及当前航段恢复信息。stage若处于航段中途，不能用“重开下一条”冒充精确续训|
|14. seed/可复现性|Online/Target初始化用Torch seed42；动作和采样共享agent独立Python Random42；每pass顺序用Random(42+pass)；warmup顺序Random42；诊断另用独立随机源|`OffPolicyAlgorithm._setup_model:174`调用BaseAlgorithm.set_random_seed；`utils.set_random_seed:29`设置Python/NumPy/Torch，action_space和env再seed。SB3采样/epsilon的消费顺序与V6不同|工程。保存实际版本、所有被使用的RNG和顺序；诊断不得消耗训练RNG。相同种子提供各自实现的复现条件，不保证两框架逐动作相同|

**已验证的初始化与精度边界：** 在实际Python3.11.5/Torch2.8.0/SB3 2.9.0环境中，seed42下两种Online网络的六组weight/bias逐元素相同，每个网络均为70,669个可训练参数；Target独立且初始复制相等。检查见[`test_network_initialization_matches_v6_and_target_is_frozen`](../tests/test_sb3_masked_dqn.py#L79)。相同初值不表示后续动作或批次相同：两个框架仍使用不同的探索/采样随机流。

V6环境状态、SOC和物理mask基于原float64计算，随后网络输入和Replay转float32；Gym观察同样为float32，但动作mask复用原始物理结果，没有从舍入后的SOC重算边界。

训练前已完成**环境3项、Masked DQN 6项，加1项短人工样本集成，共10项定向检查**，源码见[环境检查](../tests/test_sb3_environment.py)、[DQN检查](../tests/test_sb3_masked_dqn.py)与[集成检查](../tests/test_sb3_training_integration.py)。覆盖固定动作下原executor与Gym的奖励/FC/电池/SOC/下一状态/下一mask/终止一致、空后继mask的最后真实动作仅罚一次、预算关闭不伪造终止及动作前缀恢复，以及固定Q的TD、Replay实际存储采样、网络初始化、合法Greedy/探索/warmup、实际Adam/gather/Target计数和模型/Replay往返。集成检查实际进入SB3.learn，用人工两秒航段确认终止不丢、warmup首更新及续训后参数/Adam/Replay/三类RNG与不中断参照一致；报告生成不改变Replay。它们是适配必要的一致性证据，不是正式Train35/Validation8控制结果，也不证明长时间训练一定成功。

另外仅以既有V6元数据确认了报告格式器和最新版checkpoint指针解析，没有新增环境动作或梯度更新。检查记录见[v6_sb3_consistency_checks.json](v6_sb3_consistency_checks.json)。

## 4. CleanRL逐项核对：不要把不同实现强行说成相同

审查的是[上述实际commit的cleanrl/dqn.py](https://github.com/vwxyzjn/cleanrl/blob/fe8d8a03c41a7ef5b523e2e354bd01c363e786bb/cleanrl/dqn.py)，不是其他Atari/JAX变体。

|项目|CleanRL实际源码|与本轮V6/SB3对照的关系|
|---|---|---|
|网络|`QNetwork:91`：输入→120→84→动作，ReLU；`:147–150`构建Online、Adam、Target并复制参数|网络维度不同；仅审查，不能套默认网络正式训练|
|Replay|`:152–159`调用`cleanrl_utils.buffers.ReplayBuffer`，并关闭handle_timeout_termination；辅助实现`BaseBuffer.sample:218`使用NumPy randint有放回抽样|同样是实际交互入Replay，不能据此推断其批次与V6一致|
|状态链|`:173`执行env；`:183–189`截断时用final_observation替换；`:188`把terminations而非truncations写done；`:191`推进obs|检查数据链/截断的参考；船舶有限任务语义仍以原环境为准|
|动作|`:165–171`按全局步线性epsilon，随机全空间或Q argmax|无船舶合法动作mask，不能直接训练本任务|
|TD|`:197–200`在no_grad下Target全动作max，再用done停止bootstrap；Online gather实际动作|普通DQN，不是Double-DQN；需船舶mask适配|
|损失|`:201`实际为 **MSE**|并非Huber。不能声称CleanRL和本轮损失完全一致；本轮仍保持V6/SB3的SmoothL1 beta1|
|优化|`:210–212`zero_grad、backward、Adam.step，当前该文件 **没有梯度范数裁剪**|梯度确实更新，但不能编造clip10；只作为代码差异记录|
|Adam|`:148`仅传lr，默认lr2.5e-4|与本轮显式1e-4不同|
|时序|`:194–195`在global_step>learning_starts且能被train_frequency整除时更新；默认warmup10000、train_frequency10|不能套用为每环境步更新；步索引从0开始也须注意边界|
|Target|`:215–219`按全局环境步频率更新，tau默认1，频率默认500；循环位于warmup条件之内|不是每1000次Adam复制。默认train_frequency10时不能把500环境步当500优化器步|
|seed|`:133–138`设置Python/NumPy/Torch；env action_space另seed|种子机制真实存在，但不等于与其他框架相同随机流|
|保存|`:221–225`可选保存Online state_dict|不是包含Target、Adam、Replay和RNG的完整续训断点|

这些差异没有证明V6实现错误；它们证明成熟仓库本身也有不同的损失和调度选择。当前最接近V6单步Huber/clip目标的主线参考是SB3，CleanRL本轮不作为训练框架。

## 5. 训练预算必须明确：8遍与2,782,159次更新不是同一个约束

**用户已明确选择固定8遍：每遍35条Train各执行一次，实际总步数和更新数允许随策略变化。** 该口径在实际运行报告中记录为`budget_mode='eight_passes'`，不再额外执行一个固定更新预算实验。实际调度入口见[`training.run`](../src/v6_sb3/training.py)和[`TrainCurriculum`](../src/v6_sb3/curriculum.py)。

V6每条航段执行到完成或控制失败。换策略以后，同一航段的执行长度会变。因而“同8遍、每遍35条”和“恰好同2,782,159条正式转移/Adam”一般不能同时满足。不能靠补跑更多航段、重复旧经验更新或截短航段后仍把它称作同8遍，来隐藏这个差异。

|口径|停止条件|能严格匹配什么|必须披露什么|
|---|---|---|---|
|固定有效更新预算（仅解释另一种口径，本轮不执行）|预填充50,000之后，实际正式Adam达到2,782,159次停止|正式环境转移数和Adam次数；每1000次Adam的Target频率|实际遍历数/航段数可能不是8×35；最后可能在航段中途。checkpoint阶段不应冒称完整Train pass；阶段内Train统计须标明前缀/已完成样本|
|**固定8次Train遍历（本轮用户选择）**|8遍，每遍相同35条各访问一次，保持相同shuffle及按航段epsilon公式|数据访问计划、遍历数、探索调度的任务进度|总环境步与Adam由策略决定，通常不等于2,782,159；不能声称step-exact比较|

每轮报告同时记录实际环境步、Adam、Target计数和Replay状态，结果注明为pass-matched对照。不能把两种口径混用，也不能因为SB3执行得更长或更短就未经授权增加/减少遍历次数。正式训练仍由用户在PyCharm启动，本报告未执行该训练。

以下历史累计更新点可帮助理解V6各轮计算量。若新训练刚好跨过这些计数，可仅记录已有训练统计作里程碑对照；本轮不为凑这些计数额外训练或额外执行Validation：

|V6历史轮次|累计正式更新|该轮正式步|V6 Train完成|V6 Greedy Validation完成|
|---|---:|---:|---:|---:|
|1|274126|274126|20/35|7/8|
|2|632575|358449|28/35|7/8|
|3|1016154|383579|29/35|7/8|
|4|1421304|405150|31/35|4/8|
|5|1779547|358243|28/35|5/8|
|6|2146754|367207|28/35|3/8|
|7|2470152|323398|25/35|6/8|
|8|2782159|312007|25/35|3/8|

V6数字来自最终run02已提交报告，排除了曾按用户要求丢弃的第5轮部分尝试。不能把日志中的被丢弃计算再次加入最终状态分支计数。

## 6. 接入SB3时必须核对的三个时序点

**最后一步不能丢。** 在`OffPolicyAlgorithm.collect_rollouts:570–582`中，环境已经step且num_timesteps已增加，随后先调用callback.on_step，最后才_store_transition；`learn:348–354`之后才train。如果callback在预算最后一秒直接返回False，原生代码会提前退出，最后真实转移未入Replay也未Adam。停止/保存要在该转移入Replay且应有的Adam完成之后，或由learn的精确步预算结束，不能仅看计数表面增加。

**VecEnv自动reset不能错位。** `_store_transition:449–513`会使用info的terminal_observation替换自动reset的新观察。自定义Replay的next mask必须来自同一条真实转移的info，不能调用此时已reset环境的mask；终止失败的next mask可为空，但不bootstrap。

**恢复必须包括物理执行位置和随机状态。** 实现会在完整遍历边界保存；Ctrl+C还可请求在最近完整step后暂停。`TrainingContext.save_checkpoint`除SB3模型zip和Replay外，保存Python/NumPy/Torch RNG、课程顺序/计数、当前物理状态的动作前缀、Gym/action-space RNG及诊断状态，并最后写完整性标记。`resume_context`和`TrainCurriculum.restore_curriculum`用已保存动作重建原环境位置，不入Replay、不更新网络，随后恢复训练RNG。不能用“从下一条航段重开”代替这一恢复语义；完整runner中断/恢复效果仍应以实际运行记录为证，9项定向检查没有冒充一轮正式训练。

## 7. 新实现的源码接入位置

- [`environment.py`](../src/v6_sb3/environment.py)：Gymnasium观察/动作接口通过一个可关闭worker和容量1队列调用原`v4.v6_training.execute`；原物理、账本、奖励和失败处理没有另写替代模型。`step`只返回真实闭合转移，下一状态无动作时先等待原失败终止处理。
- [`masked_dqn.py`](../src/v6_sb3/masked_dqn.py)：`MaskedDQN`继承官方`DQN`，`MaskedReplayBuffer`继承官方`ReplayBuffer`。本任务没有覆盖SB3的`OffPolicyAlgorithm.learn`或`collect_rollouts`；必要扩展集中在动作采样、TD合法max、Replay mask与实际Adam时钟的Target复制。
- [`curriculum.py`](../src/v6_sb3/curriculum.py)：保持原warmup航段顺序生成规则、每pass的`Random(42+pass)`顺序及原按35条航段进度的epsilon公式。预算结束后不再执行下一动作。
- [`training.py`](../src/v6_sb3/training.py)：`_CollectionCallback._on_step`恒返回True，避免丢失已执行的最后一步；完成航段、轮末评价/保存和暂停在实际Adam完成后的`after_update`处理。warmup阶段只收集不训练，50,000条后重新从正式第一遍起点开始。
- [`reporting.py`](../src/v6_sb3/reporting.py)：Greedy Validation仅推断，不入Replay、不更新；完整逐秒CSV保留Q13/合法排序，终端不逐步打印。模型的SB3 logger只写CSV，关闭进度条；每遍结束打印一条完成率/FC变载摘要。关闭频繁输出不能单独证明已排除其他内存问题，运行时按轮保存进程内存记录供核对。

## 8. Tianshou留作后续PER参考

实际下载来源：[ReplayBuffer](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/buffer/buffer_base.py)与[PrioritizedReplayBuffer](https://github.com/thu-ml/tianshou/blob/f2402056b03acafeedf2518a5088db55add004ab/tianshou/data/buffer/prio.py)。当前文件名是`buffer_base.py`，没有按旧记忆假设它叫`base.py`。

`ReplayBuffer.add:420`和`sample_indices:503`提供基础环形存储/抽样；`PrioritizedReplayBuffer:12`使用SegmentTree，`:63`按优先级质量抽样，`:69`计算重要性权重，`:81`以绝对新权重加epsilon更新priority并施加alpha，`:109`返回带权batch。PER因此会同时改变抽样分布与损失权重，不是只在日志里记录TD误差。

本轮没有安装Tianshou、没有使用其Replay或PER。是否采用PER只能在当前Uniform SB3基线真实完成且可比后另行决定；失败终止拟合不足本身不是已证明PER有效的因果证据。

## 9. 当前能给出的结论

已读源码支持：V6确实有普通DQN的Target最大化、实际动作gather、Huber、backward、梯度裁剪、Adam和纯硬Target更新；没有发现“仅定义网络但不训练”的证据，也没有因SB3有不同默认设置就把V6判作bug。V6是否在控制上有效，与训练闭环是否真实存在是不同问题。

SB3原生DQN不能直接公平替代当前船舶控制器。本次已实现必要的动作mask、终止/VecEnv时序、Target计数和探索调度适配，并完成上述9项固定样本检查；完整正式训练仍未执行。训练将由用户在PyCharm终端启动，之后才能由真实结果判断SB3是否改善完成率、Q排序或SOC维持。
