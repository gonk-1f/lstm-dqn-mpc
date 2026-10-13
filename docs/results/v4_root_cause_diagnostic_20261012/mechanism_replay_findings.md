# 当前 v4 DDQN 训练链与终止语义只读诊断

核查 HEAD `52e9232916b4e64883cf6b09d5697fb68f6446cd`、正式 100 轮 report、既有数值 Replay 缓存及 Train 标签/边界 metadata。没有加载大型 checkpoint，没有运行优化器、反向传播、环境回放、pytest 或正式训练，没有读取 Test payload。缓存 q/y 来自先前审计最终冻结 Online/Target 网络的重计算，source 字段来自该审计的来源重建；本次不把它们当作 checkpoint 内原生历史字段。

## 确定事实

### 实际训练路径没有发现 DDQN、n8、scale 或终止 bootstrap 公式错误

- `src/v4/feedback_study.py:63` 调用 `run_monitored_training`；正式入口锁定 gamma=1、n=8、scale=.001、batch64、replay32、quota2、tau=.001、Adam=.001、7-128-128-61。reward redistribution 关闭。
- `src/v4/monitored_training.py:401` 构建 agent；`:619` 先执行完整样本；成功路径 `:665` 插入 n-step 轨迹后，`:669` 授予入池信用，再 `:671` 集中更新。失败路径 `:651` 插入经 failure view 标记的真实执行轨迹，再 `:660`/`:662` 授信/更新。replay32 不表示每32个实时控制步立即更新。
- `src/v4/dqn.py:54` 的 ReplayItem 保存状态、动作、已缩放 n-step 奖励、后继状态、done、mask、m 和结果标签，**没有历史 TD target**。`:299` 每次 learn 都用当时的 Online/Target 重新构造目标；本报告缓存 y 则用最终快照重算，不能冒充当时优化目标。
- `src/v4/dqn.py:234` 按真实执行奖励累加 m≤8；`:173` 在入池边界缩放一次；`:43` 用合法动作中的 Online argmax、`:49` 用 Target 估值且 done 禁止 bootstrap。正式每次 Adam 后 `:314`/`:444` 更新 Target，累计硬复制1次、soft update20953次。
- 最终 300000 条池内，done 等于空后继掩码、done 等于非空终止原因。全部7096条终止记录 y 逐条精确等于 `float32(stored_reward)`，说明此处没有残留 bootstrap。推理数字与状态奖励均有限。
- `src/v4/failure_replay.py:58` 仅接纳 genuine no-feasible-action；`:85` 只修改最后一个真实动作的训练 reward，追加一次失败惩罚和终止标签，不创建失败行上的虚拟动作、不修改真实账本。全程实际576个SOC失败事件，累计4608个失败终止窗口=576×8。当前1760个失败终止窗口=220个样本/轮次失败事件×8；均不能当作独立物理失败数。

### 5336 条正常终止残差是真实当前拟合缺口

5336条=667个成功样本/轮次事件×8；m=1..8各667。全部窗口到源样本最后一个 ONBOARD 决策，且没有 UNKNOWN 来源。并不是中途 SHORE 被误写为 done。

|最终池类别|条数|当前 TD MAE|abs(TD)>1|abs(TD)>5|
|---|---:|---:|---:|---:|
|全部|300000|0.354917|19020|4615|
|正常终止窗口|5336|2.388937|5336|0|
|失败终止窗口|1760|0.258196|29|6|
|失败轨迹非终止前缀|35224|0.748420|4669|2259|

正常终止 y 平均-0.083340，Online Q平均-2.472277；误差全部为正，绝对值1.052128–3.662345。同状态动作的 Target网络Q平均-2.346447；两网络都没有贴近短末端回报。由于正常终止y不含Target值，不能将该2.389误差直接解释成tau引起的bootstrap滞后。

正常末端状态平均SOC=.604778、负载21.275 kW、动作117.446 kW。池中负载≤80 kW共82106条，正常终止只占5336条（6.499%）；其余76770条平均Q=-2.932555、y=-2.972759。此为粗分布证据，不能代替逐状态的真实矛盾证明。先前审计无完全相同float32输入/动作而终止不同的组，但发现131个SOC差≤.001、各功率特征差≤1 kW的近邻，支持阶段不可辨识的疑点。

### 实际历史批次计数已经保存，逐条历史计数没有保存

bootstrap303次+正式20650次=20953次优化；实际采样位置1340992。以下由正式 report 的 bootstrap 与100轮 `td_statistics.sample_outcomes` 求和，不是理论抽样率。

|历史实际批次类别|采样位置|
|---|---:|
|成功轨迹记录（含正常终止与非终止）|1112913|
|失败轨迹记录（含前缀与失败终止）|228079|
|其中失败终止|41906|
|非失败终止 ordinary|1299086|

失败终止配额实际占3.125%。没有正常终止独立的历史采样计数，也没有每条Replay的采样次数或历史y序列。全程采样位置/插入数约2.0000，只是总量比例，**不能说每条记录实际更新两次**。全程实际0次梯度范数裁剪触发；最终一轮216次、13824个采样位置，其中failure terminal432，success11820，failure2004。

### UNKNOWN 处理导致九个Train样本未贡献经济Replay

`src/v4/control.py:282` 将UNKNOWN记为data_truncation，`:423`禁止把此前最后动作标completed。`src/v4/monitored_training.py:642` 调用 `remember_completed_prefix`，后者 `src/v4/dqn.py:273` 只保留真正done的前段。当前跨SHORE仍连接样本，UNKNOWN之前没有样本done，因此整条未闭合前缀丢弃。

含UNKNOWN的九个Train为 `_007/_017/_029/_033/_034/_037/_041/_042/_046`。全程未知前缀物理失败0次，data_truncation909次=九样本×101次（含bootstrap）。最终Replay来自上述样本的记录为0条。正式100轮丢弃515400个已执行决策；bootstrap另丢5154个；合计520554。单轮可观测前缀5154个ONBOARD决策，UNKNOWN后的6376个已标ONBOARD行从未进入此训练执行路径。

这是当前“必须完整闭合才能入池”设计的确定覆盖损失，不是伪造done或失败惩罚错误。`docs/v4_mode_labels_v3.md:15` 仍称前缀进入Replay，与当前实现/数据相矛盾，属于需要修正文档的确定错误。

## cutoff、真实任务终止和时限截断需要分开

`src/v2/data/zero_boundary_dataset.py:252` 在一个原始parent时间窗内，取持续正功率区间后的首个死区零点或正负交叉零点；Train30个末端19个OBSERVED_DEADBAND、11个CONSTRUCTED_CROSSING。它没有读取任务完成标志或港口接入事件。

`src/v2/data/formal_training_dataset.py:326` 校验末端零负载，`:345` 只取可执行完整30 s区间，尾部未执行余量0–26 s；零点本身不作为最后决策状态输入。`src/v4/control.py:390` 在数组尾加模型终端SOC补能账，`:423`标completed。末端最后监督行的历史FC均为0，负载仍为1.022–26.248 kW；`_021`最后监督行历史航速1.9 kn，说明零功率边界不能单独当作真实任务已完成的证明。

- 若研究目标明确定义为“这个有限样本的运行奖励加末端等效补能”，当前终止bootstrap为0与该有限任务目标一致。此时需检验7维状态是否能区分末端与中途；`build_state`（`:189`）没有任务进度/已知计划时域。
- 若目标是连续航行任务，而文件只因日志窗口/人为裁切结束，cutoff应当与真实terminated区别。但当前没有下一个真实负载、可行动作掩码及跨样本状态连续性，不能简单把done改False并用复制末端状态bootstrap；也不能跨独立样本连接。
- UNKNOWN是真实数据缺口截断，当前没有惩罚/虚拟补能/跨缺口连接；并非Gym式max_steps时限。源文件end零点是否是真实任务终点仍待业务/采集证据确认，**不能现阶段确定全部667事件的done语义错误**。

## 高概率机制与尚未证明的假设

较强疑点是终止样本占池少、低负载中途值与末端值相近且阶段未显式可观测，使共享MLP主要拟合占多数的中途值。Smooth L1在abs(TD)>1时对单Q的梯度饱和，也不会因终止误差2.4就得到2.4倍力度。此疑点已有分布和近邻证据，仍不等于证明无法同时拟合或加时间特征一定有效。

失败终止已有总体拟合，而早期失败前缀存在高残差；gamma=1和n8使风险进入当前y，不能保证真实采样到并及时拟合。tau、学习率、函数容量、行为策略n-step的off-policy偏差以及采样不足各自仍需要对照；现有终点快照不能给它们排出唯一因果次序。

## 针对函数的最小后续建议（本次未实施）

1. `DirectPowerDDQN._sample_replay_batch/learn`：增加关键Replay ID或局部区域的实际采样计数，并单列normal terminal的batch计数/TD统计。先测真实更新覆盖，避免用期望率代替历史事实。
2. `FormalEpisode`、`replay_episode`：明确保存原始cutoff/研究样本终点/真实任务终点元数据；业务确认连续任务之前保留现行有限样本目标。若确有在线可知任务计划，仅做任务进度单变量设计，不使用未来负载或未知样本剩余长度。
3. `remember_completed_prefix/remember_trajectory`与训练UNKNOWN分支：可单独设计“已验证前缀内完整n8窗口”的入池对照。只保留后继仍已观测且mask非空的窗口，维持done=False；靠UNKNOWN且后继未知的窗口舍弃；前缀结果标未知，不称成功，不虚构末端补能。当前授权范围没有更改此契约。
4. 正常末端/中途配对与 `_015/97` 三条已存高残差tuple：在独立模型副本固定当前目标、有限预算拟合，以区分目标易拟合但更新覆盖不足与相近状态目标干扰。不能给未执行的600 kW动作虚构监督目标；本次未运行此检查。
5. `annotate_evaluation/fully_completed/BestCheckpoint`：现行资格按固定21个完整Train和7个完整Validation，且全部UNKNOWN前缀无物理失败、未知截断数吻合。每轮都评估Validation；最后一轮Train13/21、Validation5/7，有8/2次物理失败，完整成本均null。100轮合格数0，best checkpoint为空，不能拿partial cost排序选模。保留这些资格事实并修正文档即可，没有发现选模绕过失败或打开Test的证据。

## 独立固定最终池600批抽样诊断

直接调用当前 `DirectPowerDDQN._sample_replay_batch(64)`，以 `object.__new__` 跳过构造器；对象仅含quota2、`random.Random(42)`与 `(FIFO_ID, 数值array_slot)` 两个池。没有创建ReplayItem对象、网络或优化器，没有更新Q或运行环境。最终池固定不变，池内部按FIFO ID排序。历史swap-delete顺序与当时RNG轨迹未知，**这不是原训练逐条采样次数的重建**；下列数字只是在这个独立诊断中真正抽到的次数。

600批共38400位置，抽到35901个不同记录，264099条没有抽到。其中普通池37200位置；失败终止1200位置；正常终止656位置，仅涉及615条不同正常终止记录，其余4721条正常终止未抽到。

|分组|独立诊断实际抽样位置|
|---|---:|
|非终止|36544|
|正常终止|656|
|失败终止|1200|
|失败非终止前缀|4452|
|SOC [.2,.3)|1976|
|SOC [.3,.4)|1896|
|SOC [.4,.5)|5365|
|SOC [.5,.6)|10925|
|SOC [.6,.7)|11705|
|SOC [.7,.8]|6533|
|当前负载 [0,80) kW|10315|
|当前负载 [80,300) kW|9086|
|当前负载 [300,600) kW|14626|
|当前负载 [600,900) kW|3672|
|当前负载 ≥900 kW|701|
|样本位置 [0,.25)|12780|
|样本位置 [.25,.5)|9142|
|样本位置 [.5,.75)|8305|
|样本位置 [.75,1]|8173|
|0 kW动作|101|
|160 kW动作|350|
|600 kW动作|439|

同源 `_015/97`、动作160 kW三条记录：ID493205/第76轮、ID516139/第79轮、ID584126/第88轮，独立诊断均抽0次。17条SOC±.03、六个功率特征各±30 kW邻域共抽2次，抽到的为第76轮源行100和101，各1次。三条同源记录中只有第88轮一条属于这个17条邻域；第76轮FC历史特征最大相差110 kW，第79轮SOC差.03850且FC历史最大相差40 kW。附件因此保存19条并集记录而不把两个集合混为一谈。

该诊断支持关键稀疏区域在当前采样预算下可能得到很少更新的机制，但**不能证明三条记录原历史训练抽0次，也不能证明增大预算单独就能解决共享函数拟合问题**。

数值附件：`mechanism_replay_summary.json`、`mechanism_replay_by_sample.csv`、`mechanism_replay_normal_terminal_by_sample.csv`、`mechanism_replay_train_endpoint_metadata.csv`、`mechanism_replay_fixed600_sampling.json`、`mechanism_replay_fixed600_groups.csv`、`mechanism_replay_fixed600_key015_draws.csv`。脚本位于TEMP `v4_mechanism_replay_diagnostic_20261012.py`、`v4_mechanism_replay_fixed600_20261012.py`；主脚本以exclusive-create写出，避免覆盖历史产物。
