# V4 优先经验采样独立候选及验证

## 1. 本次交付与隔离

候选位于 `.worktrees/v4-per-prep`，基于正在运行的 B 组当前源码快照，HEAD `a40afa2c8fd0a967d40d8aa30472b228232fbc0e`。原工作树仍是 `feat/direct-power-mlp-ddqn`。候选使用 detached worktree，不切换原分支，也不修改原工作树。

159 个复制源文件的 SHA256 在 [快照](v4_per_base_snapshot.json)，完成时与原工作树重新逐一核对一致。原训练 PID32160 仍在运行，最后一次检查的完整 CSV 轮次为78，经济Q累计更新28599次。没有停止、恢复或修改该训练，没有读取其续训 checkpoint；正式数据和历史输出未修改。仅读取旧审计导出的小型 Train 经验 CSV，不读取 Test。

当前运行的 B 组**不会自动采用本候选**。候选尚未提交、推送或合并；正式 PER 效果需要以后在独立实验中验证。

## 2. 文献与算法边界

采用 [Schaul et al., Prioritized Experience Replay](https://arxiv.org/html/1511.05952v4)，§3.3、§3.4、Algorithm1、附录B.2.1及Table3：按绝对 TD 误差进行随机比例采样，并用重要性采样权重修正。参数与训练奖励尺度无关的设备阈值没有新增。

| 项目 | 候选设置 | 依据与限制 |
|---|---|---|
| 默认采样 | `uniform` | 继续使用原固定失败配额的均匀抽样；候选默认关闭 |
| 显式候选 | `--replay-sampling prioritized` | 只改变更新所抽取的经验与逐样本 IS 权重 |
| alpha | 0.6，可配置 | 原论文比例 PER 的候选值；不是本船已验证最优值 |
| beta_IS | 1，可配置 | 原论文§3.4的完全校正端点；本次固定，不新增退火调度 |
| epsilon | 1e-6，可配置 | 防止零优先级的数值正数；不是 SOC 阈值或经济费用 |
| Batch/更新频率 | 64/replay32 | 保持原样，不同时增加梯度预算 |
| 失败终止配额 | 2 | 既有机制保留；不新增 SOC/FC 风险类别规则 |

论文的 Atari 实验还调整过学习率。本次没有照搬该调整，不声称复现论文全部设置；候选仍保持当前 Adam0.001。beta_IS=1的共同归一化也会改变梯度尺度，Adam历史矩、epsilon和梯度裁剪意味着实际更新不与原采样完全等价。

### 2.1 普通经验与原配额的结合

失败池按既有终止标签识别，仍均匀无放回。普通池包括正常末端、失败非终止前缀及 UNKNOWN 前有效窗口；使用有放回比例 PER，不按 SOC、FC动作或航段编号人为分类。

令 `B` 为 batch、`N_f/N_o` 为两个实时池大小，失败条数仍为：

\[
k_f=\max(\min(q,N_f,B),B-N_o),\qquad k_o=B-k_f.
\]

普通池：

\[
p_i=(|\delta_i|+\epsilon)^{\alpha},\quad P_i=p_i/\sum_jp_j,
\quad M=\max\{1,(N_oP_{\min})^{-\beta}\}.
\]

普通权重为 `(N_o P_i)^(-beta)/M`，失败权重也为 `1/M`；不能对两个类别分别归一化。beta=1时，期望损失为：

\[
\mathbb E[L]=\frac{k_f\overline\ell_f+k_o\overline\ell_o}{BM}.
\]

这是原固定配额目标的共同缩放，不是全 Replay 均匀目标。配额0时改为全池 PER/IS。普通池为空时 M=1。普通池不足时保留原失败配额回退，即使 PER 普通抽样允许重复也不减少该失败数。

逐样本损失为 `mean(w_i * SmoothL1(Q_i,y_i,reduction='none'))`。优先级取更新前、未乘 IS 的绝对 TD；重复 ID 聚合最大值；新经验以历史最大优先级进入，随后用实际 TD 更新。优先级可能陈旧，较大残差也可能来自状态歧义或噪声，因此“抽得更多”不等于“策略变好”。

### 2.2 未改变的内容

7–128–128–61、Adam0.001、gamma1、n8、tau0.001、Batch64、Replay300000、replay32、scale0.001、SOC参考0.6/beta200、失败惩罚、epsilon、四项经济成本、模式v3、UNKNOWN语义及评估规则全部保持原样。

`select_power`、合法动作掩码、Double DQN目标、n-step累计、终止处理均没有修改；不添加人工门控、强制FC出力、动作平滑、Q排名覆盖或新奖励。

## 3. 实际文件

- `src/v4/prioritized_replay.py`：稳定ID、sum/min tree；FIFO同步淘汰；抽样和优先级更新 O(log容量)，不逐次扫描 Replay。
- `src/v4/dqn.py`：显式候选采样、逐样本 Huber/IS、优先级更新、恢复及诊断。报告同时保存加权优化loss和未加权loss/TD；重复抽样计数与实际类别抽样沿用原统计。
- `src/v4/feedback_study.py`、`monitored_training.py`：将四个采样配置传到实际 Agent、报告、权重元数据及续训核对，不重新设计训练系统。
- 三个定向测试文件：`test_v4_prioritized_replay.py`、`test_v4_per_dqn.py`、`test_v4_per_training_entry.py`。
- `scripts/v4_per_frozen_probe.py`：最多每组300次更新的独立冻结标签比较，不调用正式训练，不打开数据加载器。

PER Agent checkpoint使用格式3，Uniform保持格式2；保存真实优先级、历史最大值、live ID映射和 RNG。采样参数或优先级缺失、不匹配时拒绝恢复，旧代码不能把PER checkpoint静默按Uniform读取。原 B 组 checkpoint不能用于本候选续训。

## 4. 定向验证

134项参数化检查/必要回归通过（最终9.64s），8个实现/定向测试/探针文件静态编译通过，相关diff及新文件空白检查通过。覆盖：FIFO旧引用、重复ID、可复现采样、类别不足回退、quota0、逐样本IS配对、Huber及TD目标、优先级/RNG恢复、旧配置默认不变、正式CLI→包装器→Agent参数和metadata、恢复不匹配、一次Adam对应一次soft target，以及Greedy只读性。没有运行全套仓库测试或正式训练。

另直接加载原工作树的 `dqn.py` 作为只读参考，对相同合成经验执行8次Uniform更新：每批RNG、loss、Online和Target参数逐位一致。对9个TD/策略/n-step/Target函数的AST比较也完全一致。该参考检查不写入原工作树或原checkpoint。

## 5. 有限冻结标签比较

结果：[frozen_probe.json](results/v4_per_preparation/frozen_probe.json)。使用旧审计CSV的538条不同Train Replay记录，按（样本ID、来源轮次）分组划分432条训练、106条留出。导出CSV的终止枚举与 `experience_failed` 联合核对：1为本导出中的失败终止，3为正常终止；不能猜枚举含义。

两组均从同seed42的全新网络开始，使用相同固定标签、Adam0.001、Batch64和300次更新；每组19200个抽样位置。标签是旧A最后Online/Target快照的导出目标，并非新训练的实时TD目标。Target更新0、正式环境步数0。没有加载300000条原Replay或大型checkpoint。

| 指标 | Uniform+原配额2 | PER+原配额2 |
|---|---:|---:|
| `_015`/97、Replay516139抽取次数 | 45 | 186 |
| 同一训练经验最终绝对误差 | 8.1380 | 8.2560 |
| 留出近邻584126最终绝对误差 | 8.3150 | 8.4671 |
| 全部训练经验MAE | 0.7719 | 0.8031 |
| 全部留出经验MAE | 1.0910 | 1.1163 |
| 正常终止抽样位置 | 1173 | 275 |
| 失败非终止前缀抽样位置 | 6402 | 8232 |
| 失败终止抽样位置 | 600 | 600 |

单位均为原实验scale0.001后的训练奖励单位，不能当作实际费用。关键记录抽样增加约4.13倍，但其误差和整体留出误差没有改善；因此本次只证明采样机制生效，**没有证据证明PER已经解决拟合干扰或提高闭环完成率**。

正常终止在全新初始化下标签接近0、残差较小，PER因此少抽；这不能代表旧最终网络正常终止误差2.389情况下的抽样表现。本探针是被选择的邻域数据，不代表原30万条池分布；同一样本的不同轮次仍可能分属训练/留出，留出结果不等同于新航段泛化。探针输出的61动作Q排序仅属于这些新副本，不能冒充历史动作排名。

## 6. 后续边界

先让当前 B 组完成，再按相同完成样本及成本口径分析。PER当前只作为已经接好、默认关闭的候选；不与新的SOC权重、网络或更新预算一起变更。以后若批准独立正式对照，使用空输出、新网络和空Replay，显式指定：

```text
--replay-sampling prioritized --per-alpha 0.6 --per-beta 1 --per-epsilon 0.000001
```

其他参数不变，不能从原Bcheckpoint切换采样模式续训。判断必须同时看完整可评价Train/Validation完成率、相同完成集合的四项成本、FC变载/启停、SOC分布、关键经验抽样与未加权TD、留出拟合及多seed稳定性；仅TD或抽样量变化不够。

如果关键经验多次更新后仍存在较大残差，下一优先项是区分共享表示干扰与相近状态的不同后续目标；本次不靠调大SOC权重或人工控制规则遮盖这些问题。
