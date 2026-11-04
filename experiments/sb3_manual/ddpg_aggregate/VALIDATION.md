# 独立 DDPG Validation8

入口为 `experiments.sb3_manual.ddpg_aggregate.validate`，复用原有
`AggregateShipDDPG-v0` 和 `AggregateDDPGEnv`，不调用训练入口，也不修改
`env.py`、Zoo YAML 或原有 `evaluate.py`。

## 在 PyCharm 点击运行

请打开实际 DDPG 工作树，而不是外层同名文件夹：

```text
C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline
```

共享配置位于项目根的 `.run/DDPG Validation8.run.xml`。打开此工作树后，在运行配置中
选择 **DDPG Validation8**，点击绿色运行按钮。该配置沿用项目解释器。

若手动创建 Python Run Configuration，填写：

| 项目 | 内容 |
| --- | --- |
| Run kind | Module name |
| Module name | `experiments.sb3_manual.ddpg_aggregate.validate` |
| Working directory | 上述实际 DDPG 工作树根目录 |
| Parameters | `--model-path outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip --device cpu` |
| Python interpreter | `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\.venv\Scripts\python.exe` |

无需设置 `PYTHONPATH`，也无需启用 “Add content roots/source roots to PYTHONPATH”。
入口自行定位当前工作树的 `src` 依赖。使用 Module name，不要把包内的 `validate.py`
填进 Script path。

## PyCharm Terminal / PowerShell

在上述 DDPG 项目根目录执行：

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --model-path "outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --device cpu
```

验证其他模型时，只替换 `--model-path`，相对路径以当前项目根目录为准，也可填写绝对路径。
无需复制模型、加载磁盘 Replay Buffer 或先运行训练。

默认数据根为当前项目的 `data/processed/operating_dataset_v6_1s`；默认输出根为
`outputs/ddpg_fixed_parent_005_validation_economic_v3`。可分别通过 `--data-root`、`--output-root` 指定。
默认 `--seed 42`、`--episodes 8`、`--torch-threads 1`。

## 验证范围与确定性

验证使用 `DDPG.load()` 和 `model.predict(obs, deterministic=True)`，关闭加载副本的
探索噪声。不调用 `learn()`，不做 Actor/Critic 更新或目标网络同步，不修改模型文件和
历史 Replay Buffer。程序检查推理前后参数及更新状态是否保持一致。

固定按照现有数据加载器的航段编号顺序执行全部八条 Validation：

1. `parent_002_seg01`
2. `parent_011_seg01`
3. `parent_016_seg01`
4. `parent_020_seg01`
5. `parent_033_seg01`
6. `parent_040_seg01`
7. `parent_044_seg01`
8. `parent_045_seg01`

每轮从原始航段起点及原有初始状态开始，直到完整结束或真实物理失败。
不会切换到 Train，也不读取 Test。物理失败如实计为未完成。

当前正式模型训练未使用 VecNormalize，因此默认不做额外观测归一化。验证确实使用
VecNormalize 训练的模型时，可通过 `--vecnormalize-path` 指定其对应统计文件。未指定
该参数时，程序依次检查模型旁的 `AggregateShipDDPG-v0/vecnormalize.pkl` 和
`vecnormalize.pkl`，发现后自动加载；不要在模型旁放置无关实验的统计文件。验证时设置
`training=False`、`norm_reward=False`，JSON 记录实际加载路径并检查统计量不变。

## 输出与指标口径

每次运行在输出根下建立独立时间戳目录，具体目录在控制台显示，不覆盖以前的验证。
至少生成：

| 文件 | 内容 |
| --- | --- |
| `validation_summary.csv` | 每航段累计奖励、执行长度、完成状态、覆盖率、Ramp RMS、氢耗、电池能量、经济成本及初末 SOC |
| `validation_step_data.csv` | 全部航段逐步动作、负载、实际 FC/LB 功率、SOC、奖励、终止信息及可用经济指标 |
| `validation_metrics.json` | 总体验证指标、模型身份、运行设置与推理不变性检查结果 |
| `validation_reward_curve.png` | 八条验证航段的累计奖励及数值标注，不是训练学习曲线 |
| `validation_power_curves.png` | 每航段负载、FC 功率、LB 功率，便于检查低负载充电行为 |
| `validation_soc_curves.png` | 每航段实际执行过程的 SOC |

电池功率采用环境的**母线侧**符号：正值表示放电供能，负值表示充电吸能。
电池充、放电能量分别累计真实执行区间的母线侧功率，不与电芯侧能量混用。

`no_feasible_physical_power` 失败调用没有执行新的物理一秒：逐步记录中
`step_executed=False`、结束时间为空，不伪造电池功率或新增执行动作；该调用的失败奖励
仍计入 Episode Reward。失败时 info 提供的 FC 功率表示失败前已有的实际 FC 状态。

经济成本沿用现有聚合燃料电池/电池经济模型，属于模型定义下的船上运行成本，
不等于逐堆实测退化成本，也不包含航段结束后的补电成本。提前失败的前缀成本不能当成
完成整条航段的总成本，与完整航段比较时必须先看完成状态和覆盖率。
环境或现有模型无法提供的指标标为 unavailable，不填造出来的零值。

## 有界检查

仅运行第一条 Validation 的前 100 次环境调用：

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --model-path "outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --episodes 1 --max-steps-per-episode 100 --device cpu
```

这是短程接口检查，不能作为 Validation8 结果；输出会保留未完成状态。省略
`--max-steps-per-episode` 且使用默认八轮，才是全部八条航段的正式验证流程。

查看完整参数：

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --help
```

## 当前经济奖励与模型身份

奖励版本为 `ddpg_economic_normalized_tanh_v3`。正常步仅计四项经济成本：
实际氢耗、FC退化、电池退化和有符号电池等效氢耗，`C=四项成本之和`。
正常奖励为 `tanh(C_ref/C)`；`C=0` 时为1；负总成本或非有限值明确报错。
原有真实物理不可行失败奖励保持 -10000，失败调用不制造物理经济成本。
不再包含独立SOC目标偏差成本；电池退化模型内部的SOC应力函数仍保留。

`C_ref` 固定读取 `economic_reward_config.json` 的 `c_ref`，由现有Train35基准轨迹
按新退化定义重新核算四项单步成本后取中位数，未使用Validation/Test标定。
FC累计电压损失报废阈值为142500 μV；保留四工况系数、设备价和50%更换比例。
80%高负载分界为项目建模假设，不是Zhou原文参数。电池保持F(SOC)、G(I)、
加权Ah吞吐和15000倍容量归一化，退化使用与物理SOC更新一致的实际电流。

开启FC时采用实际功率点效率，关机时采用现有曲线正功率范围计算出的峰值效率。
电池等效氢耗保持放电正、充电负的符号，不重复计算效率损失。
`total_cost_cny`和`reward_total_cost_cny`均表示同一个四项经济成本总额；
逐秒CSV另记`battery_current_a`，不再导出独立SOC惩罚字段。

仅加载完整奖励与标定身份一致的新版本模型，旧线性、旧tanh及无身份模型均拒绝。
新训练默认输出为 `outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3`；
模型需在后续从头训练后才存在。根目录对应的运行脚本可直接在PyCharm绿色运行，
无需Module配置；也可使用上述模块入口的`--model-path`选择相同版本的其他checkpoint。
