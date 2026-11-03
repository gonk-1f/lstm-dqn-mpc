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
| Parameters | `--model-path outputs/ddpg_fixed_parent_005_seg01_500ep_linear_v2/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip --device cpu` |
| Python interpreter | `C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline\.venv\Scripts\python.exe` |

无需设置 `PYTHONPATH`，也无需启用 “Add content roots/source roots to PYTHONPATH”。
入口自行定位当前工作树的 `src` 依赖。使用 Module name，不要把包内的 `validate.py`
填进 Script path。

## PyCharm Terminal / PowerShell

在上述 DDPG 项目根目录执行：

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --model-path "outputs/ddpg_fixed_parent_005_seg01_500ep_linear_v2/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --device cpu
```

验证其他模型时，只替换 `--model-path`，相对路径以当前项目根目录为准，也可填写绝对路径。
无需复制模型、加载磁盘 Replay Buffer 或先运行训练。

默认数据根为当前项目的 `data/processed/operating_dataset_v6_1s`；默认输出根为
`outputs/ddpg_fixed_parent_005_validation_linear_v2`。可分别通过 `--data-root`、`--output-root` 指定。
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
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --model-path "outputs/ddpg_fixed_parent_005_seg01_500ep_linear_v2/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --episodes 1 --max-steps-per-episode 100 --device cpu
```

这是短程接口检查，不能作为 Validation8 结果；输出会保留未完成状态。省略
`--max-steps-per-episode` 且使用默认八轮，才是全部八条航段的正式验证流程。

查看完整参数：

```powershell
& ".\.venv\Scripts\python.exe" -m experiments.sb3_manual.ddpg_aggregate.validate --help
```


## 经济奖励版本更新

当前默认采用 `ddpg_economic_linear_v2`：四项经济成本（实际氢耗、FC退化、电池退化、实时电池等效氢耗）加固定 SOC 平方惩罚后计算 `reward=-r`；物理失败固定为 -10000，不再计算其他失败补偿。
固定 `C_typ=0.17429534153424198` 为 Train35 实际执行步的四项经济成本中位数；`lambda_soc=8.0` 继续保持当前固定权重；C_typ仅保留为历史标定元数据，不参与线性奖励缩放。未使用 Validation/Test 标定，训练和验证期间均不重新计算权重。
FC关机参考效率由现有正功率曲线取峰值，开启时使用实际功率点效率；每秒记录两种效率。`total_cost_cny` 为四项经济成本，`reward_total_cost_cny` 另加SOC罚项；无额外终端补能计价。
新奖励模型默认路径需在后续从头训练完成后才会存在。不同奖励身份或无身份的模型明确拒绝加载评估。根目录对应的直接运行脚本无需Module配置。

仅允许加载奖励身份完全一致的线性奖励模型；旧tanh模型及无身份历史模型拒绝评估。当前模型必须先在新目录从头训练，历史结果保留。
