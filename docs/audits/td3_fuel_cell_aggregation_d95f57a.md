# TD3 燃料电池单曲线聚合审计

基线为 `feat/td3-economic-ems-v2` 的 `d95f57a92fd7116cc6d6679b5c54f5202c457ddc`，训练入口为 `run_td3_train_economic_ems.py`，注册环境为 `AggregateShipTD3Economic-v0`。

## 原始数据与额定边界

核对了 [FC_Data.xlsx](<C:/Users/20883/OneDrive/Desktop/氢耗/FC_Data.xlsx>) 的 `Sheet1!A2:B12`。11 组、22 个数值均与代码内嵌数据一致，文件 SHA-256 为 `906a0383f6e427a938a8343e9fb1428bfea0f8e2766f5ac5449fea0ccbde4a21`，与代码溯源记录一致。

- A 列功率以 kW 计，E 列为辅助功率，G 列电堆功率逐行等于 A 列与 E 列之和。因此 A 列是系统净输出功率。B 列效率由百分数除以 100 后使用。[同目录 MATLAB 脚本](<C:/Users/20883/OneDrive/Desktop/氢耗/figure_fc.m:30>) 也以该口径处理 A、B 列。
- 原始功率严格递增，覆盖 `0–116.668 kW`。代码先用完整原始数据的 PCHIP 插值得到 `100 kW` 的效率 `0.5476352421776669`，再使用额定范围内的原始点及此端点重建正式 PCHIP。`109.123 kW` 和 `116.668 kW` 两点仅参与原始端点插值，正式查询范围为单套 `0–100 kW`。
- 首个正功率实测点 `11.3938 kW`、效率 `0.638554`，对应聚合 `68.3628 kW`。最后一个额定范围内的实测点为 `98.3215 kW`，对应聚合 `589.929 kW`。`100 kW` 额定端点为插值点。代码禁用外推，拒绝负功率与超过 `600 kW` 的输入。以 1,201 个查询点独立核验，正功率查询效率均大于 0 且不超过 `0.638554`。[SciPy PCHIP 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- 工作簿 F 列为原有氢耗数据，配套 MATLAB 脚本以 g/min 使用。用 A、B、F 列反算的隐含热值约为 `115.08–116.92 MJ/kg`，工作簿未明示热值或偏差来源。本项目按 A、B 列的净功率和效率配合指定的 `120 MJ/kg` 计算氢耗，F 列未参与本次模型核算。

## 聚合公式与执行路径

六套均匀分配时，每套的净输出为 \(P_i=P_{\mathrm{FC}}/6\)。正式聚合曲线仅放大额定域 PCHIP 的功率轴，效率保持原值。

\[
\eta_{600}(P_{\mathrm{FC}})=\eta_{100}(P_{\mathrm{FC}}/6),\qquad
m_{\mathrm{H_2}}=\sum_{i=1}^{6}\frac{P_i\Delta t_h}{\eta_{100}(P_i)LHV}
=\frac{P_{\mathrm{FC}}\Delta t_s}{3600\,\eta_{600}(P_{\mathrm{FC}})\,(120/3.6)}.
\]

功率用 kW，时间用秒，氢耗用 kg。额定域重建之后再缩放功率轴的步骤与查询单套额定域曲线的 `P/6` 等价，效率无需再乘除 6。

1. [TD3 可行区间动作包装器](../../experiments/sb3_manual/td3_aggregate/economic_action_wrappers.py#L54) 把 Actor 的归一化动作映射到当前聚合 FC 可行区间。[共享物理环境](../../experiments/sb3_manual/ddpg_aggregate/env.py#L234) 用实际 FC 净输出执行 1 秒物理区间。
2. [EconomicMPC.interval()](../../src/v3/control.py#L135) 按 `P_bat = P_load - P_FC` 平衡总线功率，把同一个实际 FC 净功率传入[正式效率曲线与氢耗函数](../../src/v2/models/fuel_cell_efficiency.py#L316)。[负载重构](../../src/v2/data/supervisory_rules.py#L151) 来自 FC 和电池总功率。工作簿辅助功率只用于核对净输出口径，运行时无额外辅助功率扣减或氢耗加算。
3. [物理经济账本](../../src/v2/economics.py#L387) 以 `21.9 元/kg` 计算实际氢耗成本。[环境奖励明细](../../experiments/sb3_manual/ddpg_aggregate/economic_reward.py#L119) 向上提供同一笔 `h2_cost_cny`，[TD3 奖励包装器](../../experiments/sb3_manual/td3_aggregate/economic_ems.py#L197) 以权重 `1.0` 使用一次。电池等效氢耗为独立成本项，其固定效率基准不变。

零功率分支先返回 `0 kg`，不会进入效率除法。`0 < P < 68.3628 kW` 是关机点与首个正功率实测点之间的插值区，缺少正功率实测数据。当前插值在该区间靠近零点时，使 1 秒氢耗趋于 `0.000586419 kg`。该极限由插值在零点附近的斜率决定。`68.3628 kW` 仅是首个测量点。现有模型未引入最低稳定功率或单列怠速氢耗。

## 修改与核验

现有聚合效率映射与氢耗调用链符合给定模型。生产代码无差异。新增 [`test_fuel_cell_aggregation.py`](../../experiments/sb3_manual/td3_aggregate/test_fuel_cell_aggregation.py)，以从源数据独立重建的单套曲线和六套逐项求和为参照，核验额定边界、功率映射、效率范围、1 秒单位、零功率、越界拒绝、总线平衡及物理账本与 TD3 奖励对账。

在 TD3 工作树根目录执行的相关单元测试命令如下。

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src"
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
& '..\v6-sb3-baseline\.venv\Scripts\python.exe' -B -m unittest experiments.sb3_manual.td3_aggregate.test_fuel_cell_aggregation experiments.sb3_manual.td3_aggregate.test_economic_ems experiments.sb3_manual.ddpg_aggregate.test_economic_reward -q
```

最终运行耗时 `20.398 s`，全部 `24` 项通过。

### 固定 Train 航段的确定性物理回放

对 `parent_005_seg01` 使用 `seed=42` 和固定归一化动作 `0.0`，即每步取可行区间中点。该动作是物理核验用探针。用直接读取 Excel 重建的单套曲线逐秒计算六套参考氢耗，并与环境账本比对。

| 项目 | 实际结果 |
| --- | --- |
| 完成状态 | 完整执行 `9,258 / 9,258` 个 1 秒区间，无物理失败或截断 |
| FC 净功率与效率 | `300 kW`，每套 `50 kW`，效率 `0.587302169363016` |
| 总实际氢耗 | `39.4090149967 kg` |
| 氢耗成本及 TD3 氢耗分项 | 均为 `863.0574284269 元`（四舍五入） |
| 最大逐步氢耗和成本差异 | 独立六套计算与物理账本逐步一致，累计加总误差约 `4.7e-12 kg` 和 `6.9e-11 元` |
| 最大净功率平衡误差 | `2.84e-14 kW` |
| SOC | 初始 `0.6`，最终 `0.4567713855`，范围 `0.3587369407–0.6920458095` |
| 完整物理成本与四项学习成本 | `1,401.275111982 元` 与 `1,324.192401473 元` |
| 终端停机 | 一次，时长 `0 s`，终态 FC 和电池功率均为零 |
| 实际耗时 | `46.934 s` |

回放仅覆盖 `300 kW` 工作点，零功率、首个正功率测量点以下和额定端点由单元测试独立核验。未启动训练或 Optuna。回放命令及本机证据如下。

```powershell
$env:PYTHONPATH = "$PWD;$PWD\src"
$env:FC_REPLAY_WORKBOOK = 'C:\Users\20883\OneDrive\Desktop\氢耗\FC_Data.xlsx'
& '..\v6-sb3-baseline\.venv\Scripts\python.exe' -B "$env:TEMP\td3_fc_replay_d95f57a_20261108\replay.py" "$env:TEMP\td3_fc_replay_d95f57a_20261108" 0.0
```

[回放摘要](C:/Users/20883/AppData/Local/Temp/td3_fc_replay_d95f57a_20261108/fixed_replay_action_0p0_summary.json) · [逐步 CSV](C:/Users/20883/AppData/Local/Temp/td3_fc_replay_d95f57a_20261108/fixed_replay_action_0p0_steps.csv) · [回放脚本](C:/Users/20883/AppData/Local/Temp/td3_fc_replay_d95f57a_20261108/replay.py)
