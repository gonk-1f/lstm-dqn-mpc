# 独立 DDPG Test4

只评估已冻结的模型，不训练、不调参，不修改现有训练及 Validation 代码。

在实际 DDPG 项目根目录执行：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
& .\.venv\Scripts\python.exe -m experiments.sb3_manual.ddpg_aggregate.test4 --model-path "outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --device cpu
```

PyCharm 可直接选择共享配置 **DDPG Test4** 并点击运行。配置位于项目根
`.run/DDPG Test4.run.xml`，采用 Module name：
`experiments.sb3_manual.ddpg_aggregate.test4`，工作目录为上述根目录，
沿用项目解释器（该工作树的 `.venv\Scripts\python.exe`）。无需手动设置 PYTHONPATH。

更换模型只需替换 `--model-path`。默认输出根为 `outputs/ddpg_fixed_parent_005_test_economic_v3`，
可用 `--output-root` 更改；默认 seed=42、CPU、torch_threads=1。

## 数据与推理

固定按 `parent_001_seg01`、`parent_004_seg01`、`parent_013_seg01`、`parent_032_seg01`
顺序评估四条完整 Test 航段，直到正常结束或真实物理失败。没有随机抽样、
Episode 数量选择或步数截断参数。

V6 目录仅保存 Test 元数据；原始 canonical 1 秒 payload 位于元数据指向的
`v5-economic4-ddqn` 源工作树数据目录。入口直接只读这些文件，保留终端边界定义，
不复制、不重采样、不插值或重新生成数据。源目录迁移时可传入
`--test-source-root`；数据元信息与实际来源记录在结果 JSON 中。

独立 Test 环境继承原 DDPG 环境的物理、奖励、观测和动作逻辑。为复用原初始化，
构造时会只读加载 Validation8，随后替换为冻结的四条 Test；不会执行 Validation 航段。
新 Test 环境不加入 Zoo 训练注册，不改变原环境仅允许 Train/Validation 的限制。

观测与训练保持一致，为`[SOC, load/600, previous_actual_fc/600, delta_load/600,
elapsed_seconds/3600]`五维状态，每条航段首步负载变化量为0，运行时间从0开始。
旧三维状态模型不兼容；训练使用的Critic Huber损失不会在Test推理中计算。

使用原生 `DDPG.load()`、`deterministic=True`，关闭加载副本的探索噪声；
不调用 learn、梯度更新或目标网络同步，不加载或修改磁盘 Replay Buffer。
结果包含模型和更新状态的不变性检查。若模型训练使用 VecNormalize，可通过
`--vecnormalize-path` 指定训练统计，推理时冻结统计并关闭奖励归一化。

## 输出

每次运行新建 `outputs/ddpg_fixed_parent_005_test_economic_v3/test_时间戳/`，保留已有结果：

- `test_summary.csv`：每航段奖励、完成状态、覆盖率、功率变化、SOC、能量及经济指标。
- `test_step_data.csv`：逐步真实动作、功率、SOC、奖励及可用账本。
- `test_metrics.json`：总体指标、模型及数据身份、不变性检查。
- `test_reward_curve.png`：各航段累计奖励及标注，不是训练学习曲线。
- `test_power_curves.png`：FC、LB 及负载功率分配。
- `test_soc_curves.png`：各航段 SOC 轨迹与终止原因。

电池母线功率正值为放电、负值为充电。失败调用如未执行物理动作，不制造新的
执行时间、电池功率、氢耗或经济成本；失败奖励仍计入 Episode Reward。

成本复用现有聚合 FC/电池经济模型，不是逐堆实测退化成本，未结算末端 SOC 恢复。
失败前缀不能当成完成整条航段的成本，与其他模型比较时应先看完成状态和覆盖率。

## 当前经济奖励与模型身份

奖励版本为 `ddpg_economic_negative_no_start_stop_v5`。正常奖励为：
`-(C_H2+C_bat_eq+C_LB_deg+C_FC_transient+C_FC_low+C_FC_high)`。
六项成本零值对应奖励0；有限有符号合计直接取负，不使用tanh、倒数或C_ref缩放。
非有限值报错，完整经济账本的原有负值检查不变。
真实物理不可行失败奖励为 -2000，失败调用不制造物理经济成本。
不再包含独立SOC目标偏差成本；电池退化模型内部的SOC应力函数仍保留。

`C_ref` 仅保留原Train35标定历史元数据，不参与当前奖励；未使用Validation/Test标定。
FC累计电压损失报废阈值为142500 μV；保留四工况系数、设备价和50%更换比例。
80%高负载分界为项目建模假设，不是Zhou原文参数。电池保持F(SOC)、G(I)、
加权Ah吞吐和15000倍容量归一化，退化使用与物理SOC更新一致的实际电流。

电池等效项固定使用`eta_fc_avg=0.5755641791002982`；FC实际氢耗仍用当前功率点效率。
电池等效氢耗保持放电正、充电负的符号，不重复计算效率损失。
`total_cost_cny`保存含FC启停的完整经济总额，`reward_total_cost_cny`保存不含启停的六项奖励成本。
FC四工况累计电压损失及完整账本保持不变，四类成本按本步raw损失占比分摊共同EOL裁剪。
逐秒CSV记录`fuel_cell_transient_cost_cny`、`fuel_cell_low_load_cost_cny`、
`fuel_cell_high_load_cost_cny`、`fuel_cell_start_stop_cost_cny`及`battery_current_a`，
不导出独立SOC惩罚字段。

仅加载完整奖励与标定身份一致的新版本模型，旧线性、旧tanh及无身份模型均拒绝。
新训练默认输出为 `outputs/ddpg_fixed_parent_005_seg01_500ep_economic_v3`；
模型需在后续从头训练后才存在。根目录对应的运行脚本可直接在PyCharm绿色运行，
无需Module配置；也可使用上述模块入口的`--model-path`选择相同版本的其他checkpoint。
