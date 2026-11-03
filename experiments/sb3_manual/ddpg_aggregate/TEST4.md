# 独立 DDPG Test4

只评估已冻结的模型，不训练、不调参，不修改现有训练及 Validation 代码。

在实际 DDPG 项目根目录执行：

```powershell
Set-Location 'C:\Users\20883\OneDrive\Desktop\lstm-dqn-mpc\lstm-dqn-mpc\.worktrees\v6-sb3-baseline'
& .\.venv\Scripts\python.exe -m experiments.sb3_manual.ddpg_aggregate.test4 --model-path "outputs/ddpg_train_500ep_economic_v1/ddpg/AggregateShipDDPG-v0_1/AggregateShipDDPG-v0.zip" --device cpu
```

PyCharm 可直接选择共享配置 **DDPG Test4** 并点击运行。配置位于项目根
`.run/DDPG Test4.run.xml`，采用 Module name：
`experiments.sb3_manual.ddpg_aggregate.test4`，工作目录为上述根目录，
沿用项目解释器（该工作树的 `.venv\Scripts\python.exe`）。无需手动设置 PYTHONPATH。

更换模型只需替换 `--model-path`。默认输出根为 `outputs/ddpg_test_economic_v1`，
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

使用原生 `DDPG.load()`、`deterministic=True`，关闭加载副本的探索噪声；
不调用 learn、梯度更新或目标网络同步，不加载或修改磁盘 Replay Buffer。
结果包含模型和更新状态的不变性检查。若模型训练使用 VecNormalize，可通过
`--vecnormalize-path` 指定训练统计，推理时冻结统计并关闭奖励归一化。

## 输出

每次运行新建 `outputs/ddpg_test_economic_v1/test_时间戳/`，保留已有结果：

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


## 经济奖励版本更新

当前默认采用 `ddpg_economic_tanh_v1`：四项经济成本（实际氢耗、FC退化、电池退化、实时电池等效氢耗）加固定 SOC 平方惩罚后计算 `tanh(1/r)`；物理失败为 -1。
固定 `C_typ=0.17429534153424198`、`lambda_soc=17.429534153424196` 来自 Train35 实际执行步的中位数，未使用 Validation/Test 标定。
FC关机参考效率由现有正功率曲线取峰值，开启时使用实际功率点效率；每秒记录两种效率。`total_cost_cny` 为四项经济成本，`reward_total_cost_cny` 另加SOC罚项；无额外终端补能计价。
新奖励模型默认路径需在后续新训练完成后才会存在。显式加载旧策略时，JSON标记其在新奖励下评估；不同奖励身份的模型不混用。根目录对应的直接运行脚本无需Module配置。
