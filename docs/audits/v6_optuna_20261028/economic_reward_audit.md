# V6 经济评价与奖励审计（2026-10-28，阶段 2）

## 结论与边界

**已证实：`zhou_economic_only` 可以表达当前独立完整航段经济评价的负值。** 条件是独立航段从默认 SOC 0.6、FC 关闭及新寿命状态开始，`beta_soc=0`、不作 battery-energy redistribution、`gamma=1`，并将失败终止分数单独列出。代数上完全相等；二进制浮点不承诺逐 bit 相等。现有 64 条 Validation 记录的最大恒等残差为 `1.1641532182693481e-10` CNY。

**已证实：冻结 V6 奖励不包含 FC 或电池退化。** 它包含实际氢气、signed SOC 折氢、每步 SOC 辅助项和平方 FC 变载辅助项。因此，训练回报与完整经济评价有明确不同的目标。不能将“奖励实现正确”推成“经济目标已在训练中被优化”。

本次只新增审计工具、固定人工短轨迹回归和报告；没有修改既有奖励、物理参数、经济系数或训练默认配置，没有正式训练、checkpoint 推断、生成新 Train/Val 轨迹或重跑 Test。离线读取范围是 64 个既有 Val CSV.gz 与 8 个既有 Train failure-prefix CSV.gz。Test 仅人工只读已归档 `test_summary.json`；其历史字段 `test_payloads_opened=4` 描述原来的 Test 执行，并非本次执行。

## 1. 四种模式的实际语义

令 `L_t = C_H2,t + C_FCdeg,t + C_Battdeg,t`。独立航段没有 shore 电费、末端补能或人为 FC shutdown 动作。令

```text
K = 624 kWh * 21.9 CNY/kg / (eta_bar * (120/3.6) kWh/kg)
S_t = K * (SOC_t - SOC_(t+1))
U_beta,t = beta_soc * (SOC_(t+1) - 0.60)^2
U_6,t = 0.2 * ((SOC_(t+1) - 0.50) / 0.1)^2
R_Z,t = 0.0018375 * (delta_P_FC,kW)^2
R_6,t = 0.05 / 600 * (delta_P_FC,kW)^2
```

`eta_bar = 0.5711999752720877` 为既有正式效率曲线在 10、20、…、600 kW 六十个正功率点的等权平均，不是从本次控制轨迹拟合的效率。实际氢耗仍由每步正式瞬时效率曲线计算。

| 模式 | 每步奖励，未计独立失败分数 | 与完整评价的关系 |
|---|---|---|
| `full_economic` | `-L_t - U_beta,t` | 没有 signed SOC 折氢；独立末端储能没有评价修正 |
| `zhou_economic_only` | `-L_t - S_t` | 强制 `beta_soc=0`，没有 SOC/ramp 辅助项 |
| `zhou_joint` | `-L_t - S_t - U_beta,t - R_Z,t` | 保留真实退化，再额外加入平方变载和可选 SOC 偏好 |
| `v6_energy_soc_ramp` | `-C_H2,t - S_t - U_6,t - R_6,t` | 完全排除两项退化，默认权重 0.2 / 0.05 |

`full_economic` 在 legacy 任务还可附带实际/建模 shore 和 terminal settlement；不能将上述独立航段公式扩展到 legacy。该模式允许可选 redistribution；离线四模式比较明确固定为 `False`、`beta_soc=0`。工具的 `zhou_joint` 默认 `beta_soc=0`，可通过 `--zhou-beta-soc` 明示改变。`replay_episode` 的参数默认也是零；不能把文件内 `DEFAULT_BETA_SOC=200` 常量误当作该调用的隐含值。

SOC reference 的区别由回归直接验证：Zhou/通用 beta 项使用 **0.60**，V6 专用项使用 **0.50**。审计工具初版将 Zhou reference 误写为 0.50，四模式对照回归报错后按源码修正；冻结实现未改。

源码证据：`src/v4/control.py` 的 `SOC_REF`、`ledger_training_reward`、`replay_episode` 奖励分支；`src/v4/v6_training.py` 的 `CONFIG`、`execute`、`sample_record`；`src/v4/independent_economics.py` 的 `soc_equivalent_hydrogen_cost_cny` 与 `evaluate_zhou_economics`。`src/v6_sb3` 复用相同物理执行及账本。

## 2. Telescoping 与失败的严格边界

```text
sum_t S_t = K * (SOC_initial - SOC_terminal)
sum_t r_zhou_economic_only,t
  = -(sum_t L_t + K * (0.6 - SOC_terminal))
  = -evaluate_zhou_economics(original_total_ledger).total_cost_cny
```

SOC 已含正式电池充放电效率，不能在 `S_t` 再乘/除一遍效率，也不能再添加末端 SOC 修正，否则重复结算储能。末端保留最后执行的 FC 状态，不另造关闭动作。

失败分数 `F` 由 `prepare_failure_replay` 只在最后一个真实执行 transition 扣一次，不属于任何物理 ledger。对连续执行前缀，有 `sum r + executed_prefix_economic_cost + F = 0`；前缀经济成本不代表未完成航段的完整成本。首步即无动作的 event-only 失败没有可扣分的真实 transition，也不会制造一个虚拟账本步骤。

`gamma<1` 时，折现后的 SOC 差分和其他成本不再等于原完整经济评价；短回归已明确覆盖此区别。离线工具只报告 `gamma=1`。它也不把非连续 Train 记录中相隔数百秒的端点当成一个连续前缀。

## 3. FC 的 358.65、变载、运行及 EOL

正式 FC 电压损失包含四个原始分项。600 kW aggregate 到 100 kW reference 的功率映射是冻结项目模型 `1/6`；既有源码将其标记为项目建模关系，而非实船测量标定。本次核对实现，未独立重新认证文献适用性。

```text
replacement_cost = 3500 * 600 * 0.5 = 1,050,000 CNY
voltage_cost = replacement_cost / 70,000 = 15 CNY/uV
startup = 23.91 uV * 15 = 358.65 CNY / off-to-positive event
transient = 0.0441 * (1/6) * 15 * abs(delta_P) = 0.11025 CNY/kW
low-load runtime = 10.17/3600 * 15 = 0.042375 CNY/s
high-load runtime = 11.74/3600 * 15 = 0.0489166667 CNY/s
```

aggregate FC 功率达到 480 kW 时使用高负载运行项。运行项按时间缩放；启动项按事件计；变载项按绝对功率变化计，不按秒再乘一次。

**已证实，无重复累计计费：** `0 -> positive` 收一次完整启动 cycle；`positive -> 0` 不再收一次完整 cycle，但仍按其绝对功率变化收变载损失。启动时同时存在 startup 和 transient 是两个不同的正式损失分项，不是同一项重复加入 ledger。Zhou-joint 额外平方 ramp 是训练偏好，不能再称为真实 FC 账单。

真实 FC 经济成本采用 `1,050,000 * (min(loss_after/70000,1) - min(loss_before/70000,1))`。跨 EOL 只收剩余寿命；之后账单为零，原始物理损失继续累计，没有自动更换或复位。工具不人为分摊 EOL 截断后的启动/变载/运行份额，报告的是 raw 分项与 capped 总额。

64 条 Val 的最大单航段 raw FC 费用为 `1,030,760.5313333333`，约为该上限的 98.17%；没有观察到触发 FC EOL 截断。所有 Val raw FC 总额与 logged 总额的约 `1.08e-6` 差值是大量浮点累计的舍入，逐步重构最大差值仅 `1.46e-11` CNY。EOL 跨界和 post-EOL 在固定短回归中另行验证，不能因现有轨迹没有跨界就忽略该语义。

源码证据：`src/v2/models/fuel_cell_degradation.py`、`src/v2/economics.py`、`src/v3/control.py: EconomicMPC.interval`。电池退化也沿用既有 clipped 累积寿命差分，已有独立经济回归覆盖其 EOL 保留。

## 4. 每秒 SOC 和平方变载的量级

V6 SOC 项每次决策收一次，**没有乘 `duration_s`**。例如 SOC 恒为 0.60 时每步 0.2；1 Hz 的 30 秒累计为 6.0，而一次 30 秒决策只有 0.2。SOC 为 0.20 或 0.80 时每步为 1.8。把这组权重挪到不同决策周期会改变单位物理时间的控制偏好。上述常 SOC 对比是固定算术例子，不是新轨迹效果证据。

| `abs(delta_P)` kW | 真实 FC transient，CNY | V6 平方项，reward units | Zhou-joint 额外平方项 |
|---:|---:|---:|---:|
| 10 | 1.1025 | 0.00833333 | 0.18375 |
| 30 | 3.3075 | 0.075 | 1.65375 |
| 60 | 6.615 | 0.3 | 6.615 |

相同总爬升的 `30 + 30 kW` 与一次 `60 kW` 的真实线性 transient 相同，但平方项仅为后者一半；前者多一个有正运行时长的步骤时，运行费用还需另算。平方项因此偏好分散功率变化，不能当作真实线性退化的等价式。V6 在 10/60 kW 时仅为真实 transient 的约 0.76% / 4.54%，并且完全没有 358.65 启动项。该数值是实现量级对照，不是权重最优值的标定。

## 5. 既有 Train/Val 离线结果

工具读取 72 个获准归档文件，包含 1,019,485 个 Val 执行动作、64 条 Val 完整/失败轨迹，以及 47 个 Train 稀疏失败记录组。每个来源的相对路径、SHA256 和行数记录在 `reward_source_manifest.json`。未读取 Test CSV.gz 或 Test 数据源 payload。

| 验证项 | 全部 Val 的最大绝对残差 |
|---|---:|
| Zhou return + 经济评价 + 独立失败分数 | `1.1641532182693481e-10` CNY |
| 每步 signed SOC 之和 - 首末端点公式 | `5.684341886080802e-14` CNY |
| 独立 FC 重构 - 原 logged FC 单步成本 | `1.460875864722766e-11` CNY |
| 重算冻结 V6 reward - 原 logged 单步 reward | `4.440892098500626e-16` |

只有第 5、7 轮都完成相同的 8 条 Val，才在下表比较固定完整集合。

| 同一套 8 条 Val | 第 5 轮 | 第 7 轮 |
|---|---:|---:|
| 原完整经济评价，CNY | 1,579,024.0495 | 3,040,016.7651 |
| FC degradation，CNY | 1,564,363.1210 | 3,025,417.4163 |
| FC 启动次数 | 3,519 | 7,569 |
| FC 启动 raw 费用，CNY | 1,262,089.3500 | 2,714,621.8500 |
| FC transient raw，CNY | 296,004.7125 | 304,814.7900 |
| FC runtime raw，CNY | 6,269.0585 | 5,980.7763 |
| V6 SOC 辅助项之和 | 9,085.9787 | 13,962.8181 |
| V6 ramp 辅助项之和 | 7,054.3917 | 6,309.8000 |
| 原 V6 总回报 | -29,696.3316 | -33,835.5865 |
| 同轨迹改按 Zhou economic-only 计分 | -1,579,024.0495 | -3,040,016.7651 |

启动项占第 5/7 轮 FC 成本约 80.68% / 89.73%。第 7 轮 V6 平方 ramp 总项较低，但正式经济费用较高；这支持“辅助项与经济账单并不等价”，不能据此断言某一种新训练配置必然改善。

Train prefix CSV 保存的是抽取的真实状态（例如首步、距失败 1800/600 秒和最后约 300 步），不含全部中间步骤，也不含对应 FC/电池退化 ledger。因此工具只核对这些记录行的 V6 分项与 raw FC 分项，不填补缺口；其完整经济成本、capped FC 成本和经济模式回报均保留 `null`。不会把稀疏行求和冒充航段累计回报。

输出：`offline_reward_comparison.csv`（逐来源/航段或稀疏组）、`offline_reward_comparison.json`（聚合及限制）、`fc_component_scales.csv`（固定量级对照）、`reward_source_manifest.json`（来源校验）。

## 6. 既有 Test 聚合证据的允许用法

仅作问题溯源：已归档 `raw/test_round_05/test_summary.json` 的完整总费用为 `705,008.4212311902` CNY，FC degradation 为 `702,221.0009166506`。任务提供的既有启动计数为 1,756；按冻结系数直接算 `358.65 * 1756 = 629,789.40`，约占该 FC 退化总额 89.69%。**该计数本次未通过 Test 步进 payload 重新核验**。没有用这些 Test 数值制定权重、挑选 checkpoint、设计搜索边界或宣称新方法泛化性能。

## 7. 证据分类与建议

**已证实：** 四模式语义、完整独立航段的 gamma-one 恒等、FC 事件/线性/运行计费与 EOL 规则、既有 Val 分项重构、V6 当前确实排除了正式退化。固定回归和离线残差只证明这些实现事实。

**可检验假设：** 目标遗漏启动和真实退化可能促成当前频繁启动；改用经济模式后能否减少启动且保持完成率，需要新的、预先约束的 Train/Val 实验。不能从同一轨迹重评分推出新策略表现。现有失败分数约 `10,132.66`，相对常达十万元的经济航段成本可能不足以阻止较早失败；这也是待检验假设，不是已观测的策略动机。

**工程假设：** `eta_bar` 等权均值、600→100 kW 映射、±60 kW/s 实际增量限制、辅助 SOC/ramp 权重均保留其原项目/仿真假设身份；测试通过不会把它们变成实船标定。固定状态不包含累积 FC 损失，EOL 附近奖励可能依赖未显式输入的历史寿命；当前 Val 尚未跨界，不代表未来策略必然不受影响。

**建议（本阶段未改训练）：** 独立实验配置可以暴露 `zhou_economic_only`，显式记录 `gamma=1`、`beta_soc=0`、无 redistribution 以及失败分数口径。继续先按 Val 完成数和覆盖率，再按固定完整集合经济成本判断候选；不要以失败前缀成本、Loss 或重评分优势代替控制验证。失败 scale 的合理性应单列 Train-only 审计，不能直接把当前辅助奖励权重解释成经济系数，也不能依据 Test 调整。

## 8. 回归与复现记录

本机使用已有 `.venv\Scripts\python.exe`，未安装环境。TDD 先落盘测试，再实现新工具：显式 `PYTHONPATH=src` 后，已有恒等表征项 `2 passed`，新工具缺失项 `8 failed`（期望的 RED）。首次未设置 PYTHONPATH 的 collection 错误是运行入口问题，未改正式模块。工具补齐及 Zhou reference 更正后，单文件 `10 passed in 1.15s`；随后增加完整日志重构与 gamma 区别的两个检查，并与已有相关回归合跑：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.venv\Scripts\python.exe -m pytest tests/test_v6_economic_reward_audit.py tests/test_v4_independent_economics.py tests/test_v6_control.py -q
# 44 passed in 1.47s

.venv\Scripts\python.exe scripts/audit_v6_rewards.py --source-root docs/results/sb3_dqn_1s_20261028/raw --output docs/audits/v6_optuna_20261028
# exit 0; 72 sources, 64 Val trajectories, 47 sparse Train groups
# 1,019,485 Val rows; residuals as above
```

新增单文件现有 12 个收集项；上述 44 项还包含既有独立经济和 V6 控制回归。完整命令及结果也见 `regression_results.json`。没有 commit；没有运行正式训练或 Test。
