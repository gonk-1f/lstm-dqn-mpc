# 新旧环境短时 Train 性能与等价性

2026-10-29，本机 CPU、PyTorch 单线程，使用 `.venv-optuna/Scripts/python.exe`。仅加载最短 Train `parent_046_seg02`（完整长度 3272 秒）的一个 payload，未加载其余 34 条 Train、Validation 或 Test。先用冻结 Queue 生成最多 128 步合法固定策略动作，再在相同初始状态和相同动作下各运行 32/64/128 步，每项 3 次，交替新旧先后顺序。

比较共 672 对真实 step；逐步 observation、完整 `DirectTransition` dataclass（含 FC/电池功率、SOC、氢气与退化账本、reward、terminal reason 等全部字段）、successor mask 和 terminated/truncated 均通过等价比较。浮点容差为 `rel_tol=1e-11, abs_tol=1e-8`；不是承诺任意轨迹 bit 相等。完整旧 Queue 仅支持历史 `v6_energy_soc_ramp`，因此速度等价基准双方都用该模式；纯经济短验证另列。

| 前缀步数 | 重复 | Queue step 总时长中位 / s | Direct step 总时长中位 / s | Queue/Direct 实测比 | Queue reset 中位 / s | Direct reset 中位 / s |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3 | 0.093549 | 0.066032 | 1.4167 | 0.026105 | 0.023129 |
| 64 | 3 | 0.180312 | 0.125732 | 1.4341 | 0.025234 | 0.022542 |
| 128 | 3 | 0.362488 | 0.278842 | 1.3000 | 0.026714 | 0.022192 |

step 计时只包含真实 `env.step()` 调用，不含随后结果捕获和等价性断言。机器文件也逐次记录包括捕获的前缀 wall time、构造、reset、close，便于识别计时范围。这里只证明上述短前缀与机器条件下的实测比；没有预言长航段或 500 episode 速度。

额外 `zhou_economic_only` 微测只执行相同 Train 32 步：FC 保持关闭，总经济成本 2.1888949570 CNY、return −2.1888949570、恒等残差 0；单步成本范围 0.06742653–0.06940929。32 次 step 总计 0.0625182 秒、平均 1.953694 ms。FC 启动、寿命截断和失败末步没有在该窄前缀发生，不能从中推断这些状态的吞吐或完整经济分布。

在一次性 CPU `MaskedDQN`、`[256,256]`、batch=64、Adam=1e-4、单线程下，把这 32 条真实 transition 存入容量 256 的临时 Replay，再以有放回采样进行 3 个单 Adam 调用：116.407、12.262、12.596 ms。首调用明显含初始化/冷启动开销，中位 12.596 ms。没有调用 `learn()`，没有继续采集正式 episode，没有保存 checkpoint/Replay；这是 3 次明确标记的可丢弃优化微测，不是正式训练或搜索。32 条 tiny Replay 的缓存/抽样行为不等价于成熟百万 Replay。

该次总计 128 条预生成动作执行、两环境各 672 条 timed transition、32 条经济微测 transition，共 1504 条真实执行；每个前缀最长 128。CLI 强制单前缀 <=256、重复 <=10、每 adapter timed transition <=4096，拒绝非 Train `--sample-id`。

## 预算应如何解释

当前 35 条 Train 的完整长度为 3272–36660 步，均值 13073.2。若每个 episode 都完整完成，500 个独立有放回均匀抽样 episode 的步数范围是 1,636,000–18,330,000，期望 6,536,600；另有 50,000 随机预热步。真实失败会缩短 episode，首决策 event-only 失败可产生 0 步，不能把上述完整完成下界当作无条件下界。

下面是明确标记假设的工作量算术，不是完整训练时间预测。使用排序后的 35 个 Train ID，与实际 `EpisodeSchedule` 相同的独立 `random.Random(42).choice`，不执行任何物理 step，500 次有放回抽样在全部完成假设下实际共 **6,119,941** 步；100 次为 1,240,435 步，150 次为 1,863,164 步。完整按 ID 抽中次数保存在 JSON 的 `budget_scenarios`，与模型 RNG 相互独立。

短前缀单步观测值 `t_step=0.0019536937504653906 s`。Adam 原始 3 个观测完整保留；明确剔除第一项 0.1164067 s 的冷启动后，仅用后两项 0.0122624、0.0125956 s 的中位数 **0.0124290 s/update**，每 32 环境步摊销为 0.00038840625 s。因此仅计算采集中的物理 step 加 Adam：

```text
T_formal_component(N) = N * 0.0019536937504653906
                     + floor(N / 32) * 0.012429000002157409 seconds
```

| 预算 | 全完成最短/最长抽样工作量对应时长 / h | 均匀抽样期望步数 | 期望步数的采集+Adam / s (h) | seed42 实际抽样步数 | seed42 采集+Adam / s (h) |
|---|---:|---:|---:|---:|---:|
| 100 episode / Trial | 0.213–2.385 | 1,307,320 | 3,061.865 (0.851) | 1,240,435 | 2,905.215 (0.807) |
| 150 episode / Trial | 0.319–3.578 | 1,960,980 | 4,592.803 (1.276) | 1,863,164 | 4,363.706 (1.212) |
| 500 episode / 正式基线或独立重训 | 1.064–11.925 | 6,536,600 | 15,309.362 (4.253) | 6,119,941 | 14,333.512 (3.982) |

表内区间只是“每次都抽到最短/最长 Train 且全完成、每步恰好耗时相同”的极端场景范围，**不是置信区间或实际运行时间上下界**。期望步数的更新数用 `floor(E[N]/32)` 作近似；seed42 的 191,248 次正式 Adam 是对应整数步数的精确 floor。失败会缩短 episode，真实行为也会改变逐步耗时。

另从冻结 manifest **只读取 Validation 的 8 个 duration 字段**，总和 143,762 步，没有加载 Validation payload 或执行 Validation，也没有读取 Test payload/成本来估预算。按每 50 episode 一次全部完成评价计算：

- 500 episode 的 10 次评价：1,437,620 个物理 step，以相同 `t_step` 粗算 2,808.669 秒（0.780 h），不含验证策略推断。
- 100 episode 的 2 次评价：287,524 步，561.734 秒（0.156 h）。
- 150 episode 的 3 次评价：431,286 步，842.601 秒（0.234 h）。

50,000 步预热按同样固定 `t_step` 粗算 97.685 秒；这里不在预热期间计 Adam。把 seed42 正式采集+Adam、50k 预热物理 step、上述 Validation 物理 step 相加，得到 **500 episode 为 17,239.866 秒 / 4.789 h；100 episode 单 Trial 为 3,564.634 秒 / 0.990 h；150 episode 单 Trial 为 5,303.991 秒 / 1.473 h**。这三个数只对应已列出的工作项，不是可靠总时长或 SLA。

尚未实测的策略预测、成熟百万 Replay 抽样与内存行为、真实探索和启动/失败状态、重置、报告、Validation 策略推断、序列化、模型/Replay checkpoint、Windows 负载都会改变总时长；500 episode 的 draw 序列长度本身不保证全完成。不能依据表格承诺完成时间。Optuna 多 Trial 预算可先写作 `候选 Trial 数 × 同预算单 Trial 的实际时间`，但表内 100/150 episode 仅提供初步工作量，需由用户通过已交付的单 Trial 计时入口核实真实训练及验证开销，再决定 Trial 数量；本次没有启动 Trial。

## 复现及结果文件

```powershell
.venv-optuna\Scripts\python.exe scripts\benchmark_v6_environment.py --help
.venv-optuna\Scripts\python.exe scripts\benchmark_v6_environment.py --steps 32 64 128 --repeats 3 --measure-adam --adam-repeats 3 --torch-threads 1
.venv-optuna\Scripts\python.exe scripts\benchmark_v6_environment.py --budget-only
```

全部原始计时、次序、动作、payload SHA256、672 step-pair 数量、微测 32 条分项账本和 metadata 预算见 `environment_benchmark.json`。`--budget-only` 读取原有计时更新预算，不执行任何新的 step 或 Adam。Windows 调度、其他进程负载、缓存和短时抖动会影响结果；该归档保留实际测量，无未执行的正式训练结果。
