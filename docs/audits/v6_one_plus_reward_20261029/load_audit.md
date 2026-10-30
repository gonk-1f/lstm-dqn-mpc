# 35 Train 全时域小负载来源审计

日期：2026-10-29；代码基础 `2c2577e`。本报告只描述既有数据和固定诊断，不定义新的 deadband，不修改物理、经济、数据或训练代码。扫描全部 35 Train 的完整 canonical 1 s 曲线；未运行学习、Validation 或 Test 控制。机器证据为 [load_audit.json](load_audit.json)、[全部小节点及独立列](load_small_nodes.csv)、[分箱](load_bins.csv)、[逐航段计数](load_per_voyage.csv)、[全部既有诊断匹配](load_diagnostic_matches.csv)。

## 1. t=1 的 0.0017413563970556567 kW 从何而来

`parent_015_seg01` 的当前 Train 身份对应 `zero_boundary_015` / `4月28日08_00_4月28日12_00`。其 `t=1` 是 `2024-04-28 08:49:35+08:00`，映射 `source_time_s=1`、`source_1s_row=1`。该字符串与 `operating_dataset_zero_boundary_v2/train/zero_boundary_015.csv` 对应行逐字相同；全部 35 Train 的每个 canonical 节点均核验了源行负载字符串和时间戳。

生成链为：8 个 FC 通道和 12 个 BMS 电池簇通道按既有规则对齐、求和，`P_source=P_FC-P_battery_raw`；电池原始约定是充电为正，bus 约定为放电为正。总负载是派生值，不是独立负载传感器。原始 BMS 电池功率由电压、电流计算；对齐和缺测规则见 `src/v2/data/segment_power_source.py`。较长内部缺口先分别对 FC、raw-sign battery 做局部 natural cubic；随后裁剪/生成零边界，再对派生负载用 PCHIP 生成 1 s 曲线。独立航段裁剪保留已有负载字符串，V6 再复制已有 canonical 文件；这两层没有新增插值。

本次仅对这一已确认的 Train 父组载入原始通道并重做已有预处理：456 个 aligned 派生点，2 个 gap 插入 24 点，trim 后 339 个锚点，PCHIP 得到 10116 个节点。重建结果与冻结源曲线全部 10116 个节点的最大绝对差为 **0 kW**。这确认了生成机制，并不验证传感器精度或证明 1.741 W 是真实船舶需求。

| 相对秒 / 时间戳 | canonical 总功率 / kW | 生成证据 |
|---|---:|---|
| 0 / 08:49:34 | 0 | aligned 原始派生值 `0.48561800000000005`，既有起点 `OBSERVED_DEADBAND` 将其设为 0 |
| 30 / 08:50:04 | 1.1170300000000006 | aligned 派生点，未被 cubic 填入 |
| 60 / 08:50:34 | 5.358458092904364 | gap 的 cubic 插入点 |

以上前三个锚点即可复算 PCHIP 第一区间，其中 `t` 是距 08:49:34 的秒数：

```text
P(t) = -1.7248688021076253e-5 * t^3 + 0.0017586050850767329 * t^2
P(1) = 0.0017413563970556567 kW
```

08:49:35 不是组装后的 aligned 派生总功率采样时刻。这里没有声称每一个原始设备通道在该时刻都没有记录；对齐后的总功率、个别设备原始记录、插值负载是不同证据层级。原始信号的仪表分辨率、误差、独立负载校验及这么小的需求是否具有物理意义，现有证据仍不足。

## 2. 既有零边界与运行期 normalize 的语义不同

`zero_boundary_dataset.py` 的边界查找允许 `abs(P)<=1 kW` 的 observed boundary，并把选中的边界值设为 0；这是既有数据生成时仅对选定端点的处理。不能将其解释成运行中任意 `abs(P)<=1` 都归零。

`src/v2/data/supervisory_rules.py::normalize_onboard_load_kw` 的实际运行契约为：

```text
P < -1 kW    -> ValueError
-1 <= P <= 0 -> 0 kW
P > 0        -> 保留原正值，包括极小正值
```

因此正负小值处理不对称。它保留了原始 signed load 作为数据/transition 证据，实际电源平衡使用 normalize 后的负载。`t=1` 的正数会被保留；对应负数 `-0.0017413563970556567` 会被现有规则归零。该行为有既有合同测试，本次没有扩大 deadband，也没有为了奖励稳定重写负载。

## 3. 全 35 Train 描述性统计

共 **457597 个节点**，其中 **457562 个真实控制间隔**和 35 个 terminal boundary 元数据节点。以下阈值只是统计分箱，不是新的物理阈值：

| abs(raw load) / kW | 控制节点负值 | 控制节点正值 | 控制精确零 | terminal 节点 |
|---|---:|---:|---:|---:|
| 精确 0 | 0 | 0 | 25 | 29 |
| (0, 1e-6] | 0 | 0 | 0 | 0 |
| (1e-6, 1e-4] | 0 | 0 | 0 | 0 |
| (1e-4, 0.001] | 2 | 2 | 0 | 0 |
| (0.001, 0.01] | 15 | 17 | 0 | 0 |
| (0.01, 0.1] | 4245 | 218 | 0 | 1（负） |
| (0.1, 1] | 9 | 849 | 0 | 2（正） |
| >1 | 0 | 452180 | 0 | 3（正） |

`abs(P)<=1` 总计 **5414 节点**：5382 控制节点、32 terminal 节点；按 raw 符号分为负值 4272、正值 1088、精确零 54。最小严格正值为 `0.0002771791822630402 kW`。不能把这些全时域节点计数与上一轮 `35×32` 前缀的 35 个零成本步骤混为一谈：两者统计对象、状态与功率策略都不同。

全部 5414 小节点已导出 canonical 路径、父源路径、原始字符串、timestamp、source_time_s、source_1s_row 和 normalize 后负载。原始文件未改动。

## 4. source、既有固定执行和反事实分列

`load_small_nodes.csv` 使用三个独立列前缀，避免混称实测或真实 DQN 行为：

- `source_record_*`：同一源时间恰好有既有 30 s component sidecar 记录时，保留其 FC/battery bus 值及 `power_provenance`。185 个小节点存在该记录，5229 个没有，字段留空。sidecar 自身可含 aligned/cubic/PCHIP 处理，不能一律称独立实测 1 s 功率。
- `diagnostic_*`：仅关联上一轮已保存的固定策略实际执行。5414 节点中 522 个具有至少一条匹配，4892 个没有，缺失的状态和账本保持空。主列优先 `load_follow`；同一时刻多个 policy/source 的全部匹配另存 `load_diagnostic_matches.csv`，不拼接为新轨迹，也不宣称来自 DQN。
- `counterfactual_*`：经根任务明确授权，对 5382 个控制小节点分别计算一个独立 1 s 样本，固定 `SOC=.6`、FC off、上一 FC=0、两项老化累计=0，每行重新开始，实际执行负载取既有 normalize。每行保留 scenario/assumptions、FC、battery、SOC before/after/delta、四项成本与合计。32 个 terminal 元数据节点不模拟动作。这些值不补入 diagnostic 空列，不能代替该时刻真实策略状态。

独立反事实中 4296 个 `C=0`、1086 个 `C>0`；无学习、无串接长航段。FC0 对这些固定初态和负载合法；没有人为禁用低负载或空闲动作。

`parent_015_seg01,t=1` 既有固定诊断与同初态反事实一致：

```text
FC = 0 kW, battery = 0.0017413563970556567 kW
SOC: 0.6 -> 0.5999999991840248; delta=-8.159751763159306e-10
H2 = 0 CNY; FC degradation = 0 CNY
battery degradation = 9.803203879319011e-8 CNY
fixed-reference SOC equivalent = 5.856507800521894e-7 CNY
total = 6.836828188453796e-7 CNY
```

这里仍采用既有固定 SOC 参考价；本审计不改经济口径。极小正成本来自给定物理/参考价方程与插值负载，不能仅因其导致较大 `1/C` 就断言数据错误，更不能静默改成对称 deadband。

## 5. 读取范围、验证和限制

本次扫描只由冻结 manifest 选出的 35 Train canonical 文件和对应的当前 Train 父源/sidecar。只对明确的 4月28 Train 父组重做原始来源链。未打开当前 Validation/Test 航段 payload，也未用其数据选择阈值或模型。

读取事件透明记录：初期按旧编号打开过 `operating_dataset_final/metadata/aligned_30s/parent_015.csv` 的表头和前 7 行，其 parent 为 `4月18日12_00_4月18日18_00`。随后仅用当前 manifest 元数据确认它对应当前 `parent_012_seg01` / Train，而非本次目标 4月28父组。该旧文件内容全部排除本次数值、来源重建及参数判断，之后未再打开。机器结果保留该事件及元数据映射，避免掩盖编号错配。

验证：审计脚本自检通过；既有 normalize 合同和 signed raw load 保留合同 **2 passed**。机器产物另核验 5414 行总量、5382 控制反事实上限、terminal 无反事实、4892 diagnostic 缺失保留、负载源字符串逐字相同。没有正式训练、新的完整策略航段、Validation 或 Test 控制。

复现（项目根 PowerShell）：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.venv-optuna\Scripts\python.exe scripts\audit_small_train_loads.py --self-check
.venv-optuna\Scripts\python.exe scripts\audit_small_train_loads.py
```

建议保持当前数据和物理契约不变，将这份来源证据用于说明奖励设计应如何处理合法零/小正成本。若要修改数据阈值，应单独建立测量误差依据、明确端点与运行期规则、生成新的数据版本并重审物理/经济影响；本次没有执行这些修改。
