# 多时间尺度 DQN-WMPC v2：下层非线性 MPC

## 方法边界与版本

本页只定义 `MPC_OBJECTIVE_VERSION = fc_base_smooth_soc_deadband_mean_v2` 的下层控制器。下层 MPC 负责控制品质和物理可行性，不计算氢耗价格、设备价格、燃料电池或电池退化成本，也不使用上层 DQN 的经济奖励。上层 DQN 的动作只参数化三个正权重；它不改变下层目标项的定义。

当前冻结的基线时间尺度由 `TimeScaleConfig.provisional()` 给出：

- MPC 采样周期 `Ts_MPC = 30 s`；
- 预测长度 `N_MPC = 5`，即每次求解预测未来 5 个 MPC 步；
- 滚动时域每次只执行计划的第 0 步，下一控制周期重新观测和求解；
- `dqn_switch_steps=M=5` 是当前 provisional 基线，不参与下层计划长度；它与同为 5 的 `N_MPC` 仍是不同概念。

## 因果负载与基准功率

在控制时刻 `k`，预测器 API 只接收一个当前标量负载 `P_load(k)`，没有接收未来实测序列的参数。因此下层只能使用不晚于 `k` 的信息。v2 首个预测器是持久性预测：

`P_load_hat(k+i|k) = P_load(k), i = 1,...,N_MPC`。

基准负载采用一阶低通滤波。时间常数 `tau_LPF` 的单位明确为物理秒，且必须显式传入并大于 0：

`alpha = exp(-Ts_MPC / tau_LPF)`，

`P_base(k) = alpha P_base(k-1) + (1-alpha) P_load(k)`。

预测域内可在持久性负载上递推同一个滤波方程，得到 `P_base_hat`。初次观测时以当前负载初始化滤波状态，避免凭空引入历史值。

## 三项目标函数

权重为精确字段 `(q_base, q_smooth, q_soc)`。三个值都必须是有限正数，且仅允许浮点表示误差范围内的和为 1。对计划 `P_fc[0:N]`：

`J_base = (1/N) sum_i ((P_fc[i] - P_base_hat[i]) / 600 kW)^2`

`J_smooth = (1/N) sum_i (Delta P_fc[i] / 600 kW)^2`

其中 `Delta P_fc[0] = P_fc[0] - P_fc_executed_previous`，后续差分为相邻计划值。首项必须引用上个周期实际执行的燃料电池功率，而不是上一计划中的预测值。

SOC 死区函数为：

`phi(s) = ((0.40-s)/0.60)^2`，当 `s < 0.40`；

`phi(s) = 0`，当 `0.40 <= s <= 0.60`；

`phi(s) = ((s-0.60)/0.60)^2`，当 `s > 0.60`。

`J_soc = (1/N) sum_i phi(SOC[i])`。完整目标严格为：

`J = q_base J_base + q_smooth J_smooth + q_soc J_soc`。

不存在额外惩罚项或隐藏权重。三个方法尺度固定为 `600 kW / 600 kW / 0.60`；
其中 `0.60 = SOC_max-SOC_min = 0.80-0.20`。权重和为 1 只定义相对偏好，
不能替代这些 objective magnitude normalization。

## 功率平衡、能量学与硬约束

航行孤岛母线采用以下符号约定：

`P_batt_bus[i] = P_load_hat[i] - P_fc[i]`。

`P_batt_bus > 0` 表示电池放电，`P_batt_bus < 0` 表示充电。功率平衡由该定义逐步精确构造，不通过软惩罚近似。

SOC 递推只调用 `v2.models.battery_energy.next_soc`，并要求经过来源校验的 `BatteryEfficiency`。时间输入为秒，电池额定能量为 kWh；控制模块不复制或猜测充放电效率。

每个预测步必须同时满足：

- `0 <= P_fc <= P_fc_rated`；
- `P_batt_charge_min <= P_batt_bus <= P_batt_discharge_max`，其中充电下界为负数、放电上界为正数；
- 若配置了 `P_fc_ramp_per_step`，则 `abs(Delta P_fc) <= P_fc_ramp_per_step`；
- `0.20 <= SOC[i] <= 0.80`。

`[0.20,0.80]` 是物理硬约束；`[0.40,0.60]` 只是 SOC 软目标的零惩罚工作区间，
不会进入约束集合。求解结果不会裁剪燃料电池功率、电池功率或 SOC；成功结果还会经过独立物理残差复核。

## 参数状态与来源边界

`PlantConfig.research_simulation()` 中有来源记录的研究仿真值
`P_fc_rated=600 kW`、`E_batt=624 kWh`、`P_batt_min=-624 kW`、
`P_batt_max=+1248 kW`，均来自 Yang et al. (2026) Table 6。用户于
2026-09-22 明确批准将该组电池边界用于本次 objective-scale audit。它们属于当前
研究仿真 MPC 配置，不得表述成 12 簇、约 1806 kWh 原船硬件边界；原船技术规格
本身仍只给出系统额定输出不低于 900 kW。

本次 objective-scale audit 由用户于 2026-09-22 临时指定
`tau_LPF=90 s`。在名义 `Ts_MPC=30 s` 下对应
`alpha=exp(-30/90)=0.7165313106`。该值只解除本次审计的参数阻塞，来源分类为
`user_approved_provisional_audit_parameter`；它不是两篇论文直接给出的数值，也不
自动成为正式训练参数。

以下参数或证据尚未冻结，当前正式训练状态为 **NO-GO**：

- 正式训练使用的 `tau_LPF`；
- 正式 `Ts_MPC/N/M` 选择、最终 DQN state/action catalog、退化归一化、reward scale
  及最终 catalog 的 solver robustness 证据。

真实 Train objective-scale audit 已在 6 个代表 case、完整 36 个候选 action 上完成
216 次求解，active-P95 `scale_ratio=1.827863`，其独立 gate 为
**PASS / VERIFIED**。该结果不提升上述其他 gate，也不授权正式训练。

这些参数只能在 Train 切分上选择、校准和审计。Validation/Test 不得用于选择它们。代码要求显式配置，避免将临时试验值提升为方法事实。

爬坡硬约束与 `J_smooth` 的 600 kW 数值归一化严格分离。`MPCConfig` 允许
`fuel_cell_ramp_kw_per_step=None` 且默认关闭；只有调用方显式给出正的来源支持值
时才加入 hard ramp。当前 v2 baseline 在约 30 s supervisory 尺度没有可靠标定值，
因此不启用 hard ramp。仓库求解 smoke fixture 中的正数仅为测试输入。旧 v1 的
`48 kW/s` 不再作为 `J_smooth` 分母，也没有机械乘以 30 s 生成
`1440 kW/step` 的新硬约束。

## 确定性求解与执行接口

控制器使用 SciPy SLSQP。冷启动由当前基准参考和可达功率区间确定，不使用随机数。控制器不保存隐藏的上次优化解；如需热启动，调用方必须显式传入长度恰为 `N_MPC` 的向量。`shifted_warm_start(previous_plan)` 明确执行左移并复制末值。

成功结果包含：完整 `N_MPC` 步的 `P_fc`、由平衡式导出的 `P_batt_bus`、预测 SOC、持久性负载、基准参考、三个目标分量、总目标值及求解器诊断。`first_command()` 只返回索引 0 的燃料电池功率、电池功率和下一 SOC；其余预测值不属于本周期可执行命令。

负载滤波状态采用事务式更新。控制器先无副作用地预览当前观测产生的预测；输入校验、优化器失败、解提取失败或后置物理复核失败均不消费该观测。只有完整且物理有效的 `MPCPlan` 构造成功后才提交一次当前观测。因此对同一控制时刻的失败重试不会重复推进滤波器。返回计划、命令、目标分量和诊断均为冻结记录；所有向量会防御性复制为等长、非空、有限的不可变元组。

## 错误语义

`PhysicalInfeasibilityError` 表示在调用优化器前即可证明的物理不可行，例如当前状态越过硬边界，或某个预测步不存在同时满足额定功率、电池功率和爬坡限制的功率区间，或可达 SOC 与硬区间不相交。

`NumericalSolverError` 表示数值求解失败，包括 SLSQP 未成功、元数据类型或取值无效、成功标志下返回不可读取/错误长度/非有限解，或结果未通过独立物理残差复核。`success` 只接受布尔值，`status` 和迭代次数只接受非布尔整数，且迭代次数不得为负。数值失败不会被笼统标成物理不可行，也没有可能违反约束的后备控制命令。两类异常均提供机器可读的 `kind`；数值异常仅在求解器状态已经通过整数校验时保留 `status`。
