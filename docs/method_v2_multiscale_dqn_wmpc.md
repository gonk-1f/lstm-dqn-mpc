# 多时间尺度 DQN-WMPC v2：下层非线性 MPC

## 方法边界与版本

本页只定义 `MPC_OBJECTIVE_VERSION = fc_base_smooth_soc_deadband_v1` 的下层控制器。下层 MPC 负责控制品质和物理可行性，不计算氢耗价格、设备价格、燃料电池或电池退化成本，也不使用上层 DQN 的经济奖励。上层 DQN 的动作只参数化三个正权重；它不改变下层目标项的定义。

当前冻结的基线时间尺度由 `TimeScaleConfig.provisional()` 给出：

- MPC 采样周期 `Ts_MPC = 30 s`；
- 预测长度 `N_MPC = 5`，即每次求解预测未来 5 个 MPC 步；
- 滚动时域每次只执行计划的第 0 步，下一控制周期重新观测和求解；
- `dqn_switch_steps` 是上层动作保持的实际控制步数，不参与下层计划长度。即使当前暂定值也为 5，它与 `N_MPC` 仍是不同概念。

## 因果负载与基准功率

在控制时刻 `k`，预测器 API 只接收一个当前标量负载 `P_load(k)`，没有接收未来实测序列的参数。因此下层只能使用不晚于 `k` 的信息。v2 首个预测器是持久性预测：

`P_load_hat(k+i|k) = P_load(k), i = 1,...,N_MPC`。

基准负载采用一阶低通滤波。时间常数 `tau_LPF` 的单位明确为物理秒，且必须显式传入并大于 0：

`alpha = exp(-Ts_MPC / tau_LPF)`，

`P_base(k) = alpha P_base(k-1) + (1-alpha) P_load(k)`。

预测域内可在持久性负载上递推同一个滤波方程，得到 `P_base_hat`。初次观测时以当前负载初始化滤波状态，避免凭空引入历史值。

## 三项目标函数

权重为精确字段 `(q_base, q_smooth, q_soc)`。三个值都必须是有限正数，且仅允许浮点表示误差范围内的和为 1。对计划 `P_fc[0:N]`：

`J_base = sum_i ((P_fc[i] - P_base_hat[i]) / P_fc_scale)^2`

`J_smooth = sum_i (Delta P_fc[i] / Delta_P_fc_scale)^2`

其中 `Delta P_fc[0] = P_fc[0] - P_fc_executed_previous`，后续差分为相邻计划值。首项必须引用上个周期实际执行的燃料电池功率，而不是上一计划中的预测值。

SOC 死区函数为：

`phi(s) = ((L-s)/SOC_scale)^2`，当 `s < L`；

`phi(s) = 0`，当 `L <= s <= H`；

`phi(s) = ((s-H)/SOC_scale)^2`，当 `s > H`。

`J_soc = sum_i phi(SOC[i])`。完整目标严格为：

`J = q_base J_base + q_smooth J_smooth + q_soc J_soc`。

不存在额外惩罚项或隐藏权重。`P_fc_scale`、`Delta_P_fc_scale`、`SOC_scale` 都必须显式给出，单位分别为 kW、kW 和无量纲 SOC，且为有限正数。

## 功率平衡、能量学与硬约束

航行孤岛母线采用以下符号约定：

`P_batt_bus[i] = P_load_hat[i] - P_fc[i]`。

`P_batt_bus > 0` 表示电池放电，`P_batt_bus < 0` 表示充电。功率平衡由该定义逐步精确构造，不通过软惩罚近似。

SOC 递推只调用 `v2.models.battery_energy.next_soc`，并要求经过来源校验的 `BatteryEfficiency`。时间输入为秒，电池额定能量为 kWh；控制模块不复制或猜测充放电效率。

每个预测步必须同时满足：

- `0 <= P_fc <= P_fc_rated`；
- `P_batt_charge_min <= P_batt_bus <= P_batt_discharge_max`，其中充电下界为负数、放电上界为正数；
- `abs(Delta P_fc) <= P_fc_ramp_per_step`；
- `SOC_min <= SOC[i] <= SOC_max`。

SOC 硬约束可配置，当前研究参考常用 `0.2..0.8`，但实现没有把它伪装成已冻结默认值。求解结果不会裁剪燃料电池功率、电池功率或 SOC；成功结果还会经过独立物理残差复核。

## 参数状态与来源边界

`PlantConfig.research_simulation()` 中有来源记录的研究仿真额定值 `P_fc_rated = 600 kW`、`E_batt = 624 kWh`，调用方可以显式传给 MPC。该来源不提供本实现所需的电池充放电功率边界或燃料电池逐步爬坡限制，因而不能从额定功率或容量推导这些值。

以下参数尚未冻结，当前正式训练状态为 **NO-GO**：

- `tau_LPF`；
- SOC 死区 `L/H` 与 `SOC_scale`；
- `P_fc_scale` 与 `Delta_P_fc_scale`；
- 电池充电下界与放电上界；
- 燃料电池每步爬坡限制；
- 若偏离研究参考，还包括 SOC 硬边界。

这些参数只能在 Train 切分上选择、校准和审计。Validation/Test 不得用于选择它们。代码要求显式配置，避免将临时试验值提升为方法事实。

## 确定性求解与执行接口

控制器使用 SciPy SLSQP。冷启动由当前基准参考和可达功率区间确定，不使用随机数。控制器不保存隐藏的上次优化解；如需热启动，调用方必须显式传入长度恰为 `N_MPC` 的向量。`shifted_warm_start(previous_plan)` 明确执行左移并复制末值。

成功结果包含：完整 `N_MPC` 步的 `P_fc`、由平衡式导出的 `P_batt_bus`、预测 SOC、持久性负载、基准参考、三个目标分量、总目标值及求解器诊断。`first_command()` 只返回索引 0 的燃料电池功率、电池功率和下一 SOC；其余预测值不属于本周期可执行命令。

## 错误语义

`PhysicalInfeasibilityError` 表示在调用优化器前即可证明的物理不可行，例如当前状态越过硬边界，或某个预测步不存在同时满足额定功率、电池功率和爬坡限制的功率区间，或可达 SOC 与硬区间不相交。

`NumericalSolverError` 表示数值求解失败，包括 SLSQP 未成功、成功标志下返回错误长度/非有限解，或结果未通过独立物理残差复核。数值失败不会被笼统标成物理不可行，也没有可能违反约束的后备控制命令。两类异常均提供机器可读的 `kind`；数值异常在求解器提供时保留 `status`。
