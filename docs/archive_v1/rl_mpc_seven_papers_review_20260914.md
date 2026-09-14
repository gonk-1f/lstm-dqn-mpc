# 强化学习调节 MPC 权重：七篇论文原文分析与项目建议

七篇论文共同支持“用强化学习选择控制器参数、由 MPC 求解受约束控制量”的分层设计，但没有给出统一的奖励函数，更没有证明某一种奖励适用于所有能量管理系统。对当前项目，最有价值的组合是：固定且可解释的闭环评价标准、具有真实行为差异的权重候选集，以及严格一致的动作执行和奖励结算时序。

本文解释调权机制、奖励公式及其涉及的参数；车辆和燃料电池底层模型中与这些问题无直接关系的全部材料常数不逐项展开。原文未给出的数值和实现细节标为缺失；数学推导、工程判断与论文结论分别表述。

## 1. 原文与方法对照

| 编号 | 论文 | 原文依据 | RL 实际调整对象 | 奖励主要依据 |
|---|---|---|---|---|
| [1] | Sun 等，2024，Adaptive parameterized model predictive control based on reinforcement learning: A synthesis framework | 本地 13 页出版原件；TU Delft 版本额外包含封面 | 通用框架允许调整模型、控制律、目标、约束和求解设置；交通案例实际调整控制律的密度设定值 | 一个 RL 周期内的实际性能代价；案例为负总车辆停留时间 |
| [2] | Zhai 等，2026，Model-based MPC with adaptive weights for tracked vehicle trajectory tracking | 本地 17 页 PDF及出版商全文 | DQN 输出四个权重的增量，共 81 个组合 | 横向误差和航向误差的负二范数 |
| [3] | Zarrouki 等，2021，Weights-varying MPC for Autonomous Vehicle Guidance: a Deep Reinforcement Learning Approach | 本地 IEEE 原件 7 页，逐页图像核对 | TD3 输出连续权重；实际降至 6 个自由参数 | 跟踪二目标高斯；舒适模式比较 MOG 与 MOCG |
| [4] | Zarrouki 等，2024，A Safe Reinforcement Learning driven Weights-varying Model Predictive Control for Autonomous Vehicle Motion Control | 指定 arXiv v1 全文；本地 8 页 PDF | PPO 从预优化 Pareto 权重库选择离散索引 | 两次切换之间的实测误差 RMS，经缩放后送入高斯函数 |
| [5] | Yuan 等，2025，Energy management and performance improvement for fuel cell hybrid electric vehicle with reinforcement learning-based model predictive control | 本地出版原件 16 页 | 两个 Q 表分别调整上层 3 个、下层 4 个权重 | 对应 MPC 预测最优序列代价的倒数 |
| [6] | Mehndiratta 等，2018，Automated Tuning of Nonlinear Model Predictive Controller by Reinforcement Learning | 本地作者全文 6 页 | 分两阶段、按 episode 搜索 NMPC 权重 | 误差、误差变化率、jerk、稳态误差的阈值分段评分 |
| [7] | Haspolat 与 Yalcin，2023，Energy Management of P2 Hybrid Electric Vehicle Based on Event-Triggered Nonlinear Model Predictive Control and Deep Q Network | MDPI 官方 PDF，28 页 | 一套 DQN 学触发；另一套 DQN 学电机权重并联动其他权重 | 前者跟踪与触发成本；后者发动机和电机效率 |

“RL 动作”“MPC 控制输入”和“RL 奖励”应始终分开。记 RL 观测为 \(s_t\)，动作对应参数 \(w_t=\pi(s_t)\)，则 MPC 计算

\[
U_t^*=\arg\min_U J_{\mathrm{MPC}}(U;s_t,w_t),
\]

执行首个控制量后，外部评价函数生成奖励。MPC 的 \(w_t\) 决定如何求控制；奖励的固定系数决定什么行为值得学习。二者可以包含相同物理指标，但不必取相同权重，也不必使用相同时间窗口。

## 2. Sun 2024：通用参数化框架，案例并非直接调 MPC 代价权重

原文 §3.2、式(11)、(13)、(17)、(18)允许 RL 调整

\[
\theta=[\theta_F,\theta_f,\theta_J,\theta_{\mathcal G},\theta_s].
\]

这些参数依次对应预测模型、状态反馈控制律、阶段及终端目标、约束集合、求解器设置。它们不是五个必须同时训练的权重。具体应用可只开放一个部分。[1]

式(13)的性能奖励为

\[
r_{k_{rl}}=-R(\bar x(k_{rl}),\bar u(k_{rl})).
\]

\(R\) 是外部性能代价；\(\bar x\) 是区间 \([k_{rl}T_{rl},(k_{rl}+1)T_{rl})\) 中测量的状态，\(\bar u\) 是该区间实际实施的输入。这里不是把当前 QP 最小目标数值直接加负号。若同时调整求解设置，式(18)另减 \(J_{\mathcal C}(a)\)，表示求解时间等计算成本。[1]

时间尺度必须区分：\(T_s\) 为模拟采样间隔，\(T_c\) 为控制输入更新间隔，\(T_p\) 为 PMPC 优化间隔，\(T_{rl}\) 为 RL 调参间隔。原文规定 \(T_c=m_1T_s\)、\(T_p=m_2T_c\)、\(T_{rl}=m_3T_p\)，\(m_i\) 为正整数；\(N_{p,s},N_{p,c},N_{p,o}\) 是同一预测长度按三种时钟计数的步数。[1]

交通案例 §4.2–4.3 的控制律为

\[
u_{rm,i}(k_c+1)=u_{rm,i}(k_c)+u_\theta(k_p)[\theta_f(k_{rl})-\rho_i(k_c)].
\]

\(u_{rm,i}\) 为匝道放行率，\(\rho_i\) 为下游密度，\(u_\theta\) 是 PMPC 优化的增益，DQN 选择 \(\theta_f\) 密度设定值。其范围为 15–40 veh/km/lane，共 11 个等距动作。案例奖励是该 RL 周期内负 TTS；TTS 表示全网车辆总停留时间，单位 veh·h。RL-PMPC 的 \(T_s,T_c,T_p,T_{rl}\) 分别是 10、60、300、1800 s，预测长度 900 s。独立 RL 基线的更新周期另为 60 s。[1]

**工程判断：**本项目可借鉴“以实际执行过程评价参数”的原则。不能把交通案例的效果写成“DQN 调代价权重已经验证”，也不能把其 1800 s 调参周期迁移到船舶功率控制。原文结尾仍将稳定性和递归可行性列为后续研究，参数限幅本身不是完整安全证明。

## 3. Zhai 2026：81 个权重增量动作

原文式(39)–(44)给出

\[
s_t=[y_e,\phi_e,v_e,W_{t-1}],\quad
W_t=W_{t-1}+\Delta W_t,
\]

\[
\Delta W_t=[\Delta Q_{y_e},\Delta Q_{\phi_e},\Delta Q_{v_e},\Delta R],
\quad \Delta W_{t,i}\in\{-0.1,0,0.1\}.
\]

所以观测为 3 个误差加 4 个旧权重，共 7 维；动作数 \(3^4=81\)。\(Q_{y_e},Q_{\phi_e},Q_{v_e}\) 分别惩罚横向、航向和速度误差，\(R\) 惩罚控制增量。小增量让相邻时刻权重变化受限，但累计权重仍需合法范围处理。[2]

奖励原式是

\[
r_t=-\sqrt{y_e^2+\phi_e^2}.
\]

\(y_e\) 为横向位置误差，\(\phi_e\) 为航向误差。奖励中没有显式出现 \(v_e\)、控制增量或动作选出的四个权重。论文称其为负 RMS，但展示的式子是二范数，没有时间平均或分量平均因子。[2]

表2给出学习率 0.001、折扣率 0.99、1000 episodes、batch 64、replay 50,000、探索率 1.0 降至 0.01。正文对更小学习率的讨论是一般建议，不应覆盖表中实际设置。[2]

**原文疑点与工程判断：**米和弧度直接相加使隐含优先级依赖单位，迁移时应固定各自尺度。式(44)始终非正，但图14相关正文称总奖励趋近正的 800；若没有平移或其他奖励项，二者不能同时成立。文中还同时出现 DQN/DDQN 名称，展示的 max-target 更新式不足以证明实现是 Double DQN。复现应以代码或作者澄清为准，不自行补公式。当前项目的 84 个绝对权重动作与这 81 个增量动作机制不同。

## 4. Zarrouki 2021：连续权重、固定缩放、高斯与级联高斯

原文 §II–IV 最初有 11 个权重：

\[
Q=\mathrm{diag}(q_x,q_y,q_\psi,q_a),\quad
P=\mathrm{diag}(p_x,p_y,p_\psi,p_a),\quad
R=\mathrm{diag}(r_a,r_\omega,r_\vartheta).
\]

\(q_x,q_y,q_\psi\) 对应位置和航向误差；\(q_a\) 对应横向加速度；\(P\) 对应同类终端误差；\(r_a,r_\omega,r_\vartheta\) 对应纵向加速度指令、方向盘目标角速度和路径速度误差。实际取 \(P=Q,q_x=q_y\)，只剩 6 个自由权重，由 TD3 连续输出，正值下限 \(10^{-4}\)。论文将成本信号按物理跨度的一半缩放，并在训练中固定这些尺度。[3]

跟踪模式式(4)为

\[
R=A\exp\left[-\frac{e_{vel}^2}{2\sigma_v^2}-\frac{e_{lat}^2}{2\sigma_l^2}\right].
\]

一般 MOG 式(5)为

\[
R(x_1,\ldots,x_n)=A\exp\left[-\sum_{k=1}^n\frac{(x_k-x_{0,k})^2}{2\sigma_k^2}\right].
\]

\(A=1\) 是峰值；\(x_k\) 是外部评价信号，\(x_{0,k}\) 是目标中心；\(\sigma_k>0\) 是设计宽度，越小表示对该误差越严格。它不是论文从数据估计出的观测噪声。\(e_{vel},e_{lat}\) 分别为速度和横向误差。舒适模式加入横向加速度 \(a_{lat}\) 和纵向 jerk \(j_{long}\)，四个中心均为零。[3]

式(8)的级联 MOCG 则先评价跟踪和舒适两个组，再评价两组的得分偏离峰值：

\[
R_{\rm MOCG}=G_o\bigl(G_t(e_{vel},e_{lat})-1,\ G_c(a_{lat},j_{long})-1\bigr).
\]

这里用 \(G_t,G_c,G_o\) 区分原文重复使用的 \(R_2\)，属于解释性改写；减 1 是因为内层满分为 1，使满分组在外层的误差中心为 0。外层宽度应在得分单位下解释，不能不加转换地套用米或 m/s 的宽度。[3]

| 模式 | 速度宽度 \(\sigma_v\) | 横向宽度 \(\sigma_l\) | 加速度宽度 \(\sigma_a\) | jerk 宽度 \(\sigma_j\) |
|---|---:|---:|---:|---:|
| 跟踪训练 | 10 m/s | 0.5 m | 不参与 | 不参与 |
| 跟踪评价 | 2 m/s | 0.1 m | 不参与 | 不参与 |
| 舒适训练 | 7 m/s | 0.5 m | 1 m/s² | 1 m/s³ |

MPC 周期为 0.04 s，调权周期 10 s，每 episode 300 s；论文说明对区间内缓存信号取均值或极值来构造观测，不能统一说这一篇也是全部指标取 RMS。表III中 MOCG 的舒适性更好，但速度误差 RMSE 从 MOG 的 3.19 增至 4.87 m/s，说明结果包含取舍。[3]

**工程判断：**高斯奖励提供明确目标中心和容忍宽度，也会在远离目标时饱和接近零。它不能独立防止所有奖励漏洞；2021 年论文将切换系统的完整稳定性分析留作未来工作。对本项目，可以借鉴固定尺度与多目标独立评价，但不宜仅因形式漂亮就替换已有线性氢耗加平方项的执行评分。

## 5. Zarrouki 2024：先筛选权重库，再学习何时切换

原文 §III–V 的单个候选为

\[
\theta=[q_{x,y},q_\psi,q_v,r_j,r_\omega,L_1,L_2].
\]

前三项惩罚位置、航向、速度偏差；\(r_j\) 惩罚纵向 jerk，\(r_\omega\) 惩罚转向速率；\(L_1,L_2\) 对约束松弛施加线性和二次惩罚。先在直线和弯道场景上进行多目标贝叶斯优化，并检验全局可行性，再缩减密集 Pareto 候选，最终案例使用 26 组。PPO 的动作是其中一组的索引，不是每次重新输出七个任意实数。[4]

式(8)沿用多目标高斯，目标为两个零误差，但输入明确为切换窗口的实测 RMS：

\[
e_{lat}=\sqrt{\frac1n\sum_{i=1}^n e_{lat,i}^2},\quad
e_{vel}=\sqrt{\frac1n\sum_{i=1}^n e_{vel,i}^2},\quad n=T_{sw}/T_{s,sim}.
\]

\(T_{sw}=1.6\) s，\(T_{s,sim}=0.02\) s，所以每次奖励汇总 80 次测量；NMPC 离散间隔 0.08 s，预测长度与 RL 前瞻长度均 3.04 s。峰值 \(A=1\)，横向和速度缩放边界分别 [0,0.4 m]、[0,1 m/s]；正文同时给出宽度 0.1 m、0.5 m/s，并规定归一化值截断至 [0,1]。[4]

**复现注意：**正文同时谈归一化和带物理单位的 \(\sigma\)，不能只缩放误差却保持宽度含义不变。若以物理宽度为准，同步缩放应取 \(\tilde\sigma_{lat}=0.1/0.4=0.25\)、\(\tilde\sigma_{vel}=0.5/1=0.5\)；这是量纲一致的推导，不是对作者代码的确认。v1 的可行性探索系数正文为 0.8、表I为 0.9，也应保留差异。

**工程判断：**当前项目首先应检验候选权重产生的控制差异、可行性及跨工况表现。论文的安全措辞依赖筛选过程及实验范围；“每组固定权重均可行”不自动证明“任意切换序列均安全”。本项目 84 点网格未经相同筛选，不能继承该保证。引入未来信息时也应使用当时可获得的预测，不能使用事后真实负荷。

## 6. Yuan 2025：燃料电池双层 MPC 与两个代价倒数奖励

这是七篇中与燃料电池能量管理最直接相关的一篇，但其下层控制还涉及电堆电流、温度和冷却水泵，超出当前功率分配模型。[5]

### 6.1 两层目标与七个权重

上层式(17)–(22)：

\[
C_1=\sum_{i=1}^{N_p}J_1+\sum_{j=1}^{N_c}(J_2+J_3),
\]

\[
J_1=\omega_1\left(\frac{SOC-SOC_{ref}}{SOC_{max}-SOC_{ref}}\right)^2,
\quad J_2=\omega_2\left(\frac{P_{st}}{P_{st,rated}}\right)^2,
\]

\[
J_3=\omega_3\left(\frac{\Delta P_{st}}{P_{st,max}-P_{st,min}}\right)^2,
\quad\Delta P_{st}=\frac{P_{st}(k+1|k)-P_{st}(k|k)}{\Delta t},
\quad\sum_{i=1}^{3}\omega_i=1.
\]

\(\omega_1\) 管 SOC 偏离，\(\omega_2\) 管电堆功率，\(\omega_3\) 管功率变化。\(SOC_{ref}\) 为参考荷电状态，\(SOC_{max}\) 为上界；\(P_{st,rated}\) 为额定电堆功率，\(P_{st,max/min}\) 为上下界。\(N_p,N_c\) 分别为预测和控制步数，均取 5；\(\Delta t=1\) s。初始 SOC 取 0.6。这里 \(P_{st}^2\) 是电堆功率惩罚，不是物理氢耗查表值。按展示公式，变化率除以功率跨度还带时间量纲，改变采样周期时不能忽略这一点。[5]

下层式(35)–(39)对以下四项按各自预测/控制窗口求和：

\[
J_4=\omega_4\left(\frac{I_{st}-I_{st,min}}{I_{st,max}-I_{st,min}}\right)^2,
\quad J_5=\omega_5\left(\frac{T_{st}-T_{st,nom}}{T_{st,max}-T_{st,min}}\right)^2,
\]

\[
J_6=\omega_6\left(\frac{\eta_{st,nom}-\eta_{st}}{\eta_{st,nom}}\right)^2,
\quad J_7=\omega_7\left(\frac{P_{st}-P_{st,nom}}{P_{st,max}-P_{st,nom}}\right)^2,
\quad\sum_{i=4}^{7}\omega_i=1.
\]

\(I_{st}\) 是电堆电流，\(T_{st}\) 是温度，\(\eta_{st}\) 是效率；后缀 min/max 表示上下界，nom 表示参考值。四个权重依次管电流、温度、效率偏离、功率跟踪。下层控制量是电流与冷却水泵转速 \(r_p\)，其参考和允许区间由上层功率结合电堆性能图确定。[5]

### 6.2 状态与动作

上层状态为负荷变化率的四档与 SOC 相对参考值的两档。阈值 \(\delta\) 原文写 5 kW，但比较的是 \(\Delta P_{load}/\Delta t\)；严格按变化率理解应说明 kW/s，在其 1 s 离散下数值相同。下层状态为温度高低两档及效率比 \(\eta_{st}/\eta_{st,max}\) 的十档。动作编码 \(A_i=\lceil10\omega_i\rceil\)，上层每权重取 0.1–0.8，下层取 0.1–0.7，步长 0.1。[5]

若同时执行正值与和为 1 的约束，上层合法组合数是 \(\binom92=36\)，下层是 \(\binom93=84\)。这是对原文条件的组合计数推导。原文另报告动作表维度 \(8^3\) 与 \(7^4\)，可能包含未使用单元，但不能在无代码时断言无效组合如何处理。当前项目的 84 点恰好具有同一数学构造，四项物理含义却与该文下层不同。

### 6.3 奖励与复现风险

PDF 第9页 §3.2.2、式(51)明确是

\[
R_1=\frac1{C_1(u_1^*,x_1^*;\omega_1,\omega_2,\omega_3)},\quad
R_2=\frac1{C_6(u_2^*,x_2^*;\omega_4,\omega_5,\omega_6,\omega_7)}.
\]

星号表示当前 QP 求出的最优控制和状态序列。它不是 \(1/(1+C)\)，不是实测氢耗倒数，也不是两个 MPC 统一使用的实际首步评分。表述的学习率 \(\alpha=0.99\)、折扣率 \(\gamma=0.01\)，训练上限 500 episodes；因此按式子，其未来奖励影响很小，不能将二者凭经验交换。[5]

原文至少有三项实现细节需要澄清：式(52)第二个 Q 表更新的减项印为 \(Q_1\)；式(44)矩阵中权重对应关系与式(36)–(39)不一致；标准 QP 展开式省略的常数是否在奖励中恢复未交代清楚。常数虽不影响固定动作下的 argmin，却会改变作为 reward 的最优代价值。\(1/C\) 在 \(C\to0\) 时也需明确处理。

**工程判断：**可以借鉴分层目标和随工况调权，但本项目主奖励不宜退回动作自身加权代价倒数。表2的氢耗从 ECMS/GMPC/DMPC 到 QMPC 的改善在两条工况上并不相同，不能直接引用正文汇总百分比当作普遍收益；比较还应核对终端 SOC 的可比性。

## 7. Mehndiratta 2018：按控制要求构造分段评分

该文使用增量式、按 episode 的 RL，式(12)更新

\[
Q_{n+1}(s,a)=Q_n(s,a)+\frac1n[R_n-Q_n(s,a)].
\]

\(s\) 是悬停或移向设定点的飞行模式，\(a\) 是整组 NMPC 权重；\(n\) 为访问次数，\(R_n\) 为该次评价。它不是 DQN 每个控制步使用的 Bellman 目标。[6]

式(14)–(16)把位置误差 \(e\)、误差变化率 \(\dot e\)、jerk 与稳态误差 \(e_{ss}\) 分别转成评分。对任一非负指标 \(E\)，

\[
R_E=\begin{cases}
100,&E<E_{max}/100,\\
E_{max}/E,&E_{max}/100\le E\le E_{max},\\
-100,&E>E_{max}.
\end{cases}
\]

\(E_{max}\) 是应用允许阈值；100 是封顶分数，避免误差接近零时倒数发散；−100 是超阈值惩罚。式(14)只把综合评分写作 \(f(R_e,R_{\dot e},R_{jerk},R_{e_{ss}})\)，不能未经额外依据宣称它就是四项等权相加。[6]

| 最大允许指标，按 x/y/z 排列 | 阶段1：悬停 | 阶段2：移动至设定点 |
|---|---|---|
| 位置误差，m | 0.1 / 0.1 / 0.05 | 0.25 / 0.25 / 0.1 |
| 速度误差，m/s | 0.01 / 0.01 / 0.005 | 0.2 / 0.2 / 0.12 |
| jerk，m/s³ | 0.5 / 0.5 / 0.5 | 5 / 5 / 8 |
| 稳态位置误差，m | 0.1 / 0.1 / 0.05 | 0.05 / 0.05 / 0.03 |

两阶段各 50 episodes；第二阶段在第一阶段理想权重附近 10% 范围搜索。NMPC 的 \(W_x\) 对应三轴位置和速度误差，\(W_u\) 对应姿态角及总推力；终端权重设为 \(1.3W_x\)，预测步数 30。逐渐加入指标的四组消融说明：只看位置误差可能留下瞬态或平滑性差的权重。[6]

**工程判断：**本项目可以借鉴“先定义合格行为，再评价权重”的流程。表中的无人机阈值及 −100 奖励没有船舶设备依据；不能据此认定电池连续大功率阈值或直接确定 terminal failure penalty。

## 8. Haspolat 2023：调权奖励与事件触发奖励分开读

### 8.1 能量分配调权

第一层 MPC 跟踪车速、给出总需求转矩；第二层 NMPC 在约束下分配电机和发动机转矩。式(36)的目标按预测窗口累加发动机、电机、电池的三个功率相关项，分别由 \(W_{Eng},W_{Mot},W_{Batt}\) 加权。§4.2 只训练一个参数：

\[
W_{Mot}=a,\quad W_{Batt}=a,\quad W_{Eng}=5-a,\quad a\in[0.1,5].
\]

联动关系把三个权重降为一个自由度；所有权重之和是 \(5+a\)，不是常数 5。原文把动作写成实数区间，却称使用 DQN，没有清楚给出用于 DQN 的离散候选表，不能擅自解释为“每隔 0.1 的 50 个动作”。[7]

其观测来自未来四个参考速度、实际速度和阻力换算的轴侧需求转矩。式(55)中的 \(a_{Ref},a_{Veh}\) 为参考与实际加速度，\(m\) 为质量，\(F_{TotR}\) 为总阻力，\(r\) 为车轮半径，\(N_{Gear},N_{FDR}\) 为挡位和主减速传动比。该式的求和与差值括号应按 PDF 核对，不应自动改成四维向量或平均误差。

PDF 第18页式(57)是

\[
r_k=\begin{cases}
f(Mot_{Eff},Eng_{Eff}),&(Mot_{Eff}+Eng_{Eff})/2>Lim_{Eff},\\
-f(Mot_{Eff},Eng_{Eff}),&\text{其他情况}.
\end{cases}
\]

\(Mot_{Eff},Eng_{Eff}\) 为电机和发动机效率；\(Lim_{Eff}\) 为效率阈值；\(f\) 是通过查表把效率表现转为分值的函数。正文说低于阈值的差值还乘调整参数，但未提供足以重建查表的数值，也未明确给出该调整系数和阈值。因此可以理解设计思想，不能声称奖励已可逐数复现。[7]

### 8.2 事件触发是另一个学习任务

用 \(d_k\) 表示式(47)–(50)中由当前车速和前四个参考车速构成的差值指标，奖励结构为

\[
r=-w_1|d_k|+\mathbf1_{|d_k|\le Lim_V}\,g(Lim_V-|d_k|)-w_3a_k.
\]

这是对原文查表项的解释性改写：原文式(50)用 \(w_2\) 表示标定，文字称其为一维查表，此处以 \(g\) 明确它未必是常数乘法。\(w_1\) 控制跟踪惩罚，\(Lim_V\) 定义加分区域，\(w_3\) 是触发代价；\(a_k\in\{0,1\}\)，1 表示重新触发 MPC。它与上一节的调权动作 \(a\) 含义不同。[7]

**工程判断：**发动机/电机效率的简单平均不是系统能效；设备能量流差异、SOC 净消耗均可能改变结论。表10的 3.61 和 2.86 是效率百分点增加；52.01% 是触发次数减少。由表中 13.881 s 与 7.815 s 计算的 self-time 降幅约 43.70%，不能把触发数降幅直接写作运行时间降幅。对本项目，可借鉴减少调参自由度和独立物理反馈，但不能直接移植效率查表或触发阈值。

## 9. 奖励形式为什么不能只看“值越大越好”

以下为数学分析，示例不是论文实验结果。

### 9.1 自己改评分权重，再比较分数，会改变评价标准

设同一段物理轨迹的四项归一化成本是 \(c=(10,1,1,1)\)。采用两个合法权重

\[
w_A=(0.7,0.1,0.1,0.1),\quad w_B=(0.1,0.7,0.1,0.1),
\]

便得到 \(J_A=7.3,J_B=1.9\)。若奖励为 \(1/J\)，则分别约 0.1370 和 0.5263；若为 \(1/(1+J)\)，则约 0.1205 和 0.3448。物理结果完全一样，奖励却变化。这说明“和为1”排除了整体缩放，但没有解决相对权重改变评分尺度的问题。

实际不同权重也会改变控制轨迹，因此不能据此断言某篇论文一定发生奖励利用；它揭示的是需要审计的机制风险。本项目更合理的要求是：给定完全相同的实际执行记录，外部评分应完全相同。

### 9.2 单步单调变换不保证长期策略相同

在 \(C>0\) 时，\(-C\)、\(1/(1+C)\)、\(\exp(-C)\) 对单次成本排序相同。但两步累计不同：成本序列 A=(0,4)、B=(1.5,1.5)，负成本和偏好 B；\(1/(1+C)\) 的和分别为 1.2 与 0.8，偏好 A；指数得分和约为 1.0183 与 0.4463，也偏好 A。风险偏好与时间分布被改变了。

同理，“线性氢耗加二次惩罚”不等于“对它再取指数”。如果要做高斯奖励消融，应将它作为不同优化目标验证，而不是只称为数值归一化。

### 9.3 归一化尺度就是隐含优先级

对于 \(\tfrac12(e/s)^2\)，把 \(s\) 减半会把同一误差的惩罚放大四倍。所谓等权，只是缩放后的等权；不代表设备寿命、经济损失或安全风险已经等价。MPC 决策权重、reward 效用系数、物理尺度、硬约束必须分别记录。

## 10. 当前项目实际实现及参数解释

代码快照为 `ba81281da8236f0816d352a51ccfd7d176baabaf`，以下来自本次直接读取的代码。这里只描述已实现逻辑，不把历史单测记录当作训练收敛证明。

### 10.1 动作、观测和执行时序

`src/dqn/utils/action_mapper.py` 生成

\[
w=(n_h,n_b,n_s,n_f)/10,\quad n_i\ge1,\quad\sum n_i=10,
\]

共 84 个绝对权重动作，按字典序固定编号。依次对应氢耗、电池功率平方、SOC 偏离平方、燃料电池功率变化平方。它们是控制器候选，并不因位于正单纯形就成为 Pareto 优选集。

`state_builder.py` 使用 7 维因果观测：当前 SOC、上一执行 FC 功率、上一电池功率、当前负荷、向后负荷差、近10 s与近60 s平均负荷。分别使用项目固定物理尺度归一化。`DQNTrainConfig.state_normalization_enabled=False` 只表示额外在线统计归一化默认关闭，不表示这7项完全没归一化。

`dqn_mpc_weight_env.py` 中正式环境当前用当前负荷持续预测未来 t+1…t+6。FC 执行 MPC 首步；电池补偿执行时的实际负荷差额；随后更新实际 SOC 并计算奖励。这一实现事实应优先于仓库名字中出现的 LSTM，不能写成“当前正式 DQN 状态已经包含未来 LSTM 预测”。

### 10.2 当前外部奖励

`src/dqn/utils/reward.py` 的实现为

\[
r_t=-\left[h_t+\frac12b_t^2+\frac12s_t^2+\frac12f_t^2\right],
\]

\[
h_t=\frac{\dot m_{H_2}^{Dp0}(P_{fc,t}^{exec})}{\dot m_{H_2}^{Dp0}(600)},\quad
b_t=\frac{P_{batt,t}^{exec}}{624},
\]

\[
s_t=\frac{SOC_{after,t}-0.55}{0.05},\quad
f_t=\frac{P_{fc,t}^{exec}-P_{fc,prev}^{exec}}{48}.
\]

此处 t 表示被评分的执行步；环境的决策 t、执行 t+1 索引见上一节。函数不接收动作编号、预测 SOC 或 MPC 最优代价，因此相同执行记录具有相同评分。

| 参数或量 | 当前含义 | 不能误认为 |
|---|---|---|
| \(\dot m^{Dp0}_{H_2}\) | 既有 Dp0 物理氢耗曲线，单位 g/s | QP 内部的二次拟合目标值 |
| 600 kW | FC 总额定/上限功率，也是氢耗参考点 | 电池容量或普适燃料电池规格 |
| 624 kW | 电池功率归一化参考 | “超过即违反放电硬约束”的阈值 |
| 624 kWh | 电池能量容量，用于 SOC 更新 | 上一行功率尺度的单位 |
| 0.55 | 固定 SOC 参考值 | 当前任务自动跟随初始 SOC |
| 0.05 | SOC 软尺度；偏离5个百分点时 \(|s|=1\) | 统计测量噪声或认证安全边界 |
| 48 kW/s | FC 变化率限制；在1 s步长下分母为48 kW | 任意步长都直接除48 |
| 三个 1/2 | 平方项的固定效用系数 | 七篇论文联合证明的最优系数 |
| \(T_s=T_{sw}=1\) s | 每执行一步结算一次奖励 | 一次评价整个6步预测计划 |
| SOC 0.20–0.80 | 正式硬范围 | 0.50–0.60软区间没有代价 |
| 电池充/放电上限 624/1248 kW | 正式配置中的非对称功率界限 | reward 对充放电必然采用不同平方系数 |

当前 SOC 惩罚围绕0.55连续计算；即使 SOC 处于0.50–0.60内部，只要不等于0.55仍产生代价。功率平方项是应力代理，没有温度、电流、循环退化标定时不能把其数值解释成真实寿命损失。

MPC 内部也有四类指标，但在6步预测上累加并使用动作权重；氢耗使用便于 QP 的二次近似。外部评分使用实际首步和 Dp0 曲线，二者作用不同。正式配置中 `q_terminal_soc=0`；最终预测步仍含常规 SOC 阶段项，因此“没有额外终端代价”不等于“末步完全不关心 SOC”。

### 10.3 当前最值得注意的时间尺度

默认 \(\gamma=0.99\) 时，在1 s动作周期下折扣半衰期约为 \(\ln(0.5)/\ln(0.99)=68.97\) s，常用有效长度近似为100步。它不是数十分钟或整航次能量约束的替代。

按当前理想能量更新式，100 kW 持续放电1 s使624 kWh电池 SOC 下降约 \(4.45\times10^{-5}\)，即0.00445个百分点。若恰好从0.55起步，该一步产生的 SOC 平方半项仅约 \(3.96\times10^{-7}\)。这说明 SOC 的即时动作差异可能很弱；不能仅看到归一化偏差达到1时罚0.5，就认为一步奖励已经充分表达长时能量价值。

**工程判断：**这是需要闭环辨识的信用分配问题，不能据此直接把 \(\gamma\) 改大、SOC尺度改小或调权周期改长。每一项都会改变学习及控制行为，应保持其他条件固定并做 Train 内对照。

## 11. 面向本项目的下一步建议

| 优先顺序 | 建议工作 | 验收证据 |
|---|---|---|
| 1 | 保持统一执行评分，审计正常执行与失败终止的 reward/Q/TD 尺度 | 同一轨迹评分不随 action ID 改变；实际违规、预测失配、数值失败分别记录 |
| 2 | 对现有84动作做固定动作闭环和状态局部动作差异分析 | 各工况可行性、氢耗、SOC净变化、功率峰值、首步控制差异与动作冗余 |
| 3 | 比较动态调权是否优于最强固定权重和简单因果规则 | 相同预测、物理模型、初始状态、动作库与统一评价标准 |
| 4 | 按项报告收益，做有针对性的消融 | 去除SOC、电池功率或FC变化项后的真实行为变化，而非只比较总reward |
| 5 | 在确有短视证据时研究更长信用分配 | 固定Train工况比较折扣率、回报跨度；若改变调权周期，同步定义区间累计奖励 |

当前 `DQNTrainConfig` 的 `terminal_failure_penalty` 和标定记录默认均为 `None`；代码要求有 Train-only 标定证据才允许正式训练。这是已存在的实现条件。文献没有提供可直接替代本项目标定的处罚数值。

对于负阶段奖励，若提前失败仅结束轨迹而没有合适处理，可能因少累计负分而比正常完成更有利。若有可信阶段成本上界 \(c_{max}\)、剩余步数 H 和折扣率 \(\gamma\)，剩余折扣成本上界可写为

\[
c_{max}\frac{1-\gamma^H}{1-\gamma}.
\]

这可以帮助理解处罚尺度，但本项目目前没有在本文中求得这种可信全域界，不能据此填一个未经检验的大数。数值求解失败也不能自动解释成真实物理违规来处罚策略。

氢耗比较至少同时报告起末 SOC；用掉更多电池能量的策略可能显得“节氢”。电池功率平方、FC功率变化平方可以作为可解释指标，但若要宣称寿命收益，需有独立设备退化模型。奖励提高、loss下降、约束可行和真实闭环收益分别回答不同问题。

推荐保留已有84动作和统一执行奖励作为可复现实验基线，先证明不同工况确实需要不同动作，再研究筛选、增量调权或连续调权。本文的建议尚未转成控制器修改或训练结果。

## 12. 通用参数阅读表

| 符号 | 在 RL 中的含义 | 在这些论文中容易混淆的另一含义 |
|---|---|---|
| \(Q(s,a)\) | 状态动作价值，即期望未来回报 | MPC 的状态权重矩阵 Q |
| \(\alpha\) | Q更新或梯度学习率 | 贝叶斯优化获取函数也常写 alpha |
| \(\gamma\) | 未来奖励折扣率 | Yuan 的 \(\gamma_1,\gamma_2,\gamma_3\) 是代价归一化常数 |
| \(\epsilon\) | epsilon-greedy 随机探索概率 | PPO裁剪范围或BO可行性探索系数 |
| \(\theta\) | 策略网络参数，或被调控制器参数，需看上下文 | 车辆路径参数也使用 theta |
| \(N\) / batch size | 一次学习更新使用的样本数量 | MPC预测步数、轨迹样本数或训练总步数 |
| replay size | 可重放转移样本容量 | episode长度 |
| target update | 目标网络硬同步周期或软更新率 | MPC权重更新周期 |
| \(\lambda\)，PPO GAE | 优势估计的偏差与方差折中 | 目标中的物理效用系数 |
| entropy coefficient | 鼓励策略探索的系数 | 设备运行成本 |
| \(\sigma\)，高斯reward | 设计的奖励宽度/容忍尺度 | 概率模型或测量噪声的标准差 |

例如，2024安全调权论文 PPO 表II中的 \(\gamma=0.8\)、GAE \(\lambda=0.98\)、裁剪0.2和熵系数0.006，均是学习设置，不是 MPC 权重或车辆限制。它们也不能与 Yuan 的 \(\gamma=0.01\) 跨不同时间周期直接比较。

## 13. 原文来源与复查位置

1. Dingshan Sun, Anahita Jamshidnejad, Bart De Schutter. *Adaptive parameterized model predictive control based on reinforcement learning: A synthesis framework*. Engineering Applications of Artificial Intelligence 136 (2024), 109009. [DOI](https://doi.org/10.1016/j.engappai.2024.109009)，[TU Delft原文](https://repository.tudelft.nl/file/File_5f230ded-4f1a-456f-b3e4-4337cd8f90ec)。重点：正文页4–7、9–11，§3.2与§4.3。仓储PDF含额外封面，PDF页码比正文多1。
2. Li Zhai, Baichuan Shi, Chang Liu, Chengping Wang, Jianghaoyu Yan. *Model-based MPC with adaptive weights for tracked vehicle trajectory tracking*. Advances in Mechanical Engineering (2026). [出版商全文](https://journals.sagepub.com/doi/10.1177/16878132261426615)。重点：式(39)–(47)、算法1、表2、图14说明。
3. Baha Zarrouki, Verena Klös, Nikolas Heppner, Simon Schwan, Robert Ritschel, Rick Voßwinkel. *Weights-varying MPC for Autonomous Vehicle Guidance: a Deep Reinforcement Learning Approach*. European Control Conference (2021), pp.119–125. [IEEE](https://ieeexplore.ieee.org/document/9655042)，[作者公开版本](https://www.researchgate.net/publication/357540616_Weights-varying_MPC_for_Autonomous_Vehicle_Guidance_a_Deep_Reinforcement_Learning_Approach)。重点：PDF页3–6、式(4)–(8)、表III。
4. Baha Zarrouki, Marios Spanakakis, Johannes Betz. *A Safe Reinforcement Learning driven Weights-varying Model Predictive Control for Autonomous Vehicle Motion Control*. arXiv:2402.02624v1 (2024). [指定版本全文](https://arxiv.org/html/2402.02624v1)。重点：§IV–VI、式(3)、(7)、(8)、表I–III。
5. S. Yuan et al. *Energy management and performance improvement for fuel cell hybrid electric vehicle with reinforcement learning-based model predictive control*. International Journal of Hydrogen Energy 185 (2025), 151770. [DOI](https://doi.org/10.1016/j.ijhydene.2025.151770)。重点：PDF页6–10、13，式(17)–(22)、(35)–(39)、(45)–(52)、表2；出版商网页访问受限，以上分析依据本地完整原件。
6. Mohit Mehndiratta, Efe Camci, Erdal Kayacan. *Automated Tuning of Nonlinear Model Predictive Controller by Reinforcement Learning*. IEEE/RSJ IROS (2018), pp.3016–3021. [IEEE](https://ieeexplore.ieee.org/document/8594350)，[作者全文](https://www.researchgate.net/publication/330591563_Automated_Tuning_of_Nonlinear_Model_Predictive_Controller_by_Reinforcement_Learning)。重点：PDF页3–5，式(12)–(16)、表II–IV。
7. Cuneyt Haspolat, Yaprak Yalcin. *Energy Management of P2 Hybrid Electric Vehicle Based on Event-Triggered Nonlinear Model Predictive Control and Deep Q Network*. World Electric Vehicle Journal 14(6) (2023), 135. [DOI](https://doi.org/10.3390/wevj14060135)，[官方PDF](https://mdpi-res.com/d_attachment/wevj/wevj-14-00135/article_deploy/wevj-14-00135.pdf)。重点：PDF页15–18、24–26，式(36)–(57)、表10–11。

本地来源路径、SHA-256、版本与阅读范围另存于同目录 `rl_mpc_seven_papers_evidence_20260914.json`。每篇的机制和奖励均依据正文核对；“原文存在未说明参数”与“未取得全文”是不同问题。
