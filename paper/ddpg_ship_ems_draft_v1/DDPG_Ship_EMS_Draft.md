# Degradation-Aware Energy Management for a Marine Fuel Cell–Lithium-Ion Battery Hybrid Power System Using Deep Deterministic Policy Gradient

*Author names and affiliations: to be added.*

## Abstract

The economic performance of a marine fuel cell–battery propulsion system depends on both hydrogen consumption and the operating stresses imposed on its energy sources. This study develops an energy management framework based on deep deterministic policy gradient (DDPG) for a proton exchange membrane fuel cell (PEMFC) system coupled with a lithium-ion battery. A 600 kW aggregate fuel cell and a 624 kWh battery are represented using a system-efficiency map, an energy-based battery model, and literature-informed degradation accounting. Fuel cell voltage loss is associated with operating load, power changes, and aggregate starts, whereas battery wear is represented by state-of-charge- and current-weighted ampere-hour throughput. A continuous actor requests the aggregate fuel cell power; a deterministic physical projection enforces fuel cell ramp, battery power, and state-of-charge constraints. The reward combines actual hydrogen expenditure, incremental fuel cell and battery replacement costs, a signed battery-equivalent hydrogen term, and a quadratic state-of-charge penalty. The implementation uses native Stable-Baselines3 DDPG and independent deterministic validation after every 100 terminated training episodes. Reconstructed operating records are separated into 35 training, eight validation, and four held-out test voyages. A fixed-voyage, 500-episode experiment is specified as a prospective protocol, alongside the implemented random-voyage training mode. Numerical convergence, degradation, and comparative economic results remain to be established under the current reward definition.

**Keywords:** Marine hybrid propulsion; proton exchange membrane fuel cell; lithium-ion battery; degradation cost; deep deterministic policy gradient; energy management.

## Nomenclature

| Symbol | Definition | Unit |
|---|---|---|
| $t$, $\Delta t$ | Decision index and physical sampling interval | –, s |
| $P_{L,t}$ | Reconstructed onboard DC-bus load | kW |
| $P_{F,t}$, $P_{B,t}$ | Executed aggregate FC and battery bus power | kW |
| $P_{F,t}^{req}$ | Actor-requested absolute FC setpoint | kW |
| $E_N$, $V_N$, $Q_N$ | Nominal battery energy, voltage, charge capacity | kWh, V, Ah |
| $z_t$, $z_{ref}$ | Battery SOC and its reward reference | – |
| $\eta_F$, $\eta_c$, $\eta_d$ | FC system, battery charging, discharging efficiencies | – |
| $H_L$ | Hydrogen lower heating value | kJ $\mathrm{kg}^{-1}$ |
| $\Delta m_{H,t}$ | Actual FC hydrogen consumption in one interval | kg |
| $U_t$, $W_t$ | Cumulative FC voltage loss and weighted battery throughput | μV, Ah |
| $D_{F,t}$, $D_{B,t}$ | Raw normalized FC and battery life consumption | – |
| $K_F$, $K_B$ | Aggregate FC and battery replacement costs | CNY |
| $C_{H,t}$, $C_{F,t}$, $C_{B,t}$ | Hydrogen, FC wear, and battery wear costs | CNY |
| $C_{eq,t}$ | Signed battery-equivalent hydrogen cost | CNY |
| $C_{SOC,t}$, $q_t$ | SOC penalty and total reward cost | CNY |
| $o_t$, $a_t$, $r_t$ | Agent observation, normalized action, and reward | – |
| $\mu_\theta$, $Q_\phi$ | Deterministic actor and action-value critic | – |
| $\gamma$, $\tau$, $\sigma$ | Discount factor, target mixing factor, noise standard deviation | – |
| DPG, DDPG | Deterministic policy gradient; deep DPG | – |
| DP, ECMS, MPC | Dynamic programming; equivalent consumption minimization; model predictive control | – |
| EOL, FC, LB | End of life; fuel cell; lithium-ion battery | – |

## 1. Introduction

### 1.1 Background and motivation

Reducing greenhouse-gas emissions from shipping requires changes in both energy supply and onboard energy use. The 2023 International Maritime Organization strategy establishes a net-zero ambition for international shipping by or around 2050 and indicative total annual greenhouse-gas reduction checkpoints, relative to 2008, of at least 20% by 2030 and at least 70% by 2040, while striving for 30% and 80%, respectively [1]. These are sector-level ambitions rather than vessel-specific performance guarantees. For a particular vessel, the suitability of alternative propulsion depends on its route, installed capacity, energy infrastructure, and operating profile. Consequently, an energy management strategy should be assessed against a defined duty cycle and physical plant rather than presented as a universal decarbonization solution.

Hybrid electric architectures allow complementary energy sources to serve a common electrical demand. Guo et al. [2] review the relationship between marine hybrid-system configuration, control objectives, and intelligent energy management. For a hydrogen-fueled PEMFC–LB system, the fuel cell converts stored hydrogen into electrical energy, while the battery can absorb short-term power mismatches and supply transient demand. The battery may also permit the fuel cell to operate at a more favorable point on its system-efficiency curve. These opportunities are constrained by finite battery energy, conversion losses, fuel cell power limits, and the operating stresses experienced by both sources. Plant sizing and supervisory control therefore cannot be treated as independent determinants of operating cost; the ferry modeling and multi-objective optimization work of Wu and Bucknall [3] illustrates this interaction.

Hydrogen consumption alone does not characterize the economic consequences of a power split. A policy that repeatedly changes fuel cell power may reduce an instantaneous fuel-use term while increasing a voltage-loss proxy. Conversely, suppressing fuel cell fluctuations by extensive battery cycling can transfer wear to the battery. Aggressive discharge can also create an apparent saving by consuming initially stored electrical energy without accounting for the resulting SOC deficit. Actual fuel expenditure, source degradation, and the change in stored energy should therefore be reported separately, with an explicit boundary for each cost. A controller that is described as degradation-aware should identify which degradation effects are modeled, how they are monetized, and which physical aging processes remain outside its scope.

The present work addresses aggregate power allocation between one equivalent FC system and one equivalent LB system. It does not optimize the dispatch of individual fuel cell stacks or battery clusters. This distinction is important: an aggregate voltage-loss proxy can inform an upper-level power request, but cannot establish stack-specific start counts, heterogeneous aging, or the economic benefit of a particular lower-level dispatch algorithm. Such extensions require additional state information and a separately validated distribution model.

### 1.2 Literature review

Rule-based strategies remain attractive because their decisions are transparent and their computational requirements are modest. Power thresholds, SOC bands, and source-priority rules can be implemented without solving a large optimization problem online. Their effectiveness nevertheless depends on the selected thresholds and on the operating conditions for which those thresholds were designed. A rule that maintains SOC during one voyage may lead to persistent charging or battery depletion under another load profile. Reviews and marine fuel cell studies consequently place rule-based methods alongside optimization-based and learning-based alternatives rather than treating one category as universally preferable [2,3,4,5].

Equivalent consumption minimization converts the electrical contribution of a storage device into a fuel-equivalent quantity and minimizes an instantaneous objective. This approach gives a physically interpretable connection between fuel use and battery energy, but requires a suitable equivalence factor and a consistent treatment of efficiency and terminal SOC. Dynamic programming can supply a useful finite-horizon reference when the complete demand sequence and plant model are available; its computational and information requirements limit its direct use as an unrestricted online controller. Model predictive control repeatedly optimizes over a moving horizon and can enforce explicit operating constraints, while its performance depends on demand forecasts, model fidelity, horizon selection, and computational feasibility. Zhou et al. [5] combine time-scale considerations with source-lifespan accounting, showing why fuel cell and battery wear should be considered together. These approaches provide useful reference principles for evaluating a learned policy, but a fair numerical comparison requires identical demand, initial conditions, constraints, and cost boundaries.

Specific marine studies also demonstrate that these categories can be combined. Bassam et al. [6] develop a multi-scheme controller for a fuel cell passenger ship, selecting among state-based, equivalent-consumption, charge-depleting/charge-sustaining, and proportional–integral schemes. Kalikatzarakis et al. [7] formulate ship energy management with shore charging using ECMS and mixed-integer optimization, with dynamic programming providing a benchmark. For an all-electric ship with battery/flywheel storage, Hou et al. [8] use dynamic programming to investigate performance trade-offs and develop MPC with an SOC reference to address a short prediction horizon. These are informative control precedents, but their source configurations and optimization boundaries differ from the present FC–LB plant. Their published performance changes cannot be transferred directly to this study.

Reinforcement learning instead adjusts a control policy through interaction with an environment. Wu et al. [4] investigate cost-effective marine fuel cell–battery energy management using Double Q-learning. Their later work [9] uses a Double Deep Q-Network for near-optimal energy management of plug-in hybrid propulsion. The distinction is substantive: tabular Double Q-learning is not a deep neural-network implementation, and Double DQN is not DDPG. Deep Q-learning, exemplified by the foundational work of Mnih et al. [10], estimates action values for a specified set of discrete decisions. Discretization provides a tractable action-selection procedure, but the number and spacing of power levels determine the available control resolution. Enlarging that set can increase the burden of action-value approximation and exploration.

Zhu et al. [11] formulate a deep Q-learning strategy for a fuel cell hybrid ship with multiple objectives involving voyage cost, fuel cell life, and battery SOC. Their work is relevant because a fuel cell life metric is translated into economic consequences rather than retained only as a power-smoothing criterion. Jung and Chang [12] investigate a DQN-based strategy for a liquid-hydrogen-fueled hybrid electric propulsion system, including fuel-gas and balance-of-plant considerations. That article has a subsequent correction [13]; interpretations of its numerical tables should use the corrected record. Its discrete-action DQN and explicit supporting-system representation should not be conflated with the aggregate DDPG plant used here. Wu et al. [14] subsequently employ twin-delayed DDPG (TD3) in an intelligent marine hybrid-electric energy management framework with fuel cell cluster control. This provides evidence that continuous-action actor–critic methods are already relevant to marine applications; the novelty of the present work cannot reasonably be based on claiming the first use of a continuous actor in this domain.

DDPG combines deterministic policy-gradient theory [15] with neural actor–critic approximation, replay, and slowly updated target networks [16]. The actor directly generates a continuous command, avoiding a prescribed finite grid of fuel cell power levels. A critic evaluates that command and supplies the local action gradient used to improve the actor. In a related transport application, Wu et al. [17] study continuous-control energy management with traffic information for a series-parallel plug-in hybrid electric bus. This supports the methodological relevance of continuous control, while the vehicle topology and information set remain different from those of a marine FC–LB system.

Guo et al. [18] apply DDPG to a multi-energy cruise-ship case involving four diesel-generator power commands. Their actor–critic formulation, replay mechanism, and target-network treatment provide a useful application-level reference. Their fuel-cost and SOC-based reward, however, is associated with a different power-system configuration and does not define the present hydrogen and source-degradation accounts. In particular, a reward transformation taken as a mathematical form does not justify transferring a diesel fuel-cost model, an SOC coefficient, or a generator action mapping into a fuel cell system. Likewise, their reported average-reward curve must be distinguished from a raw episode return when voyage durations vary.

Table 1 summarizes the algorithmic distinctions most relevant to this study. It is intended to identify methodological relationships rather than rank publications using results obtained under different ships, datasets, and accounting assumptions.

**Table 1. Selected marine and continuous-control reinforcement-learning studies.**

| Study | Algorithm and application | Relevance and boundary |
|---|---|---|
| Wu et al. [4] | Double Q-learning; marine plug-in FC–battery system | Economic energy management; not Double DQN |
| Wu et al. [9] | Double DQN; marine plug-in propulsion | Discrete-action deep RL; not DDPG |
| Zhu et al. [11] | Deep Q-learning; FC hybrid ship | Voyage cost, FC life, and SOC objectives |
| Jung and Chang [12,13] | DQN; liquid-hydrogen hybrid propulsion | Corrected article; supporting-system model differs |
| Wu et al. [14] | TD3; hybrid-electric marine propulsion | Continuous control and cluster-level allocation |
| Wu et al. [17] | Continuous-control deep RL; hybrid electric bus | Related control theory; different vehicle and topology |
| Guo et al. [18] | DDPG; multi-energy cruise ship | Four diesel power actions; different fuel model |
| Present framework | Native DDPG; aggregate marine FC–LB plant | Joint cost accounting and explicit physical projection; performance pending |

### 1.3 Research gap and contributions

The literature establishes that reinforcement learning can be formulated for marine energy management; it does not remove the need for transparent modeling and reproducible comparison. Three practical issues motivate the present study. First, aggregate continuous power allocation must remain feasible under simultaneous source, ramp, and SOC constraints. Second, degradation penalties and monetary costs need a traceable connection: a voltage-loss fraction, a replacement-cost fraction, and a complete equipment CAPEX are different quantities. Third, battery energy valuation and source degradation must be separated so that apparent fuel savings cannot be attributed to unreported battery depletion. Existing work addresses parts of these issues, so the research gap is formulated as a specific implementation and evaluation problem rather than a general claim that joint degradation has never been considered.

The implemented contribution is an aggregate FC–LB environment in which a continuous DDPG request is passed through a physically defined admissible-power projection. Hydrogen use, FC voltage-loss cost, LB throughput cost, and signed battery-equivalent hydrogen are made explicit in a common CNY accounting convention. A second implemented contribution is an episode-count-based checkpoint and independent Validation8 workflow, which separates model inference from training updates. The planned empirical contribution is a controlled comparison of economic and degradation outcomes under the current reward definition, using both a fixed training voyage and the existing random-voyage mode. Evidence for improved cost, durability, convergence, or generalization is reserved for the numerical investigation in Section 4.

## 2. Modeling of the Hybrid Ship Power System

### 2.1 Hybrid power system configuration

The simulation represents an aggregate PEMFC system and an aggregate LB connected through a common DC-bus power balance. The selected 600 kW FC rating and 624 kWh battery energy are consistent with the capacity configuration reported by Yang et al. [19], whose physical description includes six 100 kW fuel cell units. Their architecture also contains a supercapacitor; no supercapacitor is represented here. The six-unit description motivates the reference-power scaling used below, but does not imply that six independently controlled stacks are simulated.

Figure 1 is reserved for a schematic of the modeled energy boundary. FC power and battery power are defined at the bus. Converter functions may be shown as interface blocks, but separate converter switching dynamics, an independently resolved FC auxiliary-power branch, and individually measured propulsion and hotel-load channels are not present in the current plant. The demand input is a lumped onboard electrical load. Accordingly, the diagram should not suggest that these unmodeled subsystems have been independently simulated.

**Fig. 1. Aggregate FC–LB hybrid ship power-system architecture.**

*Figure position reserved: aggregate FC system → DC interface → common bus; LB ↔ bidirectional interface ↔ common bus; bus → lumped onboard load. Dashed EMS signals indicate the FC setpoint and measured observation. Interface losses are represented only where specified in the equations.*

Table 2 lists the active plant parameters. Although several capacities and power limits agree with Yang et al. [19], the project uses an initial SOC of 0.6 and an admissible range of 0.2–0.8; these differ from the SOC settings in that source and are stated as project operating assumptions.

**Table 2. Aggregate plant parameters used in the current implementation.**

| Parameter | Value | Basis |
|---|---|---|
| FC aggregate rated power | 600 kW | Yang et al. [19]; active project plant |
| Reference FC rating for scaling | 100 kW | Aggregate-to-reference model assumption |
| FC ramp limit | 48 kW $\mathrm{s}^{-1}$ | Active project constraint |
| Battery nominal energy | 624 kWh | Yang et al. [19] |
| Battery nominal voltage | 432 V | Yang et al. [19]; fixed current-conversion basis |
| Battery bus-power limits | −624 to +1248 kW | Charge negative, discharge positive |
| Battery charge/discharge efficiencies | 0.95 / 0.95 | Active energy model; also reported in [19] |
| Initial / minimum / maximum SOC | 0.6 / 0.2 / 0.8 | Project operating assumptions |
| Initial FC power | 0 kW | Per-voyage reset |
| Decision interval | 1 s | Processed-data and environment contract |

The load records are reconstructed from recorded aggregate FC and battery powers. At the upstream source boundary, battery discharge has the opposite sign from that used by the present plant, so the source reconstruction subtracts the raw battery channel. Interpolation and subsequent one-second resampling produce the decision-grid demand. These inputs are therefore described as reconstructed operating-load records, not direct 1 Hz measurements of an independent load meter. Plant capacities adopted from [19] do not establish that these operating records were collected from the same vessel.

### 2.2 Fuel cell model

Let $P_{F,t}$ denote executed net FC bus power. Its admissible range and one-step ramp constraint are

$$0\leq P_{F,t}\leq P_F^N=600,\qquad |P_{F,t}-P_{F,t-1}|\leq 48\Delta t.\tag{1}$$

In Eq. (1), power is in kW and time is in seconds. The system-efficiency function is constructed from the project's supplied FC power–efficiency data using a shape-preserving piecewise cubic Hermite interpolating polynomial. If $p=P_{F,t}/6$ denotes the corresponding 100 kW reference-unit power, the active mapping is

$$\eta_F(P_{F,t})=\mathcal{I}_{PCHIP}(p),\qquad p=P_{F,t}/6.\tag{2}$$

The original table extends above the reference rating. The implementation first determines the 100 kW endpoint from the source interpolant, retains the support below that rating together with the endpoint, and maps it to 600 kW. Equation (2) is thus a normalized-load scaling assumption, not a newly measured 600 kW efficiency map. The measurement and test-condition provenance of the supplied workbook must be established before interpreting the curve as a validated marine system map.

The hydrogen consumed during a constant-power interval is

$$\Delta m_{H,t}=\begin{cases}\dfrac{P_{F,t}\Delta t}{\eta_F(P_{F,t})H_L},&P_{F,t}>0,\\0,&P_{F,t}=0,\end{cases}\qquad H_L=120000\ {\rm kJ\,kg^{-1}}.\tag{3}$$

Since 1 kW equals 1 kJ $\mathrm{s}^{-1}$, Eq. (3) gives kilograms directly. No additional factor of 3600 belongs in this form; the equivalent energy-unit heating value is 33.3333 kWh $\mathrm{kg}^{-1}$. The implementation treats FC output and efficiency at the selected system boundary and does not subtract a separately modeled compressor, pump, or converter parasitic load. Adding such a term without revising that boundary could count losses twice. Figure 2 will display the actual interpolated efficiency function and its corresponding hydrogen-consumption characteristic, with source points identified.

**Fig. 2. FC system-efficiency map and hydrogen-consumption characteristic.**

*Figure position reserved for curves generated from the active project efficiency data and Eq. (3); no efficiency or hydrogen-use curve has been invented.*

### 2.3 Lithium-ion battery model

The battery is represented by a lumped energy balance. Positive $P_{B,t}$ denotes power delivered from the battery to the bus; negative power denotes charging from the bus. Define the battery's internal-energy withdrawal rate as

$$P_{cell,t}=\begin{cases}P_{B,t}/\eta_d,&P_{B,t}\geq0,\\\eta_cP_{B,t},&P_{B,t}<0.\end{cases}\tag{4}$$

The SOC transition is then

$$z_{t+1}=z_t-\dfrac{P_{cell,t}\Delta t}{3600E_N},\qquad E_N=624\ {\rm kWh}.\tag{5}$$

Equations (4)–(5) include charging and discharging efficiencies once, at their respective energy-conversion boundaries. With $\eta_c=\eta_d=0.95$, discharge consumes more stored energy than the bus receives, while only a fraction of charging energy increases stored energy. The operating limits are

$$-624\leq P_{B,t}\leq1248,\qquad 0.2\leq z_t\leq0.8,\qquad z_0=0.6.\tag{6}$$

Equation (6) specifies the hard operating band, independently of the SOC reward reference introduced later. This is not an Rint or electrochemical battery model: open-circuit voltage, polarization, resistance, thermal evolution, and capacity fade feedback are not dynamically resolved. The nominal voltage introduced in Section 2.6 is a throughput-accounting conversion, not a voltage-state equation. Within each one-second interval, executed FC and battery powers are held constant. The model consequently addresses supervisory energy allocation rather than subsecond bus regulation.

### 2.4 Power balance and system constraints

The DC-bus balance is

$$P_{L,t}=P_{F,t}+P_{B,t}.\tag{7}$$

The battery supplies the residual in Eq. (7). There is no shore-power contribution during a voyage and no supercapacitor term. Battery energy-conversion losses are already included in Eq. (4), while a further independently parameterized converter efficiency is not applied to Eq. (7).

The FC request must satisfy battery energy availability as well as the nameplate limits. For a state $z_t$, define the admissible battery interval by

$$\begin{aligned}B_t^- &=\max\!\left[-624,-\dfrac{(0.8-z_t)3600E_N}{\eta_c\Delta t}\right],\\B_t^+&=\min\!\left[1248,\dfrac{(z_t-0.2)3600E_N\eta_d}{\Delta t}\right].\end{aligned}\tag{8}$$

The corresponding FC interval is

$$\begin{aligned}F_t^- &=\max(0,P_{F,t-1}-48\Delta t,P_{L,t}-B_t^+),\\F_t^+&=\min(600,P_{F,t-1}+48\Delta t,P_{L,t}-B_t^-).\end{aligned}\tag{9}$$

Equations (8)–(9) are the intersection of battery-power, SOC, FC-rating, and FC-ramp restrictions. If the interval is nonempty, the requested setpoint is projected onto it. If the intersection is empty beyond the implementation's numerical comparison tolerance, no physically admissible action exists under the modeled constraints. This produces a true task failure, rather than a reward penalty for an otherwise feasible control action. The physical ledger does not advance for such a failed attempted step.

### 2.5 Fuel cell degradation model

PEMFC degradation depends on the operating trajectory and on mechanisms not represented by hydrogen consumption alone. Open-circuit or low-load operation can expose electrodes to high potentials and promote catalyst or membrane degradation; high-load operation can aggravate reactant-transport and water-management stresses. Dynamic load changes can induce potential cycling, reactant starvation, and changing thermal or humidity conditions. Start–stop events can create local hydrogen–air boundaries and severe cathode corrosion. The reviews of Ren et al. [20] and Schmittinger and Vahidi [21] explain these mechanisms, but should not be treated as direct experimental calibration of every numerical coefficient used here.

The accounting follows the four-condition voltage-loss structure discussed by Zhou et al. [5], with numerical provenance also recorded in the project to the coastal-vessel analysis of Saponaro et al. [22]. It does not implement the ECSA model used in Yang et al. [19]. Define reference-unit power and the aggregate-start indicator as

$$p_t=P_{F,t}/6,\qquad \delta_t^{start}=\mathbf{1}\{P_{F,t-1}\leq0<P_{F,t}\}.\tag{10}$$

Using the reference power and start indicator in Eq. (10), the actual interval voltage-loss calculation is

$$\Delta U_t=\left[d_1\mathbf{1}_{0<p_t<80}+d_2\mathbf{1}_{p_t\geq80}\right]\dfrac{\Delta t}{3600}+d_3|p_t-p_{t-1}|+d_4\delta_t^{start}.\tag{11}$$

In Eq. (11), $d_1=10.17$ μV $\mathrm{h}^{-1}$, $d_2=11.74$ μV $\mathrm{h}^{-1}$, $d_3=0.0441$ μV $\mathrm{kW}^{-1}$, and $d_4=23.91$ μV per counted cycle. The two runtime branches are mutually exclusive: the code assigns the lower rate to every positive operating point below 80% of rating and the higher rate at or above 80%. It does not add both rates at high load. This broad lower-load branch is an implementation simplification and should not be interpreted as an independently calibrated near-idle threshold.

The variable-load quantity is the accumulated absolute change in reference power, measured in kW, rather than merely the number of action changes. Multiplying an unweighted event count by a coefficient in μV $\mathrm{kW}^{-1}$ would be dimensionally inconsistent. Changes into or out of zero power contribute to this transient term. The start–stop contribution is booked once at a zero-to-positive aggregate start; a later shutdown does not trigger an additional identical cycle charge. This convention is an aggregate accounting proxy, not a reconstruction of starts for six individually dispatched stacks.

Cumulative voltage loss and raw normalized FC life consumption are

$$U_{t+1}=U_t+\Delta U_t,\qquad D_{F,t}=\dfrac{U_t}{70000}.\tag{12}$$

The denominator in Eq. (12) is 70 mV, numerically equal to 10% of a 0.7 V single-cell reference. It is a frozen literature/model-based normalization, not a vessel-measured remaining-life estimate. For monetary accounting, the implementation limits cumulative charged life to one replacement equivalent:

$$\Delta\bar D_{F,t}=\min(D_{F,t+1},1)-\min(D_{F,t},1).\tag{13}$$

Equation (13) prevents repeatedly charging beyond one replacement within the cumulative ledger. It does not model a replacement event or restore the plant after EOL. The raw voltage-loss diagnostic should therefore be retained alongside the monetized fraction. Figure 3 will show the four contributions, reference-power mapping, and separate lifetime-normalization step.

**Fig. 3. Four-condition aggregate FC voltage-loss and incremental replacement-cost model.**

*Figure position reserved: mutually exclusive runtime branches, absolute reference-power change, and aggregate-start count feed cumulative voltage loss, followed by the 70 mV normalization and replacement-cost mapping.*

### 2.6 Lithium battery degradation model

Battery degradation is described by a stress-weighted charge-throughput surrogate, following the form used by Zhou et al. [5]. A directly traceable earlier formulation appears in Kwon et al. [23]. Ghaderi et al. [24] provide related health-conscious multi-stack energy-management context, while Xu et al. [25] and Han et al. [26] discuss broader battery degradation modeling and mechanisms. These references do not establish that the present coefficients and lifetime denominator are experimentally validated for the modeled marine battery.

For compatibility with the current implementation, the signed equivalent current is computed from battery bus power using a fixed nominal voltage:

$$\begin{aligned}I_t&=\dfrac{1000P_{B,t}}{V_N},\qquad V_N=432\ {\rm V},\\Q_N&=\dfrac{624000}{432}\ {\rm Ah},\qquad I_N=Q_N/(1\ {\rm h}).\end{aligned}\tag{14}$$

Thus positive current denotes discharge and negative current denotes charge. Equation (14) is the code's bus-power-based throughput convention; it is not the current of a resolved internal circuit. In particular, it is not obtained by applying the battery efficiency a second time to $P_{cell,t}$.

The SOC and current stress functions are

$$F(z_t)=1+3.25(1-z_t)^2,\tag{15}$$

$$G(I_t)=\begin{cases}1+0.45I_t/I_N,&I_t\geq0,\\1+0.55|I_t|/I_N,&I_t<0.\end{cases}\tag{16}$$

For a one-second constant-power interval, the weighted throughput increment is

$$\Delta W_t=F(z_t)G(I_t)|I_t|\dfrac{\Delta t}{3600},\qquad W_{t+1}=W_t+\Delta W_t.\tag{17}$$

The pre-transition SOC is used in Eq. (15). Within this empirical function, lower SOC increases the weighting over the admissible range; this mathematical behavior should not be generalized into a universal statement that high SOC is harmless. Likewise, Eq. (16) increases the throughput penalty with current magnitude and applies different factors to charge and discharge. Under Eq. (17), repeated cycling increases the absolute throughput even when the net change in SOC is small. Temperature, calendar aging, rainflow-resolved cycle depth, chemistry-specific mechanisms, and capacity/resistance feedback are not included.

The project's lifetime normalization and incremental charged fraction are

$$\begin{aligned}Q_{life}&=15000Q_N,\qquad D_{B,t}=W_t/Q_{life},\\\Delta\bar D_{B,t}&=\min(D_{B,t+1},1)-\min(D_{B,t},1).\end{aligned}\tag{18}$$

The factor 15000 in Eq. (18) is a fixed project lifetime-throughput assumption with secondary-literature provenance. It is neither a measured marine lifetime nor a claim of 15000 full charge–discharge cycles. Its independent quantitative calibration remains unresolved; sensitivity to this denominator is required before making strong absolute lifetime or replacement-cost claims. Figure 4 will distinguish the published weighting functions from this project-specific normalization.

**Fig. 4. SOC- and current-weighted LB throughput model and lifetime normalization.**

*Figure position reserved: bus-power-to-current conversion, pre-step SOC and signed-current weighting, cumulative weighted Ah, and the assumed lifetime-throughput denominator.*

### 2.7 Economic cost model

All interval costs are calculated in CNY, without conversion to fen. The physical operating-and-wear cost is

$$C_{phys,t}=C_{H,t}+C_{F,t}+C_{B,t},\qquad C_{H,t}=c_H\Delta m_{H,t}.\tag{19}$$

The hydrogen price in Eq. (19) is $c_H=21.9$ CNY $\mathrm{kg}^{-1}$, consistent with the parameter reported in Yang et al. [19]. It is not the 35 CNY $\mathrm{kg}^{-1}$ value reported in Zhou et al. [5]. These are scenario prices, not claims about current market quotations. Source-degradation costs are charged only for the newly consumed clipped economic-life fraction:

$$C_{F,t}=K_F\Delta\bar D_{F,t},\qquad C_{B,t}=K_B\Delta\bar D_{B,t}.\tag{20}$$

The replacement values used in Eq. (20) are

$$\begin{aligned}K_F&=\rho_Fc_FP_F^N=0.5\times3500\times600=1050000\ {\rm CNY},\\K_B&=c_BE_N=2000\times624=1248000\ {\rm CNY}.\end{aligned}\tag{21}$$

In Eq. (21), the FC fraction $\rho_F=0.5$ represents replacement expenditure relative to complete equipment CAPEX. Mylonopoulos et al. [27] assume this fraction because replacement of the degradable cells need not entail replacement of all supporting equipment, and cite the Hydrogenics presentation of Thomas [28]. The latter concerns electrolyser technology, so it is an indirect contextual source rather than independent verification that all marine FC installations have a 50% replacement fraction. The project adopts the fraction as an explicit economic assumption. It is applied once in $K_F$ and is not a reduction in physical degradation rate. Mylonopoulos et al.'s 15% voltage-loss replacement assumption is separate from the 10%/70 mV normalization used here; the two thresholds are not combined.

To account for the change in stored battery energy, the reward uses a signed equivalent-hydrogen term. Define the reference FC efficiency by

$$\eta_{eq,t}=\begin{cases}\eta_F(P_{F,t}),&P_{F,t}>0,\\\max_{0<P\leq600}\eta_F(P),&P_{F,t}=0.\end{cases}\tag{22}$$

The off-state reference in Eq. (22) is computed from the existing positive-power efficiency curve, whose present maximum is 0.638554. It is an approved project equivalence convention, not the zero-power efficiency and not a separately specified average efficiency. The battery-equivalent cost is

$$C_{eq,t}=c_H\dfrac{(z_t-z_{t+1})E_N}{\eta_{eq,t}(H_L/3600)}.\tag{23}$$

Equation (23) is positive for discharge and negative for charging. The SOC change already includes battery energy-conversion losses through Eqs. (4)–(5); no additional charge/discharge efficiency or DC/DC loss is multiplied into Eq. (23). The equivalent energy value is not an additional measured hydrogen purchase. It should therefore remain separate from $C_{phys,t}$ in economic reports. The four-term quantity used by the reward is

$$C_{4,t}=C_{H,t}+C_{F,t}+C_{B,t}+C_{eq,t}.\tag{24}$$

Because $\eta_{eq,t}$ varies with operating power, the sum of Eq. (23) does not generally telescope to a single terminal-SOC correction. It should not be presented as an exact lifetime cash-flow model or added again to a separate terminal recharging charge. Comparisons must disclose actual hydrogen, each wear cost, the signed equivalent term, and initial/final SOC individually. Table 3 summarizes the economic assumptions and their roles.

**Table 3. Monetary parameters and accounting boundaries.**

| Parameter or term | Active value or definition | Interpretation |
|---|---|---|
| Hydrogen price | 21.9 CNY $\mathrm{kg}^{-1}$ | Project scenario; distinct from Zhou's 35 |
| Complete FC CAPEX coefficient | 3500 CNY $\mathrm{kW}^{-1}$ | Equipment cost basis |
| FC replacement fraction | 0.5 | Applied once, not a degradation-rate factor |
| Battery replacement coefficient | 2000 CNY $\mathrm{kWh}^{-1}$ | Aggregate battery replacement basis |
| FC charged-life denominator | 70000 μV | Project 10% single-cell-reference convention |
| LB lifetime denominator | $15000Q_N$ | Project assumption requiring sensitivity analysis |
| Battery equivalent term | Signed Eq. (23) | Stored-energy valuation, not actual fuel purchase |
| Shore electricity during voyage | None | No voyage shore-energy cost |

## 3. Proposed Energy Management Strategy

### 3.1 Markov decision process formulation

A finite-horizon energy-management problem can be written as

$$\mathcal{M}=(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R},\gamma),\qquad J(\pi)=\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{T-1}\gamma^t r_t\right].\tag{25}$$

In Eq. (25), $\mathcal{S}$ denotes a sufficiently informative plant-and-voyage state, $\mathcal{A}$ the admissible command domain, $\mathcal{P}$ the transition law including the demand evolution and physical projection, and $\mathcal{R}$ the reward. The current implementation supplies the actor with a three-component observation rather than the complete underlying state. Voyage identity, elapsed time, future demand, and accumulated aging are not included in this observation. Reconstructed demand trajectories are exogenous, and identical observed loads need not imply identical future loads. Accordingly, the implementation is an observation-based approximation to this MDP and can be partially observed; full Markov sufficiency is not established by naming the observation a state vector.

An episode begins at the first executable interval of one complete voyage, with SOC reset to 0.6, FC power reset to zero, and fresh accounting accumulators. Successful completion occurs after the last executable interval. An empty feasible-power intersection produces a physical-failure termination. These are true environment terminations, whereas an external training-budget interruption is not evidence that the voyage was completed. This distinction is used in the bootstrap mask and in performance reporting.

### 3.2 DDPG algorithm

DDPG employs a deterministic actor $\mu_\theta$ and a scalar critic $Q_\phi$, together with target copies $\mu_{\theta'}$ and $Q_{\phi'}$ [15,16]. The actor maps the available observation to an action. During learning, exploration is introduced in normalized action space:

$$a_t=\operatorname{clip}\!\left(\mu_\theta(o_t)+\epsilon_t,-1,1\right),\qquad \epsilon_t\sim\mathcal{N}(0,\sigma^2).\tag{26}$$

Equation (26) describes the post-warm-up training policy. Before the configured learning-start threshold, SB3 samples random actions; during deterministic validation, the noise term is disabled. The action-noise standard deviation is constant in the current fixed configuration. There is no DQN-style epsilon-greedy probability or automatically decaying greedy-rate parameter.

The replay buffer stores tuples $(o_i,a_i,r_i,o_{i+1},d_i)$. For a mini-batch of size $N$, the one-step target is

$$y_i=r_i+\gamma(1-d_i)Q_{\phi'}\!\left(o_{i+1},\mu_{\theta'}(o_{i+1})\right).\tag{27}$$

Here $d_i=1$ for actual task termination, including physical failure, so such transitions do not bootstrap. The critic is updated by minimizing

$$L_Q(\phi)=\dfrac{1}{N}\sum_{i=1}^{N}\left[Q_\phi(o_i,a_i)-y_i\right]^2.\tag{28}$$

For the actor, deterministic policy-gradient theory gives the following sampled gradient [15]:

$$\nabla_\theta J\approx\dfrac{1}{N}\sum_{i=1}^{N}\left.\nabla_aQ_\phi(o_i,a)\right|_{a=\mu_\theta(o_i)}\nabla_\theta\mu_\theta(o_i).\tag{29}$$

In implementation, the equivalent actor loss is the negative mean critic value at actor-generated actions. Both networks use Adam. Replay permits off-policy reuse of past transitions; target networks reduce the rate at which the bootstrap target changes, although neither mechanism guarantees convergence with nonlinear function approximation. After an update, each target parameter follows

$$\theta'\leftarrow\tau\theta+(1-\tau)\theta',\qquad \phi'\leftarrow\tau\phi+(1-\tau)\phi'.\tag{30}$$

The implementation is native SB3 DDPG [29]. Its shared software components with TD3 do not make this experiment a TD3 controller: it uses one critic, an actor update for every gradient update, and no TD3 target-policy smoothing or minimum over twin critics. TD3's methodological distinctions are described by Fujimoto et al. [30]. The actor and critic each have two 256-unit ReLU hidden layers; the actor's final tanh produces one normalized command and the critic produces one action value. Figure 5 will illustrate these four networks and the update paths of Eqs. (27)–(30).

**Fig. 5. DDPG actor–critic, replay, and target-network architecture.**

*Figure position reserved: three observation inputs, 256–256 actor, one action output, observation–action critic with 256–256 hidden units, and soft-updated target copies. No twin-critic or TD3-specific path is included.*

### 3.3 Observation space and continuous action space

The actual three-dimensional observation is

$$o_t=\left[z_t,\ P_{L,t}/600,\ P_{F,t-1}/600\right]^{\mathsf{T}}.\tag{31}$$

The second component in Eq. (31) can exceed unity because demand may be supplied jointly by the FC and battery. It is not clipped to the FC rating. The third component is the previous executed FC power, which is required to evaluate the next ramp limit. No future-load predictor, remaining-voyage-time feature, separate stack-health vector, or cumulative wear state is supplied to the actor.

The scalar action domain and its physical mapping are

$$a_t\in[-1,1],\qquad P_{F,t}^{req}=300(a_t+1),\qquad P_{F,t}=\operatorname{clip}(P_{F,t}^{req},F_t^-,F_t^+).\tag{32}$$

Equation (32) is an absolute-power command, not a power increment. The residual battery power follows Eq. (7). This is continuous deterministic actuator projection rather than the discrete action masking used in a separate DQN route. SB3 replay records the normalized requested action, while the observation transition and economic ledger correspond to the projected physical action. The critic therefore learns the value of a requested action under the environment's actuator mapping. The request, executed setpoint, feasible interval, and projection indicator should be retained for evaluation so that policy saturation is visible.

### 3.4 Economic reward function

The active reward identity is `ddpg_economic_tanh_v1`. Its SOC penalty is evaluated at the next physical state:

$$C_{SOC,t}=\lambda_z(z_{t+1}-0.5)^2,\qquad \lambda_z=17,\qquad q_t=C_{4,t}+C_{SOC,t}.\tag{33}$$

There is no interval normalization of the SOC deviation and no additional multiplication by the one-second interval in Eq. (33). The four economic terms are those of Eq. (24); they are not replaced by an unpriced FC ramp penalty. FC power changes influence reward through the degradation account in Eq. (11). The previous independent ramp penalty has been removed from the active objective.

For a normal executed interval, and for a physically infeasible attempted transition, respectively, the reward is

$$r_t=\begin{cases}\tanh(1/q_t),&q_t>0\ \text{and physical step executed},\\1,&q_t=0\ \text{and physical step executed},\\-1,&\text{true physical failure}.\end{cases}\tag{34}$$

Costs in Eq. (34) are numerical values in CNY under the fixed monetary convention. The zero-cost branch is the right-hand limit of the positive-cost expression. Negative or nonfinite values of either the four-term cost or the total reward cost raise an implementation error rather than being replaced by an absolute value, silently clipped, or interpreted as a new physical-failure condition. On an infeasible attempted transition, the existing environment returns the failure reward and a terminal flag without executing an interval or advancing the physical cost ledger. This failure signal is a training reward, not a negative monetary expense. Normal completion adds no extra terminal bonus or terminal SOC-restoration charge.

For positive cost, Eq. (34) lies strictly between zero and one mathematically and decreases with increasing cost. Floating-point saturation can yield a value numerically equal to one for small positive costs. This transformation is monotonic per interval, but maximizing the sum of transformed, discounted rewards is not mathematically equivalent to minimizing total undiscounted CNY cost. It can compress distinctions among low-cost decisions. Consequently, convergence of reward alone cannot demonstrate an economic advantage; economic components and completion must be evaluated directly.

The SOC coefficient has a documented training-only calibration history. An existing deterministic baseline policy was run over Train35, yielding 448434 executed physical seconds, including two voyages that terminated before their full duration. The median of the four-term economic cost over those executed intervals was

$$\begin{aligned}C_{typ}&=0.17429534153424198\ {\rm CNY},\\\lambda_z^{cal}&=\dfrac{C_{typ}}{0.1^2}=17.429534153424198.\end{aligned}\tag{35}$$

The subsequently approved active coefficient is 17, not the exact calibrated value in Eq. (35). At a 0.1 SOC deviation, its penalty is 0.17 CNY. This calibration does not establish current-policy performance, contains no Validation8 or Test4 samples, and excludes unexecuted suffixes and failed attempted intervals. The baseline model's previous reward is not treated as evidence of convergence under Eq. (34).

### 3.5 Training, validation, and test procedure

The implementation uses the RL Baselines3 Zoo training infrastructure with the native SB3 DDPG algorithm, a single environment, and no VecNormalize wrapper in the active configuration. The project observation scaling in Eq. (31) is therefore distinct from learned running-statistic normalization. Table 4 gives the formal YAML baseline. Its target mixing factor and noise standard deviation are current fixed values; neither is presented as a finalized optimum for the new reward.

**Table 4. Current formal DDPG baseline and training budget.**

| Setting | Current value | Status |
|---|---|---|
| Discount factor $\gamma$ | 0.999 | Fixed baseline |
| Actor/critic learning rate | 0.001 | Adam |
| Batch size | 64 | Fixed baseline |
| Learning starts | 5000 environment steps | Random-action warm-up |
| Collection / gradient schedule | 32 steps / 1 gradient step | Single environment |
| Replay capacity | 1000000 transitions | Fixed capacity |
| Hidden layers | 256, 256 | Actor and critic, ReLU |
| Target mixing factor $\tau$ | 0.03 | YAML baseline; final selection pending |
| Gaussian action-noise $\sigma$ | 0.12 | Normalized action units; final selection pending |
| Training device | CUDA | Actual hardware/software reporting pending |
| Episode limit | 500 terminated episodes | Native episode-count callback |
| Step safety ceiling | 20000000 | Independent upper bound |
| Independent validation | Every 100 terminated episodes | All eight validation voyages |
| Seed in formal launcher | 42 | Per-trial reproducibility requires separate verification |

The dataset contains 35 Train, eight Validation, and four Test voyages. The partition is grouped by source parent record, so segments from a parent do not cross the three splits. The one-second V6 Train/Validation package preserves the processed voyage boundaries and does not re-interpolate the upstream data. Each executable interval uses its left-boundary load; the final boundary node is not an extra physical step. This avoids replacing a complete voyage with an arbitrary training window. Test demand is not needed for policy fitting, SOC-weight calibration, or hyperparameter selection.

In the implemented training mode, each reset samples one of the 35 Train voyages uniformly with replacement. The proposed controlled experiment will instead repeatedly use one selected Train voyage, internally identified as `parent_005_seg01`, for 500 episodes. It contains 9258 executable one-second intervals. This fixed-voyage selector is a planned extension, not an active feature of the current environment. If all 500 episodes completed, that design would contain 4629000 physical training intervals; early failures would change the realized count. The existing 20000000-step ceiling is only a safety budget and must not be reported as an equivalent definition of 500 episodes. The random-voyage mode is to be retained for a separate generalization-oriented comparison.

The episode counter advances when an environment episode terminates, including physical failures; it does not certify 500 successful complete voyages. At counters 100, 200, 300, 400, and 500, the callback saves a model checkpoint and launches independent deterministic evaluation on all eight Validation voyages in fixed order. In the current SB3 callback position, the snapshot is taken before insertion of the triggering transition into replay and before subsequent rollout-gradient updates; this timing is included in checkpoint metadata. The separate evaluation process loads the snapshot on CPU, disables exploration noise, calls deterministic prediction, and performs no learning or replay updates. Test4 is excluded from this periodic mechanism. Figure 6 will depict this separation explicitly.

**Fig. 6. Training–environment interaction and independent episode-triggered Validation8 workflow.**

*Figure position reserved: Train-only rollout and replay/update loop; episode-counter checkpoints; separate deterministic Validation8 process; held-out Test4 used only after selection.*

The current new-reward Optuna study is distinct from historical studies. At the inspected snapshot it contains completed and ongoing trials, but no finalized model-selection decision is established. Its current objective is Zoo's mean validation episode return, not a lexicographic completion-and-economic-cost objective. This limitation must be reported: a high objective value alone does not demonstrate the lowest full-voyage cost. Final selection should be reviewed against completion, coverage, separate economic components, and SOC, with Test4 reserved until that selection is frozen. Search-space bounds, sampler settings, per-trial model seeds, software versions, and the final chosen parameters remain experimental reporting items; old-reward Trial 18 is not substituted for this evidence.

For comparable learning plots, an episode return and a duration-normalized mean reward are distinct quantities:

$$R_k=\sum_{t=0}^{N_k-1}r_{k,t},\qquad \bar r_k=R_k/N_k.\tag{36}$$

Here $N_k$ is the number of returned training transitions. For a completed one-second voyage it equals the executed duration in seconds, making Eq. (36) consistent with averaging accumulated reward over voyage duration as described by Guo et al. [18]. Failed episodes can contain a nonexecuted terminal transition and must be marked separately with physical coverage; their shortened returns must not be interpreted as comparable full-voyage performance. Any smoothing window and whether an average is per episode or pooled per time step will be stated explicitly. A fixed-voyage experiment removes one source of episode-length variation, but does not remove exploration or policy-update variability.

Independent Validation8 and final Test4 reports will include completion status, physical-duration coverage, cumulative and mean reward, actual hydrogen consumption, FC and LB wear costs, signed equivalent hydrogen, total four-term cost, and initial/final SOC. Define duration coverage and suite completion rate as

$$\kappa_j=\dfrac{N_j^{phys}}{T_j^{planned}},\qquad CR=\dfrac{1}{M}\sum_{j=1}^{M}\mathbf{1}\{j\ \text{completed}\}.\tag{37}$$

In Eq. (37), both numerator and denominator of coverage are one-second interval counts, and $M$ is the number of voyages in the evaluation suite: eight for Validation and four for Test. A failed prefix cost is not a complete-voyage cost. Economic comparisons should therefore pair completed voyages, disclose failures, and avoid using an inexpensive early termination as evidence of improved efficiency. No numerical comparative conclusion is drawn before the experiments described next.

## 4. Simulation Results and Analysis

### 4.1 Simulation environment and parameter settings

*To be completed with the hardware, exact Python/SB3/Zoo/PyTorch versions, source revision, reward/configuration hashes, final selected hyperparameters, dataset provenance, and independent runs. The plant and baseline parameters in Tables 2–4 are model definitions, not experimental outcomes.*

**Table 5. Final experimental protocol and run identities.**

| Item | Fixed-voyage DDPG | Random-voyage DDPG | Reference strategies |
|---|---|---|---|
| Source revision / model hash | To be completed | To be completed | To be completed |
| Training voyage(s) / seed(s) | To be completed | To be completed | To be completed |
| Final $\tau$ / noise / update counts | To be completed | To be completed | Not applicable or to be completed |
| Validation checkpoint selection | To be completed | To be completed | To be completed |
| Held-out evaluation status | To be completed | To be completed | To be completed |

### 4.2 DDPG training convergence analysis

*To be completed. Present both episode return and mean reward as defined in Eq. (36), identify failed episodes, and report the smoothing convention. Add critic loss and update counts as learning diagnostics, without treating critic loss alone as evidence of policy quality. Independent-run variability is required for a robustness claim.*

**Fig. 7. Training mean-reward evolution under the current economic reward.**

*Result figure pending; no curve or convergence episode is supplied.*

### 4.3 Fuel cell and battery power distribution

*To be completed using deterministic evaluation trajectories. Plot load, requested and executed FC power, and signed LB power on a common time base; disclose projection frequency and FC ramp statistics. Examine low-load charging using the actual feasibility interval and SOC trajectory rather than inferring unreasonable operation from FC power alone.*

**Fig. 8. Deterministic FC–LB power allocation over representative complete voyages.**

*Result figure pending. Positive LB power denotes discharge; negative LB power denotes charging.*

### 4.4 Battery SOC performance

*To be completed. Report initial, final, minimum, and maximum SOC, deviation from the 0.5 reward reference, and physical completion. The hard operating band and the reward reference must be plotted as different quantities.*

**Fig. 9. Battery SOC trajectories and operating limits.**

*Result figure pending; no SOC-maintenance performance is claimed.*

### 4.5 Fuel cell and battery degradation analysis

*To be completed. Decompose FC voltage loss into the four accounted conditions and report raw as well as economically charged life fractions. Report LB unweighted/weighted throughput and sensitivity to the assumed lifetime denominator. Aggregate results do not establish individual-stack life balancing or experimentally measured service-life extension.*

**Fig. 10. FC degradation contributions and aggregate source-wear comparison.**

*Result figure pending. Corresponding LB throughput and normalization-sensitivity results will be included in the associated table or companion panel.*

### 4.6 Economic cost comparison

*To be completed. Use identical completed voyages and reset conditions for every compared controller. Present actual hydrogen, FC wear, LB wear, and signed equivalent-energy valuation separately; show the SOC penalty as a reward component rather than a physical purchase. Any percentage change must specify its baseline, denominator, and completion eligibility.*

**Table 6. Matched-voyage economic and degradation comparison.**

| Strategy / split | Completed voyages | Actual H₂ | FC wear cost | LB wear cost | Signed equivalent cost | Four-term cost | Final SOC |
|---|---|---|---|---|---|---|---|
| Reference strategy | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Fixed-voyage DDPG | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| Random-voyage DDPG | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

**Fig. 11. Disaggregated economic-cost comparison under matched operating conditions.**

*Result figure pending; no cost saving or optimality claim is supplied.*

### 4.7 Validation and generalization performance

*To be completed with the five episode-triggered Validation8 reports and a single frozen-selection Test4 assessment. Show all voyage outcomes, completion and coverage, paired complete-voyage costs, and SOC. Compare the fixed-voyage and random-voyage protocols without using Test4 to tune hyperparameters or select checkpoints. Explain any difference between the Optuna mean-return objective and the final economic interpretation.*

## 5. Conclusions

*This section will be completed after the current-reward experiments and independent evaluations. No quantitative effectiveness or superiority conclusion is asserted at this stage.*

### 5.1 Energy-management objective and modeled contributions

*To be completed: summarize the aggregate continuous-power controller, physically feasible actuator projection, and explicitly separated hydrogen, FC-wear, LB-wear, stored-energy-equivalence, and SOC-reward terms.*

### 5.2 Conclusions supported by the numerical evidence

*To be completed: state only conclusions supported by the current reward version, matched complete-voyage comparisons, independent runs, Validation8, and held-out Test4. Add numerical changes only after checking their denominator, uncertainty, and accounting boundary.*

### 5.3 Limitations and future work

*To be completed: discuss efficiency-map provenance, empirical degradation transferability, assumed LB lifetime normalization, partial observability, nonlinear reward saturation, and the difference between aggregate wear accounting and stack/cluster dispatch. Future multi-stack control will require individual health states and a validated allocation layer; it is not an implemented contribution of the present controller.*

## References

[1] International Maritime Organization. 2023 IMO Strategy on Reduction of GHG Emissions from Ships. Resolution MEPC.377(80), adopted 7 July 2023. https://www.imo.org/en/OurWork/Environment/Pages/2023-IMO-Strategy-on-Reduction-of-GHG-Emissions-from-Ships.aspx.

[2] Guo X, Lang X, Yuan Y, Tong L, Shen B, Long T, Mao W. Energy management system for hybrid ship: Status and perspectives. Ocean Engineering 2024;310:118638. https://doi.org/10.1016/j.oceaneng.2024.118638.

[3] Wu P, Bucknall R. Hybrid fuel cell and battery propulsion system modelling and multi-objective optimisation for a coastal ferry. International Journal of Hydrogen Energy 2020;45(4):3193–3208. https://doi.org/10.1016/j.ijhydene.2019.11.152.

[4] Wu P, Partridge J, Bucknall R. Cost-effective reinforcement learning energy management for plug-in hybrid fuel cell and battery ships. Applied Energy 2020;275:115258. https://doi.org/10.1016/j.apenergy.2020.115258.

[5] Zhou X, Yang X, Zhou M, Liu L, Niu S, Zhou C, Wang Y. Multi-Temporal Energy Management Strategy for Fuel Cell Ships Considering Power Source Lifespan Decay Synergy. Journal of Marine Science and Engineering 2025;13(1):34. https://doi.org/10.3390/jmse13010034.

[6] Bassam AM, Phillips AB, Turnock SR, Wilson PA. Development of a multi-scheme energy management strategy for a hybrid fuel cell driven passenger ship. International Journal of Hydrogen Energy 2017;42(1):623–635. https://doi.org/10.1016/j.ijhydene.2016.08.209.

[7] Kalikatzarakis M, Geertsma RD, Boonen EJ, Visser K, Negenborn RR. Ship energy management for hybrid propulsion and power supply with shore charging. Control Engineering Practice 2018;76:133–154. https://doi.org/10.1016/j.conengprac.2018.04.009.

[8] Hou J, Sun J, Hofmann H. Control development and performance evaluation for battery/flywheel hybrid energy storage solutions to mitigate load fluctuations in all-electric ship propulsion systems. Applied Energy 2018;212:919–930. https://doi.org/10.1016/j.apenergy.2017.12.098.

[9] Wu P, Partridge J, Anderlini E, Liu Y, Bucknall R. Near-optimal energy management for plug-in hybrid fuel cell and battery propulsion using deep reinforcement learning. International Journal of Hydrogen Energy 2021;46(80):40022–40040. https://doi.org/10.1016/j.ijhydene.2021.09.196.

[10] Mnih V, Kavukcuoglu K, Silver D, et al. Human-level control through deep reinforcement learning. Nature 2015;518(7540):529–533. https://doi.org/10.1038/nature14236.

[11] Zhu L, Liu Y, Zeng Y, Guo H, Ma K, Liu S, Zhang Q. Energy management strategy for fuel cell hybrid ships based on deep reinforcement learning with multi-optimization objectives. International Journal of Hydrogen Energy 2024;93:1258–1267. https://doi.org/10.1016/j.ijhydene.2024.10.192.

[12] Jung W, Chang D. Deep Reinforcement Learning-Based Energy Management for Liquid Hydrogen-Fueled Hybrid Electric Ship Propulsion System. Journal of Marine Science and Engineering 2023;11(10):2007. https://doi.org/10.3390/jmse11102007.

[13] Jung W, Chang D. Correction: Jung, W.; Chang, D. Deep Reinforcement Learning-Based Energy Management for Liquid Hydrogen-Fueled Hybrid Electric Ship Propulsion System. J. Mar. Sci. Eng. 2023, 11, 2007. Journal of Marine Science and Engineering 2024;12(3):484. https://doi.org/10.3390/jmse12030484.

[14] Wu P, Partridge J, Anderlini E, Liu Y, Bucknall R. An intelligent energy management framework for hybrid-electric propulsion systems using deep reinforcement learning. International Journal of Hydrogen Energy 2025;106:282–294. https://doi.org/10.1016/j.ijhydene.2025.01.380.

[15] Silver D, Lever G, Heess N, Degris T, Wierstra D, Riedmiller M. Deterministic Policy Gradient Algorithms. Proceedings of the 31st International Conference on Machine Learning, PMLR 2014;32(1):387–395. https://proceedings.mlr.press/v32/silver14.html.

[16] Lillicrap TP, Hunt JJ, Pritzel A, Heess N, Erez T, Tassa Y, Silver D, Wierstra D. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015; presented at ICLR 2016. https://arxiv.org/abs/1509.02971.

[17] Wu Y, Tan H, Peng J, Zhang H, He H. Deep reinforcement learning of energy management with continuous control strategy and traffic information for a series-parallel plug-in hybrid electric bus. Applied Energy 2019;247:454–466. https://doi.org/10.1016/j.apenergy.2019.04.021.

[18] Guo X, Tang D, Yuan Y, Yuan C, Shen B, Guerrero JM. A New Energy Management Strategy Supported by Reinforcement Learning: A Case Study of a Multi-Energy Cruise Ship. Journal of Marine Science and Engineering 2025;13(4):720. https://doi.org/10.3390/jmse13040720.

[19] Yang X, Zhou X, Li X, Wang Y, Long F, Tang T, Liu L, Liu Y. Multi-time scale energy management and optimization for hybrid power ships considering lifetime balance and extension of multi-stack fuel cell systems. Ocean Engineering 2026;358:125687. https://doi.org/10.1016/j.oceaneng.2026.125687.

[20] Ren P, Pei P, Li Y, Wu Z, Chen D, Huang S. Degradation mechanisms of proton exchange membrane fuel cell under typical automotive operating conditions. Progress in Energy and Combustion Science 2020;80:100859. https://doi.org/10.1016/j.pecs.2020.100859.

[21] Schmittinger W, Vahidi A. A review of the main parameters influencing long-term performance and durability of PEM fuel cells. Journal of Power Sources 2008;180(1):1–14. https://doi.org/10.1016/j.jpowsour.2008.01.070.

[22] Saponaro G, Stefanizzi M, Torresi M, Camporeale SM. Analysis of the degradation of a Proton Exchange Membrane Fuel Cell for propulsion of a coastal vessel. International Journal of Hydrogen Energy 2024;61:803–819. https://doi.org/10.1016/j.ijhydene.2024.02.349.

[23] Kwon L, Cho DS, Ahn C. Degradation-Conscious Equivalent Consumption Minimization Strategy for a Fuel Cell Hybrid System. Energies 2021;14(13):3810. https://doi.org/10.3390/en14133810.

[24] Ghaderi R, Kandidayeni M, Soleymani M, Boulon L, Trovão JPF. Online Health-Conscious Energy Management Strategy for a Hybrid Multi-Stack Fuel Cell Vehicle Based on Game Theory. IEEE Transactions on Vehicular Technology 2022;71(6):5704–5714. https://doi.org/10.1109/TVT.2022.3167319.

[25] Xu B, Oudalov A, Ulbig A, Andersson G, Kirschen DS. Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment. IEEE Transactions on Smart Grid 2018;9(2):1131–1140. https://doi.org/10.1109/TSG.2016.2578950.

[26] Han X, Lu L, Zheng Y, Feng X, Li Z, Li J, Ouyang M. A review on the key issues of the lithium ion battery degradation among the whole life cycle. eTransportation 2019;1:100005. https://doi.org/10.1016/j.etran.2019.100005.

[27] Mylonopoulos F, Durgaprasad S, Coraddu A, Polinder H. Lifetime design, operation, and cost analysis for the energy system of a retrofitted cargo vessel with fuel cells and batteries. International Journal of Hydrogen Energy 2024;91:1262–1273. https://doi.org/10.1016/j.ijhydene.2024.10.235.

[28] Thomas D. Cost Reduction Potential for Electrolyser Technology. Hydrogenics; 2018. Presentation cited in [27], reference 54. https://www.humsterlandenergie.nl/resources/LInks-duurzaam/Linkpagina/20180619_Hydrogenics_EU-P2G-Platform_for-distribution.pdf. No DOI assigned; direct slide-level verification remains pending.

[29] Raffin A, Hill A, Gleave A, Kanervisto A, Ernestus M, Dormann N. Stable-Baselines3: Reliable Reinforcement Learning Implementations. Journal of Machine Learning Research 2021;22(268):1–8. https://jmlr.org/papers/v22/20-1364.html.

[30] Fujimoto S, van Hoof H, Meger D. Addressing Function Approximation Error in Actor-Critic Methods. Proceedings of the 35th International Conference on Machine Learning, PMLR 2018;80:1587–1596. https://proceedings.mlr.press/v80/fujimoto18a.html.
