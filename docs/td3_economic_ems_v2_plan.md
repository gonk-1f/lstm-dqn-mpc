# TD3 Economic EMS v2

## Reward

Raw reward is the negative of the sum of actual hydrogen cost, FC transient/low/high variable degradation, signed battery-equivalent hydrogen cost, battery degradation, per-step SOC deviation, and a successful terminal SOC penalty.

r_raw = -(C_H2 + C_FC_transient + C_FC_low + C_FC_high + C_LB_eq,signed + C_LB_deg + 0.5*(SOC_next-0.5)^2 + I_terminal*20000*(SOC_final-0.5)^2).

TD3 receives 0.001*r_raw. All physical costs remain in the raw CNY ledger. The one start-stop cycle settlement remains in the complete ledger but is excluded from the learning reward because the present environment fixes one start and one stop per voyage and the Actor does not control these events. The action-dependent shutdown transient loss remains included. Do not clip the signed battery-equivalent hydrogen term: discharge is a cost and charge is a credit; the signed episode sum is needed for energy balance.

The step SOC shaping coefficient is 0.5 CNY per squared SOC deviation. The successful terminal SOC coefficient is 20,000 CNY per squared deviation from 0.5. A truncated or failed episode does not receive the successful-terminal term.

## Action parameterization

For each state, the normalized Actor action a in [-1,1] maps affinely to the intersection of FC rating, battery charge/discharge power, and next-step SOC constraints:

P_FC = P_FC,min(s) + (a+1)/2 * (P_FC,max(s)-P_FC,min(s)).

This removes the flat clipping region of the old absolute-setpoint mapping. The physical environment retains its fail-closed check. Logs distinguish raw Actor action, old-style reference power, mapped power, feasible interval width, and executed power.

## TD3 configuration

- Huber TD3, delta=1; twin critics; policy delay=2.
- batch=32; replay=1,000,000; gamma=1; tau=0.001.
- Actor/Critic learning rate=1e-4.
- train_freq=32 environment steps; gradient_steps=1.
- learning_starts=30,000 random-action steps, roughly three long-voyage equivalents of replay warm-up.
- Gaussian exploration noise std=0.05 in normalized action coordinates.
- Target policy smoothing sigma=0.2, clip=0.5.
- One environment; 500 episode budget; 6,000,000 timestep ceiling.

The exploration std of 0.05 is a controlled starting choice, not a literature-proven optimum. Compare it with 0.1 only after the economic reward and feasible mapping pass the acceptance tests.

## Voyage protocol

- Train: random sampling with replacement from all 35 Train voyages; each episode is one complete voyage.
- Fixed diagnostic: deterministic parent_005_seg01 every 20 episodes.
- Model selection: full deterministic Validation8 every 100 episodes.
- Test4 remains untouched until final frozen-checkpoint evaluation.

A fixed voyage is appropriate for debugging whether the reward and controller can learn a known trajectory. It is not a sufficient final training distribution. Random Train35 training reduces overfitting to one time-indexed load curve; fixed-voyage evaluation remains a repeatable diagnostic.

## Acceptance criteria

Do not select a checkpoint from training reward alone. Require full-voyage completion, hard SOC compliance, terminal SOC close to 0.5, no persistent Actor boundary saturation, stable FC output with transient degradation measured, battery response to load residuals, and separate reporting of H2, FC degradation, LB equivalent hydrogen, LB degradation, and total physical CNY cost.

Validation8 economic cost and terminal SOC must improve together. Use at least three random seeds before claiming convergence. A monotonic reward curve is not guaranteed by TD3 and must not be artificially imposed; report 20-episode moving averages, per-step normalized return, completion rate, and Validation8 economics.