# DQN four-weight adaptation design

## Scope and fixed boundary

This document defines only how a future DQN may select the four existing MPC weights. It does not define or add a DQN implementation, environment, training entry point, reward interface, experiment output, or new MPC mechanism. The current N=6, 1 s OSQP-QP model, objective terms, state equations, constraints, physical parameters, SOC reference, first-move receding-horizon execution, and load-input timing remain unchanged.

The DQN decision is one level above the retained MPC. At every control second, before the MPC solve, it reads the current state and selects one complete four-weight action. The existing MPC then solves with that action and the same future six load points, and only the first control move is applied.

## Decision sequence

1. At time `t`, construct the DQN state from measurements available at `t`, the previous applied powers, and the same six future load values that will be passed to MPC.
2. The DQN selects one discrete complete tuple `(q_h2, q_batt, q_soc, q_fc_var)`.
3. The unchanged N=6 MPC solves once with that tuple and its existing hard constraints.
4. If the solve and existing commit checks pass, apply only the first MPC move and advance the physical state.
5. Form the one-step reward from hydrogen use, battery power, SOC deviation, FC power change, and any solve or constraint failure. A failed solution is not applied and does not update SOC.

The six future load points let the selector identify an approaching rise, fall, or regime change before it reaches the current-load channel. Their purpose is to reduce action-switching reaction lag. In the current offline study they are ideal-foresight natural-clipped load values; replacing them with online forecasts is outside this design task.

## State

The state contains only:

- current `SOC(t)` and `SOC(t) - 0.55`;
- previous applied `P_fc(t-1)` and `P_batt(t-1)`;
- current load `P_load(t)` and current load change rate, calculated from the current and immediately previous load;
- the six future load values used by the N=6 MPC at the same decision instant.

The state must not contain samples from an excluded voyage. It also must not use future information beyond the six values already supplied to the MPC.

## Discrete actions

Every action is a complete four-weight tuple. The DQN must not select or adjust one weight independently while leaving the others implicit.

- **Balanced action:** Candidate C, exactly `(0.25, 0.4, 12.0, 20.0)`. This is the retained reference action.
- **High-dynamic direction:** favor smoother FC power and let the battery absorb more of a rapid load transient, while retaining all battery and SOC hard bounds.
- **Sudden-unload direction:** favor a sufficiently prompt FC reduction and limit excessive charging absorption by the battery, while retaining the FC hard ramp constraint.
- **SOC-recovery direction:** strengthen the tendency to return toward the existing SOC reference without adding a terminal SOC term or a charging rule.
- **Battery-protection direction:** increase the cost of battery power so the FC carries more of the sustainable load contribution, subject to its existing hard ramp and power bounds.

Only Candidate C is numerically fixed here. The other entries are physical directions for later validation, not authorized weight combinations. This document deliberately does not create a large action table or choose additional numbers.

## Reward

The one-step reward is the negative of the same four physical quantities plus a failure penalty:

`r_t = -(lambda_h2 * H2_t + lambda_batt * Batt_t + lambda_soc * SOC_error_t + lambda_fc_var * FC_change_t) - lambda_fail * I_failure`

where:

- `H2_t` is the existing hydrogen-use measure for the applied step;
- `Batt_t` is the existing battery-power use measure;
- `SOC_error_t` is the existing deviation from SOC reference;
- `FC_change_t` is the existing FC step-change measure;
- `I_failure` is one only when the MPC solve fails or an existing hard-constraint/commit check fails, and zero otherwise.

The reward coefficients `lambda_*` are fixed reward-scaling constants, distinct from the four weights selected by the action. This design does not assign their values. It adds no terminal SOC reward, slack variable, reserve term, switching bonus, new degradation model, or other aggregate mechanism.

## Data boundary and evaluation discipline

The active voyage-level split contains 35 training, 10 validation, and 5 test voyages. The 16 abnormal voyages listed in `docs/VOYAGE_DATA_QUALITY_AUDIT.md` are absent from all three sets and must never enter DQN fitting, action definition, validation, or final evaluation.

- Training voyages are the only source for a future DQN fit.
- Validation voyages are the only source for choosing future action tuples and training hyperparameters.
- Test voyages remain isolated for final evaluation.
- No control window may cross a voyage boundary.

Network structure, exploration policy, replay buffer, optimization hyperparameters, training duration, additional action tuples, and deployment forecasting are intentionally left unspecified because they require separate evidence and authorization.
