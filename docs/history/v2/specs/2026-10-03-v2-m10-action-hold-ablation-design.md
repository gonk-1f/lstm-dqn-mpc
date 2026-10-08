# v2 M=10 action-hold ablation design

## Purpose

Add an isolated ablation in which one DQN-selected weight action is held for at
most ten executed rolling MPC solves. `Ts=30 s`, `N=5`, `tau_LPF=180 s`, the
90-value history state, the 36-action catalog, the MPC objective, and
`gamma=1.0` remain unchanged. The resulting DQN decision interval is 300 s.

Holding an action for ten solves assigns a longer physical and economic outcome
to that action, but it also halves the approximate macro-transition count and
reduces control adaptability. Therefore the ablation is evaluated using both
economic cost and action-use diagnostics; lower cost is not inferred merely
from a larger per-transition reward magnitude.

## Training-volume match

Parameters measured in DQN macro steps are halved relative to the M=5 H4
configuration:

- warmup: 5,000 to 2,500;
- epsilon decay: 150,000 to 75,000;
- replay capacity: 200,000 to 100,000.

The network `(128, 128)`, learning rate `1e-3`, batch size 256, target sync
1,000 macro steps, gradient clip 10, epsilon endpoints `1.0 -> 0.05`, and all
other learning logic remain unchanged. Target sync is intentionally unchanged
because the request freezes every parameter not explicitly listed; this means
the target network is synchronized after twice as much physical time and is a
declared limitation of the one-factor ablation.

## Artifact isolation

The M=10 experiment owns a distinct output root, checkpoint semantics, and
Train-only reward-scale document. Its documents bind
`dqn_switch_steps=10`/`switch_seconds=300`; M=5 checkpoints, replay state, and
reward-scale documents must fail before runtime mutation. Reward calibration is
re-executed over Train only under fixed `w_8_1_1`, retains the established
30-second interval-CNY reference definition, and opens zero Test payloads.

## Evaluation boundary

Training may read Train and Validation only. Every round reports Validation raw
economic cost, completion/failure counts, number of unique greedy actions,
Shannon entropy, and maximum action share. Test remains unopened. A separate
Validation-only comparison evaluates the selected M=5 and M=10 policies plus
fixed `w_8_1_1`; no claim of improvement is made until those artifacts exist.
