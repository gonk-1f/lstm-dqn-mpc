# Direct FC power MLP Double-DQN (v4)

The `feat/direct-power-mlp-ddqn` branch is a direct power allocation experiment. The v2/v3 accounting code remains the source of the physical and economic model; `EconomicMPC.solve` is never called.

## Decision timing and modes

At each 30 s ONBOARD row, the controller first observes that row's exogenous load, SOC, previous FC output, and preceding observed loads. It then chooses an FC command from `0, 10, ..., 600 kW`; battery bus power is exactly `measured load - chosen FC power`. The executed command is not clipped after selection. A causal feasibility mask removes actions that violate battery power or one-step SOC bounds. If no action is feasible, replay fails with the row index. The mask guarantees only current-step feasibility, not future viability.

On departure, previous load and previous FC output reset to zero while the first physical row still uses its measured load. The SOC and cumulative degradation state carry through shore charging. The explicit `operating_mode` field gates ONBOARD and SHORE_PENDING/SHORE_CHARGING. Shore rows have FC power zero, no DQN action, accepted charging limited by the physical charger and SOC 0.6. The FC shutdown event is included once in the first shore interval's formal FC degradation ledger. Its charging electricity and battery degradation are both in the four-component actual ledger. The last ONBOARD transition receives the following shore block's cost and is terminal; a later departure starts a new DQN episode.

## State, action, reward

The eight state features, in order, are: `SOC`, current measured load / 600, current minus previous load / 600, previous minus earlier load / 600, previous FC power / 600, cumulative FC economic life fraction, cumulative battery economic life fraction, departure flag. The current measured load is available *before* the decision; later loads are never supplied to the policy. The successor state in a stored transition may contain the next measured load because it is observed after the action.

The reward is the negative actual CNY ledger sum: hydrogen + FC degradation + battery degradation + shore grid electricity. There is no `Cscale`, LPF tracking term, SOC target term, or MPC objective. FC start and shutdown costs are already inside the formal FC degradation component. The final ONBOARD reward includes the immediately following actual shore block once. Leading shore intervals have no preceding DQN action and remain in the total accounting ledger.

The MLP has hidden widths 128 and 64, ReLU activations, and 61 Q outputs. The agent uses masked Double-DQN targets, Adam at `1e-4`, replay capacity 100,000, default batch 64, target sync each training round, gradient norm cap 10, and `gamma=1.0` so the undiscounted Q return represents total CNY cost across a complete episode. Training begins with causal load-following trajectories in the replay buffer, then uses epsilon-greedy direct-power rollouts. Epsilon goes from 0.15 to 0.02 across rounds; for one round it is 0.15. These are initial experimental settings, not tuned hyperparameters.

Run a bounded pilot from the repository root or this worktree:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m v4.train --rounds 1 --max-train-episodes 1 --max-validation-episodes 1 --updates-per-episode 1
```

The runner loads authenticated Train and Validation payloads only; it checks that Test-open count remains zero. It writes `report.json` and writes `selected_agent.pt` only when bootstrap, every training round, and Validation complete **and** the economic horizon is settled. Failed episodes are listed, excluded from comparable aggregate cost, and never assigned a made-up penalty.

## Current limits of the dataset and reward

On the current formal split, all 30 Train and 8 Validation samples end in ONBOARD mode. Their remaining battery energy has no observed later shore charge in the sample. Under the requested actual-cost-only reward, a terminal low SOC can look artificially cheap. The runner reports the count of unsettled terminal ONBOARD samples and blocks checkpoint selection. A later design must either join samples into complete voyage-to-shore economic horizons, or explicitly define an economic terminal energy valuation/constraint before comparing policies.

A one-episode Train/Validation pilot found a no-feasible-action state during exploratory training at Train row 113. The causal one-step feasibility mask cannot prevent a previous action from making a later high-load row impossible. This is recorded as a failed rollout rather than silently changing the FC action or adding a non-economic penalty. A future viability rule or complete-horizon dataset is needed before claiming a trained deployable policy.

Historical v3 forecast and calibration outputs were archived in commit `2e70008` on `refactor/multiscale-dqn-wmpc-v2` before removing superseded generated output files from this branch.
