# Direct FC power MLP Double-DQN (v4)

The `feat/direct-power-mlp-ddqn` branch is a direct power allocation experiment. The v2/v3 accounting code remains the source of the physical and economic model; `EconomicMPC.solve` is never called.

## Decision timing and modes

At each 30 s ONBOARD row, the controller first observes that row's exogenous load, SOC, previous FC output, and preceding observed loads. It then chooses an FC command from `0, 10, ..., 600 kW`; battery bus power is exactly `measured load - chosen FC power`. The executed command is not clipped after selection. A causal feasibility mask removes actions that violate battery power or one-step SOC bounds. If no action is feasible, replay fails with the row index. The mask guarantees only current-step feasibility, not future viability.

On departure, previous load and previous FC output reset to zero while the first physical row still uses its measured load. The SOC and cumulative degradation state carry through shore charging. The explicit `operating_mode` field gates ONBOARD and SHORE_PENDING/SHORE_CHARGING. Shore rows have FC power zero, no DQN action, accepted charging limited by the physical charger and SOC 0.6. The FC shutdown event is included once in the first shore interval's formal FC degradation ledger. Its charging electricity and battery degradation are both in the four-component actual ledger. The last ONBOARD transition receives the following shore block's cost and is terminal; a later departure starts a new DQN episode.

## State, action, reward

The eight state features, in order, are: `SOC`, current measured load / 600, current minus previous load / 600, previous minus earlier load / 600, previous FC power / 600, cumulative FC economic life fraction, cumulative battery economic life fraction, departure flag. The current measured load is available *before* the decision; later loads are never supplied to the policy. The successor state in a stored transition may contain the next measured load because it is observed after the action.

The ONBOARD reward uses the actual post-action SOC and a configurable soft penalty. Define `phi(s)=(0.4-s)^2` for `s<0.4`, zero for `0.4<=s<=0.6`, and `(s-0.6)^2` for `s>0.6`. For an ordinary ONBOARD step,

```text
r_k = -(C_H2,k + C_FC_deg,k + C_Bat_deg,k + beta_soc * phi(SOC_k+1))
```

`beta_soc` is a nonnegative CNY coefficient and defaults to zero (`--beta-soc`). The physical one-step action mask still enforces SOC 0.2–0.8. There is no `Cscale`, LPF tracking term, FC variation penalty, or MPC objective; FC start, shutdown, and load-change degradation remain in the formal FC ledger. SHORE has no DQN action and receives no additional soft penalty. The final ONBOARD reward also includes the immediately following actual shore block once, or the modeled terminal settlement described below. These are accounting costs, not extra SOC penalties. Leading shore intervals have no preceding DQN action and remain in the total accounting ledger. Reported economic cost excludes the artificial `beta_soc * phi` term; `completed_soc_soft_penalty_cny` reports it separately.

If a sample ends after ONBOARD with no observed shore block and final SOC is below 0.6, the final reward also includes a separate **MODELED terminal settlement**. This prices an accounting-only recharge from the actual terminal SOC to 0.6. The code requests the existing battery-side charger limit of 624 kW for as many virtual 30 s intervals as needed; the existing shore model caps the last accepted interval at SOC 0.6. Grid kWh equals accepted battery-side kWh divided by 0.95. The same shore model supplies battery charging degradation. The 624 kW charge profile is an explicit valuation assumption because terminal SOC alone does not determine charging degradation. It does not add load rows, alter actual terminal SOC, or represent a measured grid purchase. If actual SHORE follows, its measured requests take precedence even when actual charging stops below 0.6; no modeled top-up is added. `total_cost_cny` remains observed cost, while `comparable_cost_cny` adds the modeled settlement when present.

The economic MLP has hidden widths 128 and 64, ReLU activations, and 61 Q outputs. The agent uses masked Double-DQN targets, Adam at `1e-4`, replay capacity 100,000, default batch 64, target sync each training round, gradient norm cap 10, and `gamma=1.0`. With `beta_soc>0`, the undiscounted Q return includes the cumulative soft penalty as well as comparable CNY cost; it must not be interpreted as economic cost alone. Training begins with causal load-following trajectories in the replay buffer, then uses epsilon-greedy direct-power rollouts. The default epsilon schedule goes from 0.15 to 0.02 across rounds; for one round it is 0.15. `--epsilon-start` and `--epsilon-end` allow controlled comparisons. These are initial experimental settings, not tuned hyperparameters.

Executed prefixes of rollouts that later reach a no-feasible-action state are retained in a separate outcome replay with their actual CNY reward and a binary observed-rollout failure label. A separate 8-128-64-61 model is trained with binary cross-entropy on those outcomes and on successful rollouts. In a sample with multiple voyages, an earlier voyage ending at SHORE is labeled successful, and only the unfinished final voyage is labeled failed. The label says that the sampled continuation failed; it does not prove that every prefix action caused failure. Unfinished failed voyages are excluded from economic TD replay because treating their last transition as a cheap terminal would bias Q values; any earlier completed voyages in the same sample remain valid economic replay. **The outcome model is diagnostic and is not consulted by action selection or checkpoint selection.** Therefore the current economic policy does not yet learn to avoid failures from these labels; completion improvement is unproven. The economic network still receives 16 updates per completed sample rollout, with no new economic updates triggered by a failed sample. The separate outcome model receives updates after both complete and failed rollouts.

Run a bounded pilot from the repository root or this worktree:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m v4.train --rounds 1 --max-train-episodes 1 --max-validation-episodes 1 --updates-per-episode 1
```

To run the 40-round full Train/Validation experiment from the PyCharm PowerShell terminal, set its working directory to this worktree and run:

```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
New-Item -ItemType Directory -Force .\outputs\v4_direct_power_40r_u16_seed42_pycharm | Out-Null
python -u -m v4.train --rounds 40 --updates-per-episode 16 --batch-size 64 --seed 42 --progress-every-steps 50 --output-dir .\outputs\v4_direct_power_40r_u16_seed42_pycharm 2>&1 | Tee-Object -FilePath .\outputs\v4_direct_power_40r_u16_seed42_pycharm\train.log
```

The progress counter counts selected ONBOARD actions during DQN training rollouts, not gradient updates. It prints every 50 selected actions, immediate failure locations, and one summary per round. The final policy is then evaluated greedily on all Train and Validation samples. The full JSON report is saved in the output directory; Test is not opened.

The runner loads authenticated Train and Validation payloads only; it checks that Test-open count remains zero. It writes `report.json` and writes `selected_agent.pt` only when there is training experience and the final greedy policy completes **every Train and Validation episode**. Failed exploratory Train episodes retain their executed prefix in the outcome replay; the next Train episode still runs. `completed_observed_cost_cny` and `completed_modeled_terminal_cost_cny` remain separate; `completed_cost_cny` is their sum over finished episodes. The comparable aggregate `cost_cny` is `null` when any episode failed. `completed_soc_soft_penalty_cny` is excluded from all economic-cost totals. Failed episodes never receive a made-up CNY penalty. Exploratory Train completion need not be 100% to continue training, while final greedy Train and Validation completion must each be 100% for model selection.

## Current limits of the dataset and reward

On the current formal split, the last *executed 30 s interval* in all 30 Train and 8 Validation samples is ONBOARD. The raw power payload also has a separate terminal 0 kW boundary row that is not an executable interval. Therefore the last operating mode cannot establish that the physical load trace has no terminal boundary. The runner reports the last executed mode as a diagnostic and does not block checkpoint selection based on it. Terminal 0 kW does not by itself restore SOC; the accounting-only terminal settlement above supplies one explicit common valuation rule. It does not establish actual post-voyage charging behavior or guarantee a complete policy.

Final Test payloads are opened only after authorized model selection. The loader rejects a Test trace unless its raw first and last load rows are both 0 kW. Train/Validation endpoint loads are not checkpoint-selection criteria.

A previous one-episode Train/Validation pilot found a no-feasible-action state during exploratory training at Train row 113. The causal one-step feasibility mask cannot prevent a previous action from making a later high-load row impossible. This is recorded as a failed rollout rather than silently changing the FC action or adding a non-economic penalty. The diagnostic outcome model does not close this future feasibility gap.

In a historical two-Train-episode pilot, `zero_boundary_002` failed but the runner continued to `zero_boundary_005`, which completed with 298 DQN transitions. Train completion was reported as 1/2; the validation trajectory completed. The full Validation split remains the meaningful selection gate.

Historical v3 forecast and calibration outputs were archived in commit `2e70008` on `refactor/multiscale-dqn-wmpc-v2` before removing superseded generated output files from this branch.
