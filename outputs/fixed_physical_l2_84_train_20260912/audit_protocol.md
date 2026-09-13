# Fixed physical L2 reward, isolated Train-only audit

The sole candidate is `r=1/(1+norm([H,B,S,F]/6))`, with the current physical
metrics, all 84 weight actions and the current six-step persistence forecast.
No anchor, clipping, denominator epsilon or per-action objective enters this reward.

1. Reuse all 1440 x 84 saved grid MPC outcomes, including the 14 states excluded
   from the prior reference reward because an anchor failed. Save distributions,
   squared shares, paired SOC comparisons, winners, gaps and neighboring-action
   control equivalence. Preserve both original solver metrics and diagnostics
   with SOC reconstructed from power balance.
2. Rescore every nonempty saved trace for the eight frozen Train windows. These
   remain trajectories of the OLD policies; do not call them new-policy rollouts.
   Zero-step anchor failures provide no trajectory to rescore. Keep this limitation.
3. Freeze 120 representative states by evenly spaced row positions within each
   (cohort, SOC) stratum (10 in each of 12 strata). Solve the full 84-action grid
   under fresh forward warm starts, reversed state/action order, reversed order
   with tighter solver tolerances, and zeroed warm starts at every solve. Compare
   rewards, winner changes, control changes and regret. Reuse saved reference
   anchor repetitions for a separately labeled comparison. No new anchors.
4. Check score reconstruction, hand-computable physical examples, paired-state
   identity and source hashes. Report failure classes with independent physical
   feasibility LPs if any new grid solve fails. Never assign a failure penalty.

This audit creates only temporary audit code and outputs. Existing source/tests,
formal reward, training settings, actions and prior audit artifacts remain unchanged.
No DQN training, Validation/Test dataset reads, commit or push. Quantile summaries
describe this intentionally stratified Train audit set, not a deployment frequency.
No new closed-loop rollout is required to diagnose scale imbalance or reward
numerical sensitivity; long-horizon control guarantees cannot be inferred from
rescoring the old policies.

After initial scoring, the 1440 snapshots were found to span only 150-688 kW.
Add a separately reported diagnostic supplement: five evenly spaced saved
states from each of the eight windows' maximum-first-FC grid policy traces.
For low_recharge use the previously saved anchor-independent control trace;
read only the first Train load row to reconstruct its first current load.
Solve all 84 actions at these 40 states to include the sustained >900 kW region.
Keep these out of the original 1440-state quantiles and stability sample.
