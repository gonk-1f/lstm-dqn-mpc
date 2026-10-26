# V6 standard DQN, one-second control

**Current completed run:** run02 uses Adam **0.0001**, SOC weight **0.2**,
and FC ramp weight **0.05**, as subsequently authorized. Its eight traversals
are complete; Greedy Validation finishes only 3/8 in round 8. See the
[full archive and analysis](results/v6_dqn_1s_20261026/README.md) and
[current reporting contract](v6_lr1e4_q_reporting.md).
The following implementation description records the initial **run01**
configuration; its 0.001 / 0.1 / 0.1 values are historical, not run02 defaults.

Based on V5 source at 39b83b5 in the independent `feat/v6-dqn-1s` worktree.
The frozen user specification is implemented: ordinary DQN, 5–256–256–13,
Adam 0.001, SmoothL1 beta1, clip10, gamma1, one-step TD, reward scale1.
Actions are actual one-second increments -60:10:60 kW, without division by30.
SOC and squared FC increment weights are both0.1, SOC reference0.5 and
denominator0.1; degradation stays in the economic ledger but outside reward.

The existing canonical one-second Train35 and Validation8 files are copied
unchanged. All original nodes and load values are preserved. Test4 appears
only in split metadata; no Test payload is copied or read. See
`data/processed/operating_dataset_v6_1s/metadata/data_processing_report.csv`.

Train-only random warmup collects exactly50,000 closed transitions without
Adam. A prefix ending at the warmup budget remains a real nonterminal prefix.
The eight training traversals then each schedule all35 Train voyages once.
Epsilon decays by traversal/episode progress over the first six traversals,
then stays0.05. Each new closed transition receives one Adam update; no
unpenalized infeasible-next-state tail is inserted. Target is copied initially
and every1000 actual Adam updates; it stays fixed between copies.

Uniform Replay uses a numeric FIFO ring to avoid copying a million-entry pool
for every minibatch. V5's DDQN agent and training modes remain available.
The V6 runner reuses `replay_episode`, `EconomicMPC.interval`, the signed SOC
equivalent cost, and the existing once-only failure finalizer.

Run `scripts/start_v6_dqn.ps1`. Each completed traversal saves model weights,
latest full training state, Validation profiles/costs/Q ranks, and the existing
four physical probes remapped to V6 causal history. Research-candidate ranking
uses Validation completion count, then duration coverage; when all eight
complete, lower full-set economic cost wins. An incomplete prefix cost is
never presented as the fixed complete-set cost. V5/V6 event rates are compared
per physical hour, with cost comparisons restricted to jointly completed IDs.

The one-second load is a derived historical trajectory, and ±60 kW/s is a
simulation action assumption; neither is a new vessel measurement/calibration.
