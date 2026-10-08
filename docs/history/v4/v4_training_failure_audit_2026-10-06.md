# v4 direct-power training failure audit (2026-10-06)

Historical snapshot: these measurements and statements about discarded failure
prefixes describe the code before the 2026-10-07 failure-outcome and terminal
settlement changes. See [current v4 status](v4_status_2026-10-07.md) for the
implemented semantics. These numbers are not results for the revised code.

## Reproduction

All runs used the frozen Train/Validation split, seed 42, 30 s control, the
0–600 kW/10 kW FC action grid, and the four-component actual-CNY reward.
No Test payload was opened.

| Controller / training setting | Train completed | Validation completed |
| --- | ---: | ---: |
| Current-load-following baseline | 29/30 | 7/8 |
| Highest currently feasible FC power | 30/30 | 8/8 |
| Current Double-DQN, 1 round, 1 update/episode | 15/30 | 5/8 |
| Current Double-DQN, 1 round, 16 updates/episode (CLI default) | 13/30 | 4/8 |
| Current Double-DQN, 20 rounds, 1 update/episode | 22/30 in final round | 6/8 |
| Current Double-DQN, 40 rounds, 1 update/episode | 10/30 in final round | 4/8 |
| Same 40-round run with training exploration forced to zero | 10/30 in final round | 4/8 |
| Full 40-round Train/Validation run, 16 updates/completed episode | 10/30 in final round | 4/8 |

The two 1-update, 40-round runs are diagnostic experiments, not formal model selection.
Both use all 30 Train and 8 Validation episodes, seed 42, and only one gradient
update per completed episode per round. The default CLI uses 16. With the
usual epsilon schedule (0.15 to 0.02), Train completion temporarily reached
30/30 in rounds 23 and 24, then fell to 10/30 by round 40. With exploration
forced to zero it reached 30/30 in rounds 22 and 31, then also fell to 10/30.
Validation used greedy actions in both runs and completed 4/8 at the final
checkpoint. These observations do not support exploration as the main cause
of the high final failure rate, and temporary Train completion does not
establish Validation success.

The reports are saved in
`outputs/v4_direct_power_40r_u1_seed42/report.json` and
`outputs/v4_direct_power_40r_u1_seed42_greedy/report.json`. Neither run
opened Test or produced an eligible selected checkpoint.

The full 40-round, 16-update run used the same split and seed, the unmodified
four-component economic reward, and the scheduled epsilon from 0.15 to 0.02.
Train completion peaked at 14/30 in round 8 and ended at 10/30. Final greedy
Validation completed 4/8. Its report and live progress log are in
`outputs/v4_direct_power_40r_u16_seed42/`. The log contains 6,649 progress
lines at exact 50-action intervals and 40 round summaries. The run selected
332,452 ONBOARD actions and made 7,008 DQN gradient updates on completed
training episodes, plus 464 bootstrap updates. Test remained
closed, and no selected checkpoint was written. The added terminal progress
logging does not alter replay, learning, reward, or failure semantics.
The log file timestamps span about 7 min 53 s. Of 1,200 nominal Train
episode attempts (40 x 30), only 438 completed and therefore triggered the
16-update loop. The 7,008 Train gradient updates are 36.5% of the nominal
19,200 if every attempt had completed. The run selected 332,452 actions versus
737,920 ONBOARD actions for 40 complete passes over the Train episodes.

An otherwise matched 40-round, 16-update run changed only the epsilon schedule
to 1.0 -> 0.05. Its first round completed 20/30 Train episodes versus 13/30
with the default schedule. Across all 40 rounds it completed 584 attempts
versus 438 and therefore made 9,344 Train updates plus 464 bootstrap updates.
Both runs ended at 10/30 Train and 4/8 greedy Validation; neither selected a
checkpoint or opened Test. The run report is in
`outputs/v4_direct_power_40r_u16_eps1_to_005_seed42/report.json`. These
single-seed results show improved exploration coverage, not improved final
policy feasibility. Keep both schedules available for further comparisons.

The tested completion classifier, trained separately from the CNY reward, did
not fix the failure: its five-round pilot completed 6/8 Validation episodes.
That experimental classifier was removed from the worktree.

## Physical cause

`zero_boundary_015` with current-load following reaches SOC 0.2036 after row
161. At row 162, load is 1093.1 kW. Even FC output 600 kW leaves roughly
493.1 kW to the battery, which would cross the hard SOC 0.2 lower bound.
The policy kept SOC around 0.6 during earlier low load, while the maximum-FC
baseline charged toward 0.8 and completed the same episode.

After one-round/one-update DQN training, `zero_boundary_036` selected FC
200 kW repeatedly while load was around 1020 kW, making the battery supply
around 820 kW. At row 84 its SOC was approximately 0.2001; row 85 had no
feasible action. This is a policy decision, not an FC power-capacity bug.

The current interval ledger makes a large FC power increase immediately costly.
For an illustrative state with SOC 0.25, previous FC 200 kW, and load 1012.8
kW, the existing accounting code gives 7.12 CNY at FC 200 kW and 53.25 CNY
at FC 600 kW; 45.57 CNY of the latter is FC degradation from the large power
change. This is one-step cost, not the long-horizon cost of staying at 600 kW.

## Training and information limits

`run_train_validation` drops the entire executed prefix of a failed episode.
The economic Q replay therefore receives no outcome for the choices that
preceded a supply failure. Its one-step TD target also propagates delayed
cost over a long episode slowly. More updates and rounds alone did not resolve
the observed failures.

Simply marking the last executed transition `done=True` while keeping only
negative economic cost would be misleading: failure would avoid all later
costs in the Bellman target and could look attractive. At a row with no
feasible action there is no physically executable FC command under the current
power and SOC constraints; continuing that voyage would require an explicit
load-shedding or violation model. Retaining the executed prefix and training
on a separate binary failure outcome is a possible constrained-RL experiment,
but it is not implemented or validated here. A previous five-round completion
classifier pilot completed only 6/8 Validation episodes.

The eight policy inputs contain current and two preceding measured loads, SOC,
previous FC power, two life fractions, and a departure flag. They contain no
future load or voyage plan. Two voyages with identical observed histories but
different upcoming high-load duration require the same causal action under
this state, even though the action that minimizes cost for one may leave too
little energy for the other. No fixed amount of DQN training can guarantee
both economic optimality and supply completion for arbitrary unseen profiles
under this information structure.

## Decision boundary

Do not select a formal DQN checkpoint from the current training procedure.
The zero-load endpoint requirement for final Test is independent of SOC
feasibility during the voyage. A defensible next controller must explicitly
represent supply completion as a hard constraint and supply an operational
demand envelope or causal voyage information for planning. Keep the four
actual-CNY components as the economic ledger; do not hide failures inside a
made-up CNY charge or invent unobserved shore energy.
