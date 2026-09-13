# Train-only 84-action ideal/reference audit

User-authorized prototype audit; no DQN, formal logic edits, failure penalty,
Validation/Test data, or git commit/push. Existing physical model and timing stay fixed.

## Frozen protocol (declared before running)

1. Generate lexicographic positive integer compositions of 10 into four parts;
   divide by 10. Keep all 84 actions. Anchors are four separate unit vectors.
2. Reuse all 840 states in the previous unified-objective Train audit. For each
   distinct (Train segment, load, previous FC, load delta), evaluate matched SOC
   0.55/0.45/0.35/0.25/0.22. Preserve their common load/FC and label counterfactuals.
3. Solve the 4 anchors and 84 actions with existing OSQP settings. Independently
   repeat anchors in reverse state order and at eps_abs=eps_rel=1e-8, max_iter=100000.
   These are audit-only instance settings, never changes to formal solver defaults.
4. Recompute H/B/S/F from physical trajectories. Also integrate the returned FC
   trajectory through the exact frozen dynamics to expose SOC residual sensitivity.
   No denominator epsilon or upper clipping. Record raw negative coordinates.
5. Call a separate objective-free HiGHS LP if a solver result fails. Classify
   physical infeasibility only when the LP certifies infeasibility; otherwise record
   a numerical failure or unresolved result. Never assign a terminal reward.
6. Save min/P1/P50/P99/max spans, reward distributions, numerical repeat sensitivity,
   Pareto nondominance and physical endpoint coverage, winners and reward gaps.
   Compare the same fixed action outcomes under each anchor repetition to isolate
   evaluation noise from action-solver changes. Pareto fractions describe the sampled
   grid, not the full physical Pareto front. Report tolerances and raw results.
7. On eight predeclared 600-step Train windows, compare immediate-reward greedy
   selection with an 84-action maximum-first-FC diagnostic. Both use persistence and
   execute only the first FC decision against the next actual Train load. Neither is
   DQN nor a proof of optimal switching. A full-window perfect-information LP is only
   an ex-post physical feasibility benchmark, never an input to either controller.
8. Separately benchmark four persistent anchors plus one selected ordinary action.
   Retain solver/execution failures and full timing distributions, including retries.
9. Verify helper edge cases, output completeness, Train-only read provenance and
   hashes of pre-existing Python source/README. Deliver a readiness decision based on
   evidence, without changing the proposed reward or reducing the action set.

## Interpretation gates

- A positive denominator is insufficient: compare span with repeat/model-residual
  uncertainty and compare winner gaps with actual reward sensitivity.
- A near tie does not imply physical equivalence; use six-step trajectory RMS too.
- Low SOC need not always produce a larger qS: inspect executed power/SOC, attainable
  protection, and complete windows. Relative reward across SOC is not absolute safety.
- A failed fixed/greedy policy does not prove no feasible switching policy exists.
- A successful 600-step window does not establish full-voyage training readiness.

Run: `python -B src/main/audit_ideal_reference_84_train.py`
Tests: `python -B -m unittest discover -s tests -p test_ideal_reference_84_audit.py -v`

## Follow-up diagnostics triggered by observed failures

- After S-only nonconvergence and payoff sensitivity appeared, check its known
  analytical solution when SOC=.55 and constant FC=load satisfies the ramp bound.
  This gives exactly S=0 and independently diagnoses payoff errors; it does not
  replace the anchors used to compute the primary reported rewards.
- After H-only failed at the low-load window's initial state, run the same 84-action
  maximum-first-FC diagnostic there without requiring an anchor or computing reward.
  This separates available control capacity from evaluator availability.
- Final five-solve timing uses 12 evenly spaced states per cohort/SOC group (144
  total), runs without other audit processes, and includes nonconverged calls.
