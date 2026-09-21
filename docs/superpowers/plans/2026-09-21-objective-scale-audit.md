# Objective Normalization and Scale Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the v2 MPC objective normalization and add a fail-closed Train-only objective magnitude and responsiveness audit.

**Architecture:** Keep objective computation in `nonlinear_mpc.py`, add a separate dependency-light offline audit module, and expose only its formal repository status to preflight. Real Train evidence remains absent, so documentation records unavailable numeric results and NO-GO.

**Tech Stack:** Python dataclasses, SciPy/Numpy MPC, `unittest`, Markdown.

---

### Task 1: Freeze and average the three MPC objectives

**Files:**
- Modify: `tests/test_v2_nonlinear_mpc.py`
- Modify: `src/v2/control/nonlinear_mpc.py`
- Modify: `docs/method_v2_multiscale_dqn_wmpc.md`

- [ ] Add failing tests for the 600/600/0.60 constants, `[0.20,0.80]` hard bounds, `[0.40,0.60]` soft band, inclusive zero penalty, horizon means, and unchanged simplex weights.
- [ ] Run `python -m unittest tests.test_v2_nonlinear_mpc.ObjectiveTests` and confirm failures are caused by the old sums/scales.
- [ ] Add named constants, divide every component sum by `N`, and make `MPCConfig` reject non-method objective scales/bounds while leaving the explicit ramp field independent.
- [ ] Update the method document with the exact formulas and unresolved ramp provenance.
- [ ] Re-run the targeted objective and nonlinear MPC tests.

### Task 2: Add the Train-only audit

**Files:**
- Create: `tests/test_v2_objective_scale_audit.py`
- Create: `src/v2/analysis/objective_scale_audit.py`
- Modify: `src/v2/analysis/__init__.py`

- [ ] Add failing tests proving held-out rejection before loader/solver access, exact statistics and percentile rules, ratio boundary statuses, dominance counts, behavioral redundancy, and tamper detection.
- [ ] Run `python -m unittest tests.test_v2_objective_scale_audit` and confirm the missing module/API failure.
- [ ] Implement exact audit cases, observations, summary records, sealed result validation, lazy Train guard, actual solver-runner traversal, and deterministic statistics.
- [ ] Re-run the audit tests and the existing Train-only audit tests.

### Task 3: Add the formal gate and reports

**Files:**
- Modify: `tests/test_v2_formal_preflight.py`
- Modify: `src/v2/preflight.py`
- Create: `docs/v2_objective_scale_audit.md`
- Modify: `docs/v2_preflight_report.md`
- Modify: `README.md`

- [ ] Add failing tests that preflight contains the objective-scale gate and remains NO-GO.
- [ ] Add the immutable repository audit status and the new preflight check without bypassing existing gates.
- [ ] Document formulas, unavailable Train statistics, dominance/responsiveness status, ramp configuration boundary, and recommendations that are not automatically applied.
- [ ] Run targeted tests, then one final full suite, two solver smoke tests, compile/import scans, and `git diff --check`.
- [ ] Commit only the scoped implementation, tests, and documentation.

