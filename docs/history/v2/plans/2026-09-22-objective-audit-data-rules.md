# Objective Audit Data Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build strict Train-only supervisory states and run the existing small objective-scale audit when nonzero eligible states exist.

**Architecture:** Pure mode/freshness rules live in a focused v2 data module. A separate audit runner reads only manifest-declared Train parents, performs causal one-to-one alignment and deterministic representative selection, then adapts cases to the existing audit API. The report remains the sole persisted audit result.

**Tech Stack:** Python dataclasses/enums, pandas/numpy for raw telemetry, scipy-backed existing nonlinear MPC, unittest.

---

### Task 1: Freeze freshness and operating-mode semantics

**Files:**
- Create: `src/v2/data/supervisory_rules.py`
- Create: `tests/test_v2_supervisory_rules.py`

- [x] Write failing tests proving the 10 s inclusive cap, future/stale rejection,
  three-state classification, two-sample shore dwell, unknown fallbacks, and
  sailing-only load reconstruction.
- [x] Run `python -m unittest tests.test_v2_supervisory_rules -v` and confirm the
  import or assertions fail because the module is absent.
- [x] Implement immutable inputs/results, frozen constants, consecutive-run
  classification, and fail-closed load reconstruction.
- [x] Re-run the test module and require zero failures.

### Task 2: Construct strict Train supervisory states

**Files:**
- Create: `src/v2/data/train_supervisory_audit.py`
- Create: `tests/test_v2_train_supervisory_audit.py`

- [x] Write failing fixture tests for exact-duplicate collapse, conflicting
  duplicate exclusion, latest-unused causal matching, all-8-FC/all-12-BMS
  completeness, provenance, gap separation, and Validation/Test pre-access
  rejection.
- [x] Run `python -m unittest tests.test_v2_train_supervisory_audit -v` and
  confirm failures identify missing behavior.
- [x] Implement the minimal raw-parent loader and state builder using the frozen
  `10.0 s`, `0.1 kn`, `8 kW`, and `45 s` rules.
- [x] Re-run the fixture tests and require zero failures.

### Task 3: Select cases and invoke the existing 36-action audit

**Files:**
- Create: `src/main/run_v2_objective_scale_audit.py`
- Modify: `docs/v2_objective_scale_audit.md`

- [x] Load only strict eligible Train states and deterministically select a small
  set covering low/medium/high load, steady/rapid change, and every available
  hard-bound-valid SOC band.
- [x] Replay the 90 s causal filter from eligible historical states, construct
  `ObjectiveAuditCase` payloads, configure MPC with `-624/+1248 kW` battery
  bounds and disabled hard FC ramp, and call `run_objective_scale_audit` with
  all 36 canonical actions.
- [x] Run the script. If cases are empty or solves fail the strict contract,
  retain NO-GO and record the precise blocker; never relax eligibility.
- [x] If successful, write counts, objective percentiles, active P95,
  `scale_ratio`, all ordered dominance rates, provenance, and PASS/WARNING/NO-GO
  to `docs/v2_objective_scale_audit.md`.

### Task 4: Verify scope and regression safety

**Files:**
- Verify all files above plus existing v2 modules.

- [x] Run `python -m unittest discover -s tests -p 'test_v2_*.py'`.
- [x] Run `git diff --check` and inspect `git diff` for changes to objectives,
  normalization constants, SOC bounds, action space, or DQN code; none are
  allowed.
