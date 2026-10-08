# V2 Parameter Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the approved battery-lifetime and control-timescale baseline values without upgrading their evidence provenance or changing model equations.

**Architecture:** Keep the existing preflight status enum and represent formal usability through `VERIFIED`; expose separate configuration-status and evidence-status constants so project-design freezing cannot be mistaken for vessel or manufacturer validation. Leave diagnostic and sensitivity code dormant and unchanged except where documentation must stop treating it as a training gate.

**Tech Stack:** Python 3, `unittest`, existing v2 dataclass/configuration contracts.

---

### Task 1: Freeze-contract tests

**Files:**
- Create: `tests/test_v2_parameter_freeze.py`

- [x] Add tests for the 15,000 battery factor, secondary-literature evidence, frozen N/M/tau values, independent N/M semantics, both 150 s intervals, and non-blocking preflight statuses.
- [x] Run `python -m unittest tests.test_v2_parameter_freeze -v` and confirm failures are caused by missing frozen contracts.

### Task 2: Minimal production status changes

**Files:**
- Modify: `src/v2/config.py`
- Modify: `src/v2/contracts.py`
- Modify: `src/v2/models/battery_degradation.py`
- Modify: `src/v2/preflight.py`

- [x] Add frozen baseline and separate evidence constants.
- [x] Replace the provisional timescale factory used by formal semantics with a frozen-baseline factory.
- [x] Mark battery lifetime, N, M, and tau checks as preflight-passing while preserving their exact evidence classifications.
- [x] Re-run the focused test and confirm it passes.

### Task 3: Contract migration, documentation, and verification

**Files:**
- Modify only directly conflicting v2 tests and v2 parameter/preflight documentation.

- [x] Replace obsolete provisional assertions with frozen-configuration assertions.
- [x] State explicitly that N/M/tau are project design values, battery evidence remains secondary literature, and none is claimed globally optimal or vessel measured.
- [x] Run focused tests, all v2 tests, solver smoke, compile/import, and `git diff --check`.
- [ ] Commit the verified working tree and record the commit SHA.
