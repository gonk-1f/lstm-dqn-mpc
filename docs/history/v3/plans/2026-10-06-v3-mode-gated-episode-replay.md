# V3 Mode-Gated Episode Replay and Train Pilot Plan

**Goal:** Replay full 30 s v3 episodes with an explicit `operating_mode` gate and run a bounded Train-only fixed-action pilot for the 16 calibrated weight pairs.

**Architecture:** A supervisor reads the frozen mode column before invoking any policy. ONBOARD rows use the existing causal `PredictiveController`; consecutive `shore_pending` and `shore_charging` rows use only the shore settlement path. Episode transitions close at the following shore block or dataset end, while SOC and degradation accounts carry into the next ONBOARD run. A separate pilot runner selects complete Train ONBOARD runs and records costs, constraints, and failures; it never loads Validation or Test payloads.

**Tech stack:** Python, frozen `FormalTrainingDataset`, existing v3 control and shore modules, pytest.

## Task 1: Explicit mode-gated replay

- [x] Write synthetic tests for ONBOARD → shore → ONBOARD, leading shore, dataset-end terminal, and fail-closed unresolved mode.
- [x] Run the tests and confirm they fail because the replay interface is absent.
- [x] Implement the episode replay with one policy call and one MPC solve per ONBOARD row, zero policy/MPC calls on shore rows, shore costs attached to the preceding terminal transition, and end SOC carried forward.
- [x] Run the synthetic tests and existing v3 controller tests.

## Task 2: Bounded Train-only action pilot

- [x] Write tests for selecting complete Train runs and rejecting a mismatched calibration artifact or forbidden split.
- [x] Implement a resumable or bounded pilot CLI that reads the 16 calibrated actions and fixed `C_nom`, evaluates a small set of complete Train ONBOARD runs, and records component costs, feasibility, SOC, and runtime per action.
- [x] Run the pilot, inspect action redundancy and physical failures, and retain the full raw report.

## Task 3: Verify and document

- [x] Run focused v3 tests and compile checks.
- [x] Update the v3 runbook with the upper-mode gate, pilot procedure, numerical findings, and limits of the small Train sample.
