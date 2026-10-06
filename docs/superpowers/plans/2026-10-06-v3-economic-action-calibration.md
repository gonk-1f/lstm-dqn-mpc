# V3 Economic Scale and Action Calibration Implementation Plan

**Goal:** Produce a reproducible Train-only `C_nom` and a screened 4 by 4 candidate weight table for the 30 s persistence DQN-MPC path.

**Architecture:** Expose the exact three unweighted terms evaluated by `EconomicMPC`, then use only frozen Train episodes to construct causal LPF decision origins. Calibrate the five-step economic numerator on a reference-power policy, derive separate weight axes from observed reference and SOC penalty scales, and screen the 16 combinations on representative Train states before publishing an artifact. Do not inspect Validation or Test payloads during calibration.

**Tech stack:** Python, NumPy, SciPy, existing v2 formal dataset and v3 MPC, pytest.

## Task 1: Objective term interface

- [ ] Add a test that the public term evaluation reproduces the MPC objective at the returned plan and exposes economic, reference, and SOC terms separately.
- [ ] Verify the test fails because the interface does not exist.
- [ ] Refactor `src/v3/control.py` so the optimizer and calibration use one term calculation, preserving the existing objective and constraints.
- [ ] Run the v3 control tests.

## Task 2: Train-only calibration

- [ ] Add tests for deterministic Train-origin selection, positive finite scales, and a 16-pair Cartesian action table with independent axes.
- [ ] Verify those tests fail before implementation.
- [ ] Implement `src/v3/action_calibration.py`: load Train only; reset the LPF at each ONBOARD run; sample origins across the Train load distribution; estimate `C_nom` as the median positive five-step economic numerator under LPF-reference FC power and SOC 0.5; use Train 95th-percentile one-step load changes as a material reference deviation; use SOC 0.4/0.5/0.6 scenarios for SOC penalty scale; construct four geometric levels on each axis; and screen actions by solver success, first-step FC differences, and predicted SOC behavior.
- [ ] Write provenance, assumptions, distributions, and 16 candidates to `outputs/v3_persistence_calibration/action_calibration.json`.

## Task 3: Execute and report

- [ ] Run the calibration with the frozen dataset and inspect the complete artifact, including skipped cases and action redundancy.
- [ ] Run focused v3 tests and check the output contains no Validation or Test-derived values.
- [ ] Document the numerical recommendation and its limitations in `docs/v3_persistence_dqn_mpc.md`.
