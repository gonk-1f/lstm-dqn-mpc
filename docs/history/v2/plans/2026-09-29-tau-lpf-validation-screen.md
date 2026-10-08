# LPF Tau Validation Screen Plan

## Scope

- Evaluate the already-selected H4 round-20 checkpoint on Validation only.
- Compare `tau_lpf_seconds` values 180 and 300 without changing the frozen 90 s baseline.
- Record completion, economics, SOC, FC variation, battery burden, and per-episode power plots.
- Mark every result as pre-retraining environment screening, not a final model result.
- Prove that no Test payload is opened.
- Remove only the explicitly retired H1, H2, H3, and pilot-selection output directories.

## TDD sequence

1. Add failing tests for per-run LPF override propagation and tau-screen diagnostics.
2. Add failing CLI tests proving both tau values use Validation and Test remains unopened.
3. Add the minimum production implementation.
4. Run focused tests, then all v2 tests, compile/import checks, solver smoke, and `git diff --check`.
5. Execute the Validation screen and inspect the generated summaries and plots.

## Non-goals

- Do not retrain H4 in this step.
- Do not read or re-evaluate Test.
- Do not change the formal baseline constant from 90 s.
- Do not select a final tau from Test behavior.
