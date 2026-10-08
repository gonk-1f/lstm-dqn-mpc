# v2 M=10 action-hold ablation implementation plan

1. Add red tests for the M=10 timescale/profile, macro-step-scaled training
   parameters, environment hold length, and independent checkpoint/reward-scale
   semantics.
2. Generalize formal MPC/backend/environment construction to accept an explicit
   immutable `TimeScaleConfig`, while retaining M=5 defaults.
3. Bind checkpoint save/load and reward-scale build/load to the requested
   timescale so cross-M reuse is rejected.
4. Add an isolated M=10 profile and CLI using warmup 2,500, epsilon decay
   75,000, replay 100,000, H4 learning settings, and a dedicated output root.
5. Add an M=10 Train-only calibration entrypoint and validation-only comparison
   reporting cost, action distribution, entropy, and maximum action share. Do
   not load Test.
6. Regenerate current authenticated economic/preflight artifacts, run focused
   tests, all v2 tests, solver smoke, compile/import, and `git diff --check`.
