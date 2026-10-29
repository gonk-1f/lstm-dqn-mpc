# Official Zoo vessel adapter implementation plan

**Goal:** Integrate the frozen V6 environment and masked DQN into actual Zoo 2.9.0 training/Optuna, isolated from the existing episode runner.

**Architecture:** The new worktree starts at 17f3da8. All adapters live in experiments/sb3_manual/zoo_vessel. A Gym composition wrapper fixes Train/Validation selection and delegates every physical step to DirectVesselEnv. ZooMaskedDQN inherits the existing masked TD update, adds SB3 model.learn compatibility, restores native timestep epsilon, and explicitly preserves the post-warmup formal Target clock. A scoped Zoo algorithm key and ExperimentManager extension call official training/optimization; callbacks evaluate eight independent Validation voyages and use completion-first CNY scoring. Native SB3 Logger writes stdout, CSV and TensorBoard with a 1000-step dump callback; no alternative table printer.

**Environment:** Dedicated .venv-zoo with Zoo2.9.0, installed-user-equivalent SB3/contrib2.9.2, CPU and one Torch thread. Existing v6-sb3-baseline source, environments, user staged CartPole files and paused checkpoints are untouched. Shared frozen dataset is read-only with explicit data-root; no Test payloads.

- [x] Verify installed primary-source extension APIs and preserve old-worktree status.
- [x] RED/GREEN env registration, seeded Train35 selection, exact Validation8 cycle, masks and real terminal metadata.
- [x] RED/GREEN model.learn action masks, successor Replay masks, terminal TD, native epsilon, warmup boundary and Target/update counts.
- [x] RED/GREEN native progress logging and complete independent Validation; do not insert synthetic costs for incomplete runs.
- [x] RED/GREEN scoped ExperimentManager integration, independent identities, custom Optuna sampler/callback and standalone evaluation.
- [x] Bounded real Train training, native table capture, save/load/evaluate all8Validation, one short official Zoo Optuna Trial; verify CSV/TensorBoard and counters.
- [x] Review, document exact commands, official/adapted boundaries and limitations. No formal500, long search or Test. Leave final experiments to the user.

Budgets use actual Zoo/SB3 timesteps (including warmup), not a claimed equivalent of 500 full voyages. A final collection block may extend to its documented rollout boundary; no artificial voyage termination is introduced. Historical episode-runner comparison requires reporting this independent exploration/budget contract.
