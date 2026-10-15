# Energy/SOC reward with preserved PER implementation plan

> **For agentic workers:** Use subagent-driven development for independent training-interface and regression-test tasks. Preserve the original worktree and all existing results.

**Goal:** Prepare a separately runnable DDQN candidate using hydrogen, purchased shore electricity and the existing SOC reference penalty, with the already preserved proportional PER implementation. Do not start formal training.

**Architecture:** Branch `feat/v4-energy-soc-per` starts at saved PER commit `6a470de`, whose parent is baseline `9cd6063`. Add `reward_cost_mode=energy_only` to the existing replay/monitoring path; default `full_economic` preserves baseline behavior. Never change the physical accountant or four-component ledger. Both intermediate fixed-target shore purchase and modeled terminal purchase attach once to actual transitions. Existing n8, failure handling, UNKNOWN boundaries and model qualification stay intact.

**Tech Stack:** Python, PyTorch, existing v4 CLI and targeted synthetic pytest checks.

## Tasks

- [x] Add synthetic regression checks before implementation: identical physical/ledger results under both reward modes; shore attachment and no duplicate terminal settlement; n8 returns containing one source shore charge; one failure penalty and no failed terminal recharge; UNKNOWN handling.
- [x] Add `REWARD_COST_MODES`, a ledger reward selector, and per-transition excluded degradation diagnostics in `control.py` / `diagnostics.py`. Retain FC and battery aging updates and all ledger charges.
- [x] Route `reward_cost_mode` through CLI, public/private monitored wrappers, bootstrap, exploration and greedy evaluation. Record weights/formula in metadata and compatibility checks; existing checkpoint hyperparameter comparison rejects a different reward mode.
- [x] Keep PER alpha=.6, IS beta=1, epsilon=1e-6 and failure quota2 exactly as preserved. Retain learning rate .001, gamma1, n8, replay32, tau .001, batch64, replay300000, betaSOC200, 100 rounds and seed42.
- [x] Run only directly relevant synthetic tests and CLI loading checks. Verify manifests by metadata only; do not read formal Validation/Test payloads or run training.
- [x] Review the diff and provide PowerShell fresh-run/resume commands with independent output path, PYTHONPATH and verified interpreter. No push requested.

## Acceptance and interpretation

- Normal candidate reward is `-(H2 + shore_purchase + betaSOC*(SOC_next-.6)**2)` before the existing single replay scale .001; failure penalty remains separate and once per source trajectory.
- Four economic ledger components, SOC, masks and actual selected power execution match the baseline for a fixed action sequence.
- Cross-shore successors are real post-charge states with `done=False`; actual sample end/failure is terminal; UNKNOWN has no invented terminal charge or successor.
- Shore billing once is compatible with overlapping n-step training windows: replay reuses a cost observation, never rebills the ledger.
- Delayed shore feedback is observable through TD, but does not guarantee successful fitting or anticipatory SOC control. Do not add immediate battery value feedback, artificial FC gating or new penalties.
