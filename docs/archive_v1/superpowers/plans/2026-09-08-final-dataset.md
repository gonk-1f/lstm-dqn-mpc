# Final operating dataset implementation plan

**Goal:** Build a new reproducible raw-telemetry dataset with distinct training, validation and natural-voyage test responsibilities. The user's attached specification is authoritative.

**Architecture:** New data-only modules read original device files, derive and freeze timing/noise/stop rules, freeze parent roles, classify physical states, select semantically valid samples, and reconstruct only accepted intervals. Old datasets, controller code, model artifacts and existing staged changes remain untouched. No controller rollout is permitted.

**Stack:** Python, NumPy, pandas, SciPy sparse linear programming, existing pure PCHIP/AIS helpers, matplotlib.

## Tasks and acceptance

- [x] Snapshot existing files and Git index. Derive original channel offset, FC shutdown distribution and contextual stationary-duration distributions without reading controller outcomes.
- [x] Add `src/utils/final_dataset_feasibility.py` and focused tests. Verify a controller-independent feasible allocation against all power, energy, SOC and ramp bounds. An unknown solver outcome is not infeasibility. Save witnesses for accepted test voyages.
- [x] Add `src/utils/final_dataset_source.py`: raw file hashes, deterministic duplicate policy, unique monotone nearest matching, repaired timestamp/channel provenance, empirical policy estimation, raw parent roles fixed before sample extraction.
- [x] Add `src/utils/final_dataset_semantics.py`: three charging classes, AIS-led stop context, explicit gap/quality exclusions and role-specific natural-boundary acceptance. Sensor ambiguity must remain ambiguous. Never synthesize missing low-speed endpoints.
- [x] Add `src/main/build_final_operating_dataset.py`: fresh output directory only, all 14 required metadata tables, per-point accounting, cleaned telemetry, 1 s samples, coverage, test figures, independent QA and provenance to original parents. Preserve old IDs only as overlap references.
- [x] Add synthetic tests for ties/reuse, small async offsets, genuine gaps, ambiguous charging, retained operational stops, natural test boundaries, parent isolation and exhaustive exclusion accounting. Run failing tests before implementation.
- [ ] Build `data/processed/operating_dataset_final` from original inputs. Do not invoke training or controller validation/test. Strictly exclude uncertain cases instead of meeting a sample-count quota.
- [ ] Review every selected test figure and feasibility witness; run independent artifact verification, deterministic second-build comparison in a separate task-owned output directory, and code/spec review. Record actual tests and limitations.
- [ ] Compare protected input/output hashes and original Git index/status. Do not stage or commit because unrelated staged deletions exist. Report actual counts and freeze recommendation; do not change controller data defaults in this task.

## Design decisions

Keep the existing chronological 46/13/7 parent assignment if the seven held-out parents supply acceptable natural voyages. Parent assignment is based on raw folder chronology only, before extraction; no historical reward, feasibility exclusion list, or controller trace may influence selection. Dataset roles and eligibility are distinct: unused parents retain their assigned role and receive explicit exclusion reasons.

Power samples are physically valid measured channels only. Conflicting duplicate measurements are flagged rather than silently averaged. Time offset tolerance is derived from unique-nearest source offsets with a conservative cap below half the source cadence. Gaps exceeding the frozen cadence-supported gap limit cannot be bridged by PCHIP.

Natural boundaries require observed low-speed endpoints and sustained sailing inside, with terminal dwell retained only for an empirically defined settling interval. Unknown/missing boundary evidence fails validation/test eligibility. Short/long stops are based on observed duration distributions plus before/after sailing context, not load alone. A missing direct external-supply sensor limits charging certainty; plausible but unconfirmed stationary FC charging is recorded as ambiguous and excluded.

Tests: `python -B -m unittest discover -s tests -p "test_final_dataset*.py" -v`. Build CLI and independent verification instructions will be documented alongside the completed implementation.

## Evidence-based split revision

The first raw-only review build found zero qualifying natural voyages in the original seven chronological Test parents, while 14 feasible natural candidates exist across 13 other parents. The user explicitly authorized redesign in this situation. `final_dataset_split.py` now surveys raw quality/natural completeness and independent physical existence, then fixes one parent role before formal extraction. Seven Test parents are chosen with deterministic coverage of duration, mean load, overload energy and load variation; remaining natural parents are prioritized for validation. Counts remain 46/13/7. All original/final roles and per-parent reasons are recorded. No boundary rule was relaxed and no controller output was inspected.

Eight selected Test figures were reviewed: seven route/short sailing tasks and one short low-speed harbor task; all retain observed stationary endpoints. None of the strict complete candidates has an internal short stop, so that coverage remains in Training. Historical model weights that saw reassigned Test parents cannot be used for a clean benchmark on this new split.
