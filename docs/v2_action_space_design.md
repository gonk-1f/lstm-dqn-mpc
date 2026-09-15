# v2 Candidate Action Space and Screening Boundary

## Status

The formal v2 DQN action catalog is **NO-GO / unset**. The repository has a
canonical candidate bank, but it does not have usable Train operating-cycle
data or a completed solver reproducibility audit. Therefore
`FINAL_DQN_ACTION_CATALOG` is `None`, and the default catalog getter fails
closed. Nothing in this implementation claims that real-data screening has
run, that a catalog size `K` has been selected, or that any candidate is fit
for formal training.

The frozen compatibility identifier is
`three_weight_simplex_behavior_filtered_v1`, matching the v2 contracts module.

## Candidate bank

An MPC action has three positive objective weights:

- `q_base`: causal base-reference tracking;
- `q_smooth`: fuel-cell power smoothing; and
- `q_soc`: SOC deadband protection.

The candidate generator enumerates every positive integer composition

`n_base + n_smooth + n_soc = 10`, with every numerator at least one,

then derives each weight as `n_i / 10`. There are exactly
`C(9, 2) = 36` candidates. The integer triple is the canonical identity;
floating-point values are derived outputs and never used for uniqueness or
enumeration. Ordering is lexicographic in `(n_base, n_smooth, n_soc)`, and IDs
are `w_<n_base>_<n_smooth>_<n_soc>`. Both order and IDs are independent of
Python hashes and runtime container ordering. Each candidate converts directly
to Task 5 `MPCWeights`.

The 36-member `CANDIDATE_ACTION_BANK` is an experimental search domain, not a
DQN catalog. Code must not substitute it for the unset final catalog.

## Train-only evidence model

Every screening record identifies an exact `DatasetProvenance` containing a
dataset version, provenance ID, and `DataSplit`. Callers cannot pass strings,
booleans, enum lookalikes, or subclasses as a split or provenance substitute.
Selection accepts only the exact `DataSplit.TRAIN` member. Validation, Test,
Unknown, mixed provenance, duplicate candidate IDs, and inconsistent metric
schemas are rejected before filtering, so even an infeasible held-out row
cannot be silently ignored and influence a decision.

Thresholds, cluster assignments supplied to medoid selection, and the final
selected IDs each require an explicit provenance argument that must exactly
match the Train records. This includes choices that would otherwise be easy to
leak from held-out data: near-duplicate distance, clustering distance,
representatives, and the final catalog. Validation and Test are reserved for
evaluation after a future Train-selected catalog is frozen.

Feasibility and solver reproducibility are separate first-class results. They
are hard gates, not extra Pareto objectives and not penalty values hidden in a
behavior vector. A reproducibility result or audit records at least two runs,
an explicit pass/fail value, provenance, and a nonempty evidence identifier or
reason.

## Behavior fingerprint contract

A fingerprint contains a candidate ID, an immutable tuple of metric
definitions, and an immutable tuple of values. Construction defensively copies
array-like inputs. All values and scales must be finite; scales must be
positive; metric names must be nonempty and unique; value length must exactly
match the schema. Each metric declares `MINIMIZE` or `MAXIMIZE` plus a scale.
The screening algorithms transform these to a normalized lower-is-better
vector. No mutable NumPy alias or NaN can enter a stored fingerprint.

The metric schema itself is intentionally not frozen here. A future Train-only
audit must state the physical metrics, directions, and engineering or
Train-derived scales it uses.

## Deterministic screening pipeline

The intended future flow is:

1. Evaluate all 36 candidates on one explicitly versioned Train dataset.
2. Apply the feasibility and repeated-solve reproducibility hard gates.
3. Compute the Pareto front using each metric's declared direction.
4. Remove near-duplicates with a declared finite nonnegative normalized
   Euclidean-distance threshold. Candidate-ID order chooses the representative
   deterministically.
5. Cluster remaining behavior fingerprints using an explicitly selected
   Train-only distance threshold. The provided implementation uses
   deterministic single-linkage connected components.
6. Select one medoid per supplied cluster by minimum total normalized distance;
   candidate ID breaks exact ties.
7. Freeze the resulting representative count as `K` only after the Train audit
   is reviewed. This module does not invent or freeze `K`.

The Pareto, de-duplication, clustering, and medoid functions are small
dependency-light algorithms that can be exercised with synthetic records now.
They are interfaces for a future empirical audit, not evidence that the audit
already happened.

## Finalization gate

The generic finalization boundary requires all of the following:

- complete, unique Train-only screening evidence for every candidate in the
  supplied bank;
- matching dataset version and provenance for screening records, threshold and
  selection decisions, data readiness, and solver audit;
- explicit usable-data readiness evidence;
- a passed repeated-solve audit covering the complete candidate bank; and
- nonempty selected IDs drawn only from the bank, each passing both hard gates.

Complete evidence may correctly show that some unselected candidates failed a
hard gate. Such failures are a reason to remove them, not a reason to pretend
the audit is incomplete. Selecting one of those failed candidates is rejected.

Passing this generic function with synthetic evidence does not mutate module
constants or publish a repository catalog. Under the current raw-data status,
the production getter remains NO-GO until a future reviewed Train-only workflow
supplies real usable data, audited solver evidence, documented thresholds, and
a frozen representative set. Validation and Test must not participate in that
selection.
