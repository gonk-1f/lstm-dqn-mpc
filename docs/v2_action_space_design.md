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

Every feasibility result, solver result, and behavior fingerprint identifies
an exact `DatasetProvenance` containing a dataset version, provenance ID, and
`DataSplit`. A `CandidateScreeningRecord` can be constructed only when all
three evidence objects carry the same exact Train provenance. Callers cannot
pass strings, booleans, enum lookalikes, subclasses, Validation, Test, or
Unknown as substitutes. Mixed provenance, duplicate candidate IDs, and
inconsistent metric schemas are rejected before filtering, so even failed
held-out evidence cannot be silently ignored and influence a decision.

The selection API is a sealed chain:

`HardGateResult -> ParetoResult -> NearDuplicateResult -> ClusteringResult -> MedoidSelectionResult`

Only the hard-gate entry point accepts raw records. Every later operation
requires the exact preceding result type. Near-duplicate and clustering
thresholds are not numeric API arguments: `derive_distance_threshold` seals a
`DistanceThresholdEvidence` tied to the exact parent object, parent digest,
Train provenance, registered rule, and audit ID. The registered rules are
`ZERO`, `MIN_POSITIVE_PAIRWISE`, and `MEDIAN_PAIRWISE`; their numeric values are
derived solely from the exact parent's normalized Train fingerprints. The
pairwise rules ignore infinite distances and deterministically fall back to
zero when no usable finite pair exists.

Threshold evidence, cluster assignments, selected representatives, audit IDs,
parent lineage, and Train provenance are embedded in immutable stage results.
Each result has a deterministic digest, and consumers recursively revalidate
parent content, derived output, object identity, and digest before proceeding.
This prevents a detached Train label from laundering a held-out numeric
threshold, arbitrary clusters, or raw selected IDs. Validation and Test are
reserved for evaluation only after a future Train-selected catalog is frozen.

This is an enforcement boundary for declared, audited provenance. Software
cannot prove that a human did not inspect held-out results before choosing a
registered threshold rule or audit decision. The numeric threshold is derived
only from Train records, but the rule choice remains an auditable human choice.
Preventing that procedural leak still requires access controls, audit review,
and documented experiment governance outside this module.

Feasibility and solver reproducibility are separate first-class results. They
are hard gates, not extra Pareto objectives and not penalty values hidden in a
behavior vector. A reproducibility result or audit records at least two runs,
an explicit pass/fail value, provenance, a nonempty evidence identifier, and a
reason. Data-readiness evidence and the complete solver audit carry
deterministic digests; finalization recomputes them so `object.__setattr__`
changes to pass status, covered IDs, provenance, repeats, audit ID, or reason
are rejected.

## Behavior fingerprint contract

A fingerprint contains a candidate ID, exact dataset provenance, an immutable
tuple of metric definitions, and an immutable tuple of values. Construction
defensively copies array-like inputs. All values, scales, and normalized values
must be finite; this includes rejecting finite value/scale inputs whose
division overflows. Scales must be positive; metric names must be nonempty and
unique; value length must exactly match the schema. Each metric declares
`MINIMIZE` or `MAXIMIZE` plus a scale. The screening algorithms transform these
to a normalized lower-is-better vector. No mutable NumPy alias or NaN can enter
a stored fingerprint.

The metric schema itself is intentionally not frozen here. A future Train-only
audit must state the physical metrics, directions, and engineering or
Train-derived scales it uses.

## Deterministic screening pipeline

The intended future flow is:

1. Evaluate all 36 candidates on one explicitly versioned Train dataset.
2. Apply the feasibility and repeated-solve reproducibility hard gates.
3. Compute the Pareto front using each metric's declared direction.
4. Choose a registered, audited threshold rule and derive sealed threshold
   evidence from the Pareto result's Train fingerprints. Remove near-duplicates
   using that evidence. Candidate-ID order chooses the representative
   deterministically. Stable `math.dist` evaluation avoids overflow from
   squaring large finite coordinates; a genuinely overflowing distance is
   treated deterministically as infinity.
5. Choose and audit a registered rule for the de-duplicated result, derive its
   sealed Train-only threshold evidence, and cluster with deterministic
   single-linkage connected components.
6. Select one medoid per supplied cluster by minimum total normalized distance;
   candidate ID breaks exact ties. Medoid scoring first computes all pair
   distances, divides finite distances by one cluster-wide finite scale, and
   uses `math.fsum`. This prevents a sum of finite extreme distances from
   overflowing into a false tie; any genuinely infinite distance still gives
   that candidate an infinite score.
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
- one recursively valid, complete `MedoidSelectionResult` derived from those
  36 source records, with no raw selected-ID or detached provenance escape;
- matching exact Train provenance for the entire pipeline, data readiness, and
  solver audit;
- explicit digest-validated usable-data readiness evidence;
- a digest-validated, passed repeated-solve audit covering the complete
  candidate bank; and
- nonempty selected records drawn only from the bank, each passing both hard
  gates.

Finalization never trusts the mutable module-level candidate-bank alias. It
generates a fresh canonical 36-action bank on every call, reconstructs and
revalidates every supplied exact `ActionCandidate` (integer fields, ID,
composition invariants, and `MPCWeights` conversion), compares against that
fresh bank, and returns fresh canonical action objects.

Complete evidence may correctly show that some unselected candidates failed a
hard gate. Such failures are a reason to remove them, not a reason to pretend
the audit is incomplete. Selecting one of those failed candidates is rejected.

Passing this generic function with synthetic evidence does not mutate module
constants or publish a repository catalog. Under the current raw-data status,
the production getter remains NO-GO until a future reviewed Train-only workflow
supplies real usable data, audited solver evidence, documented thresholds, and
a frozen representative set. Validation and Test must not participate in that
selection.
