# V6 single-configuration training result

Status: training; completed traversals: 1/8.
Warmup: 50000 genuine transitions, 0 Adam updates.
Formal training: 275000 executed transitions; Adam updates: 275000; Target copies (including initialization): 276.
Test payloads opened: 0. No second configuration was trained.

|Traversal|Train /35|Validation /8|Train steps|Adam|Target copies|Epsilon start/end|
|---|---:|---:|---:|---:|---:|---|
|1|15|0|236097|236097|236|1.000000/0.841667|

Research candidate: traversal 1.
Selection prioritizes Validation completion count, then duration coverage; full-set cost is used when all eight complete.
Incomplete-prefix costs are never represented as a fixed complete-set cost.

## Comparison with V5 round32

Different time resolution, action availability, state, objective and training budget prevent a single-factor causal comparison.
Change/reversal rates use actual physical hours; per-step event counts are not compared directly.
Jointly completed Validation IDs at the final traversal: 0.

Final recorded Validation completion: 0/8; V5 round32:5/8.
V6 FC changes/hour: 1977.787447; adjacent reversals/hour: 827.553573.
These rates include actual failed prefixes and describe different policy paths; same-ID complete costs are reported separately.
Each completed traversal preserves round_NN_failure_details.csv and .json with Train/Validation failure states and all 13 candidate legality flags.
Train failures occurred under epsilon-greedy exploration; per-action random-choice flags were not stored, so a random action cannot be identified as their cause.

Validation CSV.gz and PNG files contain load, FC power, battery power, SOC, FC increments,
all13 action Q values, legal hold rank, separate reward terms and cumulative reward terms.
Fixed probes preserve V5 physical SOC/FC/time and add V6 causal history. They have a different input/output
definition and reward unit from V5; raw numerical Q values must not be compared as identical targets.
See fixed_probe_history.json and v5_v6_validation_comparison.csv.

Data reuse preserved every canonical original anchor exactly. No new real1Hz measurements were created.
The ±60kW/s action capability remains a simulation assumption.

## Existing fixed physical probes in V6 coordinates

|Sample|Time/s|Initial hold rank /13|Latest hold rank /13|Greedy increment/kW|Greedy minus hold Q|
|---|---:|---:|---:|---:|---:|
|parent_011_seg01|750|1|2|10|0.038803|
|parent_011_seg01|780|1|2|10|0.024052|
|parent_045_seg01|870|1|1|0|0.000000|
|parent_045_seg01|900|1|1|0|0.000000|
