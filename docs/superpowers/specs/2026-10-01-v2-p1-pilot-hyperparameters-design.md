# v2 P1 Pilot Hyperparameters Design

## Objective

Add an auditable P1 training profile derived from corrected H4 while changing only replay capacity and the epsilon decay length. Preserve the historical H1-H4 profiles, dataset, state/action contracts, MPC/economic semantics, and Test isolation.

## Frozen P1 configuration

- profile id: `P1`
- reward: scaled by the existing Train-only `Cref`
- learning rate: `1e-3`
- replay capacity: `75_000`
- epsilon: linear `1.0 -> 0.05` over `100_000` global DQN macro steps
- batch size: `256`
- warmup: `5_000` transitions
- target synchronization: `1_000` optimizer steps
- seed: `42`
- first run budget: `15` rounds
- continuation budget: `30` rounds using the same `latest.pt`

## Semantics

P1 is a new pilot profile, not a rewrite of H4. Epsilon parameters belong to the checkpoint-bound training configuration so a P1 checkpoint cannot be resumed under a different exploration schedule. Existing checkpoints that predate this identity extension are accepted only with the historical epsilon defaults (`1.0`, `0.05`, `150_000`).

The first 15 rounds are an engineering and early-learning check. At roughly 55,000 transitions they cannot establish the effect of replay eviction. If healthy, the same run is resumed to 30 rounds, allowing the 75,000-capacity buffer to roll over and the 100,000-step epsilon schedule to finish.

## Validation and Test isolation

Training continues to evaluate Validation greedily after every round without replay writes or optimizer updates. Test payloads remain unopened. P1 output uses a separate directory and does not overwrite H4 artifacts.

## Verification

Focused tests must prove the P1 profile values, schedule values, legacy H4 preservation, checkpoint rejection on schedule mismatch, CLI selection, and 15-to-30 resume compatibility. Then run the relevant v2 suite, compile/import checks, and `git diff --check`.
