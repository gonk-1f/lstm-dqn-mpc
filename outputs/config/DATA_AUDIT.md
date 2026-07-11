# Data Leakage & Split Audit — 2026-06-15

## 1. Leakage Check: PASS (0 overlaps)

| Check | Overlap | Status |
|-------|---------|--------|
| TEST vs old-TRAIN (LSTM checkpoint) | 0 | ✓ |
| TEST vs new-TRAIN | 0 | ✓ |
| TEST vs VAL | 0 | ✓ |
| VAL vs TRAIN | 0 | ✓ |
| All 35 voyages assigned | 35/35 | ✓ |

## 2. Split Summary

```
TRAIN: 24 voyages, 170.9h — 22 LSTM-trained + 2 new (5月24日, 7月9日)
  Conditions: CRUISE(12) LIGHT(6) HEAVY(4) PEAK(2)
  Monthly:   Apr(2) May(13) Jun(8) Jul(1)
  Load mean: [20, 72] kW

VAL:   7 voyages, 102.9h
  Conditions: CRUISE(6) PEAK(1)
  Monthly:   May(1) Jun(3) Jul(3)
  Load mean: [35, 69] kW

TEST:  4 voyages,  43.0h
  Conditions: DOCKED(1) CRUISE(2) HEAVY(1)
  Monthly:   May(1) Jun(2) Jul(1)
  Load mean: [8, 52] kW
```

## 3. Problems Found

### 3.1 Test voyage "6月7日 00-24" is DOCKED (mean load=8kW)

This voyage has **zero LSTM training coverage** — the 22-voyage training set contains no docked/anchor conditions (minimum train mean load = 20kW). The LSTM has never seen load profiles this low.

**Impact**: Test metrics are skewed by a condition the model was never trained to handle.
**Recommendation**: Either (a) move 6月7日 to train/val and replace with a CRUISE voyage, or (b) keep it but report metrics separately for this outlier.

### 3.2 VAL lacks condition diversity

VAL has only CRUISE(6) + PEAK(1). No LIGHT, HEAVY, or DOCKED. This means hyperparameter selection (for DQN) will overfit to medium-load cruising.

### 3.3 TRAIN has 2 new voyages LSTM not trained on

5月24日 and 7月9日 are in new-TRAIN but NOT in the old 22-voyage Phase3 training set. The current LSTM checkpoint was trained without these. This is minor — they add diversity for DQN training, but LSTM hasn't benefited from them.

### 3.4 Small TEST size

4 voyages (43h) is the minimum viable test size. With a DOCKED outlier, only 3 meaningful comparison voyages remain (33h).

## 4. Operating Condition Identification

The 35 voyages span 5 distinct conditions:

| Condition | Mean Load | Count | In Train | In Val | In Test |
|-----------|----------|-------|----------|--------|---------|
| DOCKED    | <15 kW   | 1     | 0        | 0      | 1       |
| LIGHT     | 15-30 kW | 6     | 6        | 0      | 0       |
| CRUISE    | 30-50 kW | 21    | 12       | 6      | 2       |
| HEAVY     | 50-65 kW | 5     | 4        | 0      | 1       |
| PEAK      | >65 kW   | 2     | 2        | 1      | 0       |

**Verdict**: The current split is a simple random 7:2:1 without condition stratification. This is acceptable if the system is expected to operate in all conditions, but the DOCKED voyage in TEST is problematic. A stratified split would be more rigorous for paper/publication purposes.

## 5. Recommendations

1. **Immediate**: Move 6月7日 DOCKED voyage from TEST to TRAIN (it's a data gap, not a test challenge). Replace with a CRUISE or HEAVY voyage from VAL.
2. **DQN training**: The 2 new train voyages (5月24日, 7月9日) add useful diversity.
3. **Future**: Consider stratified splitting by operating condition for final evaluation.
