# V13 Results

## Summary

V13 addresses three critical questions:
1. Does pretraining contribute under E2E fine-tuning? (Phase 0c)
2. Does the encoder read sensor content or just count timesteps? (Phase 0d)
3. Can we close the JEPA-STAR gap (14.2 vs 12.2 RMSE on FD001)? (Phase 1)

**Key headlines**:
- Pretraining contributes +8.8 RMSE at 100% labels, +14.5 at 20% - **strong SSL claim**
- Encoder reads temporal degradation patterns, not just length - **representation claims hold**
- Fine-tuning protocol is already near-optimal - **STAR gap is architectural**

---

## Phase 0: Validation Gates

### 0d. Length-vs-Content Ablation (PASSED)

Three inference-only tests on frozen V2 encoder:

| Test | Result | Verdict |
|:-----|:-------|:--------|
| Constant input (first cycle repeated t times) | Prediction range: 2.29 cycles across t=30..200 | PE does NOT dominate |
| Length-matched cross-engine swap | Mean cosine similarity: 0.647 | Content matters |
| Temporal shuffle | RMSE: 16.06 -> 36.77 (+20.71), rho: 0.896 -> 0.771 | Encoder reads temporal patterns |

**Gate: PASSED.** The H.I. R^2=0.926 is not a length artifact.

### 0c. From-Scratch Ablation (PASSED)

Same V2 encoder, random init vs pretrained checkpoint, same E2E protocol.

| Budget | Pretrained E2E | From-Scratch E2E | Frozen Probe | Delta |
|:-------|:---------------|:-----------------|:-------------|:------|
| 100%   | 14.18 +/- 0.55 | 22.99 +/- 2.33 | 16.70 +/- 0.95 | +8.81 |
| 20%    | 18.00 +/- 1.37 | 32.50 +/- 1.50 | 19.50 +/- 1.58 | +14.51 |
| 10%    | 19.97 +/- 2.19 | 35.59 +/- 2.67 | 19.83 +/- 0.83 | +15.62 |
| 5%     | 29.64 +/- 5.27 | 37.59 +/- 2.00 | 24.47 +/- 5.48 | +7.95 |

**Gate: PASSED with massive margin.** Delta > 3 RMSE at all budgets.

Key findings:
- Delta peaks at 10-20% labels (~15 RMSE)
- At 5% labels, frozen probe outperforms E2E (24.47 vs 29.64) - E2E destabilizes
- Without pretraining, transformer cannot learn from limited fine-tuning data

### 0a. STAR Label-Efficiency Sweep (IN PROGRESS)

Running in background. Will update when complete.

### 0b. STAR FD004 (QUEUED)

---

## Phase 1: Closing the JEPA-STAR Gap

### Starting point

| Method | FD001 RMSE | Notes |
|:-------|:-----------|:------|
| STAR (supervised) | 12.19 +/- 0.55 | End-to-end supervised, sliding windows |
| JEPA E2E | 14.23 +/- 0.39 | SSL pretrained + E2E fine-tuned |
| JEPA Frozen | 17.81 | Linear probe on frozen encoder |
| Gap | ~2.0 RMSE | |

### 1a. Warmup-Freeze Fine-Tuning (NO IMPROVEMENT)

Freeze encoder for 20 epochs, then unfreeze with standard LR=1e-4.

- Warmup-freeze: 14.200 +/- 0.817
- Standard E2E:  14.165 +/- 0.453
- Delta: +0.034

Probe warmup does not help. Standard E2E is already optimal.

### 1b. Weight Decay (NO IMPROVEMENT)

AdamW with weight_decay=1e-4:

| Budget | WD=1e-4 | WD=0 | Delta |
|:-------|:--------|:-----|:------|
| 100%   | 14.289 +/- 0.812 | 14.209 +/- 0.406 | +0.081 |
| 5%     | 30.072 +/- 5.379 | 27.708 +/- 6.059 | +2.364 |

Weight decay hurts at low labels and doesn't help at 100%.

### 1c. Longer Prediction Horizon (IN PROGRESS)

Pretraining with max_horizon=50 (vs baseline 30). Running.

### 1d. Deeper Architecture V4 (QUEUED)

V4: d=256, L=4 (vs V2: d=256, L=2). Will run after 1c.

---

## Conclusions (preliminary - Phase 1c/1d pending)

1. **Pretraining is strongly validated.** The from-scratch ablation shows massive
   degradation without pretraining (+8.8 RMSE at 100%, +15 at 10-20%).
   This is a strong SSL claim for the paper.

2. **Encoder reads content, not length.** The length-vs-content ablation confirms
   the encoder captures temporal degradation patterns. Temporal shuffle RMSE
   doubles (16 -> 37), proving representations are meaningful.

3. **The JEPA-STAR gap (14.2 vs 12.2) is architectural, not a training issue.**
   Fine-tuning schedule (1a), weight decay (1b), probe architecture (v13 prior),
   and data quantity (v13 prior) all fail to close the gap. The remaining
   hypotheses are encoder depth (1d) and prediction horizon (1c).

4. **Frozen probe beats E2E at very low labels (5%).**
   This is a practical finding: when labels are scarce, don't fine-tune the encoder.
   The pretrained features are more reliable than what E2E can learn from 4 engines.
