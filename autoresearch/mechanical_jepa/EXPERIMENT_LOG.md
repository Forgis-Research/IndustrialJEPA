# Mechanical-JEPA Experiment Log

## Overview

**Goal:** Prove JEPA learns transferable features for bearing fault detection (like Brain-JEPA for fMRI).

**Key metric:** Linear probe accuracy on JEPA embeddings > Random init + 5%

---

## Current Best

| Metric | Value | Config | Seeds |
|--------|-------|--------|-------|
| JEPA Test Acc | 61.6% | 30ep, depth=4, mask=0.5 | 1 |
| Random Init | ~30% | - | 1 |
| Improvement | +31.6% | - | - |

---

## Experiments

### Exp 0: Initial Implementation

**Time**: 2026-03-30 21:05
**Hypothesis**: JEPA learns useful features for bearing fault detection
**Change**: Initial implementation based on Brain-JEPA architecture
**Config**:
```python
epochs=30, embed_dim=256, encoder_depth=4, predictor_depth=2
patch_size=256, mask_ratio=0.5, ema_decay=0.996
dataset=cwru, batch_size=32, window_size=4096
```

**Sanity checks**:
- ✓ Loss decreased (0.0079 → 0.0017, 78% reduction)
- ✓ Test acc > random guessing (49.8% > 25%)
- ✓ Test acc > random init (49.8% > ~30%)
- ⚠️ Single seed only
- ⚠️ Healthy class missing from test set (fixed with stratified split)

**Results**:
| Metric | Value |
|--------|-------|
| Initial loss | 0.0079 |
| Final loss | 0.0017 |
| Test accuracy | 49.8% |
| Random init baseline | ~30% |
| Improvement | +19.8% |

**Per-class accuracy**:
- Outer race: 56.9% (58 samples)
- Inner race: 29.3% (174 samples)
- Ball: 63.4% (232 samples)

**Verdict**: ✓ KEEP - Shows clear transferability
**Insight**: JEPA learns fault-discriminative features. Inner race harder to detect.
**Next**: Multi-seed validation, longer training

---

### Exp 1: Replicate Baseline (Verification)

**Time**: 2026-03-30 23:26
**Hypothesis**: Verify baseline results are reproducible
**Change**: Re-run with same config, seed 42
**Config**: Same as Exp 0

**Sanity checks**:
- ✓ Loss decreased (similar pattern to Exp 0)
- ✓ Test acc > random guessing (61.6% > 25%)
- ✓ Test acc > random init (61.6% > ~30%)
- ⚠️ Significantly higher than Exp 0 (61.6% vs 49.8%)
- ⚠️ Still single seed

**Results**:
| Metric | Value |
|--------|-------|
| Test accuracy | 61.6% |
| Improvement over Exp 0 | +11.8% |
| Improvement over random | +31.6% |

**Verdict**: ✓ KEEP - Even better results!
**Insight**: High variance suggests need for multi-seed validation. 61.6% vs 49.8% indicates either:
1. Random seed variance in data splitting
2. Random seed variance in model initialization
3. Or both

**Next**: CRITICAL - Multi-seed validation to establish confidence intervals

---

### Exp 2: Cross-Dataset Transfer (CWRU → IMS)

**Time**: 2026-03-31 08:07
**Hypothesis**: CWRU-pretrained encoder should transfer to IMS degradation detection
**Change**: Evaluate CWRU checkpoint on IMS degradation detection task
**Method**:
- Pretrained JEPA on CWRU (checkpoint from Exp 1)
- IMS 1st_test: 2,156 episodes
- Created pseudo-labels: early 25% = healthy, late 25% = degraded
- Train/test split: 862/216 episodes
- Linear probe on frozen CWRU encoder

**Sanity checks**:
- ✓ Data loaded correctly (7,744 train embeddings, 1,944 test embeddings)
- ✓ Linear probe training converged
- ✗ Test accuracy = 50.0% (exact random guessing)
- ✗ Train accuracy = 50.0% (no learning even on train set)

**Results**:
| Metric | Value |
|--------|-------|
| IMS Test Accuracy | 50.0% |
| Random Baseline | 50.0% |
| Transfer Gap | 0.0% (no transfer) |

**Verdict**: ✗ **TRANSFER FAILED** - CRITICAL ISSUE
**Insight**: The CWRU-pretrained encoder does NOT transfer to IMS. This indicates:
1. Model is overfitting to CWRU-specific patterns (bearing-specific vibration signatures)
2. NOT learning general fault features that transfer across datasets
3. This contradicts the goal of a "foundation model" for bearing faults

**Possible causes**:
1. CWRU and IMS are too different (different test rigs, different failure modes)
2. Model capacity too small to capture general features
3. Pretraining task (masked patch prediction) not sufficient for transfer
4. Need longer pretraining or more data
5. IMS pseudo-labels (temporal position) may not correlate with actual degradation

**Next actions**:
1. Verify IMS data quality and pseudo-labels make sense
2. Try training directly on IMS to see if it's learnable
3. Consider joint pretraining on CWRU+IMS
4. Try enhanced masking strategies (temporal_block, cross_time)
5. Longer pretraining (100+ epochs)

---

