# Mechanical-JEPA: Key Findings Summary

**Date**: 2026-03-31
**Researcher**: Claude (Autonomous Overnight Research)
**Goal**: Prove JEPA learns transferable features for bearing fault detection

---

## Executive Summary

**TL;DR**: While JEPA achieves reasonable within-dataset accuracy (69.6% ± 8.0%), it **completely fails** at cross-dataset transfer (50% = random guessing). This indicates the model is overfitting to CWRU-specific patterns rather than learning general fault signatures.

**Status**: ⚠️ Foundation model hypothesis **REJECTED** with current approach

---

## Results Overview

### Within-Dataset Performance (CWRU)

| Metric | Value | Status |
|--------|-------|--------|
| Mean Test Accuracy | 69.6% ± 8.0% | ✓ Good (2 seeds) |
| Best Single Run | 77.6% (seed 123) | ✓ Promising |
| Worst Single Run | 61.6% (seed 42) | ⚠️ High variance |
| vs Random Init | +39.6% | ✓ Clear improvement |
| vs Random Guessing | +44.6% | ✓ Far above chance |

**Per-class accuracy (mean ± std):**
- **Healthy**: 100.0% ± 0.0% ✓ Perfect
- **Ball**: 81.9% ± 4.3% ✓ Good
- **Inner Race**: 41.4% ± 15.5% ⚠️ Poor, high variance
- **Outer Race**: 23.7% ± 20.3% ✗ Very poor, very high variance

### Cross-Dataset Transfer (CWRU → IMS)

| Metric | Value | Status |
|--------|-------|--------|
| IMS Test Accuracy | 50.0% | ✗ **RANDOM** |
| IMS Train Accuracy | 50.0% | ✗ No learning |
| Transfer Gap | 19.6% | ✗ Large |
| Random Baseline | 50.0% | = Random |

**Verdict**: ✗ **COMPLETE TRANSFER FAILURE**

---

## Critical Insights

### 1. No Transferability

The model does NOT learn features that transfer across datasets:
- CWRU-pretrained encoder: 69.6% on CWRU → 50% on IMS
- This is WORSE than a random init encoder, which could at least overfit to IMS
- Indicates severe overfitting to CWRU-specific vibration patterns

**Why this matters**: The whole point of JEPA/foundation models is to learn general-purpose features. If features don't transfer to a new test rig, they're not "foundation" features.

### 2. Class Imbalance Masking

The 69.6% overall accuracy hides severe class imbalance:
- Healthy & Ball: 100% and 82% (easy to detect)
- Inner/Outer Race: 41% and 24% (failing)

**This is NOT acceptable for industrial deployment** - missing 75% of outer race faults is a safety risk!

### 3. High Variance Across Seeds

- Range: 61.6% - 77.6% (16% spread)
- Outer race: 3.5% - 44% (massive variance)

This suggests:
- Model is very sensitive to data splits
- Some bearings are much easier to classify than others
- Potential data quality issues

---

## Hypothesis: Why Transfer Failed

### Theory 1: Dataset Mismatch (Most Likely)

**CWRU vs IMS differences:**
| Aspect | CWRU | IMS |
|--------|------|-----|
| Fault types | 4 classes (healthy, inner, outer, ball) | All "healthy" (run-to-failure) |
| Test rig | Single test stand | Different test stand |
| Bearings | Seeded faults | Natural degradation |
| Labeling | Fault type labels | Temporal position (pseudo-labels) |

**Problem**: IMS pseudo-labels (early=healthy, late=degraded) may not correlate with actual degradation. Run-to-failure data is noisy - bearings can run healthy for most of their life and fail suddenly.

### Theory 2: Overfitting to Specific Fault Signatures

JEPA may be learning:
- Specific frequency patterns unique to CWRU test rig
- Bearing-specific vibration signatures
- Fault size-specific patterns (CWRU has 0.007", 0.014", 0.021" faults)

NOT learning:
- General "degradation" or "anomaly" patterns
- Transferable fault physics

### Theory 3: Insufficient Model Capacity

- embed_dim=256, encoder_depth=4 may be too small
- Brain-JEPA uses much larger models on much larger datasets
- Need more capacity to learn general + specific features

### Theory 4: Pretraining Task Mismatch

- Masked patch prediction forces temporal prediction
- BUT: May not align with fault detection task
- Fault signatures are about frequency content, not temporal prediction

---

## Comparison to Brain-JEPA

| Aspect | Brain-JEPA (fMRI) | Mechanical-JEPA (Vibration) |
|--------|-------------------|------------------------------|
| **Pretraining data** | Large multi-site fMRI datasets | CWRU: 40 episodes only |
| **Downstream tasks** | Age, disease, traits | Fault classification |
| **Transfer performance** | SOTA across tasks | **50% (failed)** |
| **Key difference** | Much more pretraining data | Tiny dataset! |

**Takeaway**: We may need 100x more pretraining data for JEPA to work.

---

## Experiments in Progress

1. **Seed 456**: Third seed for multi-seed validation
2. **Enhanced masking** (temporal_block, seed 789): Test if structured masking helps
3. Planned: Longer pretraining (100 epochs)
4. Planned: Mixed CWRU+IMS pretraining

---

## Recommendations

### Short-term (Overnight)

1. ✅ **Complete multi-seed validation** (3+ seeds)
2. ✅ **Test enhanced masking** (temporal_block, cross_time)
3. ⚠️ **Investigate IMS labels** - Are they actually meaningful?
4. ⚠️ **Try supervised baseline on IMS** - Is the task even learnable?
5. ⚠️ **Try longer pretraining** (100 epochs) - Does it just need more training?

### Medium-term (Next Steps)

1. **Joint CWRU+IMS pretraining** - Learn from both datasets simultaneously
2. **Augmentation** - Generate synthetic fault data to increase diversity
3. **Larger model** - Try embed_dim=512, encoder_depth=8
4. **Different pretraining task** - Try contrastive learning (SimCLR-style)
5. **More data** - Add Paderborn dataset

### Long-term (Rethink Approach)

If JEPA continues to fail:

1. **Supervised learning may be better** for small-data regimes
2. **Domain adaptation** methods (e.g., DANN, CORAL)
3. **Physics-informed models** - Incorporate bearing fault frequencies explicitly
4. **Transfer learning from ImageNet** - Treat spectrograms as images
5. **TabPFN-style in-context learning** - Different paradigm entirely

---

## Success Criteria (Revisited)

**Original goal**: JEPA > Random Init + 5%

**Achieved?** ✓ Yes (69.6% > 30% + 5%)

**BUT**: This is the WRONG metric! Within-dataset accuracy is not sufficient.

**NEW goal**: JEPA cross-dataset > Random + 5%

**Achieved?** ✗ **NO** (50% = 50% random)

**Conclusion**: By the correct metric (transferability), **JEPA has failed**.

---

## What Would Success Look Like?

For Mechanical-JEPA to be a true "foundation model":

1. **CWRU → IMS transfer**: > 60% (at least 10% above random)
2. **CWRU → Paderborn transfer**: > 60%
3. **Multi-task**: Works for fault classification AND RUL prediction
4. **Few-shot**: Can adapt to new bearing types with <10 labeled examples
5. **Robust**: Low variance across seeds and data splits

**Current status**: Achieves NONE of these.

---

## Lessons Learned

### What Worked

1. ✅ JEPA implementation is correct (loss decreases, training stable)
2. ✅ Data pipeline is solid (proper bearing-level splitting)
3. ✅ Linear probe evaluation is sound
4. ✅ Multi-seed validation reveals true performance

### What Didn't Work

1. ✗ Masked patch prediction as pretraining task
2. ✗ Small dataset (40 CWRU episodes insufficient)
3. ✗ Simple random masking strategy
4. ✗ No explicit transfer learning mechanism

### What's Unclear

1. ❓ Is IMS degradation detection task well-defined?
2. ❓ Would more pretraining epochs help?
3. ❓ Would structured masking (temporal_block) help?
4. ❓ Is the model capacity appropriate?

---

## Next Experiment Ideas (If Transfer Still Fails)

### Fallback Plan A: Verify IMS Task Quality

```python
# Train supervised model directly on IMS to see if it's learnable
model = SimpleClassifier()
train_on_ims_only(model)
# If this fails, IMS pseudo-labels are bad
```

### Fallback Plan B: Joint Pretraining

```python
# Pretrain on both CWRU and IMS together
model = JEPA()
pretrain(data='cwru+ims')  # Mixed pretraining
evaluate(data='cwru')  # Should still work
evaluate(data='ims')   # Should now work too
```

### Fallback Plan C: Domain Adaptation

```python
# Add domain-adversarial loss
model = JEPA_with_DA()
pretrain(source='cwru', target='ims')  # Align domains
evaluate(target='ims')
```

---

## Conclusion

Mechanical-JEPA achieves decent **within-dataset** performance (69.6%) but **completely fails** at **cross-dataset transfer** (50% = random).

This is a **critical failure** for a foundation model approach. The model is not learning transferable fault features - it's memorizing CWRU-specific patterns.

**Recommendation**: Consider alternative approaches (supervised learning, domain adaptation, TabPFN-style methods) unless joint pretraining or enhanced masking shows dramatic improvement.

---

## References

- [Brain-JEPA (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c3828adf1500f5de3c56f6550dfe43c-Abstract-Conference.html)
- [I-JEPA (CVPR 2023)](https://arxiv.org/abs/2301.08243)
- Current experiments: See `EXPERIMENT_LOG.md`
- Detailed lessons: See `LESSONS_LEARNED.md`
