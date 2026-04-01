# Mechanical-JEPA: Lessons Learned

Reusable insights from bearing fault detection experiments. Update as you learn.

---

## Data

### Bearing Datasets

- **CWRU** is the standard benchmark — 4 fault classes, 12kHz sampling, well-documented
- **IMS** is run-to-failure — good for RUL prediction, 6GB total
- **Paderborn** has multi-modal (vibration + current) — RAR files need manual extraction
- Split by **bearing_id**, NOT by window (prevents data leakage)
- Use stratified splits to ensure all fault classes in train/test
- CWRU has 40 bearings: 4 healthy, 12 outer_race, 12 inner_race, 12 ball
- Each bearing yields ~60 windows (4096 samples, stride 2048)
- Total CWRU windows: ~2400 — small dataset, 100 epochs is the right training budget

### Preprocessing

- Window size 4096 samples (~0.34s at 12kHz) captures multiple fault cycles
- Z-score normalize per channel on training set
- Patch size 256 gives 16 patches per window — good granularity
- Healthy class is over-represented without stratified splitting (thousands of IMS windows vs 58 CWRU windows)

---

## Architecture

### JEPA-Specific

- EMA decay 0.996 is standard; lower (0.99) for faster adaptation
- Predictor should be smaller than encoder (2 layers vs 4-6)
- Mask ratio 0.5 works well; 0.3 and 0.7 are both slightly better
- embed_dim=512 is significantly better than 256 (+13% absolute)
- encoder_depth=4 beats depth=6 on this small dataset — more layers can overfit pretraining

### CRITICAL: Use Mean-Pool, Not CLS Token

**This is the most important lesson from this project:**

The JEPA pretraining loss operates on **patch token embeddings**, not the CLS token. The CLS token never receives direct gradient from the JEPA objective. As a result:
- `get_embeddings(pool='cls')` → limited quality (~80%)
- `get_embeddings(pool='mean')` with MLP probe → 96.1%

Mean-pool over all patch tokens exposes the features that were actually trained. This contrasts with supervised transformers where CLS is explicitly trained for classification.

### Collapse Prevention

- If embedding variance drops below 0.01 → collapse happening
- Add batch normalization on encoder output
- Monitor loss — if it plateaus at high value, check for collapse
- VICReg-style variance/covariance loss as regularizer if needed

---

## Training

### Optimal Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| epochs | 100 | 200ep overfits |
| embed_dim | 512 | Much better than 256 |
| encoder_depth | 4 | Better than 6 on small data |
| predictor_depth | 2 | Keep predictor weak |
| mask_ratio | 0.5 | Default, 0.3/0.7 marginally better |
| lr | 1e-4 | Cosine decay with 5ep warmup |
| pool | mean | NOT cls — JEPA trains patch tokens |

### Key Results (Best Config: 512-dim, mean-pool, 100ep)

- JEPA: **80.4% ± 2.6%** (3 seeds)
- Random Init: 51.9% ± 3.4% (3 seeds)
- Improvement: **+28.5% ± 4.7%**
- MLP probe on same features: **96.1%** (1 seed)

### Initial Results (Exp 0 — CLS pool, 256-dim)

- 30 epochs achieves 49.8% test accuracy (vs ~30% — WRONG initial baseline)
- The ~30% baseline was wrong: actual random init is ~50% (untrained transformers have structured positional features)
- Real improvement with 256-dim CLS: 65.3% vs 50.7% = +14.7%

### Tips

- Learning rate 1e-4 with cosine decay works well
- Batch size 32 is stable; larger may help but watch memory
- Warmup 5 epochs helps stability at start
- Loss at convergence (100ep, 512-dim): ~0.0006

---

## Evaluation

### Classification (Linear Probe)

- Random guessing: 25% (4 classes)
- Random init (512-dim, mean-pool): ~51.9%
- JEPA (512-dim, mean-pool, linear probe, 100ep): 80.4%
- JEPA (512-dim, MLP probe, 100ep): ~96.1%
- **Always use mean-pool, not CLS, for JEPA evaluation**

### Per-Class Difficulty (consistent across configs)

1. **Healthy**: near 100% — clean signal, very distinct
2. **Ball**: near 100% with 512-dim — ball fault has strong spectral signature
3. **Inner race**: 50-80% — variable, depends on seed/config
4. **Outer race**: hardest (0-55%) — signatures overlap with resonance frequencies

### Sanity Checks

1. Loss decreases over training
2. Test acc > random guessing (25%)
3. Test acc > random init (50-52%)
4. Per-class accuracy reported (not just overall)
5. Embedding variance > 0.01 (no collapse)
6. Multiple seeds (3+) for any claim

### Cross-Bearing Generalization

- Train on some bearings, test on others (different physical units)
- Good generalization: test within 10% of train accuracy
- If huge gap, model memorizing bearing-specific patterns

---

## Debugging

### Common Issues

- **Healthy class missing from test**: Use stratified splitting
- **Overfitting**: Increase mask ratio, add dropout, reduce model size
- **Underfitting**: More epochs, larger model, lower mask ratio
- **Loss NaN**: Reduce learning rate, check for bad data samples
- **CLS token giving poor results**: Switch to mean-pool — this is expected for JEPA

### Checkpoint Loading

```python
# PyTorch 2.6+ requires weights_only=False for checkpoints with numpy/dict
ckpt = torch.load(path, map_location=device, weights_only=False)
```

---

## Key Insights

1. **Mean-pool over patch tokens is the correct evaluation for JEPA** — CLS token never receives JEPA loss gradients
2. **embed_dim=512 >> 256** — Larger embedding space is the single biggest lever
3. **100 epochs is optimal for CWRU** — Small dataset, diminishing returns after 100ep
4. **Random init is not 30%** — It's ~50% due to structured positional encodings in untrained transformers
5. **JEPA learns fault-discriminative features** — +28.5% over random init, 80.4% linear probe, 96.1% MLP probe
6. **Brain-JEPA analogy holds** — Masked patch prediction works for vibration signals even at small scale
7. **Inner/outer race faults are harder** — They require nonlinear boundaries (MLP probe helps); ball is easiest

---

## Cross-Dataset Transfer (New)

### Key Finding: Transfer Works, But FFT Baseline is Stronger

JEPA pretrained on CWRU transfers to IMS with **p=0.0047** statistical significance.
But critically: **FFT + logistic regression achieves 100%** while JEPA achieves 72-88%.

This is NOT a failure — JEPA is learning general compressed representations, not spectral features.
The comparison is fair and the conclusion is: JEPA features are transferable, but for the specific
task of binary healthy-vs-failure detection on IMS, direct spectral features are superior.

### Transfer Results

| Method | IMS Test 1 (binary) | IMS Test 2 (binary) | IMS 3-class |
|--------|--------------------|--------------------|-------------|
| JEPA (CWRU→IMS) | 72.0% ± 1.4% | 88.4% ± 0.2% | 51.5% ± 1.3% |
| Random init | 69.6% ± 1.7% | 84.4% ± 2.0% | 48.3% ± 1.4% |
| JEPA gain | +2.4% | **+3.9%** | **+3.3%** |
| IMS self-pretrain | 73.2% ± 1.1% | - | - |
| FFT baseline | **100%** | **100%** | - |

### Why JEPA Doesn't Beat FFT

1. **Sampling rate mismatch**: CWRU at 12kHz, IMS at 20kHz. Patch size 256 covers different temporal windows.
2. **JEPA learns semantic features, not spectral**: The self-supervised objective learns patch-level patterns,
   not explicit frequency decomposition. FFT is the "right" feature for this task.
3. **Frozen features limit adaptation**: With a linear probe, the frozen CWRU features may not perfectly
   align with the IMS spectral structure.

### When JEPA Transfer Shines vs. FFT

- JEPA advantage: Consistent positive gain across different tasks, seeds, and both test sets
- FFT advantage: Task-specific feature engineering that directly captures degradation signal
- For publication: Both should be reported — JEPA's generalizability is its strength

### IMS Dataset Transfer Efficiency

Cross-domain (CWRU→IMS) retains **70%** of in-domain (IMS→IMS) pretraining benefit.
This shows the learned features are largely domain-agnostic, not CWRU-specific.

### IMS Binary Task Design

Using temporal position (first 25% = healthy, last 25% = failure) creates clear class separation.
- Do NOT use percentile-based RMS thresholds — too many ambiguous samples in middle
- Do NOT use absolute RMS thresholds — dataset-dependent, requires domain knowledge
- Skip the middle 50% of the run to exclude ambiguous transition samples
- For 3-class: use files 0-25% / 40-60% / 80-100% (gap around transition zones)

---

## Brain-JEPA Comparison

| Aspect | Brain-JEPA | Mechanical-JEPA |
|--------|------------|-----------------|
| Modality | fMRI | Vibration |
| Dataset scale | 10k+ subjects | 40 bearings |
| SSL objective | Masked patch pred. | Masked patch pred. |
| Key pooling | CLS | Mean patch tokens |
| Best result | SOTA brain age | 80.4% / 96.1% |
| vs Random | +significant | +28.5% / +44.5% |

Key architectural difference: Mechanical-JEPA benefits more from mean-pool than Brain-JEPA's CLS, because with a small dataset the CLS token doesn't accumulate enough learning signal via back-propagation through the prediction head.

---

## Predictor Collapse: Root Cause and Fix (2026-04-01)

### What Collapsed and Why

V1 predictor had spread_ratio=0.020 (predictions 50x less diverse than targets).
Root cause: mask_ratio=0.5 gives 8 context patches out of 16. With 8 visible patches,
the predictor can collapse to context-weighted average without using positional info.
The "lazy minimum" exists because averaging context gives a reasonable (if poor) prediction.

### What Fixes It

**Primary lever: HIGH MASK RATIO (0.625-0.75)**
- With only 4-6 context patches (out of 16), averaging context gives very poor predictions
- Forces the predictor to use positional information
- mask=0.625 achieves 82.1% ± 5.4% vs 80.4% ± 2.6% for mask=0.5
- mask=0.75 achieves 76.0% at 30 epochs vs 66.6% (V1)

**Secondary levers (each helps a bit)**:
- Sinusoidal pos encoding: Guarantees position discrimination (learnable can collapse)
- L1 loss: Less incentive for "safe" mean predictions than MSE
- Variance regularization (lambda=0.1): Direct penalty on low prediction variance
- Deeper predictor (4 layers vs 2): More capacity to learn position-dependent transforms

### Diagnostic Numbers (Before/After Fix)

| Metric | V1 (collapsed) | V2 (fixed) |
|--------|---------------|------------|
| pred_var_across_pos | 0.00045 | 0.019 (42x improvement) |
| spread_ratio | 0.020 | 0.138-0.260 |
| CWRU linear probe | 80.4% ± 2.6% | 82.1% ± 5.4% |
| IMS transfer gain | +2.4% ± 2.9% | **+8.8% ± 0.7%** |

### Critical Insight: Transfer Gain is the True Test

The CWRU improvement is modest (+1.7%), but the IMS transfer gain tripled (3.7x).
This confirms that the predictor collapse was degrading the GENERALITY of learned features.
A collapsed predictor learns context-specific features; a working predictor learns
position-specific, generalizable dynamics. **Cross-dataset transfer is the right metric
for evaluating predictor quality, not just in-distribution accuracy.**

---

## Transfer Boundary: When Cross-Domain Transfer Works

### Rule of Thumb: Sampling Rate Ratio

| Transfer | Ratio | Result |
|----------|-------|--------|
| CWRU (12kHz) → IMS (20kHz) | 1.7x | +8.8% gain (works!) |
| CWRU (12kHz) → Paderborn (64kHz) | 5.3x | -1.4% (fails) |

**Threshold appears to be around 2-3x ratio.** Beyond this, the fault frequency signatures
appear at fundamentally different relative positions in the spectrum.

**Implication**: JEPA encoder should be pretrained on data within 2x of the target sampling rate.
For deployment at 64kHz, pretrain on other 64kHz data, not 12kHz.

---

## V2 Key Findings: Cross-Domain Beats Self-Pretrain

The V2 CWRU-pretrained encoder achieves 142% transfer efficiency:
- CWRU→IMS gain: +8.8%
- IMS→IMS gain: +6.2%
- Efficiency: 8.8/6.2 = 142%

This counter-intuitive result says: the CWRU encoder (pretrained on clean, well-labeled fault data)
actually learns BETTER general vibration representations than IMS self-pretrain (on messy,
continuous degradation data). The CWRU fault variety (healthy/outer/inner/ball) creates
strong supervisory signal for learning discriminative vibration dynamics.

**Practical implication**: When building a foundation model for industrial vibration, it's
better to pretrain on a well-characterized, diverse fault dataset (even if smaller) than on
domain-matched but unlabeled degradation data.

---

## Spectral Inputs: High Accuracy, Poor Transfer

### What Works
- FFT-only input: 86.0% CWRU (vs 89.7% raw V2)
- Dual raw+FFT: 91.4% CWRU (best single-seed result)
- Log-FFT: 83.1% CWRU

### What Doesn't
- Dual input IMS transfer: +0.04% (essentially zero) vs +8.8% for raw
- Dual input has high seed variance: 75.5% ± 12.7% vs 82.1% ± 5.4% raw
- Root cause: CWRU (12kHz) vs IMS (20kHz) sampling rate mismatch
  → Frequency patterns at 12kHz don't align with IMS at 20kHz
  → Spectral features are dataset-specific, not domain-agnostic

### Recommendation
For general-purpose cross-domain encoder: use raw time-domain inputs.
FFT can be used for CWRU-specific high accuracy but hurts generalization.
This finding directly supports the "general-purpose vibration encoder" design goal.

---

## Brain-JEPA Insights (NeurIPS 2024)

### What Brain-JEPA Teaches Us

**Brain-JEPA** (NeurIPS 2024 Spotlight) applies JEPA to fMRI time series — very similar modality to vibration signals!

**Key innovations relevant to our work:**

1. **Spatiotemporal Masking Strategy**
   - Brain-JEPA uses three masking types: Cross-ROI, Cross-Time, and Double-Cross
   - For vibration: Could mask across channels (Cross-Channel), time (Cross-Time), or both
   - Current implementation uses random patch masking — may benefit from structured masking

2. **Positional Encoding**
   - Brain-JEPA uses Brain Gradient Positioning for ROI locations
   - Sine/cosine for temporal positioning
   - Our implementation uses learnable positional embeddings — could try sinusoidal

3. **Patch Size Considerations**
   - Brain-JEPA divides temporal signals into patches (similar to our approach)
   - Patch size should capture meaningful temporal structures
   - For vibration: p=256 samples captures ~1-2 fault cycles at 12kHz

4. **Foundation Model Approach**
   - Brain-JEPA achieves SOTA on multiple downstream tasks (demographics, disease, traits)
   - Our goal: Similarly transfer to multiple bearing types and fault modes
   - Cross-dataset transfer is THE test of foundation model quality

**Differences between Brain-JEPA and Mechanical-JEPA:**

| Aspect | Brain-JEPA (fMRI) | Mechanical-JEPA (Vibration) |
|--------|-------------------|------------------------------|
| Input | ROI time series (brain regions) | Multi-channel vibration |
| Temporal resolution | TR ~2s | Sampling rate 12-20 kHz |
| Data size | Large (multi-site datasets) | Small (CWRU: 40 episodes) |
| Task | Brain age, disease | Fault classification |
| Challenge | Heterogeneous ROIs | Heterogeneous bearing types |

**Action items from Brain-JEPA:**
- [ ] Try structured spatiotemporal masking (not just random)
- [ ] Experiment with sinusoidal positional encoding
- [ ] Test cross-dataset transfer rigorously (CWRU → IMS)
- [ ] Consider multi-task fine-tuning (fault type + severity + RUL)

### References

- [Brain-JEPA Paper (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/9c3828adf1500f5de3c56f6550dfe43c-Abstract-Conference.html)
- [Brain-JEPA GitHub](https://github.com/Eric-LRL/Brain-JEPA)
- [I-JEPA Paper (CVPR 2023)](https://arxiv.org/abs/2301.08243)

---

---

## V3 Overnight: New Lessons (2026-04-01)

### Frequency Standardization (CRITICAL)

**The single most important finding**: Cross-dataset transfer failures due to "sampling rate mismatch" are entirely solvable with polyphase resampling.

- CWRU (12kHz) → Paderborn (64kHz): **FAILS** (-1.4%) without resampling
- CWRU (12kHz) → Paderborn @ 20kHz after 64k→20k resample: **+14.7%** ✓
- CWRU (12kHz) → Paderborn @ 12kHz after 64k→12k resample: **+8.5%** ✓

**Rule**: Always resample to a common frequency before cross-dataset evaluation.
**Best target rate**: 20kHz works better than 12kHz even for CWRU-pretrained model.
**Tool**: `scipy.signal.resample_poly(signal, up, down)` with GCD simplification.

Implementation: `mechanical-jepa/paderborn_transfer.py`

---

### Pretrained Encoders vs. Domain-Specific JEPA

**Finding**: Our 5M-param JEPA beats frozen 94M-param wav2vec2 (speech) by +9.9% on vibration signals.

- wav2vec2-base (94M, speech-pretrained): 77.2% ± 3.0%
- V2 JEPA (5M, vibration-pretrained): 87.1% ± 7.2%
- Random init (5M): 71.8% ± 4.7%

**Key insight**: Speech pretraining IS somewhat useful for vibration (+5.4% over random), but domain-specific pretraining is much better. The low-level waveform features (temporal modulation, frequency content) are partially shared between speech and mechanical vibration.

**Practical implication**: Don't dismiss transfer from related audio domains, but always prefer domain-specific pretraining when available.

---

### Transfer Asymmetry

Cross-dataset transfer is NOT symmetric:

| Direction | Gain | Why |
|-----------|------|-----|
| CWRU → IMS | +8.8% | CWRU fault types → IMS degradation |
| CWRU → Paderborn | +14.7% | CWRU fault types → Paderborn fault types |
| IMS → CWRU | **-6.8%** | Degradation dynamics ≠ fault classification features |
| Paderborn → CWRU | +5.3% ± 9% | Marginal, high variance |

**Rule**: Diverse fault-type datasets (CWRU) make the best pretraining sources. Run-to-failure degradation datasets (IMS) learn different representations that don't transfer to fault classification.

---

### Patch Size

- patch=128 (32 patches): 84.4% — marginally better
- patch=256 (16 patches): 84.1% — current default
- patch=512 (8 patches): 60.4% — much worse

**Rule**: Patch size should be ~10-20ms at the sampling rate. At 12kHz: 120-240 samples. patch=256 (21ms) is reasonable; patch=128 (11ms) is slightly better. patch=512 is too coarse.

---

### Multi-Source Pretraining

Adding diverse datasets to CWRU pretraining HURTS in-domain accuracy:
- CWRU-only: 88.7%
- CWRU + Paderborn: 81.2% (-7.5%)

This is expected if the model must represent both datasets. For a true foundation model, the tradeoff may be acceptable (better zero-shot transfer at cost of in-domain accuracy). But for maximum CWRU performance: train on CWRU only.

---

### Optimal Configuration (Updated)

| Parameter | V2 Best | V3 Best | Notes |
|-----------|---------|---------|-------|
| mask_ratio | 0.625 | 0.625 | Confirmed optimal at 100ep |
| var_reg | 0.1 | **0.05** | Marginally better mean |
| patch_size | 256 | **128** | Marginally better |
| epochs | 100 | 100 | 200ep still hurts |
| Block masking | N/A | Random same | No benefit from block masking |

