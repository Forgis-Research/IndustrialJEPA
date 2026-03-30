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
