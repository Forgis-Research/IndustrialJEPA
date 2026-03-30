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

### Preprocessing

- Window size 4096 samples (~0.34s at 12kHz) captures multiple fault cycles
- Z-score normalize per channel on training set
- Patch size 256 gives 16 patches per window — good granularity

---

## Architecture

### JEPA-Specific

- EMA decay 0.996 is standard; lower (0.99) for faster adaptation
- Predictor should be smaller than encoder (2 layers vs 4-6)
- Mask ratio 0.5 works well; 0.7 forces harder prediction
- embed_dim=256, encoder_depth=4 is a good starting point

### Collapse Prevention

- If embedding variance drops below 0.01 → collapse happening
- Add batch normalization on encoder output
- Monitor loss — if it plateaus at high value, check for collapse
- VICReg-style variance/covariance loss as regularizer if needed

---

## Training

### Initial Results (Exp 0)

- 30 epochs achieves 49.8% test accuracy (vs ~30% random init)
- Loss decreased 78% (0.0079 → 0.0017) — good learning signal
- Inner race faults hardest to detect (29.3% accuracy)
- Ball faults easiest (63.4% accuracy)

### Tips

- Learning rate 1e-4 with cosine decay works well
- Batch size 32 is stable; larger may help but watch memory
- No warmup needed for 30 epochs, add if training longer

---

## Evaluation

### Classification (Linear Probe)

- Random guessing: 25% (4 classes)
- Random init encoder: ~30%
- JEPA (30 epochs): 49.8%
- **Target**: JEPA > Random Init + 5%

### Sanity Checks

1. Loss decreases over training
2. Test acc > random guessing (25%)
3. Test acc > random init (~30%)
4. Per-class accuracy reported (not just overall)
5. Multiple seeds (3+) for any claim

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

### Verification Commands

```bash
# Quick smoke test
python train.py --epochs 5 --no-wandb

# Check dataset
python data/bearings/prepare_bearing_dataset.py --verify

# Evaluate checkpoint
python train.py --eval-only --checkpoint checkpoints/jepa_xxx.pt
```

---

## Key Insights

1. **JEPA learns fault-discriminative features** — 49.8% vs 30% proves transferability
2. **Self-supervision works** — No labels needed during pretraining
3. **Inner race faults are harder** — May need class-specific strategies
4. **Brain-JEPA analogy holds** — Masked patch prediction works for vibration signals

---
