# MTS-JEPA Replication Results

**Configuration**: d_model=128, K=64, 3 encoder layers, batch_size=32, A10G GPU
**Paper**: d_model=256, K=128, 6 encoder layers, batch_size=128, RTX 4090

## Replication Summary

| Dataset | Metric | Ours | Paper | Gap (%) | Status |
|---------|--------|------|-------|---------|--------|
| MSL | F1 | 20.57 ± 5.81 | 33.58 | 38.7% | FAILED |
| MSL | AUC | 52.59 ± 4.36 | 66.08 | 20.4% | MARGINAL |
| PSM | F1 | 50.68 ± 3.64 | 61.61 | 17.7% | MARGINAL |
| PSM | AUC | 48.96 ± 3.69 | 77.85 | 37.1% | FAILED |
| SMAP | F1 | 24.10 ± 2.22 | 33.64 | 28.3% | FAILED |
| SMAP | AUC | 56.26 ± 1.11 | 65.41 | 14.0% | GOOD |

## Per-Dataset Details


### MSL

Seeds: 5

| Seed | F1 | AUC | Precision | Recall | Best Epoch | Codebook Util |
|------|-----|-----|-----------|--------|------------|---------------|
| 1024 | 26.67 | 52.65 | 33.33 | 22.22 | 7 | 1.000 |
| 123 | 28.57 | 57.59 | 21.05 | 44.44 | 6 | 1.000 |
| 42 | 15.38 | 57.26 | 14.29 | 16.67 | 7 | 1.000 |
| 456 | 16.87 | 47.09 | 10.77 | 38.89 | 7 | 1.000 |
| 789 | 15.38 | 48.33 | 25.00 | 11.11 | 7 | 1.000 |

### PSM

Seeds: 5

| Seed | F1 | AUC | Precision | Recall | Best Epoch | Codebook Util |
|------|-----|-----|-----------|--------|------------|---------------|
| 1024 | 52.32 | 45.73 | 35.43 | 100.00 | 2 | 1.000 |
| 123 | 52.54 | 53.71 | 35.63 | 100.00 | 2 | 1.000 |
| 42 | 52.10 | 52.11 | 35.23 | 100.00 | 1 | 1.000 |
| 456 | 52.99 | 43.96 | 36.05 | 100.00 | 1 | 1.000 |
| 789 | 43.43 | 49.29 | 33.63 | 61.29 | 2 | 1.000 |

### SMAP

Seeds: 5

| Seed | F1 | AUC | Precision | Recall | Best Epoch | Codebook Util |
|------|-----|-----|-----------|--------|------------|---------------|
| 1024 | 26.17 | 56.85 | 16.24 | 67.31 | 2 | 1.000 |
| 123 | 25.28 | 57.10 | 15.73 | 64.42 | 2 | 1.000 |
| 42 | 26.07 | 56.95 | 17.01 | 55.77 | 2 | 1.000 |
| 456 | 20.62 | 54.11 | 12.01 | 73.08 | 2 | 1.000 |
| 789 | 22.38 | 56.29 | 13.06 | 77.88 | 2 | 1.000 |

---

## Diagnosis: Why the Gap?

### Primary Issue: KL Divergence Instability

The total training loss increases after 1-7 epochs due to the KL divergence term in the prediction loss. As the EMA target encoder evolves, the prediction task becomes harder, causing KL to grow. This forces early stopping at very early epochs (1-7), before the encoder has learned meaningful representations.

The paper uses batch_size=128 which likely stabilizes KL gradients through better averaging. Our batch_size=32 requires kl_scale=0.1 (vs paper's 1.0) to prevent complete divergence, which weakens the prediction signal.

### Secondary Issue: Codebook Uniformity

All experiments show 100% codebook utilization with perplexity ≈ K (64). This means the codebook assigns codes nearly uniformly — it's not learning discriminative patterns. The paper's larger codebook (K=128) with their entropy weights (0.005/0.01) may achieve a better balance between diversity and discrimination.

### Model Capacity

The medium model (d=128, 3 layers) has 1.47M params vs the paper's estimated ~5-8M params. The reduced capacity limits representational power, especially for the channel-independent encoder which must independently process V=25-33 channels.

---

## Lead-Time Analysis (Critical Finding)

**Most "anomaly predictions" are continuation detections, not genuine early warnings.**

| Dataset | TRUE_PREDICTION | CONTINUATION | Total Anomalous |
|---------|:-:|:-:|:-:|
| PSM | 15.5% (45/291) | **84.5%** (246/291) | 291 |
| MSL | 30.4% (35/115) | **69.6%** (80/115) | 115 |
| SMAP | 10.9% (67/615) | **89.1%** (548/615) | 615 |

This analysis is model-independent — it's a property of the evaluation protocol. With non-overlapping T_w=100 windows, anomalies spanning multiple windows create continuation cases. Only 11-30% of anomalous target windows have fully-normal context windows.

**Implications**:
1. MTS-JEPA's reported F1/AUC conflates prediction with detection
2. Fair evaluation must separate TRUE_PREDICTION from CONTINUATION
3. Causal masking (our CC-JEPA) would be properly evaluated on this breakdown

---

## Deviations from Paper

| Aspect | Paper | Ours | Expected Impact |
|--------|-------|------|----------------|
| d_model | 256 | 128 | -20% capacity |
| Encoder layers | 6 | 3 | -50% depth |
| Codebook K | 128 | 64 | Fewer codes |
| Batch size | 128 | 32 | KL instability |
| KL scale | 1.0 | 0.1 | Weaker prediction |
| Entropy weights | 0.005/0.01 | 0.005/0.01 | Same (paper values) |
| GPU | RTX 4090 | A10G | Similar VRAM |

---

## Path to Full Replication

To close the gap:
1. **Gradient accumulation** to simulate batch_size=128 (4 accumulation steps × batch 32)
2. **Full-size model** (d=256, K=128, 6 layers) — feasible with gradient accumulation
3. **KL warmup** — gradually increase kl_scale from 0.01 to 1.0 over first 20 epochs
4. **Separate validation metric** — use reconstruction loss for early stopping instead of total loss
