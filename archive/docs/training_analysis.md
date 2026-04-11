# Training Analysis

Analysis of training approaches and experimental results.

## The Core Problem

**Static setpoint→effort prediction fails for anomaly detection.**

| Effort Signal | R² (setpoint only) | Interpretation |
|---------------|-------------------|----------------|
| effort_torque_1 | 0.992 | Gravity—fully determined by position |
| effort_torque_2 | 0.977 | Gravity—fully determined by position |
| effort_force_z | 0.156 | Contact force—depends on load |

Contact forces are the key anomaly signal, but they're unpredictable from setpoint alone because the physical load (screw presence, material resistance) is unobserved.

### Why Static Prediction Fails

For `missing_screw` faults:
- Actual contact force: 3.94 (mean)
- Normal contact force: 8.09 (mean)
- Prediction (trained on normal): ~8.0
- **Error is lower for anomaly** → ROC-AUC = 0.391 (worse than random)

Anomalies with lower variance are missed by reconstruction error.

## Solution: Temporal Self-Prediction

Instead of predicting effort from setpoint, predict **future effort from past effort + setpoint**.

```
Input:  [setpoint(t-k:t), effort(t-k:t)]
Output: effort(t+1:t+n) in latent space
```

**Why it works**: Faults disrupt temporal dynamics even when absolute values are lower. Missing screw creates different force *profile*, not just lower values.

## Current Results (AURSAD)

| Metric | Value |
|--------|-------|
| Training loss | 0.192 → 0.060 (converged) |
| Validation loss | 0.097 → 0.035 |
| ROC-AUC | 0.538 |
| PR-AUC | 0.832 |
| Best threshold | 0.165 |

### Per-Fault Detection

| Fault Type | Count | Mean Score | Detection Rate |
|------------|-------|-----------|----------------|
| normal | 1526 | 0.102 | — |
| missing_screw | 1954 | 0.169 | 100% |
| damaged_screw | 2335 | 0.117 | 100% |
| extra_component | 1936 | 0.133 | 100% |
| damaged_thread | 27 | 0.179 | 100% |

All fault types show higher mean anomaly score than normal, enabling detection.

## Baseline Comparisons

| Method | Description | Expected Result |
|--------|-------------|-----------------|
| **TemporalPredictor** | Predict future effort embeddings | Best (current) |
| SetpointToEffort | Static raw-space prediction | Fails (R²=0.16) |
| EffortAutoencoder | Reconstruct effort only | Poor (no causal info) |
| MAE | Masked autoencoder | Moderate (wastes capacity) |
| Contrastive | Episode-level SimCLR | TBD |

## Training Configuration

```yaml
# Temporal predictor
window_size: 256
hidden_dim: 256
num_layers: 4
num_heads: 8
context_ratio: 0.5  # First half = context, second half = target

# Training
batch_size: 64
learning_rate: 1e-4
weight_decay: 1e-5
epochs: 50
ema_momentum: 0.996
```

## Dataset Filtering

AURSAD contains both tightening and loosening episodes. We filter to tightening only:

| Split | Episodes | Rows |
|-------|----------|------|
| Total | 4094 | 6.2M |
| Tightening only | 2045 | 3.1M |
| Healthy (train) | 1420 | 2.1M |
| Mixed (test) | 625 | 1.0M |

Episode-level splitting prevents train/test leakage.

## Next Steps

1. **Causal ablation**: Compare setpoint→effort vs effort-only vs wrong direction
2. **Cross-machine transfer**: Train on AURSAD, test on voraus-AD zero-shot
3. **Multi-dataset pretraining**: Combine all FactoryNet datasets
4. **Q&A heads**: Binary classifiers on embeddings for state queries

## References

- [Anomaly Detection Survey](https://arxiv.org/abs/2007.02500): Deep learning for anomaly detection
- [Time Series Anomaly Detection](https://arxiv.org/abs/2202.07105): Transformer-based approaches
