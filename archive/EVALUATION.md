# IndustrialJEPA Evaluation Plan

## Current Status

**First Training Run (2026-03-11):**
- Dataset: AURSAD (6.2M rows, 4094 episodes)
- Model: 14.2M parameters (4-layer Transformer encoder, 2-layer MLP predictor)
- Results: val_loss = 0.037 after 10 epochs (still decreasing)

## Evaluation Approaches

### 1. Prediction Error (Intrinsic)
What we measure now - MSE in latent space.
- val_loss = 0.037
- Hard to interpret in isolation, need baselines for comparison

### 2. Anomaly Detection (Main Goal)
Use prediction error as anomaly score:
- Normal operation → low prediction error (model learned the physics)
- Fault/anomaly → high prediction error (unexpected dynamics)

**AURSAD Fault Labels:**
- `normal` - 3,469 episodes (85%)
- `Damaged_screw` - 221 episodes
- `Missing_screw` - 218 episodes
- `Extra_part` - 183 episodes
- `Damaged_thread` - 3 episodes

**Metrics:**
- AUC-ROC (area under ROC curve)
- F1-score at optimal threshold
- Precision/Recall curves
- Detection latency (how early in episode is fault detected?)

### 3. Multi-step Rollout
Predict multiple steps ahead to test model stability:
- Encode obs(t) → z(t)
- Predict z(t+1), z(t+2), ..., z(t+k) autoregressively
- Decode back to observation space
- Measure error accumulation

Good world models should:
- Stay stable (no divergence)
- Accumulate error slowly
- Maintain physical plausibility

### 4. Latent Space Quality
Visualize learned representations:
- t-SNE/UMAP of latent embeddings
- Do fault episodes cluster separately?
- Do different fault types form distinct clusters?

### 5. Cross-Machine Transfer
Train on one robot, test on another:
- Train: AURSAD (UR3e, 6-DOF, current signals)
- Test: Voraus (Yu-Cobot, 6-DOF, current signals)

Both are 6-DOF robots with similar physics. If JEPA learns transferable dynamics, it should generalize.

---

## SOTA Baselines

### For Industrial Anomaly Detection

| Model | Type | Key Idea | Expected Performance |
|-------|------|----------|---------------------|
| **Autoencoder** | Reconstruction | High recon error = anomaly | F1 ~0.80 |
| **LSTM Predictor** | Sequence model | Direct next-step prediction | F1 ~0.85 |
| **USAD** | Adversarial AE | Two-phase adversarial training | F1 ~0.88 |
| **Deep SVDD** | One-class | Hypersphere around normal data | F1 ~0.85 |
| **OC-SVM** | Classical ML | One-class SVM on features | F1 ~0.75 |
| **Isolation Forest** | Tree-based | Isolation of anomalies | F1 ~0.78 |

**Published AURSAD results:** F1 ~0.85-0.95 for fault detection (varies by method and fault type)

### For World Models / Dynamics Learning

| Model | Architecture | Prediction Space |
|-------|--------------|------------------|
| **MLP Predictor** | Simple feedforward | Observation space |
| **LSTM Predictor** | Recurrent | Observation space |
| **Transformer Predictor** | Attention | Observation space |
| **VAE + Predictor** | Variational | Latent space |
| **JEPA (ours)** | Transformer + EMA | Latent space |

JEPA advantage: Predicts in latent space with EMA target encoder, avoiding reconstruction and collapse.

---

## Implementation Plan

### Phase 1: Anomaly Detection Evaluation
```python
# scripts/evaluate_anomaly_detection.py

# 1. Load trained model
model = load_checkpoint("results/world_model/best_model.pt")

# 2. Compute prediction error per window
def compute_anomaly_scores(model, dataloader):
    scores = []
    labels = []
    for batch in dataloader:
        obs_t, cmd_t, obs_t1 = batch['obs_t'], batch['cmd_t'], batch['obs_t1']

        # Get prediction error in latent space
        z_t = model.encode(obs_t)
        z_pred = model.predict(z_t, cmd_t)
        z_target = model.encode_target(obs_t1)

        error = (z_pred - z_target).pow(2).mean(dim=-1)  # per-sample error
        scores.extend(error.cpu().numpy())
        labels.extend([m['is_anomaly'] for m in batch['metadata']])

    return np.array(scores), np.array(labels)

# 3. Compute metrics
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

scores, labels = compute_anomaly_scores(model, test_loader)
auc = roc_auc_score(labels, scores)
print(f"AUC-ROC: {auc:.4f}")

# Find optimal threshold
precision, recall, thresholds = precision_recall_curve(labels, scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_f1 = f1_scores.max()
print(f"Best F1: {best_f1:.4f}")
```

### Phase 2: Implement Baselines
```bash
# Already have baseline implementations in:
# src/industrialjepa/baselines/

# Run baseline training:
python scripts/train_baseline.py --model autoencoder --epochs 10
python scripts/train_baseline.py --model lstm_predictor --epochs 10
```

### Phase 3: Comparison Table
Generate comparison table across models:

| Model | AUC-ROC | F1 | Precision | Recall | Latency |
|-------|---------|----|-----------| -------|---------|
| JEPA (ours) | ? | ? | ? | ? | ? |
| Autoencoder | ? | ? | ? | ? | ? |
| LSTM Predictor | ? | ? | ? | ? | ? |

### Phase 4: Cross-Machine Transfer
```python
# Train on AURSAD
python scripts/train_world_model.py --data-source aursad --epochs 50

# Evaluate on Voraus (zero-shot transfer)
python scripts/evaluate_transfer.py --checkpoint aursad_model.pt --test-data voraus
```

---

## Success Criteria

1. **Anomaly Detection:** AUC-ROC > 0.90, F1 > 0.85
2. **Beat Baselines:** JEPA outperforms autoencoder and LSTM predictor
3. **Transfer Learning:** >50% performance retained when transferring to new robot
4. **Stable Rollouts:** <2x error increase over 50-step rollout

---

## Related Work (For NeurIPS Positioning)

### JEPA for Time Series

| Paper | Key Contribution | Our Differentiation |
|-------|-----------------|---------------------|
| [MTS-JEPA](https://arxiv.org/abs/2602.04643) | Multi-resolution JEPA with soft codebook for anomaly prediction | We focus on cross-machine transfer via causal structure, not multi-resolution |
| [TS-JEPA](https://arxiv.org/abs/2406.04853) | JEPA for predictive remote control under limited networks | We target fault detection, not communication efficiency |
| [V-JEPA](https://arxiv.org/abs/2310.03191) | Video JEPA for visual representation learning | We adapt JEPA principles to industrial time series with causal structure |

### Industrial Anomaly Detection

| Paper | Approach | Limitation We Address |
|-------|----------|----------------------|
| USAD | Adversarial autoencoder | No causal structure, doesn't transfer |
| Deep SVDD | One-class classification | Learns hardware statistics, not physics |
| CASPER | Context-aware IoT anomaly detection | Single machine, no transfer |

### Key Insight for Originality

**Existing gap:** All prior work treats industrial time series as purely observational data.
They learn p(sensor_t+1 | sensor_t) - what sensors do next given what they did before.

**Our contribution:** We exploit the causal structure of control systems:
- Setpoint (command) → Effort (response) → Feedback (outcome)
- We learn p(effort | setpoint) - the physics of how commands cause effects
- This physics-based relationship transfers across machines of the same type

### Theoretical Grounding

The Setpoint→Effort relationship is governed by physics:
```
τ = J·α + b·ω + τ_gravity + τ_friction + τ_payload
```
Where:
- τ = joint torque (effort)
- J = inertia matrix (machine-specific but learnable)
- α = commanded acceleration (from setpoint)
- b = damping (friction)
- τ_gravity = gravity compensation
- τ_payload = external load

**Key observation:** The functional form is the same across all 6-DOF robots.
Only the parameters (J, b, etc.) differ. A model that learns this functional relationship
should transfer - it just needs to adapt the parameters.

JEPA in latent space naturally learns this functional relationship without
explicitly modeling the physics equations.

---

## Next Steps

- [ ] Implement `scripts/evaluate_anomaly_detection.py`
- [ ] Run evaluation on trained model
- [ ] Implement baseline models (autoencoder, LSTM)
- [ ] Train baselines on same data
- [ ] Generate comparison table
- [ ] Visualize latent space (t-SNE)
- [ ] Test cross-machine transfer
