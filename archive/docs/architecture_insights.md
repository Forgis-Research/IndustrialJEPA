# IndustrialJEPA: Key Architectural Insights

A summary of the core ideas that make IndustrialJEPA work.

## The Central Insight: Physics-Grounded World Models

Traditional anomaly detection asks: "Does this data look normal?"

IndustrialJEPA asks: "Does this robot behave according to physics?"

```
Commands (setpoint) ──► Robot Physics ──► Observed Forces (effort)
```

A healthy robot follows predictable dynamics. Faults break this relationship:
- Friction faults → unexpected torque
- Collision → unexpected forces
- Wear → degraded dynamics

## Why JEPA, Not Autoencoders?

### Problem with Reconstruction

Autoencoders reconstruct raw sensor values:
```
sensor_data ──► Encoder ──► z ──► Decoder ──► reconstructed_data
```

This wastes capacity on:
- Sensor noise (unpredictable, shouldn't be reconstructed)
- High-frequency vibrations (normal but noisy)
- Exact waveform details (irrelevant for fault detection)

### JEPA Solution: Predict in Latent Space

```
state(t) ──► Encoder ──► z(t) ──► Predictor ──► z_pred(t+1)
                                       ↓
state(t+1) ──► EMA Encoder ──► z_target(t+1)
                                       ↓
                        Loss = ||z_pred - z_target||²
```

Benefits:
1. **Ignores noise**: Encoder learns to discard unpredictable components
2. **Learns dynamics**: Predictor captures how states evolve
3. **Semantic representations**: Latent space captures meaningful physics

## The Setpoint→Effort Causal Structure

### Key Insight

Industrial robots have a natural causal structure:
- **Setpoint** (input): Commanded joint positions, velocities
- **Effort** (output): Resulting torques, forces

This is physics: `F = ma`, `τ = Iα`, `power = force × velocity`

### Why This Matters

1. **Grounded prediction**: We know setpoints *cause* efforts (not vice versa)
2. **Counterfactual reasoning**: "If we commanded X, we'd expect effort Y"
3. **Transfer learning**: Physics relationships transfer across robots

### Baseline Comparison

| Model | Architecture | Result |
|-------|-------------|--------|
| effort_ae | effort → effort | 0.001 loss (trivial) |
| s2e | setpoint → effort | 0.074 loss |
| temporal | z(t) → z(t+1) | 0.036 loss |
| **JEPA** | setpoint + z(t) → z(t+1) | **0.028 loss** |

JEPA beats blind temporal prediction by 22%, validating the physics-grounded structure.

## EMA Target Encoder: Stable Training

### The Collapse Problem

Without negatives, self-supervised models can collapse (output constant vectors).

### Solution: Exponential Moving Average

```python
θ_target = momentum * θ_target + (1 - momentum) * θ_online
```

The target encoder changes slowly, providing stable prediction targets.
From BYOL/I-JEPA research: momentum ~0.996 works well.

## Asymmetric Encoder-Predictor

### The Problem

If predictor = encoder, the model learns identity: `z_pred = z_context`

### Solution: Smaller Predictor

```
Encoder:   4-6 transformer layers, 256-512 dim
Predictor: 1-2 MLP layers, 128 dim
```

This asymmetry:
- Prevents trivial solutions
- Forces the encoder to learn rich representations
- The predictor does "easy" linear predictions in latent space

## One-Class Anomaly Detection

### Training: Healthy Data Only

```
Train on: Normal robot operation (no faults)
Learn:    What healthy dynamics look like
```

### Inference: Prediction Error as Anomaly Score

```python
anomaly_score = ||z_pred - z_actual||²
```

- Low score → prediction matches reality → healthy
- High score → prediction fails → anomaly

### Why This Works

Healthy dynamics are consistent and learnable.
Faults create *novel* dynamics the model hasn't seen.
Novelty = high prediction error = anomaly.

## Cross-Machine Transfer

### The Challenge

Different robots have:
- Different kinematics (joint configurations)
- Different actuators (motor constants)
- Different sensors (calibration offsets)

### The Solution: Normalize Per-Machine, Transfer Dynamics

**Non-transferable** (normalize away):
```
- Absolute joint positions → z-score normalize
- Sensor scales → per-channel normalization
- Motor constants → absorbed in normalization
```

**Transferable** (learned in latent space):
```
- Phase transitions: approach → contact → execute
- Contact dynamics: force buildup patterns
- Energy relationships: power = F × v
```

### Key Insight

Physics relationships are universal. A screw tightening on Robot A has similar dynamics to Robot B, even if absolute values differ.

## Unified Signal Schema

### Problem: Heterogeneous Datasets

Different datasets have different columns:
- AURSAD: joint torques only
- Voraus-AD: joint torques + Cartesian forces
- RH20T: different joint configuration

### Solution: Fixed-Dimension Schema with Masks

| Signal | Dimension | Contents |
|--------|-----------|----------|
| Setpoint | 14 | 7 positions + 7 velocities |
| Effort | 13 | 7 torques + 6 Cartesian |

Missing signals get `value=0, mask=0`. Model learns to ignore invalid dims.

```python
# Model sees:
input = torch.cat([setpoint, effort])  # Always same shape
mask = torch.cat([setpoint_mask, effort_mask])  # Valid indicators
```

## Experimental Validation

### Results (20 epochs on AURSAD)

```
Baseline (temporal):  val_loss = 0.036
JEPA (ours):          val_loss = 0.028  ← 22% better
```

### What This Proves

The physics-grounded structure (setpoint→effort) provides better inductive bias than blind temporal prediction.

## Summary

1. **Predict in latent space** (not raw sensors) → ignores noise
2. **Use physics structure** (setpoint→effort) → better representations
3. **EMA target encoder** → stable training without negatives
4. **Asymmetric predictor** → prevents collapse
5. **One-class training** → detects any deviation from normal
6. **Unified schema** → enables cross-dataset training

The key innovation: Combining JEPA's latent prediction with robotics' causal structure (commands cause forces) creates a physics-grounded world model for anomaly detection.
