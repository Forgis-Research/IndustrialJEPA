# World Model Architecture

This document describes the IndustrialJEPA world model architecture and design decisions.

## Overview

The world model learns to predict future robot states in latent space. Unlike reconstruction-based methods (autoencoders, MAE), JEPA predicts embeddings directly—avoiding wasted capacity on unpredictable sensor noise.

```
┌─────────────────────────────────────────────────────────────┐
│  effort(t-k:t) ──► StateEncoder ──► z(t)                   │
│                                       │                     │
│                                       ▼                     │
│                                   Predictor ──► z_pred(t+1) │
│                                       │                     │
│  effort(t+1) ───► EMA Encoder ──► z_target(t+1)            │
│                                       │                     │
│                       Loss = ||z_pred - z_target||²         │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Temporal Self-Prediction (Not Static Setpoint→Effort)

**Problem**: Static setpoint→effort prediction achieves R²=0.99 for gravity torques but only R²=0.16 for contact forces. Contact forces depend on unmeasured load (screw presence, material resistance).

**Solution**: Predict future effort from past effort + setpoint. Temporal context captures dynamics that distinguish faults even when absolute values are similar.

```python
# Static (fails for contact forces)
pred_effort = model(setpoint)  # R² = 0.16 for contact

# Temporal (captures dynamics)
pred_effort_t1 = model(effort[t-k:t], setpoint[t-k:t])  # Learns dynamics
```

### 2. EMA Target Encoder

The target encoder uses exponential moving average of the online encoder weights, following [BYOL](https://arxiv.org/abs/2006.07733) and [I-JEPA](https://arxiv.org/abs/2301.08243):

```python
target_encoder = ema_update(target_encoder, online_encoder, momentum=0.996)
```

This prevents representation collapse without negative samples.

### 3. Asymmetric Encoder-Predictor

The predictor is smaller than the encoder (1-2 layers vs 4-6). This asymmetry:
- Prevents the predictor from memorizing identity mappings
- Forces meaningful latent representations
- Follows I-JEPA architecture principles

### 4. Unified Signal Schema

All datasets map to fixed dimensions with validity masks:

| Signal Group | Dimension | Contents |
|-------------|-----------|----------|
| Setpoint | 14 | 7 joint positions + 7 velocities |
| Effort | 13 | 7 joint torques + 6 Cartesian forces |

Missing signals are zero-padded with mask=0. The model learns to ignore invalid dimensions.

## Signal Groups

### Setpoint (Command)
What the controller commanded:
- Joint positions: `setpoint_pos_0..6`
- Joint velocities: `setpoint_vel_0..6`

### Effort (Energy)
What the robot expended:
- Joint torques: `effort_torque_0..6`
- Cartesian forces: `effort_force_x, y, z`
- Cartesian torques: `effort_torque_x, y, z`

### Context (Optional)
Task phase information (dataset-specific):
- `ctx_state_tightening`: Screwing phase
- `ctx_state_loosening`: Unscrewing phase
- `ctx_state_process_ok/nok`: Success/failure

## Transfer Strategy

**Transferable** (learned in latent space):
- Phase transitions: approach → contact → execute
- Contact dynamics: force buildup patterns
- Energy relationships: power = force × velocity

**Non-transferable** (handled by normalization):
- Absolute joint positions (different kinematics)
- Motor constants (different actuators)
- Sensor scales (different calibrations)

The key insight: normalize per-machine, predict in latent space, condition on task phase.

## Anomaly Detection

Anomalies are detected by high prediction error in latent space:

```python
z_pred = predictor(encoder(effort_context))
z_target = ema_encoder(effort_future)
anomaly_score = ||z_pred - z_target||²
```

Threshold is set on validation data (healthy only) at desired false positive rate.

## Implementation

Core model: `src/industrialjepa/model/world_model.py`

```python
class JEPAWorldModel(nn.Module):
    def __init__(self, config):
        self.encoder = StateEncoder(config)
        self.predictor = StatePredictor(config)  # Smaller than encoder
        self.target_encoder = copy.deepcopy(self.encoder)  # EMA updated

    def forward(self, effort_context, effort_target):
        z_context = self.encoder(effort_context)
        z_pred = self.predictor(z_context)

        with torch.no_grad():
            z_target = self.target_encoder(effort_target)

        return F.mse_loss(z_pred, z_target)
```

## References

- [I-JEPA](https://arxiv.org/abs/2301.08243): Joint-Embedding Predictive Architecture
- [V-JEPA](https://arxiv.org/abs/2404.08471): Video extension with temporal prediction
- [BYOL](https://arxiv.org/abs/2006.07733): EMA target networks without negatives
- [World Models](https://arxiv.org/abs/1803.10122): Latent dynamics for planning
