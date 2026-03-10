# IndustrialJEPA Training Analysis

## Executive Summary

The original JEPA training objective (setpoint → effort prediction) has limited effectiveness because **setpoint alone cannot predict contact forces** - the key signals for anomaly detection. This document analyzes the problem and proposes improved training approaches.

---

## 1. Schema Validation (Completed)

The expanded effort schema is working correctly:

| Dataset | Effort Columns | Key Signals |
|---------|---------------|-------------|
| AURSAD | 12 (6 joint + 6 Cartesian) | effort_force_z (screwdriving) |
| Voraus | 6 (joint torques only) | effort_torque_* |
| RH20T | 7 (velocity-based) | effort_vel_* |

**Change made**: `unified_effort_dim` increased from 7 to 13 to include Cartesian forces.

---

## 2. Setpoint → Effort Predictability Analysis

### Linear Regression Results (100k samples from AURSAD)

| Effort Signal | R² Score | Interpretation |
|--------------|----------|----------------|
| effort_torque_1 | **0.992** | Gravity compensation - fully determined by position |
| effort_torque_2 | **0.977** | Gravity compensation - fully determined by position |
| effort_torque_0 | 0.074 | Base rotation - depends on dynamics/friction |
| effort_force_z | **0.156** | Contact force - depends on load/context |
| effort_force_x | 0.040 | Contact force - context dependent |
| effort_force_y | 0.083 | Contact force - context dependent |

### Key Insight

**Gravity torques are predictable** (joints 1,2 hold up the arm weight based on position), but **contact forces are NOT predictable** from setpoint alone. The missing information is the **physical load** (screw presence, tightness, material resistance).

---

## 3. Context Signal Analysis

### Available Context Signals

| Signal | Type | Helps Prediction? |
|--------|------|-------------------|
| `ctx_temp_*` | Joint temperatures | **No** (+0.001 R²) |
| `fault_label` | Metadata | Slight (+0.005 R²) |
| Load/contact info | **NOT AVAILABLE** | Would be critical |

### Fault Label Distribution (AURSAD tightening)

| Fault Label | Count | Mean effort_force_z | Std |
|-------------|-------|---------------------|-----|
| normal | 2.2M | 8.09 | 11.89 |
| loosening | 3.0M | 8.34 | 12.07 |
| **missing_screw** | 286k | **3.94** | 8.71 |
| damaged_screw | 336k | 8.02 | 12.01 |
| extra_component | 275k | 7.53 | 12.33 |

**Critical finding**: `missing_screw` has **2x lower force** than normal operations.

---

## 4. Anomaly Detection with Prediction Error

### Experiment: Train on Normal, Detect by Prediction Error

```
Training: Ridge regression on normal samples only
Test: Mix of normal + anomalous samples
Metric: |predicted - actual| effort_force_z
```

### Results

| Metric | Normal | Anomaly |
|--------|--------|---------|
| Mean MAE | 11.296 | 10.502 |
| Std MAE | 4.228 | 4.444 |
| **ROC-AUC** | - | **0.391** (worse than random!) |

### Why It Fails

The model predicts "average" effort. For `missing_screw`:
- Actual effort_force_z is **lower** (3.94 vs 8.09)
- Prediction is around the mean (~8)
- Error is actually **lower** for the anomaly!

**Prediction error alone cannot detect anomalies when anomalies have lower variance.**

---

## 5. Q&A Features Assessment

### Current Dataset Structure

The FactoryNet dataset has **no explicit Q&A text fields**. The "Robot Q&A" use case mentioned in documentation is aspirational.

**Available for conditioning:**
- Metadata: `task_type`, `machine_model`, `fault_label`
- Context sensors: `ctx_temp_*` (temperatures)
- Auxiliary sensors: `aux_vibration_*`, `aux_acoustic_*` (some datasets)

**NOT available:**
- Natural language task descriptions at timestep level
- Load/contact force context
- Material properties

---

## 6. Proposed Training Approaches

### Option A: Temporal Self-Prediction (Recommended)

**Idea**: Instead of setpoint → effort, predict future effort from past effort+setpoint.

```
Input:  [setpoint(t-k:t), effort(t-k:t)]
Output: effort(t+1:t+n)
```

**Why it might work**:
- Anomalies disrupt temporal patterns even if absolute values are lower
- Normal operations have predictable dynamics
- Missing screw causes different force *profile*, not just lower values

**Implementation**:
```python
class TemporalJEPA:
    """Predict future effort embeddings from past context."""

    def forward(self, setpoint, effort):
        # Split into context and target
        context_setpoint = setpoint[:, :T//2]
        context_effort = effort[:, :T//2]
        target_effort = effort[:, T//2:]

        # Encode context (both modalities)
        ctx = self.encoder(concat(context_setpoint, context_effort))

        # Predict future effort embeddings
        pred = self.predictor(ctx)
        target = self.ema_encoder(target_effort)

        return jepa_loss(pred, target)
```

---

### Option B: Residual Modeling

**Idea**: Decompose effort into predictable + unpredictable components.

```
effort = gravity_compensation(setpoint) + contact_residual + noise
```

**Step 1**: Train gravity predictor on joints 1,2 (R² > 0.97)
**Step 2**: Model residuals (what's left after gravity)
**Step 3**: Anomalies show up as unusual residual patterns

**Implementation**:
```python
class ResidualJEPA:
    def forward(self, setpoint, effort):
        # Predict gravity compensation (deterministic)
        gravity = self.gravity_head(setpoint)  # High R²

        # Compute residual
        residual = effort - gravity  # Contact forces + noise

        # JEPA on residuals
        residual_embed = self.encoder(residual)
        # ... masked prediction on residual embeddings
```

---

### Option C: Episode-Level Contrastive Learning

**Idea**: Embed entire episodes, cluster normal operations, detect anomalies as outliers.

```
Normal episodes → cluster in embedding space
Anomaly episodes → outliers from normal cluster
```

**Implementation**:
```python
class EpisodeContrastive:
    def forward(self, setpoint, effort, episode_ids):
        # Encode full episode
        episode_embed = self.encoder(concat(setpoint, effort))
        episode_repr = episode_embed.mean(dim=1)  # Pool over time

        # Contrastive: same episode type = similar, different = dissimilar
        # (Use fault_label during training)
        loss = contrastive_loss(episode_repr, fault_labels)

    def detect_anomaly(self, setpoint, effort, normal_prototypes):
        episode_repr = self.encode(setpoint, effort)
        distance = (episode_repr - normal_prototypes).norm()
        return distance  # High = anomaly
```

---

### Option D: Multi-Signal Fusion

**Idea**: Combine predictable signals (gravity torques) with contact signals.

```
Anomaly signal = mismatch between joint torques and Cartesian forces
```

**Rationale**:
- Joint torques 1,2 are predictable from setpoint
- Contact forces depend on load
- **Healthy**: torques and forces are consistent
- **Anomaly**: relationship breaks down

**Implementation**:
```python
class MultiSignalJEPA:
    def forward(self, setpoint, effort):
        # Split effort into predictable vs contact
        gravity_torques = effort[:, :, [1, 2]]  # Predictable
        contact_forces = effort[:, :, 6:9]  # force_xyz

        # Encode both streams
        torque_embed = self.torque_encoder(gravity_torques)
        force_embed = self.force_encoder(contact_forces)

        # Cross-modal prediction
        pred_force = self.predictor(torque_embed, setpoint)
        target_force = self.ema_encoder(force_embed)

        # Loss: can torques predict forces?
        return jepa_loss(pred_force, target_force)
```

---

## 7. Recommendations

### Short-term (Quick Experiments)

1. **Test Temporal Prediction**: Modify existing JEPA to predict future effort from past effort+setpoint
2. **Evaluate per-signal**: Track anomaly detection separately for each effort signal
3. **Episode-level metrics**: Aggregate timestep anomaly scores to episode level

### Medium-term (Architecture Changes)

1. **Implement Residual Modeling**: Decompose into gravity + residual
2. **Multi-Signal Fusion**: Cross-modal prediction between torques and forces
3. **Temporal context window**: Extend context to include recent effort history

### Long-term (Data Improvements)

1. **Add load context**: If possible, include gripper force, object weight
2. **Phase segmentation**: Separate approach/contact/retract phases
3. **Synthetic augmentation**: Generate anomaly patterns for training

---

## 8. Next Steps

1. [ ] Implement Temporal Self-Prediction variant
2. [ ] Run baseline comparisons on AURSAD tightening subset
3. [ ] Evaluate episode-level ROC-AUC (not just timestep)
4. [ ] Test on missing_screw vs normal (clearest signal)

---

## Appendix: Code Snippets

### Loading AURSAD with fault labels

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

ds = load_dataset('Forgis/factorynet-hackathon', data_dir='aursad', split='train')
df = ds.to_pandas()

# Load metadata
meta_path = hf_hub_download(
    repo_id='Forgis/factorynet-hackathon',
    filename='metadata/aursad_metadata.json',
    repo_type='dataset'
)
with open(meta_path) as f:
    metadata = {m['episode_id']: m for m in json.load(f)}

df['fault_label'] = df['episode_id'].map(
    lambda x: metadata.get(x, {}).get('fault_label', 'unknown')
)
```

### Computing prediction error per episode

```python
# Group by episode, compute mean prediction error
episode_errors = df.groupby('episode_id').apply(
    lambda g: ((model.predict(g[setpoint_cols]) - g[effort_cols].values) ** 2).mean()
)

# Join with fault labels
episode_errors = episode_errors.to_frame('mse')
episode_errors['fault'] = episode_errors.index.map(
    lambda x: metadata.get(x, {}).get('fault_label', 'unknown')
)
```
