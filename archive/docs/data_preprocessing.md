# Data Preprocessing Pipeline

> **Status**: Implementation complete, validation in progress

This document details the data preprocessing pipeline for IndustrialJEPA, including the unified schema design, known issues, and validation checklist.

---

## Overview

```
Raw FactoryNet (HuggingFace)
    │
    ▼
┌─────────────────────────────┐
│  1. Load subset (data_dir)  │  e.g., aursad, voraus-ad
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. Column discovery        │  Auto-detect setpoint/effort columns
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. Episode-level split     │  80/10/10 train/val/test (by episode, not row)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. Windowing               │  256 timesteps, stride 128
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5. Normalization           │  Per-window z-score
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  6. Unified padding + mask  │  Pad to (14, 7) with validity mask
└─────────────────────────────┘
    │
    ▼
Output: (setpoint, effort, metadata)
```

---

## Unified Schema Design

### The Problem

Different robots have different sensor configurations:

| Dataset | DOF | Setpoint Cols | Effort Type | Effort Cols |
|---------|-----|---------------|-------------|-------------|
| AURSAD | 6 | pos(6) + vel(6) = 12 | torque | 6 |
| voraus-AD | 6 | pos(6) + vel(6) = 12 | current | 6 |
| NASA Milling | 3 | pos(3) + vel(3) = 6 | force | 3 (xyz) |
| RH20T | 7 | pos(7) + vel(7) = 14 | torque | 7 |
| REASSEMBLE | 7 | pos(7) + vel(7) = 14 | torque | 7 |

### The Solution: Unified Dimensions + Validity Masks

```python
# Unified output dimensions
unified_setpoint_dim = 14  # 7 DOF × 2 (pos + vel)
unified_effort_dim = 7     # 7 DOF max

# For 6-DOF robot (e.g., AURSAD):
setpoint = [pos0, pos1, ..., pos5, vel0, ..., vel5, 0, 0]  # shape: (seq, 14)
effort = [torque0, ..., torque5, 0]                         # shape: (seq, 7)

setpoint_mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 12 real, 2 padded
effort_mask = [1, 1, 1, 1, 1, 1, 0]                          # 6 real, 1 padded
```

### Semantic Equivalence of Effort Signals

**Key assumption**: Torque, current, and force are semantically equivalent "effort" signals.

Physical justification:
- `τ = K_t × I` (torque = motor constant × current)
- `F = m × a` (force = mass × acceleration)

The model should learn that these signals represent "energy expended to achieve setpoint", regardless of the specific physical quantity measured.

---

## Column Discovery Logic

```python
# Priority order for effort signals (first match wins)
effort_signals = ["torque", "current", "voltage", "force"]

# Column patterns
EFFORT_PATTERNS = {
    "torque": [f"effort_torque_{i}" for i in range(7)],
    "current": [f"effort_current_{i}" for i in range(7)],
    "force": ["effort_force_x", "effort_force_y", "effort_force_z"],
}
```

The loader tries each signal type in order and uses the first one found. This allows automatic adaptation to dataset-specific schemas.

---

## Normalization Strategy

### Per-Window Z-Score (Default)

```python
# For each window independently:
setpoint_norm = (setpoint - mean(setpoint)) / (std(setpoint) + 1e-8)
effort_norm = (effort - mean(effort)) / (std(effort) + 1e-8)
```

**Pros:**
- Scale-invariant across robots with different sensor ranges
- Each window is self-contained (no train/test leakage)

**Cons:**
- Loses absolute magnitude information (is 10 Nm a lot?)
- Constant signals become 0/0 → handled by epsilon

### Global Z-Score (Alternative)

Compute mean/std over entire training set. Preserves absolute scale but requires careful handling of dataset mixing.

---

## Validation Checklist

### Data Loading
- [x] Primary dataset (Forgis/factorynet-hackathon) loads correctly
- [x] Fallback to karimm6/FactoryNet_Dataset works
- [x] Subset filtering via `data_dir` parameter
- [ ] **TODO**: Test loading ALL subsets (voraus-ad, nasa-milling, rh20t, reassemble)
- [ ] **TODO**: Test multi-dataset loading (combine subsets)

### Column Discovery
- [x] Auto-detect setpoint columns (pos + vel)
- [x] Auto-detect effort columns (torque preferred over current)
- [ ] **TODO**: Verify voraus-AD uses current (not torque)
- [ ] **TODO**: Verify NASA Milling uses force (not torque)
- [ ] **TODO**: Handle missing columns gracefully

### Unified Schema
- [x] Padding to unified dimensions
- [x] Validity masks generated correctly
- [ ] **TODO**: Verify 7-DOF robots (RH20T, REASSEMBLE) don't get padded
- [ ] **TODO**: Test model training with masks (masked loss)

### Episode Splitting
- [x] Episode-level split (no row-level leakage)
- [x] Healthy-only training mode
- [x] Reproducible splits (seed=42)
- [ ] **TODO**: Verify fault episodes only in test set when train_healthy_only=True

### Normalization
- [x] Per-window z-score normalization
- [ ] **TODO**: Handle constant signals (zero variance)
- [ ] **TODO**: Test global normalization mode

---

## Open Questions / TODOs

### High Priority

1. **Sampling Rate Alignment**
   - Different datasets may have different sampling rates (100Hz vs 500Hz)
   - Current approach: treat each timestep equally
   - **TODO**: Investigate resampling to common rate

2. **Cartesian vs Joint Space**
   - NASA Milling uses Cartesian forces (x, y, z)
   - Robot datasets use joint torques
   - **TODO**: Should we project to common space?

3. **Multi-Dataset Training**
   - How to batch samples from different datasets?
   - Option A: Random mixing (current approach)
   - Option B: Stratified sampling (equal dataset representation)
   - **TODO**: Implement and compare

### Medium Priority

4. **Timestamp Handling**
   - `timestamp` column exists but not used
   - Could enable variable-length sequences
   - **TODO**: Consider time-aware positional encoding

5. **Context Labels**
   - `ctx_anomaly_label` exists with fault types
   - Currently: binary (healthy vs any fault)
   - **TODO**: Multi-class fault classification

6. **Acceleration Signals**
   - `setpoint_acc_*` exists in some datasets
   - Currently: not used (only pos + vel)
   - **TODO**: Ablate inclusion of acceleration

### Low Priority

7. **End-Effector Signals**
   - `feedback_vel_ee_*` (end-effector velocity)
   - `ctx_temp_*` (joint temperatures)
   - **TODO**: Evaluate as additional inputs

8. **Data Augmentation**
   - Window jittering
   - Noise injection
   - **TODO**: Implement and evaluate

---

## Known Issues

### Issue 1: AURSAD Episode Count Mismatch [FIXED]

**Status**: ✅ RESOLVED

**Problem**: FactoryNet contains 4094 AURSAD episodes, but the paper reports 2045.

**Root cause**: The dataset includes 2049 supplementary "loosening" episodes (label 5) which describe a **different task** (unscrewing), not anomalies in the tightening task.

**Solution**:
1. Load fault labels from metadata JSON (`metadata/aursad_metadata.json`)
2. Added `aursad_tightening_only=True` config to filter out loosening episodes
3. Use `fault_label` field which matches original paper labels

**Verified result (matches paper exactly)**:
```
Paper: 2045 episodes (1420 normal + 625 fault)
Our loader:
  Filtered out 2049 loosening episodes (label 5, different task)
  Remaining: 2045 episodes
  Episodes: 1420 healthy, 625 fault ✅

Fault types:
  normal: 1420 ✅
  damaged_screw: 221 ✅
  extra_component: 183 ✅
  missing_screw: 218 ✅
  damaged_thread: 3 ✅
```

**Reference**: [AURSAD Paper](https://arxiv.org/abs/2102.01409)

### AURSAD Phase Structure (Important!)

The AURSAD dataset contains **paired operations**, not independent episodes:

```
Operation Cycle = Loosening (prepare/pick screw) → Tightening (screw in)
                  Episode N (odd)                   Episode N+1 (even)
```

- **2049 loosening episodes** (odd IDs: 00001, 00003, ...)
- **2045 tightening episodes** (even IDs: 00002, 00004, ...)
- **4 unpaired loosening** (131, 1290, 1469, 2154 - incomplete cycles)

**Fault labels only apply to tightening phase** - that's where anomalies occur.
Loosening is a different motion (unscrewing), always considered "normal operation".

**Config options**:
```python
# Both phases - all follow valid Setpoint→Effort physics
aursad_phase_handling = "both"  # default, 4094 episodes

# Tightening only - matches paper's 2045 episodes
aursad_phase_handling = "tightening_only"  # 2045 episodes

# Merge paired phases (TODO)
aursad_phase_handling = "merge"
```

### Issue 2: Windows Span Episode Boundaries

**Current behavior**: Windows are created per-episode, never crossing boundaries.

**Verified**: ✅ (stride resets at episode start)

### Issue 3: Memory Usage

Loading full dataset into pandas DataFrame can be memory-intensive.

**Mitigation**: Use streaming mode for exploration, full load for training.

**TODO**: Consider memory-mapped loading for large-scale training.

---

## Usage Examples

### Basic Loading

```python
from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

config = FactoryNetConfig(
    subset="AURSAD",
    window_size=256,
    stride=128,
)

train_ds = FactoryNetDataset(config, split="train")
setpoint, effort, meta = train_ds[0]

print(f"Setpoint: {setpoint.shape}")  # (256, 14)
print(f"Effort: {effort.shape}")      # (256, 7)
print(f"Mask: {meta['setpoint_mask']}")  # [1,1,1,1,1,1,1,1,1,1,1,1,0,0]
```

### Multi-Dataset Training

```python
# TODO: Implement ConcatDataset wrapper
from torch.utils.data import ConcatDataset

configs = [
    FactoryNetConfig(subset="AURSAD"),
    FactoryNetConfig(subset="voraus-AD"),
]

datasets = [FactoryNetDataset(c, split="train") for c in configs]
combined = ConcatDataset(datasets)
```

---

## References

- FactoryNet Paper: [Hackathon Documentation]
- HuggingFace Dataset: `Forgis/factorynet-hackathon`
- Implementation: `src/industrialjepa/data/factorynet.py`
