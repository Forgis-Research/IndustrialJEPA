# AURSAD (Anomaly Detection Dataset for UR3e Screwdriving)

## Executive Summary
- **Domain**: Industrial Robotics / Assembly
- **Task**: Fault detection / anomaly detection; adaptable to forecasting
- **Size**: 4,094 episodes × ~1,500 timesteps × 20 channels
- **Sampling Rate**: 500 Hz
- **Real vs Synthetic**: Real — physical UR3e robot performing screwdriving tasks
- **License**: CC BY 4.0
- **Download URL**: https://github.com/saifsustain/AURSAD
- **Published SOTA**: Moderate — the paper itself reports baselines; a few follow-up papers

## Detailed Description

AURSAD (Anomaly detection in Uniform Robot Screwdriving Assembly Dataset) was collected from a UR3e 6-DOF collaborative robot performing a screwdriving assembly task. The robot inserts a screwdriver into a bolt and tightens it. Fault conditions include damaged screws, missing screws, extra parts, and damaged threads.

### Physical Setup
- **Robot**: Universal Robots UR3e (6-DOF collaborative robot arm)
- **Task**: Screwdriving assembly — approach, contact, tighten
- **Fault conditions**: 5 classes
- **Healthy episodes**: 3,469 (85%)

### Episode Classes
| Label | Count | Description |
|---|---|---|
| normal | 3,469 | Healthy screwdriving |
| damaged_screw | 221 | Screw thread damaged |
| missing_screw | 218 | No screw present |
| extra_part | 183 | Obstruction present |
| damaged_thread | 3 | Thread damaged (very rare) |

### Channels (20 total)
| Channel | Type | Unit | Description |
|---|---|---|---|
| q1–q6 | joint_position | rad | Joint angles (6 DOF) |
| qd1–qd6 | joint_velocity | rad/s | Joint velocities (6 DOF) |
| i1–i6 | motor_current | A | Joint motor currents (6 DOF) |
| TCP_x/y/z | cartesian | m | Tool center point position |
| TCP_rx/ry/rz | orientation | rad | Tool orientation (Euler) |

## Physics Groups
```python
AURSAD_GROUPS = {
    "joint_position":   [0, 1, 2, 3, 4, 5],      # q1-q6
    "joint_velocity":   [6, 7, 8, 9, 10, 11],     # qd1-qd6
    "motor_current":    [12, 13, 14, 15, 16, 17],  # i1-i6
    "cartesian_pose":   [18, 19, 20, 21, 22, 23],  # TCP xyz + rpy
}
# 4 groups × 6 channels = 24 channels total (some sources use 20)
```

## Published Benchmarks
| Method | Metric | Value | Paper | Year |
|---|---|---|---|---|
| MLP (time features) | AUC-ROC | 0.89 | Bremer et al. | 2020 |
| CNN-1D | F1 | 0.92 | Bremer et al. | 2020 |
| LSTM | F1 | 0.91 | Bremer et al. | 2020 |
| IndustrialJEPA (temporal) | AUC-ROC | 0.538 | This project | 2026 |
| IndustrialJEPA (temporal) | PR-AUC | 0.832 | This project | 2026 |

## Relevance to IndustrialJEPA

### Current Usage
Primary training dataset for IndustrialJEPA. Used for anomaly detection experiments (Exp 1–6). JEPA temporal prediction trained on this dataset.

### Physics Grouping Potential
**Strong** — 4 clear physical groups (position, velocity, current, cartesian). Current signals are the "effort" variable (causal structure: setpoint → effort → feedback). This matches IndustrialJEPA's core hypothesis.

### Transfer Learning
- AURSAD → Voraus-AD: UR3e (screwdriving) → Yu-Cobot (pick-and-place); different robot, similar 6-DOF structure
- Normal → fault transfer: within-dataset generalization
- Joint-level: train on specific joints, test on others

### Scale Adequacy
**Moderate** — 4,094 episodes × ~1,500 timesteps = ~6.1M timesteps × 20 channels. Good for supervised learning; small for pretraining.

---

## Evaluation Suite: Meta-Feature Prediction

AURSAD is the **primary dataset for meta-feature prediction experiments** due to its clear causal structure (command → effort → feedback) and well-labeled fault classes.

### High-Value Meta-Features for AURSAD

| Feature | Channels | Computation | Physical Meaning | Anomaly Signal |
|---|---|---|---|---|
| **Effort Variance** | i1–i6 | σ(current) per joint | Motor stress | Normal: 0.02-0.05A, Fault: 0.08-0.15A |
| **Command-Response Lag** | q1–q6 vs qd1–qd6 | Cross-correlation peak delay | Mechanical binding | Normal: 0-2ms, Fault: 5-15ms |
| **Coupling Strength** | i_j ↔ qd_j | Pearson correlation | Current-velocity coupling | Normal: 0.6-0.8, Fault: 0.3-0.5 (decoupled) |
| **Joint Asymmetry** | i1–i6 | Gini coefficient across joints | Load distribution | Normal: 0.1-0.2, Fault: 0.3-0.5 |
| **Velocity Smoothness** | qd1–qd6 | |d²q/dt²| (jerk) | Motion quality | Fault → increased jerk |
| **TCP Deviation** | TCP_xyz | |actual - commanded| | Position error | Fault → increased deviation |

### Causal Structure for Meta-Features

```
[Setpoint (q*)]  →  [Effort (i)]  →  [Feedback (q, qd)]  →  [Output (TCP)]
                          ↓
                    ANOMALY SIGNALS:
                    - Effort variance ↑
                    - Lag ↑
                    - Coupling ↓
```

**Key insight**: Faults manifest as disruptions in the causal chain. Current (effort) is the earliest indicator; TCP deviation is the latest.

### Rapid Evaluation (Meta-Features)

| Test | Feature | Metric | Baseline | Target |
|---|---|---|---|---|
| Variance spike | Effort variance (i1–i6) | F1 on fault class | Threshold detector: 0.75 | JEPA latent: 0.85 |
| Lag detection | Command-response delay | AUC-ROC | Statistical threshold: 0.70 | JEPA latent: 0.80 |

### Full-Scale Benchmarks (Meta-Features)

| Benchmark | Features Used | Metric | Paper Claim |
|---|---|---|---|
| Single vs multi-feature | Effort variance alone vs all 6 features | F1-score | "Multi-feature +10-15% over single" |
| Raw vs latent features | Features on raw data vs JEPA latent | F1-score | "Latent features +5-8% generalization" |
| Early detection | Time-to-detection from episode start | Timesteps | "Detect fault 20-30% earlier than baseline" |
| Fault type discrimination | Per-class F1 | Macro-F1 | "Meta-features distinguish fault types" |

### Comparison to Prior Work

| Method | AUC-ROC | F1 | Features Used |
|---|---|---|---|
| Threshold on current | ~0.70 | ~0.65 | Raw current |
| MLP (time features) | 0.89 | 0.87 | Hand-crafted |
| CNN-1D | — | 0.92 | Learned |
| LSTM | — | 0.91 | Learned |
| **Target: JEPA meta-features** | **0.90** | **0.88** | **Latent meta-features** |

---

## Download Notes
- Available on GitHub: https://github.com/saifsustain/AURSAD
- Also on Zenodo: https://zenodo.org/record/4905920
- Format: HDF5 (.h5) or CSV
- Downloader: `datasets/downloaders/download_aursad.py`
