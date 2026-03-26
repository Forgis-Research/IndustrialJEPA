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

## Download Notes
- Available on GitHub: https://github.com/saifsustain/AURSAD
- Also on Zenodo: https://zenodo.org/record/4905920
- Format: HDF5 (.h5) or CSV
- Downloader: `datasets/downloaders/download_aursad.py`
