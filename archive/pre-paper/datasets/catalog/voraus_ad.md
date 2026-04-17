# Voraus-AD (Yu-Cobot Pick-and-Place Anomaly Dataset)

## Executive Summary
- **Domain**: Industrial Robotics / Assembly
- **Task**: Anomaly detection; adaptable to forecasting
- **Size**: ~2,000 episodes × ~2,000 timesteps × 66 channels
- **Sampling Rate**: 100 Hz
- **Real vs Synthetic**: Real — physical Yu-Cobot 6-DOF collaborative robot
- **License**: CC BY 4.0
- **Download URL**: https://github.com/voraus-io/voraus-AD-dataset
- **Published SOTA**: Limited — introduced in 2023, few follow-up papers

## Detailed Description

Voraus-AD was collected from a Yu-Cobot collaborative robot performing a pick-and-place task. It has richer signal coverage than AURSAD, including joint-level voltage, current, and position data. The dataset was designed specifically to benchmark anomaly detection methods for collaborative robots.

### Physical Setup
- **Robot**: Yu-Cobot 6-DOF collaborative arm
- **Task**: Pick-and-place (object manipulation)
- **Anomalies**: Payload variations, joint disturbances, tool changes

### Channels (66 total)
| Channel Group | Count | Type | Description |
|---|---|---|---|
| Joint position | 6 | rad | Actual joint angles |
| Joint velocity | 6 | rad/s | Actual joint velocities |
| Joint current | 6 | A | Motor drive current |
| Joint voltage | 6 | V | Motor drive voltage |
| Joint temperature | 6 | °C | Joint motor temperature |
| Joint torque | 6 | Nm | Measured joint torques |
| Cartesian position | 3 | m | TCP position |
| Cartesian orientation | 3 | rad | TCP orientation |
| Cartesian velocity | 6 | m/s, rad/s | TCP linear + angular velocity |
| Command values | 12 | various | Commanded position, velocity, torque |

## Physics Groups
```python
VORAUS_GROUPS = {
    "joint_position":    [0..5],    # 6 joint angles
    "joint_velocity":    [6..11],   # 6 joint velocities
    "joint_current":     [12..17],  # 6 motor currents
    "joint_voltage":     [18..23],  # 6 motor voltages
    "joint_temperature": [24..29],  # 6 motor temperatures
    "joint_torque":      [30..35],  # 6 joint torques
    "cartesian":         [36..47],  # TCP pose + velocity
    "command":           [48..59],  # Commanded setpoints
}
# 8 groups, 6 channels each (+ cartesian group with 12)
```

The voltage/current/torque structure directly maps to the IndustrialJEPA causal hypothesis:
- **Setpoint** (command group) → **Effort** (current, voltage, torque) → **Feedback** (position, velocity)

## Published Benchmarks
| Method | Metric | Value | Paper | Year |
|---|---|---|---|---|
| LSTM Autoencoder | AUC-ROC | 0.87 | Voraus paper | 2023 |
| OC-SVM | AUC-ROC | 0.81 | Voraus paper | 2023 |
| Isolation Forest | AUC-ROC | 0.79 | Voraus paper | 2023 |

## Relevance to IndustrialJEPA

### Current Usage
Target domain for cross-machine transfer (AURSAD → Voraus-AD). In progress per project status.

### Physics Grouping Potential
**Excellent** — 66 channels with 8 clear physical groups. Best physics-grouping structure of all currently used datasets. The command/effort/feedback causal chain is directly observable.

### Transfer Learning
- AURSAD (UR3e screwdriving) → Voraus-AD (Yu-Cobot pick-and-place): Different robot, different task, same 6-DOF kinematics
- Within-dataset: Normal → anomaly episodes

### Scale Adequacy
**Small** — ~2,000 episodes. Good for evaluation; insufficient for pretraining.

## Download Notes
- GitHub: https://github.com/voraus-io/voraus-AD-dataset
- Format: CSV files per episode
- Downloader: `datasets/downloaders/download_voraus.py`
