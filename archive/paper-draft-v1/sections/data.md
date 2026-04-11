# Data: FactoryNet Multi-Robot Benchmark

## 3.1 Dataset Overview

We use **FactoryNet**, a unified multi-robot industrial time-series dataset comprising 18.2M timesteps across 6,983 episodes from five robotic systems (Table 1).

| Dataset | Robot | DOF | Task | Episodes | Rows | Fault Types |
|---------|-------|-----|------|----------|------|-------------|
| AURSAD | UR3e | 6 | Screwdriving | 2,045 cycles* | 6.2M | damaged_screw, missing_screw, extra_component, damaged_thread |
| voraus-AD | Yu-Cobot | 6 | Pick-and-place | 2,122 | 2.3M | 12 types |
| NASA Milling | CNC | 3 | Milling | 167 | 1.5M | Tool wear (continuous) |
| RH20T | Franka Panda | 7 | Manipulation | 500 | 1.2M | — |
| REASSEMBLE | Franka Panda | 7 | Assembly | 100 | 1.0M | Task success/fail |

*AURSAD contains paired operations: each cycle = loosening phase (prepare) + tightening phase (screw in). Faults occur only during tightening. Total 4,094 phase-episodes, or 2,045 complete operation cycles.

**Table 1:** FactoryNet composition. Each dataset captures different robot hardware, tasks, and fault modalities.

## 3.2 Causal Signal Structure

Each timestep contains three semantically-grouped signal types following the causal chain of robot operation:

**Setpoint** (X): Controller commands
- Joint positions: `setpoint_pos_{0..N}` [rad]
- Joint velocities: `setpoint_vel_{0..N}` [rad/s]

**Effort** (Y): Energy expended to achieve commands
- Joint torques: `effort_torque_{0..N}` [Nm]
- Motor currents: `effort_current_{0..N}` [A] (dataset-dependent)

**Feedback** (Z): Measured outcomes (not used in JEPA objective)
- Measured positions: `feedback_pos_{0..N}` [rad]

The core insight is that under healthy operation, **Effort is a deterministic function of Setpoint** governed by robot dynamics (F = ma, τ = Iα). Faults disrupt this relationship.

## 3.3 Unified Schema for Cross-Robot Training

Since robots have varying degrees of freedom (3-7 DOF) and sensor configurations (torque vs. current), we adopt a **unified schema** with zero-padding:

- **Unified Setpoint dimension**: 14 (7 DOF × 2 signals: pos + vel)
- **Unified Effort dimension**: 7 (7 DOF)

For a 6-DOF robot, the last dimensions are zero-padded, with a **validity mask** indicating real vs. padded dimensions. This enables:
1. Batching across heterogeneous datasets
2. Shared model weights across robots
3. Zero-shot transfer evaluation

## 3.4 Preprocessing Pipeline

1. **Windowing**: Extract overlapping windows (256 timesteps, stride 128) from episodes
2. **Normalization**: Per-window z-score normalization (zero mean, unit variance)
3. **Padding**: Pad to unified dimensions with validity mask
4. **Split**: Episode-level split (80/10/10 train/val/test) ensuring no episode leakage

For anomaly detection, we train exclusively on **healthy episodes** and evaluate on held-out healthy + all fault episodes, following one-class classification methodology.

## 3.5 Data Availability

FactoryNet is publicly available on HuggingFace:
`Forgis/factorynet-hackathon` (CC BY 4.0)
