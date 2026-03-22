# Alternative Datasets for Cross-Machine Transfer Learning

## Summary

After researching available datasets, here are the best options for cross-machine transfer learning with multivariate sensor time series.

---

## Option 1: NASA C-MAPSS (Recommended for Clean Benchmark)

**What it is**: Turbofan engine degradation simulation with multiple units

**Why it's great**:
- Very clean, well-established benchmark
- Multiple subsets with different transfer scenarios:
  - FD001: 100 engines, 1 operating condition, 2 fault modes
  - FD002: 120 engines, 6 operating conditions, 1 fault mode
  - FD003: 100 engines, 2 operating conditions, 2 fault modes
  - FD004: 248 engines, 6 operating conditions, 2 fault modes
- 21 sensor channels (multivariate)
- Perfect for cross-unit transfer learning (train on some engines, test on others)
- Thousands of papers use this → easy to compare

**Limitations**:
- Not robots (jet engines)
- Simulated data (though realistic)
- Currently unavailable for new downloads (NASA reviewing)

**Download**:
- [Kaggle mirror](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
- [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

---

## Option 2: CWRU + Paderborn University (PU) Bearing Datasets

**What it is**: Two physically different bearing test rigs with identical fault types

**Why it's great**:
- Explicitly designed for cross-domain transfer research
- CWRU is THE most cited dataset in fault diagnosis
- PU provides a different machine with same fault types
- High-frequency vibration data (12-51 kHz)
- Well-understood baselines (94%+ accuracy on transfer tasks)

**Transfer scenarios**:
- CWRU → PU (cross-machine)
- Different operating speeds (600, 900, 1200 RPM)
- Different loads (0, 1, 2, 3 HP)

**Limitations**:
- Bearings only (not full robot)
- Single sensor type (vibration)

**Download**:
- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [PU Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter)
- [Awesome Bearing Datasets](https://github.com/VictorBauler/awesome-bearing-dataset)

---

## Option 3: IEEE PHM 2012 Challenge Dataset

**What it is**: Bearing RUL prediction challenge data

**Why it's great**:
- Competition dataset → well-defined evaluation protocol
- 6 training + 11 testing bearings
- Three different operating conditions
- Clear success metric (RUL prediction error)

**Download**:
- [GitHub](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset)
- [Kaggle](https://www.kaggle.com/datasets/alanhabrony/ieee-phm-2012-data-challenge)

---

## Option 4: HyQ Quadruped Proprioceptive Dataset

**What it is**: Hydraulic quadruped robot with full proprioceptive sensors

**Why it's great**:
- Real robot proprioception (torque, current, encoders, IMU)
- 1000 Hz sampling rate
- Multiple locomotion sequences (5-30 minutes each)
- Ground truth from motion capture

**Limitations**:
- Single robot (no cross-machine scenario built-in)
- Quadruped locomotion (different from manipulation)

**Download**: [IEEE DataPort](https://ieee-dataport.org/open-access/proprioceptive-sensor-dataset-quadruped-robots)

---

## Option 5: Open X-Embodiment (Not Recommended for Your Use Case)

**What it is**: 22 robot platforms, 60+ datasets, 1M+ trajectories

**Why NOT for you**:
- Primarily vision-based (RGB images, not sensor time series)
- Limited proprioceptive data available
- Designed for behavior cloning, not anomaly detection

**If you still want to explore**: [GitHub](https://github.com/google-deepmind/open_x_embodiment)

---

## Option 6: Stay with AURSAD + Voraus-AD (Current Approach)

**What it is**: What we're already using

**Pros**:
- Real industrial robots (UR3e, Yu-Cobot)
- Both manipulation tasks
- Full proprioceptive data (joint angles, torques/voltage, velocities)
- Anomaly labels available
- You already have the data and code

**Cons**:
- Different sensor semantics (torque vs voltage)
- Different number of joints (6 vs 6, but different kinematics)
- Less established benchmark (fewer comparison papers)

---

## Recommendation

For **cleanest cross-machine benchmark**: Use **CWRU + PU bearing datasets**
- Well-established, easy to compare with literature
- Explicitly designed for cross-domain transfer
- Can get results quickly to validate your approach

For **robotics relevance**: Stay with **AURSAD + Voraus-AD**
- More relevant to your research goals
- Already set up and working
- Novel contribution (no one has done this exact transfer)

For **RUL/prognostics focus**: Use **C-MAPSS or PHM 2012**
- Clean, simulated data
- Very well understood baselines

---

## Quick Comparison Table

| Dataset | Machines | Sensors | Transfer Scenario | Established |
|---------|----------|---------|-------------------|-------------|
| C-MAPSS | 100-248 engines | 21 channels | Cross-unit, cross-condition | ⭐⭐⭐⭐⭐ |
| CWRU + PU | 2 test rigs | Vibration | Cross-machine, cross-speed | ⭐⭐⭐⭐⭐ |
| PHM 2012 | 17 bearings | Vibration | Cross-condition | ⭐⭐⭐⭐ |
| HyQ | 1 quadruped | Full proprio | N/A (single robot) | ⭐⭐ |
| Open X | 22 robots | Mostly vision | Cross-embodiment | ⭐⭐⭐ |
| AURSAD+Voraus | 2 arms | Full proprio | Cross-robot | ⭐⭐ |

---

## Next Steps

1. **Quick validation**: Try CWRU → PU transfer with your current architecture
2. **If that works**: Your method is sound, continue with AURSAD → Voraus
3. **If it fails**: Debug on cleaner data before tackling harder problem
