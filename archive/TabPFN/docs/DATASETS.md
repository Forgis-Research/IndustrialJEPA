# Dataset Selection for TabPFN-TS Mechanical Forecasting

**Selection criteria**: Datasets must be (1) mechanical/industrial in nature, (2) time series with forecasting potential, (3) accessible without extensive approval processes.

---

## Primary Datasets

### 1. UCI Hydraulic System (Recommended First Test)

| Property | Value |
|----------|-------|
| **Domain** | Industrial hydraulic systems |
| **Channels** | 17 sensors |
| **Samples** | 2,205 cycles × 60s each |
| **Sampling rates** | 1-100 Hz (mixed) |
| **Data type** | Real laboratory measurements |
| **Download** | Direct (no registration) |

**Why this dataset:**
- Real mechanical system with ground truth labels
- Multiple sensor modalities (pressure, flow, temperature, vibration)
- Natural covariates: fault states as implicit conditions
- Clear physics groups for potential cross-sensor conditioning

**TabPFN-TS relevance:**
- Covariates: Can use fault labels or other sensors as covariates
- Periodic patterns: 60s load cycles create natural periodicity
- Multi-modal: Test if cross-sensor information helps

**Experiment ideas:**
1. Single-sensor forecasting (PS1-PS6, FS1-FS2)
2. Cross-sensor conditioning (use pressure to predict flow)
3. Fault-conditioned forecasting (healthy vs degraded)

---

### 2. C-MAPSS Turbofan Engine

| Property | Value |
|----------|-------|
| **Domain** | Aerospace / turbomachinery |
| **Channels** | 21 sensors + 3 operating settings |
| **Samples** | ~20k engine cycles across 4 subsets |
| **Sampling rate** | 1 sample per flight cycle |
| **Data type** | Simulated (NASA MAPSS) |
| **Download** | Kaggle or NASA portal |

**Why this dataset:**
- Widely used benchmark (easy comparison)
- Operating settings as natural covariates
- Degradation trends (tests extrapolation ability)
- Multiple fault modes and conditions

**TabPFN-TS relevance:**
- Covariates: Operating settings (altitude, throttle) directly affect sensors
- Trend forecasting: Can TabPFN-TS capture degradation trajectories?
- Multi-condition: Transfer across FD001-FD004

**Experiment ideas:**
1. Sensor forecasting with operating settings as covariates
2. Compare TabPFN-TS vs linear trend baseline on degradation
3. Cross-condition transfer (train FD001, test FD003)

**Known limitations:**
- Synthetic data (not real sensors)
- Standard task is RUL, not forecasting (no published SOTA)
- Sensors are highly correlated (degradation affects all together)

---

### 3. Paderborn Bearing

| Property | Value |
|----------|-------|
| **Domain** | Rotating machinery / bearings |
| **Channels** | 8 (vibration, thermal, electrical) |
| **Samples** | 256k per file × 33 bearings |
| **Sampling rate** | 64 kHz (vibration), lower for others |
| **Data type** | Real laboratory measurements |
| **Download** | Direct (requires unrar) |

**Why this dataset:**
- Real vibration signals with known faults
- Multiple modalities (vibration + current + temperature)
- Natural periodicity from rotation
- Cross-condition transfer potential (healthy → faulty)

**TabPFN-TS relevance:**
- Periodic signals: Rotation creates strong periodic patterns
- Multi-modal covariates: Use motor current to predict vibration
- Fault detection: Residuals might indicate developing faults

**Experiment ideas:**
1. Short-horizon vibration forecasting
2. Cross-modality conditioning (current → vibration)
3. Healthy vs faulty operating regime

**Practical notes:**
- High sampling rate requires subsampling
- Large file sizes (plan storage)
- Multiple bearing types and fault severities

---

## Secondary Datasets

### 4. AURSAD (Robot Anomaly Detection)

| Property | Value |
|----------|-------|
| **Domain** | Industrial robotics (UR3e) |
| **Channels** | 20 (position, velocity, current, Cartesian) |
| **Samples** | ~6M total (4,094 episodes) |
| **Sampling rate** | ~125 Hz |
| **Data type** | Real robot data |

**TabPFN-TS potential:**
- Strong covariates: Joint commands directly cause sensor readings
- Periodic motion: Repetitive tasks create periodic signals
- Anomaly detection via forecast residuals

### 5. Voraus-AD (Collaborative Robot)

| Property | Value |
|----------|-------|
| **Domain** | Collaborative robotics |
| **Channels** | 66 (comprehensive joint data) |
| **Samples** | ~4M total |
| **Data type** | Real robot data |

**TabPFN-TS potential:**
- Very high channel count (TabPFN handles tabular well)
- Rich covariate set
- Modern robot platform

---

## Dataset Comparison Matrix

| Dataset | Real? | Channels | Covariates | Periodicity | Degradation | Difficulty |
|---------|-------|----------|------------|-------------|-------------|------------|
| Hydraulic | Yes | 17 | Moderate | Strong | Mild | Easy |
| C-MAPSS | No | 24 | Strong | None | Strong | Medium |
| Paderborn | Yes | 8 | Weak | Strong | Strong | Medium |
| AURSAD | Yes | 20 | Strong | Strong | Weak | Medium |
| Voraus-AD | Yes | 66 | Strong | Strong | Weak | Hard |

---

## Recommended Experiment Order

1. **Start with Hydraulic**:
   - Easy download, real data, clear structure
   - Validate TabPFN-TS works at all

2. **Move to C-MAPSS**:
   - Strong covariate story (operating settings)
   - Compare to degradation baselines

3. **Then Paderborn** (if promising):
   - Higher sampling rate, more complex
   - Tests periodic signal handling

4. **AURSAD/Voraus-AD** (stretch goal):
   - Higher complexity, richer covariates
   - Cross-embodiment transfer potential

---

## Data Loading Snippets

### Hydraulic System
```python
import numpy as np
from pathlib import Path

data_dir = Path('datasets/data/hydraulic')
ps1 = np.loadtxt(data_dir / 'PS1.txt')  # Shape: (2205, 6000)
# ps1[cycle_idx, timestep] gives pressure value
```

### C-MAPSS
```python
import pandas as pd

train_file = 'datasets/data/cmapss/train_FD001.txt'
data = pd.read_csv(train_file, sep=' ', header=None)
# Columns: unit_id, cycle, setting1-3, sensor1-21
unit_1 = data[data[0] == 1]
```

### Paderborn
```python
import scipy.io

mat_file = 'datasets/data/paderborn/K001/N15_M07_F10_K001_1.mat'
data = scipy.io.loadmat(mat_file)
# Access vibration, current, etc. from data dict
```

---

## References

- Hydraulic: https://archive.ics.uci.edu/dataset/447
- C-MAPSS: https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data
- Paderborn: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter
- AURSAD: https://zenodo.org/record/4905920
- Voraus-AD: https://github.com/voraus-io/voraus-AD-dataset
