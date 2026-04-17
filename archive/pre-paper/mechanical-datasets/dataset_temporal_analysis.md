# Mechanical Vibration Datasets: Temporal and Action-Conditioning Analysis

**Compiled:** 2026-04-01
**Purpose:** Determine what temporal structure and control information is actually available in 10 identified bearing/gearbox datasets for action-conditioned JEPA models.

---

## Summary Table

| Dataset | Episode Structure | Control Setpoints | Transitions | Fault Severity | Runtime/Age | Timestamps |
|---------|------------------|-------------------|-------------|----------------|-------------|------------|
| **1. MFPT** | Independent snapshots (3-6 sec) | MEASURED only (RPM, load) | No - steady state only | Binary (healthy/faulty) | No | No |
| **2. CWRU** | Independent snapshots | MEASURED only (RPM inferred from load) | No - steady state only | Binary + fault size (diameter) | No | No |
| **3. IMS/NASA** | Time series episodes (run-to-failure) | SET: 2000 RPM, 6000 lbs (constant) | No - constant speed/load | Continuous (implicit degradation) | Yes - file timestamps show progression | Yes - 10 min intervals in filenames |
| **4. XJTU-SY** | Time series episodes (run-to-failure) | SET: 3 conditions (RPM+load fixed per test) | No - constant per run | Continuous (implicit degradation) | Yes - CSV files in sequence | Yes - 1 min intervals implicit |
| **5. Paderborn** | Independent snapshots (4 sec each) | SET: 4 conditions (RPM, load, torque fixed) | No - steady state only | Discrete levels (26 damage states) | No - 20 repeated measurements | No - only run number |
| **6. FEMTO** | Time series episodes (run-to-failure) | SET: 3 conditions (RPM+load fixed per test) | No - constant per run | Continuous (implicit degradation) | Yes - time-to-failure tracked | Yes - continuous sampling |
| **7. PHM 2009** | Independent snapshots | SET: 5 speeds (30-50 Hz), 2 loads | No - fixed per sample | Continuous (45-dim multi-label severity) | No | No |
| **8. OEDI** | Time series (1 min segments) | MEASURED: high-speed shaft RPM signals | Possibly - dynamometer testing | Binary (healthy/damaged) | No | Limited - segment structure |
| **9. MCC5-THU** | Time series with transitions (60 sec) | SET: Speed/load sequences programmed | YES - speed ramps (e.g., 0→2500→3000 RPM) | Discrete (fault types and severities) | No | No - CSV without timestamps |
| **10. Mendeley** | Time series episodes | VARIED: 680-2460 RPM (random variations) | YES - speed ramps (up/down transitions) | Binary (healthy/inner/outer fault) | No | Likely present but format varies |

---

## Detailed Findings by Dataset

### 1. MFPT (Machinery Failure Prevention Technology)

**Episode Structure:** Independent snapshots
- Baseline: 6 seconds @ 97,656 samples/sec
- Faults: 3 seconds @ 48,828 samples/sec
- Each file is a single test condition, not continuous

**Control Setpoints:** MEASURED values only
- RPM: 25 Hz input shaft rate (fixed)
- Load: 25-300 lbs (7 levels tested)
- No commanded/target values in data

**Transitions:** None
- All data at steady-state operation

**Fault Severity:** Binary + fault location
- 3 baseline (healthy)
- 3 outer race faults
- 7 inner race faults
- No continuous degradation metric

**Runtime/Age:** Not tracked
- Independent test runs

**Timestamps:** None
- Only sampling metadata

**Verdict:** Limited for action-conditioned JEPA. No temporal episodes, no control actions, no transitions.

---

### 2. CWRU (Case Western Reserve University)

**Episode Structure:** Independent snapshots
- Each .mat file is a single test condition
- Time series within file, but no cross-file continuity

**Control Setpoints:** MEASURED only (inferred)
- RPM measured: 1730-1797 RPM (varies with load)
- Load: 0-3 HP (motor load)
- Target values not explicitly stored

**Transitions:** None
- All steady-state operation
- One revolution ≈ 0.0344 sec @ 1750 RPM

**Fault Severity:** Binary + fault diameter
- Fault sizes: 0.007, 0.014, 0.021, 0.028, 0.040 inches
- No continuous degradation metric
- Fault location: inner race, outer race, ball

**Runtime/Age:** Not tracked
- Independent tests

**Timestamps:** None
- Sampling rate: 12 kHz or 48 kHz

**Verdict:** Poor for action-conditioned JEPA. Widely-used benchmark but very limited temporal/control information.

---

### 3. IMS/NASA Bearing Dataset

**Episode Structure:** Time series episodes (run-to-failure)
- 3 test sets, 4 bearings each
- Continuous monitoring until failure
- Files represent sequential snapshots

**Control Setpoints:** SET and held constant
- Speed: 2000 RPM (constant)
- Load: 6000 lbs (constant)
- No dynamic control changes

**Transitions:** None during normal operation
- Interruptions between days (gaps in timestamps)
- File timestamps show experiment resumption

**Fault Severity:** Continuous (implicit)
- No labeled degradation percentage
- Researchers derive health indicators post-hoc
- 7 states of health observed in literature

**Runtime/Age:** YES
- File timestamps show progression
- Sampling: 1.024 sec snapshots every 10 minutes
- Run-to-failure trajectory captured

**Timestamps:** YES
- File naming includes timestamps
- 10-minute intervals between samples
- Can reconstruct temporal sequence

**Verdict:** GOOD for unsupervised RUL prediction, but NO control actions. Constant operating condition throughout. Temporal structure excellent.

---

### 4. XJTU-SY Bearing Dataset

**Episode Structure:** Time series episodes (run-to-failure)
- 15 bearings total (5 per condition)
- Each bearing monitored until failure
- CSV files in sequence

**Control Setpoints:** SET per test (3 conditions)
1. 2100 RPM, 12 kN radial load
2. 2250 RPM, 11 kN radial load
3. 2400 RPM, 10 kN radial load
- Constants within each bearing test

**Transitions:** None
- Fixed condition per bearing
- No speed/load changes within test

**Fault Severity:** Continuous (implicit)
- No pre-labeled degradation metric
- Research papers construct health indicators post-hoc
- Run-to-failure data enables RUL prediction

**Runtime/Age:** YES
- CSV files numbered sequentially
- 1-minute sampling intervals
- Complete degradation trajectory

**Timestamps:** Implicit
- 1-minute intervals between samples
- File sequence indicates time progression
- No explicit timestamp column in CSV

**Verdict:** GOOD for RUL but NO control variation. Like IMS/NASA, excellent temporal structure but constant operating conditions. Cross-condition learning possible (3 conditions).

---

### 5. Paderborn University (PU) Bearing Dataset

**Episode Structure:** Independent repeated measurements
- 20 measurements of 4 seconds each per setting
- Each measurement is independent
- No temporal continuity across files

**Control Setpoints:** SET (4 operating conditions)
- **Set 0 (baseline):** 1500 RPM, 1000 N, 0.7 Nm
- **Set 1:** 900 RPM, 1000 N, 0.7 Nm
- **Set 2:** 1500 RPM, 400 N, 0.7 Nm
- **Set 3:** 1500 RPM, 1000 N, 0.1 Nm
- Constants held during each 4-sec measurement

**Transitions:** None
- Steady-state for all measurements
- Parameters constant within each run

**Fault Severity:** Discrete damage states
- 32 total states: 6 healthy, 26 damaged
- 12 artificially damaged (EDM)
- 14 real damages from accelerated life tests
- Severity implicit in damage state labels

**Runtime/Age:** NO
- Independent measurements
- 20 repetitions per condition (not sequential aging)

**Timestamps:** NO
- Only run number (1-20)
- Sampling rate: 64 kHz for 4 seconds

**Verdict:** LIMITED for JEPA. Good diversity of damage states and operating conditions, but no temporal episodes or transitions. Useful for multi-condition fault classification.

---

### 6. FEMTO/PRONOSTIA Bearing Dataset

**Episode Structure:** Time series episodes (run-to-failure)
- 17 bearings total (Bearing1-1 to Bearing3-3)
- Sub-dataset 1: 2 training + 5 test runs
- Sub-dataset 2: 2 training + 5 test runs
- Sub-dataset 3: 1 test run
- High variability: 28 min to 7 hours lifetime

**Control Setpoints:** SET per test (3 conditions)
1. 1800 RPM, 4000 N radial load
2. 1650 RPM, 4200 N radial load
3. Unknown RPM, 5000 N radial load
- Constants within each bearing test
- Test stops when vibration > 20g

**Transitions:** None
- Accelerated degradation under constant load
- PRONOSTIA platform supports variable conditions but dataset uses constant

**Fault Severity:** Continuous (implicit)
- No labeled degradation percentage
- Researchers derive health indicators
- IEEE PHM 2012 Challenge format

**Runtime/Age:** YES
- Complete run-to-failure trajectory
- High-frequency continuous monitoring
- Timestamps in data files

**Timestamps:** YES
- Horizontal and vertical accelerometers
- Temperature also recorded
- Continuous sampling (exact rate varies)

**Verdict:** GOOD for RUL prediction, NO control actions. Similar to IMS/NASA and XJTU-SY: excellent run-to-failure data but constant operating conditions. 3 distinct conditions for cross-condition learning.

---

### 7. PHM 2009 Gearbox Challenge Dataset

**Episode Structure:** Independent snapshots (560 samples)
- Each sample is a single test run
- No temporal continuity between samples

**Control Setpoints:** SET per sample
- Speed: 30, 35, 40, 45, 50 Hz shaft speed (5 levels)
- Load: High and low (2 levels)
- 10 combinations tested
- Tachometer provides 10 pulses/revolution

**Transitions:** None
- Each sample at fixed speed/load
- No ramp-up or transient data

**Fault Severity:** Continuous multi-label (45-dimensional)
- Fault type, location, and MAGNITUDE annotated
- Severity is continuous for magnitude estimation
- Ground truth labels for supervised learning

**Runtime/Age:** NO
- Independent test runs
- No aging data

**Timestamps:** NO
- Synchronous sampling from multiple accelerometers
- No timestamp metadata

**Verdict:** MODERATE for action-conditioned learning. Good diversity of operating conditions (5 speeds × 2 loads) with continuous severity labels. But no temporal episodes or transitions. Useful for steady-state condition-dependent fault diagnosis.

---

### 8. OEDI Wind Turbine Gearbox Dataset

**Episode Structure:** Time series segments
- 10 datasets of 1-minute duration
- Healthy and damaged gearbox tests
- Segment-level structure

**Control Setpoints:** MEASURED (high-speed shaft RPM)
- Generator: 1800 RPM and 1200 RPM nominal
- RPM signals recorded alongside vibration
- Dynamometer testing with controlled loading

**Transitions:** Possibly present
- Dynamometer testing suggests potential transients
- Documentation unclear on ramp-up/down
- Likely includes startup/shutdown

**Fault Severity:** Binary
- Healthy vs damaged gearbox
- Natural faults from GRC (Gearbox Reliability Collaborative)
- No continuous severity metric

**Runtime/Age:** NO
- Snapshot comparison (healthy vs damaged)
- Not run-to-failure

**Timestamps:** Implicit segment structure
- 1-minute segments
- 40 kHz sampling rate
- Limited temporal metadata

**Verdict:** MODERATE potential. Real-world wind turbine gearbox data with RPM signals. May contain transients but requires inspection. Binary fault labels limit RUL application.

---

### 9. MCC5-THU Gearbox Benchmark Dataset

**Episode Structure:** Time series with programmed transitions (60 sec)
- Each file is 60-second recording
- Speed/load changes within recording
- 12 working conditions total

**Control Setpoints:** SET (programmed sequences)
- Speed transitions (time-varying):
  - 0-500-1000 RPM
  - 0-1500-2000 RPM
  - 0-2500-3000 RPM
- Load: 10 Nm or 20 Nm
- Example: 3000 RPM @ 10-20s and 40-50s, 2500 RPM @ 25-30s

**Transitions:** YES - explicitly designed
- Speed ramps programmed
- Non-stationary conditions by design
- Excellent for testing algorithms under varying conditions

**Fault Severity:** Discrete fault types and severities
- Intentionally induced faults
- Diverse fault types
- Compound faults included
- Severity levels documented

**Runtime/Age:** NO
- Independent 60-second recordings
- Not run-to-failure

**Timestamps:** NO
- CSV format without timestamp column
- Sampling: 12.8 kHz for 60 seconds
- Time can be inferred from sample count

**Verdict:** EXCELLENT for action-conditioned JEPA! Only dataset with explicit speed transitions and time-varying control. Non-stationary by design. However: NO timestamps in files, discrete fault severities, and no degradation progression.

---

### 10. Mendeley Varying Speed Bearing Dataset

**Episode Structure:** Time series episodes
- 3 subsets available
- Multi-modal: vibration, acoustic, temperature, current
- Variable speed tests

**Control Setpoints:** VARIED (random speed variations)
- Speed range: 680-2460 RPM
- Speed transitions: increasing, decreasing, up-then-down, down-then-up
- Load: 0 Nm, 2 Nm, 4 Nm

**Transitions:** YES - explicit speed ramps
- Non-stationary speed profiles
- Multiple transition patterns tested
- Designed for varying speed fault diagnosis

**Fault Severity:** Binary + fault location
- Healthy bearing
- Inner race faults
- Outer race faults
- No continuous degradation metric

**Runtime/Age:** NO
- Independent test runs
- Not run-to-failure

**Timestamps:** Likely present (format varies by subset)
- Multiple file formats (CSV/MAT)
- Mendeley registration required
- Detailed format needs inspection

**Verdict:** EXCELLENT for action-conditioned JEPA! Explicit speed transitions with multiple ramp patterns. However: binary fault labels, no degradation tracking, multi-modal (must extract vibration only), format inconsistency across subsets.

---

## Key Insights for Action-Conditioned JEPA

### Datasets WITH Transitions (Action-Conditioning Potential):
1. **MCC5-THU** - Programmed speed sequences within 60-sec windows
2. **Mendeley** - Speed ramps (increasing, decreasing, mixed patterns)
3. **OEDI** (possibly) - Dynamometer testing may include transients (needs verification)

### Datasets WITH Temporal Episodes (RUL/Degradation):
1. **IMS/NASA** - Run-to-failure, 10-min snapshots, timestamps
2. **XJTU-SY** - Run-to-failure, 1-min intervals, 15 bearings
3. **FEMTO** - Run-to-failure, continuous, 17 bearings
4. **OEDI** - 1-min segments (limited)

### Datasets WITH Multiple Operating Conditions (Cross-Condition Learning):
1. **Paderborn** - 4 speed/load/torque combinations
2. **XJTU-SY** - 3 speed/load combinations
3. **FEMTO** - 3 speed/load combinations
4. **PHM 2009** - 5 speeds × 2 loads (10 combinations)
5. **MFPT** - 7 load levels
6. **CWRU** - 4 load levels (0-3 HP)
7. **Mendeley** - 3 load levels + varying speed

### Critical Limitations:

**NO dataset has ALL desired properties:**
- Commanded setpoints (target values) vs measured values
- Speed/load transitions within episodes
- Continuous degradation labels
- Long temporal episodes with control actions

**Most datasets are:**
- Steady-state snapshots (no transitions)
- Binary or discrete fault labels (no continuous severity)
- Constant operating conditions (no control variation)
- Missing explicit timestamps

**Only 2 datasets have transitions:**
- MCC5-THU: Time-varying but NO timestamps, NO degradation
- Mendeley: Speed ramps but binary faults, NO degradation

**Run-to-failure datasets (IMS, XJTU, FEMTO):**
- Excellent temporal structure
- BUT constant operating conditions (no control variation)
- Continuous degradation implicit, not labeled

---

## Recommendations for Industrial-JEPA

### Option 1: Focus on MCC5-THU + Mendeley
- Only datasets with explicit transitions
- Use for action-conditioned representation learning
- Limitation: No degradation progression, discrete faults

### Option 2: Augment Run-to-Failure Datasets
- Take IMS/NASA, XJTU-SY, FEMTO
- Synthetically vary playback speed or add simulated control signals
- Focus on temporal dynamics under constant conditions

### Option 3: Multi-Condition as Proxy for Control
- Treat Paderborn, PHM 2009, CWRU as "discrete actions"
- Each operating condition = discrete action token
- Learn condition-dependent representations
- Limitation: No transitions, only steady-state

### Option 4: Hybrid Approach
- Use MCC5-THU for action-conditioning (transitions)
- Use XJTU-SY/FEMTO for temporal prediction (RUL)
- Use Paderborn/PHM for multi-condition generalization
- Combine strengths across datasets

### Reality Check:
True action-conditioned JEPA (with commanded setpoints, continuous control, and degradation tracking) requires:
- Purpose-built dataset OR
- Simulation environment OR
- Novel data collection

Existing public datasets were not designed for this use case.

---

## Sources

### MFPT Dataset:
- [MFPT Fault Data Sets](https://www.mfpt.org/fault-data-sets/)
- [MFPT Documentation - GitHub](https://github.com/opprud/bearing_dataset/blob/master/3_mfpt/README.md)
- [MFPT Dataset Analysis](https://www.researchgate.net/figure/Sample-description-of-the-MFPT-dataset_tbl4_343249978)

### CWRU Dataset:
- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [CWRU NumPy Format](https://github.com/srigas/CWRU_Bearing_NumPy)
- [All About CWRU Dataset](https://medium.com/@NameerAkhter/all-you-need-to-know-about-cwru-dataset-8d391577d8f2)

### IMS/NASA Dataset:
- [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings)
- [GitHub - Failure Classification](https://github.com/Miltos-90/Failure_Classification_of_Bearings)

### XJTU-SY Dataset:
- [XJTU-SY GitHub Repository](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)
- [XJTU-SY Official Page](https://biaowang.tech/xjtu-sy-bearing-datasets/)
- [RUL Prediction Research 2025](https://link.springer.com/article/10.1007/s43684-024-00088-4)

### Paderborn Dataset:
- [Paderborn Data Center](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download)
- [Paderborn Documentation](https://yanncalec.github.io/dpmhm/datasets/paderborn/)
- [Operating Conditions Table](https://www.researchgate.net/figure/Four-Operating-Conditions-in-the-Paderborn-University-Dataset_tbl1_389351800)

### FEMTO/PRONOSTIA Dataset:
- [PRONOSTIA Platform Paper](https://hal.science/hal-00719503v1)
- [FEMTO Dataset Analysis 2025](https://arxiv.org/pdf/2203.03259)
- [Operating Conditions](https://www.researchgate.net/figure/Operating-conditions-for-the-FEMTO-ST-bearing-dataset_tbl1_383937808)

### PHM 2009 Gearbox Dataset:
- [PHM Society Public Datasets](https://phmsociety.org/public-data-sets/)
- [NASA DASHlink](https://c3.ndc.nasa.gov/dashlink/resources/997/)
- [2026 Multi-Label Framework](https://link.springer.com/article/10.1007/s44245-026-00229-4)

### OEDI Dataset:
- [OEDI Wind Turbine Gearbox](https://data.openei.org/submissions/738)
- [OEDI Documentation PDF](https://data.openei.org/files/738/Vibration%20Condition%20Monitoring%20Benchmarking%20Datasets.pdf)

### MCC5-THU Dataset:
- [MCC5-THU GitHub](https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets)
- [Multi-Mode Fault Diagnosis Paper](https://www.sciencedirect.com/science/article/pii/S2352340924004220)
- [Time-Varying Conditions Analysis 2025](https://arxiv.org/pdf/2506.17740)

### Mendeley Varying Speed Dataset:
- [Subset 1](https://data.mendeley.com/datasets/vxkj334rzv/7)
- [Subset 2](https://data.mendeley.com/datasets/x3vhp8t6hg/7)
- [Subset 3](https://data.mendeley.com/datasets/j8d8pfkvj2/7)
- [Time-Varying Speed Data](https://data.mendeley.com/datasets/v43hmbwxpm/2)

---

**Conclusion:** Only MCC5-THU and Mendeley datasets contain explicit speed transitions suitable for action-conditioned learning. However, neither has continuous degradation tracking or commanded control setpoints. A hybrid approach using multiple datasets or synthetic augmentation is recommended.
