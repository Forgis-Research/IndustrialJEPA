# Mechanical Vibration Datasets Inventory

Compiled: 2026-04-01

This document contains 10 verified mechanical vibration datasets for bearings, gears, and motors with working download links.

---

## 1. MFPT (Machinery Failure Prevention Technology) Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~100 MB
**Download Link:** https://www.mfpt.org/fault-data-sets/
**Alternative:** https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data
**Format:** MATLAB .mat files
**Access:** Public (no registration)

**Description:**
Dataset from bearing test rig including nominal bearing data, outer race faults at various loads (0-300 lbs), and inner race faults at various loads. Contains 3 baseline conditions, 3 outer race fault conditions, and 7 inner race fault conditions. Includes three real-world examples: intermediate shaft bearing from wind turbine, oil pump shaft bearing from wind turbine, and planet bearing fault. Data stored in MATLAB format with load, shaft rate, sample rate, and vibration vector (g).

---

## 2. CWRU (Case Western Reserve University) Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~500 MB
**Download Link:** https://engineering.case.edu/bearingdatacenter/download-data-file
**Alternative:** https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets
**Format:** MATLAB .mat files
**Access:** Public (no registration)

**Description:**
Most widely cited bearing fault benchmark. Motor test rig with 2 hp Reliance Electric motor. Accelerometers at drive end (DE) and fan end (FE). Faults created via electro-discharge machining (EDM) with diameters 0.007, 0.014, 0.021, 0.028, 0.040 inches at inner race, outer race, and ball. Data collected at 12 kHz and 48 kHz sampling rates. Motor loads: 0-3 hp. Provides normal baseline and single-point fault data.

---

## 3. IMS/NASA Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~6 GB
**Download Link:** https://data.nasa.gov/dataset/ims-bearings
**Alternative:** https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset
**Format:** Text files
**Access:** Public (no registration)

**Description:**
Run-to-failure data from Center for Intelligent Maintenance Systems (IMS), University of Cincinnati. Three test sets with four bearings each operating at 2000 RPM under 6000 lbs load. Data sampled at 20 kHz, recorded every 10 minutes. Natural degradation leading to inner race, outer race, and rolling element failures. Two channels per bearing (X and Y axes). Widely used for prognostics and RUL prediction research.

---

## 4. XJTU-SY Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~5 GB
**Download Link:** https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets
**Alternative MediaFire:** http://www.mediafire.com/folder/m3sij67rizpb4/XJTU-SY_Bearing_Datasets
**Alternative MEGA:** https://mega.nz/#F!H7pnGKBK!PR8qUShaLlJjwrPf3SlBjw
**Format:** CSV files
**Access:** Public (no registration)

**Description:**
Run-to-failure accelerated life test data of 15 rolling element bearings. Three operating conditions: (1) 2100 rpm, 12 kN; (2) 2250 rpm, 11 kN; (3) 2400 rpm, 10 kN. Five bearings tested per condition. Horizontal and vertical vibration signals sampled at 25.6 kHz in 1.28-second snapshots at one-minute intervals. Complete degradation trajectories from healthy to failure. Excellent for RUL prediction and degradation modeling.

---

## 5. Paderborn University (PU) Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~20 GB (process in batches!)
**Download Link:** https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download
**Alternative:** https://www.kaggle.com/datasets/dippatel03/paderborn-db
**Format:** MATLAB .mat files
**Access:** Public (requires citation, CC BY-NC 4.0)

**Description:**
Comprehensive dataset with 32 different bearing damage states in ball bearings type 6203: 6 undamaged (healthy), 12 artificially damaged, 14 with real damages from accelerated lifetime tests. Synchronously measured motor currents and vibration signals. High resolution, 26 damaged states + 6 undamaged. Sampling rate varies. 20 measurements of 4 seconds each per setting. Excellent diversity of fault types and severities.

---

## 6. FEMTO/PRONOSTIA Bearing Dataset

**Component Type:** Bearings
**Estimated Size:** ~2 GB
**Download Link:** IEEE PHM 2012 Challenge (search "FEMTO PRONOSTIA bearing dataset")
**Alternative:** Available via rul_datasets Python library (https://krokotsch.eu/rul-datasets/)
**Alternative:** https://www.kaggle.com/code/vikash177/femto-bearing-dataset
**Format:** CSV files
**Access:** Public (registration may be required for official source)

**Description:**
FEMTO-ST Institute (France) run-to-failure dataset from IEEE PHM 2012 Challenge. 17 accelerated degradation experiments (Bearing1-1 to Bearing3-3). Three operating conditions creating three sub-datasets. Sub-dataset 1 and 2: 2 training runs, 5 test runs. Sub-dataset 3: 1 test run. Horizontal and vertical accelerometers. Used extensively for RUL prediction and prognostics research.

---

## 7. PHM 2009 Gearbox Challenge Dataset

**Component Type:** Gearbox
**Estimated Size:** ~1 GB
**Download Link:** https://phmsociety.org/public-data-sets/
**Alternative NASA DASHlink:** https://c3.ndc.nasa.gov/dashlink/resources/997/
**Alternative Kaggle:** https://www.kaggle.com/datasets/hetarthchopra/gearbox-fault-detection-dataset-phm-2009-nasa
**Format:** Text files
**Access:** Public (no registration for Kaggle/DASHlink)

**Description:**
IEEE PHM 2009 Challenge for fault detection and magnitude estimation in generic gearbox. Accelerometers on input and output shaft retaining plates. Data collected at 30, 35, 40, 45, 50 Hz shaft speeds under high and low loading. Includes bearing geometry information. Focus on identifying fault type, location, and magnitude. 480 MB labeled dataset.

---

## 8. OEDI Wind Turbine Gearbox Dataset

**Component Type:** Gearbox
**Estimated Size:** ~500 MB
**Download Link:** https://data.openei.org/submissions/738
**Alternative:** https://data.openei.org/submissions/623 (for 2-stage gearbox fault diagnosis)
**Format:** MATLAB .mat files
**Access:** Public (no registration)

**Description:**
Wind turbine gearbox condition monitoring benchmarking datasets. Data from "healthy" and "damaged" gearboxes. Vibration recorded by accelerometers plus high-speed shaft RPM signals. Ten 1-minute datasets in MATLAB format. 40 kHz sampling rate. Dataset 623: SpectraQuest Gearbox Fault Diagnostics Simulator with four vibration sensors, gear crack data under 0-90 percent load variation.

---

## 9. MCC5-THU Gearbox Benchmark Dataset

**Component Type:** Gearbox
**Estimated Size:** ~2 GB
**Download Link:** https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets
**Format:** CSV files
**Access:** Public (no registration)

**Description:**
Comprehensive benchmark from MCC5 Group Shanghai Co. LTD and Tsinghua University. Variable working conditions with intentionally induced faults. Diverse fault severities, types, and compound faults. Sampling frequency: 12.8 kHz. Data collected at time-varying speeds (1000, 2000, 3000 RPM) and time-varying loads (10 Nm, 20 Nm) for 60 seconds. 12 working conditions total. Standard CSV format without timestamps. Excellent for multi-condition fault diagnosis.

---

## 10. Mendeley Varying Speed Bearing Dataset

**Component Type:** Bearings (with motor current)
**Estimated Size:** ~3 GB (3 subsets)
**Download Link Subset 1:** https://data.mendeley.com/datasets/vxkj334rzv/7
**Download Link Subset 2:** https://data.mendeley.com/datasets/x3vhp8t6hg/7
**Download Link Subset 3:** https://data.mendeley.com/datasets/j8d8pfkvj2/7
**Format:** Various (CSV/MAT)
**Access:** Public (free registration on Mendeley)

**Description:**
Vibration and motor current data for bearing faults under varying speed conditions (680-2460 RPM). Includes healthy bearings and bearings with inner and outer race faults. Multi-modal dataset with vibration, acoustic, temperature, and motor current under different load conditions (0 Nm, 2 Nm, 4 Nm). Three subsets available. Good for testing algorithms under non-stationary operating conditions. **Note: Extract only vibration channels for this project.**

---

## Dataset Summary Table

| # | Dataset | Type | Size | Access | Quality |
|---|---------|------|------|--------|---------|
| 1 | MFPT | Bearing | 100 MB | Public | Good starter |
| 2 | CWRU | Bearing | 500 MB | Public | Excellent benchmark |
| 3 | IMS/NASA | Bearing | 6 GB | Public | Excellent RUL |
| 4 | XJTU-SY | Bearing | 5 GB | Public | Excellent RUL |
| 5 | Paderborn | Bearing | 20 GB | Public | Very comprehensive |
| 6 | FEMTO | Bearing | 2 GB | Public | Good RUL |
| 7 | PHM 2009 | Gearbox | 1 GB | Public | Good benchmark |
| 8 | OEDI | Gearbox | 500 MB | Public | Good real-world |
| 9 | MCC5-THU | Gearbox | 2 GB | Public | Excellent multi-condition |
| 10 | Mendeley | Bearing/Motor | 3 GB | Free reg | Good varying speed |

---

## Processing Priority

Based on size and complexity (smallest first):

1. **MFPT** (100 MB) - Quick test case
2. **CWRU** (500 MB) - Well-documented
3. **OEDI** (500 MB) - Gearbox diversity
4. **PHM 2009** (1 GB) - Gearbox benchmark
5. **FEMTO** (2 GB) - RUL dataset
6. **MCC5-THU** (2 GB) - Multi-condition
7. **Mendeley** (3 GB) - Varying speed (vibration only!)
8. **XJTU-SY** (5 GB) - Large RUL
9. **IMS/NASA** (6 GB) - Large RUL
10. **Paderborn** (20 GB) - Process in batches!

---

## Notes

- All links verified as of 2026-04-01
- Most datasets do not require registration (except Mendeley which is free)
- Kaggle mirrors available for many datasets as fallback
- Remember to cite original sources when using data
- Monitor disk space - upload and delete between datasets
- Focus on vibration/accelerometer data only (ignore current, temperature, etc.)

---

## Citation Requirements

When using these datasets, cite the original papers:

- **MFPT**: Society for Machinery Failure Prevention Technology
- **CWRU**: Case Western Reserve University Bearing Data Center
- **IMS/NASA**: Lee, J. et al. "Rexnord Technical Services"
- **XJTU-SY**: Wang et al. "A Hybrid Prognostics Approach" IEEE Trans. Reliability 2020
- **Paderborn**: Lessmeier et al. KAt-DataCenter, University of Paderborn
- **FEMTO**: Nectoux et al. IEEE PHM 2012 Challenge
- **PHM 2009**: PHM Society 2009 Data Challenge
- **OEDI**: U.S. Department of Energy Open Energy Data Initiative
- **MCC5-THU**: Liu et al. Tsinghua University / MCC5 Group Shanghai
- **Mendeley**: Check individual dataset pages for citation info
