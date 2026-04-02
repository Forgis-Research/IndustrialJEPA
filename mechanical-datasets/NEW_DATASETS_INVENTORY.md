# Exhaustive Rotating Machinery Vibration Dataset Inventory

**Compiled:** 2026-03-31
**Purpose:** Identify ALL publicly available vibration datasets for training a cross-component transferable fault physics model.
**Scope:** Bearings, gears, shafts, couplings, motors/generators, fans/blowers, complete drivetrains.
**Exclusions:** Pumps, valves, gas/steam turbines.

---

## ALREADY INCLUDED (skip these)

CWRU, MFPT, IMS, XJTU-SY, Paderborn, FEMTO, Mendeley Varying Speed, OEDI, PHM 2009, MCC5-THU

---

## NEW DATASETS: DRIVETRAINS / MULTI-COMPONENT

### 1. SEU Drivetrain Dataset (Southeast University, China)

| Field | Details |
|-------|---------|
| **Name** | SEU Drivetrain Dynamics Simulator Dataset |
| **Institution** | Southeast University, Nanjing, China |
| **URL** | https://github.com/cathysiyu/Mechanical-datasets |
| **Alt URL** | https://github.com/Yxz3930/SEU-datasets |
| **Download** | Direct (GitHub, no registration) |
| **Size** | ~400 MB |
| **Components** | Bearings + Gears (planetary + parallel gearbox) + Motor |
| **Sensors** | Vibration (8 channels: motor, planetary gearbox x/y/z, parallel gearbox x/y/z), motor torque |
| **Sampling rate** | 12 kHz |
| **Files/samples** | 2 subdatasets (bearing + gear), 2 working conditions, ~1000 samples/class |
| **Fault types** | Bearing: ball, inner race, outer race, combined, healthy. Gear: chipped, missing, root, surface, healthy |
| **Operating conditions** | 2 speed-load configs: 20 Hz-0V, 30 Hz-2V |
| **Run-to-failure** | No |
| **License** | Public (academic use) |
| **Unique value** | **CRITICAL: Full drivetrain with BOTH bearing and gear faults from the same rig.** 8 channels spanning motor through gearbox. Perfect for cross-component transfer learning. |

### 2. MAFAULDA (Machinery Fault Database, UFRJ Brazil)

| Field | Details |
|-------|---------|
| **Name** | MAFAULDA - Machinery Fault Database |
| **Institution** | Federal University of Rio de Janeiro (UFRJ), Brazil |
| **URL** | https://www02.smt.ufrj.br/~offshore/mfs/page_01.html |
| **Alt URL** | https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset |
| **Alt URL** | https://www.kaggle.com/datasets/vuxuancu/mafaulda-full |
| **Download** | Direct download (.tgz/.zip), 7 packages totaling ~13 GB compressed |
| **Size** | ~13 GB compressed, ~26 GB uncompressed |
| **Components** | Shaft (imbalance, misalignment) + Bearings (inner, outer race) |
| **Sensors** | Vibration (triaxial underhang + triaxial overhang = 6 accel channels), tachometer, microphone (8 total) |
| **Sampling rate** | 50 kHz |
| **Files/samples** | 1,951 multivariate time-series, each 5 seconds |
| **Fault types** | Normal, imbalance (7 severity levels: 6-35g), horizontal misalignment (4 levels: 0.5-2.0mm), vertical misalignment (6 levels: 0.51-1.9mm), inner bearing fault, outer bearing fault |
| **Operating conditions** | Multiple speeds via SpectraQuest MFS-ABVT |
| **Run-to-failure** | No |
| **License** | Public |
| **Unique value** | **CRITICAL: Shaft faults (imbalance + misalignment) with multiple severity levels, PLUS bearing faults. Compound faults. Acoustic channel. High sampling rate. This fills the shaft/coupling gap.** |

### 3. COMFAULDA (Composed Fault Dataset)

| Field | Details |
|-------|---------|
| **Name** | COMFAULDA - Composed Fault Dataset |
| **Institution** | Federal University of Rio de Janeiro (UFRJ), Brazil |
| **URL** | https://ieee-dataport.org/documents/composed-fault-dataset-comfaulda (DOI: 10.21227/89ye-ap56) |
| **Download** | IEEE DataPort (subscription may be required) |
| **Size** | ~2 GB (estimated) |
| **Components** | Shaft (imbalance, misalignment) + compound combinations |
| **Sensors** | Vibration (capacitive accel x/z/y + piezo accel x/z/y), tachometer (7 channels) |
| **Sampling rate** | 50 kHz |
| **Files/samples** | 2,162 examples |
| **Fault types** | Normal, unbalance, horizontal misalignment, vertical misalignment, unbalance+horizontal, unbalance+vertical, vertical+horizontal |
| **Operating conditions** | 7 speed values |
| **Run-to-failure** | No |
| **License** | IEEE DataPort |
| **Unique value** | **Compound/combined fault scenarios. Extends MAFAULDA with fault combinations. Critical for real-world scenarios where multiple faults coexist.** |

### 4. NLN-EMP (Dutch Navy E-Motor Pump Dataset, 4TU.ResearchData)

| Field | Details |
|-------|---------|
| **Name** | Motor Current and Vibration Monitoring Dataset for E-motor-driven Centrifugal Pump |
| **Institution** | Royal Netherlands Navy / University of Twente |
| **URL** | https://data.4tu.nl/datasets/2b61183e-c14f-4131-829b-cc4822c369d0 |
| **DOI** | 10.4121/2b61183e-c14f-4131-829b-cc4822c369d0.v4 |
| **Download** | Direct (4TU.ResearchData, free registration) |
| **Size** | Large (multi-GB, details on page) |
| **Components** | Motor + Coupling + Shaft + Bearings (complete drivetrain) |
| **Sensors** | Vibration, motor current, voltage |
| **Sampling rate** | Varies |
| **Files/samples** | Extensive (multiple fault types x severity x speed) |
| **Fault types** | Bearing defects, loose foot, misalignment, unbalance, coupling degradation, stator winding short, broken rotor bar, soft foot, bent shaft |
| **Operating conditions** | Multiple motor speeds |
| **Run-to-failure** | No |
| **License** | 4TU (check terms) |
| **Unique value** | **GOLD MINE: 11+ fault types spanning motor, coupling, shaft, and bearing. Near-perfect labeling. Fills virtually every component gap. NOTE: Includes pump/impeller data too -- extract only motor-coupling-shaft-bearing channels.** |

### 5. Mendeley Gearbox Variable Conditions Testing

| Field | Details |
|-------|---------|
| **Name** | Gearbox Variable Conditions Testing |
| **Institution** | Not specified (academic) |
| **URL** | https://data.mendeley.com/datasets/whj3wxhw8j/1 |
| **Download** | Direct (Mendeley, free registration) |
| **Size** | Multi-GB (estimated) |
| **Components** | Motor + Test Gearbox (2-stage) + Load Gearbox (2-stage) + Bearings |
| **Sensors** | 8 accelerometers (motor, gearbox inputs/outputs, load motor), tachometer, 2 encoders, torque sensor, 2 current transformers |
| **Sampling rate** | Not specified (likely 10-50 kHz) |
| **Files/samples** | Multiple scenarios |
| **Fault types** | Baseline (healthy), broken gear tooth, outer race bearing failure |
| **Operating conditions** | Variable speed and load via AC motor + torque control |
| **Run-to-failure** | No |
| **License** | Mendeley (CC) |
| **Unique value** | **Complete instrumented drivetrain: motor through dual gearbox to load. Multi-encoder speed measurement. Both gear and bearing faults on same rig. Excellent for cross-component work.** |

---

## NEW DATASETS: GEARS / GEARBOXES

### 6. Mendeley Multi-Mode Gearbox Fault Dataset

| Field | Details |
|-------|---------|
| **Name** | Multi-mode Fault Diagnosis Datasets of Gearbox Under Variable Working Conditions |
| **Institution** | Tsinghua University |
| **URL** | https://data.mendeley.com/datasets/p92gj2732w/2 |
| **DOI** | 10.17632/p92gj2732w.2 |
| **Download** | Direct (Mendeley, free registration) |
| **Size** | Multi-GB |
| **Components** | Gears + Bearings (compound faults) |
| **Sensors** | Vibration (two 3-axis accelerometers: motor + intermediate shaft), speed, torque |
| **Sampling rate** | 12.8 kHz |
| **Files/samples** | Extensive (multiple fault types x severity x operating conditions) |
| **Fault types** | Gear: health, missing teeth, wear, pitting, root cracks, broken teeth. Bearing-gear compound faults. Multiple severity levels. |
| **Operating conditions** | Variable speed (time-varying) and variable load |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **Compound gear+bearing faults under truly variable (time-varying) speed. Multiple gear fault types with severity levels. From same group as MCC5-THU but different rig/faults.** |

### 7. Figshare Planetary Gearbox (University of Pretoria)

| Field | Details |
|-------|---------|
| **Name** | Planetary Gearbox Vibration Data |
| **Institution** | University of Pretoria, South Africa |
| **URL** | https://figshare.com/articles/dataset/Planetary_gearbox_vibration_data/13476525/2 |
| **Download** | Direct (Figshare) |
| **Size** | ~15.8 GB |
| **Components** | Planetary gearbox |
| **Sensors** | Accelerometers on gearbox housing |
| **Sampling rate** | Not specified |
| **Files/samples** | Large (see README.md in download) |
| **Fault types** | Varying degrees of seeded damage |
| **Operating conditions** | Multiple conditions |
| **Run-to-failure** | Possibly (thesis on prognostics) |
| **License** | CC (Figshare) |
| **Unique value** | **Planetary gearbox specifically. Most gearbox datasets are parallel/helical. Planetary gears have fundamentally different vibration signatures (modulated by planet carrier). Critical for diverse gear coverage.** |

### 8. UConn Gearbox Dataset (University of Connecticut)

| Field | Details |
|-------|---------|
| **Name** | UConn Gearbox Fault Diagnosis Dataset |
| **Institution** | University of Connecticut, DSCL Lab |
| **URL** | Contact: https://dscl.uconn.edu/gearbox-diagnosis/ |
| **Download** | **Contact lab directly** (no public download link) |
| **Size** | ~50 MB (estimated, small) |
| **Components** | Two-stage gearbox (spur gears) |
| **Sensors** | Accelerometer |
| **Sampling rate** | 20 kHz |
| **Files/samples** | 936 samples (104 per class) |
| **Fault types** | 9 categories: healthy, missing tooth, root crack, spalling, tip chipping (5 severity levels) |
| **Operating conditions** | Constant |
| **Run-to-failure** | No |
| **License** | Contact required |
| **Unique value** | **Fine-grained gear fault classification with 5 severity levels for tip chipping. Small but well-structured.** |

### 9. Figshare Gearbox Fault Diagnosis Dataset

| Field | Details |
|-------|---------|
| **Name** | Vibration-based Gearbox Fault Diagnosis Dataset |
| **Institution** | Not specified |
| **URL** | https://figshare.com/articles/dataset/Vibration-based_Gearbox_Fault_Diagnosis_Dataset/28812476 |
| **DOI** | 10.6084/m9.figshare.28812476 |
| **Download** | Direct (Figshare) |
| **Size** | Not specified |
| **Components** | Gearbox |
| **Sensors** | Vibration |
| **Sampling rate** | Not specified |
| **Files/samples** | Not specified |
| **Fault types** | Gearbox faults (details in download) |
| **Operating conditions** | Not specified |
| **Run-to-failure** | No |
| **License** | CC (Figshare) |
| **Unique value** | Recent (2025) gearbox dataset. Supporting PLOS ONE publication. |

---

## NEW DATASETS: MOTORS / GENERATORS

### 10. Mendeley Multi-Mode Motor Fault Dataset (Tsinghua)

| Field | Details |
|-------|---------|
| **Name** | Multi-mode Fault Diagnosis Datasets of Three-phase Asynchronous Motor Under Variable Working Conditions |
| **Institution** | Tsinghua University |
| **URL** | https://data.mendeley.com/datasets/6s3dggj9mw/1 |
| **Alt URL** | https://ieee-dataport.org/documents/multi-mode-fault-diagnosis-datasets-three-phase-asynchronous-motor-under-variable-working |
| **Download** | Direct (Mendeley, free registration) |
| **Size** | Multi-GB |
| **Components** | Motor (rotor, stator, bearings) |
| **Sensors** | Triaxial vibration, three-phase currents, torque, key-phase |
| **Sampling rate** | Not specified |
| **Files/samples** | Multiple fault scenarios |
| **Fault types** | Rotor unbalance, stator winding short circuits, bearing faults, AND mechanical-electrical compound faults |
| **Operating conditions** | Variable speed and load (steady + transitional) |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **CRITICAL: Motor-specific faults (stator winding + rotor unbalance) combined with bearing faults. Compound mechanical-electrical faults. From same Tsinghua group as gearbox dataset #6 -- can potentially combine for cross-component studies.** |

### 11. Figshare Motor Fault Detection Data (3-Phase Induction Motor)

| Field | Details |
|-------|---------|
| **Name** | Comprehensive Fault Diagnosis of Three-Phase Induction Motors (Motor Fault Detection Data) |
| **Institution** | Not specified (Nature Scientific Data 2025) |
| **URL** | https://figshare.com/articles/dataset/MOTOR_FAULT_DETECTION_DATA/27216219 |
| **Download** | Direct (Figshare) |
| **Size** | ~1 GB (10 CSV files) |
| **Components** | Motor (0.2 kW squirrel cage induction motor) |
| **Sensors** | Vibration, voltage, current (all synchronized) |
| **Sampling rate** | 50 kHz |
| **Files/samples** | 10 CSV files covering different operational states |
| **Fault types** | Phase removal, mechanical misalignments |
| **Operating conditions** | Multiple operational states |
| **Run-to-failure** | No |
| **License** | CC (Figshare) |
| **Unique value** | **50 kHz synchronized multi-sensor (vibration + voltage + current). Published in Nature Scientific Data 2025 -- high quality. Small motor but well-documented.** |

### 12. IEEE DataPort Broken Rotor Bar Dataset

| Field | Details |
|-------|---------|
| **Name** | Experimental Database for Detecting and Diagnosing Rotor Broken Bar in Three-Phase Induction Motor |
| **Institution** | University of Sao Paulo, Brazil |
| **URL** | https://ieee-dataport.org/open-access/experimental-database-detecting-and-diagnosing-rotor-broken-bar-three-phase-induction |
| **DOI** | 10.21227/fmnm-bn95 |
| **Download** | Direct (IEEE DataPort, **open access**) |
| **Size** | ~2 GB (estimated) |
| **Components** | Motor (rotor fault) |
| **Sensors** | 5 axial accelerometers (drive end + non-drive end), current, voltage |
| **Sampling rate** | Sensors freq range 5-2000 Hz; 18s recordings |
| **Files/samples** | 10 repetitions x multiple conditions |
| **Fault types** | Broken rotor bars (multiple severity levels in 34-bar rotor) |
| **Operating conditions** | Multiple mechanical loads, transient + steady state |
| **Run-to-failure** | No |
| **License** | Open access |
| **Unique value** | **Open access on IEEE DataPort. Rotor bar faults with severity progression. 5 accelerometers positioned at DE and NDE. Transient-to-steady-state recordings.** |

### 13. Mendeley PMSM Stator Fault Dataset (KAIST)

| Field | Details |
|-------|---------|
| **Name** | Vibration and Current Dataset of Three-Phase Permanent Magnet Synchronous Motors with Stator Faults |
| **Institution** | KAIST, South Korea |
| **URL** | https://data.mendeley.com/datasets/rgn5brrgrn/5 |
| **DOI** | 10.17632/rgn5brrgrn.5 |
| **Download** | Direct (Mendeley, free registration) |
| **Size** | ~2 GB (estimated) |
| **Components** | PMSM Motors (3 different powers) |
| **Sensors** | Vibration (accelerometer), 3-phase current (CT) |
| **Sampling rate** | Not specified (NI DAQ) |
| **Files/samples** | 3 motors x 16 stator faults each = 48+ conditions in TDMS format |
| **Fault types** | 8 inter-coil short circuit faults, 8 inter-turn short circuit faults per motor |
| **Operating conditions** | 3000 RPM rated speed, 15% rated load |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **PMSM-specific stator faults. Three different motor capacities (1.0, 1.5, 3.0 kW). Fine-grained severity levels. Bridges induction motor datasets with permanent magnet motors.** |

### 14. KAIST Industrial-Scale Motor Fault Dataset (2025)

| Field | Details |
|-------|---------|
| **Name** | Vibration, Current, Torque, RPM Dataset for Multiple Fault Conditions in Industrial-Scale Electric Motors |
| **Institution** | KAIST, South Korea |
| **URL** | https://www.sciencedirect.com/science/article/pii/S235234092500678X (links to Mendeley) |
| **Download** | Mendeley Data (check paper for exact link) |
| **Size** | >60 GB (raw signals) |
| **Components** | Motor (AC motors) + HVAC air handling unit |
| **Sensors** | 3-phase current (100 kHz), vibration (25.6 kHz), torque (25.6 kHz), RPM (100 kHz) |
| **Sampling rate** | 25.6 kHz vibration, 100 kHz current/RPM |
| **Files/samples** | Extensive; each scenario 120-300 seconds |
| **Fault types** | Coil winding faults, inter-phase short circuits, misalignment, rolling-element bearing faults, journal bearing faults, belt loosening (HVAC) |
| **Operating conditions** | **Randomized speed fluctuations (6% and 16%)**, variable load, variable frequency drive |
| **Run-to-failure** | No |
| **License** | CC (Mendeley/Data in Brief) |
| **Unique value** | **EXCEPTIONAL: Industrial-scale motors with RANDOMIZED speed variations (not discrete steps). Multi-fault, multi-severity. 60+ GB. Fills the "real-world variable speed" gap. Published 2025.** |

### 15. GitHub ITSC Dataset (Induction Motor Inter-Turn Short Circuit)

| Field | Details |
|-------|---------|
| **Name** | ITSC - Inter-Turn Short Circuit Fault Dataset |
| **Institution** | University of Guanajuato, Mexico |
| **URL** | https://github.com/ibarram/ITSC |
| **Download** | Direct (GitHub) |
| **Size** | ~200 MB (estimated) |
| **Components** | Motor (stator winding) |
| **Sensors** | Three-phase current |
| **Sampling rate** | Not specified |
| **Files/samples** | 13 categories x 5 repetitions = 65 recordings |
| **Fault types** | 12 inter-turn short-circuit faults per phase at 10%, 20%, 30%, 40% severity + healthy |
| **Operating conditions** | Multiple loads |
| **Run-to-failure** | No |
| **License** | Public (GitHub) |
| **Unique value** | **Current-only (no vibration). BUT useful for multi-modal learning if combined with vibration datasets. Fine-grained stator fault severity. Small, easy to use.** |

---

## NEW DATASETS: SHAFT / COUPLING / ROTOR FAULTS

### 16. Figshare Single and Double Faults Dataset (University of Arkansas)

| Field | Details |
|-------|---------|
| **Name** | Dataset of Single and Double Faults Scenarios Using Vibration Signals from a Rotary Machine |
| **Institution** | University of Arkansas |
| **URL** | https://figshare.com/articles/dataset/Single_and_Double_Fault_Scenarios_for_a_Rotary_Machine/22693120 |
| **DOI** | 10.6084/m9.figshare.22693120.v1 |
| **Download** | Direct (Figshare) |
| **Size** | ~500 MB (estimated) |
| **Components** | Shaft (bent) + Bearings (inner, outer race) |
| **Sensors** | Accelerometers on support bearings |
| **Sampling rate** | 6,400 Hz |
| **Files/samples** | 39 scenarios (38 fault + 1 no-fault) at 3 operating frequencies, ~10s each |
| **Fault types** | Bearing inner race, bearing outer race, bent shaft, AND all pairwise doubles: IR+OR, IR+bent, OR+bent |
| **Operating conditions** | 3 different operating frequencies |
| **Run-to-failure** | No |
| **License** | CC (Figshare) |
| **Unique value** | **Bent shaft fault explicitly included. Compound double-fault scenarios. Small and focused -- good for shaft fault representation.** |

### 17. IEEE DataPort Five Typical Rotor Fault Types

| Field | Details |
|-------|---------|
| **Name** | Example Dataset of Five Typical Fault Types in Industrial Rotor Systems |
| **Institution** | Not specified |
| **URL** | https://ieee-dataport.org/documents/example-dataset-five-typical-fault-types-industrial-rotor-systems |
| **DOI** | 10.21227/1p64-7m10 |
| **Download** | IEEE DataPort (**subscription required**) |
| **Size** | 745 MB |
| **Components** | Industrial rotors (turbine, compressor) |
| **Sensors** | Eddy-current displacement sensors |
| **Sampling rate** | 1,024 points/file |
| **Files/samples** | 13,820 to 35,305 files per dataset, 5 datasets total |
| **Fault types** | Oil film whirl, rotor-stator rubbing, compressor surge, shaft-related faults |
| **Operating conditions** | Full operational cycle: startup, normal, fault |
| **Run-to-failure** | Partial (captures fault development) |
| **License** | IEEE DataPort subscription |
| **Unique value** | **Real industrial rotor system data (not lab). Displacement sensors (not accelerometers). Covers startup transients. Published Feb 2026.** |

### 18. VBL-VA001 Lab-Scale Vibration Dataset (ITS Indonesia)

| Field | Details |
|-------|---------|
| **Name** | VBL-VA001: Lab-scale Vibration Analysis Dataset |
| **Institution** | Institut Teknologi Sepuluh Nopember (ITS), Indonesia |
| **URL** | https://zenodo.org/records/7006575 |
| **Alt URL** | https://github.com/bagustris/VBL-VA001 |
| **Download** | Direct (Zenodo, no registration) |
| **Size** | ~2 GB |
| **Components** | Shaft (misalignment, unbalance) + Bearings |
| **Sensors** | Triaxial vibration (enDAQ LOG-0002-100G sensor) |
| **Sampling rate** | 20 kHz |
| **Files/samples** | 4,000 CSV files (1,000 per condition), 5 seconds each |
| **Fault types** | Normal, bearing fault, misalignment, unbalance |
| **Operating conditions** | Constant (Panasonic GP-129JXK motor) |
| **Run-to-failure** | No |
| **License** | CC-BY-4.0 |
| **Unique value** | **Clean 4-class dataset with shaft faults (misalignment + unbalance). Triaxial. Open license. Good baseline for shaft fault detection. Well-documented with paper and baseline code.** |

---

## NEW DATASETS: BEARINGS (supplementary to existing collection)

### 19. HUST Bearing Dataset (Huazhong University)

| Field | Details |
|-------|---------|
| **Name** | HUST Bearing Dataset |
| **Institution** | Huazhong University of Science and Technology, China |
| **URL** | https://github.com/CHAOZHAO-1/HUSTbearing-dataset |
| **Alt URL** | https://data.mendeley.com/datasets/cbv7jyx4p9/1 |
| **Download** | Direct (GitHub/Mendeley) |
| **Size** | ~5 GB (estimated) |
| **Components** | Bearings (5 types: 6204, 6205, 6206, 6207, 6208) |
| **Sensors** | Accelerometer (PCB 325C33, vertical) |
| **Sampling rate** | 51,200 Hz |
| **Files/samples** | 99 raw vibration signals, 10s each |
| **Fault types** | 6 types: inner crack, outer crack, ball crack, and 3 pairwise combinations |
| **Operating conditions** | 3 loads (0W, 200W, 400W), 11 operating conditions, stable + time-varying speed |
| **Run-to-failure** | No |
| **License** | Public |
| **Unique value** | **Multiple bearing TYPES (5 sizes), combination faults, time-varying speed. 51.2 kHz high resolution. Great for domain generalization across bearing types.** |

### 20. Multi-Domain Bearing Compound Fault Dataset (University of Seoul)

| Field | Details |
|-------|---------|
| **Name** | Multi-domain Vibration Dataset with Various Bearing Types Under Compound Machine Fault Scenarios |
| **Institution** | University of Seoul, South Korea |
| **URL Subset 1** | (deep groove ball bearing - check paper for Mendeley link) |
| **URL Subset 2** | https://data.mendeley.com/datasets/7trwzz77xh/1 (cylindrical roller) |
| **URL Subset 3** | https://data.mendeley.com/datasets/2cygy6y4rk/1 (tapered roller) |
| **Download** | Direct (Mendeley, free registration) |
| **Size** | Multi-GB (3 subsets) |
| **Components** | Bearings (3 types) + Shaft (looseness, unbalance, misalignment) |
| **Sensors** | USB digital accelerometer (uniaxial) |
| **Sampling rate** | 8 kHz and 16 kHz |
| **Files/samples** | MAT files; 1,280,000 data points per file (160s @8kHz or 80s @16kHz) |
| **Fault types** | 3 single bearing faults (ball, IR, OR) + 7 single rotating component faults (looseness, unbalance 3 levels, misalignment 3 levels) + **21 compound faults** |
| **Operating conditions** | 6 rotating speeds |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **EXCEPTIONAL: Three bearing types (deep groove, cylindrical roller, tapered roller) + shaft faults + 21 compound fault combinations. Explicitly designed for compound fault research. Two sampling rates. Published 2024.** |

### 21. University of Ottawa Bearing Dataset (2023)

| Field | Details |
|-------|---------|
| **Name** | UORED-VAFCLS (University of Ottawa Rolling-element Dataset) |
| **Institution** | University of Ottawa, Canada |
| **URL** | https://data.mendeley.com/datasets/y2px5tg92h/1 |
| **Alt URL** | https://github.com/Mert-Sehri/University-of-Ottawa-Bearing-Dataset-UORED-VAFCLS- |
| **Download** | Direct (Mendeley/GitHub) |
| **Size** | ~3 GB (estimated) |
| **Components** | Bearings |
| **Sensors** | Accelerometer + microphone + load cell + hall effect + thermocouples |
| **Sampling rate** | 42,000 Hz |
| **Files/samples** | 60 distinct sets (20 bearings x 3 states: healthy/developing/faulty), 10s each |
| **Fault types** | Inner race, outer race, ball, cage faults (5 of each type) |
| **Operating conditions** | Constant 1,750 RPM, 400N load |
| **Run-to-failure** | Partial (3 stages: healthy -> developing -> faulty) |
| **License** | CC (Mendeley) |
| **Unique value** | **Cage faults (rare in datasets). Three health stages per bearing. Acoustic + vibration. 20 different physical bearings tested.** |

### 22. University of Ottawa Time-Varying Speed Bearing (2018)

| Field | Details |
|-------|---------|
| **Name** | Bearing Vibration Data under Time-varying Rotational Speed Conditions |
| **Institution** | University of Ottawa, Canada |
| **URL** | https://data.mendeley.com/datasets/v43hmbwxpm/2 |
| **Download** | Direct (Mendeley) |
| **Size** | ~1 GB (estimated, 36 files x 10s @200kHz) |
| **Components** | Bearings |
| **Sensors** | Vibration |
| **Sampling rate** | **200,000 Hz** |
| **Files/samples** | 36 datasets (3 trials x 4 speed profiles x 3 health conditions) |
| **Fault types** | Healthy, inner race defect, outer race defect |
| **Operating conditions** | Time-varying: increasing, decreasing, increase-then-decrease, decrease-then-increase |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **Ultra-high 200 kHz sampling rate. True time-varying speed profiles (ramps, not discrete steps). Perfect for action-conditioned prediction (speed ramp -> vibration state).** |

### 23. KAIST Ball Bearing Run-to-Failure Dataset

| Field | Details |
|-------|---------|
| **Name** | Ball Bearing Vibration and Temperature Run-to-Failure Dataset |
| **Institution** | KAIST, South Korea |
| **URL** | https://data.mendeley.com/datasets/5hcdd3tdvb/6 |
| **Download** | Direct (Mendeley) |
| **Size** | ~2 GB |
| **Components** | Bearings (NSK 6205) |
| **Sensors** | Vibration x/y (PCB 352C34), temperature (bearing + atmospheric) |
| **Sampling rate** | 25.6 kHz |
| **Files/samples** | 129 CSV files (hourly intervals), 78.125s each |
| **Fault types** | Natural degradation to failure |
| **Operating conditions** | Constant 1770-1780 RPM, axial 300kg + vertical 600kg load |
| **Run-to-failure** | **Yes** |
| **License** | CC (Mendeley) |
| **Unique value** | **Recent (2024) run-to-failure with temperature. Complements IMS/XJTU-SY/FEMTO with modern bearings and multi-modal (vibration + temperature).** |

### 24. University of Ferrara Run-to-Failure Dataset

| Field | Details |
|-------|---------|
| **Name** | Run-to-failure Vibration Dataset of Self-Aligning Double-Row Ball Bearings |
| **Institution** | University of Ferrara, Italy |
| **URL Part 1** | https://data.mendeley.com/datasets/htk59pp5wx/1 |
| **URL Part 2** | https://data.mendeley.com/datasets/zz8hpyx939/1 |
| **Download** | Direct (Mendeley) |
| **Size** | Multi-GB |
| **Components** | Bearings (self-aligning double-row, model 1205 ETN 9) |
| **Sensors** | Uniaxial accelerometer (radial) |
| **Sampling rate** | 25.6 kHz |
| **Files/samples** | 6 run-to-failure tests, acquired every 5 min for 5 seconds |
| **Fault types** | Natural degradation (threshold: 20g peak) |
| **Operating conditions** | Constant 40 Hz speed; loads: 3, 4, 4.7, 5 kN across tests |
| **Run-to-failure** | **Yes** |
| **License** | CC (Mendeley) |
| **Unique value** | **Self-aligning double-row bearings (unique type). Multiple load levels across tests. Clean degradation data from European university.** |

### 25. UNSW Bearing Run-to-Failure Dataset

| Field | Details |
|-------|---------|
| **Name** | Bearing Run-to-Failure Datasets of UNSW |
| **Institution** | University of New South Wales, Australia |
| **URL** | https://data.mendeley.com/datasets/h4df4mgrfb/3 |
| **Download** | Direct (Mendeley) |
| **Size** | Multi-GB |
| **Components** | Bearings |
| **Sensors** | Horizontal + vertical acceleration, 2 encoders (1024 pulses/rev), load cell, tacho |
| **Sampling rate** | Variable (see Fs in each file) |
| **Files/samples** | 4 run-to-failure tests (Test 1-4) |
| **Fault types** | Natural spall evolution |
| **Operating conditions** | Multiple shaft speeds per file |
| **Run-to-failure** | **Yes** |
| **License** | CC (Mendeley) |
| **Unique value** | **Natural spall evolution tracked over full life. Encoder signals enable precise order tracking. Load cell for force measurement. Published in MSSP 2022.** |

### 26. DLR Aerospace Bearing Spall Dataset

| Field | Details |
|-------|---------|
| **Name** | Vibration Data for Axial Ball Bearings and Spall Faults |
| **Institution** | German Aerospace Center (DLR) + Tekniker |
| **URL** | https://data.mendeley.com/datasets/chwhh9n3bf/2 |
| **Download** | Direct (Mendeley) |
| **Size** | ~500 MB (estimated, 28 files x 30s @25.6kHz) |
| **Components** | Bearings (aerospace-grade FAG QJ212TVP) |
| **Sensors** | Triaxial accelerometer (PCB 356A32) |
| **Sampling rate** | 25.6 kHz |
| **Files/samples** | 28 vibration time-series, 30s each |
| **Fault types** | Fatigue spalls of various sizes (width, height, depth) on inner/outer race |
| **Operating conditions** | Constant speed |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **Aerospace-grade bearings with precisely characterized spall geometry. Quantitative fault sizing (not just categories). DLR credibility.** |

### 27. JNU Bearing Dataset (Jiangnan University)

| Field | Details |
|-------|---------|
| **Name** | JNU Bearing Dataset |
| **Institution** | Jiangnan University, China |
| **URL** | https://github.com/ClarkGableWang/JNU-Bearing-Dataset |
| **Download** | Direct (GitHub) |
| **Size** | ~500 MB (estimated) |
| **Components** | Bearings (in centrifugal fan system) |
| **Sensors** | Accelerometer (PCB MA352A60, vertical) |
| **Sampling rate** | 50 kHz |
| **Files/samples** | 12 fault types, 20s recordings |
| **Fault types** | Healthy, inner ring fault, outer ring fault, rolling element fault |
| **Operating conditions** | 3 speeds: 600, 800, 1000 RPM |
| **Run-to-failure** | No |
| **License** | Public (GitHub) |
| **Unique value** | **Centrifugal fan context (not standard motor bench). 50 kHz high resolution. Good for fan/blower representation in training data.** |

### 28. SCA Industrial Pulp Mill Bearing Dataset

| Field | Details |
|-------|---------|
| **Name** | SCA Bearing Dataset |
| **Institution** | SCA (Swedish forestry company) |
| **URL** | https://data.mendeley.com/datasets/tdn96mkkpt/2 |
| **Download** | Direct (Mendeley) |
| **Size** | ~1 GB (estimated) |
| **Components** | Bearings (multiple types in industrial pulp mill) |
| **Sensors** | Vibration |
| **Sampling rate** | Not specified |
| **Files/samples** | 11 cases (data collected 2019-2022) |
| **Fault types** | **Naturally occurring faults** (not seeded) |
| **Operating conditions** | Real industrial operation |
| **Run-to-failure** | Partial (data before bearing replacement) |
| **License** | CC (Mendeley) |
| **Unique value** | **REAL INDUSTRIAL DATA with naturally occurring faults. Not a lab dataset. Bridges the lab-to-field gap. Multiple bearing types.** |

### 29. HIT-SM Cross-Domain Bearing Dataset (Harbin Institute)

| Field | Details |
|-------|---------|
| **Name** | HIT Bearing Datasets |
| **Institution** | Harbin Institute of Technology, China |
| **URL** | https://github.com/hitwzc/Bearing-datasets |
| **Download** | Direct (GitHub) |
| **Size** | ~1 GB (estimated) |
| **Components** | Bearings (from 2 different test rigs) |
| **Sensors** | Accelerometer |
| **Sampling rate** | 51.2 kHz |
| **Files/samples** | Multiple conditions from 2 rigs |
| **Fault types** | Same fault types across different rigs (inner, outer, ball) |
| **Operating conditions** | Multiple conditions |
| **Run-to-failure** | No |
| **License** | Public (GitHub) |
| **Unique value** | **Explicitly designed for cross-domain transfer learning. Same fault types measured on two different test rigs. Perfect for domain adaptation benchmarking.** |

### 30. Triaxial Bearing Vibration Dataset (Induction Motor)

| Field | Details |
|-------|---------|
| **Name** | Triaxial Bearing Vibration Dataset of Induction Motor under Varying Load Conditions |
| **Institution** | Not specified |
| **URL** | https://data.mendeley.com/datasets/fm6xzxnf36/2 |
| **Download** | Direct (Mendeley) |
| **Size** | ~500 MB (estimated) |
| **Components** | Bearings (in induction motor) |
| **Sensors** | MEMS triaxial accelerometer (NI myRIO) |
| **Sampling rate** | 10 kHz |
| **Files/samples** | 38 datasets |
| **Fault types** | Healthy, inner race, outer race (6 severity levels: 0.7-1.7mm) |
| **Operating conditions** | 3 load conditions: 100W, 200W, 300W |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **MEMS sensor (low-cost, industrial-relevant). 6 fault severity levels. Triaxial + varying load. Good for transfer to industrial IoT deployments.** |

### 31. Vishwakarma Institute Bearing Dataset (2025)

| Field | Details |
|-------|---------|
| **Name** | Rolling-Element Bearing Vibration Datasets under Varying Loads and Speeds |
| **Institution** | Vishwakarma Institute of Technology, Pune, India |
| **URL** | https://data.mendeley.com/datasets/zgzyxdnyv9/2 |
| **Download** | Direct (Mendeley) |
| **Size** | ~200 MB (estimated) |
| **Components** | Bearings (with gearbox at both motor ends) |
| **Sensors** | Vibration sensors |
| **Sampling rate** | 12,000 Hz |
| **Files/samples** | 50 files, 6s each |
| **Fault types** | Healthy, inner race fault, outer race fault |
| **Operating conditions** | 3 speeds (950, 1250, 1950 RPM), with/without load (1.25 kg) |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **Recent (2025). Varying speed AND load. Gearbox included in test rig. Indian institution -- geographic diversity.** |

---

## NEW DATASETS: FANS / BLOWERS / WIND TURBINE BLADES

### 32. Mendeley Wind Turbine Blade Fault Dataset

| Field | Details |
|-------|---------|
| **Name** | Wind Turbine Blades Fault Diagnosis based on Vibration Dataset Analysis |
| **Institution** | University of Technology, Iraq |
| **URL** | https://data.mendeley.com/datasets/5d7vbdp8f7/4 |
| **Download** | Direct (Mendeley) |
| **Size** | ~100 MB (estimated -- low sampling rate) |
| **Components** | Wind turbine blades (rotating) |
| **Sensors** | Uniaxial vibration |
| **Sampling rate** | 1 kHz (500 samples/channel) |
| **Files/samples** | 35 condition datasets |
| **Fault types** | Surface erosion, cracked blade, mass imbalance, twist blade fault, healthy |
| **Operating conditions** | Varying wind speeds |
| **Run-to-failure** | No |
| **License** | CC (Mendeley) |
| **Unique value** | **Blade/fan-specific faults. Covers blade-specific failure modes (erosion, cracks, twist) that map to fan/blower faults. Low sampling rate may limit utility.** |

---

## NEW DATASETS: MOTOR VIBRATION (SUPPLEMENTARY)

### 33. Zenodo Electric Motor Vibrations Dataset

| Field | Details |
|-------|---------|
| **Name** | Electric Motor Vibrations Dataset |
| **Institution** | CHIST-ERA SOON project (European) |
| **URL** | https://zenodo.org/records/6473455 |
| **Alt URL** | https://www.kaggle.com/datasets/amirberenji/electric-motor-vibrations-dataset |
| **Download** | Direct (Zenodo, CC-BY-4.0) |
| **Size** | ~1 GB (estimated) |
| **Components** | Electric motors (2 motors: tested + noise source) |
| **Sensors** | Vibration sensor |
| **Sampling rate** | Not specified |
| **Files/samples** | Multiple (m1 tested, m2 auxiliary) |
| **Fault types** | Various motor conditions (encoded in filenames) |
| **Operating conditions** | Complex environment (2 motors running) |
| **Run-to-failure** | No |
| **License** | CC-BY-4.0 |
| **Unique value** | **Two-motor environment captures realistic noise coupling. Good for noise-robust model development.** |

### 34. Politecnico di Torino Spherical Roller Bearing Dataset

| Field | Details |
|-------|---------|
| **Name** | Vibration, Temperature and Speed Measurements for Multiple Types of Localized Defects on Spherical Roller Bearings |
| **Institution** | Politecnico di Torino, Italy |
| **URL** | https://zenodo.org/records/13913254 |
| **Download** | **Requires written agreement with authors** |
| **Size** | Large (multi-GB) |
| **Components** | Spherical roller bearings (SKF 22240, OD up to 420mm) |
| **Sensors** | Vibration + temperature + speed |
| **Sampling rate** | Not specified |
| **Files/samples** | Organized as (speed)rpm_(radial)kN_(axial)kN.mat |
| **Fault types** | Multiple localized defects |
| **Operating conditions** | 10 rotation speeds, 4 load conditions (including axial) |
| **Run-to-failure** | No |
| **License** | Restricted (written agreement) |
| **Unique value** | **Medium/large-scale industrial bearings (unlike most lab datasets with small bearings). Spherical roller type. Includes axial load conditions.** |

---

## KAGGLE DATASETS (Quick-access, may overlap with above)

### 35. Kaggle Vibration Faults for Rotating Machines (sumairaziz)

| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/datasets/sumairaziz/vibration-faults-dataset-for-rotating-machines |
| **Download** | Kaggle API: `kaggle datasets download -d sumairaziz/vibration-faults-dataset-for-rotating-machines` |
| **Unique value** | Aggregate rotating machine faults. Check for overlap with MAFAULDA. |

### 36. Kaggle Vibration Analysis on Rotating Shaft

| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/datasets/jishnukoliyadan/vibration-analysis-on-rotating-shaft |
| **Download** | Kaggle API |
| **Unique value** | Shaft-specific vibration analysis. Check details on Kaggle page. |

### 37. Kaggle Mechanical Gear Vibration (hieudaotrung)

| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/datasets/hieudaotrung/gear-vibration |
| **Download** | Kaggle API |
| **Unique value** | Gear vibration data. Check for overlap with other gearbox datasets. |

### 38. Kaggle Gearbox Fault Diagnosis (brjapon)

| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/datasets/brjapon/gearbox-fault-diagnosis |
| **Download** | Kaggle API |
| **Unique value** | Gearbox fault data. Multiple elaborated/stacked versions available. May overlap with Mendeley gearbox datasets. |

---

## PRIORITY RANKING FOR YOUR PROJECT

### Tier 1: MUST HAVE (fill critical component gaps)

| Priority | Dataset | Why |
|----------|---------|-----|
| 1 | **#2 MAFAULDA** | Shaft faults (imbalance + misalignment) at multiple severity levels. 50 kHz. 8 channels. Fills shaft/coupling gap. |
| 2 | **#4 NLN-EMP (Dutch Navy)** | 11+ fault types across motor, coupling, shaft, bearing. Complete drivetrain. Near-perfect labeling. |
| 3 | **#1 SEU Drivetrain** | 8-channel drivetrain with both gear and bearing faults. Enables cross-component transfer. |
| 4 | **#10 Tsinghua Motor** | Motor stator/rotor faults + bearing faults + compound faults. Variable conditions. |
| 5 | **#6 Mendeley Multi-Mode Gearbox** | Gear faults with severity levels under truly variable (time-varying) speed + compound faults. |

### Tier 2: HIGH VALUE (extend coverage and diversity)

| Priority | Dataset | Why |
|----------|---------|-----|
| 6 | **#20 Seoul Compound Faults** | 3 bearing types + shaft faults + 21 compound combinations. Published 2024. |
| 7 | **#14 KAIST Industrial Motors** | 60+ GB, randomized speed, industrial scale. Published 2025. |
| 8 | **#7 Pretoria Planetary Gearbox** | Planetary gearbox (different physics from parallel gearbox). |
| 9 | **#19 HUST Bearing** | 5 bearing types, combination faults, time-varying speed. |
| 10 | **#22 Ottawa Time-Varying Speed** | 200 kHz, true speed ramps. Perfect for action-conditioned training. |

### Tier 3: SUPPLEMENTARY (nice to have for diversity)

| Priority | Dataset | Why |
|----------|---------|-----|
| 11 | **#5 Mendeley Gearbox Variable** | Full drivetrain with multi-encoder speed measurement. |
| 12 | **#16 Arkansas Single/Double Faults** | Bent shaft + compound faults. |
| 13 | **#3 COMFAULDA** | Compound fault combinations. |
| 14 | **#12 IEEE Broken Rotor Bar** | Motor rotor faults (open access). |
| 15 | **#28 SCA Pulp Mill** | Real industrial data with natural faults. |
| 16 | **#21 Ottawa 2023** | Cage faults + acoustic. |
| 17 | **#11 Figshare Motor** | 50 kHz synchronized vibration+current. |
| 18 | **#29 HIT Cross-Domain** | Explicitly for domain adaptation. |
| 19 | **#23 KAIST Run-to-Failure** | Modern RUL dataset with temperature. |
| 20 | **#13 KAIST PMSM** | PMSM stator faults -- different motor type. |

### Tier 4: SUPPLEMENTARY BEARING (only if time allows)

21-34: Remaining bearing datasets (Ferrara, UNSW, DLR, JNU, Vishwakarma, triaxial, wind turbine, etc.)

---

## COMPONENT COVERAGE SUMMARY

| Component | Already Have | New Datasets | Gap Status |
|-----------|-------------|-------------|------------|
| **Bearings (ball)** | CWRU, MFPT, IMS, XJTU-SY, Paderborn, FEMTO, Mendeley | HUST, Ottawa, KAIST, Ferrara, UNSW, DLR, JNU, Seoul, HIT, Vishwakarma, SCA, Triaxial | **EXCELLENT** |
| **Bearings (roller)** | Paderborn (some) | Seoul (cylindrical + tapered roller), Torino (spherical roller), Ferrara (self-aligning) | **GOOD** |
| **Gears (parallel)** | PHM 2009, OEDI, MCC5-THU | SEU, Mendeley Gearbox, Figshare Gearbox, UConn, Gearbox Var Conditions | **GOOD** |
| **Gears (planetary)** | None | SEU (has planetary), Pretoria Planetary | **IMPROVED** (was gap) |
| **Shaft (imbalance)** | None | MAFAULDA (7 levels), VBL-VA001, Seoul (3 levels), COMFAULDA | **FILLED** (was major gap) |
| **Shaft (misalignment)** | None | MAFAULDA (4+6 levels), VBL-VA001, NLN-EMP, Seoul (3 levels) | **FILLED** (was major gap) |
| **Shaft (bent/crack)** | None | Arkansas (bent), NLN-EMP (bent shaft), Rotor Systems | **PARTIALLY FILLED** |
| **Couplings** | None | NLN-EMP (coupling degradation) | **PARTIALLY FILLED** (1 dataset) |
| **Motor (rotor)** | None | IEEE Broken Rotor Bar, Tsinghua Motor (unbalance), KAIST Industrial | **FILLED** |
| **Motor (stator)** | None | Tsinghua Motor (winding short), KAIST PMSM (inter-turn/coil), ITSC, Figshare Motor | **FILLED** |
| **Motor (eccentricity)** | None | None found | **STILL GAP** |
| **Fans/Blowers** | None | Wind Turbine Blades (Mendeley), JNU (centrifugal fan context) | **PARTIALLY FILLED** |
| **Complete Drivetrains** | None | SEU, NLN-EMP, Gearbox Var Conditions | **FILLED** |
| **Compound faults** | None | COMFAULDA, Seoul, Tsinghua Motor, MAFAULDA | **FILLED** |

---

## ESTIMATED TOTAL NEW DATA VOLUME

| Tier | Estimated Size |
|------|---------------|
| Tier 1 (datasets 1-5) | ~25 GB |
| Tier 2 (datasets 6-10) | ~90 GB |
| Tier 3 (datasets 11-20) | ~15 GB |
| Tier 4 (datasets 21-34) | ~20 GB |
| **Total new data** | **~150 GB** |
| **Existing data** | ~40 GB |
| **Grand total** | **~190 GB** |

---

## DOWNLOAD METHODS SUMMARY

| Method | Datasets |
|--------|----------|
| **Direct (no registration)** | SEU, MAFAULDA, VBL-VA001, HUST, JNU, HIT, UNSW, Ferrara, DLR, Ottawa 2018, Figshare datasets, GitHub datasets |
| **Mendeley (free registration)** | Tsinghua Motor, Tsinghua Gearbox, Seoul Compound, KAIST, Triaxial, Vishwakarma, SCA, Gearbox Var Conditions, Ottawa 2023, KAIST RtF |
| **Kaggle API** | MAFAULDA mirror, various Kaggle datasets |
| **4TU (free registration)** | NLN-EMP |
| **IEEE DataPort (subscription)** | COMFAULDA, Rotor Systems |
| **Author contact required** | UConn, Politecnico di Torino |

---

## KEY REFERENCES

- Mauthe et al., "Overview of Publicly Available Degradation Data Sets for PHM," ESREL 2025, arXiv:2403.13694
- "Towards a Universal Vibration Analysis Dataset," IJPHM, 2025, arXiv:2504.11581
- awesome-bearing-dataset: https://github.com/VictorBauler/awesome-bearing-dataset
- Rotating-machine-fault-data-set: https://github.com/hustcxl/Rotating-machine-fault-data-set
- PredictiveMaintenance-and-Vibration-Resources: https://github.com/Charlie5DH/PredictiveMaintenance-and-Vibration-Resources
