# Data Quality Report -- IndustrialJEPA

**Audited by:** Data Curator Agent
**Date:** 2026-03-26
**Datasets:** ETT (4 variants), UCI Hydraulic System, CWRU Bearing, Paderborn Bearing

---

## Executive Summary

All four datasets loaded successfully with zero missing values. Data quality is high across the board. Each dataset is suitable for the multivariate time series forecasting task central to IndustrialJEPA. Detailed per-dataset findings follow.

| Dataset | Shape | Missing | Duplicates | Quality Score |
|---------|-------|---------|------------|---------------|
| ETTh1 | 17,420 x 8 | 0 | 0 | **A** |
| ETTh2 | 17,420 x 8 | 0 | 0 | **A** |
| ETTm1 | 69,680 x 8 | 0 | 0 | **A** |
| ETTm2 | 69,680 x 8 | 0 | 0 | **A** |
| Hydraulic | 2,205 cycles x 17 sensors | 0 | 0 | **A-** |
| CWRU | 4 files, 121k-244k samples | 0 | 0 | **A** |
| Paderborn | 240 files, 7 channels each | 0 | 0 | **A** |

---

## 1. ETT (Electricity Transformer Temperature)

### 1.1 Basic Stats

| Variant | Rows | Cols | Sampling | Date Range |
|---------|------|------|----------|------------|
| ETTh1 | 17,420 | 8 | 1 hour | 2016-07-01 to 2018-06-26 |
| ETTh2 | 17,420 | 8 | 1 hour | 2016-07-01 to 2018-06-26 |
| ETTm1 | 69,680 | 8 | 15 min | 2016-07-01 to 2018-06-26 |
| ETTm2 | 69,680 | 8 | 15 min | 2016-07-01 to 2018-06-26 |

**Columns:** date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (oil temperature target)
**Data types:** All float64 (numeric features), datetime (date)

### 1.2 Quality Assessment

- **Missing values:** 0 across all variants
- **Duplicate rows:** 0 across all variants
- **Temporal regularity:** PERFECT -- exactly 1-hour (h) or 15-min (m) gaps, no jumps
- **Constant columns:** None
- **Outliers (>5 std):** ETTh2 HUFL: 55 (0.32%), ETTm2 HUFL: 223 (0.32%), ETTm2 HULL: 1 (~0%). All minor.
- **Near-perfect correlations:** None detected (highest: HUFL-MUFL at 0.99 in ETTh1 -- these are physically related transformer load features)

### 1.3 Red Flags

- The HUFL-MUFL correlation (0.99) is extremely high. This reflects physics (both measure high/medium utilization of the same load) and is expected, not an error.
- ETTh1 and ETTm1 cover the same transformer station but at different temporal resolutions. Same for h2/m2.

### 1.4 Quality Score: **A**

No issues. Clean, regular, well-documented benchmark data.

---

## 2. UCI Hydraulic System

### 2.1 Basic Stats

- **2,205 working cycles**, each 60 seconds long
- **17 sensor channels** at varying sampling rates:
  - 100 Hz (6000 pts/cycle): PS1-PS6, EPS1
  - 10 Hz (600 pts/cycle): FS1, FS2
  - 1 Hz (60 pts/cycle): TS1-TS4, VS1, CE, CP, SE
- **5 label columns:** cooler condition {3,20,100}, valve condition {73,80,90,100}, pump leakage {0,1,2}, accumulator pressure {90,100,115,130}, stability flag {0,1}

### 2.2 Data Types

All sensor readings are float64. Labels are integer-coded categorical.

### 2.3 Quality Assessment

- **Missing values:** 0 across all sensors
- **Duplicate rows:** Not applicable (each row is a cycle)
- **Temporal regularity:** Uniform within each cycle (fixed sampling rate per sensor)
- **Constant columns:** None

### 2.4 Red Flags

- **PS4 has 1,238 constant rows (56%)** -- many cycles show zero variation in this pressure sensor. This is likely a physical characteristic (PS4 may be a secondary/redundant sensor that only activates under certain conditions). Not a data error but important to note for modeling.
- Label class imbalance is moderate:
  - Cooler: 3 classes (roughly balanced)
  - Valve: 4 classes (roughly balanced)
  - Pump: 3 classes
  - Accumulator: 4 classes

### 2.5 Quality Score: **A-**

Docked slightly for the PS4 constant-row issue (56% flat signal). Otherwise excellent multi-sensor industrial data.

---

## 3. CWRU Bearing Dataset

### 3.1 Basic Stats

| File | Condition | Samples | Channels |
|------|-----------|---------|----------|
| normal_0hp.mat | Normal | 243,938 | 2 (DE, FE) |
| IR007_0hp.mat | Inner Race Fault (0.007") | 121,265 | 3 (DE, FE, BA) |
| B007_0hp.mat | Ball Fault (0.007") | 122,571 | 3 (DE, FE, BA) |
| OR007_6_0hp.mat | Outer Race Fault (0.007") | 121,991 | 3 (DE, FE, BA) |

**Sampling rate:** 12,000 Hz (DE, FE), 48,000 Hz (BA)
**Channels:** Drive End accelerometer (DE), Fan End accelerometer (FE), Base accelerometer (BA, fault files only)

### 3.2 Quality Assessment

- **Missing values:** 0
- **Data types:** float64
- **Outliers (>5 std):**
  - IR007 DE: 38 (0.031%) -- expected for impulsive fault signatures
  - OR007 DE: 13 (0.011%) -- same reason
  - OR007 BA: 4 (0.003%)
- **Amplitude range varies dramatically by condition:**
  - Normal DE std: 0.073g
  - Outer Race DE std: 0.669g (9x larger -- strong fault signature)

### 3.3 Red Flags

- Normal file has only 2 channels (no BA), fault files have 3. This inconsistency must be handled during preprocessing.
- Only 4 files at 0 hp load are present. Full CWRU has many more conditions/loads. Current subset is minimal but sufficient for proof-of-concept.

### 3.4 Quality Score: **A**

Clean vibration data with clear fault signatures visible in both time and frequency domains.

---

## 4. Paderborn Bearing Dataset

### 4.1 Basic Stats

| Condition | Code | Files | Description |
|-----------|------|-------|-------------|
| Healthy | K001 | 80 | No damage |
| Outer Race Damage | KA01 | 80 | Artificial outer race fault |
| Inner Race Damage | KI01 | 80 | Artificial inner race fault |

**7 channels per measurement:**

| Channel | Sample Length | Est. Sampling Rate |
|---------|-------------|-------------------|
| force | 16,008 | ~4 kHz |
| phase_current_1 | 256,823 | ~64 kHz |
| phase_current_2 | 256,823 | ~64 kHz |
| speed | 16,008 | ~4 kHz |
| temp_2_bearing_module | 5 | ~1 Hz |
| torque | 16,008 | ~4 kHz |
| vibration_1 | 256,823 | ~64 kHz |

**Duration:** ~4 seconds per measurement

### 4.2 Quality Assessment

- **Missing values:** 0
- **Data types:** float64
- **Structure:** Nested MATLAB structs (X=independent vars, Y=measurements, 7 signal channels)
- **Consistency:** All 240 files have identical structure and channel count
- **Sample lengths:** Consistent within each channel across files

### 4.3 Red Flags

- **Mixed sampling rates:** Channels have very different lengths (5 to 256,823 samples). Alignment/resampling will be needed for multivariate modeling.
- **Temperature channel has only 5 samples** per 4-second recording -- essentially a scalar, not useful as a time series channel.
- The `.rar` archives remain alongside extracted folders (disk overhead but not a data issue).

### 4.4 Quality Score: **A**

Rich multi-channel bearing data with clear physics-based channel groupings.

---

## 5. Use-Case Assessment for IndustrialJEPA

### 5.1 Multivariate Time Series Forecasting Suitability

| Dataset | Forecasting Task | Suitability |
|---------|-----------------|-------------|
| ETT | Forecast OT from 6 load features | Excellent -- standard benchmark |
| Hydraulic | Forecast sensor evolution within/across cycles | Good -- multi-rate complicates things |
| CWRU | Forecast vibration signal evolution | Moderate -- single-channel per sensor, short recordings |
| Paderborn | Forecast vibration/current/force | Good -- rich multi-channel, needs alignment |

### 5.2 Physics-Based Channel Groupings

**ETT:**
- Load group: HUFL, HULL (high utilization)
- Load group: MUFL, MULL (medium utilization)
- Load group: LUFL, LULL (low utilization)
- Target: OT (oil temperature -- thermal response to electrical load)
- Physics: Electrical load -> heat generation -> oil temperature (thermal dynamics)

**Hydraulic:**
- Pressure group: PS1-PS6 (hydraulic circuit pressures)
- Temperature group: TS1-TS4 (thermal state)
- Flow group: FS1-FS2 (fluid dynamics)
- Electrical group: EPS1 (motor power), CE (cooling efficiency)
- Mechanical group: VS1 (vibration), CP (cooling power), SE (efficiency)
- Physics: Pressure-flow-temperature coupled through fluid dynamics and thermodynamics

**CWRU:**
- Spatial group: DE, FE, BA (different locations on same machine)
- Physics: Fault impulses propagate through mechanical structure with distance-dependent attenuation

**Paderborn:**
- Mechanical group: force, torque, speed (load conditions)
- Electrical group: phase_current_1, phase_current_2 (motor state)
- Vibration group: vibration_1 (bearing health indicator)
- Thermal group: temp_2_bearing_module (slow thermal state)
- Physics: Electromechanical coupling -- current drives torque, bearing damage modulates vibration and force

### 5.3 Transfer Learning Scenarios

| Dataset | Transfer Scenario |
|---------|-------------------|
| ETT | ETTh1 -> ETTh2 (different transformer stations, same schema) |
| ETT | ETTh1 -> ETTm1 (same station, different temporal resolution) |
| Hydraulic | Healthy -> Degraded condition transfer |
| CWRU | Normal -> Fault detection (domain adaptation) |
| CWRU | Cross-load transfer (if more load conditions downloaded) |
| Paderborn | K001 -> KA01/KI01 (healthy to fault transfer) |
| Paderborn | Cross-operating-condition (different speed/load combos in filenames) |

### 5.4 Required Preprocessing

| Dataset | Preprocessing Steps |
|---------|-------------------|
| ETT | Minimal: standard normalization, train/val/test split by time |
| Hydraulic | Multi-rate alignment (resample to common rate or process separately), cycle segmentation already done |
| CWRU | Windowing into fixed-length segments, channel alignment (normal has 2ch vs 3ch for faults), normalization |
| Paderborn | Multi-rate channel alignment (resample or select subset), windowing, extract from nested .mat structure |

---

## 6. Existing Analysis Figures Review

All four overview figures (`*_overview.png`) were reviewed visually:

- **ett_overview.png:** Correct. Shows full time series, missing values, feature distributions, correlation, and cross-variant comparison. Well-constructed.
- **hydraulic_overview.png:** Correct. Shows sensor means, condition distributions, cross-sensor correlation, healthy-vs-fault comparison. Good.
- **cwru_overview.png:** Correct. Shows time-domain signals, FFT spectra, RMS by fault type, channel counts. Clear fault signature differentiation.
- **paderborn_overview.png:** Correct. Shows vibration/current signals, FFT, RMS by channel/condition, cross-channel correlation. Comprehensive.

No errors or misleading visualizations detected. The curator audit figures (`*_curator_audit.png`) complement the overviews with additional detail on distributions, outliers, and quantitative quality metrics.

---

## 7. Audit Figures Index

| Figure | Description |
|--------|-------------|
| `ett_overview.png` | Original EDA overview |
| `ett_curator_audit.png` | Quality audit: distributions, correlations, time series |
| `hydraulic_overview.png` | Original EDA overview |
| `hydraulic_curator_audit.png` | Quality audit: sensor distributions, healthy-vs-degraded |
| `cwru_overview.png` | Original EDA overview |
| `cwru_curator_audit.png` | Quality audit: amplitude distributions, FFT, condition comparison |
| `paderborn_overview.png` | Original EDA overview |
| `paderborn_curator_audit.png` | Quality audit: vibration/current distributions, FFT, force comparison |

All figures saved in `datasets/analysis/figures/`.
