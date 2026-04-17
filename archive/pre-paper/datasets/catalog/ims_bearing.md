# IMS Bearing Dataset (NASA / University of Cincinnati)

## Executive Summary
- **Domain**: Mechanical / Rotating Machinery
- **Task**: Prognostics / RUL prediction; adaptable to forecasting
- **Size**: 3 test-to-failure runs × up to 8 channels × 20,480 samples/file × up to 4,448 files
- **Sampling Rate**: 20,000 Hz (1-second snapshots every 10 minutes)
- **Real vs Synthetic**: Real — run-to-failure experiment
- **License**: Public domain (NASA / U.S. Government Works)
- **Download URL**: https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset (Kaggle mirror)
  - Original: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/ (may redirect)
- **Published SOTA**: Moderate — dozens of RUL papers benchmark here

## Detailed Description

The IMS (Intelligent Maintenance Systems) dataset contains run-to-failure bearing vibration data collected at the University of Cincinnati. Three complete experiments were performed; each ran until at least one bearing failed. The dataset is particularly valuable for prognostics because it captures progressive degradation over weeks.

### Physical Setup
- **Test rig**: 4 bearings on a single shaft, motor-driven
- **Speed**: 2000 RPM (constant)
- **Load**: 6000 lbs radial force applied via a spring mechanism
- **Bearing type**: Rexnord ZA-2115 double row
- **Lubrication**: Oil circulated through bearings

### Experiments
| Set | Duration | Files | Channels | Failure |
|---|---|---|---|---|
| Set 1 | Oct 22 – Nov 25, 2003 | 2,156 | 8 (2/bearing: X, Y) | Bearing 3 inner race, Bearing 4 roller |
| Set 2 | Feb 12–19, 2004 | 984 | 4 (1/bearing) | Bearing 1 outer race |
| Set 3 | Mar 4 – Apr 4, 2004 | 4,448 | 4 (1/bearing) | Bearing 3 outer race |

### Channels (Set 1 — most information-rich)
| Channel | Description |
|---|---|
| ch1 | Bearing 1 X-axis vibration |
| ch2 | Bearing 1 Y-axis vibration |
| ch3 | Bearing 2 X-axis vibration |
| ch4 | Bearing 2 Y-axis vibration |
| ch5 | Bearing 3 X-axis vibration |
| ch6 | Bearing 3 Y-axis vibration |
| ch7 | Bearing 4 X-axis vibration |
| ch8 | Bearing 4 Y-axis vibration |

## Physics Groups
```python
IMS_GROUPS = {
    "bearing_1": [0, 1],   # ch1, ch2 (x, y)
    "bearing_2": [2, 3],   # ch3, ch4 (x, y)
    "bearing_3": [4, 5],   # ch5, ch6 (x, y)
    "bearing_4": [6, 7],   # ch7, ch8 (x, y)
}
```
This is a natural 4-group physics structure with clear symmetry.

## Published Benchmarks / SOTA (RUL)
| Method | Task | Metric | Value | Year |
|---|---|---|---|---|
| CNN + LSTM | RUL | RMSE (Set 1) | ~1500 samples | 2018 |
| ResNet + Attention | RUL | RMSE | ~1100 | 2020 |
| Transformer | Degradation stage | Accuracy | ~90% | 2021 |
| BiLSTM | Health indicator | Correlation | ~0.93 | 2022 |

Note: Metrics are not standardized — some use RMSE in timesteps, others use health index correlation. No published forecasting MSE baselines.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Strong** — 8 channels with natural grouping by bearing (4 bearings × 2 axes). The X/Y axis pair within each bearing is the finest grouping; the bearing-level grouping is the coarser level. This creates a 2-level hierarchy.

### Transfer Learning Scenarios
- Set 1 → Set 2: Same rig, different run (different failure modes)
- Healthy phase → degraded phase (early vs. late in each run)
- Cross-axis: X-axis model tested on Y-axis

### Scale Adequacy
**Limited** — ~3 runs × ~1000–4000 files × 20k samples = high raw sample count but only 3 "instances" at the run level. Good for single-dataset analysis, not pretraining.

### Forecasting Task Formulation
Natural: resample from 20kHz to ~200Hz (50x downsampling), then predict next 200 samples (1s) given past 1000 samples (5s). This captures bearing resonance dynamics without requiring huge windows.

### Verdict for Tier 2
**Not recommended as primary Tier 2** — only 8 vibration channels, no current/torque/temperature. Physics grouping is by bearing position (spatial), not by physical quantity type. Similar limitation to CWRU.

## Download Notes
- Available as ZIP (~1.67 GB) from Kaggle (requires account)
- Files are space-delimited ASCII text
- Original NASA URL may redirect; Kaggle mirror is reliable
- Downloader: `datasets/downloaders/download_ims.py`
