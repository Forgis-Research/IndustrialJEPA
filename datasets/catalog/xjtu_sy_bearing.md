# XJTU-SY Bearing Dataset (Xi'an Jiaotong University)

## Executive Summary
- **Domain**: Mechanical / Rotating Machinery
- **Task**: Prognostics / RUL prediction; fault detection
- **Size**: 15 bearings × ~123–2,538 files × 32,768 samples × 2 channels
- **Sampling Rate**: 25,600 Hz (1.28-second snapshots every 1 minute)
- **Real vs Synthetic**: Real — accelerated bearing degradation experiment
- **License**: Free for research (requires registration on IEEE DataPort or direct request)
- **Download URL**: https://biapy.readthedocs.io/en/latest/ (secondary); primary: IEEE DataPort
  - IEEE DataPort: https://ieee-dataport.org/open-access/xjtu-sy-bearing-datasets
- **Published SOTA**: Moderate — dozens of prognostics papers benchmark here

## Detailed Description

XJTU-SY is a more controlled bearing degradation dataset than IMS, with explicit operating conditions and a larger number of bearing runs. Each bearing is run under one of three speed-load conditions until failure, with consistent 1-minute snapshots.

### Physical Setup
- **Test rig**: Motor-driven shaft with a test bearing
- **Speed-load conditions**:
  - Condition 1: 2100 RPM, 12 kN radial load (5 bearings)
  - Condition 2: 2250 RPM, 11 kN radial load (5 bearings)
  - Condition 3: 2400 RPM, 10 kN radial load (5 bearings)
- **Bearing type**: LDK UER204 ball bearing
- **Sensors**: 2 accelerometers (horizontal + vertical) on bearing housing
- **Sample frequency**: 25,600 Hz
- **Sample duration**: 1.28 seconds per snapshot

### Dataset Structure
| Condition | Speed (RPM) | Load (kN) | Bearings | Files/Bearing | Failure Types |
|---|---|---|---|---|---|
| 1 | 2100 | 12 | 5 | 123–2,803 | Outer/inner/cage/combined |
| 2 | 2250 | 11 | 5 | 491–3,538 | Outer/inner/cage |
| 3 | 2400 | 10 | 5 | 241–2,538 | Outer/inner/combined |

## Features (2 channels per snapshot)
| Feature | Type | Unit | Description |
|---|---|---|---|
| horizontal | vibration | g | Horizontal accelerometer |
| vertical | vibration | g | Vertical accelerometer |

## Physics Groups
```python
XJTU_GROUPS = {
    "horizontal": [0],  # horizontal vibration
    "vertical":   [1],  # vertical vibration
}
```
Only 2 channels — minimal grouping structure.

## Published Benchmarks / SOTA (RUL Prediction)
| Method | Condition 1 RMSE | Condition 2 RMSE | Paper | Year |
|---|---|---|---|---|
| CNN + LSTM | ~0.12 (normalized) | ~0.14 | Pan et al. | 2020 |
| LSTM + Attention | ~0.09 | ~0.11 | Various | 2021 |
| Transformer | ~0.08 | ~0.10 | Multiple | 2022–2023 |

Note: Results reported as normalized RMSE (0–1 scale for RUL percentage). Transfer between conditions is common evaluation.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Very weak** — only 2 channels (X, Y vibration). Cannot demonstrate meaningful attention masking with 2 channels.

### Transfer Learning Scenarios
- Condition 1 → Condition 2 → Condition 3 (increasing speed/decreasing load)
- Cross-bearing: one bearing's healthy phase → another's degradation phase
- Cross-fault-type: outer race → inner race failure prediction

### Scale Adequacy
**Moderate** — 15 bearings × ~1000 files average × 32k samples = ~500M samples total. Much larger raw data than IMS. But only 2 channels limits its usefulness.

### Verdict for IndustrialJEPA
**Not recommended** — too few channels for physics-informed attention. Use only if doing ablation on bearing RUL as a secondary task.

## Download Notes
- IEEE DataPort: https://ieee-dataport.org/open-access/xjtu-sy-bearing-datasets
  (requires free IEEE account)
- Files in CSV format per bearing condition
- Downloader: `datasets/downloaders/download_xjtu_sy.py`
