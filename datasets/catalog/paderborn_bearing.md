# Paderborn University Bearing Dataset (KAT Datacenter)

## Executive Summary
- **Domain**: Mechanical / Rotating Machinery
- **Task**: Fault detection / condition monitoring (adaptable to forecasting)
- **Size**: 26 bearing conditions × ~2 hours data × 8 channels × 64 kHz
- **Sampling Rate**: 64,000 Hz (vibration); 4,000 Hz (motor current)
- **Real vs Synthetic**: Real — includes both artificially and naturally damaged bearings
- **License**: Open for research use (direct download, no registration required)
- **Download URL**: https://groups.uni-paderborn.de/kat/BearingDataCenter/
- **Published SOTA**: Moderate — several dozens of papers benchmark on this dataset

## Detailed Description

The Paderborn dataset is significantly more realistic than CWRU: it includes naturally damaged bearings (progressive wear) alongside artificially damaged ones, and captures **both vibration and electrical motor signals** simultaneously. This makes it the most motor-physics-rich bearing dataset publicly available.

### Physical Setup
- **Test rig**: Modular drive train: AC motor → coupling → bearing housing → flywheel
- **Operating conditions**: 4 combinations of load torque (0.1–0.7 Nm) × rotational speed (900–1500 RPM) × radial force (400–1000 N)
- **Bearing types**: Various, with outer race / inner race / rolling element damage
- **Damage methods**: EDM (artificial), accelerated lifetime test (natural), and undamaged

### Naming Convention
- K001–K006: Undamaged bearings (healthy)
- KA01, KA03–KA09, KA15–KA16, KA22, KA30: Outer race damage
- KB23, KB24, KB27: Combined damage
- KI01, KI03–KI08, KI14, KI16–KI18, KI21: Inner race damage

### File Structure (per bearing, per condition)
Each RAR archive (~155–175 MB) contains 20 measurement files of 4 seconds at 64 kHz.

## Features
| Feature Name | Type | Unit | Description |
|---|---|---|---|
| a1 | vibration | m/s² | Accelerometer radial (measuring bearing) |
| a2 | vibration | m/s² | Accelerometer tangential (measuring bearing) |
| a3 | vibration | m/s² | Accelerometer axial (housing) |
| v1 | vibration | mm/s | Bearing shaft velocity |
| temp1 | temperature | °C | Bearing housing temperature |
| Nm (torque) | torque | Nm | Measured motor torque |
| phase_a | current | A | Motor phase current channel A |
| phase_b | current | A | Motor phase current channel B |

## Published Benchmarks / SOTA (Fault Diagnosis)
| Method | Metric | Value | Paper | Year |
|---|---|---|---|---|
| CNN on raw vibration | Accuracy | ~95–99% | Lessmeier et al. | 2016 |
| Transfer CNN (CWRU→PB) | Accuracy | ~88–93% | Various | 2019–2022 |
| Domain adaptation methods | Accuracy | ~91–96% | Multiple | 2020–2024 |

Note: Most results use a binary (healthy/faulty) or multi-class setting on selected conditions. No published forecasting SOTA.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Moderate** — 8 channels with distinct physical semantics:
```python
PADERBORN_GROUPS = {
    "vibration_radial":    [0, 1],   # a1, a2
    "vibration_axial":     [2, 3],   # a3, v1
    "thermal_torque":      [4, 5],   # temp1, torque
    "motor_current":       [6, 7],   # phase_a, phase_b
}
```
This grouping maps cleanly to physical subsystems (mechanical vibration, thermal/load, electrical).

### Transfer Learning Scenarios
- Healthy → damaged: train on K001, test on KA01
- Artificial → natural damage: KA (EDM) → KB (natural)
- Cross-operating-condition: different load/speed combinations
- Cross-domain: train on vibration, test using only current signals

### Scale Adequacy
**Limited**. 26 bearings × 20 measurements × 4s × 64kHz × 8ch = ~107M samples total. Good for training but small for pretraining a large model.

### Forecasting Task Formulation
Viable: predict next 640 timesteps (10ms at 64kHz) given past 6400 samples. Or downsample to 1kHz for longer temporal context.

### Verdict for Tier 2
**Strong candidate** — 8 channels with clear physics grouping, real damage, motor current + vibration multimodality. Main weakness: no published forecasting SOTA (would define our own baseline).

## Download Notes
- 33 RAR archives, each ~155–175 MB (total ~5.4 GB)
- No registration required — direct HTTP download
- Files at: https://groups.uni-paderborn.de/kat/BearingDataCenter/K001.rar etc.
- Read .mat files inside RAR with scipy after extraction
- Downloader: `datasets/downloaders/download_paderborn.py`
