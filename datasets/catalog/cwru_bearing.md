# CWRU Bearing Dataset (Case Western Reserve University)

## Executive Summary
- **Domain**: Mechanical / Rotating Machinery
- **Task**: Fault detection / classification (adaptable to forecasting)
- **Size**: ~500 files × 20,480 samples × 2–4 channels per file
- **Sampling Rate**: 12,000 Hz (drive end) or 48,000 Hz (high-resolution), 12,000 Hz (fan end)
- **Real vs Synthetic**: Real — seeded fault experiment on a motor-drive test rig
- **License**: Public domain (Case Western Reserve University, free for research)
- **Download URL**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Published SOTA**: Extensive — hundreds of papers use this as a benchmark

## Detailed Description

The CWRU bearing dataset is the de-facto standard benchmark for bearing fault diagnosis ML research. Data was collected from a motor-drive test rig with an induction motor, torque transducer/encoder, dynamometer, and control electronics. Fault conditions were introduced using electro-discharge machining (EDM) at specific bearing components.

### Physical Setup
- **Bearing**: SKF deep-groove ball bearing (6205-2RS and 6203-2RS)
- **Fault locations**: Outer race, inner race, rolling element (ball)
- **Fault diameters**: 0.007, 0.014, 0.021 inches
- **Load conditions**: 0, 1, 2, 3 horsepower (motor speed: 1797–1730 RPM)
- **Sensors**: Accelerometers at drive end, fan end, base

### Data Categories
| Category | Sampling Rate | Channels |
|---|---|---|
| Normal baseline | 12 kHz | Drive end + Fan end |
| 12k Drive End | 12,000 Hz | DE + FE + BA |
| 48k Drive End | 48,000 Hz | DE + FE + BA |
| Fan End | 12,000 Hz | FE + BA |

### Known Issues / Limitations
- Single sensor type (vibration accelerometer) — no torque, current, or thermal data
- Fault sizes are artificial (EDM) not realistic progressive wear
- Laboratory conditions only — no load variation or environmental noise
- Channel count is low (2–4) — limited physics grouping structure
- Primary use case is fault **classification**, not **forecasting**
- Signal is periodic/stationary under constant load — low information per channel

## Features
| Feature Name | Type | Unit | Description |
|---|---|---|---|
| DE_time | vibration | g | Drive end accelerometer |
| FE_time | vibration | g | Fan end accelerometer |
| BA_time | vibration | g | Base accelerometer |
| RPM | scalar | rpm | Motor rotational speed |

## Published Benchmarks / SOTA (Fault Diagnosis)
| Method | Metric | Value | Paper | Year |
|---|---|---|---|---|
| CNN-1D | Accuracy | 99.7% | Zhang et al. | 2017 |
| Res-Net | Accuracy | 99.4% | Zhang et al. | 2018 |
| WDCNN | Accuracy | 99.5% | Zhang et al. | 2017 |
| GAN-augmented CNN | Accuracy | 99.8% | Various | 2019–2022 |
| Transformer | Accuracy | ~99%+ | Many | 2020–2024 |

Note: Most published results use 4-class or 10-class classification. Forecasting SOTA is not established.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Weak** — only 2–4 channels, all vibration. Grouping is: {DE, FE} by bearing location, or {drive-end-components} by fault location. Too few channels for meaningful attention mask research.

### Transfer Learning Scenarios
- Within-dataset: Different fault severity or load condition
- Cross-bearing-type: 6205 → 6203
- Cross-load: 0 HP → 3 HP (motor speed changes)

### Scale Adequacy
**Insufficient for pretraining**. ~500 files × 20k samples = ~10M samples total across all conditions, but only 2–4 channels. Good for rapid prototyping only.

### Forecasting Task Formulation
Can formulate as: predict next 96 timesteps of vibration given past 512. However, this is not the primary use case and no published SOTA exists for this formulation.

### Verdict for Tier 2
**Not recommended**. Too few channels, vibration-only, no published forecasting baselines. Good for fault classification ablations only.

## Download Notes
- Files are in MATLAB .mat format
- Can be read with `scipy.io.loadmat()`
- No registration required
- Downloader: `datasets/downloaders/download_cwru.py`
