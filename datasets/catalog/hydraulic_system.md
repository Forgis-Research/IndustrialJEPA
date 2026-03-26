# UCI Hydraulic System Condition Monitoring Dataset

## Executive Summary
- **Domain**: Industrial / Hydraulic Systems
- **Task**: Condition monitoring / fault detection (multi-label regression)
- **Size**: 2,205 cycles × 17 sensors × up to 6,000 timesteps (per cycle, varies by sensor)
- **Sampling Rate**: 1–100 Hz (mixed — pressure/power at 100 Hz, temperature at 1 Hz)
- **Real vs Synthetic**: Real — laboratory hydraulic test rig
- **License**: CC BY 4.0 (UCI ML Repository)
- **Download URL**: https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
- **Published SOTA**: Limited — primarily used for classification, not forecasting

## Detailed Description

Data was collected from a hydraulic test rig with a primary working circuit and secondary cooling-filtration circuit connected via oil tank. Each experiment consists of a 60-second load cycle recorded under varying component degradation states.

### Physical Setup
- **System**: Industrial hydraulic circuit
- **Primary components**: Hydraulic pump, pressure control valve, hydraulic accumulator, cooler
- **Load**: Constant 60-second cycles at varying conditions
- **Fault injection**: Progressive degradation of 4 components

### Sensor Channels (17 total across 2,205 cycles)
| Sensor | Rate (Hz) | Timesteps/Cycle | Unit | Description |
|---|---|---|---|---|
| PS1 | 100 | 6,000 | bar | Pressure sensor 1 |
| PS2 | 100 | 6,000 | bar | Pressure sensor 2 |
| PS3 | 100 | 6,000 | bar | Pressure sensor 3 |
| PS4 | 100 | 6,000 | bar | Pressure sensor 4 |
| PS5 | 100 | 6,000 | bar | Pressure sensor 5 |
| PS6 | 100 | 6,000 | bar | Pressure sensor 6 |
| EPS1 | 100 | 6,000 | W | Motor power |
| FS1 | 10 | 600 | l/min | Flow sensor 1 |
| FS2 | 10 | 600 | l/min | Flow sensor 2 |
| TS1 | 1 | 60 | °C | Temperature 1 |
| TS2 | 1 | 60 | °C | Temperature 2 |
| TS3 | 1 | 60 | °C | Temperature 3 |
| TS4 | 1 | 60 | °C | Temperature 4 |
| VS1 | 1 | 60 | mm/s | Vibration |
| CE | 1 | 60 | % | Cooling efficiency |
| CP | 1 | 60 | kW | Cooling power |
| SE | 1 | 60 | % | System efficiency |

### Fault Conditions (Labels)
| Component | Degradation States |
|---|---|
| Cooler | 3% efficiency, 20%, 100% (no fault) |
| Valve | 73% closing, 80%, 90%, 100% (no fault) |
| Pump | Leakage 0 ml/min, 1, 2 |
| Accumulator | 90, 100, 115, 130 bar |
| Stability | Stable / unstable cycle |

## Features (Physics Grouping)
```python
HYDRAULIC_GROUPS = {
    "pressure":     [0, 1, 2, 3, 4, 5],     # PS1-PS6
    "flow_power":   [6, 7, 8],               # EPS1, FS1, FS2
    "thermal":      [9, 10, 11, 12, 15, 16], # TS1-TS4, CE, CP, SE
    "mechanical":   [13],                    # VS1
}
```

## Published Benchmarks / SOTA
| Method | Task | Metric | Value | Year |
|---|---|---|---|---|
| MLP on hand features | Valve classification | Accuracy | ~96% | 2019 |
| CNN on raw signals | Multi-fault | F1 | ~93% | 2020 |
| LSTM | Cooler condition | Accuracy | ~98% | 2021 |

Note: No published SOTA for time series forecasting on this dataset.

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Strong** — 4 clearly defined physical subsystems (pressure, flow, thermal, vibration). The mixed sampling rates (1–100 Hz) require careful interpolation or sensor-selection for uniform time series.

### Transfer Learning Scenarios
- Different cooler degradation levels
- Different pump leakage stages
- Cross-fault-type transfer (trained on valve degradation, test on pump leakage)

### Scale Adequacy
**Small** — 2,205 cycles total. Good for fine-tuning; insufficient for pretraining.

### Forecasting Task Formulation
Viable using the 100 Hz sensors (PS1-PS6, EPS1): predict next 60 timesteps (0.6s) given past 300 samples. Must select consistent-rate sensors (e.g., PS1-PS6 + EPS1 only = 7 channels at 100 Hz).

### Verdict for Tier 2
**Possible fallback** — good physics grouping, real data. Main weakness: small scale (2,205 cycles), mixed sampling rates complicate preprocessing, and no published forecasting SOTA.

## Download Notes
- Direct download as ZIP (73.1 MB) — no registration required
- URL: https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip
- Individual sensor data as space-delimited .txt files (one file per sensor)
- Labels in profile.txt
- Downloader: `datasets/downloaders/download_hydraulic.py`
