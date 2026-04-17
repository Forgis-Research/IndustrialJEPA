# SWaT / WADI (iTrust, SUTD — Industrial Control Systems)

## Executive Summary
- **Domain**: Industrial Control Systems / Critical Infrastructure
- **Task**: Anomaly detection; adaptable to forecasting
- **Size (SWaT)**: ~946,000 timesteps × 51 channels; 11 days at 1 Hz
- **Size (WADI)**: ~1.2M timesteps × 127 channels; 16 days at 1 Hz
- **Sampling Rate**: 1 Hz
- **Real vs Synthetic**: Real — physical water treatment/distribution testbed
- **License**: Restricted — requires formal data request to iTrust (SUTD Singapore)
- **Download URL**: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/
  (Must submit data request form — typically approved for academic research)
- **Published SOTA**: Extensive for anomaly detection; limited for forecasting

## Detailed Description

SWaT (Secure Water Treatment) is a scaled-down operational water treatment plant used for cybersecurity and process anomaly research. It runs actual water treatment processes (flocculation, UF filtration, RO desalination, dechlorination) with real sensors and PLCs. WADI extends this to a water distribution network.

### Physical Setup (SWaT)
- 6 physical processes with dedicated sensors and actuators
- Processes: Raw water supply → UF filtration → chemical dosing → RO desalination → dechlorination → water storage
- Attack scenarios: 36 attacks injected over 4 days of the 11-day run
- Normal period: 7 days; Attack period: 4 days

### Channel Count and Groups (SWaT — 51 channels)
| Process | Sensors | Count |
|---|---|---|
| P1 (Water intake) | Flow, level, pressure, chemical dosing pumps | ~8 |
| P2 (Chemical dosing) | Flow sensors, pump states | ~6 |
| P3 (UF filtration) | Pressure, flow, UF status | ~8 |
| P4 (De-chlorination) | Chlorine, flow, pump | ~6 |
| P5 (RO desalination) | Conductivity, pressure, flow, RO state | ~12 |
| P6 (Product water) | Level, flow, UV, pump states) | ~11 |

### Physics Groups
```python
SWAT_GROUPS = {
    "P1_intake":    [0..7],
    "P2_chemical":  [8..13],
    "P3_filter":    [14..21],
    "P4_dechlo":    [22..27],
    "P5_RO":        [28..39],
    "P6_product":   [40..50],
}
```
These groups follow the physical process flow — excellent physics-informed structure.

## Published Benchmarks / SOTA (Anomaly Detection)
| Method | F1 | Precision | Recall | Paper | Year |
|---|---|---|---|---|---|
| LSTM-based | 0.77 | 0.99 | 0.63 | Hundman et al. | 2018 |
| OmniAnomaly | 0.84 | 0.84 | 0.84 | Su et al. | 2019 |
| THOC | 0.85 | 0.86 | 0.85 | Shen et al. | 2020 |
| GDN (Graph) | 0.88 | 0.89 | 0.87 | Deng & Hooi | 2021 |
| TranAD | 0.89 | 0.91 | 0.88 | Tuli et al. | 2022 |
| Anomaly Transformer | 0.91 | 0.94 | 0.88 | Xu et al. | 2022 |

## Relevance to IndustrialJEPA

### Physics Grouping Potential
**Excellent** — 6 process stages with clear physical meaning. The process flow P1→P6 creates a causal chain, which is exactly the kind of physics-informed structure our attention masks can exploit.

### Scale Adequacy
**Good for fine-tuning, moderate for pretraining**:
- SWaT: 946k × 51 = ~48M values
- WADI: 1.2M × 127 = ~152M values
- Both are continuous (no episodes), making them better for pretraining than bearing datasets

### Transfer Learning Scenarios
- SWaT → WADI: Same water system domain, different process stages
- Normal operation → attack periods (anomaly transfer)
- Cross-process: Train on P1-P3, test on P4-P6

### Forecasting Task Formulation
Excellent: 1 Hz data with 51 channels. Predict next 96/192 steps (96s/192s) given past 512 steps. Direct comparison with ETT benchmarks possible (same forecasting setup).

### Critical Limitation: Access
**Requires formal data request** to SUTD Singapore. Typically approved for academic research within 1–2 weeks, but not instantly downloadable. This delays iteration.

### Verdict for Tier 2
**Strong candidate if access obtained** — 51 real industrial sensors, physics-grouped by process stage, continuous 11-day run, published SOTA for comparison (even if anomaly detection, not forecasting). The process-flow physics is more interpretable than bearing vibration.

## Download Notes
- Submit data request at: https://itrust.sutd.edu.sg/testbeds/secure-water-treatment-swat/
- Approved researchers receive download link via email
- Format: CSV files for normal + attack periods
- Not downloadable without approval — cannot be scripted
- Downloader: `datasets/downloaders/download_swat.py` (includes instructions + placeholder)
