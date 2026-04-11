# TabPFN for Mechanical Systems Time Series Forecasting

**Research Question**: Can TabPFN-TS, a tabular foundation model adapted for time series, provide useful forecasting capabilities for mechanical systems such as turbofan engines, hydraulic systems, and robotic manipulators?

**Last Updated**: 2026-03-27

---

## Executive Summary

[TabPFN](https://github.com/PriorLabs/TabPFN) is a transformer-based foundation model for tabular data. [TabPFN-TS](https://github.com/PriorLabs/tabpfn-time-series) extends it to time series forecasting by treating prediction as a tabular regression problem with temporal features.

### Why This Matters for Mechanical Systems

1. **Zero-shot capability**: No training required—immediately applicable to new systems
2. **Covariate support**: Can incorporate control inputs, operating conditions, sensor readings
3. **Small but powerful**: 11M parameters outperforms much larger specialized models
4. **Interpretable features**: Temporal featurization is explicit and inspectable

### Key Hypothesis

Mechanical systems often exhibit:
- **Periodic patterns** (rotational machinery, cyclic loads)
- **Multi-modal sensor relationships** (temperature, pressure, vibration, current)
- **Operating condition dependence** (speed, load settings as covariates)

TabPFN-TS could excel here because it:
- Treats time series as tabular data with explicit temporal features
- Naturally handles covariates without special preprocessing
- Auto-detects periodicities via FFT (rotation frequencies, machine cycles)
- Native probabilistic outputs (better calibrated than quantile heads)

### Known Limitations (from paper analysis)

**Critical**: TabPFN-TS does **NOT extrapolate linear trends well** (Section 5.4 of paper).
- Degradation trajectories may flatten out instead of continuing
- C-MAPSS RUL-style predictions will be challenging
- **Mitigation**: Detrend data first, forecast residuals, add trend back

**What works well**:
- Pure periodic signals (almost perfect)
- Complex waveforms (multiple harmonics)
- Trend + periodicity combinations
- Exponential trends (better than linear)

See `docs/PAPER_INSIGHTS.md` for detailed analysis of the TabPFN-TS paper

---

## Directory Structure

```
TabPFN/
├── README.md                 # This file
├── tutorial/                 # Learn TabPFN-TS hands-on
│   ├── 01_installation.md    # Setup guide
│   ├── 02_quickstart.ipynb   # First forecasts
│   ├── 03_covariates.ipynb   # Adding external features
│   ├── 04_mechanical.ipynb   # Apply to bearing/hydraulic data
│   └── exercises/            # Practice problems
├── experiments/              # Minimal experiments for assessment
│   ├── exp01_hydraulic.py    # Hydraulic system forecasting
│   ├── exp02_cmapss.py       # Turbofan sensor prediction
│   ├── exp03_bearing.py      # Bearing vibration forecasting
│   └── baselines/            # Comparison models
└── docs/                     # Research documentation
    ├── DATASETS.md           # Dataset selection rationale
    ├── EXPERIMENT_PLAN.md    # Structured experiment outline
    ├── BREAKTHROUGH.md       # What success looks like
    └── PAPER_INSIGHTS.md     # Detailed paper analysis (critical!)
```

---

## How It Works (Technical Summary)

TabPFN-TS converts time series to tabular format via feature extraction:

```
X_t = Φ_cal(t) ⊕ Φ_auto(t) ⊕ Φ_index(t) ∈ R^28
```

| Feature Type | Dims | Description | Mechanical Relevance |
|--------------|------|-------------|---------------------|
| **Calendar** | 17 | sin/cos of day/week/month/year | Low (replace with machine cycles) |
| **Auto-Seasonal** | 10 | Top-5 FFT-detected frequencies | **High** (detects rotation freq) |
| **Running Index** | 1 | Simple timestep counter | Medium (enables trends) |

**Key insight**: The auto-seasonal features via FFT can automatically detect machine rotation frequencies, load cycles, and other domain-specific periodicities.

**No lag features**: Unlike traditional approaches, TabPFN-TS does NOT use autoregressive features (moving averages, lag terms). This enables fast multi-step forecasting.

---

## Quick Start

### Installation

```bash
# TabPFN-TS (recommended)
pip install tabpfn-time-series

# Or full TabPFN with extensions
pip install tabpfn
```

### Minimal Example

```python
import pandas as pd
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

# Your mechanical system time series
y = pressure_sensor_data  # shape: (n_timesteps,)

# Prepare data in required format (DataFrame with timestamp and target)
context_df = pd.DataFrame({
    'item_id': ['sensor'] * len(y),  # Optional but recommended
    'timestamp': pd.date_range('2024-01-01', periods=len(y), freq='s'),
    'target': y
})

# Optional: add operating conditions as covariates
# context_df['speed'] = motor_speed
# context_df['load'] = load_setting

# Create pipeline (uses cloud API by default - no GPU needed)
pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.CLIENT)

# Forecast (returns DataFrame with quantiles 0.1-0.9)
predictions_df = pipeline.predict_df(
    context_df=context_df,
    prediction_length=10  # forecast 10 steps ahead
)

# Get point predictions (median)
predictions = predictions_df['0.5'].values
```

---

## Research Directions

### 1. Single-Sensor Univariate Forecasting
Can TabPFN-TS beat naive baselines on individual mechanical sensors?
- Pressure sensors (hydraulic system)
- Vibration signals (bearing data)
- Temperature profiles (turbofan)

### 2. Covariate-Informed Forecasting
Does adding operating conditions improve mechanical system forecasts?
- Control inputs → sensor readings
- Cross-sensor conditioning (pressure predicts flow)
- Degradation state as implicit covariate

### 3. Cross-System Transfer
Can features learned on one mechanical system generalize?
- Hydraulic → pneumatic systems
- Single robot → different robot embodiment
- Healthy → degraded conditions

---

## Relevant Datasets (Already in Project)

| Dataset | Channels | Samples | Key Feature | Relevance |
|---------|----------|---------|-------------|-----------|
| **Hydraulic System** | 17 | 2,205×6k | Real industrial, multi-rate | Primary test |
| **C-MAPSS** | 21 | ~20k cycles | Turbofan simulation, RUL task | Sensor prediction |
| **Paderborn Bearing** | 8 | 256k/file | Real vibration, multi-modal | Periodic signals |
| **AURSAD** | 20 | 6M points | Robot joint states | Control covariates |

See `docs/DATASETS.md` for detailed selection rationale.

---

## Success Criteria

### Minimum Viable Result
- TabPFN-TS matches or beats seasonal naive baseline on 1+ mechanical dataset
- Interpretable: we understand *why* it works (or doesn't)

### Promising Signal
- TabPFN-TS beats ARIMA/Prophet on covariate-informed forecasting
- Adding operating conditions as covariates improves predictions

### Breakthrough
- TabPFN-TS + mechanical domain features approaches specialized SOTA
- Zero-shot transfer works across mechanical systems
- Novel insight: "treating mechanical time series as tabular works because..."

See `docs/BREAKTHROUGH.md` for detailed breakthrough definition.

---

## Timeline Estimate

| Phase | Focus | Deliverable |
|-------|-------|-------------|
| **Tutorial** | Learn TabPFN-TS API | Completed notebooks |
| **Baseline** | Establish comparison points | ARIMA, Prophet, Naive on datasets |
| **Experiments** | Run minimal assessments | Results on 3 datasets |
| **Analysis** | Interpret findings | Report with recommendations |

---

## References

### TabPFN Papers
- [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) (ICLR 2023)
- [From Tables to Time: Extending TabPFN-v2 to Time Series Forecasting](https://arxiv.org/abs/2501.02945) (NeurIPS 2024 Workshops)

### Repositories
- [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) - Main TabPFN repository
- [PriorLabs/tabpfn-time-series](https://github.com/PriorLabs/tabpfn-time-series) - Time series extension

### Benchmarks
- [GIFT-EVAL](https://github.com/ServiceNow/GIFT-EVAL) - TabPFN-TS ranked #1 (May 2025)
