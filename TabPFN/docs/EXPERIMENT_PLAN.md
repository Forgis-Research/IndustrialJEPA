# TabPFN-TS Experiment Plan for Mechanical Systems

**Objective**: Assess whether TabPFN-TS provides useful forecasting for mechanical system time series.

---

## Critical Paper Findings (Must Read First!)

Before running experiments, understand these key limitations from the TabPFN-TS paper analysis:

### ⚠️ Linear Trend Extrapolation FAILS

**This is the most critical limitation for mechanical systems.**

TabPFN-TS does **NOT** extrapolate simple linear trends well:
```
y(t) = α·t + β  →  Prediction flattens out instead of continuing
```

**Impact on experiments**:
- C-MAPSS degradation trajectories may not extrapolate correctly
- RUL-style predictions will be problematic
- Any sensor with monotonic degradation trend is at risk

**Mitigation strategies**:
1. **Detrend first**: Fit linear trend, forecast residuals, add trend back
2. **Focus on periodic components**: Use for vibration/rotation patterns, not degradation
3. **Exponential trends work better**: Curiously, exponential decay/growth is handled better

### ✅ What Works Well

- **Pure periodic signals**: Almost perfect (rotation frequencies, machine cycles)
- **Complex waveforms**: Multiple harmonics, non-sinusoidal patterns
- **Trend + periodicity combinations**: Even multiplicative
- **Covariates**: Native support is a key advantage

### Technical Configuration (from paper)

```python
# Recommended settings
max_context = 4096        # Sweet spot for context length
k_seasonalities = 5       # Top-5 FFT peaks for auto-seasonal features

# For 100 Hz data: 4096 samples = ~40 seconds of history
# For 1 Hz data: 4096 samples = ~68 minutes of history
```

See `docs/PAPER_INSIGHTS.md` for complete paper analysis.

---

## Phase 1: Validation (Quick Sanity Checks)

**Goal**: Confirm TabPFN-TS works and establish baseline expectations.

### Exp 1.1: Synthetic Mechanical Signal
- **Data**: Simulated vibration (periodic + trend + noise)
- **Task**: 10-step ahead forecasting
- **Baseline**: Naive (last value), Seasonal naive
- **Success criterion**: TabPFN-TS RMSE < Naive RMSE

### Exp 1.2: Basic Hydraulic Forecasting
- **Data**: UCI Hydraulic PS1 (single cycle)
- **Task**: 20% holdout forecasting
- **Baseline**: Naive, Moving average
- **Success criterion**: TabPFN-TS competitive or better

### Exp 1.3: Covariate Sanity Check
- **Data**: Synthetic covariate-dependent signal
- **Task**: Forecast with/without covariates
- **Success criterion**: Covariates improve RMSE by >10%

---

## Phase 2: Core Experiments

**Goal**: Systematic evaluation on real mechanical datasets.

### Exp 2.1: Hydraulic System — Single Sensor Forecasting

| Sensor | Type | Rate | Expected Difficulty |
|--------|------|------|---------------------|
| PS1-PS6 | Pressure | 100 Hz | Easy (smooth signals) |
| FS1-FS2 | Flow | 10 Hz | Medium |
| TS1-TS4 | Temperature | 1 Hz | Hard (slow dynamics) |
| VS1 | Vibration | 1 Hz | Medium |

**Protocol**:
1. Load sensor data for 100 random cycles
2. Train/test split: 80/20 within each cycle
3. Measure RMSE, MAE, and skill score vs naive
4. Report mean ± std across cycles

**Baselines**:
- Last value naive
- Seasonal naive (period = estimated from data)
- ARIMA(p,d,q) with auto-selection
- Prophet (if applicable)

### Exp 2.2: Hydraulic System — Cross-Sensor Conditioning

Test if one sensor improves prediction of another:

| Target | Covariate(s) | Hypothesis |
|--------|-------------|------------|
| FS1 (flow) | PS1-PS6 (pressure) | Pressure drives flow |
| TS1 (temp) | EPS1 (motor power) | Power → heat |
| VS1 (vibration) | All pressures | System state affects vibration |

**Protocol**:
1. Same setup as Exp 2.1
2. Compare: TabPFN-TS alone vs TabPFN-TS + covariate
3. Test with 1, 2, 3+ covariates
4. Report relative improvement

### Exp 2.3: C-MAPSS — Covariate-Informed Forecasting

**Motivation**: C-MAPSS has strong covariates (operating settings).

**⚠️ CRITICAL WARNING**: C-MAPSS sensors exhibit degradation trends. Per paper analysis, TabPFN-TS does NOT extrapolate linear trends well. This experiment focuses on **covariate benefit**, not raw forecasting performance.

**Setup**:
- Sensor: T24 (LPC outlet temperature)
- Covariates: setting1, setting2, setting3
- Units: Random 20 units from FD001

**Protocol**:
1. For each unit: forecast last 30% of cycles
2. Compare: No covariates vs operating settings as covariates
3. Baseline: Linear extrapolation (degradation trend)
4. **NEW**: Also test with detrended data (remove linear trend, forecast residuals)

**Success criterion**: Covariates provide >5% improvement.

**Expected outcome**: TabPFN-TS may struggle with degradation trends but covariates should still help capture operating-condition-dependent variations.

### Exp 2.4: C-MAPSS — Cross-Condition Transfer

Can a model trained on one condition generalize?

**⚠️ NOTE**: Given linear trend limitations, focus on whether *patterns around the trend* transfer, not absolute trajectory forecasting. Consider detrending as preprocessing.

| Train | Test | Hypothesis |
|-------|------|------------|
| FD001 (1 condition, 1 fault) | FD002 (6 conditions, 1 fault) | Operating variety hurts |
| FD001 | FD003 (1 condition, 2 faults) | Multiple faults harder |

**Protocol**:
1. Train TabPFN-TS on FD001 training set
2. Evaluate on FD002/FD003 test sets
3. Compare to models trained on target dataset
4. **NEW**: Try both raw and detrended versions

**Interpretation guidance**: If detrended versions transfer well but raw versions don't, this confirms the trend limitation is the bottleneck, not the general pattern learning.

---

## Phase 3: Advanced Experiments (If Phase 2 Promising)

### Exp 3.1: Paderborn Bearing — High-Frequency Vibration

**Challenge**: Very high sampling rate (64 kHz) requires careful handling.

**Setup**:
- Subsample to 1 kHz or 100 Hz
- Predict 10-100 samples ahead
- Compare healthy vs faulty bearings

**Questions**:
- Does TabPFN-TS capture vibration periodicity?
- Are prediction residuals higher for faulty bearings?

### Exp 3.2: Multi-Sensor Joint Forecasting

Instead of single sensor:
1. Stack multiple sensors as features
2. Predict all simultaneously
3. Compare to individual forecasts

### Exp 3.3: Anomaly Detection via Residuals

**Idea**: Large prediction errors indicate abnormal operation.

**Protocol**:
1. Train on healthy cycles only
2. Predict on mixed healthy + faulty cycles
3. Compute residual = |true - predicted|
4. Evaluate: Do faulty cycles have higher residuals?

**Metric**: AUC-ROC for fault detection

---

## Baselines

### Naive Methods
```python
# Last value
pred_naive = np.full(horizon, y_train[-1])

# Seasonal naive (period P)
pred_seasonal = np.tile(y_train[-P:], horizon // P + 1)[:horizon]

# Moving average
pred_ma = np.full(horizon, np.mean(y_train[-window:]))
```

### Statistical Methods
```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(y_train, order=(5, 1, 0))
fitted = model.fit()
pred_arima = fitted.forecast(steps=horizon)
```

### Prophet (Optional)
```python
from prophet import Prophet

df = pd.DataFrame({'ds': dates, 'y': y_train})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=horizon)
pred_prophet = model.predict(future)['yhat'].values[-horizon:]
```

---

## Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| RMSE | √(mean((y - ŷ)²)) | General comparison |
| MAE | mean(|y - ŷ|) | Robust to outliers |
| MAPE | mean(|y - ŷ|/|y|) × 100 | Percentage error |
| Skill Score | 1 - RMSE_model / RMSE_naive | Relative improvement |

**Primary metric**: RMSE (comparable to literature)
**Secondary metric**: Skill score (shows improvement over naive)

---

## Reporting Template

For each experiment, report:

```
## Exp X.Y: [Name]

**Dataset**: [Name], [samples], [channels]
**Task**: [Forecast horizon], [train/test split]

### Results

| Method | RMSE | MAE | Skill Score |
|--------|------|-----|-------------|
| TabPFN-TS | X.XXX | X.XXX | X.XX |
| TabPFN-TS (cov) | X.XXX | X.XXX | X.XX |
| ARIMA | X.XXX | X.XXX | X.XX |
| Naive | X.XXX | X.XXX | 0.00 |

### Observations
- [Key finding 1]
- [Key finding 2]

### Conclusion
[What we learned, next steps]
```

---

## Timeline

| Week | Focus | Experiments |
|------|-------|-------------|
| 1 | Setup & Validation | 1.1, 1.2, 1.3 |
| 2 | Hydraulic | 2.1, 2.2 |
| 3 | C-MAPSS | 2.3, 2.4 |
| 4 | Analysis & Writing | Report |
| 5+ | Advanced (optional) | 3.1, 3.2, 3.3 |

---

## Code Structure

```
experiments/
├── exp01_synthetic.py      # Phase 1 validation
├── exp02_hydraulic.py      # Single sensor forecasting
├── exp03_hydraulic_cov.py  # Cross-sensor conditioning
├── exp04_cmapss.py         # Covariate forecasting
├── exp05_cmapss_transfer.py # Cross-condition transfer
├── baselines/
│   ├── naive.py
│   ├── arima_baseline.py
│   └── prophet_baseline.py
└── utils/
    ├── data_loading.py
    ├── metrics.py
    └── visualization.py
```
