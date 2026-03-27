# TabPFN-TS Paper Analysis: Key Insights for Mechanical Systems

**Paper**: "From Tables to Time: How TabPFN-v2 Outperforms Specialized Time Series Forecasting Models"
**Authors**: Shi Bin Hoo, Samuel Müller, David Salinas, Frank Hutter (Prior Labs / University of Freiburg)
**arXiv**: 2501.02945v3 (May 2025)

---

## Executive Summary

TabPFN-TS achieves **#1 on GIFT-Eval** (97 benchmarking tasks) for probabilistic forecasting despite:
- Only **11M parameters** (vs TimesFM-2.0's 500M)
- **No time series pretraining** — TabPFN-v2 was trained only on synthetic tabular data
- Simple feature engineering approach

**Key insight**: Treating time series as tabular regression with the right features works surprisingly well.

---

## How TabPFN-TS Works

### Time Series → Tabular Conversion

The method does NOT use autoregressive features (lags, moving averages). Instead:

```
X_t = Φ_cal(t) ⊕ Φ_auto(t) ⊕ Φ_index(t) ∈ R^28
```

| Feature Type | Dimensions | Description |
|--------------|------------|-------------|
| **Calendar Features** | 17 | sin/cos encoding of 8 calendar cycles + year |
| **Auto-Seasonal Features** | 10 (2×k) | Top-k=5 FFT-detected periodicities |
| **Running Index** | 1 | Simple timestep counter (0, 1, 2, ...) |

### Calendar Features (17 dims)
Encodes 8 cyclic components with sin/cos:
- second_of_minute (period 60)
- minute_of_hour (period 60)
- hour_of_day (period 24)
- day_of_week (period 7)
- day_of_month (period 30.5)
- day_of_year (period 365)
- week_of_year (period 52)
- month_of_year (period 12)
- Plus: year (raw value)

### Automatic Seasonal Features (Algorithm 1)
1. **Detrend** via linear regression
2. **Apply Hann window** + zero-pad
3. **FFT** to find spectral peaks
4. **Select top-k** frequencies by magnitude
5. **Encode** as sin(2πf_i·t), cos(2πf_i·t)

**Critical for mechanical systems**: This auto-detects rotation frequencies, machine cycles, etc.

### Running Index
Simple `t = 0, 1, 2, ...` enables trend extrapolation.

---

## Performance Results

### GIFT-Eval Benchmark (97 tasks)

| Model | Params | WQL Rank | Relative WQL | Relative MASE |
|-------|--------|----------|--------------|---------------|
| **TabPFN-TS** | **11M** | **#1 (3.68)** | **0.460** | 0.692 |
| TimesFM-2.0 | 500M | #2 (3.87) | 0.465 | **0.680** |
| Chronos-Bolt-Base | 205M | #3 (3.91) | 0.485 | 0.725 |
| Chronos-Bolt-Small | 48M | #4 (4.50) | 0.487 | 0.738 |
| Seasonal Naive | - | #11 (9.64) | 1.000 | 1.000 |

**Key**: TabPFN-TS is 45× smaller than TimesFM-2.0 but performs comparably or better.

---

## Critical Ablations

### 1. Featurization Impact (Figure 3)

| Features | Relative WQL | Relative MASE |
|----------|--------------|---------------|
| Index only | 0.663 | 1.014 |
| Auto-Seasonal only | 0.644 | 0.964 |
| Calendar only | 0.466 | 0.714 |
| Index + Calendar | 0.452 | 0.685 |
| Index + Auto-Seasonal | 0.456 | 0.683 |
| **All combined** | **0.443** | **0.668** |

**Takeaway**: All feature types contribute. Auto-seasonal features almost match calendar features when combined with index.

### 2. How TabPFN Learns Harmonics (Section 5.2)

Given sin(x) and cos(x) as features, TabPFN-v2 can approximate sin(nx) for n up to ~24:

```
sin(4x) = 4·sin(x)·cos³(x) - 4·sin³(x)·cos(x)
```

This is consistent with Chebyshev polynomial expansion. **Implication**: Providing base-frequency features enables higher harmonic reconstruction without explicit frequency inputs.

### 3. CatBoost Comparison (Section A.7)

| Model | Relative MASE |
|-------|---------------|
| TabPFN-TS | 0.663 |
| CatBoost-TS (same features) | 0.782 |

**Takeaway**: TabPFN-v2's pretrained generalization matters, not just the featurization.

### 4. Context Length (Section A.6)

| Context Length | Relative WQL | Relative MASE |
|----------------|--------------|---------------|
| 1024 | 0.462 | 0.715 |
| 2048 | 0.444 | 0.681 |
| **4096** | **0.440** | 0.666 |
| 10000 | 0.451 | **0.654** |

**Takeaway**: 4096 timesteps is a good tradeoff. More context helps MASE slightly but can hurt WQL.

---

## Known Limitations (Critical for Mechanical Systems!)

### 1. Linear Trend Extrapolation FAILS (Figure 6b)
**This is the biggest weakness.**

TabPFN-TS does **not** extrapolate simple linear trends:
```
y(t) = α·t + β  →  Prediction flattens out
```

**Impact on mechanical systems**:
- Degradation trends (RUL prediction) will be problematic
- C-MAPSS sensor trajectories may not extrapolate well
- Need to handle trends separately or use detrending

### 2. Exponential Trends Work Better
Curiously, exponential growth/decay is handled better than linear.

### 3. Pure Periodicity Works Almost Perfectly
Clean periodic signals (sin waves, rotational harmonics) are well-captured.

### 4. Trend + Periodicity Combinations Work
Even multiplicative combinations (trend × seasonal) are handled.

### 5. Inference Speed
TabPFN-TS is significantly slower than Chronos-Bolt (Table 3):
- Electricity 15T/long: 17,811s vs 337s (53× slower)
- Per-series fitting overhead

---

## Implications for Mechanical Systems

### What Should Work Well

1. **Periodic Signals** (bearings, rotating machinery)
   - Vibration patterns from rotation
   - Auto-seasonal features will detect rotation frequency

2. **Operating Condition Changes**
   - Speed changes create different periodicities
   - Auto-FFT adapts to dominant frequencies

3. **Complex Waveforms**
   - Composite signals (multiple harmonics)
   - Non-sinusoidal but periodic patterns

### What May NOT Work Well

1. **Linear Degradation Trends**
   - RUL-style monotonic degradation
   - C-MAPSS sensor trajectories
   - **Mitigation**: Detrend before forecasting, add trend back

2. **Calendar Features Irrelevant**
   - Mechanical systems don't care about day-of-week
   - Need machine-specific cycle features instead

3. **Very High Frequencies**
   - 64 kHz vibration needs heavy subsampling
   - May lose important high-frequency content

### Recommended Adaptations for Mechanical Systems

1. **Replace Calendar Features** with machine-specific features:
   - Rotation cycle position (if known)
   - Operating mode indicator
   - Load setting / speed setting

2. **Keep Auto-Seasonal Features** — they detect rotation frequencies

3. **Keep Running Index** — enables trend modeling

4. **Handle Degradation Separately**:
   - Fit linear trend first, forecast residuals
   - Or: use degradation rate as a covariate

5. **Subsample Appropriately**:
   - 4096 context points is the sweet spot
   - For 100 Hz data: ~40 seconds of history

---

## Technical Configuration

### Recommended Setup (from paper)

```python
# Model checkpoint
checkpoint = "2noar4o2"  # Best performing

# Context length
max_context = 4096  # Good tradeoff

# Auto-seasonal features
k_seasonalities = 5  # Top-5 FFT peaks

# Preprocessing
# - z-normalization on targets
# - Ensemble with power-transformed targets (Box-Cox)
# - Drop missing values from training set
```

### API Usage Pattern

```python
from tabpfn_ts import TabPFNForecaster

# Basic usage
forecaster = TabPFNForecaster(horizon=H)
forecaster.fit(y_train)
predictions = forecaster.predict()

# With covariates (external features)
forecaster.fit(y_train, X=X_train)
predictions = forecaster.predict(X=X_test)

# Probabilistic output
quantiles = forecaster.predict_quantiles(quantiles=[0.1, 0.5, 0.9])
```

---

## Comparison to Other Foundation Models

| Aspect | TabPFN-TS | Chronos | TimesFM |
|--------|-----------|---------|---------|
| Architecture | Tabular transformer | T5-style | Decoder-only |
| Pretraining data | Synthetic tabular | Real + synthetic TS | Real TS |
| Covariates | Native support | Requires ChronosX | Limited |
| Probabilistic | Native (Riemann dist.) | Quantile heads | Quantile heads |
| Parameters | 11M | 9M-205M | 500M |
| Zero-shot | Yes | Yes | Yes |

**TabPFN-TS advantages**:
- Native covariate support (critical for mechanical systems with operating conditions)
- Native probabilistic outputs (better calibrated)
- Smallest model with competitive performance

---

## Key Equations

### Calendar Feature Encoding
```
Φ_cal(t) = (cos(2πt/P₁), sin(2πt/P₁), ..., cos(2πt/P₈), sin(2πt/P₈), year(t))
```

### Automatic Seasonal Features
```
Φ_auto(t) = (cos(2πf₁t), sin(2πf₁t), ..., cos(2πf_kt), sin(2πf_kt))
```
where f₁...f_k are top-k FFT-detected frequencies.

### Harmonic Reconstruction (why sin/cos features work)
```
sin(nx) = polynomial(sin(x), cos(x))
```

---

## References

- Paper: https://arxiv.org/abs/2501.02945
- Code: https://github.com/PriorLabs/tabpfn-time-series
- TabPFN-v2: https://github.com/PriorLabs/TabPFN
- GIFT-Eval: https://github.com/ServiceNow/GIFT-EVAL
