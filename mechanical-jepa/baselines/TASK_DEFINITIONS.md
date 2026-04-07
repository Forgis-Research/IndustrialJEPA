# Baseline Task Definitions — Mechanical Vibration JEPA

Date: 2026-04-07
Author: Overnight V7 research session

## Summary of Research Findings

### Why Raw Signal Forecasting Is Out

Accelerometer vibration signals are broadband stochastic processes. The signal at time t+1 is not predictable
from t (or any history) — it's essentially band-limited noise shaped by structural resonances. Published work
on vibration forecasting unanimously works in the *feature domain* (RMS, kurtosis, band energies, envelope
spectrum peaks), not the raw waveform domain. MOMENT/Chronos/TimesFM do "forecasting" of slowly-varying
sensor channels (temperature, load), not raw vibration.

**Conclusion:** Raw signal forecasting is not a meaningful task for accelerometers. Drop it.

### What Tasks ARE Well-Posed?

1. **Anomaly Classification** (fault type detection)
   - Well-posed: we have clean fault_type labels
   - Community metric: macro F1 (because classes are imbalanced)
   - Cross-domain transfer is where JEPA should shine
   - SOTA in-domain: ~99% accuracy on CWRU (nearly saturated), ~95% on Paderborn
   - Cross-domain SOTA: 60-80% F1 (domain adaptation methods)

2. **One-Class Anomaly Detection** (healthy vs. faulty, no fault labels needed)
   - Well-posed: train on healthy only, test on healthy+faulty
   - Community metric: AUROC (primary), AUPRC (secondary)
   - Uses sources with clear healthy/faulty split
   - SOTA: Deep SVDD, PatchTST reconstruction, IsolationForest ~0.85-0.92 AUROC

3. **Health Indicator Forecasting** (for RUL sources only)
   - Well-posed ONLY for FEMTO (3569 samples), XJTU-SY (1370 samples), IMS (1256 extra shard)
   - Target: RMS or kurtosis trajectory prediction
   - Community metric: RMSE, MAE, R², Spearman correlation with true degradation
   - NOT raw signal forecasting — feature-domain time series
   - Horizon: 1-step, 5-step, 10-step ahead (in units of "measurement windows")
   - Total temporal samples: ~6000 — enough for simple baselines, limited for deep models

4. **RUL Estimation** (point-in-time regression)
   - Given features of current window → predict rul_percent
   - Distinct from forecasting: no temporal context needed, just current state
   - Community metric: RMSE on rul_percent (0-1 scale)
   - Well-posed with FEMTO+XJTU_SY+IMS

## Decisions: What We Are NOT Running

- **Latent forecasting in autoencoder space**: Would require training an autoencoder first, then evaluating
  whether forecasting in that space is meaningful. Too many unknowns; skip for baselines. Note honestly
  that this requires a defined latent space first.
- **Raw signal forecasting**: Not meaningful for accelerometers (see above).
- **DTW-based 1-NN**: Signal lengths vary 2560-484k samples; DTW at this scale is infeasible overnight.
- **Deep SVDD**: Requires significant hyperparameter tuning; include simplified version.

## Final Task Set

### Task 1: Anomaly Classification (Primary)

**Definition:** Given a fixed-length window (16,384 samples at 12,800 Hz = 1.28s), classify fault type.
**Preprocessing:** Resample to 12,800 Hz, take one 16,384-sample window per signal (or multiple).
**Labels:** {healthy, inner_race, outer_race, ball, cage, gear_crack, gear_wear, compound, ...}
**Splits:**
  - In-domain: Per-source stratified 80/20
  - Cross-domain: Train on {CWRU, MAFAULDA, SEU}, test on {Ottawa, Paderborn-subset}
**Metric:** Macro F1 (primary), accuracy
**Sources:** CWRU (16 samples), MAFAULDA (800), SEU (140 bearings-only), Ottawa (180), Paderborn-subset (384)
**Seeds:** 42, 123, 456

### Task 2: One-Class Anomaly Detection (Important)

**Definition:** Train only on healthy windows; detect faulty/degrading windows at test time.
**Anomaly score:** Reconstruction error, distance-based, or density-based
**Preprocessing:** Same as Task 1 (16,384 samples at 12,800 Hz)
**Splits:**
  - In-domain: Train on healthy from source X, test on healthy+faulty from source X (80/20 healthy for train)
  - Cross-domain: Train healthy on FEMTO early-life, test on FEMTO late-life (different bearings)
**Metric:** AUROC (primary), AUPRC, F1@optimal threshold
**Sources:** FEMTO (healthy vs. degrading), CWRU (healthy vs. faulty), MAFAULDA (healthy vs. various faults)
**Seeds:** 42, 123, 456

### Task 3: Health Indicator Forecasting (For JEPA RUL Motivation)

**Definition:** Given sequence of HI values [h(t-k), ..., h(t)], predict h(t+1), ..., h(t+H)
**HI choices:** RMS of vibration signal, kurtosis, band energy (high-frequency band)
**Context length:** k=20 past windows
**Horizon:** H=1, 5, 10
**Preprocessing:** Compute HI for each temporal snapshot; normalize per-episode
**Sources:** FEMTO only (3569 samples in 5 shards + extra), since most complete run-to-failure
**Splits:** Episode-based — use Bearing1_1...Bearing2_3 as train episodes, Bearing3_x as test
**Metric:** RMSE (primary), MAE, R², Spearman with true degradation
**Seeds:** 42, 123, 456

### Task 4: RUL Estimation (Point-in-Time)

**Definition:** Given handcrafted features of current window, predict rul_percent (regression)
**Preprocessing:** Extract time+frequency domain features, normalize per source
**Sources:** FEMTO + XJTU-SY (both have rul_percent)
**Splits:** Episode-split — some full episodes for test, rest for training
**Metric:** RMSE on rul_percent (0-1), MAE, R²
**Seeds:** 42, 123, 456

## Baselines per Task

### Task 1 Baselines (Classification)
- Trivial: majority class, random stratified
- Feature-based: handcrafted features → Logistic Regression, Random Forest, XGBoost, SVM
- Deep: 1D CNN (supervised), 1D ResNet (supervised)
- Transfer comparison: same feature-based, trained source A, tested source B

### Task 2 Baselines (Anomaly Detection)
- Trivial: RMS threshold (μ+3σ), kurtosis threshold, always-healthy predictor
- Feature-based: Isolation Forest, One-Class SVM, LOF, PCA reconstruction error, Mahalanobis distance
- Deep: 1D CNN Autoencoder reconstruction error

### Task 3 Baselines (HI Forecasting)
- Trivial: last-value, moving average (k=5, 10), linear extrapolation
- Standard: Ridge regression on rolling features, ARIMA, Exponential smoothing
- Deep (if time): LSTM

### Task 4 Baselines (RUL Estimation)
- Trivial: constant (mean RUL), linear trending
- Feature-based: Ridge regression, Random Forest, XGBoost on handcrafted features
- Deep (if time): 1D CNN regressor

## Data Sources Summary

| Source      | N samples | Task 1 | Task 2 | Task 3 | Task 4 |
|-------------|-----------|--------|--------|--------|--------|
| CWRU        | ~56       | train  | train  | -      | -      |
| MAFAULDA    | 800       | train  | train  | -      | -      |
| SEU         | 140       | train  | -      | -      | -      |
| FEMTO       | 3569      | -      | test   | train  | train  |
| XJTU-SY     | 1370      | -      | -      | -      | train  |
| IMS         | 1256      | -      | -      | -      | -      |
| Ottawa      | 180       | test   | -      | -      | -      |
| Paderborn   | 384       | test   | -      | -      | -      |

Note: CWRU from extra_cwru_mfpt.parquet (40 CWRU + 20 MFPT), not the full local CWRU dataset.
The full HF bearings dataset also has VBL (800 samples faulty-only) and SCA pulpmill (2663 industrial).

## What a Self-Supervised Model Needs to Beat

To be publishable, JEPA-V7 must beat:
1. **Task 1 cross-domain F1**: Random Forest with handcrafted features (expected ~0.50-0.65 cross-domain)
2. **Task 2 AUROC**: IsolationForest with handcrafted features (expected ~0.82-0.88)
3. **Task 3 RMSE**: Linear extrapolation on HI (expected ~0.02-0.08 normalized RMSE)
4. **Task 4 RMSE**: Random Forest on handcrafted features (expected ~0.12-0.20 on 0-1 scale)

Beating SOTA requires results that reviewers would consider interesting, not just marginal.
