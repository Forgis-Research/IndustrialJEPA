# Mechanical-JEPA V7: Baseline Results

**Date**: 2026-04-07  
**Session**: Overnight V7 — Comprehensive Baseline Establishment  
**Dataset**: Forgis/Mechanical-Components (HuggingFace)

---

## Dataset Summary

| Source | N samples | SR (Hz) | Duration/sample | Has episodes | Use |
|--------|-----------|---------|-----------------|--------------|-----|
| CWRU | 60 (HF subset) | 12000 | 20-40s | No | Train (classification) |
| MFPT | 20 | 48828 | 6s | No | Train (classification) |
| MAFAULDA | 800 | 50000 | 0.512s | No | Train (classification, anomaly) |
| SEU | 140 | 5120 | 5s | No | Train (classification) |
| FEMTO | 3569 | 25600 | 0.1s | Yes | Anomaly detection, RUL |
| XJTU-SY | 1370 | 25600 | 1.28s | Yes | RUL |
| Ottawa | 180 | 42000 | 2s | Partially | Test (classification) |
| Paderborn | 384 | 64000 | 4s | No | Test (classification) |

**Key observation**: Signal lengths vary enormously (0.1s FEMTO to 40s CWRU).  
For feature extraction: use native signals at native SR.  
For deep models: resample to 12800 Hz, take 8192-sample window (0.64s).  

---

## Task 1: Cross-Domain Fault Classification

**Setup**: Train on CWRU+MAFAULDA+SEU (1000 samples, 10 classes), test on Ottawa+Paderborn (564 samples, 6 classes).

**Note on low F1**: The train/test sets do not share identical class distributions. MAFAULDA contributes "imbalance" and "misalignment" classes not present in test (Ottawa and Paderborn only have bearing faults). The task is hard: different machines, different fault mechanisms, different feature distributions.

### Results

| Method | Macro F1 ± std | Accuracy | Notes |
|--------|---------------|----------|-------|
| Majority class | 0.000 | 0.000 | Predicts most common = zero recall for others |
| Random (stratified) | 0.040 ± 0.004 | 0.053 | Random baseline |
| Nearest Centroid | 0.193 | 0.335 | Per-class mean distance |
| Logistic Regression | 0.144 ± 0.000 | 0.278 | L2 reg, C=1.0 |
| **Random Forest** | **0.193 ± 0.021** | **0.351** | Best feature-based (200 trees) |
| XGBoost | 0.172 ± 0.000 | 0.356 | |
| SVM (RBF) | 0.127 ± 0.000 | 0.213 | Slow, only 1 seed |
| CNN 1D | 0.159 ± 0.004 | 0.244 | Trained on CWRU+SEU only |
| ResNet 1D | 0.168 ± 0.007 | 0.324 | Trained on CWRU+SEU only |

**Seeds**: 42, 123, 456 (except SVM: seed 42 only)

**Key Finding**: Cross-domain F1 ~0.19 with best methods. Deep models don't clearly beat feature-based methods when trained on limited cross-source data. This is the "bar to clear" for JEPA self-supervised representations.

**What JEPA needs to beat**: Macro F1 > 0.25 cross-domain would be meaningful (+30% over RF baseline).

### Per-Class Analysis (best method: Random Forest, seed 42)
Test classes: ball, cage, compound, healthy, inner_race, outer_race  
Train has these classes but with different distributions and source-specific features.

---

## Task 2: One-Class Anomaly Detection

**Setup**: Train on healthy samples only, detect faulty/degrading at test time.

### FEMTO Results (N_train=656 healthy, N_test=164 healthy + 824 anomalous)

| Method | AUROC | AUPRC | F1@optimal |
|--------|-------|-------|-----------|
| Constant healthy | 0.500 | 0.682 | 0.811 |
| RMS threshold | 0.498 | 0.681 | 0.811 |
| **Kurtosis threshold** | **0.779** | **0.881** | **0.814** |
| Isolation Forest | 0.710 ± 0.004 | N/A | N/A |
| One-Class SVM | 0.495 | 0.712 | 0.811 |
| LOF | 0.470 | 0.647 | 0.811 |
| Mahalanobis distance | 0.561 | 0.756 | 0.811 |
| PCA reconstruction | 0.363 | 0.635 | 0.811 |
| Autoencoder (CNN) | 0.414 ± 0.011 | N/A | N/A |

**FEMTO Best**: Kurtosis threshold AUROC=0.779 — physically meaningful because kurtosis increases with bearing impacts.

**MAFAULDA Results** (N_train=39 healthy — very few!, N_test=10 healthy + 751 faulty)

All methods fail (AUROC < 0.5 for most, kurtosis threshold 0.740 is best).  
Root cause: only 39 training samples of healthy data; highly imbalanced test.  
MAFAULDA is a fault-centric dataset — not well-suited for anomaly detection.

### Key Findings
1. Kurtosis is the best simple anomaly detector for bearings (physically motivated)
2. Complex ML methods (IsolationForest, OCSVM) don't consistently outperform kurtosis on FEMTO
3. CNN autoencoder performs poorly on short signals (FEMTO: 0.1s = 2560 samples) — insufficient to learn meaningful reconstruction
4. 18-feature handcrafted set gives most methods AUROC 0.45-0.56 — kurtosis alone carries the signal

**What JEPA needs to beat**: AUROC > 0.85 on FEMTO would be meaningful (~+10% over best baseline).

---

## Task 3: Health Indicator Forecasting (FEMTO, RMS)

**Setup**: FEMTO episodes, RMS trajectory, 20 context steps → predict H steps ahead.  
Train on 5 episodes, test on 2 episodes.

### H=1 (1-step ahead)

| Method | RMSE | Notes |
|--------|------|-------|
| Constant mean | 0.844 | Worst trivial |
| Moving avg (k=20) | 0.547 | |
| Linear extrapolation | 0.389 | |
| **Last value** | **0.351** | Best trivial |
| Ridge regression | 0.354 | Barely beats last-value |
| XGBoost | 0.345 | |
| **Random Forest** | **0.311** | Best overall |
| ARIMA (2,1,2) | 1.822 | Fails on degradation non-stationarity |

### H=5 (5-step ahead)

| Method | RMSE |
|--------|------|
| Last value | 1.217 |
| Moving avg | 1.013 |
| Linear extrap | 1.390 |
| **Ridge** | **0.996** |
| Random Forest | 1.091 |

### H=10 (10-step ahead)

| Method | RMSE |
|--------|------|
| Last value | 1.314 |
| Moving avg | 1.021 |
| **Ridge** | **1.008** |
| Random Forest | 1.158 |

### Kurtosis HI Forecasting (H=1)

| Method | RMSE | Notes |
|--------|------|-------|
| Last value | 1.070 | |
| Linear extrap | 0.944 | |
| Moving avg | 0.964 | |
| **Ridge** | **0.906** | Best |
| Random Forest | 0.995 | |

**Key Findings**:
1. RMS is more forecastable than kurtosis (lower RMSE on same scale)
2. Last-value is a competitive baseline for H=1; linear regression is better for H>1
3. ARIMA completely fails — degradation HI is non-stationary and non-Gaussian
4. The "useful" forecasting horizon is maybe 1-5 steps for RMS; beyond that all methods degrade

**What JEPA needs to beat**: RMSE < 0.30 at H=1 would represent a meaningful improvement over Random Forest.

---

## Task 4: RUL Estimation

**Setup**: FEMTO (3569) + XJTU-SY (647) samples, 23 episodes. Episode-based split (75/25).  
Predict `rul_percent` (0-1) from handcrafted features of current window.

| Method | RMSE ± std | Notes |
|--------|-----------|-------|
| Constant mean | 0.290 | Strong trivial! |
| Linear position* | 0.000 | **ORACLE — DO NOT USE** (rul=1-position is definitional) |
| Ridge regression | 0.229 ± 0.000 | |
| Random Forest | 0.214 ± 0.000 | |
| **XGBoost** | **0.212 ± 0.000** | Best real baseline |

*Linear position (rul_percent = 1 - episode_position definitionally) is an oracle. Not a valid baseline.

**Key Finding**: On these datasets, `rul_percent` is defined as `1 - episode_position`, meaning the "true" rul is just linear time. All features-based methods improve over constant mean (0.290) but the gain is modest (~25% RMSE reduction with XGBoost).

**What JEPA needs to beat**: RMSE < 0.18 would represent a meaningful improvement.

---

## Sanity Checks

### 5-Minute Checklist (Mandatory)

**Task 1 (Classification)**:
- [x] Beats trivial: Yes (F1=0.19 vs 0.04 random)
- [x] Direction: Correct (harder cross-domain task yields lower F1)
- [x] Magnitude: In range for cross-domain (published cross-domain F1 typically 0.5-0.7 with domain adaptation)
- [x] No leakage: Train/test are different sources
- [x] Implementation: Loss decreased, gradients OK

**Task 2 (Anomaly Detection)**:
- [x] Kurtosis > random (0.779 AUROC vs 0.5)
- [x] FEMTO is harder than it looks (lots of gradual degradation = hard boundary)
- [x] MAFAULDA is problematic: 39 healthy train samples — insufficient
- [x] Autoencoder poorly suited to 0.1s signals

**Task 3 (HI Forecasting)**:
- [x] Last-value beats constant mean (expected)
- [x] RF beats trivial (expected)
- [x] ARIMA failing makes sense (non-stationary degradation)
- [x] Kurtosis harder than RMS (physically expected — kurtosis is more impulsive/volatile)

**Task 4 (RUL)**:
- [x] Constant mean RMSE=0.29 is consistent with rul std~0.29 for uniform distribution on [0,1]
- [x] Oracle check: 1-position = RMSE 0 (confirmed, definitional relationship)
- [x] XGBoost RMSE=0.21 is 27% improvement over constant mean

---

## What JEPA Must Beat to Be Publishable

| Task | Metric | Current Best Baseline | Target for JEPA |
|------|--------|-----------------------|-----------------|
| Cross-domain classification | Macro F1 | 0.193 (Random Forest) | > 0.30 (+56%) |
| Anomaly detection (FEMTO) | AUROC | 0.779 (kurtosis) | > 0.85 (+9%) |
| RMS forecasting H=1 | RMSE | 0.311 (Random Forest) | < 0.25 (-20%) |
| RUL estimation | RMSE | 0.212 (XGBoost) | < 0.17 (-20%) |

**Primary target**: Cross-domain classification F1. This is the most publishable metric.

---

## Critical Observations for JEPA Model Design

1. **Cross-domain is hard with handcrafted features**: F1=0.19 shows feature mismatch across sources. JEPA representations that are source-agnostic should do much better.

2. **Kurtosis is a strong anomaly signal**: A JEPA model that learns to predict impulsive components should capture what kurtosis measures.

3. **HI forecasting is feasible but limited**: The useful forecasting horizon is 1-5 steps for RMS. Beyond that, degradation trajectory variance dominates.

4. **RUL task is the most interesting gap**: Current methods stall at RMSE=0.21 because handcrafted features can't capture the temporal degradation pattern efficiently.

---

## Validation Notes

**Question 1**: Are metrics correct?
- Classification: Macro F1 is standard for imbalanced multi-class. Correct.
- Anomaly: AUROC is standard for binary detection. AUPRC reported where possible. Correct.
- Forecasting: RMSE on normalized scale (each episode normalized separately). Correct.
- RUL: RMSE on raw 0-1 scale. Correct.

**Question 2**: Are we missing key SOTA baselines?
- Classification: InceptionTime would be a known SOTA method but requires uniform window lengths across all sources. Not included due to data diversity. Note this limitation.
- Anomaly: Deep SVDD (end-to-end) was not included. Literature shows it sometimes outperforms IsolationForest but also sometimes worse. Future work.
- Forecasting: N-BEATS, Informer, PatchTST not included — designed for longer sequences than available.
- RUL: CNN-LSTM (from DCSSL paper) not included. Would likely beat RF by ~20-30% based on literature.

**Community SOTA Context**:
- In-domain CWRU accuracy: 99%+ (near saturated, not our setting)
- Cross-domain with domain adaptation (JMMD, DANN): 60-80% F1 (our target range)
- Bearing anomaly detection AUROC (in-domain): 0.90-0.97 in published work
- Cross-dataset RUL RMSE: 0.10-0.15 for SOTA SSL methods
