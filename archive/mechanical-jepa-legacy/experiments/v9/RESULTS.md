# V9 Results: Data-First JEPA

Session: 2026-04-09 (overnight)
Dataset: 31 episodes (16 FEMTO + 15 XJTU-SY), 75/25 episode-based split
V8 baselines: JEPA+LSTM=0.189±0.015, Hybrid JEPA+HC=0.055±0.004, Elapsed time=0.224

## Part B: Dataset Compatibility (COMPLETE)

Key findings from spectral analysis of 300 windows per source:

| Source | Centroid (Hz) | Kurtosis | KL vs FEMTO | Verdict |
|--------|:------------:|:-------:|:-----------:|:-------:|
| femto | 2453 ± 564 | 0.99 ± 2.02 | 0.28 | COMPATIBLE (reference) |
| xjtu_sy | 1987 ± 785 | 0.16 ± 0.46 | 0.28 | COMPATIBLE (reference) |
| cwru | 2699 ± 695 | 4.57 ± 6.18 | 1.47 | COMPATIBLE |
| ims | 2827 ± 426 | 0.60 ± 2.18 | 0.73 | COMPATIBLE |
| paderborn | 3323 ± 642 | 2.40 ± 3.42 | 0.67 | COMPATIBLE |
| ottawa | 1074 ± 649 | 3.30 ± 6.67 | 0.99 | COMPATIBLE |
| mfpt | 2753 ± 440 | 12.39 ± 16.99 | 0.54 | MARGINAL |
| **mafaulda** | **173 ± 50** | 2.91 ± 1.72 | **3.04** | **INCOMPATIBLE** |

Root cause of V8 instability: MAFAULDA spectral centroid 173Hz vs FEMTO 2453Hz (14x difference).

## Part C: Pretraining Source Comparison (COMPLETE)

| Config | Windows | Best Epoch | Val Loss | Emb Corr | RMSE ± std | vs V8 |
|--------|:-------:|:----------:|:--------:|:--------:|:----------:|:------:|
| all_8 | 33,939 | 2 | 0.0161 | 0.000 | 0.0852 ± 0.0014 | +54.9% |
| compatible_6 | 28,839 | 3 | 0.0140 | -0.121 | 0.0873 ± 0.0018 | +53.8% |
| bearing_rul_3 | 22,599 | 3 | 0.0161 | -0.123 | 0.0863 ± 0.0020 | +54.4% |

Key insight: "vs V8" driven by episode count (24 vs 18), not model improvement.

## Part D: TCN-Transformer (COMPLETE)

| Method | RMSE | ±std | Notes |
|--------|:----:|:----:|:------|
| TCN-Transformer+HC (D.1) | 0.1642 | 0.0023 | Supervised baseline |
| JEPA+TCN-Transformer (D.2) | 0.1395 | 0.0060 | Overfits with 24 eps |
| JEPA+Deviation (D.3) | 0.1795 | 0.0062 | Contaminated baseline |
| JEPA+HC+Deviation (D.4) | SKIPPED | — | D.3 failed, per plan |

## Part E: Masking Strategy (COMPLETE)

| Config | Best Epoch | Val Loss | Emb Corr | RMSE ± std | vs C.2 |
|--------|:----------:|:--------:|:--------:|:----------:|:------:|
| C.2 random masking | 3 | 0.0140 | -0.121 | 0.0873 ± 0.0018 | baseline |
| E.1 block masking | 4 | 0.0173 | 0.154 | 0.0886 ± 0.0049 | -1.5% |
| E.2 dual-channel | 4 | 0.0155 | 0.239 | 0.1119 ± 0.0057 | -28.2% |

## Part F: Probabilistic Output (COMPLETE)

| Method | RMSE ± std | PICP@90% | MPIW | Notes |
|--------|:----------:|:--------:|:----:|:-----:|
| Deterministic LSTM (C.2) | 0.0873 ± 0.0018 | N/A | N/A | Baseline |
| Heteroscedastic LSTM (F.1) | 0.0868 ± 0.0023 | 0.910 | 0.2414 | Gaussian NLL |
| Ensemble F.2 (5 seeds) | 0.0873 ± 0.0018 | N/A | — | Cross-seed uncertainty |

## Complete Results Table

| Exp | Method | RMSE | ±std | vs Elapsed | vs V8 JEPA |
|-----|--------|:----:|:----:|:----------:|:----------:|
| baseline | Elapsed time | 0.224 | — | 0% | — |
| baseline | V8 JEPA+LSTM | 0.189 | 0.015 | +15.8% | 0% |
| baseline | V8 Hybrid JEPA+HC | 0.055 | 0.004 | +75.5% | +70.9% |
| C.1 | V9 JEPA+LSTM (all_8) | 0.0852 | 0.0014 | +62.0% | +54.9% |
| C.2 | V9 JEPA+LSTM (compat_6) | 0.0873 | 0.0018 | +61.0% | +53.8% |
| C.3 | V9 JEPA+LSTM (bearing_3) | 0.0863 | 0.0020 | +61.5% | +54.4% |
| D.1 | TCN-Transformer+HC | 0.1642 | 0.0023 | +26.7% | 13.2% worse |
| D.2 | JEPA+TCN-Transformer | 0.1395 | 0.0060 | +37.7% | 26.2% worse |
| D.3 | JEPA+Deviation | 0.1795 | 0.0062 | +19.9% | 5.0% worse |
| D.4 | JEPA+HC+Deviation | SKIPPED | — | — | — |
| E.1 | JEPA[block]+LSTM | 0.0886 | 0.0049 | +60.4% | +53.1% |
| E.2 | JEPA[dual]+LSTM | 0.1119 | 0.0057 | +50.0% | +40.8% |
| F.1 | JEPA+Prob-LSTM | 0.0868 | 0.0023 | +61.2% | +54.1% |

## Published SOTA Comparison

| Reference | Method | Dataset | Metric | Value |
|-----------|--------|---------|--------|:-----:|
| CNN-GRU-MHA (2024) | Supervised CNN | FEMTO only | nRMSE | 0.044 |
| DCSSL (2024) | SSL+RUL | FEMTO only | RMSE | 0.131 |
| V8 (ours) | Hybrid JEPA+HC | FEMTO+XJTU | RMSE | 0.055 |
| V9 (ours) | Best method | FEMTO+XJTU | RMSE | 0.0852 |

Note: CNN-GRU-MHA uses FEMTO only. Our protocol uses 7 held-out test episodes (FEMTO+XJTU mixed).
