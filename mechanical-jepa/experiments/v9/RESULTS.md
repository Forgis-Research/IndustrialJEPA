# V9 Results: Data-First JEPA

Session: 2026-04-09 (overnight)
Dataset: 31 episodes (16 FEMTO + 15 XJTU-SY), 75/25 episode-based split
V8 baseline: JEPA+LSTM=0.189±0.015, Hybrid JEPA+HC=0.055±0.004, Elapsed time=0.224

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
93.9% of MAFAULDA energy in 0-500Hz vs 10.5% for FEMTO. Instance normalization cannot fix this.

Compatible pretraining group (V9): cwru, femto, xjtu_sy, ims, paderborn, ottawa

## Part C: Pretraining Source Comparison (COMPLETE)

| Config | Windows | Best Epoch | Val Loss | Emb Corr | RMSE ± std | vs V8 |
|--------|:-------:|:----------:|:--------:|:--------:|:----------:|:------:|
| all_8 | 33,939 | 2 | 0.0161 | 0.000 | 0.0852 ± 0.0014 | +54.9% |
| compatible_6 | 28,839 | 3 | 0.0140 | -0.121 | 0.0873 ± 0.0018 | +53.8% |
| bearing_rul_3 | 22,599 | 3 | 0.0161 | -0.123 | 0.0863 ± 0.0020 | +54.4% |

Key insight: "vs V8" improvement is driven by episode count (24 vs 18 train), NOT model improvement.
Apples-to-apples within V9: compatible_6 vs all_8 = -2.4% (not significant, within 1 std).
Early convergence at epoch 2-3 persists even without MAFAULDA.
Compatible filtering improves val_loss (0.0161→0.0140) and adds weak embedding structure.

## Part D: TCN-Transformer (IN PROGRESS — running)

## Part E: Masking Strategy (PENDING)

## Part F: Probabilistic Output (PENDING)

---
NOTE: Parts D-F currently running in background. Results will be appended here.
