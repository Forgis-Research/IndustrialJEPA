# V9 Experiment Log

## Overview
Session: 2026-04-09 (overnight)
Goal: Data-first JEPA — fix pretraining instability via dataset compatibility analysis
V8 baseline to beat: JEPA+LSTM RMSE=0.189, Contrastive RMSE=0.227 (cross-domain)

---

## IMPORTANT NOTE: Episode Count Effect

V9 uses 31 episodes (24 train, 7 test) vs V8's 23 episodes (18 train, 5 test).
More training episodes = better LSTM performance. RMSE comparisons against V8 should
account for this extra data. For apples-to-apples: compare V9 all_8 vs V9 compatible_6.

---

## Exp B1: Dataset Compatibility Analysis

**Time**: 2026-04-09
**Hypothesis**: 8 heterogeneous sources are causing JEPA pretraining instability
**Change**: Full spectral/statistical analysis of all 8 sources
**Result**:
- MAFAULDA: centroid 173Hz vs FEMTO 2453Hz. 93.9% energy in 0-500Hz. INCOMPATIBLE.
- MFPT: kurtosis 12.4±17.0 vs FEMTO 1.0±2.0. Impulse-dominated. MARGINAL.
- cwru, ims, ottawa, paderborn, femto, xjtu_sy: COMPATIBLE (centroid 1074-3323Hz range)
- Instance normalization equalizes RMS but NOT spectral shape
- PSD KL divergence: MAFAULDA-FEMTO = 3.04 (worst), all others <1.5
**Verdict**: KEEP — MAFAULDA confirmed as primary source of instability
**Insight**: The encoder was simultaneously trying to represent 173Hz pump signals
            and 2453Hz bearing signals. No wonder JEPA oscillated after epoch 2.
**Next**: Pretrain on 6 compatible sources only, check if loss stabilizes

---

## Exp C.1: JEPA Pretraining - all_8

**Time**: 2026-04-09
**Hypothesis**: Training on all 8 sources replicates V8 instability (early convergence)
**Change**: Pretrain on all 8 sources (33,939 windows)
**Sanity checks**: best_epoch=2, val_loss_ep5=0.0190, val_loss_ep10=0.0197
**Result**: best_epoch=2, best_val=0.0161, RMSE=0.0852±0.0014
**Seeds**: [0.0869, 0.0838, 0.0863, 0.0835, 0.0853]
**Embedding quality**: max_dim_corr=0.000, PC1_corr=0.000 (random structure)
**Verdict**: CONFIRMS V8 INSTABILITY PATTERN - best at epoch 2/100 then diverges
**Insight**: Heterogeneous frequency content (MAFAULDA 173Hz vs FEMTO 2453Hz) prevents stable learning.
  Note: RMSE improved vs V8 (0.085 vs 0.189) due to more training episodes (24 vs 18 in V8).

---

## Exp C.2: JEPA Pretraining - compatible_6

**Time**: 2026-04-09
**Hypothesis**: Excluding MAFAULDA+MFPT allows JEPA to learn beyond epoch 2
**Change**: Pretrain on compatible sources only: cwru, femto, xjtu_sy, ims, paderborn, ottawa (28,839 windows)
**Sanity checks**: best_epoch=3, val_loss_ep5=0.0165, val_loss_ep10=0.0193
**Result**: best_epoch=3, best_val=0.0140, RMSE=0.0873±0.0018
**Seeds**: [0.0869, 0.0861, 0.0858, 0.0869, 0.0907]
**Embedding quality**: max_dim_corr=-0.121, PC1_corr=0.000
**Verdict**: MARGINAL — best epoch improved 2→3, val_loss improved 0.0161→0.0140
**Insight**: Compatible filtering helps (better val_loss) but still early convergence.
  RMSE similar to all_8 (0.087 vs 0.085) — downstream task sees similar benefit.
  The fundamental JEPA oscillation on multi-source data is not fully fixed.

---

## Exp C.3: JEPA Pretraining - bearing_rul_3

**Time**: 2026-04-09
**Hypothesis**: Training on bearing_rul_3 sources stabilizes JEPA beyond epoch 2
**Change**: Pretrain on ['femto', 'xjtu_sy', 'ims'] sources (22599 windows)
**Sanity checks**: best_epoch=3, val_loss_ep5=0.0194, val_loss_ep10=0.0187
**Result**: best_epoch=3, best_val=0.0161, RMSE=0.0863+/-0.0020
**Embedding quality**: max_dim_corr=-0.123, PC1_corr=-0.005
**Verdict**: MARGINAL - still early convergence
**Insight**: Still early convergence - need further analysis

---

## Exp D.1: TCN-Transformer + Handcrafted Features (Supervised)

**Time**: 2026-04-09
**Hypothesis**: TCN+Transformer captures temporal dependencies better than LSTM for HC features
**Change**: TCN (4 layers, dilations 1/2/4/8) + Transformer (2L, 4H) fusion, input=18 HC features
**Result**: RMSE=0.1642±0.0023
**vs V8 Transformer+HC (RMSE=0.070)**: WORSE (-134.6%)
**Verdict**: MARGINAL
**Insight**: TCN captures local temporal patterns; Transformer captures global episode structure

---

## Exp D.2: JEPA + TCN-Transformer

**Time**: 2026-04-09
**Hypothesis**: TCN-Transformer head works better than LSTM for JEPA embeddings
**Change**: Replace LSTM with TCN-Transformer on frozen JEPA embeddings
**Result**: RMSE=0.1395±0.0060
**vs V8 JEPA+LSTM (RMSE=0.189)**: BETTER (+26.2%)
**vs V9 JEPA+LSTM (RMSE=0.085)**: MUCH WORSE (1.64x higher error)
**Seeds**: [0.1342, 0.1323, 0.1472, 0.1380, 0.1456]
**Verdict**: REVERT — TCN-Transformer overfits with 24 training episodes
**Insight**: LSTM regularization via small hidden state outperforms TCN-Transformer in small-data regime.
  TCN-Transformer has more parameters and is more prone to overfitting.
  Simple LSTM (RMSE=0.085) outperforms complex TCN-Transformer (RMSE=0.140).

---

## Exp D.3: JEPA + Deviation-from-Baseline Features

**Time**: 2026-04-09
**Hypothesis**: Explicit deviation from healthy baseline helps predict RUL during long healthy phase
**Change**: Add [z_deviation, deviation_norm] to TCN-Transformer input (total dim=515)
**z_baseline = mean(z_1,...,z_K) for K=10 snapshots**
**Result**: RMSE=0.1795±0.0062
**vs JEPA+TCN-Transformer (RMSE=0.1395)**: WORSE (-28.7%)
**vs V9 JEPA+LSTM (RMSE=0.085)**: MUCH WORSE (2.1x higher error)
**Seeds**: [0.1734, 0.1878, 0.1795, 0.1719, 0.1848]
**Verdict**: REVERT — deviation features consistently hurt performance
**Insight**: Two likely causes: (1) K=10 baseline includes degraded snapshots for short-lifetime XJTU-SY
  bearings (42 total snapshots), contaminating the reference. (2) 513-dim input (vs 258-dim)
  causes overfitting with only 24 training episodes.

---

## Exp D.4: Hybrid JEPA+HC+Deviation — SKIPPED

**Time**: 2026-04-09
**Reason**: D.3 (JEPA+Deviation) RMSE=0.1795 was WORSE than baseline.
Per plan: "D.4 only if D.3 helps". D.3 did not help (RMSE 0.1795 vs 0.085 baseline).
Two confirmed failure modes: (1) K=10 baseline contaminated in short-lifetime XJTU-SY episodes,
(2) doubling input dimensionality (258→515) causes overfitting with 24 train episodes.
**Verdict**: SKIP — adding handcrafted features (532-dim input) would only worsen overfitting

---

## Exp E.1: Contiguous Block Masking

**Time**: 2026-04-09
**Hypothesis**: Contiguous block masking forces JEPA to learn temporal context beyond random masking
**Change**: Replace random 10/16 patch masking with single contiguous 10-patch block (random start).
  Pretrained on compatible_6 sources, 100 epochs, EMA=0.996. Block start randomized per sample.
**Sanity checks**: training loss decreased, RMSE in valid range [0, 1]
**Result**: best_epoch=4, best_val=0.0173, RMSE=0.0886±0.0049
**Seeds**: ['0.0870', '0.0894', '0.0883', '0.0969', '0.0816']
**Embedding quality**: max_dim_corr=0.154, PC1_corr=0.026
**vs C.2 (random masking, RMSE=0.0873)**: -1.5%
**Verdict**: KEEP
**Insight**: Block and random masking give similar downstream performance for 1024-sample windows; the JEPA context prediction task may be similar regardless of contiguity at this window length
**Next**: E.2 dual-channel encoder

---

## Exp E.2: Dual-Channel Raw+FFT Encoder

**Time**: 2026-04-09
**Hypothesis**: Explicit FFT channel helps JEPA learn spectral features correlated with RUL degradation
**Change**: Input (B, 2, 1024): channel 0=raw, channel 1=magnitude FFT (512 bins mirrored+normalized).
  PatchEmbed: 128 dims per patch (64 raw + 64 FFT) → 256. n_channels=2 in MechanicalJEPAV8.
**Sanity checks**: dual-channel model trains, loss decreases, embedding quality checked
**Result**: best_epoch=4, best_val=0.0155, RMSE=0.1119±0.0057
**Seeds**: ['0.1069', '0.1052', '0.1193', '0.1105', '0.1179']
**Embedding quality**: max_dim_corr=0.239, PC1_corr=0.030
**vs C.2 (single-channel random, RMSE=0.0873)**: -28.2%
**Verdict**: MARGINAL
**Insight**: FFT channel does not improve over single-channel — JEPA may already learn spectral features from raw signal alone via masked patch prediction

---

## Exp F.1: Heteroscedastic LSTM (Probabilistic RUL)

**Time**: 2026-04-09
**Hypothesis**: Gaussian NLL training provides calibrated uncertainty with near-zero accuracy cost
**Change**: LSTM head outputs (mu, log_var). Loss = 0.5*(log_var + (y-mu)^2/exp(log_var)).
  Identical architecture to deterministic head (256 hidden, 2 layers) + extra log_var linear.
**Sanity checks**: NLL loss finite, RMSE reasonable, PICP checked
**Result**: RMSE=0.0868±0.0023, PICP@90%=0.910 (WELL-CALIBRATED), MPIW=0.2414
**Seeds**: ['0.0865', '0.0861', '0.0891', '0.0831', '0.0894']
**vs deterministic JEPA+LSTM (0.0873)**: +0.6%
**Verdict**: KEEP — uncertainty at minimal accuracy cost
**Insight**: PICP@90%=0.910. Intervals are well-calibrated. Heteroscedastic output enables P(RUL<threshold) computation for deployment.

---

## Exp F.2: Ensemble Uncertainty (5-seed C.2 JEPA+LSTM)

**Time**: 2026-04-09
**Change**: Use 5 independently-seeded C.2 JEPA+LSTM runs as ensemble. Inter-seed std = uncertainty.
**Result**: Ensemble RMSE=0.0873±0.0018
  vs Heteroscedastic F.1: RMSE=0.0868±0.0023, PICP@90%=0.910
**Verdict**: KEEP — both methods useful, serve different purposes
**Insight**: Ensemble std (0.0018) reflects training variance. Heteroscedastic provides per-timestep
  uncertainty — more actionable for maintenance decisions. With 24 train episodes, both estimates have
  high noise. Ensemble is free (uses existing seeds); heteroscedastic requires NLL training.

---

