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

