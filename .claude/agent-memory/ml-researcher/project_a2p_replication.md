---
name: A2P Replication Findings (April 2026)
description: Critical findings from replicating Park et al. ICML 2025 "When Will It Fail?" - reveals A2P evaluation flaws and NeurIPS contribution
type: project
---

## Key Facts

Paper: "When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series" (ICML 2025, Park et al.)
Code: https://github.com/KU-VGI/AP
Replication dir: `/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/`

## Critical Findings (All Confirmed, April 2026)

1. **AUROC essentially random on proper split (3-seed confirmed)**: Seed 42 (0.490), seed 1 (0.498), seed 2 (0.508). Mean=0.499 +/- 0.008. Indistinguishable from random 0.500.

2. **RANDOM SCORES BEAT A2P on ALL 3 datasets**: SVDB4: 68.10% vs 67.55%; SMD: 67.60% vs 52.07% (+15.5pp!); SVDB1: 58.91% vs 19.17% (+39.7pp). F1-tol is trivially gamed by random noise.

3. **Rolling variance (max-chan) DOMINATES A2P**: MBA SVDB4 w=50: 86.70% vs 67.55% (+19.15pp). SMD w=10: 63.95% vs 52.07% (+11.9pp). ALL window sizes beat A2P.

4. **MBA train==test data leakage (3.4x inflation)**: TranAD-derived MBA has identical train/test. With proper 70/30 split: 12.66%.

5. **F1-tolerance 8x inflation**: Raw F1=5.35%, F1-tol=43.1% (8x). Point adjustment inflates random scores 10x (6.68% -> 68.19% on SVDB4).

6. **Oracle AP AUROC = 0.347 (below random!)**: Future variance doesn't predict current anomaly labels on SVDB4. Evaluation tests detection, not prediction.

7. **Correct AP evaluation oracle = 0.720**: Define future_labels[t]=1 if anomaly in [t+100, t+150]. Oracle AUROC=0.720 (SVDB4), 0.554 (SMD), 0.692 (SVDB1). AP IS learnable with proper evaluation.

8. **SVDB1 temporal confound**: All 5 anomaly segments at t>65000 of 69120. Time index AUROC=0.954. Not a valid AP dataset.

9. **E2E training probe**: Unfreezing AAFN during joint training: +12.8pp F1 (16%->28.8%), AUROC crosses 0.5 for first time (0.490->0.507).

10. **Metric rank inversion (Spearman rho=0.000)**: F1-tol ranks A2P #1, AUROC ranks A2P last.

## 3-Seed SVDB1 Results (FINAL, April 11, 2026)

| Seed | F1-tol | AUROC |
|------|--------|-------|
| 42 | 16.06% | 0.490 |
| 1 | 22.29% | 0.498 |
| 2 | 36.41% | 0.508 |
| Mean | 24.92% ± 8.51% | 0.499 ± 0.008 |
| Paper | 67.55% | - |

## Random Score Baselines (5 seeds each)

| Dataset | Random F1-tol | A2P | Beats A2P? |
|---------|--------------|-----|------------|
| SVDB4 | 68.10% ± 0.04% | 67.55% | YES |
| SMD | 67.60% ± 0.03% | 52.07% | YES (+15.5pp) |
| SVDB1 | 58.91% ± 7.64% | 19.17% | YES (+39.7pp) |

## Oracle AP Upper Bounds (Correct Evaluation)

| Dataset | Oracle AUROC | Method |
|---------|-------------|--------|
| SVDB4 | 0.720 | Oracle future var |
| SVDB4 | 0.679 | Supervised MLP (15 features, 30 epochs) |
| SMD | 0.554 | Oracle future var |
| SMD | 0.652 | Supervised MLP (15 features, 30 epochs) |
| SVDB1 | 0.692 | Oracle future var (but confounded!) |

## Trainable AP Models on Correct Evaluation (SVDB4, April 11, 2026)

**CRITICAL: Multi-seed validation (Probe 27) reveals single-seed results are unreliable!**

| Model | AUROC | Source | Notes |
|-------|-------|--------|-------|
| Rolling var (no training) | 0.476 | deterministic | Baseline |
| Multi-scale MLP (supervised) | 0.602 | single seed=42 | May be lucky |
| JEPA temporal pretrain + finetune | 0.619 | single seed=42 | May be lucky |
| Supervised from scratch (fixed LR) | 0.625 | single seed=42 | May be lucky |
| InfoNCE contrastive pretrain | 0.641 | single seed=42 | May be lucky |
| Large-scale pretrain (4x data) | 0.632 | single seed=42 | May be lucky |
| APTransformer (cosine LR) | 0.642 | single seed=42 | WAS LUCKY (3.2 sigma above mean!) |
| **APTransformer 3-seed TRUE** | **0.524 +/- 0.037** | **3 seeds** | **Statistically validated** |
| Oracle future var | 0.720 | deterministic | Upper bound |

**KEY INSIGHT**: APTransformer 0.642 = lucky seed 42 only. True multi-seed mean = 0.524 ± 0.037.
Gap to oracle is 0.196 (not 0.078). ALL single-seed results are unvalidated.

**SSL pretraining findings (all failed)**:
- JEPA temporal: -0.023 vs APTransformer, hurts due to normalcy prior
- InfoNCE generic: -0.001 vs APTransformer, neutral
- 4x large-scale temporal: -0.010 vs APTransformer, scale doesn't fix objective mismatch

## UPDATED FINDINGS (April 11, 2026 ~19:00) - Probes 28b-46

**FINAL AUROC results (SVDB4, COMPLETE as of April 11, 2026)**:
- LR variance 8 features (stride=5, 36K seq): AUROC=0.616, AUPRC=0.100
- LR variance (stride=1, 183K seq, Probe 35): AUROC=0.5929 (more reliable)
- Oracle AUROC=0.7445 (183K seq), AUPRC=0.522
- Supervised transformer 50ep (3-seed, Probe 33): AUROC=0.6147 ± 0.0081 (50ep baseline)
- **Supervised transformer 100ep (5-seed, Probe 30 FINAL): AUROC=0.6238 ± 0.0075**
- Unsupervised 30-epoch (10-seed, Probe 28b): AUROC=0.521 ± 0.042 (NOT above random p=0.081)
- Variance augmentation HURTS (Probe 33 FINAL): 0.5771 ± 0.0014 (vs baseline 0.6147)
- Deep supervised transformer (Probe 38, 128d/4L, 150ep): RUNNING

**Statistical significance (Probe 41, 53)**:
- Transformer 30ep vs random: t=1.52, one-sided p=0.081 (NOT significant)
- LR vs transformer: t=-5.19, p=0.0006 (LR significantly better, d=1.73)
- Supervised vs unsupervised: Welch t=7.17, p=0.000026, Cohen's d=3.45 (HUGE)
- 95% CIs non-overlapping: supervised [0.613, 0.634] vs unsupervised [0.490, 0.552]
- % of learnable AUROC: unsup=8.6%, LR=38.0%, supervised=50.6%

**AUPRC (Probe 39)**:
- LR AUPRC=0.097 (1.26x above random=0.077)
- Oracle AUPRC=0.522 (6.75x)
- LR captures only 4.5% of learnable AUPRC
- Calibration (Probe 43): LR is well-calibrated; recalibration does NOT help

**"Calm before storm" (Probes 44-45, 49, 52)**:
- AP-positive windows have LOWER variance (ratio 0.77-0.80 across ALL splits)
- Pre-event variance (200 steps before): 0.0334 ± 0.0032 (CONSISTENT across all 117 events)
- Post-anomaly variance: 2.42x higher
- This is "loss of heart rate variability" - clinically established predictor
- Consistent across Train/Val/Test and across all 117 identical 100-step anomaly blocks

**Horizon analysis (Probe 51 LR, Probe 46 oracle)**:
- Near (0-50): AUROC=0.646 (LR), 65.9% of oracle - EASIEST
- A2P default (100-150): AUROC=0.624 (LR), 56% of oracle
- 25-75 step gap: AUROC=0.517 (LR), 7.8% - HARDEST (non-monotonic!!)
- Oracle AUROC identical (0.721) across ALL horizons

**Anomaly block structure (Probe 50)**:
- All 117 anomaly blocks are EXACTLY 100 steps (synthetic labeling, matches pred_len=100)
- Each event creates 149-step AP+ window (pred_len + future_window - 1)
- No events closer than 235 steps (all IEIs >= 235)

**Architecture comparison (ongoing)**:
- Supervised transformer (5-seed): 0.624 ± 0.008 (BEST)
- CNN 1D (seed=42 only so far): 0.560 (worse than transformer)
- BiLSTM (seed=42 only so far): 0.584 (worse than transformer)
- Variance augmentation HURTS transformer: -0.038 AUROC
- Deep transformer (128d, 4L, 150ep): RUNNING

## Result Files (April 2026)

All in `results/improvements/`:
- `aptransformer_seed_distribution.json`: 10-seed distribution (0.5211 ± 0.0415)
- `oracle_analysis.json`: LR captures 38%, oracle=0.7445
- `f1tol_analysis.json`: Random beats A2P F1-tol (69.57 vs 67.55%)
- `statistical_comparison.json`: Formal t-tests (Probe 41)
- `auprc_lr_analysis.json`: LR AUPRC=0.097, oracle=0.522
- `calibration_lr.json`: Calibration analysis (no improvement)
- `feature_analysis.json`: Permutation importance + coefficients
- `calm_before_storm.json`: Lead time AUROC + temporal analysis
- `supervised_ap_5seed.json`: Probe 30 final 5-seed (0.6238 ± 0.0075)
- `transformer_var_augmented.json`: Probe 33 final 3-seed (augmentation hurts)
- `horizon_comparison.json`: Probe 51 horizon analysis

## NeurIPS Narrative (10-step evidence chain, ALL VERIFIED)

1. F1-tolerance 8x inflation (raw 5.35% -> 43.1%)
2. Random scores beat A2P on ALL 3 datasets (SVDB4: 69.57% vs 67.55%)
3. AUROC rank inversion: LR>A2P, F1-tol says Random>A2P>LR (Spearman rho=0)
4. A2P transformer (30ep) NOT above random: p=0.081 (Probe 41, 10 seeds)
5. LR variance significantly beats transformer: p=0.0006, Cohen's d=1.73
6. AP signal is "calm before storm": variance DECREASES 0.78x before arrhythmia
7. AP is precision-limited: AUPRC barely above random despite decent AUROC
8. With proper training (supervised 100ep, 5-seed): reliable 0.6238 AUROC
9. Lead time matters: 0-50 step is easiest (65.9% oracle), 25-75 is paradoxically hardest
10. SVDB1 invalid (temporal confound: all AP labels at t>94%); SVDB4 is valid

Key Why: A2P's failure = wrong metric (F1-tol) + insufficient training (30ep) + single-seed evaluation
Correct approach: AUROC/AUPRC + sufficient epochs (100+) + multi-seed (≥5)
Supervised upper bound: 0.624 AUROC (50.6% of oracle 0.744)
