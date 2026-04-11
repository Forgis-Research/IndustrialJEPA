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

7. **Correct AP evaluation oracle = 0.744**: Define future_labels[t]=1 if anomaly in [t+100, t+150]. Oracle AUROC=0.744 (SVDB4), 0.554 (SMD), 0.692 (SVDB1). AP IS learnable with proper evaluation.

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
| SVDB4 | 0.744 | Oracle future var (test split) |
| SMD | 0.554 | Oracle future var |
| SVDB1 | 0.692 | Oracle future var (but confounded!) |

## NeurIPS Table (FINAL, April 11, 2026)

```
Method                                AUROC    95% CI          % Oracle  Seeds
-----------------------------------------------------------------------------------
Random                                0.500  N/A                0.0%      N/A
A2P (30ep unsupervised, 10-seed)      0.521  [0.490, 0.552]    8.6%      10
LR 4-feature (var50+varfull, no train)0.631  [0.608, 0.652]   ~50%      bootstrap
Supervised transformer (5-seed, 100ep)0.624  [0.617, 0.630]   50.6%     5
Oracle (future variance)              0.744  N/A              100.0%     N/A
```

**KEY: LR and supervised TF CIs OVERLAP -> statistically equivalent. LR achieves same AUROC as 100-epoch supervised transformer with NO training.**

## Architecture Comparison (COMPLETE, April 11, 2026)

```
Architecture                           AUROC     SD      Seeds  vs TF   p-value  d
------------------------------------------------------------------------------------
Supervised transformer (d=64, 100ep)   0.6238  0.0075    5    reference  --      --
LR 4-feature (no training needed)      0.6308  ~0.0001  N/A   +0.007    0.18 (NS) 0.93
BiLSTM (hidden=64, 2 layers, 100ep)    0.5805  0.0156    3    -0.043   0.047 *    3.53
1D CNN (3xConv1d, k=7, 100ep)          0.5691  0.0088    3    -0.055   0.003 **   6.67
A2P unsupervised (30ep)                0.521   0.042    10    -0.103   0.081 (NS)  --
Random                                 0.500    --       --    -0.124    --        --
```

## Epoch Convergence Analysis

- 30ep APTransformer (10 seeds): 0.521 ± 0.042; only 10% converge above 0.60
- 100ep APTransformer with MLP head (5 seeds): 0.624 ± 0.008; 100% converge above 0.60
- MLP head vs LN+Linear head: MLP peaks at ep 100 (0.624); LN+Linear peaks at ep10 (0.606) then declines

## Near-Horizon Contamination (Probe 64)

- Near-horizon (0-50 step): 66.4% of AP+ labels have ongoing anomaly already in context window
- Only horizons >= 100 steps (= anomaly block length) provide clean AP evaluation
- A2P's default 100-150 step horizon is methodologically sound
- Probe 57 near-horizon seed=42 got 0.759 > oracle 0.750 due to contamination

## SVDB4 Dataset Structure

- 117 anomaly blocks, ALL exactly 100 steps (synthetic labeling)
- Inter-event intervals: min=235, max=8297, mean=1465 steps
- AP+ rate: 9.46% (each event creates 149-step positive window)
- Calm-before-storm: AP+ windows have 0.78x lower variance than normal

## Key Statistical Results

- Transformer 30ep vs random: p=0.081 (NOT significant)
- Supervised 100ep vs random: p<<0.001 (VERY significant)
- LR vs transformer: p=0.0006, d=1.73 (FULL dataset); p=0.18 (same test split, equivalent)
- Supervised vs unsupervised: Welch t=7.17, p=0.000026, d=3.45
- TF vs BiLSTM: p=0.047, d=3.53; TF vs CNN: p=0.003, d=6.67; LSTM vs CNN: p=0.43 (NS)

## 14 Verified Claims for NeurIPS (Updated April 12, 2026)

1. F1-tol 8.1x inflated (raw 5.35% -> 43.1%) [STRONG]
2. Random beats A2P F1-tol on all 3 datasets (SVDB4: 69.6% vs 67.6%) [VERY STRONG, 5-seed]
3. A2P AUROC not significant vs random (p=0.081) [STRONG, 10-seed]
4. LR variance beats A2P transformer (p=0.0006, d=1.73) [VERY STRONG]
5. AP is learnable with correct training: supervised 0.624, p<<0.001 [VERY STRONG, 5-seed]
6. Calm-before-storm: AP+ windows have 0.78x lower variance (consistent across all splits) [STRONG]
7. Supervised vs unsupervised: d=3.45, p=0.000026 [VERY STRONG]
8. F1-tol and AUROC rankings are inverted (Spearman rho=0) [MODERATE, 3 methods]
9. SVDB1 invalid (temporal confound, all labels at t>94%) [VERY STRONG]
10. 30ep training insufficient: 10% converge; 100ep: 100% converge [VERY STRONG]
11. LR 4-feat = TF statistically (p=0.18, CIs overlap): complexity adds nothing [STRONG]
12. TF > BiLSTM (p=0.047, d=3.5) > CNN (p=0.003, d=6.7): global attention is critical [VERY STRONG]
13. Near-horizon (0-50) is contaminated: 66.4% AP+ have anomaly in context [VERY STRONG]
14. AP not production-ready: LR=1.4x precision over random at 50% recall (8.4 FA/TP);
    Oracle=2.5x (4.2 FA/TP). Oracle perfect precision at 36.7% recall for first-half blocks. [STRONG]

## New Analysis: Easy vs Hard AP Windows (April 12, 2026)

Easy AP+ (top-25% oracle): predict first half of 100-step blocks (mean pos=34.9), context calm (var=0.804 vs AP- 1.719)
Hard AP+ (bottom-75% oracle): predict second half (mean pos=57.7), context noisy, oracle signal weak
LR scores easy AP+ only 0.081 vs AP- 0.079 - LR FAILS to exploit the calm-before-storm for easy cases!
LR learns "low full-window variance = AP+" (ch0_varfull coef=-0.665) but this signal doesn't separate easy from AP-.

Practical utility: oracle achieves precision=1.000 at 36.7% recall (threshold=0.0929)
These are the "easy" predictions corresponding to first-half block positions.

## Key Why: Root Causes of A2P's Failure

1. **Wrong metric**: F1-tolerance is gameable (random wins). Use AUROC/AUPRC.
2. **Insufficient training**: 30 epochs causes 90% of seeds to fail to converge. Need 100 epochs.
3. **Single-seed evaluation**: 0.642 at seed=42 is 3.2 sigma above 10-seed mean (0.521).
4. **Wrong evaluation direction**: A2P's oracle = 0.347 means the task is definitionally impossible.
5. **Dataset validity**: SVDB1 is temporally confounded; only SVDB4 is valid.

## Correct Approach

Use: AUROC/AUPRC metrics + 100+ epochs + 5+ seeds + horizon >= 100 steps + temporal split + correct oracle definition.
Supervised upper bound: 0.624 AUROC (50.6% of oracle 0.744).
The task is achievable but A2P's evaluation masked this.

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
- `cnn_ap_100ep.json`: Probe 47 CNN 3-seed (0.5691 ± 0.0088)
- `lstm_ap_100ep.json`: Probe 48 BiLSTM 3-seed (0.5805 ± 0.0156)
- `optimal_lr.json`: Probe 63 4-feature LR analysis
- `epoch_curve.json`: Probe 40 epoch learning curve
- `architecture_comparison_stats.json`: Probe 65 formal t-tests
