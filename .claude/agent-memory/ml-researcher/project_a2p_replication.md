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
LR 4-feature (var50+varfull, no train)0.635  [0.612, 0.656]   ~50%    bootstrap AUPRC=0.1336
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

- Transformer 30ep vs random: p=0.162 (NOT significant, corrected from earlier p=0.081; 10-seed two-tailed t-test)
- Supervised 100ep vs random: p<<0.001 (VERY significant)
- LR vs transformer: p=0.0006, d=1.73 (FULL dataset); p=0.18 (same test split, equivalent)
- Supervised vs unsupervised: Welch t=7.17, p=0.000026, d=3.45
- TF vs BiLSTM: p=0.047, d=3.53; TF vs CNN: p=0.003, d=6.67; LSTM vs CNN: p=0.43 (NS)

## 16 Verified Claims for NeurIPS (Updated April 12, 2026 - FINAL)

1. F1-tol 8.1x inflated (raw 5.35% -> 43.1%) [STRONG]
2. Random beats A2P F1-tol on all 3 datasets (SVDB4: 69.6% vs 67.6%) [VERY STRONG, 5-seed]
3. A2P AUROC not significant vs random (p=0.162, two-tailed t-test) [STRONG, 10-seed]
4. LR variance beats A2P transformer (p=0.0006, d=1.73) [VERY STRONG]
5. AP is learnable with correct training: supervised 0.624, p<<0.001 [VERY STRONG, 5-seed]
6. Calm-before-storm: AP+ windows have 0.78x lower variance (SVDB4 only; global 200-step window signal) [STRONG]
7. Supervised vs unsupervised: d=3.45, p=0.000026 [VERY STRONG]
8. F1-tol and AUROC rankings are inverted (Spearman rho=0) [MODERATE, 3 methods]
9. SVDB1 invalid (temporal confound, all labels at t>94%) [VERY STRONG]
10. 30ep training insufficient: 10% converge; 100ep: 100% converge [VERY STRONG] + SMD 30ep=0.583 (0% above 0.60)
11. LR 4-feat ~ TF (p=0.047 borderline, bootstrap CIs overlap [0.612,0.656] vs [0.614,0.633], delta=+1.1pp) [MODERATE]
12. TF > BiLSTM (p=0.047, d=3.5) > CNN (p=0.003, d=6.7): global attention is critical [VERY STRONG]
13. Near-horizon (0-50) is contaminated: 66.4% AP+ have anomaly in context [VERY STRONG]
14. AP not production-ready: LR=1.4x precision over random at 50% recall (8.4 FA/TP) [STRONG]
15. Dataset-specific AP directions: SVDB4=calm (oracle=0.718, LR=0.628); SMD borderline (oracle=0.862 confounded); only SVDB4 is valid [VERY STRONG] (Probes 74b/75/78/79/80/81)
16. F1-tol gameable by tolerance t: random achieves 58.9% at t=200; AUROC is stable [STRONG] (Probe 76)

## Formal AP Dataset Validity Criteria (Probes 78-81)

5 criteria: Separation (<20% ctx anomaly in AP+), Learnability (oracle>0.55), Non-trivial (ctx AUROC<0.60), Sample size (>50 AP+), Temporal (AP+ in both splits)

| Dataset | Verdict | Pass/5 | Key issue |
|---------|---------|--------|-----------|
| SVDB4   | VALID   | 5/5    | None - genuine AP task |
| SVDB1   | INVALID | 1/5    | Temporal confound + 0 AP+ in train |
| SMD     | BORDERLINE | 3/5 | Clustering effect: 45% AP+ have ongoing anomaly in context |

## New Analysis: SMD AP Contamination (Probe 79/80)

- SVDB4 AP+ windows: context anomaly rate=0.012 (0.16x base) - GENUINE calm
- SMD AP+ windows: context anomaly rate=0.307 (7.4x base!) - CLUSTERING effect
- SMD context-any-anomaly AUROC=0.672 (trivially predictable!)
- 45% of SMD AP+ windows have ongoing anomaly in context -> cluster continuation, not future prediction
- SMD oracle AUROC=0.862 is inflated by clustering (vs SVDB4's clean 0.748)

## Correction Waterfall (Probe 82)

Task difficulty (oracle-random gap) = 0.244 AUROC units.
1. A2P (wrong eval, 30ep): AUROC=0.521 -> 4.2% of gap
2. Fix evaluation: oracle goes 0.347->0.744
3. Fix training (100ep): AUROC=0.624 -> 50.6% of gap  
4. No-training baseline: LR=0.631 -> 54.1% of gap (beats trained model!)
5. Remaining gap: 0.121 AUROC units

## Paper Figures Generated (Probe 83)

In `results/figures/`:
- fig1_correction_waterfall.pdf/png: AUROC waterfall
- fig2_rank_inversion.pdf/png: F1-tol vs AUROC ranking inversion
- fig3_calm_before_storm.pdf/png: Lead time AUROC profile
- fig4_architecture_comparison.pdf/png: All architecture results
- fig5_dataset_validity.pdf/png: Pass/fail criteria for all 3 datasets

## Reproducibility: 16/16 claims verified (11 automated + 5 manual)

## Probe 62: Width Ablation (April 12, 2026)

- d=32 (32K params): AUROC=0.6178 ± 0.0070
- d=64 (ref, 103K): AUROC=0.6238 ± 0.0075 (5-seed)
- d=128 (431K params): AUROC=0.6164 ± 0.0059
- p=0.846 (NOT significant) - task is capacity-saturated at d=32
- 13x more parameters gives -0.001 AUROC change
- Architecture bottleneck is signal difficulty, NOT model capacity

## Probe 85: Oracle Gap Decomposition (April 12, 2026)

- Oracle AUROC=0.747, TF=0.624, remaining gap=0.123
- Early AP+ (block pos 0-49): 4.19x oracle signal vs AP-
- Late AP+ (block pos 50-99): 2.23x oracle signal vs AP-
- 34.4% of AP+ are "hard" late-block predictions
- Top-25% oracle covers 59.6% of AP+ events
- Extended horizon: oracle goes 0.747 (h=50) -> 0.900 (h=300) - anomalies are bursty

## Probe 86: Operational Utility (April 12, 2026)

- Oracle @ 25% recall: precision=1.000 (0 false alarms!) - only "easy" early-block events
- LR @ 50% recall: precision=0.100 (1.3x lift, 9 FA/TP) - NOT production-ready
- Oracle @ 50% recall: precision=0.325 (4.2x lift) - partially useful
- Base rate: 7.7% AP+

## NeurIPS LaTeX Tables (April 12, 2026)

In `results/figures/neurips_tables.tex`:
- Table 1: Architecture comparison (9 methods, AUROC, std, seeds, params)
- Table 2: Dataset validity criteria (SVDB4/SVDB1/SMD)
- Table 3: Correction waterfall (%gap achieved at each fix step)

## New Analysis: Easy vs Hard AP Windows (April 12, 2026)

Easy AP+ (top-25% oracle): predict first half of 100-step blocks (mean pos=34.9), context calm (var=0.804 vs AP- 1.719)
Hard AP+ (bottom-75% oracle): predict second half (mean pos=57.7), context noisy, oracle signal weak
LR scores easy AP+ only 0.081 vs AP- 0.079 - LR FAILS to exploit the calm-before-storm for easy cases!
LR learns "low full-window variance = AP+" (ch0_varfull coef=-0.665) but this signal doesn't separate easy from AP-.

Practical utility: oracle achieves precision=1.000 at 25% recall (oracle@25% = 1.000, LR@25%=0.097)
These are the "easy" predictions corresponding to first-half block positions.

## Temporal Calm Signal Analysis (April 12, 2026)

Variance window analysis (AUROC vs AP+ labels):
- last-5: 0.473 (BELOW random), last-10: 0.462, last-25: 0.459 (all below random!)
- last-50: 0.519, last-100: 0.571, last-150: 0.555, last-200: 0.613 (BEST)
- Signal monotonically increases with window length -> calm is GLOBAL (200+ steps)
- Very recent variance below random because anomaly already started in last 5-25 steps

Temporal chunk analysis (50-step blocks):
- Oldest [0-50]: AUROC=0.558 (strongest single signal, ratio=0.729)  
- Middle [50-100]: AUROC=0.493 (below random!)
- Middle [100-150]: AUROC=0.527
- Newest [150-200]: AUROC=0.524
- Transformer can weight chunks, LR cannot -> partial explanation for TF > LR

CNN receptive field: RF = 7+(7-1)+(7-1) = 19 steps = 9.5% of 200-step window
This is the mechanistic reason CNN (0.569) << Transformer (0.624)

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

## NEW CRITICAL FINDINGS (April 12, 2026 - Overnight Session 2)

### Contamination Decomposition (Probes 99-101)
- 66.5% of AP+ have ongoing anomaly in [t, t+100] = NEAR-HORIZON CONTAMINATION
- Oracle on contaminated AP+: AUROC=0.809 (detection, not prediction)
- Oracle on TRUE AP+ (no near-horizon): AUROC=0.603
- **LR on TRUE AP+: AUROC=0.702 >> Oracle (0.603) by +0.099!**
- Properly defined pure-prediction AP task: oracle=0.603, LR=0.702

### Statistical Proof: LR > Oracle on Strict AP (Probe 116)
- **LR AUROC: 0.703 [95% CI: 0.688, 0.718]** (bootstrap, n=5000)
- **Oracle AUROC: 0.648 [95% CI: 0.635, 0.662]**
- **Difference: +0.055 [CI: +0.037, +0.072] - CI excludes 0**
- **Permutation p=0.0000 (highly significant)**
- Contamination REVERSES comparison: oracle wins all-AP (p<0.0001), LR wins strict-AP (p<0.0001)

### 5-Fold CV: LR/RF Beat Oracle on Strict AP (Probe 120b)
- LR: 0.759 ± 0.015 (beats oracle in ALL 5 folds)
- RF: 0.791 ± 0.013 (beats oracle in ALL 5 folds)
- Oracle: 0.648 ± 0.010
- 0.168 total AUROC swing from contamination

### Block Onset Structure (Probes 122-123)
- 97.9% of strict AP+ events are within [100, 150] steps of next anomaly block start
- ALL 1170 strict AP+ = 117 blocks × 10 predictors each (EXACT match)
- Context shows block onset: last-20 var = 1.73x AP- (onset visible in context window)
- The AP task is ENTIRELY about anomaly block boundaries (not true future prediction)

### 4-Type AP+ Classification (Probes 113-114)
- Type A (66.4%): Contaminated (detection-like). Oracle wins (0.794 vs LR 0.608)
- Type B (19.9%): Strict + Rising onset. LR wins (0.722 vs oracle 0.591)
- Type C (0.2%): Strict + Calm baseline. LR wins (0.918 vs oracle 0.399)
- Type D (13.5%): Strict + no signal. Neither wins well; irreducibly unpredictable

### 10 Verified Claims (Probe 124 - FINAL)
1. 66.5% contamination
2. LR > oracle on strict AP: p=0.0000, CI=[+0.037, +0.072]
3. 5-fold CV: LR=0.759, RF=0.791 > Oracle=0.648 in all 5 folds
4. 100% of strict AP+ are block onset windows (97.9% in [100,150] window)
5. 0.168 AUROC contamination swing
6. F1-tol 8.1x inflation; random=68.1% > A2P=67.55%
7. SMD oracle=0.346 sub-random (all channels); top-5=0.704
8. LR +10.8pp over A2P (0.636 vs 0.528)
9. Practical ceiling=0.677 (oracle ensemble); not headline 0.745
10. 13.5% genuinely unpredictable (Type D)

### Main Performance Table (FINAL, April 12, 2026)
| Method | Std AP | Strict AP | Strict CV |
|--------|--------|-----------|-----------|
| A2P (paper, MBA TranAD) | 0.528 | ~0.55? | n/a |
| LR (4 var, no training) | 0.636 | 0.703 | 0.759±0.015 |
| RF (n=100, depth=5) | 0.717 | 0.808* | 0.791±0.013 |
| Oracle (future var) | 0.745 | 0.648 | 0.648±0.010 |
| Oracle ensemble | 0.677 | n/a | n/a |

### Calm-Before-Storm in Strict AP+ (Probe 103)
- True AP+ (non-contaminated) show clear rising variance in context:
  - Steps 0-40: variance 0.35x AP- (very calm)
  - Steps 60-100: variance 1.40x AP- (rising)
  - Steps 140-160: variance 0.24x AP- (calm again)
  - Steps 180-200: variance 1.50x AP- (final rise)
- Wilcoxon p<0.0001, trend ratio = 1.62x
- Standard AP+ shows NO trend (1.02x - dominated by contamination)

### SVDB4 Artificial Block Structure (Probe 107)
- ALL 117 anomaly blocks are EXACTLY 100 steps = pred_len (std=0, min=max=100)
- Temporal position feature (cos 2πt/1372) achieves AUROC=0.632 (≈ LR 0.634)
- But LR-position correlation: rho=0.007 (LR is NOT exploiting position)

### Five Attacks on A2P (Probe 115)
1. Task definition failure: 66.5% contamination (detection not prediction)
2. Metric failure: F1-tol 8.1x inflation; Brier Skill=-0.117; random beats A2P
3. Evaluation protocol failure: detection AUROC=0.401, A2P=0.528 (only +0.127 above detection)
4. Dataset validity failure: SMD oracle=0.346 sub-random (all channels)
5. Baseline failure: LR (no training) beats A2P by +0.108 AUROC

### SMD vs SVDB4 Comparison (Probe 98)
- SMD oracle (all 38 channels): 0.346 (below random)
- SMD oracle (top-5 channels): 0.704 (cherry-picked)
- 4 root causes: channel noise, anti-correlated signal, dimensionality, implicit cherry-picking
- SVDB4 oracle: 0.747 (valid task)

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
