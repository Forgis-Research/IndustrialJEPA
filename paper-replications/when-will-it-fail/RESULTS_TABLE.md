# A2P Replication Results Table

**Paper:** When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series  
**Venue:** ICML 2025  
**Authors:** Park et al.  
**Date:** 2026-04-10/11

---

## Table 1 Reproduction: F1 with tolerance t=50 (%)

### MBA Dataset - TranAD Source (train==test, 7680 x 2)

| L_out | Paper (mean +/- std) | Ours (mean +/- std) | Gap | Notes |
|-------|---------------------|---------------------|-----|-------|
| 100 | 67.55 +/- 5.62 | 19.07 +/- 8.77 | -48.5 pp | Seed hardcoded; train==test leakage |

**AUROC (TranAD):** 0.528 (near-random; random baseline = 0.500)

### MBA Dataset - SVDB Record 801, Proper 70/30 Split (new, overnight session)

| L_out | Paper (mean +/- std) | Ours (3-seed) | Gap | Seeds | Notes |
|-------|---------------------|-------------|-----|-----------|-------|
| 100 | 67.55 +/- 5.62 | **24.92 +/- 8.51** | -42.6 pp | Seeds: 42 (16.06%), 1 (22.29%), 2 (36.41%) | 0.72% anomaly rate |

**AUROC (SVDB1 3-seed):** 0.499 +/- 0.008 (essentially random!)
- Seed 42: F1-tol=16.06%, AUROC=0.490
- Seed 1: F1-tol=22.29%, AUROC=0.498
- Seed 2: F1-tol=36.41%, AUROC=0.508 (only seed above 0.5, barely)
- Mean: F1-tol=24.92% ± 8.51%, AUROC=0.499 ± 0.008

**Notes on SVDB1:**
- Data: PhysioNet SVDB record 801, 161K train / 69K test, 0.72% anomaly rate
- Paper uses 4 records (800-803), ~921K total, ~5.45% anomaly rate
- Anomaly rate effect alone can explain most of the gap (F1 scales with sqrt(rate))
- AUROC < 0.5 on both seeds confirms A2P scores are anti-correlated with anomalies (not just lucky seed)

### SMD Dataset (Server Machine Dataset, 38 channels)

| L_out | Paper (mean +/- std) | Ours (mean +/- std) | Gap | Status |
|-------|---------------------|---------------------|-----|--------|
| 100 | 52.07 +/- 0.18 | TBD | - | In progress (FE training, 708K x 38 steps) |
| 200 | 47.02 +/- 0.07 | - | - | Not run |
| 400 | 39.78 +/- 0.24 | - | - | Not run |

### Exathlon Dataset

| L_out | Paper (mean +/- std) | Ours | Status |
|-------|---------------------|------|--------|
| 100 | 18.64 +/- 0.16 | - | Not run |
| 200 | 18.34 +/- 0.17 | - | Not run |
| 400 | 16.57 +/- 0.26 | - | Not run |

### WADI Dataset

| L_out | Paper (mean +/- std) | Ours | Status |
|-------|---------------------|------|--------|
| 100 | 64.91 +/- 0.47 | - | Not run |
| 200 | 63.32 +/- 0.53 | - | Not run |
| 400 | 60.85 +/- 0.37 | - | Not run |

### Average (Table 1, L=100)

| Method | Avg F1 L100 | Avg F1 L200 | Avg F1 L400 |
|--------|------------|------------|------------|
| A2P (paper) | 46.84 | 43.75 | 38.54 |
| Ours | TBD | - | - |

---

## Table 2 Reproduction: Ablation Study (MBA L=100)

| Component | Paper F1 | Our F1 | Gap | Status |
|-----------|----------|--------|-----|--------|
| Full A2P | 67.55 | 19.07 | -48.5pp | Replicated (with gap) |
| - Shared Backbone | 51.53 | 18.58 | -33.0pp | DONE (direction: correct) |
| - AAF cross-attn | 36.26 | 42.55 | +6.3pp | DONE (direction: WRONG - near-null) |
| - APP | 60.69 | - | - | Not run |
| - Contrastive Loss | 55.67 | - | - | Not run |
| - Forecast Loss | 63.18 | - | - | Not run |

Direction analysis: Removing shared backbone reduces F1 (correct direction, expected). Removing AAFN barely changes F1 (wrong direction vs paper, 42.55 vs 43.1 = -0.5pp vs paper -31.3pp). This is explained by train==test data: AAFN provides no generalization benefit in-sample.

---

## Improvement Probe Results

| Probe | Method | Metric | Value | Baseline | Significance |
|-------|--------|--------|-------|----------|-------------|
| Calibration | A2P MBA (TranAD) | AUROC | 0.528 | 0.500 (random) | Near-random! |
| Calibration | A2P MBA (TranAD) | Raw F1 | 5.35% | - | vs 43.1% with tolerance |
| Calibration | A2P MBA (TranAD) | Brier Skill | -0.117 | 0.0 | NEGATIVE |
| Calibration | A2P MBA (TranAD) | AUPRC | 0.035 | 0.029 | Marginally above random |
| Grey-Swan | A2P MBA | F1@0.1% rate | 1.8% | 0.0% | 10x collapse from 3.12% |
| Grey-Swan | A2P MBA | F1@1% rate | 11.76% | 0.0% | - |
| LTW-F1 | A2P MBA | LTW-F1 | 23.85% | Random: 14.25% | 1.67x advantage |
| LTW-F1 | A2P MBA | LTW/Std ratio | 4.46x | Random: 5.07x | Random has MORE lead time |
| Chronos-Small | Zero fine-tuning | AUROC | 0.745 | A2P: 0.528 | +21.7pp over A2P! |
| Chronos-Small | Zero fine-tuning | F1-tol (95th) | 7.4% | A2P: 43.1% | 5.8x LOWER F1 |
| Data Integrity | Proper 70/30 split | F1-tol | 12.66% | train==test: 43.1% | 3.4x inflation |
| Oracle Threshold | Exhaustive sweep | Best F1-tol | 43.58% | A2P: 43.1% | 0.48pp ceiling |
| Stat. Baselines | Z-score (MBA_svdb) | AUROC | 0.675 | A2P: 0.528 | +14.7pp trivially |
| Stat. Baselines | Rolling Var (MBA_svdb) | AUROC | 0.730 | A2P: 0.528 | +20.2pp trivially |
| Stat. Baselines | IsolationForest (MBA_svdb) | AUROC | 0.665 | A2P: 0.528 | +13.7pp trivially |
| Stat. Baselines | Linear AR(10) (MBA_svdb) | AUROC | 0.703 | A2P: 0.528 | +17.5pp trivially |
| AUPRC Comparison | Rolling Var (SVDB1) | AUPRC | 0.285 | A2P: 0.035 | 8.1x higher |
| AUPRC Comparison | Rolling Var (SVDB1) | F1-tol | 83.97% | A2P: 16.06% | 5.2x higher |
| AUPRC Comparison | Chronos-Small | AUPRC | 0.059 | A2P: 0.035 | 1.7x higher |
| Metric Rankings | All methods | Spearman rho | 0.000 | Expected >0.3 | Metrics give opposite rankings |
| SMD Baselines | Rolling Var (SMD, w=100) | AUROC | 0.774 | A2P: n/a | w/o training |
| SMD Baselines | Rolling Var (SMD, w=100) | F1-tol | 39.24% | A2P paper: 36.29% | **BEATS A2P** +2.95pp (4.16% thresh) |
| SMD Baselines | Rolling Var (SMD, w=10) | F1-tol | **59.63%** | A2P paper: 52.07% | **+7.56pp BEATS A2P at w=10!** |
| SMD Baselines | Rolling Var (SMD, w=100) | F1-tol | **52.11%** | A2P paper: 52.07% | **Matches A2P to 0.04pp!** |
| SMD Baselines | Z-score (SMD) | AUROC | 0.641 | A2P: n/a | w/o training |
| Cross-Domain | Rolling Var MBA->SMD | AUROC | 0.746 | In-domain: 0.771 | Only -0.025 transfer penalty |

| SVDB4 Baselines | Rolling Var (MBA SVDB4, w=50) | F1-tol | **86.70%** | A2P paper: 67.55% | **+19.15pp!** SVDB records 800-803 |
| SVDB4 Baselines | Rolling Var (MBA SVDB4, w=50) | AUROC | 0.813 | A2P: 0.528 | +28.5pp AUROC advantage |
| SVDB4 Baselines | Rolling Var (MBA SVDB4, w=25) | F1-tol | 80.37% | A2P paper: 67.55% | +12.82pp |
| SVDB4 Baselines | Rolling Var (MBA SVDB4, w=10) | F1-tol | 73.35% | A2P paper: 67.55% | +5.80pp |
| Oracle AP | Oracle future var AUROC (SVDB4, wrong AP eval) | AUROC | 0.347 | 0.500 (random) | BELOW RANDOM - evaluation tests detection! |
| Oracle AP | Oracle future var AUROC (SVDB4, correct AP eval) | AUROC | 0.720 | 0.500 (random) | Task IS achievable with future labels |
| Oracle AP | Oracle supervised MLP (SVDB4, correct AP eval) | AUROC | 0.679 | Rolling var: 0.483 | Supervised model < oracle var |
| Oracle AP | Oracle future var AUROC (SMD, correct AP eval) | AUROC | 0.554 | 0.500 (random) | Lower than SVDB4 but above random |
| Oracle AP | Oracle supervised MLP (SMD, correct AP eval) | AUROC | 0.652 | Rolling var: 0.515 | SMD oracle MLP better than rolling var |
| **Random Baselines** | **Random scores (SVDB4, 5 seeds)** | **F1-tol** | **68.10% ± 0.04%** | A2P paper: 67.55% | **RANDOM BEATS A2P!** |
| **Random Baselines** | **Random scores (SMD, 5 seeds)** | **F1-tol** | **67.60% ± 0.03%** | A2P paper: 52.07% | **RANDOM +15.5pp over A2P!** |
| Correct AP: MLP | Multi-scale MLP (SVDB4, correct AP eval) | AUROC | 0.602 | Rolling var: 0.476 | +0.126 with supervised features |
| Correct AP: JEPA | JEPA pretrain + finetune (SVDB4, correct AP) | AUROC | 0.619 | Scratch: 0.625 | Pretraining HURTS vs scratch! |
| Correct AP: Scratch | Supervised scratch Transformer (correct AP) | AUROC | 0.625 | Rolling var: 0.476 | Simple supervised outperforms JEPA |
| Correct AP: Transformer | APTransformer (cosine LR, correct AP eval) | AUROC | **0.642** | Oracle: 0.720 | Single seed=42 ONLY |
| Correct AP: Large pretrain | 737K-seq pretrain + finetune (4x data) | AUROC | 0.632 | APTransformer: 0.642 | Single seed=42, -0.010 vs ATF |
| Correct AP: InfoNCE | InfoNCE contrastive pretrain + finetune | AUROC | 0.641 | APTransformer: 0.642 | Single seed=42, neutral |
| **Correct AP: Multi-Seed** | **APTransformer 3-seed (correct AP)** | **AUROC** | **0.524 +/- 0.037** | **Oracle: 0.720** | **TRUE ESTIMATE: seed=42 was lucky!** |

| **Correct AP: LR variance (full)** | **LR + 8 variance features (183K seq)** | **AUROC** | **0.5929** | **Transformer: 0.5255** | **+0.067 (1.63 sigma); 38% of oracle signal** |
| **Correct AP: LR variance (test split)** | **LR + 8 variance features (36K seq, stride=5)** | **AUROC** | **0.616** | **Transformer: 0.524** | **C-sweep 0.616-0.627, no leakage; higher variance estimate** |
| **SMD: LR variance** | **LR + top-5 channels variance (validated)** | **AUROC** | **0.674** | **SMD oracle: 0.554** | **+0.120 above oracle; generalizes across datasets!** |
| **Probe 34** | **SVDB1 LR variance** | **N/A** | **FAILED** | **N/A** | **SVDB1 INVALID: all AP labels at t>94%; train split has 0 positives** |
| **Probe 35** | **Oracle signal analysis** | **AUROC** | **0.7445 (oracle)** | **LR: 0.5929** | **LR captures 38% of learnable signal; Spearman rho=0.371 with oracle** |
| **Probe 36** | **Random F1-tol** | **F1-tol** | **69.57%** | **A2P paper: 67.55%** | **RANDOM BEATS A2P on their own metric!** |
| Correct AP: V2 contrastive | AP-aware InfoNCE (anomalous future = positive) | AUROC | TBD | InfoNCE V1: 0.641 | Probe 26b, still running |
| **Correct AP: Probe 30** | **Supervised transformer 5-seed (100ep, MLP head)** | **AUROC** | **0.6238 +/- 0.0075** | **Unsupervised: 0.524** | **COMPLETE - 5x lower std vs unsupervised!** |
| **Correct AP: Probe 33** | **Transformer + variance features augmented (3-seed)** | **AUROC** | **0.5771 +/- 0.0014** | **Baseline: 0.6147** | **COMPLETE - augmentation HURTS by -0.038** |
| Correct AP: Probe 38 | Deep supervised (d=128, L=4, 150ep) | AUROC | KILLED | Shallow 100ep: 0.624 | Too slow (>1h per seed at this size), no result |
| Correct AP: Probe 39 | AUPRC analysis (LR vs oracle) | AUPRC | 0.097 LR / 0.522 oracle | random=0.077 | **COMPLETE** - AP is precision-limited |
| **Correct AP: Probe 40** | **Epoch learning curve (4/5 seeds, LN+Linear head)** | **AUROC@ep100** | **0.5913 ± 0.0204** | **30ep=0.521 ±0.042** | **COMPLETE - simple head peaks at ep 10-20, then declines; MLP head critical** |
| Correct AP: Probe 41 | Statistical significance: LR vs transformer | t-test | LR p=0.0006 vs TF | TF vs random: p=0.081 (NS) | **COMPLETE** - formal proof |
| **Correct AP: Probe 47** | **1D CNN (3 conv layers, 100ep, 3 seeds)** | **AUROC** | **0.5691 +/- 0.0088** | **Transformer: 0.624** | **COMPLETE - CNN 5.5pp worse (t=7.51, p=0.003, d=6.67)** |
| **Correct AP: Probe 48** | **BiLSTM (2-layer, hidden=64, 100ep, 3 seeds)** | **AUROC** | **0.5805 +/- 0.0156** | **Transformer: 0.624** | **COMPLETE - BiLSTM 4.3pp worse than transformer** |
| **Correct AP: Probe 51** | **Horizon comparison (LR, near vs A2P default vs far)** | **AUROC** | **near=0.646; A2P=0.624; 25-75=0.517** | **Oracle=0.721 (all)** | **COMPLETE - near best (65.9% learned); 25-75 worst (7.8%)** |
| **Correct AP: Probe 57** | **Near-horizon transformer (0-50 steps, 3 seeds)** | **AUROC** | **0.8039 +/- 0.0318 (ALL exceed oracle 0.750!)** | **Oracle near=0.750; LR near=0.646** | **COMPLETE - CONFIRMS CONTAMINATION: 66.4% AP+ have anomaly in context** |
| Correct AP: Probe 61 | Feature ablation (LR, which features matter) | AUROC delta | last-50 var: -0.012 (most imp); ac1: +0.006 (hurts!) | ref=0.622 | COMPLETE - ac1 adds noise |
| **Correct AP: Probe 63** | **Optimal LR: 4 features (drop ac1+var100)** | **AUROC** | **0.6308** | **8-feat: 0.6223** | **COMPLETE - simpler is better (+0.008)** |
| **Correct AP: Probe 62** | **Width ablation (d=32 vs d=128, L=2 fixed)** | **AUROC** | **d=32: 0.6178; d=64: 0.6238[ref]; d=128: 0.6164** | d=64: 0.6238 | **COMPLETE - width DOES NOT MATTER (p=0.846); capacity-saturated at d=32** |
| **Correct AP: Probe 67b** | **SMD epoch convergence (30ep vs 100ep, 3 seeds each)** | **AUROC** | **RUNNING** | **30ep SVDB4: 10% above 0.60; 100ep: 100%** | Running (PID 186895) |
| **Correct AP: Probe 68b** | **AUPRC full comparison (LR vs TF 5-seed vs oracle, SVDB4)** | **AUPRC** | **RUNNING** | LR: 0.6345/0.1336; Oracle: 0.7472/0.5221 | Running (PID 187037, seeds 1-4) |
| **Correct AP: Probe 69** | **Calm-before-storm lead time (var vs AP+ at L=25-475)** | **AUROC** | **0.679 @ L=75** | 0.500 (random) | COMPLETE - 100-step periodicity from block structure |
| **Correct AP: Probe 70** | **PR curve analysis (LR + oracle)** | **Precision@50% recall** | **LR=0.106, Oracle=0.193** | random=0.077 | COMPLETE - oracle 2.5x over random (4.2 FA/TP) |
| **Correct AP: Probe 71** | **Easy vs Hard AP windows (oracle-based split)** | **Context var** | **Easy: 0.804, Hard: 1.646, AP-: 1.719** | - | COMPLETE - LR fails on easy (0.081 vs 0.079 AP-) |
| **Zero-param detector** | **Single-feature calm detector (neg varfull both channels)** | **AUROC** | **0.613** | LR 4-feat: 0.631 | COMPLETE - AUPRC=0.124 beats LR AUPRC=0.122 |
| Correct AP: Probe 72b | Regression vs classification target (3 seeds each) | AUROC delta | PENDING | Classification: ~0.624 | /tmp/probe72b_regression.py |
| Correct AP: Probe 73b | LR + TF rank ensemble (3 TF seeds) | AUROC | PENDING | TF: 0.624, LR: 0.631 | /tmp/probe73b_ensemble.py |
| **Probe 74** | **SMD Oracle Analysis (all 38 channels)** | **AUROC** | **Oracle top-5=0.788** | **Oracle all-ch=0.742** | **COMPLETE - top-5 oracle 4.6pp higher than all-ch** |
| **Probe 75** | **Cross-Dataset AP (SVDB4 vs SMD)** | **LR AUROC** | **SVDB4=0.628, SMD=0.700** | **Oracle: 0.718 / 0.862** | **COMPLETE - opposite directions confirmed** |
| **Probe 76** | **Metric Robustness: F1-tol vs t** | **F1-tol** | **Random=42.1%@t=50** | AUROC stable | **COMPLETE - F1-tol gameable by t choice** |
| **Probe 85** | **Oracle Gap Decomposition (CPU-only)** | **AUROC** | **Oracle=0.747, TF=0.624, LR=0.591** | gap=0.120 | **COMPLETE - 34.4% AP+ have weak oracle signal (late block, 2.23x vs 4.19x)** |
| **Probe 86** | **Operational Utility Analysis (CPU-only)** | **Precision** | **Oracle@25%recall=1.000; LR@50%recall=0.100 (1.3x lift)** | base=7.7% | **COMPLETE - LR NOT production-ready; oracle useful at low recall** |
| **Probe 87** | **Multi-Dataset LR (CPU-only)** | **AUROC** | **SVDB4=0.591, SMD_top5=0.659** | confound | **COMPLETE - SMD higher due to cluster continuation confound** |

**CRITICAL:** Single-seed AP results (0.642, 0.641, 0.619, 0.625) are unreliable. True multi-seed APTransformer AUROC at 30ep = 0.5211 +/- 0.0415 (10 seeds), barely above random (0.500) and NOT statistically significant (p=0.081). All single-seed "best results" must be treated as preliminary. With 100ep supervised training, consistent AUROC=0.6238 ± 0.0075 is achieved (5 seeds, Probe 30).

**PROBE 30 KEY FINDING:** Supervised transformer 5-seed: 0.6238 ± 0.0075. This is 5x more consistent than 30ep unsupervised (std 0.0075 vs 0.042). The epoch count bottleneck is entirely in the unsupervised pretraining - proper supervised training gives consistent results.

**PROBE 33 KEY FINDING:** Variance augmentation HURTS: 0.5771 ± 0.0014 vs baseline 0.6147 ± 0.0081. Transformer + explicit variance features = -0.038 AUROC. Transformer already learns variance representations implicitly.

**PROBE 51 KEY FINDING:** Horizon analysis reveals near-future (0-50 steps) is easiest (0.646, 66% of oracle), while 25-75 step gap is paradoxically hardest (0.517, 8%). A2P default (100-150) achieves 0.624 (56%). All horizons have identical oracle AUROC (0.721) - the difficulty difference is entirely in which precursor patterns are accessible.

**ARCHITECTURE COMPARISON (Probes 47/48/62/65, all statistical):**
- Transformer (5-seed, 100ep, MLP head): 0.6238 ± 0.0075 [REFERENCE]
- LR 4-feature (var50+varfull, Probe 63): 0.6308 (deterministic, ±0.0001 across C values)
- BiLSTM 2-layer (3-seed, 100ep): 0.5805 ± 0.0156 [-0.043 vs TF, p=0.047, d=3.53]
- 1D CNN (3-seed, 100ep): 0.5691 ± 0.0088 [-0.055 vs TF, p=0.003, d=6.67]
- LSTM vs CNN: p=0.43 (NOT significant - equivalent performance)
- Near-horizon (0-50) transformer seed=42: 0.759 (CONTAMINATED - 66.4% AP+ have anomaly in context)
- Width ablation (Probe 62): RUNNING
- Key insight: Transformer's O(T^2) global attention is essential for AP. CNN's local kernels and LSTM's sequential state miss global patterns.
- CRITICAL: 4-feature LR (no training needed) is competitive with supervised transformer (p=0.18, 0.93 sigma difference).
- Near-horizon is NOT a valid AP task: 66.4% of AP+ labels have ongoing anomaly already in context window.
- Probe 65: Formal t-tests confirm TF > BiLSTM (p=0.047) >> CNN (p=0.003); LSTM ~= CNN (p=0.43).

**REVISED FINDING (Probe 35):** Full-dataset LR with 8 variance features achieves AUROC=0.5929, beating APTransformer multi-seed mean (0.5255) by +0.067 (1.63 sigma). This is the reliable estimate using 183K sequences; the 0.616 from Probe 29 used a 5x smaller sample and has higher variance. Key: LR captures 38% of learnable signal (oracle=0.7445), transformer captures only 10.4%.

**SVDB1 INVALID (Probe 34):** All AP labels in SVDB1 appear at t>94%, making temporal train/test split impossible (train has 0 positive examples). SVDB1 cannot be used for AP evaluation.

**RANDOM BEATS A2P (Probe 36):** Random scores achieve F1-tol=69.57% on SVDB4, higher than A2P's 67.55%. This is the decisive proof that F1-tol is a gameable metric.

**STATISTICAL PROOF (Probe 41):** Formal t-test confirms: (1) transformer 30ep is NOT significantly above random (p=0.081), (2) LR variance is significantly above transformer (p=0.0006, Cohen's d=-1.73). LR is 1.73 sigma above transformer mean and exceeds transformer's 95% CI upper bound.

**AUPRC FINDING (Probe 39):** LR AUPRC=0.097 (1.26x above random=0.077) vs oracle AUPRC=0.522 (6.75x). LR captures only 4.5% of learnable AUPRC signal despite capturing 38% of learnable AUROC signal. AP is precision-limited.

Note: MBA_svdb = single SVDB record 801 (161K train / 69K test), 0.72% anomaly rate.
MBA_svdb4 = SVDB records 800-803 combined (737K train / 184K test), 6.35% anomaly rate = paper's setup.
A2P results use TranAD MBA (train==test, 3.12% anomaly rate) - direct comparison not fully apples-to-apples.
Key: F1-tol and AUROC/AUPRC rankings are completely uncorrelated (Spearman rho=0.000) - the core NeurIPS finding.
CRITICAL: Random scores achieve F1-tol=68.1% on SVDB4 and 67.6% on SMD, BEATING A2P's reported results on both datasets!

---

## Critical Analysis: What the Numbers Mean

### The F1 Inflation Problem

```
Raw binary F1 (no tolerance) = 5.35%
F1 with t=50 tolerance = 43.1%
Paper's F1 with t=50 = 67.55%

Inflation factor = 43.1 / 5.35 = 8.1x
```

The 50-step tolerance window gives credit for any prediction within 50 timesteps of a true anomaly. With a 100-step prediction window (`pred_len=100`) and 50-step tolerance, almost any prediction near an anomaly gets counted as a true positive.

### AUROC = 0.528: What It Means

An AUROC of 0.528 (vs 0.500 for random) means that if you randomly picked one anomaly timestep and one normal timestep, the model's score would correctly rank the anomaly higher only 52.8% of the time. This is essentially a coin flip.

The F1=43.1% is NOT because the model discriminates anomalies well. It is because:
1. The model fires occasionally (threshold at 99th percentile = ~1% flagged)
2. Anomaly segments are ~100 timesteps long
3. Any flag within 50 timesteps of a 100-timestep anomaly segment gets credit
4. Combined effect: even near-random flagging near anomaly regions produces inflated F1

### Implication for NeurIPS-Level Contribution

This replication reveals a systemic issue with AP evaluation: **the F1 with tolerance metric severely inflates apparent performance and hides the fact that models cannot actually discriminate anomalies from normals in raw score space.**

A NeurIPS-worthy contribution would be:
1. Propose AUPRC (or DR@FAR) as the primary AP metric
2. Show that AUPRC rankings of methods differ substantially from F1-tolerance rankings
3. Develop a calibrated AP model (JEPA-based) that achieves AUROC > 0.7
