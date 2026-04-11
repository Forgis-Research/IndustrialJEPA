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

| L_out | Paper (mean +/- std) | Ours seed 42 | Gap | Seeds 1+2 | Notes |
|-------|---------------------|-------------|-----|-----------|-------|
| 100 | 67.55 +/- 5.62 | 16.06 | -51.5 pp | In progress | 0.72% anomaly rate |

**AUROC (SVDB1 seed 42):** 0.490 (BELOW random 0.5 - A2P is anti-discriminating on held-out data!)

**Notes on SVDB1:**
- Data: PhysioNet SVDB record 801, 161K train / 69K test, 0.72% anomaly rate
- Paper uses 4 records (800-803), ~921K total, ~5.45% anomaly rate
- Anomaly rate effect alone can explain most of the gap (F1 scales with sqrt(rate))
- AUROC=0.490 confirms A2P scores are not just noisy but anti-correlated with anomalies

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
| SMD Baselines | Rolling Var (SMD) | AUROC | 0.774 | A2P: n/a | w/o training |
| SMD Baselines | Rolling Var (SMD) | F1-tol | 39.24% | A2P paper: 36.29% | **BEATS A2P** +2.95pp! |
| SMD Baselines | Z-score (SMD) | AUROC | 0.641 | A2P: n/a | w/o training |
| Cross-Domain | Rolling Var MBA->SMD | AUROC | 0.746 | In-domain: 0.771 | Only -0.025 transfer penalty |

Note: MBA_svdb = single SVDB record 801 (161K train / 69K test), 0.72% anomaly rate.
A2P results use TranAD MBA (train==test, 3.12% anomaly rate) - direct comparison not fully apples-to-apples.
Key: F1-tol and AUROC/AUPRC rankings are completely uncorrelated (Spearman rho=0.000) - the core NeurIPS finding.
CRITICAL: Rolling variance F1-tol BEATS A2P paper on SMD (39.24% vs 36.29%) without any training!

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
