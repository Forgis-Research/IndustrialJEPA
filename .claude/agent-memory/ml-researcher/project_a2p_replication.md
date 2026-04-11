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

1. **AUROC=0.528 on train==test, AUROC=0.490 on proper split**: A2P literally cannot discriminate anomalies from normals on held-out data. F1=43.1% comes entirely from 50-step tolerance window (8x inflation from raw F1=5.35%).

2. **Chronos-Small beats A2P by +21.7pp AUROC**: Frozen Chronos-Small (20M, zero fine-tuning) achieves AUROC=0.745 vs A2P=0.528. A2P's specialized architecture provides negative value.

3. **MBA train==test data leakage**: TranAD-derived MBA has identical train and test sets (3.4x inflation). With proper 70/30 split: F1=12.66%.

4. **Grey-swan collapse**: At 0.1% anomaly rate (realistic industrial), F1=1.8% (24x collapse from 43.1%). Entire AP evaluation framework is broken for real industrial use.

5. **Seed bug**: Official code ignores --random_seed, hardcodes seed=20462. Variance in paper is from checkpoint stochasticity, not true seed variation.

6. **F1-tolerance/AUROC rank inversion (Spearman rho=0.000)**: Metrics give completely uncorrelated method rankings. A2P is #1 by F1-tol but #4 by AUROC. Chronos is #1 by AUROC but #3 by F1-tol.

7. **Statistical baselines beat A2P**: Rolling variance (0.730), Z-score (0.675), Linear AR (0.703), Isolation Forest (0.665) all beat A2P AUROC=0.528 on MBA_svdb.
   - Rolling variance AUPRC=0.285 vs A2P AUPRC=0.035 (8x higher)
   - Rolling variance F1-tol=83.97% vs A2P F1-tol=16.06% (5x higher) on SVDB1

8. **Anomaly rate explains gap**: SVDB1 single record (0.72% rate) gives F1=16.06%. Paper uses 4 records (~5.45% rate). Rate effect alone predicts F1~36-44%, but paper claims 67.55%. Remaining 23-31pp gap unexplained (may be record selection).

## Replication Results (Completed)

| Dataset | Paper | Ours | Gap | Notes |
|---------|-------|------|-----|-------|
| MBA L100 (TranAD train==test) | 67.55 | 19.07+/-8.77 | -48pp | Seed bug, data leakage |
| MBA L100 (TranAD 70/30 split) | 67.55 | 12.66 | -55pp | 3.4x leakage confirmed |
| SVDB1 record 801 (seed=42) | 67.55 | 16.06 | -52pp | AUROC=0.490 (below random!) |
| SVDB1 seeds 1,2 | - | In progress | - | Running as of April 2026 |
| SMD L100 | 52.07 | In progress | - | 708K steps, ~2h to complete |

## Improvement Probe Results (Completed)

All saved in `results/improvements/`:
- `calibration_analysis.json`: AUROC=0.528, Brier skill=-0.117, raw F1=5.35%
- `chronos_baseline.json`: Chronos AUROC=0.745 (+21.7pp vs A2P)
- `grey_swan_test.json`: 24x F1 collapse at 0.1% anomaly rate
- `data_integrity.json`: 3.4x leakage confirmed
- `oracle_threshold.json`: F1 ceiling = 43.58% (A2P already at ceiling)
- `svdb_baselines.json`: All classical baselines beat A2P AUROC
- `cross_dataset_transfer.json`: Only -0.025 AUROC cross-domain penalty
- `metric_ranking_analysis.json`: Spearman rho=0.000 (key NeurIPS finding)
- `auprc_method_comparison.json`: Rolling var beats A2P on AUROC+AUPRC+F1-tol
- `smd_baselines.json`: Rolling var AUROC=0.773 on SMD (vs A2P not reported)
- `ltw_f1_analysis.json`: A2P lead-time ratio 4.46x < random 5.07x (title not validated)

## Publication Figures

Four figures in `figures/`:
- `fig1_metric_ranking_inversion.png`: F1-tol vs AUROC ranking comparison
- `fig2_f1_inflation.png`: 8x inflation cascade + grey-swan + leakage
- `fig3_auroc_all_methods.png`: All methods AUROC - A2P is worst
- `fig4_neurips_contribution.png`: Ranking inversion summary table

## Why: NeurIPS Research Implication

Core NeurIPS contribution: "F1-tolerance and AUROC/AUPRC rankings are completely uncorrelated (Spearman rho=0.000). A2P evaluation framework is misleading. Propose AUPRC as primary metric. Show trivial baselines beat A2P on AUROC/AUPRC at all anomaly rates."

Next step: JEPA-AP backbone that achieves AUROC > 0.7 on multiple datasets.
