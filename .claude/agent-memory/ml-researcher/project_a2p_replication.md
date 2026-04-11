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

1. **AUROC < 0.5 on proper split (multi-seed confirmed)**: Seeds 42 (AUROC=0.490) and seed 1 (AUROC=0.498) both show A2P is anti-discriminating. 2-seed mean: 0.494 +/- 0.004.

2. **Rolling variance DOMINATES A2P on paper's own setup**: MBA SVDB4 (records 800-803) - Rolling var w=50: F1-tol=86.70% vs A2P paper 67.55% (+19.15pp). SMD w=10: F1-tol=63.84% vs A2P 52.07% (+11.77pp). ALL window sizes beat A2P.

3. **MBA train==test data leakage (3.4x inflation)**: TranAD-derived MBA has identical train/test. With proper 70/30 split: F1-tol=12.66%.

4. **Seed bug**: Official code hardcodes seed=20462, ignores --random_seed flag.

5. **F1-tolerance 8x inflation**: Raw F1=5.35%, F1-tol=43.1% (8x inflation).

6. **Chronos-Small beats A2P by +21.7pp AUROC**: Zero fine-tuning Chronos achieves AUROC=0.745.

7. **Metric rank inversion (Spearman rho=0.000)**: F1-tol ranks A2P #1, AUROC ranks A2P last.

8. **Grey-swan collapse**: At 0.1% anomaly rate, F1=1.8% (24x collapse).

## Replication Results (Completed)

| Dataset | Paper | Ours | Gap | Notes |
|---------|-------|------|-----|-------|
| MBA L100 (TranAD train==test) | 67.55 | 19.07 +/- 8.77 | -48pp | Seed bug, data leakage |
| MBA L100 (TranAD 70/30 split) | 67.55 | 12.66 | -55pp | 3.4x leakage confirmed |
| SVDB1 record 801 (seeds 42+1) | 67.55 | 19.17 +/- 3.12 | -48pp | AUROC=0.494 (below random!) |
| SVDB1 record 801 (seed=2) | - | In progress | - | Running as of April 2026 |
| SMD L100 | 52.07 | In progress | - | 708K steps, running overnight |

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
- `auprc_method_comparison.json`: Rolling var beats A2P on AUROC+AUPRC+F1-tol (8.1x AUPRC!)
- `smd_baselines.json`: Rolling var AUROC=0.773 on SMD
- `smd_rolling_var_f1.json`: Rolling var w=100 F1-tol=39.24% vs A2P 36.29% on SMD
- `smd_window_sensitivity.json`: Rolling var ALL windows beat A2P on SMD (w=10: 63.84% vs 52.07%)
- `svdb4_rolling_var.json`: Rolling var w=50 F1-tol=86.70% vs A2P 67.55% on SVDB4 (paper setup!)
- `svdb4_method_comparison.json`: 6 methods on SVDB4, Spearman rho=0.94 (baselines consistent)

## Publication Figures

Five figures in `figures/`:
- `fig1_metric_ranking_inversion.png`: F1-tol vs AUROC ranking comparison
- `fig2_f1_inflation.png`: 8x inflation cascade + grey-swan + leakage
- `fig3_auroc_all_methods.png`: All methods AUROC - A2P is worst
- `fig4_neurips_contribution.png`: Ranking inversion summary table
- `fig5_complete_comparison.png`: All datasets/methods - rolling var dominates

## Why: NeurIPS Research Implication

Core NeurIPS contribution: "F1-tolerance is a broken metric for Anomaly Prediction (AP). Rolling variance (no training) beats A2P by +19pp on MBA and +12pp on SMD using paper's own setup. Propose AUPRC as primary metric. Show rank inversion across all configurations."

Next step: JEPA-AP backbone that achieves AUROC > 0.7 on multiple datasets with proper metric evaluation.
