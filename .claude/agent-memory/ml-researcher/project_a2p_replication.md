---
name: A2P Replication Findings (April 2026)
description: Critical findings from replicating Park et al. ICML 2025 "When Will It Fail?" - reveals A2P evaluation flaws
type: project
---

## Key Facts

Paper: "When Will It Fail? Anomaly to Prompt for Forecasting Future Anomalies in Time Series" (ICML 2025, Park et al.)  
Code: https://github.com/KU-VGI/AP  
Replication dir: `/home/sagemaker-user/IndustrialJEPA/paper-replications/when-will-it-fail/`

## Critical Findings (All Confirmed, April 2026)

1. **AUROC=0.528 (near-random)**: A2P's raw anomaly scores cannot discriminate anomalies from normals. F1=43.1% comes entirely from 50-step tolerance window (8x inflation from raw F1=5.35%).

2. **Chronos-Small beats A2P by +21.7pp AUROC**: Frozen Chronos-Small (20M, zero fine-tuning) achieves AUROC=0.745 vs A2P=0.528. Using only forecast MSE as anomaly score. This means A2P's specialized architecture provides negative value for raw discrimination.

3. **MBA train==test data leakage**: TranAD-derived MBA has identical train and test sets (100% overlap). With proper 70/30 split, F1 drops from 43.1% to 12.66% (3.4x inflation). Paper uses PhysioNet SVDB records with proper temporal separation.

4. **Grey-swan collapse**: F1 degrades as sqrt(anomaly_rate). At 0.1% rate (realistic industrial): F1=1.8% vs 3.12% rate: F1=19.1% (10x collapse). Entire AP evaluation framework is broken for real industrial use.

5. **F1 metric inflates performance**: Paper reports 67.55% F1 on MBA. Our replication: 19.1% (gap = 48pp). Most of gap explained by: (1) data leakage in TranAD MBA, (2) wrong data source.

6. **Seed bug**: Official code ignores --random_seed, runs all "seeds" with hardcoded seed=20462. Variance in paper is from checkpoint stochasticity, not true seed variation.

## Replication Results

| Dataset | Paper | Ours | Gap | Notes |
|---------|-------|------|-----|-------|
| MBA L100 | 67.55 | 19.07 | -48pp | TranAD train==test data issue |
| SMD L100 | 52.07 | TBD | - | Still running after 2h+ (708K timesteps) |

## Code Locations

- Probe scripts: `probe_grey_swan.py`, `probe_calibration.py`, `probe_ltw_f1.py`, `probe_data_integrity.py`, `probe_chronos_baseline.py`
- Results: `results/improvements/` (5 JSON files)
- Quarto notebook: `notebooks/a2p_replication_summary.qmd`
- Run orchestrator: `run_replication.py`

## Why: NeurIPS Research Implication

The main contribution from this replication for NeurIPS: "A2P's F1 is mostly tolerance-window artifact. Zero-shot Chronos beats A2P on AUROC. AP evaluation framework needs AUPRC/DR@FAR, not F1-with-tolerance."
